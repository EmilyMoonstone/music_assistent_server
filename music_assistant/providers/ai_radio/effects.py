"""Sound effects a host puts around its breaks: jingles ahead of them and a music bed below."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any

from music_assistant_models.errors import InvalidDataError

from music_assistant.constants import ANNOUNCE_ALERT_FILE
from music_assistant.helpers.dsp import ComplexFilter, ComplexFilterInput

from .constants import (
    DEFAULT_JINGLE_CHANCE,
    DEFAULT_JINGLE_SELECTION,
    DEFAULT_LEAD_IN,
    DEFAULT_LEAD_IN_SECONDS,
    DEFAULT_MUSIC_BED_LEVEL,
    EFFECT_BUILTIN_JINGLE,
    JINGLE_CHOICE_INSTRUCTION,
    JINGLE_OCCASION_TAGS,
    JINGLE_SELECTION_MODES,
    JINGLE_SLOT_OCCASIONS,
    JINGLE_TIME_TAGS,
    JINGLE_VOICE_OVERLAP_SECONDS,
    LEAD_IN_MODES,
    LEAD_IN_SECONDS_RANGE,
    MAX_JINGLE_TEXT_CHARS,
    MAX_JINGLES,
    MUSIC_BED_FADE_IN_SECONDS,
    MUSIC_BED_LEVEL_RANGE,
    MUSIC_BED_TAIL_SECONDS,
)
from .helpers import coerce_float, coerce_int

# the reply line the LLM names its jingle on, at the very start of its answer
_JINGLE_CHOICE_LINE = re.compile(
    r"^\s*\**\s*JINGLE\s*:\s*\**\s*(\d+|none)\s*\**\s*$", re.IGNORECASE | re.MULTILINE
)


@dataclass(frozen=True, slots=True)
class EffectSound:
    """A jingle or bed ready to be mixed: where it is, how long it runs and the gain it needs."""

    path: str
    seconds: float
    gain_db: float


@dataclass(frozen=True, slots=True)
class ClipEffects:
    """The sounds one break is dressed with."""

    jingle: EffectSound | None = None
    bed: EffectSound | None = None


def default_effects() -> dict[str, Any]:
    """Return the effects of a host that has none configured."""
    return {
        "jingles": [],
        "jingle_chance": DEFAULT_JINGLE_CHANCE,
        "jingle_selection": DEFAULT_JINGLE_SELECTION,
        "music_bed": "",
        "music_bed_level": DEFAULT_MUSIC_BED_LEVEL,
        "lead_in": DEFAULT_LEAD_IN,
        "lead_in_seconds": DEFAULT_LEAD_IN_SECONDS,
    }


def normalize_effects(raw: Any) -> dict[str, Any]:
    """
    Validate and normalize the effects of a host.

    :param raw: The effects as stored or sent by a client, missing values fall back to off.
    """
    effects = raw if isinstance(raw, dict) else {}
    normalized = default_effects()
    raw_jingles = effects.get("jingles")
    jingles = [_normalize_jingle(item) for item in raw_jingles or [] if isinstance(item, dict)]
    # the single news and show jingle of the first version become tagged library entries
    for legacy_key, tags in (("news_jingle", ["news"]), ("show_jingle", ["intro", "outro"])):
        if source := str(effects.get(legacy_key) or "").strip():
            jingles.append(_normalize_jingle({"source": source, "tags": tags}))
    if len(jingles) > MAX_JINGLES:
        raise InvalidDataError(f"A host can have at most {MAX_JINGLES} jingles")
    normalized["jingles"] = jingles
    normalized["jingle_chance"] = min(
        100, max(0, coerce_int(effects.get("jingle_chance"), DEFAULT_JINGLE_CHANCE))
    )
    selection = str(effects.get("jingle_selection") or DEFAULT_JINGLE_SELECTION)
    normalized["jingle_selection"] = (
        selection if selection in JINGLE_SELECTION_MODES else DEFAULT_JINGLE_SELECTION
    )
    bed = str(effects.get("music_bed") or "").strip()
    if bed == EFFECT_BUILTIN_JINGLE:
        raise InvalidDataError("There is no built-in music bed, set a file path or URL")
    if bed and not is_valid_source(bed):
        raise InvalidDataError(f"The music bed must be an absolute file path or URL, got {bed!r}")
    normalized["music_bed"] = bed
    low, high = MUSIC_BED_LEVEL_RANGE
    level = coerce_float(effects.get("music_bed_level"), DEFAULT_MUSIC_BED_LEVEL)
    normalized["music_bed_level"] = min(high, max(low, level))
    lead_in = str(effects.get("lead_in") or DEFAULT_LEAD_IN)
    normalized["lead_in"] = lead_in if lead_in in LEAD_IN_MODES else DEFAULT_LEAD_IN
    low, high = LEAD_IN_SECONDS_RANGE
    seconds = coerce_float(effects.get("lead_in_seconds"), DEFAULT_LEAD_IN_SECONDS)
    normalized["lead_in_seconds"] = min(high, max(low, seconds))
    return normalized


def jingle_occasion(news: bool, weather: bool, slot_when: str) -> str:
    """
    Return what a break is, as far as picking its jingle goes.

    :param news: Whether the break reports news.
    :param weather: Whether the break talks about the weather.
    :param slot_when: The slot the break airs in, e.g. start_of_playlist.
    """
    # the news carries its own jingle even when it opens or closes a show
    if news:
        return "news"
    if slot_when in JINGLE_SLOT_OCCASIONS:
        return JINGLE_SLOT_OCCASIONS[slot_when]
    if weather:
        return "weather"
    return "transition"


def jingle_candidates(
    effects: dict[str, Any], occasion: str, hour: int, last_source: str = ""
) -> list[dict[str, Any]]:
    """
    Return the jingles that fit a break, or an empty list when it opens without one.

    :param effects: The normalized effects of the host speaking the break.
    :param occasion: What the break is, see jingle_occasion.
    :param hour: The local hour the break airs at.
    :param last_source: The jingle the host played last, avoided when there is a choice.
    """
    jingles: list[dict[str, Any]] = effects.get("jingles") or []
    general = [
        jingle
        for jingle in jingles
        if "general" in jingle["tags"] or not set(jingle["tags"]) & set(JINGLE_OCCASION_TAGS)
    ]
    if occasion == "transition":
        pool = general
    else:
        pool = [jingle for jingle in jingles if occasion in jingle["tags"]]
        # a show still gets its ident from the general jingles, news and weather do not
        if not pool and occasion in ("intro", "outro"):
            pool = general
    now_tag = time_of_day_tag(hour)
    timed = [
        jingle
        for jingle in pool
        if now_tag in jingle["tags"] or not set(jingle["tags"]) & set(JINGLE_TIME_TAGS)
    ]
    pool = timed or pool
    if len(pool) > 1 and last_source:
        pool = [jingle for jingle in pool if jingle["source"] != last_source] or pool
    return pool


def pick_jingle(
    candidates: list[dict[str, Any]], genres: set[str], rng: random.Random | None = None
) -> dict[str, Any] | None:
    """
    Pick a jingle without the LLM, preferring one tagged for the genre of the next song.

    :param candidates: The jingles that fit the break.
    :param genres: The genres of the song after the break, lowercase.
    :param rng: The random source, for tests.
    """
    if not candidates:
        return None
    rng = rng or random.Random()
    matching = [jingle for jingle in candidates if _genre_matches(jingle["tags"], genres)]
    return rng.choice(matching or candidates)


def jingle_choice_prompt(candidates: list[dict[str, Any]]) -> str:
    """Return the prompt block that asks the LLM to pick one of the given jingles."""
    lines = [JINGLE_CHOICE_INSTRUCTION]
    for number, jingle in enumerate(candidates, start=1):
        tags = ", ".join(jingle["tags"]) or "general"
        said = f'"{jingle["text"]}"' if jingle["text"] else "(no words given)"
        lines.append(f"{number}. [{tags}] {said}")
    return "\n".join(lines)


def take_jingle_choice(
    text: str, candidates: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str]:
    """
    Return the jingle the LLM named and its script with the choice line taken out.

    :param text: The LLM's reply.
    :param candidates: The jingles it was offered, in the order they were numbered.
    """
    choice: dict[str, Any] | None = None
    if (match := _JINGLE_CHOICE_LINE.search(text)) is not None:
        answer = match.group(1).lower()
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            choice = candidates[int(answer) - 1]
    # the line never reaches the listener, whether it named a valid jingle or not
    return choice, _JINGLE_CHOICE_LINE.sub("", text).strip()


def time_of_day_tag(hour: int) -> str:
    """Return the time-of-day tag covering a local hour."""
    for tag, (start, end) in JINGLE_TIME_TAGS.items():
        if (start <= hour < end) if start < end else (hour >= start or hour < end):
            return tag
    return "daytime"


def jingle_text_from_lyrics(lyrics: str | None) -> str:
    """Return the words of a jingle from its lyrics tag, without cue marks like [Spoken]."""
    if not lyrics:
        return ""
    return " ".join(re.sub(r"\[[^\]]*\]", " ", lyrics).split())[:MAX_JINGLE_TEXT_CHARS]


def resolve_source(source: str) -> str:
    """Return the path or URL ffmpeg reads for a configured sound."""
    return ANNOUNCE_ALERT_FILE if source == EFFECT_BUILTIN_JINGLE else source


def effect_filters(effects: ClipEffects, voice_seconds: int | None) -> list[str | ComplexFilter]:
    """
    Return the filters that dress an already levelled voice with its effects.

    :param effects: The sounds to add.
    :param voice_seconds: The whole seconds the voice runs, None when unknown.
    """
    filters: list[str | ComplexFilter] = []
    if (bed := effects.bed) is not None:
        # the duration is whole seconds and cut short, so the voice may run on for almost
        # a second more: the padding covers that, and the fade only starts once it is over
        filters.append(f"apad=pad_dur={MUSIC_BED_TAIL_SECONDS + 1}")
        filters.append(
            ComplexFilter(
                body="amix=inputs=2:duration=first:normalize=0",
                inputs=[
                    ComplexFilterInput(
                        path=bed.path,
                        filters=(
                            f"volume={round(bed.gain_db, 2)}dB,"
                            f"afade=t=in:d={MUSIC_BED_FADE_IN_SECONDS}"
                        ),
                        input_args=["-stream_loop", "-1"],
                    )
                ],
            )
        )
        if voice_seconds is not None:
            filters.append(f"afade=t=out:st={voice_seconds + 1}:d={MUSIC_BED_TAIL_SECONDS}")
    if (jingle := effects.jingle) is not None:
        delay_ms = max(0, round((jingle.seconds - JINGLE_VOICE_OVERLAP_SECONDS) * 1000))
        filters.append(f"adelay={delay_ms}:all=1")
        filters.append(
            ComplexFilter(
                body="amix=inputs=2:duration=longest:normalize=0",
                inputs=[
                    ComplexFilterInput(
                        path=jingle.path, filters=f"volume={round(jingle.gain_db, 2)}dB"
                    )
                ],
            )
        )
    return filters


def dressed_duration(effects: ClipEffects, voice_seconds: int | None) -> int | None:
    """
    Return how many whole seconds a break runs once its effects are added.

    :param effects: The sounds added to the break.
    :param voice_seconds: The whole seconds the voice runs, None when unknown.
    """
    if voice_seconds is None:
        return None
    seconds = float(voice_seconds)
    if effects.bed is not None:
        seconds += MUSIC_BED_TAIL_SECONDS + 1
    if effects.jingle is not None:
        seconds += max(0.0, effects.jingle.seconds - JINGLE_VOICE_OVERLAP_SECONDS)
    return math.ceil(seconds)


def is_valid_source(source: str) -> bool:
    """Return whether a configured sound names something ffmpeg can open."""
    return source.startswith(("/", "http://", "https://"))


def _normalize_jingle(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate one library jingle: where it is, what it is tagged for and what it says."""
    source = str(raw.get("source") or "").strip()
    if source != EFFECT_BUILTIN_JINGLE and not is_valid_source(source):
        raise InvalidDataError(
            f"A jingle must be the built-in gong, an absolute file path or URL, got {source!r}"
        )
    raw_tags = raw.get("tags")
    if not isinstance(raw_tags, list):
        raw_tags = []
    tags = sorted({"_".join(str(tag).lower().split()) for tag in raw_tags if str(tag).strip()})
    return {
        "source": source,
        "tags": tags,
        "text": " ".join(str(raw.get("text") or "").split())[:MAX_JINGLE_TEXT_CHARS],
    }


def _genre_matches(tags: list[str], genres: set[str]) -> bool:
    """Return whether a jingle carries a tag for one of the genres, e.g. indie in indie rock."""
    free_tags = set(tags) - set(JINGLE_OCCASION_TAGS) - set(JINGLE_TIME_TAGS)
    return any(tag.replace("_", " ") in genre for tag in free_tags for genre in genres)
