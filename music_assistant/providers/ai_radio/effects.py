"""Sound effects a host puts around its breaks: jingles around them and a music bed below."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any

from music_assistant_models.errors import InvalidDataError

from music_assistant.constants import ANNOUNCE_ALERT_FILE
from music_assistant.helpers.dsp import ComplexFilter, ComplexFilterInput

from .constants import (
    DEFAULT_JINGLE_AFTER_GAP_MINUTES,
    DEFAULT_JINGLE_CHANCE,
    DEFAULT_JINGLE_SELECTION,
    DEFAULT_JINGLE_SLOT_MODE,
    DEFAULT_LEAD_IN,
    DEFAULT_LEAD_IN_SECONDS,
    DEFAULT_MUSIC_BED_LEVEL,
    DEFAULT_POST_DUCK_PERCENT,
    EFFECT_BUILTIN_JINGLE,
    JINGLE_AFTER_ALWAYS_INSTRUCTION,
    JINGLE_AFTER_AUTO_INSTRUCTION,
    JINGLE_AFTER_EARLY_VOCAL_SECONDS,
    JINGLE_AFTER_GAP_RANGE,
    JINGLE_AFTER_ONSET_HINT,
    JINGLE_ANALYSIS_PROMPT,
    JINGLE_ANALYSIS_STYLE_TAGS,
    JINGLE_BEFORE_INSTRUCTION,
    JINGLE_CHOICE_CLOSING,
    JINGLE_LIST_HEADER,
    JINGLE_OCCASION_TAGS,
    JINGLE_PROMPT_TEXT_CHARS,
    JINGLE_SELECTION_MODES,
    JINGLE_SLOT_OCCASIONS,
    JINGLE_TIME_TAGS,
    JINGLE_VOICE_OVERLAP_SECONDS,
    LEAD_IN_MODES,
    LEAD_IN_SECONDS_RANGE,
    MAX_JINGLE_STYLE_CHARS,
    MAX_JINGLE_TEXT_CHARS,
    MAX_JINGLES,
    MIN_ECHOED_PHRASE_WORDS,
    MUSIC_BED_FADE_IN_SECONDS,
    MUSIC_BED_LEVEL_RANGE,
    MUSIC_BED_TAIL_SECONDS,
    POST_DUCK_RANGE,
    POST_GAP_RANGE,
    POST_MAX_RANGE,
    POST_TAIL_GAP,
)
from .helpers import coerce_float, coerce_int

# the reply lines the LLM names its jingles on, at the very start of its answer
_JINGLE_CHOICE_LINE = re.compile(
    r"^\s*\**\s*JINGLE\s*:\s*\**\s*(\d+|none)\s*\**\s*$", re.IGNORECASE | re.MULTILINE
)
_AFTER_CHOICE_LINE = re.compile(
    r"^\s*\**\s*AFTER\s*:\s*\**\s*(\d+|none)\s*\**\s*$", re.IGNORECASE | re.MULTILINE
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
    # the jingle closing the break, leading into the next song
    after: EffectSound | None = None


@dataclass(frozen=True, slots=True)
class JingleChoice:
    """What the LLM answered about a break's jingles."""

    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    # whether it answered on the after line at all: "none" is an answer, silence is not
    after_answered: bool = False


def default_effects() -> dict[str, Any]:
    """Return the effects of a host that has none configured."""
    return {
        "jingles": [],
        "jingle_chance": DEFAULT_JINGLE_CHANCE,
        "jingle_selection": DEFAULT_JINGLE_SELECTION,
        "jingle_after_gap_minutes": DEFAULT_JINGLE_AFTER_GAP_MINUTES,
        "music_bed": "",
        "music_bed_level": DEFAULT_MUSIC_BED_LEVEL,
        "lead_in": DEFAULT_LEAD_IN,
        "lead_in_seconds": DEFAULT_LEAD_IN_SECONDS,
        "post_gap_seconds": POST_TAIL_GAP,
        "post_max_seconds": 0.0,
        "post_duck_percent": DEFAULT_POST_DUCK_PERCENT,
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
    low_gap, high_gap = JINGLE_AFTER_GAP_RANGE
    gap = coerce_int(effects.get("jingle_after_gap_minutes"), DEFAULT_JINGLE_AFTER_GAP_MINUTES)
    normalized["jingle_after_gap_minutes"] = min(high_gap, max(low_gap, gap))
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
    for key, default, (minimum, maximum) in (
        ("post_gap_seconds", POST_TAIL_GAP, POST_GAP_RANGE),
        ("post_max_seconds", 0.0, POST_MAX_RANGE),
        ("post_duck_percent", DEFAULT_POST_DUCK_PERCENT, POST_DUCK_RANGE),
    ):
        value = coerce_float(effects.get(key), default)
        normalized[key] = min(float(maximum), max(float(minimum), value))
    return normalized


def merge_jingle_modes(modes: list[str]) -> str:
    """
    Return the jingle mode of a break merged from several sections, see JINGLE_SLOT_MODES.

    :param modes: The modes of the merged sections, "" for one that set none.
    """
    if "always" in modes:
        return "always"
    if modes and all(mode == "never" for mode in modes):
        return "never"
    return DEFAULT_JINGLE_SLOT_MODE


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
    effects: dict[str, Any],
    occasion: str,
    hour: int,
    last_source: str = "",
    always: bool = False,
) -> list[dict[str, Any]]:
    """
    Return the jingles that fit a break, or an empty list when it opens without one.

    :param effects: The normalized effects of the host speaking the break.
    :param occasion: What the break is, see jingle_occasion.
    :param hour: The local hour the break airs at.
    :param last_source: The jingle the host played last, avoided when there is a choice.
    :param always: The break's section asks for a jingle, so one without the occasion's tag
        stands in when none carries it.
    """
    jingles: list[dict[str, Any]] = effects.get("jingles") or []
    general = _general_jingles(jingles)
    if occasion == "transition":
        pool = general
    else:
        pool = [jingle for jingle in jingles if occasion in jingle["tags"]]
        # a show still gets its ident from the general jingles, news and weather do not
        if not pool and (always or occasion in ("intro", "outro")):
            pool = general
    if not pool and always:
        pool = list(jingles)
    pool = _timed(pool, hour)
    if len(pool) > 1 and last_source:
        pool = [jingle for jingle in pool if jingle["source"] != last_source] or pool
    return pool


def after_jingle_candidates(
    effects: dict[str, Any], occasion: str, hour: int, always: bool = False
) -> list[dict[str, Any]]:
    """
    Return the jingles that can close a break and lead into the next song.

    The ones tagged for the break's occasion (a news closer after the news) come first,
    then the general ones.

    :param effects: The normalized effects of the host speaking the break.
    :param occasion: What the break is, see jingle_occasion.
    :param hour: The local hour the break airs at.
    :param always: The break's section asks for a jingle, so any jingle stands in when
        none fits.
    """
    jingles: list[dict[str, Any]] = effects.get("jingles") or []
    tagged = [jingle for jingle in jingles if occasion in jingle["tags"]]
    pool = tagged + [jingle for jingle in _general_jingles(jingles) if jingle not in tagged]
    if not pool and always:
        pool = list(jingles)
    return _timed(pool, hour)


def wants_after_jingle(occasion: str, vocal_onset: float | None) -> bool:
    """
    Return whether a break closes with a jingle when no LLM decided it.

    :param occasion: What the break is, see jingle_occasion.
    :param vocal_onset: The second the next song starts singing, None when unknown.
    """
    if occasion == "news":
        return True
    return vocal_onset is not None and vocal_onset < JINGLE_AFTER_EARLY_VOCAL_SECONDS


def _general_jingles(jingles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the jingles tagged general, or for no occasion at all."""
    return [
        jingle
        for jingle in jingles
        if "general" in jingle["tags"] or not set(jingle["tags"]) & set(JINGLE_OCCASION_TAGS)
    ]


def _timed(pool: list[dict[str, Any]], hour: int) -> list[dict[str, Any]]:
    """Return the jingles of a pool that fit the hour, the whole pool when none does."""
    now_tag = time_of_day_tag(hour)
    timed = [
        jingle
        for jingle in pool
        if now_tag in jingle["tags"] or not set(jingle["tags"]) & set(JINGLE_TIME_TAGS)
    ]
    return timed or pool


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


def numbered_jingles(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the jingles offered for either end of a break, each once, in prompt order."""
    numbered = list(before)
    numbered += [jingle for jingle in after if jingle not in numbered]
    return numbered


def jingle_choice_prompt(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]] | None = None,
    after_always: bool = False,
    vocal_onset: float | None = None,
) -> str:
    """
    Return the prompt block asking the LLM to pick the jingles around a break.

    Every jingle is listed once, so a jingle on offer at both ends costs its words once.

    :param before: The jingles the break may open with, empty when it opens without one.
    :param after: The jingles that may close it, empty when it closes without one.
    :param after_always: The break closes with a jingle whatever the LLM thinks of it.
    :param vocal_onset: The second the next song starts singing, None when unknown.
    """
    after = after or []
    numbered = numbered_jingles(before, after)
    lines = [JINGLE_LIST_HEADER]
    for number, jingle in enumerate(numbered, start=1):
        tags = ", ".join(jingle["tags"]) or "general"
        words = jingle["text"][:JINGLE_PROMPT_TEXT_CHARS]
        lines.append(f'{number}. [{tags}] "{words}"' if words else f"{number}. [{tags}] -")
    if before:
        lines.append(JINGLE_BEFORE_INSTRUCTION.format(numbers=_numbers_of(before, numbered)))
    if after:
        numbers = _numbers_of(after, numbered)
        if after_always:
            lines.append(JINGLE_AFTER_ALWAYS_INSTRUCTION.format(numbers=numbers))
        else:
            onset = (
                JINGLE_AFTER_ONSET_HINT.format(seconds=vocal_onset)
                if vocal_onset is not None
                else ""
            )
            lines.append(JINGLE_AFTER_AUTO_INSTRUCTION.format(numbers=numbers, onset=onset))
    lines.append(JINGLE_CHOICE_CLOSING)
    return "\n".join(lines)


def take_jingle_choices(
    text: str, before: list[dict[str, Any]], after: list[dict[str, Any]] | None = None
) -> tuple[JingleChoice, str]:
    """
    Return the jingles the LLM named and its script with the choice lines taken out.

    A number outside the jingles offered for that end of the break counts as no choice.

    :param text: The LLM's reply.
    :param before: The jingles it was offered to open the break with.
    :param after: The jingles it was offered to close the break with.
    """
    after = after or []
    numbered = numbered_jingles(before, after)
    before_choice = _named_jingle(_JINGLE_CHOICE_LINE.search(text), numbered, before)
    after_match = _AFTER_CHOICE_LINE.search(text)
    choice = JingleChoice(
        before=before_choice,
        after=_named_jingle(after_match, numbered, after),
        after_answered=after_match is not None,
    )
    # the lines never reach the listener, whether they named a valid jingle or not
    script = _AFTER_CHOICE_LINE.sub("", _JINGLE_CHOICE_LINE.sub("", text)).strip()
    return choice, script


def strip_jingle_words(text: str, jingles: list[dict[str, Any]]) -> str:
    """
    Return a script without the phrases its jingles already say.

    The LLM is told not to repeat them, but a line it echoes anyway would air twice: once
    sung by the jingle, once read by the host. Phrases too short to be the jingle's own
    (see MIN_ECHOED_PHRASE_WORDS) are left alone.

    :param text: The script.
    :param jingles: The jingles airing around it.
    """
    cut = text
    for jingle in jingles:
        words_said = str(jingle.get("text") or "")
        # the whole line, and each of its parts on its own
        for phrase in [words_said, *_PHRASE_BREAK.split(words_said)]:
            words = [word for word in (w.strip(_PUNCTUATION) for w in phrase.split()) if word]
            if len(words) < MIN_ECHOED_PHRASE_WORDS:
                continue
            pattern = r"\b" + _WORD_GAP.join(re.escape(word) for word in words)
            cut = re.sub(pattern + r"\b[.!?,;:…]*", " ", cut, flags=re.IGNORECASE)
    if cut == text:
        return text
    text = cut
    # tidy what the cuts leave behind: doubled spaces, stray and doubled punctuation
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    text = re.sub(r"([.!?])[.!?,;:\s]*[.!?,;:]", r"\1", text)
    return text.strip(" ,;:")


_PHRASE_BREAK = re.compile(r"[.!?;:,\n]+")
# what may sit between the words of an echoed phrase: spaces, commas, dashes (en, em)
_WORD_GAP = "[\\s,;:\\-–—]+"  # noqa: RUF001
# what is trimmed off a jingle's words: punctuation, quotes of all kinds, dashes
_PUNCTUATION = ".,;:!?…\"'„“”‚‘’«»()-–—"  # noqa: RUF001


def _numbers_of(pool: list[dict[str, Any]], numbered: list[dict[str, Any]]) -> str:
    """Return the prompt numbers of a pool's jingles, like '1, 3'."""
    return ", ".join(str(numbered.index(jingle) + 1) for jingle in pool)


def _named_jingle(
    match: re.Match[str] | None, numbered: list[dict[str, Any]], pool: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return the jingle a reply line named, when it is one of the pool's."""
    if match is None or not (answer := match.group(1)).isdigit():
        return None
    number = int(answer)
    if not 1 <= number <= len(numbered) or (jingle := numbered[number - 1]) not in pool:
        return None
    return jingle


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
    return jingle_text(re.sub(r"\[[^\]]*\]", " ", lyrics))


def jingle_text(words: str) -> str:
    """Return the words of a jingle on one line, cut to the length the library keeps."""
    return " ".join(words.split())[:MAX_JINGLE_TEXT_CHARS]


def jingle_analysis_prompt(language: str) -> str:
    """
    Return the prompt asking the AI to listen to a jingle and suggest how to file it.

    :param language: The locale the style note is written for, like de-DE.
    """
    return JINGLE_ANALYSIS_PROMPT.format(
        occasions=", ".join(JINGLE_OCCASION_TAGS),
        times=", ".join(JINGLE_TIME_TAGS),
        style_tags=JINGLE_ANALYSIS_STYLE_TAGS,
        language=language,
    )


def parse_jingle_analysis(reply: str) -> dict[str, Any]:
    """
    Return the tags, words and style note the AI suggested for a jingle.

    Tags come back in the form the library stores; the occasion and time tags it knows are
    kept, other tags only up to the few style tags asked for.

    :param reply: The AI's answer, expected to hold a JSON object.
    :raises InvalidDataError: When the answer holds no JSON object.
    """
    start, end = reply.find("{"), reply.rfind("}")
    try:
        data = json.loads(reply[start : end + 1]) if 0 <= start < end else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise InvalidDataError("The AI did not answer with a description of the jingle")
    known = set(JINGLE_OCCASION_TAGS) | set(JINGLE_TIME_TAGS)
    tags: list[str] = []
    style_tags = 0
    raw_tags = data.get("tags")
    for raw_tag in raw_tags if isinstance(raw_tags, list) else []:
        tag = "_".join(str(raw_tag).lower().split())
        if not tag or tag in tags:
            continue
        if tag not in known:
            if style_tags >= JINGLE_ANALYSIS_STYLE_TAGS:
                continue
            style_tags += 1
        tags.append(tag)
    style = " ".join(str(data.get("style") or "").split())
    return {
        "tags": tags,
        "text": jingle_text(str(data.get("text") or "")),
        "style": style[:MAX_JINGLE_STYLE_CHARS],
    }


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
    if (after := effects.after) is not None:
        closer = ComplexFilterInput(path=after.path, filters=f"volume={round(after.gain_db, 2)}dB")
        overlap = _after_overlap(effects)
        # with a bed the closer comes in over its fading tail, without one right after the voice
        body = (
            f"acrossfade=d={overlap}:c1=nofade:c2=nofade" if overlap > 0 else "concat=n=2:v=0:a=1"
        )
        filters.append(ComplexFilter(body=body, inputs=[closer]))
    return filters


def _after_overlap(effects: ClipEffects) -> float:
    """Return the seconds the closing jingle overlaps the bed's tail, 0 without a bed."""
    if effects.after is None or effects.bed is None:
        return 0.0
    # acrossfade needs both sides to be at least as long as the overlap
    return round(min(MUSIC_BED_TAIL_SECONDS, effects.after.seconds / 2), 2)


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
    if effects.after is not None:
        seconds += effects.after.seconds - _after_overlap(effects)
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
