"""Sound effects a host puts around its breaks: jingles ahead of them and a music bed below."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from music_assistant_models.errors import InvalidDataError

from music_assistant.constants import ANNOUNCE_ALERT_FILE
from music_assistant.helpers.dsp import ComplexFilter, ComplexFilterInput

from .constants import (
    DEFAULT_MUSIC_BED_LEVEL,
    EFFECT_BUILTIN_JINGLE,
    EFFECT_SOURCE_KEYS,
    JINGLE_VOICE_OVERLAP_SECONDS,
    MUSIC_BED_FADE_IN_SECONDS,
    MUSIC_BED_LEVEL_RANGE,
    MUSIC_BED_TAIL_SECONDS,
    SHOW_JINGLE_SLOTS,
)
from .helpers import coerce_float


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
        "news_jingle": "",
        "show_jingle": "",
        "music_bed": "",
        "music_bed_level": DEFAULT_MUSIC_BED_LEVEL,
    }


def normalize_effects(raw: Any) -> dict[str, Any]:
    """
    Validate and normalize the effects of a host.

    :param raw: The effects as stored or sent by a client, missing values fall back to off.
    """
    effects = raw if isinstance(raw, dict) else {}
    normalized = default_effects()
    for key in EFFECT_SOURCE_KEYS:
        source = str(effects.get(key) or "").strip()
        if source == EFFECT_BUILTIN_JINGLE and key == "music_bed":
            raise InvalidDataError("There is no built-in music bed, set a file path or URL")
        if source and source != EFFECT_BUILTIN_JINGLE and not _is_valid_source(source):
            raise InvalidDataError(
                f"Effect '{key}' must be an absolute file path or an http(s) URL, got {source!r}"
            )
        normalized[key] = source
    low, high = MUSIC_BED_LEVEL_RANGE
    level = coerce_float(effects.get("music_bed_level"), DEFAULT_MUSIC_BED_LEVEL)
    normalized["music_bed_level"] = min(high, max(low, level))
    return normalized


def jingle_source(effects: dict[str, Any], news: bool, slot_when: str) -> str:
    """
    Return the jingle a break opens with, or an empty string when it has none.

    :param effects: The normalized effects of the host speaking the break.
    :param news: Whether the break reports news.
    :param slot_when: The slot the break airs in, e.g. start_of_playlist.
    """
    # the news carries its own jingle even when it opens or closes a show
    if news and effects.get("news_jingle"):
        return resolve_source(str(effects["news_jingle"]))
    if slot_when in SHOW_JINGLE_SLOTS and effects.get("show_jingle"):
        return resolve_source(str(effects["show_jingle"]))
    return ""


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


def _is_valid_source(source: str) -> bool:
    """Return whether a configured sound names something ffmpeg can open."""
    return source.startswith(("/", "http://", "https://"))
