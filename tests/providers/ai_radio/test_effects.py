"""Unit tests for the sound effects AI Radio hosts put around their breaks."""

from __future__ import annotations

import pytest
from music_assistant_models.errors import InvalidDataError

from music_assistant.constants import ANNOUNCE_ALERT_FILE
from music_assistant.helpers.dsp import ComplexFilter
from music_assistant.helpers.ffmpeg import get_ffmpeg_args
from music_assistant.providers.ai_radio.constants import (
    DEFAULT_MUSIC_BED_LEVEL,
    MUSIC_BED_LEVEL_RANGE,
    MUSIC_BED_TAIL_SECONDS,
    TTS_CLIP_PCM_FORMAT,
)
from music_assistant.providers.ai_radio.effects import (
    ClipEffects,
    EffectSound,
    default_effects,
    dressed_duration,
    effect_filters,
    jingle_source,
    normalize_effects,
)

JINGLE = EffectSound(path="/media/jingle.mp3", seconds=2.4, gain_db=-3.0)
BED = EffectSound(path="/media/bed.mp3", seconds=120.0, gain_db=-20.5)


def test_a_host_without_effects_airs_bare() -> None:
    """Missing or malformed effects fall back to every sound being off."""
    assert normalize_effects(None) == default_effects()
    assert normalize_effects("nonsense") == default_effects()
    assert default_effects()["music_bed_level"] == DEFAULT_MUSIC_BED_LEVEL


def test_effects_accept_the_builtin_gong_paths_and_urls() -> None:
    """A jingle may be the shipped gong, a file or a URL, and surrounding blanks are dropped."""
    effects = normalize_effects(
        {
            "news_jingle": "builtin",
            "show_jingle": " /media/ai_radio/ident.mp3 ",
            "music_bed": "https://example.test/bed.mp3",
            "music_bed_level": "-24",
        }
    )

    assert effects == {
        "news_jingle": "builtin",
        "show_jingle": "/media/ai_radio/ident.mp3",
        "music_bed": "https://example.test/bed.mp3",
        "music_bed_level": -24.0,
    }


@pytest.mark.parametrize(
    "effects",
    [
        {"music_bed": "builtin"},
        {"news_jingle": "media/jingle.mp3"},
        {"show_jingle": "ftp://example.test/ident.mp3"},
    ],
)
def test_effects_refuse_what_ffmpeg_cannot_open(effects: dict[str, str]) -> None:
    """There is no shipped bed, and a relative path or other scheme is rejected up front."""
    with pytest.raises(InvalidDataError):
        normalize_effects(effects)


def test_the_bed_level_is_kept_within_range() -> None:
    """A bed can neither drown the voice nor vanish below hearing."""
    low, high = MUSIC_BED_LEVEL_RANGE

    assert normalize_effects({"music_bed_level": 10})["music_bed_level"] == high
    assert normalize_effects({"music_bed_level": -90})["music_bed_level"] == low
    assert normalize_effects({"music_bed_level": "loud"})["music_bed_level"] == (
        DEFAULT_MUSIC_BED_LEVEL
    )


@pytest.mark.parametrize(
    ("news", "slot_when", "expected"),
    [
        (True, "between_songs", "/media/news.mp3"),
        (True, "start_of_playlist", "/media/news.mp3"),
        (False, "start_of_playlist", "/media/ident.mp3"),
        (False, "end_of_playlist", "/media/ident.mp3"),
        (False, "between_songs", ""),
    ],
)
def test_the_jingle_follows_what_the_break_is(news: bool, slot_when: str, expected: str) -> None:
    """News opens with the news jingle, a show's first and last break with the ident."""
    effects = normalize_effects(
        {"news_jingle": "/media/news.mp3", "show_jingle": "/media/ident.mp3"}
    )

    assert jingle_source(effects, news=news, slot_when=slot_when) == expected


def test_the_builtin_jingle_is_the_announcement_gong() -> None:
    """The shipped gong needs no file of the user's own."""
    effects = normalize_effects({"news_jingle": "builtin"})

    assert jingle_source(effects, news=True, slot_when="between_songs") == ANNOUNCE_ALERT_FILE


def test_a_break_without_effects_gets_no_filters() -> None:
    """Nothing is added to a bare break."""
    assert effect_filters(ClipEffects(), voice_seconds=10) == []


def test_the_bed_loops_under_the_voice_and_fades_out_after_it() -> None:
    """The bed fades in, runs on past the last word and fades out once the voice is done."""
    filters = effect_filters(ClipEffects(bed=BED), voice_seconds=10)

    assert filters[0] == f"apad=pad_dur={MUSIC_BED_TAIL_SECONDS + 1}"
    mix = filters[1]
    assert isinstance(mix, ComplexFilter)
    assert mix.body == "amix=inputs=2:duration=first:normalize=0"
    assert mix.inputs[0].path == BED.path
    assert mix.inputs[0].input_args == ["-stream_loop", "-1"]
    assert mix.inputs[0].filters.startswith("volume=-20.5dB,afade=t=in")
    assert filters[2] == f"afade=t=out:st=11:d={MUSIC_BED_TAIL_SECONDS}"


def test_a_bed_under_a_voice_of_unknown_length_is_not_faded() -> None:
    """Without a duration there is no point to fade from, the bed just ends with the padding."""
    filters = effect_filters(ClipEffects(bed=BED), voice_seconds=None)

    assert not any(isinstance(item, str) and item.startswith("afade=t=out") for item in filters)


def test_the_voice_comes_in_just_before_the_jingle_ends() -> None:
    """The voice is delayed by the jingle, less a short overlap, and mixed with it."""
    filters = effect_filters(ClipEffects(jingle=JINGLE), voice_seconds=10)

    assert filters[0] == "adelay=2000:all=1"
    mix = filters[1]
    assert isinstance(mix, ComplexFilter)
    assert mix.body == "amix=inputs=2:duration=longest:normalize=0"
    assert mix.inputs[0].path == JINGLE.path
    assert mix.inputs[0].filters == "volume=-3.0dB"


def test_jingle_and_bed_build_one_filtergraph() -> None:
    """Bed and jingle land in one graph, the bed under the voice and the jingle ahead of both."""
    filters = [
        "volume=2.0dB",
        *effect_filters(ClipEffects(jingle=JINGLE, bed=BED), voice_seconds=10),
        "alimiter=limit=-1.5dB",
    ]

    args = get_ffmpeg_args(
        input_format=TTS_CLIP_PCM_FORMAT,
        output_format=TTS_CLIP_PCM_FORMAT,
        filter_params=filters,
        input_path="/media/voice.mp3",
    )

    graph = args[args.index("-filter_complex") + 1]
    assert args.count("-i") == 3
    assert args[args.index("-stream_loop") + 2 : args.index("-stream_loop") + 4] == [
        "-i",
        BED.path,
    ]
    order = ["duration=first", "adelay", "duration=longest", "alimiter"]
    assert [graph.index(step) for step in order] == sorted(graph.index(step) for step in order)


def test_the_announced_duration_covers_jingle_and_bed_tail() -> None:
    """Core is told how long the dressed break really runs."""
    effects = ClipEffects(jingle=JINGLE, bed=BED)

    # 10s of voice, 3s of padded bed tail and 2.0s of jingle ahead of the voice
    assert dressed_duration(effects, 10) == 15
    assert dressed_duration(ClipEffects(), 10) == 10
    assert dressed_duration(effects, None) is None
