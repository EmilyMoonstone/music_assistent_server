"""Unit tests for the sound effects AI Radio hosts put around their breaks."""

from __future__ import annotations

import random
from typing import Any

import pytest
from music_assistant_models.errors import InvalidDataError

from music_assistant.constants import ANNOUNCE_ALERT_FILE
from music_assistant.helpers.dsp import ComplexFilter
from music_assistant.helpers.ffmpeg import get_ffmpeg_args
from music_assistant.providers.ai_radio.constants import (
    DEFAULT_JINGLE_AFTER_GAP_MINUTES,
    DEFAULT_JINGLE_CHANCE,
    DEFAULT_MUSIC_BED_LEVEL,
    MAX_JINGLES,
    MUSIC_BED_LEVEL_RANGE,
    MUSIC_BED_TAIL_SECONDS,
    TTS_CLIP_PCM_FORMAT,
)
from music_assistant.providers.ai_radio.effects import (
    ClipEffects,
    EffectSound,
    after_jingle_candidates,
    default_effects,
    dressed_duration,
    effect_filters,
    jingle_candidates,
    jingle_choice_prompt,
    jingle_occasion,
    jingle_text_from_lyrics,
    merge_jingle_modes,
    normalize_effects,
    pick_jingle,
    resolve_source,
    strip_jingle_words,
    take_jingle_choices,
    time_of_day_tag,
    wants_after_jingle,
)

JINGLE = EffectSound(path="/media/jingle.mp3", seconds=2.4, gain_db=-3.0)
BED = EffectSound(path="/media/bed.mp3", seconds=120.0, gain_db=-20.5)


def _library(*jingles: tuple[str, list[str]]) -> dict[str, Any]:
    """Build normalized effects holding the given (source, tags) jingles."""
    return normalize_effects(
        {"jingles": [{"source": source, "tags": tags} for source, tags in jingles]}
    )


def test_a_host_without_effects_airs_bare() -> None:
    """Missing or malformed effects fall back to every sound being off."""
    assert normalize_effects(None) == default_effects()
    assert normalize_effects("nonsense") == default_effects()
    assert default_effects()["music_bed_level"] == DEFAULT_MUSIC_BED_LEVEL
    assert default_effects()["jingle_chance"] == DEFAULT_JINGLE_CHANCE


def test_library_jingles_are_normalized() -> None:
    """Tags are lowercased, deduplicated and sorted, words are flattened to one line."""
    effects = normalize_effects(
        {
            "jingles": [
                {
                    "source": " /media/ai_radio/untergrund.mp3 ",
                    "tags": ["News", "news", "Late Night", " "],
                    "text": "Neues aus\n dem Untergrund.",
                },
                {"source": "builtin"},
            ],
            "jingle_chance": 150,
            "jingle_selection": "sometimes",
        }
    )

    assert effects["jingles"] == [
        {
            "source": "/media/ai_radio/untergrund.mp3",
            "tags": ["late_night", "news"],
            "text": "Neues aus dem Untergrund.",
        },
        {"source": "builtin", "tags": [], "text": ""},
    ]
    assert effects["jingle_chance"] == 100
    assert effects["jingle_selection"] == "ai"


def test_the_first_versions_single_jingles_become_library_entries() -> None:
    """A stored news or show jingle keeps working as a tagged entry."""
    effects = normalize_effects({"news_jingle": "builtin", "show_jingle": "/media/ident.mp3"})

    assert effects["jingles"] == [
        {"source": "builtin", "tags": ["news"], "text": ""},
        {"source": "/media/ident.mp3", "tags": ["intro", "outro"], "text": ""},
    ]
    assert "news_jingle" not in effects


@pytest.mark.parametrize(
    "effects",
    [
        {"music_bed": "builtin"},
        {"music_bed": "bed.mp3"},
        {"jingles": [{"source": "media/jingle.mp3"}]},
        {"jingles": [{"source": "ftp://example.test/ident.mp3"}]},
        {"jingles": [{"source": "/media/j.mp3"}] * (MAX_JINGLES + 1)},
    ],
)
def test_effects_refuse_what_ffmpeg_cannot_open(effects: dict[str, Any]) -> None:
    """There is no shipped bed, a relative path or other scheme is rejected up front."""
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
    ("news", "weather", "slot_when", "expected"),
    [
        (True, False, "start_of_playlist", "news"),
        (False, False, "start_of_playlist", "intro"),
        (False, True, "end_of_playlist", "outro"),
        (False, True, "between_songs", "weather"),
        (False, False, "between_songs", "transition"),
    ],
)
def test_the_occasion_follows_what_the_break_is(
    news: bool, weather: bool, slot_when: str, expected: str
) -> None:
    """News wins over everything, a show's ends over the weather."""
    assert jingle_occasion(news, weather, slot_when) == expected


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (5, "morning"),
        (9, "morning"),
        (10, "daytime"),
        (17, "evening"),
        (22, "late_night"),
        (3, "late_night"),
    ],
)
def test_time_of_day_tags_cover_the_whole_day(hour: int, expected: str) -> None:
    """Every hour has its tag, and the night wraps around midnight."""
    assert time_of_day_tag(hour) == expected


def test_news_only_gets_a_news_jingle() -> None:
    """Without a news-tagged jingle the news opens bare, general jingles are not used."""
    effects = _library(("/media/general.mp3", []), ("/media/news.mp3", ["news"]))

    assert [j["source"] for j in jingle_candidates(effects, "news", 12)] == ["/media/news.mp3"]
    assert jingle_candidates(_library(("/media/general.mp3", [])), "news", 12) == []
    assert jingle_candidates(_library(("/media/general.mp3", [])), "weather", 12) == []


def test_a_show_falls_back_to_general_jingles() -> None:
    """An intro without an intro jingle still gets a general one."""
    effects = _library(("/media/general.mp3", ["general"]), ("/media/news.mp3", ["news"]))

    sources = [j["source"] for j in jingle_candidates(effects, "intro", 12)]

    assert sources == ["/media/general.mp3"]


def test_transitions_draw_from_general_jingles_only() -> None:
    """A jingle tagged for an occasion is kept for it; untagged ones count as general."""
    effects = _library(
        ("/media/untagged.mp3", ["indie"]),
        ("/media/news.mp3", ["news"]),
        ("/media/both.mp3", ["general", "news"]),
    )

    sources = {j["source"] for j in jingle_candidates(effects, "transition", 12)}

    assert sources == {"/media/untagged.mp3", "/media/both.mp3"}


def test_jingles_for_another_time_of_day_are_left_out() -> None:
    """A late-night jingle does not open the morning show, an untimed one fits any time."""
    effects = _library(
        ("/media/night.mp3", ["late_night"]),
        ("/media/morning.mp3", ["morning"]),
        ("/media/any.mp3", []),
    )

    sources = {j["source"] for j in jingle_candidates(effects, "transition", 7)}

    assert sources == {"/media/morning.mp3", "/media/any.mp3"}


def test_only_off_time_jingles_are_still_better_than_none() -> None:
    """When nothing is tagged for the hour, the off-time jingles still play."""
    effects = _library(("/media/night.mp3", ["late_night"]))

    assert len(jingle_candidates(effects, "transition", 12)) == 1


def test_the_last_jingle_is_not_played_twice_in_a_row() -> None:
    """With a choice, the host skips the jingle it just played."""
    effects = _library(("/media/a.mp3", []), ("/media/b.mp3", []))

    candidates = jingle_candidates(effects, "transition", 12, "/media/a.mp3")

    assert [j["source"] for j in candidates] == ["/media/b.mp3"]
    single = _library(("/media/a.mp3", []))
    assert len(jingle_candidates(single, "transition", 12, "/media/a.mp3")) == 1


def test_random_pick_prefers_a_jingle_for_the_next_genre() -> None:
    """A jingle tagged indie opens an indie rock song when the choice is left to chance."""
    effects = _library(("/media/calm.mp3", ["calm"]), ("/media/indie.mp3", ["indie"]))
    candidates = jingle_candidates(effects, "transition", 12)

    for seed in range(10):
        chosen = pick_jingle(candidates, {"indie rock"}, random.Random(seed))
        assert chosen is not None
        assert chosen["source"] == "/media/indie.mp3"
    assert pick_jingle([], {"pop"}) is None


def test_the_llm_is_offered_numbered_jingles_with_their_words() -> None:
    """The prompt lists every candidate with its tags and what it says."""
    effects = normalize_effects(
        {"jingles": [{"source": "/media/a.mp3", "tags": ["calm"], "text": "Mika hier."}]}
    )

    prompt = jingle_choice_prompt(effects["jingles"])

    assert "'JINGLE: <number>'" in prompt
    assert "(1)" in prompt
    assert '1. [calm] "Mika hier."' in prompt
    assert "AFTER" not in prompt


@pytest.mark.parametrize(
    ("reply", "expected_source", "expected_text"),
    [
        ("JINGLE: 2\nGuten Abend.", "/media/b.mp3", "Guten Abend."),
        ("**JINGLE: 1**\n\nGuten Abend.", "/media/a.mp3", "Guten Abend."),
        ("JINGLE: 7\nGuten Abend.", None, "Guten Abend."),
        ("Guten Abend.", None, "Guten Abend."),
    ],
)
def test_the_choice_line_is_read_and_never_spoken(
    reply: str, expected_source: str | None, expected_text: str
) -> None:
    """A valid number picks its jingle, and the line is stripped either way."""
    candidates = _library(("/media/a.mp3", []), ("/media/b.mp3", []))["jingles"]

    choice, text = take_jingle_choices(reply, candidates)

    assert (choice.before["source"] if choice.before else None) == expected_source
    assert choice.after is None
    assert text == expected_text


def test_jingle_words_come_from_the_lyrics_without_cue_marks() -> None:
    """Suno's [Spoken] style marks are dropped from what the jingle says."""
    lyrics = "[Spoken, dry] Keine Floskeln. Nur gute Musik. Mika am Mikro. [End]"

    assert jingle_text_from_lyrics(lyrics) == "Keine Floskeln. Nur gute Musik. Mika am Mikro."
    assert jingle_text_from_lyrics("[Jingle]") == ""
    assert jingle_text_from_lyrics(None) == ""


def test_the_builtin_jingle_is_the_announcement_gong() -> None:
    """The shipped gong needs no file of the user's own."""
    assert resolve_source("builtin") == ANNOUNCE_ALERT_FILE
    assert resolve_source("/media/a.mp3") == "/media/a.mp3"


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


CLOSER = EffectSound(path="/media/closer.mp3", seconds=3.0, gain_db=-1.0)


def test_the_gap_between_closing_jingles_is_kept_in_range() -> None:
    """The gap defaults, and is clamped to what the editor offers."""
    assert normalize_effects(None)["jingle_after_gap_minutes"] == DEFAULT_JINGLE_AFTER_GAP_MINUTES
    assert normalize_effects({"jingle_after_gap_minutes": -5})["jingle_after_gap_minutes"] == 0
    assert normalize_effects({"jingle_after_gap_minutes": 999})["jingle_after_gap_minutes"] == 240


@pytest.mark.parametrize(
    ("modes", "expected"),
    [
        (["", "always"], "always"),
        (["no_post", "always"], "always"),
        (["no_post", "never"], "no_post"),
        (["never", "never"], "never"),
        (["never", ""], "auto"),
        (["never", "auto"], "auto"),
        ([], "auto"),
    ],
)
def test_merged_sections_share_one_jingle_mode(modes: list[str], expected: str) -> None:
    """A jingle any part asks for plays; one is left out only when every part leaves it."""
    assert merge_jingle_modes(modes) == expected


def test_a_section_asking_for_a_jingle_gets_one_even_without_a_fitting_tag() -> None:
    """News without a news jingle stays bare, unless its section asks for one every time."""
    general = _library(("/media/general.mp3", []))
    only_weather = _library(("/media/weather.mp3", ["weather"]))

    assert jingle_candidates(general, "news", 12) == []
    assert [j["source"] for j in jingle_candidates(general, "news", 12, always=True)] == [
        "/media/general.mp3"
    ]
    # nothing general either, so any jingle stands in
    assert [j["source"] for j in jingle_candidates(only_weather, "news", 12, always=True)] == [
        "/media/weather.mp3"
    ]


def test_a_closing_jingle_prefers_the_occasions_own() -> None:
    """After the news its closer comes first, then the general ones; others never close."""
    effects = _library(
        ("/media/general.mp3", []),
        ("/media/news.mp3", ["news"]),
        ("/media/weather.mp3", ["weather"]),
    )

    news = [j["source"] for j in after_jingle_candidates(effects, "news", 12)]
    plain = [j["source"] for j in after_jingle_candidates(effects, "transition", 12)]

    assert news == ["/media/news.mp3", "/media/general.mp3"]
    assert plain == ["/media/general.mp3"]
    assert after_jingle_candidates(_library(("/media/news.mp3", ["news"])), "transition", 12) == []
    assert len(after_jingle_candidates(_library(("/media/news.mp3", ["news"])), "x", 12, True)) == 1


@pytest.mark.parametrize(
    ("occasion", "onset", "expected"),
    [
        ("news", None, True),
        ("transition", None, False),
        ("transition", 0.5, True),
        ("transition", 12.0, False),
    ],
)
def test_without_the_llm_a_break_closes_after_the_news_or_into_early_vocals(
    occasion: str, onset: float | None, expected: bool
) -> None:
    """The rule closes the news, and bridges into a song that sings right away."""
    assert wants_after_jingle(occasion, onset) is expected


def test_both_ends_share_one_numbered_list() -> None:
    """A jingle on offer at both ends is listed once, and each end names its numbers."""
    effects = _library(("/media/a.mp3", ["news"]), ("/media/b.mp3", []), ("/media/c.mp3", []))
    before = effects["jingles"][:2]
    after = effects["jingles"][1:]

    prompt = jingle_choice_prompt(before, after, vocal_onset=1.4)

    assert prompt.count("/media") == 0
    assert prompt.count("[general]") == 2
    assert "'JINGLE: <number>'" in prompt
    assert "(1, 2)" in prompt
    assert "'AFTER: <number or none>'" in prompt
    assert "(2, 3)" in prompt
    assert "after 1s" in prompt


def test_a_closing_jingle_asked_for_every_time_cannot_be_declined() -> None:
    """The after line then offers no 'none'."""
    effects = _library(("/media/a.mp3", []))

    prompt = jingle_choice_prompt([], effects["jingles"], after_always=True)

    assert "JINGLE:" not in prompt
    assert "'AFTER: <number>'" in prompt


def test_long_jingle_words_are_cut_in_the_prompt() -> None:
    """Only the start of long words is sent, enough to pick by."""
    effects = normalize_effects({"jingles": [{"source": "/media/a.mp3", "text": "la " * 100}]})

    prompt = jingle_choice_prompt(effects["jingles"])

    assert len(prompt.splitlines()[1]) < 140


@pytest.mark.parametrize(
    ("reply", "before", "after", "answered"),
    [
        ("JINGLE: 1\nAFTER: 3\nHallo.", "/media/a.mp3", "/media/c.mp3", True),
        ("JINGLE: 2\nAFTER: none\nHallo.", "/media/b.mp3", None, True),
        # 1 is only on offer to open the break
        ("JINGLE: 2\nAFTER: 1\nHallo.", "/media/b.mp3", None, True),
        ("JINGLE: 1\nHallo.", "/media/a.mp3", None, False),
    ],
)
def test_both_choice_lines_are_read_and_never_spoken(
    reply: str, before: str, after: str | None, answered: bool
) -> None:
    """Each line picks from its own end's jingles, and neither is voiced."""
    effects = _library(("/media/a.mp3", []), ("/media/b.mp3", []), ("/media/c.mp3", []))

    choice, text = take_jingle_choices(reply, effects["jingles"][:2], effects["jingles"][1:])

    assert choice.before is not None
    assert choice.before["source"] == before
    assert (choice.after["source"] if choice.after else None) == after
    assert choice.after_answered is answered
    assert text == "Hallo."


def test_a_closing_jingle_follows_the_voice() -> None:
    """Without a bed the closer is appended right after the voice."""
    filters = effect_filters(ClipEffects(after=CLOSER), voice_seconds=10)

    closer = filters[-1]
    assert isinstance(closer, ComplexFilter)
    assert closer.body == "concat=n=2:v=0:a=1"
    assert closer.inputs[0].path == CLOSER.path
    assert closer.inputs[0].filters == "volume=-1.0dB"
    assert dressed_duration(ClipEffects(after=CLOSER), 10) == 13


def test_a_closing_jingle_comes_in_over_the_fading_bed() -> None:
    """With a bed the closer overlaps its tail instead of waiting for it to die out."""
    effects = ClipEffects(bed=BED, after=CLOSER)

    closer = effect_filters(effects, voice_seconds=10)[-1]

    assert isinstance(closer, ComplexFilter)
    assert closer.body == "acrossfade=d=1.5:c1=nofade:c2=nofade"
    # 10s of voice, 3s of bed tail, the 3s closer overlapping it by 1.5s
    assert dressed_duration(effects, 10) == 15


def test_opening_and_closing_jingles_build_one_filtergraph() -> None:
    """The opener leads, the closer trails, both in the same graph as the voice."""
    filters = effect_filters(ClipEffects(jingle=JINGLE, bed=BED, after=CLOSER), voice_seconds=10)

    args = get_ffmpeg_args(
        input_format=TTS_CLIP_PCM_FORMAT,
        output_format=TTS_CLIP_PCM_FORMAT,
        filter_params=filters,
        input_path="/media/voice.mp3",
    )

    graph = args[args.index("-filter_complex") + 1]
    assert args.count("-i") == 4
    order = ["duration=first", "adelay", "duration=longest", "acrossfade"]
    assert [graph.index(step) for step in order] == sorted(graph.index(step) for step in order)


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        (
            "Kurz und knapp: Was draussen passiert. In Berlin regnet es.",
            "In Berlin regnet es.",
        ),
        ("Kopf aus, Lautsprecher an! Hier ist Mika.", "Hier ist Mika."),
        ("Hier ist Mika, kopf aus lautsprecher an, und los.", "Hier ist Mika, und los."),
        # two words are too common to be the jingle's own
        ("Das ist die erste Platte heute.", "Das ist die erste Platte heute."),
        # a line of short sentences is matched as a whole, across its full stops
        ("München leuchtet. Mika hier. Heute wird es warm.", "Heute wird es warm."),
        ("Mika hier, gleich kommt Musik.", "Mika hier, gleich kommt Musik."),
    ],
)
def test_a_script_does_not_repeat_its_jingles(script: str, expected: str) -> None:
    """Phrases the jingles sing are taken out of what the host reads."""
    jingles = [
        {"text": "Kurz und knapp. Was draussen passiert."},
        {"text": "Kopf aus, Lautsprecher an"},
        {"text": "Erste Platte."},
        {"text": "München leuchtet. Mika hier."},
    ]

    assert strip_jingle_words(script, jingles) == expected


def test_a_script_without_echoes_is_left_as_it_is() -> None:
    """Line breaks and spacing survive when nothing had to be cut."""
    script = "Guten Abend.\n\nGleich kommt Musik."

    assert strip_jingle_words(script, [{"text": "Kopf aus, Lautsprecher an"}]) == script
    assert strip_jingle_words(script, []) == script
