"""Tests for starting an AI Radio break over the outro of the song before it."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from music_assistant_models.enums import ContentType, CrossfadeMode, MediaType
from music_assistant_models.media_items import AudioFormat, ProviderMapping, SoundEffect, Track
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.models.plugin import LeadIn
from music_assistant.providers.ai_radio.constants import (
    ATTR_ALLOW_TALK_OVER,
    ATTR_HOST_ID,
    JINGLE_VOICE_OVERLAP_SECONDS,
    POST_TAIL_GAP,
)
from music_assistant.providers.ai_radio.effects import (
    ClipEffects,
    EffectSound,
    effect_filters,
    normalize_effects,
)
from music_assistant.providers.ai_radio.rendering import (
    AIRadioRenderMixin,
    _bed_after_talk_over,
    _TalkOverPlan,
)

_QUEUE_ID = "player_a"
_CLIP_ID = "sess_1"
_SONG_SECONDS = 200
_VOCAL_END = 180.0
_TALK_OVER = _SONG_SECONDS - _VOCAL_END - POST_TAIL_GAP


class TalkOverRenderer(AIRadioRenderMixin):
    """Minimal harness around one queue, with the lyrics and the crossfade settings stubbed."""

    domain = "ai_radio"
    instance_id = "ai_radio--test"

    def __init__(self, order: list[QueueItem]) -> None:
        """Initialize the harness around one queue played in the given order."""
        self.logger = logging.getLogger("tests.ai_radio.talk_over")
        self.order = order
        self.vocal_end: float | None = _VOCAL_END
        self.lookups = 0
        self.crossfade_mode = CrossfadeMode.STANDARD_CROSSFADE
        self.crossfade_seconds = 30
        self._hosts: dict[str, dict[str, Any]] = {}
        cast("Any", self).mass = SimpleNamespace(
            player_queues=SimpleNamespace(
                index_by_id=self._index_by_id,
                get_item=self._get_item,
                get=lambda queue_id: SimpleNamespace(queue_id=queue_id),
                items=lambda _queue_id, limit, offset: self.order[offset : offset + limit],
                all=lambda: [SimpleNamespace(queue_id=_QUEUE_ID)],
            ),
            streams=SimpleNamespace(get_crossfade_mode=lambda _queue: self.crossfade_mode),
            config=SimpleNamespace(get_raw_core_config_value=lambda *_args: self.crossfade_seconds),
        )
        self._sessions = {}

    def _index_by_id(self, queue_id: str, queue_item_id: str) -> int | None:
        assert queue_id == _QUEUE_ID
        ids = [item.queue_item_id for item in self.order]
        return ids.index(queue_item_id) if queue_item_id in ids else None

    def _get_item(self, queue_id: str, index: int) -> QueueItem | None:
        assert queue_id == _QUEUE_ID
        return self.order[index] if 0 <= index < len(self.order) else None

    async def _resolve_vocal_end(self, queue_item: QueueItem) -> tuple[float | None, str]:
        self.lookups += 1
        return self.vocal_end, "" if self.vocal_end is not None else "no lyrics found"


def _break_item(*, allow: bool = True) -> QueueItem:
    media_item = SoundEffect(
        item_id=_CLIP_ID,
        provider="ai_radio--test",
        name="Back announce",
        provider_mappings={
            ProviderMapping(
                item_id=_CLIP_ID, provider_domain="ai_radio", provider_instance="ai_radio--test"
            )
        },
    )
    return QueueItem(
        queue_id=_QUEUE_ID,
        queue_item_id="qi_break",
        name="Back announce",
        duration=None,
        media_item=media_item,
        extra_attributes={ATTR_ALLOW_TALK_OVER: allow, ATTR_HOST_ID: "mika"},
    )


def _track_item(name: str) -> QueueItem:
    media_item = Track(
        item_id=name,
        provider="library",
        name=name,
        provider_mappings={
            ProviderMapping(item_id=name, provider_domain="filesystem", provider_instance="fs")
        },
    )
    return QueueItem(
        queue_id=_QUEUE_ID,
        queue_item_id=f"qi_{name}",
        name=name,
        duration=_SONG_SECONDS,
        media_item=media_item,
    )


def _break_streamdetails() -> StreamDetails:
    return StreamDetails(
        provider="ai_radio--test",
        item_id=_CLIP_ID,
        audio_format=AudioFormat(content_type=ContentType.MP3),
        media_type=MediaType.SOUND_EFFECT,
    )


def _renderer(*, allow: bool = True, **effects: Any) -> tuple[TalkOverRenderer, QueueItem]:
    song, clip = _track_item("song"), _break_item(allow=allow)
    renderer = TalkOverRenderer([song, clip, _track_item("next")])
    renderer._hosts["mika"] = {"effects": normalize_effects(effects)}
    return renderer, clip


async def test_a_break_starts_over_the_outro_once_the_singing_is_over() -> None:
    """The record plays on under the break, held down, from just after its last sung line."""
    renderer, clip = _renderer(post_duck_percent=50)

    plan = await renderer._plan_talk_over(clip, _CLIP_ID)

    assert plan is not None
    assert plan.seconds == pytest.approx(_TALK_OVER)
    assert renderer._talk_over_fits(clip)
    assert renderer.get_lead_in(_break_streamdetails()) == LeadIn(
        seconds=pytest.approx(_TALK_OVER), fade_in=False, duck_depth=pytest.approx(0.5)
    )


async def test_the_hosts_longest_stretch_caps_the_talk_over() -> None:
    """A host that talks over at most so long starts that long before the song ends."""
    renderer, clip = _renderer(post_max_seconds=6)

    plan = await renderer._plan_talk_over(clip, _CLIP_ID)

    assert plan is not None
    assert plan.seconds == pytest.approx(6.0)


async def test_the_talk_over_reaches_no_further_than_the_crossfade() -> None:
    """The queue only holds a song's last crossfade seconds back, so that is the most."""
    renderer, clip = _renderer()
    renderer.crossfade_seconds = 8

    plan = await renderer._plan_talk_over(clip, _CLIP_ID)

    assert plan is not None
    assert plan.seconds == pytest.approx(8.0)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param({"crossfade_mode": CrossfadeMode.DISABLED}, "crossfade", id="no crossfade"),
        pytest.param({"vocal_end": None}, "no lyrics found", id="no lyrics"),
        pytest.param({"vocal_end": 198.5}, "too little outro", id="sings to the end"),
    ],
)
async def test_a_talk_over_that_cannot_be_done_is_skipped_and_logged(
    change: dict[str, Any], reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The break then starts after the song, and the log says why."""
    renderer, clip = _renderer()
    for key, value in change.items():
        setattr(renderer, key, value)

    with caplog.at_level(logging.INFO):
        plan = await renderer._plan_talk_over(clip, _CLIP_ID)

    assert plan is None
    assert not renderer._talk_over_fits(clip)
    assert "talk-over skipped" in caplog.text
    assert reason in caplog.text


async def test_a_section_that_did_not_opt_in_never_talks_over() -> None:
    """Without the section's say-so the lyrics are not even looked up."""
    renderer, clip = _renderer(allow=False)

    assert await renderer._plan_talk_over(clip, _CLIP_ID) is None
    assert renderer.lookups == 0


async def test_the_plan_is_kept_for_a_repeat_request() -> None:
    """Asking again for the same break reuses its plan instead of reading the lyrics again."""
    renderer, clip = _renderer()

    first = await renderer._plan_talk_over(clip, _CLIP_ID)
    second = await renderer._plan_talk_over(clip, _CLIP_ID)

    assert first is second
    assert renderer.lookups == 1


async def test_a_moved_song_takes_the_talk_over_away() -> None:
    """Once another song comes before the break, it starts with the host's usual lead-in."""
    renderer, clip = _renderer(lead_in="cut")
    await renderer._plan_talk_over(clip, _CLIP_ID)

    renderer.order.insert(1, _track_item("other"))

    assert not renderer._talk_over_fits(clip)
    assert renderer.get_lead_in(_break_streamdetails()) is None


async def test_a_break_that_opens_the_queue_does_not_talk_over() -> None:
    """With no song before it there is nothing to talk over."""
    renderer, clip = _renderer()
    renderer.order.pop(0)

    assert await renderer._plan_talk_over(clip, _CLIP_ID) is None


def _sound(path: str, seconds: float) -> EffectSound:
    return EffectSound(path=path, seconds=seconds, gain_db=0.0)


def test_the_bed_comes_in_as_the_song_under_the_voice_ends() -> None:
    """The bed waits for the talk-over, less what the opener already took of it."""
    effects = ClipEffects(jingle=_sound("/j.mp3", 3.0), bed=_sound("/bed.mp3", 60.0))
    plan = _TalkOverPlan(seconds=10.0, track_item_id="qi_song")

    dressed = _bed_after_talk_over(effects, plan, voice_seconds=30)

    assert dressed is not None
    assert dressed.bed_delay == pytest.approx(10.0 - (3.0 - JINGLE_VOICE_OVERLAP_SECONDS))


def test_a_bed_that_would_only_come_in_after_the_voice_is_left_out() -> None:
    """A talk-over longer than the voice leaves no room for the bed at all."""
    effects = ClipEffects(bed=_sound("/bed.mp3", 60.0))
    plan = _TalkOverPlan(seconds=12.0, track_item_id="qi_song")

    assert _bed_after_talk_over(effects, plan, voice_seconds=10) is None


def test_a_held_back_bed_is_delayed_in_the_mix() -> None:
    """The bed is faded in first and then delayed, so it fades in when it starts."""
    effects = ClipEffects(bed=_sound("/bed.mp3", 60.0), bed_delay=7.5)

    bed_mix = effect_filters(effects, voice_seconds=20)[1]

    assert bed_mix.inputs[0].filters.endswith("afade=t=in:d=1.0,adelay=7500:all=1")


async def test_the_log_tells_how_the_break_met_the_song_before_it() -> None:
    """An armed talk-over is logged with its length, a skipped one with a reason code."""
    renderer, clip = _renderer()
    await renderer._plan_talk_over(clip, _CLIP_ID)

    renderer_skipped, clip_skipped = _renderer()
    renderer_skipped.vocal_end = None
    await renderer_skipped._plan_talk_over(clip_skipped, _CLIP_ID)

    [armed] = await renderer.get_break_log()
    [skipped] = await renderer_skipped.get_break_log()
    assert armed["from_song"] == {"kind": "talk_over", "seconds": pytest.approx(_TALK_OVER)}
    assert armed["section"] == "Back announce"
    assert skipped["from_song"] == {"kind": "cut", "reason": {"code": "no_lyrics"}}


@pytest.mark.parametrize(
    ("vocal_end", "kind", "seconds"),
    [
        pytest.param(None, "crossfade", 5.0, id="unknown singing keeps the crossfade"),
        pytest.param(197.0, "crossfade", 2.6, id="shortened to the outro"),
        pytest.param(199.5, "cut", 0.0, id="singing to the end leaves it out"),
    ],
)
async def test_the_hosts_crossfade_never_reaches_into_the_singing(
    vocal_end: float | None, kind: str, seconds: float
) -> None:
    """Without a talk-over the host's crossfade applies, but only over the outro."""
    renderer, clip = _renderer(allow=False, lead_in="crossfade", lead_in_seconds=5)
    renderer.vocal_end = vocal_end

    plan = await renderer._plan_talk_over(clip, _CLIP_ID)

    assert plan is not None
    assert (plan.kind, plan.seconds) == (kind, pytest.approx(seconds))
    assert not renderer._talk_over_fits(clip)
    expected = LeadIn(seconds=pytest.approx(seconds)) if kind == "crossfade" else None
    assert renderer.get_lead_in(_break_streamdetails()) == expected


async def test_a_host_that_cuts_needs_no_lyrics() -> None:
    """A host without a crossfade looks nothing up for a break that does not talk over."""
    renderer, clip = _renderer(allow=False, lead_in="cut")

    assert await renderer._plan_talk_over(clip, _CLIP_ID) is None
    assert renderer.lookups == 0


async def test_the_log_can_be_read_for_one_station() -> None:
    """Breaks of other stations are left out when one is asked for."""
    renderer, clip = _renderer()
    await renderer._plan_talk_over(clip, _CLIP_ID)

    assert await renderer.get_break_log(station_id="other") == []
    assert len(await renderer.get_break_log()) == 1
