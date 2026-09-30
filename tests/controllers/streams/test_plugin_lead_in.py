"""Tests for the lead-in a plugin can ask for from the track before one of its items."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ContentType, CrossfadeMode, MediaType, StreamType
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.streams.audio import StreamsAudio
from music_assistant.controllers.streams.audio_buffer import AudioBuffer
from music_assistant.controllers.streams.smart_fades.fades import StandardCrossFade
from music_assistant.models.plugin import LeadIn, PluginProvider

PCM = AudioFormat(content_type=ContentType.PCM_S16LE, sample_rate=48000, bit_depth=16, channels=2)


class LeadInPlugin(PluginProvider):
    """Plugin stand-in that answers get_lead_in with a fixed lead-in."""

    def __init__(self, lead_in: LeadIn | Exception | None) -> None:
        """Initialize with the answer to give, or the error to raise."""
        self.answer = lead_in
        self.asked: list[StreamDetails] = []

    def get_lead_in(self, streamdetails: StreamDetails) -> LeadIn | None:
        """Record the question and give the fixed answer."""
        self.asked.append(streamdetails)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _buffer() -> AudioBuffer:
    """Build a prepared incoming buffer."""
    audio_buffer = MagicMock(spec=AudioBuffer)
    audio_buffer.has_error = False
    audio_buffer.is_valid.return_value = True
    audio_buffer.duration_available = 30.0
    audio_buffer.eof = False
    audio_buffer.ready = MagicMock()
    audio_buffer.ready.is_set.return_value = True
    return audio_buffer


def _item(media_type: MediaType, provider: str = "plugin--1") -> Any:
    """Build a resolved incoming queue item of the given type."""
    streamdetails = StreamDetails(
        provider=provider,
        item_id="clip-1",
        audio_format=AudioFormat(content_type=ContentType.WAV),
        media_type=media_type,
        stream_type=StreamType.CUSTOM,
        path="/tmp/clip.wav",  # noqa: S108
        duration=20,
    )
    streamdetails.buffer = _buffer()
    return SimpleNamespace(
        queue_id="queue-1",
        queue_item_id="clip",
        media_type=media_type,
        streamdetails=streamdetails,
        media_item=None,
    )


def _audio(provider: object) -> StreamsAudio:
    """Build the audio controller with one provider behind every provider id."""
    mass = MagicMock()
    mass.get_provider.return_value = provider
    return StreamsAudio(cast("Any", mass))


def test_a_plugin_sound_effect_gets_the_lead_in_its_plugin_asks_for() -> None:
    """The plugin behind a sound effect decides how the track before blends into it."""
    plugin = LeadInPlugin(LeadIn(seconds=3, fade_in=False))
    item = _item(MediaType.SOUND_EFFECT)

    lead_in = _audio(plugin).plugin_lead_in(item)

    assert lead_in == LeadIn(seconds=3, fade_in=False)
    assert plugin.asked == [item.streamdetails]


@pytest.mark.parametrize(
    ("media_type", "answer", "provider"),
    [
        (MediaType.TRACK, LeadIn(seconds=3), "plugin"),
        (MediaType.RADIO, LeadIn(seconds=3), "plugin"),
        (MediaType.SOUND_EFFECT, None, "plugin"),
        (MediaType.SOUND_EFFECT, LeadIn(seconds=0), "plugin"),
        (MediaType.SOUND_EFFECT, RuntimeError("broken"), "plugin"),
        (MediaType.SOUND_EFFECT, LeadIn(seconds=3), "music provider"),
    ],
)
def test_everything_else_keeps_a_cut(
    media_type: MediaType, answer: LeadIn | Exception | None, provider: str
) -> None:
    """Tracks, other media, a plugin that declines or fails, and non-plugins get no lead-in."""
    backend: object = LeadInPlugin(answer) if provider == "plugin" else MagicMock()

    assert _audio(backend).plugin_lead_in(_item(media_type)) is None


def test_an_unresolved_item_gets_no_lead_in() -> None:
    """Without stream details there is nothing to ask the plugin about."""
    item = _item(MediaType.SOUND_EFFECT)
    item.streamdetails = None

    assert _audio(LeadInPlugin(LeadIn(seconds=3))).plugin_lead_in(item) is None
    assert _audio(LeadInPlugin(LeadIn(seconds=3))).plugin_lead_in(None) is None


def test_the_lead_in_replaces_the_configured_fade_with_its_own_standard_one() -> None:
    """A smart fade is not attempted, and the plugin's length wins over the configured one."""
    audio = _audio(MagicMock())

    mode, window = audio._select_buffered_crossfade(
        _item(MediaType.SOUND_EFFECT).streamdetails,
        CrossfadeMode.SMART_CROSSFADE,
        standard_crossfade_duration=8,
        fade_out_seconds=12,
        lead_in=LeadIn(seconds=3),
    )

    assert mode == CrossfadeMode.STANDARD_CROSSFADE
    assert window == 3


def test_a_queue_without_crossfade_keeps_its_cut() -> None:
    """The lead-in only shapes a crossfade the queue already does."""
    audio = _audio(MagicMock())

    mode, window = audio._select_buffered_crossfade(
        _item(MediaType.SOUND_EFFECT).streamdetails,
        CrossfadeMode.DISABLED,
        standard_crossfade_duration=8,
        fade_out_seconds=12,
        lead_in=LeadIn(seconds=3),
    )

    assert mode == CrossfadeMode.DISABLED
    assert window == 0


def test_crossfade_is_allowed_into_a_plugin_item_that_asks_for_it() -> None:
    """The track before may fade into a sound effect only when its plugin wants a lead-in."""
    queue_item = SimpleNamespace(
        queue_id="queue-1", queue_item_id="song", media_type=MediaType.TRACK, media_item=None
    )
    asking = _audio(LeadInPlugin(LeadIn(seconds=3)))
    declining = _audio(LeadInPlugin(None))
    clip = _item(MediaType.SOUND_EFFECT)

    assert asking.crossfade_allowed(
        cast("Any", queue_item),
        CrossfadeMode.STANDARD_CROSSFADE,
        "player-1",
        flow_mode=True,
        next_queue_item=clip,
    )
    assert not declining.crossfade_allowed(
        cast("Any", queue_item),
        CrossfadeMode.STANDARD_CROSSFADE,
        "player-1",
        flow_mode=True,
        next_queue_item=clip,
    )


def test_a_plugin_item_never_fades_out_into_the_next_track() -> None:
    """Only the fade into a plugin item is opened up, not the one out of it."""
    clip = _item(MediaType.SOUND_EFFECT)
    song = SimpleNamespace(queue_id="queue-1", queue_item_id="song", media_type=MediaType.TRACK)

    assert not _audio(LeadInPlugin(LeadIn(seconds=3))).crossfade_allowed(
        clip,
        CrossfadeMode.STANDARD_CROSSFADE,
        "player-1",
        flow_mode=True,
        next_queue_item=cast("Any", song),
    )


@pytest.mark.parametrize(("fade_in", "curve"), [(True, "qsin"), (False, "nofade")])
def test_a_talk_up_keeps_the_incoming_item_at_full_level(fade_in: bool, curve: str) -> None:
    """Without a fade-in the incoming audio starts at full level while the tail fades out."""
    fade = StandardCrossFade(MagicMock(), crossfade_duration=3, fade_in=fade_in)
    fade.build(PCM.pcm_sample_size * 3, PCM.pcm_sample_size * 3, PCM)

    chain = fade.filters[0].apply("[in]", "[out]")

    assert f"afade=t=in:start_sample=0:nb_samples=144000:curve={curve}" in chain[1]
    assert "curve=qsin" in chain[0]


def test_a_talk_over_ducks_the_track_before_for_the_whole_overlap() -> None:
    """A lead-in with a duck depth holds the outgoing tail down instead of fading it out."""
    fade = StandardCrossFade(MagicMock(), crossfade_duration=10, fade_in=False, duck_depth=0.6)
    fade.build(PCM.pcm_sample_size * 10, PCM.pcm_sample_size * 10, PCM)

    chain = fade.filters[0].apply("[in]", "[out]")

    assert "[duck_keep_in]volume=0.4[duck_keep]" in chain
    # the dip and the closing fade take LEAD_IN_DUCK_SECONDS at 48 kHz
    assert "afade=t=out:start_sample=0:nb_samples=38400:curve=qsin" in chain[2]
    assert "afade=t=out:start_sample=441600:nb_samples=38400" in chain[3]
    assert "curve=nofade" in chain[-2]
