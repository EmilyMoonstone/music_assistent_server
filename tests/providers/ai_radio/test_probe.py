"""Tests for rehearsing an AI Radio segment in the editor."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import AudioFormat

from music_assistant.providers.ai_radio.constants import (
    PROBE_CLIP_PREFIX,
    PROBE_EXAMPLE_SONGS,
    TTS_CLIP_PCM_FORMAT,
)
from music_assistant.providers.ai_radio.effects import ClipEffects, EffectSound
from music_assistant.providers.ai_radio.probe import AIRadioProbeMixin

_PLAYER = "player_a"


class ProbeRenderer(AIRadioProbeMixin):
    """Harness with the AI, the voice and the audio stubbed out."""

    def __init__(self, queue: list[tuple[str, MediaType]] | None = None) -> None:
        """Initialize the harness around a player playing the given queue."""
        self.logger = logging.getLogger("tests.ai_radio.probe")
        self.prompts: list[str] = []
        self.spoken: list[str] = []
        self.dressed: list[ClipEffects | None] = []
        self.reply = "Guten Abend, hier ist Mika."
        items = [
            SimpleNamespace(queue_item_id=f"qi_{index}", name=name, media_type=media_type)
            for index, (name, media_type) in enumerate(queue or [])
        ]
        by_id = {item.queue_item_id: index for index, item in enumerate(items)}

        def next_item(_queue_id: str, item_id: str) -> Any:
            index = by_id[item_id] + 1
            return items[index] if index < len(items) else None

        cast("Any", self).mass = SimpleNamespace(
            player_queues=SimpleNamespace(
                get=lambda _id: SimpleNamespace(current_item=items[0] if items else None),
                get_next_item=next_item,
            )
        )

    def _configured_now(self) -> datetime:
        return datetime(2026, 9, 30, 20, 0, tzinfo=UTC)

    async def _resolve_deferred_placeholders(self, prompt: str) -> dict[str, str]:
        return {"<timestamp>": "Mittwoch, 20:00", "<weather_hourly>": ""}

    def _apply_break_memory(self, prompt: str, host_id: str, news: bool) -> str:
        return prompt

    async def _generate_text(self, **kwargs: Any) -> str:
        self.prompts.append(kwargs["prompt"])
        return self.reply

    def _tts_language(self, language: str) -> str:
        return language or "de-DE"

    async def _render_tts_media(self, text: str, *_args: Any) -> tuple[str, StreamType, Any]:
        self.spoken.append(text)
        return "voice.mp3", StreamType.LOCAL_FILE, AudioFormat(content_type=ContentType.MP3)

    async def _probe_duration(self, _path: str) -> int:
        return 4

    async def _measure_loudness(self, _path: str) -> float:
        return -20.0

    def _wanted_loudness(self, _queue_id: str) -> float:
        return -14.0

    async def _effect_sound(self, path: str, target: float) -> EffectSound:
        return EffectSound(path=path, seconds=2.0, gain_db=0.0)

    async def _probe_pcm(self, *args: Any) -> bytes:
        self.dressed.append(args[3])
        return b"\x00" * TTS_CLIP_PCM_FORMAT.pcm_sample_size * 4


_HOST = {
    "id": "mika",
    "instructions": "Du bist Mika.",
    "effects": {"jingles": [{"source": "/media/intro.mp3", "tags": []}]},
}


async def test_a_rehearsal_names_the_songs_on_the_player() -> None:
    """The songs around the break come from the player's queue, clips passed over."""
    renderer = ProbeRenderer(
        [
            ("A - One", MediaType.TRACK),
            ("clip", MediaType.SOUND_EFFECT),
            ("B - Two", MediaType.TRACK),
        ]
    )
    section = {"prompt": "Nach <prev_songinfo> kommt <next_songinfo>, dann <very_next_songinfo>."}

    result = await renderer.render_probe(_HOST, section, _PLAYER)

    assert renderer.prompts == [f"Nach A - One kommt B - Two, dann {PROBE_EXAMPLE_SONGS[2]}."]
    assert result["text"] == "Guten Abend, hier ist Mika."
    assert result["jingle"] == "/media/intro.mp3"
    assert result["seconds"] == pytest.approx(4.0, abs=0.1)
    assert Path(result["path"]).name.startswith(PROBE_CLIP_PREFIX)
    Path(result["path"]).unlink()


async def test_a_rehearsal_opens_bare_when_the_segment_never_has_a_jingle() -> None:
    """A segment set to never open with a jingle is rehearsed without one."""
    renderer = ProbeRenderer()
    section = {"prompt": "Sag hallo.", "jingle_before": "never"}

    result = await renderer.render_probe(_HOST, section, _PLAYER)

    assert result["jingle"] == ""
    assert renderer.dressed == [None]
    Path(result["path"]).unlink()


async def test_only_the_latest_rehearsal_is_kept() -> None:
    """A new rehearsal replaces the file of the one before."""
    renderer = ProbeRenderer()
    first = await renderer.render_probe(_HOST, {"prompt": "Eins."}, _PLAYER)
    second = await renderer.render_probe(_HOST, {"prompt": "Zwei."}, _PLAYER)

    assert not Path(first["path"]).exists()
    assert Path(second["path"]).exists()
    Path(second["path"]).unlink()


async def test_a_rehearsal_keeps_to_the_segments_length() -> None:
    """The script is cut to the segment's character limit like a break on air."""
    renderer = ProbeRenderer()
    renderer.reply = "Erster Satz ist kurz. " * 20
    section = {"prompt": "Rede.", "constraints": {"max_chars": 60}}

    result = await renderer.render_probe(_HOST, section, _PLAYER)

    assert len(result["text"]) <= 80
    Path(result["path"]).unlink()


async def test_a_segment_without_a_prompt_cannot_be_rehearsed() -> None:
    """There is nothing to write without a prompt."""
    with pytest.raises(InvalidDataError):
        await ProbeRenderer().render_probe(_HOST, {"prompt": "  "}, _PLAYER)


async def test_the_weather_is_left_out_when_there_is_none() -> None:
    """An empty weather placeholder tells the AI to skip the weather."""
    renderer = ProbeRenderer()

    result = await renderer.render_probe(_HOST, {"prompt": "Wetter: <weather_hourly>"}, _PLAYER)

    assert "<weather_hourly>" not in renderer.prompts[0]
    assert renderer.prompts[0] != "Wetter: "
    Path(result["path"]).unlink()
