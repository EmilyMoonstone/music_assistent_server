"""Tests for the speech-to-text helpers."""

from __future__ import annotations

import asyncio
import shutil
import wave
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import ProviderType
from music_assistant_models.errors import MusicAssistantError

from music_assistant.helpers import stt
from music_assistant.helpers.plugin_engines import get_stt_engines, resolve_stt_engine
from music_assistant.models.plugin import STT_SAMPLE_RATE, PluginProvider, STTEngine


def _create_plugin(
    instance_id: str,
    engine_names: list[str] | None = None,
    *,
    priority: int = 50,
    available: bool = True,
    provider_type: ProviderType = ProviderType.PLUGIN,
) -> MagicMock:
    """Create a mock plugin provider exposing speech-to-text engines with the given names."""
    provider = MagicMock(spec=PluginProvider)
    provider.instance_id = instance_id
    provider.priority = priority
    provider.available = available
    provider.type = provider_type
    provider.get_stt_engines = AsyncMock(
        return_value=[
            STTEngine(id=f"stt.{name.lower()}", name=name, provider=provider)
            for name in engine_names or []
        ]
    )
    provider.speech_to_text = AsyncMock(return_value="")
    return provider


def _create_mass(*providers: MagicMock) -> MagicMock:
    """Create a mock MusicAssistant holding the given providers."""
    mass = MagicMock()
    mass.providers = list(providers)
    return mass


async def test_stt_engines_are_asked_of_every_available_plugin() -> None:
    """Every available plugin is asked, by priority and then engine name."""
    late = _create_plugin("late", ["Zulu", "Alpha"], priority=60)
    early = _create_plugin("early", ["Mike"], priority=10)
    offline = _create_plugin("offline", ["Gone"], available=False)
    music = _create_plugin("music", ["Wrong"], provider_type=ProviderType.MUSIC)

    engines = await get_stt_engines(_create_mass(late, early, offline, music))

    assert [engine.uid for engine in engines] == [
        "early/stt.mike",
        "late/stt.alpha",
        "late/stt.zulu",
    ]


async def test_stt_engines_skip_a_plugin_that_fails_to_list_them() -> None:
    """A plugin that cannot list its engines leaves the others' engines in place."""
    broken = _create_plugin("broken")
    broken.get_stt_engines.side_effect = RuntimeError("offline")
    working = _create_plugin("working", ["Whisper"])

    engines = await get_stt_engines(_create_mass(broken, working))

    assert [engine.uid for engine in engines] == ["working/stt.whisper"]


async def test_resolve_stt_engine_never_substitutes_another() -> None:
    """A selection resolves to its engine, a stale one to None."""
    mass = _create_mass(_create_plugin("hass", ["Cloud", "Whisper"]))

    engine = await resolve_stt_engine(mass, "hass/stt.whisper")

    assert engine is not None
    assert engine.id == "stt.whisper"
    assert await resolve_stt_engine(mass, "hass/stt.gone") is None
    assert await resolve_stt_engine(mass, None) is None


async def test_plugin_without_engines_lists_none() -> None:
    """The base plugin lists no speech-to-text engines."""
    assert await PluginProvider.get_stt_engines(MagicMock(spec=PluginProvider)) == []


def _decoded() -> AsyncMock:
    """Stand in for the decoding, handing back a little PCM."""
    return AsyncMock(return_value=b"\x00\x01" * 100)


async def test_transcribe_returns_the_first_words_heard() -> None:
    """The first engine that hears words answers, the ones after it are not asked."""
    plugin = _create_plugin("hass", ["Alpha", "Beta"])
    plugin.speech_to_text.side_effect = ["  Das Radio für Musikentdecker  ", "unused"]

    with patch.object(stt, "decode_for_stt", _decoded()):
        text = await stt.transcribe(_create_mass(plugin), "/media/jingle.mp3", "de-DE")

    assert text == "Das Radio für Musikentdecker"
    plugin.speech_to_text.assert_awaited_once_with(
        b"\x00\x01" * 100, language="de-DE", engine_id="stt.alpha"
    )


async def test_transcribe_hands_a_failure_or_silence_to_the_next_engine(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing engine, or one that heard nothing, hands over to the next one."""
    plugin = _create_plugin("hass", ["Alpha", "Beta", "Gamma"])
    plugin.speech_to_text.side_effect = [MusicAssistantError("quota"), "", "mit Mika"]

    with caplog.at_level("WARNING"), patch.object(stt, "decode_for_stt", _decoded()):
        text = await stt.transcribe(_create_mass(plugin), "/media/jingle.mp3")

    assert text == "mit Mika"
    assert [call.kwargs["engine_id"] for call in plugin.speech_to_text.await_args_list] == [
        "stt.alpha",
        "stt.beta",
        "stt.gamma",
    ]
    assert "hass/stt.alpha failed: quota" in caplog.text


async def test_transcribe_returns_nothing_when_no_engine_heard_words() -> None:
    """Engines that ran but heard no words give an empty text, not an error."""
    plugin = _create_plugin("hass", ["Alpha", "Beta"])
    plugin.speech_to_text.side_effect = [MusicAssistantError("quota"), ""]

    with patch.object(stt, "decode_for_stt", _decoded()):
        assert await stt.transcribe(_create_mass(plugin), "/media/jingle.mp3") == ""


async def test_transcribe_raises_when_every_engine_failed() -> None:
    """Every engine failing is an error naming each of them."""
    plugin = _create_plugin("hass", ["Alpha", "Beta"])
    plugin.speech_to_text.side_effect = [MusicAssistantError("quota"), TimeoutError()]

    with (
        patch.object(stt, "decode_for_stt", _decoded()),
        pytest.raises(MusicAssistantError, match=r"hass/stt\.alpha: quota"),
    ):
        await stt.transcribe(_create_mass(plugin), "/media/jingle.mp3")


async def test_transcribe_without_engines_does_not_decode() -> None:
    """Without an engine the file is not decoded at all."""
    decode = _decoded()

    with (
        patch.object(stt, "decode_for_stt", decode),
        pytest.raises(MusicAssistantError, match="No speech-to-text engine"),
    ):
        await stt.transcribe(_create_mass(_create_plugin("hass")), "/media/jingle.mp3")

    decode.assert_not_awaited()


async def test_query_stt_engine_caps_a_wedged_engine() -> None:
    """An engine that does not answer in time fails with a clear error."""
    plugin = _create_plugin("hass", ["Slow"])
    engine = (await plugin.get_stt_engines())[0]

    async def _never(*_args: Any, **_kwargs: Any) -> str:
        await asyncio.sleep(10)
        return ""

    plugin.speech_to_text.side_effect = _never

    with pytest.raises(MusicAssistantError, match=r"did not respond within 0\.01s"):
        await stt.query_stt_engine(engine, b"", timeout=0.01)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
async def test_decode_for_stt_gives_mono_pcm_at_the_engine_rate(tmp_path: Path) -> None:
    """A stereo 44.1 kHz file comes out as mono 16-bit PCM at the engine rate."""
    source = tmp_path / "jingle.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\x00\x00\x00\x00" * 44100)

    audio = await stt.decode_for_stt(str(source))

    # one second of mono 16-bit samples
    assert abs(len(audio) - STT_SAMPLE_RATE * 2) <= 64


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
async def test_decode_for_stt_reports_an_unreadable_file(tmp_path: Path) -> None:
    """A file ffmpeg cannot read is an error naming the file."""
    missing = tmp_path / "missing.mp3"

    with pytest.raises(MusicAssistantError, match=r"missing\.mp3"):
        await stt.decode_for_stt(str(missing))
