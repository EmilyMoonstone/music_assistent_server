"""Unit tests for AI Radio just-in-time clip rendering."""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import (
    ContentType,
    MediaType,
    StreamType,
    VolumeNormalizationMode,
)
from music_assistant_models.errors import (
    InvalidDataError,
    MediaNotFoundError,
    MusicAssistantError,
)
from music_assistant_models.media_items import AudioFormat, ProviderMapping, SoundEffect
from music_assistant_models.queue_item import QueueItem

from music_assistant.constants import (
    CONF_VALUE_DISABLED,
    CONF_VALUE_ENABLED,
    CONF_VOLUME_NORMALIZATION,
    CONF_VOLUME_NORMALIZATION_TARGET,
    CONF_VOLUME_NORMALIZATION_TRACKS,
)
from music_assistant.helpers.dsp import ComplexFilter
from music_assistant.helpers.tags import AudioTags
from music_assistant.helpers.tts import TTSLanguageNotSupportedError
from music_assistant.models.plugin import PluginProvider, TTSEngine
from music_assistant.providers.ai_radio.constants import (
    ATTR_HOST_ID,
    ATTR_JINGLE,
    ATTR_JINGLE_AFTER,
    ATTR_JINGLE_AFTER_MODE,
    ATTR_JINGLE_BEFORE_MODE,
    ATTR_MAX_CHARS,
    ATTR_PROMPT,
    ATTR_RENDERED_TEXT,
    ATTR_SESSION_ID,
    ATTR_SLOT_WHEN,
    ATTR_STATION_ID,
    ATTR_WEATHER_REQUIRED,
    ATTR_WEB_SEARCH_MODE,
    BREAK_MEMORY_BREAKS_INSTRUCTION,
    BREAK_MEMORY_EMPTY,
    BREAK_MEMORY_NEWS_INSTRUCTION,
    CLIP_STREAMDETAILS_EXPIRATION,
    CONF_BREAK_MEMORY,
    CONF_TTS_LOUDNESS_BOOST,
    DEFAULT_LLM_INSTRUCTIONS,
    MIN_LOUDNESS_REFERENCE_SECONDS,
    NO_WEATHER_DATA_INSTRUCTION,
    TTS_CLIP_PCM_FORMAT,
    TTS_PEAK_CEILING_DB,
    TTS_SPEECHNORM_FILTER,
)
from music_assistant.providers.ai_radio.effects import normalize_effects
from music_assistant.providers.ai_radio.memory import AIRadioMemoryMixin
from music_assistant.providers.ai_radio.models import SessionState
from music_assistant.providers.ai_radio.rendering import AIRadioRenderMixin


class DummyRenderer(AIRadioMemoryMixin, AIRadioRenderMixin):
    """Minimal harness exposing the render path."""

    domain = "ai_radio"
    instance_id = "ai_radio--test"

    def __init__(self) -> None:
        """Initialize the harness with recording stubs."""
        self.logger = logging.getLogger("tests.ai_radio.rendering")
        self._sessions: dict[str, Any] = {}
        self._hosts: dict[str, dict[str, Any]] = {}
        self.llm_prompts: list[str] = []
        self.tts_texts: list[str] = []
        self.tts_options: list[dict[str, Any] | None] = []
        self.weather_calls = 0
        self.fail_generation = False
        self.measure_calls: list[str] = []
        self.measured_loudness: float | None = None
        self.break_memory_enabled = True
        self.llm_reply = "Good evening, it is warm out."
        self.next_genres: set[str] | None = None
        self.memory_writes = 0

    def _configured_now(self) -> Any:
        return __import__("datetime").datetime(2026, 7, 30, 18, 30)

    async def _generate_text(
        self, instructions: str, prompt: str, web_mode: str, language: str | None = None
    ) -> str:
        # a real suspension point so concurrent callers actually interleave under
        # asyncio.gather, otherwise the lock in get_stream_details is never exercised
        await asyncio.sleep(0)
        if self.fail_generation:
            raise RuntimeError("llm down")
        self.llm_prompts.append(prompt)
        return self.llm_reply

    async def _write_break_memory(self) -> None:
        self.memory_writes += 1

    async def _prepare_weather_tokens(self) -> dict[str, str]:
        self.weather_calls += 1
        return {"<weather_hourly>": f"fresh weather {self.weather_calls}"}

    async def _render_tts_media(
        self,
        text: str,
        engine_uid: str | None = None,
        language: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, StreamType, AudioFormat]:
        self.tts_texts.append(text)
        self.tts_options.append(options)
        return (
            f"http://ha.invalid/api/tts_proxy/{len(self.tts_texts)}.mp3",
            StreamType.HTTP,
            AudioFormat(content_type=ContentType.MP3),
        )

    async def _probe_duration(self, path: str) -> int | None:
        return 9

    async def _measure_loudness(self, path: str) -> float | None:
        self.measure_calls.append(path)
        return self.measured_loudness


class RealTtsRenderer(DummyRenderer):
    """Harness that exercises the mixin's real TTS path instead of the DummyRenderer stub."""

    _render_tts_media = AIRadioRenderMixin._render_tts_media


def _tts_renderer(path: str, audio_format: AudioFormat | None = None) -> RealTtsRenderer:
    """Build a renderer whose TTS engine returns StreamDetails carrying the given path."""
    renderer = RealTtsRenderer()
    plugin = MagicMock(spec=PluginProvider)
    plugin.instance_id = "hass_1"
    plugin.get_tts_message = AsyncMock(
        return_value=SimpleNamespace(
            path=path, audio_format=audio_format or AudioFormat(content_type=ContentType.UNKNOWN)
        )
    )
    engine = TTSEngine(id="tts.cloud", name="Cloud", provider=plugin)
    cast("Any", renderer)._get_tts_engine = AsyncMock(return_value=engine)
    return renderer


def _clip_item(clip_id: str, queue_id: str = "player_a", **overrides: Any) -> QueueItem:
    """Build a queue item for a pending AI Radio clip."""
    attributes: dict[str, Any] = {
        ATTR_SESSION_ID: "sess",
        ATTR_STATION_ID: "st",
        ATTR_PROMPT: "It is <timestamp>. Weather: <weather_hourly>.",
        ATTR_MAX_CHARS: 300,
        ATTR_WEB_SEARCH_MODE: "disabled",
    }
    attributes.update(overrides)
    media_item = SoundEffect(
        item_id=clip_id,
        provider="ai_radio--test",
        name="Weather",
        provider_mappings={
            ProviderMapping(
                item_id=clip_id,
                provider_domain="ai_radio",
                provider_instance="ai_radio--test",
            )
        },
    )
    return QueueItem(
        queue_id=queue_id,
        queue_item_id=f"qi_{clip_id}",
        name="Weather",
        duration=None,
        media_item=media_item,
        extra_attributes=attributes,
    )


def _attach_queues(renderer: DummyRenderer, queues: dict[str, list[QueueItem]]) -> list[bool]:
    """Wire a minimal player_queues stub and return the signal_update call log."""
    signals: list[bool] = []
    cast("Any", renderer).mass = SimpleNamespace(
        player_queues=SimpleNamespace(
            all=lambda: tuple(SimpleNamespace(queue_id=queue_id) for queue_id in queues),
            items=lambda queue_id, limit=500, offset=0: queues.get(queue_id, [])[
                offset : offset + limit
            ],
            signal_update=lambda _queue_id, items_changed=False: signals.append(items_changed),
            get_next_item=lambda _queue_id, _item_id: (
                None
                if renderer.next_genres is None
                else SimpleNamespace(
                    media_item=SimpleNamespace(
                        metadata=SimpleNamespace(genres=renderer.next_genres)
                    )
                )
            ),
        ),
        metadata=SimpleNamespace(locale="en_US"),
    )
    _attach_normalization(renderer, queue_ids=tuple(queues))
    return signals


def _attach_queue(renderer: DummyRenderer, items: list[QueueItem]) -> list[bool]:
    """Wire a single-queue player_queues stub and return the signal_update call log."""
    return _attach_queues(renderer, {"player_a": items})


def _attach_normalization(
    renderer: DummyRenderer,
    *,
    enabled: bool = True,
    target: int = -14,
    boost: int = 3,
    tracks_mode: str = VolumeNormalizationMode.FALLBACK_DYNAMIC.value,
    queue_ids: tuple[str, ...] = ("player_a",),
) -> None:
    """Wire the queue, streams and provider config that decide the clip's loudness gain."""

    def queue_setting(queue_id: str, key: str, default: str) -> str:
        assert queue_id in queue_ids
        assert key == CONF_VOLUME_NORMALIZATION
        assert default == CONF_VALUE_ENABLED
        return CONF_VALUE_ENABLED if enabled else CONF_VALUE_DISABLED

    def streams_setting(key: str, **_kwargs: Any) -> str | int:
        if key == CONF_VOLUME_NORMALIZATION_TRACKS:
            return tracks_mode
        assert key == CONF_VOLUME_NORMALIZATION_TARGET
        return target

    def provider_setting(key: str) -> int | bool:
        if key == CONF_BREAK_MEMORY:
            return bool(cast("Any", renderer).break_memory_enabled)
        assert key == CONF_TTS_LOUDNESS_BOOST
        return boost

    mass = cast("Any", renderer).mass
    mass.config = SimpleNamespace(get_effective_player_queue_config_value=queue_setting)
    mass.streams = SimpleNamespace(get_config_value=streams_setting)
    cast("Any", renderer).config = SimpleNamespace(get_value=provider_setting)


async def test_render_resolves_deferred_placeholders_at_render_time() -> None:
    """The prompt sent to the LLM carries render-time weather, not plan-time."""
    renderer = DummyRenderer()
    item = _clip_item("sess_001")
    _attach_queue(renderer, [item])

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert renderer.weather_calls == 1
    assert "fresh weather 1" in renderer.llm_prompts[0]
    assert "<timestamp>" not in renderer.llm_prompts[0]
    assert streamdetails.media_type == MediaType.SOUND_EFFECT
    assert streamdetails.stream_type == StreamType.HTTP
    assert streamdetails.duration == 9
    assert streamdetails.expiration == 60
    assert streamdetails.can_seek is False
    assert streamdetails.allow_seek is False


async def test_render_caches_the_script_and_the_minted_media() -> None:
    """A second render within the cache window reuses both the stored script and media."""
    renderer = DummyRenderer()
    item = _clip_item("sess_001")
    signals = _attach_queue(renderer, [item])

    first = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)
    second = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert len(renderer.llm_prompts) == 1
    assert renderer.tts_texts == ["Good evening, it is warm out."]
    assert first.path == second.path
    assert item.extra_attributes[ATTR_RENDERED_TEXT] == "Good evening, it is warm out."
    assert signals == [True]


async def test_concurrent_renders_call_the_llm_once() -> None:
    """Two simultaneous requests for one clip render a single script."""
    renderer = DummyRenderer()
    _attach_queue(renderer, [_clip_item("sess_001")])

    await asyncio.gather(
        renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT),
        renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT),
    )

    assert len(renderer.llm_prompts) == 1


async def test_concurrent_renders_mint_the_clip_only_once() -> None:
    """Three simultaneous requests for one clip share a single minted TTS render."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")
    _attach_queue(renderer, [_clip_item("sess_001")])

    results = await asyncio.gather(
        renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT),
        renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT),
        renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT),
    )

    engine = cast("Any", renderer)._get_tts_engine.return_value
    engine.provider.get_tts_message.assert_awaited_once()
    assert len({result.path for result in results}) == 1


async def test_cached_media_remints_once_it_expires() -> None:
    """A render requested after the cache window elapses mints a fresh clip."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")
    _attach_queue(renderer, [_clip_item("sess_001")])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)
    cached = cast("Any", renderer)._media_cache["sess_001"]
    cached.minted_at -= CLIP_STREAMDETAILS_EXPIRATION + 1

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    engine = cast("Any", renderer)._get_tts_engine.return_value
    assert engine.provider.get_tts_message.await_count == 2


async def test_a_late_cache_hit_expires_with_the_url_it_serves() -> None:
    """A hit late in the window hands out the url's remaining life, not a fresh full window."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")
    _attach_queue(renderer, [_clip_item("sess_001")])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)
    cached = cast("Any", renderer)._media_cache["sess_001"]
    cached.minted_at -= 45

    late = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    engine = cast("Any", renderer)._get_tts_engine.return_value
    assert engine.provider.get_tts_message.await_count == 1
    assert 14 <= late.expiration <= 15


async def test_a_cache_hit_with_no_useful_life_left_remints() -> None:
    """A hit in the last seconds of the window mints again instead of serving a dying url."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")
    _attach_queue(renderer, [_clip_item("sess_001")])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)
    cached = cast("Any", renderer)._media_cache["sess_001"]
    cached.minted_at -= CLIP_STREAMDETAILS_EXPIRATION - 2

    fresh = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    engine = cast("Any", renderer)._get_tts_engine.return_value
    assert engine.provider.get_tts_message.await_count == 2
    assert fresh.expiration == CLIP_STREAMDETAILS_EXPIRATION


async def test_expired_cache_entries_are_pruned_on_the_next_mint() -> None:
    """Minting a clip drops the entries whose urls died, so the cache cannot grow forever."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")
    _attach_queue(renderer, [_clip_item("sess_001"), _clip_item("sess_002")])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)
    media_cache = cast("Any", renderer)._media_cache
    media_cache["sess_001"].minted_at -= CLIP_STREAMDETAILS_EXPIRATION + 1
    await renderer.get_stream_details("sess_002", MediaType.SOUND_EFFECT)

    assert set(media_cache) == {"sess_002"}


async def test_render_tts_media_passes_the_locale_as_language() -> None:
    """The DJ script's locale reaches the TTS engine as a hyphenated language code."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")
    _attach_queue(renderer, [_clip_item("sess_001")])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    engine = cast("Any", renderer)._get_tts_engine.return_value
    engine.provider.get_tts_message.assert_awaited_once_with(
        "Good evening, it is warm out.", language="en-US", engine_id="tts.cloud", options={}
    )


async def test_render_tts_media_falls_back_without_language_on_rejection() -> None:
    """An engine that rejects the requested language is retried once without it."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")
    _attach_queue(renderer, [_clip_item("sess_001")])
    engine = cast("Any", renderer)._get_tts_engine.return_value
    engine.provider.get_tts_message = AsyncMock(
        side_effect=[
            TTSLanguageNotSupportedError(
                "TTS engine 'tts.cloud' does not support language 'en-US'"
            ),
            SimpleNamespace(
                path="http://example.test/api/tts_proxy/abc123.mp3",
                audio_format=AudioFormat(content_type=ContentType.MP3),
            ),
        ]
    )

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.path == "http://example.test/api/tts_proxy/abc123.mp3"
    assert engine.provider.get_tts_message.await_count == 2
    first_call, second_call = engine.provider.get_tts_message.await_args_list
    assert first_call.kwargs["language"] == "en-US"
    assert second_call.kwargs["language"] is None


async def test_render_tts_media_does_not_retry_after_a_timeout_style_failure() -> None:
    """A structured MusicAssistantError is not a language rejection, so it skips the retry."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")
    _attach_queue(renderer, [_clip_item("sess_001")])
    engine = cast("Any", renderer)._get_tts_engine.return_value
    engine.provider.get_tts_message = AsyncMock(
        side_effect=MusicAssistantError("engine did not respond within 5s")
    )

    with pytest.raises(MediaNotFoundError):
        await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    engine.provider.get_tts_message.assert_awaited_once()


async def test_clip_is_found_in_the_owning_sessions_queue() -> None:
    """The session registry points the lookup at the queue that holds the clip."""
    renderer = DummyRenderer()
    session = SessionState(session_id="sess", station_id="st", queue_id="player_b")
    renderer._sessions = {"sess": session}
    _attach_queues(
        renderer,
        {"player_a": [], "player_b": [_clip_item("sess_001", queue_id="player_b")]},
    )

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.item_id == "sess_001"


async def test_clip_is_found_by_scanning_every_queue_after_a_restart() -> None:
    """With the session registry gone, the clip is still located in its persisted queue."""
    renderer = DummyRenderer()
    _attach_queues(
        renderer,
        {"player_a": [], "player_b": [_clip_item("sess_001", queue_id="player_b")]},
    )

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.item_id == "sess_001"


async def test_unknown_clip_raises_media_not_found() -> None:
    """A clip id that is not in any queue is reported as missing media."""
    renderer = DummyRenderer()
    _attach_queue(renderer, [_clip_item("sess_001")])

    with pytest.raises(MediaNotFoundError):
        await renderer.get_stream_details("sess_999", MediaType.SOUND_EFFECT)


async def test_clip_without_prompt_raises_media_not_found() -> None:
    """A clip whose attributes were lost is reported as missing media."""
    renderer = DummyRenderer()
    item = _clip_item("sess_001")
    item.extra_attributes.pop(ATTR_PROMPT)
    _attach_queue(renderer, [item])

    with pytest.raises(MediaNotFoundError):
        await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)


async def test_llm_failure_raises_media_not_found() -> None:
    """An LLM failure surfaces as missing media so the core skips the clip."""
    renderer = DummyRenderer()
    renderer.fail_generation = True
    _attach_queue(renderer, [_clip_item("sess_001")])

    with pytest.raises(MediaNotFoundError):
        await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)


async def test_render_failure_increments_the_session_skip_counter() -> None:
    """A skipped clip is recorded on its session without failing the run."""
    renderer = DummyRenderer()
    session = SessionState(session_id="sess", station_id="st")
    renderer._sessions = {"sess": session}
    _attach_queue(renderer, [_clip_item("sess_001")])
    renderer.fail_generation = True

    with pytest.raises(MediaNotFoundError):
        await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert session.skipped_sections == 1
    assert session.last_render_error


async def test_tts_failure_raises_media_not_found_and_records_skip() -> None:
    """A TTS failure surfaces as missing media and is recorded on the owning session."""

    class UnspeakableRenderer(DummyRenderer):
        async def _render_tts_media(
            self,
            text: str,
            engine_uid: str | None = None,
            language: str | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[str, StreamType, AudioFormat]:
            raise RuntimeError("tts down")

    renderer = UnspeakableRenderer()
    session = SessionState(session_id="sess", station_id="st")
    renderer._sessions = {"sess": session}
    _attach_queue(renderer, [_clip_item("sess_001")])

    with pytest.raises(MediaNotFoundError):
        await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert session.skipped_sections == 1
    assert session.last_render_error


class NoWeatherRenderer(DummyRenderer):
    """Renderer whose weather fetch always fails, leaving deferred weather tokens empty."""

    async def _prepare_weather_tokens(self) -> dict[str, str]:
        self.weather_calls += 1
        return {}


async def test_weather_required_clip_is_skipped_when_weather_is_unavailable() -> None:
    """A weather-required clip with no forecast data is skipped instead of airing a guess."""
    renderer = NoWeatherRenderer()
    session = SessionState(session_id="sess", station_id="st")
    renderer._sessions = {"sess": session}
    item = _clip_item("sess_001")
    item.extra_attributes[ATTR_WEATHER_REQUIRED] = True
    _attach_queue(renderer, [item])

    with pytest.raises(MediaNotFoundError):
        await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert session.skipped_sections == 1
    assert session.last_render_error
    assert renderer.llm_prompts == []


async def test_non_weather_required_clip_renders_with_no_data_instruction() -> None:
    """A non-weather-required clip still airs, told to leave the forecast out rather than guess."""
    renderer = NoWeatherRenderer()
    item = _clip_item("sess_001")
    item.extra_attributes[ATTR_WEATHER_REQUIRED] = False
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert NO_WEATHER_DATA_INSTRUCTION in renderer.llm_prompts[0]
    assert "<weather_hourly>" not in renderer.llm_prompts[0]


async def test_missing_weather_required_attribute_defaults_to_not_required() -> None:
    """An older queue item with no weather_required attribute must not be treated as required."""
    renderer = NoWeatherRenderer()
    item = _clip_item("sess_001")
    assert ATTR_WEATHER_REQUIRED not in item.extra_attributes
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert NO_WEATHER_DATA_INSTRUCTION in renderer.llm_prompts[0]


async def test_probe_failure_is_not_fatal() -> None:
    """A failed duration probe yields streamdetails without a duration."""

    class UnprobableRenderer(DummyRenderer):
        async def _probe_duration(self, path: str) -> int | None:
            return None

    renderer = UnprobableRenderer()
    _attach_queue(renderer, [_clip_item("sess_001")])

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.duration is None


class RealProbeRenderer(DummyRenderer):
    """Harness that exercises the mixin's real duration probe instead of the stub."""

    _probe_duration = AIRadioRenderMixin._probe_duration


def _failing_probe(message: str, monkeypatch: pytest.MonkeyPatch) -> RealProbeRenderer:
    """Build a renderer whose duration probe fails with the given ffprobe message."""

    async def _raise(*_args: Any, **_kwargs: Any) -> AudioTags:
        raise InvalidDataError(message)

    monkeypatch.setattr("music_assistant.providers.ai_radio.rendering.async_parse_tags", _raise)
    renderer = RealProbeRenderer()
    renderer._sessions = {"sess": SessionState(session_id="sess", station_id="st")}
    _attach_queue(renderer, [_clip_item("sess_001")])
    return renderer


@pytest.mark.parametrize(
    "server_error",
    ["Server returned 5XX Server Error reply", "HTTP error 500 Internal Server Error"],
)
async def test_tts_server_error_fails_the_clip_with_an_actionable_message(
    server_error: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine that hands out a URL it cannot render fails the clip, not the playback."""
    renderer = _failing_probe(
        f"Unable to retrieve info for http://ha.invalid/api/tts_proxy/1.mp3 ({server_error})",
        monkeypatch,
    )

    with pytest.raises(MediaNotFoundError):
        await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    session = renderer._sessions["sess"]
    assert session.skipped_sections == 1
    assert "Check the logs of the TTS engine" in session.last_render_error
    # the hint is a guess, so the whole probe message travels with it - the url included,
    # since that is what tells a failing engine apart from a failing tts server behind it
    assert "http://ha.invalid/api/tts_proxy/1.mp3" in session.last_render_error
    assert server_error in session.last_render_error


async def test_unmeasurable_clip_still_plays(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe that only fails to measure the audio leaves the clip playable."""
    renderer = _failing_probe(
        "Unable to retrieve info for http://ha.invalid/api/tts_proxy/1.mp3 "
        "(Invalid or unsupported media file)",
        monkeypatch,
    )

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.duration is None
    assert renderer._sessions["sess"].skipped_sections == 0


async def test_generate_script_uses_host_instructions() -> None:
    """The prompt sent to the LLM carries the resolved host's persona instructions."""
    renderer = DummyRenderer()
    renderer._hosts = {"rick": {"id": "rick", "instructions": "Persona text.", "tts_engine": ""}}
    captured: dict[str, str] = {}

    async def fake_generate_text(
        instructions: str,
        prompt: str,  # noqa: ARG001
        web_mode: str,  # noqa: ARG001
        language: str | None = None,  # noqa: ARG001
    ) -> str:
        captured["instructions"] = instructions
        return "script"

    cast("Any", renderer)._generate_text = fake_generate_text
    item = _clip_item("sess_001", **{ATTR_HOST_ID: "rick", ATTR_PROMPT: "p"})

    text = await renderer._generate_script(item, "p", "clip_1")

    assert text == "script"
    assert captured["instructions"] == "Persona text."


async def test_generate_script_falls_back_to_default_instructions() -> None:
    """A clip whose host is gone by render time still generates, using the default persona."""
    renderer = DummyRenderer()
    renderer._hosts = {}
    captured: dict[str, str] = {}

    async def fake_generate_text(
        instructions: str,
        prompt: str,  # noqa: ARG001
        web_mode: str,  # noqa: ARG001
        language: str | None = None,  # noqa: ARG001
    ) -> str:
        # mirrors the empty-to-default fallback the real _generate_text applies (runtime.py)
        captured["instructions"] = instructions.strip() or DEFAULT_LLM_INSTRUCTIONS
        return "script"

    cast("Any", renderer)._generate_text = fake_generate_text
    item = _clip_item("sess_001", **{ATTR_HOST_ID: "gone", ATTR_PROMPT: "p"})

    await renderer._generate_script(item, "p", "clip_1")

    assert captured["instructions"] == DEFAULT_LLM_INSTRUCTIONS


async def test_generate_script_forwards_the_hosts_language() -> None:
    """The host's configured language reaches _generate_text, ready to override the locale."""
    renderer = DummyRenderer()
    renderer._hosts = {
        "rick": {
            "id": "rick",
            "instructions": "Persona text.",
            "tts_engine": "",
            "language": "fr_FR",
        }
    }
    captured: dict[str, str | None] = {}

    async def fake_generate_text(
        instructions: str,  # noqa: ARG001
        prompt: str,  # noqa: ARG001
        web_mode: str,  # noqa: ARG001
        language: str | None = None,
    ) -> str:
        captured["language"] = language
        return "script"

    cast("Any", renderer)._generate_text = fake_generate_text
    item = _clip_item("sess_001", **{ATTR_HOST_ID: "rick", ATTR_PROMPT: "p"})

    await renderer._generate_script(item, "p", "clip_1")

    assert captured["language"] == "fr_FR"


async def test_generate_script_forwards_empty_language_when_host_has_none() -> None:
    """A host with no configured language forwards an empty string, not None."""
    renderer = DummyRenderer()
    renderer._hosts = {"rick": {"id": "rick", "instructions": "Persona text.", "tts_engine": ""}}
    captured: dict[str, str | None] = {}

    async def fake_generate_text(
        instructions: str,  # noqa: ARG001
        prompt: str,  # noqa: ARG001
        web_mode: str,  # noqa: ARG001
        language: str | None = None,
    ) -> str:
        captured["language"] = language
        return "script"

    cast("Any", renderer)._generate_text = fake_generate_text
    item = _clip_item("sess_001", **{ATTR_HOST_ID: "rick", ATTR_PROMPT: "p"})

    await renderer._generate_script(item, "p", "clip_1")

    assert captured["language"] == ""


async def test_render_tts_media_prefers_the_hosts_language_over_the_locale() -> None:
    """A host's configured language reaches the TTS engine, overriding the server locale."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")
    renderer._hosts = {"rick": {"id": "rick", "tts_engine": "", "language": "fr_FR"}}
    _attach_queue(renderer, [_clip_item("sess_001", **{ATTR_HOST_ID: "rick"})])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    engine = cast("Any", renderer)._get_tts_engine.return_value
    engine.provider.get_tts_message.assert_awaited_once_with(
        "Good evening, it is warm out.", language="fr-FR", engine_id="tts.cloud", options={}
    )


async def test_render_tts_media_falls_back_to_locale_when_host_language_is_empty() -> None:
    """A host with no configured language falls back to the server locale for the TTS call."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")
    renderer._hosts = {"rick": {"id": "rick", "tts_engine": ""}}
    _attach_queue(renderer, [_clip_item("sess_001", **{ATTR_HOST_ID: "rick"})])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    engine = cast("Any", renderer)._get_tts_engine.return_value
    engine.provider.get_tts_message.assert_awaited_once_with(
        "Good evening, it is warm out.", language="en-US", engine_id="tts.cloud", options={}
    )


async def test_resolve_deferred_placeholders_skips_weather_without_token() -> None:
    """A prompt with no weather placeholder never triggers a weather fetch."""
    renderer = DummyRenderer()

    values = await renderer._resolve_deferred_placeholders("Just plain text, no tokens.")

    assert renderer.weather_calls == 0
    assert values["<weather_hourly>"] == ""
    assert values["<weather_daily>"] == ""


async def test_resolve_deferred_placeholders_timestamp_spells_out_weekday() -> None:
    """The <timestamp> value names the weekday so the LLM never has to derive it."""
    renderer = DummyRenderer()

    values = await renderer._resolve_deferred_placeholders("Just plain text, no tokens.")

    assert "Thursday" in values["<timestamp>"]
    assert "July" in values["<timestamp>"]


async def test_mint_clip_media_resolves_host_tts_engine() -> None:
    """A clip whose host declares a tts_engine reaches _get_tts_engine with that override."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")
    renderer._hosts = {"rick": {"id": "rick", "tts_engine": "tts.rick_voice"}}
    cast("Any", renderer).mass = SimpleNamespace(metadata=SimpleNamespace(locale="en_US"))
    _attach_normalization(renderer)
    item = _clip_item("sess_001", **{ATTR_HOST_ID: "rick"})

    await renderer._mint_clip_media(item, "hello world", "clip_1")

    cast("Any", renderer)._get_tts_engine.assert_awaited_once_with("tts.rick_voice")


async def test_mint_clip_media_forwards_the_hosts_options() -> None:
    """A host's configured TTS options are forwarded into the render call."""
    renderer = DummyRenderer()
    renderer._hosts = {
        "rick": {
            "id": "rick",
            "tts_engine": "",
            "options": {"voice": "en_US-lessac-medium", "length_scale": 1.2},
        }
    }
    cast("Any", renderer).mass = SimpleNamespace(metadata=SimpleNamespace(locale="en_US"))
    _attach_normalization(renderer)
    item = _clip_item("sess_001", **{ATTR_HOST_ID: "rick"})

    await renderer._mint_clip_media(item, "hello world", "clip_1")

    assert renderer.tts_options == [{"voice": "en_US-lessac-medium", "length_scale": 1.2}]


async def test_mint_clip_media_sends_no_options_for_a_host_without_any() -> None:
    """A host with no configured options forwards an empty dict, not None."""
    renderer = DummyRenderer()
    renderer._hosts = {"rick": {"id": "rick", "tts_engine": ""}}
    cast("Any", renderer).mass = SimpleNamespace(metadata=SimpleNamespace(locale="en_US"))
    _attach_normalization(renderer)
    item = _clip_item("sess_001", **{ATTR_HOST_ID: "rick"})

    await renderer._mint_clip_media(item, "hello world", "clip_1")

    assert renderer.tts_options == [{}]


async def test_render_tts_media_forwards_the_hosts_tts_options() -> None:
    """A host's configured TTS options reach the engine's get_tts_message call."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")
    renderer._hosts = {
        "rick": {
            "id": "rick",
            "tts_engine": "",
            "options": {"voice": "en_US-lessac-medium", "length_scale": 1.2},
        }
    }
    _attach_queue(renderer, [_clip_item("sess_001", **{ATTR_HOST_ID: "rick"})])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    engine = cast("Any", renderer)._get_tts_engine.return_value
    engine.provider.get_tts_message.assert_awaited_once_with(
        "Good evening, it is warm out.",
        language="en-US",
        engine_id="tts.cloud",
        options={"voice": "en_US-lessac-medium", "length_scale": 1.2},
    )


async def test_render_tts_media_streams_a_url_over_http() -> None:
    """A TTS engine returning a proxy URL yields an HTTP stream with the MP3 default format."""
    renderer = _tts_renderer("http://example.test/api/tts_proxy/abc123.mp3")

    path, stream_type, audio_format = await renderer._render_tts_media("hello world")

    assert path == "http://example.test/api/tts_proxy/abc123.mp3"
    assert stream_type == StreamType.HTTP
    assert audio_format.content_type == ContentType.MP3
    engine = cast("Any", renderer)._get_tts_engine.return_value
    # the provider-scoped engine.id, never engine.uid, and never omitted
    engine.provider.get_tts_message.assert_awaited_once_with(
        "hello world", language=None, engine_id="tts.cloud", options=None
    )


async def test_render_tts_media_streams_a_local_file_from_disk(tmp_path: Path) -> None:
    """A TTS engine that renders to disk is played as a local file rather than fetched."""
    clip = tmp_path / "section.mp3"
    clip.write_bytes(b"")
    renderer = _tts_renderer(str(clip))
    _attach_queue(renderer, [_clip_item("sess_001")])

    path, stream_type, _ = await renderer._render_tts_media("hello world")

    assert path == str(clip)
    assert stream_type == StreamType.LOCAL_FILE


async def test_render_tts_media_keeps_a_declared_audio_format(tmp_path: Path) -> None:
    """A TTS engine that declares its own format has it carried into the clip streamdetails."""
    clip = tmp_path / "section.wav"
    clip.write_bytes(b"")
    renderer = _tts_renderer(str(clip), AudioFormat(content_type=ContentType.WAV))

    _, _, audio_format = await renderer._render_tts_media("hello world")

    assert audio_format.content_type == ContentType.WAV


@pytest.mark.parametrize("path", ["", "section.mp3", "/does/not/exist.mp3"])
async def test_render_tts_media_rejects_an_unplayable_path(path: str) -> None:
    """A path that is neither a URL nor an existing file fails loudly instead of degrading."""
    renderer = _tts_renderer(path)

    with pytest.raises(InvalidDataError, match="unusable stream path"):
        await renderer._render_tts_media("hello world")


async def test_render_tts_media_gives_up_on_a_stalled_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled TTS engine fails the clip instead of pinning the render path."""
    monkeypatch.setattr("music_assistant.helpers.tts.TTS_QUERY_TIMEOUT_SECONDS", 0.01)

    async def _answers_too_late(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        await asyncio.sleep(5)
        return SimpleNamespace(
            path="http://example.test/late.mp3",
            audio_format=AudioFormat(content_type=ContentType.MP3),
        )

    renderer = _tts_renderer("http://example.test/late.mp3")
    engine = cast("Any", renderer)._get_tts_engine.return_value
    engine.provider.get_tts_message = AsyncMock(side_effect=_answers_too_late)

    with pytest.raises(MusicAssistantError, match="did not respond within"):
        await renderer._render_tts_media("hello world")


async def test_render_tts_media_reports_an_engine_side_timeout_as_is() -> None:
    """A timeout raised by the TTS engine itself is not reported as our own cap."""
    renderer = _tts_renderer("http://example.test/late.mp3")
    engine = cast("Any", renderer)._get_tts_engine.return_value
    engine.provider.get_tts_message = AsyncMock(side_effect=TimeoutError)

    with pytest.raises(TimeoutError) as error:
        await renderer._render_tts_media("hello world")
    assert "did not respond within" not in str(error.value)


async def test_local_file_clip_yields_local_file_streamdetails(tmp_path: Path) -> None:
    """A disk-rendered clip reaches the core as LOCAL_FILE streamdetails."""
    clip = tmp_path / "section.mp3"
    clip.write_bytes(b"")
    renderer = _tts_renderer(str(clip))
    _attach_queue(renderer, [_clip_item("sess_001")])

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.stream_type == StreamType.LOCAL_FILE
    assert streamdetails.path == str(clip)


async def test_a_measured_clip_is_lifted_to_the_target_plus_the_boost() -> None:
    """A clip quieter than the levelled music is served through the provider's own chain."""
    renderer = DummyRenderer()
    renderer.measured_loudness = -18.0
    _attach_queue(renderer, [_clip_item("sess_001")])
    _attach_normalization(renderer, target=-14, boost=3)

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.stream_type == StreamType.CUSTOM
    assert streamdetails.decoded_audio_format == TTS_CLIP_PCM_FORMAT
    assert streamdetails.audio_format.content_type == ContentType.MP3
    assert streamdetails.data.gain_db == pytest.approx(7.0)


async def test_the_clip_is_evened_out_and_lifted_before_it_is_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The filter chain reaches ffmpeg as speechnorm, then gain, then the peak limiter."""
    captured: dict[str, Any] = {}

    async def fake_ffmpeg_stream(**kwargs: Any) -> AsyncGenerator[bytes]:
        captured.update(kwargs)
        yield b"pcm"

    monkeypatch.setattr(
        "music_assistant.providers.ai_radio.rendering.get_ffmpeg_stream", fake_ffmpeg_stream
    )
    renderer = DummyRenderer()
    renderer.measured_loudness = -18.0
    _attach_queue(renderer, [_clip_item("sess_001")])
    _attach_normalization(renderer, target=-14, boost=3)
    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    chunks = [chunk async for chunk in renderer.get_audio_stream(streamdetails)]

    assert chunks == [b"pcm"]
    assert captured["audio_input"] == streamdetails.path
    assert captured["input_format"].content_type == ContentType.MP3
    assert captured["output_format"] == TTS_CLIP_PCM_FORMAT
    assert captured["filter_params"] == [
        TTS_SPEECHNORM_FILTER,
        "volume=7.0dB",
        f"alimiter=limit={TTS_PEAK_CEILING_DB}dB:level=false:latency=true",
    ]


async def test_the_engine_reference_is_measured_once_and_reused() -> None:
    """A second clip from the same voice levels itself against the stored measurement."""
    renderer = DummyRenderer()
    renderer.measured_loudness = -18.0
    _attach_queue(renderer, [_clip_item("sess_001"), _clip_item("sess_002")])
    _attach_normalization(renderer)

    first = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)
    second = await renderer.get_stream_details("sess_002", MediaType.SOUND_EFFECT)

    assert len(renderer.measure_calls) == 1
    assert first.data.gain_db == second.data.gain_db


async def test_a_short_clip_levels_itself_but_never_becomes_the_reference() -> None:
    """A clip of a few words is too thin a sample to speak for the rest of the engine."""

    class BriefRenderer(DummyRenderer):
        async def _probe_duration(self, path: str) -> int | None:
            return MIN_LOUDNESS_REFERENCE_SECONDS - 1

    renderer = BriefRenderer()
    renderer.measured_loudness = -18.0
    _attach_queue(renderer, [_clip_item("sess_001"), _clip_item("sess_002")])
    _attach_normalization(renderer)

    first = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)
    await renderer.get_stream_details("sess_002", MediaType.SOUND_EFFECT)

    assert first.stream_type == StreamType.CUSTOM
    assert len(renderer.measure_calls) == 2
    assert cast("Any", renderer)._engine_loudness == {}


async def test_clip_plays_untouched_when_the_queue_does_not_normalize() -> None:
    """With the music unlevelled there is nothing to match, so the clip airs as rendered."""
    renderer = DummyRenderer()
    renderer.measured_loudness = -18.0
    _attach_queue(renderer, [_clip_item("sess_001")])
    _attach_normalization(renderer, enabled=False)

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.stream_type == StreamType.HTTP
    assert streamdetails.decoded_audio_format is None
    assert streamdetails.data is None


async def test_clip_plays_untouched_when_it_could_not_be_measured() -> None:
    """A failed measurement leaves the clip playable at its own level."""
    renderer = DummyRenderer()
    _attach_queue(renderer, [_clip_item("sess_001")])
    _attach_normalization(renderer)

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert renderer.measure_calls
    assert streamdetails.stream_type == StreamType.HTTP
    assert streamdetails.data is None


async def test_a_clip_above_the_wanted_level_is_trimmed_back_down() -> None:
    """The trim runs in either direction, so a loud voice is brought down to the target."""
    renderer = DummyRenderer()
    renderer.measured_loudness = -8.0
    _attach_queue(renderer, [_clip_item("sess_001")])
    _attach_normalization(renderer, target=-14, boost=3)

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.stream_type == StreamType.CUSTOM
    assert streamdetails.data.gain_db == pytest.approx(-3.0)


async def test_the_levelled_clip_does_not_hand_out_the_shared_pcm_format() -> None:
    """Core writes what ffmpeg reports onto this format, so it may not be the shared one."""
    renderer = DummyRenderer()
    renderer.measured_loudness = -18.0
    _attach_queue(renderer, [_clip_item("sess_001")])
    _attach_normalization(renderer)

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.decoded_audio_format == TTS_CLIP_PCM_FORMAT
    assert streamdetails.decoded_audio_format is not TTS_CLIP_PCM_FORMAT


# verbatim ffmpeg 7.1 output, so the parsing this depends on is covered for real
FFMPEG_LOUDNORM_OUTPUT = b"""[Parsed_loudnorm_0 @ 0x93b41d440] \n{
\t"input_i" : "-18.37",
\t"input_tp" : "-1.89",
\t"input_lra" : "0.40",
\t"input_thresh" : "-27.86",
\t"normalization_type" : "dynamic"
}
"""


async def test_the_measurement_reads_the_level_out_of_ffmpegs_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine reference comes from loudnorm's report on the rendered clip."""
    captured: dict[str, Any] = {}

    async def fake_check_output(*args: str, **_kwargs: Any) -> tuple[int, bytes]:
        captured["args"] = args
        return 0, FFMPEG_LOUDNORM_OUTPUT

    monkeypatch.setattr(
        "music_assistant.providers.ai_radio.rendering.check_output", fake_check_output
    )
    renderer = DummyRenderer()

    loudness = await AIRadioRenderMixin._measure_loudness(renderer, "http://ha.invalid/clip.mp3")

    assert loudness == -18.37
    assert "http://ha.invalid/clip.mp3" in captured["args"]
    # the reading has to come from behind speechnorm, or the gain corrects for a level
    # that never reaches it
    assert f"{TTS_SPEECHNORM_FILTER},loudnorm=print_format=json" in captured["args"]


async def test_a_failed_measurement_leaves_the_level_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ffmpeg run that did not succeed yields no reference rather than a wrong one."""

    async def fake_check_output(*_args: str, **_kwargs: Any) -> tuple[int, bytes]:
        return 1, b"ffmpeg: Invalid data found when processing input"

    monkeypatch.setattr(
        "music_assistant.providers.ai_radio.rendering.check_output", fake_check_output
    )
    renderer = DummyRenderer()

    assert (
        await AIRadioRenderMixin._measure_loudness(renderer, "http://ha.invalid/clip.mp3") is None
    )


async def test_clip_plays_untouched_when_tracks_are_not_normalized() -> None:
    """The queue switch alone does not mean the music around the clip is levelled."""
    renderer = DummyRenderer()
    renderer.measured_loudness = -18.0
    _attach_queue(renderer, [_clip_item("sess_001")])
    _attach_normalization(renderer, tracks_mode=VolumeNormalizationMode.DISABLED.value)

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.stream_type == StreamType.HTTP
    assert streamdetails.data is None


async def test_no_measurement_is_taken_when_the_reading_has_nowhere_to_go() -> None:
    """Measuring costs a fetch and a decode, so a queue that will not use it is not charged."""
    renderer = DummyRenderer()
    renderer.measured_loudness = -18.0
    _attach_queue(renderer, [_clip_item("sess_001")])
    _attach_normalization(renderer, enabled=False)

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert renderer.measure_calls == []


async def test_a_rendered_break_is_shown_to_the_hosts_next_break() -> None:
    """What a host aired reaches its next prompt, so it can avoid repeating itself."""
    renderer = DummyRenderer()
    first = _clip_item("sess_001", **{ATTR_HOST_ID: "mika"})
    second = _clip_item("sess_002", **{ATTR_HOST_ID: "mika"})
    _attach_queue(renderer, [first, second])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)
    await renderer.get_stream_details("sess_002", MediaType.SOUND_EFFECT)

    assert BREAK_MEMORY_BREAKS_INSTRUCTION not in renderer.llm_prompts[0]
    assert BREAK_MEMORY_BREAKS_INSTRUCTION in renderer.llm_prompts[1]
    assert "(Weather): Good evening, it is warm out." in renderer.llm_prompts[1]
    assert BREAK_MEMORY_NEWS_INSTRUCTION not in renderer.llm_prompts[1]
    assert renderer.memory_writes == 2


async def test_break_memory_is_kept_per_host() -> None:
    """A host never sees the breaks another host aired."""
    renderer = DummyRenderer()
    _attach_queue(
        renderer,
        [
            _clip_item("sess_001", **{ATTR_HOST_ID: "mika"}),
            _clip_item("sess_002", **{ATTR_HOST_ID: "night_owl"}),
        ],
    )

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)
    await renderer.get_stream_details("sess_002", MediaType.SOUND_EFFECT)

    assert BREAK_MEMORY_BREAKS_INSTRUCTION not in renderer.llm_prompts[1]


async def test_a_news_break_sees_the_news_already_reported() -> None:
    """A forced web search marks the news, and only the next news break is handed it."""
    renderer = DummyRenderer()
    news = {ATTR_HOST_ID: "mika", ATTR_WEB_SEARCH_MODE: "force"}
    _attach_queue(
        renderer,
        [
            _clip_item("sess_001", **news),
            _clip_item("sess_002", **news),
            _clip_item("sess_003", **{ATTR_HOST_ID: "mika"}),
        ],
    )

    for clip_id in ("sess_001", "sess_002", "sess_003"):
        await renderer.get_stream_details(clip_id, MediaType.SOUND_EFFECT)

    assert BREAK_MEMORY_NEWS_INSTRUCTION in renderer.llm_prompts[1]
    # the news is not a break to vary on, and a plain break has no use for the news
    assert BREAK_MEMORY_BREAKS_INSTRUCTION not in renderer.llm_prompts[1]
    assert BREAK_MEMORY_NEWS_INSTRUCTION not in renderer.llm_prompts[2]
    assert [entry["news"] for entry in renderer._break_memory["mika"]] == [True, True, False]


async def test_placed_memory_placeholders_are_filled_in_place() -> None:
    """A prompt that places the memory itself gets exactly that, and nothing is appended."""
    renderer = DummyRenderer()
    placed = _clip_item(
        "sess_002",
        **{ATTR_HOST_ID: "mika", ATTR_PROMPT: "Said: <recent_breaks> | News: <recent_news>"},
    )
    _attach_queue(renderer, [_clip_item("sess_001", **{ATTR_HOST_ID: "mika"}), placed])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)
    await renderer.get_stream_details("sess_002", MediaType.SOUND_EFFECT)

    prompt = renderer.llm_prompts[1]
    assert prompt.startswith("Said: - ")
    assert "Good evening, it is warm out." in prompt
    assert prompt.endswith(f"| News: {BREAK_MEMORY_EMPTY}")
    assert BREAK_MEMORY_BREAKS_INSTRUCTION not in prompt
    # asking for the news it already told makes the break a news break
    assert renderer._break_memory["mika"][-1]["news"] is True


async def test_disabled_break_memory_still_fills_placed_placeholders() -> None:
    """Turning the memory off stops the automatic reminder, not a placeholder placed on purpose."""
    renderer = DummyRenderer()
    renderer.break_memory_enabled = False
    _attach_queue(
        renderer,
        [
            _clip_item("sess_001", **{ATTR_HOST_ID: "mika"}),
            _clip_item("sess_002", **{ATTR_HOST_ID: "mika"}),
            _clip_item("sess_003", **{ATTR_HOST_ID: "mika", ATTR_PROMPT: "<recent_breaks>"}),
        ],
    )

    for clip_id in ("sess_001", "sess_002", "sess_003"):
        await renderer.get_stream_details(clip_id, MediaType.SOUND_EFFECT)

    assert BREAK_MEMORY_BREAKS_INSTRUCTION not in renderer.llm_prompts[1]
    assert "Good evening, it is warm out." in renderer.llm_prompts[2]


async def test_a_clip_without_a_host_is_not_remembered() -> None:
    """Memory is kept per host, so a clip nobody speaks leaves none behind."""
    renderer = DummyRenderer()
    _attach_queue(renderer, [_clip_item("sess_001")])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert renderer.memory_writes == 0
    assert not getattr(renderer, "_break_memory", {})


async def test_a_failed_generation_is_not_remembered() -> None:
    """Only a script that airs becomes part of the memory."""
    renderer = DummyRenderer()
    renderer.fail_generation = True
    _attach_queue(renderer, [_clip_item("sess_001", **{ATTR_HOST_ID: "mika"})])

    with pytest.raises(MediaNotFoundError):
        await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert renderer.memory_writes == 0


def _host_with_effects(renderer: DummyRenderer, **effects: Any) -> None:
    """Give the renderer a host 'mika' dressing its breaks with the given sounds."""
    renderer._hosts["mika"] = {"effects": normalize_effects(effects)}
    cast("Any", renderer)._measure_effect_loudness = AsyncMock(return_value=-20.0)


def _stub_sound_tags(monkeypatch: pytest.MonkeyPatch, seconds: float = 2.4) -> list[str]:
    """Make every sound probe report the given length, and return the probed paths."""
    probed: list[str] = []

    async def fake_parse_tags(path: str, **_kwargs: Any) -> Any:
        probed.append(path)
        return SimpleNamespace(duration=seconds)

    monkeypatch.setattr(
        "music_assistant.providers.ai_radio.rendering.async_parse_tags", fake_parse_tags
    )
    return probed


async def test_a_news_break_opens_with_the_hosts_news_jingle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The news jingle is levelled to the voice, and core learns the longer duration."""
    probed = _stub_sound_tags(monkeypatch)
    renderer = DummyRenderer()
    renderer.measured_loudness = -18.0
    _host_with_effects(renderer, news_jingle="/media/news.mp3", show_jingle="/media/ident.mp3")
    _attach_queue(
        renderer,
        [_clip_item("sess_001", **{ATTR_HOST_ID: "mika", ATTR_WEB_SEARCH_MODE: "force"})],
    )
    _attach_normalization(renderer, target=-14, boost=3)

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    effects = streamdetails.data.effects
    assert probed == ["/media/news.mp3"]
    assert effects.bed is None
    # the voice airs at -14 + 3, so the jingle measured at -20 is lifted by 9 dB
    assert effects.jingle.gain_db == pytest.approx(9.0)
    assert streamdetails.duration == 9 + 2


async def test_the_first_break_of_a_show_opens_with_the_ident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A show's intro carries the show jingle, and the bed plays under it too."""
    _stub_sound_tags(monkeypatch)
    renderer = DummyRenderer()
    renderer.measured_loudness = -18.0
    _host_with_effects(
        renderer, show_jingle="/media/ident.mp3", music_bed="/media/bed.mp3", music_bed_level=-20
    )
    _attach_queue(
        renderer,
        [_clip_item("sess_001", **{ATTR_HOST_ID: "mika", ATTR_SLOT_WHEN: "start_of_playlist"})],
    )
    _attach_normalization(renderer, target=-14, boost=3)

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    effects = streamdetails.data.effects
    assert effects.jingle.path == "/media/ident.mp3"
    assert effects.bed.path == "/media/bed.mp3"
    assert effects.bed.gain_db == pytest.approx(-11.0)


async def test_a_sound_that_cannot_be_read_is_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing file costs the effect, never the break."""

    async def broken_parse_tags(path: str, **_kwargs: Any) -> Any:
        raise InvalidDataError(f"{path} not found")

    monkeypatch.setattr(
        "music_assistant.providers.ai_radio.rendering.async_parse_tags", broken_parse_tags
    )
    renderer = DummyRenderer()
    renderer.measured_loudness = -18.0
    _host_with_effects(renderer, news_jingle="/media/missing.mp3")
    _attach_queue(
        renderer,
        [_clip_item("sess_001", **{ATTR_HOST_ID: "mika", ATTR_WEB_SEARCH_MODE: "force"})],
    )

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.data.effects is None
    assert streamdetails.duration == 9


async def test_effects_play_even_when_the_queue_does_not_normalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The voice stays untouched, and the sounds are levelled against its own reading."""
    _stub_sound_tags(monkeypatch)
    renderer = DummyRenderer()
    renderer.measured_loudness = -18.0
    _host_with_effects(renderer, music_bed="/media/bed.mp3", music_bed_level=-20)
    _attach_queue(renderer, [_clip_item("sess_001", **{ATTR_HOST_ID: "mika"})])
    _attach_normalization(renderer, enabled=False)

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert streamdetails.stream_type == StreamType.CUSTOM
    assert streamdetails.data.gain_db is None
    assert streamdetails.data.effects.bed.gain_db == pytest.approx(-18.0)


async def test_the_dressed_clip_keeps_the_limiter_last(monkeypatch: pytest.MonkeyPatch) -> None:
    """Voice levelling comes first, the effects after it and the limiter at the very end."""
    _stub_sound_tags(monkeypatch)
    captured: dict[str, Any] = {}

    async def fake_ffmpeg_stream(**kwargs: Any) -> AsyncGenerator[bytes]:
        captured.update(kwargs)
        yield b"pcm"

    monkeypatch.setattr(
        "music_assistant.providers.ai_radio.rendering.get_ffmpeg_stream", fake_ffmpeg_stream
    )
    renderer = DummyRenderer()
    renderer.measured_loudness = -18.0
    _host_with_effects(renderer, music_bed="/media/bed.mp3")
    _attach_queue(renderer, [_clip_item("sess_001", **{ATTR_HOST_ID: "mika"})])
    _attach_normalization(renderer, target=-14, boost=3)
    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    [chunk async for chunk in renderer.get_audio_stream(streamdetails)]

    params = captured["filter_params"]
    assert params[:2] == [TTS_SPEECHNORM_FILTER, "volume=7.0dB"]
    assert any(isinstance(item, ComplexFilter) for item in params)
    assert params[-1] == f"alimiter=limit={TTS_PEAK_CEILING_DB}dB:level=false:latency=true"


async def test_a_host_without_effects_probes_no_sounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare breaks never pay for a sound lookup."""
    probed = _stub_sound_tags(monkeypatch)
    renderer = DummyRenderer()
    renderer._hosts["mika"] = {"effects": normalize_effects(None)}
    _attach_queue(renderer, [_clip_item("sess_001", **{ATTR_HOST_ID: "mika"})])

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert probed == []
    assert streamdetails.data is None or streamdetails.data.effects is None


# a plain transition: no weather or news in it, so only the general jingles fit
TRANSITION = {ATTR_HOST_ID: "mika", ATTR_SLOT_WHEN: "between_songs", ATTR_PROMPT: "Talk."}


def _jingle_host(renderer: DummyRenderer, selection: str = "ai", chance: int = 100) -> None:
    """Give 'mika' a library of two general jingles and a news jingle."""
    renderer._hosts["mika"] = {
        "effects": normalize_effects(
            {
                "jingles": [
                    {"source": "/media/calm.mp3", "tags": ["calm"], "text": "Mika hier."},
                    {"source": "/media/indie.mp3", "tags": ["indie"], "text": "Indie!"},
                    {"source": "/media/news.mp3", "tags": ["news"], "text": "Neues."},
                ],
                "jingle_selection": selection,
                "jingle_chance": chance,
            }
        )
    }


async def test_the_llm_picks_the_jingle_and_the_line_is_not_spoken() -> None:
    """The script's first line names the jingle, and only the script is voiced and kept."""
    renderer = DummyRenderer()
    _jingle_host(renderer)
    renderer.llm_reply = "JINGLE: 2\nGood evening, it is warm out."
    item = _clip_item("sess_001", **TRANSITION)
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert '1. [calm] "Mika hier."' in renderer.llm_prompts[0]
    assert "news.mp3" not in renderer.llm_prompts[0]
    assert item.extra_attributes[ATTR_JINGLE] == "/media/indie.mp3"
    assert renderer.tts_texts == ["Good evening, it is warm out."]
    assert renderer._break_memory["mika"][-1]["text"] == "Good evening, it is warm out."


async def test_an_unanswered_choice_falls_back_to_the_genre_of_the_next_song() -> None:
    """When the LLM names no jingle, one tagged for the next song's genre is taken."""
    renderer = DummyRenderer()
    _jingle_host(renderer)
    renderer.next_genres = {"Indie Rock"}
    item = _clip_item("sess_001", **TRANSITION)
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert item.extra_attributes[ATTR_JINGLE] == "/media/indie.mp3"


async def test_random_selection_never_asks_the_llm() -> None:
    """Left to chance, the prompt carries no jingle options."""
    renderer = DummyRenderer()
    _jingle_host(renderer, selection="random")
    item = _clip_item("sess_001", **{ATTR_HOST_ID: "mika", ATTR_WEB_SEARCH_MODE: "force"})
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert "JINGLE" not in renderer.llm_prompts[0]
    assert item.extra_attributes[ATTR_JINGLE] == "/media/news.mp3"


async def test_a_transition_without_luck_opens_without_a_jingle() -> None:
    """At a 0% chance plain transitions stay bare, and nothing is offered to the LLM."""
    renderer = DummyRenderer()
    _jingle_host(renderer, chance=0)
    item = _clip_item("sess_001", **TRANSITION)
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert "JINGLE" not in renderer.llm_prompts[0]
    assert item.extra_attributes[ATTR_JINGLE] == ""


async def test_the_same_jingle_does_not_open_two_breaks_in_a_row() -> None:
    """The host moves on to another jingle when there is one to move on to."""
    renderer = DummyRenderer()
    _jingle_host(renderer, selection="random")
    items = [_clip_item(f"sess_00{n}", **TRANSITION) for n in range(1, 5)]
    _attach_queue(renderer, items)

    for n in range(1, 5):
        await renderer.get_stream_details(f"sess_00{n}", MediaType.SOUND_EFFECT)

    picked = [item.extra_attributes[ATTR_JINGLE] for item in items]
    assert all(first != second for first, second in itertools.pairwise(picked))


async def test_the_llm_closes_a_break_with_a_jingle_into_the_next_song() -> None:
    """With a song to follow, the LLM may name a closer; neither line is voiced."""
    renderer = DummyRenderer()
    _jingle_host(renderer)
    renderer.next_genres = set()
    renderer.llm_reply = "JINGLE: 1\nAFTER: 2\nGood evening."
    item = _clip_item("sess_001", **TRANSITION)
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert "'AFTER: <number or none>'" in renderer.llm_prompts[0]
    assert item.extra_attributes[ATTR_JINGLE] == "/media/calm.mp3"
    assert item.extra_attributes[ATTR_JINGLE_AFTER] == "/media/indie.mp3"
    assert renderer.tts_texts == ["Good evening."]


async def test_closing_jingles_keep_the_hosts_gap() -> None:
    """Once a break closed with a jingle, the next ones are not even offered one."""
    renderer = DummyRenderer()
    _jingle_host(renderer)
    renderer.next_genres = set()
    renderer.llm_reply = "JINGLE: 1\nAFTER: 2\nGood evening."
    items = [_clip_item(f"sess_00{n}", **TRANSITION) for n in (1, 2)]
    _attach_queue(renderer, items)

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)
    await renderer.get_stream_details("sess_002", MediaType.SOUND_EFFECT)

    assert "AFTER" not in renderer.llm_prompts[1]
    assert items[1].extra_attributes[ATTR_JINGLE_AFTER] == ""


async def test_a_declined_closer_keeps_the_gap_open() -> None:
    """'none' closes nothing, so the next break may still be offered one."""
    renderer = DummyRenderer()
    _jingle_host(renderer)
    renderer.next_genres = set()
    renderer.llm_reply = "JINGLE: 1\nAFTER: none\nGood evening."
    items = [_clip_item(f"sess_00{n}", **TRANSITION) for n in (1, 2)]
    _attach_queue(renderer, items)

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)
    await renderer.get_stream_details("sess_002", MediaType.SOUND_EFFECT)

    assert items[0].extra_attributes[ATTR_JINGLE_AFTER] == ""
    assert "AFTER" in renderer.llm_prompts[1]


async def test_no_closing_jingle_without_a_song_to_lead_into() -> None:
    """The last break of a show is not offered a closer."""
    renderer = DummyRenderer()
    _jingle_host(renderer)
    item = _clip_item("sess_001", **TRANSITION)
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert "AFTER" not in renderer.llm_prompts[0]
    assert item.extra_attributes[ATTR_JINGLE_AFTER] == ""


async def test_a_section_decides_its_jingles_itself() -> None:
    """'always' opens a transition despite a 0% chance, 'never' keeps the closer away."""
    renderer = DummyRenderer()
    _jingle_host(renderer, chance=0)
    renderer.next_genres = set()
    renderer.llm_reply = "JINGLE: 2\nGood evening."
    item = _clip_item(
        "sess_001",
        **TRANSITION,
        **{ATTR_JINGLE_BEFORE_MODE: "always", ATTR_JINGLE_AFTER_MODE: "never"},
    )
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert "AFTER" not in renderer.llm_prompts[0]
    assert item.extra_attributes[ATTR_JINGLE] == "/media/indie.mp3"
    assert item.extra_attributes[ATTR_JINGLE_AFTER] == ""


@pytest.mark.parametrize(("post_fits", "closer"), [(False, "/media/indie.mp3"), (True, "")])
async def test_a_section_can_close_with_a_jingle_only_when_it_cannot_post(
    post_fits: bool, closer: str
) -> None:
    """'no_post' closes every break with a jingle, unless the voice can talk over the intro."""
    renderer = DummyRenderer()
    _jingle_host(renderer, chance=0)
    renderer.next_genres = set()
    renderer.llm_reply = "AFTER: 2\nGood evening."
    renderer._post_fits = lambda _item, _onset: post_fits
    item = _clip_item(
        "sess_001",
        **TRANSITION,
        **{ATTR_JINGLE_BEFORE_MODE: "never", ATTR_JINGLE_AFTER_MODE: "no_post"},
    )
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert ("AFTER" in renderer.llm_prompts[0]) is not post_fits
    assert item.extra_attributes[ATTR_JINGLE_AFTER] == closer


@pytest.mark.parametrize(
    ("mode", "talk_over_fits", "opener"),
    [
        ("no_post", False, "/media/indie.mp3"),
        ("no_post", True, ""),
        ("auto", True, ""),
        ("always", True, "/media/indie.mp3"),
    ],
)
async def test_a_section_can_open_with_a_jingle_only_when_it_cannot_talk_over(
    mode: str, talk_over_fits: bool, opener: str
) -> None:
    """A break that starts over the song's outro opens bare, unless it asks for a jingle."""
    renderer = DummyRenderer()
    _jingle_host(renderer, chance=100 if mode == "auto" else 0)
    renderer.next_genres = set()
    renderer.llm_reply = "JINGLE: 2\nGood evening."
    renderer._talk_over_fits = lambda _item: talk_over_fits
    item = _clip_item(
        "sess_001",
        **TRANSITION,
        **{ATTR_JINGLE_BEFORE_MODE: mode, ATTR_JINGLE_AFTER_MODE: "never"},
    )
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert item.extra_attributes[ATTR_JINGLE] == opener


async def test_a_news_section_can_go_without_its_jingle() -> None:
    """'never' leaves even the news bare."""
    renderer = DummyRenderer()
    _jingle_host(renderer, selection="random")
    item = _clip_item(
        "sess_001",
        **{ATTR_HOST_ID: "mika", ATTR_WEB_SEARCH_MODE: "force", ATTR_JINGLE_BEFORE_MODE: "never"},
    )
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert item.extra_attributes[ATTR_JINGLE] == ""


async def test_without_the_llm_the_news_closes_with_another_jingle() -> None:
    """Left to chance, the news closes with a jingle other than the one it opened with."""
    renderer = DummyRenderer()
    _jingle_host(renderer, selection="random")
    renderer.next_genres = set()
    item = _clip_item("sess_001", **{ATTR_HOST_ID: "mika", ATTR_WEB_SEARCH_MODE: "force"})
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert item.extra_attributes[ATTR_JINGLE] == "/media/news.mp3"
    assert item.extra_attributes[ATTR_JINGLE_AFTER] in {"/media/calm.mp3", "/media/indie.mp3"}


async def test_a_plain_transition_left_to_chance_closes_bare() -> None:
    """Without the LLM and without a reason, a transition goes straight into the song."""
    renderer = DummyRenderer()
    _jingle_host(renderer, selection="random")
    renderer.next_genres = set()
    item = _clip_item("sess_001", **TRANSITION)
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert item.extra_attributes[ATTR_JINGLE_AFTER] == ""


async def test_the_closing_jingle_is_mixed_in_after_the_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The picked closer is levelled like the opener and lengthens the announced break."""
    _stub_sound_tags(monkeypatch, seconds=3.0)
    renderer = DummyRenderer()
    _host_with_effects(renderer, jingles=[{"source": "/media/closer.mp3", "tags": []}])
    item = _clip_item(
        "sess_001",
        **{
            ATTR_HOST_ID: "mika",
            ATTR_JINGLE_BEFORE_MODE: "never",
            ATTR_JINGLE_AFTER_MODE: "always",
        },
    )
    _attach_queue(renderer, [item])

    streamdetails = await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    effects = streamdetails.data.effects
    assert item.extra_attributes[ATTR_JINGLE_AFTER] == "/media/closer.mp3"
    assert effects.after.path == "/media/closer.mp3"
    assert streamdetails.duration == 9 + 3


async def test_the_host_does_not_read_out_the_jingles_words() -> None:
    """The LLM is told not to repeat a jingle, and an echo it writes anyway is cut."""
    renderer = DummyRenderer()
    renderer._hosts["mika"] = {
        "effects": normalize_effects(
            {
                "jingles": [
                    {"source": "/media/a.mp3", "tags": [], "text": "Kopf aus, Lautsprecher an"}
                ],
                "jingle_chance": 100,
            }
        )
    }
    renderer.llm_reply = "JINGLE: 1\nKopf aus, Lautsprecher an! Guten Abend."
    item = _clip_item("sess_001", **TRANSITION)
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert "never say, quote or paraphrase" in renderer.llm_prompts[0]
    assert renderer.tts_texts == ["Guten Abend."]


@pytest.mark.parametrize("chance", [100, 0])
async def test_the_host_does_not_read_out_any_jingle_of_the_station(chance: int) -> None:
    """Words of a jingle that does not play are cut too, whether one plays or none does."""
    renderer = DummyRenderer()
    renderer._hosts["mika"] = {
        "effects": normalize_effects(
            {
                "jingles": [
                    {"source": "/media/a.mp3", "tags": [], "text": "Kopf aus, Lautsprecher an"},
                    {"source": "/media/b.mp3", "tags": [], "text": "Staub auf der Nadel"},
                ],
                "jingle_chance": chance,
            }
        )
    }
    # with no jingle on offer the LLM is not asked to pick one, so it writes no choice line
    choice_line = "JINGLE: 1\n" if chance else ""
    renderer.llm_reply = f"{choice_line}Guten Abend. Staub auf der Nadel, Gold im Ohr."
    item = _clip_item("sess_001", **TRANSITION)
    _attach_queue(renderer, [item])

    await renderer.get_stream_details("sess_001", MediaType.SOUND_EFFECT)

    assert renderer.tts_texts == ["Guten Abend. Gold im Ohr."]
