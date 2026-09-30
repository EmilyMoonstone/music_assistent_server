"""
A rehearsal of one segment: written, spoken and dressed like a break, played on a speaker.

Lets a host be tuned in the editor, from the prompt to the voice and the jingles, without
starting a show. Nothing of it is remembered: the host's break memory stays as it was.
"""
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import asyncio
import os
import tempfile
import wave
from contextlib import aclosing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import MediaType, StreamType
from music_assistant_models.errors import AudioError, InvalidDataError, MusicAssistantError

from music_assistant.helpers.ffmpeg import get_ffmpeg_stream

from .constants import (
    CLIP_COPY_FORMAT,
    DEFAULT_EFFECT_LOUDNESS,
    DEFAULT_MUSIC_BED_LEVEL,
    NO_WEATHER_DATA_INSTRUCTION,
    PROBE_CLIP_PREFIX,
    PROBE_EXAMPLE_SONGS,
    PROBE_RENDER_TIMEOUT,
    PROBE_SONG_LOOKAHEAD,
    RECENT_NEWS_PLACEHOLDER,
    TTS_CLIP_PCM_FORMAT,
    WEATHER_PLACEHOLDER_TOKENS,
)
from .effects import (
    ClipEffects,
    jingle_candidates,
    jingle_occasion,
    normalize_effects,
    pick_jingle,
    resolve_source,
    strip_jingle_words,
)
from .helpers import coerce_float, coerce_int, soft_limit_text

if TYPE_CHECKING:
    import logging

    from music_assistant_models.queue_item import QueueItem

    from music_assistant.mass import MusicAssistant


class AIRadioProbeMixin:
    """Rehearses a segment of a host that may not even be saved yet."""

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger

    async def render_probe(
        self, host: dict[str, Any], section: dict[str, Any], player_id: str
    ) -> dict[str, Any]:
        """
        Write, speak and dress one break of a segment, and return the file and its script.

        :param host: The host as the editor has it: instructions, voice, language, effects.
        :param section: The segment: its prompt, web search, length and opening jingle.
        :param player_id: The player it will play on, whose queue lends it its songs.
        """
        prompt = str(section.get("prompt") or "").strip()
        if not prompt:
            raise InvalidDataError("The segment has no prompt to rehearse")
        effects = normalize_effects(host.get("effects"))
        web_mode = str(section.get("web_search") or "disabled")
        news = web_mode == "force" or RECENT_NEWS_PLACEHOLDER in prompt
        weather = any(token in prompt for token in WEATHER_PLACEHOLDER_TOKENS)
        resolved = await self._probe_prompt(prompt, player_id)
        resolved = self._apply_break_memory(resolved, str(host.get("id") or ""), news=news)
        jingle = self._probe_jingle(effects, section, news, weather)
        language = str(host.get("language") or "")
        text = await self._generate_text(
            instructions=str(host.get("instructions") or ""),
            prompt=resolved,
            web_mode=web_mode,
            language=language,
        )
        text = strip_jingle_words(str(text), effects.get("jingles") or [])
        constraints = section.get("constraints") or {}
        if (max_chars := coerce_int(constraints.get("max_chars"), 0)) > 0:
            text = soft_limit_text(text, max_chars=max_chars)
        if not text.strip():
            raise MusicAssistantError("The AI wrote nothing for this segment")
        path, seconds = await self._render_probe_audio(host, effects, jingle, text, player_id)
        return {
            "path": path,
            "text": text,
            "jingle": jingle["source"] if jingle else "",
            "seconds": round(seconds, 1),
        }

    async def _probe_prompt(self, prompt: str, player_id: str) -> str:
        """Return the prompt with the songs of the player's queue, or examples, filled in."""
        songs = self._probe_songs(player_id)
        values = {
            "<prev_songinfo>": songs[0],
            "<next_songinfo>": songs[1],
            "<very_next_songinfo>": songs[2],
        }
        deferred = await self._resolve_deferred_placeholders(prompt)
        for token in WEATHER_PLACEHOLDER_TOKENS:
            if token in prompt and not deferred.get(token):
                # a rehearsal airs anyway, so the LLM is told to leave the weather out
                deferred[token] = NO_WEATHER_DATA_INSTRUCTION
        values.update(deferred)
        for key, value in values.items():
            prompt = prompt.replace(key, value)
        return prompt

    def _probe_songs(self, player_id: str) -> tuple[str, str, str]:
        """Return the song playing on the player and the two after it, examples where unknown."""
        found: list[str] = []
        queue = self.mass.player_queues.get(player_id)
        item: QueueItem | None = queue.current_item if queue is not None else None
        # the songs around a break are what the placeholders name, so a clip is passed over;
        # the look ahead is bounded, a repeating queue of clips would go round for ever
        for _ in range(PROBE_SONG_LOOKAHEAD):
            if item is None or len(found) == len(PROBE_EXAMPLE_SONGS):
                break
            if item.media_type == MediaType.TRACK:
                found.append(item.name)
            item = self.mass.player_queues.get_next_item(player_id, item.queue_item_id)
        songs = [*found, *PROBE_EXAMPLE_SONGS[len(found) :]]
        return songs[0], songs[1], songs[2]

    def _probe_jingle(
        self, effects: dict[str, Any], section: dict[str, Any], news: bool, weather: bool
    ) -> dict[str, Any] | None:
        """Return the jingle the rehearsal opens with, None when the segment opens bare."""
        if not effects.get("jingles") or section.get("jingle_before") == "never":
            return None
        # a rehearsal plays the opener whenever one fits, so it can be heard
        occasion = jingle_occasion(news, weather, "")
        hour = self._configured_now().hour
        candidates = jingle_candidates(effects, occasion, hour, always=True)
        return pick_jingle(candidates, set())

    async def _render_probe_audio(
        self,
        host: dict[str, Any],
        effects: dict[str, Any],
        jingle: dict[str, Any] | None,
        text: str,
        player_id: str,
    ) -> tuple[str, float]:
        """Speak the script, level and dress it like a break, and return the file and length."""
        engine_uid = str(host.get("tts_engine") or "") or None
        language = self._tts_language(str(host.get("language") or ""))
        options = host.get("options") or {}
        path, stream_type, audio_format = await self._render_tts_media(
            text, engine_uid, language, options
        )
        copy: str | None = None
        if stream_type == StreamType.HTTP:
            copy = path = await self._keep_local_copy(path)
            audio_format = CLIP_COPY_FORMAT
        try:
            duration = await self._probe_duration(path)
            loudness = await self._measure_loudness(path)
            wanted = self._wanted_loudness(player_id)
            gain_db = wanted - loudness if wanted is not None and loudness is not None else None
            reference = wanted if wanted is not None else (loudness or DEFAULT_EFFECT_LOUDNESS)
            clip_effects = await self._probe_effects(effects, jingle, reference)
            pcm = await self._probe_pcm(path, audio_format, gain_db, clip_effects, duration)
        finally:
            if copy is not None:
                await asyncio.to_thread(Path(copy).unlink, missing_ok=True)
        target = await asyncio.to_thread(self._write_probe, pcm)
        return target, len(pcm) / TTS_CLIP_PCM_FORMAT.pcm_sample_size

    async def _probe_effects(
        self, effects: dict[str, Any], jingle: dict[str, Any] | None, reference: float
    ) -> ClipEffects | None:
        """Return the opener and bed of the rehearsal, levelled against the voice."""
        opener = (
            await self._effect_sound(resolve_source(jingle["source"]), reference)
            if jingle
            else None
        )
        bed = None
        if bed_path := resolve_source(str(effects.get("music_bed") or "")):
            level = coerce_float(effects.get("music_bed_level"), DEFAULT_MUSIC_BED_LEVEL)
            bed = await self._effect_sound(bed_path, reference + level)
        if opener is None and bed is None:
            return None
        return ClipEffects(jingle=opener, bed=bed)

    async def _probe_pcm(
        self,
        path: str,
        audio_format: Any,
        gain_db: float | None,
        effects: ClipEffects | None,
        duration: int | None,
    ) -> bytes:
        """Return the dressed rehearsal as PCM."""
        from .rendering import _levelling_filters  # noqa: PLC0415

        chunks: list[bytes] = []
        try:
            async with (
                asyncio.timeout(PROBE_RENDER_TIMEOUT),
                aclosing(
                    get_ffmpeg_stream(
                        audio_input=path,
                        input_format=audio_format,
                        output_format=TTS_CLIP_PCM_FORMAT,
                        filter_params=_levelling_filters(gain_db, effects, duration),
                    )
                ) as pcm_stream,
            ):
                async for chunk in pcm_stream:
                    chunks.append(chunk)
        except (TimeoutError, AudioError) as err:
            raise MusicAssistantError(
                f"The rehearsal could not be rendered: {str(err) or type(err).__name__}"
            ) from err
        if not chunks:
            raise MusicAssistantError("The rehearsal came out silent")
        return b"".join(chunks)

    def _write_probe(self, pcm: bytes) -> str:
        """Write the rehearsal to a WAV file, replacing the ones before, and return its path."""
        # only the latest rehearsal is ever played, and one left from before a restart too is
        # no longer anyone's
        for earlier in Path(tempfile.gettempdir()).glob(f"{PROBE_CLIP_PREFIX}*"):
            earlier.unlink(missing_ok=True)
        handle, target = tempfile.mkstemp(prefix=PROBE_CLIP_PREFIX, suffix=".wav")
        try:
            with os.fdopen(handle, "wb") as probe_file, wave.open(probe_file, "wb") as wav:
                wav.setnchannels(TTS_CLIP_PCM_FORMAT.channels)
                wav.setsampwidth(TTS_CLIP_PCM_FORMAT.bit_depth // 8)
                wav.setframerate(TTS_CLIP_PCM_FORMAT.sample_rate)
                wav.writeframes(pcm)
        except OSError:
            Path(target).unlink(missing_ok=True)
            raise
        return target
