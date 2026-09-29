"""Just-in-time clip rendering for AI Radio."""
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.enums import ContentType, StreamType, VolumeNormalizationMode
from music_assistant_models.errors import (
    InvalidDataError,
    MediaNotFoundError,
    MusicAssistantError,
)
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import (
    CONF_VALUE_DISABLED,
    CONF_VALUE_ENABLED,
    CONF_VOLUME_NORMALIZATION,
    CONF_VOLUME_NORMALIZATION_TARGET,
    CONF_VOLUME_NORMALIZATION_TRACKS,
)
from music_assistant.helpers.audio import parse_loudnorm
from music_assistant.helpers.dsp import ComplexFilter
from music_assistant.helpers.ffmpeg import get_ffmpeg_stream
from music_assistant.helpers.process import check_output
from music_assistant.helpers.tags import async_parse_tags
from music_assistant.helpers.tts import (
    query_tts_engine_with_language_fallback,
    resolve_tts_language,
    resolve_tts_stream_path,
)

from .constants import (
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
    ATTR_WEATHER_REQUIRED,
    ATTR_WEB_SEARCH_MODE,
    CLIP_STREAMDETAILS_EXPIRATION,
    CONF_TTS_LOUDNESS_BOOST,
    DEFAULT_EFFECT_LOUDNESS,
    DEFAULT_JINGLE_AFTER_GAP_MINUTES,
    DEFAULT_JINGLE_SLOT_MODE,
    DEFAULT_MUSIC_BED_LEVEL,
    DEFAULT_TTS_LOUDNESS_BOOST,
    DEFERRED_PLACEHOLDERS,
    EFFECT_MEASURE_SECONDS,
    LOUDNESS_MEASURE_TIMEOUT,
    MIN_CLIP_MEDIA_LIFETIME,
    MIN_LOUDNESS_REFERENCE_SECONDS,
    NO_WEATHER_DATA_INSTRUCTION,
    RECENT_NEWS_PLACEHOLDER,
    TTS_CLIP_PCM_FORMAT,
    TTS_PEAK_CEILING_DB,
    TTS_SERVER_ERROR_MARKERS,
    TTS_SPEECHNORM_FILTER,
    WEATHER_PLACEHOLDER_TOKENS,
)
from .effects import (
    ClipEffects,
    EffectSound,
    JingleChoice,
    after_jingle_candidates,
    dressed_duration,
    effect_filters,
    jingle_candidates,
    jingle_choice_prompt,
    jingle_occasion,
    pick_jingle,
    resolve_source,
    strip_jingle_words,
    take_jingle_choices,
    wants_after_jingle,
)
from .helpers import coerce_float, coerce_int, format_ai_radio_timestamp, soft_limit_text

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.enums import MediaType
    from music_assistant_models.queue_item import QueueItem

    from music_assistant.mass import MusicAssistant

    from .models import SessionState


@dataclass(slots=True)
class _CachedClipMedia:
    """Media previously minted for a clip, kept until it expires."""

    path: str
    stream_type: StreamType
    audio_format: AudioFormat
    duration: int | None
    minted_at: float
    loudness: float | None


@dataclass(slots=True)
class _JinglePlan:
    """The jingles a break may open and close with, decided around its script."""

    before: list[dict[str, Any]]
    after: list[dict[str, Any]]
    # the break's section asks for a closing jingle every time
    after_always: bool
    # whether the break closes with a jingle when the LLM leaves it open
    rule_wants_after: bool
    # the block asking the LLM to pick, "" when it is not asked
    prompt: str


@dataclass(slots=True)
class _ClipAudio:
    """What get_audio_stream needs to serve a levelled clip, carried on StreamDetails.data."""

    path: str
    input_format: AudioFormat
    # None leaves the voice at the level the engine rendered it at
    gain_db: float | None
    effects: ClipEffects | None = None
    voice_seconds: int | None = None


class AIRadioRenderMixin:
    """Renders an AI Radio clip at the moment MA needs its audio."""

    if TYPE_CHECKING:
        mass: MusicAssistant
        config: ProviderConfig
        logger: logging.Logger
        _hosts: dict[str, dict[str, Any]]
        _sessions: dict[str, SessionState]

    _render_locks: dict[str, asyncio.Lock]
    _media_cache: dict[str, _CachedClipMedia]
    _engine_loudness: dict[tuple[str, str, str], float]
    _effect_assets: dict[str, tuple[float, float]]
    _last_jingles: dict[str, str]
    _last_after_jingles: dict[str, float]

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """
        Render the AI Radio clip with the given id and return its StreamDetails.

        :param item_id: The clip id of the queue item MA wants to play.
        :param media_type: The media type of the requested item.
        """
        queue_item = self._find_clip_item(item_id)
        if queue_item is None:
            raise MediaNotFoundError(f"AI Radio clip {item_id} is not in any queue")
        prompt = str(queue_item.extra_attributes.get(ATTR_PROMPT) or "")
        if not prompt:
            self._record_skip(queue_item, "clip has no prompt to render")
            raise MediaNotFoundError(f"AI Radio clip {item_id} has no prompt to render")

        async with self._lock_for(item_id):
            text = str(queue_item.extra_attributes.get(ATTR_RENDERED_TEXT) or "")
            if not text:
                text = await self._generate_script(queue_item, prompt, item_id)
                queue_item.extra_attributes[ATTR_RENDERED_TEXT] = text
                # the signal is what marks the items cache dirty and schedules the persist
                self.mass.player_queues.signal_update(queue_item.queue_id, items_changed=True)
            media = await self._cached_clip_media(queue_item, text, item_id)

        streamdetails = StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=media.audio_format,
            media_type=media_type,
            stream_type=media.stream_type,
            path=media.path,
            duration=media.duration,
            # a talk clip has nothing worth seeking to, and a seek is the one path that
            # would re-fetch a possibly-expired HA url mid-playback
            can_seek=False,
            allow_seek=False,
            # a cache hit serves a url that was minted earlier, so it may only claim the life
            # that url has left or the stream outlives the token behind it
            expiration=self._remaining_media_lifetime(media),
        )
        gain_db = self._loudness_gain(queue_item.queue_id, media.loudness)
        effects = await self._clip_effects(queue_item, media.loudness)
        if gain_db is not None or effects is not None:
            # core never normalizes a sound effect, so the clip is levelled here or it
            # airs noticeably quieter than the music around it
            streamdetails.stream_type = StreamType.CUSTOM
            # core mirrors what ffmpeg reports onto this object, so it gets a copy of the
            # constant rather than a handle on the one every clip shares
            streamdetails.decoded_audio_format = replace(TTS_CLIP_PCM_FORMAT)
            streamdetails.data = _ClipAudio(
                media.path, media.audio_format, gain_db, effects, media.duration
            )
            if effects is not None:
                streamdetails.duration = dressed_duration(effects, media.duration)
        return streamdetails

    async def get_audio_stream(
        self, streamdetails: StreamDetails, seek_position: int = 0
    ) -> AsyncGenerator[bytes]:
        """
        Return the levelled audio of a spoken clip as PCM.

        :param streamdetails: The StreamDetails previously returned by get_stream_details.
        :param seek_position: Ignored, a spoken clip cannot be seeked.
        """
        clip = cast("_ClipAudio", streamdetails.data)
        filter_params: list[str | ComplexFilter] = []
        if clip.gain_db is not None:
            filter_params += [TTS_SPEECHNORM_FILTER, f"volume={round(clip.gain_db, 2)}dB"]
        if clip.effects is not None:
            filter_params += effect_filters(clip.effects, clip.voice_seconds)
        filter_params.append(f"alimiter=limit={TTS_PEAK_CEILING_DB}dB:level=false:latency=true")
        async for chunk in get_ffmpeg_stream(
            audio_input=clip.path,
            input_format=clip.input_format,
            output_format=TTS_CLIP_PCM_FORMAT,
            filter_params=filter_params,
        ):
            yield chunk

    def _lock_for(self, clip_id: str) -> asyncio.Lock:
        """Return the per-clip render lock, creating it on first use."""
        if not hasattr(self, "_render_locks"):
            self._render_locks = {}
        if clip_id not in self._render_locks:
            self._render_locks[clip_id] = asyncio.Lock()
        return self._render_locks[clip_id]

    async def _cached_clip_media(
        self, queue_item: QueueItem, text: str, clip_id: str
    ) -> _CachedClipMedia:
        """Return the clip's minted media, re-minting only once the cache entry has expired."""
        if not hasattr(self, "_media_cache"):
            self._media_cache = {}
        now = asyncio.get_running_loop().time()
        cached = self._media_cache.get(clip_id)
        if cached is not None and self._remaining_media_lifetime(cached) > MIN_CLIP_MEDIA_LIFETIME:
            return cached
        # the caller holds the per-clip render lock, so of the several uncoordinated paths
        # that resolve the same clip only the first one mints; the rest hit the cache above
        path, stream_type, audio_format, duration, loudness = await self._mint_clip_media(
            queue_item, text, clip_id
        )
        media = _CachedClipMedia(path, stream_type, audio_format, duration, now, loudness)
        # clips are minted per queue item, so without pruning the cache grows for as long as
        # the server runs. an entry past its window can never be served again anyway
        for expired_id in [
            key
            for key, entry in self._media_cache.items()
            if now - entry.minted_at >= CLIP_STREAMDETAILS_EXPIRATION
        ]:
            del self._media_cache[expired_id]
        self._media_cache[clip_id] = media
        return media

    def _remaining_media_lifetime(self, media: _CachedClipMedia) -> int:
        """Return the seconds the given minted media is still usable for."""
        elapsed = asyncio.get_running_loop().time() - media.minted_at
        return max(MIN_CLIP_MEDIA_LIFETIME, round(CLIP_STREAMDETAILS_EXPIRATION - elapsed))

    def _wanted_loudness(self, queue_id: str) -> float | None:
        """Return the level in LUFS a clip should air at, or None when it should air as is."""
        normalization = self.mass.config.get_effective_player_queue_config_value(
            queue_id, CONF_VOLUME_NORMALIZATION, CONF_VALUE_ENABLED
        )
        if normalization == CONF_VALUE_DISABLED:
            return None
        # the queue switch only says normalization may run; the tracks around the clip are
        # the ones it has to match, and their own preference can still turn it off
        tracks_mode = self.mass.streams.get_config_value(CONF_VOLUME_NORMALIZATION_TRACKS)
        if tracks_mode == VolumeNormalizationMode.DISABLED.value:
            return None
        target = self.mass.streams.get_config_value(
            CONF_VOLUME_NORMALIZATION_TARGET, return_type=int
        )
        boost = coerce_int(
            self.config.get_value(CONF_TTS_LOUDNESS_BOOST), DEFAULT_TTS_LOUDNESS_BOOST
        )
        return target + boost

    def _loudness_gain(self, queue_id: str, loudness: float | None) -> float | None:
        """Return the dB to lift the clip by, or None when it should air untouched."""
        if loudness is None or (wanted := self._wanted_loudness(queue_id)) is None:
            return None
        # the reference is taken behind speechnorm, which lands close to the target on its
        # own, so this trim is small and runs in either direction
        return wanted - loudness

    def _tts_language(self, host_language: str | None = None) -> str | None:
        """
        Return the host's language, or the server locale, as a hyphenated language code.

        :param host_language: The host's configured language override, if any.
        """
        if override := (host_language or "").strip():
            return override.replace("_", "-")
        return resolve_tts_language(self.mass)

    def _find_clip_item(self, clip_id: str) -> QueueItem | None:
        """Return the queue item holding the given clip, or None when no queue holds it."""
        for queue_id in self._candidate_queue_ids(clip_id):
            if (item := self._find_clip_in_queue(clip_id, queue_id)) is not None:
                return item
        return None

    def _candidate_queue_ids(self, clip_id: str) -> list[str]:
        """
        Return the queue ids to search for a clip, the most likely one first.

        The owning session knows its queue, but the session registry is empty after a
        restart while the clip lives on in the persisted queue, so every queue stays a
        candidate. Clip ids carry a uuid4-based session id, so a hit is unambiguous.
        """
        queue_ids = [queue.queue_id for queue in self.mass.player_queues.all()]
        session = self._sessions.get(clip_id.rpartition("_")[0])
        if session is not None and session.queue_id in queue_ids:
            queue_ids.remove(session.queue_id)
            queue_ids.insert(0, session.queue_id)
        return queue_ids

    def _find_clip_in_queue(self, clip_id: str, queue_id: str) -> QueueItem | None:
        """Return the queue item holding the given clip, paging through the queue."""
        page_size = 500
        offset = 0
        while True:
            page = self.mass.player_queues.items(queue_id, limit=page_size, offset=offset)
            if not page:
                return None
            for item in page:
                if item.media_item is not None and item.media_item.item_id == clip_id:
                    return item
            if len(page) < page_size:
                return None
            offset += page_size

    async def _generate_script(self, queue_item: QueueItem, prompt: str, clip_id: str) -> str:
        """Resolve the deferred placeholders and generate the spoken script."""
        attributes = queue_item.extra_attributes
        deferred = await self._resolve_deferred_placeholders(prompt)
        empty_weather_tokens = [
            token
            for token in WEATHER_PLACEHOLDER_TOKENS
            if token in prompt and not deferred.get(token)
        ]
        if empty_weather_tokens:
            if attributes.get(ATTR_WEATHER_REQUIRED):
                error = "weather data unavailable for a weather-required clip"
                self.logger.warning(
                    "AI Radio clip %s (%s) skipped: %s", clip_id, queue_item.name, error
                )
                self._record_skip(queue_item, error)
                raise MediaNotFoundError(f"AI Radio clip {clip_id} has no weather data")
            # weather is optional in this clip, so the LLM must skip it rather than invent it
            for token in empty_weather_tokens:
                deferred[token] = NO_WEATHER_DATA_INSTRUCTION
        resolved = prompt
        for key, value in deferred.items():
            resolved = resolved.replace(key, value)
        host_id = str(attributes.get(ATTR_HOST_ID) or "")
        host = self._hosts.get(host_id) or {}
        instructions = str(host.get("instructions") or "")
        language = str(host.get("language") or "")
        max_chars = int(attributes.get(ATTR_MAX_CHARS) or 0)
        web_mode = str(attributes.get(ATTR_WEB_SEARCH_MODE) or "disabled")
        news = _is_news_clip(queue_item)
        resolved = self._apply_break_memory(resolved, host_id, news=news)
        jingles = await self._plan_jingles(queue_item, host_id, news, prompt)
        if jingles.prompt:
            resolved = f"{resolved}\n\n{jingles.prompt}"
        try:
            text = cast(
                "str",
                await self._generate_text(
                    instructions=instructions,
                    prompt=resolved,
                    web_mode=web_mode,
                    language=language,
                ),
            )
        except Exception as err:
            self.logger.warning(
                "AI Radio clip %s (%s) failed to generate: %s", clip_id, queue_item.name, err
            )
            self._record_skip(queue_item, f"generation failed: {err}")
            raise MediaNotFoundError(f"AI Radio clip {clip_id} failed to generate") from err
        text = self._settle_jingles(queue_item, host_id, jingles, text)
        if max_chars > 0:
            text = soft_limit_text(text, max_chars=max_chars)
        self.logger.debug(
            "AI Radio clip %s (%s) rendered: %d chars", clip_id, queue_item.name, len(text)
        )
        # rendering runs just in time, so a script that renders is one that is about to air
        await self._remember_break(host_id, queue_item.name, text, news=news)
        return text

    async def _resolve_deferred_placeholders(self, prompt: str) -> dict[str, str]:
        """Return freshly resolved values for the placeholders deferred until airtime."""
        values = dict.fromkeys(DEFERRED_PLACEHOLDERS, "")
        values["<timestamp>"] = format_ai_radio_timestamp(self._configured_now())
        # weather is the only deferred placeholder that costs a network round-trip, so it is
        # only fetched when the prompt actually references it
        if any(token in prompt for token in WEATHER_PLACEHOLDER_TOKENS):
            values.update(await self._prepare_weather_tokens())
        return values

    async def _mint_clip_media(
        self, queue_item: QueueItem, text: str, clip_id: str
    ) -> tuple[str, StreamType, AudioFormat, int | None, float | None]:
        """Convert the script to playable audio via the configured TTS engine."""
        host = self._hosts.get(str(queue_item.extra_attributes.get(ATTR_HOST_ID) or "")) or {}
        engine_uid = str(host.get("tts_engine") or "") or None
        language = self._tts_language(str(host.get("language") or ""))
        options = host.get("options") or {}
        try:
            path, stream_type, audio_format = await self._render_tts_media(
                text, engine_uid, language, options
            )
            # the probe is the first fetch, so a failed render surfaces here and not in playback
            duration = await self._probe_duration(path)
        except Exception as err:
            self.logger.warning("AI Radio clip %s failed TTS: %s", clip_id, err)
            self._record_skip(queue_item, f"TTS failed: {err}")
            raise MediaNotFoundError(f"AI Radio clip {clip_id} failed TTS") from err
        # measuring costs a fetch and a decode on the just-in-time render path, so it only
        # runs where the reading has somewhere to go
        # sounds added to the break are levelled against the voice, so a host with effects
        # needs the reading even when the queue does not normalize
        loudness = (
            await self._reference_loudness(engine_uid, language, options, path, duration)
            if self._wanted_loudness(queue_item.queue_id) is not None or _has_effects(host)
            else None
        )
        return path, stream_type, audio_format, duration, loudness

    async def _reference_loudness(
        self,
        engine_uid: str | None,
        language: str | None,
        options: dict[str, Any],
        path: str,
        duration: int | None,
    ) -> float | None:
        """Return the loudness in LUFS to level this clip against, or None when unknown."""
        if not hasattr(self, "_engine_loudness"):
            self._engine_loudness = {}
        # engine, language and options together decide which voice speaks, and clips from one
        # voice land within a dB of each other, so measuring one of them is enough
        key = (engine_uid or "", language or "", json.dumps(options, sort_keys=True, default=str))
        if (cached := self._engine_loudness.get(key)) is not None:
            return cached
        if (loudness := await self._measure_loudness(path)) is None:
            return None
        if (duration or 0) >= MIN_LOUDNESS_REFERENCE_SECONDS:
            self._engine_loudness[key] = loudness
        return loudness

    async def _measure_loudness(self, path: str) -> float | None:
        """Return the integrated loudness of the given audio in LUFS, or None when it fails."""
        try:
            returncode, output = await check_output(
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                path,
                # measure behind speechnorm: it is what the gain is applied on top of, and it
                # levels the clip itself, so the reading has to come from its output or the
                # gain corrects for a level that no longer reaches it
                "-af",
                f"{TTS_SPEECHNORM_FILTER},loudnorm=print_format=json",
                "-f",
                "null",
                "-",
                timeout=LOUDNESS_MEASURE_TIMEOUT,
            )
        except (OSError, TimeoutError) as err:
            self.logger.debug("Could not measure AI Radio clip loudness: %s", err)
            return None
        if returncode != 0:
            self.logger.debug("Could not measure AI Radio clip loudness: ffmpeg failed")
            return None
        return parse_loudnorm(output)

    async def _render_tts_media(
        self,
        text: str,
        engine_uid: str | None = None,
        language: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, StreamType, AudioFormat]:
        """Ask the TTS engine for audio and return the path, stream type and format to play it."""
        engine = await self._get_tts_engine(engine_uid)
        stream_details = await query_tts_engine_with_language_fallback(
            engine, text, language, logger=self.logger, options=options
        )
        path, stream_type = await resolve_tts_stream_path(engine, stream_details)
        audio_format = stream_details.audio_format
        if audio_format.content_type == ContentType.UNKNOWN:
            audio_format = AudioFormat(content_type=ContentType.MP3)
        return path, stream_type, audio_format

    async def _probe_duration(self, path: str) -> int | None:
        """Return the clip duration in seconds, or None when it cannot be determined."""
        try:
            tags = await async_parse_tags(path, require_duration=True)
        except (InvalidDataError, OSError) as err:
            if any(marker in str(err) for marker in TTS_SERVER_ERROR_MARKERS):
                # the engine reports no reason of its own (Home Assistant answers a failed
                # render with an empty 500), so the probe's message is the only clue there is
                raise MusicAssistantError(
                    f"{err}. The TTS engine failed to generate the audio it handed out. "
                    "Check the logs of the TTS engine for the reason (for a Home Assistant "
                    "engine that is the Home Assistant core log). A cloud engine may be "
                    "out of credit or having an outage."
                ) from err
            self.logger.warning("Could not determine AI Radio clip duration: %s", err)
            return None
        return int(tags.duration) if tags.duration else None

    async def _clip_effects(
        self, queue_item: QueueItem, voice_loudness: float | None
    ) -> ClipEffects | None:
        """Return the sounds the host dresses this break with, or None when it airs bare."""
        attributes = queue_item.extra_attributes
        host = self._hosts.get(str(attributes.get(ATTR_HOST_ID) or "")) or {}
        if not _has_effects(host):
            return None
        effects = host["effects"]
        # the jingle was picked together with the script, so a replay airs the same one
        jingle_path = resolve_source(str(attributes.get(ATTR_JINGLE) or ""))
        after_path = resolve_source(str(attributes.get(ATTR_JINGLE_AFTER) or ""))
        bed_path = resolve_source(str(effects.get("music_bed") or ""))
        if not jingle_path and not after_path and not bed_path:
            return None
        # the sounds sit relative to the voice, so they follow whatever level it airs at
        reference = self._wanted_loudness(queue_item.queue_id)
        if reference is None:
            reference = voice_loudness if voice_loudness is not None else DEFAULT_EFFECT_LOUDNESS
        bed_level = coerce_float(effects.get("music_bed_level"), DEFAULT_MUSIC_BED_LEVEL)
        jingle = await self._effect_sound(jingle_path, reference) if jingle_path else None
        bed = await self._effect_sound(bed_path, reference + bed_level) if bed_path else None
        after = await self._effect_sound(after_path, reference) if after_path else None
        if jingle is None and bed is None and after is None:
            return None
        return ClipEffects(jingle=jingle, bed=bed, after=after)

    async def _plan_jingles(
        self, queue_item: QueueItem, host_id: str, news: bool, prompt: str
    ) -> _JinglePlan:
        """Return the jingles a break may open and close with, and what to ask the LLM."""
        host = self._hosts.get(host_id) or {}
        effects: dict[str, Any] = host.get("effects") or {}
        occasion = self._jingle_occasion_of(queue_item, news, prompt)
        before = self._jingle_candidates(queue_item, host_id, effects, occasion)
        after, after_always = self._after_jingle_candidates(queue_item, host_id, effects, occasion)
        # the song's timing only matters to a closing jingle that is still to be decided
        vocal_onset = (
            await self._next_vocal_onset(queue_item) if after and not after_always else None
        )
        # a break that can carry over the song's intro does that rather than close with a
        # jingle, which would keep it off the intro
        if after and not after_always and self._post_fits(queue_item, vocal_onset):
            after = []
        # everything is decided in the call that writes the script, so it costs no extra
        # request, only the few lines listing the jingles
        ask_llm = bool(before or after) and effects.get("jingle_selection") == "ai"
        return _JinglePlan(
            before=before,
            after=after,
            after_always=after_always,
            rule_wants_after=wants_after_jingle(occasion, vocal_onset),
            prompt=(
                jingle_choice_prompt(before, after, after_always, vocal_onset) if ask_llm else ""
            ),
        )

    def _settle_jingles(
        self, queue_item: QueueItem, host_id: str, plan: _JinglePlan, text: str
    ) -> str:
        """Record the jingles around a break, returning its script without the choice lines."""
        choice = JingleChoice()
        if plan.prompt:
            choice, text = take_jingle_choices(text, plan.before, plan.after)
        genres = self._next_track_genres(queue_item) if plan.before or plan.after else set()
        jingle = choice.before
        if plan.before and jingle is None:
            jingle = pick_jingle(plan.before, genres)
        closer = _closing_jingle(
            choice, plan.after, plan.after_always, plan.rule_wants_after, jingle, genres
        )
        attributes = queue_item.extra_attributes
        attributes[ATTR_JINGLE] = jingle["source"] if jingle else ""
        attributes[ATTR_JINGLE_AFTER] = closer["source"] if closer else ""
        if jingle:
            self._last_jingles_by_host()[host_id] = jingle["source"]
        if closer:
            self._last_after_jingles_by_host()[host_id] = time.monotonic()
        # the listeners know every jingle of the station, and the LLM saw the words of all it
        # was offered, so no jingle's words are read out, played now or not
        library = (self._hosts.get(host_id) or {}).get("effects", {}).get("jingles") or []
        return strip_jingle_words(text, library)

    @staticmethod
    def _jingle_occasion_of(queue_item: QueueItem, news: bool, prompt: str) -> str:
        """Return what a break is as far as its jingles go, see jingle_occasion."""
        weather = any(token in prompt for token in WEATHER_PLACEHOLDER_TOKENS)
        slot_when = str(queue_item.extra_attributes.get(ATTR_SLOT_WHEN) or "")
        return jingle_occasion(news, weather, slot_when)

    def _jingle_candidates(
        self,
        queue_item: QueueItem,
        host_id: str,
        effects: dict[str, Any],
        occasion: str,
    ) -> list[dict[str, Any]]:
        """Return the jingles this break may open with, empty when it opens without one."""
        if not effects.get("jingles"):
            return []
        mode = _jingle_mode(queue_item, ATTR_JINGLE_BEFORE_MODE)
        if mode == "never":
            return []
        always = mode == "always"
        # news, weather and a show's ends always get theirs, a plain transition only now and
        # then, unless its section asks for one every time
        chance = coerce_int(effects.get("jingle_chance"), 0)
        if not always and occasion == "transition" and random.random() * 100 >= chance:
            return []
        last = self._last_jingles_by_host().get(host_id, "")
        return jingle_candidates(effects, occasion, self._configured_now().hour, last, always)

    def _after_jingle_candidates(
        self,
        queue_item: QueueItem,
        host_id: str,
        effects: dict[str, Any],
        occasion: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Return the jingles this break may close with, and whether it closes with one for sure.

        Left to itself a break closes with a jingle only now and then: never twice within the
        host's gap, and never when no song follows to lead into.
        """
        if not effects.get("jingles"):
            return [], False
        mode = _jingle_mode(queue_item, ATTR_JINGLE_AFTER_MODE)
        if mode == "never":
            return [], False
        always = mode == "always"
        if not always:
            next_item = self.mass.player_queues.get_next_item(
                queue_item.queue_id, queue_item.queue_item_id
            )
            if next_item is None:
                return [], False
            gap_minutes = coerce_int(
                effects.get("jingle_after_gap_minutes"), DEFAULT_JINGLE_AFTER_GAP_MINUTES
            )
            last = self._last_after_jingles_by_host().get(host_id)
            if last is not None and time.monotonic() - last < gap_minutes * 60:
                return [], False
        hour = self._configured_now().hour
        return after_jingle_candidates(effects, occasion, hour, always), always

    async def _next_vocal_onset(self, queue_item: QueueItem) -> float | None:
        """
        Return the second the song after a clip starts singing, None when unknown.

        Lets a jingle bridge into a song that sings right away. Only the post support knows
        the vocal timing, so without it this stays unknown.

        :param queue_item: The clip whose next song to look at.
        """
        return None

    def _post_fits(self, queue_item: QueueItem, vocal_onset: float | None) -> bool:
        """
        Return whether a break is set to carry over the next song's intro, and it has one.

        Without the post support no break carries over, so a closing jingle is never held
        back for one.

        :param queue_item: The clip to look at.
        :param vocal_onset: The second the next song starts singing, None when unknown.
        """
        return False

    def _next_track_genres(self, queue_item: QueueItem) -> set[str]:
        """Return the lowercase genres of the track after a clip, empty when unknown."""
        next_item = self.mass.player_queues.get_next_item(
            queue_item.queue_id, queue_item.queue_item_id
        )
        media_item = next_item.media_item if next_item is not None else None
        genres = media_item.metadata.genres if media_item is not None else None
        return {genre.lower() for genre in genres or ()}

    def _last_jingles_by_host(self) -> dict[str, str]:
        """Return the jingle each host played last, creating the record on first use."""
        if not hasattr(self, "_last_jingles"):
            self._last_jingles = {}
        return self._last_jingles

    def _last_after_jingles_by_host(self) -> dict[str, float]:
        """Return when each host last closed a break with a jingle, in monotonic seconds."""
        if not hasattr(self, "_last_after_jingles"):
            self._last_after_jingles = {}
        return self._last_after_jingles

    async def _effect_sound(self, path: str, target: float) -> EffectSound | None:
        """
        Return a jingle or bed levelled to the given loudness, or None when it cannot be used.

        A sound that is missing or unreadable is left out, the break still airs without it.

        :param path: The file path or URL of the sound.
        :param target: The loudness in LUFS the sound should play at.
        """
        if not hasattr(self, "_effect_assets"):
            self._effect_assets = {}
        if (cached := self._effect_assets.get(path)) is None:
            try:
                async with asyncio.timeout(LOUDNESS_MEASURE_TIMEOUT):
                    tags = await async_parse_tags(path, require_duration=True)
            except (InvalidDataError, OSError, TimeoutError) as err:
                self.logger.warning(
                    "AI Radio sound %s cannot be played, leaving it out: %s", path, err
                )
                return None
            loudness = await self._measure_effect_loudness(path)
            if not tags.duration or loudness is None:
                self.logger.warning("AI Radio sound %s could not be measured, leaving it out", path)
                return None
            cached = (float(tags.duration), loudness)
            self._effect_assets[path] = cached
        seconds, loudness = cached
        return EffectSound(path=path, seconds=seconds, gain_db=target - loudness)

    async def _measure_effect_loudness(self, path: str) -> float | None:
        """Return the integrated loudness of a jingle or bed in LUFS, or None when it fails."""
        try:
            returncode, output = await check_output(
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                path,
                "-t",
                str(EFFECT_MEASURE_SECONDS),
                "-af",
                "loudnorm=print_format=json",
                "-f",
                "null",
                "-",
                timeout=LOUDNESS_MEASURE_TIMEOUT,
            )
        except (OSError, TimeoutError) as err:
            self.logger.debug("Could not measure AI Radio sound %s: %s", path, err)
            return None
        if returncode != 0:
            self.logger.debug("Could not measure AI Radio sound %s: ffmpeg failed", path)
            return None
        return parse_loudnorm(output)

    def _record_skip(self, queue_item: QueueItem, error: str) -> None:
        """Record a skipped clip on its owning session."""
        session_id = str(queue_item.extra_attributes.get(ATTR_SESSION_ID) or "")
        if (session := self._sessions.get(session_id)) is None:
            return
        session.skipped_sections += 1
        session.last_render_error = error


def _is_news_clip(queue_item: QueueItem) -> bool:
    """Return whether a clip reports news."""
    attributes = queue_item.extra_attributes
    # a section that has to search the web is the news, as is one asking what it reported
    return str(attributes.get(ATTR_WEB_SEARCH_MODE) or "") == "force" or (
        RECENT_NEWS_PLACEHOLDER in str(attributes.get(ATTR_PROMPT) or "")
    )


def _jingle_mode(queue_item: QueueItem, key: str) -> str:
    """Return what a clip's section says about one of its jingles, see JINGLE_SLOT_MODES."""
    return str(queue_item.extra_attributes.get(key) or DEFAULT_JINGLE_SLOT_MODE)


def _closing_jingle(
    choice: JingleChoice,
    after: list[dict[str, Any]],
    always: bool,
    rule_wants_one: bool,
    opener: dict[str, Any] | None,
    genres: set[str],
) -> dict[str, Any] | None:
    """
    Return the jingle that closes a break, None when it goes straight into the song.

    :param choice: What the LLM answered, if it was asked.
    :param after: The jingles offered to close the break.
    :param always: The break's section asks for a closing jingle every time.
    :param rule_wants_one: Whether the break closes with one when the LLM did not decide.
    :param opener: The jingle opening the same break.
    :param genres: The genres of the next song, lowercase.
    """
    if not after:
        return None
    closer = choice.after
    if closer is None and (always or (not choice.after_answered and rule_wants_one)):
        # the same jingle at both ends sounds like a mistake, so another one closes if it can
        others = [jingle for jingle in after if opener is None or jingle is not opener]
        closer = pick_jingle(others or after, genres)
    if closer is not None and closer is opener and not always:
        return None
    return closer


def _has_effects(host: dict[str, Any]) -> bool:
    """Return whether a host dresses its breaks with any sound."""
    effects = host.get("effects") or {}
    return bool(effects.get("jingles") or effects.get("music_bed"))
