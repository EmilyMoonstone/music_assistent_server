"""Helpers to transcribe audio through the speech-to-text engines exposed by plugins."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from music_assistant_models.errors import MusicAssistantError

from music_assistant.constants import MASS_LOGGER_NAME
from music_assistant.helpers.plugin_engines import get_stt_engines
from music_assistant.helpers.process import communicate
from music_assistant.models.plugin import STT_SAMPLE_RATE

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant
    from music_assistant.models.plugin import STTEngine

LOGGER = logging.getLogger(f"{MASS_LOGGER_NAME}.helpers.stt")

# last-resort guard so a wedged engine fails over to the next one instead of hanging
STT_QUERY_TIMEOUT_SECONDS = 90
# the longest stretch of a file handed to an engine; speech engines are built for short
# commands, and the ones behind a cloud service cap the length they take
STT_MAX_SECONDS = 120


async def decode_for_stt(source: str, max_seconds: float = STT_MAX_SECONDS) -> bytes:
    """
    Return the audio of a file or URL as the raw PCM a speech-to-text engine takes.

    :param source: The file path or URL to read.
    :param max_seconds: How much of the audio to decode, from its start.
    """
    returncode, stdout, stderr = await communicate(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-t",
            str(max_seconds),
            "-i",
            source,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(STT_SAMPLE_RATE),
            "-f",
            "s16le",
            "-",
        ]
    )
    if returncode != 0 or not stdout:
        details = stderr.decode(errors="replace").strip() or f"exit code {returncode}"
        raise MusicAssistantError(f"Could not decode {source} for speech-to-text: {details}")
    return stdout


async def query_stt_engine(
    engine: STTEngine,
    audio: bytes,
    language: str | None = None,
    timeout: float | None = None,
) -> str:
    """
    Transcribe speech through a speech-to-text engine.

    :param engine: The engine to transcribe with.
    :param audio: The speech as raw PCM, see ``PluginProvider.speech_to_text``.
    :param language: Optional language code, omit to use the engine's own default.
    :param timeout: Seconds to wait for the engine, defaults to STT_QUERY_TIMEOUT_SECONDS.
    """
    if timeout is None:
        timeout = STT_QUERY_TIMEOUT_SECONDS
    try:
        async with asyncio.timeout(timeout) as query_timeout:
            return await engine.provider.speech_to_text(
                audio, language=language, engine_id=engine.id
            )
    except TimeoutError as err:
        # expired() tells our own cap apart from a timeout raised inside the engine
        if not query_timeout.expired():
            raise
        raise MusicAssistantError(
            f"STT engine '{engine.uid}' did not respond within {timeout}s"
        ) from err


async def transcribe(
    mass: MusicAssistant,
    source: str,
    language: str | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """
    Return the words spoken in a sound file, asking each speech-to-text engine in turn.

    An engine that fails (a spent quota, a timeout, a language it does not know) or hears no
    words hands over to the next one, so a single broken engine does not end the attempt.

    :param mass: The Music Assistant instance to find the engines on.
    :param source: The file path or URL of the sound.
    :param language: Optional language code of the speech, like 'de-DE'.
    :param logger: Optional logger to report failing engines on.
    :return: The recognised text, empty when the engines heard no words.
    :raises MusicAssistantError: When no engine is available or every engine failed.
    """
    engines = await get_stt_engines(mass)
    if not engines:
        raise MusicAssistantError(
            "No speech-to-text engine is available. Set up a plugin that provides "
            "speech-to-text, for example Home Assistant with an STT entity."
        )
    audio = await decode_for_stt(source)
    errors: list[str] = []
    for engine in engines:
        try:
            if text := (await query_stt_engine(engine, audio, language)).strip():
                return text
        except Exception as err:
            (logger or LOGGER).warning("Speech-to-text with %s failed: %s", engine.uid, err)
            errors.append(f"{engine.uid}: {err}")
    if len(errors) == len(engines):
        raise MusicAssistantError(f"Every speech-to-text engine failed ({'; '.join(errors)})")
    return ""
