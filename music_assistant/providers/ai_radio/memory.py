"""Break memory for AI Radio: what each host said on air, fed back so it stops repeating itself."""
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import asyncio
import datetime
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import aiofiles

from music_assistant.helpers.datetime import utc
from music_assistant.helpers.json import async_json_loads

from .constants import (
    BREAK_MEMORY_BREAK_CHARS,
    BREAK_MEMORY_BREAKS_INSTRUCTION,
    BREAK_MEMORY_EMPTY,
    BREAK_MEMORY_MAX_BREAKS,
    BREAK_MEMORY_MAX_NEWS,
    BREAK_MEMORY_NEWS_CHARS,
    BREAK_MEMORY_NEWS_INSTRUCTION,
    BREAK_MEMORY_PLACEHOLDERS,
    BREAK_MEMORY_RETENTION_HOURS,
    CONF_BREAK_MEMORY,
    DEFAULT_BREAK_MEMORY,
    RECENT_BREAKS_IN_PROMPT,
    RECENT_BREAKS_PLACEHOLDER,
    RECENT_NEWS_PLACEHOLDER,
    RECENT_NEWS_WINDOW_HOURS,
)
from .helpers import soft_limit_text

if TYPE_CHECKING:
    from pathlib import Path

    from music_assistant_models.config_entries import ProviderConfig


class AIRadioMemoryMixin:
    """Remembers the breaks each host aired and feeds them back into its later prompts."""

    if TYPE_CHECKING:
        config: ProviderConfig
        logger: logging.Logger
        _memory_file: Path

        def _configured_now(self) -> datetime.datetime:
            """Return the current time in the configured timezone."""

        async def _write_json_file(self, target: Path, payload: dict[str, Any]) -> None:
            """Write a JSON payload to disk without corrupting the target on failure."""

    _break_memory: dict[str, list[dict[str, Any]]]
    _break_memory_lock: asyncio.Lock

    async def get_break_memory(self, host_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
        """
        Return the breaks remembered per host, oldest first.

        :param host_id: Only return the memory of this host.
        """
        memory = self._memory_store()
        if host_id is not None:
            return {host_id: deepcopy(memory.get(host_id, []))}
        return deepcopy(memory)

    async def clear_break_memory(self, host_id: str | None = None) -> None:
        """
        Forget remembered breaks, so the hosts start over without any memory.

        :param host_id: Only forget the memory of this host.
        """
        memory = self._memory_store()
        if host_id is None:
            memory.clear()
        elif memory.pop(host_id, None) is None:
            return
        await self._write_break_memory()

    async def _load_break_memory(self) -> None:
        """Load the remembered breaks from disk."""
        self._break_memory = {}
        try:
            if not await asyncio.to_thread(self._memory_file.exists):
                return
            async with aiofiles.open(self._memory_file) as file_handle:
                content = await file_handle.read()
            payload = await async_json_loads(content)
        except (OSError, ValueError) as err:
            # losing the memory only costs some variety, it must never keep the plugin down;
            # the file is replaced by the next break that airs
            self.logger.warning("Break memory could not be loaded, starting without it: %s", err)
            return
        hosts = payload.get("hosts") if isinstance(payload, dict) else None
        if not isinstance(hosts, dict):
            return
        for host_id, raw_entries in hosts.items():
            if not isinstance(raw_entries, list):
                continue
            entries = [entry for raw in raw_entries if (entry := _normalize_entry(raw))]
            if entries := _prune_entries(entries):
                self._break_memory[str(host_id)] = entries

    async def _write_break_memory(self) -> None:
        """Persist the remembered breaks to disk."""
        async with self._memory_lock():
            payload = {"version": 1, "hosts": deepcopy(self._memory_store())}
            await self._write_json_file(self._memory_file, payload)

    async def _remember_break(self, host_id: str, section_name: str, text: str, news: bool) -> None:
        """
        Remember a break the host is about to air.

        :param host_id: The host speaking the break.
        :param section_name: The display name of the section(s) the break was written for.
        :param text: The script of the break.
        :param news: Whether the break reports news.
        """
        if not host_id or not (text := text.strip()):
            return
        memory = self._memory_store()
        entries = memory.setdefault(host_id, [])
        entries.append(
            {
                "aired_at": utc().isoformat(),
                "section": section_name,
                "news": news,
                "text": text,
            }
        )
        memory[host_id] = _prune_entries(entries)
        try:
            await self._write_break_memory()
        except OSError as err:
            # the break still airs, and stays remembered until the next write succeeds
            self.logger.warning("Could not persist the AI Radio break memory: %s", err)

    def _apply_break_memory(self, prompt: str, host_id: str, news: bool) -> str:
        """
        Return the prompt with the host's recent breaks worked in.

        A prompt that places the memory placeholders itself gets exactly those filled in,
        whether or not the automatic memory is enabled. Otherwise the recent breaks, and for
        a news break the recent news, are appended when the automatic memory is enabled.

        :param prompt: The prompt with all other placeholders already resolved.
        :param host_id: The host speaking the break.
        :param news: Whether the break reports news.
        """
        if placed := [token for token in BREAK_MEMORY_PLACEHOLDERS if token in prompt]:
            values = {
                RECENT_BREAKS_PLACEHOLDER: self._recent_breaks_text(host_id),
                RECENT_NEWS_PLACEHOLDER: self._recent_news_text(host_id),
            }
            for token in placed:
                prompt = prompt.replace(token, values[token] or BREAK_MEMORY_EMPTY)
            return prompt
        blocks: list[str] = []
        if recent_breaks := self._recent_breaks_text(host_id):
            blocks.append(f"{BREAK_MEMORY_BREAKS_INSTRUCTION}\n{recent_breaks}")
        if news and (recent_news := self._recent_news_text(host_id)):
            blocks.append(f"{BREAK_MEMORY_NEWS_INSTRUCTION}\n{recent_news}")
        if not blocks or not self._break_memory_enabled():
            return prompt
        return "\n\n".join([prompt, *blocks])

    def _break_memory_enabled(self) -> bool:
        """Return whether recent breaks are appended to prompts that do not place them."""
        value = self.config.get_value(CONF_BREAK_MEMORY)
        return DEFAULT_BREAK_MEMORY if value is None else bool(value)

    def _recent_breaks_text(self, host_id: str) -> str:
        """Return the host's latest non-news breaks as a list, or an empty string."""
        entries = [entry for entry in self._memory_store().get(host_id, []) if not entry["news"]]
        return self._format_entries(entries[-RECENT_BREAKS_IN_PROMPT:], BREAK_MEMORY_BREAK_CHARS)

    def _recent_news_text(self, host_id: str) -> str:
        """Return the news the host reported within the news window, or an empty string."""
        cutoff = utc() - datetime.timedelta(hours=RECENT_NEWS_WINDOW_HOURS)
        entries = [
            entry
            for entry in self._memory_store().get(host_id, [])
            if entry["news"] and _aired_at(entry) >= cutoff
        ]
        return self._format_entries(entries, BREAK_MEMORY_NEWS_CHARS)

    def _format_entries(self, entries: list[dict[str, Any]], max_chars: int) -> str:
        """Render remembered breaks as one line each, stamped with their local airtime."""
        timezone = self._configured_now().tzinfo
        lines: list[str] = []
        for entry in entries:
            aired = _aired_at(entry).astimezone(timezone)
            text = soft_limit_text(" ".join(entry["text"].split()), max_chars=max_chars)
            lines.append(f"- {aired:%H:%M} ({entry['section']}): {text}")
        return "\n".join(lines)

    def _memory_store(self) -> dict[str, list[dict[str, Any]]]:
        """Return the per-host break memory, creating it on first use."""
        if not hasattr(self, "_break_memory"):
            self._break_memory = {}
        return self._break_memory

    def _memory_lock(self) -> asyncio.Lock:
        """Return the lock serializing writes of the memory file, creating it on first use."""
        if not hasattr(self, "_break_memory_lock"):
            self._break_memory_lock = asyncio.Lock()
        return self._break_memory_lock


def _aired_at(entry: dict[str, Any]) -> datetime.datetime:
    """Return when a remembered break aired."""
    return datetime.datetime.fromisoformat(entry["aired_at"])


def _normalize_entry(raw: Any) -> dict[str, Any] | None:
    """Return a remembered break read from disk, or None when it is unusable."""
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("text") or "").strip()
    if not text:
        return None
    try:
        aired_at = datetime.datetime.fromisoformat(str(raw.get("aired_at") or ""))
    except ValueError:
        return None
    if aired_at.tzinfo is None:
        aired_at = aired_at.replace(tzinfo=datetime.UTC)
    return {
        "aired_at": aired_at.isoformat(),
        "section": str(raw.get("section") or ""),
        "news": bool(raw.get("news")),
        "text": text,
    }


def _prune_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the breaks that are too old or too many to still matter, keeping the order."""
    cutoff = utc() - datetime.timedelta(hours=BREAK_MEMORY_RETENTION_HOURS)
    recent = [entry for entry in entries if _aired_at(entry) >= cutoff]
    # capped per kind, so a chatty run of transitions cannot push the news out of the window
    breaks = [entry for entry in recent if not entry["news"]][-BREAK_MEMORY_MAX_BREAKS:]
    news = [entry for entry in recent if entry["news"]][-BREAK_MEMORY_MAX_NEWS:]
    keep = {id(entry) for entry in (*breaks, *news)}
    return [entry for entry in recent if id(entry) in keep]
