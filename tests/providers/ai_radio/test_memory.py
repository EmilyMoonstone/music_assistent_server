"""Unit tests for the AI Radio break memory."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from music_assistant.helpers.datetime import utc
from music_assistant.providers.ai_radio.constants import (
    BREAK_MEMORY_MAX_BREAKS,
    BREAK_MEMORY_MAX_NEWS,
    BREAK_MEMORY_RETENTION_HOURS,
    RECENT_BREAKS_IN_PROMPT,
    RECENT_NEWS_WINDOW_HOURS,
)
from music_assistant.providers.ai_radio.memory import AIRadioMemoryMixin
from music_assistant.providers.ai_radio.storage import AIRadioStorageMixin


class MemoryHarness(AIRadioMemoryMixin, AIRadioStorageMixin):
    """Minimal harness persisting the memory through the real storage path."""

    def __init__(self, memory_file: Path, enabled: bool | None = True) -> None:
        """Initialize the harness around the given memory file."""
        self.logger = logging.getLogger("tests.ai_radio.memory")
        self.config = cast("Any", SimpleNamespace(get_value=lambda _key: enabled))
        self._memory_file = memory_file

    def _configured_now(self) -> datetime.datetime:
        return utc()


def _entry(text: str, hours_ago: float = 0, news: bool = False) -> dict[str, Any]:
    """Build a remembered break that aired the given number of hours ago."""
    aired_at = utc() - datetime.timedelta(hours=hours_ago)
    return {"aired_at": aired_at.isoformat(), "section": "Talk", "news": news, "text": text}


async def test_the_memory_survives_a_restart(tmp_path: Path) -> None:
    """Remembered breaks are persisted and read back on the next load."""
    memory_file = tmp_path / "break_memory.json"
    harness = MemoryHarness(memory_file)
    await harness._remember_break("mika", "Transition", "Isar sunset again.", news=False)
    await harness._remember_break("mika", "News", "Parliament votes.", news=True)

    restarted = MemoryHarness(memory_file)
    await restarted._load_break_memory()

    assert await restarted.get_break_memory() == await harness.get_break_memory()
    assert [entry["text"] for entry in restarted._break_memory["mika"]] == [
        "Isar sunset again.",
        "Parliament votes.",
    ]


async def test_a_missing_memory_file_starts_empty(tmp_path: Path) -> None:
    """A fresh install has no memory yet."""
    harness = MemoryHarness(tmp_path / "break_memory.json")

    await harness._load_break_memory()

    assert await harness.get_break_memory() == {}


async def test_a_corrupt_memory_file_starts_empty(tmp_path: Path) -> None:
    """A broken file costs the memory, never the plugin."""
    memory_file = tmp_path / "break_memory.json"
    memory_file.write_text("{not json")
    harness = MemoryHarness(memory_file)

    await harness._load_break_memory()

    assert await harness.get_break_memory() == {}


async def test_unusable_and_expired_entries_are_dropped_on_load(tmp_path: Path) -> None:
    """Entries without text or a readable airtime, or past retention, are not loaded."""
    memory_file = tmp_path / "break_memory.json"
    naive = (utc() - datetime.timedelta(hours=1)).replace(tzinfo=None).isoformat()
    memory_file.write_text(
        json.dumps(
            {
                "version": 1,
                "hosts": {
                    "mika": [
                        "not an entry",
                        {"aired_at": utc().isoformat(), "text": "  "},
                        {"aired_at": "yesterday", "text": "No time"},
                        _entry("Too old", hours_ago=BREAK_MEMORY_RETENTION_HOURS + 1),
                        {"aired_at": naive, "text": "Naive time"},
                        _entry("Kept"),
                    ],
                    "gone": [_entry("Too old", hours_ago=BREAK_MEMORY_RETENTION_HOURS + 1)],
                    "broken": "not a list",
                },
            }
        )
    )
    harness = MemoryHarness(memory_file)

    await harness._load_break_memory()

    memory = await harness.get_break_memory()
    assert list(memory) == ["mika"]
    assert [entry["text"] for entry in memory["mika"]] == ["Naive time", "Kept"]
    assert all(entry["section"] is not None for entry in memory["mika"])


async def test_each_kind_is_capped_on_its_own(tmp_path: Path) -> None:
    """A long run of transitions cannot push the news out of the memory."""
    harness = MemoryHarness(tmp_path / "break_memory.json")
    await harness._remember_break("mika", "News", "Early news.", news=True)
    for index in range(BREAK_MEMORY_MAX_BREAKS + 5):
        await harness._remember_break("mika", "Transition", f"Break {index}.", news=False)
    for index in range(BREAK_MEMORY_MAX_NEWS):
        await harness._remember_break("mika", "News", f"News {index}.", news=True)

    entries = harness._break_memory["mika"]

    assert len([entry for entry in entries if not entry["news"]]) == BREAK_MEMORY_MAX_BREAKS
    assert len([entry for entry in entries if entry["news"]]) == BREAK_MEMORY_MAX_NEWS
    assert entries[0]["text"] == "Break 5."
    assert "Early news." not in [entry["text"] for entry in entries]


async def test_only_the_latest_breaks_reach_the_prompt(tmp_path: Path) -> None:
    """The prompt sees a handful of recent breaks, each flattened onto one line."""
    harness = MemoryHarness(tmp_path / "break_memory.json")
    for index in range(RECENT_BREAKS_IN_PROMPT + 2):
        await harness._remember_break("mika", "Transition", f"Break {index}.\nMore.", news=False)

    lines = harness._recent_breaks_text("mika").splitlines()

    assert len(lines) == RECENT_BREAKS_IN_PROMPT
    assert lines[0].endswith("(Transition): Break 2. More.")
    assert lines[-1].endswith(f"Break {RECENT_BREAKS_IN_PROMPT + 1}. More.")


async def test_news_outside_the_window_is_not_shown(tmp_path: Path) -> None:
    """Only the news of the last hours counts as already reported."""
    harness = MemoryHarness(tmp_path / "break_memory.json")
    harness._break_memory = {
        "mika": [
            _entry("Stale story.", hours_ago=RECENT_NEWS_WINDOW_HOURS + 1, news=True),
            _entry("Fresh story.", hours_ago=1, news=True),
        ]
    }

    news = harness._recent_news_text("mika")

    assert "Fresh story." in news
    assert "Stale story." not in news


async def test_nothing_is_appended_while_the_host_has_no_memory(tmp_path: Path) -> None:
    """A host without memory gets its prompt untouched."""
    harness = MemoryHarness(tmp_path / "break_memory.json")

    assert harness._apply_break_memory("Say hi.", "mika", news=True) == "Say hi."


async def test_the_memory_is_on_by_default(tmp_path: Path) -> None:
    """An install that never saved the setting gets the memory."""
    harness = MemoryHarness(tmp_path / "break_memory.json", enabled=None)

    assert harness._break_memory_enabled() is True


async def test_clearing_one_host_keeps_the_others(tmp_path: Path) -> None:
    """Clearing a host forgets only that host, clearing all forgets everyone."""
    memory_file = tmp_path / "break_memory.json"
    harness = MemoryHarness(memory_file)
    await harness._remember_break("mika", "Transition", "Hello.", news=False)
    await harness._remember_break("night_owl", "Transition", "Evening.", news=False)

    await harness.clear_break_memory("mika")

    assert list(await harness.get_break_memory()) == ["night_owl"]
    assert list(json.loads(memory_file.read_text())["hosts"]) == ["night_owl"]

    await harness.clear_break_memory()

    assert await harness.get_break_memory() == {}
    assert json.loads(memory_file.read_text())["hosts"] == {}


async def test_the_returned_memory_is_a_copy(tmp_path: Path) -> None:
    """A caller editing the returned memory does not change what the host remembers."""
    harness = MemoryHarness(tmp_path / "break_memory.json")
    await harness._remember_break("mika", "Transition", "Hello.", news=False)

    returned = await harness.get_break_memory("mika")
    returned["mika"].clear()

    assert len(harness._break_memory["mika"]) == 1
    assert await harness.get_break_memory("unknown") == {"unknown": []}


async def test_a_failed_write_keeps_the_break_in_memory(tmp_path: Path) -> None:
    """A disk problem must not fail the clip, and the break stays remembered."""
    harness = MemoryHarness(tmp_path / "missing_dir" / "break_memory.json")

    await harness._remember_break("mika", "Transition", "Hello.", news=False)

    assert [entry["text"] for entry in harness._break_memory["mika"]] == ["Hello."]
