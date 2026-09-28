"""Unit tests for the running order the AI puts together for an AI Radio show."""

from __future__ import annotations

import datetime
import logging
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from music_assistant_models.errors import MusicAssistantError

from music_assistant.providers.ai_radio.constants import (
    AI_ORDER_MAX_TRACKS_RANGE,
    AI_ORDER_REPLY_INSTRUCTION,
    DEFAULT_AI_ORDER_MAX_TRACKS,
    DEFAULT_AI_ORDER_PROMPT,
)
from music_assistant.providers.ai_radio.runtime import (
    AIRadioRuntimeMixin,
    _describe_track,
    _parse_track_order,
)
from music_assistant.providers.ai_radio.storage import AIRadioStorageMixin


class OrderRuntime(AIRadioRuntimeMixin):
    """Runtime harness whose AI answers with a canned reply."""

    def __init__(self, reply: str | Exception = "[]") -> None:
        """Initialize with the reply the AI gives."""
        self.logger = logging.getLogger("tests.ai_radio.running_order")
        self.reply = reply
        self.queries: list[str] = []

    def _configured_now(self) -> datetime.datetime:
        return datetime.datetime(2026, 9, 28, 22, 15, tzinfo=ZoneInfo("Europe/Berlin"))

    async def _ask_ai(self, query: str) -> str:
        self.queries.append(query)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class StationNormalizer(AIRadioStorageMixin):
    """Storage harness that knows one host, so stations validate."""

    def __init__(self) -> None:
        """Initialize with a single known host."""
        self._hosts = {"mika": {"id": "mika"}}


def _tracks(count: int) -> list[dict[str, Any]]:
    """Build shuffled-looking source tracks with index and songinfo."""
    return [
        {"index": n, "songinfo": f"Artist {n} - Song {n}", "duration": 200 + n}
        for n in range(count)
    ]


def _station(**overrides: Any) -> dict[str, Any]:
    """Build a raw station as a client would send it."""
    station = {
        "id": "musikentdecker",
        "name": "Musikentdecker",
        "source_playlist_id": "42",
        "host_id": "mika",
    }
    station.update(overrides)
    return station


@pytest.mark.parametrize(
    ("reply", "expected", "followed"),
    [
        ("[3, 1, 2]", [2, 0, 1], 3),
        ("Here you go:\n```json\n[2, 3, 1]\n```", [1, 2, 0], 3),
        ("[2, 2, 9, 0, 1]", [1, 0, 2], 2),
        ("[3]", [2, 0, 1], 1),
        ("no idea", [0, 1, 2], 0),
        ("[1, 2", [0, 1, 2], 0),
        ('{"order": [3, 2, 1]}', [2, 1, 0], 3),
    ],
)
def test_every_song_plays_once_whatever_the_ai_answers(
    reply: str, expected: list[int], followed: int
) -> None:
    """Placed songs lead, repeats and wrong numbers are ignored, the rest keep their draw."""
    assert _parse_track_order(reply, 3) == (expected, followed)


def test_a_song_is_described_with_album_year_genre_and_length() -> None:
    """The AI sees what it needs to judge whether two songs belong together."""
    track = {
        "songinfo": "Bilderbuch - Maschin",
        "duration": 245,
        "media_item": SimpleNamespace(
            album=SimpleNamespace(name="Schick Schock", year=2015),
            metadata=SimpleNamespace(genres={"indie pop", "austropop"}),
        ),
    }

    assert _describe_track(track) == (
        "Bilderbuch - Maschin (Schick Schock; 2015; austropop, indie pop; 4:05)"
    )
    assert _describe_track({"songinfo": "Solo - Song"}) == "Solo - Song"


async def test_the_ai_order_is_applied_and_renumbered() -> None:
    """The show plays in the AI's order, each song numbered by its new position."""
    runtime = OrderRuntime(reply="[3, 1, 2]")

    ordered = await runtime._apply_ai_order(_tracks(3), {"ai_order_max_tracks": 100})

    assert [track["songinfo"] for track in ordered] == [
        "Artist 2 - Song 2",
        "Artist 0 - Song 0",
        "Artist 1 - Song 1",
    ]
    assert [track["index"] for track in ordered] == [0, 1, 2]
    assert [track["source_index"] for track in ordered] == [2, 0, 1]


async def test_only_the_stations_pick_of_songs_is_ordered() -> None:
    """A long playlist is cut to the station's maximum before the AI sees it."""
    runtime = OrderRuntime(reply="[]")

    ordered = await runtime._apply_ai_order(_tracks(30), {"ai_order_max_tracks": 10})

    assert len(ordered) == 10
    assert "11. " not in runtime.queries[0]


async def test_an_unreachable_ai_plays_the_pick_as_drawn() -> None:
    """A failing AI costs the running order, never the show."""
    runtime = OrderRuntime(reply=MusicAssistantError("engine offline"))

    ordered = await runtime._apply_ai_order(_tracks(4), {"ai_order_max_tracks": 3})

    assert [track["songinfo"] for track in ordered] == [
        "Artist 0 - Song 0",
        "Artist 1 - Song 1",
        "Artist 2 - Song 2",
    ]


def test_the_default_prompt_is_anchored_to_the_show_start() -> None:
    """Without a custom prompt the built-in one asks for time of day, variety and fit."""
    runtime = OrderRuntime()

    query = runtime._ai_order_query(_tracks(2), {})

    assert query.startswith(DEFAULT_AI_ORDER_PROMPT.split("<timestamp>", maxsplit=1)[0])
    assert "Monday 28 September 2026, 22:15 CEST" in query
    assert "1. Artist 0 - Song 0" in query
    assert query.endswith(AI_ORDER_REPLY_INSTRUCTION)


def test_a_custom_prompt_and_the_listeners_wish_shape_the_order() -> None:
    """The station's own prompt replaces the default, and the wish for the show joins it."""
    runtime = OrderRuntime()
    station = {
        "ai_order_prompt": "Only slow songs after <timestamp>.",
        "listener_wish": "calm, we are cooking",
    }

    query = runtime._ai_order_query(_tracks(2), station)

    assert query.startswith("Only slow songs after Monday 28 September 2026, 22:15 CEST.")
    assert DEFAULT_AI_ORDER_PROMPT not in query
    assert "The listener's wish for this show: calm, we are cooking." in query
    # the reply format is not the prompt's to change, or the answer could not be read
    assert query.endswith(AI_ORDER_REPLY_INSTRUCTION)


@pytest.mark.parametrize(
    ("raw", "track_order", "shuffle"),
    [
        ({"shuffle_source_tracks": True}, "shuffle", True),
        ({"shuffle_source_tracks": False}, "playlist", False),
        ({"track_order": "ai", "shuffle_source_tracks": False}, "ai", True),
        ({"track_order": "playlist", "shuffle_source_tracks": True}, "playlist", False),
        ({"track_order": "nonsense", "shuffle_source_tracks": False}, "playlist", False),
    ],
)
def test_the_track_order_follows_what_the_station_asked_for(
    raw: dict[str, Any], track_order: str, shuffle: bool
) -> None:
    """A station saved before the running order existed keeps the order it had."""
    station = StationNormalizer()._normalize_station(_station(**raw))

    assert station["track_order"] == track_order
    assert station["shuffle_source_tracks"] is shuffle


def test_the_running_order_settings_are_normalized() -> None:
    """The song pick stays within range, and the prompt is stored trimmed."""
    normalizer = StationNormalizer()
    low, high = AI_ORDER_MAX_TRACKS_RANGE

    default = normalizer._normalize_station(_station())
    tiny = normalizer._normalize_station(_station(ai_order_max_tracks=1))
    huge = normalizer._normalize_station(
        _station(ai_order_max_tracks="9999", ai_order_prompt="  Chill.  ")
    )

    assert default["ai_order_max_tracks"] == DEFAULT_AI_ORDER_MAX_TRACKS
    assert default["ai_order_prompt"] == ""
    assert tiny["ai_order_max_tracks"] == low
    assert huge["ai_order_max_tracks"] == high
    assert huge["ai_order_prompt"] == "Chill."


def test_the_wish_is_not_stored_with_the_station() -> None:
    """A wish belongs to one show, so saving a station drops it."""
    station = StationNormalizer()._normalize_station(_station(listener_wish="party"))

    assert "listener_wish" not in station
