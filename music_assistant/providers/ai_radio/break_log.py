"""
What happened around each break: its script, its jingles and how it met the songs around it.

Kept in memory for the editor to show, so a post or a talk-over that did not happen can be
told apart from one that was never set up. The reasons are codes a client can translate.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

# the reasons _resolve_lyric_time gives, by how they start, and the code each is logged as
_LYRICS_REASON_CODES = (
    ("no track details", "no_track_details"),
    ("lyrics lookup took", "lyrics_timeout"),
    ("lyrics lookup failed", "lyrics_failed"),
    ("synced lyrics have no sung line", "no_sung_line"),
    ("only unsynced lyrics", "unsynced_lyrics"),
    ("no lyrics found", "no_lyrics"),
)


def lyrics_reason_code(reason: str) -> str:
    """
    Return the code for a reason the vocal timing of a song is unknown.

    :param reason: The reason as the lyrics lookup put it.
    """
    for prefix, code in _LYRICS_REASON_CODES:
        if reason.startswith(prefix):
            return code
    return "no_lyrics"


def new_break_entry(
    queue_item_id: str, section: str, session_id: str, station_id: str, host_id: str
) -> dict[str, Any]:
    """
    Return a fresh log entry for a break, before anything about it is known.

    :param queue_item_id: The break's queue item.
    :param section: The break's name, as its section is called.
    :param session_id: The run the break belongs to, "" for none.
    :param station_id: The station of that run, "" for none.
    :param host_id: The host speaking the break.
    """
    return {
        "queue_item_id": queue_item_id,
        "at": datetime.now(UTC).isoformat(),
        "section": section,
        "session_id": session_id,
        "station_id": station_id,
        "host_id": host_id,
        "text": "",
        "jingle_before": "",
        "jingle_after": "",
        # a break cuts in after its song and goes straight into the next, unless noted
        "from_song": {"kind": "cut"},
        "into_song": {"kind": "direct"},
        "skipped": None,
    }


def trim_log(log: OrderedDict[str, dict[str, Any]], size: int) -> None:
    """
    Drop the oldest entries beyond the given size.

    :param log: The log, oldest entry first.
    :param size: How many entries to keep.
    """
    while len(log) > size:
        log.popitem(last=False)


def reason(code: str, **values: float) -> dict[str, Any]:
    """
    Return a reason as logged: its code and the numbers a client fills into its text.

    :param code: What happened, e.g. early_vocal.
    :param values: The seconds that go with it, rounded to one decimal.
    """
    return {"code": code, **{key: round(value, 1) for key, value in values.items()}}
