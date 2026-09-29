"""Tests for the Home Assistant provider helpers."""

from __future__ import annotations

import pytest

from music_assistant.providers.hass.helpers import media_source_id


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/media/jingle.mp3", "media-source://media_source/local/jingle.mp3"),
        ("/media/a b/ü.mp3", "media-source://media_source/local/a b/ü.mp3"),
        ("/media", None),
        ("/media/../data/x.mp3", None),
        ("/mediafiles/x.mp3", None),
        ("/data/x.mp3", None),
        ("media/x.mp3", None),
    ],
)
def test_media_source_id(path: str, expected: str | None) -> None:
    """Only files inside the shared media folder map to a local media source item."""
    assert media_source_id(path) == expected
