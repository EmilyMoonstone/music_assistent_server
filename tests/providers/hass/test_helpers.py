"""Tests for the Home Assistant provider helpers."""

from __future__ import annotations

import pytest

from music_assistant.providers.hass.helpers import media_source_id, pick_stt_language


@pytest.mark.parametrize(
    ("wanted", "languages", "expected"),
    [
        ("de-DE", ["en-US", "de-DE"], "de-DE"),
        ("de_de", ["en-US", "de-DE"], "de-DE"),
        ("de", ["de-AT", "de-DE"], "de-DE"),
        ("de-CH", ["en-US", "de-AT"], "de-AT"),
        ("en", ["de-DE", "en-GB", "en-US"], "en-GB"),
        ("fr-FR", ["de-DE", "en-US"], None),
        (None, ["de-DE", "en-US"], "de-DE"),
        ("", ["en-US"], "en-US"),
        ("de-DE", [], None),
    ],
)
def test_pick_stt_language(wanted: str | None, languages: list[str], expected: str | None) -> None:
    """Prefer the language itself, then another region of it, never a different language."""
    assert pick_stt_language(wanted, languages) == expected


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
