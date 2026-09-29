"""Tests for the Home Assistant provider helpers."""

from __future__ import annotations

import pytest

from music_assistant.providers.hass.helpers import pick_stt_language


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
