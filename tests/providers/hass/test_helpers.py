"""Tests for the Home Assistant provider helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from music_assistant.providers.hass.helpers import (
    AI_ATTACHMENT_MAX_AGE,
    AI_ATTACHMENT_STAGING_DIR,
    is_safe_attachment_path,
    media_source_id,
    remove_staged_attachments,
    stage_attachment,
)


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


@pytest.mark.parametrize(
    ("path", "safe"),
    [
        ("/media/ai_radio/jingles/einschalten.mp3", True),
        ("/media/ai_radio/Kopf-aus_2.mp3", True),
        ("/media/Für die Szene.mp3", False),
        ("/media/ai_radio/Erster Kaffee, Erste Platte.mp3", False),
        ("/media/Das Radio für Musikentdecker/a.mp3", False),
    ],
)
def test_only_plain_file_names_are_handed_over_as_they_are(path: str, safe: bool) -> None:
    """Anything beyond letters, digits, dot, dash and underscore goes over as a copy."""
    assert is_safe_attachment_path(path) is safe


def test_a_copy_gets_a_neutral_name_and_old_ones_are_cleared(tmp_path: Path) -> None:
    """The copy keeps the audio and its extension, and leftovers of lost queries go."""
    source = tmp_path / "Für die Szene.mp3"
    source.write_bytes(b"jingle")
    staging = tmp_path / AI_ATTACHMENT_STAGING_DIR
    staging.mkdir()
    leftover = staging / "attachment_old.mp3"
    leftover.write_bytes(b"old")
    past = time.time() - AI_ATTACHMENT_MAX_AGE - 60
    os.utime(leftover, (past, past))
    recent = staging / "attachment_recent.mp3"
    recent.write_bytes(b"recent")

    copy = stage_attachment(str(source), media_root=tmp_path.as_posix())

    name = copy.rsplit("/", 1)[1]
    assert copy == f"{tmp_path.as_posix()}/{AI_ATTACHMENT_STAGING_DIR}/{name}"
    assert name.startswith("attachment_")
    assert name.endswith(".mp3")
    assert is_safe_attachment_path("/media/" + name)
    assert (staging / name).read_bytes() == b"jingle"
    assert not leftover.exists()
    assert recent.exists()

    remove_staged_attachments([copy, copy])
    assert not (staging / name).exists()


def test_an_odd_extension_is_left_off_the_copy(tmp_path: Path) -> None:
    """An extension with special characters is dropped rather than passed on."""
    source = tmp_path / "jingle.mp³"
    source.write_bytes(b"jingle")

    copy = stage_attachment(str(source), media_root=tmp_path.as_posix())

    assert "." not in copy.rsplit("/", 1)[1]
