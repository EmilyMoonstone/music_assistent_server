"""Helpers for interpreting Home Assistant entities as player controls."""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4

from .constants import MediaPlayerEntityFeature, parse_supported_features

if TYPE_CHECKING:
    import logging

    from hass_client.models import State

# Home Assistant entity IDs are a domain and an object ID, both lowercase, joined by a dot
ENTITY_ID_PATTERN = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")
# the media folder Music Assistant shares with Home Assistant as an app, which Home Assistant
# serves as its "local" media source
HA_MEDIA_ROOT = PurePosixPath("/media")
HA_LOCAL_MEDIA_SOURCE = "media-source://media_source/local"
# Home Assistant's AI integrations pass a file's name on, Google's in an HTTP header that
# only takes ASCII; a file named with anything beyond these characters is handed over as a
# copy under a neutral name, in a hidden folder Home Assistant's media browser leaves out
SAFE_ATTACHMENT_NAME = re.compile(r"[A-Za-z0-9._-]+")
AI_ATTACHMENT_STAGING_DIR = ".music_assistant_ai"
# a copy left behind by a query that never finished is removed once it is this old
AI_ATTACHMENT_MAX_AGE = 3600


class ControlCapabilities(NamedTuple):
    """The player control roles a Home Assistant entity can serve."""

    power: bool = False
    volume: bool = False
    mute: bool = False


def media_source_id(path: str) -> str | None:
    """
    Return the Home Assistant media source id of a file in the shared media folder.

    :param path: The absolute path of the file, as Music Assistant sees it.
    :return: The media source id, or None when the file is outside the media folder, so
        Home Assistant cannot reach it.
    """
    file = PurePosixPath(path)
    if not file.is_absolute() or ".." in file.parts or not file.is_relative_to(HA_MEDIA_ROOT):
        return None
    relative = file.relative_to(HA_MEDIA_ROOT)
    if not relative.parts:
        return None
    return f"{HA_LOCAL_MEDIA_SOURCE}/{relative.as_posix()}"


def is_safe_attachment_path(path: str) -> bool:
    """
    Return whether a file can be handed to an AI integration under its own name.

    :param path: The absolute path of the file.
    """
    return all(SAFE_ATTACHMENT_NAME.fullmatch(part) for part in PurePosixPath(path).parts[1:])


def stage_attachment(source: str, media_root: str = str(HA_MEDIA_ROOT)) -> str:
    """
    Copy a file to a neutral name in the shared media folder, and return the copy's path.

    Blocking: run it in an executor. Copies older than AI_ATTACHMENT_MAX_AGE, left by
    queries that never finished, are removed on the way.

    :param source: The file to copy.
    :param media_root: The shared media folder, as Music Assistant sees it.
    """
    staging = Path(media_root) / AI_ATTACHMENT_STAGING_DIR
    staging.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - AI_ATTACHMENT_MAX_AGE
    for leftover in staging.iterdir():
        try:
            if leftover.is_file() and leftover.stat().st_mtime < cutoff:
                leftover.unlink()
        except OSError:
            # another query removed it first
            continue
    suffix = PurePosixPath(source).suffix
    if not SAFE_ATTACHMENT_NAME.fullmatch(suffix or "."):
        suffix = ""
    target = staging / f"attachment_{uuid4().hex}{suffix}"
    shutil.copyfile(source, target)
    return f"{media_root.rstrip('/')}/{AI_ATTACHMENT_STAGING_DIR}/{target.name}"


def remove_staged_attachments(paths: list[str]) -> None:
    """Remove the copies stage_attachment made, ignoring ones already gone. Blocking."""
    for path in paths:
        Path(path).unlink(missing_ok=True)


def is_entity_id(value: str) -> bool:
    """
    Return whether the given value has the shape of a Home Assistant entity ID.

    :param value: The value to inspect.
    """
    return bool(ENTITY_ID_PATTERN.match(value))


def get_control_capabilities(state: State, logger: logging.Logger) -> ControlCapabilities:
    """
    Return the player control roles the given Home Assistant entity can serve.

    :param state: The current state of the entity to inspect.
    :param logger: Logger to report an unparsable supported_features attribute on.
    :return: The supported roles; all False when the entity is unusable as a player control.
    """
    entity_platform = state["entity_id"].split(".")[0]
    if entity_platform in ("switch", "input_boolean"):
        # simple on/off controls are suitable as power and mute controls
        return ControlCapabilities(power=True, mute=True)
    if entity_platform in ("number", "input_number"):
        # number and input_number are very similar, both are suitable for volume control
        return ControlCapabilities(volume=True)
    # media player can be used as control, depending on features
    if entity_platform != "media_player":
        return ControlCapabilities()
    if "mass_player_type" in state["attributes"]:
        # filter out mass players
        return ControlCapabilities()
    supported_features = parse_supported_features(
        state["attributes"].get("supported_features"),
        state["entity_id"],
        logger,
    )
    return ControlCapabilities(
        power=(
            MediaPlayerEntityFeature.TURN_ON in supported_features
            and MediaPlayerEntityFeature.TURN_OFF in supported_features
        ),
        volume=MediaPlayerEntityFeature.VOLUME_SET in supported_features,
        mute=MediaPlayerEntityFeature.VOLUME_MUTE in supported_features,
    )


def get_control_name(entity_id: str, state: State | None) -> str:
    """
    Return the human readable name to present a Home Assistant entity control under.

    :param entity_id: The entity the control is based on.
    :param state: The entity's current state, if known.
    """
    if state and (friendly_name := state["attributes"].get("friendly_name")):
        return f"{friendly_name} ({entity_id})"
    return entity_id
