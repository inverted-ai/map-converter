"""The vocabulary dictionary: names an OpenDRIVE file uses, translated into names the converter understands.

Exporters name things their own way (a road mark ``custom``, an object orientation ``undefined``, a sign country
``OpenDRIVE``). A dictionary is a YAML file mapping such names, per element attribute, onto names the converter
already understands, so a new dialect needs a new dictionary, not a code change::

    translations:
      object.orientation: {undefined: none}
      roadMark.type: {custom: solid}

``convert_map.py`` reads ``vocabularies/default.yaml`` unless ``vocabulary=`` names another file; edit a copy to
extend it. Every place a translation applies is recorded (``kind="translated"``) in the conversion report.
A dictionary can only map onto meanings the converter has; it cannot teach it new ones, and it is refused when it
names an attribute it cannot translate or a target the converter does not understand.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from typing import Optional

import yaml

from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser import vocabulary as reader_vocabulary
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLanes import Lane
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadObject import (
    CUSTOM_OBJECT_TYPES,
    ROAD_OBJECT_TYPES,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(REPO_ROOT, "vocabularies", "default.yaml")
ORIENTATIONS = ("+", "-", "none")
# Per translatable element attribute, the names the converter understands (None: any name, e.g. a country).
UNDERSTOOD: dict[str, Optional[tuple[str, ...]]] = {
    "lane.type": tuple(Lane.laneTypes),
    # the road-mark types plane_group.convert_line_marking reads
    "roadMark.type": (
        "solid", "broken", "solid solid", "broken broken", "curb", "none",
        "solid broken", "solid dashed", "broken solid", "dashed solid",
    ),
    "object.type": tuple(ROAD_OBJECT_TYPES + CUSTOM_OBJECT_TYPES),
    "object.orientation": ORIENTATIONS,
    "signal.orientation": ORIENTATIONS,
    "signalReference.orientation": ORIENTATIONS,
    "signal.dynamic": ("yes", "no"),
    "signal.country": None,
    "connection.contactPoint": ("start", "end"),
}


@dataclasses.dataclass(slots=True, frozen=True)
class Vocabulary:
    """A loaded dictionary.

    Attributes:
        path: Absolute path of the file.
        sha256: Its content hash (recorded in the provenance).
        translations: Per element attribute, ``{name in the file: name the converter understands}``.
    """

    path: str
    sha256: str
    translations: dict[str, dict[str, str]]


def load(*, path: str) -> Vocabulary:
    """Read and check a dictionary file.

    Raises:
        ValueError: the file is missing or not YAML, has a key other than ``translations``, names an attribute
            the converter cannot translate, or translates into a name the converter does not understand.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ValueError(f"vocabulary {path} does not exist")
    with open(path, "rb") as f:
        raw = f.read()
    try:
        content = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"vocabulary {path} is not YAML: {exc}") from exc
    unknown = sorted(set(content) - {"translations"})
    if unknown:
        raise ValueError(f"vocabulary {path}: unknown key(s) {', '.join(map(str, unknown))}; expected: translations")
    translations = {}
    for element, table in (content.get("translations") or {}).items():
        if element not in UNDERSTOOD:
            raise ValueError(
                f"vocabulary {path}: {element} cannot be translated; translatable: {', '.join(UNDERSTOOD)}"
            )
        understood = UNDERSTOOD[element]
        table = {str(name): str(meaning) for name, meaning in (table or {}).items()}
        for name, meaning in table.items():
            if understood is not None and meaning not in understood:
                raise ValueError(
                    f"vocabulary {path}: {element} {name!r} -> {meaning!r}, which the converter does not understand; "
                    f"understood: {', '.join(understood)}"
                )
        translations[element] = table
    return Vocabulary(path=path, sha256=hashlib.sha256(raw).hexdigest(), translations=translations)


def install(*, vocabulary: Vocabulary) -> None:
    """Make the OpenDRIVE reader use ``vocabulary`` (process-wide, like crdesigner's own config objects)."""
    reader_vocabulary.TRANSLATIONS = {element: dict(table) for element, table in vocabulary.translations.items()}
