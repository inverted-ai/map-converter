"""The vocabulary dictionary: names an OpenDRIVE file uses, translated into names the converter understands.

Exporters name things their own way (a road mark ``custom``, an object orientation ``undefined``, a sign country
``OpenDRIVE``). A dictionary is a YAML file mapping such names, per element attribute, onto names the converter
already understands, so a new dialect needs a new dictionary, not a code change::

    translations:
      object.orientation: {undefined: none}
      roadMark.type: {custom: solid}

``convert_map.py`` reads ``vocabularies/default.yaml`` unless ``vocabulary=`` names another file; edit a copy to
extend it. A vendor dictionary (a file in ``vocabularies/vendors/``, or in the folder ``vocabulary_vendors=``
names) adds a ``match`` on the file header, e.g. ``match: {vendor: "DeepMap, Inc."}``; every vendor dictionary
whose match fits the header applies on top, whatever the mode. Every place a translation applies is recorded
(``kind="translated"``) in the conversion report.
A dictionary can only map onto meanings the converter has; it cannot teach it new ones, and it is refused when it
names an attribute it cannot translate or a target the converter does not understand.

A ``signals`` section reads codes rather than single names::

    signals:
      traffic_light_types: ["100\\d{4}"]   # a signal marked dynamic is a traffic light only with one of these codes
      countries_by_code: {USA: [R1-1]}      # a sign with no country and one of these codes uses that country's table
      lane_zero_validity_means_none: true   # a <validity> naming only lane 0 names no lane

A key left out reads the file literally (every dynamic signal is a light; no country is added; lane 0 is a lane).
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
from typing import Any, Optional, Sequence

import yaml
from lxml import etree

from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser import vocabulary as reader_vocabulary
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLanes import Lane
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadObject import (
    CUSTOM_OBJECT_TYPES,
    ROAD_OBJECT_TYPES,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(REPO_ROOT, "vocabularies", "default.yaml")
VENDORS_DIR = os.path.join(REPO_ROOT, "vocabularies", "vendors")
HEADER_FIELDS = ("vendor", "name", "revMajor", "revMinor")
SIGNAL_KEYS = ("traffic_light_types", "countries_by_code", "lane_zero_validity_means_none")
DEGRADATION_LOGGER = "crdesigner.degradation"
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
        match: For a vendor dictionary, the header fields a file must have (``vendor``, ``name``,
            ``revMajor``, ``revMinor``); empty for the base dictionary.
        signals: The ``signals`` section, only the keys the file gives (see the module docstring).
    """

    path: str
    sha256: str
    translations: dict[str, dict[str, str]]
    match: dict[str, str] = dataclasses.field(default_factory=dict)
    signals: dict[str, Any] = dataclasses.field(default_factory=dict)


def load(*, path: str, vendor: bool = False) -> Vocabulary:
    """Read and check a dictionary file (a vendor dictionary when ``vendor``: it needs a ``match``).

    Raises:
        ValueError: the file is missing or not YAML, has an unknown key, a vendor dictionary has no usable
            ``match``, or it names an attribute the converter cannot translate or a name it does not understand.
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
    keys = {"translations", "signals", "match"} if vendor else {"translations", "signals"}
    unknown = sorted(set(content) - keys)
    if unknown:
        raise ValueError(f"vocabulary {path}: unknown key(s) {', '.join(map(str, unknown))}; expected: {', '.join(sorted(keys))}")
    match = {str(k): str(v) for k, v in (content.get("match") or {}).items()}
    if vendor and (not match or set(match) - set(HEADER_FIELDS)):
        raise ValueError(f"vocabulary {path}: a vendor dictionary needs a match on {', '.join(HEADER_FIELDS)}")
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
    return Vocabulary(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        translations=translations,
        match=match,
        signals=_signals(path=path, section=content.get("signals") or {}),
    )


def _signals(*, path: str, section: dict) -> dict[str, Any]:
    unknown = sorted(set(section) - set(SIGNAL_KEYS))
    if unknown:
        raise ValueError(f"vocabulary {path}: signals: unknown key(s) {', '.join(unknown)}; known: {', '.join(SIGNAL_KEYS)}")
    signals: dict[str, Any] = {}
    if "traffic_light_types" in section:
        patterns = [str(p) for p in section["traffic_light_types"] or []]
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"vocabulary {path}: traffic_light_types {pattern!r} is not a regular expression: {exc}") from exc
        signals["traffic_light_types"] = tuple(patterns)
    if "countries_by_code" in section:
        signals["countries_by_code"] = {
            str(country): tuple(str(code).upper() for code in codes or [])
            for country, codes in (section["countries_by_code"] or {}).items()
        }
    if "lane_zero_validity_means_none" in section:
        signals["lane_zero_validity_means_none"] = bool(section["lane_zero_validity_means_none"])
    return signals


def load_vendors(*, folder: str) -> list[Vocabulary]:
    """Every vendor dictionary (``*.yaml``) in ``folder``, by file name; none when the folder does not exist."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return []
    names = sorted(name for name in os.listdir(folder) if name.endswith((".yaml", ".yml")))
    return [load(path=os.path.join(folder, name), vendor=True) for name in names]


def read_header(*, xodr_path: str) -> dict[str, Optional[str]]:
    """The ``<header>`` fields a vendor dictionary can match (the rest of the file is not read)."""
    for _, element in etree.iterparse(xodr_path, events=("start",), huge_tree=True):
        if isinstance(element.tag, str) and etree.QName(element).localname == "header":
            return {field: element.get(field) for field in HEADER_FIELDS}
    return {field: None for field in HEADER_FIELDS}


def matching(*, vendors: list[Vocabulary], header: dict[str, Optional[str]]) -> list[Vocabulary]:
    """The vendor dictionaries whose every ``match`` field equals the header's."""
    return [v for v in vendors if all(header.get(field) == value for field, value in v.match.items())]


def combine(*, vocabularies: Sequence[Vocabulary]) -> Vocabulary:
    """One dictionary from several: later ones win per translated name and per ``signals`` key."""
    translations: dict[str, dict[str, str]] = {}
    signals: dict[str, Any] = {}
    for words in vocabularies:
        for element, table in words.translations.items():
            translations[element] = {**translations.get(element, {}), **table}
        signals.update(words.signals)
    first = vocabularies[0]
    return Vocabulary(path=first.path, sha256=first.sha256, translations=translations, signals=signals)


def install(*, vocabulary: Vocabulary) -> None:
    """Make the OpenDRIVE reader use ``vocabulary`` (process-wide, like crdesigner's own config objects)."""
    reader_vocabulary.TRANSLATIONS = {element: dict(table) for element, table in vocabulary.translations.items()}


def is_traffic_light(*, signal_type: Optional[str], vocabulary: Vocabulary) -> bool:
    """Whether a signal marked dynamic is a traffic light: its type matches ``traffic_light_types`` (when given)."""
    patterns = vocabulary.signals.get("traffic_light_types")
    if patterns is None:
        return True
    return any(re.fullmatch(pattern, signal_type or "") for pattern in patterns)


def country_for_code(
    *, country: Optional[str], signal_type: Optional[str], subtype: Optional[str], vocabulary: Vocabulary
) -> Optional[tuple[str, str, str]]:
    """``(country, type, subtype)`` to read a sign with no country by, from ``countries_by_code``; else ``None``.

    Codes compare upper-cased as ``TYPE-SUBTYPE`` or ``TYPE``; the returned type and subtype are upper-cased.
    """
    if (country or "").strip() or not signal_type:
        return None
    code_type, code_subtype = signal_type.upper(), (subtype or "").upper()
    for table_country, codes in vocabulary.signals.get("countries_by_code", {}).items():
        if f"{code_type}-{code_subtype}" in codes or code_type in codes:
            return table_country, code_type, code_subtype
    return None


def names_no_lane(*, validity_from: object, validity_to: object, vocabulary: Vocabulary) -> bool:
    """Whether a ``<validity>`` naming only lane 0 is read as naming no lane (``lane_zero_validity_means_none``)."""
    if not vocabulary.signals.get("lane_zero_validity_means_none"):
        return False
    try:
        return float(validity_from) == 0.0 and float(validity_to) == 0.0
    except (TypeError, ValueError):
        return False


def record(*, element: str, element_id: object, value: object, meaning: object, message: str, location: Optional[Sequence[float]]) -> None:
    """Record one place the dictionary's ``signals`` section changed how the file is read (``kind="translated"``)."""
    logging.getLogger(DEGRADATION_LOGGER).warning(
        "%s %s: %s (vocabulary)",
        element,
        element_id,
        message,
        extra={
            "kind": "translated",
            "element": element,
            "id": element_id,
            "value": value,
            "action": f"read_as_{meaning}",
            "location": None if location is None else [round(float(v), 2) for v in location[:2]],
        },
    )
