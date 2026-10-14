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
``signals`` also holds ``unknown_country_table`` (the sign table for a country that has none, default ``ZAM``) and
``sign_type_replacements`` (per country, sign types read as another, e.g. German 252 as 260). A ``markings``
section names the stop/yield stencil objects and the painted stop lines (by name pattern, object type, signal type);
left out, none are recognised. A ``defaults`` section gives ``speed_unit``: the unit of a lane speed that names none
(left out, such a speed stays unset).
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

from commonroad.scenario.traffic_sign import TrafficSignIDCountries

from crdesigner.map_conversion.opendrive.odr2cr.opendrive_conversion import network as reader_network
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_conversion.plane_elements import (
    traffic_signals as reader_signals,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser import parser as reader_parser
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
SIGNAL_KEYS = (
    "traffic_light_types",
    "countries_by_code",
    "lane_zero_validity_means_none",
    "unknown_country_table",
    "sign_type_replacements",
)
MARKING_KEYS = ("stop_stencils", "yield_stencils", "painted_stop_lines")
PAINTED_KEYS = ("names", "object_types", "signal_types")
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
    "speed.unit": tuple(reader_parser.SPEED_UNIT_DIVISORS),
}
DEFAULT_KEYS = ("speed_unit",)


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
        markings: The ``markings`` section, only the keys the file gives.
        defaults: The ``defaults`` section, only the keys the file gives.
    """

    path: str
    sha256: str
    translations: dict[str, dict[str, str]]
    match: dict[str, str] = dataclasses.field(default_factory=dict)
    signals: dict[str, Any] = dataclasses.field(default_factory=dict)
    markings: dict[str, Any] = dataclasses.field(default_factory=dict)
    defaults: dict[str, Any] = dataclasses.field(default_factory=dict)


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
    keys = {"translations", "signals", "markings", "defaults"} | ({"match"} if vendor else set())
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
        markings=_markings(path=path, section=content.get("markings") or {}),
        defaults=_defaults(path=path, section=content.get("defaults") or {}),
    )


def _defaults(*, path: str, section: dict) -> dict[str, Any]:
    unknown = sorted(set(section) - set(DEFAULT_KEYS))
    if unknown:
        raise ValueError(f"vocabulary {path}: defaults: unknown key(s) {', '.join(unknown)}; known: {', '.join(DEFAULT_KEYS)}")
    defaults: dict[str, Any] = {}
    if section.get("speed_unit") is not None:
        unit = str(section["speed_unit"])
        if unit not in reader_parser.SPEED_UNIT_DIVISORS:
            raise ValueError(
                f"vocabulary {path}: defaults: speed_unit {unit} is not one of {', '.join(reader_parser.SPEED_UNIT_DIVISORS)}"
            )
        defaults["speed_unit"] = unit
    return defaults


def _markings(*, path: str, section: dict) -> dict[str, Any]:
    unknown = sorted(set(section) - set(MARKING_KEYS))
    if unknown:
        raise ValueError(f"vocabulary {path}: markings: unknown key(s) {', '.join(unknown)}; known: {', '.join(MARKING_KEYS)}")
    markings: dict[str, Any] = {}
    for key in ("stop_stencils", "yield_stencils"):
        if key in section:
            markings[key] = tuple(str(name) for name in section[key] or [])
    if "painted_stop_lines" in section:
        painted = section["painted_stop_lines"] or {}
        unknown = sorted(set(painted) - set(PAINTED_KEYS))
        if unknown:
            raise ValueError(f"vocabulary {path}: painted_stop_lines: unknown key(s) {', '.join(unknown)}")
        for pattern in painted.get("names") or []:
            try:
                re.compile(str(pattern))
            except re.error as exc:
                raise ValueError(f"vocabulary {path}: painted_stop_lines name {pattern!r} is not a regular expression: {exc}") from exc
        markings["painted_stop_lines"] = {
            "names": tuple(str(p) for p in painted.get("names") or []),
            "object_types": tuple(str(t).lower() for t in painted.get("object_types") or []),
            "signal_types": tuple(str(t) for t in painted.get("signal_types") or []),
        }
    return markings


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
    if "unknown_country_table" in section:
        table = str(section["unknown_country_table"])
        if table not in TrafficSignIDCountries:
            raise ValueError(
                f"vocabulary {path}: unknown_country_table {table} has no sign table; known: {', '.join(TrafficSignIDCountries)}"
            )
        signals["unknown_country_table"] = table
    if "sign_type_replacements" in section:
        replacements = {}
        for country, table in (section["sign_type_replacements"] or {}).items():
            if str(country) not in TrafficSignIDCountries:
                raise ValueError(f"vocabulary {path}: sign_type_replacements: {country} has no sign table")
            known = {item.value for item in TrafficSignIDCountries[str(country)]}
            for code, meaning in (table or {}).items():
                if str(meaning) not in known:
                    raise ValueError(f"vocabulary {path}: sign_type_replacements: {country} {code} -> {meaning}, not a {country} sign type")
            replacements[str(country)] = {str(code): str(meaning) for code, meaning in (table or {}).items()}
        signals["sign_type_replacements"] = replacements
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
    markings: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    for words in vocabularies:
        for element, table in words.translations.items():
            translations[element] = {**translations.get(element, {}), **table}
        signals.update(words.signals)
        markings.update(words.markings)
        defaults.update(words.defaults)
    first = vocabularies[0]
    return Vocabulary(
        path=first.path,
        sha256=first.sha256,
        translations=translations,
        signals=signals,
        markings=markings,
        defaults=defaults,
    )


def install(*, vocabulary: Vocabulary) -> None:
    """Make the OpenDRIVE reader use ``vocabulary`` (process-wide, like crdesigner's own config objects)."""
    reader_vocabulary.TRANSLATIONS = {element: dict(table) for element, table in vocabulary.translations.items()}
    reader_network.STOP_STENCIL_NAMES = list(vocabulary.markings.get("stop_stencils", ()))
    reader_network.YIELD_STENCIL_NAMES = list(vocabulary.markings.get("yield_stencils", ()))
    reader_signals.UNKNOWN_COUNTRY_TABLE = vocabulary.signals.get("unknown_country_table", "ZAM")
    reader_parser.DEFAULT_SPEED_UNIT = vocabulary.defaults.get("speed_unit")
    reader_signals.SIGN_TYPE_REPLACEMENTS = {
        country: dict(table) for country, table in vocabulary.signals.get("sign_type_replacements", {}).items()
    }


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
