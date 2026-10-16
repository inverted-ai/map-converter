"""The vocabulary dictionary: names an OpenDRIVE file uses, translated into names the converter understands.

Exporters name things their own way (a road mark ``custom``, an object orientation ``undefined``, a sign country
``OpenDRIVE``). A dictionary is a YAML file mapping such names, per element attribute, onto names the converter
already understands, so a new dialect needs a new dictionary, not a code change::

    translations:
      object.orientation: {undefined: none}
      roadMark.type: {custom: solid}

``convert_map.py`` reads ``vocabularies/default.yaml`` unless ``vocabulary=`` names another file; edit a copy to
extend it. A vendor dictionary (a file in ``vocabularies/vendors/``, or in the folder ``vocabulary_vendors=``
names) adds a ``match`` on the file header, e.g. ``match: {vendor: "Example Maps Ltd"}``; every vendor dictionary
whose match fits the header applies on top, whatever the mode. Every place a translation applies is recorded
(``kind="translated"``) in the conversion report.
A dictionary can only map onto meanings the converter has; it cannot teach it new ones, and it is refused when it
names an attribute it cannot translate or a target the converter does not understand.

A ``signals`` section reads codes rather than single names::

    signals:
      traffic_light_types: ["100\\d{4}"]   # a signal marked dynamic is a traffic light only with one of these codes
      non_vehicle_light_types: ["1000002"]  # lights with these codes (pedestrian, bicycle) make no vehicle stoplines
      countries_by_code: {USA: [R1-1]}      # a sign with no country and one of these codes uses that country's table
      lane_zero_validity_means_none: true   # a <validity> naming only lane 0 names no lane

A key left out reads the file literally (every dynamic signal is a vehicle light; no country is added; lane 0 is a
lane).
``signals`` also holds ``unknown_country_table`` (the sign table for a country that has none, default ``ZAM``) and
``sign_type_replacements`` (per country, sign types read as another, e.g. German 252 as 260). A ``markings``
section names the stop/yield stencil objects and the painted stop lines (by name pattern, object type, signal type);
left out, none are recognised.
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

from conversion_report import DEGRADATION_LOGGER, record_location

from commonroad.scenario.traffic_sign import TrafficSignIDCountries

from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser import vocabulary as reader_vocabulary
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLanes import Lane
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadtype import RoadType
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
    "non_vehicle_light_types",
    "countries_by_code",
    "lane_zero_validity_means_none",
    "unknown_country_table",
    "sign_type_replacements",
)
MARKING_KEYS = ("stop_stencils", "yield_stencils", "painted_stop_lines")
PAINTED_KEYS = ("names", "object_types", "signal_types")
# Per translatable element attribute, the names the converter understands (None: any name, e.g. a country).
UNDERSTOOD: dict[str, Optional[tuple[str, ...]]] = {
    **reader_vocabulary.ALLOWED,
    "lane.type": tuple(Lane.laneTypes),
    "object.type": tuple(ROAD_OBJECT_TYPES + CUSTOM_OBJECT_TYPES),
    "road.type": tuple(RoadType.allowedTypes),
    "signal.country": None,
}


@dataclasses.dataclass(slots=True, frozen=True)
class Signals:
    """The ``signals`` section (see the module docstring); a key no file gives keeps the literal reading."""

    traffic_light_types: Optional[tuple[str, ...]] = None  # None: every dynamic signal is a light
    non_vehicle_light_types: tuple[str, ...] = ()
    countries_by_code: dict[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
    lane_zero_validity_means_none: bool = False
    unknown_country_table: str = "ZAM"
    sign_type_replacements: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True, frozen=True)
class PaintedStopLines:
    """``markings.painted_stop_lines``: name patterns, object types (lower case) and signal types."""

    names: tuple[str, ...] = ()
    object_types: tuple[str, ...] = ()
    signal_types: tuple[str, ...] = ()


@dataclasses.dataclass(slots=True, frozen=True)
class Markings:
    """The ``markings`` section; left out, no object is a stencil or a painted stop line."""

    stop_stencils: tuple[str, ...] = ()
    yield_stencils: tuple[str, ...] = ()
    painted_stop_lines: PaintedStopLines = dataclasses.field(default_factory=PaintedStopLines)


@dataclasses.dataclass(slots=True, frozen=True)
class Vocabulary:
    """A loaded dictionary.

    Attributes:
        path: Absolute path of the file.
        sha256: Its content hash (recorded in the provenance).
        translations: Per element attribute, ``{name in the file: name the converter understands}``.
        match: For a vendor dictionary, the header fields a file must have (``vendor``, ``name``,
            ``revMajor``, ``revMinor``); empty for the base dictionary.
        signals: The ``signals`` section.
        markings: The ``markings`` section.
        given: The ``signals.<key>`` and ``markings.<key>`` the file gives (:func:`combine` takes only those).
        vendors: ``{path, sha256}`` of each vendor dictionary :func:`combine` applied on top of this one.
    """

    path: str
    sha256: str
    translations: dict[str, dict[str, str]]
    match: dict[str, str] = dataclasses.field(default_factory=dict)
    signals: Signals = dataclasses.field(default_factory=Signals)
    markings: Markings = dataclasses.field(default_factory=Markings)
    given: frozenset[str] = frozenset()
    vendors: tuple[dict[str, str], ...] = ()


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
    keys = {"translations", "signals", "markings"} | ({"match"} if vendor else set())
    _check_keys(path=path, where="", section=content, known=sorted(keys))
    match = {str(k): str(v) for k, v in (content.get("match") or {}).items()}
    if vendor and (not match or set(match) - set(HEADER_FIELDS)):
        raise ValueError(f"vocabulary {path}: a vendor dictionary needs a match on {', '.join(HEADER_FIELDS)}")
    signals = _signals(path=path, section=content.get("signals") or {})
    markings = _markings(path=path, section=content.get("markings") or {})
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
        signals=Signals(**signals),
        markings=Markings(**markings),
        given=frozenset([*(f"signals.{key}" for key in signals), *(f"markings.{key}" for key in markings)]),
    )


def _check_keys(*, path: str, where: str, section: dict, known: Sequence[str]) -> None:
    unknown = sorted(map(str, set(section) - set(known)))
    if unknown:
        raise ValueError(f"vocabulary {path}: {where}unknown key(s) {', '.join(unknown)}; known: {', '.join(known)}")


def _patterns(*, path: str, where: str, values) -> tuple[str, ...]:
    """``values`` as strings, each checked to be a regular expression."""
    patterns = tuple(str(p) for p in values or [])
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"vocabulary {path}: {where} {pattern!r} is not a regular expression: {exc}") from exc
    return patterns


def _markings(*, path: str, section: dict) -> dict[str, Any]:
    _check_keys(path=path, where="markings: ", section=section, known=MARKING_KEYS)
    markings: dict[str, Any] = {}
    for key in ("stop_stencils", "yield_stencils"):
        if key in section:
            markings[key] = tuple(str(name) for name in section[key] or [])
    if "painted_stop_lines" in section:
        painted = section["painted_stop_lines"] or {}
        _check_keys(path=path, where="painted_stop_lines: ", section=painted, known=PAINTED_KEYS)
        markings["painted_stop_lines"] = PaintedStopLines(
            names=_patterns(path=path, where="painted_stop_lines name", values=painted.get("names")),
            object_types=tuple(str(t).lower() for t in painted.get("object_types") or []),
            signal_types=tuple(str(t) for t in painted.get("signal_types") or []),
        )
    return markings


def _signals(*, path: str, section: dict) -> dict[str, Any]:
    _check_keys(path=path, where="signals: ", section=section, known=SIGNAL_KEYS)
    signals: dict[str, Any] = {}
    if "traffic_light_types" in section:
        signals["traffic_light_types"] = _patterns(
            path=path,
            where="traffic_light_types",
            values=section["traffic_light_types"],
        )
    if "non_vehicle_light_types" in section:
        signals["non_vehicle_light_types"] = tuple(str(code) for code in section["non_vehicle_light_types"] or [])
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
    """One dictionary from several: later ones win per translated name and per ``signals``/``markings`` key given."""
    translations: dict[str, dict[str, str]] = {}
    signals, markings = Signals(), Markings()
    for words in vocabularies:
        for element, table in words.translations.items():
            translations[element] = {**translations.get(element, {}), **table}
        signals = dataclasses.replace(signals, **_given(words=words, section="signals"))
        markings = dataclasses.replace(markings, **_given(words=words, section="markings"))
    first = vocabularies[0]
    return Vocabulary(
        path=first.path,
        sha256=first.sha256,
        translations=translations,
        signals=signals,
        markings=markings,
        given=frozenset().union(*(words.given for words in vocabularies)),
        vendors=tuple({"path": v.path, "sha256": v.sha256} for v in vocabularies[1:]),
    )


def _given(*, words: Vocabulary, section: str) -> dict[str, Any]:
    prefix = f"{section}."
    values = getattr(words, section)
    return {key[len(prefix):]: getattr(values, key[len(prefix):]) for key in words.given if key.startswith(prefix)}


def speed_unit_ok(*, speed_unit: str) -> bool:
    """Whether ``speed_unit`` is a speed unit name the reader knows or a positive number (the factor to m/s)."""
    return speed_unit in reader_vocabulary.SPEED_UNIT_DIVISORS or reader_vocabulary.speed_factor(speed_unit) is not None


def install(*, vocabulary: Vocabulary, speed_unit: str) -> None:
    """Make the OpenDRIVE reader use ``vocabulary`` and read speeds without a unit in ``speed_unit`` (process-wide,
    like crdesigner's own config objects)."""
    reader_vocabulary.TRANSLATIONS = {element: dict(table) for element, table in vocabulary.translations.items()}
    reader_vocabulary.STOP_STENCIL_NAMES = list(vocabulary.markings.stop_stencils)
    reader_vocabulary.YIELD_STENCIL_NAMES = list(vocabulary.markings.yield_stencils)
    reader_vocabulary.UNKNOWN_COUNTRY_TABLE = vocabulary.signals.unknown_country_table
    reader_vocabulary.NON_VEHICLE_LIGHT_TYPES = vocabulary.signals.non_vehicle_light_types
    reader_vocabulary.SIGN_TYPE_REPLACEMENTS = {
        country: dict(table) for country, table in vocabulary.signals.sign_type_replacements.items()
    }
    reader_vocabulary.UNITLESS_SPEED_UNIT = speed_unit


def is_traffic_light(*, signal_type: Optional[str], vocabulary: Vocabulary) -> bool:
    """Whether a signal marked dynamic is a traffic light: its type matches ``traffic_light_types`` (when given)."""
    patterns = vocabulary.signals.traffic_light_types
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
    for table_country, codes in vocabulary.signals.countries_by_code.items():
        if f"{code_type}-{code_subtype}" in codes or code_type in codes:
            return table_country, code_type, code_subtype
    return None


def names_no_lane(*, validity_from: object, validity_to: object, vocabulary: Vocabulary) -> bool:
    """Whether a ``<validity>`` naming only lane 0 is read as naming no lane (``lane_zero_validity_means_none``)."""
    if not vocabulary.signals.lane_zero_validity_means_none:
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
            "location": record_location(location=location),
        },
    )
