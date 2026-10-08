"""Every rule the converter applies that guesses what an OpenDRIVE author meant.

OpenDRIVE leaves room for interpretation and exporters fill it differently, so some conversions
need a guess. Each guess lives here as one :class:`Heuristic`: a stable id (``H1``, ``H2``, ...), a
concise name that is also its :class:`HeuristicsConfig` switch, and the docstring of its rule
function, which states the assumption, the evidence behind it and the kind of map that breaks it.
When a map breaks one, switch it off or retune it here (``heuristics.<name>=...`` on the
``convert_map.py`` command line) rather than adding a special case elsewhere.

Every time a heuristic changes a conversion it logs a ``kind="heuristic"`` degradation record
(:func:`record`) whose ``element`` is its id, so ``<map>_conversion_report.json`` lists per
heuristic how often it fired and on which elements.

Guesses that are not configurable here yet:

- ``crdesigner/.../opendrive_conversion/network.py`` ``Network._stop_lines_from_road``: the object
  names read as painted stop markings (``Stencil_STOP``) and yield markings (``Stencil_GiveWay``,
  ``Stencil_Yield_Triangles``).
- ``crdesigner/.../plane_elements/traffic_signals.py`` ``assign_traffic_signals_to_road``: a sign
  whose country has no sign table is read with the ``ZAM`` table; dynamic types ``1000002`` and
  static ``1000003``/``1000004`` are skipped; German type ``252`` is read as ``260``.
- ``crdesigner/map_conversion/lanelet2/cr2lanelet.py`` ``SHARED_NODE_TOLERANCE``: linked lanelet
  ends closer than this are one seam.
- ``traffic_light_fsm.py`` ``FsmConfig`` (its own command-line flags): how lights are grouped into
  intersections and phases.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import re
from typing import Callable, Optional

from commonroad.scenario.traffic_sign import TrafficSignIDUsa

from conversion_report import DEGRADATION_LOGGER

US_STOP_YIELD_CODES = frozenset(item.value.upper() for item in TrafficSignIDUsa if item.name in ("STOP", "YIELD"))
STOP_LINE_NAME = re.compile(r"stop_?line", re.IGNORECASE)


@dataclasses.dataclass(slots=True)
class HeuristicsConfig:
    """Switches and parameters of the :data:`HEURISTICS`; each switch is named after its heuristic.

    Attributes:
        light_type_code: H1 on.
        light_type_pattern: H1: regular expression a traffic light's ``type`` must match in full.
        centre_line_validity: H2 on.
        facing_lanes: H3 on.
        stop_at_signal_s: H4 on.
        sign_stencil_match: H5 on.
        sign_stencil_match_m: H5: largest sign-to-stencil distance (m) that is the same control.
        sign_signal_stoplines: H6 on (off by default).
        non_vehicle_light: H7 on.
        us_sign_codes: H8 on.
        painted_stop_line: H9 on.
    """

    light_type_code: bool = True
    light_type_pattern: str = r"100\d{4}"
    centre_line_validity: bool = True
    facing_lanes: bool = True
    stop_at_signal_s: bool = True
    sign_stencil_match: bool = True
    sign_stencil_match_m: float = 10.0
    sign_signal_stoplines: bool = False
    non_vehicle_light: bool = True
    us_sign_codes: bool = True
    painted_stop_line: bool = True


@dataclasses.dataclass(slots=True, frozen=True)
class Heuristic:
    """One registered guess: its id, its name (= its :class:`HeuristicsConfig` switch) and its rule."""

    id: str
    name: str
    rule: Callable

    @property
    def doc(self) -> str:
        """The rule's docstring: assumption, evidence, and what breaks it."""
        return inspect.getdoc(self.rule) or ""


HEURISTICS: list[Heuristic] = []


def _heuristic(*, id: str, name: str) -> Callable[[Callable], Callable]:
    def register(rule: Callable) -> Callable:
        HEURISTICS.append(Heuristic(id=id, name=name, rule=rule))
        return rule

    return register


def by_id(*, heuristic_id: str) -> Heuristic:
    """The registered heuristic with this id."""
    return next(h for h in HEURISTICS if h.id == heuristic_id)


def check_config(*, config: HeuristicsConfig) -> None:
    """Raise ``ValueError`` naming the first unusable parameter."""
    if not config.sign_stencil_match_m >= 0:
        raise ValueError(f"heuristics.sign_stencil_match_m must be >= 0, got {config.sign_stencil_match_m}")
    try:
        re.compile(config.light_type_pattern)
    except re.error as exc:
        raise ValueError(f"heuristics.light_type_pattern is not a regular expression: {exc}") from exc


def record(*, heuristic_id: str, element_id: object, message: str, action: str) -> None:
    """Log one application of a heuristic as a ``kind="heuristic"`` degradation record."""
    name = by_id(heuristic_id=heuristic_id).name
    logging.getLogger(DEGRADATION_LOGGER).warning(
        "%s %s: %s",
        heuristic_id,
        name,
        message,
        extra={"kind": "heuristic", "element": heuristic_id, "id": element_id, "value": name, "action": action},
    )


@_heuristic(id="H1", name="light_type_code")
def is_traffic_light(*, signal_type: Optional[str], config: HeuristicsConfig) -> bool:
    """A ``dynamic="yes"`` signal is a traffic light only if its ``type`` is a ``100xxxx`` light code.

    Assumes: exporters mark some static signs ``dynamic="yes"``; the type code is more reliable
    than the flag. A dynamic signal with another type is converted as a static sign.

    Evidence: in the corpus every real light (about 1,500 across 22 maps, named ``*Light*`` or
    ``Signal_*``) has a type 1000001-1002101. The only dynamic signals with other types are one
    exporter's 537 static signs (US sign codes such as ``r1`` STOP, ``r2`` YIELD, or ``-1``).

    Breaks when: a source codes its traffic lights outside ``light_type_pattern``. Its lights would
    disappear from the stoplines and be listed under H1 in the conversion report; widen the
    pattern or switch the heuristic off.
    """
    return not config.light_type_code or re.fullmatch(config.light_type_pattern, signal_type or "") is not None


@_heuristic(id="H2", name="centre_line_validity")
def names_only_the_centre_line(*, validity_from: object, validity_to: object, config: HeuristicsConfig) -> bool:
    """A ``<validity>`` naming only lane 0 names no lane.

    Assumes: lane 0 is the reference line, not a lane; an exporter that writes
    ``fromLane="0" toLane="0"`` gives no lane scope, so the signal is treated as having no
    validity (H3 then decides its lanes).

    Evidence: 546 corpus signals have only this validity (one exporter's 530 static signs and 16
    lights of one map); none has another usable lane scope.

    Breaks when: a source means lane 0 as "the lanes next to the centre line" on a road with lanes
    on both sides and the signal facing both ways; H3 would then bind the lanes its orientation
    faces, not both inner lanes.
    """
    if not config.centre_line_validity:
        return False
    try:
        return float(validity_from) == 0.0 and float(validity_to) == 0.0
    except (TypeError, ValueError):
        return False


@_heuristic(id="H3", name="facing_lanes")
def faces(*, orientation: Optional[str], travels_with_s: bool, config: HeuristicsConfig) -> bool:
    """A control no lane names covers the drivable lanes that travel the way it faces.

    Assumes: OpenDRIVE ``orientation`` ``+`` means valid for traffic travelling in +s, ``-`` in -s,
    ``none`` both; without a lane scope a control applies to every drivable lane of its road in
    that direction. A control with ``<signalReference>`` s applies on the referencing roads (each
    with the reference's orientation and s), else on its own road. Only a control that no
    ``<validity>`` binds to any lane falls back to this.

    Evidence: 103 corpus lights name no usable lane: no validity and no reference (46), only
    lane 0 (16), or references without validity (41).

    Breaks when: a control on a multi-lane road governs only some of its lanes (e.g. a turn-lane
    light); the other lanes get a stopline too.
    """
    if not config.facing_lanes:
        return False
    if orientation == "+":
        return travels_with_s
    if orientation == "-":
        return not travels_with_s
    return True


@_heuristic(id="H4", name="stop_at_signal_s")
def stops_at_signal_s(*, on_junction_road: bool, config: HeuristicsConfig) -> bool:
    """Outside a junction a control's stopline sits at the signal's (or reference's) s on each lane.

    Assumes: a control on a junction's connecting road (the CARLA layout) stops traffic where the
    lane enters the junction, as before; one on an ordinary road stands where traffic stops, so
    the stopline goes at its s rather than at the start of the lane.

    Evidence: lights bound on ordinary roads exist in two corpus maps (142 references with a lane
    validity), most near the end of the road before a junction; their old stoplines were at the
    start of the road. No CARLA map binds a light on an ordinary road.

    Breaks when: a source puts a light on the road after the junction (the far-side layout) and
    binds it with a validity there; its stopline lands after the junction.
    """
    return config.stop_at_signal_s and not on_junction_road


@_heuristic(id="H5", name="sign_stencil_match")
def matches_stencil(*, distance_m: float, config: HeuristicsConfig) -> bool:
    """A stop/yield stencil of the same kind within ``sign_stencil_match_m`` of a sign is that sign's control.

    Assumes: a painted STOP/YIELD stencil and a nearby sign of the same kind are one control, and
    the stencil marks where traffic stops; the sign then needs no stopline of its own. With H6 on,
    the same holds lane by lane: a sign row is not written where a same-kind stencil lies inside
    that lane within the distance of it.

    Evidence: CARLA paints stencils a few metres from their sign, rarely on the sign's own road
    (one map: 6 of 62); 48 of 62 of its signs have a same-kind stencil within 10 m.

    Breaks when: two separate controls of the same kind stand within the distance (e.g. a stop
    sign on a side road next to another approach's stencil); one of them is taken as covered.
    """
    return config.sign_stencil_match and distance_m <= config.sign_stencil_match_m


@_heuristic(id="H6", name="sign_signal_stoplines")
def writes_sign_signal_stoplines(*, config: HeuristicsConfig) -> bool:
    """Stop and yield sign signals write their own stoplines (off by default).

    Assumes: a STOP/YIELD sign signal with no matching stencil (H5) stops traffic on the lanes it
    is bound to (validity, references, or H3), at its s outside a junction (H4). A sign's
    ``<validity>`` names lanes of the lane section its signal or reference stands in; on a
    junction road its stopline is where that lane enters the road (lane links followed across
    sections). One row per lane, also when several references or signs stop the same lane at the
    same place.

    Evidence: CARLA maps and others carry many signs without a stencil (one map 136 yield signs,
    another 39 signs and no stencil). Off until the rows it would add are reviewed.

    Breaks when: a source places signs that do not mean a stop at that spot (e.g. advance
    warnings coded as STOP/YIELD); they become stoplines.
    """
    return config.sign_signal_stoplines


@_heuristic(id="H7", name="non_vehicle_light")
def is_vehicle_light(*, has_drivable_lane: bool, config: HeuristicsConfig) -> bool:
    """A light whose road has no drivable lane on either side is a pedestrian or cyclist signal.

    Assumes: a light that no lane names, on a road with only sidewalk, shoulder or other
    non-driving lanes, controls no vehicle; it is left out of the vehicle-light count instead of
    failing validation.

    Evidence: 11 lights of one map sit on junction roads with only sidewalk and shoulder lanes.

    Breaks when: a vehicle light sits on a non-driving road next to the lanes it controls and
    names them only through a reference that failed to parse; it is not counted as missing.
    """
    return not config.non_vehicle_light or has_drivable_lane


@_heuristic(id="H8", name="us_sign_codes")
def us_sign_code(
    *, country: Optional[str], signal_type: Optional[str], subtype: Optional[str], config: HeuristicsConfig
) -> Optional[tuple[str, str]]:
    """A sign with no country whose type is the US (MUTCD) code of a stop or yield sign is read with the US table.

    Assumes: ``type="r1" subtype="1"`` with an empty country is MUTCD R1-1 (STOP); the code
    letters are case-insensitive. Only codes the US table reads as STOP or YIELD are taken; every
    other sign is read as before. Returns the upper-cased ``(type, subtype)`` to read, or ``None``
    when the rule does not apply.

    Evidence: one exporter writes US sign codes (``r1``, ``r2``, ``r6``, ``w11``...) with
    ``country=""``; the default ZAM table does not know R1-1. Reading all its codes with the US
    table turned 15 valueless R2-1 signs into -1 m/s speed limits and broke routing, hence the
    restriction to stop and yield.

    Breaks when: a source with no country uses a non-US code that spells R1-1; it becomes a STOP
    sign. CommonRoad's US table has no YIELD (R1-2), so US yield signs are not recognised.
    """
    if not config.us_sign_codes or (country or "").strip() or not signal_type:
        return None
    code_type, code_subtype = signal_type.upper(), (subtype or "").upper()
    for code in (f"{code_type}-{code_subtype}", code_type):
        if code in US_STOP_YIELD_CODES:
            return code_type, code_subtype
    return None


@_heuristic(id="H9", name="painted_stop_line")
def is_painted_stop_line(*, tag: str, name: Optional[str], element_type: Optional[str], config: HeuristicsConfig) -> bool:
    """An object of type ``stopLine`` or named ``StopLine``, or a signal of type 294 or named ``StopLine``, is a painted stop line.

    Assumes: these are the painted lines at stops. They are listed in the conversion report
    (``kind="not_converted"``) and never become, move or invent a stopline: a stop line with no
    control is an unsigned stop, and the painted position differs from where CARLA's lights stop.

    Evidence: corpus encodings: objects named ``StopLine`` (CARLA, one other exporter), objects of
    type ``stopline`` (one map, 222), signals named ``StopLine`` (one exporter, 114).

    Breaks when: a source names other markings ``StopLine``; they are listed as stop lines.
    """
    if not config.painted_stop_line:
        return False
    if tag == "signal" and element_type == "294":
        return True
    if tag == "object" and (element_type or "").lower() == "stopline":
        return True
    return STOP_LINE_NAME.fullmatch(name or "") is not None
