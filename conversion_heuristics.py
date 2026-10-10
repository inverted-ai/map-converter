"""Every rule the converter applies that guesses what an OpenDRIVE author meant.

OpenDRIVE leaves room for interpretation and exporters fill it differently, so some conversions
need a guess. Each guess lives here as one :class:`Heuristic`: a stable id (``H1``, ``H2``, ...), a
concise name that is also its :class:`HeuristicsConfig` entry, and the docstring of its rule
function, which states the assumption, the evidence behind it and the kind of map that breaks it.
When a map breaks one, switch it off or retune it (``autofix.heuristics.<name>.enabled=false`` on
the ``convert_map.py`` command line, or in its ``config=`` file) rather than adding a special case
elsewhere. ``autofix.enabled=false`` switches every heuristic off at once (:class:`AutoFixConfig`).

Each heuristic is split into detection and application. Its rule function only detects: it says
whether the situation it fixes is present (using the heuristic's parameters, never its switch).
Every detection is logged as a ``kind="heuristic"`` degradation record (:func:`record`) whose
``element`` is the heuristic's id, with ``applied`` (its switch, after ``autofix.enabled``) and the
``location`` in the converter's local frame; the converter acts on it only when ``applied`` is
true. ``<map>_conversion_report.json`` therefore lists both what AutoFix changed against
BasicConversion (the conversion with every heuristic off) and what a disabled heuristic would have
changed. A detection is made where the conversion reaches the heuristic's decision: with one
heuristic off, those that would have acted on its result are not reached.

Guesses that are not configurable here yet:

- ``crdesigner/.../opendrive_conversion/network.py`` ``Network._stop_lines_from_road``: the object
  names read as painted stop markings (``Stencil_STOP``) and yield markings (``Stencil_GiveWay``,
  ``Stencil_Yield_Triangles``).
- ``crdesigner/.../plane_elements/traffic_signals.py`` ``assign_traffic_signals_to_road``: a sign
  whose country has no sign table is read with the ``ZAM`` table; dynamic types ``1000002`` and
  static ``1000003``/``1000004`` are skipped; German type ``252`` is read as ``260``.
- ``traffic_light_fsm.py`` ``FsmConfig`` (its own command-line flags): how lights are grouped into
  intersections and phases.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import re
from typing import Callable, Optional, Sequence

from commonroad.scenario.traffic_sign import TrafficSignIDUsa

from conversion_report import DEGRADATION_LOGGER

US_STOP_YIELD_CODES = frozenset(item.value.upper() for item in TrafficSignIDUsa if item.name in ("STOP", "YIELD"))


@dataclasses.dataclass(slots=True)
class Switch:
    """A heuristic with no parameter: whether it is applied."""

    enabled: bool = True


@dataclasses.dataclass(slots=True)
class LightTypeCodeConfig:
    """H1.

    Attributes:
        enabled: Whether it is applied.
        pattern: Regular expression a traffic light's ``type`` must match in full.
    """

    enabled: bool = True
    pattern: str = r"100\d{4}"


@dataclasses.dataclass(slots=True)
class SignStencilMatchConfig:
    """H5.

    Attributes:
        enabled: Whether it is applied.
        max_distance_m: Largest sign-to-stencil distance (m) that is the same control.
    """

    enabled: bool = True
    max_distance_m: float = 10.0


@dataclasses.dataclass(slots=True)
class SignSignalStoplinesConfig:
    """H6.

    Attributes:
        enabled: Whether it is applied (off by default).
        max_junction_distance_m: A sign row is written only on a lane that enters a junction within
            this distance (m) of the sign, along the road; ``None`` writes it whatever the distance.
            30 m: on the corpus fast tier every H6 sign's nearest row is either within 3.9 m of the
            junction (273 of 313 signs) or more than 95 m from it or on a lane reaching none (mid-road
            rows, decision 77.7); stop and yield signs stand at the approach.
        stencil_lane_distance_m: A lane gets no sign row where a same-kind stencil lies within this
            distance (m) along it of the row (the stencil is the stop).
    """

    enabled: bool = False
    max_junction_distance_m: Optional[float] = 30.0
    stencil_lane_distance_m: float = 10.0


@dataclasses.dataclass(slots=True)
class SeamToleranceConfig:
    """H10.

    Attributes:
        enabled: Whether it is applied.
        max_gap_m: Largest distance (m) between the ends of linked lanelets that are made one seam.
    """

    enabled: bool = True
    max_gap_m: float = 1.0


@dataclasses.dataclass(slots=True)
class SignalPastRoadEndConfig:
    """H14.

    Attributes:
        enabled: Whether it is applied.
        max_overshoot_m: Largest distance (m) past the road end at which a signal is placed at the
            end instead of dropped.
    """

    enabled: bool = True
    max_overshoot_m: float = 1.0


@dataclasses.dataclass(slots=True)
class FsmDistanceClustersConfig:
    """H11.

    Attributes:
        enabled: Whether it is applied.
        max_distance_m: Largest distance (m) linking two traffic lights into one intersection.
    """

    enabled: bool = True
    max_distance_m: float = 50.0


@dataclasses.dataclass(slots=True)
class FsmAxisSplitConfig:
    """H12.

    Attributes:
        enabled: Whether it is applied.
        min_angle_rad: Smallest angular gap (radians, on ``[0, pi)``) between stopline axes that
            splits an intersection into two phases.
    """

    enabled: bool = True
    min_angle_rad: float = 0.5


@dataclasses.dataclass(slots=True)
class FsmTopologyRefinementConfig:
    """H13.

    Attributes:
        enabled: Whether it is applied.
        max_hops: Hops over the routing graph within which two lights' lanelets must meet to stay
            in one intersection.
    """

    enabled: bool = True
    max_hops: int = 3


@dataclasses.dataclass(slots=True)
class HeuristicsConfig:
    """Settings of the :data:`HEURISTICS`, one entry per heuristic, named after it."""

    light_type_code: LightTypeCodeConfig = dataclasses.field(default_factory=LightTypeCodeConfig)
    centre_line_validity: Switch = dataclasses.field(default_factory=Switch)
    facing_lanes: Switch = dataclasses.field(default_factory=Switch)
    stop_at_signal_s: Switch = dataclasses.field(default_factory=Switch)
    sign_stencil_match: SignStencilMatchConfig = dataclasses.field(default_factory=SignStencilMatchConfig)
    sign_signal_stoplines: SignSignalStoplinesConfig = dataclasses.field(default_factory=SignSignalStoplinesConfig)
    non_vehicle_light: Switch = dataclasses.field(default_factory=Switch)
    us_sign_codes: Switch = dataclasses.field(default_factory=Switch)
    seam_tolerance: SeamToleranceConfig = dataclasses.field(default_factory=SeamToleranceConfig)
    fsm_distance_clusters: FsmDistanceClustersConfig = dataclasses.field(default_factory=FsmDistanceClustersConfig)
    fsm_axis_split: FsmAxisSplitConfig = dataclasses.field(default_factory=FsmAxisSplitConfig)
    fsm_topology_refinement: FsmTopologyRefinementConfig = dataclasses.field(
        default_factory=FsmTopologyRefinementConfig
    )
    signal_past_road_end: SignalPastRoadEndConfig = dataclasses.field(default_factory=SignalPastRoadEndConfig)


@dataclasses.dataclass(slots=True)
class AutoFixConfig:
    """Whether the converter guesses the author's intent, and how.

    Attributes:
        enabled: Apply the enabled heuristics; off applies none, whatever their settings.
        heuristics: Each heuristic's settings.
    """

    enabled: bool = True
    heuristics: HeuristicsConfig = dataclasses.field(default_factory=HeuristicsConfig)


def applied_heuristics(*, autofix: AutoFixConfig) -> HeuristicsConfig:
    """The heuristic settings the conversion uses: ``autofix.heuristics``, every one disabled when AutoFix is off."""
    if autofix.enabled:
        return autofix.heuristics
    return HeuristicsConfig(
        **{
            f.name: dataclasses.replace(getattr(autofix.heuristics, f.name), enabled=False)
            for f in dataclasses.fields(HeuristicsConfig)
        }
    )


@dataclasses.dataclass(slots=True, frozen=True)
class Heuristic:
    """One registered guess: its id, its name (= its :class:`HeuristicsConfig` entry) and its rule."""

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
    prefix = "autofix.heuristics"
    if not config.sign_stencil_match.max_distance_m >= 0:
        raise ValueError(
            f"{prefix}.sign_stencil_match.max_distance_m must be >= 0, got {config.sign_stencil_match.max_distance_m}"
        )
    for name, value in (
        ("seam_tolerance.max_gap_m", config.seam_tolerance.max_gap_m),
        ("sign_signal_stoplines.stencil_lane_distance_m", config.sign_signal_stoplines.stencil_lane_distance_m),
        ("sign_signal_stoplines.max_junction_distance_m", config.sign_signal_stoplines.max_junction_distance_m or 0.0),
        ("fsm_distance_clusters.max_distance_m", config.fsm_distance_clusters.max_distance_m),
        ("fsm_axis_split.min_angle_rad", config.fsm_axis_split.min_angle_rad),
        ("fsm_topology_refinement.max_hops", config.fsm_topology_refinement.max_hops),
        ("signal_past_road_end.max_overshoot_m", config.signal_past_road_end.max_overshoot_m),
    ):
        if not value >= 0:
            raise ValueError(f"{prefix}.{name} must be >= 0, got {value}")
    try:
        re.compile(config.light_type_code.pattern)
    except re.error as exc:
        raise ValueError(f"{prefix}.light_type_code.pattern is not a regular expression: {exc}") from exc


def applies(*, heuristic_id: str, config: HeuristicsConfig) -> bool:
    """Whether the heuristic acts: its entry in ``config`` is enabled."""
    return getattr(config, by_id(heuristic_id=heuristic_id).name).enabled


def record(
    *,
    heuristic_id: str,
    element_id: object,
    message: str,
    action: str,
    config: HeuristicsConfig,
    location: Optional[Sequence[float]] = None,
) -> bool:
    """Log one detection of a heuristic as a ``kind="heuristic"`` degradation record.

    Args:
        heuristic_id: The heuristic's id.
        element_id: The OpenDRIVE (or lanelet) id it concerns.
        message: What the heuristic does there.
        action: Short name of what it does.
        config: The settings in use; they decide whether it is applied.
        location: ``(x, y)`` in the converter's local frame, if known.

    Returns:
        Whether the heuristic is applied (:func:`applies`).
    """
    name = by_id(heuristic_id=heuristic_id).name
    applied = applies(heuristic_id=heuristic_id, config=config)
    logging.getLogger(DEGRADATION_LOGGER).warning(
        "%s %s%s: %s",
        heuristic_id,
        name,
        "" if applied else " (not applied)",
        message,
        extra={
            "kind": "heuristic",
            "element": heuristic_id,
            "id": element_id,
            "value": name,
            "action": action,
            "applied": applied,
            "location": None if location is None else [round(float(v), 2) for v in location[:2]],
        },
    )
    return applied


@_heuristic(id="H1", name="light_type_code")
def is_traffic_light(*, signal_type: Optional[str], config: HeuristicsConfig) -> bool:
    """A ``dynamic="yes"`` signal is a traffic light only if its ``type`` is a ``100xxxx`` light code.

    Assumes: exporters mark some static signs ``dynamic="yes"``; the type code is more reliable
    than the flag. A dynamic signal with another type is converted as a static sign.

    Evidence: in the corpus every real light (about 1,500 across 22 maps, named ``*Light*`` or
    ``Signal_*``) has a type 1000001-1002101. The only dynamic signals with other types are one
    exporter's 537 static signs (US sign codes such as ``r1`` STOP, ``r2`` YIELD, or ``-1``).

    Breaks when: a source codes its traffic lights outside ``light_type_code.pattern``. Its lights would
    disappear from the stoplines and be listed under H1 in the conversion report; widen the
    pattern or switch the heuristic off.
    """
    return re.fullmatch(config.light_type_code.pattern, signal_type or "") is not None


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
    return not on_junction_road


@_heuristic(id="H5", name="sign_stencil_match")
def matches_stencil(*, distance_m: float, config: HeuristicsConfig) -> bool:
    """A stop/yield stencil of the same kind within ``max_distance_m`` of a sign is that sign's control.

    Assumes: a painted STOP/YIELD stencil and a nearby sign of the same kind are one control, and
    the stencil marks where traffic stops; the sign then needs no stopline of its own. With H6 on,
    the same holds lane by lane: a sign row is not written where a same-kind stencil lies inside
    that lane within the distance of it.

    Evidence: CARLA paints stencils a few metres from their sign, rarely on the sign's own road
    (one map: 6 of 62); 48 of 62 of its signs have a same-kind stencil within 10 m.

    Breaks when: two separate controls of the same kind stand within the distance (e.g. a stop
    sign on a side road next to another approach's stencil); one of them is taken as covered.
    """
    return distance_m <= config.sign_stencil_match.max_distance_m


@_heuristic(id="H6", name="sign_signal_stoplines")
def writes_sign_signal_stoplines(*, covered_by_stencil: bool) -> bool:
    """Stop and yield sign signals no stencil covers (H5) write their own stoplines (off by default).

    Assumes: a STOP/YIELD sign signal with no matching stencil (H5) stops traffic on the lanes it
    is bound to (validity, references, or H3), at its s outside a junction (H4). A sign's
    ``<validity>`` names lanes of the lane section its signal or reference stands in; on a
    junction road its stopline is where that lane enters the road (lane links followed across
    sections). One row per lane, also when several references or signs stop the same lane at the
    same place; none on a lane that enters a junction farther than ``max_junction_distance_m`` along
    the road (when set), or where a same-kind stencil lies within ``stencil_lane_distance_m`` along it.

    Evidence: CARLA maps and others carry many signs without a stencil (one map 136 yield signs,
    another 39 signs and no stencil). Off until the rows it would add are reviewed.

    Breaks when: a source places signs that do not mean a stop at that spot (e.g. advance
    warnings coded as STOP/YIELD); they become stoplines.
    """
    return not covered_by_stencil


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
    return has_drivable_lane


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
    if (country or "").strip() or not signal_type:
        return None
    code_type, code_subtype = signal_type.upper(), (subtype or "").upper()
    for code in (f"{code_type}-{code_subtype}", code_type):
        if code in US_STOP_YIELD_CODES:
            return code_type, code_subtype
    return None



READER_S_TOLERANCE_M = 0.01
READER_S_TOLERANCE_FRACTION = 0.01
BASIC_SEAM_TOLERANCE_M = 0.01


@_heuristic(id="H10", name="seam_tolerance")
def joins_seam(*, gap_m: float, config: HeuristicsConfig) -> bool:
    """Linked lanelets whose ends lie up to ``max_gap_m`` apart share their end nodes: one seam.

    Assumes: a predecessor/successor link in the OpenDRIVE means the lanes meet; ends a little apart
    are exporter rounding or geometry error, and Lanelet2 routes only across shared end nodes.
    BasicConversion shares the nodes of ends within ``BASIC_SEAM_TOLERANCE_M`` (1 cm) only. The gap
    is the larger of the left and right boundary gaps of a predecessor/successor pair of CommonRoad
    lanelets.

    Evidence: the converter shared linked ends up to 1 m apart before AutoFix existed (decision 85
    keeps that as the default); on the CARLA maps BasicConversion (1 cm) gives the same light rows
    and equal or better routing (probe 2026-10-09).

    Breaks when: lanes the OpenDRIVE links really are apart (a lane that starts later, a loop); the
    seam drags one lanelet's boundary ends onto the other's, folding its shape by up to the gap.
    """
    return BASIC_SEAM_TOLERANCE_M < gap_m <= config.seam_tolerance.max_gap_m


def shared_node_tolerance(*, config: HeuristicsConfig) -> float:
    """How far apart (m) linked lanelet ends may be and still share nodes: H10's ``max_gap_m`` when applied."""
    return config.seam_tolerance.max_gap_m if config.seam_tolerance.enabled else BASIC_SEAM_TOLERANCE_M


@_heuristic(id="H11", name="fsm_distance_clusters")
def merges_intersections(*, declared_count: int) -> bool:
    """Traffic lights closer than ``max_distance_m`` are one intersection (``traffic_light_fsm.py``).

    Detects a distance cluster that joins ``declared_count`` > 1 declared intersections (an
    OpenDRIVE junction's lights, or a light no junction claims). Two declared junctions are never
    joined; BasicConversion keeps each junction, and each light no junction claims, alone.

    Assumes: lights a few tens of metres apart that the OpenDRIVE does not group belong to one
    intersection (exporters often leave junction membership out); single linkage, shortest link
    first.

    Evidence: with H12 it reproduces the 8 hand-authored intersections of CARLA Town15
    (``test_town15_matches_manual_groupings``).

    Breaks when: two separate intersections stand closer than the distance (they share one cycle),
    or one intersection is spread wider (it is split).
    """
    return declared_count > 1


@_heuristic(id="H12", name="fsm_axis_split")
def changes_phases(*, axis_groups: list[list[int]], declared_groups: list[list[int]]) -> bool:
    """An intersection's lights are split into at most two phases by stopline axis (``traffic_light_fsm.py``).

    Detects an intersection whose axis-split phases hold other lights together than the phases
    the OpenDRIVE declares (one per ``<controller>``, in ``sequence`` order, and one per light no
    controller names), which BasicConversion uses. Phase order alone is not a difference.

    Assumes: lights whose stoplines run along the same axis (within ``min_angle_rad``) serve
    opposing approaches of one road and go green together; the crossing road gets the other phase.

    Evidence: reproduces the phases of the 8 hand-authored intersections of CARLA Town15, where
    the controllers hold one light each (``test_town15_matches_manual_groupings``).

    Breaks when: an intersection needs protected turns or more than two phases, or its roads
    cross at a shallow angle (below the split angle they share one phase).
    """
    return sorted(map(sorted, axis_groups)) != sorted(map(sorted, declared_groups))


@_heuristic(id="H13", name="fsm_topology_refinement")
def splits_intersection(*, parts: int) -> bool:
    """A low-certainty distance cluster is split where its lights' lanelets do not meet within ``max_hops``.

    Detects a cluster (certainty below ``FsmConfig.certainty_threshold``) whose lights fall into
    ``parts`` > 1 groups that the routing graph does not connect within the hops. Only distance
    clusters (H11) are refined; declared junctions never are.

    Assumes: lights of one intersection control lanes that lead into a common junction within a
    few lanelets; lights whose lanes never meet belong to different intersections.

    Evidence: the FSM tool's answer to over-merging that one distance cannot detect (closely
    spaced intersections); its corpus applications are counted in the FSM review files.

    Breaks when: the ``.osm`` lacks the routing links of the junction (a light's lanelet is
    isolated), so lights of one intersection are split apart.
    """
    return parts > 1


@_heuristic(id="H14", name="signal_past_road_end")
def road_end_placement(*, s: float, road_length: float, config: HeuristicsConfig) -> Optional[str]:
    """A signal whose ``s`` lies past its road's end is moved to the end if it is close, else dropped.

    Returns ``None`` for a signal the OpenDRIVE reader can place (``s`` at most
    ``READER_S_TOLERANCE_M`` + ``READER_S_TOLERANCE_FRACTION`` of the planView length past the
    end: the reader's own rounding tolerance), ``"placed_at_road_end"`` within ``max_overshoot_m``,
    else ``"dropped"``. The reader cannot place any other signal, so BasicConversion refuses the map.

    Assumes: a signal a little past the end of its road belongs at the end (the author measured
    ``s`` along a slightly longer road); one far past the end has a wrong ``s`` or road, so where
    it belongs is unknown.

    Evidence: 6 corpus maps crashed on such signals (0.36-518 m past the end); in 2 of them every
    such signal is within 1 m.

    Breaks when: a signal far past the end was meant for the road's end (it is dropped), or one
    within ``max_overshoot_m`` belongs on the next road (it controls the wrong road's lanes).
    """
    overshoot = s - road_length
    if overshoot <= READER_S_TOLERANCE_M + READER_S_TOLERANCE_FRACTION * road_length:
        return None
    return "placed_at_road_end" if overshoot <= config.signal_past_road_end.max_overshoot_m else "dropped"
