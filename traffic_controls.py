"""Stopline rows for traffic lights and stop/yield signs, and which of those controls got one.

A control (an OpenDRIVE ``<signal>``) is bound to lanes by the ``<validity>`` of the signal or of a
``<signalReference>`` with its id, naming lanes of the road's first lane section. Each bound lane
gets one row across it, where the lane enters (the connecting-road layout of junctions). The
guesses beyond that, for controls no validity binds, for controls on ordinary roads, and for
stop/yield signs, are :mod:`conversion_heuristics` rules H2-H7.

Every row is in the converter's local frame (``frame.to_local``), like the ``.osm``.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import re
from typing import Any, Callable, Optional

import numpy as np
from torchdrivesim.map import Stopline

import conversion_heuristics as heuristics
import conversion_report
import map_quality
import map_validation
import vocabulary

logger = logging.getLogger(__name__)

SIGN_AGENT_TYPES = {"STOP": "stop_sign", "YIELD": "yield_sign"}  # sign-table element names -> stopline agent_type


@dataclasses.dataclass(slots=True, frozen=True)
class LaneStop:
    """Where one stopline crosses one lane: the lane's inner and outer border points, ``(x, y, z)`` in file coordinates.

    ``junction_distance_m``: for a sign stop, how far along its road the lane enters a junction
    (:func:`_junction_distance`); unset (inf) for lights. ``key``: the stop position (stops with one key are one
    stopline: an approach lane's junction entry, or one lane at one s); ``lane``: the lane it stops, as the lane
    leaving its road (``("approach", road id, lane id)``), for listing lanes with several stop positions.
    """

    inner: np.ndarray
    outer: np.ndarray
    junction_distance_m: float = math.inf
    key: tuple = ()
    lane: tuple = ()


@dataclasses.dataclass(slots=True, frozen=True)
class Target:
    """Where a control applies when no lane is named: a road, the s on it and the orientation."""

    road: object
    s: float
    orientation: Optional[str]


@dataclasses.dataclass(slots=True)
class ControlStoplines:
    """The stopline rows of one conversion, the :class:`map_validation.ControlCoverage` they give, and
    per light and sign signal with rows the distance from the signal to its nearest row (``{id, location,
    distance_m}``, location = the signal)."""

    rows: list[dict]
    coverage: map_validation.ControlCoverage
    distances: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    stopline_ids: dict[str, list[str]] = dataclasses.field(default_factory=dict)  # H15: row id -> signals it stands for
    several_stops: list[dict[str, Any]] = dataclasses.field(default_factory=list)  # lanes with several stop positions


@dataclasses.dataclass(slots=True)
class _Index:
    """Lookups over the parsed OpenDRIVE and its parametric lanes."""

    planes: dict[str, object]
    signals: dict[str, tuple[object, object]]  # signal id -> (road, signal)
    references: dict[str, list[tuple[object, object]]]  # signal id -> [(road, signalReference)]
    bound: dict[str, list[tuple[object, int]]]  # signal id -> [(road, lane id)], first lane section, in parse order
    words: vocabulary.Vocabulary  # how names and codes are read (H2's lane-0 reading among them)
    roads: dict[str, object]  # road id -> road
    junctions: dict[str, object]  # junction id -> junction


def _index(*, opendrive, road_network, words: vocabulary.Vocabulary) -> _Index:
    planes = {plane.id_: plane for plane in road_network._planes}
    signals, references, bound = {}, collections.defaultdict(list), collections.defaultdict(list)
    roads = {str(road.id): road for road in opendrive.roads}
    junctions = {str(junction.id): junction for junction in opendrive.junctions}
    for road in opendrive.roads:
        for signal in road.signals:
            signals.setdefault(str(signal.id), (road, signal))
        for reference in road.signal_reference:
            references[str(reference.id)].append((road, reference))
        if not road.lanes.lane_sections:
            continue
        first = road.lanes.lane_sections[0]
        for lane in first.right_lanes + first.left_lanes:
            for signal_id in getattr(lane, "signal_references", []):
                if f"{road.id}.{first.idx}.{lane.id}.-1" in planes:
                    bound[str(signal_id)].append((road, lane.id))
    return _Index(
        planes=planes, signals=signals, references=references, bound=dict(bound), words=words, roads=roads,
        junctions=junctions,
    )


def _plane(*, index: _Index, road, section, lane_id: int):
    return index.planes.get(f"{road.id}.{section.idx}.{lane_id}.-1")


def _travels_with_s(*, plane) -> bool:
    return not plane.parametric_lanes[0].reverse


def _lane_stop(*, lane, distance: float) -> LaneStop:
    """Where a (parametric) lane is ``distance`` along it, computed exactly as the vendored lanelet vertices are."""
    elevation = lane.calc_elevation(distance)
    inner = lane.calc_border("inner", distance)[0]
    outer = lane.calc_border("outer", distance, compute_curvature=False)[0]
    return LaneStop(inner=np.array([*inner, elevation]), outer=np.array([*outer, elevation]))


def _lane_entry(*, plane) -> LaneStop:
    """Where the lane enters its section."""
    return _lane_stop(lane=plane.parametric_lanes[0], distance=0.0)


def _section_at(*, road, s: float):
    sections = road.lanes.lane_sections
    return next((sec for sec in reversed(sections) if sec.sPos <= s + 1e-9), sections[0])


def _lane_id_in(*, road, lane_id: int, section) -> int:
    """Follow lane successor links from the first lane section to ``section``."""
    for sec in road.lanes.lane_sections:
        if sec is section:
            return lane_id
        lane = sec.get_lane(lane_id)
        successor = None if lane is None or lane.link is None else lane.link.successorId
        lane_id = lane_id if successor is None else int(successor)
    return lane_id


def _junction_distance(*, road, s: float, travels_with_s: bool) -> float:
    """How far (m) from ``s`` traffic in this direction travels along ``road`` before entering a junction.

    0 on a junction's connecting road; inf when the road's end in that direction links to no junction.
    """
    if road.junction is not None:
        return 0.0
    link = road.link.successor if travels_with_s else road.link.predecessor
    if link is None or link.elementType != "junction":
        return math.inf
    return max(float(road.plan_view.length) - s, 0.0) if travels_with_s else max(s, 0.0)


def _lane_at_s(*, plane, section, s: float) -> LaneStop:
    """Where the lane is at road position ``s`` (inside ``section``)."""
    local = min(max(s - section.sPos, 0.0), plane.length)
    distance = local if _travels_with_s(plane=plane) else plane.length - local
    return _lane_stop(lane=plane, distance=distance)


def _keyed(*, stop: LaneStop, key: tuple, lane: Optional[tuple] = None) -> LaneStop:
    return dataclasses.replace(stop, key=key, lane=key if lane is None else lane)


def _heading(*, stop: LaneStop) -> float:
    """The travel heading across which ``stop`` lies (as a stopline row's ``orientation``)."""
    across = stop.inner[:2] - stop.outer[:2]
    return float(np.arctan2(across[1], across[0]) - np.pi / 2)


def _lane_id_of(*, plane) -> int:
    return int(plane.id_.split(".")[2])


def _exit_lane(*, road, section, plane) -> tuple:
    """The lane of ``plane`` (in ``section``) where it leaves ``road`` in its travel direction, as a ``lane`` key."""
    sections = road.lanes.lane_sections
    end = sections[-1] if _travels_with_s(plane=plane) else sections[0]
    lane_id = _lane_across_sections(road=road, lane_id=_lane_id_of(plane=plane), start=section, end=end)
    return ("approach", str(road.id), lane_id) if lane_id is not None else ("lane", plane.id_)


def _connecting_lanes(*, index: _Index, junction, road_id: str, lane_id: int) -> list:
    """The parametric lanes, at their entries, that ``junction`` connects lane ``lane_id`` of road ``road_id`` into."""
    found = []
    for connection in junction.connections:
        road = index.roads.get(str(connection.connectingRoad))
        if str(connection.incomingRoad) != road_id or road is None or not road.lanes.lane_sections:
            continue
        forward = connection.contactPoint != "end"
        section = road.lanes.lane_sections[0 if forward else -1]
        for link in connection.laneLinks:
            plane = _plane(index=index, road=road, section=section, lane_id=link.toId) if link.fromId == lane_id else None
            if plane is not None and _travels_with_s(plane=plane) == forward:
                found.append(plane)
    return found


def _approach_stop(*, index: _Index, junction, road, lane_id: int) -> Optional[LaneStop]:
    """The one stop of the approach lane ``lane_id`` of ``road`` (its id where the road meets ``junction``).

    It is the entry of the connecting lane that continues straightest (entry heading closest to the approach lane's
    end heading): connecting lanes leaving one approach lane may start apart or turn at once. Its
    ``junction_distance_m`` is 0.
    """
    candidates = _connecting_lanes(index=index, junction=junction, road_id=str(road.id), lane_id=lane_id)
    key = ("approach", str(road.id), lane_id)
    successor = road.link.successor
    at_end = successor is not None and successor.elementType == "junction" and str(successor.element_id) == str(junction.id)
    approach = _plane(index=index, road=road, section=road.lanes.lane_sections[-1 if at_end else 0], lane_id=lane_id)
    if approach is None:
        return None
    end_heading = _heading(stop=_lane_stop(lane=approach, distance=approach.length))
    if not candidates:
        return _keyed(stop=_lane_stop(lane=approach, distance=approach.length), key=key)

    def bend(plane) -> float:
        return abs(math.remainder(_heading(stop=_lane_entry(plane=plane)) - end_heading, math.tau))

    straightest = min(candidates, key=lambda plane: (bend(plane), plane.id_))
    return dataclasses.replace(_keyed(stop=_lane_entry(plane=straightest), key=key), junction_distance_m=0.0)


def _connecting_approach(*, index: _Index, road, section, plane) -> Optional[LaneStop]:
    """The approach stop of the lane that the junction connecting road ``road`` takes ``plane``'s lane from."""
    sections = road.lanes.lane_sections
    for connection in road.junction.connections:
        incoming = index.roads.get(str(connection.incomingRoad))
        if str(connection.connectingRoad) != str(road.id) or incoming is None:
            continue
        forward = connection.contactPoint != "end"
        entry_id = _lane_across_sections(
            road=road, lane_id=_lane_id_of(plane=plane), start=section, end=sections[0 if forward else -1]
        )
        if entry_id is None or _travels_with_s(plane=plane) != forward:
            continue
        link = next((link for link in connection.laneLinks if link.toId == entry_id), None)
        if link is not None:
            return _approach_stop(index=index, junction=road.junction, road=incoming, lane_id=link.fromId)
    return None


def _junction_ahead(*, index: _Index, road, section, plane, s: float, config: heuristics.HeuristicsConfig) -> Optional[LaneStop]:
    """The approach stop of ``plane``'s lane where ``road`` enters a junction, if it does so within H6's
    ``max_junction_distance_m`` of ``s``; ``None`` otherwise (the control is on no approach lane)."""
    forward = _travels_with_s(plane=plane)
    distance = _junction_distance(road=road, s=s, travels_with_s=forward)
    limit = config.sign_signal_stoplines.max_junction_distance_m
    lane = _exit_lane(road=road, section=section, plane=plane)
    if math.isinf(distance) or (limit is not None and distance > limit) or lane[0] != "approach":
        return None
    link = road.link.successor if forward else road.link.predecessor
    junction = index.junctions.get(str(link.element_id))
    return None if junction is None else _approach_stop(index=index, junction=junction, road=road, lane_id=lane[2])


def element_position(*, road, s: float, t: float, frame) -> tuple[float, float]:
    """Local ``(x, y)`` of a road element at ``s`` (clamped to the road) and lateral offset ``t``."""
    s = min(max(s, 0.0), road.plan_view.length)
    position, tangent, _, _ = road.plan_view.calc(s, compute_curvature=False)
    xy = np.array(position[:2]) + t * np.array([-np.sin(tangent), np.cos(tangent)])
    x, y = frame.to_local(xy[0], xy[1])
    return float(x), float(y)


def _location(*, index: _Index, signal_id: str, frame) -> Optional[tuple[float, float]]:
    """Local ``(x, y)`` of the signal ``signal_id``, if it is on a road."""
    road_signal = index.signals.get(signal_id)
    if road_signal is None:
        return None
    road, signal = road_signal
    return element_position(road=road, s=float(signal.s), t=float(signal.t), frame=frame)


def _element_s(*, index: _Index, road, signal_id: str) -> float:
    """s of the signal ``signal_id`` on ``road``, else of the road's reference to it."""
    road_signal = index.signals.get(signal_id)
    if road_signal is not None and road_signal[0] is road:
        return float(road_signal[1].s)
    return next(float(ref.s) for ref_road, ref in index.references.get(signal_id, []) if ref_road is road)


def _placed_stop(
    *, index: _Index, road, section, plane, s: float, config: heuristics.HeuristicsConfig
) -> tuple[LaneStop, bool]:
    """Where a control stops the lane of ``plane`` (in ``section``; the control stands at ``s``), and whether H4
    makes it (or, switched off, would make it) one of its approach lane's controls.

    On a junction's connecting road: the stop of the approach lane it leaves (:func:`_approach_stop`), else where
    the lane enters the road. Elsewhere with H4: at ``s``, keyed as its approach lane's when the road enters a
    junction within the limit (:func:`_junction_ahead`), so the controls of one approach lane share the position
    of the one nearest the junction (``junction_distance_m``); without H4: where the lane enters the road.
    """
    lane = _exit_lane(road=road, section=section, plane=plane)

    def entry() -> LaneStop:
        return _keyed(stop=_road_entry(index=index, road=road, section=section, plane=plane), key=("entry",) + lane, lane=lane)

    if road.junction is not None:
        return _connecting_approach(index=index, road=road, section=section, plane=plane) or entry(), False
    ahead = _junction_ahead(index=index, road=road, section=section, plane=plane, s=s, config=config)
    if not heuristics.applies(heuristic_id="H4", config=config):
        return entry(), ahead is not None
    if ahead is not None:
        distance = _junction_distance(road=road, s=s, travels_with_s=_travels_with_s(plane=plane))
        at = _lane_at_s(plane=plane, section=section, s=s)
        return dataclasses.replace(at, key=ahead.key, lane=ahead.lane, junction_distance_m=distance), True
    distance = _junction_distance(road=road, s=s, travels_with_s=_travels_with_s(plane=plane))
    at_s = _keyed(stop=_lane_at_s(plane=plane, section=section, s=s), key=("at_s", plane.id_, round(s, 3)), lane=lane)
    return dataclasses.replace(at_s, junction_distance_m=distance), False


def _bound_stops(
    *, index: _Index, signal_id: str, config: heuristics.HeuristicsConfig
) -> tuple[list[LaneStop], bool, bool]:
    """The stops of the lanes a validity binds ``signal_id`` to, whether H4 detected any of them, and whether it
    moves any to its junction entry."""
    stops, found, moved = [], False, False
    for road, lane_id in index.bound.get(signal_id, []):
        first = road.lanes.lane_sections[0]
        section, plane, s = first, _plane(index=index, road=road, section=first, lane_id=lane_id), 0.0
        detected = heuristics.stops_at_signal_s(on_junction_road=road.junction is not None)
        if detected:
            s = _element_s(index=index, road=road, signal_id=signal_id)
            at = _section_at(road=road, s=s)
            at_plane = _plane(index=index, road=road, section=at, lane_id=_lane_id_in(road=road, lane_id=lane_id, section=at))
            if at_plane is not None:
                section, plane = at, at_plane
        stop, ahead = _placed_stop(index=index, road=road, section=section, plane=plane, s=s, config=config)
        stops.append(stop)
        found, moved = found or detected, moved or ahead
    return stops, found, moved


def _has_validity(*, element) -> bool:
    """Whether a signal or reference has a ``<validity>``."""
    return getattr(element, "validity_from", None) is not None


def _names_no_lane(*, index: _Index, element) -> bool:
    """Whether the dictionary reads this signal's or reference's validity (lane 0 only) as naming no lane."""
    return vocabulary.names_no_lane(
        validity_from=getattr(element, "validity_from", None),
        validity_to=getattr(element, "validity_to", None),
        vocabulary=index.words,
    )


def _record_no_lane(*, index: _Index, signal_id: str, frame) -> None:
    vocabulary.record(
        element="signal.validity", element_id=signal_id, value="0..0", meaning="none",
        message=f"signal {signal_id}: validity names only lane 0 - read as no validity",
        location=_location(index=index, signal_id=signal_id, frame=frame),
    )


def _targets(*, index: _Index, signal_id: str, config: heuristics.HeuristicsConfig, frame) -> Optional[list[Target]]:
    """Where a control no validity binds applies (H2/H3), or ``None`` when it named lanes it did not get."""
    elements = [index.signals[signal_id]] if signal_id in index.signals else []
    references = index.references.get(signal_id, [])
    centre_only = False
    for _, element in elements + references:
        if _names_no_lane(index=index, element=element):
            centre_only = True
        elif _has_validity(element=element):
            return None
    if centre_only:
        _record_no_lane(index=index, signal_id=signal_id, frame=frame)
    chosen = references or elements
    return [Target(road=road, s=float(element.s), orientation=element.orientation) for road, element in chosen]


def _facing_stops(
    *, index: _Index, targets: list[Target], drivable_types: frozenset[str], config: heuristics.HeuristicsConfig
) -> tuple[list[LaneStop], bool, bool, bool]:
    """Stops on the drivable lanes the targets face (H3, placed per H4), whether any target road has a drivable
    lane, whether H4 detected a target off a junction with a facing lane, and whether it moves any stop to its
    junction entry."""
    stops, any_drivable, placed_at_s, moved = [], False, False, False
    for target in targets:
        road = target.road
        sections = road.lanes.lane_sections
        if not sections:
            continue
        detected = heuristics.stops_at_signal_s(on_junction_road=road.junction is not None)
        at_s = detected and heuristics.applies(heuristic_id="H4", config=config)
        candidates = []
        for section in sections:
            for lane in section.right_lanes + section.left_lanes:
                plane = _plane(index=index, road=road, section=section, lane_id=lane.id)
                if plane is None or plane.type not in drivable_types:
                    continue
                any_drivable = True
                candidates.append((section, plane))
        if at_s:
            section = _section_at(road=road, s=target.s)
            chosen = [(sec, p) for sec, p in candidates if sec is section]
        else:
            entry_sections = {True: sections[0], False: sections[-1]}
            chosen = [(sec, p) for sec, p in candidates if sec is entry_sections[_travels_with_s(plane=p)]]
        for section, plane in chosen:
            if not heuristics.faces(orientation=target.orientation, travels_with_s=_travels_with_s(plane=plane)):
                continue
            stop, ahead = _placed_stop(index=index, road=road, section=section, plane=plane, s=target.s, config=config)
            distance = _junction_distance(road=road, s=target.s, travels_with_s=_travels_with_s(plane=plane))
            stops.append(dataclasses.replace(stop, junction_distance_m=distance))
            placed_at_s, moved = placed_at_s or detected, moved or ahead
    return stops, any_drivable, placed_at_s, moved


def _lane_across_sections(*, road, lane_id: int, start, end) -> Optional[int]:
    """The id in lane section ``end`` of lane ``lane_id`` of section ``start``, following lane links (``None`` if one is missing)."""
    sections = road.lanes.lane_sections
    i, j = sections.index(start), sections.index(end)
    step = 1 if j > i else -1
    for k in range(i, j, step):
        lane = sections[k].get_lane(lane_id)
        link = None if lane is None else lane.link
        following = None if link is None else (link.successorId if step == 1 else link.predecessorId)
        if following is None:
            return None
        lane_id = int(following)
    return lane_id


def _road_entry(*, index: _Index, road, section, plane) -> LaneStop:
    """Where the lane of ``plane`` (in ``section``) enters ``road`` in its travel direction."""
    sections = road.lanes.lane_sections
    entry = sections[0] if _travels_with_s(plane=plane) else sections[-1]
    lane_id = _lane_across_sections(road=road, lane_id=int(plane.id_.split(".")[2]), start=section, end=entry)
    entry_plane = None if lane_id is None else _plane(index=index, road=road, section=entry, lane_id=lane_id)
    return _lane_entry(plane=entry_plane if entry_plane is not None else plane)


def _sign_stops(
    *, index: _Index, signal_id: str, drivable_types: frozenset[str], config: heuristics.HeuristicsConfig, frame
) -> list[LaneStop]:
    """The stops of a stop/yield sign (H6): its validity lanes read in the lane section at each element's s.

    Unlike light rows (which keep the first-section binding their existing rows were made with), a sign's
    ``<validity>`` names lanes of the lane section its signal or reference stands in; the stop is placed by
    :func:`_placed_stop` (its approach lane's stop on a junction road, H4 elsewhere). A sign no validity binds
    falls back to H2/H3.
    """
    elements = ([index.signals[signal_id]] if signal_id in index.signals else []) + index.references.get(signal_id, [])
    scoped = [
        (road, element)
        for road, element in elements
        if _has_validity(element=element) and not _names_no_lane(index=index, element=element)
    ]
    location = _location(index=index, signal_id=signal_id, frame=frame)
    if scoped and any(_names_no_lane(index=index, element=element) for _, element in elements):
        _record_no_lane(index=index, signal_id=signal_id, frame=frame)
    if not scoped:
        stops, _ = _unscoped_stops(
            index=index, signal_id=signal_id, drivable_types=drivable_types, config=config, frame=frame
        )
        return stops
    stops, found, moved = [], False, False
    for road, element in scoped:
        if not road.lanes.lane_sections:
            continue
        s = float(element.s)
        section = _section_at(road=road, s=s)
        low, high = sorted((int(element.validity_from), int(element.validity_to)))
        for lane_id in range(low, high + 1):
            plane = None if lane_id == 0 else _plane(index=index, road=road, section=section, lane_id=lane_id)
            if plane is None or plane.type not in drivable_types:
                continue
            stop, ahead = _placed_stop(index=index, road=road, section=section, plane=plane, s=s, config=config)
            found = found or heuristics.stops_at_signal_s(on_junction_road=road.junction is not None)
            moved = moved or ahead
            distance = _junction_distance(road=road, s=s, travels_with_s=_travels_with_s(plane=plane))
            stops.append(dataclasses.replace(stop, junction_distance_m=distance))
    if found:
        _record_placed_at_s(signal_id=signal_id, config=config, location=location, moved=moved)
    return stops


def _row(*, stop: LaneStop, frame, actor_id, agent_type: str, within: Callable[..., bool]) -> Optional[dict]:
    left = np.array([*frame.to_local(stop.inner[0], stop.inner[1]), *stop.inner[2:]])
    right = np.array([*frame.to_local(stop.outer[0], stop.outer[1]), *stop.outer[2:]])
    center = (left + right) / 2
    right_to_left = left - right
    x, y, z = float(center[0]), float(center[1]), float(center[2])
    if not within(x=x, y=y, z=z):
        return None
    stopline = Stopline(
        actor_id=actor_id, agent_type=agent_type, x=x, y=y,
        length=1.0, width=float(np.linalg.norm(left - right)),
        orientation=float(np.arctan2(right_to_left[1], right_to_left[0]) - (np.pi / 2)),
    )
    return dataclasses.asdict(stopline)


def _record_placed_at_s(*, signal_id: str, config: heuristics.HeuristicsConfig, location, moved: bool) -> None:
    where = "where the last control of its approach lane stands" if moved else "at its s"
    heuristics.record(
        heuristic_id="H4", element_id=signal_id, action="shares_approach_stop" if moved else "placed_at_s",
        message=f"signal {signal_id}: on an ordinary road - stopline {where}, not at the lane start",
        config=config, location=location,
    )


def _unscoped_stops(
    *, index: _Index, signal_id: str, drivable_types: frozenset[str], config: heuristics.HeuristicsConfig, frame
) -> tuple[list[LaneStop], bool]:
    """The stops H3 gives a control no validity binds, and whether any of its roads has a drivable lane.

    Without H3 (or for a control that named lanes it did not get) there are none, and the control
    counts as on a drivable road.
    """
    targets = _targets(index=index, signal_id=signal_id, config=config, frame=frame)
    if targets is None:
        return [], True
    stops, any_drivable, placed_at_s, moved = _facing_stops(
        index=index, targets=targets, drivable_types=drivable_types, config=config
    )
    location = _location(index=index, signal_id=signal_id, frame=frame)
    if not heuristics.record(
        heuristic_id="H3", element_id=signal_id, action="bound" if stops else "no_facing_lane",
        message=f"signal {signal_id}: no lane named - {len(stops)} lane(s) it faces",
        config=config, location=location,
    ):
        return [], True
    if placed_at_s:
        _record_placed_at_s(signal_id=signal_id, config=config, location=location, moved=moved)
    return stops, any_drivable


def _control_stops(
    *, index: _Index, signal_id: str, drivable_types: frozenset[str], config: heuristics.HeuristicsConfig, frame
) -> tuple[list[LaneStop], bool]:
    """The stops of one control and whether it is a vehicle control (H7 decides for unbound lights)."""
    stops, found, moved = _bound_stops(index=index, signal_id=signal_id, config=config)
    if found:
        _record_placed_at_s(
            signal_id=signal_id, config=config, location=_location(index=index, signal_id=signal_id, frame=frame),
            moved=moved,
        )
    if index.bound.get(signal_id):
        return stops, True
    return _unscoped_stops(index=index, signal_id=signal_id, drivable_types=drivable_types, config=config, frame=frame)


def _junction_phrase(*, distance_m: float) -> str:
    return "on lanes that reach no junction" if math.isinf(distance_m) else f"{distance_m:.1f} m before a junction"


def _stop_spot(*, stop: LaneStop) -> dict:
    """A stop's centre and heading in the file frame, compared with :func:`_same_spot`."""
    centre = (stop.inner[:2] + stop.outer[:2]) / 2
    return {"x": float(centre[0]), "y": float(centre[1]), "orientation": _heading(stop=stop)}


def _same_spot(*, a: dict, b: dict) -> bool:
    """Two rows across the same lane at the same place (references stacking on one lane, or two signs of one stop)."""
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < 0.1 and abs(a["orientation"] - b["orientation"]) < 1e-3


def _stencil_gap_on_lane(*, row: dict, stencils: np.ndarray) -> float:
    """Distance along travel from ``row`` to the nearest stencil inside the lane ``row`` crosses (inf if none)."""
    if not len(stencils):
        return math.inf
    heading = np.array([math.cos(row["orientation"]), math.sin(row["orientation"])])
    offsets = stencils - np.array([row["x"], row["y"]])
    across = np.abs(offsets @ np.array([-heading[1], heading[0]]))
    along = np.abs(offsets @ heading)
    inside = across <= row["width"] / 2
    return float(along[inside].min()) if inside.any() else math.inf


def _sign_kind(*, sign) -> Optional[str]:
    names = {element.traffic_sign_element_id.name for element in sign.traffic_sign_elements}
    return next((SIGN_AGENT_TYPES[name] for name in sorted(names) if name in SIGN_AGENT_TYPES), None)


def _stencil_rows(*, road_network, frame, within: Callable[..., bool]) -> list[dict]:
    """Rows of the painted stop/yield stencils inside the trim."""
    rows = []
    stencils = [("stop_sign", x) for x in road_network._iai_stop_signs] + [
        ("yield_sign", x) for x in road_network._iai_yield_signs
    ]
    for agent_type, (xy, orientation, length, width, object_id) in stencils:
        z = float(xy[2]) if len(xy) > 2 else 0.0
        x, y = (float(v) for v in frame.to_local(xy[0], xy[1]))
        if not within(x=x, y=y, z=z):
            continue
        stopline = Stopline(
            actor_id=object_id, agent_type=agent_type, x=x, y=y,
            length=1.0, width=float(width), orientation=float(orientation),
        )
        rows.append(dataclasses.asdict(stopline))
    return rows


def control_stoplines(
    *,
    opendrive,
    road_network,
    lanelet_network,
    frame,
    within: Callable[..., bool],
    drivable_types: frozenset[str],
    config: heuristics.HeuristicsConfig,
    words: vocabulary.Vocabulary,
) -> ControlStoplines:
    """Every stopline row and the coverage of lights and stop/yield signs.

    Args:
        opendrive: The parsed OpenDRIVE.
        road_network: The vendored ``Network`` after ``load_opendrive``.
        lanelet_network: The exported CommonRoad network; its traffic lights get rows, in its order.
        frame: The converter's ``CustomTransfomer``.
        within: ``within(x=, y=, z=None)``: whether a local point survives the trim (``z`` checks
            the vertical limits).
        drivable_types: OpenDRIVE lane types converted to lanelets.
        config: The heuristics settings.
        words: The vocabulary dictionary in use (how validities naming only lane 0 are read).

    Returns:
        The rows (lights, then stop/yield stencils, then sign signals) and the coverage.
    """
    index = _index(opendrive=opendrive, road_network=road_network, words=words)

    def local(position) -> np.ndarray:
        return np.array(frame.to_local(position[0], position[1]))

    def in_area(position) -> bool:
        x, y = local(position)
        return within(x=float(x), y=float(y))

    light_stops: dict[str, list[LaneStop]] = {}
    vehicle_lights: set[str] = set()
    for light in road_network._traffic_lights:
        signal_id = str(light.opendrive_id)
        if signal_id in light_stops:
            continue
        stops, is_vehicle = _control_stops(
            index=index, signal_id=signal_id, drivable_types=drivable_types, config=config, frame=frame
        )
        light_stops[signal_id] = stops
        if heuristics.is_vehicle_light(has_drivable_lane=is_vehicle) or not heuristics.record(
            heuristic_id="H7", element_id=signal_id, action="not_a_vehicle_light",
            message=f"light {signal_id}: its road has no drivable lane - not counted as a vehicle light",
            config=config, location=_location(index=index, signal_id=signal_id, frame=frame),
        ):
            vehicle_lights.add(signal_id)

    by_position: dict[tuple, list[tuple[LaneStop, str, Any]]] = {}
    keys_on_lane: dict[tuple, list[tuple]] = collections.defaultdict(list)
    for light in lanelet_network.traffic_lights:
        signal_id = str(getattr(light, "opendrive_id", ""))
        if signal_id not in vehicle_lights:
            continue
        for stop in light_stops[signal_id]:
            key = next(
                (
                    key
                    for key in keys_on_lane[stop.lane]
                    if _same_spot(a=_stop_spot(stop=by_position[key][0][0]), b=_stop_spot(stop=stop))
                ),
                stop.key,
            )
            if key not in by_position:
                keys_on_lane[stop.lane].append(key)
            at = by_position.setdefault(key, [])
            if all(other != signal_id for _, other, _ in at):
                at.append((stop, signal_id, light.opendrive_id))
    stencil_rows = _stencil_rows(road_network=road_network, frame=frame, within=within)
    stencil_xy = {
        agent_type: np.array([[r["x"], r["y"]] for r in stencil_rows if r["agent_type"] == agent_type]).reshape(-1, 2)
        for agent_type in SIGN_AGENT_TYPES.values()
    }
    signs, signs_missing = len(stencil_rows), []
    pending: list[tuple[Any, str, str, Any, list[LaneStop]]] = []
    for sign in road_network._traffic_signs:
        agent_type = _sign_kind(sign=sign)
        if agent_type is None or not in_area(sign.position):
            continue
        signs += 1
        signal_id = str(sign.opendrive_id)
        distances = np.linalg.norm(stencil_xy[agent_type] - local(sign.position), axis=1)
        nearest = float(distances.min()) if distances.size else float("inf")
        location = _location(index=index, signal_id=signal_id, frame=frame)
        covered = heuristics.matches_stencil(distance_m=nearest, config=config) and heuristics.record(
            heuristic_id="H5", element_id=signal_id, action="stencil_is_its_stopline",
            message=f"sign {signal_id}: a {agent_type} stencil {nearest:.1f} m away is its stopline",
            config=config, location=location,
        )
        if not heuristics.writes_sign_signal_stoplines(covered_by_stencil=covered):
            continue
        if not heuristics.applies(heuristic_id="H6", config=config):
            heuristics.record(
                heuristic_id="H6", element_id=signal_id, action="stopline_not_written",
                message=f"sign {signal_id}: no stencil covers it - its own {agent_type} stopline rows are not written",
                config=config, location=location,
            )
            signs_missing.append(f"signal {signal_id}")
            continue
        stops = _sign_stops(
            index=index, signal_id=signal_id, drivable_types=drivable_types, config=config, frame=frame
        )
        pending.append((sign, agent_type, signal_id, location, stops))

    all_stops = [lights[0][0] for lights in by_position.values()] + [stop for *_, stops in pending for stop in stops]
    anchors = _lane_anchors(
        index=index, stops=all_stops, limit=config.sign_signal_stoplines.max_junction_distance_m
    )
    by_position = _anchored(by_position=by_position, anchors=anchors)
    first = {stop.key: stop for stop in reversed(all_stops)}
    nearest = _nearest_stops(
        groups=[(key, [stop]) for key, stop in first.items() if anchors.get(key, key) == key]
        + [(key, [stop for stop, _, _ in lights]) for key, lights in by_position.items()]
        + [(anchors.get(stop.key, stop.key), [stop]) for *_, stops in pending for stop in stops]
    )
    rows, lit, stopline_ids, placed = _light_rows(
        by_position=by_position, nearest=nearest, frame=frame, within=within, config=config
    )
    rows += stencil_rows
    sign_rows_written: list[dict] = []
    for sign, agent_type, signal_id, location, stops in pending:
        sign_rows, by_stencil, by_sign = [], False, None
        limit = config.sign_signal_stoplines.max_junction_distance_m
        too_far, written_distances, lanes_with_stencil = [], [], 0
        for stop in stops:
            if limit is not None and stop.junction_distance_m > limit:
                too_far.append(stop.junction_distance_m)
                continue
            row = _row(
                stop=nearest.get(anchors.get(stop.key, stop.key), stop), frame=frame, actor_id=sign.opendrive_id,
                agent_type=agent_type, within=within,
            )
            if row is None or any(_same_spot(a=row, b=kept) for kept in sign_rows):
                continue
            same = next(
                (kept for kept in sign_rows_written if kept["agent_type"] == agent_type and _same_spot(a=row, b=kept)), None
            )
            if same is not None:
                by_sign = same["actor_id"]
                continue
            gap = _stencil_gap_on_lane(row=row, stencils=stencil_xy[agent_type])
            if gap <= config.sign_signal_stoplines.stencil_lane_distance_m:
                lanes_with_stencil += 1
                if heuristics.applies(heuristic_id="H5", config=config):
                    by_stencil = True
                    continue
            sign_rows.append(row)
            placed.append((stop.lane, row))
            written_distances.append(stop.junction_distance_m)
        if by_stencil and not sign_rows:
            heuristics.record(
                heuristic_id="H5", element_id=signal_id, action="stencil_is_its_stopline",
                message=f"sign {signal_id}: a {agent_type} stencil on the lane it binds is its stopline",
                config=config, location=location,
            )
            continue
        if lanes_with_stencil:
            heuristics.record(
                heuristic_id="H5", element_id=signal_id, action="stencil_on_some_lanes",
                message=f"sign {signal_id}: a {agent_type} stencil on {lanes_with_stencil} of the lanes it binds is "
                "their stopline",
                config=config, location=location,
            )
        if by_sign is not None and not sign_rows:
            heuristics.record(
                heuristic_id="H6", element_id=signal_id, action="same_stop_as_another_sign",
                message=f"sign {signal_id}: stops the same lane(s) at the same place as sign {by_sign}",
                config=config, location=location,
            )
            continue
        if sign_rows:
            rows += sign_rows
            sign_rows_written += sign_rows
            heuristics.record(
                heuristic_id="H6", element_id=signal_id, action="stopline_written",
                message=f"sign {signal_id}: {len(sign_rows)} {agent_type} stopline row(s), "
                + _junction_phrase(distance_m=min(written_distances)),
                config=config, location=location,
            )
        elif too_far:
            heuristics.record(
                heuristic_id="H6", element_id=signal_id, action="beyond_junction_limit",
                message=f"sign {signal_id}: no {agent_type} row - its lanes are "
                + _junction_phrase(distance_m=min(too_far))
                + f" (limit {limit:g} m)",
                config=config, location=location,
            )
            signs_missing.append(f"signal {signal_id}")
        else:
            signs_missing.append(f"signal {signal_id}")
    rows, placed = _light_priority(rows=rows, placed=placed, config=config)

    lights_in_area = sorted(
        {str(light.opendrive_id) for light in road_network._traffic_lights if in_area(light.position)} & vehicle_lights
    )
    lights_missing = [i for i in lights_in_area if i not in lit]
    positions = {
        **{f"signal {sign.opendrive_id}": sign.position for sign in road_network._traffic_signs},
        **{str(light.opendrive_id): light.position for light in road_network._traffic_lights},
    }
    coverage = map_validation.ControlCoverage(
        lights=len(lights_in_area),
        lights_without_stopline=lights_missing,
        stop_yield_signs=signs,
        signs_without_stopline=sorted(signs_missing),
        positions={
            element: [round(float(v), 2) for v in local(positions[element])]
            for element in lights_missing + signs_missing
            if element in positions
        },
    )
    return ControlStoplines(
        rows=rows, coverage=coverage, distances=_signal_row_distances(index=index, rows=rows, frame=frame),
        stopline_ids=stopline_ids, several_stops=_lanes_with_several_stops(placed=placed),
    )


def _id_order(*, signal_id: str) -> tuple:
    """Numeric signal ids in numeric order, before any other."""
    numeric = signal_id.lstrip("-").isdigit()
    return (0, int(signal_id), signal_id) if numeric else (1, 0, signal_id)


def _lane_anchors(*, index: _Index, stops: list[LaneStop], limit: Optional[float]) -> dict[tuple, tuple]:
    """Per stop placed at its own s (no junction anchors its lane), the key of its lane's most downstream such stop
    within ``limit`` along the road ahead of it (``None``: any distance), taken again from the next stop behind
    that group: where the controls of such a lane share one position."""
    lanes: dict[tuple, dict[tuple, float]] = collections.defaultdict(dict)
    for stop in stops:
        if stop.key and stop.key[0] == "at_s":
            s = float(stop.key[2])
            lanes[stop.lane][stop.key] = s if _travels_with_s(plane=index.planes[stop.key[1]]) else -s
    anchors: dict[tuple, tuple] = {}
    for along in lanes.values():
        remaining = sorted(along.items(), key=lambda item: item[1], reverse=True)
        while remaining:
            head_key, head = remaining[0]
            group = [item for item in remaining if limit is None or head - item[1] <= limit]
            anchors.update({key: head_key for key, _ in group})
            remaining = [item for item in remaining if item not in group]
    return anchors


def _anchored(
    *, by_position: dict[tuple, list[tuple[LaneStop, str, Any]]], anchors: dict[tuple, tuple]
) -> dict[tuple, list[tuple[LaneStop, str, Any]]]:
    """``by_position`` with each key's lights moved to its anchor's key (:func:`_lane_anchors`), one entry per light."""
    merged = {key: list(lights) for key, lights in by_position.items() if anchors.get(key, key) == key}
    for key, lights in by_position.items():
        target = anchors.get(key, key)
        if target != key:
            at = merged.setdefault(target, [])
            at.extend(entry for entry in lights if all(other != entry[1] for _, other, _ in at))
    return merged


def _nearest_stops(*, groups: list[tuple[tuple, list[LaneStop]]]) -> dict[tuple, LaneStop]:
    """Per stop key, the stop nearest the junction (least ``junction_distance_m``): where every control of that
    approach lane stops, so no stopline is moved past a control of its lane."""
    nearest: dict[tuple, LaneStop] = {}
    for key, stops in groups:
        for stop in stops:
            if key not in nearest or stop.junction_distance_m < nearest[key].junction_distance_m:
                nearest[key] = stop
    return nearest


def _light_rows(
    *, by_position: dict[tuple, list[tuple[LaneStop, str, Any]]], nearest: dict[tuple, LaneStop], frame,
    within: Callable[..., bool], config: heuristics.HeuristicsConfig,
) -> tuple[list[dict], set[str], dict[str, list[str]], list[tuple[tuple, dict]]]:
    """The light rows, one per light and stop position (``nearest``'s for its key), or with H15 one per position.

    Returns:
        The rows, the lights they cover, per row id the lights it stands for (H15 merges), and each row's lane.
    """
    rows, lit, placed = [], set(), []
    merged: dict[str, set[str]] = collections.defaultdict(set)
    for key, lights in by_position.items():
        lights = [(nearest[key], signal_id, actor_id) for _, signal_id, actor_id in lights]
        ids = [signal_id for _, signal_id, _ in lights]
        if heuristics.shares_one_row(lights_at_stop=len(ids)):
            first = min(lights, key=lambda light: _id_order(signal_id=light[1]))
            row = _row(stop=first[0], frame=frame, actor_id=first[2], agent_type="traffic_light", within=within)
            if row is None:
                continue
            others = sorted(ids, key=lambda i: _id_order(signal_id=i))
            if heuristics.record(
                heuristic_id="H15", element_id=first[1], action="shared_id",
                message=f"lights {', '.join(others)} stop one approach lane - one stopline row with id {first[1]}",
                config=config, location=(row["x"], row["y"]),
            ):
                lights = [first]
                merged[first[1]].update(ids)
                lit.update(ids)
        for stop, signal_id, actor_id in lights:
            row = _row(stop=stop, frame=frame, actor_id=actor_id, agent_type="traffic_light", within=within)
            if row is not None:
                rows.append(row)
                lit.add(signal_id)
                placed.append((stop.lane, row))
    stopline_ids = {
        row_id: sorted(ids, key=lambda i: _id_order(signal_id=i))
        for row_id, ids in sorted(merged.items(), key=lambda item: _id_order(signal_id=item[0]))
    }
    return rows, lit, stopline_ids, placed


def _light_priority(
    *, rows: list[dict], placed: list[tuple[tuple, dict]], config: heuristics.HeuristicsConfig
) -> tuple[list[dict], list[tuple[tuple, dict]]]:
    """The rows and placed rows without the stop and yield rows a traffic light's row stops on the same lane (H16)."""
    lights = [r for r in rows if r["agent_type"] == "traffic_light"]
    dropped: set[int] = set()
    for row in rows:
        if row["agent_type"] == "traffic_light":
            continue
        same_way = [light for light in lights if math.cos(light["orientation"] - row["orientation"]) > 0]
        xy = np.array([[light["x"], light["y"]] for light in same_way]).reshape(-1, 2)
        gap = _stencil_gap_on_lane(row=row, stencils=xy)
        if heuristics.light_takes_priority(gap_m=gap, config=config) and heuristics.record(
            heuristic_id="H16", element_id=row["actor_id"], action="light_takes_priority",
            message=f"{row['agent_type']} {row['actor_id']}: a traffic light's stopline {gap:.1f} m along its lane "
            "stops it - no row",
            config=config, location=(row["x"], row["y"]),
        ):
            dropped.add(id(row))
    return [r for r in rows if id(r) not in dropped], [(lane, r) for lane, r in placed if id(r) not in dropped]


def _lanes_with_several_stops(*, placed: list[tuple[tuple, dict]]) -> list[dict[str, Any]]:
    """Lanes whose rows (of any control type) lie at more than one position, for AutoCritic to list."""
    positions: dict[tuple, dict[tuple[float, float], None]] = collections.defaultdict(dict)
    for lane, row in placed:
        if lane:
            positions[lane][(round(row["x"], 2), round(row["y"], 2))] = None
    listed = []
    for lane, places in positions.items():
        if len(places) < 2:
            continue
        points = list(places)
        span = max(math.dist(a, b) for a in points for b in points)
        name = f"road {lane[1]} lane {lane[2]}" if lane[0] == "approach" else f"lane {lane[1]}"
        listed.append({
            "id": name, "location": list(points[0]),
            "detail": f"{len(points)} stop positions on one lane, up to {span:.1f} m apart",
        })
    return listed


def _signal_row_distances(*, index: _Index, rows: list[dict], frame) -> list[dict[str, Any]]:
    """Per signal with rows, the distance (m) from the signal to its nearest row."""
    by_actor: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for row in rows:
        by_actor[str(row["actor_id"])].append((row["x"], row["y"]))
    distances = []
    for signal_id, centres in sorted(by_actor.items()):
        location = _location(index=index, signal_id=signal_id, frame=frame)
        if location is None:
            continue
        nearest = min(math.dist(location, centre) for centre in centres)
        distances.append(
            {"id": f"signal {signal_id}", "location": [round(v, 2) for v in location], "distance_m": round(nearest, 2)}
        )
    return distances


def is_painted_stop_line(
    *, tag: str, name: Optional[str], element_type: Optional[str], words: vocabulary.Vocabulary
) -> bool:
    """Whether an object or signal is a painted stop line, by the dictionary's ``markings.painted_stop_lines``.

    Painted stop lines never become, move or invent a stopline (a stop line with no control is an
    unsigned stop, and the painted position differs from where CARLA's lights stop); AutoCritic
    lists them as not converted.
    """
    painted = words.markings.painted_stop_lines
    if tag == "signal" and element_type in painted.signal_types:
        return True
    if tag == "object" and (element_type or "").lower() in painted.object_types:
        return True
    return any(re.fullmatch(pattern, name or "", re.IGNORECASE) for pattern in painted.names)


def report_painted_stop_lines(*, opendrive, frame, words: vocabulary.Vocabulary) -> list[dict[str, Any]]:
    """List every painted stop line in the conversion report; none becomes a stopline row.

    Returns:
        Each one as ``{id, location, detail}`` (local frame).
    """
    degradation = logging.getLogger(conversion_report.DEGRADATION_LOGGER)
    listed = []
    for road in opendrive.roads:
        elements = [("object", obj, obj.type) for obj in road.objects] + [("signal", sig, sig.type) for sig in road.signals]
        for tag, element, element_type in elements:
            if not is_painted_stop_line(tag=tag, name=element.name, element_type=element_type, words=words):
                continue
            position = element_position(road=road, s=float(element.s), t=float(element.t), frame=frame)
            x, y = (round(v, 2) for v in position)
            degradation.warning(
                "%s %s: painted stop line at (%.2f, %.2f) - not converted",
                tag, element.id, x, y,
                extra={"kind": "not_converted", "element": tag, "id": str(element.id), "value": f"({x}, {y})",
                       "action": "listed"},
            )
            listed.append(map_quality.fault(element=f"{tag} {element.id}", location=(x, y), detail="painted stop line"))
    if listed:
        logger.info(f"{len(listed)} painted stop line(s) listed in the conversion report; none becomes a stopline")
    return listed
