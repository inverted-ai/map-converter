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
from typing import Callable, Optional

import numpy as np
from torchdrivesim.map import Stopline

import conversion_heuristics as heuristics
import conversion_report
import map_validation

logger = logging.getLogger(__name__)

SIGN_AGENT_TYPES = {"STOP": "stop_sign", "YIELD": "yield_sign"}  # sign-table element names -> stopline agent_type


@dataclasses.dataclass(slots=True, frozen=True)
class LaneStop:
    """Where one stopline crosses one lane: the lane's inner and outer border points, ``(x, y, z)`` in file coordinates."""

    inner: np.ndarray
    outer: np.ndarray


@dataclasses.dataclass(slots=True, frozen=True)
class Target:
    """Where a control applies when no lane is named: a road, the s on it and the orientation."""

    road: object
    s: float
    orientation: Optional[str]


@dataclasses.dataclass(slots=True)
class ControlStoplines:
    """The stopline rows of one conversion and the :class:`map_validation.ControlCoverage` they give."""

    rows: list[dict]
    coverage: map_validation.ControlCoverage


@dataclasses.dataclass(slots=True)
class _Index:
    """Lookups over the parsed OpenDRIVE and its parametric lanes."""

    planes: dict[str, object]
    signals: dict[str, tuple[object, object]]  # signal id -> (road, signal)
    references: dict[str, list[tuple[object, object]]]  # signal id -> [(road, signalReference)]
    bound: dict[str, list[tuple[object, int]]]  # signal id -> [(road, lane id)], first lane section, in parse order


def _index(*, opendrive, road_network) -> _Index:
    planes = {plane.id_: plane for plane in road_network._planes}
    signals, references, bound = {}, collections.defaultdict(list), collections.defaultdict(list)
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
    return _Index(planes=planes, signals=signals, references=references, bound=dict(bound))


def _plane(*, index: _Index, road, section, lane_id: int):
    return index.planes.get(f"{road.id}.{section.idx}.{lane_id}.-1")


def _travels_with_s(*, plane) -> bool:
    return not plane.parametric_lanes[0].reverse


def _lane_entry(*, plane) -> LaneStop:
    """Where the lane enters its section, computed exactly as the vendored lanelet vertices are."""
    first = plane.parametric_lanes[0]
    elevation = first.calc_elevation(0.0)
    inner = first.calc_border("inner", 0.0)[0]
    outer = first.calc_border("outer", 0.0, compute_curvature=False)[0]
    return LaneStop(inner=np.array([*inner, elevation]), outer=np.array([*outer, elevation]))


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


def _lane_at_s(*, plane, section, s: float) -> LaneStop:
    """Where the lane is at road position ``s`` (inside ``section``)."""
    local = min(max(s - section.sPos, 0.0), plane.length)
    distance = local if _travels_with_s(plane=plane) else plane.length - local
    elevation = plane.calc_elevation(distance)
    inner = plane.calc_border("inner", distance)[0]
    outer = plane.calc_border("outer", distance, compute_curvature=False)[0]
    return LaneStop(inner=np.array([*inner, elevation]), outer=np.array([*outer, elevation]))


def _element_s(*, index: _Index, road, signal_id: str) -> float:
    """s of the signal ``signal_id`` on ``road``, else of the road's reference to it."""
    road_signal = index.signals.get(signal_id)
    if road_signal is not None and road_signal[0] is road:
        return float(road_signal[1].s)
    return next(float(ref.s) for ref_road, ref in index.references.get(signal_id, []) if ref_road is road)


def _bound_stops(
    *, index: _Index, signal_id: str, config: heuristics.HeuristicsConfig
) -> tuple[list[LaneStop], bool]:
    """The stops of the lanes a validity binds ``signal_id`` to, and whether H4 placed any of them."""
    stops, moved = [], False
    for road, lane_id in index.bound.get(signal_id, []):
        first = road.lanes.lane_sections[0]
        if not heuristics.stops_at_signal_s(on_junction_road=road.junction is not None, config=config):
            stops.append(_lane_entry(plane=_plane(index=index, road=road, section=first, lane_id=lane_id)))
            continue
        s = _element_s(index=index, road=road, signal_id=signal_id)
        section = _section_at(road=road, s=s)
        plane = _plane(index=index, road=road, section=section, lane_id=_lane_id_in(road=road, lane_id=lane_id, section=section))
        if plane is None:
            plane, section = _plane(index=index, road=road, section=first, lane_id=lane_id), first
        stops.append(_lane_at_s(plane=plane, section=section, s=s))
        moved = True
    return stops, moved


def _scoped(*, element) -> tuple[bool, bool]:
    """(has a validity, the validity names only the centre line) of a signal or reference."""
    validity_from = getattr(element, "validity_from", None)
    validity_to = getattr(element, "validity_to", None)
    return validity_from is not None, validity_from is not None and validity_to is not None and (
        float(validity_from) == 0.0 and float(validity_to) == 0.0
    )


def _targets(*, index: _Index, signal_id: str, config: heuristics.HeuristicsConfig) -> Optional[list[Target]]:
    """Where a control no validity binds applies (H2/H3), or ``None`` when it named lanes it did not get."""
    elements = [index.signals[signal_id]] if signal_id in index.signals else []
    references = index.references.get(signal_id, [])
    centre_only = False
    for _, element in elements + references:
        scoped, centre = _scoped(element=element)
        if centre and heuristics.names_only_the_centre_line(
            validity_from=element.validity_from, validity_to=element.validity_to, config=config
        ):
            centre_only = True
        elif scoped:
            return None
    if centre_only:
        heuristics.record(
            heuristic_id="H2", element_id=signal_id, action="no_lane_scope",
            message=f"signal {signal_id}: validity names only lane 0 - treated as no validity",
        )
    chosen = references or elements
    return [Target(road=road, s=float(element.s), orientation=element.orientation) for road, element in chosen]


def _facing_stops(
    *, index: _Index, targets: list[Target], drivable_types: frozenset[str], config: heuristics.HeuristicsConfig
) -> tuple[list[LaneStop], bool]:
    """Stops on the drivable lanes the targets face (H3, placed per H4), and whether any target road has a drivable lane."""
    stops, any_drivable = [], False
    for target in targets:
        road = target.road
        sections = road.lanes.lane_sections
        if not sections:
            continue
        at_s = heuristics.stops_at_signal_s(on_junction_road=road.junction is not None, config=config)
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
            if not heuristics.faces(orientation=target.orientation, travels_with_s=_travels_with_s(plane=plane), config=config):
                continue
            stops.append(_lane_at_s(plane=plane, section=section, s=target.s) if at_s else _lane_entry(plane=plane))
    return stops, any_drivable


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
    *, index: _Index, signal_id: str, drivable_types: frozenset[str], config: heuristics.HeuristicsConfig
) -> list[LaneStop]:
    """The stops of a stop/yield sign (H6): its validity lanes read in the lane section at each element's s.

    Unlike light rows (which keep the first-section binding their existing rows were made with), a sign's
    ``<validity>`` names lanes of the lane section its signal or reference stands in; on a junction road the
    stop is where that lane enters the road, elsewhere at the element's s (H4). A sign no validity binds
    falls back to H2/H3.
    """
    elements = ([index.signals[signal_id]] if signal_id in index.signals else []) + index.references.get(signal_id, [])
    scoped = [
        (road, element)
        for road, element in elements
        if _scoped(element=element)[0]
        and not heuristics.names_only_the_centre_line(
            validity_from=element.validity_from, validity_to=element.validity_to, config=config
        )
    ]
    if not scoped:
        targets = _targets(index=index, signal_id=signal_id, config=config) if config.facing_lanes else None
        if targets is None:
            return []
        stops, _ = _facing_stops(index=index, targets=targets, drivable_types=drivable_types, config=config)
        heuristics.record(
            heuristic_id="H3", element_id=signal_id, action="bound" if stops else "no_facing_lane",
            message=f"signal {signal_id}: no lane named - {len(stops)} lane(s) it faces",
        )
        return stops
    stops, moved = [], False
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
            if heuristics.stops_at_signal_s(on_junction_road=road.junction is not None, config=config):
                stops.append(_lane_at_s(plane=plane, section=section, s=s))
                moved = True
            else:
                stops.append(_road_entry(index=index, road=road, section=section, plane=plane))
    if moved:
        heuristics.record(
            heuristic_id="H4", element_id=signal_id, action="placed_at_s",
            message=f"signal {signal_id}: on an ordinary road - stopline at its s, not at the lane start",
        )
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


def _control_stops(
    *, index: _Index, signal_id: str, drivable_types: frozenset[str], config: heuristics.HeuristicsConfig
) -> tuple[list[LaneStop], bool]:
    """The stops of one control and whether it is a vehicle control (H7 decides for unbound lights)."""
    stops, moved = _bound_stops(index=index, signal_id=signal_id, config=config)
    if moved:
        heuristics.record(
            heuristic_id="H4", element_id=signal_id, action="placed_at_s",
            message=f"signal {signal_id}: on an ordinary road - stopline at its s, not at the lane start",
        )
    if index.bound.get(signal_id) or not config.facing_lanes:
        return stops, True
    targets = _targets(index=index, signal_id=signal_id, config=config)
    if targets is None:
        return stops, True
    stops, any_drivable = _facing_stops(index=index, targets=targets, drivable_types=drivable_types, config=config)
    heuristics.record(
        heuristic_id="H3", element_id=signal_id, action="bound" if stops else "no_facing_lane",
        message=f"signal {signal_id}: no lane named - {len(stops)} lane(s) it faces",
    )
    return stops, any_drivable


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


def _stencil_rows(*, road_network, frame, within: Callable[..., bool]) -> tuple[list[dict], int]:
    """Rows of the painted stop/yield stencils inside the trim, and how many there are."""
    rows, count = [], 0
    stencils = [("stop_sign", x) for x in road_network._iai_stop_signs] + [
        ("yield_sign", x) for x in road_network._iai_yield_signs
    ]
    for agent_type, (xy, orientation, length, width, object_id) in stencils:
        z = float(xy[2]) if len(xy) > 2 else 0.0
        x, y = (float(v) for v in frame.to_local(xy[0], xy[1]))
        if not within(x=x, y=y, z=z):
            continue
        count += 1
        stopline = Stopline(
            actor_id=object_id, agent_type=agent_type, x=x, y=y,
            length=1.0, width=float(width), orientation=float(orientation),
        )
        rows.append(dataclasses.asdict(stopline))
    return rows, count


def control_stoplines(
    *,
    opendrive,
    road_network,
    lanelet_network,
    frame,
    within: Callable[..., bool],
    drivable_types: frozenset[str],
    config: heuristics.HeuristicsConfig,
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

    Returns:
        The rows (lights, then stop/yield stencils, then sign signals) and the coverage.
    """
    index = _index(opendrive=opendrive, road_network=road_network)

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
        stops, is_vehicle = _control_stops(index=index, signal_id=signal_id, drivable_types=drivable_types, config=config)
        light_stops[signal_id] = stops
        if heuristics.is_vehicle_light(has_drivable_lane=is_vehicle, config=config):
            vehicle_lights.add(signal_id)
        else:
            heuristics.record(
                heuristic_id="H7", element_id=signal_id, action="not_a_vehicle_light",
                message=f"light {signal_id}: its road has no drivable lane - not counted as a vehicle light",
            )

    rows: list[dict] = []
    lit: set[str] = set()
    for light in lanelet_network.traffic_lights:
        signal_id = str(getattr(light, "opendrive_id", ""))
        if signal_id not in vehicle_lights:
            continue
        for stop in light_stops[signal_id]:
            row = _row(stop=stop, frame=frame, actor_id=light.opendrive_id, agent_type="traffic_light", within=within)
            if row is not None:
                rows.append(row)
                lit.add(signal_id)

    stencil_rows, stencil_count = _stencil_rows(road_network=road_network, frame=frame, within=within)
    rows += stencil_rows
    stencil_xy = {
        agent_type: np.array([[r["x"], r["y"]] for r in stencil_rows if r["agent_type"] == agent_type]).reshape(-1, 2)
        for agent_type in SIGN_AGENT_TYPES.values()
    }
    signs, signs_missing = stencil_count, []
    sign_rows_written: list[dict] = []
    for sign in road_network._traffic_signs:
        agent_type = _sign_kind(sign=sign)
        if agent_type is None or not in_area(sign.position):
            continue
        signs += 1
        signal_id = str(sign.opendrive_id)
        distances = np.linalg.norm(stencil_xy[agent_type] - local(sign.position), axis=1)
        nearest = float(distances.min()) if distances.size else float("inf")
        if heuristics.matches_stencil(distance_m=nearest, config=config):
            heuristics.record(
                heuristic_id="H5", element_id=signal_id, action="stencil_is_its_stopline",
                message=f"sign {signal_id}: a {agent_type} stencil {nearest:.1f} m away is its stopline",
            )
            continue
        sign_rows, by_stencil, by_sign = [], False, None
        if heuristics.writes_sign_signal_stoplines(config=config):
            stops = _sign_stops(index=index, signal_id=signal_id, drivable_types=drivable_types, config=config)
            for stop in stops:
                row = _row(stop=stop, frame=frame, actor_id=sign.opendrive_id, agent_type=agent_type, within=within)
                if row is None or any(_same_spot(a=row, b=kept) for kept in sign_rows):
                    continue
                same = next(
                    (kept for kept in sign_rows_written if kept["agent_type"] == agent_type and _same_spot(a=row, b=kept)), None
                )
                if same is not None:
                    by_sign = same["actor_id"]
                    continue
                gap = _stencil_gap_on_lane(row=row, stencils=stencil_xy[agent_type])
                if heuristics.matches_stencil(distance_m=gap, config=config):
                    by_stencil = True
                    continue
                sign_rows.append(row)
        if by_stencil and not sign_rows:
            heuristics.record(
                heuristic_id="H5", element_id=signal_id, action="stencil_is_its_stopline",
                message=f"sign {signal_id}: a {agent_type} stencil on the lane it binds is its stopline",
            )
            continue
        if by_sign is not None and not sign_rows:
            heuristics.record(
                heuristic_id="H6", element_id=signal_id, action="same_stop_as_another_sign",
                message=f"sign {signal_id}: stops the same lane(s) at the same place as sign {by_sign}",
            )
            continue
        if sign_rows:
            rows += sign_rows
            sign_rows_written += sign_rows
            heuristics.record(
                heuristic_id="H6", element_id=signal_id, action="stopline_written",
                message=f"sign {signal_id}: {len(sign_rows)} {agent_type} stopline row(s)",
            )
        else:
            signs_missing.append(f"signal {signal_id}")

    lights_in_area = sorted(
        {str(light.opendrive_id) for light in road_network._traffic_lights if in_area(light.position)} & vehicle_lights
    )
    coverage = map_validation.ControlCoverage(
        lights=len(lights_in_area),
        lights_without_stopline=[i for i in lights_in_area if i not in lit],
        stop_yield_signs=signs,
        signs_without_stopline=sorted(signs_missing),
    )
    return ControlStoplines(rows=rows, coverage=coverage)


def report_painted_stop_lines(*, opendrive, frame, config: heuristics.HeuristicsConfig) -> int:
    """List every painted stop line (H9) in the conversion report; none becomes a stopline row.

    Returns:
        How many were listed.
    """
    degradation = logging.getLogger(conversion_report.DEGRADATION_LOGGER)
    count = 0
    for road in opendrive.roads:
        elements = [("object", obj, obj.type) for obj in road.objects] + [("signal", sig, sig.type) for sig in road.signals]
        for tag, element, element_type in elements:
            if not heuristics.is_painted_stop_line(tag=tag, name=element.name, element_type=element_type, config=config):
                continue
            s = min(max(float(element.s), 0.0), road.plan_view.length)
            position, tangent, _, _ = road.plan_view.calc(s, compute_curvature=False)
            xy = np.array(position[:2]) + float(element.t) * np.array([-np.sin(tangent), np.cos(tangent)])
            x, y = (round(float(v), 2) for v in frame.to_local(xy[0], xy[1]))
            degradation.warning(
                "%s %s: painted stop line at (%.2f, %.2f) - not converted (H9 painted_stop_line)",
                tag, element.id, x, y,
                extra={"kind": "not_converted", "element": tag, "id": str(element.id), "value": f"({x}, {y})",
                       "action": "listed"},
            )
            count += 1
    if count:
        logger.info(f"{count} painted stop line(s) listed in the conversion report; none becomes a stopline")
    return count
