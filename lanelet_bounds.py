"""Lanelet bound geometry between the OpenDRIVE export and the Lanelet2 writer.

Lanelet2's loader (``geometry::align``) judges each bound's direction from its middle vertex and inverts a
bound it finds backwards. A lane edge that loops back on itself (an edge offset farther into a bend than the
bend's radius) can put that middle vertex on the backward stretch, so Lanelet2 turns the bound around and the
mesh fills the space between the two bounds' opposite ends with false road. :func:`cut_loops` removes such
loops; :func:`sharp_kinks` lists the planView kinks that make them; :func:`align_safe` adds vertices on a bound's
own segments where Lanelet2 would still misjudge it. :func:`road_mesh` triangulates each lanelet pairing its
bounds by fraction of length (torchdrivesim's builder alternates vertices and joins points far apart).

Also here: the Douglas-Peucker simplification of the bounds (:func:`simplify_lanelet_network`).
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Optional

import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
from shapely.geometry import LineString

DEGRADATION_LOGGER = "crdesigner.degradation"
SHARP_KINK_DEG = 90.0
SPIKE_DEG = 150.0  # a vertex turning back by more than this is a spike, not a corner


@dataclasses.dataclass(slots=True)
class Loop:
    """A stretch of a bound removed by :func:`cut_loops`.

    Attributes:
        location: ``(x, y)`` where the bound crosses itself, in the polyline's frame.
        length: Length (m, x/y) of the removed stretch.
    """

    location: np.ndarray
    length: float


@dataclasses.dataclass(slots=True)
class Kink:
    """A heading jump between consecutive ``<planView>`` records of a road.

    Attributes:
        road_id: The road's OpenDRIVE id.
        s: Where the later record starts.
        position: ``(x, y)`` of that start, in the file's frame.
        jump_deg: The heading change, in degrees (sign: positive is to the left).
    """

    road_id: str
    s: float
    position: np.ndarray
    jump_deg: float


def cut_loops(*, polyline: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, list[Loop]]:
    """``polyline`` without the folds Lanelet2 would misread: loops traversed against the lane's turn, and spikes.

    An edge offset farther into a bend than the bend's radius runs backwards for a stretch: it either crosses
    itself, making a small loop that turns the opposite way to the lane (a swallowtail), or doubles straight back
    (a spike). A loop that turns the same way as the lane (a turnaround's inner edge crossing its way in, a ramp
    passing over itself) is real geometry and is kept. The lane's turn over a loop is the turn of ``reference``,
    the lanelet's other bound, between the points nearest the loop's ends.

    The result does not depend on direction: the reversed input (and reference) gives the reversed output, so two
    lanelets that share a bound in opposite directions keep sharing it.

    Args:
        polyline: ``(n, 2)`` or ``(n, 3)`` vertices; z is interpolated at a crossing.
        reference: The lanelet's other bound, running the same way, ``(m, 2+)``.

    Returns:
        The polyline with each fold replaced by its crossing point (a spike by its base), and the removed folds.
    """
    forward = _runs_forward(polyline=polyline)
    canonical = polyline if forward else polyline[::-1]
    reference = (reference if forward else reference[::-1])[:, :2]
    loops = []
    canonical, spikes = _remove_spikes(polyline=canonical)
    loops += spikes
    if len(canonical) >= 4 and not LineString(canonical[:, :2]).is_simple:
        while True:
            crossing = _innermost_fold(polyline=canonical, reference=reference)
            if crossing is None:
                break
            i, j, t = crossing
            point = canonical[i] + t * (canonical[i + 1] - canonical[i])
            lengths = np.linalg.norm(np.diff(canonical[:, :2], axis=0), axis=1)
            removed = (
                float(np.linalg.norm(canonical[i + 1, :2] - point[:2]))
                + float(lengths[i + 1 : j].sum())
                + float(np.linalg.norm(point[:2] - canonical[j, :2]))
            )
            loops.append(Loop(location=point[:2].copy(), length=removed))
            canonical = np.vstack([canonical[: i + 1], point[None, :], canonical[j + 1 :]])
    if not loops:
        return polyline, []
    return (canonical if forward else canonical[::-1]), loops


def _remove_spikes(*, polyline: np.ndarray) -> tuple[np.ndarray, list[Loop]]:
    """Drop every inner vertex where the polyline turns back by more than ``SPIKE_DEG``."""
    loops = []
    while len(polyline) >= 3:
        steps = np.diff(polyline[:, :2], axis=0)
        lengths = np.linalg.norm(steps, axis=1)
        cosines = (steps[:-1] * steps[1:]).sum(axis=1) / np.maximum(lengths[:-1] * lengths[1:], 1e-12)
        spikes = np.flatnonzero((cosines < math.cos(math.radians(SPIKE_DEG))) & (lengths[:-1] > 0) & (lengths[1:] > 0))
        if not len(spikes):
            break
        k = int(spikes[0]) + 1
        loops.append(Loop(location=polyline[k, :2].copy(), length=float(lengths[k - 1] + lengths[k])))
        polyline = np.delete(polyline, k, axis=0)
    return polyline, loops


def _runs_forward(*, polyline: np.ndarray) -> bool:
    """A direction-independent choice of one of the two orientations of ``polyline``."""
    for a, b in zip(polyline[:, :2], polyline[::-1, :2]):
        first, last = tuple(np.round(a, 6)), tuple(np.round(b, 6))
        if first != last:
            return first < last
    return True


def _innermost_fold(*, polyline: np.ndarray, reference: np.ndarray) -> Optional[tuple[int, int, float]]:
    """``(i, j, t)`` of the crossing segments ``i < j`` closest together in the polyline whose loop turns against
    ``reference`` (smallest ``i`` on ties); ``t`` locates the crossing on segment ``i``. None without one."""
    xy = polyline[:, :2]
    start, end = xy[:-1], xy[1:]
    direction = end - start
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(direction, axis=1))])
    along = along / along[-1] if along[-1] > 0 else along
    folds = []
    for i in range(len(direction) - 2):
        others = slice(i + 2, len(direction))
        d_other = direction[others]
        denominator = direction[i, 0] * d_other[:, 1] - direction[i, 1] * d_other[:, 0]
        offset = start[others] - start[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (offset[:, 0] * d_other[:, 1] - offset[:, 1] * d_other[:, 0]) / denominator
            u = (offset[:, 0] * direction[i, 1] - offset[:, 1] * direction[i, 0]) / denominator
        for hit in np.flatnonzero((np.abs(denominator) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
            j = i + 2 + int(hit)
            point = start[i] + t[hit] * direction[i]
            loop = np.vstack([point, xy[i + 1 : j + 1]])
            if _signed_area(ring=loop) * _turn(polyline=reference, start=along[i], end=along[j + 1]) < 0:
                folds.append((j - i, i, j, float(t[hit])))
    if not folds:
        return None
    _, i, j, t = min(folds)
    return i, j, t


def _signed_area(*, ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _turn(*, polyline: np.ndarray, start: float, end: float) -> float:
    """Signed heading change (rad, left positive) of ``polyline`` between the fractions ``start`` < ``end`` of its
    length (matched by length, not by nearest point: near a crossing both passes are equally near)."""
    lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))])
    if lengths[-1] == 0.0:
        return 0.0
    a, b = np.searchsorted(lengths / lengths[-1], [start, end])
    # one more step each side: a fold at a corner spans a stretch the other bound may turn in at a single vertex
    steps = np.diff(polyline[max(int(a) - 2, 0) : int(b) + 2], axis=0)
    steps = steps[np.linalg.norm(steps, axis=1) > 0]
    if len(steps) < 2:
        return 0.0
    headings = np.arctan2(steps[:, 1], steps[:, 0])
    return float(np.sum(np.remainder(np.diff(headings) + math.pi, math.tau) - math.pi))


def misjudged_by_align(*, left: np.ndarray, right: np.ndarray) -> bool:
    """Whether Lanelet2's loader (``geometry::align``) would turn one of these bounds around.

    ``align`` inverts ``left`` unless the middle vertex of ``right`` lies right of it, then ``right`` unless the
    middle vertex of (the possibly inverted) ``left`` lies left of it; "middle vertex" is ``ls[size / 2]``, or the
    middle of the end points for two vertices. The side comes from Lanelet2's own geometry (:func:`_side_of`).
    """
    if len(left) < 2 or len(right) < 2:
        return False
    if _side_of(line=left)(_middle(points=right)) >= 0:
        return True
    return _side_of(line=right)(_middle(points=left)) <= 0


def align_safe(*, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``left`` and ``right`` with vertices added on their own segments (no change of shape) so that the vertex
    Lanelet2's ``align`` takes as each bound's middle is one it reads on the correct side; unchanged when ``align``
    already reads them as written, or when no such vertex exists."""
    if not misjudged_by_align(left=left, right=right):
        return left, right
    side_of_left, side_of_right = _side_of(line=left), _side_of(line=right)
    right = _with_middle(points=right, readable=lambda p: side_of_left(p) < 0)
    left = _with_middle(points=left, readable=lambda p: side_of_right(p) > 0)
    return left, right


def _middle(*, points: np.ndarray) -> np.ndarray:
    return points[len(points) // 2, :2] if len(points) > 2 else (points[0, :2] + points[-1, :2]) / 2


def _with_middle(*, points: np.ndarray, readable) -> np.ndarray:
    """``points`` with segments split so that a vertex ``readable`` accepts, nearest the middle, is ``ls[size / 2]``.

    Only an inner vertex can become the middle one; when none reads right, the midpoint of the longest segment whose
    midpoint does is added first."""
    n = len(points)
    if n > 2 and readable(points[n // 2, :2]):
        return points
    inner = [k for k in range(1, n - 1) if readable(points[k, :2])]
    out = points
    if not inner:
        segments = [s for s in range(n - 1) if readable((points[s, :2] + points[s + 1, :2]) / 2)]
        if not segments:
            return points
        s = max(segments, key=lambda s: float(np.linalg.norm(points[s + 1, :2] - points[s, :2])))
        out = np.insert(points, s + 1, (points[s] + points[s + 1]) / 2, axis=0)
        inner = [s + 1]
    k = min(inner, key=lambda c: abs(c - len(out) // 2))
    while k < len(out) // 2:  # add vertices before k
        out, split = _split_longest(points=out, last=k)
        if split is None:
            return points
        k += 1
    while len(out) // 2 < k:  # add vertices after k
        out, split = _split_longest(points=out, first=k)
        if split is None:
            return points
    return out


def _split_longest(*, points: np.ndarray, first: int = 0, last: Optional[int] = None) -> tuple[np.ndarray, Optional[int]]:
    """``points`` with the longest segment between vertices ``first`` and ``last`` split at its middle."""
    last = len(points) - 1 if last is None else last
    if last <= first:
        return points, None
    lengths = np.linalg.norm(np.diff(points[first : last + 1, :2], axis=0), axis=1)
    s = first + int(np.argmax(lengths))
    return np.insert(points, s + 1, (points[s] + points[s + 1]) / 2, axis=0), s


def _side_of(*, line: np.ndarray):
    """``point -> signed distance`` to ``line``, positive left of it, computed by Lanelet2 itself
    (``geometry.toArcCoordinates``, the side test ``align`` uses), so the prediction is the loader's."""
    import lanelet2
    from lanelet2.core import BasicPoint2d, LineString3d, Point3d, getId

    linestring = lanelet2.geometry.to2D(
        LineString3d(getId(), [Point3d(getId(), float(x), float(y), 0.0) for x, y in line[:, :2]])
    )
    return lambda point: float(lanelet2.geometry.toArcCoordinates(linestring, BasicPoint2d(float(point[0]), float(point[1]))).distance)


def loop_free_bounds(*, network: LaneletNetwork, to_local) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """New ``(left, right)`` bounds of every lanelet with a looping bound, each cut recorded (``bound_loop_cut``).

    Args:
        network: The lanelet network, read only.
        to_local: ``(x, y) -> (x, y)`` from the network's frame to the map's local frame, for the records.

    Returns:
        Per lanelet id, its bounds with the loops cut; lanelets without loops are absent.
    """
    changed = {}
    for lanelet in network.lanelets:
        bounds = {}
        for side, polyline, reference in (
            ("left", lanelet.left_vertices, lanelet.right_vertices),
            ("right", lanelet.right_vertices, lanelet.left_vertices),
        ):
            bounds[side], loops = cut_loops(polyline=polyline, reference=reference)
            for loop in loops:
                record_geometry(
                    element="bound_loop_cut",
                    element_id=lanelet.lanelet_id,
                    value=round(loop.length, 2),
                    action="cut",
                    message=f"lanelet {lanelet.lanelet_id}: {side} bound looped back over {loop.length:.1f} m - loop cut",
                    location=to_local(*loop.location),
                )
        if bounds["left"] is not lanelet.left_vertices or bounds["right"] is not lanelet.right_vertices:
            changed[lanelet.lanelet_id] = (bounds["left"], bounds["right"])
    return changed


def align_safe_bounds(*, network: LaneletNetwork, to_local) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """New ``(left, right)`` bounds of every lanelet Lanelet2's loader would misread (:func:`align_safe`), and of
    the neighbours sharing a changed bound (they get the same vertices, so the converter keeps writing one way).

    Each lanelet changed for itself is recorded (``bound_vertices_added``); a lanelet still misread afterwards
    is left to AutoCritic (``crossed_lanelets``, kind ``inverted_on_load``).

    Args:
        network: The lanelet network, read only.
        to_local: ``(x, y) -> (x, y)`` from the network's frame to the map's local frame, for the records.

    Returns:
        Per lanelet id, its new bounds; unchanged lanelets are absent.
    """
    changed: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def bounds_of(lanelet) -> tuple[np.ndarray, np.ndarray]:
        return changed.get(lanelet.lanelet_id, (lanelet.left_vertices, lanelet.right_vertices))

    for lanelet in network.lanelets:
        left, right = bounds_of(lanelet)
        new_left, new_right = align_safe(left=left, right=right)
        if new_left is left and new_right is right:
            continue
        changed[lanelet.lanelet_id] = (new_left, new_right)
        added = len(new_left) + len(new_right) - len(left) - len(right)
        record_geometry(
            element="bound_vertices_added",
            element_id=lanelet.lanelet_id,
            value=added,
            action="vertices_added",
            message=f"lanelet {lanelet.lanelet_id}: {added} vertices added on its bounds so Lanelet2 reads them as written",
            location=to_local(*_middle(points=new_right)),
        )
        for old, new, neighbour_id, same_direction, neighbour_side in (
            (left, new_left, lanelet.adj_left, lanelet.adj_left_same_direction, "right"),
            (right, new_right, lanelet.adj_right, lanelet.adj_right_same_direction, "left"),
        ):
            neighbour = None if new is old or neighbour_id is None else network.find_lanelet_by_id(neighbour_id)
            if neighbour is None:
                continue
            if not same_direction:
                neighbour_side = "left" if neighbour_side == "right" else "right"
            n_left, n_right = bounds_of(neighbour)
            shared = n_right if neighbour_side == "right" else n_left
            expected = old if same_direction else old[::-1]
            if shared.shape == expected.shape and np.allclose(shared, expected, atol=1e-3):
                replacement = new if same_direction else new[::-1]
                changed[neighbour.lanelet_id] = (n_left, replacement) if neighbour_side == "right" else (replacement, n_right)
    return changed


def record_sharp_kinks(*, opendrive, to_local) -> int:
    """Record every :func:`sharp_kinks` kink (``sharp_kink``); return how many."""
    kinks = sharp_kinks(opendrive=opendrive)
    for kink in kinks:
        record_geometry(
            element="sharp_kink",
            element_id=kink.road_id,
            value=round(kink.jump_deg, 1),
            action="reported",
            message=f"road {kink.road_id}: heading turns {kink.jump_deg:+.0f} deg at s={kink.s:.1f} between planView records",
            location=to_local(*kink.position),
        )
    return len(kinks)


def sharp_kinks(*, opendrive, min_deg: float = SHARP_KINK_DEG) -> list[Kink]:
    """Every heading jump of more than ``min_deg`` between consecutive ``<planView>`` records."""
    kinks = []
    for road in opendrive.roads:
        geometries = road.plan_view._geometries
        s = 0.0
        for before, after in zip(geometries, geometries[1:]):
            s += float(before.length)
            end_heading = before.calc_position(before.length, compute_curvature=False)[1]
            jump = math.degrees(math.remainder(float(after.heading) - float(end_heading), math.tau))
            if abs(jump) > min_deg:
                kinks.append(Kink(road_id=str(road.id), s=s, position=np.array(after.start_position[:2]), jump_deg=jump))
    return kinks


def record_geometry(*, element: str, element_id: object, value: object, action: str, message: str, location) -> None:
    """Log one geometry repair or finding as a ``kind="geometry"`` degradation record with a local location."""
    logging.getLogger(DEGRADATION_LOGGER).warning(
        message,
        extra={
            "kind": "geometry",
            "element": element,
            "id": element_id,
            "value": value,
            "action": action,
            "location": [round(float(v), 2) for v in location[:2]],
        },
    )


def road_mesh(*, lanelet_map):
    """The road mesh of every lanelet: ``torchdrivesim.lanelet2.road_mesh_from_lanelet_map``'s vertices and face
    count, with each lanelet's strip of triangles pairing its bounds by fraction of length.

    torchdrivesim's builder steps one vertex along each bound in turn, so where one bound has more or bunched
    vertices (curves, cul-de-sacs, bounds simplified separately) it joins points far apart and its faces cover
    ground that is not road.

    Args:
        lanelet_map: A loaded Lanelet2 map.

    Returns:
        A ``torchdrivesim.mesh.BaseMesh`` with a batch of one.
    """
    import torch
    from torchdrivesim.mesh import BaseMesh

    point_index = {}
    vertices = np.zeros((len(lanelet_map.pointLayer), 2), dtype=np.float32)
    for k, point in enumerate(lanelet_map.pointLayer):
        point_index[point.id] = k
        vertices[k] = (point.x, point.y)
    faces = []
    for lanelet in lanelet_map.laneletLayer:
        left = [point_index[p.id] for p in lanelet.leftBound]
        right = [point_index[p.id] for p in lanelet.rightBound]
        if len(left) + len(right) < 3:
            continue
        faces.extend(_strip(left=left, right=right, vertices=vertices))
    faces = torch.from_numpy(np.array(faces, dtype=np.int64).reshape(-1, 3))
    return BaseMesh(verts=torch.from_numpy(vertices).unsqueeze(0), faces=faces.unsqueeze(0))


def _strip(*, left: list[int], right: list[int], vertices: np.ndarray) -> list[list[int]]:
    """Triangles between two bounds (vertex indices), advancing along whichever bound is behind by fraction of length."""

    def fractions(indices: list[int]) -> np.ndarray:
        lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(vertices[indices], axis=0), axis=1))])
        return lengths / lengths[-1] if lengths[-1] > 0 else np.linspace(0.0, 1.0, len(indices))

    along_left, along_right = fractions(left), fractions(right)
    i = j = 0
    faces = []
    while i < len(left) - 1 or j < len(right) - 1:
        if j == len(right) - 1 or (i < len(left) - 1 and along_left[i + 1] <= along_right[j + 1]):
            faces.append([left[i], right[j], left[i + 1]])
            i += 1
        else:
            faces.append([left[i], right[j], right[j + 1]])
            j += 1
    return faces


def set_bounds(*, lanelet, left: np.ndarray, right: np.ndarray) -> None:
    """Set ``lanelet``'s bounds, its centre line from them, and the caches commonroad-io's setters leave stale."""
    lanelet.left_vertices = left
    lanelet.right_vertices = right
    lanelet.center_vertices = _midline(left=left, right=right)
    _refresh_lanelet_caches(lanelet=lanelet)


def simplify_lanelet_network(*, network: LaneletNetwork, tolerance: float) -> int:
    """Apply 2D Douglas-Peucker to each lanelet's left/right boundaries; rebuild centerlines.

    Mutates ``network`` in place. Endpoints are preserved by shapely's simplify,
    so lanelet-to-lanelet connectivity (predecessor/successor handshakes) is
    unaffected. Because RDP is deterministic, two adjacent lanelets whose shared
    boundary arrays were identical pre-simplification produce identical
    post-simplification arrays, so cr2lanelet's 0.001 m shared-way dedup still
    works.

    Args:
        network: The CommonRoad lanelet network whose lanelets will be simplified.
        tolerance: Max permissible lateral deviation per vertex, in metres.

    Returns:
        The total number of vertices removed across all lanelets (for logging).
    """
    removed = 0
    # First pass: normalize every lanelet to 3D so adjacent boundaries always
    # share a trailing dimension. cr2lanelet's `_vertices_are_equal` does a raw
    # numpy subtract with no trailing-axis guard; if an adj-pair ends up mixed
    # 2D/3D it crashes the OSM export.
    for lanelet in network.lanelets:
        lanelet.left_vertices = _ensure_3d(polyline=lanelet.left_vertices, lanelet_id=lanelet.lanelet_id)
        lanelet.right_vertices = _ensure_3d(polyline=lanelet.right_vertices, lanelet_id=lanelet.lanelet_id)
        lanelet.center_vertices = _ensure_3d(polyline=lanelet.center_vertices, lanelet_id=lanelet.lanelet_id)
        _refresh_lanelet_caches(lanelet=lanelet)
    for lanelet in network.lanelets:
        left = lanelet.left_vertices
        right = lanelet.right_vertices
        if left.shape[0] < 3 and right.shape[0] < 3:
            continue
        new_left = _simplify_polyline(polyline=left, tolerance=tolerance)
        new_right = _simplify_polyline(polyline=right, tolerance=tolerance)
        removed += (left.shape[0] - new_left.shape[0]) + (right.shape[0] - new_right.shape[0])
        # Centerline must be re-derived from the simplified boundaries; if left and
        # right end up with different vertex counts, resample both to the longer one
        # on its own arc-length axis before averaging.
        set_bounds(lanelet=lanelet, left=new_left, right=new_right)
    return removed


def _refresh_lanelet_caches(*, lanelet) -> None:
    """Mirror the cache initialization done by Lanelet.__init__ after vertex mutation."""
    from commonroad.geometry.shape import Polygon as CRPolygon

    lanelet._polygon = CRPolygon(np.concatenate((lanelet.right_vertices, np.flip(lanelet.left_vertices, 0))))
    lanelet._distance = None
    lanelet._inner_distance = None


def _ensure_3d(*, polyline: np.ndarray, lanelet_id: int) -> np.ndarray:
    """Pad a 2D polyline with a z=0 column (logged as a degradation); pass 3D through unchanged."""
    if polyline.shape[1] == 3:
        return polyline
    logging.getLogger(DEGRADATION_LOGGER).warning(
        "lanelet %s: boundary has no heights - using z=0",
        lanelet_id,
        extra={"kind": "missing_value", "element": "lanelet.vertices.z", "id": lanelet_id, "value": None, "action": "defaulted"},
    )
    return np.column_stack([polyline, np.zeros(polyline.shape[0], dtype=polyline.dtype)])


def _simplify_polyline(*, polyline: np.ndarray, tolerance: float) -> np.ndarray:
    """Run Douglas-Peucker on the XY projection; preserve z by arc-length interpolation if present.

    The output preserves the input's trailing dimension (2 or 3), so adjacent
    lanelets that share a boundary continue to have matching vertex shapes
    (cr2lanelet's `_vertices_are_equal` requires this).
    """
    if polyline.shape[0] < 3:
        return polyline
    xy = polyline[:, :2]
    simplified_xy = np.asarray(LineString(xy).simplify(tolerance, preserve_topology=False).coords, dtype=polyline.dtype)
    if polyline.shape[1] == 2:
        return simplified_xy
    # 3D path: interpolate z onto the surviving XY points using cumulative XY arc length.
    orig_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    new_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(simplified_xy, axis=0), axis=1))])
    if orig_s[-1] > 0 and new_s[-1] > 0:
        new_s = new_s * (orig_s[-1] / new_s[-1])
    z = np.interp(new_s, orig_s, polyline[:, 2])
    return np.column_stack([simplified_xy, z.astype(polyline.dtype)])


def _midline(*, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute the centerline of two boundary polylines.

    If the two polylines have the same vertex count after simplification, we
    average pointwise. Otherwise we resample both onto a common arc-length axis
    using the denser of the two as the reference, then average. This is the same
    convention used elsewhere in commonroad-io's lanelet construction.
    """
    if left.shape[0] == right.shape[0]:
        return (left + right) / 2.0
    n = max(left.shape[0], right.shape[0])
    return (_resample_polyline(polyline=left, n=n) + _resample_polyline(polyline=right, n=n)) / 2.0


def _resample_polyline(*, polyline: np.ndarray, n: int) -> np.ndarray:
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))])
    if cumulative[-1] == 0.0:
        return np.repeat(polyline[:1], n, axis=0)
    targets = np.linspace(0.0, cumulative[-1], n)
    out = np.column_stack([np.interp(targets, cumulative, polyline[:, i]) for i in range(polyline.shape[1])])
    return out.astype(polyline.dtype)
