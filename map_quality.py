"""Quality metrics over the output files of one ``convert_map.py`` conversion.

Each metric is a pure function of :class:`ConversionOutputs`, which loads every output file of a
map directory once. ``map_report.py`` records these metrics per map, and they are the single
implementation of each check: nothing else recomputes them.

Usage as a library::

    outputs = ConversionOutputs.load(map_dir="/abs/path/to/converted/map")
    metrics, errors = compute_all(outputs=outputs)
"""

from __future__ import annotations

import collections
import dataclasses
import glob
import hashlib
import json
import logging
import math
import os
from typing import Any, Callable, Optional, Sequence

import numpy as np
from lxml import etree

from traffic_light_fsm import load_metadata, resolve_map_path, utm_projector, vehicle_routing_graph

logger = logging.getLogger(__name__)

DEFAULT_VISUALIZATION_FOV = 800.0
SEAM_TOLERANCE_M = 0.1
Z_CONFLICT_MIN_M = 0.01
CROSSED_BOUNDS_GAP_M = 1.0
STOP_YIELD_SIGN_NUMBERS = ("206", "205")


@dataclasses.dataclass(slots=True)
class OsmTagCounts:
    """Tag-level counts of a ``.osm`` that lanelet2 does not expose.

    Attributes:
        way_types: ``way`` count per ``type`` tag value.
        untyped_ways: ``way`` elements without a ``type`` tag.
        one_way_no: lanelet relations tagged ``one_way=no``.
        stop_yield_sign_ways: ``traffic_sign`` ways for stop or yield signs (subtype ``<country>206`` /
            ``<country>205``, the sign numbers every supported country table shares).
    """

    way_types: collections.Counter
    untyped_ways: int
    one_way_no: int
    stop_yield_sign_ways: int = 0


def osm_tag_counts(*, osm_path: str) -> OsmTagCounts:
    """Count way types and ``one_way=no`` lanelets in one streaming pass over the ``.osm``."""
    way_types: collections.Counter = collections.Counter()
    untyped = 0
    one_way_no = 0
    stop_yield = 0
    for _, elem in etree.iterparse(osm_path, events=("end",), tag=("way", "relation"), huge_tree=True):
        tags = {tag.get("k"): tag.get("v") for tag in elem.iter("tag")}
        if elem.tag == "way":
            if "type" in tags:
                way_types[tags["type"]] += 1
            else:
                untyped += 1
            if tags.get("type") == "traffic_sign" and tags.get("subtype", "").endswith(STOP_YIELD_SIGN_NUMBERS):
                stop_yield += 1
        elif tags.get("type") == "lanelet" and tags.get("one_way") == "no":
            one_way_no += 1
        elem.clear()
    return OsmTagCounts(
        way_types=way_types, untyped_ways=untyped, one_way_no=one_way_no, stop_yield_sign_ways=stop_yield
    )


def point_array(*, lanelet_map) -> np.ndarray:
    """``(N, 3)`` array of every point's local ``(x, y, z)``."""
    return np.array([(p.x, p.y, p.z) for p in lanelet_map.pointLayer], dtype=float).reshape(-1, 3)


@dataclasses.dataclass(slots=True)
class ConversionOutputs:
    """Every output of one conversion, each loaded once.

    Attributes:
        map_dir: Absolute path of the directory holding the outputs.
        metadata: Parsed ``metadata.json``.
        osm_path: Absolute path of the Lanelet2 ``.osm``.
        osm_bytes: Size of the ``.osm`` in bytes.
        osm_tags: Tag-level counts of the ``.osm``.
        lanelet_map: The ``.osm`` loaded by ``lanelet2.io.loadRobust`` at the metadata origin.
        points: Local ``(x, y, z)`` of every point in ``lanelet_map``.
        load_errors: Error messages ``loadRobust`` reported.
        stoplines: Parsed ``*_stoplines.json`` rows.
        mesh: The road mesh (``torchdrivesim.mesh.BirdviewMesh``), or ``None`` if absent.
        visualization_fov: Field of view (metres) the visualization was rendered with.
    """

    map_dir: str
    metadata: dict
    osm_path: str
    osm_bytes: int
    osm_tags: OsmTagCounts
    lanelet_map: Any
    points: np.ndarray
    load_errors: list[str]
    stoplines: list[dict]
    mesh: Any
    visualization_fov: float

    @classmethod
    def load(cls, *, map_dir: str) -> "ConversionOutputs":
        """Load the outputs in ``map_dir``.

        Args:
            map_dir: Directory containing ``metadata.json`` and the files it references.

        Returns:
            The loaded outputs.

        Raises:
            FileNotFoundError: If ``metadata.json`` or the ``.osm`` it names is missing.
        """
        import lanelet2

        map_dir = os.path.abspath(map_dir)
        metadata = load_metadata(map_dir=map_dir)
        osm_path = resolve_map_path(map_dir=map_dir, name=metadata["lanelet_path"])
        origin_lat, origin_lon = metadata["lanelet_map_origin"]
        lanelet_map, load_errors = lanelet2.io.loadRobust(
            osm_path, utm_projector(origin_lat=origin_lat, origin_lon=origin_lon)
        )
        stoplines: list[dict] = []
        if metadata.get("stoplines_path"):
            with open(resolve_map_path(map_dir=map_dir, name=metadata["stoplines_path"])) as f:
                stoplines = json.load(f)
        mesh = None
        if metadata.get("mesh_path"):
            from torchdrivesim.mesh import BirdviewMesh

            mesh = BirdviewMesh.load(resolve_map_path(map_dir=map_dir, name=metadata["mesh_path"]))
        return cls(
            map_dir=map_dir,
            metadata=metadata,
            osm_path=osm_path,
            osm_bytes=os.path.getsize(osm_path),
            osm_tags=osm_tag_counts(osm_path=osm_path),
            lanelet_map=lanelet_map,
            points=point_array(lanelet_map=lanelet_map),
            load_errors=[str(e) for e in load_errors],
            stoplines=stoplines,
            mesh=mesh,
            visualization_fov=_rendered_visualization_fov(map_dir=map_dir),
        )


def _rendered_visualization_fov(*, map_dir: str) -> float:
    """FOV the converter recorded in ``<map>_provenance.json``; outputs without one used the old default."""
    for path in glob.glob(os.path.join(map_dir, "*_provenance.json")):
        with open(path) as f:
            fov = json.load(f).get("visualization_fov")
        if fov is not None:
            return float(fov)
    return DEFAULT_VISUALIZATION_FOV


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def lanelet_count(*, outputs: ConversionOutputs) -> int:
    """Number of lanelets in the ``.osm``."""
    return len(outputs.lanelet_map.laneletLayer)


def load_error_count(*, outputs: ConversionOutputs) -> int:
    """Errors ``lanelet2.io.loadRobust`` reported while loading the ``.osm``."""
    return len(outputs.load_errors)


def _lanelet_centre(*, lanelet) -> list[float]:
    ends = [lanelet.leftBound[0], lanelet.leftBound[-1], lanelet.rightBound[0], lanelet.rightBound[-1]]
    return [round(sum(p.x for p in ends) / 4, 2), round(sum(p.y for p in ends) / 4, 2)]


def fault(*, element: str, location: Sequence[float], detail: str = "") -> dict[str, Any]:
    """One place a check fails (or a finding): ``{id, location, detail}``, location rounded to 1 cm."""
    return {"id": element, "location": [round(float(v), 2) for v in location[:2]], "detail": detail}


def unrouted_lanelets(*, outputs: ConversionOutputs) -> list[dict[str, Any]]:
    """Lanelets with no routing-graph edge (Germany, vehicle rules), with their centres."""
    lanelets = list(outputs.lanelet_map.laneletLayer)
    if not lanelets:
        return []
    graph = vehicle_routing_graph(lanelet_map=outputs.lanelet_map)
    return [
        fault(element=f"lanelet {ll.id}", location=_lanelet_centre(lanelet=ll))
        for ll in lanelets
        if not (
            graph.following(ll)
            or graph.previous(ll)
            or graph.left(ll)
            or graph.right(ll)
            or graph.adjacentLeft(ll)
            or graph.adjacentRight(ll)
        )
    ]


def routing_edge_pct(*, outputs: ConversionOutputs) -> Optional[float]:
    """Percentage of lanelets with at least one routing-graph edge (Germany, vehicle rules).

    An edge is a following, previous, left, right, adjacent-left or adjacent-right relation.
    ``None`` for a map without lanelets.
    """
    count = len(outputs.lanelet_map.laneletLayer)
    if not count:
        return None
    return 100.0 * (count - len(unrouted_lanelets(outputs=outputs))) / count


def seam_handshake_pct(*, outputs: ConversionOutputs) -> Optional[float]:
    """Percentage of geometric successor seams whose boundary endpoints share point ids.

    A seam is a lanelet pair where both boundary end points of one lie within
    ``SEAM_TOLERANCE_M`` (XY) of the matching start points of the other. Routing needs the
    shared point ids, so this predicts routability independently of the routing graph.
    ``None`` when the map has no seams.
    """
    seams, unshaken = _seams(outputs=outputs)
    return 100.0 * (seams - len(unshaken)) / seams if seams else None


def unshaken_seams(*, outputs: ConversionOutputs) -> list[dict[str, Any]]:
    """The geometric seams (see :func:`seam_handshake_pct`) whose end points do not share ids, at the seam."""
    return _seams(outputs=outputs)[1]


def _seams(*, outputs: ConversionOutputs) -> tuple[int, list[dict[str, Any]]]:
    from scipy.spatial import cKDTree

    lanelets = list(outputs.lanelet_map.laneletLayer)
    if len(lanelets) < 2:
        return 0, []
    left_starts = np.array([(ll.leftBound[0].x, ll.leftBound[0].y) for ll in lanelets])
    right_starts = np.array([(ll.rightBound[0].x, ll.rightBound[0].y) for ll in lanelets])
    left_ends = np.array([(ll.leftBound[-1].x, ll.leftBound[-1].y) for ll in lanelets])
    right_ends = np.array([(ll.rightBound[-1].x, ll.rightBound[-1].y) for ll in lanelets])
    candidates = cKDTree(left_starts).query_ball_point(left_ends, r=SEAM_TOLERANCE_M)
    seams = 0
    unshaken = []
    for i, js in enumerate(candidates):
        ll = lanelets[i]
        for j in js:
            if j == i or np.hypot(*(right_starts[j] - right_ends[i])) > SEAM_TOLERANCE_M:
                continue
            other = lanelets[j]
            seams += 1
            if not (
                ll.leftBound[-1].id == other.leftBound[0].id
                and ll.rightBound[-1].id == other.rightBound[0].id
            ):
                unshaken.append(
                    fault(
                        element=f"lanelets {ll.id} -> {other.id}",
                        location=(left_ends[i] + right_ends[i]) / 2,
                    )
                )
    return seams, unshaken


def z_conflicts(*, outputs: ConversionOutputs) -> dict[str, float]:
    """Points at the same XY (1 cm grid) whose elevations differ by more than 1 cm.

    Returns:
        ``count``: XY positions with conflicting z; ``max_dz``: largest z spread (metres).
    """
    z_by_xy: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
    for x, y, z in outputs.points.tolist():
        z_by_xy[(round(x * 100), round(y * 100))].append(z)
    spreads = [max(zs) - min(zs) for zs in z_by_xy.values() if len(zs) > 1]
    conflicting = [float(s) for s in spreads if s > Z_CONFLICT_MIN_M]
    return {"count": len(conflicting), "max_dz": max(conflicting, default=0.0)}


def stoplines_by_agent_type(*, outputs: ConversionOutputs) -> dict[str, int]:
    """Stopline rows per ``agent_type``."""
    return dict(sorted(collections.Counter(row["agent_type"] for row in outputs.stoplines).items()))


def sign_ways_vs_stoplines(*, outputs: ConversionOutputs) -> dict[str, int]:
    """``traffic_sign``/``traffic_light`` ways in the ``.osm`` against the stoplines emitted.

    Stop/yield sign ways with no stop/yield stopline mean traffic controls were drawn but not exported.
    """
    by_agent = stoplines_by_agent_type(outputs=outputs)
    return {
        "traffic_sign_ways": outputs.osm_tags.way_types.get("traffic_sign", 0),
        "stop_yield_sign_ways": outputs.osm_tags.stop_yield_sign_ways,
        "traffic_light_ways": outputs.osm_tags.way_types.get("traffic_light", 0),
        "sign_stoplines": by_agent.get("stop_sign", 0) + by_agent.get("yield_sign", 0),
        "light_stoplines": by_agent.get("traffic_light", 0),
    }


def untyped_way_count(*, outputs: ConversionOutputs) -> int:
    """``.osm`` ways that carry no ``type`` tag."""
    return outputs.osm_tags.untyped_ways


def one_way_no_count(*, outputs: ConversionOutputs) -> int:
    """Lanelets tagged ``one_way=no`` (bidirectional)."""
    return outputs.osm_tags.one_way_no


def lanelets_without_subtype(*, outputs: ConversionOutputs) -> int:
    """Lanelets with no ``subtype`` tag (consumers that filter by subtype drop them)."""
    return len(subtypeless_lanelets(outputs=outputs))


def subtypeless_lanelets(*, outputs: ConversionOutputs) -> list[dict[str, Any]]:
    """The lanelets with no ``subtype`` tag, with their centres."""
    return [
        fault(element=f"lanelet {ll.id}", location=_lanelet_centre(lanelet=ll))
        for ll in outputs.lanelet_map.laneletLayer
        if "subtype" not in ll.attributes
    ]


def point_extent_m(*, outputs: ConversionOutputs) -> Optional[float]:
    """Largest ``|x|`` or ``|y|`` of any point in the local frame; ``None`` for an empty map.

    Large values mean the local origin is far from the map (the float32 mesh loses precision).
    """
    if not len(outputs.points):
        return None
    return float(np.abs(outputs.points[:, :2]).max())


def furthest_point(*, outputs: ConversionOutputs) -> list[dict[str, Any]]:
    """The point that sets :func:`point_extent_m`."""
    if not len(outputs.points):
        return []
    index = int(np.abs(outputs.points[:, :2]).max(axis=1).argmax())
    return [fault(element="point", location=outputs.points[index, :2])]


def osm_size_bytes(*, outputs: ConversionOutputs) -> int:
    """Size of the ``.osm`` in bytes."""
    return outputs.osm_bytes


def origin(*, outputs: ConversionOutputs) -> list[float]:
    """``lanelet_map_origin`` (lat, lon) from ``metadata.json``."""
    return [float(v) for v in outputs.metadata["lanelet_map_origin"]]


def xy_geometry_hash(*, outputs: ConversionOutputs) -> str:
    """sha256 of the sorted local (x, y) of every point, rounded to 1 cm.

    Unchanged when only elevation, ids or tags change.
    """
    coords = sorted((round(x * 100), round(y * 100)) for x, y, _ in outputs.points.tolist())
    return hashlib.sha256(json.dumps(coords).encode()).hexdigest()


CROSSED_KINDS = ("reversed_bounds", "invalid_polygons", "inverted_on_load")


def crossed_geometry(*, outputs: ConversionOutputs) -> dict[str, int]:
    """Lanelets whose geometry folds over itself, or that Lanelet2 reads against their written direction.

    Returns:
        ``reversed_bounds``: lanelets whose left and right bounds run in opposite directions,
        i.e. ``d(l0,r0) + d(l1,r1)`` exceeds ``d(l0,r1) + d(l1,r0)`` by more than
        ``CROSSED_BOUNDS_GAP_M``. ``invalid_polygons``: lanelet polygons shapely reports as
        invalid (self-intersecting). ``inverted_on_load``: lanelets with a bound Lanelet2's loader turned
        around although it is stored their way (:func:`misread_lanelet_ids`). ``lanelets``: lanelets of any kind.
    """
    faults = crossed_lanelets(outputs=outputs)
    return {
        **{kind: sum(1 for f in faults if kind in f["detail"].split()) for kind in CROSSED_KINDS},
        "lanelets": len(faults),
    }


def crossed_lanelets(*, outputs: ConversionOutputs) -> list[dict[str, Any]]:
    """The lanelets :func:`crossed_geometry` counts, with their centres; ``detail`` names the kinds."""
    from shapely.geometry import Polygon

    misread = misread_lanelet_ids(lanelet_map=outputs.lanelet_map)
    faults = []
    for ll in outputs.lanelet_map.laneletLayer:
        left = [(p.x, p.y) for p in ll.leftBound]
        right = [(p.x, p.y) for p in ll.rightBound]
        straight = math.dist(left[0], right[0]) + math.dist(left[-1], right[-1])
        crossed = math.dist(left[0], right[-1]) + math.dist(left[-1], right[0])
        ring = left + right[::-1]
        kinds = [
            kind
            for kind, found in (
                ("reversed_bounds", straight - crossed > CROSSED_BOUNDS_GAP_M),
                ("invalid_polygons", len(ring) >= 3 and not Polygon(ring).is_valid),
                ("inverted_on_load", ll.id in misread),
            )
            if found
        ]
        if kinds:
            faults.append(fault(element=f"lanelet {ll.id}", location=_lanelet_centre(lanelet=ll), detail=" ".join(kinds)))
    return faults


def misread_lanelet_ids(*, lanelet_map) -> set[int]:
    """Lanelets whose bound Lanelet2's loader inverted where the way's other users show it should not have.

    The converter stores a way in the direction of the lanelet that creates it; a neighbour shares it as the
    other side of the line (same direction: not inverted) or as the same side (opposite direction: exactly one
    of the two reads it inverted). A way that only one lanelet uses is never inverted. Ways with more than two
    users are not judged.
    """
    users: dict[int, list[tuple[int, str, bool]]] = collections.defaultdict(list)
    for ll in lanelet_map.laneletLayer:
        users[ll.leftBound.id].append((ll.id, "left", ll.leftBound.inverted()))
        users[ll.rightBound.id].append((ll.id, "right", ll.rightBound.inverted()))
    misread = set()
    for uses in users.values():
        if len(uses) == 1 and uses[0][2]:
            misread.add(uses[0][0])
        elif len(uses) == 2:
            (a, side_a, inverted_a), (b, side_b, inverted_b) = uses
            if side_a != side_b:
                misread.update(lanelet for lanelet, inverted in ((a, inverted_a), (b, inverted_b)) if inverted)
            elif inverted_a == inverted_b:
                misread.update((a, b))
    return misread


def mesh_in_frame_fraction(*, outputs: ConversionOutputs) -> Optional[float]:
    """Fraction of mesh vertices inside the square visualization frame.

    The frame is ``visualization_fov`` metres wide, centred on ``metadata.center``.
    ``None`` without a mesh or centre.
    """
    center = outputs.metadata.get("center")
    if outputs.mesh is None or center is None:
        return None
    verts = outputs.mesh.verts.reshape(-1, 2)
    if verts.shape[0] == 0:
        return None
    half = outputs.visualization_fov / 2.0
    dx = (verts[:, 0] - float(center[0])).abs()
    dy = (verts[:, 1] - float(center[1])).abs()
    return float(((dx <= half) & (dy <= half)).float().mean())


METRICS: dict[str, Callable[..., Any]] = {
    "lanelet_count": lanelet_count,
    "load_error_count": load_error_count,
    "routing_edge_pct": routing_edge_pct,
    "seam_handshake_pct": seam_handshake_pct,
    "z_conflicts": z_conflicts,
    "stoplines_by_agent_type": stoplines_by_agent_type,
    "sign_ways_vs_stoplines": sign_ways_vs_stoplines,
    "untyped_way_count": untyped_way_count,
    "osm_size_bytes": osm_size_bytes,
    "origin": origin,
    "xy_geometry_hash": xy_geometry_hash,
    "crossed_geometry": crossed_geometry,
    "one_way_no_count": one_way_no_count,
    "mesh_in_frame_fraction": mesh_in_frame_fraction,
    "lanelets_without_subtype": lanelets_without_subtype,
    "point_extent_m": point_extent_m,
}


# Per validation check, the function listing the places that fail it (each ``{id, location, detail}``).
FAULT_LOCATIONS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "routing_edge_pct": unrouted_lanelets,
    "seam_handshake_pct": unshaken_seams,
    "crossed_lanelets": crossed_lanelets,
    "lanelets_without_subtype": subtypeless_lanelets,
    "point_extent_m": furthest_point,
}


def compute_all(*, outputs: ConversionOutputs) -> tuple[dict[str, Any], dict[str, str]]:
    """Every metric in :data:`METRICS`; a metric that raises is recorded, not propagated.

    Returns:
        ``(metrics, errors)``: values keyed by metric name, and ``"<Class>: <message>"`` for
        each metric that failed.
    """
    metrics: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, metric in METRICS.items():
        try:
            metrics[name] = metric(outputs=outputs)
        except Exception as exc:  # noqa: BLE001 - one broken metric must not hide the others
            errors[name] = f"{type(exc).__name__}: {exc}"
    return metrics, errors
