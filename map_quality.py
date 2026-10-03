"""Quality metrics over the output files of one ``convert_map.py`` conversion.

Each metric is a pure function of :class:`ConversionOutputs`, which loads every output file of a
map directory once. ``map_report.py`` records these metrics per map, and they are the single
implementation of each check: nothing else recomputes them.

Usage as a library::

    outputs = ConversionOutputs.load(map_dir="/abs/path/to/converted/map")
    metrics = compute_all(outputs=outputs)
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
from typing import Any, Callable, Optional

from lxml import etree

logger = logging.getLogger(__name__)

DEFAULT_VISUALIZATION_FOV = 800.0
SEAM_TOLERANCE_M = 0.1
Z_CONFLICT_MIN_M = 0.01
CROSSED_BOUNDS_GAP_M = 1.0


def utm_projector(*, origin_lat: float, origin_lon: float):
    """Lanelet2 ``UtmProjector`` at a map's ``lanelet_map_origin``, as every IAI tool loads maps."""
    import lanelet2
    from lanelet2.projection import UtmProjector

    return UtmProjector(lanelet2.io.Origin(origin_lat, origin_lon))


@dataclasses.dataclass(slots=True)
class ConversionOutputs:
    """Every output of one conversion, each loaded once.

    Attributes:
        map_dir: Absolute path of the directory holding the outputs.
        metadata: Parsed ``metadata.json``.
        osm_path: Absolute path of the Lanelet2 ``.osm``.
        osm_bytes: Size of the ``.osm`` in bytes.
        osm_root: Parsed ``.osm`` XML, for tag-level counts lanelet2 does not expose.
        lanelet_map: The ``.osm`` loaded by ``lanelet2.io.loadRobust`` at the metadata origin.
        load_errors: Error messages ``loadRobust`` reported.
        stoplines: Parsed ``*_stoplines.json`` rows.
        mesh: The road mesh (``torchdrivesim.mesh.BirdviewMesh``), or ``None`` if absent.
        visualization_fov: Field of view (metres) the visualization was rendered with.
    """

    map_dir: str
    metadata: dict
    osm_path: str
    osm_bytes: int
    osm_root: Any
    lanelet_map: Any
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

        from traffic_light_fsm import load_metadata

        map_dir = os.path.abspath(map_dir)
        metadata = load_metadata(map_dir=map_dir)
        osm_path = _resolve(map_dir=map_dir, name=metadata["lanelet_path"])
        origin_lat, origin_lon = metadata["lanelet_map_origin"]
        lanelet_map, load_errors = lanelet2.io.loadRobust(
            osm_path, utm_projector(origin_lat=origin_lat, origin_lon=origin_lon)
        )
        stoplines: list[dict] = []
        if metadata.get("stoplines_path"):
            with open(_resolve(map_dir=map_dir, name=metadata["stoplines_path"])) as f:
                stoplines = json.load(f)
        mesh = None
        if metadata.get("mesh_path"):
            from torchdrivesim.mesh import BirdviewMesh

            mesh = BirdviewMesh.load(_resolve(map_dir=map_dir, name=metadata["mesh_path"]))
        return cls(
            map_dir=map_dir,
            metadata=metadata,
            osm_path=osm_path,
            osm_bytes=os.path.getsize(osm_path),
            osm_root=etree.parse(osm_path).getroot(),
            lanelet_map=lanelet_map,
            load_errors=[str(e) for e in load_errors],
            stoplines=stoplines,
            mesh=mesh,
            visualization_fov=_read_visualization_fov(map_dir=map_dir),
        )


def _resolve(*, map_dir: str, name: str) -> str:
    return name if os.path.isabs(name) else os.path.join(map_dir, name)


def _read_visualization_fov(*, map_dir: str) -> float:
    """FOV recorded in the provenance sidecar, else the converter's historical default."""
    for path in glob.glob(os.path.join(map_dir, "*_provenance.json")):
        with open(path) as f:
            provenance = json.load(f)
        fov = provenance.get("visualization_fov") or provenance.get("config", {}).get(
            "visualization_fov"
        )
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


def routing_edge_pct(*, outputs: ConversionOutputs) -> Optional[float]:
    """Percentage of lanelets with at least one routing-graph edge (Germany, vehicle rules).

    An edge is a following, previous, left, right, adjacent-left or adjacent-right relation.
    ``None`` for a map without lanelets.
    """
    import lanelet2.traffic_rules as tr
    from lanelet2.routing import RoutingGraph

    lanelets = list(outputs.lanelet_map.laneletLayer)
    if not lanelets:
        return None
    graph = RoutingGraph(
        outputs.lanelet_map, tr.create(tr.Locations.Germany, tr.Participants.Vehicle)
    )
    connected = sum(
        1
        for ll in lanelets
        if graph.following(ll)
        or graph.previous(ll)
        or graph.left(ll)
        or graph.right(ll)
        or graph.adjacentLeft(ll)
        or graph.adjacentRight(ll)
    )
    return 100.0 * connected / len(lanelets)


def seam_handshake_pct(*, outputs: ConversionOutputs) -> Optional[float]:
    """Percentage of geometric successor seams whose boundary endpoints share point ids.

    A seam is a lanelet pair where both boundary end points of one lie within
    ``SEAM_TOLERANCE_M`` (XY) of the matching start points of the other. Routing needs the
    shared point ids, so this predicts routability independently of the routing graph.
    ``None`` when the map has no seams.
    """
    import numpy as np

    lanelets = list(outputs.lanelet_map.laneletLayer)
    if len(lanelets) < 2:
        return None
    starts = np.array(
        [[ll.leftBound[0].x, ll.leftBound[0].y, ll.rightBound[0].x, ll.rightBound[0].y] for ll in lanelets]
    )
    seams = 0
    shaken = 0
    for ll in lanelets:
        end = np.array([ll.leftBound[-1].x, ll.leftBound[-1].y, ll.rightBound[-1].x, ll.rightBound[-1].y])
        left_d = np.hypot(starts[:, 0] - end[0], starts[:, 1] - end[1])
        right_d = np.hypot(starts[:, 2] - end[2], starts[:, 3] - end[3])
        for j in np.nonzero((left_d <= SEAM_TOLERANCE_M) & (right_d <= SEAM_TOLERANCE_M))[0]:
            other = lanelets[int(j)]
            if other.id == ll.id:
                continue
            seams += 1
            if (
                ll.leftBound[-1].id == other.leftBound[0].id
                and ll.rightBound[-1].id == other.rightBound[0].id
            ):
                shaken += 1
    return 100.0 * shaken / seams if seams else None


def z_conflicts(*, outputs: ConversionOutputs) -> dict[str, float]:
    """Points at the same XY (1 cm grid) whose elevations differ by more than 1 cm.

    Returns:
        ``count``: XY positions with conflicting z; ``max_dz``: largest z spread (metres).
    """
    z_by_xy: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
    for p in outputs.lanelet_map.pointLayer:
        z_by_xy[(round(p.x * 100), round(p.y * 100))].append(p.z)
    spreads = [max(zs) - min(zs) for zs in z_by_xy.values() if len(zs) > 1]
    conflicting = [s for s in spreads if s > Z_CONFLICT_MIN_M]
    return {"count": len(conflicting), "max_dz": max(conflicting, default=0.0)}


def stoplines_by_agent_type(*, outputs: ConversionOutputs) -> dict[str, int]:
    """Stopline rows per ``agent_type``."""
    return dict(sorted(collections.Counter(row["agent_type"] for row in outputs.stoplines).items()))


def sign_ways_vs_stoplines(*, outputs: ConversionOutputs) -> dict[str, int]:
    """``traffic_sign``/``traffic_light`` ways in the ``.osm`` against the stoplines emitted.

    Sign ways with no stop/yield stopline mean traffic controls were drawn but not exported.
    """
    by_type = collections.Counter(_way_tags(way=w).get("type") for w in outputs.osm_root.iter("way"))
    by_agent = stoplines_by_agent_type(outputs=outputs)
    return {
        "traffic_sign_ways": by_type.get("traffic_sign", 0),
        "traffic_light_ways": by_type.get("traffic_light", 0),
        "sign_stoplines": by_agent.get("stop_sign", 0) + by_agent.get("yield_sign", 0),
        "light_stoplines": by_agent.get("traffic_light", 0),
    }


def untyped_way_count(*, outputs: ConversionOutputs) -> int:
    """``.osm`` ways that carry no ``type`` tag."""
    return sum(1 for w in outputs.osm_root.iter("way") if "type" not in _way_tags(way=w))


def one_way_no_count(*, outputs: ConversionOutputs) -> int:
    """Lanelets tagged ``one_way=no`` (bidirectional)."""
    return sum(
        1
        for r in outputs.osm_root.iter("relation")
        if _way_tags(way=r).get("type") == "lanelet" and _way_tags(way=r).get("one_way") == "no"
    )


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
    coords = sorted((round(p.x * 100), round(p.y * 100)) for p in outputs.lanelet_map.pointLayer)
    return hashlib.sha256(json.dumps(coords).encode()).hexdigest()


def crossed_geometry(*, outputs: ConversionOutputs) -> dict[str, int]:
    """Lanelets whose geometry folds over itself.

    Returns:
        ``reversed_bounds``: lanelets whose left and right bounds run in opposite directions,
        i.e. ``d(l0,r0) + d(l1,r1)`` exceeds ``d(l0,r1) + d(l1,r0)`` by more than
        ``CROSSED_BOUNDS_GAP_M``. ``invalid_polygons``: lanelet polygons shapely reports as
        invalid (self-intersecting).
    """
    from shapely.geometry import Polygon

    reversed_bounds = 0
    invalid = 0
    for ll in outputs.lanelet_map.laneletLayer:
        left = [(p.x, p.y) for p in ll.leftBound]
        right = [(p.x, p.y) for p in ll.rightBound]
        straight = math.dist(left[0], right[0]) + math.dist(left[-1], right[-1])
        crossed = math.dist(left[0], right[-1]) + math.dist(left[-1], right[0])
        if straight - crossed > CROSSED_BOUNDS_GAP_M:
            reversed_bounds += 1
        ring = left + right[::-1]
        if len(ring) >= 3 and not Polygon(ring).is_valid:
            invalid += 1
    return {"reversed_bounds": reversed_bounds, "invalid_polygons": invalid}


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


def _way_tags(*, way) -> dict[str, str]:
    return {tag.get("k"): tag.get("v") for tag in way.iter("tag")}


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
}


def compute_all(*, outputs: ConversionOutputs) -> dict[str, Any]:
    """Every metric in :data:`METRICS`, keyed by name."""
    return {name: metric(outputs=outputs) for name, metric in METRICS.items()}
