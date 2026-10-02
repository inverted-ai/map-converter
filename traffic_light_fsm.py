"""Auto-generate traffic-light finite state machines (FSMs) from converted-map artifacts.

This is a standalone tool. Given a map directory containing ``metadata.json``, a Lanelet2
``.osm`` and a ``*_stoplines.json`` (the artifacts produced by ``convert_map.py``), it
clusters traffic-light stoplines into intersections, splits each intersection into at most
two phase groups, and emits a per-intersection FSM in the format consumed by the IAI stack
(see ``maps/Carla_UE4_old/Town15/carla_Town15_light_group_states.json``).

The tool never relies on OpenDRIVE, CommonRoad, or traffic-light positions: ``actor_id`` is
treated as an opaque grouping key already present in ``stoplines.json``, and the only
geometry it trusts is the stoplines themselves.

Two strategies are combined:
  * Geometry (always): cluster by position, sub-group by stopline axis (orientation mod pi).
  * Topology (only where geometry is uncertain): associate stoplines to lanelets in the
    ``.osm`` and use lane connectivity/adjacency to correct the geometric guess.

Run with ``python traffic_light_fsm.py <map-dir>``.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import logging
import math
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

GREEN = "green"
YELLOW = "yellow"
RED = "red"


@dataclasses.dataclass(slots=True)
class FsmConfig:
    """Tunable parameters for FSM generation.

    Attributes:
        cluster_eps: Max distance (metres) linking two lights into the same intersection.
        axis_split_threshold: Min angular gap (radians, on ``[0, pi)``) required to split an
            intersection into two phase groups rather than one.
        green_base: Green duration (seconds) for a single-approach group.
        green_per_extra_lane: Extra green seconds added per additional stopline in a group.
        green_min: Lower clamp on green duration (seconds).
        green_max: Upper clamp on green duration (seconds).
        yellow: Yellow duration (seconds).
        all_red: All-red clearance duration (seconds) between phases.
        certainty_threshold: Below this, an intersection is refined with lanelet topology.
        topology_max_hops: Hops over the routing graph when testing shared-junction links.
    """

    cluster_eps: float = 50.0
    axis_split_threshold: float = 0.5
    green_base: float = 20.0
    green_per_extra_lane: float = 2.5
    green_min: float = 15.0
    green_max: float = 30.0
    yellow: float = 5.0
    all_red: float = 3.0
    certainty_threshold: float = 0.5
    topology_max_hops: int = 3


@dataclasses.dataclass(slots=True)
class Actor:
    """One controllable traffic light, aggregated from all stoplines sharing an ``actor_id``."""

    actor_id: int
    center: np.ndarray  # shape (2,)
    axis: float  # orientation mod pi, in [0, pi)
    lane_count: int  # number of distinct stopline segments (approach-width proxy)


# --------------------------------------------------------------------------------------
# Input loading
# --------------------------------------------------------------------------------------


def load_metadata(*, map_dir: str) -> dict:
    """Read ``metadata.json`` from a map directory."""
    path = os.path.join(map_dir, "metadata.json")
    with open(path) as f:
        return json.load(f)


def find_stoplines_path(*, map_dir: str) -> str:
    """Locate the ``*_stoplines.json`` file in a map directory."""
    matches = sorted(glob.glob(os.path.join(map_dir, "*_stoplines.json")))
    if not matches:
        raise FileNotFoundError(f"No *_stoplines.json found in {map_dir}")
    return matches[0]


def load_actors(*, stoplines_path: str) -> list[Actor]:
    """Aggregate ``stoplines.json`` rows into one :class:`Actor` per traffic-light ``actor_id``.

    Only ``agent_type == "traffic_light"`` rows are used; stop and yield signs are static and
    excluded. Rows sharing an ``actor_id`` are merged: the centre is the mean of segment
    centres, the axis is the circular mean of ``orientation mod pi``, and ``lane_count`` is
    the number of distinct segments (a proxy for how wide the approaching road is).
    """
    with open(stoplines_path) as f:
        rows = json.load(f)

    by_id: dict[int, list[dict]] = {}
    for row in rows:
        if row.get("agent_type") != "traffic_light":
            continue
        by_id.setdefault(row["actor_id"], []).append(row)

    actors: list[Actor] = []
    for actor_id, group in by_id.items():
        centers = np.array([(r["x"], r["y"]) for r in group], dtype=float)
        unique_centers = {(round(r["x"], 2), round(r["y"], 2)) for r in group}
        orientations = np.array([r["orientation"] for r in group], dtype=float)
        actors.append(
            Actor(
                actor_id=actor_id,
                center=centers.mean(axis=0),
                axis=_circular_mean_mod_pi(angles=orientations),
                lane_count=len(unique_centers),
            )
        )
    return actors


# --------------------------------------------------------------------------------------
# Angle helpers (axes live on the circle [0, pi); opposite directions share an axis)
# --------------------------------------------------------------------------------------


def _circular_mean_mod_pi(*, angles: np.ndarray) -> float:
    """Circular mean of angles taken modulo pi, returned in ``[0, pi)``."""
    doubled = 2.0 * angles
    mean = math.atan2(float(np.sin(doubled).mean()), float(np.cos(doubled).mean()))
    return (mean / 2.0) % math.pi


def _axis_distance(*, a: float, b: float) -> float:
    """Distance between two axes on the circle of circumference pi."""
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


# --------------------------------------------------------------------------------------
# Stage 1: cluster lights into intersections
# --------------------------------------------------------------------------------------


def cluster_intersections(*, actors: list[Actor], config: FsmConfig) -> list[list[int]]:
    """Group actors into intersections by spatial proximity.

    Returns a list of intersections, each a list of indices into ``actors``. Uses DBSCAN
    with ``min_samples=1`` so an isolated light becomes its own (single-phase) intersection.
    """
    if not actors:
        return []
    if len(actors) == 1:
        return [[0]]

    from sklearn.cluster import DBSCAN

    positions = np.array([a.center for a in actors])
    labels = DBSCAN(eps=config.cluster_eps, min_samples=1).fit_predict(positions)

    intersections: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        intersections.setdefault(int(label), []).append(idx)
    return [intersections[k] for k in sorted(intersections)]


# --------------------------------------------------------------------------------------
# Stage 2: split an intersection into <= 2 phase groups by stopline axis
# --------------------------------------------------------------------------------------


def split_axes_into_groups(*, axes: list[float], config: FsmConfig) -> list[int]:
    """Cluster axes on the circle ``[0, pi)`` into one or two groups.

    The axes are split into two groups only if the second-largest circular gap between
    neighbouring axes exceeds ``axis_split_threshold``; the two largest gaps then become the
    cut points dividing the circle into two arcs. This handles the wrap between angles near 0
    and near pi and makes no assumption that the two roads cross at 90 degrees. Returns a
    group label (0 or 1) per input axis.
    """
    n = len(axes)
    if n <= 1:
        return [0] * n

    order = sorted(range(n), key=lambda i: axes[i])
    sorted_axes = [axes[i] for i in order]

    # Circular gap after each sorted axis (the last wraps from the max axis back to the min).
    gaps = [sorted_axes[i + 1] - sorted_axes[i] for i in range(n - 1)]
    gaps.append(sorted_axes[0] + math.pi - sorted_axes[-1])

    largest_two = sorted(range(n), key=lambda i: gaps[i], reverse=True)[:2]
    if gaps[largest_two[1]] < config.axis_split_threshold:
        return [0] * n

    # Walk the sorted ring; toggle group each time we cross one of the two cut gaps.
    cut_positions = set(largest_two)
    labels = [0] * n
    label = 0
    for pos in range(n):
        labels[order[pos]] = label
        if pos in cut_positions:
            label ^= 1
    return labels


def group_actors(*, intersection: list[int], actors: list[Actor], config: FsmConfig) -> list[list[int]]:
    """Return up to two phase groups (lists of indices into ``actors``) for one intersection."""
    axes = [actors[i].axis for i in intersection]
    labels = split_axes_into_groups(axes=axes, config=config)
    groups: dict[int, list[int]] = {}
    for idx, label in zip(intersection, labels):
        groups.setdefault(label, []).append(idx)
    # Larger group first for deterministic, stable output ordering.
    return sorted(groups.values(), key=lambda g: (-len(g), min(actors[i].actor_id for i in g)))


# --------------------------------------------------------------------------------------
# Stage 3: assemble the FSM
# --------------------------------------------------------------------------------------


def _green_duration(*, group: list[int], actors: list[Actor], config: FsmConfig) -> int:
    """Green time for a phase group, scaled by the size of the approaching road."""
    size = sum(actors[i].lane_count for i in group)
    raw = config.green_base + config.green_per_extra_lane * (size - 1)
    return int(round(min(config.green_max, max(config.green_min, raw))))


def _state(*, actor_states: dict[str, str], duration: int, state: int, next_state: int) -> dict:
    return {
        "actor_states": actor_states,
        "duration": str(duration),
        "state": state,
        "next_state": next_state,
    }


def build_fsm(*, groups: list[list[int]], actors: list[Actor], config: FsmConfig) -> list[dict]:
    """Build the looping state list for one intersection.

    One group: green -> yellow -> all-red. Two groups: each cycles green -> yellow while the
    other holds red, with an all-red clearance between phases.
    """
    yellow = int(round(config.yellow))
    all_red = int(round(config.all_red))
    ids_by_group = [[str(actors[i].actor_id) for i in g] for g in groups]
    all_ids = [aid for ids in ids_by_group for aid in ids]

    def states_for(active: int, color: str) -> dict[str, str]:
        return {aid: color if aid in ids_by_group[active] else RED for aid in all_ids}

    if len(groups) == 1:
        green = _green_duration(group=groups[0], actors=actors, config=config)
        return [
            _state(actor_states=states_for(0, GREEN), duration=green, state=0, next_state=1),
            _state(actor_states=states_for(0, YELLOW), duration=yellow, state=1, next_state=2),
            _state(actor_states=states_for(0, RED), duration=all_red, state=2, next_state=0),
        ]

    fsm: list[dict] = []
    per_phase = 3  # green, yellow, all-red
    for phase, group in enumerate(groups):
        green = _green_duration(group=group, actors=actors, config=config)
        base = phase * per_phase
        last = (base + per_phase) % (per_phase * len(groups))
        fsm.append(_state(actor_states=states_for(phase, GREEN), duration=green, state=base, next_state=base + 1))
        fsm.append(_state(actor_states=states_for(phase, YELLOW), duration=yellow, state=base + 1, next_state=base + 2))
        fsm.append(_state(actor_states={aid: RED for aid in all_ids}, duration=all_red, state=base + 2, next_state=last))
    return fsm


# --------------------------------------------------------------------------------------
# Certainty metric: how much to trust the geometric grouping of an intersection
# --------------------------------------------------------------------------------------


def intersection_certainty(
    *,
    intersection: list[int],
    groups: list[list[int]],
    actors: list[Actor],
    all_centroids: np.ndarray,
    centroid_index: int,
    config: FsmConfig,
) -> float:
    """Score in ``[0, 1]`` for how confidently geometry resolved this intersection.

    Combines (via min) spatial separation from the nearest other intersection and the
    quality of the axis split. Low scores flag candidates for topology refinement.
    """
    spatial = 1.0
    if len(all_centroids) > 1:
        deltas = all_centroids - all_centroids[centroid_index]
        dists = np.linalg.norm(deltas, axis=1)
        dists[centroid_index] = np.inf
        nearest = float(dists.min())
        spatial = min(1.0, nearest / (2.0 * config.cluster_eps))

    axis_quality = 1.0
    if len(groups) == 2:
        axes0 = [actors[i].axis for i in groups[0]]
        axes1 = [actors[i].axis for i in groups[1]]
        mean0 = _circular_mean_mod_pi(angles=np.array(axes0))
        mean1 = _circular_mean_mod_pi(angles=np.array(axes1))
        between = _axis_distance(a=mean0, b=mean1)
        within = max(
            (_axis_distance(a=actors[i].axis, b=mean0) for i in groups[0]),
            default=0.0,
        )
        within = max(
            within,
            max((_axis_distance(a=actors[i].axis, b=mean1) for i in groups[1]), default=0.0),
        )
        axis_quality = min(1.0, between / (within + config.axis_split_threshold))

    return min(spatial, axis_quality)


# --------------------------------------------------------------------------------------
# Topology refinement (Plan B): used only for low-certainty intersections
# --------------------------------------------------------------------------------------


class LaneletTopology:
    """Lazy wrapper over a Lanelet2 map + routing graph, keyed by stopline geometry."""

    def __init__(self, *, osm_path: str, origin_lat: float, origin_lon: float):
        import lanelet2
        from lanelet2.projection import UtmProjector
        import lanelet2.traffic_rules as tr
        from lanelet2.routing import RoutingGraph

        projector = UtmProjector(lanelet2.io.Origin(origin_lat, origin_lon))
        self._map = lanelet2.io.load(osm_path, projector)
        rules = tr.create(tr.Locations.Germany, tr.Participants.Vehicle)
        self._graph = RoutingGraph(self._map, rules)
        self._endpoints = [
            (ll, np.array([ll.centerline[-1].x, ll.centerline[-1].y])) for ll in self._map.laneletLayer
        ]

    def lanelet_for(self, *, center: np.ndarray) -> Optional[object]:
        """Lanelet whose terminal centerline point is nearest the stopline centre."""
        best, best_d = None, math.inf
        for lanelet, endpoint in self._endpoints:
            d = float(np.linalg.norm(endpoint - center))
            if d < best_d:
                best, best_d = lanelet, d
        return best

    def shares_junction(self, *, a: object, b: object, max_hops: int) -> bool:
        """True if lanelets ``a`` and ``b`` reach a common lanelet within ``max_hops``."""
        if a is None or b is None:
            return False
        reach_a = self._reachable(start=a, max_hops=max_hops)
        reach_b = self._reachable(start=b, max_hops=max_hops)
        return bool(reach_a & reach_b)

    def _reachable(self, *, start: object, max_hops: int) -> set[int]:
        seen = {start.id}
        frontier = [start]
        for _ in range(max_hops):
            nxt = []
            for lanelet in frontier:
                for follower in self._graph.following(lanelet):
                    if follower.id not in seen:
                        seen.add(follower.id)
                        nxt.append(follower)
            frontier = nxt
        return seen


def refine_intersection_with_topology(
    *,
    intersection: list[int],
    actors: list[Actor],
    topology: LaneletTopology,
    config: FsmConfig,
) -> list[list[int]]:
    """Split a low-certainty intersection by lane connectivity.

    Actors whose approach lanelets reach a common junction stay together; topologically
    disconnected actors are separated into distinct intersections. This corrects spatial
    over-merging that a single global ``eps`` cannot detect.
    """
    lanelets = {i: topology.lanelet_for(center=actors[i].center) for i in intersection}

    parent = {i: i for i in intersection}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a_idx in intersection:
        for b_idx in intersection:
            if a_idx < b_idx and topology.shares_junction(
                a=lanelets[a_idx], b=lanelets[b_idx], max_hops=config.topology_max_hops
            ):
                parent[find(a_idx)] = find(b_idx)

    components: dict[int, list[int]] = {}
    for i in intersection:
        components.setdefault(find(i), []).append(i)
    return list(components.values())


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def resolve_intersection_groups(
    *,
    actors: list[Actor],
    config: FsmConfig,
    topology: Optional[LaneletTopology] = None,
) -> list[tuple[list[int], list[list[int]]]]:
    """Cluster, sub-group, and (where uncertain) topology-refine the actors.

    Returns one entry per final intersection: a tuple of (actor indices in the intersection,
    ordered phase groups). Both :func:`generate_fsms` and the visualization tool consume this
    so the rendered picture reflects exactly the groupings the FSM is built from.
    """
    intersections = cluster_intersections(actors=actors, config=config)
    centroids = np.array(
        [np.mean([actors[i].center for i in inter], axis=0) for inter in intersections]
    ) if intersections else np.empty((0, 2))

    resolved: list[list[int]] = []
    for idx, inter in enumerate(intersections):
        groups = group_actors(intersection=inter, actors=actors, config=config)
        certainty = intersection_certainty(
            intersection=inter,
            groups=groups,
            actors=actors,
            all_centroids=centroids,
            centroid_index=idx,
            config=config,
        )
        if topology is not None and certainty < config.certainty_threshold:
            logger.info(
                "Intersection %d certainty %.2f below %.2f - refining with topology",
                idx,
                certainty,
                config.certainty_threshold,
            )
            resolved.extend(
                refine_intersection_with_topology(
                    intersection=inter, actors=actors, topology=topology, config=config
                )
            )
        else:
            if topology is None and certainty < config.certainty_threshold:
                logger.warning(
                    "Intersection %d certainty %.2f below %.2f but no topology available "
                    "- emitting geometry-only result for review",
                    idx,
                    certainty,
                    config.certainty_threshold,
                )
            resolved.append(inter)

    return [
        (inter, group_actors(intersection=inter, actors=actors, config=config))
        for inter in resolved
    ]


def generate_fsms(
    *,
    actors: list[Actor],
    config: FsmConfig,
    topology: Optional[LaneletTopology] = None,
) -> list[list[dict]]:
    """Run the full pipeline and return one FSM (state list) per intersection."""
    resolved = resolve_intersection_groups(actors=actors, config=config, topology=topology)
    return [build_fsm(groups=groups, actors=actors, config=config) for _, groups in resolved]


def _load_topology(*, map_dir: str, metadata: dict) -> Optional[LaneletTopology]:
    """Build a :class:`LaneletTopology`, or return ``None`` if unavailable."""
    osm_name = metadata.get("lanelet_path")
    origin = metadata.get("lanelet_map_origin")
    if not osm_name or not origin:
        logger.warning("metadata.json lacks lanelet_path/lanelet_map_origin - topology disabled")
        return None
    osm_path = osm_name if os.path.isabs(osm_name) else os.path.join(map_dir, osm_name)
    if not os.path.exists(osm_path):
        logger.warning("OSM file %s not found - topology disabled", osm_path)
        return None
    try:
        return LaneletTopology(osm_path=osm_path, origin_lat=origin[0], origin_lon=origin[1])
    except Exception as exc:  # noqa: BLE001 - degrade gracefully to geometry-only
        logger.warning("Could not load lanelet topology (%s) - topology disabled", exc)
        return None


def run(
    *,
    map_dir: str,
    config: FsmConfig,
    use_topology: bool = True,
    visualize: bool = False,
    viz_config: Optional[object] = None,
) -> str:
    """Generate FSMs for a map directory and write the controller JSON.

    If ``visualize`` is set, also render a birdview mosaic of the map with the intersection
    clusters and phase subgroups overlaid (see :mod:`fsm_visualization`).

    Returns the path to the written ``*_traffic_light_controller.json``.
    """
    metadata = load_metadata(map_dir=map_dir)
    stoplines_path = find_stoplines_path(map_dir=map_dir)
    actors = load_actors(stoplines_path=stoplines_path)
    logger.info("Loaded %d traffic-light actors from %s", len(actors), stoplines_path)

    topology = _load_topology(map_dir=map_dir, metadata=metadata) if use_topology else None
    resolved = resolve_intersection_groups(actors=actors, config=config, topology=topology)
    fsms = [build_fsm(groups=groups, actors=actors, config=config) for _, groups in resolved]
    logger.info("Generated FSMs for %d intersections", len(fsms))

    location = metadata.get("name") or os.path.basename(os.path.normpath(map_dir))
    out_path = os.path.join(map_dir, f"{location}_traffic_light_controller.json")
    with open(out_path, "w") as f:
        json.dump(fsms, f, indent=4)
    logger.info("Wrote traffic light controller to %s", out_path)

    metadata["traffic_light_controller_path"] = os.path.basename(out_path)
    with open(os.path.join(map_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    if visualize:
        import fsm_visualization  # lazy: keeps torch/torchdrivesim off the core import path

        viz_path = fsm_visualization.render_mosaic(
            map_dir=map_dir,
            metadata=metadata,
            actors=actors,
            resolved=resolved,
            config=viz_config or fsm_visualization.VizConfig(),
        )
        logger.info("Wrote FSM visualization to %s", viz_path)

    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    d = FsmConfig()  # slots=True means class attributes are descriptors; read defaults off an instance
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_dir", help="Map directory with metadata.json, .osm and *_stoplines.json")
    parser.add_argument("--cluster-eps", type=float, default=d.cluster_eps)
    parser.add_argument("--axis-split-threshold", type=float, default=d.axis_split_threshold)
    parser.add_argument("--green-base", type=float, default=d.green_base)
    parser.add_argument("--green-per-extra-lane", type=float, default=d.green_per_extra_lane)
    parser.add_argument("--green-min", type=float, default=d.green_min)
    parser.add_argument("--green-max", type=float, default=d.green_max)
    parser.add_argument("--yellow", type=float, default=d.yellow)
    parser.add_argument("--all-red", type=float, default=d.all_red)
    parser.add_argument("--certainty-threshold", type=float, default=d.certainty_threshold)
    parser.add_argument("--topology-max-hops", type=int, default=d.topology_max_hops)
    parser.add_argument("--no-topology", action="store_true", help="Disable lanelet topology refinement")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also render a birdview mosaic with intersections/subgroups overlaid",
    )
    parser.add_argument(
        "--px-per-m",
        type=float,
        default=None,
        help="Visualization pixels per metre (default from VizConfig; only used with --visualize)",
    )
    parser.add_argument(
        "--max-mosaic-px",
        type=int,
        default=None,
        help="Visualization max image dimension in pixels (default from VizConfig)",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()
    config = FsmConfig(
        cluster_eps=args.cluster_eps,
        axis_split_threshold=args.axis_split_threshold,
        green_base=args.green_base,
        green_per_extra_lane=args.green_per_extra_lane,
        green_min=args.green_min,
        green_max=args.green_max,
        yellow=args.yellow,
        all_red=args.all_red,
        certainty_threshold=args.certainty_threshold,
        topology_max_hops=args.topology_max_hops,
    )

    viz_config = None
    if args.visualize:
        import fsm_visualization

        viz_config = fsm_visualization.VizConfig()
        if args.px_per_m is not None:
            viz_config.target_px_per_m = args.px_per_m
        if args.max_mosaic_px is not None:
            viz_config.max_mosaic_px = args.max_mosaic_px

    run(
        map_dir=args.map_dir,
        config=config,
        use_topology=not args.no_topology,
        visualize=args.visualize,
        viz_config=viz_config,
    )


if __name__ == "__main__":
    main()
