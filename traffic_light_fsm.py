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
from typing import Any, Optional

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
        review_threshold: Below this, an intersection is flagged for review in
            ``<map>_fsm_review.json``.
        topology_max_hops: Hops over the routing graph when testing shared-junction links.
        use_xodr_junctions: Keep each OpenDRIVE junction's lights in one intersection (needs the
            ``<map>_signal_groups.json`` sidecar). Off: distance clustering only.
        use_xodr_groups: Also take phase groups from the junction's controllers (one controller,
            one phase). Forces ``use_xodr_junctions``.
        use_xodr_sequence: Also order phases by controller ``sequence``. Forces ``use_xodr_groups``.
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
    review_threshold: float = 0.7
    topology_max_hops: int = 3
    use_xodr_junctions: bool = True
    use_xodr_groups: bool = False
    use_xodr_sequence: bool = False


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


def resolve_map_path(*, map_dir: str, name: str) -> str:
    """Absolute path of a file named in ``metadata.json`` (stored relative to the map directory)."""
    return name if os.path.isabs(name) else os.path.join(map_dir, name)


def utm_projector(*, origin_lat: float, origin_lon: float):
    """Lanelet2 ``UtmProjector`` at a map's ``lanelet_map_origin``, as every IAI tool loads maps."""
    import lanelet2
    from lanelet2.projection import UtmProjector

    return UtmProjector(lanelet2.io.Origin(origin_lat, origin_lon))


def vehicle_routing_graph(*, lanelet_map):
    """Lanelet2 routing graph for vehicles under German traffic rules."""
    import lanelet2.traffic_rules as tr
    from lanelet2.routing import RoutingGraph

    return RoutingGraph(lanelet_map, tr.create(tr.Locations.Germany, tr.Participants.Vehicle))


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


class _UnionFind:
    """Disjoint sets over integer keys (path halving)."""

    def __init__(self, *, keys) -> None:
        self._parent = {k: k for k in keys}

    def find(self, key: int) -> int:
        while self._parent[key] != key:
            self._parent[key] = self._parent[self._parent[key]]
            key = self._parent[key]
        return key

    def union(self, a: int, b: int) -> int:
        """Join the sets of ``a`` and ``b``; returns the surviving root."""
        root_a, root_b = self.find(a), self.find(b)
        self._parent[root_a] = root_b
        return root_b


def cluster_intersections(
    *, actors: list[Actor], config: FsmConfig, membership: Optional[dict[str, str]] = None
) -> list[list[int]]:
    """Group actors into intersections by spatial proximity, never merging two declared junctions.

    Returns a list of intersections, each a list of indices into ``actors``. Lights closer than
    ``cluster_eps`` are linked, shortest link first (single linkage, which equals DBSCAN with
    ``min_samples=1``), so an isolated light becomes its own intersection. ``membership`` maps an
    ``actor_id`` (as a string) to its OpenDRIVE junction: a junction's lights start joined, and a
    link that would join two different junctions is skipped, so a chain of unassigned lights can
    never bridge them.
    """
    if not actors:
        return []
    from sklearn.neighbors import NearestNeighbors

    membership = membership or {}
    sets = _UnionFind(keys=range(len(actors)))
    junctions: dict[int, set[str]] = {}
    first_of_junction: dict[str, int] = {}
    for i, actor in enumerate(actors):
        junction = membership.get(str(actor.actor_id))
        junctions[i] = {junction} if junction else set()
        if junction in first_of_junction:
            root = sets.union(i, first_of_junction[junction])
            junctions[root] = {junction}
        elif junction:
            first_of_junction[junction] = i

    positions = np.array([a.center for a in actors])
    distances, neighbours = NearestNeighbors(radius=config.cluster_eps).fit(positions).radius_neighbors(positions)
    edges = sorted(
        (float(d), i, int(k)) for i in range(len(actors)) for d, k in zip(distances[i], neighbours[i]) if i < k
    )
    skipped: dict[tuple[str, str], float] = {}
    for distance, i, k in edges:
        root_i, root_k = sets.find(i), sets.find(k)
        if root_i == root_k:
            continue
        if junctions[root_i] and junctions[root_k] and junctions[root_i] != junctions[root_k]:
            pair = tuple(sorted(junctions[root_i] | junctions[root_k]))
            skipped[pair] = min(distance, skipped.get(pair, distance))
            continue
        root = sets.union(i, k)
        junctions[root] = junctions[root_i] | junctions[root_k]
    for (a, b), distance in sorted(skipped.items()):
        logger.warning("Junctions %s and %s have lights %.1f m apart (< cluster_eps) - kept separate", a, b, distance)

    intersections: dict[int, list[int]] = {}
    for i in range(len(actors)):
        intersections.setdefault(sets.find(i), []).append(i)
    return sorted(intersections.values(), key=min)

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


def spatial_certainty(*, all_centroids: np.ndarray, centroid_index: int, config: FsmConfig) -> float:
    """Separation from the nearest other intersection, in ``[0, 1]`` (1 at ``2 * cluster_eps`` or more)."""
    if len(all_centroids) <= 1:
        return 1.0
    dists = np.linalg.norm(all_centroids - all_centroids[centroid_index], axis=1)
    dists[centroid_index] = np.inf
    return min(1.0, float(dists.min()) / (2.0 * config.cluster_eps))


def axis_certainty(*, groups: list[list[int]], actors: list[Actor], config: FsmConfig) -> float:
    """How cleanly two phase groups' axes separate, in ``[0, 1]`` (1 for a single group)."""
    if len(groups) != 2:
        return 1.0
    means = [_circular_mean_mod_pi(angles=np.array([actors[i].axis for i in g])) for g in groups]
    between = _axis_distance(a=means[0], b=means[1])
    within = max(_axis_distance(a=actors[i].axis, b=mean) for g, mean in zip(groups, means) for i in g)
    return min(1.0, between / (within + config.axis_split_threshold))


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

    The lower of :func:`spatial_certainty` and :func:`axis_certainty`. Low scores flag
    candidates for topology refinement.
    """
    return min(
        spatial_certainty(all_centroids=all_centroids, centroid_index=centroid_index, config=config),
        axis_certainty(groups=groups, actors=actors, config=config),
    )


# --------------------------------------------------------------------------------------
# Topology refinement (Plan B): used only for low-certainty intersections
# --------------------------------------------------------------------------------------


class LaneletTopology:
    """Lazy wrapper over a Lanelet2 map + routing graph, keyed by stopline geometry."""

    def __init__(self, *, osm_path: str, origin_lat: float, origin_lon: float):
        import lanelet2

        projector = utm_projector(origin_lat=origin_lat, origin_lon=origin_lon)
        self._map = lanelet2.io.load(osm_path, projector)
        self._graph = vehicle_routing_graph(lanelet_map=self._map)
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

    sets = _UnionFind(keys=intersection)
    for a_idx in intersection:
        for b_idx in intersection:
            if a_idx < b_idx and topology.shares_junction(
                a=lanelets[a_idx], b=lanelets[b_idx], max_hops=config.topology_max_hops
            ):
                sets.union(a_idx, b_idx)

    components: dict[int, list[int]] = {}
    for i in intersection:
        components.setdefault(sets.find(i), []).append(i)
    return list(components.values())


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


LEVEL_OFF = "off"
LEVEL_JUNCTIONS = "junctions"
LEVEL_GROUPS = "groups"
LEVEL_SEQUENCE = "sequence"


@dataclasses.dataclass(slots=True)
class IntersectionResult:
    """One final intersection.

    Attributes:
        actors: Indices into the actor list.
        groups: Ordered phase groups (indices into the actor list).
        junction: OpenDRIVE junction id the intersection was seeded from, or ``None``.
        level: XODR structure actually used: ``off``, ``junctions``, ``groups`` or ``sequence``.
        spatial: :func:`spatial_certainty` among the final intersections; 1.0 when an XODR
            junction declares the membership.
        axis: :func:`axis_certainty` of the phase groups; 1.0 when XODR controllers declare them.
        topology_refined: Split out of a low-certainty intersection by lanelet topology.
    """

    actors: list[int]
    groups: list[list[int]]
    junction: Optional[str]
    level: str
    spatial: float = 1.0
    axis: float = 1.0
    topology_refined: bool = False

    @property
    def certainty(self) -> float:
        """The lower of the spatial and axis parts."""
        return min(self.spatial, self.axis)

    @property
    def reason(self) -> str:
        """``declared`` when XODR fixes both parts, ``clear`` at 1.0, else the limiting part."""
        if self.level in (LEVEL_GROUPS, LEVEL_SEQUENCE):
            return "declared"
        if self.certainty >= 1.0:
            return "clear"
        return "spatial" if self.spatial <= self.axis else "axis"


def effective_xodr_levels(*, config: FsmConfig) -> FsmConfig:
    """Apply the level cascade: ``use_xodr_sequence`` forces groups on, groups force junctions on."""
    levels = dataclasses.replace(config)
    levels.use_xodr_groups = levels.use_xodr_groups or levels.use_xodr_sequence
    levels.use_xodr_junctions = levels.use_xodr_junctions or levels.use_xodr_groups
    return levels


def xodr_membership(*, signal_groups: Optional[dict], levels: FsmConfig) -> dict[str, str]:
    """``actor_id`` (as a string) -> junction, when junctions are in use and a sidecar exists."""
    if signal_groups is None or not levels.use_xodr_junctions:
        return {}
    return {sid: entry["junction"] for sid, entry in signal_groups["membership"].items()}


def unassigned_light_ids(*, actors: list[Actor], membership: dict[str, str]) -> list[int]:
    """Lights no XODR junction claims (empty when junctions are not in use)."""
    return sorted(a.actor_id for a in actors if membership and str(a.actor_id) not in membership)


def _junction_of(*, intersection: list[int], actors: list[Actor], membership: dict[str, str]) -> Optional[str]:
    junctions = {membership.get(str(actors[i].actor_id)) for i in intersection} - {None}
    return next(iter(junctions)) if len(junctions) == 1 else None


def _controller_groups(
    *, intersection: list[int], actors: list[Actor], signal_groups: dict, use_sequence: bool
) -> tuple[Optional[list[list[int]]], str]:
    """Phase groups from the XODR controllers covering this intersection's lights.

    Each controller's lights form one group; a light no controller names joins the group whose
    mean axis is nearest its own. Ordered by controller ``sequence`` when ``use_sequence`` and every
    group has one, else larger group first. Returns ``(None, level)`` when no controller applies.
    """
    by_id = {str(actors[i].actor_id): i for i in intersection}
    groups: list[tuple[Optional[int], list[int]]] = []
    placed: set[int] = set()
    for controller in signal_groups.get("controllers", []):
        members = [by_id[sid] for sid in controller["signal_ids"] if sid in by_id and by_id[sid] not in placed]
        if members:
            groups.append((controller.get("sequence"), members))
            placed.update(members)
    if not groups:
        return None, LEVEL_JUNCTIONS
    for i in intersection:
        if i in placed:
            continue
        means = [_circular_mean_mod_pi(angles=np.array([actors[j].axis for j in g])) for _, g in groups]
        nearest = min(range(len(groups)), key=lambda k: _axis_distance(a=actors[i].axis, b=means[k]))
        groups[nearest][1].append(i)
    if len(groups) > 1 and all(len(g) == 1 for _, g in groups):
        logger.warning(
            "Intersection with lights %s: every XODR controller holds a single light - each light gets its own phase",
            sorted(actors[i].actor_id for i in intersection),
        )
    if use_sequence and all(seq is not None for seq, _ in groups):
        ordered = sorted(groups, key=lambda sg: (sg[0], min(actors[i].actor_id for i in sg[1])))
        return [g for _, g in ordered], LEVEL_SEQUENCE
    ordered = sorted(groups, key=lambda sg: (-len(sg[1]), min(actors[i].actor_id for i in sg[1])))
    return [g for _, g in ordered], LEVEL_GROUPS


def resolve_intersection_groups(
    *,
    actors: list[Actor],
    config: FsmConfig,
    topology: Optional[LaneletTopology] = None,
    signal_groups: Optional[dict] = None,
) -> list[IntersectionResult]:
    """Cluster, sub-group, and (where uncertain) topology-refine the actors.

    With ``signal_groups`` (the ``<map>_signal_groups.json`` contents) and the XODR levels enabled
    in ``config``, declared junctions seed the intersections, and controllers can fix phase groups
    and their order. Both :func:`generate_fsms` and the visualization tool consume the result, so
    the rendered picture reflects exactly the groupings the FSM is built from.
    """
    levels = effective_xodr_levels(config=config)
    if levels != config:
        logger.info(
            "XODR levels in use (higher levels imply lower): junctions=%s groups=%s sequence=%s",
            levels.use_xodr_junctions,
            levels.use_xodr_groups,
            levels.use_xodr_sequence,
        )
    if signal_groups is None and levels.use_xodr_junctions:
        logger.warning("No signal groups sidecar - XODR junctions, groups and sequence are not used")
    membership = xodr_membership(signal_groups=signal_groups, levels=levels)
    intersections = cluster_intersections(actors=actors, config=config, membership=membership)
    centroids = np.array(
        [np.mean([actors[i].center for i in inter], axis=0) for inter in intersections]
    ) if intersections else np.empty((0, 2))

    resolved: list[tuple[list[int], bool]] = []  # (intersection, topology_refined)
    for idx, inter in enumerate(intersections):
        if _junction_of(intersection=inter, actors=actors, membership=membership) is not None:
            resolved.append((inter, False))
            continue
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
                (part, True)
                for part in refine_intersection_with_topology(
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
            resolved.append((inter, False))

    final_centroids = np.array([np.mean([actors[i].center for i in inter], axis=0) for inter, _ in resolved])
    results = []
    for idx, (inter, refined) in enumerate(resolved):
        junction = _junction_of(intersection=inter, actors=actors, membership=membership)
        groups, level = None, LEVEL_OFF
        if junction is not None:
            level = LEVEL_JUNCTIONS
            if levels.use_xodr_groups:
                groups, level = _controller_groups(
                    intersection=inter, actors=actors, signal_groups=signal_groups, use_sequence=levels.use_xodr_sequence
                )
        if groups is None:
            groups = group_actors(intersection=inter, actors=actors, config=config)
        results.append(
            IntersectionResult(
                actors=inter,
                groups=groups,
                junction=junction,
                level=level,
                spatial=1.0
                if junction is not None
                else spatial_certainty(all_centroids=final_centroids, centroid_index=idx, config=config),
                axis=1.0
                if level in (LEVEL_GROUPS, LEVEL_SEQUENCE)
                else axis_certainty(groups=groups, actors=actors, config=config),
                topology_refined=refined,
            )
        )
    unassigned = unassigned_light_ids(actors=actors, membership=membership)
    if unassigned:
        placed = {actors[i].actor_id: r.junction for r in results for i in r.actors}
        logger.warning(
            "%d light(s) belong to no XODR junction; placed by distance: %s",
            len(unassigned),
            ", ".join(f"{aid}->{placed[aid] or 'own intersection'}" for aid in unassigned),
        )
    return results


def flagged_intersections(*, resolved: list[IntersectionResult], config: FsmConfig) -> list[int]:
    """Indices of the intersections whose certainty is below ``config.review_threshold``."""
    return [i for i, r in enumerate(resolved) if r.certainty < config.review_threshold]


def generate_fsms(
    *,
    actors: list[Actor],
    config: FsmConfig,
    topology: Optional[LaneletTopology] = None,
    signal_groups: Optional[dict] = None,
) -> list[list[dict]]:
    """Run the full pipeline and return one FSM (state list) per intersection."""
    resolved = resolve_intersection_groups(actors=actors, config=config, topology=topology, signal_groups=signal_groups)
    return [build_fsm(groups=r.groups, actors=actors, config=config) for r in resolved]


def stopline_fsm_join(*, stoplines_path: str, fsms: list[list[dict]]) -> dict[str, Any]:
    """Check that the controller file and the stoplines name the same traffic lights.

    Every traffic-light ``actor_id`` in the stoplines should appear in exactly one FSM, and the FSMs
    should name no other actor.
    """
    with open(stoplines_path) as f:
        stopline_ids = {str(row["actor_id"]) for row in json.load(f) if row.get("agent_type") == "traffic_light"}
    seen: dict[str, int] = {}
    for fsm in fsms:
        for aid in {aid for state in fsm for aid in state["actor_states"]}:
            seen[aid] = seen.get(aid, 0) + 1
    missing = sorted(stopline_ids - set(seen), key=_id_order)
    unknown = sorted(set(seen) - stopline_ids, key=_id_order)
    several = sorted((aid for aid, n in seen.items() if n > 1), key=_id_order)
    return {
        "stopline_lights": len(stopline_ids),
        "fsm_lights": len(seen),
        "missing_from_fsm": missing,
        "not_in_stoplines": unknown,
        "in_several_intersections": several,
        "ok": not (missing or unknown or several),
    }


def _id_order(actor_id: str) -> tuple:
    return (0, int(actor_id), "") if actor_id.lstrip("-").isdigit() else (1, 0, actor_id)


def find_signal_groups_path(*, stoplines_path: str) -> Optional[str]:
    """``<map>_signal_groups.json`` beside ``<map>_stoplines.json``, if it exists."""
    path = stoplines_path[: -len("_stoplines.json")] + "_signal_groups.json"
    return path if os.path.exists(path) else None


def load_signal_groups(*, stoplines_path: str, override: Optional[str] = None) -> tuple[Optional[str], Optional[dict]]:
    """The signal groups sidecar the FSM uses: ``override`` if given, else the one beside the stoplines.

    Returns:
        ``(absolute path, contents)``, or ``(None, None)`` when there is no sidecar.
    """
    path = override or find_signal_groups_path(stoplines_path=stoplines_path)
    if path is None:
        return None, None
    path = os.path.abspath(path)
    with open(path) as f:
        groups = json.load(f)
    logger.info("Using XODR signal groups from %s", path)
    return path, groups


def _load_topology(*, map_dir: str, metadata: dict) -> Optional[LaneletTopology]:
    """Build a :class:`LaneletTopology`, or return ``None`` if unavailable."""
    osm_name = metadata.get("lanelet_path")
    origin = metadata.get("lanelet_map_origin")
    if not osm_name or not origin:
        logger.warning("metadata.json lacks lanelet_path/lanelet_map_origin - topology disabled")
        return None
    osm_path = resolve_map_path(map_dir=map_dir, name=osm_name)
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
    signal_groups_path: Optional[str] = None,
) -> str:
    """Generate FSMs for a map directory and write the controller JSON and the review JSON.

    ``signal_groups_path`` overrides the ``<map>_signal_groups.json`` found beside the stoplines.
    If ``visualize`` is set, also render a birdview mosaic of the map with the intersection
    clusters and phase subgroups overlaid (see :mod:`fsm_visualization`).

    Returns the path to the written ``*_traffic_light_controller.json``.
    """
    map_dir = os.path.abspath(map_dir)
    metadata = load_metadata(map_dir=map_dir)
    stoplines_path = find_stoplines_path(map_dir=map_dir)
    actors = load_actors(stoplines_path=stoplines_path)
    logger.info("Loaded %d traffic-light actors from %s", len(actors), stoplines_path)
    signal_groups_path, signal_groups = load_signal_groups(stoplines_path=stoplines_path, override=signal_groups_path)

    topology = _load_topology(map_dir=map_dir, metadata=metadata) if use_topology else None
    resolved = resolve_intersection_groups(actors=actors, config=config, topology=topology, signal_groups=signal_groups)
    fsms = [build_fsm(groups=r.groups, actors=actors, config=config) for r in resolved]
    logger.info("Generated FSMs for %d intersections", len(fsms))

    location = metadata.get("name") or os.path.basename(os.path.normpath(map_dir))
    out_path = os.path.join(map_dir, f"{location}_traffic_light_controller.json")
    with open(out_path, "w") as f:
        json.dump(fsms, f, indent=4)
    logger.info("Wrote traffic light controller to %s", out_path)
    review_path = write_review(
        map_dir=map_dir,
        location=location,
        resolved=resolved,
        actors=actors,
        config=config,
        signal_groups_path=signal_groups_path,
        signal_groups=signal_groups,
        stopline_join=stopline_fsm_join(stoplines_path=stoplines_path, fsms=fsms),
    )
    logger.info("Wrote FSM review to %s", review_path)

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


def write_review(
    *,
    map_dir: str,
    location: str,
    resolved: list[IntersectionResult],
    actors: list[Actor],
    config: FsmConfig,
    signal_groups_path: Optional[str],
    signal_groups: Optional[dict],
    stopline_join: Optional[dict] = None,
) -> str:
    """Write ``<map>_fsm_review.json``: per intersection, its lights, phase groups, XODR level and certainty.

    ``flagged`` repeats the intersections below ``config.review_threshold``. Also lists the lights no
    XODR junction claims (placed by distance), when junctions are in use, and ``stopline_join``
    (:func:`stopline_fsm_join`) when given.
    """
    levels = effective_xodr_levels(config=config)
    intersections = [
        {
            "junction": r.junction,
            "level": r.level,
            "lights": sorted(actors[i].actor_id for i in r.actors),
            "groups": [sorted(actors[i].actor_id for i in g) for g in r.groups],
            "certainty": round(r.certainty, 3),
            "spatial": round(r.spatial, 3),
            "axis": round(r.axis, 3),
            "reason": r.reason,
            "topology_refined": r.topology_refined,
        }
        for r in resolved
    ]
    review = {
        "levels_requested": {
            "use_xodr_junctions": levels.use_xodr_junctions,
            "use_xodr_groups": levels.use_xodr_groups,
            "use_xodr_sequence": levels.use_xodr_sequence,
        },
        "signal_groups": signal_groups_path,
        "unassigned_lights": unassigned_light_ids(
            actors=actors, membership=xodr_membership(signal_groups=signal_groups, levels=levels)
        ) if signal_groups is not None and levels.use_xodr_junctions else None,
        "review_threshold": config.review_threshold,
        "intersections": intersections,
        "flagged": [
            {"index": i, **intersections[i]} for i in flagged_intersections(resolved=resolved, config=config)
        ],
        "stopline_join": stopline_join,
    }
    path = os.path.join(map_dir, f"{location}_fsm_review.json")
    with open(path, "w") as f:
        json.dump(review, f, indent=2)
    return path


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
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=d.review_threshold,
        help="Flag intersections below this certainty in <map>_fsm_review.json",
    )
    parser.add_argument("--topology-max-hops", type=int, default=d.topology_max_hops)
    parser.add_argument("--no-topology", action="store_true", help="Disable lanelet topology refinement")
    parser.add_argument(
        "--xodr-junctions",
        action=argparse.BooleanOptionalAction,
        default=d.use_xodr_junctions,
        help="Keep each XODR junction's lights in one intersection (default on)",
    )
    parser.add_argument(
        "--xodr-groups",
        action=argparse.BooleanOptionalAction,
        default=d.use_xodr_groups,
        help="Also take phase groups from XODR controllers (implies --xodr-junctions)",
    )
    parser.add_argument(
        "--xodr-sequence",
        action=argparse.BooleanOptionalAction,
        default=d.use_xodr_sequence,
        help="Also order phases by controller sequence (implies --xodr-groups)",
    )
    parser.add_argument(
        "--signal-groups",
        default=None,
        help="Signal groups JSON (default: <map>_signal_groups.json beside the stoplines)",
    )
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
    """CLI entry point: parse arguments and run :func:`run`."""
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
        review_threshold=args.review_threshold,
        topology_max_hops=args.topology_max_hops,
        use_xodr_junctions=args.xodr_junctions,
        use_xodr_groups=args.xodr_groups,
        use_xodr_sequence=args.xodr_sequence,
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
        signal_groups_path=args.signal_groups,
    )


if __name__ == "__main__":
    main()
