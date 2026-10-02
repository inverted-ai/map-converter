"""Tests for the standalone traffic-light FSM generator (``traffic_light_fsm.py``)."""

import math
import os

import numpy as np
import pytest

import traffic_light_fsm as t

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOWN15_STOPLINES = os.path.join(
    REPO_ROOT, "maps", "Carla_UE4_old", "Town15", "Town15_stoplines.json"
)


def _actor(*, actor_id: int, x: float, y: float, axis: float, lane_count: int = 1) -> t.Actor:
    return t.Actor(actor_id=actor_id, center=np.array([x, y]), axis=axis % math.pi, lane_count=lane_count)


def _groupings(*, actors: list[t.Actor], config: t.FsmConfig) -> list[list[list[int]]]:
    """Return sorted per-intersection actor-id groupings for easy comparison."""
    out = []
    for inter in t.cluster_intersections(actors=actors, config=config):
        groups = t.group_actors(intersection=inter, actors=actors, config=config)
        out.append(sorted(sorted(actors[i].actor_id for i in g) for g in groups))
    return sorted(out, key=lambda x: min(min(s) for s in x))


# --------------------------------------------------------------------------------------
# Ground-truth regression on the hand-authored Town15 map
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(TOWN15_STOPLINES), reason="Town15 fixture absent")
def test_town15_matches_manual_groupings():
    config = t.FsmConfig()
    actors = t.load_actors(stoplines_path=TOWN15_STOPLINES)
    got = _groupings(actors=actors, config=config)

    # Sub-groups (sets of actor ids sharing a phase) hand-authored in
    # maps/Carla_UE4_old/Town15/carla_Town15_light_group_states.json.
    expected = sorted(
        [
            [[1866, 1868, 1896], [1867]],
            [[1869, 1871], [1870]],
            [[1877, 1879], [1878]],
            [[1880, 1881], [1882, 1883]],
            [[1884, 1885], [1886, 1887]],
            [[1888, 1890], [1889]],
            [[1891, 1893], [1892, 1897]],
            [[1894, 1895]],
        ],
        key=lambda x: min(min(s) for s in x),
    )
    assert got == expected


# --------------------------------------------------------------------------------------
# Synthetic geometry units (no .osm needed)
# --------------------------------------------------------------------------------------


def test_perpendicular_four_way_splits_into_two_groups():
    actors = [
        _actor(actor_id=1, x=-5, y=0, axis=0.0),  # E-W approach
        _actor(actor_id=2, x=5, y=0, axis=math.pi - 0.01),  # opposing E-W (mod pi ~ 0)
        _actor(actor_id=3, x=0, y=-5, axis=math.pi / 2),  # N-S approach
        _actor(actor_id=4, x=0, y=5, axis=math.pi / 2),  # opposing N-S
    ]
    groups = _groupings(actors=actors, config=t.FsmConfig())[0]
    assert groups == [[1, 2], [3, 4]]


def test_skewed_four_way_no_orthogonality_assumption():
    # Two roads crossing at ~60 degrees, not 90.
    skew = math.pi / 3
    actors = [
        _actor(actor_id=1, x=-5, y=0, axis=0.0),
        _actor(actor_id=2, x=5, y=0, axis=0.02),
        _actor(actor_id=3, x=0, y=-5, axis=skew),
        _actor(actor_id=4, x=0, y=5, axis=skew + 0.02),
    ]
    groups = _groupings(actors=actors, config=t.FsmConfig())[0]
    assert groups == [[1, 2], [3, 4]]


def test_three_way_t_keeps_collinear_pair_together():
    actors = [
        _actor(actor_id=1, x=-5, y=0, axis=0.0),  # through road
        _actor(actor_id=2, x=5, y=0, axis=0.0),  # through road, opposite
        _actor(actor_id=3, x=0, y=-5, axis=math.pi / 2),  # the stem
    ]
    groups = _groupings(actors=actors, config=t.FsmConfig())[0]
    assert groups == [[1, 2], [3]]


def test_single_onramp_is_one_group():
    # Two near-parallel lights (a merge) should not be split.
    actors = [
        _actor(actor_id=1, x=0, y=0, axis=0.1),
        _actor(actor_id=2, x=4, y=0, axis=0.12),
    ]
    groups = _groupings(actors=actors, config=t.FsmConfig())[0]
    assert groups == [[1, 2]]


def test_eps_merges_when_close_and_splits_when_far():
    near = [
        _actor(actor_id=1, x=0, y=0, axis=0.0),
        _actor(actor_id=2, x=0, y=30, axis=math.pi / 2),
    ]
    far = [
        _actor(actor_id=1, x=0, y=0, axis=0.0),
        _actor(actor_id=2, x=0, y=300, axis=math.pi / 2),
    ]
    config = t.FsmConfig(cluster_eps=50)
    assert len(t.cluster_intersections(actors=near, config=config)) == 1
    assert len(t.cluster_intersections(actors=far, config=config)) == 2


# --------------------------------------------------------------------------------------
# FSM assembly validity
# --------------------------------------------------------------------------------------


def _assert_loops(fsm: list[dict]):
    by_state = {s["state"]: s for s in fsm}
    assert len(by_state) == len(fsm)  # unique state indices
    visited, cur = set(), fsm[0]["state"]
    for _ in range(len(fsm)):
        visited.add(cur)
        cur = by_state[cur]["next_state"]
    assert cur == fsm[0]["state"]  # closes back to start
    assert visited == set(by_state)  # single cycle visits every state


def test_two_group_fsm_structure():
    actors = [
        _actor(actor_id=1, x=-5, y=0, axis=0.0),
        _actor(actor_id=2, x=0, y=-5, axis=math.pi / 2),
    ]
    groups = t.group_actors(intersection=[0, 1], actors=actors, config=t.FsmConfig())
    fsm = t.build_fsm(groups=groups, actors=actors, config=t.FsmConfig())
    assert len(fsm) == 6
    _assert_loops(fsm)
    for state in fsm:
        assert isinstance(state["duration"], str)
        non_red = [v for v in state["actor_states"].values() if v != t.RED]
        # At most one group is non-red at any time; never green and yellow simultaneously.
        assert set(non_red) in ({}, set(), {t.GREEN}, {t.YELLOW})


def test_single_group_fsm_structure():
    actors = [_actor(actor_id=1, x=0, y=0, axis=0.0)]
    groups = t.group_actors(intersection=[0], actors=actors, config=t.FsmConfig())
    fsm = t.build_fsm(groups=groups, actors=actors, config=t.FsmConfig())
    assert len(fsm) == 3
    _assert_loops(fsm)
    assert [s["actor_states"]["1"] for s in fsm] == [t.GREEN, t.YELLOW, t.RED]


def test_green_scales_with_group_size():
    config = t.FsmConfig()
    small = t.build_fsm(
        groups=[[0]],
        actors=[_actor(actor_id=1, x=0, y=0, axis=0.0, lane_count=1)],
        config=config,
    )
    big = t.build_fsm(
        groups=[[0]],
        actors=[_actor(actor_id=1, x=0, y=0, axis=0.0, lane_count=4)],
        config=config,
    )
    assert int(big[0]["duration"]) > int(small[0]["duration"])
    assert config.green_min <= int(small[0]["duration"]) <= config.green_max


# --------------------------------------------------------------------------------------
# Topology refinement logic (with a duck-typed stub; no lanelet2/.osm required)
# --------------------------------------------------------------------------------------


def test_topology_splits_disconnected_actors():
    actors = [
        _actor(actor_id=10, x=0, y=0, axis=0.0),
        _actor(actor_id=11, x=5, y=0, axis=math.pi / 2),
        _actor(actor_id=12, x=10, y=0, axis=0.0),
    ]

    # Map each actor to a sentinel lanelet, and declare only 10<->11 share a junction.
    sentinels = {i: object() for i in range(len(actors))}
    linked = {(0, 1)}

    class Topo:
        def lanelet_for(self, *, center):
            for i, a in enumerate(actors):
                if np.allclose(a.center, center):
                    return sentinels[i]
            return None

        def shares_junction(self, *, a, b, max_hops):
            ia = next(i for i, s in sentinels.items() if s is a)
            ib = next(i for i, s in sentinels.items() if s is b)
            return (min(ia, ib), max(ia, ib)) in linked

    components = t.refine_intersection_with_topology(
        intersection=[0, 1, 2], actors=actors, topology=Topo(), config=t.FsmConfig()
    )
    sets = sorted(sorted(actors[i].actor_id for i in c) for c in components)
    assert sets == [[10, 11], [12]]


# --------------------------------------------------------------------------------------
# Topology integration (requires lanelet2 + the Town15 .osm)
# --------------------------------------------------------------------------------------

TOWN15_DIR = os.path.dirname(TOWN15_STOPLINES)


@pytest.mark.skipif(not os.path.exists(TOWN15_STOPLINES), reason="Town15 fixture absent")
def test_topology_does_not_break_high_certainty_town15():
    pytest.importorskip("lanelet2")
    config = t.FsmConfig()
    actors = t.load_actors(stoplines_path=TOWN15_STOPLINES)
    metadata = t.load_metadata(map_dir=TOWN15_DIR)
    topo = t._load_topology(map_dir=TOWN15_DIR, metadata=metadata)
    assert topo is not None
    fsms = t.generate_fsms(actors=actors, config=config, topology=topo)
    assert len(fsms) == 8  # topology must not fragment the clean Town15 intersections
