"""Tests for the standalone traffic-light FSM generator (``traffic_light_fsm.py``)."""

import math
import os

import numpy as np
import pytest

import traffic_light_fsm as t

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS_DIR = os.path.abspath(os.environ.get("IAI_MAPS_DIR", os.path.join(REPO_ROOT, "maps")))
TOWN15_STOPLINES = os.path.join(MAPS_DIR, "Carla_UE4_old", "Town15", "Town15_stoplines.json")


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


# --------------------------------------------------------------------------------------
# XODR structure levels (junctions / controller groups / sequence)
# --------------------------------------------------------------------------------------

TOWN15_XODR = os.path.join(MAPS_DIR, "Carla_UE4_old", "Town15", "Town15.xodr")
TOWN15_EXPECTED = sorted(
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


def _signal_groups(*, membership: dict[str, str], controllers: list[dict] = ()) -> dict:
    return {
        "membership": {sid: {"junction": j, "source": "controller"} for sid, j in membership.items()},
        "controllers": list(controllers),
    }


def _resolved_ids(*, resolved: list[t.IntersectionResult], actors: list[t.Actor]) -> list[list[list[int]]]:
    out = [sorted(sorted(actors[i].actor_id for i in g) for g in r.groups) for r in resolved]
    return sorted(out, key=lambda x: min(min(s) for s in x))


@pytest.fixture(scope="module")
def town15_signal_groups() -> dict:
    """Town15's signal groups, extracted in memory (nothing is written into maps/)."""
    if not os.path.exists(TOWN15_XODR):
        pytest.skip("Town15 fixture absent")
    import opendrive_signal_groups

    return opendrive_signal_groups.extract_signal_groups(xodr_path=TOWN15_XODR)


def test_levels_off_is_todays_distance_clustering():
    actors = [
        _actor(actor_id=1, x=0, y=0, axis=0.0),
        _actor(actor_id=2, x=30, y=0, axis=0.0),
        _actor(actor_id=3, x=300, y=0, axis=math.pi / 2),
    ]
    off = t.FsmConfig(use_xodr_junctions=False)
    groups = _signal_groups(membership={"1": "A", "2": "B"})
    assert t.cluster_intersections(actors=actors, config=off) == [[0, 1], [2]]
    resolved = t.resolve_intersection_groups(actors=actors, config=off, signal_groups=groups)
    assert [r.actors for r in resolved] == [[0, 1], [2]]
    assert {r.level for r in resolved} == {"off"}


def test_town15_level_a_matches_manual_groupings(town15_signal_groups):
    actors = t.load_actors(stoplines_path=TOWN15_STOPLINES)
    resolved = t.resolve_intersection_groups(actors=actors, config=t.FsmConfig(), signal_groups=town15_signal_groups)
    assert _resolved_ids(resolved=resolved, actors=actors) == TOWN15_EXPECTED
    assert {r.level for r in resolved} == {"junctions"}


def test_unassigned_light_joins_junction_within_eps_only():
    actors = [
        _actor(actor_id=1, x=0, y=0, axis=0.0),
        _actor(actor_id=2, x=10, y=0, axis=0.0),
        _actor(actor_id=3, x=40, y=0, axis=0.0),  # unassigned, 30 m from the junction
        _actor(actor_id=4, x=500, y=0, axis=0.0),  # unassigned, far away
    ]
    membership = {"1": "J1", "2": "J1"}
    assert t.cluster_intersections(actors=actors, config=t.FsmConfig(), membership=membership) == [[0, 1, 2], [3]]


def test_unassigned_chain_never_bridges_two_junctions():
    actors = [
        _actor(actor_id=1, x=0, y=0, axis=0.0),
        _actor(actor_id=2, x=40, y=0, axis=0.0),  # unassigned link
        _actor(actor_id=3, x=80, y=0, axis=0.0),
    ]
    membership = {"1": "J1", "3": "J2"}
    clusters = t.cluster_intersections(actors=actors, config=t.FsmConfig(), membership=membership)
    assert len(clusters) == 2
    assert [0] in clusters or [0, 1] in clusters
    assert all(not ({0, 2} <= set(c)) for c in clusters)


def test_two_close_junctions_are_never_merged(caplog):
    actors = [_actor(actor_id=1, x=0, y=0, axis=0.0), _actor(actor_id=2, x=5, y=0, axis=0.0)]
    clusters = t.cluster_intersections(actors=actors, config=t.FsmConfig(), membership={"1": "J1", "2": "J2"})
    assert clusters == [[0], [1]]
    assert "Junctions J1 and J2 have lights 5.0 m apart" in caplog.text


def test_town15_level_b_gives_three_phases_and_singleton_warning(town15_signal_groups, caplog):
    actors = t.load_actors(stoplines_path=TOWN15_STOPLINES)
    config = t.FsmConfig(use_xodr_groups=True)
    resolved = t.resolve_intersection_groups(actors=actors, config=config, signal_groups=town15_signal_groups)
    junction_322 = next(r for r in resolved if r.junction == "322")
    assert junction_322.level == "groups"
    assert len(junction_322.groups) == 3
    assert "every XODR controller holds a single light" in caplog.text


def test_level_c_orders_phases_by_sequence():
    actors = [
        _actor(actor_id=1, x=0, y=0, axis=0.0),
        _actor(actor_id=2, x=5, y=5, axis=math.pi / 2),
        _actor(actor_id=3, x=5, y=-5, axis=math.pi / 2),
    ]
    groups = _signal_groups(
        membership={"1": "J", "2": "J", "3": "J"},
        controllers=[
            {"id": "a", "signal_ids": ["2", "3"], "sequence": 1},
            {"id": "b", "signal_ids": ["1"], "sequence": 0},
        ],
    )
    b = t.resolve_intersection_groups(actors=actors, config=t.FsmConfig(use_xodr_groups=True), signal_groups=groups)
    c = t.resolve_intersection_groups(actors=actors, config=t.FsmConfig(use_xodr_sequence=True), signal_groups=groups)
    assert [[actors[i].actor_id for i in g] for g in b[0].groups] == [[2, 3], [1]]  # larger group first
    assert [[actors[i].actor_id for i in g] for g in c[0].groups] == [[1], [2, 3]]  # by sequence
    assert c[0].level == "sequence"


def test_sequence_alone_forces_groups_and_junctions():
    levels = t.effective_xodr_levels(config=t.FsmConfig(use_xodr_junctions=False, use_xodr_sequence=True))
    assert (levels.use_xodr_junctions, levels.use_xodr_groups, levels.use_xodr_sequence) == (True, True, True)


def test_no_light_is_dropped_at_any_level(town15_signal_groups):
    actors = t.load_actors(stoplines_path=TOWN15_STOPLINES)
    for config in (
        t.FsmConfig(use_xodr_junctions=False),
        t.FsmConfig(),
        t.FsmConfig(use_xodr_groups=True),
        t.FsmConfig(use_xodr_sequence=True),
    ):
        resolved = t.resolve_intersection_groups(actors=actors, config=config, signal_groups=town15_signal_groups)
        assert sorted(i for r in resolved for g in r.groups for i in g) == list(range(len(actors)))
