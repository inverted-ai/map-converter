"""Tests for the standalone traffic-light FSM generator (``traffic_light_fsm.py``)."""

import json
import math
import os

import numpy as np
import pytest

import conversion_heuristics as ch
import conversion_report
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
    config = t.FsmConfig()
    assert config.cluster_eps == 50
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


# --------------------------------------------------------------------------------------
# Certainty and the review file
# --------------------------------------------------------------------------------------


def _four_way(*, first_id: int, x: float) -> list[t.Actor]:
    return [
        _actor(actor_id=first_id, x=x - 5, y=0, axis=0.0),
        _actor(actor_id=first_id + 1, x=x + 5, y=0, axis=0.0),
        _actor(actor_id=first_id + 2, x=x, y=-5, axis=math.pi / 2),
        _actor(actor_id=first_id + 3, x=x, y=5, axis=math.pi / 2),
    ]


def test_clean_four_way_is_certain_and_not_flagged():
    actors = _four_way(first_id=1, x=0)
    [result] = t.resolve_intersection_groups(actors=actors, config=t.FsmConfig(use_xodr_junctions=False))
    assert result.certainty == 1.0
    assert not result.topology_refined
    assert t.flagged_intersections(resolved=[result], config=t.FsmConfig()) == []


def test_intersection_near_another_is_flagged_for_spacing():
    # Centres 62 m apart (nearest lights 52 m, beyond eps 50 m): separate, spatial part 0.62.
    actors = _four_way(first_id=1, x=0) + _four_way(first_id=11, x=62)
    resolved = t.resolve_intersection_groups(actors=actors, config=t.FsmConfig(use_xodr_junctions=False))
    assert [round(r.spatial, 2) for r in resolved] == [0.62, 0.62]
    assert {r.reason for r in resolved} == {"spatial"}
    assert t.flagged_intersections(resolved=resolved, config=t.FsmConfig()) == [0, 1]


def test_ambiguous_axes_are_flagged_for_the_axis_split():
    # The least clear split four lights can give under the default split threshold scores ~0.83.
    actors = [
        _actor(actor_id=1, x=-5, y=0, axis=1.3),
        _actor(actor_id=2, x=5, y=0, axis=1.8),
        _actor(actor_id=3, x=0, y=-5, axis=1.9),
        _actor(actor_id=4, x=0, y=5, axis=2.4),
    ]
    config = t.FsmConfig(use_xodr_junctions=False, review_threshold=0.9)
    [result] = t.resolve_intersection_groups(actors=actors, config=config)
    assert len(result.groups) == 2
    assert round(result.axis, 2) == 0.83
    assert result.reason == "axis"
    assert t.flagged_intersections(resolved=[result], config=config) == [0]


def test_declared_phase_groups_score_one():
    actors = _four_way(first_id=1, x=0) + _four_way(first_id=11, x=62)
    groups = _signal_groups(
        membership={str(a.actor_id): ("A" if a.actor_id < 10 else "B") for a in actors},
        controllers=[
            {"id": "a1", "signal_ids": ["1", "2"]},
            {"id": "a2", "signal_ids": ["3", "4"]},
            {"id": "b1", "signal_ids": ["11", "12"]},
            {"id": "b2", "signal_ids": ["13", "14"]},
        ],
    )
    resolved = t.resolve_intersection_groups(actors=actors, config=t.FsmConfig(use_xodr_groups=True), signal_groups=groups)
    assert [(r.certainty, r.reason) for r in resolved] == [(1.0, "declared"), (1.0, "declared")]


def test_topology_refined_intersections_are_tagged_and_rescored():
    actors = [
        _actor(actor_id=10, x=0, y=0, axis=0.0),
        _actor(actor_id=11, x=5, y=0, axis=0.3),
        _actor(actor_id=12, x=10, y=0, axis=0.6),
    ]
    sentinels = [object() for _ in actors]

    class Topo:
        def lanelet_for(self, *, center):
            return next(s for a, s in zip(actors, sentinels) if np.allclose(a.center, center))

        def shares_junction(self, *, a, b, max_hops):
            return {sentinels.index(a), sentinels.index(b)} == {0, 1}

    resolved = t.resolve_intersection_groups(
        actors=actors, config=t.FsmConfig(use_xodr_junctions=False, certainty_threshold=1.01), topology=Topo()
    )
    assert len(resolved) == 2
    assert all(r.topology_refined for r in resolved)
    assert all(r.spatial < 1.0 for r in resolved)  # the split parts sit metres apart


def test_review_file_lists_certainty_and_flagged(tmp_path):
    actors = _four_way(first_id=1, x=0) + _four_way(first_id=11, x=62) + _four_way(first_id=21, x=1000)
    config = t.FsmConfig(use_xodr_junctions=False)
    resolved = t.resolve_intersection_groups(actors=actors, config=config)
    path = t.write_review(
        map_dir=str(tmp_path),
        location="m",
        resolved=resolved,
        actors=actors,
        config=config,
        signal_groups_path=None,
        signal_groups=None,
    )
    with open(path) as f:
        review = json.load(f)
    assert review["review_threshold"] == 0.7
    assert [i["certainty"] for i in review["intersections"]] == [0.62, 0.62, 1.0]
    assert [f["lights"] for f in review["flagged"]] == [[1, 2, 3, 4], [11, 12, 13, 14]]
    assert review["flagged"][0]["reason"] == "spatial"


# --------------------------------------------------------------------------------------
# BasicConversion FSM (H11-H13 off) and the FSM heuristics' detections
# --------------------------------------------------------------------------------------


def _basic(*, use_xodr_junctions: bool = True) -> t.FsmConfig:
    return t.FsmConfig(
        heuristics=ch.applied_heuristics(autofix=ch.AutoFixConfig(enabled=False)), use_xodr_junctions=use_xodr_junctions
    )


def _junction_and_strays() -> tuple[list[t.Actor], dict]:
    """Junction J (lights 1-3, controllers {1} then {2, 3}), light 4 near it, lights 5 and 6 near each other far away."""
    actors = [
        _actor(actor_id=1, x=-5, y=0, axis=0.0),
        _actor(actor_id=2, x=0, y=-5, axis=math.pi / 2),
        _actor(actor_id=3, x=0, y=5, axis=math.pi / 2),
        _actor(actor_id=4, x=20, y=0, axis=0.0),
        _actor(actor_id=5, x=500, y=0, axis=0.0),
        _actor(actor_id=6, x=510, y=0, axis=math.pi / 2),
    ]
    groups = _signal_groups(
        membership={"1": "J", "2": "J", "3": "J"},
        controllers=[
            {"id": "b", "signal_ids": ["2", "3"], "sequence": 1},
            {"id": "a", "signal_ids": ["1"], "sequence": 0},
        ],
    )
    return actors, groups


def _ids(*, resolved: list[t.IntersectionResult], actors: list[t.Actor]) -> list[list[list[int]]]:
    """Per intersection, its phase groups as actor ids, in phase order."""
    return [[[actors[i].actor_id for i in g] for g in r.groups] for r in resolved]


def test_basic_fsm_uses_declared_junctions_controllers_and_sequence():
    actors, groups = _junction_and_strays()
    resolved = t.resolve_intersection_groups(actors=actors, config=_basic(), signal_groups=groups)
    assert _ids(resolved=resolved, actors=actors) == [[[1], [2, 3]], [[4]], [[5]], [[6]]]
    assert resolved[0].level == "sequence"


def test_default_fsm_clusters_by_distance_and_splits_by_axis():
    actors, groups = _junction_and_strays()
    resolved = t.resolve_intersection_groups(actors=actors, config=t.FsmConfig(), signal_groups=groups)
    assert _resolved_ids(resolved=resolved, actors=actors) == [[[1, 4], [2, 3]], [[5], [6]]]


def test_basic_fsm_gives_each_light_no_controller_names_its_own_phase():
    actors = _four_way(first_id=1, x=0)
    groups = _signal_groups(membership={str(a.actor_id): "J" for a in actors}, controllers=[{"id": "a", "signal_ids": ["1", "2"]}])
    resolved = t.resolve_intersection_groups(actors=actors, config=_basic(), signal_groups=groups)
    assert _ids(resolved=resolved, actors=actors) == [[[1, 2], [3], [4]]]
    no_controller = t.resolve_intersection_groups(actors=actors, config=_basic(), signal_groups=_signal_groups(membership={"1": "J", "2": "J", "3": "J", "4": "J"}))
    assert _ids(resolved=no_controller, actors=actors) == [[[1], [2], [3], [4]]]
    assert no_controller[0].level == "junctions"


def test_basic_fsm_without_a_sidecar_makes_every_light_its_own_intersection():
    actors = _four_way(first_id=1, x=0)
    resolved = t.resolve_intersection_groups(actors=actors, config=_basic())
    assert _ids(resolved=resolved, actors=actors) == [[[1]], [[2]], [[3]], [[4]]]


def _fsm_findings(*, actors, config, signal_groups=None, topology=None) -> list[dict]:
    report = conversion_report.attach()
    try:
        t.resolve_intersection_groups(actors=actors, config=config, signal_groups=signal_groups, topology=topology)
    finally:
        conversion_report.detach(handler=report)
    return [r for r in report.records if r["kind"] == "heuristic"]


@pytest.mark.parametrize("basic", [False, True])
def test_fsm_heuristics_record_where_they_change_the_grouping(basic):
    actors, groups = _junction_and_strays()
    config = _basic() if basic else t.FsmConfig()
    findings = _fsm_findings(actors=actors, config=config, signal_groups=groups)
    by_id = {(r["element"], r["id"]): r for r in findings}
    # In Basic, junction J's axis split and its declared phases hold the same lights (only their order differs).
    expected = {("H11", "lights 1,2,3,4"), ("H11", "lights 5,6")} | (set() if basic else {("H12", "lights 1,2,3,4")})
    assert set(by_id) == expected
    assert all(r["applied"] is not basic for r in findings)
    assert by_id[("H11", "lights 5,6")]["location"] == pytest.approx([505.0, 0.0])


def test_topology_refinement_off_keeps_the_intersection_and_records_the_split():
    actors = [
        _actor(actor_id=10, x=0, y=0, axis=0.0),
        _actor(actor_id=11, x=5, y=0, axis=0.3),
        _actor(actor_id=12, x=10, y=0, axis=0.6),
    ]
    sentinels = [object() for _ in actors]

    class Topo:
        def lanelet_for(self, *, center):
            return next(s for a, s in zip(actors, sentinels) if np.allclose(a.center, center))

        def shares_junction(self, *, a, b, max_hops):
            return {sentinels.index(a), sentinels.index(b)} == {0, 1}

    heuristics = ch.HeuristicsConfig()
    heuristics.fsm_topology_refinement.enabled = False
    config = t.FsmConfig(use_xodr_junctions=False, certainty_threshold=1.01, heuristics=heuristics)
    resolved = t.resolve_intersection_groups(actors=actors, config=config, topology=Topo())
    assert len(resolved) == 1 and not resolved[0].topology_refined
    findings = _fsm_findings(actors=actors, config=config, topology=Topo())
    assert [(r["element"], r["applied"]) for r in findings if r["element"] == "H13"] == [("H13", False)]


def test_fsm_follows_the_autofix_settings_the_map_was_converted_with(tmp_path):
    (tmp_path / "m_provenance.json").write_text(json.dumps({"config": {"autofix": {"enabled": False}}}))
    assert t.autofix_from_provenance(map_dir=str(tmp_path)) == ch.AutoFixConfig(enabled=False)
    assert t.autofix_from_provenance(map_dir=str(tmp_path / "missing")) == ch.AutoFixConfig()



def test_phase_greening_crossing_approaches_is_flagged_whatever_the_certainty():
    actors = _four_way(first_id=1, x=0)
    groups = _signal_groups(
        membership={str(a.actor_id): "J" for a in actors},
        controllers=[{"id": "a", "signal_ids": ["1", "3"]}, {"id": "b", "signal_ids": ["2", "4"]}],
    )
    [result] = t.resolve_intersection_groups(actors=actors, config=t.FsmConfig(use_xodr_groups=True), signal_groups=groups)
    assert result.certainty == 1.0  # declared junction and phases
    assert result.conflicting_phases == [0, 1] and result.reason == "conflict"
    assert t.flagged_intersections(resolved=[result], config=t.FsmConfig()) == [0]


def test_axis_split_phases_have_no_conflict():
    [result] = t.resolve_intersection_groups(actors=_four_way(first_id=1, x=0), config=t.FsmConfig(use_xodr_junctions=False))
    assert result.conflicting_phases == [] and result.reason == "clear"
