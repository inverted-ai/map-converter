"""Self-validation: threshold checks on a conversion's outputs and the map verifier run."""

import json
from typing import Optional

import numpy as np
import pytest
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork

import conversion_common
import map_validation as mv
import traffic_light_fsm as tlf
from tests.iai import xodr_builder as xb

CLEAN_METRICS = {
    "lanelet_count": 100,
    "routing_edge_pct": 100.0,
    "seam_handshake_pct": 100.0,
    "crossed_geometry": {"reversed_bounds": 0, "invalid_polygons": 0},
    "lanelets_without_subtype": 0,
    "point_extent_m": 500.0,
    "origin": [49.0, 8.0],
}
CLEAN_CONTROLS = mv.ControlCoverage(lights=2, lights_without_stopline=[], stop_yield_signs=1, signs_without_stopline=[])


def _failed(
    *,
    metrics: dict,
    profile: str = "perfect",
    metric_errors: Optional[dict] = None,
    controls: mv.ControlCoverage = CLEAN_CONTROLS,
) -> list[str]:
    checks = mv.evaluate(
        metrics=metrics,
        metric_errors=metric_errors or {},
        controls=controls,
        config=mv.PROFILES[profile],
    )
    return [c.name for c in checks if not c.passed]


def _crossed(*, count: int, lanelets: int) -> dict:
    return {**CLEAN_METRICS, "lanelet_count": lanelets, "crossed_geometry": {"reversed_bounds": count, "invalid_polygons": 0}}


@pytest.mark.parametrize("profile", ["perfect", "strict"])
def test_clean_metrics_pass_every_check(profile):
    assert _failed(metrics=CLEAN_METRICS, profile=profile) == []


def test_perfect_fails_a_routing_gap_strict_accepts():
    metrics = {**CLEAN_METRICS, "routing_edge_pct": 96.0}
    assert (_failed(metrics=metrics), _failed(metrics=metrics, profile="strict")) == (["routing_edge_pct"], [])


@pytest.mark.parametrize(
    "count, lanelets, perfect, strict",
    [
        (1, 100, ["crossed_lanelets"], []),
        (2, 10, ["crossed_lanelets"], []),  # 20%, but strict needs at least 3 lanelets
        (3, 100, ["crossed_lanelets"], ["crossed_lanelets"]),  # 3%
        (3, 200, ["crossed_lanelets"], []),  # 1.5%
    ],
)
def test_crossed_lanelets(count, lanelets, perfect, strict):
    metrics = _crossed(count=count, lanelets=lanelets)
    assert (_failed(metrics=metrics), _failed(metrics=metrics, profile="strict")) == (perfect, strict)


@pytest.mark.parametrize("profile", ["perfect", "strict"])
def test_each_light_and_sign_needs_a_stopline(profile):
    controls = mv.ControlCoverage(
        lights=2, lights_without_stopline=["13"], stop_yield_signs=3, signs_without_stopline=["signal 5"]
    )
    failed = _failed(metrics=CLEAN_METRICS, profile=profile, controls=controls)
    assert failed == ["traffic_light_stoplines", "stop_yield_stoplines"]


@pytest.mark.parametrize("extent, failed", [(50_000.0, []), (150_000.0, ["point_extent_m"])])
@pytest.mark.parametrize("profile", ["perfect", "strict"])
def test_point_extent_limit_is_100_km(profile, extent, failed):
    assert _failed(metrics={**CLEAN_METRICS, "point_extent_m": extent}, profile=profile) == failed


def test_metric_that_raised_fails_its_check():
    metrics = {k: v for k, v in CLEAN_METRICS.items() if k != "routing_edge_pct"}
    assert _failed(metrics=metrics, metric_errors={"routing_edge_pct": "RuntimeError: boom"}) == ["routing_edge_pct"]


def test_metric_with_nothing_to_measure_passes():
    assert _failed(metrics={**CLEAN_METRICS, "seam_handshake_pct": None}) == []


def test_outcome_message_points_to_strict_when_only_perfect_fails():
    message = mv.outcome_message(profile="perfect", failed={"perfect": ["routing_edge_pct"], "strict": []})
    assert "routing_edge_pct" in message
    assert "autocritic.preset=strict" in message
    assert "review" in message


def test_outcome_message_names_strict_failures_too():
    failed = {"perfect": ["routing_edge_pct", "crossed_lanelets"], "strict": ["routing_edge_pct"]}
    message = mv.outcome_message(profile="perfect", failed=failed)
    assert "strict profile fails too (routing_edge_pct)" in message
    assert "autocritic.preset=strict" not in message


def test_outcome_message_is_none_on_a_pass():
    assert mv.outcome_message(profile="perfect", failed={"perfect": [], "strict": []}) is None


def test_verifier_past_its_cap_records_timeout():
    left = np.array([[0.0, 3.0], [10.0, 3.0]])
    right = np.array([[0.0, 0.0], [10.0, 0.0]])
    network = LaneletNetwork()
    network.add_lanelet(Lanelet(left_vertices=left, center_vertices=(left + right) / 2, right_vertices=right, lanelet_id=1))
    result = mv.run_map_verifier(lanelet_network=network, timeout_s=0.001, partition_above=200)
    assert (result["status"], result["partitioned"], result["lanelets"]) == ("timeout", False, 1)


def _convert(*, tmp_path, roads, junctions=(), controllers=(), **config: str) -> tuple[dict, str]:
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers))
    result = xb.run_convert_map(map_dir=str(tmp_path), **config)
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "case_conversion_report.json") as f:
        return json.load(f)["validation"], result.stderr


def _failed_names(*, verdict: dict) -> set[str]:
    return {c["name"] for c in verdict["checks"] if not c["passed"]}


def test_conversion_report_holds_the_verdict(tmp_path):
    # One two-way road: its two lanelets have no routing edges, and its stop sign has no stopline.
    road = xb.straight_road(id="1", length=30.0, signals=[xb.SignalSpec(id="5", s=10.0, dynamic="no", type="206")])
    verdict, stderr = _convert(tmp_path=tmp_path, roads=[road])
    assert _failed_names(verdict=verdict) == {"routing_edge_pct", "stop_yield_stoplines"}
    assert (verdict["status"], verdict["profile"]) == ("fail", "perfect")
    assert verdict["profiles"] == {
        "perfect": {"status": "fail", "failed": ["routing_edge_pct", "stop_yield_stoplines"]},
        "strict": {"status": "fail", "failed": ["routing_edge_pct", "stop_yield_stoplines"]},
    }
    assert verdict["controls"] == {
        "lights": 0,
        "lights_without_stopline": [],
        "stop_yield_signs": 1,
        "signs_without_stopline": ["signal 5"],
    }
    assert verdict["verifier"]["status"] == "skipped"
    assert "Validation: routing_edge_pct = 0.0" in stderr
    assert "strict profile fails too" in stderr


def test_verifier_runs_on_request(tmp_path):
    verdict, _ = _convert(tmp_path=tmp_path, roads=[xb.straight_road(id="1", length=30.0)], verifier_timeout_s="300")
    assert (verdict["verifier"]["status"], verdict["verifier"]["partitioned"]) == ("ok", False)


def test_strict_profile_is_chosen_on_request(tmp_path):
    verdict, _ = _convert(tmp_path=tmp_path, roads=[xb.straight_road(id="1", length=30.0)], **{"autocritic.preset": "strict"})
    assert verdict["profile"] == "strict"
    assert verdict["status"] == verdict["profiles"]["strict"]["status"]


def test_stencil_near_the_sign_is_its_stopline(tmp_path):
    road = xb.straight_road(
        id="1",
        length=30.0,
        signals=[xb.SignalSpec(id="5", s=25.0, dynamic="no", type="206")],
        objects=[xb.ObjectSpec(id="6", s=24.0)],
    )
    verdict, _ = _convert(tmp_path=tmp_path, roads=[road])
    assert (verdict["controls"]["stop_yield_signs"], verdict["controls"]["signs_without_stopline"]) == (2, [])
    assert "stop_yield_stoplines" not in _failed_names(verdict=verdict)


def test_stencil_of_the_other_kind_is_not_the_signs_stopline(tmp_path):
    road = xb.straight_road(
        id="1",
        length=30.0,
        signals=[xb.SignalSpec(id="5", s=25.0, dynamic="no", type="206")],
        objects=[xb.ObjectSpec(id="6", s=24.0, name="Stencil_GiveWay")],
    )
    verdict, _ = _convert(tmp_path=tmp_path, roads=[road])
    assert verdict["controls"]["signs_without_stopline"] == ["signal 5"]


def test_stencil_outside_the_vertical_limits_is_not_counted(tmp_path):
    road = xb.straight_road(id="1", length=30.0, objects=[xb.ObjectSpec(id="6", s=24.0)])
    verdict, _ = _convert(tmp_path=tmp_path, roads=[road], trim_vertical_limits="[5.0,10.0]")
    assert (verdict["controls"]["stop_yield_signs"], verdict["controls"]["signs_without_stopline"]) == (0, [])


def test_light_without_a_stopline_fails_its_check(tmp_path):
    roads, junctions, controllers = xb.junction_map()
    roads[1].signals.append(xb.SignalSpec(id="13", s=10.0, t=0.0, orientation="none", validity=(-3, -3)))
    verdict, _ = _convert(tmp_path=tmp_path, roads=roads, junctions=junctions, controllers=controllers)
    assert (verdict["controls"]["lights"], verdict["controls"]["lights_without_stopline"]) == (3, ["13"])
    assert "traffic_light_stoplines" in _failed_names(verdict=verdict)


def test_every_light_with_a_stopline_passes_its_check(tmp_path):
    roads, junctions, controllers = xb.junction_map()
    verdict, _ = _convert(tmp_path=tmp_path, roads=roads, junctions=junctions, controllers=controllers)
    assert (verdict["controls"]["lights"], verdict["controls"]["lights_without_stopline"]) == (2, [])
    assert "traffic_light_stoplines" not in _failed_names(verdict=verdict)


def test_autocritic_off_records_no_verdict(tmp_path):
    verdict, _ = _convert(tmp_path=tmp_path, roads=[xb.straight_road(id="1", length=30.0)], **{"autocritic.enabled": "false"})
    assert verdict is None


@pytest.mark.parametrize("field, value", [("verifier_timeout_s", "-1"), ("autocritic.preset", "lenient")])
def test_bad_validation_settings_are_refused(tmp_path, field, value):
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)]))
    result = xb.run_convert_map(map_dir=str(tmp_path), **{field: value})
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
    assert field in result.stderr


def test_stopline_fsm_join_reports_lights_missing_from_the_fsms(tmp_path):
    actors = [tlf.Actor(actor_id=i, center=np.zeros(2), axis=0.0, lane_count=1) for i in (1, 2, 3)]
    controller = tmp_path / "m_traffic_light_controller.json"
    fsms = [[{"actor_states": {"1": "green", "2": "red"}}], [{"actor_states": {"2": "green", "7": "red"}}]]
    controller.write_text(json.dumps(fsms))
    join = tlf.stopline_fsm_join(actors=actors, controller_path=str(controller))
    assert join == {
        "stopline_lights": 3,
        "fsm_lights": 3,
        "missing_from_fsm": ["3"],
        "not_in_stoplines": ["7"],
        "in_several_intersections": ["2"],
        "ok": False,
    }
