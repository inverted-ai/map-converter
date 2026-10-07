"""Self-validation: threshold checks on a conversion's outputs and the map verifier run."""

import json
from typing import Optional

import numpy as np
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork

import conversion_common
import map_validation as mv
import traffic_light_fsm as tlf
from tests.iai import xodr_builder as xb

CLEAN_METRICS = {
    "lanelet_count": 10,
    "routing_edge_pct": 100.0,
    "seam_handshake_pct": 100.0,
    "crossed_geometry": {"reversed_bounds": 0, "invalid_polygons": 0},
    "sign_ways_vs_stoplines": {
        "traffic_sign_ways": 4,
        "stop_yield_sign_ways": 2,
        "traffic_light_ways": 2,
        "sign_stoplines": 2,
        "light_stoplines": 4,
    },
    "lanelets_without_subtype": 0,
    "point_extent_m": 500.0,
    "origin": [49.0, 8.0],
}


def _failed(*, metrics: dict, metric_errors: Optional[dict] = None) -> list[str]:
    checks = mv.evaluate(metrics=metrics, metric_errors=metric_errors or {}, config=mv.ValidationConfig())
    return [c.name for c in checks if not c.passed]


def test_clean_metrics_pass_every_check():
    assert _failed(metrics=CLEAN_METRICS) == []


def test_isolated_lanelets_fail_routing():
    assert _failed(metrics={**CLEAN_METRICS, "routing_edge_pct": 0.0}) == ["routing_edge_pct"]


def test_stop_sign_ways_without_stoplines_fail():
    signs = {**CLEAN_METRICS["sign_ways_vs_stoplines"], "sign_stoplines": 0}
    assert _failed(metrics={**CLEAN_METRICS, "sign_ways_vs_stoplines": signs}) == ["stop_yield_stoplines"]


def test_metric_that_raised_fails_its_check():
    metrics = {k: v for k, v in CLEAN_METRICS.items() if k != "routing_edge_pct"}
    assert _failed(metrics=metrics, metric_errors={"routing_edge_pct": "RuntimeError: boom"}) == ["routing_edge_pct"]


def test_metric_with_nothing_to_measure_passes():
    assert _failed(metrics={**CLEAN_METRICS, "seam_handshake_pct": None}) == []


def test_verifier_past_its_cap_records_timeout():
    left = np.array([[0.0, 3.0], [10.0, 3.0]])
    right = np.array([[0.0, 0.0], [10.0, 0.0]])
    network = LaneletNetwork()
    network.add_lanelet(Lanelet(left_vertices=left, center_vertices=(left + right) / 2, right_vertices=right, lanelet_id=1))
    result = mv.run_map_verifier(lanelet_network=network, timeout_s=0.001, partition_above=200)
    assert (result["status"], result["partitioned"], result["lanelets"]) == ("timeout", False, 1)


def test_conversion_report_holds_the_verdict(tmp_path):
    # One two-way road: its two lanelets have no routing edges, and its stop sign has no stopline.
    road = xb.straight_road(id="1", length=30.0, signals=[xb.SignalSpec(id="5", s=10.0, dynamic="no", type="206")])
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=[road]))
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "case_conversion_report.json") as f:
        verdict = json.load(f)["validation"]
    failed = {c["name"] for c in verdict["checks"] if not c["passed"]}
    assert failed == {"routing_edge_pct", "stop_yield_stoplines"}
    assert (verdict["status"], verdict["ratified"]) == ("warn", False)
    assert verdict["verifier"]["status"] == "ok"
    assert verdict["verifier"]["partitioned"] is False
    assert "Validation: routing_edge_pct = 0.0" in result.stderr


def test_negative_verifier_timeout_is_refused(tmp_path):
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)]))
    result = xb.run_convert_map(map_dir=str(tmp_path), verifier_timeout_s="-1")
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
    assert "verifier_timeout_s" in result.stderr


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
