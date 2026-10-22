"""Self-validation: threshold checks on a conversion's outputs and the map verifier run."""

import json
import math
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
    "crossed_geometry": {"reversed_bounds": 0, "invalid_polygons": 0, "inverted_on_load": 0, "lanelets": 0},
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
    return {**CLEAN_METRICS, "lanelet_count": lanelets, "crossed_geometry": {"reversed_bounds": count, "invalid_polygons": 0, "inverted_on_load": 0, "lanelets": count}}


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
    # One two-way road, open at both ends (which routing allows); its stop sign has no stopline.
    road = xb.straight_road(id="1", length=30.0, signals=[xb.SignalSpec(id="5", s=10.0, dynamic="no", type="206")])
    verdict, stderr = _convert(tmp_path=tmp_path, roads=[road])
    assert _failed_names(verdict=verdict) == {"stop_yield_stoplines"}
    assert (verdict["status"], verdict["profile"]) == ("not_judged", "none")
    assert verdict["profiles"] == {
        "perfect": {"status": "fail", "failed": ["stop_yield_stoplines"]},
        "strict": {"status": "fail", "failed": ["stop_yield_stoplines"]},
    }
    assert verdict["controls"] == {
        "lights": 0,
        "lights_without_stopline": [],
        "stop_yield_signs": 1,
        "signs_without_stopline": ["signal 5"],
        "positions": {"signal 5": [10.0, -5.0]},
    }
    assert verdict["verifier"]["status"] == "skipped"
    by_name = {c["name"]: c for c in verdict["checks"]}
    assert [(f["id"], f["location"]) for f in by_name["stop_yield_stoplines"]["locations"]] == [("signal 5", [10.0, -5.0])]
    assert by_name["routing_edge_pct"]["value"] == 100.0


def test_verifier_checks_only_the_formulas_asked_for(tmp_path):
    verdict, _ = _convert(
        tmp_path=tmp_path,
        roads=[xb.straight_road(id="1", length=30.0)],
        verifier_timeout_s="300",
        verifier_formulas="[UNIQUE_ID,SAME_VERTICES_SIZE]",
    )
    assert verdict["verifier"]["status"] == "ok"
    assert verdict["verifier"]["formulas"] == ["UNIQUE_ID", "SAME_VERTICES_SIZE"]
    assert set(verdict["verifier"]["invalid_states"]) <= {"UNIQUE_ID", "SAME_VERTICES_SIZE"}


def test_verifier_runs_on_request(tmp_path):
    verdict, _ = _convert(tmp_path=tmp_path, roads=[xb.straight_road(id="1", length=30.0)], verifier_timeout_s="300")
    assert (verdict["verifier"]["status"], verdict["verifier"]["partitioned"]) == ("ok", False)


def test_strict_profile_is_chosen_on_request(tmp_path):
    road = xb.straight_road(id="1", length=30.0)
    config = {"autocritic.preset": "strict", "autocritic.thresholds.min_routing_edge_pct": "0"}
    verdict, _ = _convert(tmp_path=tmp_path, roads=[road], **config)
    assert (verdict["profile"], verdict["status"]) == ("strict", "pass")


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


MAP_OUTPUTS = ("case.osm", "case_stoplines.json", "case_mesh.json", "metadata.json", "visualization.png", "case_signal_groups.json")


UNLINKED_ROADS = [xb.straight_road(id="1", length=30.0), xb.straight_road(id="2", length=30.0, x=30.0)]


def _unlinked_roads(*, tmp_path) -> None:
    """Two two-way roads meeting end to start without a link: half their lanelets meet another without a routing
    edge (routing_edge_pct 50%) and the meeting ends share no points (seam_handshake_pct 0%); both fail every preset."""
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=UNLINKED_ROADS))


def test_failing_preset_withholds_the_outputs_and_exits_4(tmp_path):
    _unlinked_roads(tmp_path=tmp_path)
    result = xb.run_convert_map(map_dir=str(tmp_path), **{"autocritic.preset": "perfect"})
    assert result.returncode == conversion_common.REJECTED_EXIT_CODE, result.stderr
    assert "routing_edge_pct" in result.stderr.strip().splitlines()[-1]
    assert [name for name in MAP_OUTPUTS if (tmp_path / name).exists()] == []
    with open(tmp_path / "case_conversion_report.json") as f:
        report = json.load(f)
    assert (report["outcome"], report["validation"]["status"]) == ("rejected", "fail")
    assert "Validation: routing_edge_pct = 50.0" in result.stderr
    assert "strict profile fails too" in result.stderr
    assert (tmp_path / "case_provenance.json").exists()


def test_threshold_override_decides_with_the_preset(tmp_path):
    _unlinked_roads(tmp_path=tmp_path)
    config = {
        "autocritic.preset": "perfect",
        "autocritic.thresholds.min_routing_edge_pct": "0",
        "autocritic.thresholds.min_seam_handshake_pct": "0",
    }
    verdict, _ = _convert(tmp_path=tmp_path, roads=UNLINKED_ROADS, **config)
    assert (verdict["status"], verdict["profile"]) == ("pass", "perfect")
    assert verdict["thresholds"]["min_routing_edge_pct"] == 0.0
    assert verdict["profiles"]["perfect"]["status"] == "fail"  # the preset itself, without the override
    assert all((tmp_path / name).exists() for name in MAP_OUTPUTS)


def test_without_a_preset_nothing_is_judged_and_the_outputs_are_kept(tmp_path):
    _unlinked_roads(tmp_path=tmp_path)
    verdict, _ = _convert(tmp_path=tmp_path, roads=UNLINKED_ROADS)
    assert verdict["status"] == "not_judged"
    assert verdict["profiles"]["perfect"]["failed"] == ["routing_edge_pct", "seam_handshake_pct"]
    assert all((tmp_path / name).exists() for name in MAP_OUTPUTS)


@pytest.mark.parametrize(
    "field, value",
    [
        ("verifier_timeout_s", "-1"),
        ("autocritic.preset", "lenient"),
        ("autocritic.thresholds.no_such_limit", "1"),
        ("verifier_formulas", "[NO_SUCH_FORMULA]"),
    ],
)
def test_bad_validation_settings_are_refused(tmp_path, field, value):
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)]))
    result = xb.run_convert_map(map_dir=str(tmp_path), **{field: value})
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
    assert field.split(".")[-1] in result.stderr


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


def _check(*, verdict: dict, name: str) -> dict:
    return next(c for c in verdict["checks"] if c["name"] == name)


def test_signal_dropped_past_the_road_end_is_listed(tmp_path):
    sign = xb.SignalSpec(id="7", s=55.0, dynamic="no", type="206", validity=(-1, -1))
    verdict, _ = _convert(tmp_path=tmp_path, roads=[xb.straight_road(id="1", length=50.0, signals=[sign])])
    check = _check(verdict=verdict, name="dropped_signals")
    # Without a preset the checks are perfect's, for information: perfect allows no dropped signal.
    assert (check["value"], check["passed"], check["limit"]) == (1, False, 0)
    assert verdict["status"] == "not_judged"
    assert check["locations"][0]["id"] == "signal 7"


def test_linked_lanelets_farther_apart_than_the_seam_tolerance_are_listed(tmp_path):
    road1 = xb.straight_road(id="1", length=50.0, successor=xb.LinkSpec(element_type="road", element_id="2", contact_point="end"))
    road2 = xb.straight_road(
        id="2", length=50.0, x=101.5, hdg=math.pi, successor=xb.LinkSpec(element_type="road", element_id="1", contact_point="end")
    )
    for lane in road1.lane_sections[0].lanes + road2.lane_sections[0].lanes:
        lane.successor = -lane.id
    verdict, _ = _convert(tmp_path=tmp_path, roads=[road1, road2])
    check = _check(verdict=verdict, name="linked_gaps")
    assert check["value"] == 2
    assert all(place["location"][0] == pytest.approx(50.75, abs=0.01) for place in check["locations"])


def test_signal_far_from_its_stopline_is_listed_and_judged_on_request(tmp_path):
    light = xb.SignalSpec(id="21", s=40.0, t=-70.0, validity=(-1, -1))
    road = xb.straight_road(id="1", length=50.0, signals=[light])
    verdict, _ = _convert(tmp_path=tmp_path, roads=[road])
    check = _check(verdict=verdict, name="far_signals")
    assert [place["id"] for place in check["locations"]] == ["signal 21"]
    assert check["passed"]
    xb.write_xodr(map_dir=tmp_path / "judged", name="case", text=xb.build_xodr(roads=[road]))
    result = xb.run_convert_map(
        map_dir=str(tmp_path / "judged"),
        **{
            "autocritic.preset": "strict",
            "autocritic.thresholds.min_routing_edge_pct": "0",
            "autocritic.thresholds.max_far_signals": "0",
        },
    )
    assert result.returncode == conversion_common.REJECTED_EXIT_CODE
    assert "far_signals" in result.stderr.strip().splitlines()[-1]


def test_painted_stop_lines_are_listed_as_not_converted(tmp_path):
    road = xb.straight_road(id="1", length=50.0, objects=[xb.ObjectSpec(id="91", s=30.0, name="StopLine", type="-1", width=0.0, length=3.5)])
    verdict, _ = _convert(tmp_path=tmp_path, roads=[road])
    check = _check(verdict=verdict, name="not_converted")
    assert [(place["id"], place["detail"]) for place in check["locations"]] == [("object 91", "painted stop line")]
    assert check["limit"] is None


def test_a_light_code_marked_static_is_listed_without_failing_a_preset(tmp_path):
    static_light = xb.SignalSpec(id="7", s=20.0, dynamic="no", type="1000001")
    road = xb.straight_road(id="1", length=50.0, signals=[static_light])
    verdict, _ = _convert(tmp_path=tmp_path, roads=[road])
    check = _check(verdict=verdict, name="static_light_codes")
    assert [place["id"] for place in check["locations"]] == ["signal 7"]
    assert check["limit"] is None and check["passed"]
    assert "static_light_codes" not in verdict["profiles"]["perfect"]["failed"]



def test_a_lane_with_stoplines_at_two_places_is_listed_without_failing_a_preset(tmp_path):
    # No junction ahead and 40 m apart (beyond the 30 m approach distance): two stop positions on one lane.
    lights = [xb.SignalSpec(id=i, s=s, validity=(-1, -1)) for i, s in (("21", 5.0), ("22", 45.0))]
    verdict, _ = _convert(tmp_path=tmp_path, roads=[xb.straight_road(id="1", length=50.0, signals=lights)])
    check = _check(verdict=verdict, name="lanes_with_several_stops")
    assert check["value"] == 1 and check["limit"] is None and check["passed"]
    assert "2 stop positions" in check["locations"][0]["detail"]
    assert "lanes_with_several_stops" not in verdict["profiles"]["perfect"]["failed"]


def test_perfect_allows_no_linked_gap_and_no_dropped_signal():
    assert (mv.PROFILES["perfect"].max_linked_gaps, mv.PROFILES["perfect"].max_dropped_signals) == (0, 0)
    assert (mv.PROFILES["strict"].max_linked_gaps, mv.PROFILES["strict"].max_dropped_signals) == (None, None)
    assert mv.PROFILES["perfect"].max_far_signals is None
