"""The AutoCritic report: numbered AutoFix, would-have-fixed and metric items, as JSON and self-contained HTML."""

import json

import numpy as np
import pytest

import autocritic_report as ar
import conversion_common
from tests.iai import xodr_builder as xb


def _record(*, heuristic_id: str, element_id: str, applied: bool, location=(10.0, 0.0)) -> dict:
    return {
        "kind": "heuristic",
        "element": heuristic_id,
        "id": element_id,
        "value": "name",
        "action": "act",
        "message": f"{heuristic_id} {element_id}",
        "applied": applied,
        "location": list(location),
    }


VERDICT = {
    "status": "fail",
    "profile": "perfect",
    "checks": [
        {"name": "has_lanelets", "value": 2, "limit": 1, "passed": True, "detail": "", "locations": []},
        {
            "name": "routing_edge_pct",
            "value": 0.0,
            "limit": 100.0,
            "passed": False,
            "detail": "routing",
            "locations": [{"id": "lanelet 1", "location": [5.0, 1.0], "detail": ""}],
        },
        {"name": "not_converted", "value": 0, "limit": None, "passed": True, "detail": "", "locations": []},
    ],
}
RECORDS = [
    _record(heuristic_id="H4", element_id="21", applied=True),
    _record(heuristic_id="H4", element_id="22", applied=True, location=(30.0, 0.0)),
    _record(heuristic_id="H6", element_id="31", applied=False),
    {"kind": "unknown_value", "element": "lane.type", "id": "1", "value": "x", "action": "defaulted", "message": "m"},
]
POLYLINES = [np.array([[0.0, 0.0], [50.0, 0.0]]), np.array([[0.0, 3.5], [50.0, 3.5]])]


def test_items_are_numbered_per_section():
    report = ar.build(records=RECORDS, verdict=VERDICT, autofix_enabled=True, would_have_fixed=True)
    sections = {(s["kind"], s["key"]): s for s in report["sections"]}
    assert [i["number"] for i in sections[("autofix", "H4")]["items"]] == [1, 2]
    assert [i["element"] for i in sections[("would_have_fixed", "H6")]["items"]] == ["31"]
    assert [i["element"] for i in sections[("metric", "routing_edge_pct")]["items"]] == ["lanelet 1"]
    assert ("metric", "has_lanelets") not in sections and ("metric", "not_converted") not in sections
    assert report["summary"] == {"autofix": True, "preset": "perfect", "status": "fail"}


def test_dictionary_translations_get_their_own_section():
    translated = {"kind": "translated", "element": "signal.dynamic", "id": 71, "value": "yes",
                  "action": "read_as_no", "message": "signal.dynamic 71: no light code (vocabulary)", "location": [40.0, -5.0]}
    report = ar.build(records=[*RECORDS, translated], verdict=None, autofix_enabled=True, would_have_fixed=True)
    (section,) = [s for s in report["sections"] if s["kind"] == "vocabulary"]
    assert (section["key"], [i["element"] for i in section["items"]]) == ("signal.dynamic", ["71"])
    assert section["items"][0]["location"] == [40.0, -5.0]


def test_would_have_fixed_can_be_left_out():
    report = ar.build(records=RECORDS, verdict=VERDICT, autofix_enabled=True, would_have_fixed=False)
    assert all(s["kind"] != "would_have_fixed" for s in report["sections"])


@pytest.mark.parametrize(
    "switches, images",
    [({}, 6), ({"closeups": False}, 3), ({"birdview": False}, 3), ({"birdview": False, "closeups": False}, 0)],
)
def test_html_is_self_contained_and_follows_the_switches(switches, images):
    report = ar.build(records=RECORDS, verdict=VERDICT, autofix_enabled=True, would_have_fixed=True)
    html = ar.render_html(report=report, polylines=POLYLINES, config=ar.ReportConfig(**switches), title="case")
    assert html.count("<img src='data:image/png;base64,") == images  # 3 sections, a birdview and a close-up page each
    assert "src='http" not in html and "<script" not in html
    assert ("<li" in html) is True


def test_text_off_leaves_only_images():
    report = ar.build(records=RECORDS, verdict=VERDICT, autofix_enabled=True, would_have_fixed=True)
    html = ar.render_html(report=report, polylines=POLYLINES, config=ar.ReportConfig(text=False), title="case")
    assert "<li" not in html


def test_close_ups_are_capped():
    records = [_record(heuristic_id="H4", element_id=str(i), applied=True, location=(i, 0.0)) for i in range(30)]
    report = ar.build(records=records, verdict=None, autofix_enabled=True, would_have_fixed=True)
    html = ar.render_html(report=report, polylines=POLYLINES, config=ar.ReportConfig(max_closeups=24), title="case")
    assert html.count("<img") == 2  # one birdview, one close-up page of 24
    assert "6 more items have no close-up" in html


def _two_way_road(*, map_dir) -> None:
    sign = xb.SignalSpec(id="5", s=10.0, dynamic="no", type="206")
    xb.write_xodr(map_dir=map_dir, name="case", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0, signals=[sign])]))


def test_conversion_writes_the_report_next_to_the_outputs(tmp_path):
    _two_way_road(map_dir=tmp_path)
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "case_autocritic.json").read_text())
    keys = {(s["kind"], s["key"]) for s in report["sections"]}
    assert ("autofix", "H6") in keys  # H6 decided the sign gets no row: its lane reaches no junction
    assert ("metric", "stop_yield_stoplines") in keys  # that sign has no stopline
    assert "data:image/png;base64," in (tmp_path / "case_autocritic.html").read_text()


def test_rejected_conversion_still_writes_the_report(tmp_path):
    _two_way_road(map_dir=tmp_path)
    result = xb.run_convert_map(map_dir=str(tmp_path), **{"autocritic.preset": "perfect"})
    assert result.returncode == conversion_common.REJECTED_EXIT_CODE
    assert (tmp_path / "case_autocritic.html").exists() and not (tmp_path / "case.osm").exists()


@pytest.mark.parametrize("setting", ["report.enabled", "autocritic.enabled"])
def test_report_can_be_switched_off(tmp_path, setting):
    _two_way_road(map_dir=tmp_path)
    result = xb.run_convert_map(map_dir=str(tmp_path), **{setting: "false"})
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "case_autocritic.html").exists() and not (tmp_path / "case_autocritic.json").exists()


# --- the "Map" section: converted map with stoplines, beside the raw XODR -----------------------------------


def _row(*, agent_type: str, x: float, orientation: float = 0.0) -> dict:
    return {"actor_id": 1, "agent_type": agent_type, "x": x, "y": 0.0, "length": 1.0, "width": 3.5, "orientation": orientation}


def test_a_stopline_row_is_a_segment_across_its_lane():
    segment = ar.stopline_segment(row=_row(agent_type="stop_sign", x=10.0))
    np.testing.assert_allclose(segment, [[10.0, -1.75], [10.0, 1.75]])
    across_north = ar.stopline_segment(row=_row(agent_type="stop_sign", x=0.0, orientation=np.pi / 2))
    np.testing.assert_allclose(across_north, [[1.75, 0.0], [-1.75, 0.0]], atol=1e-9)


def _layers(*, raw_note=None) -> ar.MapLayers:
    face = np.array([[[0.0, -2.0], [20.0, -2.0], [20.0, 2.0]], [[0.0, -2.0], [20.0, 2.0], [0.0, 2.0]]])
    rows = [_row(agent_type=t, x=x) for t, x in (("traffic_light", 5.0), ("stop_sign", 10.0), ("yield_sign", 15.0))]
    raw = [] if raw_note else [("Driving", np.array([[0.0, -2.0], [20.0, -2.0], [20.0, 2.0], [0.0, 2.0]]))]
    return ar.MapLayers(
        road_faces=face,
        polylines=POLYLINES,
        stoplines=[(r["agent_type"], ar.stopline_segment(row=r)) for r in rows],
        raw_lanes=raw,
        raw_signals=np.zeros((0, 2)),
        raw_note=raw_note,
    )


def test_the_map_section_comes_first_with_every_stopline_type_counted():
    report = ar.build(records=[], verdict=None, autofix_enabled=True, would_have_fixed=True)
    html = ar.render_html(report=report, polylines=POLYLINES, config=ar.ReportConfig(), title="case", layers=_layers())
    assert html.index("<h2>Map") < html.index("No heuristic applied")
    for label in ("traffic light: 1", "stop sign: 1", "yield sign: 1"):
        assert label in html
    assert html.count("<img src='data:image/png;base64,") == 1
    assert "Raw XODR" in html


def test_the_map_section_says_why_the_raw_xodr_is_missing():
    report = ar.build(records=[], verdict=None, autofix_enabled=True, would_have_fixed=True)
    html = ar.render_html(
        report=report, polylines=POLYLINES, config=ar.ReportConfig(), title="case", layers=_layers(raw_note="no reader")
    )
    assert "Raw XODR not drawn: no reader" in html


def _light_rows(*, map_dir) -> int:
    rows = json.loads(next(map_dir.glob("*_stoplines.json")).read_text())
    return sum(r["agent_type"] == "traffic_light" for r in rows)


def test_every_report_draws_the_map_with_its_stoplines_and_the_raw_xodr(tmp_path):
    roads, junctions, controllers = xb.junction_map()
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers))
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    html = (tmp_path / "case_autocritic.html").read_text()
    assert _light_rows(map_dir=tmp_path) > 0
    assert f"traffic light: {_light_rows(map_dir=tmp_path)}" in html
    summary = json.loads((tmp_path / "case_autocritic.json").read_text())["map"]
    assert summary["stoplines"]["traffic_light"] == _light_rows(map_dir=tmp_path)
    assert summary["raw_xodr"]["note"] is None and summary["raw_xodr"]["lanes"] > 0


def test_the_raw_xodr_lies_on_the_converted_map_in_the_local_frame(tmp_path):
    road = xb.straight_road(id="1", length=60.0, x=1000.0, y=-500.0)
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=[road], offset=(100.0, 50.0, 0.0, 0.0)))
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    raw_extent = np.array(json.loads((tmp_path / "case_autocritic.json").read_text())["map"]["raw_xodr"]["extent"])
    points = np.concatenate(ar.load_polylines(map_dir=str(tmp_path)))
    np.testing.assert_allclose(raw_extent, [*points.min(axis=0), *points.max(axis=0)], atol=0.5)


def test_a_report_without_findings_still_draws_the_map(tmp_path):
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)]))
    result = xb.run_convert_map(map_dir=str(tmp_path), **{"report.raw_xodr": "false"})
    assert result.returncode == 0, result.stderr
    html = (tmp_path / "case_autocritic.html").read_text()
    assert "<h2>Map" in html and "data:image/png;base64," in html
    assert "Raw XODR not drawn: switched off (report.raw_xodr)" in html


def test_a_rejected_map_report_still_has_its_stoplines(tmp_path):
    roads, junctions, controllers = xb.junction_map()
    roads += [xb.straight_road(id="7", length=30.0, y=300.0), xb.straight_road(id="8", length=30.0, x=30.0, y=300.0)]
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers))
    result = xb.run_convert_map(map_dir=str(tmp_path), **{"autocritic.preset": "perfect"})
    assert result.returncode == conversion_common.REJECTED_EXIT_CODE, result.stderr
    summary = json.loads((tmp_path / "case_autocritic.json").read_text())["map"]
    assert summary["stoplines"]["traffic_light"] > 0


# --- plain-language explanations ----------------------------------------------------------------------------


def test_every_heuristic_and_check_has_a_plain_explanation():
    import conversion_heuristics
    import map_validation

    for heuristic in conversion_heuristics.HEURISTICS:
        assert 40 <= len(heuristic.plain) <= 400, heuristic.id
    controls = map_validation.ControlCoverage(
        lights=0, lights_without_stopline=[], stop_yield_signs=0, signs_without_stopline=[], positions={}
    )
    checks = map_validation.evaluate(metrics={}, metric_errors={}, controls=controls, config=map_validation.PROFILES["perfect"])
    names = {c.name for c in checks}
    assert len(names) >= 14 and names <= set(map_validation.CHECK_EXPLANATIONS)


def test_each_section_explains_itself_in_plain_language():
    import conversion_heuristics

    report = ar.build(records=RECORDS, verdict=VERDICT, autofix_enabled=True, would_have_fixed=True)
    html = ar.render_html(report=report, polylines=POLYLINES, config=ar.ReportConfig(), title="case")
    import html as html_text

    import map_validation

    assert html_text.escape(conversion_heuristics.by_id(heuristic_id="H4").plain) in html
    assert html_text.escape(conversion_heuristics.by_id(heuristic_id="H6").plain) in html
    assert "switched off" in html  # the would-have-fixed section says what it lists
    assert html_text.escape(map_validation.CHECK_EXPLANATIONS["routing_edge_pct"]) in html
    assert "How to read this report" in html


def test_geometry_repairs_get_their_own_explained_section():
    cut = {"kind": "geometry", "element": "bound_loop_cut", "id": 7, "value": 2.5, "action": "cut",
           "message": "lanelet 7: right bound looped back over 2.5 m - loop cut", "location": [3.0, 4.0]}
    report = ar.build(records=[cut], verdict=None, autofix_enabled=True, would_have_fixed=True)
    sections = {(s["kind"], s["key"]): s for s in report["sections"]}
    assert [i["location"] for i in sections[("geometry", "bound_loop_cut")]["items"]] == [[3.0, 4.0]]
    assert sections[("geometry", "bound_loop_cut")]["explanation"]
