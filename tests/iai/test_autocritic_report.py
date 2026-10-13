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
    assert ("metric", "routing_edge_pct") in keys
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
