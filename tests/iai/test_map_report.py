"""Tests for ``map_report.py``: outcome classification, input statistics, digests, compare."""

import json
import os
import subprocess
import sys

import pytest

import map_report as mr
from tests.iai import xodr_builder as xb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _child(*, tmp_path, code: str, timeout_s: float = 30.0) -> mr.ChildResult:
    script = tmp_path / "child.py"
    script.write_text(code)
    return mr.run_child(
        argv=[sys.executable, str(script)], cwd=str(tmp_path), timeout_s=timeout_s, log_prefix=str(tmp_path / "child")
    )


def test_ok_outcome_and_rss(tmp_path):
    result = _child(tmp_path=tmp_path, code="x = bytearray(50 * 1024 * 1024)\n")
    assert mr.classify(result=result, root=str(tmp_path)) == {"outcome": "ok"}
    assert result.peak_rss_mb >= 50


def test_exception_outcome_names_class_and_frame(tmp_path):
    result = _child(tmp_path=tmp_path, code="def f():\n    float('1,5')\nf()\n")
    assert mr.classify(result=result, root=str(tmp_path)) == {
        "outcome": "exception",
        "exception": "ValueError",
        "location": "child.py:2",
    }


def test_refusal_exit_code_is_refused_with_message(tmp_path):
    result = _child(tmp_path=tmp_path, code="import sys\nprint('Cannot convert: no geo-reference', file=sys.stderr)\nsys.exit(3)\n")
    assert mr.classify(result=result, root=str(tmp_path)) == {"outcome": "refused", "message": "Cannot convert: no geo-reference"}


def test_missing_script_is_an_exception_not_a_refusal(tmp_path):
    result = mr.run_child(argv=[sys.executable, str(tmp_path / "absent.py")], cwd=str(tmp_path), timeout_s=30, log_prefix=str(tmp_path / "absent"))
    assert mr.classify(result=result, root=str(tmp_path))["outcome"] == "exception"


def test_breakpoint_is_pdb_not_hang(tmp_path):
    result = _child(tmp_path=tmp_path, code="breakpoint()\nraise SystemExit(0)\n", timeout_s=20.0)
    assert not result.timed_out
    assert mr.classify(result=result, root=str(tmp_path))["outcome"] == "pdb"


def test_timeout_kills_process_group(tmp_path):
    code = "import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\ntime.sleep(60)\n"
    result = _child(tmp_path=tmp_path, code=code, timeout_s=1.0)
    assert result.timed_out
    assert mr.classify(result=result, root=str(tmp_path)) == {"outcome": "timeout"}
    assert result.wall_s < 10


def _fixture_xodr() -> str:
    """Junction with 2 controllers, an LHT road with ``broken solid``/``custom`` marks, an isolated road."""
    roads, junctions, controllers = xb.junction_map()
    lht = xb.straight_road(id="8", length=40.0, x=0.0, y=-100.0, rule="LHT")
    lht.lane_sections[0].lanes[0].road_mark = "broken solid"
    lht.lane_sections[0].lanes[1].road_mark = "custom"
    roads += [lht, xb.straight_road(id="9", length=30.0, x=0.0, y=200.0)]
    return xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers, vendor="SyntheticVendor")


def test_xodr_statistics(tmp_path):
    path = tmp_path / "fixture.xodr"
    path.write_text(_fixture_xodr())
    stats = mr.xodr_statistics(path=str(path))
    assert stats["vendor"] == "SyntheticVendor"
    assert (stats["rev_major"], stats["rev_minor"]) == ("1", "4")
    # 2 root controllers + 2 junction controller references.
    assert stats["counts"] == {"controller": 4, "control": 2, "signal": 2, "signalReference": 0, "object": 0}
    assert stats["road_marks"]["broken solid"] == 1
    assert stats["road_marks"]["custom"] == 1


def test_xodr_statistics_namespaced_root(tmp_path):
    path = tmp_path / "namespaced.xodr"
    path.write_text(_fixture_xodr().replace("<OpenDRIVE>", '<OpenDRIVE xmlns="http://www.opendrive.org">'))
    stats = mr.xodr_statistics(path=str(path))
    assert stats["vendor"] == "SyntheticVendor"
    assert stats["counts"]["signal"] == 2


def test_handled_road_marks_follow_converter():
    handled = mr.handled_road_mark_types(converter=REPO_ROOT, mark_types=["solid", "broken", "broken solid", "custom"])
    assert handled == ["broken", "broken solid", "solid"]


def test_canonical_mesh_digest_ignores_category_order(tmp_path):
    a = {"verts": [[0, 0]], "faces": [], "categories": ["road", "lane"], "vert_category": [[0, 1, 1]], "colors": {}, "zs": {}}
    b = {"verts": [[0, 0]], "faces": [], "categories": ["lane", "road"], "vert_category": [[1, 0, 0]], "colors": {}, "zs": {}}
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))
    assert mr.canonical_mesh_digest(path=str(tmp_path / "a.json")) == mr.canonical_mesh_digest(path=str(tmp_path / "b.json"))
    b["verts"] = [[0, 1]]
    (tmp_path / "b.json").write_text(json.dumps(b))
    assert mr.canonical_mesh_digest(path=str(tmp_path / "a.json")) != mr.canonical_mesh_digest(path=str(tmp_path / "b.json"))


def test_discover_maps_exclude_globs(tmp_path):
    for rel in ("a/Town12/Town12.xodr", "old/x/x.xodr", "b/keep.xodr"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("<OpenDRIVE/>")
    found = mr.discover_maps(maps_dir=str(tmp_path), only=None, exclude=("old/*", "*/Town12.xodr"))
    assert found == ["b/keep.xodr"]


def test_moved_rules():
    assert mr.moved(key="metrics.routing_edge_pct", before=90.0, after=90.5) is False
    assert mr.moved(key="metrics.routing_edge_pct", before=90.0, after=92.0) is True
    assert mr.moved(key="metrics.lanelet_count", before=10, after=11) is True
    assert mr.moved(key="digests.<map>.osm", before="a", after="b") is True
    assert mr.moved(key="conversion.wall_s", before=1.0, after=100.0) is False
    assert mr.moved(key="conversion.peak_rss_mb", before=1.0, after=100.0) is False


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    """The same two maps reported into two different output directories."""
    maps = tmp_path_factory.mktemp("maps")
    (maps / "good").mkdir()
    (maps / "good" / "fixture.xodr").write_text(_fixture_xodr())
    (maps / "bad").mkdir()
    roads = [xb.straight_road(id="1", length=30.0)]
    text = xb.build_xodr(roads=roads)
    text = text[: text.index("<lanes>")] + text[text.index("</lanes>") + len("</lanes>") :]  # a road with no lanes
    (maps / "bad" / "no_lanes.xodr").write_text(text)
    reports = []
    for name, extra in (("run_a", []), ("run_b", []), ("run_parallel", ["--workers", "2", "--memory-budget-mb", "8000"])):
        out = tmp_path_factory.mktemp(name)
        subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "map_report.py"), "run", "--maps", str(maps), "--out", str(out), "--fsm", *extra],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=900,
        )
        with open(out / "report.json") as f:
            reports.append(json.load(f))
    return reports


def test_run_records_outcomes(two_runs):
    report = two_runs[0]
    by_rel = {m["relpath"]: m for m in report["maps"]}
    good, bad = by_rel["good/fixture.xodr"], by_rel["bad/no_lanes.xodr"]
    assert good["conversion"]["outcome"] == "ok"
    assert good["metrics"]["stoplines_by_agent_type"] == {"traffic_light": 2}
    assert good["fsm"]["outcome"] == "ok"
    assert good["fsm"]["lights"] == 2
    assert good["fsm"]["junction_seeded_intersections"] >= 1
    assert good["fsm"]["unassigned_lights"] == 0
    assert isinstance(good["fsm"]["flagged"], int)
    assert bad["conversion"]["outcome"] == "exception"
    assert bad["conversion"]["exception"] == "Exception"
    assert bad["conversion"]["location"].startswith("crdesigner/map_conversion/opendrive/odr2cr/opendrive_parser/")
    assert "custom" not in report["road_mark_types_handled"]


def test_run_paths_are_absolute(two_runs):
    report = two_runs[0]
    for key in ("maps_dir", "out_dir", "converter", "ruler"):
        assert os.path.isabs(report[key])
    for m in report["maps"]:
        assert os.path.isabs(m["outputs_dir"])
        assert os.path.isabs(m["input"]["path"])
    assert os.path.isfile(os.path.join(report["out_dir"], "report.html"))


def test_digests_identical_across_output_dirs(two_runs):
    a, b = ({m["relpath"]: m for m in r["maps"]} for r in two_runs[:2])
    digests = a["good/fixture.xodr"]["digests"]
    assert {"metadata.json", "<map>.osm", "<map>_stoplines.json", "<map>_mesh.json"} <= set(digests)
    assert digests == b["good/fixture.xodr"]["digests"]
    assert a["good/fixture.xodr"]["fsm"]["grouping_digest"] == b["good/fixture.xodr"]["fsm"]["grouping_digest"]


def test_compare_identical_runs_has_no_moves(two_runs, tmp_path):
    comparison = mr.compare_reports(before=two_runs[0], after=two_runs[1])
    assert comparison["maps_with_moves"] == []
    assert len(comparison["maps"]) == 2


def test_parallel_run_matches_serial_run(two_runs):
    serial, parallel = two_runs[0], two_runs[2]
    assert [m["relpath"] for m in parallel["maps"]] == [m["relpath"] for m in serial["maps"]]
    assert mr.compare_reports(before=serial, after=parallel)["maps_with_moves"] == []
    assert (parallel["workers"], parallel["memory_budget_mb"]) == (2, 8000.0)


def _costs(**peaks_mb: float) -> dict[str, mr.MapCost]:
    return {rel: mr.MapCost(wall_s=1.0, peak_rss_mb=peak) for rel, peak in peaks_mb.items()}


def test_next_to_start_waits_for_a_free_worker():
    costs = _costs(a=100.0)
    assert mr.next_to_start(pending=["a"], costs=costs, running=["x", "y"], workers=2, budget_mb=None) is None


def test_next_to_start_skips_a_map_over_the_remaining_budget():
    costs = _costs(big=900.0, small=200.0, x=500.0)
    assert mr.next_to_start(pending=["big", "small"], costs=costs, running=["x"], workers=4, budget_mb=1000.0) == "small"


def test_next_to_start_runs_an_oversized_map_alone():
    costs = _costs(huge=5000.0, x=10.0)
    assert mr.next_to_start(pending=["huge"], costs=costs, running=[], workers=4, budget_mb=1000.0) == "huge"
    assert mr.next_to_start(pending=["huge"], costs=costs, running=["x"], workers=4, budget_mb=1000.0) is None


def test_next_to_start_without_budget_takes_the_head():
    costs = _costs(a=1e6, b=1.0)
    assert mr.next_to_start(pending=["a", "b"], costs=costs, running=[], workers=2, budget_mb=None) == "a"


def test_cost_hints_take_the_larger_of_conversion_and_fsm_peak(tmp_path):
    report = {"maps": [
        {"relpath": "a.xodr", "conversion": {"wall_s": 3.0, "peak_rss_mb": 100.0}, "fsm": {"wall_s": 2.0, "peak_rss_mb": 300.0}},
        {"relpath": "b.xodr", "conversion": {"wall_s": 1.0, "peak_rss_mb": 50.0}},
    ]}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    hints = mr.load_cost_hints(report_path=str(path))
    assert hints == {"a.xodr": mr.MapCost(wall_s=5.0, peak_rss_mb=300.0), "b.xodr": mr.MapCost(wall_s=1.0, peak_rss_mb=50.0)}


def test_workers_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        mr.run_report(maps_dir=str(tmp_path), out_dir=str(tmp_path / "out"), converter=REPO_ROOT, only=None, timeout_s=10, with_fsm=False, workers=0)


def test_convert_args_reach_the_converter(tmp_path):
    maps = tmp_path / "maps" / "nogeo"
    maps.mkdir(parents=True)
    (maps / "nogeo.xodr").write_text(xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)], geo_reference="+proj=merc +lat_ts=0"))
    out = tmp_path / "out"
    script = os.path.join(REPO_ROOT, "map_report.py")
    for extra, outcome in (([], "refused"), (["--convert-arg", "allow_zero_origin=true"], "ok")):
        subprocess.run(
            [sys.executable, script, "run", "--maps", str(tmp_path / "maps"), "--out", str(out), *extra],
            check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=300,
        )
        with open(out / "report.json") as f:
            assert json.load(f)["maps"][0]["conversion"]["outcome"] == outcome


def test_refusal_message_is_comparable_across_runs(tmp_path):
    maps = tmp_path / "maps" / "nogeo"
    maps.mkdir(parents=True)
    (maps / "nogeo.xodr").write_text(xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)], geo_reference="+proj=merc +lat_ts=0"))
    messages = []
    for name in ("a", "b"):
        report = mr.run_report(maps_dir=str(tmp_path / "maps"), out_dir=str(tmp_path / name), converter=REPO_ROOT, only=None, timeout_s=300, with_fsm=False)
        with open(report) as f:
            messages.append(json.load(f)["maps"][0]["conversion"]["message"])
    assert messages[0] == messages[1]
    assert "<RUN_DIR>" in messages[0]
