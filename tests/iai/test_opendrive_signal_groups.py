"""Tests for ``opendrive_signal_groups.py``: junction membership sources, controllers, the CLI."""

import json
import os
import subprocess
import sys

import opendrive_signal_groups as osg
from tests.iai import xodr_builder as xb


def _extract(*, tmp_path, roads, junctions=(), controllers=()) -> dict:
    path = tmp_path / "map.xodr"
    path.write_text(xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers))
    return osg.extract_signal_groups(xodr_path=str(path))


def test_membership_from_junction_controllers(tmp_path):
    roads, junctions, controllers = xb.junction_map()
    groups = _extract(tmp_path=tmp_path, roads=roads, junctions=junctions, controllers=controllers)
    assert groups["membership"] == {
        "11": {"junction": "100", "source": "controller"},
        "12": {"junction": "100", "source": "controller"},
    }
    assert [(c["id"], c["signal_ids"], c["sequence"], c["junction"]) for c in groups["controllers"]] == [
        ("1001", ["11"], 0, "100"),
        ("1002", ["12"], 1, "100"),
    ]


def test_membership_from_road_junction_attribute(tmp_path):
    roads, junctions, _ = xb.junction_map(with_controllers=False)
    groups = _extract(tmp_path=tmp_path, roads=roads, junctions=junctions)
    assert groups["membership"]["11"] == {"junction": "100", "source": "road_junction"}


def test_membership_from_approach_road_within_reach(tmp_path):
    roads, junctions, _ = xb.junction_map(with_signals=False, approach_length=200.0)
    roads[0].signals = [xb.SignalSpec(id="21", s=190.0), xb.SignalSpec(id="22", s=10.0)]
    groups = _extract(tmp_path=tmp_path, roads=roads, junctions=junctions)
    assert groups["membership"]["21"] == {"junction": "100", "source": "approach_road"}
    assert "22" not in groups["membership"]  # 190 m from the junction end


def test_disagreeing_sources_warn_and_higher_priority_wins(tmp_path):
    roads, junctions, controllers = xb.junction_map()
    roads[1].junction = "999"  # the road says another junction than the controller does
    groups = _extract(tmp_path=tmp_path, roads=roads, junctions=junctions, controllers=controllers)
    assert groups["membership"]["11"] == {"junction": "100", "source": "controller"}
    assert any("road_junction says 999" in w for w in groups["warnings"])


def test_non_integer_signal_ids_are_listed_not_joined(tmp_path):
    roads, junctions, _ = xb.junction_map(with_controllers=False)
    roads[1].signals[0].id = "light-a"
    groups = _extract(tmp_path=tmp_path, roads=roads, junctions=junctions)
    assert groups["non_integer_signal_ids"] == ["light-a"]
    assert "light-a" not in groups["membership"]


def test_cli_writes_sidecar_and_prints_absolute_path(tmp_path):
    roads, junctions, controllers = xb.junction_map()
    xodr = tmp_path / "map.xodr"
    xodr.write_text(xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers))
    out = tmp_path / "map_signal_groups.json"
    result = subprocess.run(
        [sys.executable, os.path.join(xb.REPO_ROOT, "opendrive_signal_groups.py"), "--xodr", str(xodr), "--out", "map_signal_groups.json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"wrote {out}" in result.stdout
    assert json.loads(out.read_text())["membership"]["12"]["junction"] == "100"
