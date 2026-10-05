"""Behaviour of ``convert_map.py`` on synthetic OpenDRIVE input (run in a subprocess)."""

import hashlib
import json
import os
import time

import pytest

import convert_map as cm
from tests.iai import xodr_builder as xb


def _write(*, map_dir, name: str, text: str, encoding: str = "utf-8") -> str:
    map_dir.mkdir(exist_ok=True)
    path = map_dir / f"{name}.xodr"
    path.write_bytes(text.encode(encoding))
    return str(path)


def test_latin1_xodr_with_non_ascii_name_converts(tmp_path):
    road = xb.straight_road(id="1", length=40.0, name="Hauptstraße Süd")
    _write(map_dir=tmp_path, name="latin1", text=xb.build_xodr(roads=[road], encoding="ISO-8859-1"), encoding="iso-8859-1")
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "latin1.osm").exists()


def test_incomplete_signal_validity_is_skipped_not_debugged(tmp_path):
    roads, junctions, controllers = xb.junction_map()
    connecting = roads[1]
    connecting.signals[0].validity = (None, -1)  # no fromLane
    connecting.signals[1].validity = (-7, -7)  # lane absent from the road
    _write(map_dir=tmp_path, name="validity", text=xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers))
    result = xb.run_convert_map(map_dir=str(tmp_path), timeout_s=30)
    assert result.returncode == 0, result.stderr
    assert "(Pdb)" not in result.stdout
    assert "validity without fromLane/toLane" in result.stderr
    assert "validity lane -7 not found" in result.stderr


def test_provenance_records_input_config_and_versions(tmp_path):
    xodr = _write(map_dir=tmp_path, name="prov", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)]))
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "prov_provenance.json") as f:
        provenance = json.load(f)
    with open(xodr, "rb") as f:
        assert provenance["input"]["sha256"] == hashlib.sha256(f.read()).hexdigest()
    assert provenance["input"]["path"] == os.path.abspath(xodr)
    assert provenance["config"]["dir_path"] == str(tmp_path)
    assert provenance["config"]["visualization_fov"] is None
    assert provenance["visualization_fov"] > 0
    assert set(provenance["converter"]) == {"commit", "dirty"}
    assert provenance["packages"]["torchdrivesim"]
    assert provenance["python"]
    assert provenance["created_utc"].endswith("+00:00")


def test_git_identity_outside_a_checkout_is_null(tmp_path):
    assert cm.git_identity(repo=str(tmp_path)) == {"commit": None, "dirty": None}


def test_conversion_writes_signal_groups_sidecar(tmp_path):
    roads, junctions, controllers = xb.junction_map()
    _write(map_dir=tmp_path, name="junction", text=xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers))
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "junction_signal_groups.json") as f:
        groups = json.load(f)
    assert groups["membership"]["11"] == {"junction": "100", "source": "controller"}
    assert groups["source_xodr"] == os.path.abspath(tmp_path / "junction.xodr")


@pytest.mark.parametrize(
    "setup, config, message",
    [
        ("none", {}, "no .xodr file"),
        ("two", {}, "2 .xodr files"),
        ("one", {"trim_radius": "100"}, "trim_radius requires center"),
    ],
)
def test_invalid_config_fails_fast_and_writes_nothing(tmp_path, setup, config, message):
    road = xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)])
    if setup in ("one", "two"):
        _write(map_dir=tmp_path, name="a", text=road)
    if setup == "two":
        _write(map_dir=tmp_path, name="b", text=road)
    before = sorted(os.listdir(tmp_path))
    start = time.monotonic()
    result = xb.run_convert_map(map_dir=str(tmp_path), timeout_s=30, **config)
    assert time.monotonic() - start < 5
    assert result.returncode != 0
    assert message in result.stderr
    assert sorted(os.listdir(tmp_path)) == before


def _long_map(*, tmp_path) -> None:
    roads = [xb.straight_road(id="1", length=3000.0), xb.straight_road(id="2", length=200.0, x=0.0, y=1500.0, hdg=1.0)]
    _write(map_dir=tmp_path, name="long", text=xb.build_xodr(roads=roads))


def test_default_fov_covers_the_whole_mesh(tmp_path):
    import map_quality as mq

    _long_map(tmp_path=tmp_path)
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    outputs = mq.ConversionOutputs.load(map_dir=str(tmp_path))
    assert outputs.visualization_fov > 3000
    assert mq.mesh_in_frame_fraction(outputs=outputs) == 1.0


def test_explicit_fov_is_kept(tmp_path):
    _long_map(tmp_path=tmp_path)
    result = xb.run_convert_map(map_dir=str(tmp_path), visualization_fov="800")
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "long_provenance.json") as f:
        assert json.load(f)["visualization_fov"] == 800


@pytest.mark.parametrize("rules, expected", [((None, None), "RHT"), (("LHT", "LHT"), "LHT"), (("LHT", "RHT"), "mixed")])
def test_traffic_side_recorded_in_provenance(tmp_path, rules, expected):
    roads = [xb.straight_road(id=str(i), length=30.0, y=20.0 * i, rule=rule) for i, rule in enumerate(rules, start=1)]
    _write(map_dir=tmp_path, name="side", text=xb.build_xodr(roads=roads))
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "side_provenance.json") as f:
        assert json.load(f)["traffic_side"] == expected
