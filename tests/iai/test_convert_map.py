"""Behaviour of ``convert_map.py`` on synthetic OpenDRIVE input (run in a subprocess)."""

import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import conversion_common
import convert_map as cm
from tests.iai import xodr_builder as xb


def test_latin1_xodr_with_non_ascii_name_converts(tmp_path):
    road = xb.straight_road(id="1", length=40.0, name="Hauptstraße Süd")
    xb.write_xodr(map_dir=tmp_path, name="latin1", text=xb.build_xodr(roads=[road], encoding="ISO-8859-1"), encoding="iso-8859-1")
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "latin1.osm").exists()


def test_incomplete_signal_validity_is_skipped_not_debugged(tmp_path):
    roads, junctions, controllers = xb.junction_map()
    connecting = roads[1]
    connecting.signals[0].validity = (None, -1)  # no fromLane
    connecting.signals[1].validity = (-7, -7)  # lane absent from the road
    xb.write_xodr(map_dir=tmp_path, name="validity", text=xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers))
    result = xb.run_convert_map(map_dir=str(tmp_path), timeout_s=30)
    assert result.returncode == 0, result.stderr
    assert "(Pdb)" not in result.stdout
    assert "validity without fromLane/toLane" in result.stderr
    assert "validity lane -7 not found" in result.stderr


def test_provenance_records_input_config_and_versions(tmp_path):
    xodr = xb.write_xodr(map_dir=tmp_path, name="prov", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)]))
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
    assert conversion_common.git_identity(repo=str(tmp_path)) == {"commit": None, "dirty": None}


def test_conversion_writes_signal_groups_sidecar(tmp_path):
    roads, junctions, controllers = xb.junction_map()
    xb.write_xodr(map_dir=tmp_path, name="junction", text=xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers))
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
        xb.write_xodr(map_dir=tmp_path, name="a", text=road)
    if setup == "two":
        xb.write_xodr(map_dir=tmp_path, name="b", text=road)
    before = sorted(os.listdir(tmp_path))
    start = time.monotonic()
    result = xb.run_convert_map(map_dir=str(tmp_path), timeout_s=30, **config)
    assert time.monotonic() - start < 5
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
    assert message in result.stderr
    assert sorted(os.listdir(tmp_path)) == before


@pytest.mark.parametrize("with_dir_path", [False, True], ids=["neither", "both"])
def test_dir_path_and_xodr_path_are_exclusive(tmp_path, with_dir_path):
    xodr = xb.write_xodr(map_dir=tmp_path, name="a", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)]))
    config = {"xodr_path": xodr} if with_dir_path else {}
    result = xb.run_convert_map(map_dir=str(tmp_path) if with_dir_path else None, timeout_s=30, **config)
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
    assert "exactly one of dir_path and xodr_path" in result.stderr
    assert sorted(os.listdir(tmp_path)) == ["a.xodr"]


def test_xodr_path_writes_every_output_beside_it_and_returns_the_osm(tmp_path):
    xodr = xb.write_xodr(map_dir=tmp_path, name="town", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)]))
    xb.write_xodr(map_dir=tmp_path, name="other", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)]))
    code = f"import convert_map as cm; print(cm.convert_map(cm.MapConversionConfig(xodr_path={xodr!r})))"
    result = subprocess.run([sys.executable, "-c", code], cwd=xb.REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    out_dir = tmp_path / "town"
    assert result.stdout.strip().splitlines()[-1] == str(out_dir / "town.osm")
    expected = {"town.osm", "town_stoplines.json", "town_mesh.json", "town_signal_groups.json",
                "town_provenance.json", "metadata.json", "visualization.png"}
    assert expected <= set(os.listdir(out_dir))


def _long_map(*, tmp_path) -> None:
    roads = [xb.straight_road(id="1", length=3000.0), xb.straight_road(id="2", length=200.0, x=0.0, y=1500.0, hdg=1.0)]
    xb.write_xodr(map_dir=tmp_path, name="long", text=xb.build_xodr(roads=roads))


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
    xb.write_xodr(map_dir=tmp_path, name="side", text=xb.build_xodr(roads=roads))
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "side_provenance.json") as f:
        assert json.load(f)["traffic_side"] == expected


@pytest.mark.parametrize(
    "text, lat, lon, translation",
    [
        ("+proj=tmerc +lat_0=49 +lon_0=8 +k=1 +x_0=0 +y_0=0 +ellps=WGS84", 49.0, 8.0, (0.0, 0.0)),
        ("+proj=tmerc +lat_0=4.9e+01 +lon_0=8.0e+00 +ellps=WGS84", 49.0, 8.0, (0.0, 0.0)),
        ("+proj=tmerc +lat_0=0 +lon_0=-117 +k=0.9996 +x_0=19951.4 +y_0=-3.63093e+06 +datum=WGS84", 0.0, -117.0, (19951.4, -3630930.0)),
    ],
)
def test_parse_tmerc_geo_reference(text, lat, lon, translation):
    geo = cm.parse_geo_reference(text=text, centre=None)
    assert (geo.lat, geo.lon) == pytest.approx((lat, lon))
    assert geo.translation == pytest.approx(translation)


def test_parse_keeps_every_digit_of_the_origin():
    geo = cm.parse_geo_reference(text="+proj=tmerc +lat_0=37.33241890014467 +lon_0=-121.8928770999713 +k=1", centre=None)
    assert (geo.lat, geo.lon) == (37.33241890014467, -121.8928770999713)


def test_parse_utm_geo_reference_centres_origin_on_the_map():
    geo = cm.parse_geo_reference(text="+proj=utm +zone=32 +ellps=WGS84 +units=m", centre=(500000.0, 5428000.0))
    assert geo.lon == pytest.approx(9.0)  # zone 32 central meridian
    assert 48.9 < geo.lat < 49.1
    assert geo.translation == (500000.0, 5428000.0)


@pytest.mark.parametrize("text", [None, "", "not a projection", "+proj=utm +ellps=WGS84", "+proj=merc +lat_ts=0"])
def test_unusable_geo_reference_is_none(text):
    assert cm.parse_geo_reference(text=text, centre=(0.0, 0.0)) is None


def test_missing_geo_reference_converts_at_zero_with_warning(tmp_path):
    text = xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)], geo_reference=None)
    xb.write_xodr(map_dir=tmp_path, name="nogeo", text=text)
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "has no <geoReference> - converting at latitude/longitude (0, 0)" in result.stderr
    with open(tmp_path / "metadata.json") as f:
        assert json.load(f)["lanelet_map_origin"] == [0.0, 0.0]
    with open(tmp_path / "nogeo_provenance.json") as f:
        assert json.load(f)["geo_reference"] == "absent"


def test_unusable_geo_reference_fails_unless_allowed(tmp_path):
    text = xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)], geo_reference="+proj=merc +lat_ts=0")
    xb.write_xodr(map_dir=tmp_path, name="badgeo", text=text)
    refused = xb.run_convert_map(map_dir=str(tmp_path))
    assert refused.returncode == conversion_common.REFUSED_EXIT_CODE
    assert "allow_zero_origin" in refused.stderr
    assert sorted(os.listdir(tmp_path)) == ["badgeo.xodr", "badgeo_conversion_report.json"]  # no map outputs
    with open(tmp_path / "badgeo_conversion_report.json") as f:
        assert json.load(f)["error"]["type"] == "InputError"
    allowed = xb.run_convert_map(map_dir=str(tmp_path), allow_zero_origin="true")
    assert allowed.returncode == 0, allowed.stderr
    with open(tmp_path / "badgeo_provenance.json") as f:
        assert json.load(f)["geo_reference"] == "unusable_zero_origin_allowed"


def test_malformed_geo_reference_tokens_are_ignored():
    text = "++proj=tmerc +lon_0=-73.99735 +lat_0=40.736652 +=alt_0=0 +ellps=WGS84 +units=m"
    geo = cm.parse_geo_reference(text=text, centre=None)
    assert (geo.lat, geo.lon) == (40.736652, -73.99735)


def test_undefined_object_orientation_is_treated_as_none(tmp_path):
    roads, junctions, controllers = xb.junction_map()
    text = xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers).replace(
        "<objects/>",
        '<objects><object id="900" s="5.0" t="0.0" type="pole" name="pole" orientation="undefined" '
        'zOffset="0.0" length="0.5" width="0.5" height="1.0" hdg="0.0" pitch="0.0" roll="0.0"/></objects>',
        1,
    )
    xb.write_xodr(map_dir=tmp_path, name="orient", text=text)
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "orientation 'undefined' - using 'none'" in result.stderr


def test_header_offset_applies_to_stoplines(tmp_path):
    import map_quality as mq
    from shapely.geometry import LineString, Point

    roads, junctions, controllers = xb.junction_map()
    text = xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers, offset=(120.0, -40.0, 0.0, 0.0))
    xb.write_xodr(map_dir=tmp_path, name="offset", text=text)
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    outputs = mq.ConversionOutputs.load(map_dir=str(tmp_path))
    centrelines = [LineString([(p.x, p.y) for p in ll.centerline]) for ll in outputs.lanelet_map.laneletLayer]
    assert outputs.stoplines
    for row in outputs.stoplines:
        assert min(c.distance(Point(row["x"], row["y"])) for c in centrelines) < 0.5


def _write_joined_roads(*, map_dir, offset, geo_reference=xb.DEFAULT_GEO_REFERENCE) -> None:
    """Road 1 (0..50 m) joined to road 2 (50..100 m) as ``<map_dir>/joined.xodr``."""
    road1 = xb.straight_road(id="1", length=50.0, successor=xb.LinkSpec(element_type="road", element_id="2", contact_point="start"))
    road2 = xb.straight_road(id="2", length=50.0, x=50.0, predecessor=xb.LinkSpec(element_type="road", element_id="1", contact_point="end"))
    xb.write_xodr(map_dir=map_dir, name="joined", text=xb.build_xodr(roads=[road1, road2], geo_reference=geo_reference, offset=offset))


def _joined_roads_centrelines(*, map_dir, offset, geo_reference=xb.DEFAULT_GEO_REFERENCE, **convert_args) -> tuple[np.ndarray, str]:
    """Convert the joined roads; sorted lanelet centreline end points and stderr."""
    import map_quality as mq

    _write_joined_roads(map_dir=map_dir, offset=offset, geo_reference=geo_reference)
    result = xb.run_convert_map(map_dir=str(map_dir), **convert_args)
    assert result.returncode == 0, result.stderr
    outputs = mq.ConversionOutputs.load(map_dir=str(map_dir))
    ends = [((ll.centerline[0].x, ll.centerline[0].y), (ll.centerline[-1].x, ll.centerline[-1].y)) for ll in outputs.lanelet_map.laneletLayer]
    return np.array(sorted(np.round(ends, 3).tolist())).reshape(-1, 2), result.stderr


def test_header_offset_shifts_file_coordinates(tmp_path):
    plain, _ = _joined_roads_centrelines(map_dir=tmp_path / "plain", offset=None)
    shifted, _ = _joined_roads_centrelines(map_dir=tmp_path / "shifted", offset=(100.0, 50.0, 0.0, 0.0))
    expected = plain + np.array([100.0, 50.0])
    assert shifted == pytest.approx(expected, abs=1e-3)


def test_header_offset_rotation_is_ignored_with_a_warning(tmp_path):
    plain, _ = _joined_roads_centrelines(map_dir=tmp_path / "plain", offset=None)
    rotated, stderr = _joined_roads_centrelines(map_dir=tmp_path / "rotated", offset=(0.0, 0.0, 0.0, 0.5))
    assert rotated == pytest.approx(plain, abs=1e-3)  # roads stay joined, not torn apart
    assert "hdg=0.5 rad" in stderr and "ignored" in stderr


@pytest.mark.parametrize(
    "geo_reference, offset",
    [
        ("+proj=tmerc +lat_0=51.7 +lon_0=8.7 +k=1 +x_0=0 +y_0=0 +datum=WGS84", (479701.4, 5730797.1, 0.0, 0.0)),
        ("+proj=tmerc", (3926727.07, -12809834.93, 0.0, 0.0)),
    ],
)
def test_offset_outside_the_projection_is_refused_unless_allowed(tmp_path, geo_reference, offset):
    plain, _ = _joined_roads_centrelines(map_dir=tmp_path / "plain", offset=None)
    map_dir = tmp_path / "far"
    _write_joined_roads(map_dir=map_dir, offset=offset, geo_reference=geo_reference)
    refused = xb.run_convert_map(map_dir=str(map_dir))
    assert refused.returncode == conversion_common.REFUSED_EXIT_CODE, refused.stderr
    assert "<offset>" in refused.stderr and "allow_zero_origin" in refused.stderr
    assert sorted(os.listdir(map_dir)) == ["joined.xodr", "joined_conversion_report.json"]
    allowed, stderr = _joined_roads_centrelines(map_dir=map_dir, offset=offset, geo_reference=geo_reference, allow_zero_origin="true")
    assert "converting at (0, 0) in file coordinates" in stderr
    assert allowed == pytest.approx(plain, abs=1e-3)
    with open(map_dir / "metadata.json") as f:
        assert json.load(f)["lanelet_map_origin"] == [0.0, 0.0]
    with open(map_dir / "joined_provenance.json") as f:
        assert json.load(f)["geo_reference"] == "unusable_zero_origin_allowed"


def test_parse_false_origin_tmerc_centres_origin_on_the_map():
    text = "+proj=tmerc +lat_0=0 +lon_0=-117 +k=0.9996 +x_0=19951.4 +y_0=-3630930 +datum=WGS84"
    geo = cm.parse_geo_reference(text=text, centre=(19951.4, 0.0))
    assert geo.lon == pytest.approx(-117.0)
    assert 32.5 < geo.lat < 33.1
    assert geo.translation == (19951.4, 0.0)


def test_false_northing_places_map_at_true_latitude(tmp_path):
    import map_quality as mq
    import lanelet2

    geo = "+proj=tmerc +lat_0=0 +lon_0=-117 +k=0.9996 +x_0=0 +y_0=-3630930 +datum=WGS84"
    xb.write_xodr(map_dir=tmp_path, name="sd", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)], geo_reference=geo))
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    outputs = mq.ConversionOutputs.load(map_dir=str(tmp_path))
    x, y, _ = outputs.points[0]
    lat0, lon0 = outputs.metadata["lanelet_map_origin"]
    gps = mq.utm_projector(origin_lat=lat0, origin_lon=lon0).reverse(lanelet2.core.BasicPoint3d(x, y, 0.0))
    assert 32.5 < gps.lat < 33.1  # y_0 = -3,630,930 m puts the road ~32.8 deg north, not on the equator
    assert abs(outputs.points[:, :2]).max() < 100.0  # local origin at the map centre, not 3,630 km away


def _write_config(*, path, text: str) -> str:
    path.write_text(text)
    return str(path)


def test_config_file_sets_options_and_the_command_line_overrides_them(tmp_path):
    config = _write_config(
        path=tmp_path / "c.yaml",
        text="domain: from_file\nautocritic: {preset: strict}\nautofix:\n  heuristics:\n    facing_lanes: {enabled: false}\n",
    )
    cfg = cm.load_config(argv=[f"config={config}", "domain=from_cli"])
    assert (cfg.domain, cfg.autocritic.preset) == ("from_cli", "strict")
    assert not cfg.autofix.heuristics.facing_lanes.enabled
    assert cfg.autofix.heuristics.stop_at_signal_s.enabled


def test_defaults_without_a_config_file():
    assert cm.load_config(argv=[]) == cm.MapConversionConfig()


@pytest.mark.parametrize(
    "argv, named",
    [
        (["heuristics.facing_lanes=false"], "heuristics"),
        (["validation_profile=strict"], "validation_profile"),
        (["config=/nonexistent/c.yaml"], "/nonexistent/c.yaml"),
    ],
)
def test_unknown_keys_and_missing_config_files_are_refused(argv, named):
    with pytest.raises(cm.ConfigError, match=named):
        cm.load_config(argv=argv)


def test_unknown_key_in_a_config_file_is_refused(tmp_path):
    config = _write_config(path=tmp_path / "c.yaml", text="autofix: {heuristics: {no_such_rule: {enabled: true}}}\n")
    with pytest.raises(cm.ConfigError, match="no_such_rule"):
        cm.load_config(argv=[f"config={config}"])


def test_config_file_is_read_by_the_command_line(tmp_path):
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)]))
    config = _write_config(path=tmp_path / "c.yaml", text="autocritic: {preset: strict}\n")
    result = xb.run_convert_map(map_dir=str(tmp_path), config=config)
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "case_conversion_report.json") as f:
        assert json.load(f)["validation"]["profile"] == "strict"


def _report(*, map_dir, name: str) -> dict:
    with open(map_dir / f"{name}_conversion_report.json") as f:
        return json.load(f)


def _past_end_findings(*, report: dict) -> list[tuple[str, str, bool]]:
    return [(str(r["id"]), r["action"], r["applied"]) for r in report["records"] if r["kind"] == "heuristic" and r["element"] == "H14"]


def _write_road_with_signal_at(*, map_dir, s: float) -> None:
    sign = xb.SignalSpec(id="7", s=s, dynamic="no", type="206", validity=(-1, -1))
    xb.write_xodr(map_dir=map_dir, name="case", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=50.0, signals=[sign])]))


@pytest.mark.parametrize("s, action", [(50.6, "placed_at_road_end"), (55.0, "dropped")])
def test_signal_past_the_road_end_is_placed_or_dropped(tmp_path, s, action):
    _write_road_with_signal_at(map_dir=tmp_path, s=s)
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    report = _report(map_dir=tmp_path, name="case")
    assert _past_end_findings(report=report) == [("7", action, True)]
    assert report["validation"]["controls"]["stop_yield_signs"] == (1 if action == "placed_at_road_end" else 0)


def test_signal_past_the_road_end_is_refused_without_autofix(tmp_path):
    _write_road_with_signal_at(map_dir=tmp_path, s=50.6)
    result = xb.run_convert_map(map_dir=str(tmp_path), **{"autofix.enabled": "false"})
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
    assert "signal 7" in result.stderr
    assert _past_end_findings(report=_report(map_dir=tmp_path, name="case")) == [("7", "placed_at_road_end", False)]
    assert not (tmp_path / "case.osm").exists()


def test_signal_past_the_road_end_by_a_rounding_error_is_left_to_the_reader(tmp_path):
    _write_road_with_signal_at(map_dir=tmp_path, s=50.0001)
    result = xb.run_convert_map(map_dir=str(tmp_path), **{"autofix.enabled": "false"})
    assert result.returncode == 0, result.stderr
    assert _past_end_findings(report=_report(map_dir=tmp_path, name="case")) == []


def _write_roads_with_gap(*, map_dir, gap: float) -> None:
    """Road 1 (0..50 m) and road 2, drawn the other way, meeting end to end ``gap`` metres apart at x=50.

    Lanes of opposite id signs are never concatenated, so the seam stays (a straight chain of
    lanelets less than 1 m apart is concatenated into one lanelet).
    """
    road1 = xb.straight_road(id="1", length=50.0, successor=xb.LinkSpec(element_type="road", element_id="2", contact_point="end"))
    road2 = xb.straight_road(
        id="2", length=50.0, x=100.0 + gap, hdg=math.pi, successor=xb.LinkSpec(element_type="road", element_id="1", contact_point="end")
    )
    for lane in road1.lane_sections[0].lanes + road2.lane_sections[0].lanes:
        lane.successor = -lane.id
    xb.write_xodr(map_dir=map_dir, name="gap", text=xb.build_xodr(roads=[road1, road2]))


@pytest.mark.parametrize("autofix, applied, shared", [("true", True, True), ("false", False, False)])
def test_linked_lanes_a_little_apart_share_their_seam_only_with_autofix(tmp_path, autofix, applied, shared):
    import map_quality as mq

    _write_roads_with_gap(map_dir=tmp_path, gap=0.5)
    result = xb.run_convert_map(map_dir=str(tmp_path), **{"autofix.enabled": autofix})
    assert result.returncode == 0, result.stderr
    findings = [r for r in _report(map_dir=tmp_path, name="gap")["records"] if r["kind"] == "heuristic" and r["element"] == "H10"]
    assert findings and all(r["applied"] is applied for r in findings)
    assert all(r["location"][0] == pytest.approx(50.25, abs=0.01) for r in findings)
    outputs = mq.ConversionOutputs.load(map_dir=str(tmp_path))
    assert (mq.routing_edge_pct(outputs=outputs) == 100.0) is shared


def test_linked_lanes_that_meet_need_no_seam_heuristic(tmp_path):
    _write_roads_with_gap(map_dir=tmp_path, gap=0.0)
    result = xb.run_convert_map(map_dir=str(tmp_path), **{"autofix.enabled": "false"})
    assert result.returncode == 0, result.stderr
    assert not [r for r in _report(map_dir=tmp_path, name="gap")["records"] if r["element"] == "H10"]
