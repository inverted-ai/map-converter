"""Behaviour of ``convert_map.py`` on synthetic OpenDRIVE input (run in a subprocess)."""

import hashlib
import json
import os
import time

import pytest

import conversion_common
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
    assert conversion_common.git_identity(repo=str(tmp_path)) == {"commit": None, "dirty": None}


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
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
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
    _write(map_dir=tmp_path, name="nogeo", text=text)
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "has no <geoReference> - converting at latitude/longitude (0, 0)" in result.stderr
    with open(tmp_path / "metadata.json") as f:
        assert json.load(f)["lanelet_map_origin"] == [0.0, 0.0]
    with open(tmp_path / "nogeo_provenance.json") as f:
        assert json.load(f)["geo_reference"] == "absent"


def test_unusable_geo_reference_fails_unless_allowed(tmp_path):
    text = xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)], geo_reference="+proj=merc +lat_ts=0")
    _write(map_dir=tmp_path, name="badgeo", text=text)
    refused = xb.run_convert_map(map_dir=str(tmp_path))
    assert refused.returncode == conversion_common.REFUSED_EXIT_CODE
    assert "allow_zero_origin" in refused.stderr
    assert sorted(os.listdir(tmp_path)) == ["badgeo.xodr"]
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
    _write(map_dir=tmp_path, name="orient", text=text)
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "orientation 'undefined' - using 'none'" in result.stderr


def test_header_offset_applies_to_stoplines(tmp_path):
    import map_quality as mq
    from shapely.geometry import LineString, Point

    roads, junctions, controllers = xb.junction_map()
    text = xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers, offset=(120.0, -40.0, 0.0, 0.0))
    _write(map_dir=tmp_path, name="offset", text=text)
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    outputs = mq.ConversionOutputs.load(map_dir=str(tmp_path))
    centrelines = [LineString([(p.x, p.y) for p in ll.centerline]) for ll in outputs.lanelet_map.laneletLayer]
    assert outputs.stoplines
    for row in outputs.stoplines:
        assert min(c.distance(Point(row["x"], row["y"])) for c in centrelines) < 0.5


def test_false_northing_places_map_at_true_latitude(tmp_path):
    import map_quality as mq
    import lanelet2

    geo = "+proj=tmerc +lat_0=0 +lon_0=-117 +k=0.9996 +x_0=0 +y_0=-3630930 +datum=WGS84"
    _write(map_dir=tmp_path, name="sd", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0)], geo_reference=geo))
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    outputs = mq.ConversionOutputs.load(map_dir=str(tmp_path))
    x, y, _ = outputs.points[0]
    lat0, lon0 = outputs.metadata["lanelet_map_origin"]
    gps = mq.utm_projector(origin_lat=lat0, origin_lon=lon0).reverse(lanelet2.core.BasicPoint3d(x, y, 0.0))
    assert 32.5 < gps.lat < 33.1  # y_0 = -3,630,930 m puts the road ~32.8 deg north, not on the equator
