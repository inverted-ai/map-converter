"""Tests for ``map_quality.py``.

Exact metric values are checked on small Lanelet2 maps built in memory; the end-to-end test
converts a synthetic XODR in a subprocess (crdesigner config is process-global) and checks
the metrics that follow from the fixture.
"""

import collections
import os

import lanelet2
import pytest
from lanelet2.core import Lanelet, LaneletMap, LineString3d, Point3d, getId

import map_quality as mq
from tests.iai import xodr_builder as xb


def _point(*, x: float, y: float, z: float = 0.0) -> Point3d:
    return Point3d(getId(), x, y, z)


def _lanelet(*, left: list[Point3d], right: list[Point3d]) -> Lanelet:
    ll = Lanelet(getId(), LineString3d(getId(), left), LineString3d(getId(), right))
    ll.attributes["subtype"] = "road"
    return ll


def _outputs(*, lanelets: list[Lanelet], extra_points: list[Point3d] = ()) -> mq.ConversionOutputs:
    lanelet_map = LaneletMap()
    for ll in lanelets:
        lanelet_map.add(ll)
    for p in extra_points:
        lanelet_map.add(p)
    return mq.ConversionOutputs(
        map_dir="/nonexistent",
        metadata={"lanelet_map_origin": [49.0, 8.0], "center": [0.0, 0.0]},
        osm_path="/nonexistent/x.osm",
        osm_bytes=0,
        osm_tags=mq.OsmTagCounts(way_types=collections.Counter(), untyped_ways=0, one_way_no=0),
        lanelet_map=lanelet_map,
        points=mq.point_array(lanelet_map=lanelet_map),
        load_errors=[],
        stoplines=[],
        mesh=None,
        visualization_fov=800.0,
    )


def _chain() -> tuple[Lanelet, Lanelet]:
    """Two lanelets in sequence sharing their seam points (a routable handshake)."""
    l0, l1, l2 = _point(x=0, y=2), _point(x=10, y=2), _point(x=20, y=2)
    r0, r1, r2 = _point(x=0, y=0), _point(x=10, y=0), _point(x=20, y=0)
    return _lanelet(left=[l0, l1], right=[r0, r1]), _lanelet(left=[l1, l2], right=[r1, r2])


def test_routing_counts_isolated_lanelet():
    a, b = _chain()
    isolated = _lanelet(left=[_point(x=0, y=52), _point(x=10, y=52)], right=[_point(x=0, y=50), _point(x=10, y=50)])
    pct = mq.routing_edge_pct(outputs=_outputs(lanelets=[a, b, isolated]))
    assert pct == pytest.approx(200.0 / 3.0)


def test_seam_handshake_shared_vs_duplicated_points():
    a, b = _chain()
    assert mq.seam_handshake_pct(outputs=_outputs(lanelets=[a, b])) == 100.0
    # Same geometry, but the successor uses its own copies of the seam points.
    c = _lanelet(left=[_point(x=10, y=2), _point(x=20, y=2)], right=[_point(x=10, y=0), _point(x=20, y=0)])
    assert mq.seam_handshake_pct(outputs=_outputs(lanelets=[a, c])) == 0.0
    assert mq.seam_handshake_pct(outputs=_outputs(lanelets=[a])) is None


def test_z_conflicts_count_and_max():
    a, _ = _chain()
    extra = [_point(x=0, y=2, z=3.0), _point(x=10, y=0, z=0.005)]  # 3 m conflict; 5 mm is noise
    z = mq.z_conflicts(outputs=_outputs(lanelets=[a], extra_points=extra))
    assert z == {"count": 1, "max_dz": pytest.approx(3.0)}


def test_crossed_geometry_reversed_bounds():
    a, _ = _chain()
    reversed_right = _lanelet(
        left=[_point(x=0, y=12), _point(x=10, y=12)], right=[_point(x=10, y=10), _point(x=0, y=10)]
    )
    crossed = mq.crossed_geometry(outputs=_outputs(lanelets=[a, reversed_right]))
    assert crossed == {"reversed_bounds": 1, "invalid_polygons": 1}


def test_xy_hash_ignores_z_and_ids():
    a, _ = _chain()
    b = _lanelet(
        left=[_point(x=0, y=2, z=5.0), _point(x=10, y=2, z=5.0)],
        right=[_point(x=0, y=0, z=5.0), _point(x=10, y=0, z=5.0)],
    )
    assert mq.xy_geometry_hash(outputs=_outputs(lanelets=[a])) == mq.xy_geometry_hash(outputs=_outputs(lanelets=[b]))


def test_osm_tag_counts(tmp_path):
    osm = tmp_path / "tags.osm"
    osm.write_text(
        '<osm><way id="1"><tag k="type" v="traffic_sign"/></way><way id="2"/>'
        '<way id="3"><tag k="type" v="line_thin"/></way>'
        '<relation id="4"><tag k="type" v="lanelet"/><tag k="one_way" v="no"/></relation></osm>'
    )
    outputs = _outputs(lanelets=[])
    outputs.osm_tags = mq.osm_tag_counts(osm_path=str(osm))
    outputs.stoplines = [{"agent_type": "stop_sign"}, {"agent_type": "traffic_light"}]
    assert mq.untyped_way_count(outputs=outputs) == 1
    assert mq.one_way_no_count(outputs=outputs) == 1
    assert mq.sign_ways_vs_stoplines(outputs=outputs) == {
        "traffic_sign_ways": 1,
        "traffic_light_ways": 0,
        "sign_stoplines": 1,
        "light_stoplines": 1,
    }


@pytest.fixture(scope="module")
def converted_junction(tmp_path_factory) -> str:
    """A junction map with two lights plus an isolated road, converted in a subprocess."""
    map_dir = tmp_path_factory.mktemp("junction")
    roads, junctions, controllers = xb.junction_map()
    roads.append(xb.straight_road(id="9", length=30.0, x=0.0, y=200.0))
    (map_dir / "junction.xodr").write_text(xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers))
    result = xb.run_convert_map(map_dir=str(map_dir))
    assert result.returncode == 0, result.stderr
    return str(map_dir)


def test_metrics_on_converted_map(converted_junction):
    metrics, errors = mq.compute_all(outputs=mq.ConversionOutputs.load(map_dir=converted_junction))
    assert errors == {}
    assert metrics["stoplines_by_agent_type"] == {"traffic_light": 2}
    assert metrics["sign_ways_vs_stoplines"]["light_stoplines"] == 2
    assert metrics["load_error_count"] == 0
    assert metrics["origin"] == [49.0, 8.0]
    assert metrics["crossed_geometry"] == {"reversed_bounds": 0, "invalid_polygons": 0}
    assert metrics["z_conflicts"]["count"] == 0
    assert metrics["one_way_no_count"] == 0
    assert metrics["mesh_in_frame_fraction"] == 1.0
    assert metrics["lanelet_count"] >= 4
    assert metrics["routing_edge_pct"] < 100.0
    assert metrics["osm_size_bytes"] == os.path.getsize(os.path.join(converted_junction, "junction.osm"))


def test_outputs_paths_are_absolute(converted_junction):
    outputs = mq.ConversionOutputs.load(map_dir=os.path.relpath(converted_junction))
    assert os.path.isabs(outputs.map_dir)
    assert os.path.isabs(outputs.osm_path)


def test_projector_matches_lanelet2():
    projector = mq.utm_projector(origin_lat=49.0, origin_lon=8.0)
    gps = projector.reverse(lanelet2.core.BasicPoint3d(0.0, 0.0, 0.0))
    assert (gps.lat, gps.lon) == pytest.approx((49.0, 8.0))
