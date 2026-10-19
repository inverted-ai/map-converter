"""Lanelet bounds that Lanelet2 reads the way the converter writes them.

Lanelet2's loader (``geometry::align``) decides each bound's direction from its middle vertex and inverts a
bound it judges backwards. A lane edge that loops back on itself, or left/right bounds on the wrong sides,
make it turn lanes around; these tests convert synthetic maps and load the result with Lanelet2.
"""

import json
import math

import numpy as np
import pytest

import lanelet_bounds
import map_quality
from tests.iai import xodr_builder as xb

QUARTER_TURN = math.pi / 2


def _convert(*, tmp_path, roads, **config) -> map_quality.ConversionOutputs:
    map_dir = tmp_path / "case"
    xb.write_xodr(map_dir=map_dir, name="case", text=xb.build_xodr(roads=roads))
    result = xb.run_convert_map(map_dir=str(map_dir), **config)
    assert result.returncode == 0, result.stderr[-3000:]
    return map_quality.ConversionOutputs.load(map_dir=str(map_dir))


def _records(*, tmp_path) -> list[dict]:
    return json.loads((tmp_path / "case" / "case_conversion_report.json").read_text())["records"]


def _direction(*, lanelet) -> np.ndarray:
    centre = lanelet.centerline
    return np.array([centre[-1].x - centre[0].x, centre[-1].y - centre[0].y])


def _bend_road(*, n_right: int, radius: float, width: float = 3.2) -> xb.RoadSpec:
    """Straight, a right-hand quarter turn of ``radius``, straight; right lanes only (inside of the bend)."""
    arc = QUARTER_TURN * radius
    return xb.RoadSpec(
        id="1",
        length=40.0 + arc + 40.0,
        lane_sections=[
            xb.LaneSectionSpec(s=0.0, lanes=[xb.LaneSpec(id=-i, width=width) for i in range(1, n_right + 1)])
        ],
        geometries=[
            xb.GeometrySpec(length=40.0),
            xb.GeometrySpec(length=arc, curvature=-1.0 / radius),
            xb.GeometrySpec(length=40.0),
        ],
    )


# --- cut_loops ---------------------------------------------------------------------------------------------


# The inner (left) edge of a left turn: it overshoots the corner and comes back across itself, against the turn.
FOLDED_INNER_EDGE = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [8.0, -2.0, 0.0], [8.0, 5.0, 0.0]])
OUTER_EDGE_OF_LEFT_TURN = np.array([[0.0, -3.0], [11.0, -3.0], [11.0, 5.0]])


def test_a_fold_against_the_turn_is_replaced_by_the_crossing_point():
    cut, loops = lanelet_bounds.cut_loops(polyline=FOLDED_INNER_EDGE, reference=OUTER_EDGE_OF_LEFT_TURN)

    np.testing.assert_allclose(cut[:, :2], [[0.0, 0.0], [8.0, 0.0], [8.0, 5.0]])
    assert len(loops) == 1
    np.testing.assert_allclose(loops[0].location, [8.0, 0.0])
    assert loops[0].length == pytest.approx(2.0 + math.hypot(2.0, 2.0) + 2.0)


def test_cutting_a_reversed_bound_gives_the_reversed_result():
    polyline = np.vstack([FOLDED_INNER_EDGE[:, :2], [[20.0, 5.0]]])
    reference = np.vstack([OUTER_EDGE_OF_LEFT_TURN, [[20.0, 2.0]]])

    forward, _ = lanelet_bounds.cut_loops(polyline=polyline, reference=reference)
    backward, _ = lanelet_bounds.cut_loops(polyline=polyline[::-1], reference=reference[::-1])

    np.testing.assert_allclose(backward, forward[::-1])


def test_a_turnaround_loop_that_turns_with_the_lane_is_kept():
    # A cul-de-sac: the inner edge circles the island and crosses its own way in at the neck.
    inner = np.array([[0.0, 0.0], [20.0, 0.0], [30.0, 10.0], [20.0, 20.0], [10.0, 10.0], [15.0, -5.0]])
    outer = np.array([[0.0, -3.0], [22.0, -3.0], [34.0, 10.0], [20.0, 24.0], [6.0, 10.0], [12.0, -8.0]])

    cut, loops = lanelet_bounds.cut_loops(polyline=inner, reference=outer)

    np.testing.assert_array_equal(cut, inner)
    assert loops == []


def _loop_ramp(*, offset: float) -> np.ndarray:
    """A ramp heading east that turns left through 270 degrees around (40, 20) and leaves southwards across its
    own approach; ``offset`` metres to the right of the centre line."""
    radius = 20.0 + offset
    angles = np.linspace(-math.pi / 2, math.pi, 40)
    arc = np.column_stack([40.0 + radius * np.cos(angles), 20.0 + radius * np.sin(angles)])
    return np.vstack([[[-10.0, -offset]], arc, [[20.0 - offset, -30.0]]])


def test_a_loop_ramp_crossing_its_own_approach_is_kept():
    right, left = _loop_ramp(offset=1.75), _loop_ramp(offset=-1.75)

    cut, loops = lanelet_bounds.cut_loops(polyline=right, reference=left)

    np.testing.assert_array_equal(cut, right)
    assert loops == []


def test_a_spike_that_folds_straight_back_is_removed():
    polyline = np.array([[0.0, 0.0], [10.0, 0.0], [16.0, 1.0], [11.0, -1.0], [20.0, -1.0]])

    cut, loops = lanelet_bounds.cut_loops(polyline=polyline, reference=polyline + [0.0, -3.0])

    np.testing.assert_allclose(cut, [[0.0, 0.0], [10.0, 0.0], [11.0, -1.0], [20.0, -1.0]])
    assert len(loops) == 1


def test_a_polyline_without_crossings_is_unchanged():
    polyline = np.array([[0.0, 0.0], [10.0, 0.0], [12.0, 3.0], [12.0, 10.0]])

    cut, loops = lanelet_bounds.cut_loops(polyline=polyline, reference=polyline + [3.0, 0.0])

    np.testing.assert_array_equal(cut, polyline)
    assert loops == []


# --- whole conversions -------------------------------------------------------------------------------------


@pytest.mark.parametrize("autofix", ["true", "false"])
def test_a_lane_wider_than_its_bend_radius_is_read_as_written(tmp_path, autofix):
    outputs = _convert(tmp_path=tmp_path, roads=[_bend_road(n_right=1, radius=2.5)], **{"autofix.enabled": autofix})

    lanelets = list(outputs.lanelet_map.laneletLayer)
    assert lanelets
    for ll in lanelets:
        assert not ll.leftBound.inverted() and not ll.rightBound.inverted()
    assert map_quality.crossed_geometry(outputs=outputs)["reversed_bounds"] == 0
    cuts = [r for r in _records(tmp_path=tmp_path) if r["kind"] == "geometry" and r["element"] == "bound_loop_cut"]
    assert cuts and all(r["location"] is not None for r in cuts)


def test_lanes_through_a_tight_bend_still_share_their_common_bound(tmp_path):
    outputs = _convert(tmp_path=tmp_path, roads=[_bend_road(n_right=2, radius=4.0)])

    by_offset = sorted(outputs.lanelet_map.laneletLayer, key=lambda ll: -ll.centerline[0].y)
    inner, outer = by_offset[0], by_offset[-1]
    assert inner.rightBound.id == outer.leftBound.id


def test_a_kink_sharper_than_a_right_angle_is_recorded(tmp_path):
    road = xb.RoadSpec(
        id="1",
        length=60.0,
        lane_sections=[xb.LaneSectionSpec(s=0.0, lanes=xb.standard_lanes(n_left=1, n_right=1))],
        geometries=[xb.GeometrySpec(length=30.0), xb.GeometrySpec(length=30.0, heading_jump=math.radians(120))],
    )
    _convert(tmp_path=tmp_path, roads=[road])

    kinks = [r for r in _records(tmp_path=tmp_path) if r["kind"] == "geometry" and r["element"] == "sharp_kink"]
    assert len(kinks) == 1
    assert kinks[0]["id"] == "1"
    np.testing.assert_allclose(kinks[0]["location"], [30.0, 0.0], atol=0.01)


@pytest.mark.parametrize("rule, along_s_side", [("RHT", -1), ("LHT", 1)])
def test_lanes_drive_on_the_traffic_side_after_lanelet2_loads_them(tmp_path, rule, along_s_side):
    outputs = _convert(tmp_path=tmp_path, roads=[xb.straight_road(id="1", length=100.0, rule=rule)])

    lanelets = list(outputs.lanelet_map.laneletLayer)
    assert len(lanelets) == 2
    for ll in lanelets:
        side = 1 if ll.centerline[0].y > 0 else -1
        heading_x = _direction(lanelet=ll)[0]
        assert (heading_x > 0) == (side == along_s_side), f"{rule}: lane at side {side} runs {heading_x:+.0f} m in x"
        left_offset = np.mean([(p.x, p.y) for p in ll.leftBound], axis=0) - np.mean(
            [(p.x, p.y) for p in ll.rightBound], axis=0
        )
        assert np.cross(_direction(lanelet=ll), left_offset) > 0, "left bound must lie left of the direction"


# --- road mesh ---------------------------------------------------------------------------------------------


def _bend_lanelet_map(*, inner_points: int, outer_points: int):
    """One lanelet around a 90-degree left bend (inner radius 10 m, outer 13.5 m), its bounds sampled unevenly."""
    from lanelet2.core import Lanelet, LaneletMap, LineString3d, Point3d, getId

    def arc(*, radius: float, n: int) -> LineString3d:
        angles = np.linspace(-math.pi / 2, 0.0, n) ** 3 / (math.pi / 2) ** 2 * -1  # bunched near the start
        return LineString3d(getId(), [Point3d(getId(), radius * math.cos(a), 20.0 + radius * math.sin(a), 0.0) for a in angles])

    lanelet_map = LaneletMap()
    lanelet_map.add(Lanelet(getId(), arc(radius=10.0, n=inner_points), arc(radius=13.5, n=outer_points)))
    return lanelet_map


def test_the_road_mesh_covers_each_lanelet_and_nothing_else():
    from shapely.geometry import Polygon

    lanelet_map = _bend_lanelet_map(inner_points=8, outer_points=40)
    ll = next(iter(lanelet_map.laneletLayer))
    outline = Polygon([(p.x, p.y) for p in ll.leftBound] + [(p.x, p.y) for p in reversed(list(ll.rightBound))])

    mesh = lanelet_bounds.road_mesh(lanelet_map=lanelet_map)

    triangles = mesh.verts[0][mesh.faces[0]].numpy()
    areas = [Polygon(t).area for t in triangles]
    outside = sum(Polygon(t).difference(outline.buffer(1e-6)).area for t in triangles)
    assert sum(areas) == pytest.approx(outline.area, rel=1e-3)
    assert outside < 1e-3
    assert mesh.faces.shape[1] == len(ll.leftBound) + len(ll.rightBound) - 2


# --- bounds Lanelet2's align reads as written ---------------------------------------------------------------


def _loaded_inversions(*, tmp_path, left: np.ndarray, right: np.ndarray) -> tuple[bool, bool]:
    """Write one lanelet with these bounds to an .osm, load it with Lanelet2; which bounds came back inverted."""
    import lanelet2
    from lanelet2.core import Lanelet, LaneletMap, LineString3d, Point3d, getId
    from lanelet2.io import Origin

    def line(points: np.ndarray) -> LineString3d:
        return LineString3d(getId(), [Point3d(getId(), float(x), float(y), 0.0) for x, y in points[:, :2]])

    projector = lanelet2.projection.UtmProjector(Origin(49.0, 8.0))
    lanelet_map = LaneletMap()
    lanelet_map.add(Lanelet(getId(), line(left), line(right)))
    path = str(tmp_path / "one.osm")
    lanelet2.io.write(path, lanelet_map, projector)
    loaded = next(iter(lanelet2.io.load(path, projector).laneletLayer))
    return loaded.leftBound.inverted(), loaded.rightBound.inverted()


# A V-shaped connector: down-right, a tight apex with a tangle on the right bound, back down-left.
V_LEFT = np.array(
    [[-99.2, 133.3], [-82.2, 117.9], [-62.6, 100.7], [-40.5, 84.1], [-35.1, 77.1], [13.6, 36.7], [15.1, 34.6],
     [14.5, 31.8], [15.4, 30.7], [17.8, 31.3], [18.3, 28.3], [16.5, 21.8], [13.8, 17.7], [-14.8, -16.2], [-47.4, -56.1]]
)
V_RIGHT = np.array(
    [[-101.3, 130.9], [-84.3, 115.5], [-64.8, 98.4], [-42.5, 81.6], [-38.5, 75.8], [10.5, 35.1], [12.0, 33.8],
     [12.8, 32.4], [18.3, 35.8], [12.1, 35.7], [11.4, 30.8], [13.7, 28.0], [14.4, 24.8], [11.9, 20.6],
     [-11.1, -7.2], [-49.9, -54.0]]
)


def test_lanelet2_misreads_the_v_shaped_connector_as_written(tmp_path):
    # The fixture's premise: unchanged, Lanelet2's loader turns a bound around.
    assert any(_loaded_inversions(tmp_path=tmp_path, left=V_LEFT, right=V_RIGHT))
    assert lanelet_bounds.misjudged_by_align(left=V_LEFT, right=V_RIGHT)


def test_align_safe_bounds_are_read_as_written_with_the_same_shape(tmp_path):
    from shapely.geometry import LineString

    left, right = lanelet_bounds.align_safe(left=V_LEFT, right=V_RIGHT)

    assert _loaded_inversions(tmp_path=tmp_path, left=left, right=right) == (False, False)
    assert not lanelet_bounds.misjudged_by_align(left=left, right=right)
    for before, after in ((V_LEFT, left), (V_RIGHT, right)):
        assert LineString(after).hausdorff_distance(LineString(before)) < 1e-9
        assert len(after) >= len(before)


def test_a_bound_whose_only_readable_vertices_are_its_ends_gets_a_readable_middle(tmp_path):
    # A connector turning right: the right bound folds back near the corner, so the left bound's only inner vertex
    # reads on the wrong side; only its ends read right, and an end can never be the middle vertex.
    left = np.array([[-27.6, -18.9], [-22.3, -7.0], [19.6, -7.0]])
    right = np.array([[-24.7, -20.2], [-19.4, -8.3], [-22.3, -10.2], [19.4, -10.1]])
    assert _loaded_inversions(tmp_path=tmp_path, left=left, right=right) == (False, True)

    new_left, new_right = lanelet_bounds.align_safe(left=left, right=right)

    assert _loaded_inversions(tmp_path=tmp_path, left=new_left, right=new_right) == (False, False)


def test_bounds_align_already_reads_are_left_alone():
    left, right = np.array([[0.0, 3.0], [10.0, 3.0], [20.0, 3.0]]), np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])

    assert lanelet_bounds.align_safe(left=left, right=right) == (left, right)
