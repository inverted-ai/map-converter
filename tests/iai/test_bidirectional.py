"""Bidirectional lanes: kept, tagged ``one_way=no``, and never linked across a gap (subprocess)."""

import math

import map_quality as mq
from tests.iai import xodr_builder as xb

WEST = math.pi


def _single_lane_road_map() -> str:
    """Road 1 (x 0 -> 50) into a one-lane bidirectional road 2, then road 3 (x 100 -> 150).

    Roads 2 and 3 are drawn westwards, so road 3's westbound lane drives head-on into the
    bidirectional lanelet's end: the case the OpenDRIVE conversion links the wrong way round.
    """
    road1 = xb.RoadSpec(
        id="1",
        length=50.0,
        successor=xb.LinkSpec(element_type="road", element_id="2", contact_point="end"),
        lane_sections=[xb.LaneSectionSpec(s=0.0, lanes=[xb.LaneSpec(id=1, successor=-1), xb.LaneSpec(id=-1, successor=1)])],
    )
    road2 = xb.RoadSpec(
        id="2",
        length=50.0,
        x=100.0,
        hdg=WEST,
        predecessor=xb.LinkSpec(element_type="road", element_id="3", contact_point="end"),
        successor=xb.LinkSpec(element_type="road", element_id="1", contact_point="end"),
        lane_sections=[xb.LaneSectionSpec(s=0.0, lanes=[xb.LaneSpec(id=1, type="bidirectional", predecessor=-1, successor=-1)])],
    )
    road3 = xb.RoadSpec(
        id="3",
        length=50.0,
        x=150.0,
        hdg=WEST,
        successor=xb.LinkSpec(element_type="road", element_id="2", contact_point="start"),
        lane_sections=[xb.LaneSectionSpec(s=0.0, lanes=[xb.LaneSpec(id=1), xb.LaneSpec(id=-1, successor=1)])],
    )
    return xb.build_xodr(roads=[road1, road2, road3])


def test_bidirectional_lane_is_kept_tagged_and_not_linked_head_on(tmp_path):
    xb.write_xodr(map_dir=tmp_path, name="bidir", text=_single_lane_road_map())
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    outputs = mq.ConversionOutputs.load(map_dir=str(tmp_path))
    lanelets = list(outputs.lanelet_map.laneletLayer)

    assert len(lanelets) == 5
    assert outputs.osm_tags.one_way_no == 1
    for ll in lanelets:  # a link across the gap would collapse a boundary onto the far lanelet's end
        for bound in (ll.leftBound, ll.rightBound):
            assert math.dist((bound[0].x, bound[0].y), (bound[-1].x, bound[-1].y)) > 40.0
    assert mq.crossed_geometry(outputs=outputs) == {"reversed_bounds": 0, "invalid_polygons": 0}
    graph = mq.vehicle_routing_graph(lanelet_map=outputs.lanelet_map)
    bidirectional = next(ll for ll in lanelets if "one_way" in ll.attributes and ll.attributes["one_way"] == "no")
    assert any(graph.following(ll) == [bidirectional] for ll in lanelets)
