"""Inputs the converter cannot use as written degrade, are recorded, and the conversion continues."""

import json
import logging
from pathlib import Path

import pytest

import conversion_report
from tests.iai import xodr_builder as xb
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.parser import parse_opendrive


def _road_text(**road_kwargs) -> str:
    return xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0, **road_kwargs)])


def _replace_first(*, text: str, old: str, new: str) -> str:
    assert old in text, old
    return text.replace(old, new, 1)


def _object(**attrs: str) -> str:
    base = {"id": "7", "type": "pole", "s": "5.0", "t": "-6.0", "zOffset": "0.0", "orientation": "none", "hdg": "0.0"}
    base.update(attrs)
    return f"<objects><object {xb._attrs(**base)}/></objects>"


def _junction_text(*, contact_point: str) -> str:
    roads, junctions, controllers = xb.junction_map(with_signals=False)
    junctions[0].connections[1].contact_point = contact_point
    return xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers)


EXTRA_GEOMETRY = (
    '<geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="30.0"><line/></geometry>'
    '<geometry s="30.0" x="30.0" y="0.0" hdg="0.0" length="5.0"><clothoid/></geometry>'
)
LANE_SPEED = '<speed sOffset="0.0" max="10.0"/></lane>'

# (case id, xodr text, expected (kind, element) records)
CASES = [
    ("unknown_object_type", _replace_first(text=_road_text(), old="<objects/>", new=_object(type="weirdThing")),
     [("unknown_value", "object.type")]),
    ("stopline_object", _replace_first(text=_road_text(), old="<objects/>", new=_object(type="stopline")),
     [("translated", "object.type")]),
    ("unknown_object_orientation", _replace_first(text=_road_text(), old="<objects/>", new=_object(orientation="sideways")),
     [("unknown_value", "object.orientation")]),
    ("unknown_lane_type", xb.build_xodr(roads=[xb.RoadSpec(id="1", length=30.0, lane_sections=[
        xb.LaneSectionSpec(s=0.0, lanes=[xb.LaneSpec(id=1), xb.LaneSpec(id=-1, type="fastLane")])])]),
     [("unknown_value", "lane.type")]),
    ("unknown_country", _road_text(signals=[xb.SignalSpec(id="5", s=10.0, country="GBR", dynamic="no", type="206")]),
     [("unknown_value", "signal.country")]),
    ("signal_without_country", _road_text(signals=[xb.SignalSpec(id="5", s=10.0, country=None)]), []),
    ("degenerate_spiral",_replace_first(
        text=_road_text(), old="<line/>", new='<spiral curvStart="0.01" curvEnd="0.01"/>'),
     [("degenerate_geometry", "geometry.spiral")]),
    ("unknown_geometry_child", _replace_first(
        text=xb.build_xodr(roads=[xb.straight_road(id="1", length=30.0), xb.straight_road(id="2", length=30.0, y=50.0)]),
        old='<geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="30.0"><line/></geometry>',
        new=EXTRA_GEOMETRY),
     [("unknown_value", "road")]),
    ("signal_missing_dynamic_undefined_orientation",
     _road_text(signals=[xb.SignalSpec(id="5", s=10.0, dynamic=None, orientation="undefined")]),
     [("missing_attribute", "signal.dynamic"), ("unknown_value", "signal.orientation")]),
    ("connection_contact_point_none", _junction_text(contact_point="none"),
     [("unknown_value", "connection.contactPoint")]),
    ("speed_without_unit", _replace_first(text=_road_text(), old="</lane>", new=LANE_SPEED),
     [("missing_attribute", "speed.unit")]),
    ("decimal_comma", _road_text().replace('length="30.0"', 'length="30,0"'),
     [("decimal_comma", "geometry.length"), ("decimal_comma", "road.length")]),
    ("unparseable_width", _replace_first(text=_road_text(), old='a="3.5"', new='a="wide"'),
     [("unparseable_number", "width")]),
    ("unparseable_elevation", _road_text(elevation=[(0.0, 1.0, 0.0, 0.0, 0.0)]).replace('a="1.0"', 'a="high"'),
     [("unparseable_number", "elevation")]),
    ("unmapped_road_mark", xb.build_xodr(roads=[xb.RoadSpec(id="1", length=30.0, lane_sections=[
        xb.LaneSectionSpec(s=0.0, lanes=xb.standard_lanes(n_left=1, n_right=1, road_mark="custom"))])]),
     [("unknown_value", "roadMark.type")]),
    ("unknown_link_element_type", _replace_first(
        text=_road_text(), old="<link></link>", new='<link><predecessor elementType="lane" elementId="2"/></link>'),
     [("unknown_value", "link.elementType")]),
    ("unknown_link_contact_point", _replace_first(
        text=_road_text(), old="<link></link>",
        new='<link><successor elementType="road" elementId="2" contactPoint="middle"/></link>'),
     [("unknown_value", "link.contactPoint")]),
    ("link_without_element_type", _replace_first(
        text=_road_text(), old="<link></link>", new='<link><predecessor elementId="2"/></link>'),
     [("missing_attribute", "link.elementType")]),
    ("unknown_neighbor_side_and_direction", _replace_first(
        text=_road_text(), old="<link></link>",
        new='<link><neighbor side="up" elementId="2" direction="same"/>'
            '<neighbor side="left" elementId="3" direction="sideways"/></link>'),
     [("unknown_value", "neighbor.side"), ("unknown_value", "neighbor.direction")]),
    ("unknown_road_type", _replace_first(text=_road_text(), old='type="town"', new='type="suburban"'),
     [("unknown_value", "road.type")]),
    ("road_type_without_type", _replace_first(text=_road_text(), old=' type="town"', new=""),
     [("missing_attribute", "road.type")]),
    ("unknown_single_side", _replace_first(
        text=_road_text(), old='<laneSection s="0.0">', new='<laneSection s="0.0" singleSide="yes">'),
     [("unknown_value", "laneSection.singleSide")]),
    ("road_type_speed_not_a_number", _replace_first(
        text=_road_text(), old='<type s="0.0" type="town"/>',
        new='<type s="0.0" type="town"><speed max="fast" unit="km/h"/></type>'),
     [("unparseable_number", "speed")]),
    ("road_type_speed_without_unit", _replace_first(
        text=_road_text(), old='<type s="0.0" type="town"/>', new='<type s="0.0" type="town"><speed max="50"/></type>'),
     [("missing_attribute", "speed.unit")]),
    ("sign_value_not_a_number", _road_text(
        signals=[xb.SignalSpec(id="5", s=10.0, dynamic="no", type="274")]).replace('value="-1"', 'value="fast" unit="km/h"'),
     [("unparseable_number", "signal.value")]),
    ("unknown_road_mark_weight", _road_text().replace('weight="standard"', 'weight="heavy"'),
     [("unknown_value", "roadMark.weight")]),
]


@pytest.mark.parametrize("text,expected", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_degraded_input_converts_and_is_recorded(tmp_path, text, expected):
    xb.write_xodr(map_dir=tmp_path, name="case", text=text)
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "case_conversion_report.json") as f:
        report = json.load(f)
    assert report["outcome"] == "ok"
    recorded = {(r["kind"], r["element"]) for r in report["records"]}
    assert set(expected) <= recorded, recorded
    first = {}
    for record in report["records"]:
        first.setdefault((record["kind"], record["element"], str(record["value"])), record["message"])
    for message in first.values():
        assert message in result.stderr  # printed once per kind/element/value for the person running it


def test_clean_map_has_no_records(tmp_path):
    # Lanes beyond +-1 get the converter's own "unknown" inner marking, which is not an input problem.
    road = xb.straight_road(id="1", length=30.0, n_left=2, n_right=2)
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=[road]))
    assert xb.run_convert_map(map_dir=str(tmp_path), verifier_timeout_s="0").returncode == 0
    with open(tmp_path / "case_conversion_report.json") as f:
        assert json.load(f)["records"] == []


def test_report_counts_dropped_elements(tmp_path):
    text = _replace_first(text=_road_text(), old='a="3.5"', new='a="wide"')
    xb.write_xodr(map_dir=tmp_path, name="case", text=text)
    assert xb.run_convert_map(map_dir=str(tmp_path)).returncode == 0
    with open(tmp_path / "case_conversion_report.json") as f:
        report = json.load(f)
    assert report["elements"]["width"] == {"seen": 2, "converted": 1, "degraded": 0, "dropped": 1}
    assert report["elements"]["road"]["dropped"] == 0


def test_unrecognised_tokens_are_counted_with_their_locations(tmp_path):
    signals = [xb.SignalSpec(id=str(i), s=5.0 + i, country="GBR", dynamic="no", type="206") for i in (1, 2, 3)]
    xb.write_xodr(map_dir=tmp_path, name="case", text=_road_text(signals=signals))
    assert xb.run_convert_map(map_dir=str(tmp_path)).returncode == 0
    with open(tmp_path / "case_conversion_report.json") as f:
        report = json.load(f)
    [token] = [t for t in report["unrecognised_tokens"] if t["element"] == "signal.country"]
    assert (token["value"], token["count"], sorted(token["ids"])) == ("GBR", 3, [1, 2, 3])


def test_junction_zero_is_a_junction(tmp_path):
    roads, junctions, controllers = xb.junction_map(with_signals=False)
    text = xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers)
    text = text.replace('"100"', '"0"')  # the junction id, everywhere it is referenced
    path = Path(xb.write_xodr(map_dir=tmp_path, name="case", text=text))
    opendrive = parse_opendrive(path)
    connecting = next(r for r in opendrive.roads if r.id == 2)
    assert connecting.junction is not None and connecting.junction.id == 0


def test_handler_collects_records_and_detaches():
    handler = conversion_report.attach()
    logging.getLogger(conversion_report.DEGRADATION_LOGGER).warning(
        "lane type 'x'", extra={"kind": "unknown_value", "element": "lane.type", "id": 3, "value": "x", "action": "defaulted"}
    )
    conversion_report.detach(handler=handler)
    logging.getLogger(conversion_report.DEGRADATION_LOGGER).warning("after detach")
    assert [(r["kind"], r["element"], r["id"], r["value"]) for r in handler.records] == [
        ("unknown_value", "lane.type", 3, "x")
    ]
    summary = handler.summary(elements_seen={"lane": 4})
    assert summary["elements"]["lane"] == {"seen": 4, "converted": 4, "degraded": 1, "dropped": 0}
    assert summary["unrecognised_tokens"] == [{"element": "lane.type", "value": "x", "count": 1, "ids": [3]}]



def test_user_data_is_summarised_by_element_and_code(tmp_path):
    signals = [xb.SignalSpec(id=str(i), s=5.0 + i, dynamic="no", type="206") for i in (1, 2)]
    text = _road_text(signals=signals)
    block = '<userData code="vectorSignal" value="x"/></signal>'
    text = text.replace("></signal>", ">" + block, 2)
    text = _replace_first(text=text, old="</road>", new='<userData code="vectorRoad"/></road>')
    xb.write_xodr(map_dir=tmp_path, name="case", text=text)
    assert xb.run_convert_map(map_dir=str(tmp_path)).returncode == 0
    with open(tmp_path / "case_conversion_report.json") as f:
        user_data = json.load(f)["user_data"]
    assert [(u["element"], u["code"], u["count"]) for u in user_data] == [("signal", "vectorSignal", 2), ("road", "vectorRoad", 1)]
    assert sorted(user_data[0]["ids"]) == ["1", "2"]



def test_a_bad_road_link_is_skipped_and_the_road_kept(tmp_path):
    text = _replace_first(
        text=_road_text(), old="<link></link>", new='<link><predecessor elementType="lane" elementId="2"/></link>'
    )
    opendrive = parse_opendrive(Path(xb.write_xodr(map_dir=tmp_path, name="case", text=text)))
    assert [road.id for road in opendrive.roads] == [1]
    assert opendrive.roads[0].link.predecessor is None


def test_an_unknown_road_type_is_read_as_unknown_and_keeps_its_speed(tmp_path):
    text = _replace_first(
        text=_road_text(), old='<type s="0.0" type="town"/>',
        new='<type s="0.0" type="suburban"><speed max="50" unit="km/h"/></type>',
    )
    road_type = parse_opendrive(Path(xb.write_xodr(map_dir=tmp_path, name="case", text=text))).roads[0].types[0]
    assert (road_type.use_type, road_type.speed.max) == ("unknown", "50")


def test_link_and_road_type_names_are_translatable(tmp_path):
    text = _replace_first(
        text=_road_text(), old="<link></link>",
        new='<link><successor elementType="Road" elementId="2" contactPoint="End"/></link>',
    ).replace('type="town"', 'type="urban"')
    words = tmp_path / "w.yaml"
    words.write_text(
        "translations:\n  link.elementType: {Road: road}\n  link.contactPoint: {End: end}\n  road.type: {urban: town}\n"
    )
    xb.write_xodr(map_dir=tmp_path / "map", name="case", text=text)
    assert xb.run_convert_map(map_dir=str(tmp_path / "map"), vocabulary=str(words)).returncode == 0
    report = json.loads((tmp_path / "map" / "case_conversion_report.json").read_text())
    assert {(r["element"], str(r["value"])) for r in report["records"] if r["kind"] == "translated"} >= {
        ("link.elementType", "Road"), ("link.contactPoint", "End"), ("road.type", "urban")
    }
    assert not [r for r in report["records"] if r["kind"] != "translated" and r["element"].startswith(("link.", "road.type"))]


def test_a_speed_sign_without_a_value_converts_without_a_speed_limit(tmp_path):
    text = _road_text(signals=[xb.SignalSpec(id="5", s=10.0, dynamic="no", type="274")]).replace(' value="-1"', "")
    xb.write_xodr(map_dir=tmp_path, name="case", text=text)
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "case_conversion_report.json").read_text())
    assert ("missing_attribute", "signal.value") in {(r["kind"], r["element"]) for r in report["records"]}
