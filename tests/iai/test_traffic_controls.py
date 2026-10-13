"""Stopline rows for traffic lights and stop/yield signs, and the heuristics that place them.

Every map here is straight along +x from the origin, so a lane's row centre is at (s, -1.75) for
lane -1 and (s, +1.75) for lane 1 (3.5 m lanes), heading +x and -x respectively.
"""

import json
import math

import pytest

import conversion_common
import conversion_heuristics as ch
from tests.iai import xodr_builder as xb

LANE_RIGHT_Y = -1.75
H6_ON = {"autofix.heuristics.sign_signal_stoplines.enabled": "true"}
# A plain road links to no junction; these tests exercise the rows, not the junction-distance limit.
H6_ON_ANY_DISTANCE = {**H6_ON, "autofix.heuristics.sign_signal_stoplines.max_junction_distance_m": "null"}
LANE_LEFT_Y = 1.75


def _convert(*, tmp_path, roads, junctions=(), controllers=(), **config: str) -> dict:
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=roads, junctions=junctions, controllers=controllers))
    result = xb.run_convert_map(map_dir=str(tmp_path), **config)
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "case_stoplines.json") as f:
        rows = json.load(f)
    with open(tmp_path / "case_conversion_report.json") as f:
        report = json.load(f)
    return {
        "rows": rows,
        "controls": report["validation"]["controls"],
        "heuristics": report["heuristics"],
        "not_applied": report["heuristics_not_applied"],
        "report": report,
    }


def _rows(*, out: dict, agent_type: str = "traffic_light") -> list[tuple[str, float, float, float]]:
    """(actor id, x, y, heading) of each row of ``agent_type``, rounded to the millimetre."""
    return sorted(
        (str(r["actor_id"]), round(r["x"], 3), round(r["y"], 3), round(r["orientation"], 3))
        for r in out["rows"]
        if r["agent_type"] == agent_type
    )


def _plain_road(*, signals=(), n_left: int = 1, n_right: int = 1, **kwargs) -> xb.RoadSpec:
    return xb.straight_road(id="1", length=50.0, n_left=n_left, n_right=n_right, signals=list(signals), **kwargs)


def _heuristic_ids(*, out: dict, heuristic_id: str) -> list[str]:
    return [str(i) for i in out["heuristics"].get(heuristic_id, {}).get("ids", [])]


def test_junction_road_lights_stop_where_their_lane_enters(tmp_path):
    # The connecting-road placement every CARLA map uses: unchanged by any heuristic.
    roads, junctions, controllers = xb.junction_map()
    out = _convert(tmp_path=tmp_path, roads=roads, junctions=junctions, controllers=controllers)
    assert _rows(out=out) == [("11", 50.0, LANE_RIGHT_Y, 0.0), ("12", 70.0, LANE_LEFT_Y, round(-math.pi, 3))]
    assert out["controls"]["lights_without_stopline"] == []


def test_light_on_a_plain_road_stops_at_its_own_s(tmp_path):
    light = xb.SignalSpec(id="21", s=40.0, validity=(-1, -1))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light])])
    assert _rows(out=out) == [("21", 40.0, LANE_RIGHT_Y, 0.0)]
    assert _heuristic_ids(out=out, heuristic_id="H4") == ["21"]


def test_plain_road_light_stops_at_the_lane_start_without_the_heuristic(tmp_path):
    light = xb.SignalSpec(id="21", s=40.0, validity=(-1, -1))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light])], **{"autofix.heuristics.stop_at_signal_s.enabled": "false"})
    assert _rows(out=out) == [("21", 0.0, LANE_RIGHT_Y, 0.0)]
    assert "H4" not in out["heuristics"]


def test_plain_road_light_in_a_later_lane_section_stops_at_its_s(tmp_path):
    road = _plain_road(signals=[xb.SignalSpec(id="21", s=40.0, validity=(-1, -1))])
    road.lane_sections.append(xb.LaneSectionSpec(s=25.0, lanes=xb.standard_lanes(n_left=1, n_right=1)))
    out = _convert(tmp_path=tmp_path, roads=[road])
    assert _rows(out=out) == [("21", 40.0, LANE_RIGHT_Y, 0.0)]


def test_centre_line_validity_binds_the_lanes_the_light_faces(tmp_path):
    light = xb.SignalSpec(id="31", s=40.0, orientation="-", validity=(0, 0))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light])])
    assert _rows(out=out) == [("31", 40.0, LANE_LEFT_Y, round(-math.pi, 3))]
    assert _heuristic_ids(out=out, heuristic_id="H2") == ["31"]
    assert _heuristic_ids(out=out, heuristic_id="H3") == ["31"]


def test_centre_line_validity_binds_nothing_without_the_heuristic(tmp_path):
    light = xb.SignalSpec(id="31", s=40.0, orientation="-", validity=(0, 0))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light])], **{"autofix.heuristics.centre_line_validity.enabled": "false"})
    assert _rows(out=out) == []
    assert out["controls"]["lights_without_stopline"] == ["31"]


def test_light_without_validity_binds_every_lane_it_faces(tmp_path):
    light = xb.SignalSpec(id="41", s=40.0, orientation="+")
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light], n_right=2)])
    assert _rows(out=out) == [("41", 40.0, -5.25, 0.0), ("41", 40.0, LANE_RIGHT_Y, 0.0)]
    assert _heuristic_ids(out=out, heuristic_id="H3") == ["41"]


def test_light_without_validity_binds_nothing_without_the_heuristic(tmp_path):
    light = xb.SignalSpec(id="41", s=40.0, orientation="+")
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light])], **{"autofix.heuristics.facing_lanes.enabled": "false"})
    assert _rows(out=out) == []
    assert out["controls"]["lights_without_stopline"] == ["41"]


def test_light_bound_by_validity_never_falls_back(tmp_path):
    # Facing lanes apply only to a light no lane names; here lane -1 is named, lane 1 is not.
    light = xb.SignalSpec(id="42", s=40.0, orientation="none", validity=(-1, -1))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light])])
    assert [row[:3] for row in _rows(out=out)] == [("42", 40.0, LANE_RIGHT_Y)]
    assert "H3" not in out["heuristics"]


def test_reference_without_validity_binds_the_facing_lanes_of_its_road(tmp_path):
    roads, junctions, _ = xb.junction_map(with_signals=False)
    roads[0].signals.append(xb.SignalSpec(id="51", s=45.0, t=5.0, orientation="+"))
    roads[1].signal_references.append(xb.SignalReferenceSpec(id="51", s=0.0, orientation="+"))
    out = _convert(tmp_path=tmp_path, roads=roads, junctions=junctions)
    # The reference's junction road, at its lane's entry; not the signal's own road.
    assert _rows(out=out) == [("51", 50.0, LANE_RIGHT_Y, 0.0)]
    assert _heuristic_ids(out=out, heuristic_id="H3") == ["51"]


def test_light_on_a_road_without_drivable_lanes_is_not_a_vehicle_light(tmp_path):
    walkway = xb.RoadSpec(
        id="5",
        length=20.0,
        y=30.0,
        lane_sections=[xb.LaneSectionSpec(s=0.0, lanes=[xb.LaneSpec(id=1, type="sidewalk"), xb.LaneSpec(id=-1, type="sidewalk")])],
        signals=[xb.SignalSpec(id="61", s=10.0, orientation="+", validity=(0, 0))],
    )
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(), walkway])
    assert (out["controls"]["lights"], out["controls"]["lights_without_stopline"]) == (0, [])
    assert _heuristic_ids(out=out, heuristic_id="H7") == ["61"]


def test_light_facing_away_from_every_lane_stays_without_a_stopline(tmp_path):
    light = xb.SignalSpec(id="62", s=40.0, orientation="-")
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light], n_left=0)])
    assert _rows(out=out) == []
    assert out["controls"]["lights_without_stopline"] == ["62"]
    assert "H7" not in out["heuristics"]


def test_dynamic_signal_without_a_light_type_is_a_sign(tmp_path):
    stop = xb.SignalSpec(id="71", s=40.0, type="r1", subtype="1", country="", validity=(-1, -1))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[stop])])
    assert out["controls"]["lights"] == 0
    assert _heuristic_ids(out=out, heuristic_id="H1") == ["71"]
    # Read with the US sign table it is a STOP sign (no stopline: its lane reaches no junction within H6's 30 m).
    assert _heuristic_ids(out=out, heuristic_id="H8") == ["71"]
    assert (out["controls"]["stop_yield_signs"], out["controls"]["signs_without_stopline"]) == (1, ["signal 71"])


def test_dynamic_signal_without_a_light_type_stays_a_light_without_the_heuristic(tmp_path):
    stop = xb.SignalSpec(id="71", s=40.0, type="r1", subtype="1", country="", validity=(-1, -1))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[stop])], **{"autofix.heuristics.light_type_code.enabled": "false"})
    assert out["controls"]["lights"] == 1
    assert "H1" not in out["heuristics"]


def test_us_codes_other_than_stop_and_yield_are_read_as_before(tmp_path):
    # MUTCD R2-1 is a speed limit; read with the US table its value -1 became a -1 m/s limit.
    speed = xb.SignalSpec(id="72", s=40.0, dynamic="no", type="r2", subtype="1", country="", validity=(-1, -1))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[speed])])
    assert "H8" not in out["heuristics"]
    with open(tmp_path / "case.osm") as f:
        assert 'v="-1.0"' not in f.read()


def test_sign_signals_write_no_stopline_by_default(tmp_path):
    stop = xb.SignalSpec(id="81", s=40.0, dynamic="no", type="206", validity=(-1, -1))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[stop])])
    assert _rows(out=out, agent_type="stop_sign") == []
    assert out["controls"]["signs_without_stopline"] == ["signal 81"]


def test_sign_signal_stoplines_on_request(tmp_path):
    stop = xb.SignalSpec(id="81", s=40.0, dynamic="no", type="206", validity=(-1, -1))
    give_way = xb.SignalSpec(id="82", s=20.0, dynamic="no", type="205", orientation="-")
    out = _convert(
        tmp_path=tmp_path, roads=[_plain_road(signals=[stop, give_way])], **H6_ON_ANY_DISTANCE
    )
    assert _rows(out=out, agent_type="stop_sign") == [("81", 40.0, LANE_RIGHT_Y, 0.0)]
    assert _rows(out=out, agent_type="yield_sign") == [("82", 20.0, LANE_LEFT_Y, round(-math.pi, 3))]
    assert out["controls"]["signs_without_stopline"] == []
    assert sorted(_heuristic_ids(out=out, heuristic_id="H6")) == ["81", "82"]


def test_sign_next_to_its_stencil_writes_only_the_stencil_row(tmp_path):
    stop = xb.SignalSpec(id="81", s=40.0, dynamic="no", type="206", validity=(-1, -1))
    road = _plain_road(signals=[stop], objects=[xb.ObjectSpec(id="9", s=38.0)])
    out = _convert(tmp_path=tmp_path, roads=[road], **H6_ON_ANY_DISTANCE)
    assert [row[0] for row in _rows(out=out, agent_type="stop_sign")] == ["9"]
    assert out["controls"]["signs_without_stopline"] == []
    assert _heuristic_ids(out=out, heuristic_id="H5") == ["81"]


def test_road_without_lane_sections_does_not_stop_the_controls(tmp_path):
    empty = xb.RoadSpec(id="9", length=0.0, y=40.0, lane_sections=[])
    light = xb.SignalSpec(id="21", s=40.0, validity=(-1, -1))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light]), empty])
    assert _rows(out=out) == [("21", 40.0, LANE_RIGHT_Y, 0.0)]


def test_sign_row_next_to_a_stencil_row_on_its_lane_is_not_written(tmp_path):
    # The sign stands 12 m from the stencil (beyond H5), but its row would land 2 m from the stencil's row.
    stop = xb.SignalSpec(id="81", s=40.0, t=-12.0, dynamic="no", type="206", validity=(-1, -1))
    road = _plain_road(signals=[stop], objects=[xb.ObjectSpec(id="9", s=38.0)])
    out = _convert(tmp_path=tmp_path, roads=[road], **H6_ON_ANY_DISTANCE)
    assert [row[0] for row in _rows(out=out, agent_type="stop_sign")] == ["9"]
    assert out["controls"]["signs_without_stopline"] == []
    assert _heuristic_ids(out=out, heuristic_id="H5") == ["81"]


def test_sign_referenced_twice_on_one_lane_writes_one_row(tmp_path):
    roads, junctions, _ = xb.junction_map(with_signals=False)
    roads[0].signals.append(xb.SignalSpec(id="83", s=45.0, t=-5.0, dynamic="no", type="206", validity=(0, 0)))
    roads[1].signal_references += [
        xb.SignalReferenceSpec(id="83", s=0.0, validity=(-1, -1)),
        xb.SignalReferenceSpec(id="83", s=0.0, validity=(-1, -1)),
    ]
    out = _convert(tmp_path=tmp_path, roads=roads, junctions=junctions, **H6_ON)
    assert _rows(out=out, agent_type="stop_sign") == [("83", 50.0, LANE_RIGHT_Y, 0.0)]


def test_sign_reference_lanes_are_read_in_the_lane_section_at_its_s(tmp_path):
    # Lane -2 is a shoulder in the first section and the driving lane in the second (fed by lane -1).
    roads, junctions, _ = xb.junction_map(with_signals=False)
    road2 = roads[1]
    road2.lane_sections = [
        xb.LaneSectionSpec(s=0.0, lanes=[xb.LaneSpec(id=1, predecessor=1, successor=1), xb.LaneSpec(id=-1, predecessor=-1, successor=-2),
                                         xb.LaneSpec(id=-2, type="shoulder", width=0.5)]),
        xb.LaneSectionSpec(s=10.0, lanes=[xb.LaneSpec(id=1, predecessor=1, successor=1), xb.LaneSpec(id=-1, type="shoulder", width=0.5),
                                          xb.LaneSpec(id=-2, predecessor=-1, successor=-1)]),
    ]
    roads[0].signals.append(xb.SignalSpec(id="84", s=45.0, t=-5.0, dynamic="no", type="206", validity=(0, 0)))
    road2.signal_references.append(xb.SignalReferenceSpec(id="84", s=15.0, validity=(-2, -2)))
    out = _convert(tmp_path=tmp_path, roads=roads, junctions=junctions, **H6_ON)
    # Where driving lane -1 enters the connecting road, not on the first section's shoulder.
    assert _rows(out=out, agent_type="stop_sign") == [("84", 50.0, LANE_RIGHT_Y, 0.0)]


def test_two_signs_stopping_the_same_lane_write_one_row(tmp_path):
    stops = [
        xb.SignalSpec(id="85", s=40.0, t=-5.0, dynamic="no", type="206", validity=(-1, -1)),
        xb.SignalSpec(id="86", s=40.0, t=5.0, dynamic="no", type="206", orientation="+"),
    ]
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=stops)], **H6_ON_ANY_DISTANCE)
    assert _rows(out=out, agent_type="stop_sign") == [("85", 40.0, LANE_RIGHT_Y, 0.0)]
    assert out["controls"]["signs_without_stopline"] == []



def test_sign_farther_from_a_junction_than_the_limit_writes_no_row(tmp_path):
    # A plain road links to no junction: its lanes never reach one (default limit 30 m).
    stop = xb.SignalSpec(id="81", s=40.0, dynamic="no", type="206", validity=(-1, -1))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[stop])], **H6_ON)
    assert _rows(out=out, agent_type="stop_sign") == []
    assert out["controls"]["signs_without_stopline"] == ["signal 81"]
    (finding,) = _findings(out=out, heuristic_id="H6")
    assert "no junction" in finding["message"]


def test_sign_within_the_limit_of_a_junction_writes_its_row(tmp_path):
    roads, junctions, _ = xb.junction_map(with_signals=False)
    roads[0].signals.append(xb.SignalSpec(id="87", s=45.0, dynamic="no", type="206", validity=(-1, -1)))
    limit = {**H6_ON, "autofix.heuristics.sign_signal_stoplines.max_junction_distance_m": "30"}
    out = _convert(tmp_path=tmp_path, roads=roads, junctions=junctions, **limit)
    assert _rows(out=out, agent_type="stop_sign") == [("87", 45.0, LANE_RIGHT_Y, 0.0)]
    (finding,) = _findings(out=out, heuristic_id="H6")
    assert "5.0 m before a junction" in finding["message"]


def test_stencil_check_along_the_lane_has_its_own_distance(tmp_path):
    # As in test_sign_row_next_to_a_stencil_row_on_its_lane_is_not_written, but the lane check reaches only 1 m.
    stop = xb.SignalSpec(id="81", s=40.0, t=-12.0, dynamic="no", type="206", validity=(-1, -1))
    road = _plain_road(signals=[stop], objects=[xb.ObjectSpec(id="9", s=38.0)])
    narrow = {**H6_ON_ANY_DISTANCE, "autofix.heuristics.sign_signal_stoplines.stencil_lane_distance_m": "1"}
    out = _convert(tmp_path=tmp_path, roads=[road], **narrow)
    assert sorted(row[0] for row in _rows(out=out, agent_type="stop_sign")) == ["81", "9"]


def test_painted_stop_lines_are_reported_and_write_no_row(tmp_path):
    road = _plain_road(objects=[xb.ObjectSpec(id="91", s=30.0, name="StopLine", type="-1", width=0.0, length=3.5)])
    out = _convert(tmp_path=tmp_path, roads=[road])
    assert out["rows"] == []
    assert [r["id"] for r in out["report"]["records"] if r["kind"] == "not_converted"] == ["91"]


@pytest.mark.parametrize(
    "setting, named",
    [
        ("autofix.heuristics.no_such_rule.enabled=true", "no_such_rule"),
        ("autofix.heuristics.sign_stencil_match.max_distance_m=-1", "sign_stencil_match.max_distance_m"),
    ],
)
def test_bad_heuristics_settings_are_refused(tmp_path, setting, named):
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=[_plain_road()]))
    key, value = setting.split("=")
    result = xb.run_convert_map(map_dir=str(tmp_path), **{key: value})
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
    assert named in result.stderr


def test_autofix_off_applies_no_heuristic(tmp_path):
    light = xb.SignalSpec(id="21", s=40.0, validity=(-1, -1))
    every_switch_off = {f"autofix.heuristics.{h.name}.enabled": "false" for h in ch.HEURISTICS}
    switched_off = _convert(tmp_path=tmp_path / "switches", roads=[_plain_road(signals=[light])], **every_switch_off)
    autofix_off = _convert(tmp_path=tmp_path / "off", roads=[_plain_road(signals=[light])], **{"autofix.enabled": "false"})
    assert autofix_off["heuristics"] == {}
    assert _rows(out=autofix_off) == _rows(out=switched_off)


def test_autofix_off_leaves_the_heuristic_settings_but_disables_them():
    autofix = ch.AutoFixConfig(enabled=False)
    applied = ch.applied_heuristics(autofix=autofix)
    assert not any(getattr(applied, f.name).enabled for f in ch.dataclasses.fields(applied))
    assert applied.light_type_code.pattern == autofix.heuristics.light_type_code.pattern
    assert autofix.heuristics.light_type_code.enabled
    assert ch.applied_heuristics(autofix=ch.AutoFixConfig()) == ch.HeuristicsConfig()


def _findings(*, out: dict, heuristic_id: str) -> list[dict]:
    return [r for r in out["report"]["records"] if r["kind"] == "heuristic" and r["element"] == heuristic_id]


def test_an_applied_heuristic_records_where_it_acted(tmp_path):
    light = xb.SignalSpec(id="21", s=40.0, validity=(-1, -1))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light])])
    (finding,) = _findings(out=out, heuristic_id="H4")
    assert finding["applied"] is True
    assert finding["location"] == pytest.approx([40.0, -5.0], abs=0.01)  # the signal, at s=40, t=-5
    assert "H4" not in out["not_applied"]


def test_a_disabled_heuristic_records_what_it_would_have_changed(tmp_path):
    light = xb.SignalSpec(id="21", s=40.0, validity=(-1, -1))
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light])], **{"autofix.heuristics.stop_at_signal_s.enabled": "false"})
    (finding,) = _findings(out=out, heuristic_id="H4")
    assert finding["applied"] is False
    assert finding["location"] == pytest.approx([40.0, -5.0], abs=0.01)
    assert "H4" not in out["heuristics"]
    assert [str(i) for i in out["not_applied"]["H4"]["ids"]] == ["21"]


def test_switched_off_sign_signal_stoplines_are_reported_as_not_applied(tmp_path):
    stop = xb.SignalSpec(id="31", s=40.0, dynamic="no", type="206", validity=(-1, -1))
    out = _convert(
        tmp_path=tmp_path, roads=[_plain_road(signals=[stop])], **{"autofix.heuristics.sign_signal_stoplines.enabled": "false"}
    )
    assert [row for row in out["rows"] if row["agent_type"] == "stop_sign"] == []
    assert [str(i) for i in out["not_applied"]["H6"]["ids"]] == ["31"]


def test_facing_lanes_off_reports_the_unscoped_light(tmp_path):
    light = xb.SignalSpec(id="41", s=40.0, orientation="+")
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light])], **{"autofix.heuristics.facing_lanes.enabled": "false"})
    assert _rows(out=out) == []
    assert [str(i) for i in out["not_applied"]["H3"]["ids"]] == ["41"]


def test_every_heuristic_has_an_id_a_name_a_switch_and_a_full_docstring():
    numbers = [int(h.id.removeprefix("H")) for h in ch.HEURISTICS]
    assert numbers == sorted(set(numbers)) and all(h.id == f"H{n}" for h, n in zip(ch.HEURISTICS, numbers))
    switches = {f.name for f in ch.dataclasses.fields(ch.HeuristicsConfig)}
    for h in ch.HEURISTICS:
        assert h.name in switches, h.name
        assert all(part in h.doc for part in ("Assumes:", "Evidence:", "Breaks when:")), h.id



def test_sign_signal_stoplines_are_on_by_default():
    assert ch.HeuristicsConfig().sign_signal_stoplines.enabled


def test_unscoped_light_placed_at_its_s_records_h4(tmp_path):
    # No validity: H3 binds the lane it faces, H4 places the row at the signal's s (both recorded).
    light = xb.SignalSpec(id="43", s=40.0, orientation="+")
    out = _convert(tmp_path=tmp_path, roads=[_plain_road(signals=[light])])
    assert [row[:2] for row in _rows(out=out)] == [("43", 40.0)]
    assert _heuristic_ids(out=out, heuristic_id="H3") == ["43"]
    assert _heuristic_ids(out=out, heuristic_id="H4") == ["43"]


def test_partly_scoped_sign_records_its_centre_line_validity(tmp_path):
    roads, junctions, _ = xb.junction_map(with_signals=False)
    roads[0].signals.append(xb.SignalSpec(id="88", s=45.0, t=-5.0, dynamic="no", type="206", validity=(0, 0)))
    roads[1].signal_references.append(xb.SignalReferenceSpec(id="88", s=0.0, validity=(-1, -1)))
    out = _convert(tmp_path=tmp_path, roads=roads, junctions=junctions)
    assert _rows(out=out, agent_type="stop_sign") == [("88", 50.0, LANE_RIGHT_Y, 0.0)]
    assert _heuristic_ids(out=out, heuristic_id="H2") == ["88"]


def test_stencil_covering_some_of_a_signs_lanes_is_recorded(tmp_path):
    # The sign binds lanes -1 and -2; a stop stencil lies on lane -1 only, so lane -2 gets the sign's row.
    stop = xb.SignalSpec(id="89", s=40.0, t=-12.0, dynamic="no", type="206", validity=(-2, -1))
    road = _plain_road(signals=[stop], n_right=2, objects=[xb.ObjectSpec(id="9", s=38.0)])
    out = _convert(tmp_path=tmp_path, roads=[road], **H6_ON_ANY_DISTANCE)
    assert sorted(row[0] for row in _rows(out=out, agent_type="stop_sign")) == ["89", "9"]
    (finding,) = [r for r in _findings(out=out, heuristic_id="H5") if r["id"] == "89"]
    assert finding["action"] == "stencil_on_some_lanes" and finding["applied"]
