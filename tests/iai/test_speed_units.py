"""Speeds reach the map in m/s: speed_unit= says what a speed with no unit is in; signs convert by their unit."""

import json
import math
import re

import pytest

import conversion_common
from tests.iai import xodr_builder as xb


def _road_text(*, lane_speed: str = "", type_speed: str = "", signals=()) -> str:
    text = xb.build_xodr(roads=[xb.straight_road(id="1", length=50.0, signals=list(signals))])
    if lane_speed:
        text = text.replace("</lane>", f"{lane_speed}</lane>")
    if type_speed:
        text = text.replace('<type s="0.0" type="town"/>', f'<type s="0.0" type="town">{type_speed}</type>', 1)
    return text


def _speed_limits(*, map_dir) -> list[float]:
    """Every speed limit the .osm holds (its regulatory elements' sign_type values)."""
    with open(map_dir / "case.osm") as f:
        osm = f.read()
    return sorted({float(v) for v in re.findall(r'<tag k="sign_type" v="([-0-9.e]+)"', osm)})


def _convert(*, map_dir, text: str, **config: str) -> dict:
    xb.write_xodr(map_dir=map_dir, name="case", text=text)
    result = xb.run_convert_map(map_dir=str(map_dir), **config)
    assert result.returncode == 0, result.stderr
    return json.loads((map_dir / "case_conversion_report.json").read_text())


@pytest.mark.parametrize("setting", ["furlongs", "0", "-2", "nan"])
def test_an_unusable_speed_unit_is_refused(tmp_path, setting):
    xb.write_xodr(map_dir=tmp_path, name="case", text=_road_text())
    result = xb.run_convert_map(map_dir=str(tmp_path), speed_unit=setting)
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
    assert "speed_unit" in result.stderr


def test_a_speed_without_unit_is_read_as_metres_per_second_by_default(tmp_path):
    report = _convert(map_dir=tmp_path, text=_road_text(lane_speed='<speed sOffset="0.0" max="13.89"/>'))
    assert _speed_limits(map_dir=tmp_path) == [13.89]
    records = [r for r in report["records"] if r["element"] == "speed.unit"]
    assert records and {(r["kind"], r["action"]) for r in records} == {("missing_attribute", "read_as_m/s")}


@pytest.mark.parametrize("setting, written", [("km/h", 50 / 3.6), ("mph", 50 / 2.237), ("0.3048", 50 * 0.3048)])
def test_speed_unit_names_the_unit_of_speeds_without_one(tmp_path, setting, written):
    _convert(map_dir=tmp_path, text=_road_text(lane_speed='<speed sOffset="0.0" max="50"/>'), speed_unit=setting)
    assert [round(v, 6) for v in _speed_limits(map_dir=tmp_path)] == [round(written, 6)]


def test_a_speed_that_names_its_unit_ignores_the_setting(tmp_path):
    text = _road_text(lane_speed='<speed sOffset="0.0" max="36" unit="km/h"/>')
    _convert(map_dir=tmp_path, text=text, speed_unit="mph")
    assert _speed_limits(map_dir=tmp_path) == [10.0]


@pytest.mark.parametrize("setting, written", [("m/s", 20.0), ("0.5", 10.0)])
def test_road_type_speeds_without_unit_follow_the_setting(tmp_path, setting, written):
    _convert(map_dir=tmp_path, text=_road_text(type_speed='<speed max="20"/>'), speed_unit=setting)
    assert _speed_limits(map_dir=tmp_path) == [written]


def test_a_dictionary_can_no_longer_set_the_speed_unit(tmp_path):
    words = tmp_path / "w.yaml"
    words.write_text("defaults: {speed_unit: km/h}\n")
    xb.write_xodr(map_dir=tmp_path / "map", name="case", text=_road_text())
    result = xb.run_convert_map(map_dir=str(tmp_path / "map"), vocabulary=str(words))
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
    assert "defaults" in result.stderr


def _speed_sign(*, value: str, unit: str) -> xb.SignalSpec:
    return xb.SignalSpec(id="5", s=10.0, dynamic="no", type="274", subtype="-1", validity=(-1, -1), unit=unit, value=value)


@pytest.mark.parametrize("unit, written", [("mph", 30 / 2.237), ("km/h", 30 / 3.6), ("m/s", 30.0)])
def test_a_speed_sign_is_written_in_metres_per_second(tmp_path, unit, written):
    _convert(map_dir=tmp_path, text=_road_text(signals=[_speed_sign(value="30", unit=unit)]))
    assert any(math.isclose(v, written, rel_tol=1e-9) for v in _speed_limits(map_dir=tmp_path))


def test_a_speed_sign_without_unit_keeps_its_number(tmp_path):
    _convert(map_dir=tmp_path, text=_road_text(signals=[_speed_sign(value="30", unit=None)]), speed_unit="km/h")
    assert 30.0 in _speed_limits(map_dir=tmp_path)
