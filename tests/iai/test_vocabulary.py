"""The vocabulary dictionary: names an OpenDRIVE file uses, translated into names the converter understands."""

import hashlib
import json
import os

import pytest

import conversion_common
import convert_map as cm
import vocabulary
from tests.iai import xodr_builder as xb


def _road(*, objects=(), road_mark: str = "solid") -> xb.RoadSpec:
    road = xb.straight_road(id="1", length=50.0, objects=list(objects))
    for lane in road.lane_sections[0].lanes:
        lane.road_mark = road_mark
    return road


def _convert(*, tmp_path, road: xb.RoadSpec, **config: str):
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=[road]))
    result = xb.run_convert_map(map_dir=str(tmp_path), **config)
    report = json.loads((tmp_path / "case_conversion_report.json").read_text()) if result.returncode == 0 else None
    return result, report


def _records(*, report: dict, kind: str, element: str) -> list[dict]:
    return [r for r in report["records"] if r["kind"] == kind and r["element"] == element]


def _dictionary(*, path, text: str) -> str:
    path.write_text(text)
    return str(path)


def test_the_default_dictionary_is_an_absolute_file_shipped_with_the_converter():
    path = cm.MapConversionConfig().vocabulary
    assert os.path.isabs(path) and os.path.isfile(path)
    assert vocabulary.load(path=path).translations["object.type"] == {"-1": "none", "parking": "parkingSpace", "stopline": "roadMark"}


def test_default_dictionary_translations_are_recorded(tmp_path):
    road = _road(objects=[xb.ObjectSpec(id="91", s=30.0, name="pole", type="parking")])
    result, report = _convert(tmp_path=tmp_path, road=road)
    assert result.returncode == 0, result.stderr
    (translated,) = _records(report=report, kind="translated", element="object.type")
    assert (str(translated["id"]), translated["value"], translated["action"]) == ("91", "parking", "read_as_parkingSpace")
    assert not _records(report=report, kind="unknown_value", element="object.type")


def test_a_users_dictionary_translates_names_the_converter_has_never_seen(tmp_path):
    road = _road(objects=[xb.ObjectSpec(id="92", s=30.0, orientation="undefined")], road_mark="custom")
    words = _dictionary(
        path=tmp_path / "words.yaml",
        text="translations:\n  object.orientation: {undefined: none}\n  roadMark.type: {custom: solid}\n",
    )
    plain, plain_report = _convert(tmp_path=tmp_path / "plain", road=road)
    result, report = _convert(tmp_path=tmp_path / "words", road=road, vocabulary=words)
    assert plain.returncode == 0 and result.returncode == 0, result.stderr
    assert _records(report=plain_report, kind="unknown_value", element="object.orientation")
    assert not _records(report=report, kind="unknown_value", element="object.orientation")
    assert [r["value"] for r in _records(report=report, kind="translated", element="roadMark.type")]
    assert not [r for r in report["records"] if r.get("action") == "unknown_marking"]
    assert [r for r in plain_report["records"] if r.get("action") == "unknown_marking"]


@pytest.mark.parametrize(
    "text, named",
    [
        ("translations:\n  lane.type: {walking: footpath}\n", "footpath"),
        ("translations:\n  lane.colour: {red: blue}\n", "lane.colour"),
        ("words: {}\n", "words"),
    ],
)
def test_a_dictionary_naming_what_the_converter_cannot_understand_is_refused(tmp_path, text, named):
    words = _dictionary(path=tmp_path / "words.yaml", text=text)
    result, _ = _convert(tmp_path=tmp_path, road=_road(), vocabulary=words)
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
    assert named in result.stderr


def test_provenance_records_the_dictionary_used(tmp_path):
    result, _ = _convert(tmp_path=tmp_path, road=_road())
    assert result.returncode == 0, result.stderr
    provenance = json.loads((tmp_path / "case_provenance.json").read_text())
    path = cm.MapConversionConfig().vocabulary
    assert provenance["vocabulary"] == {"path": path, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
