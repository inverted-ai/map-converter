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
    words = vocabulary.load(path=path)
    assert words.translations["object.type"] == {"-1": "none", "parking": "parkingSpace", "stopline": "roadMark", "Pole": "pole"}
    assert words.translations["signal.country"] == {"OpenDRIVE": "DEU"}


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
    assert provenance["vocabulary"] == {
        "path": path,
        "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
        "vendors": [],
    }


def _vendor_dictionary(*, folder, vendor: str) -> str:
    folder.mkdir()
    (folder / "acme.yaml").write_text(
        f"match: {{vendor: '{vendor}'}}\ntranslations:\n  object.orientation: {{sideways: none}}\n"
    )
    return str(folder)


def _sideways_object_map(*, map_dir, vendor: str) -> None:
    road = _road(objects=[xb.ObjectSpec(id="93", s=30.0, orientation="sideways")])
    xb.write_xodr(map_dir=map_dir, name="case", text=xb.build_xodr(roads=[road], vendor=vendor))


@pytest.mark.parametrize("vendor, applies", [("Acme", True), ("Other", False)])
def test_a_vendors_dictionary_applies_to_that_vendors_files_only(tmp_path, vendor, applies):
    vendors = _vendor_dictionary(folder=tmp_path / "vendors", vendor="Acme")
    _sideways_object_map(map_dir=tmp_path / "map", vendor=vendor)
    result = xb.run_convert_map(map_dir=str(tmp_path / "map"), vocabulary_vendors=vendors)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "map" / "case_conversion_report.json").read_text())
    assert bool(_records(report=report, kind="translated", element="object.orientation")) is applies
    assert bool(_records(report=report, kind="unknown_value", element="object.orientation")) is not applies
    provenance = json.loads((tmp_path / "map" / "case_provenance.json").read_text())
    expected = ["acme.yaml"] if applies else []
    assert [os.path.basename(v["path"]) for v in provenance["vocabulary"]["vendors"]] == expected


def test_a_vendor_dictionary_without_a_match_is_refused(tmp_path):
    (tmp_path / "vendors").mkdir()
    (tmp_path / "vendors" / "acme.yaml").write_text("translations:\n  object.orientation: {sideways: none}\n")
    _sideways_object_map(map_dir=tmp_path / "map", vendor="Acme")
    result = xb.run_convert_map(map_dir=str(tmp_path / "map"), vocabulary_vendors=str(tmp_path / "vendors"))
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
    assert "match" in result.stderr


def test_the_shipped_vendor_dictionaries_load_and_deepmap_matches_its_header():
    vendors = vocabulary.load_vendors(folder=cm.MapConversionConfig().vocabulary_vendors)
    deepmap = {"vendor": "DeepMap, Inc.", "name": "x", "revMajor": "1", "revMinor": "4"}
    assert [v.translations for v in vocabulary.matching(vendors=vendors, header=deepmap)] == [
        {"object.orientation": {"undefined": "none"}}
    ]
    assert vocabulary.matching(vendors=vendors, header={"vendor": "VectorZero"}) == []



def test_the_default_dictionary_reads_light_codes_us_codes_and_lane_zero():
    words = vocabulary.load(path=cm.MapConversionConfig().vocabulary)
    assert vocabulary.is_traffic_light(signal_type="1000001", vocabulary=words)
    assert not vocabulary.is_traffic_light(signal_type="r1", vocabulary=words)
    assert vocabulary.country_for_code(country="", signal_type="r1", subtype="1", vocabulary=words) == ("USA", "R1", "1")
    assert vocabulary.country_for_code(country="DEU", signal_type="r1", subtype="1", vocabulary=words) is None
    assert vocabulary.names_no_lane(validity_from="0", validity_to="0", vocabulary=words)


def test_a_dictionary_without_signals_reads_codes_literally(tmp_path):
    words = vocabulary.load(path=_dictionary(path=tmp_path / "literal.yaml", text="translations: {}\n"))
    assert vocabulary.is_traffic_light(signal_type="r1", vocabulary=words)
    assert vocabulary.country_for_code(country="", signal_type="r1", subtype="1", vocabulary=words) is None
    assert not vocabulary.names_no_lane(validity_from="0", validity_to="0", vocabulary=words)


def test_a_vendor_dictionary_overrides_the_signals_it_names(tmp_path):
    base = vocabulary.load(path=cm.MapConversionConfig().vocabulary)
    vendor = vocabulary.load(
        path=_dictionary(path=tmp_path / "v.yaml", text="match: {vendor: A}\nsignals: {lane_zero_validity_means_none: false}\n"),
        vendor=True,
    )
    combined = vocabulary.combine(vocabularies=[base, vendor])
    assert not combined.signals["lane_zero_validity_means_none"]
    assert combined.signals["countries_by_code"] == {"USA": ("R1-1",)}


@pytest.mark.parametrize("text, named", [
    ("signals: {colour_codes: []}\n", "colour_codes"),
    ("signals: {traffic_light_types: ['100(']}\n", "regular expression"),
])
def test_a_bad_signals_section_is_refused(tmp_path, text, named):
    with pytest.raises(ValueError, match=named):
        vocabulary.load(path=_dictionary(path=tmp_path / "bad.yaml", text=text))



def _stoplines(*, map_dir) -> list[dict]:
    return json.loads((map_dir / "case_stoplines.json").read_text())


@pytest.mark.parametrize("words, rows", [("markings: {stop_stencils: [MyStop]}\n", ["94"]), ("translations: {}\n", [])])
def test_stencil_names_come_from_the_dictionary(tmp_path, words, rows):
    road = _road(objects=[xb.ObjectSpec(id="94", s=30.0, name="MyStop")])
    result, _ = _convert(tmp_path=tmp_path / "map", road=road, vocabulary=_dictionary(path=tmp_path / "w.yaml", text=words))
    assert result.returncode == 0, result.stderr
    assert [str(r["actor_id"]) for r in _stoplines(map_dir=tmp_path / "map") if r["agent_type"] == "stop_sign"] == rows


def test_without_stencil_entries_a_stencil_object_is_only_an_object(tmp_path):
    road = _road(objects=[xb.ObjectSpec(id="95", s=30.0)])  # named Stencil_STOP
    words = _dictionary(path=tmp_path / "w.yaml", text="translations: {}\n")
    result, _ = _convert(tmp_path=tmp_path / "map", road=road, vocabulary=words)
    assert result.returncode == 0, result.stderr
    assert _stoplines(map_dir=tmp_path / "map") == []


def test_painted_stop_line_names_come_from_the_dictionary(tmp_path):
    road = _road(objects=[xb.ObjectSpec(id="96", s=30.0, name="Haltlinie", type="roadMark")])
    words = _dictionary(path=tmp_path / "w.yaml", text="markings: {painted_stop_lines: {names: [haltlinie]}}\n")
    result, report = _convert(tmp_path=tmp_path / "map", road=road, vocabulary=words)
    assert result.returncode == 0, result.stderr
    assert [r["id"] for r in report["records"] if r["kind"] == "not_converted"] == ["96"]


def test_the_default_dictionary_holds_the_built_in_readings():
    words = vocabulary.load(path=cm.MapConversionConfig().vocabulary)
    assert words.markings["stop_stencils"] == ("Stencil_STOP",)
    assert words.markings["yield_stencils"] == ("Stencil_GiveWay", "Stencil_Yield_Triangles")
    assert words.signals["unknown_country_table"] == "ZAM"
    assert words.signals["sign_type_replacements"] == {"DEU": {"252": "260"}, "ZAM": {"252": "260"}}


@pytest.mark.parametrize("text, named", [
    ("signals: {unknown_country_table: XYZ}\n", "XYZ"),
    ("signals: {sign_type_replacements: {DEU: {'252': '99999'}}}\n", "99999"),
    ("markings: {kerbs: []}\n", "kerbs"),
])
def test_bad_readings_are_refused(tmp_path, text, named):
    with pytest.raises(ValueError, match=named):
        vocabulary.load(path=_dictionary(path=tmp_path / "bad.yaml", text=text))


def test_a_replaced_sign_type_is_recorded(tmp_path):
    sign = xb.SignalSpec(id="97", s=30.0, dynamic="no", type="252", subtype="-1", validity=(-1, -1))
    road = xb.straight_road(id="1", length=50.0, signals=[sign])
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=[road]))
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "case_conversion_report.json").read_text())
    assert [(r["value"], r["action"]) for r in _records(report=report, kind="translated", element="signal.type")] == [
        ("252", "read_as_260")
    ]



def _lane_speed_map(*, map_dir, unit: str = "") -> None:
    unit_attr = f' unit="{unit}"' if unit else ""
    text = xb.build_xodr(roads=[xb.straight_road(id="1", length=50.0)])
    text = text.replace("</lane>", f'<speed sOffset="0.0" max="50.0"{unit_attr}/></lane>', 1)
    xb.write_xodr(map_dir=map_dir, name="case", text=text)


@pytest.mark.parametrize(
    "unit, words, translated",
    [("", "defaults: {speed_unit: km/h}\n", [None]), ("kmh", "translations: {speed.unit: {kmh: km/h}}\n", ["kmh"])],
)
def test_speed_units_come_from_the_dictionary(tmp_path, unit, words, translated):
    _lane_speed_map(map_dir=tmp_path / "map", unit=unit)
    result = xb.run_convert_map(map_dir=str(tmp_path / "map"), vocabulary=_dictionary(path=tmp_path / "w.yaml", text=words))
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "map" / "case_conversion_report.json").read_text())
    assert [r["value"] for r in _records(report=report, kind="translated", element="speed.unit")] == translated
    assert not [r for r in report["records"] if r["element"] == "speed.unit" and r["kind"] != "translated"]


def test_without_a_speed_default_a_speed_without_unit_stays_unset(tmp_path):
    _lane_speed_map(map_dir=tmp_path)
    result = xb.run_convert_map(map_dir=str(tmp_path))
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "case_conversion_report.json").read_text())
    assert _records(report=report, kind="missing_attribute", element="speed.unit")


def test_a_speed_default_must_be_a_unit_the_converter_reads(tmp_path):
    with pytest.raises(ValueError, match="furlongs"):
        vocabulary.load(path=_dictionary(path=tmp_path / "w.yaml", text="defaults: {speed_unit: furlongs}\n"))


def test_provenance_records_the_opendrive_version(tmp_path):
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=[xb.straight_road(id="1", length=50.0)], rev_minor="6"))
    assert xb.run_convert_map(map_dir=str(tmp_path)).returncode == 0
    assert json.loads((tmp_path / "case_provenance.json").read_text())["opendrive_version"] == "1.6"
