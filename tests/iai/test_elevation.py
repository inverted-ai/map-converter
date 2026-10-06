"""Lanelet heights on a road climbing at a constant grade (``convert_map.py`` in a subprocess)."""

import numpy as np

import map_quality as mq
from tests.iai import xodr_builder as xb

GRADE = 0.1
LENGTH = 100.0


def _convert_slope(*, map_dir, lane_sections: list[xb.LaneSectionSpec]) -> mq.ConversionOutputs:
    road = xb.RoadSpec(
        id="1",
        length=LENGTH,
        lane_sections=lane_sections,
        elevation=[(0.0, 0.0, GRADE, 0.0, 0.0)],
    )
    map_dir.mkdir()
    (map_dir / "slope.xodr").write_text(xb.build_xodr(roads=[road]))
    result = xb.run_convert_map(map_dir=str(map_dir))
    assert result.returncode == 0, result.stderr
    return mq.ConversionOutputs.load(map_dir=str(map_dir))


def _worst_height_error(*, outputs: mq.ConversionOutputs) -> float:
    """Largest gap between a point's height and the road's height at its x (the road runs along x)."""
    x, z = outputs.points[:, 0], outputs.points[:, 2]
    return float(np.abs(z - GRADE * x).max())


def test_every_lane_follows_the_grade(tmp_path):
    lanes = xb.standard_lanes(n_left=2, n_right=2)
    outputs = _convert_slope(map_dir=tmp_path / "m", lane_sections=[xb.LaneSectionSpec(s=0.0, lanes=lanes)])
    assert outputs.points[:, 2].max() > 0.9 * GRADE * LENGTH
    assert _worst_height_error(outputs=outputs) < 0.01


def test_second_lane_section_keeps_the_grade(tmp_path):
    sections = [
        xb.LaneSectionSpec(s=s, lanes=xb.standard_lanes(n_left=2, n_right=2)) for s in (0.0, LENGTH / 2)
    ]
    outputs = _convert_slope(map_dir=tmp_path / "m", lane_sections=sections)
    assert _worst_height_error(outputs=outputs) < 0.01


def test_merging_lane_keeps_its_height(tmp_path):
    lanes = xb.standard_lanes(n_left=1, n_right=2) + [
        xb.LaneSpec(id=-3, width=3.5, width_slope=-3.5 / LENGTH)
    ]
    outputs = _convert_slope(map_dir=tmp_path / "m", lane_sections=[xb.LaneSectionSpec(s=0.0, lanes=lanes)])
    assert _worst_height_error(outputs=outputs) < 0.01
