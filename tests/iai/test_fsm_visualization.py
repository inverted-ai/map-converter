"""Pure-function tests for fsm_visualization (no rendering / no torch).

The rendering path needs a mesh + GPU and is verified manually; here we lock down the
resolution/cap arithmetic and the world->pixel transform that overlays rely on.
"""

import numpy as np

import fsm_visualization as fv


def test_ppm_uses_target_when_under_cap():
    bounds = (0.0, 1000.0, 0.0, 500.0)  # extent_x=1000 (rows), extent_y=500 (cols)
    ppm, width, height = fv.compute_ppm_and_size(bounds=bounds, config=fv.VizConfig())
    assert ppm == 2.0
    assert height == 2000  # rows track world x
    assert width == 1000  # cols track world y


def test_ppm_scaled_down_to_honour_cap():
    bounds = (0.0, 10000.0, 0.0, 4000.0)
    config = fv.VizConfig(target_px_per_m=2.0, max_mosaic_px=8192)
    ppm, width, height = fv.compute_ppm_and_size(bounds=bounds, config=config)
    assert ppm < config.target_px_per_m
    assert max(width, height) <= config.max_mosaic_px
    assert height == 8192  # larger axis pinned to the cap


def test_world_to_pixel_corner_and_orientation():
    xmax, ymax, ppm = 100.0, 50.0, 2.0
    # top-left corner of the cropped mosaic is (xmax, ymax)
    col, row = fv.world_to_pixel(x=np.array(100.0), y=np.array(50.0), xmax=xmax, ymax=ymax, ppm=ppm)
    assert float(col) == 0.0 and float(row) == 0.0
    # larger x -> smaller row (x points up); larger y -> smaller col (y points left)
    _, row_lo_x = fv.world_to_pixel(x=np.array(0.0), y=np.array(50.0), xmax=xmax, ymax=ymax, ppm=ppm)
    col_lo_y, _ = fv.world_to_pixel(x=np.array(100.0), y=np.array(0.0), xmax=xmax, ymax=ymax, ppm=ppm)
    assert float(row_lo_x) == 200.0  # (100-0)*2
    assert float(col_lo_y) == 100.0  # (50-0)*2


def test_world_to_pixel_is_vectorised():
    col, row = fv.world_to_pixel(
        x=np.array([10.0, 20.0]), y=np.array([5.0, 15.0]), xmax=30.0, ymax=30.0, ppm=1.0
    )
    assert np.allclose(col, [25.0, 15.0])
    assert np.allclose(row, [20.0, 10.0])
