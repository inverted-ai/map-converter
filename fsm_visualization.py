"""Render a full-map birdview mosaic with traffic-light intersections and phase subgroups.

Companion to ``traffic_light_fsm.py``. It renders the converted map's road birdview (reusing
the same ``torchdrivesim`` path that ``convert_map.py`` uses for ``visualization.png``) and
overlays the FSM tool's output so a human can verify the automatic grouping at a glance:

  * stoplines are drawn as markers coloured by their phase subgroup,
  * a dashed grey convex-hull outline marks each intersection's membership,
  * each traffic-light ``actor_id`` is labelled.

For large maps the birdview is rendered as a grid of square tiles and stitched into one image,
at a fixed pixels-per-metre detail capped to a maximum dimension (both tunable in
:class:`VizConfig`).

Runs standalone (``python fsm_visualization.py <map-dir>``) or is invoked by
``traffic_light_fsm.py --visualize``. Heavy dependencies (torch, torchdrivesim, PIL) are
imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from typing import TYPE_CHECKING, Optional

import numpy as np

import traffic_light_fsm as tlf

if TYPE_CHECKING:  # avoid importing the heavy package at module load
    from traffic_light_fsm import Actor

logger = logging.getLogger(__name__)

# RGB colours for overlays.
_HULL_COLOR = (140, 140, 140)
# Phase colours in order (blue, orange, aqua, magenta); XODR controller groups can give > 2 phases.
_SUBGROUP_COLORS = ((40, 120, 255), (255, 140, 0), (27, 175, 122), (232, 123, 164))
_SINGLE_GROUP_COLOR = (40, 200, 90)  # green: single-phase intersection (e.g. on-ramp)
_TEXT_COLOR = (255, 255, 255)
_TEXT_HALO = (0, 0, 0)


@dataclasses.dataclass(slots=True)
class VizConfig:
    """Tunable visualization parameters.

    Attributes:
        target_px_per_m: Desired pixels per metre of the rendered mosaic.
        max_mosaic_px: Hard cap on the larger mosaic dimension; px/m is reduced to honour it.
        world_margin_m: Margin added around the map's vert bounds (metres).
        marker_radius_px: Radius of the per-stopline circle markers.
        hull_pad_px: Outward padding of each intersection's hull outline.
        hull_dash_px: Dash length (and gap) of the hull outline.
        font_size: Point size of the id / index text.
        draw_ids: Whether to label each stopline with its actor id.
    """

    target_px_per_m: float = 2.0
    max_mosaic_px: int = 8192
    world_margin_m: float = 20.0
    marker_radius_px: int = 6
    hull_pad_px: float = 14.0
    hull_dash_px: int = 12
    font_size: int = 16
    draw_ids: bool = True


# --------------------------------------------------------------------------------------
# Pure geometry helpers (no torch / no rendering -> unit-testable)
# --------------------------------------------------------------------------------------


def compute_ppm_and_size(
    *, bounds: tuple[float, float, float, float], config: VizConfig
) -> tuple[float, int, int]:
    """Pixels-per-metre and mosaic (width, height) in pixels for given world bounds.

    ``bounds`` is ``(xmin, xmax, ymin, ymax)``. Uses ``target_px_per_m`` unless that would
    exceed ``max_mosaic_px`` on the larger axis, in which case px/m is scaled down to fit.

    The torchdrivesim CV2 renderer transposes its output so that world **x maps to image rows**
    and world **y maps to image columns** (cv2.py: ``image.transpose(-2, -3)``). Hence mosaic
    height tracks the x-extent and width tracks the y-extent.
    """
    xmin, xmax, ymin, ymax = bounds
    extent_x = max(xmax - xmin, 1e-6)
    extent_y = max(ymax - ymin, 1e-6)
    ppm = min(config.target_px_per_m, config.max_mosaic_px / max(extent_x, extent_y))
    width = max(1, int(round(extent_y * ppm)))  # columns <- world y
    height = max(1, int(round(extent_x * ppm)))  # rows    <- world x
    return ppm, width, height


def world_to_pixel(
    *, x: np.ndarray, y: np.ndarray, xmax: float, ymax: float, ppm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map world ``(x, y)`` to mosaic pixel ``(col, row)`` (PIL's ``(x, y)`` draw order).

    Matches the torchdrivesim CV2 camera, which transposes its render so that world x drives
    the image row and world y drives the column. With ``camera_sc = [0, 1]`` the result is:
    larger x -> smaller row (x points up), larger y -> smaller column (y points left), so that
    overlays register exactly on the rendered tiles.
    """
    col = (ymax - np.asarray(y)) * ppm
    row = (xmax - np.asarray(x)) * ppm
    return col, row


# --------------------------------------------------------------------------------------
# Tile rendering (torch / torchdrivesim)
# --------------------------------------------------------------------------------------


def _mesh_bounds(*, verts_xy: np.ndarray, margin: float) -> tuple[float, float, float, float]:
    xmin, ymin = verts_xy.min(axis=0)
    xmax, ymax = verts_xy.max(axis=0)
    return (xmin - margin, xmax + margin, ymin - margin, ymax + margin)


def render_birdview_mosaic(
    *, mesh_path: str, left_handed: bool, config: VizConfig
) -> tuple[np.ndarray, tuple[float, float, float, float], float]:
    """Render the whole map birdview and crop to a tight bounding box.

    The map is rendered in a single square frame centred on the map (the torchdrivesim CV2
    renderer only supports square resolutions), then cropped to the map's bounds so the result
    is exactly ``height`` x ``width`` from :func:`compute_ppm_and_size`. This keeps the camera
    transform trivially consistent with :func:`world_to_pixel`. A single render at up to
    ``max_mosaic_px`` covers maps comfortably; tiling/stitching for detail beyond the cap is a
    deferred extension.

    Returns ``(mosaic_rgb_uint8, bounds, ppm)`` where ``bounds`` is ``(xmin, xmax, ymin, ymax)``
    and ``ppm`` is the pixels-per-metre actually used.
    """
    import torch
    from torchdrivesim.mesh import BirdviewMesh, BirdviewRGBMeshGenerator
    from torchdrivesim.rendering import RendererConfig, renderer_from_config
    from torchdrivesim.utils import Resolution

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = BirdviewMesh.load(mesh_path).to(device)
    verts_xy = mesh.verts.reshape(-1, 2).cpu().numpy()
    bounds = _mesh_bounds(verts_xy=verts_xy, margin=config.world_margin_m)
    xmin, xmax, ymin, ymax = bounds
    ppm, width, height = compute_ppm_and_size(bounds=bounds, config=config)

    side = max(width, height)  # square render; world x -> rows, world y -> cols
    fov = side / ppm
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    logger.info(
        "Rendering %dx%d px at %.3f px/m (square render %d px, fov %.1f m, bounds x[%.1f,%.1f] y[%.1f,%.1f])",
        width,
        height,
        ppm,
        side,
        fov,
        xmin,
        xmax,
        ymin,
        ymax,
    )

    renderer = renderer_from_config(RendererConfig(left_handed_coordinates=left_handed))
    generator = BirdviewRGBMeshGenerator(mesh, renderer.color_map, renderer.rendering_levels).to(device)
    rgb_mesh = generator.generate(num_cameras=1)
    dtype = mesh.verts.dtype

    image = renderer.render_frame(
        rgb_mesh,
        camera_xy=torch.tensor([[[cx, cy]]], device=device, dtype=dtype),
        camera_sc=torch.tensor([[[0.0, 1.0]]], device=device, dtype=dtype),
        res=Resolution(side, side),
        fov=fov,
    )
    full = image[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

    # Crop the centred square render to the tight (height x width) map bounds.
    row0 = (side - height) // 2
    col0 = (side - width) // 2
    mosaic = np.ascontiguousarray(full[row0 : row0 + height, col0 : col0 + width])
    return mosaic, bounds, ppm


# --------------------------------------------------------------------------------------
# Overlays (PIL, drawn directly onto the mosaic for exact resolution)
# --------------------------------------------------------------------------------------


def _load_font(*, size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _dashed_polygon(draw, *, points: list[tuple[float, float]], color, dash: int, width: int):
    n = len(points)
    for i in range(n):
        p0 = np.array(points[i], dtype=float)
        p1 = np.array(points[(i + 1) % n], dtype=float)
        seg = p1 - p0
        length = float(np.hypot(*seg))
        if length < 1e-6:
            continue
        direction = seg / length
        pos = 0.0
        while pos < length:
            start = p0 + direction * pos
            end = p0 + direction * min(pos + dash, length)
            draw.line([tuple(start), tuple(end)], fill=color, width=width)
            pos += 2 * dash


def _padded_hull(*, pts: np.ndarray, pad: float) -> list[tuple[float, float]]:
    """Convex hull of pixel points, padded outward from the centroid.

    Falls back to a padded bounding box for fewer than 3 points or degenerate (collinear) sets.
    """
    centroid = pts.mean(axis=0)
    if len(pts) >= 3:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(pts)
            ordered = pts[hull.vertices]
            out = []
            for p in ordered:
                v = p - centroid
                norm = float(np.hypot(*v))
                out.append(tuple(p + (v / norm) * pad if norm > 1e-6 else p))
            return out
        except Exception:  # noqa: BLE001 - degenerate hull -> bbox fallback
            pass
    xmin, ymin = pts.min(axis=0) - pad
    xmax, ymax = pts.max(axis=0) + pad
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def _text_with_halo(draw, *, xy, text, font):
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, fill=_TEXT_HALO, font=font)
    draw.text((x, y), text, fill=_TEXT_COLOR, font=font)


def _draw_legend(draw, *, font):
    rows = [(color, f"phase {phase}") for phase, color in enumerate(_SUBGROUP_COLORS)] + [
        (_SINGLE_GROUP_COLOR, "single-phase intersection"),
        (_HULL_COLOR, "intersection outline"),
    ]
    pad, line_h, swatch = 10, font.size + 8, font.size
    width = 270
    height = pad * 2 + line_h * len(rows)
    draw.rectangle([8, 8, 8 + width, 8 + height], fill=(0, 0, 0), outline=(255, 255, 255))
    for i, (color, label) in enumerate(rows):
        y = 8 + pad + i * line_h
        draw.rectangle([18, y, 18 + swatch, y + swatch], fill=color)
        draw.text((18 + swatch + 8, y), label, fill=(255, 255, 255), font=font)


def draw_overlays(
    *,
    mosaic: np.ndarray,
    resolved: list["tlf.IntersectionResult"],
    actors: list["Actor"],
    bounds: tuple[float, float, float, float],
    ppm: float,
    config: VizConfig,
) -> np.ndarray:
    """Draw hulls, subgroup-coloured markers, ids and a legend onto the mosaic."""
    from PIL import Image, ImageDraw

    _, xmax, _, ymax = bounds
    image = Image.fromarray(mosaic)
    draw = ImageDraw.Draw(image)
    font = _load_font(size=config.font_size)
    r = config.marker_radius_px

    for index, result in enumerate(resolved):
        intersection, groups = result.actors, result.groups
        xs = np.array([actors[i].center[0] for i in intersection])
        ys = np.array([actors[i].center[1] for i in intersection])
        us, vs = world_to_pixel(x=xs, y=ys, xmax=xmax, ymax=ymax, ppm=ppm)
        pts = np.column_stack([us, vs])

        hull = _padded_hull(pts=pts, pad=config.hull_pad_px)
        _dashed_polygon(draw, points=hull, color=_HULL_COLOR, dash=config.hull_dash_px, width=2)

        hull_top = min(hull, key=lambda p: p[1])
        _text_with_halo(draw, xy=(hull_top[0], hull_top[1] - config.font_size - 4),
                        text=f"#{index}", font=font)

        single = len(groups) == 1
        for phase, group in enumerate(groups):
            color = _SINGLE_GROUP_COLOR if single else _SUBGROUP_COLORS[phase % len(_SUBGROUP_COLORS)]
            for i in group:
                u, v = world_to_pixel(
                    x=np.array(actors[i].center[0]), y=np.array(actors[i].center[1]),
                    xmax=xmax, ymax=ymax, ppm=ppm,
                )
                u, v = float(u), float(v)
                draw.ellipse([u - r, v - r, u + r, v + r], fill=color, outline=(0, 0, 0))
                if config.draw_ids:
                    _text_with_halo(draw, xy=(u + r + 2, v - config.font_size / 2),
                                    text=str(actors[i].actor_id), font=font)

    _draw_legend(draw, font=font)
    return np.asarray(image)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def render_mosaic(
    *,
    map_dir: str,
    metadata: dict,
    actors: list["Actor"],
    resolved: list["tlf.IntersectionResult"],
    config: VizConfig,
) -> str:
    """Render the mosaic + overlays for an already-resolved FSM grouping and write the PNG.

    Returns the output image path.
    """
    import imageio.v2 as imageio

    mesh_name = metadata.get("mesh_path")
    if not mesh_name:
        raise ValueError("metadata.json has no mesh_path - cannot render birdview")
    mesh_path = mesh_name if os.path.isabs(mesh_name) else os.path.join(map_dir, mesh_name)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mosaic, bounds, ppm = render_birdview_mosaic(
        mesh_path=mesh_path,
        left_handed=bool(metadata.get("left_handed_coordinates", False)),
        config=config,
    )
    mosaic = draw_overlays(
        mosaic=mosaic, resolved=resolved, actors=actors, bounds=bounds, ppm=ppm, config=config
    )

    location = metadata.get("name") or os.path.basename(os.path.normpath(map_dir))
    out_path = os.path.join(map_dir, f"{location}_fsm_visualization.png")
    imageio.imsave(out_path, mosaic)
    return out_path


def visualize(
    *,
    map_dir: str,
    fsm_config: tlf.FsmConfig,
    viz_config: VizConfig,
    use_topology: bool = True,
    signal_groups_path: Optional[str] = None,
) -> str:
    """Standalone entry: resolve groupings from artifacts in ``map_dir`` and render the mosaic.

    ``signal_groups_path`` overrides the sidecar beside the stoplines, as in ``traffic_light_fsm.py``.
    """
    metadata = tlf.load_metadata(map_dir=map_dir)
    stoplines_path = tlf.find_stoplines_path(map_dir=map_dir)
    actors = tlf.load_actors(stoplines_path=stoplines_path)
    topology = tlf._load_topology(map_dir=map_dir, metadata=metadata) if use_topology else None
    _, signal_groups = tlf.load_signal_groups(stoplines_path=stoplines_path, override=signal_groups_path)
    resolved = tlf.resolve_intersection_groups(
        actors=actors, config=fsm_config, topology=topology, signal_groups=signal_groups
    )
    logger.info("Resolved %d intersections; rendering visualization", len(resolved))
    return render_mosaic(
        map_dir=map_dir, metadata=metadata, actors=actors, resolved=resolved, config=viz_config
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    fd = tlf.FsmConfig()
    vd = VizConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_dir", help="Map directory with metadata.json, mesh and *_stoplines.json")
    parser.add_argument("--cluster-eps", type=float, default=fd.cluster_eps)
    parser.add_argument("--no-topology", action="store_true", help="Disable lanelet topology refinement")
    parser.add_argument(
        "--signal-groups",
        default=None,
        help="Signal groups JSON (default: <map>_signal_groups.json beside the stoplines)",
    )
    parser.add_argument("--px-per-m", type=float, default=vd.target_px_per_m)
    parser.add_argument("--max-mosaic-px", type=int, default=vd.max_mosaic_px)
    parser.add_argument("--font-size", type=int, default=vd.font_size)
    parser.add_argument("--marker-radius-px", type=int, default=vd.marker_radius_px)
    return parser


def main() -> None:
    """CLI entry point: render the FSM mosaic for one map directory."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()
    fsm_config = tlf.FsmConfig(cluster_eps=args.cluster_eps)
    viz_config = VizConfig(
        target_px_per_m=args.px_per_m,
        max_mosaic_px=args.max_mosaic_px,
        font_size=args.font_size,
        marker_radius_px=args.marker_radius_px,
    )
    out_path = visualize(
        map_dir=args.map_dir,
        fsm_config=fsm_config,
        viz_config=viz_config,
        use_topology=not args.no_topology,
        signal_groups_path=args.signal_groups,
    )
    logger.info("Wrote FSM visualization to %s", out_path)


if __name__ == "__main__":
    main()
