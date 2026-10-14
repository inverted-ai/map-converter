"""AutoCritic's report of one conversion: ``<map>_autocritic.json`` and a self-contained ``<map>_autocritic.html``.

Three kinds of section, each with its own numbered items:

- ``autofix``: per heuristic, every place it changed the conversion against BasicConversion;
- ``would_have_fixed``: per heuristic that was off (or with AutoFix off), every place it would have acted;
- ``vocabulary``: per element attribute, every place the vocabulary dictionary translated a name or code;
- ``metric``: per validation check that failed, or that lists places (L10-L13), every place it found.

The HTML gives each section a numbered text list, a birdview of the whole map with the items circled and
numbered, and close-ups of each item (24 per image), all embedded; :class:`ReportConfig` switches the parts.
Item locations are in the local frame of the outputs. The FSM heuristics H11-H13 are reported by
``traffic_light_fsm.py`` in ``<map>_fsm_review.json``.
"""

from __future__ import annotations

import base64
import dataclasses
import html
import io
import json
import math
import os
from typing import Any, Optional

import numpy as np

from conversion_report import heuristic_number
from traffic_light_fsm import load_metadata, resolve_map_path, utm_projector

CLOSEUPS_PER_PAGE = 24
CLOSEUP_COLUMNS = 4
CLOSEUP_HALF_WIDTH_M = 25.0
CLOSEUP_TILE_PX = 240
CLOSEUP_GAP_PX = 4
CLOSEUP_MARKER_PX = 14
BIRDVIEW_DPI = 100
PNG_COLOURS = 32  # line art: a small palette keeps each image a fraction of its RGB size
NUMBERED_ON_BIRDVIEW = 300  # more items than this are circled without numbers
MARKER = "#d03b3b"
ROAD = "#8a8f94"


@dataclasses.dataclass(slots=True)
class ReportConfig:
    """Which parts of the AutoCritic report are written.

    Attributes:
        enabled: Write ``<map>_autocritic.json`` and ``.html`` (needs ``autocritic.enabled``).
        text: The numbered list of each section.
        birdview: One whole-map image per section with its items numbered.
        closeups: One close-up per item, 24 per image.
        max_closeups: Most close-ups per section; the remaining items are listed only.
    """

    enabled: bool = True
    text: bool = True
    birdview: bool = True
    closeups: bool = True
    max_closeups: int = 240


def _items(*, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"number": n, **entry} for n, entry in enumerate(entries, 1)]


def _rounded(*, value: Any) -> Any:
    return round(value, 2) if isinstance(value, float) else value


def build(
    *, records: list[dict[str, Any]], verdict: Optional[dict[str, Any]], autofix_enabled: bool, would_have_fixed: bool
) -> dict[str, Any]:
    """The report's content: a summary and its sections of numbered items.

    Args:
        records: The conversion report's records; the ``kind="heuristic"`` ones are used.
        verdict: The validation verdict as a dict, or ``None``.
        autofix_enabled: Whether AutoFix was on.
        would_have_fixed: Include the places switched-off heuristics would have acted.
    """
    by_heuristic: dict[tuple[bool, str], list[dict[str, Any]]] = {}
    names: dict[str, str] = {}
    for r in records:
        if r.get("kind") != "heuristic":
            continue
        names[r["element"]] = r["value"]
        by_heuristic.setdefault((bool(r["applied"]), r["element"]), []).append(
            {
                "element": str(r["id"]),
                "location": r.get("location"),
                "detail": r["message"].split(": ", 1)[-1],  # without the "<id> <name>: " prefix
            }
        )
    sections = []
    for applied, kind in ((True, "autofix"), (False, "would_have_fixed")):
        if not applied and not would_have_fixed:
            continue
        for key in sorted((k for a, k in by_heuristic if a == applied), key=heuristic_number):
            entries = by_heuristic[(applied, key)]
            sections.append(
                {
                    "kind": kind,
                    "key": key,
                    "title": f"{key} {names[key]}",
                    "items": _items(entries=entries),
                }
            )
    by_element: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r.get("kind") == "translated":
            by_element.setdefault(r["element"], []).append(
                {"element": str(r["id"]), "location": r.get("location"), "detail": r["message"]}
            )
    for element in sorted(by_element):
        sections.append(
            {
                "kind": "vocabulary",
                "key": element,
                "title": f"{element}: {len(by_element[element])} translated",
                "items": _items(entries=by_element[element]),
            }
        )
    for check in (verdict or {}).get("checks", []):
        if check["passed"] and not check.get("locations"):
            continue
        sections.append(
            {
                "kind": "metric",
                "key": check["name"],
                "title": f"{check['name']}: {_rounded(value=check['value'])} (limit {check['limit']}) - "
                + ("passed" if check["passed"] else "failed"),
                "detail": check.get("detail", ""),
                "items": _items(
                    entries=[
                        {"element": str(p["id"]), "location": p.get("location"), "detail": p.get("detail", "")}
                        for p in check.get("locations", [])
                    ]
                ),
            }
        )
    return {
        "summary": {
            "autofix": autofix_enabled,
            "preset": None if verdict is None else verdict["profile"],
            "status": None if verdict is None else verdict["status"],
        },
        "sections": sections,
    }


def load_polylines(*, map_dir: str) -> list[np.ndarray]:
    """Every lanelet boundary of the map in ``map_dir``, as ``(n, 2)`` local-frame arrays."""
    import lanelet2

    metadata = load_metadata(map_dir=map_dir)
    origin_lat, origin_lon = metadata["lanelet_map_origin"]
    lanelet_map, _ = lanelet2.io.loadRobust(
        resolve_map_path(map_dir=map_dir, name=metadata["lanelet_path"]),
        utm_projector(origin_lat=origin_lat, origin_lon=origin_lon),
    )
    return [
        np.array([(p.x, p.y) for p in bound])
        for ll in lanelet_map.laneletLayer
        for bound in (ll.leftBound, ll.rightBound)
        if len(bound) >= 2
    ]


def _image_uri(*, image) -> str:
    """A PIL image as a PNG data URI, quantized to ``PNG_COLOURS``."""
    out = io.BytesIO()
    image.convert("RGB").quantize(colors=PNG_COLOURS).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def _png(*, fig, dpi: int) -> str:
    import matplotlib.pyplot as plt
    from PIL import Image

    rendered = io.BytesIO()
    fig.savefig(rendered, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    rendered.seek(0)
    return _image_uri(image=Image.open(rendered))


def _draw_roads(*, ax, polylines: list[np.ndarray], linewidth: float) -> None:
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(polylines, colors=ROAD, linewidths=linewidth))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _bounding_boxes(*, polylines: list[np.ndarray]) -> np.ndarray:
    """``(n, 4)``: each polyline's ``min x, min y, max x, max y``."""
    return np.array([[*p.min(axis=0), *p.max(axis=0)] for p in polylines]).reshape(-1, 4)


def _birdview(*, polylines: list[np.ndarray], boxes: np.ndarray, items: list[dict[str, Any]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_roads(ax=ax, polylines=polylines, linewidth=0.3)
    points = np.array([i["location"] for i in items])
    ax.scatter(points[:, 0], points[:, 1], s=140, facecolors="none", edgecolors=MARKER, linewidths=2.0)
    if len(items) <= NUMBERED_ON_BIRDVIEW:
        for item in items:
            ax.annotate(str(item["number"]), item["location"], color=MARKER, fontsize=7, xytext=(4, 4), textcoords="offset points")
    low, high = points.min(axis=0), points.max(axis=0)
    if len(boxes):
        low, high = np.minimum(low, boxes[:, :2].min(axis=0)), np.maximum(high, boxes[:, 2:].max(axis=0))
    pad = max(10.0, 0.03 * float((high - low).max()))
    ax.set_xlim(low[0] - pad, high[0] + pad)
    ax.set_ylim(low[1] - pad, high[1] + pad)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    return _png(fig=fig, dpi=BIRDVIEW_DPI)


def _closeup_tile(*, polylines: list[np.ndarray], boxes: np.ndarray, item: dict[str, Any]):
    """One close-up: the roads within ``CLOSEUP_HALF_WIDTH_M`` of the item, its marker and its number."""
    from PIL import Image, ImageDraw

    x, y = item["location"]
    h = CLOSEUP_HALF_WIDTH_M
    scale = CLOSEUP_TILE_PX / (2 * h)
    tile = Image.new("RGB", (CLOSEUP_TILE_PX, CLOSEUP_TILE_PX), "white")
    draw = ImageDraw.Draw(tile)
    near = (boxes[:, 0] <= x + h) & (boxes[:, 2] >= x - h) & (boxes[:, 1] <= y + h) & (boxes[:, 3] >= y - h)
    for line in (polylines[k] for k in np.flatnonzero(near)):
        pixels = np.column_stack(((line[:, 0] - (x - h)) * scale, ((y + h) - line[:, 1]) * scale))
        draw.line([tuple(v) for v in pixels], fill=ROAD, width=1)
    centre, r = CLOSEUP_TILE_PX / 2, CLOSEUP_MARKER_PX
    draw.ellipse((centre - r, centre - r, centre + r, centre + r), outline=MARKER, width=2)
    draw.rectangle((0, 0, CLOSEUP_TILE_PX, 13), fill="white")
    draw.text((3, 1), f"{item['number']}. {item['element']}"[:40], fill="black")
    draw.rectangle((0, 0, CLOSEUP_TILE_PX - 1, CLOSEUP_TILE_PX - 1), outline="#d8dddc")
    return tile


def _closeup_pages(*, polylines: list[np.ndarray], boxes: np.ndarray, items: list[dict[str, Any]]) -> list[str]:
    """Close-ups of ``items``, ``CLOSEUPS_PER_PAGE`` per image (drawn with PIL: far faster than one axes each)."""
    from PIL import Image

    pages = []
    for start in range(0, len(items), CLOSEUPS_PER_PAGE):
        page_items = items[start : start + CLOSEUPS_PER_PAGE]
        rows = math.ceil(len(page_items) / CLOSEUP_COLUMNS)
        step = CLOSEUP_TILE_PX + CLOSEUP_GAP_PX
        page = Image.new("RGB", (CLOSEUP_COLUMNS * step, rows * step), "white")
        for k, item in enumerate(page_items):
            tile = _closeup_tile(polylines=polylines, boxes=boxes, item=item)
            page.paste(tile, ((k % CLOSEUP_COLUMNS) * step, (k // CLOSEUP_COLUMNS) * step))
        pages.append(_image_uri(image=page))
    return pages


CSS = """
:root{--bg:#fff;--fg:#1b1d1f;--muted:#5f6670;--line:#d8dddc}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171a;--fg:#e6e8ea;--muted:#9aa1a9;--line:#33383d}}
:root[data-theme="dark"]{--bg:#15171a;--fg:#e6e8ea;--muted:#9aa1a9;--line:#33383d}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}
h1{font-size:22px}h2{font-size:18px;border-top:1px solid var(--line);padding-top:12px}
.muted{color:var(--muted)}img{max-width:100%;height:auto;background:#fff;border:1px solid var(--line)}
ol{padding-left:2.2em}li{overflow-wrap:anywhere}
"""

SECTION_HEADINGS = {
    "autofix": "AutoFix",
    "would_have_fixed": "Would have fixed",
    "vocabulary": "Vocabulary",
    "metric": "AutoCritic metrics",
}


def render_html(*, report: dict[str, Any], polylines: list[np.ndarray], config: ReportConfig, title: str) -> str:
    """The self-contained HTML of ``report`` (images embedded as PNG data URIs)."""
    summary = report["summary"]
    boxes = _bounding_boxes(polylines=polylines)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{html.escape(title)} AutoCritic</title><style>{CSS}</style></head><body>",
        f"<h1>{html.escape(title)}: AutoCritic report</h1>",
        f"<p>AutoFix {'on' if summary['autofix'] else 'off'}; preset {html.escape(str(summary['preset']))}; "
        f"verdict {html.escape(str(summary['status']))}. Numbers in each section match its images.</p>",
    ]
    if not report["sections"]:
        parts.append("<p>No heuristic applied, none would have, and no check found anything.</p>")
    for number, section in enumerate(report["sections"], 1):
        items = section["items"]
        placed = [i for i in items if i.get("location") is not None]
        parts.append(
            f"<h2>{number}. {html.escape(SECTION_HEADINGS[section['kind']])}: {html.escape(section['title'])}</h2>"
        )
        if section.get("detail"):
            parts.append(f"<p class=muted>{html.escape(section['detail'])}</p>")
        parts.append(f"<p class=muted>{len(items)} item(s), {len(items) - len(placed)} without a position.</p>")
        if config.text and items:
            parts.append("<ol>")
            for item in items:
                where = "" if item.get("location") is None else f" at ({item['location'][0]:.1f}, {item['location'][1]:.1f})"
                parts.append(
                    f"<li value={item['number']}>{html.escape(item['element'])}{where}{': ' + html.escape(item['detail']) if item['detail'] else ''}</li>"
                )
            parts.append("</ol>")
        if config.birdview and placed:
            parts.append(f"<img src='{_birdview(polylines=polylines, boxes=boxes, items=placed)}' alt='birdview'>")
        if config.closeups and placed:
            shown = placed[: config.max_closeups]
            for page in _closeup_pages(polylines=polylines, boxes=boxes, items=shown):
                parts.append(f"<img src='{page}' alt='close-ups'>")
            if len(placed) > len(shown):
                parts.append(f"<p class=muted>{len(placed) - len(shown)} more items have no close-up (report.max_closeups).</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def write(
    *,
    map_dir: str,
    location: str,
    records: list[dict[str, Any]],
    verdict: Optional[dict[str, Any]],
    autofix_enabled: bool,
    would_have_fixed: bool,
    config: ReportConfig,
) -> tuple[str, str]:
    """Write ``<map>_autocritic.json`` and ``<map>_autocritic.html`` into ``map_dir`` (its outputs must still be there).

    Returns:
        The absolute paths of the JSON and the HTML.
    """
    report = build(records=records, verdict=verdict, autofix_enabled=autofix_enabled, would_have_fixed=would_have_fixed)
    json_path = os.path.abspath(os.path.join(map_dir, f"{location}_autocritic.json"))
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    images = config.birdview or config.closeups
    polylines = load_polylines(map_dir=map_dir) if images and report["sections"] else []
    html_path = os.path.abspath(os.path.join(map_dir, f"{location}_autocritic.html"))
    with open(html_path, "w") as f:
        f.write(render_html(report=report, polylines=polylines, config=config, title=location))
    return json_path, html_path
