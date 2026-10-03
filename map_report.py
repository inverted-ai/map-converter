"""Convert a folder of OpenDRIVE maps and report which converted, and how well.

``run`` converts every ``.xodr`` under ``--maps`` with a converter checkout, one child process
per map, and records the outcome, time, peak memory, input statistics, the
:mod:`map_quality` metrics and a digest of every output file. ``compare`` lists every metric
that moved between two runs, e.g. two converter versions on the same maps.

Usage::

    python /abs/clone/map_report.py run --maps /abs/maps --out /abs/report_a
    python /abs/clone/map_report.py run --maps /abs/maps --out /abs/report_b --converter /abs/other_clone
    python /abs/clone/map_report.py compare /abs/report_a/report.json /abs/report_b/report.json --out /abs/cmp

Outputs (all paths printed absolute): ``<out>/report.json`` and ``<out>/report.html`` for
``run``; ``<out>/compare.json`` and ``<out>/compare.html`` for ``compare``. Each map is
converted in ``<out>/maps/<relative path without .xodr>/``, and those outputs are kept.
"""

from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import datetime
import glob
import hashlib
import html
import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Optional

from lxml import etree

import map_quality
from traffic_light_fsm import load_metadata, resolve_map_path

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
XML_COUNTED_TAGS = ("controller", "control", "signal", "signalReference", "object")
CORE_OUTPUT_SUFFIXES = (".osm", "_stoplines.json", "metadata.json")
EXCLUDED_OUTPUT_SUFFIXES = (".png", ".xodr", ".log")
NEVER_MOVES = ("wall_s", "peak_rss_mb")
FLOAT_MOVE_FRACTION = 0.01
THUMBNAIL_PX = 384
POLL_S = 0.2


# --------------------------------------------------------------------------------------
# Child processes
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class ChildResult:
    """How one child process ended."""

    returncode: Optional[int]
    timed_out: bool
    wall_s: float
    peak_rss_mb: float
    stdout: str
    stderr: str


def run_child(*, argv: list[str], cwd: str, timeout_s: float, log_prefix: str) -> ChildResult:
    """Run ``argv`` with stdin closed and ``PYTHONHASHSEED=0``; kill its process group on timeout.

    Peak RSS comes from ``os.wait4`` on this child alone, so siblings never inflate it.
    stdout/stderr are written to ``<log_prefix>.stdout.log`` / ``.stderr.log``.
    """
    env = dict(os.environ, PYTHONHASHSEED="0")
    stdout_path = f"{log_prefix}.stdout.log"
    stderr_path = f"{log_prefix}.stderr.log"
    start = time.monotonic()
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        timed_out = False
        while True:
            pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() - start > timeout_s:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                _, status, rusage = os.wait4(proc.pid, 0)
                break
            time.sleep(POLL_S)
    # wait4 already reaped the child; tell Popen so it never waits on the pid again.
    proc.returncode = returncode = os.waitstatus_to_exitcode(status)
    return ChildResult(
        returncode=None if timed_out else returncode,
        timed_out=timed_out,
        wall_s=round(time.monotonic() - start, 3),
        peak_rss_mb=round(rusage.ru_maxrss / 1024.0, 1),
        stdout=_read_text(path=stdout_path),
        stderr=_read_text(path=stderr_path),
    )


def _read_text(*, path: str) -> str:
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


_FRAME_RE = re.compile(r'^\s*File "([^"]+)", line (\d+)', re.MULTILINE)
_EXCEPTION_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Quit|Warning)?)(?::|$)")


def classify(*, result: ChildResult, root: str) -> dict[str, Any]:
    """Outcome of a conversion child: ``ok``, ``exception``, ``timeout`` or ``pdb``.

    pdb prints its ``(Pdb)`` prompt on stdout and, with stdin closed, quits via ``BdbQuit``,
    so both streams are checked. For an exception, the class and the innermost frame
    (relative to ``root`` when inside it) are recorded.
    """
    if result.timed_out:
        return {"outcome": "timeout"}
    text = result.stdout + result.stderr
    if "(Pdb)" in text or "bdb.BdbQuit" in text:
        return {"outcome": "pdb", **_last_frame(stderr=result.stderr, root=root)}
    if result.returncode == 0:
        return {"outcome": "ok"}
    return {
        "outcome": "exception",
        "exception": _exception_class(stderr=result.stderr),
        **_last_frame(stderr=result.stderr, root=root),
    }


def _last_frame(*, stderr: str, root: str) -> dict[str, Optional[str]]:
    frames = _FRAME_RE.findall(stderr)
    if not frames:
        return {"location": None}
    path, line = frames[-1]
    if os.path.isabs(path) and os.path.commonpath([path, root]) == root:
        path = os.path.relpath(path, root)
    return {"location": f"{path}:{line}"}


def _exception_class(*, stderr: str) -> Optional[str]:
    lines = [ln for ln in stderr.strip().splitlines() if ln and not ln.startswith(" ")]
    for line in reversed(lines):
        match = _EXCEPTION_RE.match(line)
        if match and not line.startswith(("Traceback", "File ")):
            return match.group(1)
    return None


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def sha256_file(*, path: str) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xodr_statistics(*, path: str) -> dict[str, Any]:
    """Header identity and element counts of one ``.xodr``, streamed so large files fit.

    Returns:
        ``vendor``, ``rev_major``, ``rev_minor``, ``counts`` (per tag in
        :data:`XML_COUNTED_TAGS`) and ``road_marks`` (``roadMark@type`` → count).
    """
    counts = dict.fromkeys(XML_COUNTED_TAGS, 0)
    road_marks: collections.Counter[str] = collections.Counter()
    header: dict[str, Optional[str]] = {}
    for _, elem in etree.iterparse(path, events=("end",), huge_tree=True, recover=True):
        tag = etree.QName(elem).localname if isinstance(elem.tag, str) else ""
        if tag in counts:
            counts[tag] += 1
        elif tag == "roadMark":
            road_marks[elem.get("type", "")] += 1
        elif tag == "header" and not header:
            header = {k: elem.get(k) for k in ("vendor", "revMajor", "revMinor")}
        if tag != "header":
            elem.clear()
    return {
        "vendor": header.get("vendor") or None,
        "rev_major": header.get("revMajor"),
        "rev_minor": header.get("revMinor"),
        "counts": counts,
        "road_marks": dict(sorted(road_marks.items())),
    }


_HANDLED_MARKS_SNIPPET = """
import json, sys, types
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_conversion.plane_elements.plane_group import convert_line_marking
from commonroad.scenario.lanelet import LineMarking
types_ = json.loads(sys.argv[1])
print(json.dumps(sorted(t for t in types_ if convert_line_marking(types.SimpleNamespace(type=t, weight="standard")) != LineMarking.UNKNOWN)))
"""


def handled_road_mark_types(*, converter: str, mark_types: list[str]) -> list[str]:
    """Subset of ``mark_types`` the converter tree's road-mark mapper recognises."""
    if not mark_types:
        return []
    completed = subprocess.run(
        [sys.executable, "-c", _HANDLED_MARKS_SNIPPET, json.dumps(sorted(mark_types))],
        cwd=converter,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------------------
# Output digests
# --------------------------------------------------------------------------------------


def canonical_mesh_digest(*, path: str) -> str:
    """sha256 of a ``_mesh.json`` with categories sorted and ``vert_category`` remapped.

    torchdrivesim's ``merge`` orders categories nondeterministically; geometry is unaffected.
    """
    with open(path) as f:
        mesh = json.load(f)
    categories = mesh.get("categories", [])
    order = sorted(range(len(categories)), key=lambda i: categories[i])
    remap = {old: new for new, old in enumerate(order)}
    mesh["categories"] = [categories[i] for i in order]
    mesh["vert_category"] = [[remap.get(c, c) for c in row] for row in mesh.get("vert_category", [])]
    return hashlib.sha256(json.dumps(mesh, sort_keys=True).encode()).hexdigest()


def output_digests(*, map_dir: str) -> dict[str, str]:
    """Digest of every output in ``map_dir`` except the PNG, provenance, logs and FSM output.

    ``metadata.json``, the ``.osm`` and ``_stoplines.json`` are hashed raw; ``_mesh.json``
    canonically; any other JSON sidecar after replacing ``map_dir`` with ``<RUN_DIR>``.
    Keys are file names with the map stem replaced by ``<map>``.
    """
    digests = {}
    for path in sorted(glob.glob(os.path.join(map_dir, "*"))):
        name = os.path.basename(path)
        if not os.path.isfile(path) or name.endswith(EXCLUDED_OUTPUT_SUFFIXES):
            continue
        if name.endswith("_mesh.json"):
            digest = canonical_mesh_digest(path=path)
        elif name.endswith(CORE_OUTPUT_SUFFIXES):
            digest = sha256_file(path=path)
        else:
            text = _read_text(path=path).replace(map_dir, "<RUN_DIR>")
            digest = hashlib.sha256(text.encode()).hexdigest()
        digests[_generic_name(name=name, map_dir=map_dir)] = digest
    return digests


def _generic_name(*, name: str, map_dir: str) -> str:
    stem = os.path.basename(map_dir)
    return "<map>" + name[len(stem):] if name.startswith(stem) else name


# --------------------------------------------------------------------------------------
# FSM
# --------------------------------------------------------------------------------------


def fsm_summary(*, controller_path: str) -> dict[str, Any]:
    """Groupings recovered from a traffic-light controller file.

    A phase group is the set of lights green together in some state. The grouping digest is
    the sha256 of every intersection's sorted groups, so it ignores timings and ordering.
    """
    with open(controller_path) as f:
        fsms = json.load(f)
    intersections = []
    for fsm in fsms:
        groups = {
            tuple(sorted(aid for aid, colour in state["actor_states"].items() if colour == "green"))
            for state in fsm
        }
        intersections.append(sorted(list(g) for g in groups if g))
    intersections.sort()
    return {
        "intersections": len(intersections),
        "phase_groups": sum(len(groups) for groups in intersections),
        "lights": sum(len(g) for groups in intersections for g in groups),
        "max_phases": max((len(groups) for groups in intersections), default=0),
        "grouping_digest": hashlib.sha256(json.dumps(intersections).encode()).hexdigest(),
    }


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


def discover_maps(*, maps_dir: str, only: Optional[list[str]]) -> list[str]:
    """Relative paths of every ``.xodr`` under ``maps_dir`` (sorted), optionally filtered."""
    found = sorted(
        os.path.relpath(p, maps_dir)
        for p in glob.glob(os.path.join(maps_dir, "**", "*.xodr"), recursive=True)
    )
    if only:
        wanted = set(only)
        missing = wanted - set(found)
        if missing:
            raise FileNotFoundError(f"--map not found under {maps_dir}: {sorted(missing)}")
        found = [p for p in found if p in wanted]
    return found


def convert_one(
    *, xodr_path: str, map_dir: str, converter: str, timeout_s: float, with_fsm: bool
) -> dict[str, Any]:
    """Convert one map in a fresh ``map_dir`` and measure it."""
    if os.path.exists(map_dir):
        shutil.rmtree(map_dir)
    os.makedirs(map_dir)
    shutil.copy2(xodr_path, os.path.join(map_dir, os.path.basename(xodr_path)))
    result = run_child(
        argv=[sys.executable, os.path.join(converter, "convert_map.py"), f"dir_path={map_dir}"],
        cwd=map_dir,
        timeout_s=timeout_s,
        log_prefix=os.path.join(map_dir, "convert"),
    )
    record: dict[str, Any] = {
        "conversion": {
            **classify(result=result, root=converter),
            "returncode": result.returncode,
            "wall_s": result.wall_s,
            "peak_rss_mb": result.peak_rss_mb,
        },
        "outputs_dir": map_dir,
        "metrics": {},
        "metric_errors": {},
        "digests": {},
    }
    if not os.path.exists(os.path.join(map_dir, "metadata.json")):
        return record
    record["digests"] = output_digests(map_dir=map_dir)
    try:
        outputs = map_quality.ConversionOutputs.load(map_dir=map_dir)
    except Exception as exc:  # noqa: BLE001 - a report records failures, it never aborts
        record["metric_errors"]["load"] = f"{type(exc).__name__}: {exc}"
        return record
    record["metrics"], record["metric_errors"] = map_quality.compute_all(outputs=outputs)
    del outputs  # release the loaded map before the FSM child needs the memory
    if with_fsm and record["conversion"]["outcome"] == "ok":
        record["fsm"] = run_fsm(map_dir=map_dir, converter=converter, timeout_s=timeout_s)
    return record


def run_fsm(*, map_dir: str, converter: str, timeout_s: float) -> dict[str, Any]:
    """Run the converter tree's ``traffic_light_fsm.py`` on a converted map and summarise it."""
    result = run_child(
        argv=[sys.executable, os.path.join(converter, "traffic_light_fsm.py"), map_dir],
        cwd=map_dir,
        timeout_s=timeout_s,
        log_prefix=os.path.join(map_dir, "fsm"),
    )
    fsm: dict[str, Any] = {
        **classify(result=result, root=converter),
        "wall_s": result.wall_s,
        "peak_rss_mb": result.peak_rss_mb,
    }
    controller = load_metadata(map_dir=map_dir).get("traffic_light_controller_path")
    if controller:
        fsm.update(fsm_summary(controller_path=resolve_map_path(map_dir=map_dir, name=controller)))
    return fsm


def git_identity(*, repo: str) -> dict[str, Any]:
    """Commit and dirty flag of a checkout; ``None`` values when it is not a git checkout."""
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if head.returncode != 0:
        return {"commit": None, "dirty": None}
    status = subprocess.run(
        ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


def run_report(
    *,
    maps_dir: str,
    out_dir: str,
    converter: str,
    only: Optional[list[str]],
    timeout_s: float,
    with_fsm: bool,
) -> str:
    """Convert and measure every map; write ``report.json`` and ``report.html``.

    ``report.json`` is rewritten after each map, so an interrupted run keeps its progress.

    Returns:
        Absolute path of ``report.json``.
    """
    maps_dir = os.path.abspath(maps_dir)
    out_dir = os.path.abspath(out_dir)
    converter = os.path.abspath(converter)
    if not os.path.isfile(os.path.join(converter, "convert_map.py")):
        raise FileNotFoundError(f"No convert_map.py in --converter {converter}")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "report.json")
    report: dict[str, Any] = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "maps_dir": maps_dir,
        "out_dir": out_dir,
        "converter": converter,
        "converter_git": git_identity(repo=converter),
        "ruler": REPO_ROOT,
        "ruler_git": git_identity(repo=REPO_ROOT),
        "python": sys.version.split()[0],
        "timeout_s": timeout_s,
        "fsm": with_fsm,
        "maps": [],
    }
    first_by_sha: dict[str, str] = {}
    for index, rel in enumerate(discover_maps(maps_dir=maps_dir, only=only), start=1):
        xodr_path = os.path.join(maps_dir, rel)
        sha = sha256_file(path=xodr_path)
        logger.info("[%d] converting %s", index, xodr_path)
        record = {
            "relpath": rel,
            "input": {
                "path": xodr_path,
                "sha256": sha,
                "duplicate_of": first_by_sha.get(sha),
                "size_bytes": os.path.getsize(xodr_path),
                **xodr_statistics(path=xodr_path),
            },
            **convert_one(
                xodr_path=xodr_path,
                map_dir=os.path.join(out_dir, "maps", rel[: -len(".xodr")]),
                converter=converter,
                timeout_s=timeout_s,
                with_fsm=with_fsm,
            ),
        }
        first_by_sha.setdefault(sha, rel)
        logger.info(
            "    %s in %.1f s, peak %.0f MB",
            record["conversion"]["outcome"],
            record["conversion"]["wall_s"],
            record["conversion"]["peak_rss_mb"],
        )
        report["maps"].append(record)
        _write_json(path=report_path, data=report)
    seen = sorted({t for m in report["maps"] for t in m["input"]["road_marks"]})
    report["road_mark_types_handled"] = handled_road_mark_types(converter=converter, mark_types=seen)
    _write_json(path=report_path, data=report)
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w") as f:
        f.write(render_report_html(report=report))
    print(f"wrote {report_path}")
    print(f"wrote {html_path}")
    return report_path


def _write_json(*, path: str, data: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------------------


def flatten(*, record: dict[str, Any]) -> dict[str, Any]:
    """Comparable scalar fields of one map record, keyed by dotted path.

    Run-specific fields (paths, logs, return codes) are left out.
    """
    flat: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, sub in value.items():
                walk(f"{prefix}.{key}" if prefix else key, sub)
        else:
            flat[prefix] = value

    for section in ("conversion", "metrics", "digests", "fsm"):
        if section in record:
            walk(section, record[section])
    flat.pop("conversion.returncode", None)
    flat["input.sha256"] = record["input"]["sha256"]
    return flat


def moved(*, key: str, before: Any, after: Any) -> bool:
    """Whether a field moved: floats beyond 1%, everything else on any change; time/RSS never."""
    if key.endswith(NEVER_MOVES):
        return False
    if isinstance(before, float) or isinstance(after, float):
        if before is None or after is None:
            return before != after
        if before == 0:
            return after != 0
        return abs(after - before) / abs(before) > FLOAT_MOVE_FRACTION
    return before != after


def compare_reports(*, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Every field that moved, per map present in both reports."""
    before_maps = {m["relpath"]: m for m in before["maps"]}
    after_maps = {m["relpath"]: m for m in after["maps"]}
    maps: dict[str, Any] = {}
    for rel in sorted(before_maps.keys() & after_maps.keys()):
        b, a = flatten(record=before_maps[rel]), flatten(record=after_maps[rel])
        moves = {
            key: [b.get(key), a.get(key)]
            for key in sorted(b.keys() | a.keys())
            if moved(key=key, before=b.get(key), after=a.get(key))
        }
        maps[rel] = {
            "moved": moves,
            "time_rss": {
                key: [b.get(key), a.get(key)]
                for key in ("conversion.wall_s", "conversion.peak_rss_mb")
            },
            "thumbnails": [
                os.path.join(before_maps[rel]["outputs_dir"], "visualization.png"),
                os.path.join(after_maps[rel]["outputs_dir"], "visualization.png"),
            ],
        }
    moved_keys = collections.Counter(k for m in maps.values() for k in m["moved"])
    return {
        "before": before.get("out_dir"),
        "after": after.get("out_dir"),
        "only_in_before": sorted(before_maps.keys() - after_maps.keys()),
        "only_in_after": sorted(after_maps.keys() - before_maps.keys()),
        "maps_with_moves": sorted(rel for rel, m in maps.items() if m["moved"]),
        "moved_key_counts": dict(moved_keys.most_common()),
        "maps": maps,
    }


def largest_movers(*, comparison: dict[str, Any], limit: int = 20) -> list[tuple[str, str, Any, Any, float]]:
    """Numeric moves ranked by relative change (map, key, before, after, relative change)."""
    rows = []
    for rel, entry in comparison["maps"].items():
        for key, (b, a) in entry["moved"].items():
            if isinstance(b, (int, float)) and isinstance(a, (int, float)) and not isinstance(b, bool):
                rel_change = abs(a - b) / abs(b) if b else float("inf")
                rows.append((rel, key, b, a, rel_change))
    rows.sort(key=lambda r: -r[4])
    return rows[:limit]


def run_compare(*, before_path: str, after_path: str, out_dir: str) -> str:
    """Compare two ``report.json`` files; write ``compare.json`` and ``compare.html``.

    Returns:
        Absolute path of ``compare.json``.
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)
    comparison = compare_reports(before=before, after=after)
    comparison["before_report"] = os.path.abspath(before_path)
    comparison["after_report"] = os.path.abspath(after_path)
    json_path = os.path.join(out_dir, "compare.json")
    _write_json(path=json_path, data=comparison)
    html_path = os.path.join(out_dir, "compare.html")
    with open(html_path, "w") as f:
        f.write(render_compare_html(comparison=comparison))
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    print(f"{len(comparison['maps_with_moves'])} of {len(comparison['maps'])} maps moved")
    return json_path


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------

_CSS = """
:root{--bg:#fbfbfa;--fg:#1d2327;--muted:#5f6b70;--line:#d8dddc;--ok:#2f7d4f;--bad:#b3261e;--warn:#9a6a00}
@media (prefers-color-scheme:dark){:root{--bg:#141819;--fg:#e3e8e8;--muted:#9aa7aa;--line:#2c3437;--ok:#7fcf9d;--bad:#ff8f86;--warn:#e6c063}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}
.meta{color:var(--muted);font-size:12.5px;margin-bottom:14px;word-break:break-all}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}
th{position:sticky;top:0;background:var(--bg);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--ok);font-weight:600}.exception,.pdb,.timeout{color:var(--bad);font-weight:600}
img.thumb{width:160px;height:160px;object-fit:contain;border:1px solid var(--line);border-radius:4px;background:#fff}
.charts img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff;margin:4px 0}
code{font-size:12px}
"""


def _thumbnail_data_uri(*, path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    from PIL import Image

    with Image.open(path) as image:
        image.thumbnail((THUMBNAIL_PX, THUMBNAIL_PX))
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _img(*, path: str) -> str:
    uri = _thumbnail_data_uri(path=path)
    return f'<img class="thumb" src="{uri}" alt="">' if uri else "—"


def _fmt(*, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, str) and len(value) == 64:
        return f"<code>{value[:10]}…</code>"
    return html.escape(str(value))


def _page(*, title: str, body: str) -> str:
    return (
        f"<!doctype html><html><head><meta charset=utf-8>"
        f'<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>{body}</body></html>"
    )


OUTCOME_COLOURS = {"ok": "#0ca30c", "exception": "#d03b3b", "timeout": "#ec835a", "pdb": "#fab219"}
SERIES_COLOUR = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
DISTRIBUTIONS = (
    ("routing %", lambda m: m.get("metrics", {}).get("routing_edge_pct")),
    ("seam handshake %", lambda m: m.get("metrics", {}).get("seam_handshake_pct")),
    ("z-conflicts", lambda m: (m.get("metrics", {}).get("z_conflicts") or {}).get("count")),
    ("OSM MB", lambda m: _scaled(value=m.get("metrics", {}).get("osm_size_bytes"), factor=1e-6)),
    ("conversion s", lambda m: m["conversion"]["wall_s"]),
    ("peak RSS GB", lambda m: m["conversion"]["peak_rss_mb"] / 1024),
)


def _scaled(*, value: Optional[float], factor: float) -> Optional[float]:
    return value * factor if value else None


def _png_data_uri(*, fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _style_axes(*, ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8dddc")
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(axis="x", color="#ecefee", linewidth=0.8)
    ax.set_axisbelow(True)


def report_charts_html(*, report: dict[str, Any]) -> str:
    """Outcome-by-vendor bars and per-metric distributions as embedded PNGs.

    Duplicate inputs are counted once so a map copied into two folders does not weigh double.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique = [m for m in report["maps"] if not m["input"]["duplicate_of"]]
    if not unique:
        return ""
    by_vendor: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for m in unique:
        by_vendor[m["input"]["vendor"] or "none declared"][m["conversion"]["outcome"]] += 1
    vendors = sorted(by_vendor, key=lambda v: (-sum(by_vendor[v].values()), v))
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(vendors) + 1.2))
    _style_axes(ax=ax)
    lefts = [0] * len(vendors)
    for outcome, colour in OUTCOME_COLOURS.items():
        widths = [by_vendor[v][outcome] for v in vendors]
        if not any(widths):
            continue
        bars = ax.barh(vendors, widths, left=lefts, color=colour, edgecolor=SURFACE, linewidth=2, label=outcome, height=0.6)
        for bar, width in zip(bars, widths):
            if width:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(width),
                        ha="center", va="center", fontsize=8, color=INK)
        lefts = [a + b for a, b in zip(lefts, widths)]
    ax.invert_yaxis()
    ax.set_xlabel("unique maps", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Outcome by declared vendor", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower right", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    parts = [f'<img alt="Outcome by vendor" src="{_png_data_uri(fig=fig)}">']

    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6))
    for ax, (label, getter) in zip(axes.flat, DISTRIBUTIONS):
        _style_axes(ax=ax)
        values = sorted(v for v in (getter(m) for m in unique) if v is not None)
        ys = [((i * 0.37) % 1.0) - 0.5 for i in range(len(values))]
        ax.scatter(values, ys, s=22, color=SERIES_COLOUR, edgecolor=SURFACE, linewidth=1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.9, 0.9)
        ax.set_title(f"{label} (n={len(values)})", loc="left", color=INK, fontsize=9)
    fig.suptitle("Distributions over unique maps (one dot per map)", x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    parts.append(f'<img alt="Metric distributions" src="{_png_data_uri(fig=fig)}">')
    return "".join(parts)


def render_report_html(*, report: dict[str, Any]) -> str:
    """Self-contained HTML for a run: per-map table, charts and thumbnails."""
    rows = []
    for m in report["maps"]:
        conv = m["conversion"]
        metrics = m.get("metrics", {})
        z = metrics.get("z_conflicts") or {}
        crossed = metrics.get("crossed_geometry") or {}
        stoplines = metrics.get("stoplines_by_agent_type") or {}
        fsm = m.get("fsm") or {}
        detail = conv.get("exception") or ""
        if conv.get("location"):
            detail += f" @ {conv['location']}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(m['relpath'])}{' (dup)' if m['input']['duplicate_of'] else ''}</td>"
            f"<td>{html.escape(str(m['input']['vendor']))}</td>"
            f"<td class='{conv['outcome']}'>{conv['outcome']}</td><td>{html.escape(detail)}</td>"
            f"<td class=n>{conv['wall_s']:.1f}</td><td class=n>{conv['peak_rss_mb']:.0f}</td>"
            f"<td class=n>{_fmt(value=metrics.get('lanelet_count'))}</td>"
            f"<td class=n>{_fmt(value=metrics.get('routing_edge_pct'))}</td>"
            f"<td class=n>{_fmt(value=metrics.get('seam_handshake_pct'))}</td>"
            f"<td class=n>{_fmt(value=z.get('count'))}</td>"
            f"<td class=n>{_fmt(value=crossed.get('reversed_bounds'))}/{_fmt(value=crossed.get('invalid_polygons'))}</td>"
            f"<td class=n>{_fmt(value=sum(stoplines.values()))}</td>"
            f"<td class=n>{_fmt(value=m['input']['counts']['controller'])}</td>"
            f"<td class=n>{_fmt(value=fsm.get('intersections'))}</td>"
            f"<td>{_img(path=os.path.join(m['outputs_dir'], 'visualization.png'))}</td>"
            "</tr>"
        )
    header = (
        "<tr><th>map</th><th>vendor</th><th>outcome</th><th>detail</th><th>time s</th>"
        "<th>peak MB</th><th>lanelets</th><th>routing %</th><th>seam %</th><th>z-conflicts</th>"
        "<th>crossed rev/invalid</th><th>stoplines</th><th>&lt;controller&gt;</th>"
        "<th>FSM intersections</th><th>visualization</th></tr>"
    )
    outcomes = collections.Counter(m["conversion"]["outcome"] for m in report["maps"])
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(outcomes.items()))
    body = (
        f"<h1>Map conversion report</h1><div class=meta>converter {html.escape(report['converter'])} "
        f"@ {html.escape(str(report['converter_git']['commit']))}"
        f"{' (dirty)' if report['converter_git']['dirty'] else ''} · maps {html.escape(report['maps_dir'])}"
        f" · {len(report['maps'])} maps · {summary} · {html.escape(report['created_utc'])}</div>"
        f"<h2>Charts</h2><div class=charts>{report_charts_html(report=report)}</div>"
        f"<h2>Maps</h2><div class=wrap><table>{header}{''.join(rows)}</table></div>"
    )
    return _page(title="Map conversion report", body=body)


def render_compare_html(*, comparison: dict[str, Any]) -> str:
    """Self-contained HTML listing every moved field per map with before/after thumbnails."""
    movers = "".join(
        f"<tr><td>{html.escape(rel)}</td><td><code>{html.escape(key)}</code></td>"
        f"<td class=n>{_fmt(value=b)}</td><td class=n>{_fmt(value=a)}</td>"
        f"<td class=n>{'new' if change == float('inf') else f'{change:+.1%}'}</td></tr>"
        for rel, key, b, a, change in largest_movers(comparison=comparison)
    )
    sections = []
    for rel, entry in comparison["maps"].items():
        (wb, wa), (rb, ra) = entry["time_rss"]["conversion.wall_s"], entry["time_rss"]["conversion.peak_rss_mb"]
        moves = "".join(
            f"<tr><td><code>{html.escape(key)}</code></td><td>{_fmt(value=b)}</td><td>{_fmt(value=a)}</td></tr>"
            for key, (b, a) in entry["moved"].items()
        ) or "<tr><td colspan=3>no moves</td></tr>"
        thumbs = (
            f"{_img(path=entry['thumbnails'][0])} {_img(path=entry['thumbnails'][1])}"
            if entry["moved"]
            else ""
        )
        sections.append(
            f"<h2>{html.escape(rel)}</h2><div class=meta>time {_fmt(value=wb)} → {_fmt(value=wa)} s · "
            f"peak RSS {_fmt(value=rb)} → {_fmt(value=ra)} MB</div>{thumbs}"
            f"<div class=wrap><table><tr><th>field</th><th>before</th><th>after</th></tr>{moves}</table></div>"
        )
    body = (
        f"<h1>Map report comparison</h1><div class=meta>before {html.escape(str(comparison['before_report']))}<br>"
        f"after {html.escape(str(comparison['after_report']))}<br>{len(comparison['maps_with_moves'])} of "
        f"{len(comparison['maps'])} maps moved · only in before: {len(comparison['only_in_before'])} · "
        f"only in after: {len(comparison['only_in_after'])}</div>"
        f"<h2>Largest numeric movers</h2><div class=wrap><table><tr><th>map</th><th>field</th>"
        f"<th>before</th><th>after</th><th>change</th></tr>{movers or '<tr><td colspan=5>none</td></tr>'}"
        f"</table></div>{''.join(sections)}"
    )
    return _page(title="Map report comparison", body=body)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Convert and measure every .xodr under --maps")
    run.add_argument("--maps", required=True, help="Directory searched recursively for .xodr files")
    run.add_argument("--out", required=True, help="Output directory for report.json, report.html and maps/")
    run.add_argument("--converter", default=REPO_ROOT, help="Converter checkout to run (default: this clone)")
    run.add_argument("--map", action="append", dest="only", help="Only this .xodr (path relative to --maps); repeatable")
    run.add_argument("--timeout-s", type=float, default=1800.0, help="Per-map timeout in seconds (default 1800)")
    run.add_argument("--fsm", action="store_true", help="Also run traffic_light_fsm.py on each converted map")
    compare = sub.add_parser("compare", help="List every metric that moved between two runs")
    compare.add_argument("before", help="report.json of the earlier run")
    compare.add_argument("after", help="report.json of the later run")
    compare.add_argument("--out", required=True, help="Output directory for compare.json and compare.html")
    return parser


def main() -> None:
    """CLI entry point for the ``run`` and ``compare`` subcommands."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()
    if args.command == "run":
        run_report(
            maps_dir=args.maps,
            out_dir=args.out,
            converter=args.converter,
            only=args.only,
            timeout_s=args.timeout_s,
            with_fsm=args.fsm,
        )
    else:
        run_compare(before_path=args.before, after_path=args.after, out_dir=args.out)


if __name__ == "__main__":
    main()
