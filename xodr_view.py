"""The raw OpenDRIVE map as CARLA's OpenDRIVE reader sees it: lane outlines by lane type and signal positions.

An independent reading of the ``.xodr`` (it shares no code with the vendored converter), drawn beside the
converted map in the AutoCritic report so a difference between the two points at a reader or conversion error.
CARLA's Python package (``carla``, optional: no wheel for Python 3.13) reads the file without a simulator; it
runs in a child process (:func:`read`) because its C++ reader can abort on a malformed file.

Coordinates are the file's own ``(x, y)`` (CARLA's mirrored y is undone); the caller maps them into its frame.

usage (child process): xodr_view.py <xodr> <out json>
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
import tempfile
from typing import Optional

STEP_M = 1.0
END_STEP_M = 0.01  # how close to its end a lane piece is drawn
TIMEOUT_S = 600.0


@dataclasses.dataclass(slots=True)
class RawMap:
    """What the reader found, or why it found nothing.

    Attributes:
        lanes: ``(lane type, outline)`` per lane piece; outline is a closed ``[[x, y], ...]`` ring.
        signals: ``(x, y)`` of every signal and landmark.
        error: Why the map could not be read (``None`` when it was).
    """

    lanes: list[tuple[str, list[list[float]]]]
    signals: list[list[float]]
    error: Optional[str] = None


def read(*, xodr_path: str, timeout_s: float = TIMEOUT_S) -> RawMap:
    """Read ``xodr_path`` with CARLA's reader in a child process."""
    with tempfile.TemporaryDirectory() as folder:
        out = os.path.join(folder, "raw.json")
        try:
            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__), os.path.abspath(xodr_path), out],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return RawMap(lanes=[], signals=[], error=f"the reader took longer than {timeout_s:.0f} s")
        if result.returncode != 0 or not os.path.exists(out):
            message = (result.stderr.strip().splitlines() or [f"exit code {result.returncode}"])[-1]
            return RawMap(lanes=[], signals=[], error=message[:300])
        with open(out) as f:
            data = json.load(f)
    return RawMap(lanes=[tuple(lane) for lane in data["lanes"]], signals=data["signals"])


def _read_here(*, xodr_path: str) -> dict:
    try:
        import carla
    except ImportError:
        raise SystemExit("the optional package carla is not installed (pip install carla; Python 3.10-3.12)")
    with open(xodr_path, encoding="utf-8", errors="replace") as f:
        carla_map = carla.Map("raw", f.read())
    pieces: dict[tuple, list] = {}
    for start in carla_map.generate_waypoints(STEP_M):
        for step in ("get_left_lane", "get_right_lane"):
            current, visited = start, set()
            # crossing the centre line turns left into right, so a walk can come back: stop at a visited lane
            while (
                current is not None
                and current.road_id == start.road_id
                and current.section_id == start.section_id
                and current.lane_id not in visited
            ):
                visited.add(current.lane_id)
                pieces.setdefault((current.road_id, current.section_id, current.lane_id), {})[round(current.s, 2)] = current
                current = getattr(current, step)()
    lanes = []
    for by_s in pieces.values():
        waypoints = [by_s[s] for s in sorted(by_s)]
        waypoints += _beyond(carla_map=carla_map, waypoint=waypoints[-1])
        if len(waypoints) < 2:
            continue
        left, right = [], []
        for w in waypoints:
            location, r, half = w.transform.location, w.transform.get_right_vector(), w.lane_width / 2
            left.append([location.x - r.x * half, -(location.y - r.y * half)])
            right.append([location.x + r.x * half, -(location.y + r.y * half)])
        ring = [p for p in left + right[::-1] if math.isfinite(p[0]) and math.isfinite(p[1])]  # the reader can give NaN widths
        if len(ring) >= 3:
            lanes.append((str(waypoints[0].lane_type), ring))
    signals = [
        [lm.transform.location.x, -lm.transform.location.y]
        for lm in carla_map.get_all_landmarks()
        if math.isfinite(lm.transform.location.x) and math.isfinite(lm.transform.location.y)
    ]
    return {"lanes": lanes, "signals": signals}


def _beyond(*, carla_map, waypoint) -> list:
    """The waypoint at the far end (largest s) of ``waypoint``'s lane piece, when the samples stopped short of it.

    Found by bisection with ``get_waypoint_xodr``, which returns None past the end (the reader's walks to a lane's
    end can abort the process)."""
    def at(s: float):
        w = carla_map.get_waypoint_xodr(waypoint.road_id, waypoint.lane_id, s)
        return w if w is not None and w.section_id == waypoint.section_id else None

    low, high = waypoint.s, waypoint.s + STEP_M
    if at(high) is not None:
        return []
    while high - low > END_STEP_M:
        middle = (low + high) / 2
        low, high = (middle, high) if at(middle) is not None else (low, middle)
    return [at(low)] if low > waypoint.s else []


if __name__ == "__main__":
    data = _read_here(xodr_path=sys.argv[1])
    with open(sys.argv[2], "w") as f:
        json.dump(data, f)
