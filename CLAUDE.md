# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Cascade declarations (required by `~/.claude/CLAUDE.md` §1)

- **Purpose:** Experimental converter that turns OpenDRIVE (`.xodr`) maps into the artifact set TorchDriveSim / the Inverted AI API consumes: a Lanelet2 `.osm`, a stoplines JSON, a road-mesh JSON, a `metadata.json` (`MapConfig`), and a top-down `visualization.png`.
- **Audience:** Internal IAI tool. Not a client-facing SDK. Run ad-hoc by humans converting individual maps; no production deployment.
- **Primary language:** Python (3.10–3.13). Linux only.
- **BC stance:** §6 backwards-compatibility does **not** apply. The only "public surface" we own is the `convert_map.py` invocation shape. Everything under `crdesigner/` is vendored upstream and is not our public API.
- **Overrides of global rules:**
  - **§5 OICO / consolidation:** Do not refactor or de-duplicate inside `crdesigner/` to make it tidier. That directory is a near-verbatim snapshot of upstream — refactors there cost us on the next merge from upstream. OICO applies normally to repo-root scripts and any new IAI-specific files.
  - **§3 style / §4 keyword-args:** Match upstream style inside `crdesigner/` even where it conflicts with the global PEP8 / keyword-only conventions. Apply the global style rules to repo-root IAI scripts only.
  - **§7 TDD:** The `tests/` suite covers upstream behavior; there is no IAI-specific test harness for `convert_map.py`. Changes to `convert_map.py` fall under §7 case (b)/(c) — confirm with the user before declaring "no new tests needed."

## Repo shape

This repo is a fork of [CommonRoad Scenario Designer](https://github.com/CommonRoad/commonroad-scenario-designer) v0.8.5. Most code is vendored upstream; the IAI-specific surface is small.

- `convert_map.py` — **the IAI entry point.** OpenDRIVE → CommonRoad → Lanelet2 → TorchDriveSim artifacts. See "Conversion pipeline" below.
- `convert_lanelet_to_opendrive.py` — reverse direction via CARLA's tools.
- `crdesigner/` — **vendored** CommonRoad Scenario Designer. Treat as upstream.
- `tests/` — upstream pytest suite (pytest + pytest-qt).
- `maps/`, `xodr_output/`, `carla_xodr_path/` — local working dirs for map data, untracked. Do not commit map binaries.

## Common commands

- Install deps: `poetry install --with tests` (add `docs,tutorials` if needed).
- **Run a conversion:** `python convert_map.py dir_path=<folder-with-single-.xodr>`. Other `MapConversionConfig` fields are passed as OmegaConf dotlist args: `domain=`, `center='[x,y]'`, `trim_radius=`, `trim_vertical_limits='[lo,hi]'`, `visualization_fov=`, `lane_marking_join_threshold=`.
- Tests (whole suite): `poetry run pytest tests/`.
- Tests with coverage (matches CI): `poetry run coverage run --source ./crdesigner -m pytest tests/`.
- Single test: `poetry run pytest tests/path/to/test_file.py::TestClass::test_name`.
- Lint: `ruff check .` and `ruff format --check .`. Config in `pyproject.toml` (line-length 100, double quotes, rules `E4,E7,E9,I`). The ANTLR-generated parser at `crdesigner/verification_repairing/verification/hol/parser/gen/` is excluded — do not lint or reformat it.
- GUI (upstream, not the IAI workflow): `python crdesigner/ui/gui/start_gui.py`. Requires system Qt libs and is rarely needed for IAI work.

The upstream `crdesigner` CLI exists (`crdesigner odrcr --input-file ... --output-file ...`) but the IAI workflow uses `convert_map.py` because the upstream CLI does not emit the TorchDriveSim artifacts (mesh, stoplines JSON, MapConfig).

## Conversion pipeline (`convert_map.py`)

A future Claude editing this file needs the big picture, since it spans `crdesigner/` and `torchdrivesim` boundaries:

1. Read a single `.xodr` from `cfg.dir_path` (the filename, minus extension, becomes the map name).
2. Parse the OpenDRIVE header to extract a `+proj=tmerc` geo-reference (lat/lon) and an optional inertial-frame offset. Falls back to `(0, 0)` with a warning if absent.
3. `parse_opendrive` (`crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.parser`) → `Network.load_opendrive` → `Network.export_lanelet_network` builds a CommonRoad lanelet network in the OpenDRIVE inertial frame. `filter_types` controls which OpenDRIVE lane types survive (default: `driving`, `onRamp`, `offRamp`, `exit`, `entry`).
4. `CR2LaneletConverter` (`crdesigner/map_conversion/lanelet2/cr2lanelet.py`) with a `CustomTransfomer` wrapping a Lanelet2 `UtmProjector` writes `<map>.osm`.
5. If `trim_radius` is set, trim the Lanelet2 map to a circle around `center` (and an optional elevation band via `trim_vertical_limits`).
6. `torchdrivesim.lanelet2.road_mesh_from_lanelet_map` / `lanelet_map_to_lane_mesh` build a road + lane-marking `BirdviewMesh`; written to `<map>_mesh.json`.
7. Stoplines: pull `traffic_light.iai_stoplines` from each traffic light, plus `Network._iai_stop_signs` / `_iai_yield_signs`. Apply trim filters. Written to `<map>_stoplines.json`.
8. Write `MapConfig` to `metadata.json` and render `visualization.png` via `torchdrivesim.rendering`.

All outputs land in `cfg.dir_path` alongside the input `.xodr`.

## Where the IAI seams live

These are the names/files that diverge from upstream. Search for them when changing the OpenDRIVE → CommonRoad path:

- `convert_map.py`, `convert_lanelet_to_opendrive.py` — IAI-authored, not upstream.
- `crdesigner/map_conversion/lanelet2/cr2lanelet.py` — small IAI tweaks (import / type-hint cleanup).
- Custom attributes on upstream objects: `iai_stoplines`, `_iai_stop_signs`, `_iai_yield_signs`, `opendrive_id`. Grep for `iai_` inside `crdesigner/` to find where they're attached.
- `git diff master -- crdesigner/` (or whatever the upstream-tracking branch is) is the authoritative way to enumerate IAI deltas.

## Dependencies that are not a single `pip install`

- `lanelet2` — system / conda package, not pure PyPI. If `import lanelet2` fails, that's the cause.
- `torchdrivesim` — Inverted AI internal package. Provides `Stopline`, `MapConfig`, `BirdviewMesh`, the renderer, and mesh utilities used at the top of `convert_map.py`.
- CI (`.github/workflows/python-app.yml`) runs Ubuntu only on Python 3.10–3.13. macOS and Windows are not supported by upstream.
