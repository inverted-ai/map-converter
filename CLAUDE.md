# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Cascade declarations (required by `~/.claude/CLAUDE.md` §1)

- **Purpose:** Converter that turns OpenDRIVE (`.xodr`) maps into the artifact set TorchDriveSim / the Inverted AI API consumes: a Lanelet2 `.osm`, a stoplines JSON, a road-mesh JSON, a `metadata.json` (`MapConfig`), and a top-down `visualization.png`, plus tools for traffic-light FSMs and for measuring conversions.
- **Audience:** Public tool (`inverted-ai/map-converter` is a public repo). Customers clone it, install `requirements.txt` into a venv and run the scripts from the clone. It is never pip-installed or packaged.
- **Primary language:** Python (3.10–3.13). Linux x86_64 only.
- **BC stance:** §6 applies to the IAI surface: the `convert_map.py` command line and `MapConversionConfig` fields, the other root tools' command lines, and the output files and their schemas (`.osm`, `_stoplines.json`, `_mesh.json`, `metadata.json`, and the sidecar JSONs). `metadata.json` must stay loadable by torchdrivesim's `MapConfig(**json)`, which rejects unknown keys — new outputs go in sidecar files. Everything under `crdesigner/` is vendored upstream and is not our public API.
- **Public-repo rule:** customer names and customer map data never enter a commit. Test fixtures are synthetic XODR built with `tests/iai/xodr_builder.py`; code and comments stay vendor-neutral.
- **Overrides of global rules:**
  - **§5 OICO / consolidation:** Do not refactor or de-duplicate inside `crdesigner/` to make it tidier. That directory is a near-verbatim snapshot of upstream — refactors there cost us on the next merge from upstream. OICO applies normally to repo-root scripts and any new IAI-specific files.
  - **§3 style / §4 keyword-args:** Match upstream style inside `crdesigner/` even where it conflicts with the global PEP8 / keyword-only conventions. Apply the global style rules to repo-root IAI scripts only.
  - **§7 TDD:** IAI tests live in `tests/iai/`. Tests that convert a map run `convert_map.py` in a subprocess, because crdesigner's config objects are process-global.

## Repo shape

This repo is a fork of [CommonRoad Scenario Designer](https://github.com/CommonRoad/commonroad-scenario-designer) v0.8.5. Most code is vendored upstream; the IAI-specific surface is the repo-root scripts.

- `convert_map.py` — **the IAI entry point.** OpenDRIVE → CommonRoad → Lanelet2 → TorchDriveSim artifacts. See "Conversion pipeline" below.
- `traffic_light_fsm.py`, `fsm_visualization.py` — traffic-light FSM generation and review rendering from a converted map.
- `opendrive_signal_groups.py` — reads the XODR's junction membership and controllers into `<map>_signal_groups.json` (called by `convert_map.py`; also a CLI).
- `map_report.py`, `map_quality.py` — batch conversion report / comparison, and the quality metrics it records.
- `conversion_common.py` — dependency-free helpers both `convert_map.py` and `map_report.py` use (file sha256, git identity, the refusal exit code 3).
- `prune_osm_deletes.py` — applies OSM-editor `action="delete"` markers.
- `convert_lanelet_to_opendrive.py` — reverse direction via CARLA's tools.
- `crdesigner/` — **vendored** CommonRoad Scenario Designer. Treat as upstream.
- `tests/` — upstream pytest suite (pytest + pytest-qt); `tests/iai/` — IAI tests and the synthetic XODR builder.
- `maps/`, `xodr_output/`, `carla_xodr_path/`, `_private/` — local, git-ignored. Do not commit map data.

## Common commands

All commands: `PYTHONPATH` unset (a sourced ROS environment breaks pytest), the repo's venv python, absolute paths.

- Setup: `python3 -m venv <repo>/.venv && <repo>/.venv/bin/pip install -r <repo>/requirements-tests.txt`. `requirements.txt` is the runtime manifest; `pyproject.toml` is upstream's poetry packaging and is left untouched (`tests/iai/test_requirements.py` keeps the two in step).
- **Run a conversion:** `<repo>/.venv/bin/python <repo>/convert_map.py dir_path=<folder-with-single-.xodr>` (or `xodr_path=<file.xodr>`, outputs in `<file folder>/<stem>/`; `convert_map()` returns the `.osm` path). Other `MapConversionConfig` fields are OmegaConf dotlist args; see README.
- **Batch report:** `<repo>/.venv/bin/python <repo>/map_report.py run --maps <dir> --out <dir> [--fsm]`; `compare <a/report.json> <b/report.json> --out <dir>`.
- Tests: `QT_QPA_PLATFORM=offscreen <repo>/.venv/bin/python -m pytest <repo>/tests` (set `IAI_MAPS_DIR` to a maps folder for the Town15 FSM tests). The upstream suite writes files into the tree; they are git-ignored.
- Lint: `ruff check .` and `ruff format --check .`. Config in `pyproject.toml` (line-length 100, double quotes, rules `E4,E7,E9,I`). The ANTLR-generated parser at `crdesigner/verification_repairing/verification/hol/parser/gen/` is excluded — do not lint or reformat it.
- GUI (upstream, not the IAI workflow): `python crdesigner/ui/gui/start_gui.py`. Requires system Qt libs and is rarely needed for IAI work.

The upstream `crdesigner` CLI exists (`python -m crdesigner.ui.command_line --input-file ... odrcr`) but the IAI workflow uses `convert_map.py` because the upstream CLI does not emit the TorchDriveSim artifacts (mesh, stoplines JSON, MapConfig).

## Conversion pipeline (`convert_map.py`)

`convert_map()` calls one keyword-only stage function per step, in this order; each stage writes its own output:

1. `validate_config` — check the config before anything is read (`ConfigError`, exit code 3 from the CLI) and return the `.xodr`: `cfg.xodr_path`, or the single one in `cfg.dir_path` (the filename, minus extension, becomes the map name). With `xodr_path`, `convert_map` points `dir_path` at `<xodr folder>/<stem>/` for the remaining stages.
2. `parse_xodr` — `parse_opendrive` (`crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.parser`), then `parse_geo_reference` (pyproj; `tmerc` and `utm`) and the header `<offset>`, combined in one `CustomTransfomer` local frame; no geo-reference → origin (0, 0) with a warning; a present but unusable one raises `InputError` unless `allow_zero_origin`.
3. `export_osm` — `Network.load_opendrive` → `Network.export_lanelet_network` builds a CommonRoad lanelet network in the OpenDRIVE inertial frame (`filter_types`: `driving`, `onRamp`, `offRamp`, `exit`, `entry`), simplified, then `CR2LaneletConverter` (`crdesigner/map_conversion/lanelet2/cr2lanelet.py`) with a `CustomTransfomer` wrapping a Lanelet2 `UtmProjector` writes `<map>.osm`.
4. `trim` — if `trim_radius` is set, rewrite the `.osm` trimmed to a circle around `center` (and `trim_vertical_limits`).
5. `extract_stoplines` — `traffic_light.iai_stoplines` from each traffic light, plus `Network._iai_stop_signs` / `_iai_yield_signs`, trim-filtered → `<map>_stoplines.json`; then `write_signal_groups` → `<map>_signal_groups.json`.
6. `build_mesh` — `torchdrivesim.lanelet2.road_mesh_from_lanelet_map` / `lanelet_map_to_lane_mesh` → `<map>_mesh.json`.
7. `render_and_finish` — `MapConfig` → `metadata.json`; `visualization.png` via `torchdrivesim.rendering`; `write_provenance` → `<map>_provenance.json`.

Frame invariant: the vendored parser leaves road geometry in file coordinates; `CustomTransfomer.to_local` adds the header `<offset>` x/y shift (its `hdg` and `z` are ignored with a warning) and subtracts `GeoReference.translation` (the map centre for UTM and false-origin tmerc maps). The `.osm` (hence the mesh) and the stoplines both go through `to_local`, so they share one local frame.

## Where the IAI seams live

These are the names/files that diverge from upstream. Search for them when changing the OpenDRIVE → CommonRoad path:

- The repo-root scripts — IAI-authored, not upstream.
- `crdesigner/map_conversion/lanelet2/cr2lanelet.py` — small IAI tweaks (import / type-hint cleanup, the narrowed same-direction marking rule) and `_share_seam_nodes`, which merges coinciding predecessor/successor end nodes so routing sees the link; `_nodes_at` (`SHARED_NODE_TOLERANCE`) stops linked lanelets that do not meet from sharing end nodes; `one_way=no` for bidirectional vehicle lanes.
- `crdesigner/map_conversion/common/conversion_lanelet_network.py` — `check_concatenation_potential` never concatenates bidirectional lanelets. `convert_map.bidirectional_fixes` drops the OpenDRIVE conversion's wrong links around them.
- `crdesigner/map_conversion/opendrive/odr2cr/opendrive_conversion/plane_elements/` — elevation: `Border.calc_elevation` follows the reference chain, `ParametricLane.calc_elevation` (reversed lanes) and `ParametricLaneGroup.calc_elevation` (merge/split mirroring).
- Vendored code reports degradations through `logging.getLogger("crdesigner.degradation")` with `extra={kind, element, id, value, action}` (no IAI imports inside `crdesigner/`).
- Custom attributes on upstream objects: `iai_stoplines`, `_iai_stop_signs`, `_iai_yield_signs`, `opendrive_id`. Grep for `iai_` inside `crdesigner/` to find where they're attached.
- `git diff master -- crdesigner/` (or whatever the upstream-tracking branch is) is the authoritative way to enumerate IAI deltas.

## Dependencies worth knowing

- `torchdrivesim` — Inverted AI package (public). Provides `Stopline`, `MapConfig`, `BirdviewMesh`, the renderer, and mesh utilities. Pinned to a git commit in `requirements.txt`: `convert_map.py` needs `BirdviewRGBMeshGenerator`, which no PyPI release (≤ 0.2.3) has.
- `lanelet2` — installed from PyPI wheels (cp310–cp313, Linux).
- `click<8.2` — upstream pins `typer ^0.9`, which fails on newer click.
- CI (`.github/workflows/python-app.yml`) runs Ubuntu only on Python 3.10–3.13. macOS and Windows are not supported by upstream.
