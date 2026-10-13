"""Package a converted map as the ZIP the Inverted AI map service accepts, stamped with the converter version.

The ZIP holds, at its root, ``metadata.json``, the files it names (``.osm``, mesh, stoplines and, when present,
the traffic-light controller) and ``<map>_provenance.json``. It is written next to the outputs as
``<map>_<converter commit, 12 characters>.zip`` (``unversioned`` outside a git checkout), with the commit, its
dirty flag and the conversion time in the ZIP comment. A map whose outputs were withheld (no ``metadata.json``)
or whose ``.osm`` exceeds the map service's limit is refused (exit code 3).

usage: package_map.py <map dir> [--out <zip>] [--max-osm-mb 30]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

from conversion_common import MAP_SERVICE_MAX_OSM_MB, REFUSED_EXIT_CODE

METADATA_PATH_FIELDS = ("lanelet_path", "mesh_path", "stoplines_path", "traffic_light_controller_path")


class PackageRefused(Exception):
    """The map cannot be packaged as it is."""


def package(*, map_dir: str, out: str | None = None, max_osm_mb: float = MAP_SERVICE_MAX_OSM_MB) -> str:
    """Write the map service ZIP of the converted map in ``map_dir``.

    Args:
        map_dir: The conversion's output folder.
        out: Where to write the ZIP; default ``<map dir>/<map>_<commit>.zip``.
        max_osm_mb: The map service's ``.osm`` limit in MiB.

    Returns:
        Absolute path of the ZIP.

    Raises:
        PackageRefused: no ``metadata.json``, a file it names is missing, no provenance, or the ``.osm``
            exceeds ``max_osm_mb``.
    """
    map_dir = os.path.abspath(map_dir)
    metadata_path = os.path.join(map_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise PackageRefused(f"{map_dir} has no metadata.json (not converted, or its outputs were withheld)")
    with open(metadata_path) as f:
        metadata = json.load(f)
    name = metadata["name"]
    files = [metadata_path]
    for field in METADATA_PATH_FIELDS:
        if metadata.get(field):
            path = os.path.join(map_dir, metadata[field])
            if not os.path.isfile(path):
                raise PackageRefused(f"{path} (metadata.json {field}) does not exist")
            files.append(path)
    provenance_path = os.path.join(map_dir, f"{name}_provenance.json")
    if not os.path.isfile(provenance_path):
        raise PackageRefused(f"{provenance_path} does not exist; the package could not be stamped")
    files.append(provenance_path)
    osm_path = os.path.join(map_dir, metadata["lanelet_path"])
    osm_mb = os.path.getsize(osm_path) / (1 << 20)
    if osm_mb > max_osm_mb:
        raise PackageRefused(f"{osm_path} is {osm_mb:.1f} MB, over the {max_osm_mb:g} MB limit of the map service")
    with open(provenance_path) as f:
        provenance = json.load(f)
    converter = provenance.get("converter") or {}
    stamp = (converter.get("commit") or "unversioned")[:12]
    out = os.path.abspath(out or os.path.join(map_dir, f"{name}_{stamp}.zip"))
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=os.path.basename(path))
        archive.comment = (
            f"{name}: converter {converter.get('commit')} dirty={converter.get('dirty')} "
            f"converted {provenance.get('created_utc')}"
        ).encode()
    return out


def main() -> None:
    """CLI entry point: package one map directory and print the ZIP's absolute path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("map_dir", help="A converted map's output folder (with metadata.json)")
    parser.add_argument("--out", default=None, help="Where to write the ZIP (default: in the map folder)")
    parser.add_argument("--max-osm-mb", type=float, default=MAP_SERVICE_MAX_OSM_MB, help="The map service's .osm limit")
    args = parser.parse_args()
    try:
        path = package(map_dir=args.map_dir, out=args.out, max_osm_mb=args.max_osm_mb)
    except PackageRefused as exc:
        print(f"Cannot package: {exc}", file=sys.stderr)
        sys.exit(REFUSED_EXIT_CODE)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
