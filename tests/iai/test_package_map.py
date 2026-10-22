"""``package_map.py``: a converted map as the ZIP the map service accepts, stamped with the converter version."""

import json
import subprocess
import sys
import zipfile

import conversion_common
from tests.iai import xodr_builder as xb


def _converted(*, tmp_path, roads=None, **config: str):
    roads = roads or [xb.straight_road(id="1", length=30.0)]
    xb.write_xodr(map_dir=tmp_path, name="case", text=xb.build_xodr(roads=roads))
    return xb.run_convert_map(map_dir=str(tmp_path), **config)


def _package(*, map_dir, extra: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, f"{xb.REPO_ROOT}/package_map.py", str(map_dir), *extra],
        capture_output=True,
        text=True,
    )


def test_package_holds_what_the_map_service_reads_and_the_provenance(tmp_path):
    assert _converted(tmp_path=tmp_path).returncode == 0
    result = _package(map_dir=tmp_path)
    assert result.returncode == 0, result.stderr
    commit = json.loads((tmp_path / "case_provenance.json").read_text())["converter"]["commit"]
    stamp = (commit or "unversioned")[:12]
    path = (tmp_path / f"case_{stamp}.zip").resolve()
    assert str(path) in result.stdout
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == sorted(
            ["metadata.json", "case.osm", "case_mesh.json", "case_stoplines.json", "case_provenance.json"]
        )
        assert stamp in archive.comment.decode()


def test_a_rejected_map_is_not_packaged(tmp_path):
    # two roads meeting end to start without a link: routing fails the perfect preset
    unlinked = [xb.straight_road(id="1", length=30.0), xb.straight_road(id="2", length=30.0, x=30.0)]
    result = _converted(tmp_path=tmp_path, roads=unlinked, **{"autocritic.preset": "perfect"})
    assert result.returncode == conversion_common.REJECTED_EXIT_CODE
    packaged = _package(map_dir=tmp_path)
    assert packaged.returncode == conversion_common.REFUSED_EXIT_CODE
    assert "metadata.json" in packaged.stderr


def test_an_osm_over_the_limit_is_not_packaged(tmp_path):
    assert _converted(tmp_path=tmp_path).returncode == 0
    result = _package(map_dir=tmp_path, extra=("--max-osm-mb", "0.0001"))
    assert result.returncode == conversion_common.REFUSED_EXIT_CODE
    assert "MB limit" in result.stderr
    assert not list(tmp_path.glob("*.zip"))


def test_the_default_limit_is_the_map_service_one():
    assert conversion_common.MAP_SERVICE_MAX_OSM_MB == 30
