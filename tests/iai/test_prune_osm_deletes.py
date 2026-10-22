"""``prune_osm_deletes.py``: applies action="delete" markers and keeps the original file."""

import subprocess
import sys

import prune_osm_deletes as pod
from tests.iai.xodr_builder import REPO_ROOT

OSM = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6">
  <node id="1" lat="0" lon="0"/>
  <node id="2" lat="0" lon="0.001"/>
  <node id="3" lat="0" lon="0.002"/>
  <way id="10"><nd ref="1"/><nd ref="2"/></way>
  <way id="11" action="delete"><nd ref="2"/><nd ref="3"/></way>
</osm>
"""


def test_prune_removes_marked_elements_and_keeps_the_original(tmp_path):
    path = tmp_path / "m.osm"
    path.write_text(OSM)
    backup = pod.prune(path=path)
    assert backup == (tmp_path / "m.osm.orig").resolve()
    assert backup.read_text() == OSM
    pruned = path.read_text()
    assert 'id="11"' not in pruned and 'id="3"' not in pruned and 'id="10"' in pruned


def test_a_second_prune_never_overwrites_the_original(tmp_path):
    path = tmp_path / "m.osm"
    path.write_text(OSM)
    pod.prune(path=path)
    pod.prune(path=path)
    assert (tmp_path / "m.osm.orig").read_text() == OSM


def test_command_line_prints_absolute_paths(tmp_path):
    path = tmp_path / "m.osm"
    path.write_text(OSM)
    result = subprocess.run(
        [sys.executable, f"{REPO_ROOT}/prune_osm_deletes.py", "--path", "m.osm"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"Pruned {path.resolve()}" in result.stdout
    assert f"original kept at {(tmp_path / 'm.osm.orig').resolve()}" in result.stdout
