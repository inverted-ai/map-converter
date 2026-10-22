"""Apply action="delete" markers in an OSM file and repair referential damage.

Used to shrink converted OSM maps that exceed the backend size threshold without
lowering node resolution. The user marks unwanted nodes/ways/relations with
action="delete" during manual editing; this script removes those elements,
strips dangling <nd>/<member> references from surviving ways and relations,
cascades to ways that fall below 2 nodes and relations that fall below 2 members,
and finally sweeps any newly-orphaned nodes. The file is overwritten in place; the
first prune keeps the original as ``<file>.orig`` (later prunes never overwrite it).
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def _has_action_delete(*, elem: ET.Element) -> bool:
    """Return True if elem carries action="delete" as attribute or child <tag>."""
    if elem.get("action") == "delete":
        return True
    for child in elem.findall("tag"):
        if child.get("k") == "action" and child.get("v") == "delete":
            return True
    return False


def prune(*, path: Path) -> Path:
    """Apply action="delete" markers in `path` and overwrite the file in place, keeping the original.

    Args:
        path: Path to an OSM XML file. Must exist and have <osm> as root.

    Returns:
        Absolute path of the kept original, ``<path>.orig`` (written by the first prune only).

    Raises:
        FileNotFoundError: If `path` does not exist or is not a regular file.
        ValueError: If the root element is not <osm>.
    """
    if not path.is_file():
        raise FileNotFoundError(f"OSM file not found: {path}")

    size_before = path.stat().st_size

    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "osm":
        raise ValueError(f"Expected root element <osm>, got <{root.tag}>")

    nodes_before = len(root.findall("node"))
    ways_before = len(root.findall("way"))
    relations_before = len(root.findall("relation"))

    deleted_nodes: set[str] = set()
    deleted_ways: set[str] = set()
    deleted_relations: set[str] = set()

    for node in list(root.findall("node")):
        if _has_action_delete(elem=node):
            deleted_nodes.add(node.get("id"))
            root.remove(node)
    for way in list(root.findall("way")):
        if _has_action_delete(elem=way):
            deleted_ways.add(way.get("id"))
            root.remove(way)
    for relation in list(root.findall("relation")):
        if _has_action_delete(elem=relation):
            deleted_relations.add(relation.get("id"))
            root.remove(relation)

    explicit_node_deletes = len(deleted_nodes)
    explicit_way_deletes = len(deleted_ways)
    explicit_relation_deletes = len(deleted_relations)

    single_member_relation_deletes = 0

    while True:
        changed = False

        for way in list(root.findall("way")):
            for nd in list(way.findall("nd")):
                if nd.get("ref") in deleted_nodes:
                    way.remove(nd)
                    changed = True
            if len(way.findall("nd")) < 2:
                deleted_ways.add(way.get("id"))
                root.remove(way)
                changed = True

        for relation in list(root.findall("relation")):
            for member in list(relation.findall("member")):
                mtype = member.get("type")
                mref = member.get("ref")
                if (
                    (mtype == "node" and mref in deleted_nodes)
                    or (mtype == "way" and mref in deleted_ways)
                    or (mtype == "relation" and mref in deleted_relations)
                ):
                    relation.remove(member)
                    changed = True
            member_count = len(relation.findall("member"))
            if member_count < 2:
                if member_count == 1:
                    single_member_relation_deletes += 1
                deleted_relations.add(relation.get("id"))
                root.remove(relation)
                changed = True

        if not changed:
            break

    cascade_way_deletes = len(deleted_ways) - explicit_way_deletes
    cascade_relation_deletes = (
        len(deleted_relations) - explicit_relation_deletes - single_member_relation_deletes
    )

    referenced_node_ids: set[str] = set()
    for way in root.findall("way"):
        for nd in way.findall("nd"):
            referenced_node_ids.add(nd.get("ref"))
    for relation in root.findall("relation"):
        for member in relation.findall("member"):
            if member.get("type") == "node":
                referenced_node_ids.add(member.get("ref"))

    orphan_node_deletes = 0
    for node in list(root.findall("node")):
        if node.get("id") not in referenced_node_ids:
            root.remove(node)
            orphan_node_deletes += 1

    path = path.resolve()
    backup = path.with_name(path.name + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    tree.write(path, xml_declaration=True, encoding="UTF-8")

    size_after = path.stat().st_size
    nodes_after = len(root.findall("node"))
    ways_after = len(root.findall("way"))
    relations_after = len(root.findall("relation"))

    mb = 1024 * 1024
    print(f"Pruned {path}")
    print(
        f"  nodes:     {nodes_before} -> {nodes_after} "
        f"(explicit={explicit_node_deletes}, orphan={orphan_node_deletes})"
    )
    print(
        f"  ways:      {ways_before} -> {ways_after} "
        f"(explicit={explicit_way_deletes}, cascade={cascade_way_deletes})"
    )
    print(
        f"  relations: {relations_before} -> {relations_after} "
        f"(explicit={explicit_relation_deletes}, cascade={cascade_relation_deletes}, "
        f"single_member={single_member_relation_deletes})"
    )
    print(f"  size:      {size_before / mb:.2f} MB -> {size_after / mb:.2f} MB")
    print(f"  original kept at {backup}")
    return backup


def main() -> None:
    """Parse args and prune the given OSM file in place."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to the OSM file to prune in place",
    )
    args = parser.parse_args()
    prune(path=args.path)


if __name__ == "__main__":
    main()
