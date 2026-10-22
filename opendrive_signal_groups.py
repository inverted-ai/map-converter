"""Read the traffic-light structure an OpenDRIVE file declares: junction membership and controllers.

``convert_map.py`` writes the result to ``<map>_signal_groups.json``; ``traffic_light_fsm.py``
reads it to keep each junction's lights in one intersection and, if asked, to take phase groups
and their order from the file's controllers. Run it on its own to add the sidecar to a map
converted before it existed::

    python /abs/clone/opendrive_signal_groups.py --xodr /abs/maps/x/x.xodr --out /abs/maps/x/x_signal_groups.json

Junction membership of a signal comes from the first of these that applies:

1. ``controller``: a ``<junction>`` references a ``<controller>`` whose ``<control>`` names it.
2. ``road_junction``: the road carrying the ``<signal>`` or a ``<signalReference>`` to it lies in a
   junction (``road@junction``).
3. ``approach_road``: the road's link at the nearer end points to a junction and the signal is
   within ``max_distance`` (metres along the road) of that end.

Signal ids are normalised with ``str(int(id))`` so they join stopline ``actor_id`` values; ids that
are not integers are listed and left out.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from typing import Optional

from lxml import etree

from traffic_light_fsm import FsmConfig

logger = logging.getLogger(__name__)

SOURCES = ("controller", "road_junction", "approach_road")
DEFAULT_MAX_DISTANCE_M = FsmConfig().cluster_eps


def normalise_signal_id(*, raw: Optional[str]) -> Optional[str]:
    """``str(int(raw))``, or ``None`` when ``raw`` is not an integer."""
    try:
        return str(int((raw or "").strip()))
    except ValueError:
        return None


@dataclasses.dataclass(slots=True)
class _Road:
    junction: Optional[str]
    length: Optional[float]
    predecessor_junction: Optional[str]
    successor_junction: Optional[str]


def _localname(*, elem) -> str:
    return etree.QName(elem).localname if isinstance(elem.tag, str) else ""


def _children(*, elem, name: str) -> list:
    return [c for c in elem if _localname(elem=c) == name]


def _number(*, raw: Optional[str], cast):
    """``cast(raw)``, or ``None`` when ``raw`` is missing or malformed."""
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return None


def _road(*, road) -> _Road:
    links = {"predecessor": None, "successor": None}
    for link in _children(elem=road, name="link"):
        for end in links:
            for target in _children(elem=link, name=end):
                if target.get("elementType") == "junction":
                    links[end] = target.get("elementId")
    junction = road.get("junction")
    return _Road(
        junction=None if junction in (None, "", "-1") else junction,
        length=_number(raw=road.get("length"), cast=float),
        predecessor_junction=links["predecessor"],
        successor_junction=links["successor"],
    )


def extract_signal_groups(*, xodr_path: str, max_distance: float = DEFAULT_MAX_DISTANCE_M) -> dict:
    """Junction membership, controllers and id problems declared in one ``.xodr``.

    The file is streamed one top-level element at a time, so large maps are not held in memory.

    Args:
        xodr_path: The OpenDRIVE file.
        max_distance: Source 3 reach (metres along the road from the junction end).

    Returns:
        A JSON-ready dict with ``membership`` (signal id → ``{junction, source}``),
        ``controllers``, ``conflicts``, ``non_integer_signal_ids`` and ``warnings``.
    """
    candidates: dict[str, dict[str, str]] = {}  # signal id -> {source: junction}
    non_integer: set[str] = set()

    def signal_id(raw: Optional[str]) -> Optional[str]:
        normalised = normalise_signal_id(raw=raw)
        if normalised is None:
            non_integer.add(str(raw))
        return normalised

    def offer(*, raw_id: Optional[str], source: str, junction: Optional[str]) -> None:
        normalised = signal_id(raw_id)
        if normalised is not None and junction is not None:
            candidates.setdefault(normalised, {}).setdefault(source, junction)

    controllers: dict[str, dict] = {}
    junction_refs: list[tuple[str, str, Optional[int]]] = []  # (junction, controller, sequence)
    for _, elem in etree.iterparse(os.path.abspath(xodr_path), events=("end",), huge_tree=True):
        parent = elem.getparent()
        if parent is None or parent.getparent() is not None:
            continue  # only the root's children are handled; nested elements are read through them
        kind = _localname(elem=elem)
        if kind == "controller":
            signal_ids = [signal_id(c.get("signalId")) for c in _children(elem=elem, name="control")]
            controllers[elem.get("id")] = {
                "id": elem.get("id"),
                "name": elem.get("name"),
                "signal_ids": [s for s in signal_ids if s is not None],
                "sequence": _number(raw=elem.get("sequence"), cast=int),
                "junction": None,
            }
        elif kind == "junction":
            for ref in _children(elem=elem, name="controller"):
                junction_refs.append((elem.get("id"), ref.get("id"), _number(raw=ref.get("sequence"), cast=int)))
        elif kind == "road":
            road = _road(road=elem)
            for signals in _children(elem=elem, name="signals"):
                for signal in signals:
                    if _localname(elem=signal) not in ("signal", "signalReference"):
                        continue
                    s = _number(raw=signal.get("s"), cast=float)
                    offer(raw_id=signal.get("id"), source="road_junction", junction=road.junction)
                    offer(raw_id=signal.get("id"), source="approach_road", junction=_approach_junction(road=road, s=s, max_distance=max_distance))
        elem.clear()
        while elem.getprevious() is not None:
            del parent[0]

    for junction, controller_id, sequence in junction_refs:
        controller = controllers.get(controller_id)
        if controller is None:
            continue
        controller["junction"] = junction
        if sequence is not None:
            controller["sequence"] = sequence
        for normalised in controller["signal_ids"]:
            offer(raw_id=normalised, source="controller", junction=junction)

    membership: dict[str, dict[str, str]] = {}
    conflicts = []
    for normalised, by_source in sorted(candidates.items(), key=lambda kv: int(kv[0])):
        chosen = next(source for source in SOURCES if source in by_source)
        membership[normalised] = {"junction": by_source[chosen], "source": chosen}
        others = {s: j for s, j in by_source.items() if j != by_source[chosen]}
        if others:
            conflicts.append({"signal_id": normalised, "chosen": membership[normalised], "disagreeing": others})
    warnings = [
        f"signal {c['signal_id']}: {c['chosen']['source']} says junction {c['chosen']['junction']}, "
        + ", ".join(f"{s} says {j}" for s, j in c["disagreeing"].items())
        + f" - using {c['chosen']['junction']}"
        for c in conflicts
    ]
    if non_integer:
        warnings.append(f"{len(non_integer)} signal id(s) are not integers and cannot join stoplines: {sorted(non_integer)[:20]}")
    for warning in warnings:
        logger.warning(warning)
    return {
        "source_xodr": os.path.abspath(xodr_path),
        "membership": membership,
        "controllers": sorted(controllers.values(), key=lambda c: c["id"]),
        "conflicts": conflicts,
        "non_integer_signal_ids": sorted(non_integer),
        "warnings": warnings,
    }


def _approach_junction(*, road: _Road, s: Optional[float], max_distance: float) -> Optional[str]:
    """Junction at the nearer end of ``road`` if the signal at ``s`` is within ``max_distance`` of it."""
    if s is None or road.length is None:
        return None
    ends = [(s, road.predecessor_junction), (road.length - s, road.successor_junction)]
    reachable = [(distance, junction) for distance, junction in ends if junction is not None and distance <= max_distance]
    return min(reachable)[1] if reachable else None


def write_signal_groups(
    *,
    xodr_path: str,
    out_path: str,
    max_distance: float = DEFAULT_MAX_DISTANCE_M,
    stopline_ids: Optional[dict[str, list[str]]] = None,
) -> str:
    """Extract and write the sidecar; returns its absolute path.

    ``stopline_ids`` (from a conversion): per stopline row id, the signals it stands for when several lights share
    one approach lane's row (H15), written as ``stopline_ids``.
    """
    out_path = os.path.abspath(out_path)
    groups = extract_signal_groups(xodr_path=xodr_path, max_distance=max_distance)
    if stopline_ids is not None:
        groups["stopline_ids"] = stopline_ids
    with open(out_path, "w") as f:
        json.dump(groups, f, indent=2)
    logger.info("Wrote %d signal memberships and %d controllers to %s", len(groups["membership"]), len(groups["controllers"]), out_path)
    return out_path


def main() -> None:
    """CLI entry point: write ``--out`` from ``--xodr``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xodr", required=True, help="OpenDRIVE file")
    parser.add_argument("--out", required=True, help="Output JSON (conventionally <map>_signal_groups.json)")
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE_M, help=f"Approach-road reach in metres (default {DEFAULT_MAX_DISTANCE_M:g}, the FSM cluster_eps)")
    args = parser.parse_args()
    path = write_signal_groups(xodr_path=args.xodr, out_path=args.out, max_distance=args.max_distance)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
