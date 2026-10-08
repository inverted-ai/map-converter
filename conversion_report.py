"""Collect the converter's degradation records and write ``<map>_conversion_report.json``.

Vendored code reports every input it cannot use as written through
``logging.getLogger("crdesigner.degradation")`` with
``extra={"kind", "element", "id", "value", "action"}``:

- ``kind``: what was wrong, e.g. ``unknown_value``, ``missing_attribute``, ``decimal_comma``,
  ``unparseable_number``, ``degenerate_geometry``, ``unknown_reference``; ``heuristic`` when a
  :mod:`conversion_heuristics` rule decided (``element`` is its id, ``value`` its name);
  ``not_converted`` for an element listed but deliberately not converted (painted stop lines).
- ``element``: ``<tag>`` when the whole element was dropped (``action="dropped"``), else
  ``<tag>.<attribute>``.
- ``id``: the enclosing road, signal, object or junction id, when known.
- ``value``: the offending value (or a count, for aggregated records).
- ``action``: what the converter did instead, e.g. ``dropped``, ``defaulted``, ``coerced``.
"""

from __future__ import annotations

import collections
import json
import logging
import os
from typing import Any, Optional

from lxml import etree

DEGRADATION_LOGGER = "crdesigner.degradation"
UNRECOGNISED_KINDS = ("unknown_value",)
MAX_IDS_PER_TOKEN = 20
COUNTED_TAGS = (
    "road",
    "geometry",
    "elevation",
    "superelevation",
    "crossfall",
    "shape",
    "laneOffset",
    "laneSection",
    "lane",
    "width",
    "border",
    "roadMark",
    "speed",
    "access",
    "object",
    "signal",
    "signalReference",
    "junction",
    "connection",
)


class ConversionReport(logging.Handler):
    """A logging handler that keeps every degradation record of one conversion."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(
            {
                "kind": getattr(record, "kind", None),
                "element": getattr(record, "element", None),
                "id": _jsonable(value=getattr(record, "id", None)),
                "value": _jsonable(value=getattr(record, "value", None)),
                "action": getattr(record, "action", None),
                "message": record.getMessage(),
            }
        )

    def summary(self, *, elements_seen: dict[str, int]) -> dict[str, Any]:
        """Element counts, unrecognised tokens, the heuristics that fired and record counts by kind."""
        dropped: collections.Counter = collections.Counter()
        degraded: collections.Counter = collections.Counter()
        for r in self.records:
            element = r["element"] or ""
            if r["action"] == "dropped" and "." not in element:
                dropped[element] += 1
            else:
                degraded[element.split(".")[0]] += 1
        tags = sorted(set(elements_seen) | set(dropped) | set(degraded))
        elements = {
            tag: {
                "seen": elements_seen.get(tag),
                "converted": None if tag not in elements_seen else elements_seen[tag] - dropped[tag],
                "degraded": degraded[tag],
                "dropped": dropped[tag],
            }
            for tag in tags
        }
        tokens: dict[tuple, dict[str, Any]] = {}
        for r in self.records:
            if r["kind"] not in UNRECOGNISED_KINDS:
                continue
            entry = tokens.setdefault(
                (r["element"], str(r["value"])), {"element": r["element"], "value": r["value"], "count": 0, "ids": []}
            )
            entry["count"] += 1
            if r["id"] is not None and len(entry["ids"]) < MAX_IDS_PER_TOKEN:
                entry["ids"].append(r["id"])
        heuristics: dict[str, dict[str, Any]] = {}
        for r in self.records:
            if r["kind"] != "heuristic":
                continue
            entry = heuristics.setdefault(r["element"], {"name": r["value"], "count": 0, "ids": []})
            entry["count"] += 1
            if r["id"] is not None and len(entry["ids"]) < MAX_IDS_PER_TOKEN:
                entry["ids"].append(r["id"])
        return {
            "elements": elements,
            "unrecognised_tokens": sorted(tokens.values(), key=lambda t: (-t["count"], str(t["element"]))),
            "heuristics": dict(sorted(heuristics.items(), key=lambda item: int(item[0][1:]))),
            "records_by_kind": dict(sorted(collections.Counter(str(r["kind"]) for r in self.records).items())),
        }

    def write(
        self,
        *,
        path: str,
        xodr_path: str,
        error: Optional[BaseException] = None,
        validation: Optional[dict[str, Any]] = None,
    ) -> str:
        """Write the report for ``xodr_path``.

        Args:
            path: Where to write it.
            xodr_path: The converted input.
            error: The exception that ended the run, if any.
            validation: The validation verdict (as a dict), for a run that got that far.

        Returns:
            Absolute path of the written file.
        """
        report = {
            "input": os.path.abspath(xodr_path),
            "outcome": "ok" if error is None else "failed",
            "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
            "validation": validation,
            **self.summary(elements_seen=count_elements(xodr_path=xodr_path)),
            "records": self.records,
        }
        path = os.path.abspath(path)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return path


class FirstOccurrenceFilter(logging.Filter):
    """Pass each (kind, element, value) degradation once; the report keeps every occurrence."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[tuple] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        key = (getattr(record, "kind", None), getattr(record, "element", None), str(getattr(record, "value", None)))
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def count_elements(*, xodr_path: str) -> dict[str, int]:
    """How many of each :data:`COUNTED_TAGS` element the file holds (namespaces ignored)."""
    counts: collections.Counter = collections.Counter()
    try:
        for _, elem in etree.iterparse(xodr_path, events=("end",), huge_tree=True):
            tag = etree.QName(elem).localname if isinstance(elem.tag, str) else None
            if tag in COUNTED_TAGS:
                counts[tag] += 1
            if tag == "road":
                elem.clear()  # roads hold almost every counted element; keep memory flat
    except etree.XMLSyntaxError:
        return {}
    return {tag: counts[tag] for tag in COUNTED_TAGS if counts[tag]}


def attach() -> ConversionReport:
    """Start collecting ``crdesigner.degradation`` records."""
    handler = ConversionReport()
    logging.getLogger(DEGRADATION_LOGGER).addHandler(handler)
    return handler


def detach(*, handler: ConversionReport) -> None:
    """Stop collecting into ``handler``."""
    logging.getLogger(DEGRADATION_LOGGER).removeHandler(handler)


def _jsonable(*, value: Any) -> Any:
    return value if value is None or isinstance(value, (bool, int, float, str)) else str(value)
