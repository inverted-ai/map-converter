"""Self-validation of one conversion: threshold checks on its outputs plus the CommonRoad map verifier.

``convert_map.py`` runs :func:`validate` after writing the outputs and stores the :class:`Verdict` in
``<map>_conversion_report.json``. Every number comes from :mod:`map_quality`; the thresholds live in
:class:`ValidationConfig`. Until they are ratified (``ValidationConfig.ratified``), a failed check
only warns.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import os
import queue
import signal
import time
from typing import Any, Optional

import map_quality

logger = logging.getLogger(__name__)

VERIFIER_EXAMPLES = 5


@dataclasses.dataclass(slots=True)
class ValidationConfig:
    """Thresholds of the self-validation checks (proposed from the corpus, not yet ratified).

    Attributes:
        min_routing_edge_pct: Lowest share of lanelets with a routing edge.
        min_seam_handshake_pct: Lowest share of geometric seams whose end nodes are shared.
        max_crossed_fraction: Highest share of lanelets whose outline folds or touches itself.
        max_point_extent_m: Furthest a point may lie from the local origin.
        max_lanelets_without_subtype: Most lanelets allowed without a ``subtype`` tag.
        verifier_partition_above: Partition the map verifier's work above this many lanelets.
        ratified: Whether a failed check fails the verdict (``fail``) or only warns (``warn``).
    """

    min_routing_edge_pct: float = 95.0
    min_seam_handshake_pct: float = 95.0
    max_crossed_fraction: float = 0.02
    max_point_extent_m: float = 20_000.0
    max_lanelets_without_subtype: int = 0
    verifier_partition_above: int = 200
    ratified: bool = False


@dataclasses.dataclass(slots=True)
class Check:
    """One threshold check: ``value`` against ``limit``."""

    name: str
    value: Any
    limit: Any
    passed: bool
    detail: str = ""


@dataclasses.dataclass(slots=True)
class Verdict:
    """The outcome of :func:`validate`.

    Attributes:
        status: ``pass``, ``warn`` (a check failed, thresholds not ratified) or ``fail``.
        ratified: Whether the thresholds were ratified.
        checks: Every check, passed or not.
        metric_errors: Metrics that raised, as ``"<Class>: <message>"``.
        verifier: The map verifier's result (see :func:`run_map_verifier`).
    """

    status: str
    ratified: bool
    checks: list[Check]
    metric_errors: dict[str, str]
    verifier: dict[str, Any]


def evaluate(*, metrics: dict[str, Any], config: ValidationConfig) -> list[Check]:
    """The threshold checks over :func:`map_quality.compute_all` metrics; a missing metric fails its check."""
    checks = []

    def at_least(name: str, value: Optional[float], limit: float, detail: str) -> None:
        checks.append(Check(name=name, value=value, limit=limit, passed=value is None or value >= limit, detail=detail))

    def at_most(name: str, value: Optional[float], limit: float, detail: str) -> None:
        checks.append(Check(name=name, value=value, limit=limit, passed=value is not None and value <= limit, detail=detail))

    lanelets = metrics.get("lanelet_count")
    checks.append(Check(name="has_lanelets", value=lanelets, limit=1, passed=bool(lanelets), detail="the map is not empty"))
    at_least(
        "routing_edge_pct",
        metrics.get("routing_edge_pct"),
        config.min_routing_edge_pct,
        "share of lanelets with a predecessor, successor or neighbour in the routing graph",
    )
    at_least(
        "seam_handshake_pct",
        metrics.get("seam_handshake_pct"),
        config.min_seam_handshake_pct,
        "share of meeting lanelet ends that share their nodes",
    )
    crossed = metrics.get("crossed_geometry")
    at_most(
        "crossed_fraction",
        None if crossed is None or not lanelets else sum(crossed.values()) / lanelets,
        config.max_crossed_fraction,
        "share of lanelets whose outline folds or touches itself",
    )
    signs = metrics.get("sign_ways_vs_stoplines")
    if signs is None:
        checks.append(Check(name="stoplines", value=None, limit=None, passed=False, detail="metric unavailable"))
    else:
        checks.append(
            Check(
                name="traffic_light_stoplines",
                value=signs["light_stoplines"],
                limit=1 if signs["traffic_light_ways"] else 0,
                passed=not signs["traffic_light_ways"] or signs["light_stoplines"] > 0,
                detail=f"{signs['traffic_light_ways']} traffic-light ways in the .osm",
            )
        )
        checks.append(
            Check(
                name="stop_yield_stoplines",
                value=signs["sign_stoplines"],
                limit=1 if signs["stop_yield_sign_ways"] else 0,
                passed=not signs["stop_yield_sign_ways"] or signs["sign_stoplines"] > 0,
                detail=f"{signs['stop_yield_sign_ways']} stop/yield sign ways in the .osm",
            )
        )
    at_most(
        "lanelets_without_subtype",
        metrics.get("lanelets_without_subtype"),
        config.max_lanelets_without_subtype,
        "lanelets consumers that filter by subtype would drop",
    )
    at_most(
        "point_extent_m",
        metrics.get("point_extent_m"),
        config.max_point_extent_m,
        "furthest point from the local origin",
    )
    origin = metrics.get("origin")
    checks.append(
        Check(
            name="origin_in_range",
            value=origin,
            limit=[90.0, 180.0],
            passed=origin is not None and abs(origin[0]) <= 90.0 and abs(origin[1]) <= 180.0,
            detail="lanelet_map_origin is a valid latitude/longitude",
        )
    )
    return checks


def run_map_verifier(*, lanelet_network: Optional[Any], timeout_s: float, partition_above: int) -> dict[str, Any]:
    """Run CommonRoad's ``MapVerifier`` (report-only) in a child process killed after ``timeout_s``.

    The verifier assumes each lanelet's bounds have equal vertex counts, so pass the network before
    boundary simplification. Partitioned runs verify each block in its own process; a block that
    raises there is lost without a trace (upstream behaviour), so a partitioned ``ok`` is a lower bound.

    Returns:
        ``status`` (``ok``, ``timeout``, ``error`` or ``skipped`` when ``timeout_s <= 0``),
        ``partitioned``, ``lanelets``, ``seconds`` and, when ``ok``, ``invalid_states``: per formula,
        the count and up to :data:`VERIFIER_EXAMPLES` locations (element ids).
    """
    from crdesigner.verification_repairing.partitioning.map_partition import pymetis_imported

    lanelets = 0 if lanelet_network is None else len(lanelet_network.lanelets)
    partitioned = lanelets > partition_above and pymetis_imported
    result: dict[str, Any] = {"status": "skipped", "partitioned": partitioned, "lanelets": lanelets, "seconds": 0.0}
    if timeout_s <= 0 or lanelet_network is None:
        return result
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    child = context.Process(target=_verify_in_child, kwargs={"lanelet_network": lanelet_network, "partitioned": partitioned, "results": results})
    start = time.monotonic()
    child.start()
    outcome = None
    while outcome is None and time.monotonic() - start < timeout_s:
        try:
            outcome = results.get(timeout=1.0)
        except queue.Empty:
            if not child.is_alive():
                outcome = {"status": "error", "error": f"verifier process exited with code {child.exitcode}"}
    if outcome is None:
        outcome = {"status": "timeout"}
        _kill_process_group(pid=child.pid)
    child.join(timeout=10.0)
    if child.is_alive():
        _kill_process_group(pid=child.pid)
        child.join()
    result.update(outcome)
    result["seconds"] = round(time.monotonic() - start, 1)
    return result


def _verify_in_child(*, lanelet_network: Any, partitioned: bool, results: Any) -> None:
    os.setpgrp()  # the partitioned verifier starts its own processes; a timeout kills the whole group
    try:
        from crdesigner.verification_repairing.config import MapVerParams
        from crdesigner.verification_repairing.verification.formula_ids import extract_formula_ids
        from crdesigner.verification_repairing.verification.map_verifier import MapVerifier

        config = MapVerParams()
        config.verification.formulas = extract_formula_ids()
        config.evaluation.partitioned = partitioned
        states = MapVerifier(lanelet_network, config).verify()
        results.put(
            {
                "status": "ok",
                "invalid_states": {
                    getattr(formula, "name", str(formula)): {
                        "count": len(locations),
                        "examples": [list(location) for location in sorted(locations)[:VERIFIER_EXAMPLES]],
                    }
                    for formula, locations in sorted(states.items(), key=lambda item: str(item[0]))
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - report-only: any verifier failure is recorded, not raised
        results.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def _kill_process_group(*, pid: Optional[int]) -> None:
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def validate(
    *,
    map_dir: str,
    lanelet_network: Optional[Any],
    verifier_timeout_s: float,
    config: Optional[ValidationConfig] = None,
) -> Verdict:
    """Check a converted map: :mod:`map_quality` metrics against ``config`` and the map verifier.

    Args:
        map_dir: The conversion's output folder (with ``metadata.json``).
        lanelet_network: The CommonRoad lanelet network before boundary simplification (see
            :func:`run_map_verifier`); ``None`` skips the verifier.
        verifier_timeout_s: Cap on the map verifier; ``0`` skips it.
        config: Thresholds; the defaults when ``None``.

    Returns:
        The verdict. Failed checks are logged as warnings.
    """
    config = config or ValidationConfig()
    outputs = map_quality.ConversionOutputs.load(map_dir=map_dir)
    metrics, errors = map_quality.compute_all(outputs=outputs)
    checks = evaluate(metrics=metrics, config=config)
    failed = [c for c in checks if not c.passed]
    for check in failed:
        logger.warning(f"Validation: {check.name} = {check.value} (limit {check.limit}): {check.detail}")
    verifier = run_map_verifier(
        lanelet_network=lanelet_network,
        timeout_s=verifier_timeout_s,
        partition_above=config.verifier_partition_above,
    )
    logger.info(f"Map verifier: {verifier['status']} in {verifier['seconds']} s")
    status = "pass" if not failed else ("fail" if config.ratified else "warn")
    return Verdict(status=status, ratified=config.ratified, checks=checks, metric_errors=errors, verifier=verifier)
