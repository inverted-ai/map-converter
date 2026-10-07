"""Self-validation of one conversion: threshold checks on its outputs plus the CommonRoad map verifier.

``convert_map.py`` runs :func:`validate` after writing the outputs and stores the :class:`Verdict` in
``<map>_conversion_report.json``. Two :data:`PROFILES` set the thresholds: ``perfect`` (the default,
no fault of any kind) and ``strict`` (bounded faults). Every run is judged against both; the verdict
is the chosen one's. The output metrics come from :mod:`map_quality`; whether each traffic light and
stop/yield sign got a stopline (:class:`ControlCoverage`) comes from the converter, because the
``.osm`` does not carry the OpenDRIVE ids of the controls.
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
VERIFIER_PARTITION_ABOVE = 200
LISTED_IDS = 10


@dataclasses.dataclass(slots=True, frozen=True)
class ValidationConfig:
    """Thresholds of the self-validation checks.

    Attributes:
        min_routing_edge_pct: Lowest share of lanelets with a routing edge.
        min_seam_handshake_pct: Lowest share of geometric seams whose end nodes are shared.
        min_crossed_count: Crossed lanelets (outline folds or touches itself) fail the check only
            from this many on ...
        max_crossed_fraction: ... and only above this share of all lanelets.
        max_point_extent_m: Furthest a point may lie from the local origin.
        max_lanelets_without_subtype: Most lanelets allowed without a ``subtype`` tag.
    """

    min_routing_edge_pct: float
    min_seam_handshake_pct: float
    min_crossed_count: int
    max_crossed_fraction: float
    max_point_extent_m: float = 100_000.0
    max_lanelets_without_subtype: int = 0


PROFILES = {
    "perfect": ValidationConfig(
        min_routing_edge_pct=100.0,
        min_seam_handshake_pct=100.0,
        min_crossed_count=1,
        max_crossed_fraction=0.0,
    ),
    "strict": ValidationConfig(
        min_routing_edge_pct=95.0,
        min_seam_handshake_pct=95.0,
        min_crossed_count=3,
        max_crossed_fraction=0.02,
    ),
}
DEFAULT_PROFILE = "perfect"


@dataclasses.dataclass(slots=True)
class ControlCoverage:
    """The traffic lights and stop/yield signs in the converted area, and those without a stopline.

    Attributes:
        lights: Traffic lights (OpenDRIVE dynamic signals converted to lights).
        lights_without_stopline: OpenDRIVE ids of the lights no stopline row names.
        stop_yield_signs: Stop and yield signs: signals the sign tables read as STOP or YIELD, plus
            stop/yield stencil objects.
        signs_without_stopline: Those with no stopline, as ``"signal <id>"`` / ``"object <id>"``.
    """

    lights: int
    lights_without_stopline: list[str]
    stop_yield_signs: int
    signs_without_stopline: list[str]


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
        status: ``pass`` or ``fail`` under ``profile``.
        profile: The profile the verdict uses.
        profiles: Per profile, its ``status`` and the names of its ``failed`` checks.
        checks: Every check of ``profile``, passed or not.
        controls: The lights and signs checked for stoplines.
        metric_errors: Metrics that raised, as ``"<Class>: <message>"``.
        verifier: The map verifier's result (see :func:`run_map_verifier`).
    """

    status: str
    profile: str
    profiles: dict[str, dict[str, Any]]
    checks: list[Check]
    controls: ControlCoverage
    metric_errors: dict[str, str]
    verifier: dict[str, Any]


def evaluate(
    *,
    metrics: dict[str, Any],
    metric_errors: dict[str, str],
    controls: ControlCoverage,
    config: ValidationConfig,
) -> list[Check]:
    """The threshold checks over :func:`map_quality.compute_all` results and ``controls``.

    A metric that raised (in ``metric_errors``) fails its check; one that returned ``None`` has nothing
    to measure (e.g. no seams) and passes.
    """
    checks = []

    def add(*, name: str, metric: Optional[str], value: Any, limit: Any, passed: bool, detail: str) -> None:
        if metric in metric_errors:
            value, passed, detail = None, False, f"metric failed: {metric_errors[metric]}"
        checks.append(Check(name=name, value=value, limit=limit, passed=passed, detail=detail))

    def at_least(*, name: str, limit: float, detail: str) -> None:
        value = metrics.get(name)
        add(name=name, metric=name, value=value, limit=limit, passed=value is None or value >= limit, detail=detail)

    def at_most(*, name: str, metric: str, value: Optional[float], limit: float, detail: str) -> None:
        add(name=name, metric=metric, value=value, limit=limit, passed=value is None or value <= limit, detail=detail)

    def every_control(*, name: str, total: int, missing: list[str], what: str) -> None:
        listed = ", ".join(missing[:LISTED_IDS]) + (", ..." if len(missing) > LISTED_IDS else "")
        add(
            name=name,
            metric=None,
            value=len(missing),
            limit=0,
            passed=not missing,
            detail=f"{len(missing)} of {total} {what} have no stopline" + (f": {listed}" if missing else ""),
        )

    lanelets = metrics.get("lanelet_count")
    add(name="has_lanelets", metric="lanelet_count", value=lanelets, limit=1, passed=bool(lanelets), detail="the map is not empty")
    at_least(
        name="routing_edge_pct",
        limit=config.min_routing_edge_pct,
        detail="share of lanelets with a predecessor, successor or neighbour in the routing graph",
    )
    at_least(
        name="seam_handshake_pct",
        limit=config.min_seam_handshake_pct,
        detail="share of meeting lanelet ends that share their nodes",
    )
    crossed = metrics.get("crossed_geometry")
    count = None if crossed is None else sum(crossed.values())
    fraction = None if count is None or not lanelets else count / lanelets
    add(
        name="crossed_lanelets",
        metric="crossed_geometry",
        value=None if count is None else {"count": count, "fraction": fraction},
        limit={"min_count": config.min_crossed_count, "max_fraction": config.max_crossed_fraction},
        passed=count is None or count < config.min_crossed_count or (fraction or 0.0) <= config.max_crossed_fraction,
        detail="lanelets whose outline folds or touches itself; fails at min_count or more above max_fraction",
    )
    every_control(
        name="traffic_light_stoplines",
        total=controls.lights,
        missing=controls.lights_without_stopline,
        what="traffic lights",
    )
    every_control(
        name="stop_yield_stoplines",
        total=controls.stop_yield_signs,
        missing=controls.signs_without_stopline,
        what="stop/yield signs",
    )
    at_most(
        name="lanelets_without_subtype",
        metric="lanelets_without_subtype",
        value=metrics.get("lanelets_without_subtype"),
        limit=config.max_lanelets_without_subtype,
        detail="lanelets consumers that filter by subtype would drop",
    )
    at_most(
        name="point_extent_m",
        metric="point_extent_m",
        value=metrics.get("point_extent_m"),
        limit=config.max_point_extent_m,
        detail="furthest point from the local origin",
    )
    origin = metrics.get("origin")
    add(
        name="origin_in_range",
        metric="origin",
        value=origin,
        limit=[90.0, 180.0],
        passed=origin is not None and abs(origin[0]) <= 90.0 and abs(origin[1]) <= 180.0,
        detail="lanelet_map_origin is a valid latitude/longitude",
    )
    return checks


def outcome_message(*, profile: str, failed: dict[str, list[str]]) -> Optional[str]:
    """The warning for a failed verdict, or ``None`` when ``profile`` passed.

    Args:
        profile: The chosen profile.
        failed: Per profile, the names of its failed checks.
    """
    if not failed[profile]:
        return None
    message = f"Validation ({profile}) failed: {', '.join(failed[profile])}."
    if profile == "perfect":
        if failed["strict"]:
            message += f" The strict profile fails too ({', '.join(failed['strict'])}): review the map by hand."
        else:
            message += (
                " The strict profile passes: convert with validation_profile=strict to accept these faults,"
                " and review them by hand."
            )
    else:
        message += " Review the map by hand."
    return message


def run_map_verifier(*, lanelet_network: Optional[Any], timeout_s: float, partition_above: int) -> dict[str, Any]:
    """Run CommonRoad's ``MapVerifier`` (report-only) in a child process killed after ``timeout_s``.

    The verifier assumes each lanelet's bounds have equal vertex counts, so pass the network before
    boundary simplification. Partitioned runs verify each block in its own process; a block that
    raises there is lost without a trace (upstream behaviour), so a partitioned ``ok`` is a lower bound.

    Returns:
        ``status`` (``ok``, ``timeout``, ``error``, or ``skipped`` with no network or ``timeout_s <= 0``),
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
    controls: ControlCoverage,
    verifier_timeout_s: float,
    profile: str = DEFAULT_PROFILE,
) -> Verdict:
    """Check a converted map: :mod:`map_quality` metrics and ``controls`` under every profile, and the map verifier.

    Args:
        map_dir: The conversion's output folder (with ``metadata.json``).
        lanelet_network: The CommonRoad lanelet network before boundary simplification (see
            :func:`run_map_verifier`); ``None`` skips the verifier (``convert_map`` passes ``None``
            for a trimmed map, whose untrimmed network the other checks do not judge).
        controls: The lights and stop/yield signs and which of them got a stopline.
        verifier_timeout_s: Cap on the map verifier; ``0`` skips it.
        profile: The :data:`PROFILES` entry the verdict uses.

    Returns:
        The verdict. Its failed checks, and :func:`outcome_message`, are logged as warnings.
    """
    outputs = map_quality.ConversionOutputs.load(map_dir=map_dir)
    metrics, errors = map_quality.compute_all(outputs=outputs)
    checks = {
        name: evaluate(metrics=metrics, metric_errors=errors, controls=controls, config=config)
        for name, config in PROFILES.items()
    }
    failed = {name: [c.name for c in profile_checks if not c.passed] for name, profile_checks in checks.items()}
    for check in checks[profile]:
        if not check.passed:
            logger.warning(f"Validation: {check.name} = {check.value} (limit {check.limit}): {check.detail}")
    message = outcome_message(profile=profile, failed=failed)
    if message is not None:
        logger.warning(message)
    verifier = run_map_verifier(
        lanelet_network=lanelet_network,
        timeout_s=verifier_timeout_s,
        partition_above=VERIFIER_PARTITION_ABOVE,
    )
    logger.info(f"Map verifier: {verifier['status']} in {verifier['seconds']} s")
    return Verdict(
        status="fail" if failed[profile] else "pass",
        profile=profile,
        profiles={name: {"status": "fail" if names else "pass", "failed": names} for name, names in failed.items()},
        checks=checks[profile],
        controls=controls,
        metric_errors=errors,
        verifier=verifier,
    )
