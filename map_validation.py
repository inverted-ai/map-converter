"""Self-validation of one conversion: threshold checks on its outputs plus the CommonRoad map verifier.

``convert_map.py`` runs :func:`validate` after writing the outputs and stores the :class:`Verdict` in
``<map>_conversion_report.json``. Two :data:`PROFILES` set the thresholds: ``perfect`` (no fault of
any kind) and ``strict`` (bounded faults). Every run is measured against both; the verdict is the one
of the preset :class:`AutoCriticConfig` chooses, with its threshold overrides. The default preset,
``none``, judges nothing: the verdict is ``not_judged`` and lists the checks ``perfect`` would fail. The output metrics come from :mod:`map_quality`; whether each traffic light and
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
        max_linked_gaps: Most linked lanelet pairs whose ends lie farther apart than the seam
            tolerance (``None``: listed, not judged).
        max_dropped_signals: Most signals dropped because they lie past their road's end (``None``:
            listed, not judged).
        far_signal_m: A light or sign signal whose nearest stopline row lies farther than this is far.
        max_far_signals: Most far signals (``None``: listed, not judged).
    """

    min_routing_edge_pct: float
    min_seam_handshake_pct: float
    min_crossed_count: int
    max_crossed_fraction: float
    max_point_extent_m: float = 100_000.0
    max_lanelets_without_subtype: int = 0
    max_linked_gaps: Optional[int] = None
    max_dropped_signals: Optional[int] = None
    far_signal_m: float = 50.0
    max_far_signals: Optional[int] = None


@dataclasses.dataclass(slots=True)
class ConversionFindings:
    """What the conversion itself found, each item ``{id, location, detail}`` in the local frame.

    Attributes:
        linked_gaps: Linked lanelet pairs whose ends lie farther apart than the seam tolerance in use.
        dropped_signals: Signals dropped because they lie past their road's end (H14).
        signal_distances: Per light and sign signal with stopline rows, its distance to the nearest
            (``distance_m`` instead of ``detail``).
        painted_stop_lines: Painted stop lines, listed and never converted.
    """

    linked_gaps: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    dropped_signals: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    signal_distances: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    painted_stop_lines: list[dict[str, Any]] = dataclasses.field(default_factory=list)


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
NO_PRESET = "none"
PRESETS = (NO_PRESET, *PROFILES)
THRESHOLD_NAMES = tuple(f.name for f in dataclasses.fields(ValidationConfig))


@dataclasses.dataclass(slots=True)
class AutoCriticConfig:
    """Whether and how the converter judges its own output.

    Attributes:
        enabled: Run :func:`validate` and record the verdict; off records none.
        preset: ``none`` (judge nothing), or the :data:`PROFILES` entry the verdict uses:
            ``perfect`` (no fault of any kind) or ``strict`` (bounded faults). Both profiles are
            always measured. A failing preset withholds the map outputs (``convert_map`` exit code 4).
        thresholds: Overrides of the chosen preset's :class:`ValidationConfig` thresholds, by field
            name, e.g. ``{"min_routing_edge_pct": 98}``; ignored without a preset.
    """

    enabled: bool = True
    preset: str = NO_PRESET
    thresholds: dict[str, float] = dataclasses.field(default_factory=dict)


def check_config(*, config: AutoCriticConfig) -> None:
    """Raise ``ValueError`` naming an unknown preset or threshold."""
    if config.preset not in PRESETS:
        raise ValueError(f"autocritic.preset must be one of {', '.join(PRESETS)}, not {config.preset!r}")
    unknown = sorted(set(config.thresholds) - set(THRESHOLD_NAMES))
    if unknown:
        raise ValueError(f"autocritic.thresholds: unknown {', '.join(unknown)}; known: {', '.join(THRESHOLD_NAMES)}")


def preset_config(*, preset: str, thresholds: dict[str, float]) -> Optional[ValidationConfig]:
    """The thresholds ``preset`` judges with, overrides applied; ``None`` for ``none``."""
    if preset == NO_PRESET:
        return None
    annotations = {f.name: str(f.type) for f in dataclasses.fields(ValidationConfig)}
    overrides = {name: (int if "int" in annotations[name] else float)(value) for name, value in thresholds.items()}
    return dataclasses.replace(PROFILES[preset], **overrides)


@dataclasses.dataclass(slots=True)
class ControlCoverage:
    """The traffic lights and stop/yield signs in the converted area, and those without a stopline.

    Attributes:
        lights: Traffic lights (OpenDRIVE dynamic signals converted to lights).
        lights_without_stopline: OpenDRIVE ids of the lights no stopline row names.
        stop_yield_signs: Stop and yield signs: signals the sign tables read as STOP or YIELD, plus
            stop/yield stencil objects.
        signs_without_stopline: Those with no stopline, as ``"signal <id>"`` / ``"object <id>"``.
        positions: Local ``[x, y]`` of each light and sign without a stopline, by the ids above.
    """

    lights: int
    lights_without_stopline: list[str]
    stop_yield_signs: int
    signs_without_stopline: list[str]
    positions: dict[str, list[float]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class Check:
    """One threshold check: ``value`` against ``limit``; a failed one lists where it fails (``locations``)."""

    name: str
    value: Any
    limit: Any
    passed: bool
    detail: str = ""
    locations: list[dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class Verdict:
    """The outcome of :func:`validate`.

    Attributes:
        status: ``pass`` or ``fail`` under ``profile``; ``not_judged`` without a preset.
        profile: The preset the verdict uses (``none``, ``perfect`` or ``strict``).
        profiles: Per profile, its ``status`` and the names of its ``failed`` checks (no overrides).
        checks: Every check of ``profile`` with ``thresholds``, passed or not; without a preset,
            those of ``perfect``, for information.
        thresholds: The thresholds the verdict used (``{}`` without a preset).
        controls: The lights and signs checked for stoplines.
        metric_errors: Metrics that raised, as ``"<Class>: <message>"``.
        verifier: The map verifier's result (see :func:`run_map_verifier`).
    """

    status: str
    profile: str
    profiles: dict[str, dict[str, Any]]
    checks: list[Check]
    thresholds: dict[str, Any]
    controls: ControlCoverage
    metric_errors: dict[str, str]
    verifier: dict[str, Any]


def evaluate(
    *,
    metrics: dict[str, Any],
    metric_errors: dict[str, str],
    controls: ControlCoverage,
    config: ValidationConfig,
    findings: Optional[ConversionFindings] = None,
) -> list[Check]:
    """The threshold checks over :func:`map_quality.compute_all` results, ``controls`` and ``findings``.

    A metric that raised (in ``metric_errors``) fails its check; one that returned ``None`` has nothing
    to measure (e.g. no seams) and passes. A check whose limit is ``None`` lists its places and passes.
    """
    findings = findings or ConversionFindings()
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

    def listed(*, name: str, places: list[dict[str, Any]], limit: Optional[int], detail: str) -> None:
        passed = limit is None or len(places) <= limit
        checks.append(Check(name=name, value=len(places), limit=limit, passed=passed, detail=detail, locations=places))

    listed(
        name="linked_gaps",
        places=findings.linked_gaps,
        limit=config.max_linked_gaps,
        detail="linked lanelet pairs whose ends lie farther apart than the seam tolerance (not routable)",
    )
    listed(
        name="dropped_signals",
        places=findings.dropped_signals,
        limit=config.max_dropped_signals,
        detail="signals dropped because they lie past their road's end",
    )
    far = [
        {"id": d["id"], "location": d["location"], "detail": f"{d['distance_m']} m from its nearest stopline"}
        for d in findings.signal_distances
        if d["distance_m"] > config.far_signal_m
    ]
    listed(
        name="far_signals",
        places=far,
        limit=config.max_far_signals,
        detail=f"lights and signs whose nearest stopline lies more than {config.far_signal_m:g} m away",
    )
    listed(
        name="not_converted",
        places=findings.painted_stop_lines,
        limit=None,
        detail="painted stop lines, never converted (unrecognised and dropped elements: see the report summary)",
    )
    return checks


def outcome_message(*, profile: str, failed: dict[str, list[str]]) -> Optional[str]:
    """The warning for a failed verdict, or ``None`` when ``profile`` passed.

    Args:
        profile: The chosen profile.
        failed: Per profile, the names of its failed checks.
    """
    if profile == NO_PRESET or not failed[profile]:
        return None
    message = f"Validation ({profile}) failed: {', '.join(failed[profile])}."
    if profile == "perfect":
        if failed["strict"]:
            message += f" The strict profile fails too ({', '.join(failed['strict'])}): review the map by hand."
        else:
            message += (
                " The strict profile passes: convert with autocritic.preset=strict to accept these faults,"
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



def fault_locations(
    *, check: str, outputs: map_quality.ConversionOutputs, controls: ControlCoverage
) -> list[dict[str, Any]]:
    """Where ``check`` fails: each ``{id, location, detail}``, location in the local frame (empty when it has none)."""
    if check == "traffic_light_stoplines":
        missing = controls.lights_without_stopline
    elif check == "stop_yield_stoplines":
        missing = controls.signs_without_stopline
    else:
        listing = map_quality.FAULT_LOCATIONS.get(check)
        try:
            return [] if listing is None else listing(outputs=outputs)
        except Exception as exc:  # noqa: BLE001 - the check already failed; a missing listing must not hide it
            logger.warning(f"Could not list where {check} fails: {type(exc).__name__}: {exc}")
            return []
    return [
        {"id": element, "location": controls.positions.get(element), "detail": "no stopline"} for element in missing
    ]


def validate(
    *,
    map_dir: str,
    lanelet_network: Optional[Any],
    controls: ControlCoverage,
    verifier_timeout_s: float,
    profile: str = NO_PRESET,
    thresholds: Optional[dict[str, float]] = None,
    findings: Optional[ConversionFindings] = None,
) -> Verdict:
    """Check a converted map: :mod:`map_quality` metrics and ``controls`` under every profile, and the map verifier.

    Args:
        map_dir: The conversion's output folder (with ``metadata.json``).
        lanelet_network: The CommonRoad lanelet network before boundary simplification (see
            :func:`run_map_verifier`); ``None`` skips the verifier (``convert_map`` passes ``None``
            for a trimmed map, whose untrimmed network the other checks do not judge).
        controls: The lights and stop/yield signs and which of them got a stopline.
        verifier_timeout_s: Cap on the map verifier; ``0`` skips it.
        profile: The preset the verdict uses (:data:`PRESETS`).
        thresholds: Overrides of the preset's thresholds (:func:`preset_config`).
        findings: What the conversion found (linked gaps, dropped and far signals, painted stop lines).

    Returns:
        The verdict. Its failed checks, and :func:`outcome_message`, are logged as warnings.
    """
    outputs = map_quality.ConversionOutputs.load(map_dir=map_dir)
    metrics, errors = map_quality.compute_all(outputs=outputs)
    checks = {
        name: evaluate(metrics=metrics, metric_errors=errors, controls=controls, config=config, findings=findings)
        for name, config in PROFILES.items()
    }
    failed = {name: [c.name for c in profile_checks if not c.passed] for name, profile_checks in checks.items()}
    config = preset_config(preset=profile, thresholds=thresholds or {})
    judged = (
        checks["perfect"]
        if config is None
        else evaluate(metrics=metrics, metric_errors=errors, controls=controls, config=config, findings=findings)
    )
    judged = [
        check
        if check.passed or check.locations
        else dataclasses.replace(
            check, locations=fault_locations(check=check.name, outputs=outputs, controls=controls)
        )
        for check in judged
    ]
    judged_failed = [c.name for c in judged if not c.passed]
    for check in judged:
        if not check.passed:
            log = logger.info if config is None else logger.warning
            log(f"Validation: {check.name} = {check.value} (limit {check.limit}): {check.detail}")
    message = outcome_message(profile=profile, failed={**failed, profile: judged_failed})
    if message is not None:
        logger.warning(message)
    verifier = run_map_verifier(
        lanelet_network=lanelet_network,
        timeout_s=verifier_timeout_s,
        partition_above=VERIFIER_PARTITION_ABOVE,
    )
    logger.info(f"Map verifier: {verifier['status']} in {verifier['seconds']} s")
    return Verdict(
        status="not_judged" if config is None else "fail" if judged_failed else "pass",
        profile=profile,
        profiles={name: {"status": "fail" if names else "pass", "failed": names} for name, names in failed.items()},
        checks=judged,
        thresholds={} if config is None else dataclasses.asdict(config),
        controls=controls,
        metric_errors=errors,
        verifier=verifier,
    )
