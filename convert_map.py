"""
The input is either a folder `dir_path` with a single *.xodr file in it, or the file itself as
`xodr_path`. The name of that file will be the map name.
This script produces the required files for TorchDriveSim in `dir_path`, or for `xodr_path` in
the folder `<xodr folder>/<map name>/`.
"""
import copy
import dataclasses
import datetime
import glob
import importlib.metadata
import json
import os
import re
import sys
import logging
import math
from typing import Tuple, Optional

import imageio
import numpy as np
from pathlib import Path

import torch.cuda
from lxml import etree
import pyproj
import yaml
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

import lanelet2
from torchdrivesim.map import Stopline, MapConfig, store_map_config, resolve_paths_to_absolute
from torchdrivesim.lanelet2 import LaneletMap, load_lanelet_map, lanelet_map_to_lane_mesh
from torchdrivesim.mesh import BirdviewMesh, BirdviewRGBMeshGenerator
from torchdrivesim.rendering import renderer_from_config, RendererConfig
from torchdrivesim.map import traffic_controls_from_map_config
from torchdrivesim.utils import Resolution

from crdesigner.common.config.general_config import GeneralConfig
from crdesigner.common.config.lanelet2_config import lanelet2_config
from crdesigner.map_conversion.common.utils import clean_projection_string
from crdesigner.map_conversion.lanelet2.cr2lanelet import CR2LaneletConverter
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.parser import parse_opendrive
from crdesigner.common.config.opendrive_config import OpenDriveConfig
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_conversion.network import Network
from commonroad.scenario.lanelet import LaneletNetwork

import autocritic_report
import conversion_heuristics
import conversion_report
import lanelet_bounds
import map_quality
import map_validation
import traffic_controls
import vocabulary
from conversion_common import (
    MAP_SERVICE_MAX_OSM_MB,
    REFUSED_EXIT_CODE,
    REJECTED_EXIT_CODE,
    MapRejected,
    git_identity,
    sha256_file,
)
from opendrive_signal_groups import write_signal_groups


logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VISUALIZATION_FOV = 800.0
FOV_MARGIN = 1.05
CONVERTED_LANE_TYPES = ('driving', 'onRamp', 'offRamp', 'exit', 'entry', 'bidirectional')  # OpenDRIVE lane types made lanelets
PROVENANCE_PACKAGES = (
    'torchdrivesim', 'torch', 'lanelet2', 'commonroad-io', 'numpy', 'lxml', 'pyproj', 'shapely', 'omegaconf',
)


@dataclasses.dataclass
class MapConversionConfig:
    """Options for one ``convert_map.py`` run, given on the command line as ``field=value``.

    Attributes:
        dir_path: Directory holding exactly one ``.xodr``; all outputs are written here.
        xodr_path: The ``.xodr`` itself, instead of ``dir_path``; outputs go to
            ``<xodr folder>/<map name>/``. Exactly one of the two is required.
        domain: Prefix for ``iai_location_name`` in ``metadata.json`` (``<domain>:<map>``);
            unset leaves it null.
        lane_marking_join_threshold: Distance (m) under which the markings of two neighbouring
            lanes are drawn as one line in the mesh.
        visualization_fov: Width (m) of the square area ``visualization.png`` shows; unset fits the
            whole mesh around ``center`` with a 5% margin.
        center: Map centre ``[x, y]`` in local metres, written to ``metadata.json`` and used as
            the render and trim centre; unset uses the mesh centre.
        trim_radius: Keep only lanelets and stoplines within this many metres of ``center``.
        trim_vertical_limits: ``[min_z, max_z]``; stoplines outside it are dropped, and with
            ``trim_radius`` lanelet points outside it are trimmed too.
        simplify_tolerance: Maximum lateral error (m) of the Douglas-Peucker simplification of
            lanelet boundaries before export; 0 or None disables it.
        osm_target_bytes: Size budget for the ``.osm``; the tolerance widens until the file fits
            and the run fails if it never does. None disables the budget. Default: the map
            service's limit (``MAP_SERVICE_MAX_OSM_MB``, 30 MiB).
        simplify_max_iterations: Attempts at fitting the size budget.
        simplify_tolerance_factor: Factor the tolerance grows by per attempt.
        allow_zero_origin: Convert a map whose ``<geoReference>`` is present but unusable at
            latitude/longitude (0, 0) instead of failing; when the ``<geoReference>`` and header
            ``<offset>`` put road geometry outside the projection, also drop the offset (file
            coordinates). A map with no ``<geoReference>`` always converts at (0, 0), with a warning.
        verifier_timeout_s: Cap (s) on the CommonRoad map verifier the validation stage runs; 0
            (the default) skips it. A run that hits the cap records ``timeout`` and continues.
        verifier_formulas: The verifier's formulas to check, by name (e.g. ``[UNIQUE_ID]``); empty
            (the default) checks all of them.
        autofix: The guesses about the author's intent (:mod:`conversion_heuristics`): all of
            them on or off, and each one's settings, e.g.
            ``autofix.heuristics.sign_signal_stoplines.enabled=true``.
        autocritic: Whether the converter judges its output, under which preset and thresholds
            (:mod:`map_validation`), e.g. ``autocritic.preset=strict``. A failing preset withholds the
            map outputs (exit code 4); the reports are still written.
        report: Which parts of the AutoCritic report (:mod:`autocritic_report`) are written, e.g.
            ``report.closeups=false``.
        vocabulary: The vocabulary dictionary (:mod:`vocabulary`) that translates names the file uses into
            names the converter understands; default ``<repo>/vocabularies/default.yaml``.
        vocabulary_vendors: Folder of vendor dictionaries; each whose ``match`` fits the file header applies on
            top of ``vocabulary``. Default ``<repo>/vocabularies/vendors``.
        speed_unit: The unit of the map's speeds that name none: ``m/s`` (the default), ``km/h``, ``mph``, or a
            number that converts the map's unit to m/s (``0.3048`` for feet per second). Speeds reach the map in m/s.
    """

    dir_path: Optional[str] = None
    xodr_path: Optional[str] = None
    domain: Optional[str] = None
    lane_marking_join_threshold: float = 0.1
    visualization_fov: Optional[float] = None
    center: Optional[Tuple[float, float]] = None
    trim_radius: Optional[float] = None
    trim_vertical_limits: Optional[Tuple[float, float]] = None
    # Raise (e.g. 0.5) for very large maps where file size matters more than curve fidelity.
    simplify_tolerance: Optional[float] = 0.2
    osm_target_bytes: Optional[int] = MAP_SERVICE_MAX_OSM_MB << 20
    simplify_max_iterations: int = 4
    simplify_tolerance_factor: float = 2.0
    allow_zero_origin: bool = False
    verifier_timeout_s: float = 0.0
    verifier_formulas: list[str] = dataclasses.field(default_factory=list)
    autofix: conversion_heuristics.AutoFixConfig = dataclasses.field(default_factory=conversion_heuristics.AutoFixConfig)
    autocritic: map_validation.AutoCriticConfig = dataclasses.field(default_factory=map_validation.AutoCriticConfig)
    report: autocritic_report.ReportConfig = dataclasses.field(default_factory=autocritic_report.ReportConfig)
    # vocabulary_vendors first: the field named vocabulary shadows the module in this class body
    vocabulary_vendors: str = vocabulary.VENDORS_DIR
    vocabulary: str = vocabulary.DEFAULT_PATH
    speed_unit: str = "m/s"


@dataclasses.dataclass
class GeoOffset:
    """The OpenDRIVE header ``<offset>``: projected = file coordinates + ``(x, y)``.

    ``hdg`` (radians) and ``z`` are read but not applied: ``parse_xodr`` warns when they are non-zero.
    """

    x: float
    y: float
    z: float
    hdg: float

    @classmethod
    def from_dict(cls, offset_dict: dict):
        """Build from the parsed header ``offset`` attribute dict."""
        x = float(offset_dict['x'])
        y = float(offset_dict['y'])
        z = float(offset_dict['z'])
        hdg = float(offset_dict['hdg'])
        return cls(x, y, z, hdg)

    def apply_offset(self, x, y):
        """File ``(x, y)`` in projected coordinates."""
        return x + float(self.x), y + float(self.y)

@dataclasses.dataclass
class GeoReference:
    """Latitude/longitude of the map's local origin.

    ``translation`` is subtracted from projected (geo-referenced) coordinates to get local
    coordinates: the map centre for ``utm`` and for ``tmerc`` with a false easting/northing, else zero.
    """

    lat: float
    lon: float
    translation: Tuple[float, float] = (0.0, 0.0)
    zone: Optional[int] = None

    @property
    def origin(self):
        """``(lat, lon)``, as written to ``metadata.json``'s ``lanelet_map_origin``."""
        return self.lat, self.lon

    @property
    def lanelet2_origin(self):
        """The origin as a ``lanelet2.io.Origin``."""
        return lanelet2.io.Origin(self.lat, self.lon)

    @property
    def proj_string(self):
        """A transverse-Mercator PROJ string centred on the origin."""
        return f'+proj=tmerc +lat_0={self.lat} +lon_0={self.lon}'


class InputError(ValueError):
    """The OpenDRIVE input cannot be converted as configured (reported, never written around)."""


def parse_geo_reference(*, text: Optional[str], centre: Optional[Tuple[float, float]]) -> Optional[GeoReference]:
    """Origin and local-frame translation from an OpenDRIVE ``<geoReference>`` PROJ string.

    Args:
        text: The ``<geoReference>`` contents.
        centre: The map centre in projected coordinates; needed for ``utm``.

    Returns:
        ``tmerc`` without a false easting/northing: origin ``(lat_0, lon_0)``. ``utm``, and ``tmerc``
        with one: origin at the map centre, translation = the centre. ``None`` for a missing,
        unparseable or other projection.
    """
    if not text or not text.strip():
        return None
    text = clean_projection_string(text)
    # Vendor exports carry malformed tokens (e.g. "+=alt_0=0"); keep well-formed +key[=value] ones.
    tokens = []
    for token in text.split():
        candidate = "+" + token.lstrip("+")
        if re.fullmatch(r"\+[A-Za-z_]\w*(=\S+)?", candidate):
            tokens.append(candidate)
        else:
            logger.warning(f'Ignoring malformed <geoReference> token {token!r}')
    text = " ".join(tokens)
    try:
        crs = pyproj.CRS.from_user_input(text)
    except pyproj.exceptions.CRSError:
        return None
    # pyproj validates; values are read as written, since its PROJ round trip keeps only 15 digits.
    params = dict(token[1:].split("=", 1) for token in text.split() if token.startswith("+") and "=" in token)
    proj = params.get("proj")
    false_origin = (float(params.get("x_0", 0.0)), float(params.get("y_0", 0.0)))
    if proj == "tmerc" and (false_origin == (0.0, 0.0) or centre is None):
        return GeoReference(lat=float(params.get("lat_0", 0.0)), lon=float(params.get("lon_0", 0.0)), translation=false_origin)
    # UTM and false-origin coordinates run to millions of metres, past float32 mesh precision,
    # so the local origin moves to the map centre.
    if proj in ("tmerc", "utm") and centre is not None:
        lon, lat = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(*centre)
        zone = int(params["zone"]) if proj == "utm" and "zone" in params else None
        return GeoReference(lat=lat, lon=lon, translation=(float(centre[0]), float(centre[1])), zone=zone)
    return None


def check_geo_reference(*, geo_reference: GeoReference, source: str) -> None:
    """Fail on an origin no real map can have.

    Raises:
        InputError: ``|lat| >= 85`` or a UTM origin outside its declared zone.
    """
    if abs(geo_reference.lat) >= 85.0:
        raise InputError(f"{source}: origin latitude {geo_reference.lat:.3f} is outside the supported range")
    if geo_reference.zone is not None:
        zone = int((geo_reference.lon + 180.0) // 6.0) % 60 + 1
        if zone != geo_reference.zone:
            raise InputError(f"{source}: UTM origin longitude {geo_reference.lon:.3f} is outside zone {geo_reference.zone}")


@dataclasses.dataclass(slots=True)
class BidirectionalFixes:
    """Topology to drop around bidirectional lanelets (see :func:`bidirectional_fixes`)."""

    links: list[tuple[int, int]]  # (predecessor id, successor id) pairs to unlink
    reverse_connectors: list[int]  # lanelets retracing a bidirectional lanelet the other way


def bidirectional_fixes(*, lanelet_network: LaneletNetwork, tolerance: float = 5.0) -> BidirectionalFixes:
    """Find the links the OpenDRIVE conversion gets wrong around bidirectional lanelets.

    A bidirectional lanelet has one drawn direction. A lane meeting it at its far end, or head-on, is
    linked as if it met the near end, so the ``.osm`` writer shares nodes across the gap and the shape
    folds. A link touching a bidirectional lanelet is kept only where the predecessor ends at the
    successor's start. A predecessor running from the lanelet's end back to its start duplicates it
    in reverse; ``one_way=no`` covers travel against the drawn direction instead.

    Args:
        lanelet_network: The CommonRoad network exported from OpenDRIVE.
        tolerance: Metres within which an end point counts as touching a lanelet end.

    Returns:
        The links to unlink and the reverse connectors to remove.
    """
    links = []
    reverse_connectors = []
    for lanelet in lanelet_network.lanelets:
        if not lanelet.user_bidirectional:
            continue
        start, end = lanelet.left_vertices[0][:2], lanelet.left_vertices[-1][:2]
        for pred_id in lanelet.predecessor:
            pred = lanelet_network.find_lanelet_by_id(pred_id)
            if pred is None:
                continue
            pred_start, pred_end = pred.left_vertices[0][:2], pred.left_vertices[-1][:2]
            if math.dist(pred_start, end) < tolerance and math.dist(pred_end, start) < tolerance:
                reverse_connectors.append(pred_id)
            elif math.dist(pred_end, start) >= tolerance:
                links.append((pred_id, lanelet.lanelet_id))
        for succ_id in lanelet.successor:
            succ = lanelet_network.find_lanelet_by_id(succ_id)
            if succ is not None and math.dist(succ.left_vertices[0][:2], end) >= tolerance:
                links.append((lanelet.lanelet_id, succ_id))
    return BidirectionalFixes(links=links, reverse_connectors=reverse_connectors)


def trim_map(lanelet_map, center, radius, vertical_limits=None):
    """
    Trims a lanelet map to a circular neighbourhood around a given center.
    """
    trimmed_map = lanelet2.core.LaneletMap()
    center = lanelet2.core.Point3d(lanelet2.core.getId(), center[0], center[1], 0)
    if vertical_limits is None:
        min_ele, max_ele = -100000, 1000000
    else:
        min_ele, max_ele = vertical_limits
    for lanelet in lanelet_map.laneletLayer:
        left_boundary = [p for p in lanelet.leftBound
                         if lanelet2.geometry.distance(center, p) <= radius
                         and min_ele <= p.z <= max_ele]
        right_boundary = [p for p in lanelet.rightBound
                          if lanelet2.geometry.distance(center, p) <= radius
                          and min_ele <= p.z <= max_ele]
        if len(left_boundary) > 0 and len(right_boundary) > 0:
            left_boundary = lanelet2.core.LineString3d(lanelet.leftBound.id, left_boundary)
            right_boundary = lanelet2.core.LineString3d(lanelet.rightBound.id, right_boundary)
            trimmed_lanelet = lanelet2.core.Lanelet(lanelet.id, left_boundary, right_boundary)
            # trimmed_lanelet.attributes['type'] = lanelet.attributes['type']
            # trimmed_lanelet.attributes['subtype'] = lanelet.attributes['subtype']
            # trimmed_lanelet.attributes['is_intersection'] = lanelet.attributes['is_intersection']
            trimmed_map.add(trimmed_lanelet)
    return trimmed_map


class CustomTransfomer:
    """
    I could not find a way to construct a pyproj transformer equivalent to the Lanelet2 UtmProjector,
    so I opted to use a wrapper instead.

    ``to_local`` maps parsed OpenDRIVE coordinates into the map's local frame (header offset,
    then ``translation``); the ``.osm``, the mesh and the stoplines all use that frame.
    """
    def __init__(self, projector, offset: Optional[GeoOffset] = None, translation: Tuple[float, float] = (0.0, 0.0)):
        self.projector = projector
        self.offset = offset
        self.translation = translation

    def to_local(self, x, y):
        """Parsed OpenDRIVE ``(x, y)`` in the map's local metric frame."""
        if self.offset is not None:
            x, y = self.offset.apply_offset(x, y)
        return x - self.translation[0], y - self.translation[1]

    def transform(self, x, y):
        """Local ``(x, y)`` to ``(lat, lon)``, applying the header offset first."""
        x, y = self.to_local(x, y)
        transformed = self.projector.reverse(lanelet2.core.BasicPoint3d(x, y, 0))
        return transformed.lat, transformed.lon

@dataclasses.dataclass(slots=True)
class ParsedXodr:
    """The parsed OpenDRIVE input and its Lanelet2 geo-reference."""

    location: str
    opendrive: object
    geo_reference: GeoReference
    projector: object
    frame: CustomTransfomer
    words: vocabulary.Vocabulary  # the dictionary in use: base, then matching vendor dictionaries, combined
    traffic_side: Optional[str]  # "RHT", "LHT" or "mixed"
    geo_reference_status: str  # "declared", "absent" or "unusable_zero_origin_allowed"
    dropped_signals: list[dict] = dataclasses.field(default_factory=list)  # H14 drops: {id, location, detail}


@dataclasses.dataclass(slots=True)
class ExportedOsm:
    """The CommonRoad network built from the input and the ``.osm`` written from it.

    ``unsimplified_network`` is the network before boundary simplification, kept for the map verifier
    (which assumes a lanelet's bounds have equal vertex counts); ``None`` when the verifier is off or
    the map is trimmed.
    """

    road_network: Network
    lanelet_network: LaneletNetwork
    osm_path: str
    linked_gaps: list[dict]  # linked lanelet ends farther apart than the seam tolerance in use: {id, location, detail}
    unsimplified_network: Optional[LaneletNetwork] = None


@dataclasses.dataclass(slots=True)
class BuiltMesh:
    """The road + lane-marking mesh and where it was written."""

    combined_mesh: BirdviewMesh
    mesh_path: str


class ConfigError(ValueError):
    """An invalid :class:`MapConversionConfig`, found before anything is read or written."""


def load_config(*, argv: list[str]) -> MapConversionConfig:
    """The run's settings from ``key=value`` arguments: defaults, then the ``config=`` YAML file, then the other arguments.

    Raises:
        ConfigError: ``config=`` given twice, a missing or unreadable config file, an unknown key or
            a value of the wrong type.
    """
    config_paths = [arg.split('=', 1)[1] for arg in argv if arg.startswith('config=')]
    if len(config_paths) > 1:
        raise ConfigError('give config= at most once')
    layers = [OmegaConf.structured(MapConversionConfig)]
    for path in map(os.path.abspath, config_paths):
        if not os.path.isfile(path):
            raise ConfigError(f'config file {path} does not exist')
        try:
            layers.append(OmegaConf.load(path))
        except yaml.YAMLError as exc:
            raise ConfigError(f'config file {path} is not YAML: {exc}') from exc
    try:
        layers.append(OmegaConf.from_dotlist([arg for arg in argv if not arg.startswith('config=')]))
        return OmegaConf.to_object(OmegaConf.merge(*layers))
    except OmegaConfBaseException as exc:
        raise ConfigError(str(exc).splitlines()[0]) from exc


def map_name(*, xodr_path: str) -> str:
    """The map name: the ``.xodr`` file name without its extension."""
    return Path(xodr_path).stem


def validate_config(*, cfg: MapConversionConfig) -> str:
    """Check the configuration before parsing and return the ``.xodr`` to convert.

    Raises:
        ConfigError: neither or both of ``dir_path`` and ``xodr_path``, ``trim_radius`` without
            ``center``, a negative ``verifier_timeout_s``, an unknown ``autocritic.preset`` or
            threshold, an unusable ``autofix.heuristics`` parameter or ``vocabulary`` file, ``xodr_path`` not an existing ``.xodr`` file, or
            ``dir_path`` not holding exactly one ``.xodr``.
    """
    if (cfg.dir_path is None) == (cfg.xodr_path is None):
        raise ConfigError('give exactly one of dir_path and xodr_path')
    if cfg.trim_radius is not None and cfg.center is None:
        raise ConfigError('trim_radius requires center')
    if not vocabulary.speed_unit_ok(speed_unit=str(cfg.speed_unit)):
        raise ConfigError(
            f'speed_unit {cfg.speed_unit!r} is neither m/s, km/h, mph nor a positive number (the factor to m/s)'
        )
    if cfg.verifier_timeout_s < 0:
        raise ConfigError('verifier_timeout_s must be 0 (skip the verifier) or positive')
    unknown = sorted(set(cfg.verifier_formulas) - set(map_validation.verifier_formula_names()))
    if unknown:
        raise ConfigError(f'verifier_formulas: unknown {", ".join(unknown)}')
    try:
        vocabulary.load(path=cfg.vocabulary)
        vocabulary.load_vendors(folder=cfg.vocabulary_vendors)
        map_validation.check_config(config=cfg.autocritic)
        conversion_heuristics.check_config(config=cfg.autofix.heuristics)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if cfg.xodr_path is not None:
        if not (cfg.xodr_path.endswith('.xodr') and os.path.isfile(cfg.xodr_path)):
            raise ConfigError(f'xodr_path {os.path.abspath(cfg.xodr_path)} is not an existing .xodr file')
        return os.path.abspath(cfg.xodr_path)
    xodr_files = sorted(glob.glob(os.path.join(cfg.dir_path, '*.xodr')))
    if len(xodr_files) != 1:
        found = 'no .xodr file' if not xodr_files else (
            f'{len(xodr_files)} .xodr files: ' + ', '.join(os.path.basename(f) for f in xodr_files)
        )
        raise ConfigError(f'{os.path.abspath(cfg.dir_path)} must hold exactly one .xodr; found {found}')
    return xodr_files[0]


def projected_centre(*, opendrive, geo_offset: Optional[GeoOffset]) -> Optional[Tuple[float, float]]:
    """Map centre in projected coordinates: the header bounding box, else the road start points."""
    box = [float(getattr(opendrive.header, side) or 0.0) for side in ("north", "south", "east", "west")]
    if any(box):
        north, south, east, west = box
        centre = (east + west) / 2.0, (north + south) / 2.0
    else:
        starts = [g.start_position for road in opendrive.roads for g in road.plan_view._geometries]
        if not starts:
            return None
        xs, ys = zip(*starts)
        centre = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    return geo_offset.apply_offset(*centre) if geo_offset else centre


def _signal_position(*, road, signal, frame: CustomTransfomer) -> tuple[float, float]:
    return traffic_controls.element_position(road=road, s=float(signal.s), t=float(signal.t), frame=frame)


def signal_corrections(
    *, opendrive, words: vocabulary.Vocabulary, frame: CustomTransfomer
) -> list[tuple[object, dict]]:
    """The signal attributes the dictionary's ``signals`` section changes: ``(signal, {attribute: value})``.

    A signal marked dynamic whose type is no traffic-light code becomes a static sign; a sign with no
    country whose code the dictionary lists is read with that country's table. Each is recorded, and so is
    each light the reader skips as no vehicle light (``non_vehicle_light_types``).
    """
    corrections = []
    for road in opendrive.roads:
        for signal in road.signals:
            changes = {}
            if signal.dynamic == 'yes' and not vocabulary.is_traffic_light(signal_type=signal.type, vocabulary=words):
                changes['dynamic'] = 'no'
                vocabulary.record(
                    element='signal.dynamic', element_id=signal.id, value='yes', meaning='no',
                    message=f'type {signal.type!r} is no traffic-light code - a static sign',
                    location=_signal_position(road=road, signal=signal, frame=frame),
                )
            if changes.get('dynamic', signal.dynamic) == 'yes' and signal.type in words.signals.non_vehicle_light_types:
                vocabulary.record(
                    element='signal.type', element_id=signal.id, value=signal.type, meaning='not_a_vehicle_light',
                    message=f'light type {signal.type!r} is no vehicle light - no stoplines',
                    location=_signal_position(road=road, signal=signal, frame=frame),
                )
            if changes.get('dynamic', signal.dynamic) != 'yes':
                read_as = vocabulary.country_for_code(
                    country=signal.country, signal_type=signal.type, subtype=signal.subtype, vocabulary=words
                )
                if read_as is not None:
                    country, code_type, code_subtype = read_as
                    changes.update(country=country, type=code_type, subtype=code_subtype)
                    vocabulary.record(
                        element='signal.country', element_id=signal.id, value=signal.country or '', meaning=country,
                        message=f'no country, code {signal.type!r} - read as {country} sign {code_type}-{code_subtype}',
                        location=_signal_position(road=road, signal=signal, frame=frame),
                    )
            if changes:
                corrections.append((signal, changes))
    return corrections


def static_light_codes(*, opendrive, words: vocabulary.Vocabulary, frame: CustomTransfomer) -> list[dict]:
    """Signals with a traffic-light code (the dictionary's ``traffic_light_types``) marked ``dynamic="no"``.

    The reader takes only dynamic signals for lights, so these become static signs with no stoplines; whether the
    author meant a light is not decided here, AutoCritic lists them (``static_light_codes``).
    """
    if words.signals.traffic_light_types is None:
        return []
    return [
        map_quality.fault(
            element=f'signal {signal.id}',
            location=_signal_position(road=road, signal=signal, frame=frame),
            detail=f'traffic-light code {signal.type} marked dynamic="no": a static sign, no stoplines',
        )
        for road in opendrive.roads
        for signal in road.signals
        if signal.dynamic == 'no' and vocabulary.is_traffic_light(signal_type=signal.type, vocabulary=words)
    ]


def signals_past_road_end(
    *, opendrive, heuristics: conversion_heuristics.HeuristicsConfig, frame: CustomTransfomer
) -> list[tuple[object, object, Optional[float]]]:
    """Signals past their road's end that the reader cannot place (H14): ``(road, signal, new s, or None to drop it)``."""
    found = []
    for road in opendrive.roads:
        length = float(road.plan_view.length)
        for signal in road.signals:
            placement = conversion_heuristics.road_end_placement(s=float(signal.s), road_length=length, config=heuristics)
            if placement is None:
                continue
            outcome = 'placed at the road end' if placement == 'placed_at_road_end' else 'dropped'
            conversion_heuristics.record(
                heuristic_id='H14', element_id=signal.id, action=placement,
                message=(
                    f'signal {signal.id}: s={float(signal.s):.2f} lies {float(signal.s) - length:.2f} m past the end '
                    f'of road {road.id} ({length:.2f} m) - {outcome}'
                ),
                config=heuristics,
                location=traffic_controls.element_position(road=road, s=length, t=float(signal.t), frame=frame),
            )
            found.append((road, signal, length if placement == 'placed_at_road_end' else None))
    return found


def parse_xodr(
    *,
    opendrive_path: str,
    allow_zero_origin: bool = False,
    heuristics: Optional[conversion_heuristics.HeuristicsConfig] = None,
    dictionaries: list[vocabulary.Vocabulary],
    speed_unit: str = "m/s",
) -> ParsedXodr:
    """Parse the OpenDRIVE file, build the Lanelet2 projector from its geo-reference, and apply the signal heuristics.

    Raises:
        InputError: the map cannot be placed (see ``allow_zero_origin``), or a signal lies past its
            road's end and H14 is not applied.
    """
    heuristics = heuristics or conversion_heuristics.HeuristicsConfig()
    logger.info(f'Using {opendrive_path} as input')
    location = map_name(xodr_path=opendrive_path)
    words = vocabulary.combine(vocabularies=dictionaries)
    vocabulary.install(vocabulary=words, speed_unit=speed_unit)
    opendrive = parse_opendrive(Path(opendrive_path))

    # Construct Lanelet2 projector
    if opendrive.header.offset is not None:
        geo_offset = GeoOffset.from_dict(opendrive.header.offset)
        if geo_offset.hdg != 0.0 or geo_offset.z != 0.0:
            # The standard does not pin the rotation's direction, and no sample exists to check it on.
            logger.warning(
                f'{os.path.abspath(opendrive_path)} header <offset> has hdg={geo_offset.hdg} rad and '
                f'z={geo_offset.z} m; both are ignored, only the x/y shift is applied'
            )
    else:
        geo_offset = None
    geo_reference = parse_geo_reference(
        text=opendrive.header.geo_reference, centre=projected_centre(opendrive=opendrive, geo_offset=geo_offset)
    )
    geo_reference_status = 'declared'
    if geo_reference is None and not (opendrive.header.geo_reference or '').strip():
        # Nothing to misplace: the coordinates are local, so (0, 0) is the honest origin.
        logger.warning(f'{os.path.abspath(opendrive_path)} has no <geoReference> - converting at latitude/longitude (0, 0)')
        geo_reference, geo_reference_status = GeoReference(0.0, 0.0), 'absent'
    elif geo_reference is None:
        problem = f'an unusable <geoReference> ({opendrive.header.geo_reference!r})'
        if not allow_zero_origin:
            raise InputError(
                f'{os.path.abspath(opendrive_path)} has {problem}, so the map cannot be placed. '
                f'Set allow_zero_origin=true to convert it at latitude/longitude (0, 0).'
            )
        logger.warning(f'{opendrive_path} has {problem} - converting at (0, 0) because allow_zero_origin is set')
        geo_reference, geo_reference_status = GeoReference(0.0, 0.0), 'unusable_zero_origin_allowed'
    elif geo_reference.origin == (0.0, 0.0) and geo_reference.translation == (0.0, 0.0):
        logger.warning(f'{opendrive_path} declares its origin at latitude/longitude (0, 0)')
    check_geo_reference(geo_reference=geo_reference, source=os.path.abspath(opendrive_path))
    projector = lanelet2.projection.UtmProjector(geo_reference.lanelet2_origin)
    frame = CustomTransfomer(projector, geo_offset, geo_reference.translation)
    outside = unprojectable_road_start(opendrive=opendrive, frame=frame)
    if outside is not None:
        problem = (
            f'a <geoReference> ({(opendrive.header.geo_reference or "").strip()!r}) and header <offset> '
            f'({opendrive.header.offset}) that put road geometry outside the projection ({outside})'
        )
        if not allow_zero_origin:
            raise InputError(
                f'{os.path.abspath(opendrive_path)} has {problem}, so the map cannot be placed. '
                f'Set allow_zero_origin=true to convert it at latitude/longitude (0, 0) in file coordinates.'
            )
        logger.warning(f'{opendrive_path} has {problem} - converting at (0, 0) in file coordinates because allow_zero_origin is set')
        geo_reference, geo_reference_status = GeoReference(0.0, 0.0), 'unusable_zero_origin_allowed'
        projector = lanelet2.projection.UtmProjector(geo_reference.lanelet2_origin)
        frame = CustomTransfomer(projector)

    for signal, changes in signal_corrections(opendrive=opendrive, words=words, frame=frame):
        for attribute, value in changes.items():
            setattr(signal, attribute, value)
    past_end = signals_past_road_end(opendrive=opendrive, heuristics=heuristics, frame=frame)
    if past_end and not conversion_heuristics.applies(heuristic_id='H14', config=heuristics):
        listed = ', '.join(f'signal {signal.id} on road {road.id}' for road, signal, _ in past_end)
        raise InputError(
            f'{os.path.abspath(opendrive_path)}: {listed} lie past the end of their road, where they cannot be '
            f'placed. Correct their s, or convert with autofix.enabled=true and '
            f'autofix.heuristics.signal_past_road_end.enabled=true.'
        )
    dropped_signals = []
    for road, signal, s in past_end:
        if s is None:
            dropped_signals.append(
                map_quality.fault(
                    element=f'signal {signal.id}',
                    location=_signal_position(road=road, signal=signal, frame=frame),
                    detail=f'past the end of road {road.id}',
                )
            )
            road.signals.remove(signal)
        else:
            signal.s = s
    return ParsedXodr(
        words=words,
        dropped_signals=dropped_signals,
        location=location,
        opendrive=opendrive,
        geo_reference=geo_reference,
        projector=projector,
        frame=frame,
        traffic_side=traffic_side_of(opendrive=opendrive),
        geo_reference_status=geo_reference_status,
    )


def unprojectable_road_start(*, opendrive, frame: CustomTransfomer) -> Optional[str]:
    """Why the first road start point ``frame`` cannot turn into latitude/longitude fails, or None."""
    for road in opendrive.roads:
        for geometry in road.plan_view._geometries:
            try:
                frame.transform(*geometry.start_position)
            except RuntimeError as exc:
                return f'road {road.id}: {exc}'
    return None


def traffic_side_of(*, opendrive) -> Optional[str]:
    """``RHT`` or ``LHT`` if every road has that rule, ``mixed`` if they differ, ``None`` without roads."""
    sides = {'RHT' if road.driving_direction else 'LHT' for road in opendrive.roads}
    return None if not sides else sides.pop() if len(sides) == 1 else 'mixed'


def export_osm(*, cfg: MapConversionConfig, parsed: ParsedXodr) -> ExportedOsm:
    """Convert OpenDRIVE to CommonRoad, then write the Lanelet2 ``.osm`` within the size budget."""
    heuristics = conversion_heuristics.applied_heuristics(autofix=cfg.autofix)
    tolerance = conversion_heuristics.shared_node_tolerance(config=heuristics)
    # Convert OpenDRIVE to CommonRoad
    open_drive_config = OpenDriveConfig()
    open_drive_config.min_delta_s = 1.5
    open_drive_config.filter_types = list(CONVERTED_LANE_TYPES)
    road_network = Network()
    road_network.load_opendrive(parsed.opendrive)
    # for index in range(len(road_network._traffic_lights)):
    #     road_network._traffic_lights[index]._traffic_light_id = abs(
    #         road_network._traffic_lights[index].traffic_light_id
    #     )
    # We leave everything in the original OpenDRIVE inertial coordinate frame
    lanelet_network = road_network.export_lanelet_network(transformer=None, filter_types=open_drive_config.filter_types)
    fixes = bidirectional_fixes(lanelet_network=lanelet_network)
    for pred_id, succ_id in fixes.links:
        pred, succ = lanelet_network.find_lanelet_by_id(pred_id), lanelet_network.find_lanelet_by_id(succ_id)
        pred.successor = [i for i in pred.successor if i != succ_id]
        succ.predecessor = [i for i in succ.predecessor if i != pred_id]
    for lanelet_id in fixes.reverse_connectors:
        lanelet_network.remove_lanelet(lanelet_id)
    if fixes.links or fixes.reverse_connectors:
        logger.info(
            f'Bidirectional lanelets: unlinked {len(fixes.links)} wrong links, '
            f'removed {len(fixes.reverse_connectors)} reverse connectors'
        )
    # A bound that loops back makes Lanelet2's loader turn it around (lanelet_bounds module docstring).
    for lanelet_id, (left, right) in lanelet_bounds.loop_free_bounds(
        network=lanelet_network, to_local=parsed.frame.to_local
    ).items():
        lanelet_bounds.set_bounds(lanelet=lanelet_network.find_lanelet_by_id(lanelet_id), left=left, right=right)
    lanelet_bounds.record_sharp_kinks(opendrive=parsed.opendrive, to_local=parsed.frame.to_local)

    # Export CommonRoad to Lanelet2, iterating tolerance until the OSM fits the size budget.
    commonroad_config = GeneralConfig()
    commonroad_config.proj_string_cr = parsed.geo_reference.proj_string  # not currently used - see CustomTransformer

    tol = cfg.simplify_tolerance or 0.0
    unsimplified_network = None  # the verifier skips trimmed maps: the other checks judge the trimmed outputs
    if cfg.verifier_timeout_s > 0 and cfg.trim_radius is None:
        unsimplified_network = copy.deepcopy(lanelet_network) if tol > 0 else lanelet_network
    size_budget_active = tol > 0 and cfg.osm_target_bytes is not None
    max_attempts = max(1, cfg.simplify_max_iterations) if size_budget_active else 1
    osm_bytes = None

    for attempt in range(max_attempts):
        if tol > 0:
            removed = lanelet_bounds.simplify_lanelet_network(network=lanelet_network, tolerance=tol)
            logger.info(
                f'Simplify attempt {attempt + 1}/{max_attempts}: tol={tol:.3f} m '
                f'(removed {removed} vertices across {len(lanelet_network.lanelets)} lanelets)'
            )
        # After simplification, which moves the vertex Lanelet2's loader judges a bound's direction by.
        for lanelet_id, (left, right) in lanelet_bounds.align_safe_bounds(
            network=lanelet_network, to_local=parsed.frame.to_local
        ).items():
            lanelet_bounds.set_bounds(lanelet=lanelet_network.find_lanelet_by_id(lanelet_id), left=left, right=right)
        l2osm = CR2LaneletConverter(
            config=lanelet2_config,
            cr_config=commonroad_config,
            shared_node_tolerance=tolerance,
        )
        osm = l2osm.convert_lanelet_network(lanelet_network, transformer=parsed.frame)
        if next(osm.iter('node'), None) is None:
            raise InputError(f'No lanelets of the converted lane types ({", ".join(open_drive_config.filter_types)}) - the map is empty')
        osm_bytes = etree.tostring(osm, xml_declaration=True, encoding="UTF-8", pretty_print=True)
        size_mb = len(osm_bytes) / (1024 * 1024)
        if not size_budget_active:
            logger.info(f'OSM size: {size_mb:.2f} MB (no size budget enforced)')
            break
        target_mb = cfg.osm_target_bytes / (1024 * 1024)
        logger.info(f'  OSM size: {size_mb:.2f} MB (budget {target_mb:.1f} MB)')
        if len(osm_bytes) <= cfg.osm_target_bytes:
            break
        tol = max(tol, 0.1) * cfg.simplify_tolerance_factor
    else:
        raise RuntimeError(
            f'OSM size {len(osm_bytes) / (1024 * 1024):.2f} MB exceeds budget '
            f'{cfg.osm_target_bytes / (1024 * 1024):.1f} MB after {max_attempts} simplification attempts. '
            f'Final tolerance was {tol / cfg.simplify_tolerance_factor:.2f} m. '
            f'Raise simplify_max_iterations, raise simplify_tolerance, or relax osm_target_bytes.'
        )

    gaps = linked_end_gaps(lanelet_network=lanelet_network, frame=parsed.frame)
    record_seam_gaps(gaps=gaps, heuristics=heuristics)
    linked_gaps = [
        map_quality.fault(element=f'lanelets {pred} -> {succ}', location=middle, detail=f'{gap:.2f} m apart')
        for pred, succ, gap, middle in gaps
        if gap > tolerance
    ]
    osm_path = os.path.join(cfg.dir_path, f"{parsed.location}.osm")
    logger.info(f'Writing converted Lanelet2 map to {osm_path}')
    with open(osm_path, "wb") as file_out:
        file_out.write(osm_bytes)
    return ExportedOsm(
        road_network=road_network,
        lanelet_network=lanelet_network,
        osm_path=osm_path,
        linked_gaps=linked_gaps,
        unsimplified_network=unsimplified_network,
    )


def linked_end_gaps(
    *, lanelet_network: LaneletNetwork, frame: CustomTransfomer
) -> list[tuple[int, int, float, tuple[float, float]]]:
    """Per predecessor/successor pair: their ids, the larger of the left and right boundary gaps (m), and the local
    middle of the seam."""
    gaps = []
    for lanelet in lanelet_network.lanelets:
        for successor_id in lanelet.successor:
            successor = lanelet_network.find_lanelet_by_id(successor_id)
            if successor is None:
                continue
            gap = max(
                float(np.linalg.norm(lanelet.left_vertices[-1][:2] - successor.left_vertices[0][:2])),
                float(np.linalg.norm(lanelet.right_vertices[-1][:2] - successor.right_vertices[0][:2])),
            )
            middle = (lanelet.center_vertices[-1][:2] + successor.center_vertices[0][:2]) / 2
            x, y = frame.to_local(middle[0], middle[1])
            gaps.append((lanelet.lanelet_id, successor_id, gap, (float(x), float(y))))
    return gaps


def record_seam_gaps(
    *, gaps: list[tuple[int, int, float, tuple[float, float]]], heuristics: conversion_heuristics.HeuristicsConfig
) -> None:
    """Record H10 for every pair of :func:`linked_end_gaps` whose ends lie a little apart."""
    for pred, succ, gap, middle in gaps:
        if conversion_heuristics.joins_seam(gap_m=gap, config=heuristics):
            conversion_heuristics.record(
                heuristic_id='H10', element_id=f'{pred}->{succ}', action='seam_shared',
                message=f'lanelets {pred} -> {succ}: linked ends {gap:.2f} m apart - one seam',
                config=heuristics, location=middle,
            )


def trim(*, cfg: MapConversionConfig, osm_path: str, projector) -> None:
    """Rewrite the ``.osm`` trimmed to ``cfg.trim_radius`` around ``cfg.center``, if requested."""
    # Trim the map if requested
    if cfg.trim_radius is not None:
        lanelet_map = lanelet2.io.load(osm_path, projector)
        trimmed_map = trim_map(lanelet_map, center=cfg.center, radius=cfg.trim_radius,
                               vertical_limits=cfg.trim_vertical_limits)
        lanelet2.io.write(osm_path, trimmed_map, projector)


def _within_trim(*, cfg: MapConversionConfig, x: float, y: float, z: Optional[float] = None) -> bool:
    """Whether a local point survives ``trim_radius`` around ``center`` and, given ``z``, ``trim_vertical_limits``."""
    if cfg.center is not None and cfg.trim_radius is not None:
        if (x - cfg.center[0]) ** 2 + (y - cfg.center[1]) ** 2 > cfg.trim_radius ** 2:
            return False
    return z is None or cfg.trim_vertical_limits is None or (
        cfg.trim_vertical_limits[0] <= z <= cfg.trim_vertical_limits[1]
    )


@dataclasses.dataclass(slots=True)
class ExtractedStoplines:
    """The written stoplines file, the coverage of lights and stop/yield signs it gives, each signal's
    distance to its nearest row, and the painted stop lines listed as not converted."""

    path: str
    controls: map_validation.ControlCoverage
    distances: list[dict]
    painted_stop_lines: list[dict]


def extract_stoplines(
    *,
    cfg: MapConversionConfig,
    location: str,
    opendrive,
    lanelet_network: LaneletNetwork,
    road_network: Network,
    frame: CustomTransfomer,
    words: vocabulary.Vocabulary,
) -> ExtractedStoplines:
    """Write traffic-light, stop-sign and yield-sign stoplines to ``<map>_stoplines.json``.

    Rows and coverage come from :func:`traffic_controls.control_stoplines`; painted stop lines are
    listed in the conversion report (:func:`traffic_controls.report_painted_stop_lines`).
    """
    heuristics = conversion_heuristics.applied_heuristics(autofix=cfg.autofix)
    built = traffic_controls.control_stoplines(
        opendrive=opendrive,
        road_network=road_network,
        lanelet_network=lanelet_network,
        frame=frame,
        within=lambda *, x, y, z=None: _within_trim(cfg=cfg, x=x, y=y, z=z),
        drivable_types=frozenset(CONVERTED_LANE_TYPES),
        config=heuristics,
        words=words,
    )
    painted = traffic_controls.report_painted_stop_lines(opendrive=opendrive, frame=frame, words=words)
    stoplines_path = os.path.join(cfg.dir_path, f"{location}_stoplines.json")
    with open(stoplines_path, 'w') as f:
        logger.info(f'Writing extracted stoplines to {stoplines_path}')
        json.dump(built.rows, f, indent=4)
    return ExtractedStoplines(
        path=stoplines_path, controls=built.coverage, distances=built.distances, painted_stop_lines=painted
    )


def build_mesh(*, cfg: MapConversionConfig, location: str, osm_path: str, projector) -> BuiltMesh:
    """Build the road and lane-marking mesh from the written ``.osm`` and save ``<map>_mesh.json``."""
    # Export road mesh
    mesh_path = os.path.join(cfg.dir_path, f"{location}_mesh.json")
    lanelet_map = lanelet2.io.load(osm_path, projector)
    logger.info(f'Computing road mesh')
    road_mesh = lanelet_bounds.road_mesh(lanelet_map=lanelet_map)
    road_mesh = BirdviewMesh.set_properties(road_mesh, category='road').to(road_mesh.device)
    lane_mesh = lanelet_map_to_lane_mesh(
        lanelet_map, left_handed=False,
        left_right_marking_join_threshold=cfg.lane_marking_join_threshold
    )
    combined_mesh = lane_mesh.merge(road_mesh)
    logger.info(f'Writing road mesh to {mesh_path}')
    combined_mesh.save(mesh_path)
    return BuiltMesh(combined_mesh=combined_mesh, mesh_path=mesh_path)


def render_and_finish(
    *,
    cfg: MapConversionConfig,
    location: str,
    geo_reference: GeoReference,
    osm_path: str,
    stoplines_path: str,
    mesh: BuiltMesh,
    opendrive_path: str,
    traffic_side: Optional[str],
    geo_reference_status: str,
    words: vocabulary.Vocabulary,
    opendrive_version: Optional[str],
) -> None:
    """Pick the map centre, write ``metadata.json``, render ``visualization.png``, record provenance."""
    if cfg.center is None:
        center = mesh.combined_mesh.center.squeeze(0).numpy().tolist()
    else:
        center = list(cfg.center)

    # Write metadata
    map_cfg = MapConfig(
        name=location, center=center, lanelet_map_origin=geo_reference.origin,
        iai_location_name=f'{cfg.domain}:{location}' if cfg.domain else None,
        left_handed_coordinates=False,
        lanelet_path=os.path.abspath(osm_path),
        mesh_path=mesh.mesh_path,
        stoplines_path=stoplines_path,
    )
    metadata_path = os.path.join(cfg.dir_path, 'metadata.json')
    logger.info(f'Writing metadata to {metadata_path}')
    store_map_config(map_cfg, metadata_path)

    # Visualize results
    fov = cfg.visualization_fov
    if fov is None:
        fov = fitting_visualization_fov(verts=mesh.combined_mesh.verts, center=center)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = Resolution(2048, 2048)
    map_cfg = resolve_paths_to_absolute(map_cfg, root=cfg.dir_path)
    driving_surface_mesh = map_cfg.road_mesh.to(device)
    renderer_cfg = RendererConfig(left_handed_coordinates=map_cfg.left_handed_coordinates)
    renderer = renderer_from_config(renderer_cfg)
    traffic_controls = traffic_controls_from_map_config(map_cfg)
    mesh_generator = BirdviewRGBMeshGenerator(
        driving_surface_mesh,
        renderer.color_map,
        renderer.rendering_levels,
        traffic_controls=traffic_controls,
    ).to(device)
    camera_xy = torch.tensor(center, dtype=driving_surface_mesh.verts.dtype, device=device).reshape(1, 1, 2)
    # Zero rotation (sin=0, cos=1) -> top-down, north-up.
    camera_sc = torch.tensor([[[0.0, 1.0]]], dtype=driving_surface_mesh.verts.dtype, device=device)
    map_image = renderer.render_frame(
        mesh_generator.generate(num_cameras=1),
        camera_xy=camera_xy,
        camera_sc=camera_sc,
        res=res,
        fov=fov,
    )
    viz_path = os.path.join(cfg.dir_path, 'visualization.png')
    logger.info(f'Saving visualization to {viz_path}')
    imageio.imsave(viz_path, map_image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    write_provenance(
        cfg=cfg,
        xodr_path=opendrive_path,
        out_dir=cfg.dir_path,
        visualization_fov=fov,
        traffic_side=traffic_side,
        geo_reference_status=geo_reference_status,
        words=words,
        opendrive_version=opendrive_version,
    )


def fitting_visualization_fov(*, verts: torch.Tensor, center: list[float]) -> float:
    """Width (m) of the square around ``center`` that holds every mesh vertex, plus 5%."""
    offsets = (verts.reshape(-1, 2) - torch.tensor(center, dtype=verts.dtype)).abs()
    return float(2.0 * offsets.max() * FOV_MARGIN) if offsets.numel() else DEFAULT_VISUALIZATION_FOV


def _vocabularies(*, cfg: MapConversionConfig, xodr_path: str) -> list[vocabulary.Vocabulary]:
    """The base dictionary, then every vendor dictionary matching the file's header."""
    vendors = vocabulary.load_vendors(folder=cfg.vocabulary_vendors)
    header = vocabulary.read_header(xodr_path=xodr_path)
    return [vocabulary.load(path=cfg.vocabulary), *vocabulary.matching(vendors=vendors, header=header)]


def _opendrive_version(*, header) -> Optional[str]:
    """``revMajor.revMinor`` from the parsed file header, or None when the header gives neither."""
    if header is None or (header.revMajor is None and header.revMinor is None):
        return None
    return f"{header.revMajor}.{header.revMinor}"


def write_provenance(
    *,
    cfg: MapConversionConfig,
    xodr_path: str,
    out_dir: str,
    visualization_fov: float,
    traffic_side: Optional[str],
    geo_reference_status: str,
    words: vocabulary.Vocabulary,
    opendrive_version: Optional[str],
) -> str:
    """Write ``<map>_provenance.json``: what was converted, with which converter and settings.

    Also records values the conversion derived: the visualization FOV it rendered with, the
    traffic side (``RHT``/``LHT``/``mixed``) of the input's roads and whether the geo-reference was
    ``declared``, ``absent`` (placed at (0, 0)) or ``unusable_zero_origin_allowed``.

    Returns:
        Absolute path of the written file.
    """
    versions = {}
    for package in PROVENANCE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    provenance = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input": {"path": os.path.abspath(xodr_path), "sha256": sha256_file(path=xodr_path)},
        "config": OmegaConf.to_container(OmegaConf.structured(cfg)),
        "vocabulary": {"path": words.path, "sha256": words.sha256, "vendors": [dict(v) for v in words.vendors]},
        "opendrive_version": opendrive_version,
        "visualization_fov": visualization_fov,
        "traffic_side": traffic_side,
        "geo_reference": geo_reference_status,
        "converter": git_identity(repo=REPO_ROOT),
        "python": sys.version.split()[0],
        "packages": versions,
    }
    location = map_name(xodr_path=xodr_path)
    path = os.path.abspath(os.path.join(out_dir, f"{location}_provenance.json"))
    with open(path, "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info(f'Writing provenance to {path}')
    return path


def convert_map(cfg: MapConversionConfig) -> str:
    """Convert ``cfg.xodr_path``, or the single ``.xodr`` in ``cfg.dir_path``, into the TorchDriveSim artifacts.

    Writes ``<map>.osm``, ``<map>_stoplines.json``, ``<map>_mesh.json``, ``metadata.json``,
    ``visualization.png``, ``<map>_signal_groups.json`` and ``<map>_provenance.json`` into
    ``cfg.dir_path``, or for ``xodr_path`` into ``<xodr folder>/<map name>/``. Once the input is
    known, ``<map>_conversion_report.json`` (:mod:`conversion_report`) is written whether or not the
    conversion succeeds; after a successful one it holds the :mod:`map_validation` verdict.

    Args:
        cfg: The run's settings, as a dataclass or the command line's ``DictConfig``.

    Returns:
        Absolute path of the written ``.osm``.
    """
    if not dataclasses.is_dataclass(cfg):
        cfg = OmegaConf.to_object(cfg)
    opendrive_path = validate_config(cfg=cfg)
    location = map_name(xodr_path=opendrive_path)
    if cfg.xodr_path is not None:
        out_dir = os.path.join(os.path.dirname(opendrive_path), location)
        os.makedirs(out_dir, exist_ok=True)
        cfg = dataclasses.replace(cfg, dir_path=out_dir)
    report = conversion_report.attach()
    error, verdict = None, None
    report_path = os.path.join(cfg.dir_path, f"{location}_conversion_report.json")
    try:
        osm_path, verdict, frame = _convert(cfg=cfg, opendrive_path=opendrive_path)
        if verdict is not None and cfg.report.enabled:
            json_path, html_path = autocritic_report.write(
                map_dir=cfg.dir_path,
                location=location,
                records=report.records,
                verdict=dataclasses.asdict(verdict),
                autofix_enabled=cfg.autofix.enabled,
                would_have_fixed=cfg.autocritic.would_have_fixed,
                config=cfg.report,
                xodr_path=opendrive_path,
                to_local=frame.to_local,
            )
            logger.info(f'Wrote AutoCritic report to {html_path} and {json_path}')
        if verdict is not None and verdict.status == 'fail':
            withhold_outputs(map_dir=cfg.dir_path, location=location)
            failed = [check.name for check in verdict.checks if not check.passed]
            raise MapRejected(
                f'AutoCritic ({verdict.profile}) rejected the map: {", ".join(failed)} - map outputs withheld; '
                f'see {os.path.abspath(report_path)}'
            )
        return osm_path
    except BaseException as exc:
        error = exc
        raise
    finally:
        conversion_report.detach(handler=report)
        path = report.write(
            path=report_path,
            xodr_path=opendrive_path,
            error=error,
            validation=None if verdict is None else dataclasses.asdict(verdict),
        )
        logger.info(f'Wrote conversion report ({len(report.records)} degradation records) to {path}')


def map_output_paths(*, map_dir: str, location: str) -> list[str]:
    """The map outputs a conversion writes into ``map_dir`` (not its reports)."""
    names = (
        f'{location}.osm',
        f'{location}_stoplines.json',
        f'{location}_mesh.json',
        f'{location}_signal_groups.json',
        'metadata.json',
        'visualization.png',
    )
    return [os.path.abspath(os.path.join(map_dir, name)) for name in names]


def withhold_outputs(*, map_dir: str, location: str) -> list[str]:
    """Delete the map outputs of a rejected conversion; return those deleted."""
    deleted = [path for path in map_output_paths(map_dir=map_dir, location=location) if os.path.exists(path)]
    for path in deleted:
        os.remove(path)
    return deleted


def _convert(
    *, cfg: MapConversionConfig, opendrive_path: str
) -> tuple[str, Optional[map_validation.Verdict], CustomTransfomer]:
    """The pipeline stages of :func:`convert_map`, on a validated config: the ``.osm`` path, the verdict (``None``
    with AutoCritic off) and the frame the outputs are in."""
    parsed = parse_xodr(
        opendrive_path=opendrive_path,
        allow_zero_origin=cfg.allow_zero_origin,
        heuristics=conversion_heuristics.applied_heuristics(autofix=cfg.autofix),
        dictionaries=_vocabularies(cfg=cfg, xodr_path=opendrive_path),
        speed_unit=str(cfg.speed_unit),
    )
    exported = export_osm(cfg=cfg, parsed=parsed)
    trim(cfg=cfg, osm_path=exported.osm_path, projector=parsed.projector)
    stoplines = extract_stoplines(
        cfg=cfg,
        location=parsed.location,
        opendrive=parsed.opendrive,
        lanelet_network=exported.lanelet_network,
        road_network=exported.road_network,
        frame=parsed.frame,
        words=parsed.words,
    )
    write_signal_groups(
        xodr_path=opendrive_path, out_path=os.path.join(cfg.dir_path, f"{parsed.location}_signal_groups.json")
    )
    mesh = build_mesh(cfg=cfg, location=parsed.location, osm_path=exported.osm_path, projector=parsed.projector)
    render_and_finish(
        cfg=cfg,
        location=parsed.location,
        geo_reference=parsed.geo_reference,
        osm_path=exported.osm_path,
        stoplines_path=stoplines.path,
        mesh=mesh,
        opendrive_path=opendrive_path,
        traffic_side=parsed.traffic_side,
        geo_reference_status=parsed.geo_reference_status,
        words=parsed.words,
        opendrive_version=_opendrive_version(header=parsed.opendrive.header),
    )
    if not cfg.autocritic.enabled:
        return os.path.abspath(exported.osm_path), None, parsed.frame
    verdict = map_validation.validate(
        map_dir=cfg.dir_path,
        lanelet_network=exported.unsimplified_network,
        controls=stoplines.controls,
        verifier_timeout_s=cfg.verifier_timeout_s,
        verifier_formulas=cfg.verifier_formulas,
        profile=cfg.autocritic.preset,
        thresholds=cfg.autocritic.thresholds,
        findings=map_validation.ConversionFindings(
            linked_gaps=exported.linked_gaps,
            dropped_signals=parsed.dropped_signals,
            signal_distances=stoplines.distances,
            painted_stop_lines=stoplines.painted_stop_lines,
            static_light_codes=static_light_codes(opendrive=parsed.opendrive, words=parsed.words, frame=parsed.frame),
        ),
    )
    logger.info(f'Validation: {verdict.status}')
    return os.path.abspath(exported.osm_path), verdict, parsed.frame


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    console = logging.StreamHandler()
    console.addFilter(conversion_report.FirstOccurrenceFilter())
    logging.getLogger(conversion_report.DEGRADATION_LOGGER).addHandler(console)
    logging.getLogger(map_validation.__name__).addHandler(logging.StreamHandler())
    logging.getLogger(map_validation.__name__).setLevel(logging.INFO)
    try:
        convert_map(load_config(argv=sys.argv[1:]))
    except ConfigError as exc:
        logger.error(f'Invalid configuration: {exc}')
        sys.exit(REFUSED_EXIT_CODE)
    except InputError as exc:
        logger.error(f'Cannot convert: {exc}')
        sys.exit(REFUSED_EXIT_CODE)
    except MapRejected as exc:
        logger.error(str(exc))
        sys.exit(REJECTED_EXIT_CODE)
