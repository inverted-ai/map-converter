"""
The required input is a folder `dir_path` with a single *.xodr file in it.
The name of that file will be the map name.
This script produces the required files for TorchDriveSim in the `dir_path` folder.
"""
import dataclasses
import datetime
import glob
import importlib.metadata
import json
import os
import re
import sys
import logging
from typing import Tuple, Optional

import imageio
import math
import numpy as np
from pathlib import Path

import torch.cuda
from lxml import etree
import pyproj
from omegaconf import OmegaConf
from shapely.geometry import LineString

import lanelet2
from torchdrivesim.map import Stopline, MapConfig, store_map_config, resolve_paths_to_absolute
from torchdrivesim.lanelet2 import LaneletMap, load_lanelet_map, road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh
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

from conversion_common import REFUSED_EXIT_CODE, git_identity, sha256_file
from opendrive_signal_groups import write_signal_groups


logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VISUALIZATION_FOV = 800.0
FOV_MARGIN = 1.05
PROVENANCE_PACKAGES = (
    'torchdrivesim', 'torch', 'lanelet2', 'commonroad-io', 'numpy', 'lxml', 'pyproj', 'shapely', 'omegaconf',
)


@dataclasses.dataclass
class MapConversionConfig:
    """Options for one ``convert_map.py`` run, given on the command line as ``field=value``.

    Attributes:
        dir_path: Directory holding exactly one ``.xodr``; all outputs are written here.
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
            and the run fails if it never does. None disables the budget.
        simplify_max_iterations: Attempts at fitting the size budget.
        simplify_tolerance_factor: Factor the tolerance grows by per attempt.
        allow_zero_origin: Convert a map whose ``<geoReference>`` is present but unusable at
            latitude/longitude (0, 0) instead of failing. A map with no ``<geoReference>`` always
            converts at (0, 0), with a warning.
    """

    dir_path: str
    domain: Optional[str] = None
    lane_marking_join_threshold: float = 0.1
    visualization_fov: Optional[float] = None
    center: Optional[Tuple[float, float]] = None
    trim_radius: Optional[float] = None
    trim_vertical_limits: Optional[Tuple[float, float]] = None
    # Raise (e.g. 0.5) for very large maps where file size matters more than curve fidelity.
    simplify_tolerance: Optional[float] = 0.2
    osm_target_bytes: Optional[int] = 1000 * 1024 * 1024 #10 * 1024 * 1024
    simplify_max_iterations: int = 1
    simplify_tolerance_factor: float = 2.0
    allow_zero_origin: bool = False


@dataclasses.dataclass
class GeoOffset:
    """The OpenDRIVE header ``<offset>``: a translation (m) then a rotation (degrees as stored)."""

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
        """Translate ``(x, y)`` by the offset, then rotate it by ``hdg``."""
        x_translated = x + float(self.x)
        y_translated = y + float(self.y)
        hdg_rad = math.radians(float(self.hdg))
        x_rotated = x_translated * math.cos(hdg_rad) - y_translated * math.sin(hdg_rad)
        y_rotated = x_translated * math.sin(hdg_rad) + y_translated * math.cos(hdg_rad)
        return x_rotated, y_rotated

@dataclasses.dataclass
class GeoReference:
    """Latitude/longitude of the map's local origin.

    ``translation`` is subtracted from projected (geo-referenced) coordinates to get local
    coordinates: the false easting/northing for ``tmerc``, the map centre for ``utm``.
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
        ``tmerc``: origin ``(lat_0, lon_0)``, translation ``(x_0, y_0)``. ``utm``: origin at the map
        centre, translation = the centre. ``None`` for a missing, unparseable or other projection.
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
    if params.get("proj") == "tmerc":
        return GeoReference(
            lat=float(params.get("lat_0", 0.0)),
            lon=float(params.get("lon_0", 0.0)),
            translation=(float(params.get("x_0", 0.0)), float(params.get("y_0", 0.0))),
        )
    if params.get("proj") == "utm" and centre is not None:
        lon, lat = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(*centre)
        return GeoReference(lat=lat, lon=lon, translation=(float(centre[0]), float(centre[1])), zone=int(params["zone"]) if "zone" in params else None)
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


def simplify_lanelet_network(*, network: LaneletNetwork, tolerance: float) -> int:
    """Apply 2D Douglas-Peucker to each lanelet's left/right boundaries; rebuild centerlines.

    Mutates ``network`` in place. Endpoints are preserved by shapely's simplify,
    so lanelet-to-lanelet connectivity (predecessor/successor handshakes) is
    unaffected. Because RDP is deterministic, two adjacent lanelets whose shared
    boundary arrays were identical pre-simplification produce identical
    post-simplification arrays, so cr2lanelet's 0.001 m shared-way dedup still
    works.

    Args:
        network: The CommonRoad lanelet network whose lanelets will be simplified.
        tolerance: Max permissible lateral deviation per vertex, in metres.

    Returns:
        The total number of vertices removed across all lanelets (for logging).
    """
    removed = 0
    # First pass: normalize every lanelet to 3D so adjacent boundaries always
    # share a trailing dimension. cr2lanelet's `_vertices_are_equal` does a raw
    # numpy subtract with no trailing-axis guard; if an adj-pair ends up mixed
    # 2D/3D it crashes the OSM export.
    for lanelet in network.lanelets:
        lanelet.left_vertices = _ensure_3d(polyline=lanelet.left_vertices)
        lanelet.right_vertices = _ensure_3d(polyline=lanelet.right_vertices)
        lanelet.center_vertices = _ensure_3d(polyline=lanelet.center_vertices)
        _refresh_lanelet_caches(lanelet=lanelet)
    for lanelet in network.lanelets:
        left = lanelet.left_vertices
        right = lanelet.right_vertices
        if left.shape[0] < 3 and right.shape[0] < 3:
            continue
        new_left = _simplify_polyline(polyline=left, tolerance=tolerance)
        new_right = _simplify_polyline(polyline=right, tolerance=tolerance)
        # Centerline must be re-derived from the simplified boundaries; if left and
        # right end up with different vertex counts, resample both to the longer one
        # on its own arc-length axis before averaging.
        new_center = _midline(left=new_left, right=new_right)
        removed += (left.shape[0] - new_left.shape[0]) + (right.shape[0] - new_right.shape[0])
        lanelet.left_vertices = new_left
        lanelet.right_vertices = new_right
        lanelet.center_vertices = new_center
        # commonroad-io's vertex setters do not refresh the polygon / arc-length
        # caches built in Lanelet.__init__, so do it here.
        _refresh_lanelet_caches(lanelet=lanelet)
    return removed


def _refresh_lanelet_caches(*, lanelet) -> None:
    """Mirror the cache initialization done by Lanelet.__init__ after vertex mutation."""
    from commonroad.geometry.shape import Polygon as CRPolygon
    lanelet._polygon = CRPolygon(np.concatenate((lanelet.right_vertices, np.flip(lanelet.left_vertices, 0))))
    lanelet._distance = None
    lanelet._inner_distance = None


def _ensure_3d(*, polyline: np.ndarray) -> np.ndarray:
    """Pad a 2D polyline with a z=0 column; pass 3D through unchanged."""
    if polyline.shape[1] == 3:
        return polyline
    return np.column_stack([polyline, np.zeros(polyline.shape[0], dtype=polyline.dtype)])


def _simplify_polyline(*, polyline: np.ndarray, tolerance: float) -> np.ndarray:
    """Run Douglas-Peucker on the XY projection; preserve z by arc-length interpolation if present.

    The output preserves the input's trailing dimension (2 or 3), so adjacent
    lanelets that share a boundary continue to have matching vertex shapes
    (cr2lanelet's `_vertices_are_equal` requires this).
    """
    if polyline.shape[0] < 3:
        return polyline
    xy = polyline[:, :2]
    simplified_xy = np.asarray(LineString(xy).simplify(tolerance, preserve_topology=False).coords,
                               dtype=polyline.dtype)
    if polyline.shape[1] == 2:
        return simplified_xy
    # 3D path: interpolate z onto the surviving XY points using cumulative XY arc length.
    orig_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    new_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(simplified_xy, axis=0), axis=1))])
    if orig_s[-1] > 0 and new_s[-1] > 0:
        new_s = new_s * (orig_s[-1] / new_s[-1])
    z = np.interp(new_s, orig_s, polyline[:, 2])
    return np.column_stack([simplified_xy, z.astype(polyline.dtype)])


def _midline(*, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute the centerline of two boundary polylines.

    If the two polylines have the same vertex count after simplification, we
    average pointwise. Otherwise we resample both onto a common arc-length axis
    using the denser of the two as the reference, then average. This is the same
    convention used elsewhere in commonroad-io's lanelet construction.
    """
    if left.shape[0] == right.shape[0]:
        return (left + right) / 2.0
    n = max(left.shape[0], right.shape[0])
    return (_resample_polyline(polyline=left, n=n) + _resample_polyline(polyline=right, n=n)) / 2.0


def _resample_polyline(*, polyline: np.ndarray, n: int) -> np.ndarray:
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))])
    if cumulative[-1] == 0.0:
        return np.repeat(polyline[:1], n, axis=0)
    targets = np.linspace(0.0, cumulative[-1], n)
    out = np.column_stack([np.interp(targets, cumulative, polyline[:, i]) for i in range(polyline.shape[1])])
    return out.astype(polyline.dtype)


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

    @property
    def rotation(self) -> float:
        """Radians the header offset rotates local coordinates by."""
        return math.radians(float(self.offset.hdg)) if self.offset is not None else 0.0

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
    traffic_side: Optional[str]  # "RHT", "LHT" or "mixed"
    geo_reference_status: str  # "declared", "absent" or "unusable_zero_origin_allowed"


@dataclasses.dataclass(slots=True)
class ExportedOsm:
    """The CommonRoad network built from the input and the ``.osm`` written from it."""

    road_network: Network
    lanelet_network: LaneletNetwork
    osm_path: str


@dataclasses.dataclass(slots=True)
class BuiltMesh:
    """The road + lane-marking mesh and where it was written."""

    combined_mesh: BirdviewMesh
    mesh_path: str


class ConfigError(ValueError):
    """An invalid :class:`MapConversionConfig`, found before anything is read or written."""


def validate_config(*, cfg: MapConversionConfig) -> str:
    """Check the configuration before parsing and return the ``.xodr`` to convert.

    Raises:
        ConfigError: ``trim_radius`` without ``center``, or ``dir_path`` not holding exactly one ``.xodr``.
    """
    if cfg.trim_radius is not None and cfg.center is None:
        raise ConfigError('trim_radius requires center')
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
        return (east + west) / 2.0, (north + south) / 2.0
    starts = [g.start_position for road in opendrive.roads for g in road.plan_view._geometries]
    if not starts:
        return None
    xs, ys = zip(*(geo_offset.apply_offset(x, y) if geo_offset else (x, y) for x, y in starts))
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def parse_xodr(*, opendrive_path: str, allow_zero_origin: bool = False) -> ParsedXodr:
    """Parse the OpenDRIVE file and build the Lanelet2 projector from its geo-reference."""
    logger.info(f'Using {opendrive_path} as input')
    location = os.path.basename(opendrive_path)[:-5]
    opendrive = parse_opendrive(Path(opendrive_path))

    # Construct Lanelet2 projector
    if opendrive.header.offset is not None:
        geo_offset = GeoOffset.from_dict(opendrive.header.offset)
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
    return ParsedXodr(
        location=location,
        opendrive=opendrive,
        geo_reference=geo_reference,
        projector=projector,
        frame=CustomTransfomer(projector, geo_offset, geo_reference.translation),
        traffic_side=traffic_side_of(opendrive=opendrive),
        geo_reference_status=geo_reference_status,
    )


def traffic_side_of(*, opendrive) -> Optional[str]:
    """``RHT`` or ``LHT`` if every road has that rule, ``mixed`` if they differ, ``None`` without roads."""
    sides = {'RHT' if road.driving_direction else 'LHT' for road in opendrive.roads}
    return None if not sides else sides.pop() if len(sides) == 1 else 'mixed'


def export_osm(*, cfg: MapConversionConfig, parsed: ParsedXodr) -> ExportedOsm:
    """Convert OpenDRIVE to CommonRoad, then write the Lanelet2 ``.osm`` within the size budget."""
    # Convert OpenDRIVE to CommonRoad
    open_drive_config = OpenDriveConfig()
    open_drive_config.min_delta_s = 1.5
    open_drive_config.filter_types = [
        "driving",
        # "restricted",
        "onRamp",
        "offRamp",
        "exit",
        "entry",
        # "sidewalk",
        # "shoulder",
        # "crosswalk",
        # "bidirectional",
    ]
    road_network = Network()
    road_network.load_opendrive(parsed.opendrive)
    # for index in range(len(road_network._traffic_lights)):
    #     road_network._traffic_lights[index]._traffic_light_id = abs(
    #         road_network._traffic_lights[index].traffic_light_id
    #     )
    # We leave everything in the original OpenDRIVE inertial coordinate frame
    lanelet_network = road_network.export_lanelet_network(transformer=None, filter_types=open_drive_config.filter_types)

    # Export CommonRoad to Lanelet2, iterating tolerance until the OSM fits the size budget.
    commonroad_config = GeneralConfig()
    commonroad_config.proj_string_cr = parsed.geo_reference.proj_string  # not currently used - see CustomTransformer

    tol = cfg.simplify_tolerance or 0.0
    size_budget_active = tol > 0 and cfg.osm_target_bytes is not None
    max_attempts = max(1, cfg.simplify_max_iterations) if size_budget_active else 1
    osm_bytes = None

    for attempt in range(max_attempts):
        if tol > 0:
            removed = simplify_lanelet_network(network=lanelet_network, tolerance=tol)
            logger.info(
                f'Simplify attempt {attempt + 1}/{max_attempts}: tol={tol:.3f} m '
                f'(removed {removed} vertices across {len(lanelet_network.lanelets)} lanelets)'
            )
        l2osm = CR2LaneletConverter(config=lanelet2_config, cr_config=commonroad_config)
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

    osm_path = os.path.join(cfg.dir_path, f"{parsed.location}.osm")
    logger.info(f'Writing converted Lanelet2 map to {osm_path}')
    with open(osm_path, "wb") as file_out:
        file_out.write(osm_bytes)
    return ExportedOsm(road_network=road_network, lanelet_network=lanelet_network, osm_path=osm_path)


def trim(*, cfg: MapConversionConfig, osm_path: str, projector) -> None:
    """Rewrite the ``.osm`` trimmed to ``cfg.trim_radius`` around ``cfg.center``, if requested."""
    # Trim the map if requested
    if cfg.trim_radius is not None:
        lanelet_map = lanelet2.io.load(osm_path, projector)
        trimmed_map = trim_map(lanelet_map, center=cfg.center, radius=cfg.trim_radius,
                               vertical_limits=cfg.trim_vertical_limits)
        lanelet2.io.write(osm_path, trimmed_map, projector)


def extract_stoplines(
    *,
    cfg: MapConversionConfig,
    location: str,
    lanelet_network: LaneletNetwork,
    road_network: Network,
    frame: CustomTransfomer,
) -> str:
    """Write traffic-light, stop-sign and yield-sign stoplines to ``<map>_stoplines.json``.

    Returns:
        Path of the written stoplines file.
    """
    # Export stoplines
    stoplines = []
    for traffic_light in lanelet_network.traffic_lights:
        if not hasattr(traffic_light, 'iai_stoplines'):
            continue
        agent_type = 'traffic_light'
        opendrive_id = traffic_light.opendrive_id
        for left, right in traffic_light.iai_stoplines:
            left = np.array([*frame.to_local(left[0], left[1]), *left[2:]])
            right = np.array([*frame.to_local(right[0], right[1]), *right[2:]])
            center = (left + right) / 2
            right_to_left = left - right
            x, y, z = float(center[0]), float(center[1]), float(center[2])

            include_stopline = True
            if cfg.center is not None and cfg.trim_radius is not None:
                within_radius = (x - cfg.center[0]) ** 2 + (y - cfg.center[1]) ** 2 <= cfg.trim_radius ** 2
                include_stopline = include_stopline and within_radius
            if cfg.trim_vertical_limits is not None:
                within_limits = cfg.trim_vertical_limits[0] <= z <= cfg.trim_vertical_limits[1]
                include_stopline = include_stopline and within_limits

            if include_stopline:
                stopline = Stopline(
                    actor_id=opendrive_id, agent_type=agent_type, x=float(center[0]), y=float(center[1]),
                    length=1.0, width=float(np.linalg.norm(left - right)),
                    orientation=float(np.arctan2(right_to_left[1], right_to_left[0]) - (np.pi / 2)),
                )
                stoplines.append(dataclasses.asdict(stopline))
    traffic_signs = [('stop_sign', x) for x in road_network._iai_stop_signs] +\
                    [('yield_sign', x) for x in road_network._iai_yield_signs]
    for agent_type, (xy, orientation, length, width, opendrive_id) in traffic_signs:
        length = 1.0  # We could also adjust width and placement based on lanelet information
        if len(xy) > 2:
            z = float(xy[2])
        else:
            z = 0.0
        x, y = (float(v) for v in frame.to_local(xy[0], xy[1]))
        orientation = orientation + frame.rotation

        include_stopline = True
        if cfg.center is not None and cfg.trim_radius is not None:
            within_radius = (x - cfg.center[0]) ** 2 + (y - cfg.center[1]) ** 2 <= cfg.trim_radius ** 2
            include_stopline = include_stopline and within_radius
        if cfg.trim_vertical_limits is not None:
            within_limits = cfg.trim_vertical_limits[0] <= z <= cfg.trim_vertical_limits[1]
            include_stopline = include_stopline and within_limits

        if include_stopline:
            stopline = Stopline(
                actor_id=opendrive_id, agent_type=agent_type, x=x, y=y,
                length=float(length), width=float(width), orientation=float(orientation),
            )
            stoplines.append(dataclasses.asdict(stopline))
    stoplines_path = os.path.join(cfg.dir_path, f"{location}_stoplines.json")
    with open(stoplines_path, 'w') as f:
        logger.info(f'Writing extracted stoplines to {stoplines_path}')
        json.dump(stoplines, f, indent=4)
    return stoplines_path


def build_mesh(*, cfg: MapConversionConfig, location: str, osm_path: str, projector) -> BuiltMesh:
    """Build the road and lane-marking mesh from the written ``.osm`` and save ``<map>_mesh.json``."""
    # Export road mesh
    mesh_path = os.path.join(cfg.dir_path, f"{location}_mesh.json")
    lanelet_map = lanelet2.io.load(osm_path, projector)
    logger.info(f'Computing road mesh')
    road_mesh = road_mesh_from_lanelet_map(lanelet_map)
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
    )


def fitting_visualization_fov(*, verts: torch.Tensor, center: list[float]) -> float:
    """Width (m) of the square around ``center`` that holds every mesh vertex, plus 5%."""
    offsets = (verts.reshape(-1, 2) - torch.tensor(center, dtype=verts.dtype)).abs()
    return float(2.0 * offsets.max() * FOV_MARGIN) if offsets.numel() else DEFAULT_VISUALIZATION_FOV


def write_provenance(
    *,
    cfg: MapConversionConfig,
    xodr_path: str,
    out_dir: str,
    visualization_fov: float,
    traffic_side: Optional[str],
    geo_reference_status: str,
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
        "visualization_fov": visualization_fov,
        "traffic_side": traffic_side,
        "geo_reference": geo_reference_status,
        "converter": git_identity(repo=REPO_ROOT),
        "python": sys.version.split()[0],
        "packages": versions,
    }
    location = os.path.basename(xodr_path)[:-5]
    path = os.path.abspath(os.path.join(out_dir, f"{location}_provenance.json"))
    with open(path, "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info(f'Writing provenance to {path}')
    return path


def convert_map(cfg: MapConversionConfig) -> None:
    """Convert the single ``.xodr`` in ``cfg.dir_path`` into the TorchDriveSim artifact set.

    Writes ``<map>.osm``, ``<map>_stoplines.json``, ``<map>_mesh.json``, ``metadata.json``,
    ``visualization.png``, ``<map>_signal_groups.json`` and ``<map>_provenance.json`` into
    ``cfg.dir_path``.
    """
    opendrive_path = validate_config(cfg=cfg)
    parsed = parse_xodr(opendrive_path=opendrive_path, allow_zero_origin=cfg.allow_zero_origin)
    exported = export_osm(cfg=cfg, parsed=parsed)
    trim(cfg=cfg, osm_path=exported.osm_path, projector=parsed.projector)
    stoplines_path = extract_stoplines(
        cfg=cfg,
        location=parsed.location,
        lanelet_network=exported.lanelet_network,
        road_network=exported.road_network,
        frame=parsed.frame,
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
        stoplines_path=stoplines_path,
        mesh=mesh,
        opendrive_path=opendrive_path,
        traffic_side=parsed.traffic_side,
        geo_reference_status=parsed.geo_reference_status,
    )


if __name__ == '__main__':
    cfg: MapConversionConfig = OmegaConf.structured(
        MapConversionConfig(**OmegaConf.from_dotlist(sys.argv[1:]))
    )
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    try:
        convert_map(cfg)
    except ConfigError as exc:
        logger.error(f'Invalid configuration: {exc}')
        sys.exit(REFUSED_EXIT_CODE)
    except InputError as exc:
        logger.error(f'Cannot convert: {exc}')
        sys.exit(REFUSED_EXIT_CODE)
