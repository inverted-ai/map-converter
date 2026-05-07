"""
The required input is a folder `dir_path` with a single *.xodr file in it.
The name of that file will be the map name.
This script produces the required files for TorchDriveSim in the `dir_path` folder.
"""
import dataclasses
import glob
import os
import re
import sys
import logging
from typing import Tuple, Optional
import math
from pathlib import Path
from lxml import etree
from omegaconf import OmegaConf
import lanelet2

from crdesigner.common.config.general_config import GeneralConfig
from crdesigner.common.config.lanelet2_config import lanelet2_config
from crdesigner.map_conversion.lanelet2.cr2lanelet import CR2LaneletConverter
from crdesigner.map_conversion.opendrive.opendrive_parser.parser import parse_opendrive
from crdesigner.common.config.opendrive_config import OpenDriveConfig
from crdesigner.map_conversion.opendrive.opendrive_conversion.network import Network
from crdesigner.map_conversion.common.conversion_lanelet_network import ConversionLaneletNetwork

# Prevent bidirectional lanelets from being concatenated with adjacent junction connectors.
# Without this, a bidirectional road (going, e.g., SE) gets merged with a junction connector
# that has swapped inner/outer boundaries (going NW), producing crossed 65-vertex geometry.
_original_check_concat = ConversionLaneletNetwork.check_concatenation_potential

def _no_concat_for_bidir(self, lanelet, adjacent_direction):
    if lanelet.user_bidirectional:
        return None
    # Also skip if any successor is bidirectional — concatenating into a bidir lanelet
    # swaps its inner/outer boundaries relative to the predecessor, producing crossed geometry.
    for succ_id in lanelet.successor:
        succ = self.find_lanelet_by_id(succ_id)
        if succ and succ.user_bidirectional:
            return None
    return _original_check_concat(self, lanelet, adjacent_direction)

ConversionLaneletNetwork.check_concatenation_potential = _no_concat_for_bidir


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MapConversionConfig:
    xodr_path: str
    center: Optional[Tuple[float, float]] = None  # world center in local coordinates - by default road mesh center
    trim_radius: Optional[float] = None  # trim the map to this radius around the center
    trim_vertical_limits: Optional[Tuple[float, float]] = None  # lower and upper elevation bounds for trimming


@dataclasses.dataclass
class GeoOffset:
    x: float
    y: float
    z: float
    hdg: float

    @classmethod
    def from_dict(cls, offset_dict: dict):
        x = float(offset_dict['x'])
        y = float(offset_dict['y'])
        z = float(offset_dict['z'])
        hdg = float(offset_dict['hdg'])
        return cls(x, y, z, hdg)

    def apply_offset(self, x, y):
        x_translated = x + float(self.x)
        y_translated = y + float(self.y)
        hdg_rad = math.radians(float(self.hdg))
        x_rotated = x_translated * math.cos(hdg_rad) - y_translated * math.sin(hdg_rad)
        y_rotated = x_translated * math.sin(hdg_rad) + y_translated * math.cos(hdg_rad)
        return x_rotated, y_rotated

@dataclasses.dataclass
class GeoReference:
    lat: float
    lon: float

    @property
    def origin(self):
        return self.lat, self.lon

    @property
    def lanelet2_origin(self):
        return lanelet2.io.Origin(self.lat, self.lon)

    @property
    def proj_string(self):
        return f'+proj=tmerc +lat_0={self.lat} +lon_0={self.lon}'
    

def extract_geo_reference(geo_reference: str) -> Optional[GeoReference]:
    proj_pattern = r"\+proj=([\w]+)"
    lat_pattern = r"\+lat_0=([-\d.]+)"
    lon_pattern = r"\+lon_0=([-\d.]+)"
    proj_match = re.search(proj_pattern, geo_reference)
    lat_match = re.search(lat_pattern, geo_reference)
    lon_match = re.search(lon_pattern, geo_reference)

    if proj_match is not None and proj_match.group(1) == 'tmerc' and lat_match and lon_match:
        latitude = float(lat_match.group(1))
        longitude = float(lon_match.group(1))
        return GeoReference(latitude, longitude)
    else:
        return None


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
            trimmed_map.add(trimmed_lanelet)
    return trimmed_map


class CustomTransfomer:
    """
    I could not find a way to construct a pyproj transformer equivalent to the Lanelet2 UtmProjector,
    so I opted to use a wrapper instead.
    """
    def __init__(self, projector, offset: Optional[GeoOffset] = None):
        self.projector = projector
        self.offset = offset

    def transform(self, x, y):
        if self.offset is not None:
            x, y = self.offset.apply_offset(x, y)
        transformed = self.projector.reverse(lanelet2.core.BasicPoint3d(x, y, 0))
        return transformed.lat, transformed.lon


def _fix_bidirectional_topology(lanelet_network, tolerance: float = 5.0):
    """
    Fix incorrect predecessor/successor connections for bidirectional lanelets.

    When a road connects to a bidirectional lane at that lane's END (e.g. contactPoint=end
    in the xodr successor link), the converter incorrectly registers it as a predecessor
    of the forward-direction lanelet.  The predecessor's endpoint ends up near the
    bidirectional lanelet's END rather than its START, which causes cr2lanelet to share
    the wrong OSM nodes and produces a criss-cross shape in the output.

    We remove such connections here so that:
    - predecessors always terminate geometrically near the bidirectional lane's START, and
    - successors always originate geometrically near the bidirectional lane's END.

    The one_way=no tag (added separately) lets the routing engine traverse the bidirectional
    lane in the reverse direction without explicit reverse-direction topology links.
    """
    for lanelet in lanelet_network.lanelets:
        if not lanelet.user_bidirectional:
            continue

        la_start = lanelet.left_vertices[0][:2]
        la_end = lanelet.left_vertices[-1][:2]

        wrong_preds = []
        for pred_id in list(lanelet.predecessor):
            pred = lanelet_network.find_lanelet_by_id(pred_id)
            if pred is None:
                continue
            pred_end = pred.left_vertices[-1][:2]
            d_to_start = math.dist(pred_end, la_start)
            d_to_end = math.dist(pred_end, la_end)
            if d_to_end < tolerance and d_to_end < d_to_start:
                wrong_preds.append(pred_id)

        for pred_id in wrong_preds:
            pred = lanelet_network.find_lanelet_by_id(pred_id)
            lanelet.predecessor.remove(pred_id)
            if pred is not None and lanelet.lanelet_id in pred.successor:
                pred.successor.remove(lanelet.lanelet_id)

        wrong_succs = []
        for succ_id in list(lanelet.successor):
            succ = lanelet_network.find_lanelet_by_id(succ_id)
            if succ is None:
                continue
            succ_start = succ.left_vertices[0][:2]
            d_to_end = math.dist(succ_start, la_end)
            d_to_start = math.dist(succ_start, la_start)
            if d_to_start < tolerance and d_to_start < d_to_end:
                wrong_succs.append(succ_id)

        for succ_id in wrong_succs:
            succ = lanelet_network.find_lanelet_by_id(succ_id)
            lanelet.successor.remove(succ_id)
            if succ is not None and lanelet.lanelet_id in succ.predecessor:
                succ.predecessor.remove(lanelet.lanelet_id)

        # Detect "reverse connector" predecessors: a lanelet whose START is near bidir.END
        # and whose END is near bidir.START.  These are physically redundant roads that
        # traverse the bidirectional lane's space in the opposite direction.  We remove them
        # from the network entirely — the bidir one_way=no handles reverse routing.
        reverse_connector_preds = []
        for pred_id in list(lanelet.predecessor):
            pred = lanelet_network.find_lanelet_by_id(pred_id)
            if pred is None:
                continue
            pred_start = pred.left_vertices[0][:2]
            pred_end = pred.left_vertices[-1][:2]
            if (math.dist(pred_start, la_end) < tolerance and
                    math.dist(pred_end, la_start) < tolerance):
                reverse_connector_preds.append(pred_id)

        for pred_id in reverse_connector_preds:
            pred = lanelet_network.find_lanelet_by_id(pred_id)
            lanelet.predecessor.remove(pred_id)
            # Remove back-references from the reverse connector's own predecessors
            if pred is not None:
                for pp_id in list(pred.predecessor):
                    pp = lanelet_network.find_lanelet_by_id(pp_id)
                    if pp is not None and pred_id in pp.successor:
                        pp.successor.remove(pred_id)
            # Remove the reverse connector lanelet entirely so it won't be rendered
            lanelet_network.remove_lanelet(pred_id)


def _tag_bidirectional_relations(osm_tree):
    """Add one_way=no to every lanelet relation that has any one_way:* = no tag."""
    for rel in osm_tree.findall('.//relation'):
        tags = {t.get('k'): t.get('v') for t in rel.findall('tag')}
        if tags.get('type') != 'lanelet':
            continue
        has_bidirectional_tag = any(
            k.startswith('one_way:') and v == 'no' for k, v in tags.items()
        )
        if has_bidirectional_tag and 'one_way' not in tags:
            rel.append(etree.Element('tag', k='one_way', v='no'))


def convert_map(cfg: MapConversionConfig) -> str:
    # Find and parse OpenDRIVE file
    opendrive_path = cfg.xodr_path
    logger.info(f'Using {opendrive_path} as input')
    opendrive = parse_opendrive(Path(opendrive_path))

    # Construct Lanelet2 projector
    if opendrive.header.geo_reference is None:
        logger.warning(f'Geo reference not found in {opendrive_path} - Lanelet2 map will not be properly geo referenced')
        geo_reference = GeoReference(0.0, 0.0)
    else:
        geo_reference = extract_geo_reference(geo_reference=opendrive.header.geo_reference)
        if geo_reference is None:
            logger.warning(f'Unable to parse geo reference - Lanelet2 map will not be properly geo referenced')
            geo_reference = GeoReference(0.0, 0.0)
    if opendrive.header.offset is not None:
        geo_offset = GeoOffset.from_dict(opendrive.header.offset)
        geo_reference.offset = geo_offset
    else:
        geo_offset = None
    projector = lanelet2.projection.UtmProjector(geo_reference.lanelet2_origin)

    # Convert OpenDRIVE to CommonRoad
    open_drive_config = OpenDriveConfig()
    open_drive_config.min_delta_s = 1.0
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
        "bidirectional",
    ]
    road_network = Network()
    road_network.load_opendrive(opendrive)
    lanelet_network = road_network.export_lanelet_network(transformer=None, filter_types=open_drive_config.filter_types)

    # Fix incorrect predecessor/successor connections for bidirectional lanelets
    _fix_bidirectional_topology(lanelet_network)

    # Export CommonRoad to Lanelet2
    commonroad_config = GeneralConfig()
    commonroad_config.proj_string_cr = geo_reference.proj_string  # not currently used - see CustomTransformer
    l2osm = CR2LaneletConverter(config=lanelet2_config, cr_config=commonroad_config)
    osm = l2osm.convert_lanelet_network(lanelet_network, transformer=CustomTransfomer(projector, geo_offset))

    # Add one_way=no to all bidirectional lanelet relations
    _tag_bidirectional_relations(osm)

    osm_path = os.path.splitext(cfg.xodr_path)[0] + '.osm'
    with open(osm_path, "wb") as file_out:
        logger.info(f'Writing converted Lanelet2 map to {osm_path}')
        file_out.write(etree.tostring(osm, xml_declaration=True, encoding="UTF-8", pretty_print=True))

    # Trim the map if requested
    if cfg.trim_radius is not None:
        assert cfg.center is not None, "Must specify center for trimming"
        lanelet_map = lanelet2.io.load(osm_path, projector)
        trimmed_map = trim_map(lanelet_map, center=cfg.center, radius=cfg.trim_radius,
                               vertical_limits=cfg.trim_vertical_limits)
        lanelet2.io.write(osm_path, trimmed_map, projector)
    return osm_path