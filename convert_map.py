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


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MapConversionConfig:
    dir_path: str
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


def convert_map(cfg: MapConversionConfig) -> None:
    # Find and parse OpenDRIVE file
    xodr_files = glob.glob(os.path.join(cfg.dir_path, '*.xodr'))
    if not xodr_files:
        logger.error(f'No .xodr files found in {cfg.dir_path} - aborting')
        return
    opendrive_path = xodr_files[0]
    logger.info(f'Using {opendrive_path} as input')
    location = os.path.basename(opendrive_path)[:-5]
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

    # Export CommonRoad to Lanelet2
    commonroad_config = GeneralConfig()
    commonroad_config.proj_string_cr = geo_reference.proj_string  # not currently used - see CustomTransformer
    l2osm = CR2LaneletConverter(config=lanelet2_config, cr_config=commonroad_config)
    osm = l2osm.convert_lanelet_network(lanelet_network, transformer=CustomTransfomer(projector, geo_offset))
    osm_path = os.path.join(cfg.dir_path, f"{location}.osm")
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


if __name__ == '__main__':
    cfg: MapConversionConfig = OmegaConf.structured(
        MapConversionConfig(**OmegaConf.from_dotlist(sys.argv[1:]))
    )
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    convert_map(cfg)
