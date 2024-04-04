"""
The required input is a folder `dir_path` with a single *.xodr file in it.
The name of that file will be the map name.
This script produces the required files for TorchDriveSim in the `dir_path` folder.
"""
import dataclasses
import glob
import json
import os
import re
import sys
import logging
from typing import Tuple, Optional

import imageio
import numpy as np
from pathlib import Path

import torch.cuda
from lxml import etree
from omegaconf import OmegaConf

import lanelet2
from torchdrivesim.map import Stopline, MapConfig, store_map_config, resolve_paths_to_absolute
from torchdrivesim.lanelet2 import LaneletMap, load_lanelet_map, road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh
from torchdrivesim.mesh import BirdviewMesh
from torchdrivesim.rendering import renderer_from_config, RendererConfig
from torchdrivesim.traffic_controls import traffic_controls_from_map_config
from torchdrivesim.utils import Resolution

from crdesigner.common.config.general_config import GeneralConfig
from crdesigner.common.config.lanelet2_config import lanelet2_config
from crdesigner.map_conversion.lanelet2.cr2lanelet import CR2LaneletConverter
from crdesigner.map_conversion.opendrive.opendrive_parser.parser import parse_opendrive
from crdesigner.common.config.opendrive_config import OpenDriveConfig
from crdesigner.map_conversion.opendrive.opendrive_conversion.network import Network


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Config:
    dir_path: str
    domain: Optional[str] = None


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

    if proj_match.group(1) == 'tmerc' and lat_match and lon_match:
        latitude = float(lat_match.group(1))
        longitude = float(lon_match.group(1))
        return GeoReference(latitude, longitude)
    else:
        return None


class CustomTransfomer:
    """
    I could not find a way to construct a pyproj transformer equivalent to the Lanelet2 UtmProjector,
    so I opted to use a wrapper instead.
    """
    def __init__(self, projector):
        self.projector = projector

    def transform(self, x, y):
        transformed = self.projector.reverse(lanelet2.core.BasicPoint3d(x, y, 0))
        return transformed.lat, transformed.lon


if __name__ == '__main__':
    cfg: Config = OmegaConf.structured(
        Config(**OmegaConf.from_dotlist(sys.argv[1:]))
    )

    # Find and parse OpenDRIVE file
    xodr_files = glob.glob(os.path.join(cfg.dir_path, '*.xodr'))
    if not xodr_files:
        logger.error(f'No .xodr files found in {cfg.dir_path} - aborting')
    opendrive_path = xodr_files[0]
    logger.info(f'Using {opendrive_path} as input')
    location = os.path.basename(opendrive_path)[:-5]
    opendrive = parse_opendrive(Path(opendrive_path))

    # Construct Lanelet2 projector
    if opendrive.header.geo_reference is None:
        geo_reference = GeoReference(0.0, 0.0)
    else:
        geo_reference = extract_geo_reference(geo_reference=opendrive.header.geo_reference)
        if geo_reference is None:
            logger.warning(f'Unable to parse geo reference - Lanelet2 map will not be properly geo referenced')
            geo_reference = GeoReference(0.0, 0.0)
    projector = lanelet2.projection.UtmProjector(geo_reference.lanelet2_origin)

    # Convert OpenDRIVE to CommonRoad
    open_drive_config = OpenDriveConfig()
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
    road_network.load_opendrive(opendrive)
    # for index in range(len(road_network._traffic_lights)):
    #     road_network._traffic_lights[index]._traffic_light_id = abs(
    #         road_network._traffic_lights[index].traffic_light_id
    #     )
    # We leave everything in the original OpenDRIVE inertial coordinate frame
    lanelet_network = road_network.export_lanelet_network(transformer=None, filter_types=open_drive_config.filter_types)

    # Export CommonRoad to Lanelet2
    commonroad_config = GeneralConfig()
    commonroad_config.proj_string_cr = geo_reference.proj_string  # not currently used - see CustomTransformer
    l2osm = CR2LaneletConverter(config=lanelet2_config, cr_config=commonroad_config)
    osm = l2osm.convert_lanelet_network(lanelet_network, transformer=CustomTransfomer(projector))
    osm_path = os.path.join(cfg.dir_path, f"{location}.osm")
    with open(osm_path, "wb") as file_out:
        file_out.write(etree.tostring(osm, xml_declaration=True, encoding="UTF-8", pretty_print=True))

    # Export stoplines
    stoplines = []
    for traffic_light in lanelet_network.traffic_lights:
        if not hasattr(traffic_light, 'iai_stoplines'):
            continue
        agent_type = 'traffic-light'
        opendrive_id = traffic_light.opendrive_id
        for left, right in traffic_light.iai_stoplines:
            center = (left + right) / 2
            right_to_left = left - right
            stopline = Stopline(
                actor_id=opendrive_id, agent_type=agent_type, x=float(center[0]), y=float(center[1]),
                length=1.0, width=float(np.linalg.norm(left - right)),
                orientation=float(np.arctan2(right_to_left[1], right_to_left[0]) - (np.pi / 2)),
            )
            stoplines.append(dataclasses.asdict(stopline))
    traffic_signs = [('stop-sign', x) for x in road_network._iai_stop_signs] +\
                    [('yield-sign', x) for x in road_network._iai_yield_signs]
    for agent_type, (xy, orientation, length, width, opendrive_id) in traffic_signs:
        length = 1.0  # We could also adjust width and placement based on lanelet information
        stopline = Stopline(
            actor_id=opendrive_id, agent_type=agent_type, x=float(xy[0]), y=float(xy[1]),
            length=float(length), width=float(width), orientation=float(orientation),
        )
        stoplines.append(dataclasses.asdict(stopline))
    stoplines_path = os.path.join(cfg.dir_path, f"{location}_stoplines.json")
    with open(stoplines_path, 'w') as f:
        json.dump(stoplines, f, indent=4)

    # Export road mesh
    mesh_path = os.path.join(cfg.dir_path, f"{location}_mesh.json")
    lanelet_map = lanelet2.io.load(osm_path, projector)
    road_mesh = road_mesh_from_lanelet_map(lanelet_map)
    road_mesh = BirdviewMesh.set_properties(road_mesh, category='road').to(road_mesh.device)
    lane_mesh = lanelet_map_to_lane_mesh(lanelet_map, left_handed=False)
    combined_mesh = lane_mesh.merge(road_mesh)
    combined_mesh.save(mesh_path)

    # Write metadata
    center = combined_mesh.center.numpy().tolist()
    map_cfg = MapConfig(
        name=location, center=center, lanelet_map_origin=geo_reference.origin,
        iai_location_name=f'{cfg.domain}:{location}' if cfg.domain else None,
        left_handed_coordinates=False,
        lanelet_path=os.path.abspath(osm_path),
        mesh_path=mesh_path,
        stoplines_path=stoplines_path,
    )
    store_map_config(map_cfg, os.path.join(cfg.dir_path, 'metadata.json'))

    # Visualize results
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = Resolution(2048, 2048)
    map_cfg = resolve_paths_to_absolute(map_cfg, root=cfg.dir_path)
    driving_surface_mesh = map_cfg.road_mesh.to(device)
    renderer_cfg = RendererConfig(left_handed_coordinates=map_cfg.left_handed_coordinates)
    renderer = renderer_from_config(
        renderer_cfg, device=device, static_mesh=driving_surface_mesh
    )
    traffic_controls = traffic_controls_from_map_config(map_cfg)
    controls_mesh = renderer.make_traffic_controls_mesh(traffic_controls).to(renderer.device)
    renderer.add_static_meshes([controls_mesh])
    map_image = renderer.render_static_meshes(res=res, fov=800)
    imageio.imsave(
        os.path.join(cfg.dir_path, 'visualization.png'), map_image[0].cpu().numpy().astype(np.uint8)
    )
