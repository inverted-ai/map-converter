import os
import argparse
import carla
from carla import Osm2OdrSettings, Osm2Odr

import invertedai as iai
from typing import Optional, List

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Tag

from crdesigner.common.config.lanelet2_config import lanelet2_config
from crdesigner.common.file_writer import CRDesignerFileWriter, OverwriteExistingFile
from crdesigner.map_conversion.map_conversion_interface import lanelet_to_commonroad
from crdesigner.map_conversion.map_conversion_interface import commonroad_to_opendrive


def main(
    output_dir: str,
    osm_path: Optional[str] = None,
    map_list: Optional[List[str]] = None,
    use_carla: Optional[bool] = False
):
    """
    A utility function to create a set of Regions to be passed into :func:`large_initialize` in
    a single convenient entry point. 

    Arguments
    ----------
    output_dir:
        Directory to output the final XODR maps.

    osm_path:
        Optional: If a path is provided, this one OSM map will be converted.

    map_list:
        Optional: A list of IAI formatted map strings to pull from location_info and save the OSM file to the output_dir.
    """

    
    lanelet2_path_list = []
    if osm_path is not None:
        lanelet2_path_list = [osm_path]

    if map_list is not None:
        for map_name in map_list:
            loc_info_res = iai.location_info(
                location=map_name,
                include_map_source=True
            )
            osm_map_path = os.path.join(output_dir,map_name.split(":")[-1]+".osm")
            loc_info_res.osm_map.save_osm_file(osm_map_path)
            lanelet2_path_list.append(osm_map_path)


    for osm_map in lanelet2_path_list:
        print(f"Now processing map: {osm_map}")
        
        try:
            map_name = osm_map.split("/")[-1].split(".osm")[0]
            xml_path = f"{output_dir}{map_name}.xml"
            output_path = f"{output_dir}{map_name}.xodr"
            if not use_carla:
                ###################################################################################
                #Convert Lanelet2 to CR
                lanelet2_config.adjacencies = True

                # load lanelet/lanelet2 file, parse it, and convert it to a CommonRoad scenario
                scenario = lanelet_to_commonroad(osm_map, lanelet2_conf=lanelet2_config)

                # store converted file as CommonRoad scenario
                writer = CRDesignerFileWriter(
                    scenario=scenario,
                    planning_problem_set=PlanningProblemSet(),
                    author="Sebastian Maierhofer",
                    affiliation="Technical University of Munich",
                    source="CommonRoad Scenario Designer",
                    tags={Tag.URBAN},
                )
                writer.write_to_file(xml_path, OverwriteExistingFile.ALWAYS)

                ###################################################################################
                #Convert CR to XODR
                commonroad_to_opendrive(xml_path, output_path)
            else:

                # Read the .osm data

                f = open(osm_map, 'r')
                osm_data = f.read()
                f.close()

                # Define the desired settings. In this case, default values.
                settings = carla.Osm2OdrSettings()
                # settings.center_map = True
                settings.generate_traffic_lights = True
                # Set OSM road types to export to OpenDRIVE
                settings.set_osm_way_types(["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential"])
                # Convert to .xodr
                xodr_data = carla.Osm2Odr.convert(osm_data, settings)

                # save opendrive file
                f = open(output_path, 'w')
                f.write(xodr_data)
                f.close()


        except Exception as e:
            print(f"{e}")
            print(f"Failed while converting map: {osm_map}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--osm_path',
        type=str,
        help=f"Full path to an OSM map to convert",
        default='None'
    )
    argparser.add_argument(
        '--output_dir',
        type=str,
        help=f"Directory to output converted OpenDrive map.",
        default='None'
    )
    args = argparser.parse_args()

    main(
        output_dir=args.output_dir,
        osm_path=args.osm_path if args.osm_path != "None" else None,
        use_carla=True,
    )