import os
import argparse
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
    map_list: Optional[List[str]] = None
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
            ###################################################################################
            #Convert Lanelet2 to CR

            map_name = osm_map.split("/")[-1].split(".osm")[0]
            xml_path = f"{output_dir}{map_name}.xml"
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
            output_path = f"{output_dir}{map_name}.xodr"
            commonroad_to_opendrive(xml_path, output_path)
            
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
        map_list=['can:yukon_and_2nd', 'can:king_edward_and_columbia', 'can:victoria_drive_and_marine_drive', 'can:victoria_drive_and_41st', 'can:victoria_drive_and_41st_3', 'can:152_street_and_102a_avenue_surrey', 'can:fraser_hwy_and_160_street_surrey', 'usa:district_drive_and_blue_ridge_road_united_states', 'usa:rogers_road_and_heritage_branch_road_united_states', 'can:browns_line_and_coules_court_canada', 'usa:ligon_mill_road_and_south_main_street_united_states', 'usa:capital_boulevard_and_calvary_drive_united_states', 'usa:capital_boulevard_and_oak_forest_drive_united_states', 'usa:perry_creek_road_and_mcguire_drive_united_states', 'usa:west_division_street_and_north_orleans_street_united_states', 'usa:old_knight_road_and_knightdale_boulevard_united_states', 'usa:liles_dean_road_and_wendell_boulevard_united_states', 'usa:foundation_drive_and_forestville_road_united_states', 'usa:lee_street_sw_and_139_united_states', 'usa:langhorn_street_sw_and_139_united_states', 'usa:us_highway_41_and_mlk_jr_drive_sw_united_states', 'usa:w_division_street_and_n_halsted_street_united_states', 'usa:culver_drive_and_irvine_boulevard_united_states', 'usa:irvine_boulevard_and_old_myford_road_united_states', 'usa:exit_lane_to_irvine_boulevard_united_states', 'usa:west_diversity_parkway_and_north_sheridan_road_united_states', 'usa:edinger_avenue_and_brookhurst_street_united_states', 'usa:west_division_street_and_north_humboldt_drive_united_states_1', 'usa:north_lincoln_avenue_and_north_clark_street_united_states', 'grc:leof_iasonidou_and_dim_gounari_greece', 'usa:carroll_way_and_17th_st_united_states', 'can:williams_rd_and_no_2_rd_canada', 'can:e_41st_ave_and_rupert_st_canada', 'can:rupert_st_and_kingsway_canada', 'can:e_41st_ave_and_victoria_dr_canada', 'can:appleby_line_and_dryden_ave_canada', 'can:dundas_st_and_tim_dobbie_dr_canada', 'can:florence_ave_and_yonge_st_canada', 'can:mc_nicoll_ave_and_markham_rd_canada', 'can:dynamic_dr_and_mcnicoll_ave_canada', 'can:avondale_ave_and_bales_ave_canada', 'can:cherry_blossom_rd_and_fountain_st_n_canada', 'can:ironstone_dr_and_appleby_line_canada', 'can:wilson_dr_and_main_st_e_canada', 'can:harrison_ct_and_appleby_line_canada', 'can:upper_middle_rd_and_country_club_dr_canada', 'can:n_service_rd_and_appleby_line_canada', 'can:ontario_st_s_and_main_st_e_canada']
    )