import os
from pathlib import Path

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Tag

from crdesigner.common.config.lanelet2_config import lanelet2_config
from crdesigner.common.file_writer import CRDesignerFileWriter, OverwriteExistingFile
from crdesigner.map_conversion.map_conversion_interface import lanelet_to_commonroad
from crdesigner.map_conversion.map_conversion_interface import commonroad_to_opendrive


input_path = (
    Path.cwd().parent.parent / "tests/map_conversion/test_maps/lanelet2/merging_lanelets_utm.osm"
)
output_path = Path.cwd() / "example_files/lanelet2/merging_lanelets_utm.xml"

lanelet2_config.adjacencies = True

# load lanelet/lanelet2 file, parse it, and convert it to a CommonRoad scenario
scenario = lanelet_to_commonroad(str(input_path), lanelet2_conf=lanelet2_config)

# store converted file as CommonRoad scenario
writer = CRDesignerFileWriter(
    scenario=scenario,
    planning_problem_set=PlanningProblemSet(),
    author="Sebastian Maierhofer",
    affiliation="Technical University of Munich",
    source="CommonRoad Scenario Designer",
    tags={Tag.URBAN},
)

# create a folder for the example file if it does not exist
if os.path.exists(Path.cwd() / "example_files") is False:
    os.mkdir(Path.cwd() / "example_files")
if os.path.exists(Path.cwd() / "example_files/lanelet2") is False:
    os.mkdir(Path.cwd() / "example_files/lanelet2")

writer.write_to_file(str(output_path), OverwriteExistingFile.ALWAYS)


input_path = (
    Path.cwd().parent.parent / "tests/map_conversion/test_maps/cr2odr/ARG_Carcarana-1_1_T-1.xml"
)
output_name = Path.cwd() / "example_files/opendrive/ARG_Carcarana-1_1_T-1.xodr"

if not (Path.cwd() / "example_files/opendrive").exists():
    (Path.cwd() / "example_files/opendrive").mkdir(parents=True, exist_ok=True)

commonroad_to_opendrive(input_path, output_name)
