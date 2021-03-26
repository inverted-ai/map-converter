from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from typing import Union

from crdesigner.io.scenario_designer.toolboxes.toolbox_ui import CheckableComboBox
from crdesigner.io.scenario_designer.gui_resources.scenario_saving_dialog_ui import ScenarioDialogUI

from commonroad.scenario.scenario import Tag, TimeOfDay, Weather, Underground
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.scenario import Scenario, SCENARIO_VERSION
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile


class ScenarioDialog:
    def __init__(self):
        self.save_window = ScenarioDialogUI()
        self.connect_gui_elements()
        self.current_scenario: Union[None, Scenario] = None
        self.directory = ""

    def connect_gui_elements(self):
        self.save_window.button_directory.clicked.connect(lambda: self.select_directory())
        self.save_window.button_save.clicked.connect(lambda: self.save_scenario())

    def show(self, scenario: Scenario):
        self.current_scenario = scenario
        self.save_window.label_benchmark_id.setText(scenario.scenario_id.__str__())

        self.save_window.scenario_author.setText(self.current_scenario.author)
        self.save_window.scenario_affiliation.setText(self.current_scenario.affiliation)
        self.save_window.scenario_source.setText(self.current_scenario.source)
        self.save_window.scenario_time_step_size.setText(str(self.current_scenario.dt))
        self.save_window.scenario_tags.set_checked_items([t.value for t in self.current_scenario.tags]
                                                         if self.current_scenario.tags else [])
        self.save_window.scenario_config_id.setValue(self.current_scenario.scenario_id.configuration_id
                                                     if self.current_scenario.scenario_id.configuration_id else 1)
        self.save_window.cooperative_scenario.setChecked(self.current_scenario.scenario_id.cooperative
                                                         if self.current_scenario.scenario_id.cooperative else False)
        self.save_window.country.setCurrentText(self.current_scenario.scenario_id.country_id)
        self.save_window.scenario_scene_id.setValue(self.current_scenario.scenario_id.map_id)
        self.save_window.scenario_scene_name.setText(self.current_scenario.scenario_id.map_name)
        self.save_window.prediction_type.setCurrentText(self.current_scenario.scenario_id.obstacle_behavior)
        self.save_window.scenario_prediction_id.setValue(self.current_scenario.scenario_id.prediction_id
                                                         if self.current_scenario.scenario_id.prediction_id else 1)

        if scenario.location:
            self.save_window.scenario_geo_anme_id.setText(str(scenario.location.geo_name_id))
            self.save_window.scenario_latitude.setText(str(scenario.location.gps_latitude))
            self.save_window.scenario_longitude.setText(str(scenario.location.gps_longitude))
            if scenario.location.environment:
                self.save_window.scenario_time_of_day.setCurrentText(
                    self.current_scenario.location.environment.time_of_day.value)
                self.save_window.scenario_weather.setCurrentText(
                    self.current_scenario.location.environment.weather.value)
                self.save_window.scenario_underground.setCurrentText(
                    self.current_scenario.location.environment.underground.value)
                self.save_window.scenario_time_hour.setValue(self.current_scenario.location.environment.time.hours)
                self.save_window.scenario_time_minute.setValue(self.current_scenario.location.environment.time.minutes)
        else:
            self.save_window.scenario_geo_anme_id.setText("-1")
            self.save_window.scenario_latitude.setText("-999.9")
            self.save_window.scenario_longitude.setText("-999.9")
        self.save_window.show()

    def select_directory(self):
        self.directory = QFileDialog.getExistingDirectory(self.save_window, "Dir", options=QFileDialog.Options())
        if dir:
            self.save_window.label_directory.setText(self.directory)

    def save_scenario(self):
        self.current_scenario.author = self.save_window.scenario_author.text()
        self.current_scenario.affiliation = self.save_window.scenario_affiliation.text()
        self.current_scenario.source = self.save_window.scenario_source.text()
        self.current_scenario.tags = [Tag(t) for t in self.save_window.scenario_tags.get_checked_items()]
        self.current_scenario.scenario_id.configuration_id = int(self.save_window.scenario_config_id.text())
        self.current_scenario.scenario_id.cooperative = self.save_window.cooperative_scenario.isChecked()
        self.current_scenario.scenario_id.country_id = self.save_window.country.currentText()
        self.current_scenario.scenario_id.map_id = int(self.save_window.scenario_scene_id.text())
        self.current_scenario.scenario_id.map_name = self.save_window.scenario_scene_name.text()
        self.current_scenario.scenario_id.obstacle_behavior = self.save_window.prediction_type.currentText()
        self.current_scenario.scenario_id.prediction_id = int(self.save_window.scenario_prediction_id.text())
        self.current_scenario.scenario_id.scenario_version = SCENARIO_VERSION

        try:
            writer = CommonRoadFileWriter(
                scenario=self.current_scenario,
                planning_problem_set=None,
                author=self.current_scenario.author,
                affiliation=self.current_scenario.affiliation,
                source=self.current_scenario.source,
                tags=set(self.current_scenario.tags),
            )
            filename = self.directory + "/" + self.current_scenario.scenario_id.__str__() + ".xml"
            writer.write_scenario_to_file(filename, OverwriteExistingFile.ALWAYS)
            self.save_window.close()
        except IOError as e:
            QMessageBox.critical(
                self.save_window,
                "CommonRoad file not created!",
                "The CommonRoad file was not saved due to an error.\n\n" +
                "{}".format(e),
                QMessageBox.Ok,
            )
