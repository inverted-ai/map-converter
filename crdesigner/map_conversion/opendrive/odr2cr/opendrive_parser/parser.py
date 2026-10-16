import collections
import contextlib
import logging
import re
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from lxml import etree

from crdesigner.common.config.opendrive_config import open_drive_config
from crdesigner.map_conversion.common.utils import (
    clean_projection_string,
    generate_unique_id,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.junction import (
    Connection as JunctionConnection,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.junction import (
    Junction,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.junction import (
    LaneLink as JunctionConnectionLaneLink,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.opendrive import (
    Header,
    OpenDrive,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.road import (
    Road,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadElevationProfile import (
    ElevationRecord as RoadElevationProfile,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLanes import (
    Lane as RoadLaneSectionLane,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLanes import (
    LaneBorder as RoadLaneSectionLaneBorder,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLanes import (
    LaneOffset as RoadLanesLaneOffset,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLanes import (
    LaneSection as RoadLanesSection,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLanes import (
    LaneWidth as RoadLaneSectionLaneWidth,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLanes import (
    RoadMark as RoadLaneRoadMark,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLateralProfile import (
    Crossfall as RoadLateralProfileCrossfall,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLateralProfile import (
    Shape as RoadLateralProfileShape,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLateralProfile import (
    Superelevation as RoadLateralProfileSuperelevation,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLink import (
    Neighbor as RoadLinkNeighbor,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLink import (
    Predecessor as RoadLinkPredecessor,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadLink import (
    Successor as RoadLinkSuccessor,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadObject import (
    Object as RoadObject,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadObject import (
    ObjectOutlineCorner,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadSignal import (
    Signal as RoadSignal,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadSignal import (
    SignalReference,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser import vocabulary
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.vocabulary import record, translate
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadtype import (
    RoadType,
)
from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.elements.roadtype import (
    Speed as RoadTypeSpeed,
)


# A number written with a decimal comma, e.g. length="29,7078"
DECIMAL_COMMA = re.compile(r"^\s*[+-]?\d+,\d+(?:[eE][+-]?\d+)?\s*$")
# Attributes that hold ids, names or enums, never a number to re-read with a decimal point
NON_NUMERIC_ATTRIBUTES = {
    "id", "name", "type", "subtype", "country", "countryRevision", "elementId", "elementType",
    "junction", "incomingRoad", "connectingRoad", "linkedRoad", "signalId", "text", "unit", "rule",
    "contactPoint", "orientation", "dynamic", "side", "restriction",
}


class UnparseableNumber(ValueError):
    """An attribute that should hold a number does not; the enclosing element is dropped."""

    kind = "unparseable_number"

    def __init__(self, name: str, value: Optional[str]):
        super().__init__(f"{name}={value!r} is not a number")
        self.name = name
        self.value = value


class UnknownGeometry(ValueError):
    """A ``<geometry>`` record of an unknown type; the road it belongs to is dropped."""

    kind = "unknown_value"


def _record_dropped(element: str, element_id, error: ValueError):
    record(
        getattr(error, "kind", type(error).__name__), element, element_id, str(error), "dropped",
        "%s %s dropped: %s", element, element_id, error,
    )


def _validity_lanes(validity: etree.ElementTree, element: str, element_id) -> Optional[range]:
    """The lane ids a ``<validity>`` covers; ``None`` (recorded) if they are not integers."""
    from_lane, to_lane = validity.get("fromLane"), validity.get("toLane")
    try:
        return range(int(from_lane), int(to_lane) + 1)
    except ValueError:
        record(
            "unparseable_number", f"{element}.validity", element_id, f"{from_lane}..{to_lane}", "skipped",
            "%s %s: validity lanes %s..%s are not integers - validity skipped", element, element_id, from_lane, to_lane,
        )
        return None


@contextlib.contextmanager
def _dropped_on_error(element: str, element_id):
    """Drop (and record) the record built inside the block if one of its numbers does not parse."""
    try:
        yield
    except UnparseableNumber as error:
        _record_dropped(element, element_id, error)


def parse_opendrive(file_path: Path, odr_conf: open_drive_config = open_drive_config) -> OpenDrive:
    """
    Tries to parse XML tree, returns OpenDRIVE object

    :param file_path: path to OpenDRIVE file
    :param odr_conf: OpenDRIVE configuration.
    :return: Object representing an OpenDrive specification
    """
    generate_unique_id(odr_conf.initial_cr_id)  # reset IDs

    decimal_commas = collections.Counter()
    with file_path.open("rb") as file_in:
        root_node = etree.parse(file_in)

        for elem in root_node.getiterator():
            if not (
                isinstance(elem, etree._Comment) or isinstance(elem, etree._ProcessingInstruction)
            ):
                elem.tag = etree.QName(elem).localname
                for attr, value in list(elem.attrib.items()):
                    if "," in value and attr not in NON_NUMERIC_ATTRIBUTES and DECIMAL_COMMA.match(value):
                        elem.set(attr, value.replace(",", "."))
                        decimal_commas[f"{elem.tag}.{attr}"] += 1
        etree.cleanup_namespaces(root_node)
    for element, count in sorted(decimal_commas.items()):
        record(
            "decimal_comma", element, None, count, "read_as_decimal_point",
            "%d %s value(s) written with a decimal comma - read as a decimal point", count, element,
        )

    opendrive = OpenDrive()

    # Header
    header = root_node.find("header")
    if header is not None:
        parse_opendrive_header(opendrive, header)

    # Junctions
    for junction in root_node.findall("junction"):
        try:
            parse_opendrive_junction(opendrive, junction)
        except ValueError as error:
            _record_dropped("junction", junction.get("id"), error)

    # Load roads
    for road in root_node.findall("road"):
        try:
            parse_opendrive_road(opendrive, road)
        except ValueError as error:
            _record_dropped("road", road.get("id"), error)

    return opendrive


def parse_opendrive_road_link(new_road: Road, opendrive_road_link: etree.ElementTree):
    """
    Parses OpenDRIVE Road Link element

    :param new_road: Road element where link should be added
    :param opendrive_road_link: Loaded OpenDRIVE link
    """
    predecessor = opendrive_road_link.find("predecessor")

    if predecessor is not None:
        new_road.link.predecessor = _road_link(new_road.id, predecessor, RoadLinkPredecessor)

    successor = opendrive_road_link.find("successor")

    if successor is not None:
        new_road.link.successor = _road_link(new_road.id, successor, RoadLinkSuccessor)

    for neighbor in opendrive_road_link.findall("neighbor"):
        values = (
            ("neighbor.side", neighbor.get("side"), ("left", "right")),
            ("neighbor.elementId", neighbor.get("elementId"), None),
            ("neighbor.direction", neighbor.get("direction"), ("same", "opposite")),
        )
        if not _readable(new_road.id, "neighbor", values):
            continue
        new_neighbor = RoadLinkNeighbor(
            neighbor.get("side"), neighbor.get("elementId"), neighbor.get("direction")
        )

        new_road.link.neighbors.append(new_neighbor)


def _road_link(road_id, link: etree.ElementTree, link_class):
    """
    A road's predecessor or successor, or None when the reader cannot use it (recorded). (IAI)

    :param road_id: ID of the road the link belongs to
    :param link: XML predecessor or successor element
    :param link_class: RoadLinkPredecessor or RoadLinkSuccessor
    :return: The link, or None
    """
    element_type = translate("link.elementType", link.get("elementType"), road_id)
    contact_point = translate("link.contactPoint", link.get("contactPoint"), road_id)
    values = (
        ("link.elementType", element_type, vocabulary.ALLOWED["link.elementType"]),
        ("link.elementId", link.get("elementId"), None),
        ("link.contactPoint", contact_point, vocabulary.CONTACT_POINTS + (None,)),
    )
    if not _readable(road_id, link.tag, values):
        return None
    return link_class(element_type, link.get("elementId"), contact_point)


def _readable(road_id, tag: str, values) -> bool:
    """
    Whether every value is present and, where allowed values are given, one of them; the first that is not is
    recorded and its element skipped. (IAI)

    :param road_id: ID of the road the element belongs to
    :param tag: Tag of the element, for the message
    :param values: (element attribute, value, allowed values or None for any present value) triples
    :return: True when the element can be read
    """
    for element, value, allowed in values:
        if (allowed is None and value is None) or (allowed is not None and value not in allowed):
            record(
                "unknown_value" if value is not None else "missing_attribute", element, road_id, value, "skipped",
                "road %s %s: %s %r - %s skipped", road_id, tag, element, value, tag,
            )
            return False
    return True


def parse_opendrive_road_type(road: Road, opendrive_xml_road_type: etree.ElementTree):
    """
    Parse OpenDRIVE road type and appends it to road object.

    :param road: Road to append the parsed road_type to types
    :param opendrive_xml_road_type: XML element which contains the information
    """
    speed = None
    if opendrive_xml_road_type.find("speed") is not None:
        speed = _road_type_speed(road.id, opendrive_xml_road_type.find("speed"))

    use_type = translate("road.type", opendrive_xml_road_type.get("type"), road.id)
    if use_type not in RoadType.allowedTypes:
        record(
            "unknown_value" if use_type is not None else "missing_attribute",
            "road.type",
            road.id,
            use_type,
            "read_as_unknown",
            "road %s: type %r - read as unknown", road.id, use_type,
        )
        use_type = "unknown"
    road_type = RoadType(
        s_pos=opendrive_xml_road_type.get("s"),
        use_type=use_type,
        speed=speed,
    )
    road.types.append(road_type)


def _road_type_speed(road_id, speed: etree.ElementTree) -> Optional[RoadTypeSpeed]:
    """
    The speed of a road type, or None when its maximum is not a number (recorded). A unit the reader does not know,
    or none and no dictionary default, is read as m/s (recorded). (IAI)

    :param road_id: ID of the road
    :param speed: XML speed element of the road type
    :return: The speed, or None
    """
    max_speed = speed.get("max")
    if max_speed not in ["no limit", "undefined"]:
        try:
            parse_required_float(max_speed, "speed.max")
        except UnparseableNumber as error:
            _record_dropped("speed", road_id, error)
            return None
    unit = _speed_unit(road_id, "type", speed.get("unit"))
    if unit not in vocabulary.SPEED_UNIT_DIVISORS:
        record(
            "unknown_value" if unit is not None else "missing_attribute", "speed.unit", road_id, unit, "read_as_m/s",
            "road %s type: speed unit %r not recognised - read as m/s", road_id, unit,
        )
    return RoadTypeSpeed(max_speed=max_speed, unit=unit)


def _speed_unit(road_id, where: str, unit: Optional[str]) -> Optional[str]:
    """
    The unit of a speed: the translated attribute, or the dictionary default when it names none (recorded). (IAI)

    :param road_id: ID of the road
    :param where: Element holding the speed, for the message
    :param unit: Unit attribute as read from the file
    :return: The unit, or None
    """
    unit = translate("speed.unit", unit, road_id)
    if unit is None and vocabulary.DEFAULT_SPEED_UNIT is not None:
        record(
            "translated", "speed.unit", road_id, None, f"read_as_{vocabulary.DEFAULT_SPEED_UNIT}",
            "road %s %s: speed without unit read as %s (vocabulary)", road_id, where, vocabulary.DEFAULT_SPEED_UNIT,
        )
        unit = vocabulary.DEFAULT_SPEED_UNIT
    return unit


def defaultval(val: Optional[str], name: str, default: str = "0") -> str:
    """
    Replace a given value with a default value should it not exist in the XML and print a warning.

    :param val: Value that is inspected to be None
    :param name: Name of the variable inspected
    :param default: Alternative value given a val is None
    """
    default = str(default)
    if val is None or val.strip() == "":
        warnings.warn(
            "Parser could not find value for " + name + ", " + default + " is used per default."
        )
        return default
    else:
        return val


def parse_float(val: Optional[str], name: str, default: str = "0") -> float:
    """
    :func:`defaultval`, read as a float.

    :raises UnparseableNumber: if the value is present but not a number
    """
    text = defaultval(val, name, default)
    try:
        return float(text)
    except ValueError:
        raise UnparseableNumber(name, val) from None


def parse_required_float(val: Optional[str], name: str) -> float:
    """
    A float the element cannot do without.

    :raises UnparseableNumber: if the value is missing or not a number
    """
    try:
        return float(val)
    except (TypeError, ValueError):
        raise UnparseableNumber(name, val) from None


def parse_opendrive_road_geometry(
    new_road: Road, road_geometry: etree.ElementTree, offset: Dict[str, str]
):
    """
    Parse OpenDRIVE road geometry and appends it to road object.

    :param new_road: Road to append the parsed road geometry
    :param road_geometry: XML element which contains the information
    :param offset: Offset defined in OpenDRIVE.
    :raises UnparseableNumber: if a required number is missing or not a number
    :raises UnknownGeometry: if the record is of an unknown type
    """
    start_coord = np.array(
        [
            parse_required_float(road_geometry.get("x"), "geometry.x") - float(offset["x"]),
            parse_required_float(road_geometry.get("y"), "geometry.y") - float(offset["y"]),
        ]
    )
    hdg = parse_required_float(road_geometry.get("hdg"), "geometry.hdg")
    length = parse_required_float(road_geometry.get("length"), "geometry.length")
    local_hdg = hdg - float(offset["hdg"])

    def add_arc_or_line(curvature: float):
        if curvature != 0:
            new_road.plan_view.add_arc(start_coord, local_hdg, length, curvature)
        else:
            new_road.plan_view.add_line(start_coord, local_hdg, length)

    if road_geometry.find("line") is not None:
        new_road.plan_view.add_line(start_coord, local_hdg, length)

    elif road_geometry.find("spiral") is not None:
        curv_start = parse_float(road_geometry.find("spiral").get("curvStart"), "spiral.curvStart")
        curv_end = parse_float(road_geometry.find("spiral").get("curvEnd"), "spiral.curvEnd")
        if np.isclose(curv_start, curv_end):
            record(
                "degenerate_geometry", "geometry.spiral", new_road.id, curv_start, "read_as_arc",
                "road %s: spiral with equal start and end curvature %s - read as an arc", new_road.id, curv_start,
            )
            add_arc_or_line(curv_start)
        else:
            new_road.plan_view.add_spiral(
                start_coord,
                local_hdg,
                length,
                curv_start,
                curv_end,
            )
    elif road_geometry.find("arc") is not None:
        add_arc_or_line(parse_required_float(road_geometry.find("arc").get("curvature"), "arc.curvature"))

    elif road_geometry.find("poly3") is not None:
        new_road.plan_view.add_poly3(
            start_coord,
            hdg,
            length,
            parse_float(road_geometry.find("poly3").get("a"), "poly3.a"),
            parse_float(road_geometry.find("poly3").get("b"), "poly3.b"),
            parse_float(road_geometry.find("poly3").get("c"), "poly3.c"),
            parse_float(road_geometry.find("poly3").get("d"), "poly3.d"),
        )
        # NotImplementedError()

    elif road_geometry.find("paramPoly3") is not None:
        if road_geometry.find("paramPoly3").get("pRange"):
            if road_geometry.find("paramPoly3").get("pRange") == "arcLength":
                p_max = length
            else:
                p_max = None
        else:
            p_max = None

        new_road.plan_view.add_param_poly3(
            start_coord,
            hdg,
            length,
            parse_float(road_geometry.find("paramPoly3").get("aU"), "paramPoly3.aU"),
            parse_float(road_geometry.find("paramPoly3").get("bU"), "paramPoly3.bU", 1),
            parse_float(road_geometry.find("paramPoly3").get("cU"), "paramPoly3.cU"),
            parse_float(road_geometry.find("paramPoly3").get("dU"), "paramPoly3.dU"),
            parse_float(road_geometry.find("paramPoly3").get("aV"), "paramPoly3.aV"),
            parse_float(road_geometry.find("paramPoly3").get("bV"), "paramPoly3.bV"),
            parse_float(road_geometry.find("paramPoly3").get("cV"), "paramPoly3.cV"),
            parse_float(road_geometry.find("paramPoly3").get("dV"), "paramPoly3.dV"),
            p_max,
        )

    else:
        child = next((c.tag for c in road_geometry if isinstance(c.tag, str)), None)
        raise UnknownGeometry(f"unknown geometry type {child!r}")


def parse_opendrive_road_elevation_profile(
    new_road: Road, road_elevation_profile: etree.ElementTree
):
    """
    Parse OpenDRIVE road elevation profile and appends it to road object.

    :param new_road: Road to append the parsed road elevation profile
    :param road_elevation_profile: XML element which contains the information
    """
    for elevation in road_elevation_profile.findall("elevation"):
        with _dropped_on_error("elevation", new_road.id):
            new_elevation = RoadElevationProfile(
                parse_float(elevation.get("a"), "elevation.a"),
                parse_float(elevation.get("b"), "elevation.b"),
                parse_float(elevation.get("c"), "elevation.c"),
                parse_float(elevation.get("d"), "elevation.d"),
                start_pos=parse_float(elevation.get("s"), "elevation.s"),
            )

            new_road.elevation_profile.elevations.append(new_elevation)


def parse_opendrive_road_lateral_profile(new_road: Road, road_lateral_profile: etree.ElementTree):
    """
    Parse OpenDRIVE road lateral profile and appends it to road object.

    :param new_road: Road to append the parsed road lateral profile
    :param road_lateral_profile: XML element which contains the information
    """
    for super_elevation in road_lateral_profile.findall("superelevation"):
        with _dropped_on_error("superelevation", new_road.id):
            new_super_elevation = RoadLateralProfileSuperelevation(
                parse_float(super_elevation.get("a"), "super_elevation.a"),
                parse_float(super_elevation.get("b"), "super_elevation.b"),
                parse_float(super_elevation.get("c"), "super_elevation.c"),
                parse_float(super_elevation.get("d"), "super_elevation.d"),
                start_pos=parse_float(super_elevation.get("s"), "super_elevation.s"),
            )

            new_road.lateral_profile.superelevations.append(new_super_elevation)

    for crossfall in road_lateral_profile.findall("crossfall"):
        with _dropped_on_error("crossfall", new_road.id):
            new_crossfall = RoadLateralProfileCrossfall(
                parse_float(crossfall.get("a"), "crossfall.a"),
                parse_float(crossfall.get("b"), "crossfall.b"),
                parse_float(crossfall.get("c"), "crossfall.c"),
                parse_float(crossfall.get("d"), "crossfall.d"),
                side=defaultval(crossfall.get("side"), "crossfall.side", "both"),
                start_pos=parse_float(crossfall.get("s"), "crossfall.s"),
            )

            new_road.lateral_profile.crossfalls.append(new_crossfall)

    for shape in road_lateral_profile.findall("shape"):
        with _dropped_on_error("shape", new_road.id):
            new_shape = RoadLateralProfileShape(
                parse_float(shape.get("a"), "shape.a"),
                parse_float(shape.get("b"), "shape.b"),
                parse_float(shape.get("c"), "shape.c"),
                parse_float(shape.get("d"), "shape.d"),
                start_pos=parse_float(shape.get("s"), "shape.s"),
                start_pos_t=parse_float(shape.get("t"), "shape.t"),
            )

            new_road.lateral_profile.shapes.append(new_shape)


def parse_opendrive_road_lane_offset(new_road: Road, lane_offset: etree.ElementTree):
    """
    Parse OpenDRIVE road lane offset and appends it to road object.

    :param new_road: Road to append the parsed road lane offset
    :param lane_offset: XML element which contains the information
    """
    new_lane_offset = RoadLanesLaneOffset(
        parse_float(lane_offset.get("a"), "lane_offset.a"),
        parse_float(lane_offset.get("b"), "lane_offset.b"),
        parse_float(lane_offset.get("c"), "lane_offset.c"),
        parse_float(lane_offset.get("d"), "lane_offset.d"),
        start_pos=parse_float(lane_offset.get("s"), "lane_offset.s"),
    )

    new_road.lanes.lane_offsets.append(new_lane_offset)


def parse_opendrive_road_lane_section(
    new_road: Road, lane_section_id: int, lane_section: etree.ElementTree
):
    """
    Parse OpenDRIVE road lane section and appends it to road object.

    :param new_road: Road to append the parsed road lane section
    :param lane_section_id: ID which should be assigned to lane section
    :param lane_section: XML element which contains the information
    """

    new_lane_section = RoadLanesSection(road=new_road)

    # NOTE: Manually enumerate lane sections for referencing purposes
    new_lane_section.idx = lane_section_id

    new_lane_section.sPos = parse_required_float(lane_section.get("s"), "laneSection.s")
    single_side = lane_section.get("singleSide")
    if single_side not in ["true", "false", None]:
        record(
            "unknown_value", "laneSection.singleSide", new_road.id, single_side, "read_as_false",
            "road %s lane section %s: singleSide %r - read as false", new_road.id, lane_section_id, single_side,
        )
        single_side = "false"
    new_lane_section.single_side = single_side

    sides = dict(
        left=new_lane_section.left_lanes,
        center=new_lane_section.center_lanes,
        right=new_lane_section.right_lanes,
    )

    for sideTag, newSideLanes in sides.items():
        side = lane_section.find(sideTag)

        # It is possible one side is not present
        if side is None:
            continue

        for lane in side.findall("lane"):
            new_lane = RoadLaneSectionLane(parentRoad=new_road, lane_section=new_lane_section)
            new_lane.id = lane.get("id")
            new_lane.type = lane.get("type")

            # In some sample files the level is not specified according to the OpenDRIVE spec
            new_lane.level = "true" if lane.get("level") in [1, "1", "true"] else "false"

            # Lane Links
            if lane.find("link") is not None:
                if lane.find("link").find("predecessor") is not None:
                    new_lane.link.predecessorId = lane.find("link").find("predecessor").get("id")

                if lane.find("link").find("successor") is not None:
                    new_lane.link.successorId = lane.find("link").find("successor").get("id")

            # Width
            for widthIdx, width in enumerate(lane.findall("width")):
                with _dropped_on_error("width", new_road.id):
                    new_width = RoadLaneSectionLaneWidth(
                        parse_float(width.get("a"), "width.a"),
                        parse_float(width.get("b"), "width.b"),
                        parse_float(width.get("c"), "width.c"),
                        parse_float(width.get("d"), "width.d"),
                        idx=widthIdx,
                        start_offset=parse_float(width.get("sOffset"), "width.sOffset"),
                    )

                    new_lane.widths.append(new_width)

            # Border
            for borderIdx, border in enumerate(lane.findall("border")):
                with _dropped_on_error("border", new_road.id):
                    new_border = RoadLaneSectionLaneBorder(
                        parse_float(border.get("a"), "border.a"),
                        parse_float(border.get("b"), "border.b"),
                        parse_float(border.get("c"), "border.c"),
                        parse_float(border.get("d"), "border.d"),
                        idx=borderIdx,
                        start_offset=parse_float(border.get("sOffset"), "border.sOffset"),
                    )

                    new_lane.borders.append(new_border)

            if lane.find("width") is None and lane.find("border") is not None:
                new_lane.widths = new_lane.borders
                new_lane.has_border_record = True

            # Road Marks
            for mark in lane.findall("roadMark"):
                road_mark = RoadLaneRoadMark()

                road_mark.type = mark.get("type")
                if mark.get("weight") is not None:
                    road_mark.weight = mark.get("weight")
                road_mark.SOffset = mark.get("sOffset")

                new_lane.road_mark.append(road_mark)

            # Material and Rules are not implemented in CommonRoad -> no need

            # Visiblility -> not part of ASAM OpenDRIVE 1.7.0 anymore
            # TODO implementation

            # Speed
            new_lane.speed = None
            speed = lane.find("speed")
            if speed is not None:
                # speed is always converted to m/s
                unit = _speed_unit(new_road.id, f"lane {new_lane.id}", speed.get("unit"))
                divisor = vocabulary.SPEED_UNIT_DIVISORS.get(unit)
                if divisor is None:
                    record(
                        "unknown_value" if unit is not None else "missing_attribute",
                        "speed.unit",
                        new_road.id,
                        unit,
                        "speed_unset",
                        "road %s lane %s: speed unit %r not recognised - speed left unset",
                        new_road.id,
                        new_lane.id,
                        unit,
                    )
                else:
                    with _dropped_on_error("speed", new_road.id):
                        new_lane.speed = parse_required_float(speed.get("max"), "speed.max") / divisor

            # Access
            for _access in lane.findall("access"):
                with _dropped_on_error("access", new_road.id):
                    new_lane.access += [
                        [
                            str(_access.get("restriction")),
                            str(_access.get("rule")),
                            parse_float(_access.get("sOffset"), "access.sOffset"),
                        ]
                    ]
            # Lane Height
            # TODO implementation

            newSideLanes.append(new_lane)

    new_road.lanes.lane_sections.append(new_lane_section)


def parse_opendrive_road_signal(new_road: Road, road_signal: etree.ElementTree):
    """
    Parse OpenDRIVE road signal and appends it to road object.

    :param new_road: Road to append the parsed road lane section
    :param road_signal: XML element which contains the information
    """
    new_signal = RoadSignal()
    new_signal.id = road_signal.get("id")
    new_signal.s = road_signal.get("s")  # position along the reference curve
    new_signal.t = road_signal.get("t")  # position away from the reference curve
    new_signal.name = road_signal.get("name")
    new_signal.country = road_signal.get("country")
    new_signal.type = road_signal.get("type")
    new_signal.subtype = road_signal.get("subtype")
    new_signal.orientation = road_signal.get("orientation")
    new_signal.dynamic = road_signal.get("dynamic")
    try:
        new_signal.signal_value = road_signal.get("value")
    except ValueError:
        record(
            "unparseable_number", "signal.value", new_signal.id, road_signal.get("value"), "value_dropped",
            "signal %s: value %r is not a number - value dropped", new_signal.id, road_signal.get("value"),
        )
        new_signal.signal_value = None
    new_signal.unit = road_signal.get("unit")
    new_signal.text = road_signal.get("text")
    if (
        road_signal.find("validity") is not None
        and road_signal.find("validity").get("fromLane") is not None
        and road_signal.find("validity").get("toLane") is not None
    ):
        new_signal.validity_from = road_signal.find("validity").get("fromLane")
        new_signal.validity_to = road_signal.find("validity").get("toLane")
    if (
        road_signal.find("userData") is not None
        and road_signal.find("userData").find("vectorSignal") is not None
    ):
        new_signal.signal_id = road_signal.find("userData").find("vectorSignal").get("signalId")

    for child in road_signal.iterchildren():
        if child.tag == 'validity':
            if 'fromLane' not in child.keys() or 'toLane' not in child.keys():
                record(
                    "missing_attribute", "signal.validity", new_signal.id, None, "skipped",
                    "signal %s: validity without fromLane/toLane - validity skipped", new_signal.id,
                )
                continue
            lanes = _validity_lanes(child, "signal", new_signal.id)
            if lanes is None:
                continue
            for lane_idx in lanes:
                lane = new_road.lanes.lane_sections[0].get_lane(lane_idx)
                if lane is None:
                    record(
                        "unknown_reference", "signal.validity", new_signal.id, lane_idx, "skipped",
                        "signal %s: validity lane %s not found in the first lane section - lane skipped",
                        new_signal.id,
                        lane_idx,
                    )
                    continue
                if hasattr(lane, 'signal_references'):
                    lane.signal_references.append(new_signal.id)
                else:
                    lane.signal_references = [new_signal.id]

    new_road.add_signal(new_signal)



def parse_opendrive_road_signal_reference(new_road: Road, road_signal_reference: etree.ElementTree):
    """
    Parse OpenDRIVE road signal reference and appends it to road object.

    :param new_road: Road to append the parsed road signal reference
    :param road_signal_reference: XML element which contains the information
    """
    new_signal_reference = SignalReference()
    new_signal_reference.id = road_signal_reference.get("id")
    new_signal_reference.s = road_signal_reference.get("s")  # position along the reference curve
    new_signal_reference.t = road_signal_reference.get(
        "t"
    )  # position away from the reference curve
    new_signal_reference.orientation = road_signal_reference.get("orientation")
    if (
        road_signal_reference.find("validity") is not None
        and road_signal_reference.find("validity").get("fromLane") is not None
        and road_signal_reference.find("validity").get("toLane") is not None
    ):
        new_signal_reference.validity_from = road_signal_reference.find("validity").get("fromLane")
        new_signal_reference.validity_to = road_signal_reference.find("validity").get("toLane")
    if (
        road_signal_reference.find("userData") is not None
        and road_signal_reference.find("userData").find("vectorSignal") is not None
    ):
        if road_signal_reference.find("userData").find("vectorSignal").get("signalId") is not None:
            new_signal_reference.signal_id = (
                road_signal_reference.find("userData").find("vectorSignal").get("signalId")
            )
        if (
            road_signal_reference.find("userData").find("vectorSignal").get("turnRelation")
            is not None
        ):
            new_signal_reference.turn_relation = (
                road_signal_reference.find("userData").find("vectorSignal").get("turnRelation")
            )

    for child in road_signal_reference.iterchildren():
        if child.tag == 'validity':
            if 'fromLane' not in child.keys() or 'toLane' not in child.keys():
                record(
                    "missing_attribute", "signalReference.validity", new_signal_reference.id, None, "skipped",
                    "signalReference %s: validity without fromLane/toLane - validity skipped", new_signal_reference.id,
                )
                continue
            lanes = _validity_lanes(child, "signalReference", new_signal_reference.id)
            if lanes is None:
                continue
            for lane_idx in lanes:
                lane = new_road.lanes.lane_sections[0].get_lane(lane_idx)
                if lane is None:
                    record(
                        "unknown_reference", "signalReference.validity", new_signal_reference.id, lane_idx, "skipped",
                        "signalReference %s: validity lane %s not found in the first lane section - lane skipped",
                        new_signal_reference.id,
                        lane_idx,
                    )
                    continue
                if hasattr(lane, 'signal_references'):
                    lane.signal_references.append(new_signal_reference.id)
                else:
                    lane.signal_references = [new_signal_reference.id]

    new_road.add_signal_reference(new_signal_reference)



def parse_opendrive_road(opendrive: OpenDrive, road: etree.ElementTree):
    """
    Parse OpenDRIVE road and appends it to OpenDRIVE object.

    :param opendrive: OpenDRIVE object to append the parsed road
    :param road: XML element which contains the information
    """

    new_road = Road()

    new_road.id = int(road.get("id"))
    new_road.name = road.get("name")
    new_road.driving_direction = True  # RHT default
    driving_direction = road.get("rule")
    if driving_direction is not None:
        if driving_direction == "LHT":
            new_road.driving_direction = False

    junction = road.get("junction")
    junction_id = None if junction in (None, "-1") else int(junction)

    if junction_id is not None:
        new_road.junction = opendrive.getJunction(junction_id)

    # TODO verify road length
    # length had no getter/setter and attribute was therefore deleted in road.py
    # new_road.length = float(road.get("length"))

    # Links
    opendrive_road_link = road.find("link")
    if opendrive_road_link is not None:
        parse_opendrive_road_link(new_road, opendrive_road_link)

    # Type
    for opendrive_xml_road_type in road.findall("type"):
        parse_opendrive_road_type(new_road, opendrive_xml_road_type)

    # Plan view
    for road_geometry in road.find("planView").findall("geometry"):
        # Geometry stays in file coordinates; the header offset is applied when projecting.
        parse_opendrive_road_geometry(new_road, road_geometry, {"x": "0", "y": "0", "hdg": "0"})
    # Elevation profile
    road_elevation_profile = road.find("elevationProfile")
    if road_elevation_profile is not None:
        parse_opendrive_road_elevation_profile(new_road, road_elevation_profile)

    # Lateral profile
    road_lateral_profile = road.find("lateralProfile")
    if road_lateral_profile is not None:
        parse_opendrive_road_lateral_profile(new_road, road_lateral_profile)

    # Lanes
    lanes = road.find("lanes")

    if lanes is None:
        raise Exception("Road must have lanes element")

    # Lane offset
    for lane_offset in lanes.findall("laneOffset"):
        with _dropped_on_error("laneOffset", new_road.id):
            parse_opendrive_road_lane_offset(new_road, lane_offset)

    # Lane sections
    for lane_section_id, lane_section in enumerate(road.find("lanes").findall("laneSection")):
        parse_opendrive_road_lane_section(new_road, lane_section_id, lane_section)

    # Objects
    if road.find("objects") is not None:
        for obj in road.find("objects").findall("object"):
            if obj is not None:
                try:
                    parse_opendrive_road_object(new_road, obj)
                except ValueError as error:
                    _record_dropped("object", obj.get("id"), error)

    # Signals
    if road.find("signals") is not None:
        for road_signal in road.find("signals").findall("signal"):
            if road_signal is not None:
                try:
                    parse_opendrive_road_signal(new_road, road_signal)
                except ValueError as error:
                    _record_dropped("signal", road_signal.get("id"), error)
        for road_signal_reference in road.find("signals").findall("signalReference"):
            # Parsing signal reference element
            if road_signal_reference is not None:
                try:
                    parse_opendrive_road_signal_reference(new_road, road_signal_reference)
                except ValueError as error:
                    _record_dropped("signalReference", road_signal_reference.get("id"), error)
    else:
        pass

    calculate_lane_section_lengths(new_road)

    opendrive.roads.append(new_road)


def parse_opendrive_road_object(new_road: Road, obj: etree.ElementTree):
    """Parses opendrive road object, creates roadObject from it and adds it to the road.

    :param new_road: The road to add the object to.
    :param obj: XML road element which is parsed.
    """
    corners = []
    if obj.find("outlines") is not None:
        outlines = obj.find("outlines").findall("outline")
    elif (
        obj.find("outline") is not None
    ):  # based on the OpenDRIVE specification outline should be within outlines;
        # probably a bug in a previous RoadRunner version
        outlines = [obj.find("outline")]
    else:
        outlines = []

    if len(outlines) > 0:
        if len(outlines) > 1:
            logging.warning(
                "We do not support outlines consisting of several shapes at the moment."
            )
        for corner_local in outlines[0].findall("cornerLocal"):
            if corner_local is not None:
                corner = ObjectOutlineCorner()
                corner.u = corner_local.get("u")
                corner.v = corner_local.get("v")
                corner.z = corner_local.get("z")
                corners.append(corner)

    road_object = RoadObject()
    try:
        if obj.get("id") is not None:
            road_object.id = obj.get("id")
        if obj.get("type") is not None:
            road_object.type = obj.get("type")
        if obj.get("s") is not None:
            road_object.s = obj.get("s")
        if obj.get("t") is not None:
            road_object.t = obj.get("t")
        if obj.get("name") is not None:
            road_object.name = obj.get("name")
        if obj.get("width") is not None:
            road_object.width = obj.get("width")
        if obj.get("height") is not None:
            road_object.height = obj.get("height")
        if obj.get("length") is not None:
            road_object.validLength = obj.get("length")
        if obj.get("zOffset") is not None:
            road_object.zOffset = obj.get("zOffset")
        if obj.get("validLength") is not None:
            road_object.validLength = obj.get("validLength")
        if obj.get("orientation") is not None:
            road_object.orientation = obj.get("orientation")
        if obj.get("hdg") is not None:
            road_object.hdg = obj.get("hdg")
        if obj.get("pitch") is not None:
            road_object.pitch = obj.get("pitch")
        if obj.get("roll") is not None:
            road_object.roll = obj.get("roll")

        road_object.outline = corners
    except KeyError:
        logging.error(
            "parse_opendrive_road_object: Error during parsing of road objects. "
            "Object id {}".format(obj.attrib.get("id"))
        )

    new_road.add_object(road_object)


def calculate_lane_section_lengths(new_road: Road):
    """
    Calculates lane section length for OpenDRIVE road.

    :param new_road: OpenDRIVE road for which lane section length should be calculated
    """

    # OpenDRIVE does not provide lane section lengths by itself, calculate them by ourselves
    for lane_section in new_road.lanes.lane_sections:
        # Last lane section in road
        if lane_section.idx + 1 >= len(new_road.lanes.lane_sections):
            lane_section.length = new_road.plan_view.length - lane_section.sPos

        # All but the last lane section end at the succeeding one
        else:
            lane_section.length = (
                new_road.lanes.lane_sections[lane_section.idx + 1].sPos - lane_section.sPos
            )

    # OpenDRIVE does not provide lane width lengths by itself, calculate them by ourselves
    for lane_section in new_road.lanes.lane_sections:
        for lane in lane_section.all_lanes:
            widths_poses = np.array([x.start_offset for x in lane.widths] + [lane_section.length])
            widths_lengths = widths_poses[1:] - widths_poses[:-1]

            for widthIdx, width in enumerate(lane.widths):
                width.length = widths_lengths[widthIdx]


def parse_opendrive_header(opendrive: OpenDrive, header: etree.ElementTree):
    """
    Parse OpenDRIVE header and appends it to OpenDRIVE object.

    :param opendrive: OpenDRIVE object to append the parsed header
    :param header: XML element which contains the information
    """

    # Generates object out of the attributes of the header
    parsed_header = Header(
        header.get("revMajor"),
        header.get("revMinor"),
        header.get("name"),
        header.get("version"),
        header.get("date"),
        header.get("north"),
        header.get("south"),
        header.get("east"),
        header.get("west"),
        header.get("vendor"),
        header.get("offset")
    )

    # Reference
    if header.find("geoReference") is not None:
        parsed_header.geo_reference = clean_projection_string(header.find("geoReference").text)

    # offset {x: , y: , z: , hdg:}
    if header.find("offset") is not None:
        parsed_header.offset = header.find("offset").attrib

    opendrive.header = parsed_header


def parse_opendrive_junction(opendrive: OpenDrive, junction: etree.ElementTree):
    """
    Parse OpenDRIVE junction and appends it to OpenDRIVE object.

    :param opendrive: OpenDRIVE object to append the parsed junction
    :param junction: XML element which contains the information
    """

    new_junction = Junction()

    new_junction.id = int(junction.get("id"))
    new_junction.name = junction.get("name")

    for connection in junction.findall("connection"):
        new_connection = JunctionConnection()

        new_connection.id = connection.get("id")
        new_connection.incomingRoad = connection.get("incomingRoad")
        
        connecting_road = connection.get("connectingRoad")
        contact_point = translate("connection.contactPoint", connection.get("contactPoint"), new_junction.id)
        if contact_point not in vocabulary.CONTACT_POINTS:
            record(
                "unknown_value" if contact_point is not None else "missing_attribute",
                "connection.contactPoint",
                new_junction.id,
                contact_point,
                "skipped",
                "junction %s connection %s: contactPoint %r - connection skipped",
                new_junction.id,
                connection.get("id"),
                contact_point,
            )
            continue
        if connecting_road is None:  # sometimes we observed linkedRoad being used instead
            connecting_road = connection.get("linkedRoad")
        new_connection.connectingRoad = connecting_road
        new_connection.contactPoint = contact_point

        for laneLink in connection.findall("laneLink"):
            new_lane_link = JunctionConnectionLaneLink()

            new_lane_link.fromId = laneLink.get("from")
            new_lane_link.toId = laneLink.get("to")

            new_connection.addLaneLink(new_lane_link)

        new_junction.addConnection(new_connection)

    opendrive.junctions.append(new_junction)
