import logging
import math

# IAI: the reader's tables, in one place. The repo-root vocabulary.py sets them from the vocabulary dictionary; the
# values here are what the reader uses without one.

# Names an OpenDRIVE file may use, translated into names the reader understands, per element attribute,
# e.g. {"object.orientation": {"undefined": "none"}}.
TRANSLATIONS = {}
# Object names read as painted stop / yield stencils.
STOP_STENCIL_NAMES = ["Stencil_STOP"]
YIELD_STENCIL_NAMES = ["Stencil_GiveWay", "Stencil_Yield_Triangles"]
# The sign table for a country without one.
UNKNOWN_COUNTRY_TABLE = "ZAM"
# Per country, sign types read as another type of that country's table.
SIGN_TYPE_REPLACEMENTS = {"DEU": {"252": "260"}, "ZAM": {"252": "260"}}
# Dynamic signal types that are no vehicle light (pedestrian and bicycle lights).
NON_VEHICLE_LIGHT_TYPES = ("1000002", "1000007", "1000013")
# Speed units (ASAM OpenDRIVE e_unitSpeed) and what divides a value into m/s.
SPEED_UNIT_DIVISORS = {"m/s": 1.0, "km/h": 3.6, "mph": 2.237}
# The unit of a speed that names none: a SPEED_UNIT_DIVISORS name, or a number that converts it to m/s. None leaves a
# lane speed without unit unset and reads a road-type one as m/s.
UNITLESS_SPEED_UNIT = None

ORIENTATIONS = ("+", "-", "none")
CONTACT_POINTS = ("start", "end")
# The values the reader understands, per element attribute (a translation must name one of them).
ALLOWED = {
    "object.orientation": ORIENTATIONS,
    "signal.orientation": ORIENTATIONS,
    "signalReference.orientation": ORIENTATIONS,
    "signal.dynamic": ("yes", "no"),
    "connection.contactPoint": CONTACT_POINTS,
    "link.elementType": ("road", "junction"),
    "link.contactPoint": CONTACT_POINTS,
    # the road-mark types plane_group.convert_line_marking reads
    "roadMark.type": (
        "solid", "broken", "solid solid", "broken broken", "curb", "none",
        "solid broken", "solid dashed", "broken solid", "dashed solid",
    ),
    "roadMark.weight": ("standard", "bold"),
    "speed.unit": tuple(SPEED_UNIT_DIVISORS),
}


def speed_factor(unit):
    """
    The factor to m/s that ``unit`` gives as a number (finite and positive), or None.

    :param unit: A speed unit setting, e.g. ``"0.3048"`` for feet per second.
    :return: The factor, or None when ``unit`` is no such number.
    """
    try:
        factor = float(unit)
    except (TypeError, ValueError):
        return None
    return factor if math.isfinite(factor) and factor > 0 else None


def to_metres_per_second(value: float, unit):
    """
    ``value`` in m/s.

    :param value: A speed.
    :param unit: A SPEED_UNIT_DIVISORS name, or a number that converts the unit to m/s.
    :return: The speed in m/s, or None when ``unit`` is neither.
    """
    if unit in SPEED_UNIT_DIVISORS:
        return value / SPEED_UNIT_DIVISORS[unit]
    factor = speed_factor(unit)
    return None if factor is None else value * factor


def record(kind: str, element: str, element_id, value, action, message: str, *args):
    """
    Record one value the reader could not use as written (the conversion report collects these).

    :param kind: Record kind, e.g. ``unknown_value`` (see conversion_report.py).
    :param element: The element attribute, e.g. ``link.contactPoint``.
    :param element_id: Id of the element (or its road).
    :param value: The value as read from the file.
    :param action: What the reader did instead, e.g. ``skipped``.
    :param message: Log message, %-formatted with ``args``.
    """
    logging.getLogger("crdesigner.degradation").warning(
        message,
        *args,
        extra={"kind": kind, "element": element, "id": element_id, "value": value, "action": action},
    )


def translate(element: str, value, element_id=None):
    """
    The name the reader understands for ``value`` of ``element``, recording a translation that applies.

    :param element: The element attribute, e.g. ``object.type``.
    :param value: The value as read from the file.
    :param element_id: Id of the element (or its road), for the record.
    :return: The translated value, or ``value`` itself when no translation applies.
    """
    table = TRANSLATIONS.get(element)
    if not table or value is None or str(value) not in table:
        return value
    translated = table[str(value)]
    record(
        "translated", element, element_id, value, f"read_as_{translated}",
        "%s %s: %r read as %r (vocabulary)", element, element_id, value, translated,
    )
    return translated
