import logging

# Names an OpenDRIVE file may use, translated into names the reader understands, per element attribute,
# e.g. {"object.orientation": {"undefined": "none"}}. Empty unless a vocabulary is installed (IAI).
TRANSLATIONS = {}


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
    logging.getLogger("crdesigner.degradation").warning(
        "%s %s: %r read as %r (vocabulary)",
        element,
        element_id,
        value,
        translated,
        extra={"kind": "translated", "element": element, "id": element_id, "value": value,
               "action": f"read_as_{translated}"},
    )
    return translated
