"""Builder for small synthetic OpenDRIVE maps used as IAI test fixtures.

Every IAI test builds its XODR here and writes it to ``tmp_path``; corpus maps are never
committed. Geometry is straight lines only, which keeps expected values easy to compute.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

DEFAULT_GEO_REFERENCE = "+proj=tmerc +lat_0=49.0 +lon_0=8.0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84"


@dataclasses.dataclass(slots=True)
class LaneSpec:
    """One lane; positive ids are left of the reference line, negative ids right."""

    id: int
    type: str = "driving"
    width: float = 3.5
    road_mark: str = "solid"
    predecessor: Optional[int] = None
    successor: Optional[int] = None


@dataclasses.dataclass(slots=True)
class LaneSectionSpec:
    s: float
    lanes: list[LaneSpec]
    center_road_mark: str = "broken"


@dataclasses.dataclass(slots=True)
class LinkSpec:
    element_type: str  # "road" or "junction"
    element_id: str
    contact_point: Optional[str] = "start"


@dataclasses.dataclass(slots=True)
class SignalSpec:
    id: str
    s: float
    t: float = -5.0
    type: str = "1000001"
    subtype: str = "-1"
    dynamic: Optional[str] = "yes"
    orientation: Optional[str] = "+"
    country: str = "DEU"
    name: str = ""
    validity: Optional[tuple[Optional[int], Optional[int]]] = None  # (fromLane, toLane)


@dataclasses.dataclass(slots=True)
class SignalReferenceSpec:
    id: str
    s: float
    t: float = -5.0
    orientation: str = "+"


@dataclasses.dataclass(slots=True)
class ObjectSpec:
    id: str
    s: float
    t: float = 0.0
    type: str = "roadMark"
    name: str = ""
    orientation: str = "+"
    width: float = 3.5
    length: float = 0.5


@dataclasses.dataclass(slots=True)
class RoadSpec:
    """A straight road from ``(x, y)`` along heading ``hdg``."""

    id: str
    length: float
    lane_sections: list[LaneSectionSpec]
    x: float = 0.0
    y: float = 0.0
    hdg: float = 0.0
    junction: str = "-1"
    predecessor: Optional[LinkSpec] = None
    successor: Optional[LinkSpec] = None
    elevation: list[tuple[float, float, float, float, float]] = dataclasses.field(
        default_factory=list
    )  # (s, a, b, c, d)
    rule: Optional[str] = None  # "RHT" / "LHT"
    signals: list[SignalSpec] = dataclasses.field(default_factory=list)
    signal_references: list[SignalReferenceSpec] = dataclasses.field(default_factory=list)
    objects: list[ObjectSpec] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class ConnectionSpec:
    id: str
    incoming_road: str
    connecting_road: str
    contact_point: str = "start"
    lane_links: list[tuple[int, int]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class JunctionSpec:
    id: str
    connections: list[ConnectionSpec]
    controllers: list[tuple[str, Optional[int]]] = dataclasses.field(
        default_factory=list
    )  # (controller id, sequence)


@dataclasses.dataclass(slots=True)
class ControllerSpec:
    id: str
    signal_ids: list[str]
    name: str = ""
    sequence: Optional[int] = None


def standard_lanes(
    *, n_left: int, n_right: int, width: float = 3.5, road_mark: str = "solid"
) -> list[LaneSpec]:
    """Driving lanes ``n_left..1`` and ``-1..-n_right`` with equal widths and marks."""
    ids = list(range(n_left, 0, -1)) + list(range(-1, -n_right - 1, -1))
    return [LaneSpec(id=i, width=width, road_mark=road_mark) for i in ids]


def straight_road(
    *,
    id: str,
    length: float,
    x: float = 0.0,
    y: float = 0.0,
    hdg: float = 0.0,
    n_left: int = 1,
    n_right: int = 1,
    **kwargs,
) -> RoadSpec:
    """Convenience: a straight road with one lane section of standard lanes."""
    return RoadSpec(
        id=id,
        length=length,
        x=x,
        y=y,
        hdg=hdg,
        lane_sections=[LaneSectionSpec(s=0.0, lanes=standard_lanes(n_left=n_left, n_right=n_right))],
        **kwargs,
    )


def _num(*, value: float) -> str:
    return repr(float(value))


def _attrs(**attrs: Optional[object]) -> str:
    return " ".join(f'{k}="{v}"' for k, v in attrs.items() if v is not None)


def _link(*, tag: str, link: Optional[LinkSpec]) -> str:
    if link is None:
        return ""
    attrs = _attrs(
        elementType=link.element_type, elementId=link.element_id, contactPoint=link.contact_point
    )
    return f"<{tag} {attrs}/>"


def _lane(*, lane: LaneSpec) -> str:
    links = ""
    if lane.predecessor is not None:
        links += f'<predecessor id="{lane.predecessor}"/>'
    if lane.successor is not None:
        links += f'<successor id="{lane.successor}"/>'
    return (
        f'<lane id="{lane.id}" type="{lane.type}" level="false"><link>{links}</link>'
        f'<width sOffset="0.0" a="{_num(value=lane.width)}" b="0.0" c="0.0" d="0.0"/>'
        f'<roadMark sOffset="0.0" type="{lane.road_mark}" weight="standard" color="standard" '
        f'width="0.13"/></lane>'
    )


def _lane_section(*, section: LaneSectionSpec) -> str:
    left = sorted((ln for ln in section.lanes if ln.id > 0), key=lambda ln: -ln.id)
    right = sorted((ln for ln in section.lanes if ln.id < 0), key=lambda ln: -ln.id)
    out = f'<laneSection s="{_num(value=section.s)}">'
    if left:
        out += "<left>" + "".join(_lane(lane=ln) for ln in left) + "</left>"
    out += (
        f'<center><lane id="0" type="none" level="false"><link/>'
        f'<roadMark sOffset="0.0" type="{section.center_road_mark}" weight="standard" '
        f'color="standard" width="0.13"/></lane></center>'
    )
    if right:
        out += "<right>" + "".join(_lane(lane=ln) for ln in right) + "</right>"
    return out + "</laneSection>"


def _signal(*, signal: SignalSpec) -> str:
    validity = ""
    if signal.validity is not None:
        validity = f"<validity {_attrs(fromLane=signal.validity[0], toLane=signal.validity[1])}/>"
    attrs = _attrs(
        s=_num(value=signal.s),
        t=_num(value=signal.t),
        id=signal.id,
        name=signal.name,
        dynamic=signal.dynamic,
        orientation=signal.orientation,
        zOffset="0.0",
        country=signal.country,
        type=signal.type,
        subtype=signal.subtype,
        value="-1",
        height="1.0",
        width="0.5",
    )
    return f"<signal {attrs}>{validity}</signal>"


def _road(*, road: RoadSpec) -> str:
    attrs = _attrs(
        name=f"road{road.id}",
        length=_num(value=road.length),
        id=road.id,
        junction=road.junction,
        rule=road.rule,
    )
    link = _link(tag="predecessor", link=road.predecessor) + _link(
        tag="successor", link=road.successor
    )
    geometry = (
        f'<geometry s="0.0" x="{_num(value=road.x)}" y="{_num(value=road.y)}" '
        f'hdg="{_num(value=road.hdg)}" length="{_num(value=road.length)}"><line/></geometry>'
    )
    elevations = "".join(
        f'<elevation s="{_num(value=s)}" a="{_num(value=a)}" b="{_num(value=b)}" '
        f'c="{_num(value=c)}" d="{_num(value=d)}"/>'
        for s, a, b, c, d in road.elevation
    )
    signals = "".join(_signal(signal=sig) for sig in road.signals) + "".join(
        f'<signalReference {_attrs(s=_num(value=ref.s), t=_num(value=ref.t), id=ref.id, orientation=ref.orientation)}/>'
        for ref in road.signal_references
    )
    objects = "".join(
        f"<object {_attrs(id=o.id, s=_num(value=o.s), t=_num(value=o.t), type=o.type, name=o.name, orientation=o.orientation, zOffset='0.0', length=_num(value=o.length), width=_num(value=o.width), height='0.0', hdg='0.0', pitch='0.0', roll='0.0')}/>"
        for o in road.objects
    )
    sections = "".join(_lane_section(section=sec) for sec in road.lane_sections)
    return (
        f"<road {attrs}><link>{link}</link><type s=\"0.0\" type=\"town\"/>"
        f"<planView>{geometry}</planView>"
        f"<elevationProfile>{elevations}</elevationProfile><lateralProfile/>"
        f"<lanes>{sections}</lanes><objects>{objects}</objects><signals>{signals}</signals></road>"
    )


def _junction(*, junction: JunctionSpec) -> str:
    out = f'<junction id="{junction.id}" name="junction{junction.id}">'
    for conn in junction.connections:
        links = "".join(f'<laneLink from="{a}" to="{b}"/>' for a, b in conn.lane_links)
        out += (
            f'<connection id="{conn.id}" incomingRoad="{conn.incoming_road}" '
            f'connectingRoad="{conn.connecting_road}" contactPoint="{conn.contact_point}">'
            f"{links}</connection>"
        )
    for controller_id, sequence in junction.controllers:
        out += f"<controller {_attrs(id=controller_id, sequence=sequence)}/>"
    return out + "</junction>"


def _controller(*, controller: ControllerSpec) -> str:
    attrs = _attrs(id=controller.id, name=controller.name, sequence=controller.sequence)
    controls = "".join(f'<control signalId="{sid}" type="0"/>' for sid in controller.signal_ids)
    return f"<controller {attrs}>{controls}</controller>"


def build_xodr(
    *,
    roads: list[RoadSpec],
    junctions: tuple[JunctionSpec, ...] | list[JunctionSpec] = (),
    controllers: tuple[ControllerSpec, ...] | list[ControllerSpec] = (),
    geo_reference: Optional[str] = DEFAULT_GEO_REFERENCE,
    offset: Optional[tuple[float, float, float, float]] = None,
    vendor: str = "synthetic",
    rev_minor: int = 4,
    encoding: str = "UTF-8",
) -> str:
    """Return OpenDRIVE XML text for the given roads, junctions and controllers.

    Args:
        roads: Roads to emit.
        junctions: Junctions to emit.
        controllers: Root-level signal controllers to emit.
        geo_reference: PROJ string for ``<geoReference>``; ``None`` omits the element.
        offset: Header ``<offset>`` as ``(x, y, z, hdg)``; ``None`` omits it.
        vendor: Header ``vendor`` attribute.
        rev_minor: Header ``revMinor``.
        encoding: Value written in the XML declaration (the caller encodes the text).

    Returns:
        The XODR document as a string.
    """
    header = f'<header revMajor="1" revMinor="{rev_minor}" name="synthetic" version="1.0" vendor="{vendor}">'
    if geo_reference is not None:
        header += f"<geoReference><![CDATA[{geo_reference}]]></geoReference>"
    if offset is not None:
        x, y, z, hdg = offset
        header += f'<offset x="{_num(value=x)}" y="{_num(value=y)}" z="{_num(value=z)}" hdg="{_num(value=hdg)}"/>'
    header += "</header>"
    body = (
        "".join(_road(road=r) for r in roads)
        + "".join(_controller(controller=c) for c in controllers)
        + "".join(_junction(junction=j) for j in junctions)
    )
    return f'<?xml version="1.0" encoding="{encoding}"?>\n<OpenDRIVE>{header}{body}</OpenDRIVE>\n'


def junction_map(
    *,
    with_signals: bool = True,
    with_controllers: bool = True,
    approach_length: float = 50.0,
    rule: Optional[str] = None,
) -> tuple[list[RoadSpec], list[JunctionSpec], list[ControllerSpec]]:
    """A straight-through junction: road 1 -> connecting road 2 (junction 100) -> road 3.

    With signals, connecting road 2 carries light ``11`` at its entry for lane -1 and light ``12``
    at its far end for lane 1; each light gets its own controller with a sequence.
    """
    road1 = straight_road(
        id="1",
        length=approach_length,
        successor=LinkSpec(element_type="junction", element_id="100", contact_point=None),
        rule=rule,
    )
    road2 = straight_road(
        id="2",
        length=20.0,
        x=approach_length,
        junction="100",
        predecessor=LinkSpec(element_type="road", element_id="1", contact_point="end"),
        successor=LinkSpec(element_type="road", element_id="3", contact_point="start"),
        rule=rule,
    )
    road3 = straight_road(
        id="3",
        length=approach_length,
        x=approach_length + 20.0,
        predecessor=LinkSpec(element_type="junction", element_id="100", contact_point=None),
        rule=rule,
    )
    for lane in road2.lane_sections[0].lanes:
        lane.predecessor = lane.id
        lane.successor = lane.id
    if with_signals:
        # CARLA-style placement: on the connecting road at its entry, scoped by <validity>.
        road2.signals.append(SignalSpec(id="11", s=0.5, t=0.0, orientation="none", validity=(-1, -1)))
        road2.signals.append(SignalSpec(id="12", s=19.5, t=0.0, orientation="none", validity=(1, 1)))
    connections = [
        ConnectionSpec(id="0", incoming_road="1", connecting_road="2", contact_point="start", lane_links=[(-1, -1)]),
        ConnectionSpec(id="1", incoming_road="3", connecting_road="2", contact_point="end", lane_links=[(1, 1)]),
    ]
    controllers: list[ControllerSpec] = []
    junction = JunctionSpec(id="100", connections=connections)
    if with_signals and with_controllers:
        controllers = [
            ControllerSpec(id="1001", signal_ids=["11"], sequence=0),
            ControllerSpec(id="1002", signal_ids=["12"], sequence=1),
        ]
        junction.controllers = [("1001", 0), ("1002", 1)]
    return [road1, road2, road3], [junction], controllers
