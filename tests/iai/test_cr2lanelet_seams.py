"""Predecessor/successor seams share node ids even when a boundary reuses a neighbour's way."""

import numpy as np
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork

from crdesigner.map_conversion.lanelet2.cr2lanelet import CR2LaneletConverter


class _DegreesTransformer:
    """Local metres to (lat, lon) at roughly 111 km per degree."""

    def transform(self, x: float, y: float) -> tuple[float, float]:
        return y / 111_000.0, x / 111_000.0


def _lanelet(*, lanelet_id: int, left: list, right: list, **links) -> Lanelet:
    left, right = np.array(left, dtype=float), np.array(right, dtype=float)
    return Lanelet(
        left_vertices=left,
        center_vertices=(left + right) / 2,
        right_vertices=right,
        lanelet_id=lanelet_id,
        **links,
    )


def test_successor_reusing_an_opposite_neighbours_way_shares_the_seam_nodes():
    # 1 -> 3 eastbound; 2 runs westbound left of 3 and is converted before it, so 3's left
    # boundary reuses 2's way, whose end node sits where 1's last left node already is.
    network = LaneletNetwork()
    network.add_lanelet(_lanelet(lanelet_id=1, left=[[0, 3], [10, 3]], right=[[0, 0], [10, 0]], successor=[3]))
    network.add_lanelet(
        _lanelet(
            lanelet_id=2,
            left=[[20, 3], [10, 3]],
            right=[[20, 6], [10, 6]],
            adjacent_left=3,
            adjacent_left_same_direction=False,
        )
    )
    network.add_lanelet(
        _lanelet(
            lanelet_id=3,
            left=[[10, 3], [20, 3]],
            right=[[10, 0], [20, 0]],
            predecessor=[1],
            adjacent_left=2,
            adjacent_left_same_direction=False,
        )
    )
    converter = CR2LaneletConverter()
    converter.convert_lanelet_network(network, transformer=_DegreesTransformer())
    ways = converter.osm.ways

    assert converter.left_ways[3] == converter.left_ways[2]  # the case under test: a reused way
    shared_way = ways[converter.left_ways[3]].nodes
    assert ways[converter.left_ways[1]].nodes[-1] == shared_way[-1]  # reversed: 3 starts at its end
    assert ways[converter.right_ways[1]].nodes[-1] == ways[converter.right_ways[3]].nodes[0]
    assert set(converter.osm.nodes) >= {n for way in ways.values() for n in way.nodes}


def test_linked_lanelets_that_do_not_meet_keep_their_own_end_nodes():
    # 2 is linked as 1's successor but starts 10 m further on: sharing 1's end nodes would
    # stretch 2's boundaries back across the gap.
    network = LaneletNetwork()
    network.add_lanelet(_lanelet(lanelet_id=1, left=[[0, 3], [10, 3]], right=[[0, 0], [10, 0]], successor=[2]))
    network.add_lanelet(_lanelet(lanelet_id=2, left=[[20, 3], [30, 3]], right=[[20, 0], [30, 0]], predecessor=[1]))
    converter = CR2LaneletConverter()
    converter.convert_lanelet_network(network, transformer=_DegreesTransformer())
    ways, nodes = converter.osm.ways, converter.osm.nodes

    assert ways[converter.left_ways[1]].nodes[-1] != ways[converter.left_ways[2]].nodes[0]
    first_right = nodes[ways[converter.right_ways[2]].nodes[0]]
    assert abs(float(first_right.lon) * 111_000.0 - 20.0) < 0.01  # 2 still starts at x = 20 m
