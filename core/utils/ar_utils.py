#!/usr/bin/env python3

"""
utils for both abstraction and refinement
"""

from core.utils.abstraction_utils import choose_weight_func
from core.data_structures.Network import Network

def calculate_weight_of_edge_between_two_part_groups(network: Network,
                                                     group_a: list,
                                                     group_b: list) -> float:
    """
    calculate the weight of the edge
    from the union_node of group @group_a to group @group_b
    :param network: Network
    :param group_a: list of parts in i'th layer
    (each part is a name of node in the original network after preprocessing)
    :param group_b: list of parts in (i+1)'th layer
    (each part is a name of node in the original network after preprocessing)
    :return: weight between union node of @group_a and union node of @group_b
    """
    assert group_a
    assert group_b
    assert group_a[0] in network.orig_name2node_map.keys() or group_b[0] in network.orig_name2node_map.keys()
    a2b_weight = 0
    if group_a[0] not in network.orig_name2node_map.keys() and len(group_a) == 1:
        group_a_list = []
        if len(group_a[0].split("_")) == 3:
            suffixes = ["_inc","_dec"]
            for suffix in suffixes:
                group_a_list.append(group_a[0] + suffix)
                a2b_weight += calculate_weight(network,group_a_list,group_b)
        
        return a2b_weight
    else:
        return calculate_weight(network,group_a,group_b)
def calculate_weight (network:Network, group_a:list, group_b:list)->float:

    # dict from couple of parts, first from group_a and second from consecutive
    # layer's node, to the edge in between
    a_part_dest2edge = {}
    # print("group_a")
    # print(group_a)
    # print(type(group_a))
    for a_part in group_a:
        # print("a_part")
        # print(a_part)
        orig_a_part_node = network.orig_name2node_map[a_part]
        for out_edge in orig_a_part_node.out_edges:
            a_part_dest2edge[(a_part, out_edge.dest)] = out_edge

    # stage 1: calculate dictionary of weights from each part of @group_a to
    # group_b. done by choosing the right max/min function and iterate over all
    # parts of group_b
    a_part2weight = {}  # stores for each part in group_a the max/min value from
    # it to group_b

    a0_node = network.orig_name2node_map[group_a[0]]
    b0_node = network.orig_name2node_map[group_b[0]]
    max_min_func, initial_weight = choose_weight_func(a0_node, b0_node)

    for a_part in group_a:
        for b_part in group_b:
            w1 = a_part2weight.get(a_part, initial_weight)
            w2 = a_part_dest2edge.get((a_part, b_part), None)
            if w2 is not None:
                a_part2weight[a_part] = max_min_func(w1, w2.weight)

    # stage 2: summing the weights from all parts of @group_a to @group_b to be
    # the output weight
    a2b_weight = None
    for a_part in group_a:
        val = a_part2weight.get(a_part, None)
        if val is not None:
            if a2b_weight is None:
                a2b_weight = 0.0
            a2b_weight += val
    return a2b_weight