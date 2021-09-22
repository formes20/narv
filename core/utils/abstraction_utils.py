#!/usr/bin/env python3

from typing import Dict, Tuple, Callable
from core.configuration.consts import (
    INT_MAX, INT_MIN, VERBOSE, FIRST_ABSTRACT_LAYER
)
from core.utils.debug_utils import debug_print
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode


def choose_weight_func(node:ARNode, dest:ARNode) \
        -> Tuple[Callable[[float], float], int]:
    """
    :param node: ARNode, current node
    :param dest: ARNode, destination node
    :return: the appropriate func to choose weight by, and its initial weight
    func is min/max, initial weight is sys.maxint/-sys.maxint (accordingly)
    """
    d = {
        ("inc"): (max, INT_MIN),
        ("dec"): (min, INT_MAX)
    }
    #positivity = "neg" if ("neg" in node.name) else "pos"
    return d[(dest.ar_type)]


def finish_abstraction(network, next_layer_part2union:Dict,
                       verbose:bool=VERBOSE) -> None:
    # fix input layer edges dest names (to fit layer 1 test_abstraction results)
    fix_prev_layer_out_edges_dests(network=network,
                                   prev_layer_index=FIRST_ABSTRACT_LAYER - 1,
                                   updated_names=next_layer_part2union)
    if verbose:
        debug_print("net after fix_prev_layer_out_edges_dests")
        print(network)
    fix_prev_layer_min_max_edges(network=network,
                                 prev_layer_index=FIRST_ABSTRACT_LAYER - 1)
    if verbose:
        debug_print("net after fix_prev_layer_min_max_edges")
        print(network)
    fix_cur_layer_in_edges(network=network,
                           layer_index=FIRST_ABSTRACT_LAYER)
    network.biases = network.generate_biases()
    network.weights = network.generate_weights()
    if verbose:
        debug_print("net after fix_cur_layer_in_edges")
        print(network)
    if verbose:
        debug_print("finish_abstraction finished")


def fix_prev_layer_out_edges_dests(network, prev_layer_index:int,
                                   updated_names:Dict) -> None:
    """
    fix the dest names of out edges of nodes in a layer to the names of the
    updated node names in the next layer
    used to update previous layer edges dest names after changing a layer
    (after both test_abstraction and test_refinement)
    @layer index index of layer in self.layers
    @updated_names dictionary from previous name to current name
    """
    layer = network.layers[prev_layer_index]
    for node in layer.nodes:
        for out_edge in node.out_edges:
            out_edge.dest = updated_names.get(out_edge.dest, out_edge.dest)

def fix_prev_layer_min_max_edges(network, prev_layer_index:int) -> None:
    prev_layer = network.layers[prev_layer_index]
    cur_layer = network.layers[prev_layer_index + 1]
    node2dest_edges = {node.name: {} for node in prev_layer.nodes}

    # calculate min/max edges from prev layer's nodes to their dest nodes
    for node in prev_layer.nodes:
        for out_edge in node.out_edges:
            dest = network.name2node_map[out_edge.dest]
            max_min_func, cur_weight = choose_weight_func(node, dest)
            node2dest_edges[node.name].setdefault(dest.name, cur_weight)
            cur_weight = node2dest_edges[node.name][dest.name]
            node2dest_edges[node.name][dest.name] = \
                max_min_func(cur_weight, out_edge.weight)
    # append min/max edges only as new edges from node to each dest
    # the strange names a_node,a_dest were chosen because for some reason
    # the intepreter confused when using the name "node"
    for a_node in node2dest_edges:
        for a_dest in node2dest_edges[a_node]:
            new_edge = Edge(a_node, a_dest, node2dest_edges[a_node][a_dest])
            dest_node = network.name2node_map[a_dest]
            try:
                dest_node.new_in_edges.append(new_edge)
            except AttributeError:
                dest_node.new_in_edges = [new_edge]
            node = network.name2node_map[a_node]
            try:
                node.new_out_edges.append(new_edge)
            except AttributeError:
                node.new_out_edges = [new_edge]

    # update edges to new edges
    for node in prev_layer.nodes:
        if hasattr(node, "new_out_edges"):
            node.out_edges = node.new_out_edges
            del node.new_out_edges
    for node in cur_layer.nodes:
        if hasattr(node, "new_in_edges"):
            node.in_edges = node.new_in_edges
            del node.new_in_edges


def fix_cur_layer_in_edges(network, layer_index:int) -> None:
    assert layer_index != 0
    layer = network.layers[layer_index]
    for node in layer.nodes:
        node.in_edges = []
    prev_layer = network.layers[layer_index - 1]
    for node in prev_layer.nodes:
        for out_edge in node.out_edges:
            # out_edge is an in_edge of its dest node
            network.name2node_map[out_edge.dest].in_edges.append(out_edge)
