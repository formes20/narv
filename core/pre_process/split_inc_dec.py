from core.configuration.consts import (
    VERBOSE, FIRST_INC_DEC_LAYER
)
from core.data_structures.Network import Network
from core.utils.debug_utils import debug_print
from core.data_structures.Edge import Edge


# def adjust_layer_after_split_inc_dec(network:Network,
#                                      layer_index:int=FIRST_INC_DEC_LAYER):
#     cur_layer = network.layers[layer_index]
#     next_layer = network.layers[layer_index + 1]
#     count = 0
#     for cur_node in cur_layer.nodes:
#         cur_node.new_out_edges = []
#     for next_node in next_layer.nodes:
#         next_node.new_in_edges = []
#         parts = next_node.name.split("+")
#         if len(parts) > 1:
#             print(parts[1])
#         for part in parts:
#             for cur_node in cur_layer.nodes:
#                 for out_edge in cur_node.out_edges:
#                     for suffix in ["_inc", "_dec"]:
#                         if (out_edge.dest + suffix) == part:
#                             weight = out_edge.weight
#                             edge = Edge(cur_node.name,
#                                         next_node.name,
#                                         weight)
#                             cur_node.new_out_edges.append(edge)
#                             next_node.new_in_edges.append(edge)
#                             count +=1 
#         next_node.in_edges = next_node.new_in_edges
#         del next_node.new_in_edges
#     for node in cur_layer.nodes:
#         node.out_edges = node.new_out_edges
#         del node.new_out_edges
#     print(count)

def adjust_layer_after_split_inc_dec(network: Network,
                                     layer_index: int = FIRST_INC_DEC_LAYER):
    cur_layer = network.layers[layer_index]
    next_layer = network.layers[layer_index + 1]
    count = 0
    for node in cur_layer.nodes:
        node.new_out_edges = []
    for next_node in next_layer.nodes:
        next_node.in_edges = []
    for node in cur_layer.nodes:
        for out_edge in node.out_edges:
            for suffix in ["_inc", "_dec"]:
                linked_node = network.name2node_map.get(out_edge.dest + suffix, None)
                if linked_node:
                    weight = out_edge.weight
                    edge = Edge(node.name, linked_node.name, weight)
                    node.new_out_edges.append(edge)
                    linked_node.in_edges.append(edge)
                    count += 1
        node.out_edges = node.new_out_edges
    print(count)


def preprocess_split_inc_dec(network: Network) -> None:
    """
    split net nodes to increasing/decreasing nodes
    preprocess all layers except input layer (from last to first)
    """
    if VERBOSE:
        debug_print("preprocess_split_inc_dec()")
    for i in range(len(network.layers) - 1, FIRST_INC_DEC_LAYER, -1):
        network.layers[i].split_inc_dec(network.name2node_map)
    network.generate_name2node_map()
    adjust_layer_after_split_inc_dec(network, layer_index=FIRST_INC_DEC_LAYER)
    if VERBOSE:
        debug_print("after preprocess_split_inc_dec()")
        print(network)
