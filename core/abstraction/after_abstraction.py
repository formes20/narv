from core.data_structures.Network import Network
from core.data_structures.ARNode import ARNode
from core.pre_process.pre_process import preprocess
from core.utils.debug_utils import debug_print
from core.utils.abstraction_utils import finish_abstraction
from core.visualization.visualize_network import visualize_network
from core.pre_process.pre_process_mine import do_process_before, do_process_after
# from core.utils.alg2_utils import has_violation, get_limited_random_inputs
from core.abstraction.step import union_couple_of_nodes
from core.utils.abstraction_utils import finish_abstraction
from core.configuration.consts import (
    VERBOSE, FIRST_ABSTRACT_LAYER, INT_MIN, INT_MAX
)
from core.data_structures.Edge import Edge
from core.utils.activation_functions import relu
from core.utils.verification_properties_utils import TEST_PROPERTY_ACAS
from core.utils.alg2_utils import has_violation, get_limited_random_inputs
from core.utils.cal_contribution import calculate_contribution
from core.utils.cal_layer_average_in_weight import cal_average_in_weight
from core.utils.combine_influ_of2nodes import combin_influnce
from core.data_structures.Abstract_action import Abstract_action
from core.utils.find_relation import find_relation
import copy
import time


def node_reunion(network: Network, i: int, j: int):
    node_name = "x_" + str(i) + "_" + str(j)
    suffixes = ["_inc", "_dec"]
    four_nodes = []
    for suffix in suffixes:
        four_nodes.append(network.name2node_map[node_name + suffix])
    union_node = ARNode(name="x_" + str(i) + "_" + str(j),
                        ar_type="",
                        activation_func=relu,
                        in_edges=[],
                        out_edges=[],
                        bias=four_nodes[0].bias
                        )
    for in_edge in four_nodes[0].in_edges:
        edge = Edge(src=in_edge.src, dest=union_node.name, weight=in_edge.weight)
        union_node.in_edges.append(edge)
        src_node = network.name2node_map[in_edge.src]
        src_node.out_edges.append(edge)

    node_a = four_nodes[0]
    node_b = four_nodes[1]

    out_edges_a = sorted(node_a.out_edges, key=lambda x: x.dest)
    out_edges_b = sorted(node_b.out_edges, key=lambda x: x.dest)

    assert len(out_edges_b) == len(out_edges_a)
    for k in range(len(out_edges_a)):
        out_edge_a = out_edges_a[k]
        out_edge_b = out_edges_b[k]
        # print(out_edge_b.dest+"vs"+out_edge_a.dest)
        assert out_edge_a.dest == out_edge_b.dest
        edge = Edge(src=union_node.name, dest=out_edge_a.dest, weight=out_edge_a.weight + out_edge_b.weight)
        union_node.out_edges.append(edge)
        dest_node = network.name2node_map[out_edge_a.dest]
        dest_node.in_edges.append(edge)

    assert len(union_node.out_edges) == len(network.layers[i + 1].nodes)
    for node in four_nodes:
        network.remove_node(node, i)

    network.layers[i].nodes.append(union_node)


# def node_reunion_inc(network: Network,i: int,j: int):
#     node_name = "x_"+str(i)+"_"+str(j)
#     suffixes = ["_pos_inc","_neg_inc"]
#     two_nodes = []
#     for suffix in suffixes:
#         two_nodes.append(network.name2node_map[node_name + suffix])
#     union_node = ARNode(name="x_"+str(i)+"_"+str(j)+"_inc",
#                         ar_type="",
#                         activation_func=relu,
#                         in_edges=[],
#                         out_edges=[],
#                         bias=two_nodes[0].bias
#                         )
#     for in_edge in two_nodes[0].in_edges:
#         edge = Edge(src=in_edge.src, dest=union_node.name, weight=in_edge.weight)
#         union_node.in_edges.append(edge)
#         src_node = network.name2node_map[in_edge.src]
#         src_node.out_edges.append(edge)

#     node_a = two_nodes[0]
#     node_b = two_nodes[1]
#     out_edges_a = sorted(node_a.out_edges, key=lambda x: x.dest)
#     out_edges_b = sorted(node_b.out_edges, key=lambda x: x.dest)
#     assert len(out_edges_b) == len(out_edges_a)
#     for k in range(len(out_edges_a)):
#         out_edge_a = out_edges_a[k]
#         out_edge_b = out_edges_b[k]
#         #print(out_edge_b.dest+"vs"+out_edge_a.dest)
#         assert out_edge_a.dest == out_edge_b.dest
#         edge = Edge(src=union_node.name, dest=out_edge_a.dest, weight=out_edge_a.weight + out_edge_b.weight)
#         union_node.out_edges.append(edge)
#         dest_node = network.name2node_map[out_edge_a.dest]
#         dest_node.in_edges.append(edge)
#     assert len(union_node.out_edges) == len(network.layers[i+1].nodes)   
#     for node in two_nodes:
#         network.remove_node(node,i)

#     network.layers[i].nodes.append(union_node)

# def node_reunion_dec(network: Network,i: int,j: int):
#     node_name = "x_"+str(i)+"_"+str(j)
#     suffixes = ["_neg_dec","_pos_dec"]
#     two_nodes = []
#     for suffix in suffixes:
#         two_nodes.append(network.name2node_map[node_name + suffix])
#     union_node = ARNode(name="x_"+str(i)+"_"+str(j)+"_dec",
#                         ar_type="",
#                         activation_func=relu,
#                         in_edges=[],
#                         out_edges=[],
#                         bias=two_nodes[0].bias
#                         )
#     for in_edge in two_nodes[0].in_edges:
#         edge = Edge(src=in_edge.src, dest=union_node.name, weight=in_edge.weight)
#         union_node.in_edges.append(edge)
#         src_node = network.name2node_map[in_edge.src]
#         src_node.out_edges.append(edge)

#         node_a = two_nodes[0]
#     node_b = two_nodes[1]
#     out_edges_a = sorted(node_a.out_edges, key=lambda x: x.dest)
#     out_edges_b = sorted(node_b.out_edges, key=lambda x: x.dest)
#     assert len(out_edges_b) == len(out_edges_a)
#     for k in range(len(out_edges_a)):
#         out_edge_a = out_edges_a[k]
#         out_edge_b = out_edges_b[k]
#         #print(out_edge_b.dest+"vs"+out_edge_a.dest)
#         assert out_edge_a.dest == out_edge_b.dest
#         edge = Edge(src=union_node.name, dest=out_edge_a.dest, weight=out_edge_a.weight + out_edge_b.weight)
#         union_node.out_edges.append(edge)
#         dest_node = network.name2node_map[out_edge_a.dest]
#         dest_node.in_edges.append(edge)

#     assert len(union_node.out_edges) == len(network.layers[i+1].nodes)   
#     for node in two_nodes:
#         network.remove_node(node,i)

#     network.layers[i].nodes.append(union_node)

def after_abstraction(network: Network, free_layer: int, ACAS: bool):
    count = 0
    network.generate_name2node_map()
    if ACAS:
        NUMBER_OF_LAYER = 50
    else:
        NUMBER_OF_LAYER = 512
    for i in range(FIRST_ABSTRACT_LAYER, len(network.layers) - 2):
        for j in range(NUMBER_OF_LAYER):
            node_name = "x_" + str(i) + "_" + str(j)
            free = True
            if i <= free_layer:
                suffixes = ["_inc", "_dec"]
                for suffix in suffixes:
                    if node_name + suffix not in network.name2node_map.keys():
                        free = False
                    elif network.name2node_map[node_name + suffix].deleted:
                        free = False
                if free:
                    node_reunion(network, i, j)
                    print(node_name + "  is free")
                    network.generate_name2node_map()
                    count += 1
    print("________________________________________AFTER ABSTRACTION")
    print(count)
