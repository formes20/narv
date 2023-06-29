from collections import Counter
from itertools import groupby

from core.abstraction.step import union_couple_of_nodes
from core.configuration.consts import VERBOSE, FIRST_ABSTRACT_LAYER
from core.data_structures.Network import Network
from core.pre_process.pre_process import preprocess
from core.utils.debug_utils import debug_print
from core.utils.abstraction_utils import finish_abstraction
from core.visualization.visualize_network import visualize_network
from core.pre_process.pre_process_mine import do_process_before, do_process_after
import copy


def is_completely_abstract_network(network: Network):
    """
    :param network: a given network
    :return: bool, True iff network is completely abstract and no more ar_type unions
    are needed
    """
    for i, layer in enumerate(network.layers[FIRST_ABSTRACT_LAYER:-1]):
        if not is_completely_abstract_layer(layer):
            return False
    return True


def is_completely_abstract_layer(layer):
    ar_types = ["_".join(node.name.split("_")[-2:]) for node in layer.nodes]
    # count number of nodes for each of pos_inc/pos_dec/neg_inc/neg_dec ar_type
    artype_counter = Counter(ar_types)
    # layer is fully abstract iff for each ar_type there is at most one node
    return all(v <= 1 for v in artype_counter.values())


# def abstract_network(
#         network:Network,
#         do_preprocess:bool=True,
#         visualize:bool=False,
#         verbose:bool=VERBOSE
# ) -> Network:
#     """
#     abstract a given network until saturaion
#     :param network: Network represent the network
#     :param do_preprocess: Bool, is pre-proprocess required before abstraction
#     :param visualize: Bool, visualize flag, currently not in use
#     :param verbose: Bool, verbosity flag
#     :return: Network, the full abstraction of @network
#     """
#     if do_preprocess:
#         preprocess(network)
#
#     import time
#     t0 = time.time()
#     #for i,layer in enumerate(network.layers[FIRST_ABSTRACT_LAYER:-1]):
#         # print(f"abstract the {i+FIRST_ABSTRACT_LAYER}'th' layer")
#     for i in range(len(network.layers) - 2, FIRST_ABSTRACT_LAYER - 1, -1):
#         print(f"abstract the {i}'th' layer")
#         layer = network.layers[i]
#         # get the list of every ar_type related nodes
#         ar_types = ["_".join(node.name.split("_")[-2:]) for node in layer.nodes]
#         ar_types_nodes = list(zip(ar_types, layer.nodes))
#         nodes_groupby_ar_type = groupby(
#             sorted(ar_types_nodes, key=lambda x:x[0]),
#             key=lambda x:x[0]
#         )
#
#         # technical note
#         # nodes_groupby_ar_type maps from ar_type to list of (ar_type, node):
#         #   "pos_inc": [("pos_inc", node1), ("pos_inc", node2), ...]
#         # the following loop generate mapping from ar_type to list of nodes:
#         #   "pos_inc": [node1, node2, ...]
#         ar_type2nodes = {}
#         for ar_type, group in nodes_groupby_ar_type:
#             ar_type2nodes[ar_type] = [x[1] for x in group]
#
#         # union ar_type related nodes incrementally (one by one)
#         for ar_type, nodes in ar_type2nodes.items():
#             for j, node in enumerate(nodes):
#                 # print(f"abstract {j}'th node in {i+FIRST_ABSTRACT_LAYER}'th' layer")
#                 # in the 1'st iteration, current node is union_node
#                 if j == 0:
#                     union_node = node
#                     continue
#                 # from 2'nd iteration and on, add current node to union node
#                 union_name = "+".join([union_node.name, node.name])
#                 union_couple_of_nodes(network, union_node, node)
#                 nl_p2u = {union_node.name: union_name, node.name: union_name}
#                 finish_abstraction(network=network,
#                                    next_layer_part2union=nl_p2u,
#                                    verbose=verbose)
#     print(f"full abstraction time={time.time()-t0}")
#     return network


def abstract_network(network: Network, do_preprocess: bool = True, visualize: bool = False, verbose: bool = VERBOSE) -> (Network, Network):
    if VERBOSE:
        debug_print("original net:")
        print(network)
    if do_preprocess:
        preprocess(network)
        # do_process_after(network)
    processed_net = copy.deepcopy(network)
    next_layer_part2union = {}
    for i in range(len(network.layers) - 1, FIRST_ABSTRACT_LAYER - 1, -1):
        layer = network.layers[i]
        next_layer_part2union = layer.abstract(network.name2node_map,
                                               next_layer_part2union)
        # update name2node_map - add new union nodes and remove inner nodes
        # removal precedes addition for equal names case (e.g output layer)
        for part in next_layer_part2union.keys():
            del network.name2node_map[part]
        network.generate_name2node_map()
        # print (i)
        if visualize:
            title = "after layer {} test_abstraction".format(i)
            visualize_network(network_layers=network.layers,
                              title=title,
                              next_layer_part2union=next_layer_part2union,
                              debug=False)
        if verbose:
            debug_print("net after abstract {}'th layer:".format(i))
            print(network)
    finish_abstraction(network, next_layer_part2union, verbose=verbose)
    return network, processed_net
