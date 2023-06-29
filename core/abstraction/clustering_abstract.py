import copy
import random
from core.data_structures.Network import Network
from core.utils.alg2_utils import has_violation, get_limited_random_inputs
from core.pre_process.pre_process import preprocess
from core.abstraction.step import union_couple_of_nodes
from core.utils.abstraction_utils import finish_abstraction
from core.configuration.consts import (
    VERBOSE, FIRST_ABSTRACT_LAYER, INT_MIN
)
from core.utils.cluster import kMeans


def heuristic_abstract_clustering(network: Network, test_property: dict,
                                  do_preprocess: bool = True, sequence_length: int = 50,
                                  verbose: bool = VERBOSE) -> Network:
    """
    the main idea:
    find the best network to start query.
    if we know that specific abstract network is SAT wrt some input, we should
    avoid query its abstractions.
    Start with full abstraction and check if some point is SAT:
    - if not, exponentialy increase number of clusters and try again until
    convergence to original network.
    - if SAT is found, return this network
    method:
    abstract network using clustering, exponentialy increase number of clusters
    in consequtive iterations, (so 1'st abstract network is fully abstracted,
    last abstract network is the original network)

    abstract a network until @test_property holds in resulted abstract network
    the abstraction is done using clustering with kmeans, each time k increase,
    hence network increase from full abstract to original network


    but has equal number in every layer (same number of pairs are grouped in
    each layer)
    :param network: Network represent the network
    :param test_property: Dict represents the property to check
    :param do_preprocess: Bool, is pre-process required before abstraction
    :param sequence_length: Int, number of abstraction steps in each sequence
    :param verbose: Bool, verbosity flag
    :return: Network, abstract network according to alg 2 in the paper
    """
    if do_preprocess:
        preprocess(network)
    orig_net = copy.deepcopy(network)

    input_size = len(network.layers[0].nodes)
    # generate random inputs in the bound of the test property
    random_inputs = get_limited_random_inputs(
        input_size=input_size,
        test_property=test_property
    )
    # map from layer index to number of clusters in layer
    layer_index2number_of_clusters = {}
    # map from layer index and ar_type to number of clusters of ar_type in layer
    layer_index_ar_type2number_of_clusters = {}
    # while no violation occurs, continue to abstract
    while not has_violation(network, test_property, random_inputs):
        # check that the network is not fully abstracted
        if all(n <= 4 for n in layer_index2number_of_clusters.values()):
            # if all(len(layer.nodes)<=4 for layer in network.layers[1:-1]):
            break

        # in every layer, generate clusters
        for i, layer in enumerate(network.layers[FIRST_ABSTRACT_LAYER:-1]):
            layer_index = i + 1
            prev_layer = network.layers[layer_index - 1]
            # get for each type the relevant nodes
            ar_type2nodes = layer.get_ar_type2nodes()
            for ar_type, nodes in ar_type2nodes.items():
                # calc the maximal diffs between edges of any couple of nodes
                for n1, n2 in layer_couples:
                    max_diff_n1_n2 = INT_MIN
                    for prev_node in prev_layer.nodes:
                        in_edge_n1 = nodes2edge_between_map.get((prev_node.name, n1.name), None)
                        a = 0 if in_edge_n1 is None else in_edge_n1.weight
                        in_edge_n2 = nodes2edge_between_map.get((prev_node.name, n2.name), None)
                        b = 0 if in_edge_n2 is None else in_edge_n2.weight
                        if abs(a - b) > max_diff_n1_n2:
                            max_diff_n1_n2 = abs(a - b)
                    union_pairs.append(([n1, n2], max_diff_n1_n2))

        # abstract constant number of random nodes
        layer_couples = []
        for i, layer in enumerate(network.layers[FIRST_ABSTRACT_LAYER:-1]):
            layer_couples.extend(layer.get_couples_of_same_ar_type())
        random.shuffle(layer_couples)

        # union these couples
        best_sequence_pairs = layer_couples[:sequence_length]
        for pair in best_sequence_pairs:
            if pair[0].name not in network.name2node_map or pair[1].name not in network.name2node_map:
                continue
            union_name = "+".join([pair[0].name, pair[1].name])
            union_couple_of_nodes(network, pair[0], pair[1])
            nl_p2u = {pair[0].name: union_name, pair[1].name: union_name}
            finish_abstraction(network=network,
                               next_layer_part2union=nl_p2u,
                               verbose=verbose)
    return network


def heuristic_abstract_alg2(network: Network, test_property: dict,
                            do_preprocess: bool = True, sequence_length: int = 50,
                            verbose: bool = VERBOSE) -> Network:
    """
    abstract a given network according to alg 2 method in the paper
    the abstraction is done on @sequence_length pairs that are chosen a.t alg2
    in the paper, those with the biggest difference which is the smallest
    :param network: Network represent the network
    :param test_property: Dict represents the property to check
    :param do_preprocess: Bool, is pre-proprocess required before abstraction
    :param sequence_length: Int, number of abstraction steps in each sequence
    :param verbose: Bool, verbosity flag
    :return: Network, abstract network according to alg 2 in the paper
    """
    if do_preprocess:
        preprocess(network)
    input_size = len(network.layers[0].nodes)
    # generate random inputs in the bound of the test property
    random_inputs = get_limited_random_inputs(
        input_size=input_size,
        test_property=test_property
    )
    nodes2edge_between_map = network.get_nodes2edge_between_map()
    union_pairs = None  # set initial value to verify that the loop's condition holds in the first iteration
    loop_iterations = 0
    # while no violation occurs, continue to abstract
    while not has_violation(network, test_property, random_inputs) and union_pairs != []:
        loop_iterations += 1
        # the minimal value over all max differences between input edges' weights of couples of nodes of same ar_type.
        union_pairs = []
        for i, layer in enumerate(network.layers[FIRST_ABSTRACT_LAYER:-1]):
            layer_index = i + 1
            prev_layer = network.layers[layer_index - 1]
            layer_couples = layer.get_couples_of_same_ar_type()
            # calc max difference between input edges' weights of couples
            for n1, n2 in layer_couples:
                max_diff_n1_n2 = INT_MIN
                for prev_node in prev_layer.nodes:
                    in_edge_n1 = nodes2edge_between_map.get((prev_node.name, n1.name), None)
                    a = 0 if in_edge_n1 is None else in_edge_n1.weight
                    in_edge_n2 = nodes2edge_between_map.get((prev_node.name, n2.name), None)
                    b = 0 if in_edge_n2 is None else in_edge_n2.weight
                    if abs(a - b) > max_diff_n1_n2:
                        max_diff_n1_n2 = abs(a - b)
                union_pairs.append(([n1, n2], max_diff_n1_n2))
        if union_pairs:  # if union_pairs != []:
            # take the couples whose maximal difference is minimal
            best_sequence_pairs = sorted(union_pairs, key=lambda x: x[1])
            cur_abstraction_seq_len = 0
            for (pair, diff) in best_sequence_pairs:
                if cur_abstraction_seq_len >= sequence_length:
                    break
                if pair[0].name not in network.name2node_map or pair[1].name not in network.name2node_map:
                    continue
                cur_abstraction_seq_len += 1
                union_name = "+".join([pair[0].name, pair[1].name])
                union_couple_of_nodes(network, pair[0], pair[1])
                nl_p2u = {pair[0].name: union_name, pair[1].name: union_name}
                finish_abstraction(network=network,
                                   next_layer_part2union=nl_p2u,
                                   verbose=verbose)
    return network
