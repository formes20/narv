#!/usr/bin/env python3

from core.data_structures.Network import Network
from core.utils.alg2_utils import has_violation, get_limited_random_inputs
from core.pre_process.pre_process import preprocess
from core.abstraction.step import union_couple_of_nodes
from core.utils.abstraction_utils import finish_abstraction
from core.configuration.consts import (
    VERBOSE, FIRST_ABSTRACT_LAYER, INT_MIN
)


def heuristic_abstract_alg2(
        network: Network,
        test_property:dict,
        random_inputs={},
        do_preprocess:bool=True,
        sequence_length:int=50,
        verbose:bool=VERBOSE
) -> Network:
    print("1111111111111111111111111111111111111111111111111111111111111111111111")
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
    sat = False
    if do_preprocess:
        preprocess(network)

    if not random_inputs:
        input_size = len(network.layers[0].nodes)
        # generate random inputs in the bound of the test property
        random_inputs = get_limited_random_inputs(
            input_size=input_size,
            test_property=test_property
        )
    #print(random_inputs)
    #print(random_inputs)
    nodes2edge_between_map = network.get_nodes2edge_between_map()
    #print(nodes2edge_between_map.values())
    union_pairs = None  # set initial value to verify that the loop's condition holds in the first iteration
    loop_iterations = 0
    # while no violation occurs, continue to abstract
    while not has_violation(network, test_property, random_inputs) and union_pairs != []:
        loop_iterations += 1
        # the minimal value over all max differences between input edges' weights of couples of nodes of same ar_type.
        union_pairs = []
        for i, layer in enumerate(network.layers[FIRST_ABSTRACT_LAYER:-1]):
            # print(layer)
            layer_index = i + FIRST_ABSTRACT_LAYER - 1
            prev_layer = network.layers[layer_index]
            # print(i)
            # print(prev_layer)
            #sleep(10000000)
            layer_couples = layer.get_couples_of_same_ar_type()
            # calc max difference between input edges' weights of couples
            for n1, n2 in layer_couples:
                max_diff_n1_n2 = INT_MIN
                for prev_node in prev_layer.nodes:
                    # print(prev_node)
                    # print(n1.name)
                    # print("..................................")
                    in_edge_n1 = nodes2edge_between_map.get((prev_node.name, n1.name), None)
                    a = 0 if in_edge_n1 is None else in_edge_n1.weight
                    in_edge_n2 = nodes2edge_between_map.get((prev_node.name, n2.name), None)
                    b = 0 if in_edge_n2 is None else in_edge_n2.weight
                    if abs(a - b) > max_diff_n1_n2:
                        max_diff_n1_n2 = abs(a - b)
                        #print(max_diff_n1_n2)
                union_pairs.append(([n1, n2], max_diff_n1_n2))                
        if union_pairs:  # if union_pairs != []:
            # take the couples whose maximal difference is minimal
            best_sequence_pairs = sorted(union_pairs, key=lambda x: x[1])
            #print(best_sequence_pairs)
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
    if loop_iterations == 0:
        sat = True
    print("alg2 operation time")
    print(loop_iterations)
    return network,sat
