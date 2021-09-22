#!/usr/bin/env python3

import random
from core.data_structures.Network import Network
from core.utils.alg2_utils import has_violation, get_limited_random_inputs
from core.pre_process.pre_process import preprocess
from core.abstraction.step import union_couple_of_nodes
from core.utils.abstraction_utils import finish_abstraction
from core.configuration.consts import VERBOSE, FIRST_ABSTRACT_LAYER


def heuristic_abstract_random(
        network: Network,
        test_property:dict,
        do_preprocess:bool=True,
        sequence_length:int=50,
        verbose:bool=VERBOSE
) -> Network:
    """
    abstract a network until @test_property holds in resulted abstract network
    the abstraction is done on @sequence_length pairs that are chosen randomly
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
    # while no violation occurs, continue to abstract
    while not has_violation(network, test_property, random_inputs):
        # check that the network is not fully abstracted
        if all(len(layer.nodes)<=4 for layer in network.layers[1:-1]):
            break
        # abstract constant number of random nodes
        layer_couples = []
        for i, layer in enumerate(network.layers[FIRST_ABSTRACT_LAYER:-1]):
            layer_couples.extend(layer.get_couples_of_same_ar_type())
        random.shuffle(layer_couples)

        # union these couples
        best_sequence_pairs = layer_couples
        cur_abstraction_seq_len = 0
        for pair in best_sequence_pairs:
            if cur_abstraction_seq_len >= sequence_length:
                break
            if pair[0].name not in network.name2node_map or \
                    pair[1].name not in network.name2node_map:
                continue
            cur_abstraction_seq_len += 1
            union_name = "+".join([pair[0].name, pair[1].name])
            union_couple_of_nodes(network, pair[0], pair[1])
            nl_p2u = {pair[0].name: union_name, pair[1].name: union_name}
            finish_abstraction(network=network,
                               next_layer_part2union=nl_p2u,
                               verbose=verbose)
    return network
