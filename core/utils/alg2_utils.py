import random
from typing import List, Dict

from core.utils.assertions_utils import is_evaluation_result_equal
from core.utils.verification_properties_utils import is_satisfying_assignment

def get_limited_random_input(input_size, test_property) -> Dict[int, float]:
    """
    :param input_size: network's input layer's size: len(network.layers[0].nodes)
    :param test_property:
    :return: random input a.t. the boundaries in test_property
    """
    #print("input_size="+str(input_size))
    random_input = {}
    for i in range(input_size):
        bounds = test_property["input"][i][1]
        random_input[i] = random.uniform(bounds["Lower"], bounds["Upper"])
    return random_input


def get_limited_random_inputs(input_size, test_property)\
        -> List[Dict[int, float]]:
    """
    :param input_size: network's input layer's size: len(network.layers[0].nodes)
    :param test_property: dictionary with lower/upper bound for each variable
    :return: List of inputs (Dicts from input var to value a.t. the bounds)
    """
    random_inputs = [get_limited_random_input(input_size=input_size,
                                              test_property=test_property)
                     for _ in range(5)]
    return random_inputs


def has_violation(network, test_property, inputs) -> bool:
    """
    check if outputs of evaluating @inputs in @network violates @test_property
    :param network: Network
    :param test_property: Dict, test_property
    :param inputs: List of inputs, each input is Dict from input var to value
    :return: Bool is there violation or not
    """
    _, variables2nodes = network.get_variables()
    for x in inputs:
        # output = network.evaluate(x)
        speedy_output = network.speedy_evaluate(x)
        #print(speedy_output)
        # assert is_evaluation_result_equal(output.items(), speedy_output)
        # print(f"output={output}")
        # print(f"speedy_output={speedy_output}")
        if is_satisfying_assignment(network, test_property, speedy_output, variables2nodes):
            return True  # if x is a satisfying assignment, there is a violation, because we want UNSAT
    return False
