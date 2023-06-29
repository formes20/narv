import random
import numpy as np
from core.nnet.read_nnet import network_from_nnet_file, get_all_acas_nets


# from core.configuration.consts import DELTA

def get_random_input(input_layer_size):
    return {
        # i: random.uniform(-1,1) for i in range(input_layer_size)

        # these are the defined ranges of input values in ACASXU networks
        0: random.uniform(-0.3284228772, 0.6798577687),
        1: random.uniform(-0.5000000551, 0.5000000551),
        2: random.uniform(-0.5000000551, 0.5000000551),
        3: random.uniform(-0.5000000551, 0.5000000551),
        4: random.uniform(-0.5000000551, 0.5000000551)
    }


def get_epsilon_similar_input_values(input_values, delta):
    changed_input_values = {}
    for i, x in input_values.items():
        val = x + random.uniform(delta * (-1), delta)
        changed_input_values[i] = val
    return changed_input_values


"""
find properties to check with marabou
generate random inputs and check that their result is equal wrt all networks.
if it is, add property to maybe_list
then check that for each input in maybe_list, changing in a bit does not 
change the result in all networks. if so, it is a property that we want to 
prove, otherwise, it is a property that we want to negate
"""


def find_properties_to_verify():
    DELTA = 0.04
    wanted_number_of_properties = 5
    current_number_of_properties = 0
    number_of_networks = 45
    # print(list(range(number_of_networks)))
    networks = get_all_acas_nets(list(range(number_of_networks)))
    # print(len(networks))
    # assume that input layer is constant in all networks
    input_layer_size = len(networks[0].layers[0].nodes)

    # print("start to find properties")
    tries_counter = 0
    good_properties = []
    outputs = []
    while current_number_of_properties < wanted_number_of_properties:
        tries_counter += 1
        input_values = get_random_input(input_layer_size=input_layer_size)
        is_good_property = True

        # pre-check: use only stable inputs, i.e inputs with equal result on all networks
        result = None
        original_output = None
        wrong = False
        for i, network in enumerate(networks):
            output = network.speedy_evaluate(input_values=input_values)
            current_result = output.argmin()
            for j, output_value in enumerate(output):
                if j != current_result and output_value == output[current_result]:
                    wrong = True
            if wrong:
                # print("发生了两个输出相同")
                break
            # print(i, output, current_result)
            # first network - assign value to result
            if result is None:
                result = current_result
                original_output = output
            else:
                if result != current_result:
                    # print(f"input values={input_values}, {i}'th network result is different")
                    is_good_property = False
                    break
        if not is_good_property:
            # print("property doesn't pass pre check")
            continue

        # check that in some DELTA env the results are equal
        changed_input_values = get_epsilon_similar_input_values(input_values,
                                                                delta=0.04)
        for i, network in enumerate(networks):
            output = network.speedy_evaluate(input_values=changed_input_values)
            # print(output)
            # print(type(output))
            current_result = output.argmin()
            if result != current_result:
                # print(f"{i}'th network result is different for changed_input_values={changed_input_values}")
                is_good_property = False
                print(f"property doesn't pass DELTA={DELTA} check")
                break

        # if the input property is OK, add it to the good_properties
        if is_good_property:
            current_number_of_properties += 1
            # print(f"current_number_of_properties={current_number_of_properties}")
            good_properties.append(input_values)
            outputs.append(original_output)
    # print(f'len(good_properties)={len(good_properties)}')
    # print(f'len(outputs)={len(outputs)}')

    for i, good_property in enumerate(good_properties):
        print(f'"adversarial_{i}":', end=" ")
        print(r"{")
        # input
        print(f'\t"type": "adversarial",')
        print(f'\t"input":')
        print('\t\t[')
        for j in range(len(good_property)):
            print(f'\t\t\t({j},', end=" ")
            print(r"{", end="")
            print(f'"Lower": {good_property[j] - 0.04}, "Upper": {good_property[j] + 0.04}', end="")
            print(r"}),")
        print(f'\t\t],')
        # output
        print(f'\t"output":')
        print('\t\t[')
        for j in range(len(outputs[i])):
            print(f'\t\t\t({j},', end=" ")
            print(r"{", end="")
            print(f'"Lower": {outputs[i][j]}, "Upper": {outputs[i][j]}', end="")
            print(r"}),")
        print(f'\t\t]')
        print(r"},")

        # print(f"{i}'th good_property is: {good_property}")
    # print(f"tries_counter={tries_counter}")


if __name__ == '__main__':
    find_properties_to_verify()
