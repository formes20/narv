from core.data_structures.Network import Network


def read_bounds_from_property(network: Network, test_property: dict) -> None:
    # property_content = TEST_PROPERTY_ACAS[test_property]["input"]
    input_layer = network.layers[0]
    for i, node in enumerate(input_layer.nodes):
        node.lower_bound = test_property["input"][i][1]['Lower']
        node.upper_bound = test_property["input"][i][1]['Upper']
        # print(str(i)+str(node.lower_bound))
        # print(str(i)+str(node.upper_bound))
