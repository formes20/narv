#/usr/bin/python3

from import_marabou import dynamically_import_marabou
from core.visualization.visualize_network import visualize_network


def get_query_1_net():
    """
    return the newt
    :return: simple 3 layers net from CAV 2017 Reluplex paper
    """
    from core.data_structures.Network import Network
    from core.data_structures.Layer import Layer
    from core.data_structures.ARNode import ARNode
    from core.data_structures.Edge import Edge

    e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=1)
    e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=-1)
    e_10_y = Edge(src="x_1_0", dest="y", weight=1)
    e_11_y = Edge(src="x_1_1", dest="y", weight=1)

    # nodes
    x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11])
    x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10], out_edges=[e_10_y])
    x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11], out_edges=[e_11_y])
    y = ARNode(name="y", ar_type=None, in_edges=[e_10_y, e_11_y], out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0])
    h1_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Network(layers=[input_layer, h1_layer, output_layer])
    return net

def get_test_property_1():
    return {
        "type": "adversarial",
        "input":
            [
                (0, {"Lower": 0.0, "Upper": 1.0})
            ],
        "output":
            [
                (0, {"Lower": 0.5, "Upper": 1.0}),
            ]
    }


def test_get_query_1():
    """
    checks that the query that is generated is correct wrt net from CAV 2017
    Reluplex paper
    """
    test_property = get_test_property_1()
    dynamically_import_marabou(query_type=test_property["type"])
    net = get_query_1_net()

    from core.utils.marabou_query_utils import get_query

    print(get_query(network=net, test_property=test_property))



# ========================================

def get_query_2_net():
    """
    :return: Network
    """
    from core.data_structures.Network import Network
    from core.data_structures.Layer import Layer
    from core.data_structures.ARNode import ARNode
    from core.data_structures.Edge import Edge

    e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=1)
    e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=1)
    e_02_10 = Edge(src="x_0_2", dest="x_1_0", weight=1)
    e_03_10 = Edge(src="x_0_3", dest="x_1_0", weight=1)
    e_04_10 = Edge(src="x_0_4", dest="x_1_0", weight=1)
    e_10_20 = Edge(src="x_1_0", dest="x_2_0", weight=1)
    e_20_30 = Edge(src="x_2_0", dest="x_3_0", weight=1)
    e_20_31 = Edge(src="x_2_0", dest="x_3_1", weight=-1)
    e_30_40 = Edge(src="x_3_0", dest="x_4_0", weight=1)
    e_30_41 = Edge(src="x_3_0", dest="x_4_1", weight=1)
    e_30_42 = Edge(src="x_3_0", dest="x_4_2", weight=1)
    e_30_43 = Edge(src="x_3_0", dest="x_4_3", weight=1)
    e_30_44 = Edge(src="x_3_0", dest="x_4_4", weight=1)
    e_31_40 = Edge(src="x_3_1", dest="x_4_0", weight=1)
    e_31_41 = Edge(src="x_3_1", dest="x_4_1", weight=1)
    e_31_42 = Edge(src="x_3_1", dest="x_4_2", weight=1)
    e_31_43 = Edge(src="x_3_1", dest="x_4_3", weight=1)
    e_31_44 = Edge(src="x_3_1", dest="x_4_4", weight=1)

    # nodes
    x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10])
    x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10])
    x_0_2 = ARNode(name="x_0_2", ar_type=None, in_edges=[], out_edges=[e_02_10])
    x_0_3 = ARNode(name="x_0_3", ar_type=None, in_edges=[], out_edges=[e_03_10])
    x_0_4 = ARNode(name="x_0_4", ar_type=None, in_edges=[], out_edges=[e_04_10])
    x_1_0 = ARNode(name="x_1_0", ar_type=None,
                   in_edges=[e_00_10, e_01_10, e_02_10, e_03_10, e_04_10],
                   out_edges=[e_10_20])
    x_2_0 = ARNode(name="x_2_0", ar_type=None, in_edges=[e_10_20],
                   out_edges=[e_20_30, e_20_31])
    x_3_0 = ARNode(name="x_3_0", ar_type=None, in_edges=[e_20_30],
                   out_edges=[e_30_40, e_30_41, e_30_42, e_30_43, e_30_44])
    x_3_1 = ARNode(name="x_3_1", ar_type=None, in_edges=[e_20_31],
                   out_edges=[e_31_40, e_31_41, e_31_42, e_31_43, e_31_44])
    x_4_0 = ARNode(name="x_4_0", ar_type=None, in_edges=[e_30_40, e_31_40],
                   out_edges=[])
    x_4_1 = ARNode(name="x_4_1", ar_type=None, in_edges=[e_30_41, e_31_41],
                   out_edges=[])
    x_4_2 = ARNode(name="x_4_2", ar_type=None, in_edges=[e_30_42, e_31_42],
                   out_edges=[])
    x_4_3 = ARNode(name="x_4_3", ar_type=None, in_edges=[e_30_43, e_31_43],
                   out_edges=[])
    x_4_4 = ARNode(name="x_4_4", ar_type=None, in_edges=[e_30_44, e_31_44],
                   out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1, x_0_2, x_0_3, x_0_4])
    h1_layer = Layer(type_name="hidden", nodes=[x_1_0])
    h2_layer = Layer(type_name="hidden", nodes=[x_2_0])
    h3_layer = Layer(type_name="hidden", nodes=[x_3_0, x_3_1])
    output_layer = Layer(type_name="output",
                         nodes=[x_4_0, x_4_1, x_4_2, x_4_3, x_4_4])

    # net
    net = Network(
        layers=[input_layer, h1_layer, h2_layer, h3_layer, output_layer])
    return net


def get_test_property_2():
    return {
        "type": "basic",
        "input":
            [
                (0, {"Lower": 0.0, "Upper": 1.0}),
                (1, {"Lower": 0.0, "Upper": 1.0}),
                (2, {"Lower": 0.0, "Upper": 1.0}),
                (3, {"Lower": 0.0, "Upper": 1.0}),
                (4, {"Lower": 0.0, "Upper": 1.0}),
            ],
        "output":
            [
                (0, {"Lower": 0.5}),
                (1, {"Lower": 0.0}),
                (2, {"Lower": 0.0}),
                (3, {"Lower": 0.0}),
                (4, {"Lower": 0.0}),
            ]
    }


def test_get_query_2():
    """
    checks property_2 wrt net_2
    """
    test_property = get_test_property_2()
    dynamically_import_marabou(query_type=test_property["type"])
    from core.abstraction.naive import abstract_network
    net = get_query_2_net()
    visualize_network(network_layers=net.layers, title="original_net")
    from core.utils.marabou_query_utils import get_query

    vanilla_result = (get_query(network=net, test_property=test_property))

    net = abstract_network(net)
    visualize_network(network_layers=net.layers, title="abstracted_net")
    ar_result = (get_query(network=net, test_property=test_property))
    print(f"vanilla_result={vanilla_result}")
    print(f"ar_result={ar_result}")

    assert (vanilla_result[-1] == ar_result[-1])


if __name__ == '__main__':
    test_get_query_2()