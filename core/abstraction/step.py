from core.utils.debug_utils import embed_ipython
from core.data_structures.ARNode import ARNode
from core.data_structures.Edge import Edge
from core.data_structures.Network import Network
from core.utils.ar_utils import (
    calculate_weight_of_edge_between_two_part_groups
)

def union_couple_of_nodes(network: Network, node_1: ARNode, node_2: ARNode)\
        -> None:
    """
    union two nodes into one node, update in/out edges in next/prev layers resp.
    :param network: Network
    :param node_1: ARNode
    :param node_2: ARNode
    """
    # union two node of diferrent types is forbidden
    assert node_1.ar_type == node_2.ar_type
    layer_index = None
    try:
        layer_index = int(node_1.name.split("_")[1])
    except IndexError:
        print("IndexError in core.test_abstraction.step.union_couple_of_nodes")
        embed_ipython()
    layer = network.layers[layer_index]
    next_layer = network.layers[layer_index + 1]
    prev_layer = network.layers[layer_index - 1]
    if node_1.ar_type == "inc":
        sum_biases = max(node_1.bias, node_2.bias)
    else:
        sum_biases = max(node_1.bias, node_2.bias)
    if node_1.ar_type is None:
        print("node_1.ar_type is None")
        embed_ipython()
    # generate the union node
    union_node = ARNode(name="+".join([node_1.name, node_2.name]),
                        ar_type=node_1.ar_type,
                        activation_func=node_1.activation_func,
                        in_edges=[],
                        out_edges=[],
                        bias=sum_biases
                        )

    if node_1.ar_type == "dec":
        if node_1.upper_bound < node_2.upper_bound:
            union_node.upper_bound = node_1.upper_bound
            union_node.lower_bound = 0
        else:
            union_node.upper_bound = node_2.upper_bound
            union_node.lower_bound = 0
    else:
        if node_1.lower_bound < node_2.lower_bound:
            union_node.lower_bound = node_2.lower_bound
            union_node.upper_bound = node_1.upper_bound+node_2.upper_bound
        else:
            union_node.lower_bound = node_1.lower_bound
            union_node.upper_bound = node_1.upper_bound+node_2.upper_bound
    # update out edges (and in edges of next layer)
    for next_layer_node in next_layer.nodes:
        group_a = union_node.name.split("+")
        group_b = next_layer_node.name.split("+")
        out_edge_weight = calculate_weight_of_edge_between_two_part_groups(
            network, group_a, group_b)
        if out_edge_weight is not None:
            out_edge = Edge(union_node.name,
                            next_layer_node.name,
                            out_edge_weight)
            union_node.out_edges.append(out_edge)
            next_layer_node.in_edges.append(out_edge)
    # update in edges (and out edges of next layer)
    for prev_layer_node in prev_layer.nodes:
        group_a = prev_layer_node.name.split("+")
        group_b = union_node.name.split("+")
        in_edge_weight = calculate_weight_of_edge_between_two_part_groups(
            network, group_a, group_b)
        if in_edge_weight is not None:
            in_edge = Edge(prev_layer_node.name,
                           union_node.name,
                           in_edge_weight)
            union_node.in_edges.append(in_edge)
            prev_layer_node.out_edges.append(in_edge)
    layer.nodes.append(union_node)
    network.remove_node(node_2, layer_index)
    network.remove_node(node_1, layer_index)
    network.generate_name2node_map()
