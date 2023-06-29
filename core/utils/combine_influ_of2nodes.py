from core.data_structures.Network import Network
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Layer import Layer


def combin_influnce(node_1: ARNode, node_2: ARNode, network: Network, layer_index: float,
                    nodes2edge_between_map) -> float:
    assert node_1.name.split("_")[3].split("+")[0] == node_2.name.split("_")[3].split("+")[0]
    assert layer_index >= 1
    # print(node_1)
    # print(node_2)
    # assert node_1.sum_in_edges > 0 
    # assert node_2.sum_in_edges > 0
    # nodes2edge_between_map = network.get_nodes2edge_between_map()
    max_in_edges = (node_1.ar_type == "inc")
    change_node_1 = 0
    change_node_2 = 0
    # print(nodes2edge_between_map)
    for prev_node in network.layers[layer_index - 1].nodes:
        # print(prev_node)
        in_edge_n1 = nodes2edge_between_map.get((prev_node.name, node_1.name), None)
        a = 0 if in_edge_n1 is None else in_edge_n1.weight
        # print(in_edge_n1)
        in_edge_n2 = nodes2edge_between_map.get((prev_node.name, node_2.name), None)
        b = 0 if in_edge_n2 is None else in_edge_n2.weight
        if max_in_edges:
            if a >= b:
                change_node_2 = change_node_2 + abs(a - b)
            else:
                change_node_1 = change_node_1 + abs(a - b)
        else:
            if a < b:
                change_node_2 = change_node_2 + abs(a - b)
            else:
                change_node_1 = change_node_1 + abs(a - b)
    if node_1.sum_in_edges != 0:
        node_1_influ = change_node_1 * node_1.cal_contri() / node_1.sum_in_edges
    else:
        if change_node_1 == 0:
            node_1_influ = 0
        else:
            node_1_influ = 10000000
    if node_2.sum_in_edges != 0:
        node_2_influ = change_node_2 * node_2.cal_contri() / node_2.sum_in_edges
    else:
        if change_node_2 == 0:
            node_2_influ = 0
        else:
            node_2_influ = 10000000
    return node_1_influ + node_2_influ
