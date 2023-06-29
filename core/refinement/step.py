from typing import AnyStr

from core.utils.debug_utils import debug_print
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Network import Network
from core.utils.ar_utils import calculate_weight_of_edge_between_two_part_groups
from core.pre_process.pre_process import fill_zero_edges


def split_back(network: Network, part: AnyStr) -> None:
    """
    implement the refinement step. split back part from the union node it was
    grouped into into a separated node
    :param network: Network
    :param part: Str of the name of the original node that is part of the union
    """
    # assume that layer_index is in [2, ..., L-1] (L = num of layers)
    try:
        layer_index = int(part.split("_")[1])
    except IndexError:
        debug_print("IndexError in core.test_refinement.step.split_back()")
        import IPython
        IPython.embed()
    layer = network.layers[layer_index]
    next_layer = network.layers[layer_index + 1]
    prev_layer = network.layers[layer_index - 1]
    part2node_map = network.get_part2node_map()
    union_node = network.name2node_map[part2node_map[part]]
    parts = union_node.name.split("+")
    other_parts = [p for p in parts if p != part]
    if not other_parts:
        return
    part_node = ARNode(name=part,
                       ar_type=union_node.ar_type,
                       activation_func=union_node.activation_func,
                       in_edges=[],
                       out_edges=[],
                       bias=network.orig_name2node_map[part].bias
                       )
    bias = sum([network.orig_name2node_map[other_part].bias
                for other_part in other_parts])
    other_parts_node = ARNode(name="+".join(other_parts),
                              ar_type=union_node.ar_type,
                              activation_func=union_node.activation_func,
                              in_edges=[],
                              out_edges=[],
                              bias=bias
                              )
    splitting_nodes = [part_node, other_parts_node]

    for splitting_node in splitting_nodes:
        # print("splitting_node.name={}".format(splitting_node.name))
        for next_layer_node in next_layer.nodes:
            group_a = splitting_node.name.split("+")
            group_b = next_layer_node.name.split("+")
            # print("call 1 - group_a")
            # print(group_a)
            # print("call 1 - group_b")
            # print(group_b)
            out_edge_weight = calculate_weight_of_edge_between_two_part_groups(
                network=network, group_a=group_a, group_b=group_b)

            if out_edge_weight is not None:
                out_edge = Edge(splitting_node.name,
                                next_layer_node.name,
                                out_edge_weight)
                splitting_node.out_edges.append(out_edge)
                next_layer_node.in_edges.append(out_edge)
            # fill_zero_edges(network)
        for prev_layer_node in prev_layer.nodes:
            group_a = prev_layer_node.name.split("+")
            group_b = splitting_node.name.split("+")
            # print("call 2 - group_a")
            # print(group_a)
            # print("call 2 - group_b")
            # print(group_b)
            in_edge_weight = calculate_weight_of_edge_between_two_part_groups(
                network=network, group_a=group_a, group_b=group_b)
            if in_edge_weight is not None:
                in_edge = Edge(prev_layer_node.name,
                               splitting_node.name,
                               in_edge_weight)
                splitting_node.in_edges.append(in_edge)
                prev_layer_node.out_edges.append(in_edge)
            # fill_zero_edges(network)
        layer.nodes.append(splitting_node)
        fill_zero_edges(network)
    network.remove_node(union_node, layer_index)
    network.generate_name2node_map()


def split_back_fixed(network: Network, part) -> None:
    """
    implement the refinement step. split back part from the union node it was
    grouped into into a separated node
    :param network: Network
    :param part: Str of the name of the original node that is part of the union
    """
    # assume that layer_index is in [2, ..., L-1] (L = num of layers)
    try:
        layer_index = int(part[0].split("_")[1])
    except IndexError:
        debug_print("IndexError in core.test_refinement.step.split_back()")
        import IPython
        IPython.embed()
    layer = network.layers[layer_index]
    next_layer = network.layers[layer_index + 1]  # if in_edge_weight is not None:
    #     in_edge_weights_sum += out_edge_weight
    prev_layer = network.layers[layer_index - 1]
    part2node_map = network.get_part2node_map()
    union_node = network.name2node_map[part2node_map[part[0].split("+")[0]]]
    parts = union_node.name.split("+")
    other_parts = [p for p in parts if p not in part]
    if not other_parts:
        return
    if union_node.ar_type == "inc":
        bias = max([network.orig_name2node_map[a_part].bias
                    for a_part in part])
    else:
        bias = min([network.orig_name2node_map[a_part].bias
                    for a_part in part])
    part_node = ARNode(name="+".join(part),
                       ar_type=union_node.ar_type,
                       activation_func=union_node.activation_func,
                       in_edges=[],
                       out_edges=[],
                       bias=bias
                       )
    if union_node.ar_type == "inc":
        bias = sum([network.orig_name2node_map[other_part].bias
                    for other_part in other_parts])
    else:
        bias = min([network.orig_name2node_map[a_part].bias
                    for a_part in part])
    other_parts_node = ARNode(name="+".join(other_parts),
                              ar_type=union_node.ar_type,
                              activation_func=union_node.activation_func,
                              in_edges=[],
                              out_edges=[],
                              bias=bias
                              )
    splitting_nodes = [part_node, other_parts_node]

    for splitting_node in splitting_nodes:
        # print("splitting_node.name={}".format(splitting_node.name))
        for next_layer_node in next_layer.nodes:
            out_edge_weights = []
            group_a = splitting_node.name.split("+")
            group_b = next_layer_node.name.split("+")
            # print("call 1 - group_a")
            # print(group_a)
            # print("call 1 - group_b")
            # print(group_b)
            for group_a_elem in group_a:
                group_list = []
                group_list.append(group_a_elem)
            out_edge_weight = 0
            if not next_layer_node.deleted:
                out_edge_weight = calculate_weight_of_edge_between_two_part_groups(
                    network=network, group_a=group_list, group_b=group_b)
            # if out_edge_weight is not None:
            #     out_edge_weights.append(out_edge_weight)
            # #print(group_b[0].split('_')[4])
            # if group_b[0].split('_')[3] == "inc":
            #     out_edge_weight = max(out_edge_weights)
            # else:
            #     out_edge_weight = min(out_edge_weights)
            # fill_zero_edges(network)
            out_edge = Edge(splitting_node.name,
                            next_layer_node.name,
                            out_edge_weight)
            splitting_node.out_edges.append(out_edge)
            next_layer_node.in_edges.append(out_edge)
        for prev_layer_node in prev_layer.nodes:
            # in_edge_weights_sum = 0
            group_a = prev_layer_node.name.split("+")
            group_b = splitting_node.name.split("+")
            # print("call 2 - group_a")
            # print(group_a)
            # print("call 2 - group_b")
            # print(group_b)
            for group_b_elem in group_b:
                group_list = []
                group_list.append(group_b_elem)
            in_edge_weight = calculate_weight_of_edge_between_two_part_groups(
                network=network, group_a=group_a, group_b=group_list)
            for deleted_name in network.deleted_name2node.keys():
                if int(deleted_name.split("_")[1]) == layer_index - 1:
                    deleted_node = network.deleted_name2node[deleted_name]
                    delete_list = []
                    delete_list.append(deleted_name)
                    edge_between = calculate_weight_of_edge_between_two_part_groups(
                        network=network, group_a=delete_list, group_b=group_list)
                    if deleted_node.ar_type == "inc":
                        splitting_node.bias += deleted_node.upper_bound * edge_between
                    else:
                        splitting_node.bias += deleted_node.lower_bound * edge_between
            # if in_edge_weight is not None:
            #     in_edge_weights_sum += out_edge_weight
            in_edge = Edge(prev_layer_node.name,
                           splitting_node.name,
                           in_edge_weight)
            splitting_node.in_edges.append(in_edge)
            prev_layer_node.out_edges.append(in_edge)
            # fill_zero_edges(network)
        layer.nodes.append(splitting_node)
        fill_zero_edges(network)
    network.remove_node(union_node, layer_index)
    network.generate_name2node_map()


def add_back(network: Network, part) -> None:
    print(part)
    print("part name")
    layer_index = int(part[0].split("_")[1])
    layer = network.layers[layer_index]
    next_layer = network.layers[layer_index + 1]
    prev_layer = network.layers[layer_index - 1]
    # part2node_map = network.get_part2node_map()
    # union_node = network.name2node_map[part2node_map[part]]
    # parts = union_node.name.split("+")
    # other_parts = [p for p in parts if p != part]
    # if not other_parts:
    #    return
    one_part = part[0].split("+")[0]
    one_node = network.name2node_map[one_part]
    # print("node.bound{}".format(one_node.bias))
    deleted_bias = network.deleted_name2node[one_part].bias
    network.remove_node(one_node, layer_index)
    bias = sum([network.orig_name2node_map[a_part].bias
                for a_part in part])
    # print("node.bias{}".format(bias))
    part_node = ARNode(name="+".join(part),
                       ar_type=one_node.ar_type,
                       activation_func=one_node.activation_func,
                       in_edges=[],
                       out_edges=[],
                       bias=bias
                       )
    part_node.added = True
    # bias = sum([network.orig_name2node_map[other_part].bias
    #             for other_part in other_parts])
    # other_parts_node = ARNode(name="+".join(other_parts),
    #                           ar_type=union_node.ar_type,
    #                           activation_func=union_node.activation_func,
    #                           in_edges=[],
    #                           out_edges=[],
    #                           bias=bias
    #                           )
    # splitting_nodes = [part_node, other_parts_node]

    #    for splitting_node in splitting_nodes:
    # print("splitting_node.name={}".format(splitting_node.name))
    # deleted_node = network.deleted_name2node[part_node.name]
    for next_layer_node in next_layer.nodes:
        # out_edge_weights = []
        if not next_layer_node.deleted:
            # print(next_layer_node.name)
            group_a = part_node.name.split("+")
            group_b = next_layer_node.name.split("+")
            # print("call 1 - group_a")
            # print(group_a)
            # print("call 1 - group_b")
            # print(group_b)
            # group_a = part
            out_edge_weight = calculate_weight_of_edge_between_two_part_groups(
                network=network, group_a=group_a, group_b=group_b)
            #     if out_edge_weight is not None:
            #         out_edge_weights.append(out_edge_weight)
            # #print(group_b.split('_')[4])
            # if group_b[0].split('_')[3] == "inc":
            #     out_edge_weight = max(out_edge_weights)
            # else:
            #     out_edge_weight = min(out_edge_weights)
            # fill_zero_edges(network)
            out_edge = Edge(part_node.name,
                            next_layer_node.name,
                            out_edge_weight)
            part_node.out_edges.append(out_edge)
            next_layer_node.in_edges.append(out_edge)
            # fill_zero_edges(network)
            # next_layer_node.bias -= deleted_bias * out_edge_weight
        else:
            out_edge = Edge(part_node.name,
                            next_layer_node.name,
                            0)
            part_node.out_edges.append(out_edge)
            next_layer_node.in_edges.append(out_edge)

    for prev_layer_node in prev_layer.nodes:
        # in_edge_weights_sum = 0
        group_a = prev_layer_node.name.split("+")
        group_b = part_node.name.split("+")
        # print("call 2 - group_a")
        # print(group_a)
        # print("call 2 - group_b")
        # print(group_b)
        # for group_b_elem in group_b:
        #     group_list = []
        #     group_list.append(group_b_elem)
        in_edge_weight = calculate_weight_of_edge_between_two_part_groups(
            network=network, group_a=group_a, group_b=group_b)
        # if in_edge_weight is not None:
        #     in_edge_weights_sum += out_edge_weight
        in_edge = Edge(prev_layer_node.name,
                       part_node.name,
                       in_edge_weight)
        part_node.in_edges.append(in_edge)
        prev_layer_node.out_edges.append(in_edge)
        # fill_zero_edges(network)
    # for deleted_name in network.deleted_name2node.keys():
    #     if int(deleted_name.split("_")[1]) == layer_index - 1:
    #         deleted_node = network.deleted_name2node[deleted_name]
    #         delete_list = []
    #         delete_list.append(deleted_name)
    #         edge_between = calculate_weight_of_edge_between_two_part_groups(
    #     network=network, group_a=delete_list, group_b=group_b)
    #         if deleted_node.ar_type == "inc":
    #             part_node.bias += deleted_node.upper_bound * edge_between
    #         else:
    #             part_node.bias += deleted_node.lower_bound * edge_between

    layer.nodes.append(part_node)
    del network.deleted_name2node[part_node.name]
    # fill_zero_edges(network)
    # network.deleted_nodename.remove(part)
    network.generate_name2node_map()
