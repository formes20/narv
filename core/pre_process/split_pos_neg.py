from core.utils.debug_utils import debug_print
from core.configuration.consts import VERBOSE, FIRST_POS_NEG_LAYER
from core.data_structures.Edge import Edge
from core.data_structures.Network import Network
import time


# def adjust_layer_after_split_pos_neg(network:Network,
#                                      layer_index: int=
#                                      FIRST_POS_NEG_LAYER) -> None:
#     # debug_print("adjust_layer_after_split_pos_neg")
#     cur_layer = network.layers[layer_index]
#     next_layer = network.layers[layer_index + 1]
#     count = 0
#     p1 = 0
#     p2 = 0
#     p3 = 0
#     p4 = 0
#     p5 = 0
#     p6 = 0
#     p7 = 0
#     p8 = 0
#     t10 = time.time()
#     for node in cur_layer.nodes:
#         node.new_out_edges = []
#     for next_node in next_layer.nodes:
#         next_node.new_in_edges = []
#         for cur_node in cur_layer.nodes:
#             for out_edge in cur_node.out_edges:
#                 #for suffix in ["", "_pos", "_neg"]:
#                     #if out_edge.dest + suffix == next_node.name:
#                 #if out_edge.dest.split("_")[2] == next_node.name.split("_")[2]:
#                 if (out_edge.dest)[4:] == next_node.name[4:-4]:
#                     t1 = time.time()
#                     if count > 0:
#                         p8 += (t1 - t4)
#                     weight = out_edge.weight
#                     t2 = time.time()
#                     p1 += (t2 - t1)
#                     edge = Edge(cur_node.name, next_node.name, weight)
#                     t3 = time.time()
#                     p2 += (t3 - t2)
#                     cur_node.new_out_edges.append(edge)
#                     t4 = time.time()
#                     p3 += (t4 - t3)
#                     next_node.new_in_edges.append(edge)
#                     t5 = time.time()
#                     p4 += (t5 - t4)
#                     count +=1
#         t6 = time.time()
#         next_node.in_edges = next_node.new_in_edges
#         t7 = time.time()
#         p5 += (t7 - t6)
#         #del next_node.new_in_edges
#     for node in cur_layer.nodes:
#         t8 = time.time()
#         node.out_edges = node.new_out_edges
#         t9 = time.time()
#         p6 += (t9 - t8)
#         #del node.new_out_edges
#     print("p1:{}".format(p1))
#     print("p2:{}".format(p2))
#     print("p3:{}".format(p3))
#     print("p4:{}".format(p4))
#     print("p5:{}".format(p5))
#     print("p6:{}".format(p6))
#     print("p8:{}".format(p8))
#     print(count)
#     t11 = time.time()
#     print(t11 - t10)
#     if VERBOSE:
#         debug_print("after adjust_layer_after_split_pos_neg()")
#         print(network)

def adjust_layer_after_split_pos_neg(network: Network,
                                     layer_index: int =
                                     FIRST_POS_NEG_LAYER) -> None:
    # debug_print("adjust_layer_after_split_pos_neg")
    cur_layer = network.layers[layer_index]
    next_layer = network.layers[layer_index + 1]
    count = 0
    for node in cur_layer.nodes:
        node.new_out_edges = []
    for next_node in next_layer.nodes:
        next_node.in_edges = []
    for node in cur_layer.nodes:
        for out_edge in node.out_edges:
            for suffix in ["_pos", "_neg"]:
                linked_node = network.name2node_map.get(out_edge.dest + suffix, None)
                if linked_node:
                    weight = out_edge.weight
                    edge = Edge(node.name, linked_node.name, weight)
                    node.new_out_edges.append(edge)
                    linked_node.in_edges.append(edge)
                    count += 1
        node.out_edges = node.new_out_edges
    print(count)


def preprocess_split_pos_neg(network: Network) -> None:
    """
    split net nodes to nodes with only positive/negative out edges
    preprocess all hidden layers (from last to first), then adjust input
    layer
    """
    if VERBOSE:
        debug_print("preprocess_split_pos_neg()")
    # orig_input_layer = copy.deepcopy(network.layers[0])
    t1 = time.time()
    for i in range(len(network.layers) - 2, FIRST_POS_NEG_LAYER, -1):
        network.layers[i].split_pos_neg(network.name2node_map)
    # splited_input_layer = self.layers[0]
    # for node in orig_input_layer.nodes:
    #    node.out_edges = []
    #    for splitted_node in splited_input_layer:
    #        if splitted_node.name[:-4] == node.name:  # suffix=_oos/_neg
    #            edge = Edge(src=node.name, dest=splitted_node.name, weight=1.0)
    #            node.out_edges.append(edge)
    #            splitted_node.in_edges.append(edge)
    t2 = time.time()
    print("split-time{}".format(t2 - t1))
    network.generate_name2node_map()
    t3 = time.time()
    print("name2node-time{}".format(t3 - t2))
    # print(self)
    adjust_layer_after_split_pos_neg(network, layer_index=FIRST_POS_NEG_LAYER)
