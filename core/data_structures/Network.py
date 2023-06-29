import sys
import copy
import numpy as np
from typing import Tuple, Dict, List, TypeVar

AnyType = TypeVar('T')

sys.path.append("/home/artifact/narv")
from import_marabou import dynamically_import_marabou

dynamically_import_marabou()
from maraboupy import MarabouNetworkNNet

from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Layer import Layer
from core.configuration.consts import VERBOSE
from core.utils.comjoin import conjunction, join_atoms, is_subseq


class Network:
    """
    This class represents a neural network that supports abstraction and
    refinement steps in the process of verification. A Net has list of
    Layers and some metadata about the ARNodes in it. The major part of the
    functionality in the class deals with the metadata manipulation, but there
    are also some functions for input evaluation and for interfacing with the
    verifier, Marabou, and for generating Network from .nnet file
    """

    def __init__(self, layers: List, weights: List = None,
                 biases: List = None, acasxu_net: MarabouNetworkNNet = None):
        self.layers = layers
        self.acasxu_net = acasxu_net
        self.orig_layers = None
        self.orig_name2node_map = None

        self.name2node_map = None
        self.generate_name2node_map()
        self.initial_name2node_map = copy.deepcopy(self.name2node_map)
        self.weights = self.generate_weights() if weights is None else weights
        self._biases = self.generate_biases() if biases is None else biases
        self.biases = self.generate_biases()
        self.deleted_name2node = {}

    def generate_weights(self) -> List:
        """
        :return: matrix of incoming edges' weights in the network
        The matrix includes the incoming edges' weights of the nodes in each layer,
        from layer 1 (first hidden layer) to output layer.
        For example, weights[0] includes the incoming edges' weights of layer 1 (first hidden layer),
        i.e the weights from layer 0 to layer 1.
        """
        if len(self.layers) < 2:
            raise NotImplementedError("try to extract weights to network with not enough layers")
        # maybe can be used in future...
        # the difference between _weights and weights is that weights include 0-weighted edges and _weights not.
        # _weights = []
        # for layer in self.layers[1:]:
        #     layer_weights = []
        #     for node in layer.nodes:
        #         # list of weights of in_edges of the node. initialize with zeros and fill
        #         node_weights = []
        #         for in_edge in node.in_edges:
        #             node_weights.append(in_edge.weight)
        #         layer_weights.append(node_weights)
        #     _weights.append(layer_weights)

        weights = []
        for prev_layer_index, layer in enumerate(self.layers[1:]):
            prev_layer = self.layers[prev_layer_index]
            # map from name of node in previous layer to its index (in the previous layer)
            prev_layer_name2index = {node.name: index for (index, node) in enumerate(prev_layer.nodes)}
            layer_weights = []
            for node in layer.nodes:
                # list of weights of in_edges of the node. initialize with zeros and fill
                node_weights = [0.0] * len(self.layers[prev_layer_index].nodes)
                for in_edge in node.in_edges:
                    src_index = prev_layer_name2index[in_edge.src]
                    node_weights[src_index] = in_edge.weight
                layer_weights.append(node_weights)
            weights.append(layer_weights)
        # print(weights)
        # print(_weights)
        return weights

    def generate_biases(self) -> List:
        """
        :return: matrix of biases of the network
        The matrix includes the biases of the nodes in each layer except input layer,
        i.e. biases from layer 1 (first hidden layer) to output layer.
        """
        if len(self.layers) < 2:
            raise NotImplementedError("try to extract biases to network with not enough layers")
        biases = []
        for layer in self.layers[1:]:
            layer_biases = []
            for node in layer.nodes:
                layer_biases.append(node.bias)
            biases.append(layer_biases)
        return biases

    def __eq__(self, other: AnyType, verbose: bool = VERBOSE) -> bool:
        if self.get_general_net_data() != other.get_general_net_data():
            if verbose:
                print("self.get_general_net_data() ({}) != other.get_general_net_data() ({})".format(
                    self.get_general_net_data(), other.get_general_net_data()))
            return False
        for i, layer in enumerate(self.layers):
            if layer != other.layers[i]:
                if verbose:
                    print("self.layers[{}] != other.layers[{}]".format(i, i))
                return False
        return True

    def layer_index2layer_size(self) -> Dict:
        return {i: len(self.layers[i].nodes) for i in range(len(self.layers))}

    def get_general_net_data(self) -> Dict:
        """
        return dict with general data on the network, of the following form:
        {
            "layer_sizes": dict from layer_index to layer size
            "num_nodes": total number of nodes in net
            "num_layers": number of layers
            "num_hidden_layers": number of hidden layers
        }
        """
        layer_index2layer_size = self.layer_index2layer_size()
        num_nodes = sum([size for (ind, size) in layer_index2layer_size.items()])
        num_layers = len(layer_index2layer_size)
        num_hidden_layers = sum([l.type_name == "hidden" for l in self.layers])
        return {
            "layer_sizes": layer_index2layer_size,
            "num_nodes": num_nodes,
            "num_layers": num_layers,
            "num_hidden_layers": num_hidden_layers
        }

    def get_variable2layer_index(self, variables2nodes: Dict) -> Dict:
        variable2layer_index = {}
        node2layer_map = self.get_node2layer_map()
        for variable, node in variables2nodes.items():
            if node.endswith("_b") or node.endswith("_f"):
                node = node[:-2]  # remove suffix
            variable2layer_index[variable] = node2layer_map[node]
        return variable2layer_index

    def evaluate(self, input_values: Dict) -> Dict:
        # print("input_values={}".format(input_values.items()))
        nodes2variables, variables2nodes = self.get_variables()
        # variable2layer_index = self.get_variable2layer_index(variables2nodes)
        cur_node2val = {}
        node2val = {}
        for node in self.layers[0].nodes:
            var = nodes2variables[node.name]
            input_list = [v for (k, v) in sorted(input_values.items(), key=lambda x: x[0])[:len(self.layers[0].nodes)]]
            # print(var)
            # print(node.name)
            cur_node2val[node.name] = input_list[var]
            node2val.update(cur_node2val)
        for i, cur_layer in enumerate(self.layers[1:]):
            # print("evaluate():\t", i, cur_node2val.items())
            prev_node2val = cur_node2val
            layer_index = i + 1
            prev_layer = self.layers[i]
            cur_node2val = {node.name: node.bias for node in cur_layer.nodes}
            # prev_node2val = {node.name: 0.0 for node in prev_layer.nodes}

            for node in prev_layer.nodes:
                for out_edge in node.out_edges:
                    # prev_node2val include names with or without "_b" suffix
                    # if out_edge.src in prev_node2val:
                    if layer_index == 1:
                        src_val = prev_node2val[out_edge.src]
                    else:
                        src_val = prev_node2val[out_edge.src + "_f"]
                    add_val = out_edge.weight * src_val
                    cur_node2val[out_edge.dest] += add_val
            # apply activation function (from x_ij_b to x_ij_f)
            # don't apply to output layer
            if layer_index < len(self.layers) - 1:
                # print("layer_index={}, cur_layer is not output layer".format(layer_index))
                # print("before activation, cur_node2val.items() = {}".format(cur_node2val.items()))
                activation_vals = {}
                for k, v in cur_node2val.items():
                    activation_func = self.name2node_map[k].activation_func
                    activation_vals[k + "_f"] = activation_func(v)
                cur_node2val.update(activation_vals)
                node2val.update(cur_node2val)
                # print("after activation, cur_node2val.items() = {}".format(cur_node2val.items()))

            # cur_values = layer.evaluate(cur_values, nodes2variables, next,
            #                            variables2nodes, variable2layer_index)
            node2val.update(cur_node2val)
        return node2val

    def speedy_evaluate(self, input_values: Dict) -> List:
        assert self.weights is not None
        assert self.biases is not None
        input_list = [v for (k, v) in sorted(input_values.items(), key=lambda x: x[0])[:len(self.layers[0].nodes)]]
        # if net was generated by net_from_nnet_file() func
        # and haven't been abstracted yet, use acasxu evaluation
        # if hasattr(self, "acasxu_net"):
        #     input_list = np.array(input_list).reshape((1, -1))
        #     return self.acasxu_net.evaluate(input_list)
        current_inputs = np.array(input_list)
        for layer in range(len(self.layers) - 2):
            # assumes that activation function is relu (otherwise replace np.maximum with something else)
            # print(layer)
            # print("speedy_evaluate():\t{}\n{}".format(layer, current_inputs))
            # print(self.weights[layer])
            current_inputs = np.maximum(np.dot(self.weights[layer], current_inputs) + self.biases[layer], 0.0)
        # print("speedy_evaluate():\t{}\n{}".format(layer, current_inputs))
        outputs = np.dot(self.weights[-1], current_inputs) + self.biases[-1]
        # print("speedy_evaluate()\t outputs:\n{}".format(current_inputs))
        return outputs

    def generate_name2node_map(self) -> None:
        name2node_map = {}
        for layer in self.layers:
            for node in layer.nodes:
                name2node_map[node.name] = node
        self.name2node_map = name2node_map

    def remove_node(self, node: ARNode, layer_index: float) -> None:
        node = self.name2node_map[node.name]
        layer = self.layers[layer_index]
        for in_edge in node.in_edges:
            src_node = self.name2node_map[in_edge.src]
            # less effective
            # src_node.out_edges = [oe for oe in src_node.out_edges if oe != in_edge]
            if src_node.out_edges:
                for i, oe in enumerate(src_node.out_edges):
                    if oe == in_edge:
                        break
                del (src_node.out_edges[i])
            del in_edge
        for out_edge in node.out_edges:
            dest_node = self.name2node_map[out_edge.dest]
            # less effective
            # dest_node.in_edges = [ie for ie in dest_node.in_edges if ie != out_edge]
            if dest_node.in_edges:
                for i, ie in enumerate(dest_node.in_edges):
                    if ie == out_edge:
                        break
                del (dest_node.in_edges[i])
            del out_edge
        # less effective
        # layer.nodes = [n for n in layer.nodes if n != node]
        for i, cur_node in enumerate(layer.nodes):
            if cur_node == node:
                break
        del layer.nodes[i]
        del node
        self.generate_name2node_map

    def get_part2loss_map(self, example: Dict = {}) -> Dict:
        part2loss = {}
        nodes2edge_between_map = self.get_nodes2edge_between_map()
        part2node_map = self.get_part2node_map()
        for layer in self.layers[2:]:
            layer_part2loss_map = \
                self.get_layer_part2loss_map(self.orig_name2node_map,
                                             nodes2edge_between_map,
                                             example)
            part2loss.update(layer_part2loss_map)
        return part2loss

    def get_nodes2edge_between_map(self) -> Dict:
        nodes2edge_between_map = {}
        for layer in self.layers:
            for node in layer.nodes:
                for edge in node.out_edges:
                    nodes2edge_between_map[(edge.src, edge.dest)] = edge
        return nodes2edge_between_map

    def get_part2node_map(self) -> Dict:
        part2node_map = {}
        for layer in self.layers:
            for node in layer.nodes:
                parts = node.name.split("+")
                for part in parts:
                    part2node_map[part] = node.name
        return part2node_map

    def get_node2layer_map(self) -> Dict:
        """
        returns map from node name to layer index (in self.layers)
        """
        node2layer_map = {}
        for i, layer in enumerate(self.layers):
            for node in layer.nodes:
                node2layer_map[node.name] = i
        return node2layer_map

    def get_layer_part2loss_map(self,
                                orig_name2node_map: Dict,
                                nodes2edge_between_map: Dict,
                                example: Dict = {}) -> Dict:
        part2loss = {}
        part2node = self.get_part2node_map()
        nodes2variables, variables2nodes = self.get_variables()
        for node_name in self.name2node_map:
            parts = node_name.split("+")
            if len(parts) <= 1:
                continue
            for part in parts:
                part2loss.setdefault(part, 0.0)
                orig_part_node = orig_name2node_map[part]
                for edge in orig_part_node.out_edges:
                    dest_union = part2node[edge.dest]
                    abstract_edge = nodes2edge_between_map[(node_name,
                                                            dest_union)]
                    diff = abs(edge.weight - abstract_edge.weight)
                    node_var = nodes2variables.get(node_name + "_f",
                                                   nodes2variables.get(node_name + "_b",
                                                                       nodes2variables.get(node_name)))
                    diff *= example.get(node_var, 1.0)
                    part2loss[part] += diff
        return part2loss

    @staticmethod
    def get_global_refine_part(self,
                               orig_name2node_map: Dict,
                               example,
                               orig_net,
                               ori_var2val,
                               actions) -> list:
        refineables = []
        node2loss = {}
        node2eff = {}
        name2action = {}
        # part2node = self.get_part2node_map()
        # nodes2variables, variables2nodes = self.get_variables()
        # ori_nodes2variables, ori_variables2nodes = orig_net.get_variables()
        # print(orig_net)
        cur_var2val = self.evaluate(example)
        part2node_map = self.get_part2node_map()
        self.generate_name2node_map()
        # print(list(cur_var2val.values())[-5:])
        ##caculate max activation of each layer in original network with example input##
        # print(ori_var2val.items())
        max_of_each_layer = []
        for ori_layer in orig_net.layers:
            max_activation = -sys.maxsize - 1
            for ori_node in ori_layer.nodes:
                node_val = ori_var2val.get(ori_node.name + "_f",
                                           ori_var2val.get(ori_node.name + "_b",
                                                           ori_var2val.get(ori_node.name, None)))
                if node_val != None:
                    if node_val > max_activation:
                        max_activation = node_val
            max_of_each_layer.append(max_activation)
        # print(max_of_each_layer)
        print(max_of_each_layer)

        for action in actions:
            # print("name")
            # print(action.name_1+"+"+action.name_2)
            # print("relyed")
            # for i in action.relyed:
            #     print(i.name_1+"+"+i.name_2)
            if action.refineable():
                refineables.append(action)
        if refineables:
            print("refineables")
            for refineable_action in refineables:
                if refineable_action.types == "combine":

                    union_node_name = part2node_map.get(refineable_action.name_1.split("+")[0])
                    # print(refineable_action.name_1)
                    # print("refineable_action.name_1")
                    # print(refineable_action.name_2)
                    # print("refineable_action.name_2")
                    # print(union_node_name)
                    # print('union_node_name')
                    name2action[union_node_name] = refineable_action
                    if union_node_name and (union_node_name not in node2loss.keys()):
                        layer_index = refineable_action.layer
                        a_node = self.name2node_map[union_node_name]
                        a_activation = 0
                        ori_activation = 0
                        a_node_val = cur_var2val.get(a_node.name + "_f",
                                                     cur_var2val.get(a_node.name + "_b",
                                                                     cur_var2val.get(a_node.name)))
                        num_of_outedges = 0
                        for out_edge in a_node.out_edges:
                            a_activation += abs(
                                out_edge.weight) * a_node_val  # abstracted node's activation x abs(outedges)
                            num_of_outedges += 1
                        parts = a_node.name.split("+")
                        # print(parts)
                        assert len(parts) > 1
                        for part in parts:
                            ori_part_activation = 0
                            orig_part_node = orig_name2node_map[part]
                            node_val = ori_var2val.get(orig_part_node.name + "_f",
                                                       ori_var2val.get(orig_part_node.name + "_b",
                                                                       ori_var2val.get(orig_part_node.name, None)))
                            for out_edge in orig_part_node.out_edges:
                                ori_part_activation += abs(out_edge.weight) * node_val

                            ori_activation += ori_part_activation
                        node2eff[a_node.name] = ori_activation
                        diff_activation = abs(abs(a_activation) - abs(ori_activation))
                        if max_of_each_layer[layer_index] == 0:
                            if diff_activation != 0:
                                node2loss[a_node.name] = 10000000
                            else:
                                node2loss[a_node.name] = 0
                        else:
                            node2loss[a_node.name] = diff_activation / (
                                        max_of_each_layer[layer_index] * num_of_outedges)
        else:
            refine_list = []
            refine_layer = 10
            deleted_node_inf = 0
            for node_name in self.deleted_name2node.keys():
                if int(node_name.split("_")[1]) < refine_layer:
                    refine_list = []
                    refine_layer = int(node_name.split("_")[1])
                    refine_list.append(node_name)
                elif int(node_name.split("_")[1]) == refine_layer:
                    refine_list.append(node_name)
                else:
                    continue
            print("refine_list")
            print(refine_list)
            # deleted_name = refineable_action.name_1
            # print(deleted_name)
            # print("deleted_name")
            # name2action[deleted_name] = refineable_action
            for deleted_name in refine_list:
                deleted_node_activation = self.deleted_name2node[deleted_name].bias
                # parts = deleted_name.split("+")
                # if len(parts) == 1:
                deleted_node = orig_name2node_map[deleted_name]
                deleted_node_val = ori_var2val.get(deleted_name + "_f",
                                                   ori_var2val.get(deleted_name + "_b",
                                                                   ori_var2val.get(deleted_name)))
                num_of_outedges = 0
                for out_edge in deleted_node.out_edges:
                    deleted_node_inf += abs(out_edge.weight) * abs(deleted_node_val - deleted_node_activation)
                    num_of_outedges += 1
            # else:
            #     assert False
            #     for part in parts:
            #         dec = part.split("_")[3] == "dec"
            #         part_eff = 0
            #         if neg:
            #             other_parts_bound = sys.maxsize
            #         else:
            #             other_parts_bound = -sys.maxsize-1
            #         parts_out_edges_sum = 0
            #         part_node_val = ori_var2val.get(part + "_f",
            #                                         ori_var2val.get(part + "_b",
            #                                                             ori_var2val.get(part)))
            #         part_node = orig_name2node_map.get(part)
            #         if dec:
            #             if part_node_val < other_parts_bound:
            #                 other_parts_bound = part_node_val
            #         else:
            #             if part_node_val > other_parts_bound:
            #                 other_parts_bound = part_node_val
            #         num_of_outedges = 0
            #         for out_edge in part_node.out_edges:
            #             parts_out_edges_sum += abs(out_edge.weight)
            #             #if out_edge.weight != 0:
            #             num_of_outedges += 1
            #     part_eff = abs(parts_out_edges_sum * other_parts_bound)
            #     deleted_node_activation = part_eff
            if max_of_each_layer[int(deleted_name.split("_")[1])] != 0:
                # node2loss[deleted_name] = abs(deleted_node_activation) / max_of_each_layer[deleted_name.split("_")[1]]
                node2loss[deleted_name] = abs(deleted_node_inf) / (
                            max_of_each_layer[int(deleted_name.split("_")[1])] * num_of_outedges)
            # node2loss[deleted_name] = abs(deleted_node_activation) / num_of_outedges
            else:
                if deleted_node_inf != 0:
                    node2loss[deleted_name] = 10000000
                else:
                    node2loss[deleted_name] = 0

        # for layer_index,a_layer in enumerate(self.layers[1:-2],1):
        #     for a_node in a_layer.nodes:
        #         a_activation = 0
        #         ori_activation = 0
        #         a_node_val = cur_var2val.get(a_node.name + "_f",
        #                                            cur_var2val.get(a_node.name + "_b",
        #                                                                cur_var2val.get(a_node.name)))
        #         num_of_outedges = 0
        #         for out_edge in a_node.out_edges:
        #             a_activation += abs(out_edge.weight) * a_node_val
        #             num_of_outedges += 1
        #         parts = a_node.name.split("+")
        #         if len(parts) <= 1:
        #             node2loss[a_node.name] = -1
        #             node2eff[a_node.name] = ori_var2val.get(a_node.name + "_f",
        #                                            ori_var2val.get(a_node.name + "_b",
        #                                                                ori_var2val.get(a_node.name,None)))
        #             continue
        #         for part in parts:
        #             ori_part_activation = 0
        #             orig_part_node = orig_name2node_map[part]
        #             node_val = ori_var2val.get(orig_part_node.name + "_f",
        #                                            ori_var2val.get(orig_part_node.name + "_b",
        #                                                                ori_var2val.get(orig_part_node.name,None)))
        #             for out_edge in orig_part_node.out_edges:
        #                 ori_part_activation += abs(out_edge.weight) * node_val
        #             ori_activation += ori_part_activation

        #         node2eff[a_node.name] = ori_activation
        #         diff_activation = abs(abs(a_activation)-abs(ori_activation))
        #         if max_of_each_layer[layer_index] == 0:
        #             if diff_activation != 0:
        #                 node2loss[a_node.name] = 10000000
        #             else:
        #                 node2loss[a_node.name] = 0
        #         else:
        #             node2loss[a_node.name] = diff_activation / (max_of_each_layer[layer_index] * num_of_outedges)
        #         #node2loss[a_node.name] = diff_activation / max_of_each_layer[layer_index]
        #         #node2loss[a_node.name] = diff_activation / num_of_outedges
        # for deleted_name in self.deleted_nodename:
        #     deleted_node_activation = 0
        #     parts = deleted_name.split("+")
        #     if len(parts) == 1:  
        #         deleted_node = orig_name2node_map[deleted_name]
        #         deleted_node_val = ori_var2val.get(deleted_name + "_f",
        #                                         ori_var2val.get(deleted_name + "_b",
        #                                                             ori_var2val.get(deleted_name)))
        #         num_of_outedges = 0                                                    
        #         for out_edge in deleted_node.out_edges:
        #             deleted_node_activation += abs(out_edge.weight) * deleted_node_val
        #             num_of_outedges += 1
        #     else:
        #         for part in parts:
        #             neg = part.split("_")[3] == "neg"
        #             part_eff = 0
        #             if neg:
        #                 other_parts_bound = sys.maxsize
        #             else:
        #                 other_parts_bound = -sys.maxsize-1
        #             parts_out_edges_sum = 0
        #             part_node_val = ori_var2val.get(part + "_f",
        #                                             ori_var2val.get(part + "_b",
        #                                                                 ori_var2val.get(part)))
        #             part_node = orig_name2node_map.get(part)
        #             if neg:
        #                 if part_node_val < other_parts_bound:
        #                     other_parts_bound = part_node_val
        #             else:
        #                 if part_node_val > other_parts_bound:
        #                     other_parts_bound = part_node_val
        #             num_of_outedges = 0
        #             for out_edge in orig_part_node.out_edges:
        #                 parts_out_edges_sum += abs(out_edge.weight)
        #                 num_of_outedges += 1
        #         part_eff = abs(parts_out_edges_sum * other_parts_bound)
        #     if max_of_each_layer[int(deleted_name.split("_")[1])] != 0 :
        #     #node2loss[deleted_name] = abs(deleted_node_activation) / max_of_each_layer[deleted_name.split("_")[1]] 
        #         node2loss[deleted_name] = abs(deleted_node_activation) / (max_of_each_layer[int(deleted_name.split("_")[1])] * num_of_outedges)             
        #     #node2loss[deleted_name] = abs(deleted_node_activation) / num_of_outedges
        #     else:
        #         if deleted_node_activation != 0:
        #             node2loss[deleted_name] = 10000000
        #         else:
        #             node2loss[deleted_name] = 0
        # self.generate_name2node_map()
        top_diff_node = sorted(node2loss.items(), key=lambda x: x[1])[-1][0]
        operations = []
        # atoms = []
        print(top_diff_node)
        print('top_diff_node')
        if refineables:

            # combined node
            # parts = top_diff_node.split("+")
            # #pos = parts[0].split('_')[3]=="pos"
            # inc = parts[0].split('_')[3]=="inc"
            # for action in refineables:
            #     if inc == action.inc:
            #         print(action.name_1)
            #         print('action.name_1')
            #         print(action.name_2)
            #         print('action.name_2')
            #         if conjunction(action.name_1.split("+"),parts) and conjunction(action.name_2.split("+"),parts):
            #             print("yes")
            #             operations.append(action)
            #             atoms = join_atoms(atoms, parts, action.name_1.split('+'))
            #             atoms = join_atoms(atoms, parts, action.name_2.split('+'))
            # print(atoms)
            # assert len(atoms) > 0
            # total_ori_eff = node2eff[top_diff_node]
            # atom_attri = []
            # for atom in atoms:
            #     if inc:
            #         parts_bound = -sys.maxsize-1
            #     else:
            #         parts_bound = sys.maxsize
            #     parts_out_edges_sum = 0
            #     for part in atom:
            #         part_eff = 0
            #         part_out_edges_sum = 0
            #         part_node_val = ori_var2val.get(part + "_f",
            #                                         ori_var2val.get(part + "_b",
            #                                                             ori_var2val.get(part)))
            #         if inc:
            #             if part_node_val > parts_bound:
            #                 parts_bound = part_node_val
            #         else:
            #             if part_node_val < part_bound:
            #                 parts_bound = part_node_val

            #         part_node = orig_name2node_map.get(part)
            #         for out_edge in part_node.out_edges:
            #             part_out_edges_sum += abs(out_edge.weight) 
            #         part_eff = part_out_edges_sum * part_node_val
            #         parts_out_edges_sum += part_out_edges_sum
            #     atom_attri.append([parts_bound,parts_out_edges_sum])

            # atoms_num = len(atoms)
            # max_split = int((1<<atoms_num)/2)
            # cost_max = -sys.maxsize
            # for i in range(1, max_split):
            #     left = []
            #     right = []
            #     left_edge_sum = 0
            #     right_edge_sum = 0
            #     cost = 0
            #     if inc:
            #         left_bound = 0
            #         right_bound = 0
            #     else:
            #         left_bound = sys.maxsize
            #         right_bound = sys.maxsize
            #     for j in range(atoms_num):
            #         if(1 << j) & i:
            #             for item in atoms[j]:
            #                 left.append(item)
            #                 if inc:
            #                     if atom_attri[j][0] > left_bound:
            #                         left_bound = atom_attri[j][0]
            #                 else:
            #                     if atom_attri[j][0] < left_bound:
            #                         left_bound = atom_attri[j][0]
            #                 left_edge_sum += atom_attri[j][1]
            #         else:
            #             for item in atoms[j]:
            #                 right.append(item)
            #                 if inc:
            #                     if atom_attri[j][0] > right_bound:
            #                         right_bound = atom_attri[j][0]
            #                 else:
            #                     if atom_attri[j][0] < right_bound:
            #                         right_bound = atom_attri[j][0]
            #                 right_edge_sum += atom_attri[j][1]                            
            #     cost = abs(node2eff[top_diff_node] - left_edge_sum*left_bound - right_edge_sum*right_bound)
            #     if cost > cost_max:
            #         cost_max = cost
            #         part_1 = left
            #         part_2 = right
            refine_action = name2action[top_diff_node]
            for rely in refine_action.rely:
                rely.relyed.remove(refine_action)
            actions.remove(refine_action)
            # for action in actions:
            #     if conjunction(action.name_1.split('+'),part_1) and not conjunction(action.name_1.split('+'),part_2):
            #         if conjunction(action.name_2.split('+'),part_2) and not conjunction(action.name_2.split('+'),part_1):
            #             for rely in action.rely:
            #                 rely.relyed.remove(action)
            #             actions.remove(action)
            #             break
            #     elif conjunction(action.name_1.split('+'),part_2) and not conjunction(action.name_1.split('+'),part_1):
            #         if conjunction(action.name_2.split('+'),part_1) and not conjunction(action.name_2.split('+'),part_2):
            #             for rely in action.rely:
            #                 rely.relyed.remove(action)
            #             actions.remove(action)
            #             break    
            part_name = refine_action.name_1.split("+")
        else:
            # refine_action = name2action[top_diff_node]
            # for rely in refine_action.rely:
            #     rely.relyed.remove(refine_action)
            # actions.remove(refine_action)

            part_name = []
            part_name.append(top_diff_node)
            # for action in actions:
            #     if action.types != "combine":
            #         if is_subseq(action.name_1.split("+"),top_diff_node.split("+")) and is_subseq(action.name_2.split("+"),top_diff_node.split("+")):
            #             for rely in action.rely:
            #                 rely.relyed.remove(action)
            #             actions.remove(action)
            #             break

            # ori_eff = 0
            # a_node_val = cur_var2val.get(top_diff_node + "_f",
            #                                        cur_var2val.get(top_diff_node + "_b",
            #                                                            cur_var2val.get(top_diff_node)))
            # node = self.name2node_map[top_diff_node]
            # print(node)
            # print("###################refine node##################")
            # for out_edge in node.out_edges:
            #     ori_eff += abs(out_edge.weight) * a_node_val

            # for part in parts:
            #     neg = part.split("_")[3] == "neg"
            #     other_parts = [p for p in parts if p != part]
            #     part_eff = 0
            #     other_parts_diff = 0
            #     other_parts_bound = sys.maxsize
            #     other_parts_out_edges_sum = 0
            #     max_diff_part = 0
            #     part_node_val = ori_var2val.get(part + "_f",
            #                                        ori_var2val.get(part + "_b",
            #                                                            ori_var2val.get(part)))
            #     part_node = orig_name2node_map.get(part)
            #     for out_edge in part_node.out_edges:
            #         part_eff += abs(out_edge.weight) * part_node_val
            #     for other_part in other_parts:
            #         other_part_val = ori_var2val.get(other_part + "_f",
            #                                             ori_var2val.get(other_part + "_b",
            #                                                         ori_var2val.get(other_part)))
            #         #other_part_node = orig_name2node_map.get(other_part)
            #         if neg:
            #             if other_part_val < other_parts_bound:
            #                 other_parts_bound = other_part_val
            #         else:
            #             if other_part_val > other_parts_bound:
            #                 other_parts_bound = other_part_val

            #         for out_edge in orig_part_node.out_edges:
            #             other_parts_out_edges_sum += abs(out_edge.weight)
            #     other_parts_diff = abs(node2eff[top_diff_node] - (other_parts_out_edges_sum * other_parts_bound + part_eff))
            #     if other_parts_diff >= max_diff_part:
            #         part_name = part
            #         max_diff_part = other_parts_diff

        return part_name

    def get_next_nodes(current_values: List) -> List:
        next_nodes = set([])
        for node in current_values:
            for edge in node.out_edges:
                next_nodes.add(edge.dest)
        return list(next_nodes)

    def get_part2example_change(self, example: Dict) -> Dict:
        """
        run example, calculate difference in each node between its outputed
        value and the original netework outputed value,
        @example: a dict from node name no value
        @return map from part to the sum of differences
        """
        part2diffs = {}
        cur_layer_values = example
        while True:
            next_layer_nodes = self.get_next_nodes(cur_layer_values)
            if not next_layer_values:
                break
            next_layer_values = {name: 0.0 for name in next_layer_nodes}
            for node_name in cur_layer_values.keys():
                node = self.name2node_map[node_name]
                for edge in node.out_edges:
                    cur_value = cur_layer_values[node.name]
                    next_layer_values[edge.dest] += edge.weight * cur_value
            for node, val in next_layer_values.items():
                part2diffs[node] = val
            cur_layer_values = next_layer_values
        return part2diffs

    def get_variables(self, property_type: str = "basic") -> Tuple[Dict, Dict]:
        nodes2variables = {}
        variables2nodes = {}
        var_index = 0
        is_acas_xu_conjunction = property_type == "acas_xu_conjunction"
        # is_adversarial = property_type == "adversarial"
        for l_index, layer in enumerate(self.layers):
            for node in layer.nodes:
                if layer.type_name in ["input", "output"]:
                    nodes2variables[node.name] = var_index
                    variables2nodes[var_index] = node.name
                    var_index += 1
                else:  # hidden layer, all nodes with relu activation
                    if is_acas_xu_conjunction and l_index == len(self.layers) - 3:
                        # prev output layer, no relu
                        suffices = ["_b"]
                    # elif is_adversarial and l_index == len(self.layers)-2:
                    #     # prev output layer, no relu
                    #     suffices = ["_b"]
                    else:
                        suffices = ["_b", "_f"]
                    for suffix in suffices:
                        nodes2variables[node.name + suffix] = var_index
                        variables2nodes[var_index] = node.name + suffix
                        var_index += 1
        return nodes2variables, variables2nodes

    def get_large(self) -> float:
        # some silly heuristic to get upper bound for value in the network
        # without returning too big number
        # in each layer multiply the result in the max absolute value
        # of an out edge
        large = 1.0
        for layer in self.layers:
            for node in layer.nodes:
                out_weights = [abs(edge.weight) for edge in node.out_edges]
                large *= max(out_weights, default=1.0)
        return min(20, "large = {}\n\n".format(large))

    def __str__(self) -> str:
        s = ""
        net_data = self.get_general_net_data()
        for k, v in net_data.items():
            s += "{}: {}\n".format(k, v)
        s += "\n"
        s += "\n\n".join(layer.__str__() for layer in self.layers)
        return s

    def generate_in_edge_weight(self) -> None:
        for i in range(0, len(self.layers)):
            self.layers[i].generate_in_edge_weight_sum()
