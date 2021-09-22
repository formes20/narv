#pylint: disable=all

"""
/* ********************/
/*! \file test_abstraction.py
 ** \verbatim
 ** Top contributor:
 **   Yizhak Yisrael Elboher
 ** All rights reserved.
 ** See the file COPYING in this directory for licensing information.
 ** \endverbatim
 **
 ** \brief [[ Add one-line brief description here ]]
 ** implements test_abstraction and test_refinement of Relu based neural networkd
 ** according to the method suggested by Yizhak Yisrael Elboher & Guy Kats.
 **
 ** [[ Add lengthier description here ]]
 **/
"""


"""
usage:
nohup python3 test_abstraction.py  > ~/Desktop/AR_results/log_`date | sed -e 's/ /_/g'`.txt &
should generate a log file and fig into the dir
"""


import os
import sys
import json
import copy
import time
import random
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# marabou related imports and prerequisits
import import_marabou
from maraboupy import MarabouCore
from maraboupy import MarabouNetworkNNet as mnn

from core.configuration.consts import EPSILON, PATH_TO_MARABOU_ACAS_EXAMPLES, PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES, \
    results_directory, COMPLETE_ABSTRACTION, FIRST_POS_NEG_LAYER, FIRST_INC_DEC_LAYER, FIRST_ABSTRACT_LAYER, \
    DO_CEGAR, INPUT_UPPER_BOUND, INPUT_LOWER_BOUND, UNSAT_EXIT_CODE, SAT_EXIT_CODE, VISUAL_WEIGHT_CONST, INT_MIN, \
    INT_MAX, VERBOSE, SORTING_COLOR_MAP, LAYER_INTERVAL, NODE_INTERVAL, LAYER_TYPE_NAME2COLOR, \
    COMPARE_TO_PREPROCESSED_NET
from core.utils.debug_utils import debug_print, embed_ipython
from core.utils.assertions_utils import is_evaluation_result_equal
from core.utils.verification_properties_utils import get_test_property_acas, get_test_property_tiny, \
    get_test_property_input_2_output_1, get_test_property_input_3_output_1


"""
# notes:
net is fully connected
net is relu based (except input/output layers)
"""




# problem: networkx draw_nodes func get color values unsorted
# solution: sort them
# method: using sorting_color_key function that uses sorting_color_map
def sorting_color_key(color):
    return SORTING_COLOR_MAP[color]

ar_type2sign = {
    "inc": "+",
    "dec": "-"
}


class ARTypeException(Exception):
    pass


# test_abstraction functions
def relu(x):
    """
    :param x: float
    :return: float, Relu(x)
    """
    return max(x, 0)


#TODO: MAYBE remove next choose_weight_func and return this func back
"""
def choose_weight_func(dest_ar_type, node_positivity):
    # return the appropriate func to choose weight by, and its initial weight
    # func is min/max, initial weight is sys.maxint/-sys.maxint (accordingly)
    d = {
        ("inc", "pos"): (min, INT_MAX),
        ("inc", "neg"): (max, INT_MIN),
        ("dec", "pos"): (max, INT_MIN),
        ("dec", "neg"): (min, INT_MAX)
    }
    return d[(dest_ar_type, node_positivity)]
"""
    # equivalent code:
    #if ar_node.ar_type == "inc":
    #    if ar_dest.ar_type == "inc":
    #        return min, sys.maxint
    #    else:
    #        return max, -sys.maxint
    #else:  # node.ar_type == "dec":
    #    if ar_dest.ar_type == "inc":
    #        return max, -sys.maxint
    #    else:
    #        return min, sys.maxint

"""
def choose_weight_func(dest_node):
    # return the appropriate func to choose weight by, and its initial weight
    # func is min/max, initial weight is sys.maxint/-sys.maxint (accordingly)

    d = {
        ("inc", "pos"): (min, INT_MAX),
        ("inc", "neg"): (max, INT_MIN),
        ("dec", "pos"): (max, INT_MIN),
        ("dec", "neg"): (min, INT_MAX)
    }

    assert dest_node.in_edges
    is_negative = any([e.weight < 0 for e in dest_node.in_edges])
    dest_positivity = "neg" if is_negative else "pos"
    return d[(dest_node.ar_type, dest_positivity)]
"""


def choose_weight_func(node, dest):
    """
    :param node: ARNode, current node
    :param dest: ARNode, destination node
    :return: the appropriate func to choose weight by, and its initial weight
    func is min/max, initial weight is sys.maxint/-sys.maxint (accordingly)
    """
    d = {
        ("inc", "pos"): (max, INT_MIN),
        ("inc", "neg"): (min, INT_MAX),
        ("dec", "pos"): (min, INT_MAX),
        ("dec", "neg"): (max, INT_MIN)
    }
    positivity = "neg" if ("neg" in node.name) else "pos"
    return d[(dest.ar_type, positivity)]


def net_from_nnet_file(nnet_filename):
    """
    generate Net instance which is equivalent to given nnet formatted network
    @nnet_filename fullpath of nnet file, see maraboupy.MarabouNetworkNNet
    under the root dir of git marabou project
    :return: Net object
    """
    #nnet_dir = "/home/yizhak/Research/Code/Marabou/maraboupy/examples/networks/"
    #nnet_dir = "/home/yizhak/Research/Code/MarabouApplications/acas/nnet/"
    # filename = "ACASXU_TF12_run3_DS_Minus15_Online_tau0_pra1_200Epochs.nnet"
    #filename = "ACASXU_run2a_1_1_batch_2000.nnet"
    #nnet_filename = os.path.join(nnet_dir, filename)

    # read nnetfile into Marabou Network
    acasxu_net = mnn.MarabouNetworkNNet(filename=nnet_filename)

    # list of layers, layer include list of nodes, node include list of Edge-s
    # notice the acasxu.weights include weights of input edges
    edges = []  # list of list of list of Edge instances
    # edges[i] is list of list of edges between layer i to layer i+1
    # edges[i][j] is list of out edges from node j in layer i
    # edges[i][j][k] is the k'th out edge of "node" j in "layer" i
    for i in range(len(acasxu_net.layerSizes)-1):
        # add cell for each layer, includes all edges in that layer
        edges.append([])
        for j in range(len(acasxu_net.weights[i])):
            # add cell for each node, includes all edges in that node
            edges[i].append([])
            for k in range(len(acasxu_net.weights[i][j])):
                # acasxu.weights include input edges, so the edge is from the
                # k'th node in layer i to the j'th node in layer i+1
                src = "x_{}_{}".format(i,k)
                dest = "x_{}_{}".format(i+1,j)
                weight = acasxu_net.weights[i][j][k]
                edge = Edge(src=src, dest=dest, weight=weight)
                edges[i][j].append(edge)
                # print(i,j,k,mn.weights[i][j][k])

    # validate sizes
    assert(len(acasxu_net.weights) == len(edges))
    for i in range(len(acasxu_net.layerSizes)-1):
        assert(len(acasxu_net.weights[i]) == len(edges[i]))
        for j in range(len(acasxu_net.weights[i])):
            assert(len(acasxu_net.weights[i][j]) == len(edges[i][j]))
            for k in range(len(acasxu_net.weights[i][j])):
                if acasxu_net.weights[i][j][k] != edges[i][j][k].weight:
                    print("wrong edges: {},{},{}".format(i,j,k))
                    assert False

    nodes = []  # list of list of ARNode instances
    # nodes[i] is list of nodes in layer i
    # nodes[i][j] is ARNode instance of node j in layer i
    name2node_map = {}  # map name to arnode instance, for adding edges later
    for i,layer in enumerate(edges):
        nodes.append([])  # add layer
        for j,node in enumerate(layer):
            for k,edge in enumerate(node):
                src_name = edge.src
                if src_name not in name2node_map.keys():
                    src_arnode = ARNode(name=src_name,
                                        ar_type=None,
                                        in_edges=[],
                                        out_edges=[],
                                        activation_func=relu,
                                        bias=0.0  # assigned later
                                       )
                    nodes[i].append(src_arnode)
                    name2node_map[src_name] = src_arnode

    # add output layer nodes
    # equal to add the next line
    # layer = acasxu_net.weights[len(acasxu_net.layerSizes)-1]
    nodes.append([])
    for j,node in enumerate(layer):
        for k,edge in enumerate(node):
            dest_name = edge.dest
            if dest_name not in name2node_map.keys():
                dest_arnode = ARNode(name=dest_name,
                                     ar_type=None,
                                     in_edges=[],
                                     out_edges=[],
                                     activation_func=relu,
                                     bias=0.0  # assigned later
                                    )
                nodes[i+1].append(dest_arnode)
                name2node_map[dest_name] = dest_arnode

    # after all nodes instances exist, add input and output edges
    for i,layer in enumerate(edges):  # layer is list of list of edges
        for j,node in enumerate(layer):  # node is list of edges
            for k,edge in enumerate(node):  # edge is Edge instance
                #print (i,j,k)
                src_node = name2node_map[edge.src]
                dest_node = name2node_map[edge.dest]
                src_node.out_edges.append(edge)
                dest_node.in_edges.append(edge)

    #TODO create nodes using the edges
    layers = []
    for i,layer in enumerate(nodes):
        if i == 0:
            type_name = "input"
        elif i == len(acasxu_net.layerSizes) - 1:
            type_name = "output"
        else:
            type_name = "hidden"
        layers.append(Layer(type_name=type_name, nodes=nodes[i]))

    for i, biases in enumerate(acasxu_net.biases):
        layer = layers[i + 1]
        for j, node in enumerate(layer.nodes):
            node.bias = biases[j]

    #TODO create layers using the nodes
    net = Net(layers=layers, weights=acasxu_net.weights, biases=acasxu_net.biases, acasxu_net=acasxu_net)

    # for i,biases in enumerate(acasxu_net.biases):
        # layer = net.layers[i+1]
        # for j,node in enumerate(layer.nodes):
        #     node.bias = biases[j]
    return net


# data structures
class Net:
    def __init__(self, layers, weights=None, biases=None, acasxu_net=None):
        self.layers = layers
        # map from name of node to its instance
        self.generate_name2node_map()
        # save first map to extract valid refinements of specific test_abstraction
        self.initial_name2node_map = copy.deepcopy(self.name2node_map)
        #self.visualize(title="init")
        self.weights = self.generate_weights() if weights is None else weights
        self._biases = self.generate_biases() if biases is None else biases
        self.biases = self.generate_biases()
        self.acasxu_net = acasxu_net

    def generate_weights(self):
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

    def generate_biases(self):
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

    def __eq__(self, other, verbose=VERBOSE):
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

    def visualize(self, title="figure 1", next_layer_part2union=None,
                  out_image_path=None, debug=False):
        nn = nx.Graph()
        # adding nodes
        for i,layer in enumerate(self.layers):
            color = LAYER_TYPE_NAME2COLOR[layer.type_name]
            for j,node in enumerate(layer.nodes):
                nn.add_node(node.name,
                            pos=(i*LAYER_INTERVAL, j*NODE_INTERVAL),
                            label=ar_type2sign.get(node.ar_type, node.name),
                            color=node.name, # color,
                            style="filled",
                            fillcolor=color)
        # adding out_edges (no need to iterate over output layer)
        for i,layer in enumerate(self.layers[:-1]):
            for j,node in enumerate(layer.nodes):
                for edge in node.out_edges:
                    if next_layer_part2union is None or edge.dest not in next_layer_part2union:
                        dest = edge.dest
                    else:
                        dest = next_layer_part2union[edge.dest]
                    visual_weight = round(edge.weight, VISUAL_WEIGHT_CONST)
                    nn.add_edge(edge.src, dest, weight=visual_weight)

        pos = nx.get_node_attributes(nn, 'pos')
        node_labels = nx.get_node_attributes(nn, 'label')
        node_colors = nx.get_node_attributes(nn, 'fillcolor').values()
#        node_colors = sorted(node_colors.values(), key=sorting_color_key)

        weights = nx.get_edge_attributes(nn,'weight')

        if debug:
            print("visualize() debug=True")
            import IPython
            IPython.embed()

        #nx.draw_networkx(nn,pos,edge_labels=labels, label_pos=0.8)
        plt.figure(title)
        # to set title inside the figure
        #nx.draw_networkx(nn,pos, title=title)
        nx.draw(nn,pos, title=title)
        nx.draw_networkx_nodes(nn,pos, node_color=node_colors)
        nx.draw_networkx_labels(nn,pos,labels=node_labels,node_color=node_colors)
        nx.draw_networkx_edge_labels(nn,pos,edge_labels=weights, label_pos=0.75)
        if out_image_path is not None:
            plt.savefig(out_image_path)
        else:
            plt.show()

    def layer_index2layer_size(self):
        return {i:len(self.layers[i].nodes) for i in range(len(self.layers))}

    def get_general_net_data(self):
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

    def get_variable2layer_index(self, variables2nodes):
        variable2layer_index = {}
        node2layer_map = self.get_node2layer_map()
        for variable, node in variables2nodes.items():
            if node.endswith("_b") or node.endswith("_f"):
                node = node[:-2]  # remove suffix
            variable2layer_index[variable] = node2layer_map[node]
        return variable2layer_index

    def evaluate(self, input_values):
        #print("input_values={}".format(input_values.items()))
        nodes2variables, variables2nodes = self.get_variables()
        # variable2layer_index = self.get_variable2layer_index(variables2nodes)
        cur_node2val = {}
        for node in self.layers[0].nodes:
            var = nodes2variables[node.name]
            cur_node2val[node.name] = input_values[var]
        for i, cur_layer in enumerate(self.layers[1:]):
            # print("evaluate():\t", i, cur_node2val.items())
            prev_node2val = cur_node2val
            layer_index = i+1
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
                #print("layer_index={}, cur_layer is not output layer".format(layer_index))
                #print("before activation, cur_node2val.items() = {}".format(cur_node2val.items()))
                activation_vals = {}
                for k,v in cur_node2val.items():
                    activation_func = self.name2node_map[k].activation_func
                    activation_vals[k + "_f"] = activation_func(v)
                cur_node2val.update(activation_vals)
                #print("after activation, cur_node2val.items() = {}".format(cur_node2val.items()))

            #cur_values = layer.evaluate(cur_values, nodes2variables, next,
            #                            variables2nodes, variable2layer_index)
        return cur_node2val

    def speedy_evaluate(self, input_values):
        assert self.weights is not None
        assert self.biases is not None
        input_list = [v for (k,v) in sorted(input_values.items(), key=lambda x: x[0])]
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

    def does_property_holds(self, test_property, output, variables2nodes):
        """
        returns if output is valid a.t. test_property output bounds
        @output - dict from node_name to value
        @test_property - see get_query() method documentation

        # for the reader, e.g:
        output
        {'x_7_0': 2.4388628607018603,
         'x_7_1': 0.1540985927922452,
         'x_7_2': 0.44519896688589594,
         'x_7_3': 1.0787132557294057,
         'x_7_4': 0.49820921502029597}

        output_vars
        [(605, 'x_7_0'),
         (606, 'x_7_1'),
         (607, 'x_7_2'),
         (608, 'x_7_3'),
         (609, 'x_7_4')]

        test_property["output"][0]
        (0, {'Upper': 3.9911256459})

        test_property["output"][0][0]
        0

        output_vars[test_property["output"][0][0]]
        (605, 'x_7_0')

        output_vars[test_property["output"][0][0]][1]
        'x_7_0'

        output[output_vars[test_property["output"][0][0]][1]]
        2.4388628607018603
        """
        output_property = test_property["output"]
        sorted_vars = sorted(variables2nodes.items(), key=lambda x: x[0])
        output_size = len(self.layers[-1].nodes)
        # couples of (variable index, node name) of output layer only
        output_vars = sorted_vars[-len(self.layers[-1].nodes):]
        for variable, bounds in output_property:
            index_name = output_vars[variable]
            node_name = index_name[1]
            if "Lower" in bounds.keys():
                lower_bound = bounds["Lower"]
                if lower_bound > output[node_name]:
                    return False
            if "Upper" in bounds.keys():
                upper_bound = bounds["Upper"]
                if output[node_name] > upper_bound:
                    return False
        return True

    def generate_name2node_map(self):
        name2node_map = {}
        for layer in self.layers:
            for node in layer.nodes:
                name2node_map[node.name] = node
        self.name2node_map = name2node_map

    def preprocess(self):
        self.preprocess_split_pos_neg()
        self.preprocess_split_inc_dec()
        if VERBOSE:
            debug_print("after preprocess")
            print(self)
            #self.visualize(title="after preprocess")
        self.orig_layers = copy.deepcopy(self.layers)
        self.orig_name2node_map = copy.deepcopy(self.name2node_map)
        self.weights = self.generate_weights()
        self.biases = self.generate_biases()

    def preprocess_split_pos_neg(self):
        """
        split net nodes to nodes with only positive/negative out edges
        preprocess all hidden layers (from last to first), then adjust input
        layer
        """
        #debug_print("preprocess_split_pos_neg()")
        orig_input_layer = copy.deepcopy(self.layers[0])
        for i in range(len(self.layers)-2, FIRST_POS_NEG_LAYER, -1):
            self.layers[i].split_pos_neg(self.name2node_map)
        #splited_input_layer = self.layers[0]
        #for node in orig_input_layer.nodes:
        #    node.out_edges = []
        #    for splitted_node in splited_input_layer:
        #        if splitted_node.name[:-4] == node.name:  # suffix=_oos/_neg
        #            edge = Edge(src=node.name, dest=splitted_node.name, weight=1.0)
        #            node.out_edges.append(edge)
        #            splitted_node.in_edges.append(edge)
        self.generate_name2node_map()
        #print(self)
        self.adjust_layer_after_split_pos_neg(layer_index=FIRST_POS_NEG_LAYER)

    def preprocess_split_inc_dec(self):
        """
        split net nodes to increasing/decreasing nodes
        preprocess all layers except input layer (from last to first)
        """
        if VERBOSE:
            debug_print("preprocess_split_inc_dec()")
        for i in range(len(self.layers)-1, FIRST_INC_DEC_LAYER, -1):
            self.layers[i].split_inc_dec(self.name2node_map)
        self.generate_name2node_map()
        self.adjust_layer_after_split_inc_dec(layer_index=FIRST_INC_DEC_LAYER)
        if VERBOSE:
            debug_print("after preprocess_split_inc_dec()")
            print(self)

    def adjust_input_layer_after_abstraction(self):
        """
        the test_abstraction works layer excluding the input layer, therefore the
        names of the out edges of input layer nodes have to be updated.
        this function fix the names of output edges of the first layer to be
        the updated names of union nodes of hidden layer 1 after test_abstraction
        """
        pass

    def adjust_layer_after_split_pos_neg(self, layer_index=FIRST_POS_NEG_LAYER):
        #debug_print("adjust_layer_after_split_pos_neg")
        cur_layer = self.layers[layer_index]
        next_layer = self.layers[layer_index+1]
        for node in cur_layer.nodes:
            node.new_out_edges = []
        for next_node in next_layer.nodes:
            next_node.new_in_edges = []
            for cur_node in cur_layer.nodes:
                for out_edge in cur_node.out_edges:
                    for suffix in ["", "_pos", "_neg"]:
                        if out_edge.dest + suffix == next_node.name:
                            weight = out_edge.weight
                            edge = Edge(cur_node.name, next_node.name, weight)
                            cur_node.new_out_edges.append(edge)
                            next_node.new_in_edges.append(edge)
            next_node.in_edges = next_node.new_in_edges
            del next_node.new_in_edges
        for node in cur_layer.nodes:
            node.out_edges = node.new_out_edges
            del node.new_out_edges
        if VERBOSE:
            debug_print("after adjust_layer_after_split_pos_neg()")
            print(self)

        """
        TODO: remove this code that (not yet) sum edges of same src,dest to one
        # union multiple edges with same dest to one edge with sum of weights
        src_dest2edges = {}
        for node in cur_layer.nodes:
            for out_edge in cur_node.out_edges:
                src_dest2edges.setdefault((edge.src,edge.dest), []).append(edge)
        new_edges = []
        for src_dest, edges in dest2edges.items():
            src,dest = src_dest
            new_edge = Edge(src=src, weight=0.0, dest=dest)
            for edge in edges:
                new_edge.weight += edge.weight
        # TODO assign the new edges as the in/out edges of cur/next layer
        """

    def adjust_layer_after_split_inc_dec(self, layer_index=FIRST_INC_DEC_LAYER):
        cur_layer = self.layers[layer_index]
        next_layer = self.layers[layer_index+1]
        for cur_node in cur_layer.nodes:
            cur_node.new_out_edges = []
        for next_node in next_layer.nodes:
            next_node.new_in_edges = []
            parts = next_node.name.split("+")
            for part in parts:
                for cur_node in cur_layer.nodes:
                    for out_edge in cur_node.out_edges:
                        for suffix in ["_inc", "_dec"]:
                            if (out_edge.dest + suffix) == part:
                                weight = out_edge.weight
                                edge = Edge(cur_node.name,
                                            next_node.name,
                                            weight)
                                cur_node.new_out_edges.append(edge)
                                next_node.new_in_edges.append(edge)
            next_node.in_edges = next_node.new_in_edges
            del next_node.new_in_edges
        for node in cur_layer.nodes:
            node.out_edges = node.new_out_edges
            del node.new_out_edges

    def fix_prev_layer_out_edges_dests(self, prev_layer_index, updated_names):
        """
        fix the dest names of out edges of nodes in a layer to the names of the
        updated node names in the next layer
        used to update previous layer edges dest names after changing a layer
        (after both test_abstraction and test_refinement)
        @layer index index of layer in self.layers
        @updated_names dictionary from previous name to current name
        """
        layer = self.layers[prev_layer_index]
        for node in layer.nodes:
            for out_edge in node.out_edges:
                out_edge.dest = updated_names.get(out_edge.dest, out_edge.dest)

    def fix_prev_layer_min_max_edges(self, prev_layer_index, updated_names):
        prev_layer = self.layers[prev_layer_index]
        cur_layer = self.layers[prev_layer_index+1]
        node2dest_edges = {node.name: {} for node in prev_layer.nodes}

        # calculate min/max edges from prev layer's nodes to their dest nodes
        for node in prev_layer.nodes:
            for out_edge in node.out_edges:
                dest = self.name2node_map[out_edge.dest]
                max_min_func, cur_weight = choose_weight_func(node, dest)
                node2dest_edges[node.name].setdefault(dest.name, cur_weight)
                cur_weight = node2dest_edges[node.name][dest.name]
                node2dest_edges[node.name][dest.name] = max_min_func(cur_weight,
                                                                     out_edge.weight)
        # append min/max edges only as new edges from node to each dest
        # the strange names a_node,a_dest were chosen because for some reason
        # the intepreter confused when using the name "node"
        for a_node in node2dest_edges:
            for a_dest in node2dest_edges[a_node]:
                new_edge = Edge(a_node, a_dest, node2dest_edges[a_node][a_dest])
                dest_node = self.name2node_map[a_dest]
                try:
                    dest_node.new_in_edges.append(new_edge)
                except:
                    dest_node.new_in_edges = [new_edge]
                node = self.name2node_map[a_node]
                try:
                    node.new_out_edges.append(new_edge)
                except:
                    node.new_out_edges = [new_edge]

        # update edges to new edges
        for node in prev_layer.nodes:
            if hasattr(node, "new_out_edges"):
                node.out_edges = node.new_out_edges
                del node.new_out_edges
        for node in cur_layer.nodes:
            if hasattr(node, "new_in_edges"):
                node.in_edges = node.new_in_edges
                del node.new_in_edges

    """
    def fix_prev_layer_min_max_edges(self, prev_layer_index, updated_names):
        prev_layer = self.layers[prev_layer_index]
        cur_layer = self.layers[prev_layer_index+1]
        dest2node_edges = {dest: {} for dest in cur_layer.nodes}

        # get a dictionary {dest: min/max weight from specific part to it}
        for part in node_parts:
            for edge in name2node_map[part].out_edges:
                dest = dest_part2dest[edge.dest]
                dest_node = name2node_map[dest]
                max_min_func, cur_weight = choose_weight_func(node, dest_node)
                cur_weight = dest2part_weights[dest].get(part, cur_weight)
                dest2part_weights[dest][part] = max_min_func(cur_weight,
                                                             edge.weight)

        for node in prev_layer.nodes:
            for out_edge in node.out_edges:
                dest = self.name2node_map[part2union[dest]]
                dest2part_edges[dest.name].setdefault(node.name, {})
                dest2part_edges[dest.name][node.name]
        for dest, node2edges in dest2part_edges.items()
            src = self.name2node_map[node]
            max_min_func, cur_weight = choose_weight_func(src, dest)
            weight = max_min_func([e.weight for e in edges])
            new_edge = Edge(src.name, dest.name, weight)
            dest.new_in_edges.append(new_edge)
            try:
                src.new_out_edges.append(new_edge)
            except:
                src.new_out_edges = [new_edge]
        for node in prev_layer:
            node.out_edges = node.new_out_edges
            del node.new_out_edges
        for node in cur_layer:
            node.in_edges = node.new_in_edges
            del node.new_in_edges
    """

    def fix_cur_layer_in_edges(self, layer_index):
        assert layer_index != 0
        layer = self.layers[layer_index]
        for node in layer.nodes:
            node.in_edges = []
        prev_layer = self.layers[layer_index-1]
        for node in prev_layer.nodes:
            for out_edge in node.out_edges:
                # out_edge is an in_edge of its dest node
                self.name2node_map[out_edge.dest].in_edges.append(out_edge)

    def abstract(self, do_preprocess=True, visualize=False, verbose=VERBOSE):
        if VERBOSE:
            debug_print("original net:")
            print(self)
        if do_preprocess:
            self.preprocess()
        #if VERBOSE:
        #    debug_print("net after preprocessing:")
        #    print(self)
        #    self.visualize(title="after preprocess")
        next_layer_part2union = {}
        for i in range(len(self.layers)-1, FIRST_ABSTRACT_LAYER-1, -1):
            layer = self.layers[i]
            next_layer_part2union = layer.abstract(self.name2node_map,
                                                   next_layer_part2union)
            # update name2node_map - add new union nodes and remove inner nodes
            # removal precedes addition for equal names case (e.g output layer)
            for part in next_layer_part2union.keys():
                del self.name2node_map[part]
            self.generate_name2node_map()
            #print (i)
            if visualize:
                self.visualize(title="after layer {} test_abstraction".format(i),
                               next_layer_part2union=next_layer_part2union,
                               debug=False)
            if verbose:
                debug_print("net after abstract {}'th layer:".format(i))
                print(self)
        self.finish_abstraction(next_layer_part2union, verbose=verbose)
        return self

    def finish_abstraction(self, next_layer_part2union, verbose=VERBOSE):
        # fix input layer edges dest names (to fit layer 1 test_abstraction results)
        self.fix_prev_layer_out_edges_dests(FIRST_ABSTRACT_LAYER-1, next_layer_part2union)
        if verbose:
            debug_print("net after fix_prev_layer_out_edges_dests")
            print(self)
        self.fix_prev_layer_min_max_edges(FIRST_ABSTRACT_LAYER-1, next_layer_part2union)
        if verbose:
            debug_print("net after fix_prev_layer_min_max_edges")
            print(self)
        self.fix_cur_layer_in_edges(layer_index=FIRST_ABSTRACT_LAYER)
        self.biases = self.generate_biases()
        self.weights = self.generate_weights()
        if verbose:
            debug_print("net after fix_cur_layer_in_edges")
            print(self)
        if verbose:
            debug_print("finish_abstraction finished")

    def union_couple_of_nodes(self, node_1, node_2):
        assert node_1.ar_type == node_2.ar_type
        try:
            layer_index = int(node_1.name.split("_")[1])
        except:
            print("union_couple_of_nodes - except")
            import IPython
            IPython.embed()
        layer = self.layers[layer_index]
        next_layer = self.layers[layer_index+1]
        prev_layer = self.layers[layer_index-1]
        sum_biases = node_1.bias + node_2.bias
        if node_1.ar_type is None:
            print("node_1.ar_type is None")
            import IPython
            IPython.embed()
        union_node = ARNode(name="+".join([node_1.name, node_2.name]),
                            ar_type=node_1.ar_type,
                            activation_func=node_1.activation_func,
                            in_edges=[],
                            out_edges=[],
                            bias=sum_biases
                           )
        for next_layer_node in next_layer.nodes:
            group_a = union_node.name.split("+")
            group_b = next_layer_node.name.split("+")
            out_edge_weight = self.calculate_weight_of_edge_between_two_part_groups(group_a, group_b)
            if out_edge_weight is not None:
                out_edge = Edge(union_node.name, next_layer_node.name, out_edge_weight)
                union_node.out_edges.append(out_edge)
                next_layer_node.in_edges.append(out_edge)
        for prev_layer_node in prev_layer.nodes:
            group_a = prev_layer_node.name.split("+")
            group_b = union_node.name.split("+")
            in_edge_weight = self.calculate_weight_of_edge_between_two_part_groups(group_a, group_b)
            if in_edge_weight is not None:
                in_edge = Edge(prev_layer_node.name, union_node.name, in_edge_weight)
                union_node.in_edges.append(in_edge)
                prev_layer_node.out_edges.append(in_edge)
        layer.nodes.append(union_node)
        self.remove_node(node_2, layer_index)
        self.remove_node(node_1, layer_index)
        self.generate_name2node_map()

    def get_limited_random_input(self, test_property):
        random_input = {}
        for i in range(len(self.layers[0].nodes)):
            bounds = test_property["input"][i][1]
            random_input[i] = random.uniform(bounds["Lower"], bounds["Upper"])
        return random_input

    def get_limited_random_inputs(self, test_property):
        """

        :param test_property: dictionary with lower/upper bound for each variable
        :return:
        """
        random_inputs = [self.get_limited_random_input(test_property) for j in range(100)]
        return random_inputs

    def has_violation(self, test_property, inputs):
        _, variables2nodes = self.get_variables()
        for x in inputs:
            output = self.evaluate(x)
            speedy_output = self.speedy_evaluate(x)
            assert is_evaluation_result_equal(output.items(), speedy_output)
            # TODO: rename does_property_holds -> is_satisfying_assignment
            if self.does_property_holds(test_property, output, variables2nodes):
                return True  # if x is a satisfying assignment, there is a violation, because we want UNSAT
        return False

    # def heuristic_abstract_orig(self, test_property, do_preprocess=True):
    #     # first heuristic - choose the best test_abstraction pair in every iteration
    #     # this function is deprecated, can be replaced by calling:
    #     # heuristic_abstract_orig(self, test_property, do_preprocess, sequence_length=1)
    #     if do_preprocess:
    #         self.preprocess()
    #     random_inputs = self.get_limited_random_inputs(test_property)
    #     nodes2edge_between_map = self.get_nodes2edge_between_map()
    #     best_pair = [None, None]  # set initial value to verify that the loop's condition holds in the first iteration
    #     while not self.has_violation(test_property, random_inputs) and best_pair is not None:
    #         # the minimal value over all max differences between input edges' weights of couples of nodes of same ar_type.
    #         min_max_diff = INT_MAX
    #         best_pair = None
    #         for i, layer in enumerate(self.layers[FIRST_ABSTRACT_LAYER:-1]):
    #             layer_index = i + 1
    #             prev_layer = self.layers[layer_index - 1]
    #             layer_couples = layer.get_couples_of_same_ar_type()
    #             for n1, n2 in layer_couples:
    #                 max_diff_n1_n2 = INT_MIN
    #                 for prev_node in prev_layer.nodes:
    #                     in_edge_n1 = nodes2edge_between_map.get((prev_node.name, n1.name), None)
    #                     a = 0 if in_edge_n1 is None else in_edge_n1.weight
    #                     in_edge_n2 = nodes2edge_between_map.get((prev_node.name, n2.name), None)
    #                     b = 0 if in_edge_n2 is None else in_edge_n2.weight
    #                     if abs(a-b) > max_diff_n1_n2:
    #                         max_diff_n1_n2 = abs(a-b)
    #                 if max_diff_n1_n2 < min_max_diff:
    #                     min_max_diff = max_diff_n1_n2
    #                     best_pair = (n1, n2)
    #                     print("update: best_pair = ({})".format((n1.name, n2.name)))
    #         if best_pair is not None:
    #             union_name = "+".join([best_pair[0].name, best_pair[1].name])
    #             self.union_couple_of_nodes(best_pair[0], best_pair[1])
    #             self.finish_abstraction({best_pair[0].name: union_name,
    #                                      best_pair[1].name: union_name})
    #     return self

    def heuristic_abstract(self, test_property, do_preprocess=True, sequence_length=50):
        if do_preprocess:
            self.preprocess()
        random_inputs = self.get_limited_random_inputs(test_property)
        nodes2edge_between_map = self.get_nodes2edge_between_map()
        union_pairs = None  # set initial value to verify that the loop's condition holds in the first iteration
        # print("heuristic test_abstraction: starting loop")
        loop_iterations = 0
        while not self.has_violation(test_property, random_inputs) and union_pairs != []:
            loop_iterations += 1
            # print("heuristic test_abstraction: loop_iteration #{}".format(loop_iterations))
            # the minimal value over all max differences between input edges' weights of couples of nodes of same ar_type.
            min_max_diff = INT_MAX
            union_pairs = []
            for i, layer in enumerate(self.layers[FIRST_ABSTRACT_LAYER:-1]):
                layer_index = i + 1
                prev_layer = self.layers[layer_index - 1]
                layer_couples = layer.get_couples_of_same_ar_type()
                for n1, n2 in layer_couples:
                    max_diff_n1_n2 = INT_MIN
                    for prev_node in prev_layer.nodes:
                        in_edge_n1 = nodes2edge_between_map.get((prev_node.name, n1.name), None)
                        a = 0 if in_edge_n1 is None else in_edge_n1.weight
                        in_edge_n2 = nodes2edge_between_map.get((prev_node.name, n2.name), None)
                        b = 0 if in_edge_n2 is None else in_edge_n2.weight
                        if abs(a-b) > max_diff_n1_n2:
                            max_diff_n1_n2 = abs(a-b)
                    union_pairs.append(([n1, n2], max_diff_n1_n2))
            if union_pairs:  # if union_pairs != []:
                # print("len(union_pairs)={}".format(len(union_pairs)))
                best_sequence_pairs = sorted(union_pairs, key=lambda x: x[1])
                cur_abstraction_seq_len = 0
                for pair, diff in best_sequence_pairs:
                    if cur_abstraction_seq_len >= sequence_length:
                        break
                    if pair[0].name not in self.name2node_map or pair[1].name not in self.name2node_map:
                        continue
                    cur_abstraction_seq_len += 1
                    union_name = "+".join([pair[0].name, pair[1].name])
                    self.union_couple_of_nodes(pair[0], pair[1])
                    self.finish_abstraction({pair[0].name: union_name,
                                             pair[1].name: union_name})
        return self

    def refine(self, sequence_length=1, visualize=False, **kws):
        # use huristics to get best test_refinement sequence
        r_sequence = self.get_refinement_sequence(sequence_len=sequence_length, **kws)
        print("best {} test_refinement steps are: {}".format(sequence_length,
                                                        r_sequence))
        # given test_refinement sequence, actually do test_refinement
        for part in r_sequence:
            self.split_back(part)
            #debug_print("after split_back({})".format(part))
            if visualize:
                self.visualize(title="after refine {}".format(part))
        self.biases = self.generate_biases()
        self.weights = self.generate_weights()
        return self

    def calculate_weight_of_edge_between_two_part_groups(self, group_a, group_b):
        """
        calculate the weight of the edge from the union_node of group @group_a to group @group_b
        :param group_a: list of parts in i'th layer(each part is a name of node in the original network after preprocessing)
        :param group_b: list of parts in (i+1)'th layer (each part is a name of node in the original network after preprocessing)
        :return: the weight between union node of @group_a and union node of @group_b
        """
        assert group_a
        assert group_b

        # dict from couple of parts, first from group_a and second from consecutive layer's node, to the edge in between
        a_part_dest2edge = {}
        for a_part in group_a:
            orig_a_part_node = self.orig_name2node_map[a_part]
            for out_edge in orig_a_part_node.out_edges:
                a_part_dest2edge[(a_part, out_edge.dest)] = out_edge

        # stage 1: calculate dictionary of weights from each part of @group_a to group_b
        # done by choosing the right max/min function and iterate over all parts of group_b
        a_part2weight = {}  # stores for each part in group_a the max/min value from it to group_b

        a0_node = self.orig_name2node_map[group_a[0]]
        b0_node = self.orig_name2node_map[group_b[0]]
        max_min_func, initial_weight = choose_weight_func(a0_node, b0_node)

        for a_part in group_a:
            for b_part in group_b:
                w1 = a_part2weight.get(a_part, initial_weight)
                w2 = a_part_dest2edge.get((a_part, b_part), None)
                if w2 is not None:
                    a_part2weight[a_part] = max_min_func(w1, w2.weight)

        # stage 2: summing the weights from all parts of @group_a to @group_b to be the output weight
        a2b_weight = None
        for a_part in group_a:
            val = a_part2weight.get(a_part, None)
            if val is not None:
                if a2b_weight is None:
                    a2b_weight = 0.0
                a2b_weight += val
        # note that the returned weight is 0 in two cases: in case that the sum is 0 and in case of no edges at all
        return a2b_weight

    def split_back(self, part):
        # assume that layer_index is in [2, ..., L-1] (L = num of layers)
        try:
            layer_index = int(part.split("_")[1])
        except:
            import IPython
            IPython.embed()
        layer = self.layers[layer_index]
        next_layer = self.layers[layer_index+1]
        prev_layer = self.layers[layer_index-1]
        part2node_map = self.get_part2node_map()
        union_node = self.name2node_map[part2node_map[part]]
        parts = union_node.name.split("+")
        other_parts = [p for p in parts if p != part]
        if not other_parts:
            return
        part_node = ARNode(name=part,
                           ar_type=union_node.ar_type,
                           activation_func=union_node.activation_func,
                           in_edges=[],
                           out_edges=[],
                           bias=self.orig_name2node_map[part].bias
                          )
        other_parts_node = ARNode(name="+".join(other_parts),
                           ar_type=union_node.ar_type,
                           activation_func=union_node.activation_func,
                           in_edges=[],
                           out_edges=[],
                           bias=sum([self.orig_name2node_map[other_part].bias for other_part in other_parts])
                          )
        splitting_nodes = [part_node, other_parts_node]
        for splitting_node in splitting_nodes:
            #print("splitting_node.name={}".format(splitting_node.name))
            for next_layer_node in next_layer.nodes:
                group_a = splitting_node.name.split("+")
                group_b = next_layer_node.name.split("+")
                #print("call 1 - group_a")
                #print(group_a)
                #print("call 1 - group_b")
                #print(group_b)
                out_edge_weight = self.calculate_weight_of_edge_between_two_part_groups(group_a, group_b)
                if out_edge_weight is not None:
                    out_edge = Edge(splitting_node.name, next_layer_node.name, out_edge_weight)
                    splitting_node.out_edges.append(out_edge)
                    next_layer_node.in_edges.append(out_edge)
            for prev_layer_node in prev_layer.nodes:
                group_a = prev_layer_node.name.split("+")
                group_b = splitting_node.name.split("+")
                #print("call 2 - group_a")
                #print(group_a)
                #print("call 2 - group_b")
                #print(group_b)
                in_edge_weight = self.calculate_weight_of_edge_between_two_part_groups(group_a, group_b)
                if in_edge_weight is not None:
                    in_edge = Edge(prev_layer_node.name, splitting_node.name, in_edge_weight)
                    splitting_node.in_edges.append(in_edge)
                    prev_layer_node.out_edges.append(in_edge)
            #import IPython
            #IPython.embed()
            layer.nodes.append(splitting_node)
        self.remove_node(union_node, layer_index)
        self.generate_name2node_map()

    def remove_node(self, node, layer_index):
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
                for i,ie in enumerate(dest_node.in_edges):
                    if ie == out_edge:
                        break
                del(dest_node.in_edges[i])
            del out_edge
        # less effective
        # layer.nodes = [n for n in layer.nodes if n != node]
        for i,cur_node in enumerate(layer.nodes):
            if cur_node == node:
                break
        del layer.nodes[i]
        del node

    """
    def split_back(self, part):
        #debug_print("start split_back {}".format(part))
        orig_part_node = self.orig_name2node_map[part]
        part2node_map = self.get_part2node_map()
        union_node = self.name2node_map[part2node_map[part]]
        #print("split part {} from union {}".format(part, union_node.name))
        union_parts = union_node.name.split("+")
        assert part in union_parts
        if len(union_parts) == 1:
            # no parts except part
            return
        node2layer_map = self.get_node2layer_map()
        layer = self.layers[node2layer_map[union_node.name]]

        # define a dictionary from union_part and dest to edge in between
        # define a dictionary from src and union_part to edge in between
        union_part_dest2edge = {}
        union_part_src2edge = {}
        # fill dicts values (for each couple of nodes, assign an edge)
        for union_part in union_parts:
            orig_union_part_node = self.orig_name2node_map[union_part]
            for out_edge in orig_union_part_node.out_edges:
                union_part_dest2edge[(union_part, out_edge.dest)] = out_edge
            for in_edge in orig_union_part_node.in_edges:
                union_part_src2edge[(in_edge.src, union_part)] = in_edge

        # generate the original part node
        part_node = ARNode(name=part,
                           ar_type=union_node.ar_type,
                           activation_func=union_node.activation_func,
                           in_edges=[],
                           out_edges=[],
                           bias=orig_part_node.bias
                          )

        orig_union_name = union_node.name
        union_new_name = "+".join([up for up in union_parts if up != part])
        # since src/dest names are changed during splitting, in/out edges have
        # to be updated. updated_names_map is dict from old to new name of node
        updated_names_map = {orig_union_name:union_new_name}

        union_node.new_out_edges = []
        union_node.new_in_edges = []

        # assign out edges from part and from union without part
        # part_union2weight: {(union_part,union_dest)->min/max edge in between}
        part_union2weight = {}
        for union_part in union_parts:
            for out_edge in self.orig_name2node_map[union_part].out_edges:
                dest_part = out_edge.dest
                edge = union_part_dest2edge[(union_part, dest_part)]
                dest_union = self.name2node_map[part2node_map[dest_part]]
                max_min_func, cur_weight = choose_weight_func(union_node,
                                                              dest_union)
                cur_weight = part_union2weight.get(
                             (union_part, dest_union.name), cur_weight)
                part_union2weight[(union_part, dest_union.name)] = \
                                        max_min_func(cur_weight, edge.weight)
        # union_dest2weight: {union_dest->weight of edge from union_node
        union_dest2weight = {}
        # part_dest2weight: {union_dest->weight of edge from splitted_node
        part_dest2weight = {}
        for union_part in union_parts:
            # couple = (union_part, dest_union)
            for couple, weight in part_union2weight.items():
                if union_part == couple[0]:
                    if union_part == part:
                        part_dest2weight[couple[1]] = w
                    else:
                        w = union_dest2weight.get(couple[1], 0.0) + weight
                        union_dest2weight[couple[1]] = w
        
        #                w = part_dest2weight.get(couple[1], 0.0) + weight
        ## dict from part and dest union to weight of edge in between
        #part_dest2weight = {}
        ## dict from union and dest union to weight of edge in between
        #union_dest2weight = {}
        #union_positivity = union_node.get_positivity()
        #for out_edge in union_node.out_edges:
        #    union_dest = self.name2node_map[out_edge.dest]
        #    # TODO: maybe wrong change from this line to next
        #    #max_min_func, cur_weight = choose_weight_func(union_dest.ar_type, union_positivity)
        #    max_min_func, cur_weight = choose_weight_func(union_dest)
        #    part_dest2weight[union_dest.name] = cur_weight
        #    union_dest2weight[union_dest.name] = cur_weight
        #    union_dest_parts = union_dest.name.split("+")
        #    for dest_part in union_dest_parts:
        #        for union_part in union_parts:
        #            w2 = union_part_dest2edge.get((union_part, dest_part), None)
        #            if w2 is None:
        #                continue
        #            else:
        #                w2 = w2.weight
        #            if union_part == part:
        #                w1 = part_dest2weight[union_dest.name]
        #                cur_weight = max_min_func(w1, w2)
        #                part_dest2weight[union_dest.name] = cur_weight
        #            else:
        #                w1 = union_dest2weight[union_dest.name]
        #                cur_weight = max_min_func(w1, w2)
        #                union_dest2weight[union_dest.name] = cur_weight
        
        # actually update out_edges of part and union without part
        for out_edge in union_node.out_edges:
            c = 0
            if out_edge.dest in part_dest2weight:
                c += 1
                weight = part_dest2weight[out_edge.dest]
                part_out_edge = Edge(part, out_edge.dest, weight)
                part_node.out_edges.append(part_out_edge)
            if out_edge.dest in union_dest2weight:
                c += 1
                weight = union_dest2weight[out_edge.dest]
                union_out_edge = Edge(union_new_name, out_edge.dest, weight)
                union_node.new_out_edges.append(union_out_edge)
            #assert c > 0
            if c <= 0:
                print("c == 0 (a)")
                import IPython
                IPython.embed()

            # update the in_edges of union dest to include the new edges
            # union_dest = self.name2node_map[out_edge.dest]
            # notice that out_edge is in edge of union src
            # TODO: the following for loop is not a must, can be done in O(1).
            # if i had some mapping e.g. from edge name to edge, it was cheaper

        #    for splitted_edge in union_dest.in_edges:
        #        if out_edge == splitted_edge:  # same (src,dest) for both edges
        #            union_dest.in_edges.remove(splitted_edge)
        #            if weight is not None:
        #                union_dest.in_edges.append(part_out_edge)
        #            else:
        #                union_dest.in_edges.append(union_out_edge)
        
        union_node.out_edges = union_node.new_out_edges
        del union_node.new_out_edges

        dests = {}
        for out_edge in part_node.out_edges + union_node.out_edges:
            dests[out_edge.dest] = self.name2node_map[out_edge.dest]
            dests[out_edge.dest].new_in_edges = []
        for out_edge in union_node.out_edges:
            dests[out_edge.dest].new_in_edges.append(out_edge)
        for name,dest_node in dests.items():
            dest_node.update_in_edges(updated_names_map)
            #dest_node.in_edges = dest_node.new_in_edges
            #del dest_node.new_in_edges
        for out_edge in part_node.out_edges:
            dest_node = self.name2node_map[out_edge.dest]
            dest_node.in_edges.append(out_edge)

        # assign in edges to part and to union without part
        # part_src2weight: {(source_part,union_dest)->min/max edge in between}
        part_src2weight = {}
        for union_part in union_parts:
            for in_edge in self.orig_name2node_map[union_part].in_edges:
                src_part = in_edge.src
                edge = union_part_src2edge[(src_part, union_part)]
                src_union = self.name2node_map[part2node_map[src_part]]
                max_min_func, cur_weight = choose_weight_func(src_union,
                                                              union_node)
                cur_weight = part_src2weight.get(
                             (src_union.name, union_part), cur_weight)
                part_src2weight[(src_union.name, union_part)] = \
                                        max_min_func(cur_weight, edge.weight)
        # union_src2weight: {union_src->weight to union_node}
        union_src2weight = {}
        # splitted_src2weight: {union_src->weight to splitted_node}
        splitted_src2weight = {}
        for union_part in union_parts:
            # couple = (source_part, dest_union)
            for couple, weight in part_src2weight.items():
                if union_part == couple[1]:
                    if union_part == part:
                        w = splitted_src2weight.get(couple[0], 0.0) + weight
                        splitted_src2weight[couple[0]] = w
                    else:
                        w = union_src2weight.get(couple[0], 0.0) + weight
                        union_src2weight[couple[0]] = w

        ## dict from part and src union to weight of edge in between
        #part_src2weight = {}
        ## dict from union and src union to weight of edge in between
        #union_src2weight = {}
        #union_positivity = union_node.get_positivity()
        #for in_edge in union_node.in_edges:
        #    union_src = self.name2node_map[in_edge.src]
        #    # TODO: maybe wrong change from this line to next
        #    #max_min_func, cur_weight = choose_weight_func(part_node.ar_type, union_positivity)
        #    max_min_func, cur_weight = choose_weight_func(union_node)
        #    part_src2weight[union_src.name] = cur_weight
        #    union_src2weight[union_src.name] = cur_weight
        #    union_src_parts = union_src.name.split("+")
        #    for src_part in union_src_parts:
        #        for union_part in union_parts:
        #            w2 = union_part_src2edge.get((src_part, union_part), None)
        #            if w2 is None:
        #                continue
        #            else:
        #                w2 = w2.weight
        #            if union_part == part:
        #                w1 = part_src2weight[union_src.name]
        #                cur_weight = max_min_func(w1, w2)
        #                part_src2weight[union_src.name] = cur_weight
        #            else:
        #                w1 = union_src2weight[union_src.name]
        #                cur_weight = max_min_func(w1, w2)
        #                union_src2weight[union_src.name] = cur_weight
        
        # actually update in_edges of part and union without part
        union_node.new_in_edges = []
        for in_edge in union_node.in_edges:
            # counter validate that in_edge is in_edge of at least one of part_node, union_node
            c = 0
            weight = splitted_src2weight.get(in_edge.src, None)
            if weight is not None:
                c += 1
                part_in_edge = Edge(in_edge.src, part, weight)
                part_node.in_edges.append(part_in_edge)
            weight = union_src2weight.get(in_edge.src, None)
            if weight is not None:
                union_in_edge = Edge(in_edge.src, union_new_name,
                                      union_src2weight[in_edge.src])
                union_node.new_in_edges.append(union_in_edge)
                c += 1
            #assert c >= 1
            if c < 1:
                print("c == 0 (b)")
                import IPython
                IPython.embed()

            # update the out_edges of union src to include the new edges
            union_src = self.name2node_map[in_edge.src]
            # notice that in_edge is out edge of union src
            # TODO: the following for loop is not a must, can be done in O(1).
            # if i had some mapping e.g. from edge name to edge, it was cheaper
            
        #    for splitted_edge in union_src.out_edges:
        #        if in_edge == splitted_edge:
        #            union_src.out_edges.remove(splitted_edge)
        #            if weight is not None:
        #                union_src.out_edges.append(part_in_edge)
        #            else:
        #                union_src.out_edges.append(union_in_edge)
        
                # update the out_edges of union src to include the new edges

        union_node.in_edges = union_node.new_in_edges
        del union_node.new_in_edges

        srcs = {}
        for in_edge in part_node.in_edges + union_node.in_edges:
            srcs[in_edge.src] = self.name2node_map[in_edge.src]
            srcs[in_edge.src].new_out_edges = []
        for in_edge in union_node.in_edges:
            srcs[in_edge.src].new_out_edges.append(in_edge)
        #if part == "x_3_22_pos_inc":
        #    import IPython
        #    IPython.embed()
        for name,src_node in srcs.items():
            src_node.update_out_edges(updated_names_map)
            #src_node.out_edges = src_node.new_out_edges
            #del src_node.new_out_edges
        for in_edge in part_node.in_edges:
            src_node = self.name2node_map[in_edge.src]
            src_node.out_edges.append(in_edge)


        #union_src.new_out_edges = []
        #for in_edge in part_node.in_edges + union_node.in_edges:
        #    union_src.new_out_edges.append(in_edge)
        #union_dest.out_edges = union_dest.new_out_edges
        #del union_node.new_out_edges
        
        
        # assign max_min bias to union_node without splitted nodes
        orig_union_parts_nodes = [self.orig_name2node_map[up] for up in union_parts if up != part]
        union_node.bias = max_min_func([n.bias for n in orig_union_parts_nodes])

        layer.nodes.append(part_node)
        union_node.name = union_new_name
        layer_index = node2layer_map[orig_union_name]
        self.generate_name2node_map()
        self.fix_prev_layer_out_edges_dests(layer_index-1, updated_names_map)
        #print("aaa")
        #import IPython
        #IPython.embed()
        self.fix_cur_layer_in_edges(layer_index)
        """

    def get_refinement_sequence(self, sequence_len=1, **kws):
        """
        huristic function, guess the best test_refinement sequence
        1) a.t. CETAR method, choose max-lossy parts:
        on each iteration, the "most lossy" part in union node will be splitted
        2) a.t. CEGAR method, choose max-lossy parts a.t. counter example:
        use the values of network nodes in a counter example as multipliers of
        the "loss" of each part in the CETAR method
        run example on network and split the part which most changed the result

        @sequence_len - indicates the number of top test_refinement steps to return
        @kws - dict that include "example" key in case of h2 and None otherwise
        @example - marabou's counter example (node names instead of variables)
        """
        #if kws is not None and "example" in kws.keys():
            # h2 - cegar method
        #    part2loss_map = self.get_part2example_change(example)
        #else:
            # h1 - max-loss method
        #    part2loss_map = self.get_part2loss_map()
        if kws is not None and "example" in kws.keys():
            # h2 - cegar method
            example = kws.get("example")
        else:
            # h1 - max-loss method
            example = {}
        part2loss_map = self.get_part2loss_map(example=example)
        top_part_loss = sorted(part2loss_map.items(),
                               key=lambda x:x[1],
                               reverse=True
                              )[:sequence_len]
        # p2l stands for "part2loss", pl[0] is part name
        return [p2l[0] for p2l in top_part_loss]

    def get_part2loss_map(self, example={}):
        part2loss = {}
        nodes2edge_between_map = self.get_nodes2edge_between_map()
        part2node_map = self.get_part2node_map()
        for layer in self.layers[2:]:
            layer_part2loss_map = \
            self.get_layer_part2loss_map(layer,
                                         self.name2node_map,
                                         self.orig_name2node_map,
                                         part2node_map,
                                         nodes2edge_between_map,
                                         example)
            part2loss.update(layer_part2loss_map)
        return part2loss

    def get_nodes2edge_between_map(self):
        nodes2edge_between_map = {}
        for layer in self.layers:
            for node in layer.nodes:
                for edge in node.out_edges:
                    nodes2edge_between_map[(edge.src, edge.dest)] = edge
        return nodes2edge_between_map

    def get_part2node_map(self):
        part2node_map = {}
        for layer in self.layers:
            for node in layer.nodes:
                parts = node.name.split("+")
                for part in parts:
                    part2node_map[part] = node.name
        return part2node_map

    def get_node2layer_map(self):
        """
        returns map from node name to layer index (in self.layers)
        """
        node2layer_map = {}
        for i,layer in enumerate(self.layers):
            for node in layer.nodes:
                node2layer_map[node.name] = i
        return node2layer_map

    def get_layer_part2loss_map(self,
                                layer,
                                name2node_map,
                                orig_name2node_map,
                                part2node_map,
                                nodes2edge_between_map,
                                example={}):
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

    def get_next_nodes(current_values):
        next_nodes = set([])
        for node in current_values:
            for edge in node.out_edges:
                next_nodes.add(edge.dest)
        return list(next_nodes)

    def get_part2example_change(self, example):
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
            for node,val in next_layer_values.items():
                part2diffs[node] = val
            cur_layer_values = next_layer_values
        return part2diffs

    def get_variables(self):
        nodes2variables = {}
        variables2nodes = {}
        var_index = 0
        for layer in self.layers:
            for node in layer.nodes:
                if layer.type_name in ["input", "output"]:
                    nodes2variables[node.name] = var_index
                    variables2nodes[var_index] = node.name
                    var_index += 1
                else:  # hidden layer, all nodes with relu activation
                    for suffix in ["_b", "_f"]:
                        nodes2variables[node.name + suffix] = var_index
                        variables2nodes[var_index] = node.name + suffix
                        var_index += 1
        return nodes2variables, variables2nodes

    def get_large(self):
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

    def initiate_query(self):
        input_query = "\nfrom maraboupy import MarabouCore\n\n"
        input_query += "inputQuery = MarabouCore.InputQuery()\n\n"
        return input_query

    def get_num_vars(self, variables2nodes):
        set_num_vars = "inputQuery.setNumberOfVariables({})\n\n"
        set_num_vars = set_num_vars.format(len(variables2nodes.keys()))
        return set_num_vars

    def finish_query(self):
        end_query = "vars1, stats1 = MarabouCore.solve(inputQuery, \"\", 0)\n\n"
        end_query += "\n".join(["print(vars1, stats1)",
                                "if len(vars1)>0:",
                                "\tprint('SAT')",
                                "\texit({})".format(SAT_EXIT_CODE),
                                "else:",
                                "\tprint('UNSAT')",
                                "\texit({})".format(UNSAT_EXIT_CODE)])
        return end_query + "\n"

    def get_query(self, test_property, verbose=VERBOSE):
        """
        @test_property is a property to check in the network, of the form:
        {
            layer_name:
            [
                (variable_name, {"Lower": l_value, "Upper": u_value}),
                (variable_name, {"Lower": l_value, "Upper": u_value})
                ...
                (variable_name, {"Lower": l_value, "Upper": u_value})
            ],
            ...
        }
        e.g:
        {
            "input":
                [
                    (0, {"Lower": 0, "Upper": 1}),
                    (1, "Lower", 2),
                    (2, "Upper", -1),
                ],
            "output":
                [
                    (0, {"Lower": 0, "Upper": 1}),
                    (1, "Lower", -4),
                    (2, "Upper", 1.6),
                ]
        }
        @return Marabou query - is test_property holds in the network?
        """
        inputQuery = MarabouCore.InputQuery()
        # large
        large = 1.0
        # some heuristic to approximate large, TODO: not sure it is valid
        for layer in self.layers:
            for node in layer.nodes:
                out_weights = [abs(edge.weight) for edge in node.out_edges]
                large *= max(out_weights, default=1.0)
        large = min(INT_MAX, large)
        large = INT_MAX
        nodes2variables, variables2nodes = self.get_variables()
        # setNumberOfVariables
        inputQuery.setNumberOfVariables(len(variables2nodes.keys()))
        # bounds
        out_layer_var_index = len(nodes2variables) - len(self.layers[-1].nodes)

        # "fix" test_property: add [-large,large] bounds to missing variables
        for i in range(len(self.layers[0].nodes)):
            if i not in [x[0] for x in test_property["input"]]:
                print("var {} is missing in test_property".format(i))
                missing_var_bounds = (i, {"Lower": INPUT_LOWER_BOUND,
                                          "Upper": INPUT_UPPER_BOUND})
                test_property["input"].append(missing_var_bounds)
        #for i in range(len(self.layers[-1].nodes)):
        #    if i not in [x[0] for x in test_property["output"]]:
        #        missing_var_bounds = (i, {"Lower": -large, "Upper": large})
        #        test_property["output"].append(missing_var_bounds)

        for layer_name, bounds_list in test_property.items():
            for (var_index, var_bounds_dict) in bounds_list:
                lower_bound = var_bounds_dict.get("Lower", -large)
                upper_bound = var_bounds_dict.get("Upper", large)
                if layer_name == "output":
                    var_index = out_layer_var_index + var_index
                inputQuery.setLowerBound(var_index, lower_bound)
                if verbose:
                    print("setLowerBound({}, {})".format(var_index, lower_bound))
                inputQuery.setUpperBound(var_index, upper_bound)
                if verbose:
                    print("setUpperBound({}, {})".format(var_index, upper_bound))

        # equations
        i = 0
        for layer_index, layer in enumerate(self.layers):
            if layer.type_name == "input":
                continue
            for node in layer.nodes:
                equation = MarabouCore.Equation()
                # test_abstraction does not occur on input layer, therefore after
                # test_abstraction, in_edges of layer 1 nodes may include multiple
                # edges with same dest (if two dests were unioned, the
                # union in_edges include both edges), therefore the weights into
                # each union node have to be summed
                # example:
                # assume that start edges are: x00--(1)-->x10, x00--(2)-->x11
                # assume that during test_abstraction x10 and x11 became x1u (union)
                # so current edges: x00--(1)-->x1u, x00--(2)-->x1u
                # should avoid two "addAddend"s with variable x00 by summing
                # x00--(1+2)-->x1u
                if layer_index == 1:
                    #print("layer_index == 1")
                    edge2weight = {}
                    dest_seen = {}
                    # eg: x1u.in_edges = [x00--(1)-->x1u, x00--(2)-->x1u]
                    # generate {(x00,x1u): 3}
                    for in_edge in node.in_edges:
                        # "_f" suffix does not exist in input layer node names
                        src_variable = nodes2variables[in_edge.src]
                        dest_variable = nodes2variables[in_edge.dest + "_b"]
                        key = (src_variable, dest_variable)
                        var_weight = edge2weight.get(key, 0.0)
                        new_weight = var_weight + in_edge.weight
                        edge2weight[(src_variable,dest_variable)] = new_weight
                    for ((src_var, dest_var), weight) in edge2weight.items():
                        equation.addAddend(weight, src_var)
                        if verbose:
                            print("eq {}: addAddend({}, {})".format(i,
                                                                    weight,
                                                                    src_var))
                        if not dest_seen.get(dest_var, False):
                            equation.addAddend(-1, dest_var)
                            dest_seen[dest_var] = True
                            if verbose:
                                print("eq {}: addAddend({}, {})".format(i,
                                                                        -1,
                                                                        dest_var))
                else:
                    if verbose:
                        print("layer_index({}) != 1".format(layer_index))
                    for in_edge in node.in_edges:
                        src_variable = nodes2variables[in_edge.src + "_f"]
                        equation.addAddend(in_edge.weight, src_variable)
                        if verbose:
                            print("eq {}: addAddend({}, {})".format(i,
                                                                    in_edge.weight,
                                                                    src_variable))
                    dest_variable = nodes2variables.get(in_edge.dest+"_b", None)
                    if dest_variable is None:
                        dest_variable = nodes2variables[in_edge.dest]
                    equation.addAddend(-1, dest_variable)
                    if verbose:
                        print("eq {}: addAddend(-1, {})".format(i, dest_variable))
                if verbose:
                    print("eq {}: setScalar({})".format(i, -node.bias))
                equation.setScalar(-node.bias)
                inputQuery.addEquation(equation)
                if verbose:
                    print("eq {}: addEquation".format(i))
                i += 1
        # relu constraints
        for layer in self.layers:
            if layer.type_name != "hidden":
                continue
            for node in layer.nodes:
                node_b_index = nodes2variables.get(node.name + "_b", None)
                node_f_index = nodes2variables.get(node.name + "_f", None)
                if (node_b_index is None) or (node_f_index is None):
                    continue
                #print("add Relu constraint: {}\t{}".format(node_b_index, node_f_index))
                MarabouCore.addReluConstraint(inputQuery,
                                              node_b_index,
                                              node_f_index)

        # solve query
        #print("bbb")
        #import IPython
        #IPython.embed()
        vars1, stats1 = MarabouCore.solve(inputQuery, "", 0)
        result = 'SAT' if len(vars1)>0 else 'UNSAT'
        return vars1, stats1, result

    def get_query_str(self, test_property):
        """
        @test_property is a property to check in the network, of the form:
        {
            layer_name:
            [
                (variable_name, {"Lower": l_value, "Upper": u_value}),
                (variable_name, {"Lower": l_value, "Upper": u_value})
                ...
                (variable_name, {"Lower": l_value, "Upper": u_value})
            ],
            ...
        }
        e.g:
        {
            "input":
                [
                    (0, {"Lower": 0, "Upper": 1}),
                    (1, "Lower", 2),
                    (2, "Upper", -1),
                ],
            "output":
                [
                    (0, {"Lower": 0, "Upper": 1}),
                    (1, "Lower", -4),
                    (2, "Upper", 1.6),
                ]
        }
        @return Marabou query - is test_property holds in the network?
        """
        query = ""
        query += self.initiate_query()
        query += self.get_large()
        nodes2variables, variables2nodes = self.get_variables()
        query += self.get_num_vars(variables2nodes)
        query += self.get_bounds(nodes2variables, test_property)
        query += self.get_equations(nodes2variables)
        query += self.get_relu_constraints(nodes2variables)
        query += self.finish_query()
        return query

    def __str__(self):
        s = ""
        net_data = self.get_general_net_data()
        for k,v in net_data.items():
            s += "{}: {}\n".format(k, v)
        s += "\n"
        s += "\n\n".join(layer.__str__() for layer in self.layers)
        return s

    def get_bounds(self, nodes2variables, test_property):
        """
        returns the part in marabou query which is related to variables bounds.
        for details about @test_property, see get_query() method documentation.
        """
        bounds = ""
        out_layer_var_index = len(nodes2variables) - len(self.layers[-1].nodes)
        for layer_name, bounds_list in test_property.items():
            for (var_index, var_bounds_dict) in bounds_list:
                lower_bound = var_bounds_dict.get("Lower", "-large")
                upper_bound = var_bounds_dict.get("Upper", "large")
                if layer_name == "output":
                    var_index = out_layer_var_index + var_index
                lower = "inputQuery.setLowerBound({}, {})\n"
                lower = lower.format(var_index, lower_bound)
                upper = "inputQuery.setUpperBound({}, {})\n"
                upper = upper.format(var_index, upper_bound)
                bounds += lower
                bounds += upper
            bounds += "\n"
        return bounds

    def get_equations(self, nodes2variables):
        equations = ""
        eq_index = 0
        for layer in self.layers:
            if layer.type_name == "input":
                continue
            for node in layer.nodes:
                equations += self.get_equation(node, nodes2variables, eq_index)
                eq_index += 1
        return equations

    def get_equation(self, node, nodes2variables, eq_index):
        # print(nodes2variables.items())
        equation = "equation{} = MarabouCore.Equation()\n".format(eq_index)
        for in_edge in node.in_edges:
            # default in case of input layer where there is no "f" in node name
            node_variable = nodes2variables.get(in_edge.src + "f", None)
            if node_variable is None:
                node_variable = nodes2variables.get(in_edge.src)
            equation_part = "equation{}.addAddend({}, {})\n"
            equation_part = equation_part.format(eq_index,
                                                 in_edge.weight,
                                                 node_variable)
            equation += equation_part
        # default in case of input layer where there is no "b" in node name
        node_variable = nodes2variables.get(in_edge.dest + "b", None)
        if node_variable is None:
            node_variable = nodes2variables.get(in_edge.dest)
        equation_part = "equation{}.addAddend({}, {})\n"
        equation_part = equation_part.format(eq_index, -1, node_variable)
        equation += equation_part

        equation_part = "equation{}.setScalar(0)\n".format(eq_index)
        equation_part += "inputQuery.addEquation(equation{})\n".format(eq_index)
        equation += equation_part
        equation += "\n"
        return equation


    def get_relu_constraints(self, nodes2variables):
        """
        return string that includes the relu constraints in marabou query
        one constraint is derived from each node in every hidden layer
        """
        # TODO: maybe, not must - can use self for node2layer_type_name map
        constrains = ""
        for layer in self.layers:
            if layer.type_name != "hidden":
                continue
            for node in layer.nodes:
                node_b_index = nodes2variables.get(node.name + "b", None)
                node_f_index = nodes2variables.get(node.name + "f", None)
                if (node_b_index is None) or (node_f_index is None):
                    continue
                relu_con = "MarabouCore.addReluConstraint(inputQuery, {}, {})\n"
            constrains += relu_con.format(node_b_index, node_f_index)
            # MarabouCore.addReluConstraint(inputQuery,3,4)
        constrains += "\n"
        return constrains



class Layer:
    def __init__(self, type_name="hidden", nodes=[]):
        self.nodes = nodes
        self.type_name = type_name  # type_name is one of hidden/input/output

    def __eq__(self, other):
        if len(self.nodes) != len(other.nodes):
            return False
        other_nodes_sorted = sorted(other.nodes, key=lambda node:node.name)
        for i, node in enumerate(sorted(self.nodes, key=lambda node:node.name)):
            if node != other_nodes_sorted[i]:
                print("self.nodes[{}] ({}) != other.nodes[{}] ({})".format(i, node, i, other_nodes_sorted[i]))
                return False
        return True

    def evaluate(self, cur_values, nodes2variables, next,
                 variables2nodes, variable2layer_index):
        "return the next layer values, given cur_values as self inputs"
        cur_var2val = {nodes2variables[node.name] for node in self.nodes}
        next_var2val = {nodes2variables[node.name] for node in next.nodes}
        out_values = []
        for i, val in enumerate(cur_values):
            for out_edge in self.nodes[i].out_edges:
                pass
        return out_values

    def split_pos_neg(self, name2node_map):
        if self.type_name == "output":
            # all nodes are increasing nodes
            new_nodes = []
            for node in self.nodes:
                node.in_edges = []
        else:
            assert self.type_name == "hidden"
            new_nodes = []
            for node in self.nodes:
                # add the 2 nodes (that are splitted from node) to new_nodes
                splitted_nodes = node.split_pos_neg(name2node_map)
                new_nodes.extend(splitted_nodes)
                for sn in splitted_nodes:
                    name2node_map[sn.name] = sn
                del(name2node_map[node.name])
            self.nodes = new_nodes

    def split_inc_dec(self, name2node_map):
        if self.type_name == "output":
            # all nodes are increasing nodes
            new_nodes = []
            for node in self.nodes:
                new_node = ARNode(name=node.name+"_inc",
                          ar_type="inc",
                          activation_func=node.activation_func,
                          in_edges=[],
                          out_edges=[],
                          bias=node.bias
                         )
                new_nodes.append(new_node)
                name2node_map[new_node.name] = new_node
                del(name2node_map[node.name])
            self.nodes = new_nodes
        else:
            # split to increasing and decreasing nodes
            new_nodes = []
            for node in self.nodes:
                # add the 2 nodes (that are splitted from node) to new_nodes
                splitted_nodes = node.split_inc_dec(name2node_map)
                new_nodes.extend(splitted_nodes)
                for sn in splitted_nodes:
                    name2node_map[sn.name] = sn
                del(name2node_map[node.name])
            self.nodes = new_nodes

    def get_couples_of_same_ar_type(self):
        layer_couples = list(itertools.combinations(self.nodes, 2))
        layer_couples = [couple for couple in layer_couples if
                         couple[0].ar_type == couple[1].ar_type and
                         ((couple[0].out_edges[0].weight >= 0) == (couple[1].out_edges[0].weight >= 0))]
        return layer_couples


    def abstract(self, name2node_map, next_layer_part2union):
        if self.type_name == "output":
            # output layer and its nodes' names are not changed
            # return trivial next_layer_part2union
            return {node.name: node.name for node in self.nodes}

        node_inc_pos = ARNode(name="",
                              ar_type="inc",
                              activation_func=relu,
                              in_edges=[],
                              out_edges=[],
                              bias=0.0
                             )
        node_inc_neg = ARNode(name="",
                              ar_type="inc",
                              activation_func=relu,
                              in_edges=[],
                              out_edges=[],
                              bias=0.0
                             )
        node_dec_pos = ARNode(name="",
                              ar_type="dec",
                              activation_func=relu,
                              in_edges=[],
                              out_edges=[],
                              bias=0.0
                             )
        node_dec_neg = ARNode(name="",
                              ar_type="dec",
                              activation_func=relu,
                              in_edges=[],
                              out_edges=[],
                              bias=0.0
                             )

        current_layer_part2union = {}

        # name of union node include its inner nodes' names with "+" in between
        # e.g name of union node of x11_inc,x12_inc is "x11_inc+x12_inc"
        for node in self.nodes:
            if node.ar_type == "inc":
                if node.out_edges[0].weight >= 0:
                    union_node = node_inc_pos
                else:
                    union_node = node_inc_neg
            if node.ar_type == "dec":
                if node.out_edges[0].weight >= 0:
                    union_node = node_dec_pos
                else:
                    union_node = node_dec_neg
            union_node.name += ("+" if union_node.name else "") + node.name
            current_layer_part2union[node.name] = union_node
            # union_node.out_edges.extend(node.out_edges)

        # update current_layer_part2union to include updated names
        # "updated" means after listing all parts in the for loop above
        for part_node, union_node in current_layer_part2union.items():
            current_layer_part2union[part_node] = union_node.name

        abstract_layer_nodes = []
        for node in [node_inc_pos, node_inc_neg, node_dec_pos, node_dec_neg]:
            if node.name:
                abstract_layer_nodes.append(node)

        dest_part2dest = {}
        dest2part_weights = {
            # notice: handle duplicated dests by using dict
            dest: {} for dest in next_layer_part2union.values()
        }
        dest_names = set([])

        for node in abstract_layer_nodes:
            node_positivity = node.get_positivity()
            node.new_out_edges = []

            # get parts
            node_parts = node.name.split("+")

            # get dests
            current_dest_names = set([])
            for part in node_parts:
                for edge in name2node_map[part].out_edges:
                    dest = next_layer_part2union[edge.dest]
                    current_dest_names.add(dest)
                    dest_names.add(dest)

            # get dest_part2dest
            for dest in current_dest_names:
                dest_node_parts = name2node_map[dest].name.split("+")
                for dest_part in dest_node_parts:
                    dest_part2dest[dest_part] = dest

            # get a dictionary {dest: min/max weight from specific part to it}
            for part in node_parts:
                for edge in name2node_map[part].out_edges:
                    dest = dest_part2dest[edge.dest]
                    dest_node = name2node_map[dest]
                    max_min_func, cur_weight = choose_weight_func(node, dest_node)
                    cur_weight = dest2part_weights[dest].get(part, cur_weight)
                    dest2part_weights[dest][part] = max_min_func(cur_weight,
                                                                 edge.weight)
            # choose the minimal/maximal weight as the weight from node to dest
            dest_weights = {}
            for dest, part_weights in dest2part_weights.items():
                if dest not in current_dest_names:
                    continue
                #dest_node = name2node_map[dest]
                #max_min_func, cur_weight = choose_weight_func(dest_node)
                #print("max_min_func = {}".format(max_min_func))
                #print("cur_weight =  = {}".format(cur_weight))
                #print("dest_node.ar_type={}".format(dest_node.ar_type))
                #print("node_positivity={}".format(node_positivity))
                #print("dest={}".format(str(name2node_map[dest])))
                #print("node={}".format(str(node)))
                #dest_weights[dest] = cur_weight
                part_weights = {p:ws for p,ws in part_weights.items()
                                if p in node_parts
                               }
                #for weight in part_weights.values():
                #    cur_weight = max_min_func(weight, dest_weights[dest])
                #    dest_weights[dest] = cur_weight
                dest_weights[dest] = sum(part_weights.values())

                # define the node bias to be the min/max bias among all parts
                part_nodes = [name2node_map[np] for np in node_parts]
                parts_biases = [part_node.bias for part_node in part_nodes]
                node.bias = max_min_func(parts_biases)

            # update node.out_edges and dest.in_edges
            for dest, weight in dest_weights.items():
                # print "{} - ({})".format(dest, weight)
                edge = Edge(src=node.name, dest=dest, weight=weight)
                node.new_out_edges.append(edge)
                dest_node = name2node_map[dest]
                try:
                    dest_node.new_in_edges.append(edge)
                except AttributeError:
                    dest_node.new_in_edges = [edge]


            # update the out edges of the node
            node.out_edges = node.new_out_edges
            del node.new_out_edges

        # update the in edges of the dest node
        for dest in dest_names:
            # union_dest is the node which include dest
            union_dest = name2node_map[dest]
            union_dest.in_edges = union_dest.new_in_edges
            del union_dest.new_in_edges
        self.nodes = abstract_layer_nodes
        return current_layer_part2union

    def get_loss(self, part, node, part2node_map):
        total_loss = 0
        for part_edge in part.out_edges:
            # get the dest to get its node to get the edge to decrease
            part_dest_node = part2node_map[part_edge.dest]
            # TODO: improve performance by pre-calculating node_dest2edge_map
            # node_edge = node_dest2edge_map[(node,part_dest_node)]
            for node_edge in node.out_edges:
                if node_edge.dest == part_dest_node:
                    break
            total_loss += abs(part_edge.weight - node_edge.weight)
        return total_loss

    def __str__(self):
        return self.type_name + "\n\t" + \
               "\n\t".join(node.__str__() for node in self.nodes)


class ARNode:
    def __init__(self, name, ar_type, in_edges, out_edges,
                 bias=0.0, activation_func=relu):
        self.name = name
        self.ar_type = ar_type
        self.activation_func = activation_func
        self.in_edges = in_edges
        self.out_edges = out_edges
        self.bias = bias

    def __eq__(self, other, verbose=VERBOSE):
        if self.name != other.name:
            if verbose:
                print("self.name ({}) != other.name ({})".format(self.name, other.name))
            return False
        if self.ar_type != other.ar_type:
            if verbose:
                print("self.ar_type ({}) != other.ar_type ({})".format(self.ar_type, other.ar_type))
            return False
        if self.activation_func != other.activation_func:
            if verbose:
                print("self.activation_func ({}) != other.activation_func ({})".format(self.activation_func, other.activation_func))
            return False
        if self.bias != other.bias:
            if verbose:
                print("self.bias ({}) != other.bias ({})".format(self.bias, other.bias))
            return False
        if len(self.in_edges) != len(other.in_edges):
            if verbose:
                print("len(self.in_edges) ({}) != len(other.in_edges) ({})".format(len(self.in_edges), len(other.in_edges)))
            return False
        other_in_edges_sorted = sorted(other.in_edges, key=lambda edge:(edge.src,edge.dest))
        for i,edge in enumerate(sorted(self.in_edges, key=lambda edge:(edge.src,edge.dest))):
            if edge != other_in_edges_sorted[i]:
                if verbose:
                    print("self.in_edges[{}] ({}) != other.in_edges[{}] ({})".format(i, self.in_edges[i], i, other.in_edges[i]))
                return False
        if len(self.out_edges) != len(other.out_edges):
            if verbose:
                print("len(self.out_edges) ({}) != len(other.out_edges) ({})".format(len(self.out_edges), len(other.out_edges)))
            return False
        other_out_edges_sorted = sorted(other.out_edges, key=lambda edge:(edge.src,edge.dest))
        for i,edge in enumerate(sorted(self.out_edges, key=lambda edge:(edge.src,edge.dest))):
            if edge != other_out_edges_sorted[i]:
                if verbose:
                    print("self.out_edges[{}] ({}) != other.out_edges[{}] ({})".format(i, self.out_edges[i], i, other.out_edges[i]))
                return False
        return True

    def get_positivity(self):
        if self.out_edges:
            is_positive = any([e.weight < 0 for e in self.out_edges])
        else:
            is_positive = True
        return "pos" if is_positive else "neg"

    def split_pos_neg(self, name2node_map):
        node_pos = ARNode(name=self.name+"_pos",
                          ar_type="",
                          activation_func=self.activation_func,
                          in_edges=[],
                          out_edges=[],
                          bias=self.bias
                         )
        node_neg = ARNode(name=self.name+"_neg",
                          ar_type="",
                          activation_func=self.activation_func,
                          in_edges=[],
                          out_edges=[],
                          bias=self.bias
                         )
        for edge in self.out_edges:
            if edge.weight >= 0:
                src_node = node_pos
                src_suffix = "_pos"
            else:
                src_node= node_neg
                src_suffix = "_neg"
            # flag to validate that at least one edge is generated from original
            at_least_one_edge_flag = False
            for dest_suffix in ["", "_pos", "_neg"]:  # "" if next layer is output
                dest_node = name2node_map.get(edge.dest + dest_suffix, None)
                if dest_node is not None:
                    new_edge = Edge(src=edge.src + src_suffix,
                                    dest=edge.dest + dest_suffix,
                                    weight=edge.weight)
                    src_node.out_edges.append(new_edge)
                    dest_node.in_edges.append(new_edge)
                    at_least_one_edge_flag = True
            assert at_least_one_edge_flag
        # add splitted node to result if it has out edges
        splitted_nodes = []
        if node_pos.out_edges:
            splitted_nodes.append(node_pos)
        if node_neg.out_edges:
            splitted_nodes.append(node_neg)
        #print(node_inc)
        #print(node_dec)
        #print("+"*30)
        return splitted_nodes

    def split_inc_dec(self, name2node_map):
        node_inc = ARNode(name=self.name+"_inc",
                          ar_type="inc",
                          activation_func=self.activation_func,
                          in_edges=[],
                          out_edges=[],
                          bias=self.bias
                         )
        node_dec = ARNode(name=self.name+"_dec",
                          ar_type="dec",
                          activation_func=self.activation_func,
                          in_edges=[],
                          out_edges=[],
                          bias=self.bias
                         )

        for edge in self.out_edges:
            #print(edge)
            if edge.dest + "_inc" in name2node_map.keys():
                dest_node = name2node_map[edge.dest + "_inc"]
                #print("edge to inc")
                if edge.weight >= 0:
                    #print("edge>=0")
                    new_edge = Edge(src=edge.src+"_inc",
                                    dest=edge.dest+"_inc",
                                    weight=edge.weight)
                    node_inc.out_edges.append(new_edge)
                else:
                    #print("edge<0")
                    new_edge = Edge(src=edge.src+"_dec",
                                    dest=edge.dest+"_inc",
                                    weight=edge.weight)
                    node_dec.out_edges.append(new_edge)
                dest_node.in_edges.append(new_edge)
            # not "elif" but "if". both conditions can hold simultaneously
            if edge.dest + "_dec" in name2node_map.keys():
                dest_node = name2node_map[edge.dest + "_dec"]
                #print("edge to dec")
                if edge.weight >= 0:
                    #print("edge>=0")
                    new_edge = Edge(src=edge.src+"_dec",
                                    dest=edge.dest+"_dec",
                                    weight=edge.weight)
                    node_dec.out_edges.append(new_edge)
                else:
                    #print("edge<0")
                    new_edge = Edge(src=edge.src+"_inc",
                                    dest=edge.dest+"_dec",
                                    weight=edge.weight)
                    node_inc.out_edges.append(new_edge)
                dest_node.in_edges.append(new_edge)

        # add splitted node to result if it has out edges
        splitted_nodes = []
        if node_inc.out_edges:
            splitted_nodes.append(node_inc)
        if node_dec.out_edges:
            splitted_nodes.append(node_dec)
        #print(node_inc)
        #print(node_dec)
        #print("+"*30)
        return splitted_nodes

    def update_in_edges(self, updated_names_map):
        # update in_edges to include new_in_edges
        # e.g. after split a node, the edges are updated
        # @updated_names_map maps old name to new name during update,
        # all in_edges with dest not in updated_names_map will be the same
        # all in_edges with dest in updated_names_map are replaced by new edge
        #print("update_in_edges()")
        if not hasattr(self, "new_in_edges"):
            return
        updated_in_edges = []
        #import IPython
        #IPython.embed()
        for nie in self.new_in_edges:
            for ie in self.in_edges:
                replace = False
                if ie.src in updated_names_map.keys():
                    if ie.dest == nie.dest and updated_names_map[ie.src] == nie.src:
                        replace = True
                if replace:
                    updated_in_edges.append(nie)
                else:
                    updated_in_edges.append(ie)
            #print("replace={}".format(replace))
        self.in_edges = updated_in_edges
        del self.new_in_edges

    def update_out_edges(self, updated_names_map):
        # update out_edges to include new_out_edges
        #print("update_out_edges({})".format(updated_names_map.items()))
        if not hasattr(self, "new_out_edges") or self.new_out_edges == []:
            return
        updated_out_edges = []
        # updated_names_map include the source name of union node, new_union2new_split include the new node
        # e.g if updated_names_map={"a_b": "b"}, then new_union2new_split={"a":"b"}
        """
        new_union2new_split = {}
        for src_union, splitted_node in updated_names_map.items():
            parts = src_union.split("+")
            new_union_name = "+".join([p for p in parts if p !! splitted_node]) 
            new_union2new_split[new_union_name] = splitted_node
        """
        #if updated_names_map == {'x_3_22_pos_inc+x_3_27_pos_inc+x_3_41_pos_inc': 'x_3_27_pos_inc+x_3_41_pos_inc'} and self.name == "x_2_0_pos_inc":
        #    import IPython
        #    IPython.embed()
        for noe in self.new_out_edges:
            #print("noe={}".format(noe))
            #import IPython
            #IPython.embed()

            for oe in self.out_edges:
                replace = False
                if oe.dest in updated_names_map.keys():
                    if noe.src == oe.src and updated_names_map[oe.dest] == noe.dest:
                        #print("suitable oe = {}".format(oe))
                        replace = True
                #        break
                if replace:
                    updated_out_edges.append(noe)
                else:
                    updated_out_edges.append(oe)
            #print("replace={}".format(replace))
        self.out_edges = updated_out_edges
        del self.new_out_edges

    def __str__(self):
        return "\n\t\t".join(
            [
             "name={}".format(self.name),
             "ar_type={}".format(self.ar_type),
             "activation_func={}".format(self.activation_func),
             "in_edges={}".format(", ".join(e.__str__()
                                            for e in self.in_edges)),
             "out_edges={}".format(", ".join(e.__str__()
                                             for e in self.out_edges))
            ])


class Edge:
    def __init__(self, src, dest, weight):
        # src, dest are names (ids) of nodes (strings)
        self.src = src
        self.dest = dest
        self.weight = weight

    def __eq__(self, other, verbose=VERBOSE):
        if self.src != other.src:
            if verbose:
                print("self.src ({}) != other.src ({})".format(self.src, other.src))
            return False
        if self.dest != other.dest:
            if verbose:
                print("self.dest ({}) != other.dest ({})".format(self.dest, other.dest))
            return False
        if self.weight != other.weight:
            if verbose:
                print("self.weight ({}) != other.weight ({})".format(self.weight, other.weight))
            return False
        return True

    def __str__(self):
        return "{}--({})-->{}".format(self.src, self.weight, self.dest)


def example_01():
    # edges
    e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=0.5)
    e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=0.33)
    e_00_12 = Edge(src="x_0_0", dest="x_1_2", weight=0.17)
    e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=0.5)
    e_01_11 = Edge(src="x_0_1", dest="x_1_1", weight=0.0)
    e_01_12 = Edge(src="x_0_1", dest="x_1_2", weight=0.5)
    e_02_10 = Edge(src="x_0_2", dest="x_1_0", weight=-0.5)
    e_02_11 = Edge(src="x_0_2", dest="x_1_1", weight=-0.25)
    e_02_12 = Edge(src="x_0_2", dest="x_1_2", weight=-0.25)
    e_10_20 = Edge(src="x_1_0", dest="x_2_0", weight=1.0)
    e_10_21 = Edge(src="x_1_0", dest="x_2_1", weight=1.0)
    e_10_22 = Edge(src="x_1_0", dest="x_2_2", weight=2.0)
    e_11_20 = Edge(src="x_1_1", dest="x_2_0", weight=-1.0)
    e_11_21 = Edge(src="x_1_1", dest="x_2_1", weight=-2.0)
    e_11_22 = Edge(src="x_1_1", dest="x_2_2", weight=-2.0)
    e_12_20 = Edge(src="x_1_2", dest="x_2_0", weight=-2.0)
    e_12_21 = Edge(src="x_1_2", dest="x_2_1", weight=-1.0)
    e_12_22 = Edge(src="x_1_2", dest="x_2_2", weight=-1.0)
    e_20_y = Edge(src="x_2_0", dest="y", weight=1.0)
    e_21_y = Edge(src="x_2_1", dest="y", weight=2.0)
    e_22_y = Edge(src="x_2_2", dest="y", weight=-2.0)

    # nodes
    x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11, e_00_12])
    x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11, e_01_12])
    x_0_2 = ARNode(name="x_0_2", ar_type=None, in_edges=[], out_edges=[e_02_10, e_02_11, e_02_12])
    x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10, e_02_10], out_edges=[e_10_20, e_10_21, e_10_22], bias=1.0)
    x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11, e_02_11], out_edges=[e_11_20, e_11_21, e_11_22], bias=2.0)
    x_1_2 = ARNode(name="x_1_2", ar_type=None, in_edges=[e_00_12, e_01_12, e_02_12], out_edges=[e_12_20, e_12_21, e_12_22], bias=-1.0)
    x_2_0 = ARNode(name="x_2_0", ar_type=None, in_edges=[e_10_20, e_11_20, e_12_20], out_edges=[e_20_y], bias=1.0)
    x_2_1 = ARNode(name="x_2_1", ar_type=None, in_edges=[e_10_21, e_11_21, e_12_21], out_edges=[e_21_y], bias=-1.0)
    x_2_2 = ARNode(name="x_2_2", ar_type=None, in_edges=[e_10_22, e_11_22, e_12_22], out_edges=[e_22_y], bias=-2.0)
    y = ARNode(name="y", ar_type=None, in_edges=[e_20_y, e_21_y, e_22_y], out_edges=[], bias=3.0)

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1, x_0_2])
    h1_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1, x_1_2])
    h2_layer = Layer(type_name="hidden", nodes=[x_2_0, x_2_1, x_2_2])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h1_layer, h2_layer, output_layer])
    return net


def example_02():
    # edges
    e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=0.5)
    e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=0.33)
    e_00_12 = Edge(src="x_0_0", dest="x_1_2", weight=0.17)
    e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=0.5)
    e_01_11 = Edge(src="x_0_1", dest="x_1_1", weight=0.0)
    e_01_12 = Edge(src="x_0_1", dest="x_1_2", weight=0.5)
    e_02_10 = Edge(src="x_0_2", dest="x_1_0", weight=-0.5)
    e_02_11 = Edge(src="x_0_2", dest="x_1_1", weight=-0.25)
    e_02_12 = Edge(src="x_0_2", dest="x_1_2", weight=-0.25)
    e_10_20 = Edge(src="x_1_0", dest="x_2_0", weight=1.0)
    e_10_21 = Edge(src="x_1_0", dest="x_2_1", weight=1.0)
    e_10_22 = Edge(src="x_1_0", dest="x_2_2", weight=2.0)
    e_11_20 = Edge(src="x_1_1", dest="x_2_0", weight=-1.0)
    e_11_21 = Edge(src="x_1_1", dest="x_2_1", weight=-2.0)
    e_11_22 = Edge(src="x_1_1", dest="x_2_2", weight=-2.0)
    e_12_20 = Edge(src="x_1_2", dest="x_2_0", weight=-2.0)
    e_12_21 = Edge(src="x_1_2", dest="x_2_1", weight=-1.0)
    e_12_22 = Edge(src="x_1_2", dest="x_2_2", weight=-1.0)
    e_20_30 = Edge(src="x_2_0", dest="x_3_0", weight=-0.7)
    e_20_31 = Edge(src="x_2_0", dest="x_3_1", weight=0.8)
    e_20_32 = Edge(src="x_2_0", dest="x_3_2", weight=0.3)
    e_21_30 = Edge(src="x_2_1", dest="x_3_0", weight=-0.2)
    e_21_31 = Edge(src="x_2_1", dest="x_3_1", weight=0.2)
    e_21_32 = Edge(src="x_2_1", dest="x_3_2", weight=-0.6)
    e_22_30 = Edge(src="x_2_2", dest="x_3_0", weight=-0.5)
    e_22_31 = Edge(src="x_2_2", dest="x_3_1", weight=0.0)
    e_22_32 = Edge(src="x_2_2", dest="x_3_2", weight=1.0)
    e_30_y = Edge(src="x_3_0", dest="y", weight=1.0)
    e_31_y = Edge(src="x_3_1", dest="y", weight=2.0)
    e_32_y = Edge(src="x_3_2", dest="y", weight=-2.0)

    # nodes
    x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11, e_00_12])
    x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11, e_01_12])
    x_0_2 = ARNode(name="x_0_2", ar_type=None, in_edges=[], out_edges=[e_02_10, e_02_11, e_02_12])
    x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10, e_02_10], out_edges=[e_10_20, e_10_21, e_10_22], bias=1.0)
    x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11, e_02_11], out_edges=[e_11_20, e_11_21, e_11_22], bias=2.0)
    x_1_2 = ARNode(name="x_1_2", ar_type=None, in_edges=[e_00_12, e_01_12, e_02_12], out_edges=[e_12_20, e_12_21, e_12_22], bias=-1.0)
    x_2_0 = ARNode(name="x_2_0", ar_type=None, in_edges=[e_10_20, e_11_20, e_12_20], out_edges=[e_20_30, e_20_31, e_20_32], bias=1.0)
    x_2_1 = ARNode(name="x_2_1", ar_type=None, in_edges=[e_10_21, e_11_21, e_12_21], out_edges=[e_21_30, e_21_31, e_21_32], bias=-1.0)
    x_2_2 = ARNode(name="x_2_2", ar_type=None, in_edges=[e_10_22, e_11_22, e_12_22], out_edges=[e_22_30, e_22_31, e_22_32], bias=-2.0)
    x_3_0 = ARNode(name="x_3_0", ar_type=None, in_edges=[e_20_30, e_21_30, e_22_30], out_edges=[e_30_y], bias=1.0)
    x_3_1 = ARNode(name="x_3_1", ar_type=None, in_edges=[e_20_31, e_21_31, e_22_31], out_edges=[e_31_y], bias=-1.0)
    x_3_2 = ARNode(name="x_3_2", ar_type=None, in_edges=[e_20_32, e_21_32, e_22_32], out_edges=[e_32_y], bias=-2.0)
    y = ARNode(name="y", ar_type=None, in_edges=[e_30_y, e_31_y, e_32_y], out_edges=[], bias=3.0)

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1, x_0_2])
    h0_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1, x_1_2])
    h1_layer = Layer(type_name="hidden", nodes=[x_2_0, x_2_1, x_2_2])
    h2_layer = Layer(type_name="hidden", nodes=[x_3_0, x_3_1, x_3_2])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h0_layer, h1_layer, h2_layer, output_layer])
    return net


def example_03():
    # edges
    e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=0.5)
    e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=0.33)
    e_00_12 = Edge(src="x_0_0", dest="x_1_2", weight=0.17)
    e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=0.5)
    e_01_11 = Edge(src="x_0_1", dest="x_1_1", weight=0.0)
    e_01_12 = Edge(src="x_0_1", dest="x_1_2", weight=0.5)
    e_02_10 = Edge(src="x_0_2", dest="x_1_0", weight=-0.5)
    e_02_11 = Edge(src="x_0_2", dest="x_1_1", weight=-0.25)
    e_02_12 = Edge(src="x_0_2", dest="x_1_2", weight=-0.25)
    e_10_20 = Edge(src="x_1_0", dest="x_2_0", weight=1.0)
    e_10_21 = Edge(src="x_1_0", dest="x_2_1", weight=1.0)
    e_10_22 = Edge(src="x_1_0", dest="x_2_2", weight=2.0)
    e_11_20 = Edge(src="x_1_1", dest="x_2_0", weight=-1.0)
    e_11_21 = Edge(src="x_1_1", dest="x_2_1", weight=-2.0)
    e_11_22 = Edge(src="x_1_1", dest="x_2_2", weight=-2.0)
    e_12_20 = Edge(src="x_1_2", dest="x_2_0", weight=-2.0)
    e_12_21 = Edge(src="x_1_2", dest="x_2_1", weight=-1.0)
    e_12_22 = Edge(src="x_1_2", dest="x_2_2", weight=-1.0)
    e_20_30 = Edge(src="x_2_0", dest="x_3_0", weight=-0.7)
    e_20_31 = Edge(src="x_2_0", dest="x_3_1", weight=0.8)
    e_20_32 = Edge(src="x_2_0", dest="x_3_2", weight=0.3)
    e_21_30 = Edge(src="x_2_1", dest="x_3_0", weight=-0.2)
    e_21_31 = Edge(src="x_2_1", dest="x_3_1", weight=0.2)
    e_21_32 = Edge(src="x_2_1", dest="x_3_2", weight=-0.6)
    e_22_30 = Edge(src="x_2_2", dest="x_3_0", weight=-0.5)
    e_22_31 = Edge(src="x_2_2", dest="x_3_1", weight=0.0)
    e_22_32 = Edge(src="x_2_2", dest="x_3_2", weight=1.0)
    e_30_40 = Edge(src="x_3_0", dest="x_4_0", weight=-0.7)
    e_30_41 = Edge(src="x_3_0", dest="x_4_1", weight=0.8)
    e_30_42 = Edge(src="x_3_0", dest="x_4_2", weight=0.3)
    e_31_40 = Edge(src="x_3_1", dest="x_4_0", weight=-0.7)
    e_31_41 = Edge(src="x_3_1", dest="x_4_1", weight=0.8)
    e_31_42 = Edge(src="x_3_1", dest="x_4_2", weight=0.3)
    e_32_40 = Edge(src="x_3_2", dest="x_4_0", weight=-0.7)
    e_32_41 = Edge(src="x_3_2", dest="x_4_1", weight=0.8)
    e_32_42 = Edge(src="x_3_2", dest="x_4_2", weight=0.3)
    e_40_y = Edge(src="x_4_0", dest="y", weight=1.0)
    e_41_y = Edge(src="x_4_1", dest="y", weight=2.0)
    e_42_y = Edge(src="x_4_2", dest="y", weight=-2.0)

    # nodes
    x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11, e_00_12])
    x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11, e_01_12])
    x_0_2 = ARNode(name="x_0_2", ar_type=None, in_edges=[], out_edges=[e_02_10, e_02_11, e_02_12])
    x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10, e_02_10], out_edges=[e_10_20, e_10_21, e_10_22], bias=1.0)
    x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11, e_02_11], out_edges=[e_11_20, e_11_21, e_11_22], bias=2.0)
    x_1_2 = ARNode(name="x_1_2", ar_type=None, in_edges=[e_00_12, e_01_12, e_02_12], out_edges=[e_12_20, e_12_21, e_12_22], bias=-1.0)
    x_2_0 = ARNode(name="x_2_0", ar_type=None, in_edges=[e_10_20, e_11_20, e_12_20], out_edges=[e_20_30, e_20_31, e_20_32], bias=1.0)
    x_2_1 = ARNode(name="x_2_1", ar_type=None, in_edges=[e_10_21, e_11_21, e_12_21], out_edges=[e_21_30, e_21_31, e_21_32], bias=-1.0)
    x_2_2 = ARNode(name="x_2_2", ar_type=None, in_edges=[e_10_22, e_11_22, e_12_22], out_edges=[e_22_30, e_22_31, e_22_32], bias=-2.0)
    x_3_0 = ARNode(name="x_3_0", ar_type=None, in_edges=[e_20_30, e_21_30, e_22_30], out_edges=[e_30_40, e_30_41, e_30_42], bias=1.0)
    x_3_1 = ARNode(name="x_3_1", ar_type=None, in_edges=[e_20_31, e_21_31, e_22_31], out_edges=[e_31_40, e_31_41, e_31_42], bias=-1.0)
    x_3_2 = ARNode(name="x_3_2", ar_type=None, in_edges=[e_20_32, e_21_32, e_22_32], out_edges=[e_32_40, e_32_41, e_32_42], bias=-2.0)
    x_4_0 = ARNode(name="x_4_0", ar_type=None, in_edges=[e_30_40, e_31_40, e_32_40], out_edges=[e_40_y], bias=1.0)
    x_4_1 = ARNode(name="x_4_1", ar_type=None, in_edges=[e_30_41, e_31_41, e_32_41], out_edges=[e_41_y], bias=-1.0)
    x_4_2 = ARNode(name="x_4_2", ar_type=None, in_edges=[e_30_42, e_31_42, e_32_42], out_edges=[e_42_y], bias=-2.0)
    y = ARNode(name="y", ar_type=None, in_edges=[e_40_y, e_41_y, e_42_y], out_edges=[], bias=3.0)

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1, x_0_2])
    h0_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1, x_1_2])
    h1_layer = Layer(type_name="hidden", nodes=[x_2_0, x_2_1, x_2_2])
    h2_layer = Layer(type_name="hidden", nodes=[x_3_0, x_3_1, x_3_2])
    h3_layer = Layer(type_name="hidden", nodes=[x_4_0, x_4_1, x_4_2])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h0_layer, h1_layer, h2_layer, h3_layer, output_layer])
    return net


def example_1():
    net = example_01()
    print(net)

    net.visualize("original net")
    net.abstract()
    #print(net)
    net.visualize("after abstract")
    net.refine()
    print(net)
    net.visualize("after refine")


def get_all_acas_nets():
    """return list of Net objects of acas networks"""
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    l = []
    for filename in os.listdir(nnet_dir):
        nnet_filename = os.path.join(nnet_dir, filename)
        l.append(net_from_nnet_file(nnet_filename))
    return l


def test_net_eq_sanity_check():
    """checks that == operator of Net works"""
    l1 = get_all_acas_nets()
    l2 = [copy.deepcopy(net) for net in l1]
    assert l1 == l2

    # change in edge src
    orig_src = l2[10].layers[3].nodes[5].in_edges[1].src
    l2[10].layers[3].nodes[5].in_edges[1].src = "yosi"
    assert l1 != l2
    l2[10].layers[3].nodes[5].in_edges[1].src = orig_src

    # change in edge dest
    orig_dest = l2[10].layers[3].nodes[5].in_edges[1].dest
    l2[10].layers[3].nodes[5].in_edges[1].dest = "yosi"
    assert l1 != l2
    l2[10].layers[3].nodes[5].in_edges[1].dest = orig_dest

    # change in edge weight
    orig_weight = l2[10].layers[3].nodes[5].in_edges[1].weight
    l2[10].layers[3].nodes[5].in_edges[1].weight = random.random()
    assert l1 != l2
    l2[10].layers[3].nodes[5].in_edges[1].weight = orig_weight

    # change out edge src
    orig_src = l2[10].layers[3].nodes[5].out_edges[1].src
    l2[10].layers[3].nodes[5].out_edges[1].src = "yosi"
    assert l1 != l2
    l2[10].layers[3].nodes[5].out_edges[1].src = orig_src

    # change out edge dest
    orig_dest = l2[10].layers[3].nodes[5].out_edges[1].dest
    l2[10].layers[3].nodes[5].out_edges[1].dest = "yosi"
    assert l1 != l2
    l2[10].layers[3].nodes[5].out_edges[1].dest = orig_dest

    # change out edge weight
    orig_dest = l2[10].layers[3].nodes[5].out_edges[1].weight
    l2[10].layers[3].nodes[5].out_edges[1].weight = random.random()
    assert l1 != l2
    l2[10].layers[3].nodes[5].out_edges[1].weight = orig_weight

    # change node name
    orig_name = l2[10].layers[3].nodes[5].name
    l2[10].layers[3].nodes[5].name = "yosi"
    assert l1 != l2
    l2[10].layers[3].nodes[5].name = orig_name

    # change node ar_type
    orig_ar_type = l2[10].layers[3].nodes[5].ar_type
    l2[10].layers[3].nodes[5].ar_type = "yosi"
    assert l1 != l2
    l2[10].layers[3].nodes[5].ar_type = orig_ar_type

    # change node activation function
    orig_activation_func = l2[10].layers[3].nodes[5].activation_func
    l2[10].layers[3].nodes[5].activation_func = lambda x:-x
    assert l1 != l2
    l2[10].layers[3].nodes[5].activation_func = orig_activation_func

    # change node bias
    orig_bias = l2[10].layers[3].nodes[5].bias
    l2[10].layers[3].nodes[5].bias = random.random()
    assert l1 != l2
    l2[10].layers[3].nodes[5].bias = orig_bias

    # change node in_edges - remove one edge
    orig_in_edges = l2[10].layers[3].nodes[5].in_edges
    del l2[10].layers[3].nodes[5].in_edges[0]
    assert l1 != l2
    l2[10].layers[3].nodes[5].in_edges = orig_in_edges

    # change node out_edges - remove one edge
    orig_out_edges = l2[10].layers[3].nodes[5].out_edges
    del l2[10].layers[3].nodes[5].out_edges[0]
    assert l1 != l2
    l2[10].layers[3].nodes[5].out_edges = orig_out_edges


def test_minimal_abstract_01():
    test_property = {
        "input":
            [
                (0, {"Lower": 0.0, "Upper": 0.9}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5})
            ],
        "output":
            [
                (0, {"Lower": 3.9911256459})
            ]
    }
    net = example_01()
    net.heuristic_abstract(test_property=test_property)


def test_minimal_abstract_all_acas_nets():
    test_property = get_test_property_acas()
    acas_nets = get_all_acas_nets()
    for i,net in enumerate(acas_nets):
        print(i)
        net.heuristic_abstract(test_property=test_property)


def test_abstract_and_refine_result_with_orig_1():
    net = example_01()
    net.preprocess()
    orig_net = copy.deepcopy(net)
    debug_print("original net")
    print(orig_net)
    net.abstract(do_preprocess=False)
    debug_print("abstract net")
    print(net)
    net.refine(orig_net.get_general_net_data()['num_nodes'])
    debug_print("refined net")
    print(net)
    # import IPython
    # IPython.embed()
    assert net == orig_net


def test_abstract_and_refine_result_with_orig_2():
    """test on 4 layers"""
    net = example_02()
    net.preprocess()
    orig_net = copy.deepcopy(net)
    debug_print("original net")
    print(orig_net)
    net.abstract(do_preprocess=False)
    net.visualize(title="abstract net")
    debug_print("abstract net")
    print(net)
    net.refine(orig_net.get_general_net_data()['num_nodes'])
    debug_print("refined net")
    print(net)
    assert net == orig_net


def test_abstract_and_refine_result_with_orig_3():
    """test on 5 layers"""
    net = example_03()
    net.preprocess()
    orig_net = copy.deepcopy(net)
    debug_print("original net")
    print(orig_net)
    net.abstract(do_preprocess=False)
    debug_print("abstract net")
    print(net)
    net.refine(orig_net.get_general_net_data()['num_nodes'])
    debug_print("refined net")
    print(net)
    assert net == orig_net


def test_abstract_and_refine_result_with_orig_acas():
    net = example_4()
    net.preprocess()
    orig_net = copy.deepcopy(net)
    net.abstract(do_preprocess=False)
    #net.minimal_abstract(test_property=get_test_property_acas(), do_preprocess=False)
    net.refine(orig_net.get_general_net_data()['num_nodes'])
    assert net == orig_net


def test_abstract_and_refine_result_with_orig_all_acas():
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    for i,filename in enumerate(os.listdir(nnet_dir)):
        print("check {} - filename = {}".format(i, filename ))
        nnet_filename = os.path.join(nnet_dir, filename)
        net = net_from_nnet_file(nnet_filename)
        net.preprocess()
        orig_net = copy.deepcopy(net)
        net.abstract(do_preprocess=False)
        net.refine(orig_net.get_general_net_data()['num_nodes'])
        assert net == orig_net


def example_1_1():
    # edges
    e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=0.5)
    e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=-0.33)
    e_00_12 = Edge(src="x_0_0", dest="x_1_2", weight=0.17)
    e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=-0.5)
    e_01_11 = Edge(src="x_0_1", dest="x_1_1", weight=0.0)
    e_01_12 = Edge(src="x_0_1", dest="x_1_2", weight=0.5)
    e_02_10 = Edge(src="x_0_2", dest="x_1_0", weight=-0.5)
    e_02_11 = Edge(src="x_0_2", dest="x_1_1", weight=0.25)
    e_02_12 = Edge(src="x_0_2", dest="x_1_2", weight=-0.25)
    e_10_20 = Edge(src="x_1_0", dest="x_2_0", weight=1.0)
    e_10_21 = Edge(src="x_1_0", dest="x_2_1", weight=1.0)
    e_10_22 = Edge(src="x_1_0", dest="x_2_2", weight=2.0)
    e_11_20 = Edge(src="x_1_1", dest="x_2_0", weight=-1.0)
    e_11_21 = Edge(src="x_1_1", dest="x_2_1", weight=2.0)
    e_11_22 = Edge(src="x_1_1", dest="x_2_2", weight=-2.0)
    e_12_20 = Edge(src="x_1_2", dest="x_2_0", weight=-2.0)
    e_12_21 = Edge(src="x_1_2", dest="x_2_1", weight=1.0)
    e_12_22 = Edge(src="x_1_2", dest="x_2_2", weight=-1.0)
    e_20_y = Edge(src="x_2_0", dest="y", weight=1.0)
    e_21_y = Edge(src="x_2_1", dest="y", weight=2.0)
    e_22_y = Edge(src="x_2_2", dest="y", weight=-2.0)

    # nodes
    x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11, e_00_12])
    x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11, e_01_12])
    x_0_2 = ARNode(name="x_0_2", ar_type=None, in_edges=[], out_edges=[e_02_10, e_02_11, e_02_12])
    x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10, e_02_10], out_edges=[e_10_20, e_10_21, e_10_22])
    x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11, e_02_11], out_edges=[e_11_20, e_11_21, e_11_22])
    x_1_2 = ARNode(name="x_1_2", ar_type=None, in_edges=[e_00_12, e_01_12, e_02_12], out_edges=[e_12_20, e_12_21, e_12_22])
    x_2_0 = ARNode(name="x_2_0", ar_type=None, in_edges=[e_10_20, e_11_20, e_12_20], out_edges=[e_20_y])
    x_2_1 = ARNode(name="x_2_1", ar_type=None, in_edges=[e_10_21, e_11_21, e_12_21], out_edges=[e_21_y])
    x_2_2 = ARNode(name="x_2_2", ar_type=None, in_edges=[e_10_22, e_11_22, e_12_22], out_edges=[e_22_y])
    y = ARNode(name="y", ar_type=None, in_edges=[e_20_y, e_21_y, e_22_y], out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0,x_0_1,x_0_2])
    h1_layer = Layer(type_name="hidden", nodes=[x_1_0,x_1_1,x_1_2])
    h2_layer = Layer(type_name="hidden", nodes=[x_2_0,x_2_1,x_2_2])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h1_layer, h2_layer, output_layer])

    print("-"*80)
    print("original net")
    print("-"*80)
    print(net)
    # import IPython
    # IPython.embed()
    net.visualize(title="orig net", out_image_path="/tmp/orig_net")
    net.abstract()
    print("-"*80)
    print("abstract net")
    print("-"*80)
    print(net)
    net.visualize("abstract net", out_image_path="/tmp/abstract_net")


def example_2():
    # edges
    e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=10)
    e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=5)
    e_00_12 = Edge(src="x_0_0", dest="x_1_2", weight=1)
    e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=2)
    e_01_11 = Edge(src="x_0_1", dest="x_1_1", weight=4)
    e_01_12 = Edge(src="x_0_1", dest="x_1_2", weight=7)
    e_02_10 = Edge(src="x_0_2", dest="x_1_0", weight=-3)
    e_02_11 = Edge(src="x_0_2", dest="x_1_1", weight=-8)
    e_02_12 = Edge(src="x_0_2", dest="x_1_2", weight=-7)
    e_10_20 = Edge(src="x_1_0", dest="x_2_0", weight=5)
    e_10_21 = Edge(src="x_1_0", dest="x_2_1", weight=4)
    e_10_22 = Edge(src="x_1_0", dest="x_2_2", weight=9)
    e_11_20 = Edge(src="x_1_1", dest="x_2_0", weight=-7)
    e_11_21 = Edge(src="x_1_1", dest="x_2_1", weight=-1)
    e_11_22 = Edge(src="x_1_1", dest="x_2_2", weight=-2)
    e_12_20 = Edge(src="x_1_2", dest="x_2_0", weight=-5)
    e_12_21 = Edge(src="x_1_2", dest="x_2_1", weight=-3)
    e_12_22 = Edge(src="x_1_2", dest="x_2_2", weight=-6)
    e_20_y = Edge(src="x_2_0", dest="y", weight=4)
    e_21_y = Edge(src="x_2_1", dest="y", weight=3)
    e_22_y = Edge(src="x_2_2", dest="y", weight=-6)

    # nodes
    x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11, e_00_12])
    x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11, e_01_12])
    x_0_2 = ARNode(name="x_0_2", ar_type=None, in_edges=[], out_edges=[e_02_10, e_02_11, e_02_12])
    x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10, e_02_10], out_edges=[e_10_20, e_10_21, e_10_22])
    x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11, e_02_11], out_edges=[e_11_20, e_11_21, e_11_22])
    x_1_2 = ARNode(name="x_1_2", ar_type=None, in_edges=[e_00_12, e_01_12, e_02_12], out_edges=[e_12_20, e_12_21, e_12_22])
    x_2_0 = ARNode(name="x_2_0", ar_type=None, in_edges=[e_10_20, e_11_20, e_12_20], out_edges=[e_20_y])
    x_2_1 = ARNode(name="x_2_1", ar_type=None, in_edges=[e_10_21, e_11_21, e_12_21], out_edges=[e_21_y])
    x_2_2 = ARNode(name="x_2_2", ar_type=None, in_edges=[e_10_22, e_11_22, e_12_22], out_edges=[e_22_y])
    y = ARNode(name="y", ar_type=None, in_edges=[e_20_y, e_21_y, e_22_y], out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1, x_0_2])
    h1_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1, x_1_2])
    h2_layer = Layer(type_name="hidden", nodes=[x_2_0, x_2_1, x_2_2])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h1_layer, h2_layer, output_layer])
    #print(net)

    print("-"*80)
    print("do test_abstraction")
    print("-"*80)

    orig_net = copy.deepcopy(net)
    abstracted_net = net.abstract()
    #abstracted_net.visualize(title="abstracted_net")

    # do_cegar = False
    # if do_cegar:
    #     p = None  # property to check
    #     counter_example = abstracted_net.verify_property(p)
    #     if counter_example is None:
    #         return True  # p holds in abstracted_net -> p holds in net
    #     else:
    #         output = orig_net.run(counter_example)
    #         if not does_property_holds(p, output):
    #             return False
    #         else:
    #             refined_net = abtracted_net
    #             i = 0
    #             while True:
    #                 i += 1  # number of refinements
    #                 refined_net = refined_net.refine(sequence_len=1)
    #                 counter_example = net.verify_property(refined_net, p)
    #                 if counter_example is None:
    #                     return True, i
    #                 else:
    #                     output = orig_net.run(counter_example)
    #                     if not does_holds(p, output):
    #                         return False, i
    #
    #
    #
    #
    #
    # else:
    #     orig_abstracted_net = copy.deepcopy(abstracted_net)
    #     #debug_print("abstracted_net")
    #     #print(abstracted_net)
    #     abstracted_net.visualize(title="refined net")
    #     refined_net = abstracted_net.refine(sequence_length=1)
    #     #print(refined_net)
    #     refined_net.visualize(title="refined net")


def example_3():
    names = [
        "x11", "x12", "x13",
        "x21b", "x22b", "x23b",
        "x21f", "x22f", "x23f",
        "x31b", "x32b", "x33b",
        "x31f", "x32f", "x33f",
        "y"
    ]

    # edges
    e_11_21 = Edge(src="x11", dest="x21", weight=10)
    e_11_22 = Edge(src="x11", dest="x22", weight=5)
    e_11_23 = Edge(src="x11", dest="x23", weight=1)
    e_12_21 = Edge(src="x12", dest="x21", weight=2)
    e_12_22 = Edge(src="x12", dest="x22", weight=4)
    e_12_23 = Edge(src="x12", dest="x23", weight=7)
    e_13_21 = Edge(src="x13", dest="x21", weight=-3)
    e_13_22 = Edge(src="x13", dest="x22", weight=-8)
    e_13_23 = Edge(src="x13", dest="x23", weight=-7)
    e_21_31 = Edge(src="x21", dest="x31", weight=5)
    e_21_32 = Edge(src="x21", dest="x32", weight=4)
    e_21_33 = Edge(src="x21", dest="x33", weight=9)
    e_22_31 = Edge(src="x22", dest="x31", weight=-7)
    e_22_32 = Edge(src="x22", dest="x32", weight=-1)
    e_22_33 = Edge(src="x22", dest="x33", weight=-2)
    e_23_31 = Edge(src="x23", dest="x31", weight=-5)
    e_23_32 = Edge(src="x23", dest="x32", weight=-3)
    e_23_33 = Edge(src="x23", dest="x33", weight=-6)
    e_31_y = Edge(src="x31", dest="y", weight=4)
    e_32_y = Edge(src="x32", dest="y", weight=3)
    e_33_y = Edge(src="x33", dest="y", weight=-6)

    # nodes
    x11 = ARNode(name="x11", ar_type=None, in_edges=[],
                 out_edges=[e_11_21, e_11_22, e_11_23])
    x12 = ARNode(name="x12", ar_type=None, in_edges=[],
                 out_edges=[e_12_21, e_12_22, e_12_23])
    x13 = ARNode(name="x13", ar_type=None, in_edges=[],
                 out_edges=[e_13_21, e_13_22, e_13_23])
    x21 = ARNode(name="x21", ar_type=None,
                 in_edges=[e_11_21, e_12_21, e_13_21],
                 out_edges=[e_21_31, e_21_32, e_21_33])
    x22 = ARNode(name="x22", ar_type=None,
                 in_edges=[e_11_22, e_12_22, e_13_22],
                 out_edges=[e_22_31, e_22_32, e_22_33])
    x23 = ARNode(name="x23", ar_type=None,
                 in_edges=[e_11_23, e_12_23, e_13_23],
                 out_edges=[e_23_31, e_23_32, e_23_33])
    x31 = ARNode(name="x31", ar_type=None,
                 in_edges=[e_21_31, e_22_31, e_23_31],
                 out_edges=[e_31_y])
    x32 = ARNode(name="x32", ar_type=None,
                 in_edges=[e_21_32, e_22_32, e_23_32],
                 out_edges=[e_32_y])
    x33 = ARNode(name="x33", ar_type=None,
                 in_edges=[e_21_33, e_22_33, e_23_33],
                 out_edges=[e_33_y])
    y = ARNode(name="y", ar_type=None,
               in_edges=[e_31_y, e_32_y, e_33_y],
               out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x11,x12,x13])
    h1_layer = Layer(type_name="hidden", nodes=[x21,x22,x23])
    h2_layer = Layer(type_name="hidden", nodes=[x31,x32,x33])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h1_layer, h2_layer, output_layer])
    #net = example_4()
    net = net.abstract()
    print(net)
    #net.visualize()
    """test_property = {
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.62}),
                (1, {"Lower": 0.0, "Upper": 0.02}),
                (2, {"Lower": 0.0, "Upper": 0.02}),
                (3, {"Lower": 0.0, "Upper": 0.02}),
                (4, {"Lower": 0.0, "Upper": 0.02}),
            ],
        "output":
            [
                (0, {"Lower": 3.9911256459})
            ]
    }"""
    test_property = {
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                #(3, {"Lower": 0.45, "Upper": 0.5}),
                #(4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                #(0, {"Lower": 3.9911256459})
            ]
    }

    vars1, stats1, result = net.get_query(test_property)
    debug_print(vars1)
    debug_print(stats1)
    debug_print(result)
    #print(query)
    #with open("/tmp/query.py", "w") as f:
    #    f.write(query)
    #import os
    #import shutil
    #import subprocess
    #MARABOU_EXAMPLES_DIR = "/home/yizhak/Research/Code/Marabou/maraboupy/examples"
    #shutil.move(f.name, os.path.join(MARABOU_EXAMPLES_DIR, "query.py"))
    #os.chdir(MARABOU_EXAMPLES_DIR)
    #cmd = ["python3 {}".format(f.name.rpartition("/")[-1])]
    #cp = subprocess.run(cmd, shell=True)
    #print(cp)

def example_4():

    """
    generate Net from acasxu network that is represented by nnet format file
    """
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    # filename = "ACASXU_TF12_run3_DS_Minus15_Online_tau0_pra1_200Epochs.nnet"
    filename = "ACASXU_run2a_1_1_batch_2000.nnet"
    nnet_filename = os.path.join(nnet_dir, filename)
    net = net_from_nnet_file(nnet_filename)
    return net

def example_5():
    net = example_4()
    print(net)
    #net.visualize(title="acasxu_net")
    abstracted_net = net.abstract()
    abstracted_net.visualize("abstracted_net")
    refine_1_net = abstracted_net.refine(sequence_length=3)
    refine_1_net.visualize("refine 1")

def experiment_1():
    # time to check property on net with marabou
    marabou_times = []
    # time to check property on net with marabou using CEGAR
    cegar_times = []
    # time to check property on the last network in CEGAR
    last_net_cegar_times = []

    # time to check property on net with marabou using CETAR
    cetar_times = []
    # time to check property on the last network in CETAR
    last_net_cetar_times = []

    results = []
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    #for i,filename in enumerate(os.listdir(nnet_dir)):  # eg ACASXU_run2a_1_1_batch_2000.nnet
    for i,filename in enumerate(["ACASXU_run2a_1_9_batch_2000.nnet"]):
        nnet_filename = os.path.join(nnet_dir, filename)
        debug_print("{}, {}".format(i, nnet_filename))

        net = net_from_nnet_file(nnet_filename)
        """
        test_property_1 = {
            "input":
                [
                    (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                    (1, {"Lower": -0.5, "Upper": 0.5}),
                    (2, {"Lower": -0.5, "Upper": 0.5}),
                    (3, {"Lower": 0.45, "Upper": 0.5}),
                    (4, {"Lower": -0.5, "Upper": -0.45}),
                ],
            "output":
                [
                    (0, {"Lower": 3.9911256459})
                ]
        }
        """
        test_property_2 = {
            "input":
                [
                    (0, {"Lower": 0.6798577687, "Upper": 0.7}),
                    (1, {"Lower": 0.5, "Upper": 0.52}),
                    (2, {"Lower": -0.52, "Upper": -0.5}),
                    (3, {"Lower": 0.43, "Upper": 0.45}),
                    (4, {"Lower": -0.52, "Upper": -0.5}),
                ],
            "output":
                [
                    (0, {"Lower": 3.9911256459})
                ]
        }
        """
        test_property_3 = {
            "input":
                [
                    (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                    (1, {"Lower": -0.5, "Upper": 0.5}),
                    (2, {"Lower": -0.5, "Upper": 0.5}),
                    (3, {"Lower": 0.45, "Upper": 0.5}),
                    (4, {"Lower": -0.5, "Upper": -0.45}),
                ],
            "output":
                [
                    (0, {"Upper": 1000})
                ]
        }
        test_property_4 = {
            "input":
                [
                    (0, {"Lower": 0.6, "Upper": 0.62}),
                    (1, {"Lower": -0.5, "Upper": -0.48}),
                    (2, {"Lower": -0.5, "Upper": -0.48}),
                    (3, {"Lower": 0.45, "Upper": 0.47}),
                    (4, {"Lower": -0.5, "Upper": -0.48}),
                ],
            "output":
                [
                    (0, {"Lower": 3.9911256459})
                ]
        }
        test_property_5 = {
            'input':
            [
                (0, {'Lower': 0.6, 'Upper': 0.62}),
                (1, {'Lower': -0.5, 'Upper': -0.48}),
                (2, {'Lower': -0.5, 'Upper': -0.48}),
                (3, {'Lower': 0.45, 'Upper': 0.47}),
                (4, {'Lower': -0.5, 'Upper': -0.48})
            ],
            'output':
            [
                (0, {'Upper': 3.9911256459}),
                (1, {'Lower': -20, 'Upper': 20}),
                (2, {'Lower': -20, 'Upper': 20}),
                (3, {'Lower': -20, 'Upper': 20}),
                (4, {'Lower': -20, 'Upper': 20})
            ]
        }
        """
        test_property = test_property_2
        orig_net = copy.deepcopy(net)

        # query original net
        debug_print("query orig_net")
        t0 = time.time()
        #vars1, stats1, result = orig_net.get_query(test_property)
        #sys.exit(0)
        #debug_print(result)
        t1 = time.time()
        marabou_times.append(t1 - t0)
        debug_print("orig_net query time ={}".format(t1-t0))

        debug_print("query AR net")
        t2 = time.time()
        net = net.abstract()
        num_of_refine_steps = 0
        while True:  # CEGAR / CETAR method
            t4 = time.time()
            vars1, stats1, result = net.get_query(test_property)
            if result == "UNSAT":
                # if always y'<3.99 then also always y<3.99
                debug_print("UNSAT")
                break
            if result == "SAT":
                debug_print("SAT")
                debug_print(vars1)
                orig_net_output = orig_net.evaluate(vars1)
                speedy_orig_net_output = orig_net.speedy_evaluate(vars1)
                assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
                nodes2variables, variables2nodes = orig_net.get_variables()
                # we got y'>3.99, check if also y'>3.99 for the same input
                if orig_net.does_property_holds(test_property,
                                                orig_net_output,
                                                variables2nodes):
                    print("property holds also in orig - SAT")
                    break  # also counter example for orig_net
                else:
                    print("property doesn't holds in orig - spurious example")
                    num_of_refine_steps += 1
                    print("refine step #{}".format(num_of_refine_steps))
                    if DO_CEGAR:
                        net = net.refine(sequence_length=100, example=vars1)
                    else:
                        net = net.refine(sequence_length=100)
        t3 = time.time()
        cegar_times.append(t3 - t2)
        last_net_cegar_times.append(t3 - t4)
        print("ar query time ={}".format(t1-t0))

        results.append({
            "result": result,
            "orig_query_times": marabou_times[i],
            "num_of_refine_steps": num_of_refine_steps,
            "last_net_data": net.get_general_net_data(),
            "ar_query_time": cegar_times[i],
            "last_query_time": last_net_cegar_times[i]
        })
    for i,res in enumerate(results):
        print("res #{}: {}".format(i,res))
    return results


def experiment_2_tiny():
    # time to check property on net with marabou
    marabou_times = []
    # time to check property on net with marabou using CEGAR
    cegar_times = []
    # time to check property on the last network in CEGAR
    last_net_cegar_times = []

    # time to check property on net with marabou using CETAR
    cetar_times = []
    # time to check property on the last network in CETAR
    last_net_cetar_times = []

    results = []
    nnet_dir = PATH_TO_MARABOU_ACAS_EXAMPLES

    #for i,filename in enumerate(os.listdir(nnet_dir)):  # eg ACASXU_run2a_1_1_batch_2000.nnet
    for i,filename in enumerate(["ACASXU_run2a_1_1_tiny_3.nnet"]):
        nnet_filename = os.path.join(nnet_dir, filename)
        net = net_from_nnet_file(nnet_filename)
        test_property_1 = {
            "input":
                [
                    (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                    (1, {"Lower": -0.5, "Upper": 0.5}),
                    (2, {"Lower": -0.5, "Upper": 0.5}),
                    (3, {"Lower": 0.45, "Upper": 0.5}),
                    (4, {"Lower": -0.5, "Upper": -0.45}),
                ],
            "output":
                [
                    (0, {"Lower": 20})
                ]
        }
        test_property = test_property_1

        import time
        t0 = time.time()

        debug_print("query orig_net")
        orig_net = copy.deepcopy(net)
        vars1, stats1, result = orig_net.get_query(test_property)
        debug_print(result)
        t1 = time.time()
        marabou_times.append(t1 - t0)
        debug_print("orig_net query time ={}".format(t1-t0))
        #print(vars1),
        #print(stats1),
        #print(result)

        t2 = time.time()
        net = net.abstract()

        num_of_refine_steps = 0
        debug_print("net number {}".format(i))
        while True:  # CEGAR / CETAR method
            t4 = time.time()
            vars1, stats1, result = net.get_query(test_property)
            if result == "UNSAT":
                debug_print("UNSAT")
                break
            if result == "SAT":
                """
                debug_print("SAT")
                debug_print(vars1)
                orig_net_output = orig_net.evaluate(vars1)
                nodes2variables, variables2nodes = orig_net.get_variables()
                if not orig_net.does_property_holds(test_property,
                                                    orig_net_output,
                                                    variables2nodes):
                    debug_print("property not hold also in orig")
                    break  # also counter example for orig_net
                """
                if False:
                    pass
                else:
                    debug_print("property holds in orig")
                    num_of_refine_steps += 1
                    debug_print("refine step")
                    net = net.refine(sequence_length=10)
        t3 = time.time()
        cegar_times.append(t3 - t2)
        last_net_cegar_times.append(t3 - t4)
        debug_print("ar query time ={}".format(t1-t0))

        results.append({
            "result": result,
            "orig_query_times": marabou_times[i],
            "num_of_refine_steps": num_of_refine_steps,
            "last_net_data": net.get_general_net_data(),
            "ar_query_time": cegar_times[i],
            "last_query_time": last_net_cegar_times[i]
        })
    for i,res in enumerate(results):
        debug_print("res #{}: {}".format(i,res))
    return results


def one_experiment(nnet_filename=None, is_tiny=None, do_cegar=True, complete_abstraction=COMPLETE_ABSTRACTION):
    if nnet_filename is None:
        nnet_filename = sys.argv[1]
    if is_tiny is None:
        is_tiny = sys.argv[2]
    debug_print("nnet_filename = {}, is_tiny={}".format(nnet_filename, is_tiny))
    debug_print("do_cegar={}, complete_abstraction={}".format(do_cegar, complete_abstraction))

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    results_filename = "experiment_NNET_{}_CEGAR_{}_COMPLETE_ABSTRACT_{}_DATETIME_{}".format(os.path.basename(nnet_filename), do_cegar, complete_abstraction, consts.cur_time_str)
    test_property = get_test_property_tiny() if is_tiny else get_test_property_acas()

    debug_print("work on net: {}".format(nnet_filename))
    net = net_from_nnet_file(nnet_filename)
    orig_net = copy.deepcopy(net)
    if COMPARE_TO_PREPROCESSED_NET:
        orig_net.preprocess()

    # query original net
    print("query orig_net")
    t0 = time.time()
    debug_print("orig_net.get_general_net_data(): {}".format(orig_net.get_general_net_data()))
    vars1, stats1, query_result = orig_net.get_query(test_property)
    t1 = time.time()
    #print(query_result)
    #import IPython
    #IPython.embed()

    # time to check property on net with marabou
    marabou_time = t1 - t0
    print("orig_net query time ={}".format(marabou_time))

    print("query using AR")
    t2 = time.time()
    if COMPLETE_ABSTRACTION:
        net = net.abstract()
    else:
        net = net.heuristic_abstract(test_property=test_property)
    num_of_refine_steps = 0
    ar_times = []
    ar_sizes = []
    while True:  # CEGAR / CETAR method
        t4 = time.time()
        vars1, stats1, query_result = net.get_query(test_property)
        t5 = time.time()
        ar_times.append(t5 - t4)
        ar_sizes.append(net.get_general_net_data()["num_nodes"])
        print("query time after A and {} R steps is {}".format(num_of_refine_steps, t5-t4))
        debug_print(net.get_general_net_data())
        if query_result == "UNSAT":
            # if always y'<3.99 then also always y<3.99
            print("UNSAT (finish)")
            break
        if query_result == "SAT":
            print("SAT (have to check example on original net)")
            print(vars1)
            orig_net_output = orig_net.evaluate(vars1)
            speedy_orig_net_output = orig_net.speedy_evaluate(vars1)
            assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
            nodes2variables, variables2nodes = orig_net.get_variables()
            # we got y'>3.99, check if also y'>3.99 for the same input
            if orig_net.does_property_holds(test_property,
                                            orig_net_output,
                                            variables2nodes):
                print("property holds also in orig - SAT (finish)")
                break  # also counter example for orig_net
            else:
                print("property doesn't holds in orig - spurious example")
                num_of_refine_steps += 1
                print("refine step #{}".format(num_of_refine_steps))
                if DO_CEGAR:
                    net = net.refine(sequence_length=50, example=vars1)
                else:
                    net = net.refine(sequence_length=50)
    t3 = time.time()

    # time to check property on net with marabou using CEGAR
    total_ar_time = t3 - t2
    print("ar query time = {}".format(total_ar_time))

    # time to check property on the last network in CEGAR
    last_net_ar_time = t3 - t4
    print("last ar net query time = {}".format(last_net_ar_time))

    res = [
        ("net name", nnet_filename),
        ("query_result", query_result),
        ("orig_query_time", marabou_time),
        ("num_of_refine_steps", num_of_refine_steps),
        ("total_ar_query_time", total_ar_time),
        ("ar_times", json.dumps(ar_times)),
        ("ar_sizes", json.dumps(ar_sizes)),
        ("last_net_data", json.dumps(net.get_general_net_data())),
        ("last_query_time", last_net_ar_time)
    ]
    with open(os.path.join(results_directory, results_filename), "w") as fw:
        for (k,v) in res:
            fw.write("{}: {}\n".format(k,v))
    return res


def experiments(do_cegar=True, advanced_abstraction=False):
    debug_print("do_cegar={}, advanced_abstraction={}".format(do_cegar, advanced_abstraction))

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    results_filename = "experiment_CEGAR_{}_ADVANCE_ABSTRACT_{}_DATETIME_{}".format(do_cegar, advanced_abstraction, cur_time_str)


    # time to check property on net with marabou
    marabou_times = []
    # time to check property on net with marabou using CEGAR
    ar_times = []
    # time to check property on the last network in CEGAR
    last_net_ar_times = []

    exp_results = []
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    nnet_filenames = [os.path.join(nnet_dir, nnet_filename) for nnet_filename in
        [
        "ACASXU_run2a_1_9_batch_2000.nnet",
        "ACASXU_run2a_1_8_batch_2000.nnet",
        "ACASXU_run2a_1_7_batch_2000.nnet",
        "ACASXU_run2a_1_1_batch_2000.nnet",
        "ACASXU_run2a_1_2_batch_2000.nnet",
        "ACASXU_run2a_5_3_batch_2000.nnet",
        "ACASXU_run2a_1_6_batch_2000.nnet",
        ]
    ]
    tiny_nnet_dir = PATH_TO_MARABOU_ACAS_EXAMPLES
    tiny_nets = [fname for fname in os.listdir(tiny_nnet_dir) if "tiny" in fname and fname.endswith(".nnet")]
    tiny_nnet_filenames = [os.path.join(tiny_nnet_dir, tiny_net_filename) for tiny_net_filename in tiny_nets]

    test_property_1 = {
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 3.9911256459})
            ]
    }
    test_property_2 = {
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 20.0})
            ]
    }
    nnets = tiny_nnet_filenames + nnet_filenames
    test_properties = [test_property_2] * len(tiny_nnet_filenames) + \
                      [test_property_1] * len(nnet_filenames)
    #nnets = tiny_nnet_filenames[:-1]
    #test_properties = [test_property_2] * (len(tiny_nnet_filenames) - 1)
    #for i,(test_property, nnet_filename) in enumerate(zip(test_properties, nnets)):
    for i,(test_property, nnet_filename) in enumerate(zip(test_properties, nnets)):
        debug_print("work on net #{}: {}".format(i, nnet_filename))
        net = net_from_nnet_file(nnet_filename)
        orig_net = copy.deepcopy(net)

        # query original net
        print("query orig_net")
        t0 = time.time()
        vars1, stats1, query_result = orig_net.get_query(test_property)
        t1 = time.time()
        print(query_result)
        marabou_times.append(t1 - t0)
        print("i={}, orig_net query time ={}".format(i, t1-t0))

        print("query using AR")
        t2 = time.time()
        net = net.abstract()
        num_of_refine_steps = 0
        while True:  # CEGAR / CETAR method
            t4 = time.time()
            vars1, stats1, query_result = net.get_query(test_property)
            t5 = time.time()
            print("i={}, query time after A and {} R steps is {}".format(i, num_of_refine_steps, t5-t4))
            if query_result == "UNSAT":
                # if always y'<3.99 then also always y<3.99
                print("UNSAT (finish)")
                break
            if query_result == "SAT":
                print("SAT (have to check example on original net)")
                print(vars1)
                orig_net_output = orig_net.evaluate(vars1)
                speedy_orig_net_output = orig_net.speedy_evaluate(vars1)
                assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
                nodes2variables, variables2nodes = orig_net.get_variables()
                # we got y'>3.99, check if also y'>3.99 for the same input
                if orig_net.does_property_holds(test_property,
                                                orig_net_output,
                                                variables2nodes):
                    print("property holds also in orig - SAT (finish)")
                    break  # also counter example for orig_net
                else:
                    print("property doesn't holds in orig - spurious example")
                    num_of_refine_steps += 1
                    print("refine step #{}".format(num_of_refine_steps))
                    if DO_CEGAR:
                        net = net.refine(sequence_length=10, example=vars1)
                    else:
                        net = net.refine(sequence_length=10)
        t3 = time.time()
        ar_times.append(t3 - t2)
        last_net_ar_times.append(t3 - t4)
        print("i={}, ar query time ={}".format(i, t3-t2))

        exp_results.append({
            "net index": i,
            "net name": nnet_filename,
            "query_result": query_result,
            "orig_query_time": marabou_times[i],
            "num_of_refine_steps": num_of_refine_steps,
            "last_net_data": net.get_general_net_data(),
            "ar_query_time": ar_times[i],
            "last_query_time": last_net_ar_times[i]
        })
    for i,res in enumerate(exp_results):
        print("-"*40)
        for k,v in res.items():
            print("res #{}: {}".format(i,res.items()))
    df = pd.DataFrame(exp_results)
    df1 = df[["orig_query_time", "ar_query_time"]]
    ar_times = []
    orig_times = []
    for row in df1.itertuples():
        orig_times.append(row[1])
        ar_times.append(row[2])
    plt.scatter(x=ar_times, y=orig_times)
    plt.savefig(os.path.join(results_directory, results_filename))
    #plt.show()
    #import IPython
    #IPython.embed()
    return exp_results


def test_evaluate_1():
    # edges
    e_00_10 = Edge(src="x00", dest="x10", weight=1)
    e_00_11 = Edge(src="x00", dest="x11", weight=2)
    e_01_10 = Edge(src="x01", dest="x10", weight=3)
    e_01_11 = Edge(src="x01", dest="x11", weight=4)
    e_10_20 = Edge(src="x10", dest="x20", weight=-1)
    e_10_21 = Edge(src="x10", dest="x21", weight=2)
    e_11_20 = Edge(src="x11", dest="x20", weight=2)
    e_11_21 = Edge(src="x11", dest="x21", weight=-3)
    e_20_y = Edge(src="x20", dest="y", weight=-4)
    e_21_y = Edge(src="x21", dest="y", weight=6)

    # nodes
    x_0_0 = ARNode(name="x00", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11])
    x_0_1 = ARNode(name="x01", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11])
    x_1_0 = ARNode(name="x10", ar_type=None, in_edges=[e_00_10, e_01_10], out_edges=[e_10_20, e_10_21])
    x_1_1 = ARNode(name="x11", ar_type=None, in_edges=[e_00_11, e_01_11], out_edges=[e_11_20, e_11_21])
    x_2_0 = ARNode(name="x20", ar_type=None, in_edges=[e_10_20, e_11_20], out_edges=[e_20_y])
    x_2_1 = ARNode(name="x21", ar_type=None, in_edges=[e_10_21, e_11_21], out_edges=[e_21_y])
    y = ARNode(name="y", ar_type=None, in_edges=[e_20_y, e_21_y], out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1])
    h1_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1])
    h2_layer = Layer(type_name="hidden", nodes=[x_2_0, x_2_1])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h1_layer, h2_layer, output_layer])
    # net.visualize()
    eval_output = net.evaluate(input_values={0:2, 1:1})
    speedy_eval_output = net.speedy_evaluate(input_values={0:2, 1:1})
    assert is_evaluation_result_equal(eval_output.items(), speedy_eval_output)
    assert eval_output['y'] == -44


def test_evaluate_2():
    e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=10)
    e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=5)
    e_00_12 = Edge(src="x_0_0", dest="x_1_2", weight=1)
    e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=2)
    e_01_11 = Edge(src="x_0_1", dest="x_1_1", weight=4)
    e_01_12 = Edge(src="x_0_1", dest="x_1_2", weight=7)
    e_02_10 = Edge(src="x_0_2", dest="x_1_0", weight=-3)
    e_02_11 = Edge(src="x_0_2", dest="x_1_1", weight=-8)
    e_02_12 = Edge(src="x_0_2", dest="x_1_2", weight=-7)
    e_10_20 = Edge(src="x_1_0", dest="x_2_0", weight=5)
    e_10_21 = Edge(src="x_1_0", dest="x_2_1", weight=4)
    e_10_22 = Edge(src="x_1_0", dest="x_2_2", weight=9)
    e_11_20 = Edge(src="x_1_1", dest="x_2_0", weight=-7)
    e_11_21 = Edge(src="x_1_1", dest="x_2_1", weight=-1)
    e_11_22 = Edge(src="x_1_1", dest="x_2_2", weight=-2)
    e_12_20 = Edge(src="x_1_2", dest="x_2_0", weight=-5)
    e_12_21 = Edge(src="x_1_2", dest="x_2_1", weight=-3)
    e_12_22 = Edge(src="x_1_2", dest="x_2_2", weight=-6)
    e_20_y = Edge(src="x_2_0", dest="y", weight=4)
    e_21_y = Edge(src="x_2_1", dest="y", weight=3)
    e_22_y = Edge(src="x_2_2", dest="y", weight=-6)

    # nodes
    x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11, e_00_12])
    x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11, e_01_12])
    x_0_2 = ARNode(name="x_0_2", ar_type=None, in_edges=[], out_edges=[e_02_10, e_02_11, e_02_12])
    x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10, e_02_10], out_edges=[e_10_20, e_10_21, e_10_22])
    x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11, e_02_11], out_edges=[e_11_20, e_11_21, e_11_22])
    x_1_2 = ARNode(name="x_1_2", ar_type=None, in_edges=[e_00_12, e_01_12, e_02_12], out_edges=[e_12_20, e_12_21, e_12_22])
    x_2_0 = ARNode(name="x_2_0", ar_type=None, in_edges=[e_10_20, e_11_20, e_12_20], out_edges=[e_20_y])
    x_2_1 = ARNode(name="x_2_1", ar_type=None, in_edges=[e_10_21, e_11_21, e_12_21], out_edges=[e_21_y])
    x_2_2 = ARNode(name="x_2_2", ar_type=None, in_edges=[e_10_22, e_11_22, e_12_22], out_edges=[e_22_y])
    y = ARNode(name="y", ar_type=None, in_edges=[e_20_y, e_21_y, e_22_y], out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1, x_0_2])
    h1_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1, x_1_2])
    h2_layer = Layer(type_name="hidden", nodes=[x_2_0, x_2_1, x_2_2])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h1_layer, h2_layer, output_layer])
    eval_output = net.evaluate(input_values={0:2, 1:1, 2:-1})
    speedy_eval_output = net.speedy_evaluate(input_values={0:2, 1:1, 2:-1})
    assert is_evaluation_result_equal(eval_output.items(), speedy_eval_output)
    assert eval_output['y'] == -420


def test_evaluate_3():
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    filename = "ACASXU_run2a_1_1_batch_2000.nnet"
    nnet_filename = os.path.join(nnet_dir, filename)
    net = net_from_nnet_file(nnet_filename)
    res = net.evaluate({0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
    speedy_res = net.speedy_evaluate({0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
    assert is_evaluation_result_equal(res.items(), speedy_res)
    #for i in range(len(net.layers[-1].nodes)):
    #    print(i)
    #    assert res["x_7_{}".format(i)] == 0.0
    eval_output = net.evaluate(input_values={0: 0.25, 1: 0.3, 2: -0.2, 3: -0.1, 4: 0.15})
    speedy_eval_output = net.speedy_evaluate(input_values={0: 0.25, 1: 0.3, 2: -0.2, 3: -0.1, 4: 0.15})
    assert is_evaluation_result_equal(eval_output.items(), speedy_eval_output)
    # print(eval_output)
    # print(speedy_eval_output)


def test_evaluate_marabou_equality():
    """
    validates equality between the outputs of marabou/Net evaluations for random inputs.
    :param nnet_filname: path to nnet file (file that includes representation of neural network)
    """
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    filename = "ACASXU_run2a_1_1_batch_2000.nnet"
    nnet_filename = os.path.join(nnet_dir, filename)
    # prepare 2 nets
    ar_net = net_from_nnet_file(nnet_filename)
    marabou_net = MarabouCore.load_network(nnet_filename)

    # prepare 2 input (and output) parameters
    input_vals = [random.random() for i in range(MarabouCore.num_inputs(marabou_net))]
    ar_input = {i:val for i,val in enumerate(input_vals)}
    output = [0.0 for i in range(MarabouCore.num_outputs(marabou_net))]

    # run 2 evaluations and validate equality
    ar_res = ar_net.evaluate(ar_input)
    speedy_ar_res = ar_net.speedy_evaluate(ar_input)
    assert is_evaluation_result_equal(ar_res.items(), speedy_ar_res)
    MarabouCore.evaluate_network(marabou_net, input_vals, output, False, False)
    print(f"ar_res={ar_res}")
    print(f"speedy_ar_res ={speedy_ar_res }")
    print(f"output={output}")
    print(len(ar_res))
    print(len(output))
    assert len(ar_res) == len(output)
    assert ar_res.values() == output
    MarabouCore.destroy_network(marabou_net)


def test_split_pos_neg_1():
    e_11_21 = Edge(src="x11", dest="x21", weight=1)
    e_11_22 = Edge(src="x11", dest="x22", weight=2)
    e_12_21 = Edge(src="x12", dest="x21", weight=3)
    e_12_22 = Edge(src="x12", dest="x22", weight=4)
    e_21_31 = Edge(src="x21", dest="x31", weight=-1)
    e_21_32 = Edge(src="x21", dest="x32", weight=2)
    e_22_31 = Edge(src="x22", dest="x31", weight=2)
    e_22_32 = Edge(src="x22", dest="x32", weight=-3)
    e_31_y = Edge(src="x31", dest="y", weight=-4)
    e_32_y = Edge(src="x32", dest="y", weight=6)

    # nodes
    x11 = ARNode(name="x11", ar_type=None, in_edges=[],
                 out_edges=[e_11_21, e_11_22])
    x12 = ARNode(name="x12", ar_type=None, in_edges=[],
                 out_edges=[e_12_21, e_12_22])
    x21 = ARNode(name="x21", ar_type=None,
                 in_edges=[e_11_21, e_12_21],
                 out_edges=[e_21_31, e_21_32])
    x22 = ARNode(name="x22", ar_type=None,
                 in_edges=[e_11_22, e_12_22],
                 out_edges=[e_22_31, e_22_32])
    x31 = ARNode(name="x31", ar_type=None,
                 in_edges=[e_21_31, e_22_31],
                 out_edges=[e_31_y])
    x32 = ARNode(name="x32", ar_type=None,
                 in_edges=[e_21_32, e_22_32],
                 out_edges=[e_32_y])
    y = ARNode(name="y", ar_type=None,
               in_edges=[e_31_y, e_32_y],
               out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x11,x12])
    h1_layer = Layer(type_name="hidden", nodes=[x21,x22])
    h2_layer = Layer(type_name="hidden", nodes=[x31,x32])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h1_layer, h2_layer, output_layer])
    net.preprocess_split_pos_neg()
    print(net)
    net.visualize(title="test_split_pos_neg_1")


def test_split_pos_neg_2():
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    filename = "ACASXU_run2a_1_1_batch_2000.nnet"
    nnet_filename = os.path.join(nnet_dir, filename)
    net = net_from_nnet_file(nnet_filename)
    net.preprocess_split_pos_neg()
    print(net)
    #net.visualize(title="test_split_pos_neg_1")



def test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_1():
    e_11_21 = Edge(src="x11", dest="x21", weight=-1)
    e_11_22 = Edge(src="x11", dest="x22", weight=2)
    e_12_21 = Edge(src="x12", dest="x21", weight=-3)
    e_12_22 = Edge(src="x12", dest="x22", weight=4)
    e_21_31 = Edge(src="x21", dest="x31", weight=-1)
    e_21_32 = Edge(src="x21", dest="x32", weight=-2)
    e_22_31 = Edge(src="x22", dest="x31", weight=2)
    e_22_32 = Edge(src="x22", dest="x32", weight=-3)
    e_31_y = Edge(src="x31", dest="y", weight=-4)
    e_32_y = Edge(src="x32", dest="y", weight=6)

    # nodes
    x11 = ARNode(name="x11", ar_type=None, in_edges=[],
                 out_edges=[e_11_21, e_11_22])
    x12 = ARNode(name="x12", ar_type=None, in_edges=[],
                 out_edges=[e_12_21, e_12_22])
    x21 = ARNode(name="x21", ar_type=None,
                 in_edges=[e_11_21, e_12_21],
                 out_edges=[e_21_31, e_21_32])
    x22 = ARNode(name="x22", ar_type=None,
                 in_edges=[e_11_22, e_12_22],
                 out_edges=[e_22_31, e_22_32])
    x31 = ARNode(name="x31", ar_type=None,
                 in_edges=[e_21_31, e_22_31],
                 out_edges=[e_31_y])
    x32 = ARNode(name="x32", ar_type=None,
                 in_edges=[e_21_32, e_22_32],
                 out_edges=[e_32_y])
    y = ARNode(name="y", ar_type=None,
               in_edges=[e_31_y, e_32_y],
               out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x11,x12])
    h1_layer = Layer(type_name="hidden", nodes=[x21,x22])
    h2_layer = Layer(type_name="hidden", nodes=[x31,x32])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h1_layer, h2_layer, output_layer])

    N = 100
    orig_net = copy.deepcopy(net)
    abstract_net = copy.deepcopy(net.abstract())
    refined_net = net.refine()
    for j in range(N):
        input_values = {i:(0.5-random.random()) * 2 for i in range(len(net.layers[0].nodes))}
        orig_net_output = sorted(orig_net.evaluate(input_values).items(), key=lambda x: x[0])
        speedy_orig_net_output = orig_net.speedy_evaluate(input_values)
        assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
        abstract_net_output = sorted(abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
        speedy_abstract_net_output = abstract_net.speedy_evaluate(input_values)
        assert is_evaluation_result_equal(abstract_net_output, speedy_abstract_net_output)
        refined_net_output = sorted(refined_net.evaluate(input_values).items(), key=lambda x: x[0])
        speedy_refined_net_output = refined_net.speedy_evaluate(input_values)
        assert is_evaluation_result_equal(refined_net_output, speedy_refined_net_output)
        assert(len(orig_net_output) == len(abstract_net_output))
        assert(len(refined_net_output) == len(abstract_net_output))
        for k in range(len(orig_net_output)):
            assert orig_net_output[k][1] <= refined_net_output[k][1] <= abstract_net_output[k][1]


def test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_2():
    e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=10)
    e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=5)
    e_00_12 = Edge(src="x_0_0", dest="x_1_2", weight=1)
    e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=2)
    e_01_11 = Edge(src="x_0_1", dest="x_1_1", weight=4)
    e_01_12 = Edge(src="x_0_1", dest="x_1_2", weight=7)
    e_02_10 = Edge(src="x_0_2", dest="x_1_0", weight=-3)
    e_02_11 = Edge(src="x_0_2", dest="x_1_1", weight=-8)
    e_02_12 = Edge(src="x_0_2", dest="x_1_2", weight=-7)
    e_10_20 = Edge(src="x_1_0", dest="x_2_0", weight=5)
    e_10_21 = Edge(src="x_1_0", dest="x_2_1", weight=4)
    e_10_22 = Edge(src="x_1_0", dest="x_2_2", weight=9)
    e_11_20 = Edge(src="x_1_1", dest="x_2_0", weight=-7)
    e_11_21 = Edge(src="x_1_1", dest="x_2_1", weight=-1)
    e_11_22 = Edge(src="x_1_1", dest="x_2_2", weight=-2)
    e_12_20 = Edge(src="x_1_2", dest="x_2_0", weight=-5)
    e_12_21 = Edge(src="x_1_2", dest="x_2_1", weight=-3)
    e_12_22 = Edge(src="x_1_2", dest="x_2_2", weight=-6)
    e_20_y = Edge(src="x_2_0", dest="y", weight=4)
    e_21_y = Edge(src="x_2_1", dest="y", weight=3)
    e_22_y = Edge(src="x_2_2", dest="y", weight=-6)

    # nodes
    x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11, e_00_12])
    x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11, e_01_12])
    x_0_2 = ARNode(name="x_0_2", ar_type=None, in_edges=[], out_edges=[e_02_10, e_02_11, e_02_12])
    x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10, e_02_10], out_edges=[e_10_20, e_10_21, e_10_22])
    x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11, e_02_11], out_edges=[e_11_20, e_11_21, e_11_22])
    x_1_2 = ARNode(name="x_1_2", ar_type=None, in_edges=[e_00_12, e_01_12, e_02_12], out_edges=[e_12_20, e_12_21, e_12_22])
    x_2_0 = ARNode(name="x_2_0", ar_type=None, in_edges=[e_10_20, e_11_20, e_12_20], out_edges=[e_20_y])
    x_2_1 = ARNode(name="x_2_1", ar_type=None, in_edges=[e_10_21, e_11_21, e_12_21], out_edges=[e_21_y])
    x_2_2 = ARNode(name="x_2_2", ar_type=None, in_edges=[e_10_22, e_11_22, e_12_22], out_edges=[e_22_y])
    y = ARNode(name="y", ar_type=None, in_edges=[e_20_y, e_21_y, e_22_y], out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1, x_0_2])
    h1_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1, x_1_2])
    h2_layer = Layer(type_name="hidden", nodes=[x_2_0, x_2_1, x_2_2])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h1_layer, h2_layer, output_layer])
    net.visualize(title="orig")
    N = 1000
    orig_net = copy.deepcopy(net)
    abstract_net = copy.deepcopy(net.abstract())
    print("before refine")
    refined_net = net.refine()
    print("after refine")
    for j in range(N):
        input_values = {i:(0.5-random.random()) * 2 for i in range(len(net.layers[0].nodes))}
        orig_net_output = sorted(orig_net.evaluate(input_values).items(), key=lambda x: x[0])
        speedy_orig_net_output = orig_net.speedy_evaluate(input_values)
        assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
        abstract_net_output = sorted(abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
        speedy_abstract_net_output = abstract_net.speedy_evaluate(input_values)
        assert is_evaluation_result_equal(abstract_net_output, speedy_abstract_net_output)
        refined_net_output = sorted(refined_net.evaluate(input_values).items(), key=lambda x: x[0])
        speedy_refined_net_output = refined_net.speedy_evaluate(input_values)
        assert is_evaluation_result_equal(refined_net_output, speedy_refined_net_output)
        assert(len(orig_net_output) == len(abstract_net_output))
        assert(len(refined_net_output) == len(abstract_net_output))
        for k in range(len(orig_net_output)):
            if (orig_net_output[k][1] > refined_net_output[k][1]
                and abs(orig_net_output[k][1] - refined_net_output[k][1]) > EPSILON)\
                    or (refined_net_output[k][1] > abstract_net_output[k][1]
                        and abs(refined_net_output[k][1] - abstract_net_output[k][1]) > EPSILON):
                print ("error: orig is bigger than test_refinement's output or test_refinement is bigger than abstract output")
                print("input_values: {}".format(input_values.items()))
                print("orig_net_output[k][1] = {}".format(orig_net_output[k][1]))
                print("refined_net_output[k][1] = {}".format(refined_net_output[k][1]))
                print("abstract_net_output[k][1] ({}) = {}".format(abstract_net_output[k][1]))
                assert False


def make_all_edges_positive(net):
    for layer in net.layers:
        for node in layer.nodes:
            for out_edge in node.out_edges:
                out_edge.weight = abs(out_edge.weight)
            for in_edge in node.in_edges:
                in_edge.weight = abs(in_edge.weight)
    return net


def test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_3():
    """
    test to validate that test_abstraction outputs >= test_refinement outputs >= original outputs:

    for every ACASXU network, generate test_abstraction then evaluate 100 inputs and
    assert that the original net output is smaller than the refined net output
    and that the refined net output is smaller than the abstract net output
    """
    N = 100
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    # random.seed(9001)
    for i,filename in enumerate(os.listdir(nnet_dir)):  # eg ACASXU_run2a_1_1_batch_2000.nnet
        debug_print("{} {}".format(i, filename))
        nnet_filename = os.path.join(nnet_dir, filename)
        net = net_from_nnet_file(nnet_filename)
        orig_net = copy.deepcopy(net)
        abstract_net = copy.deepcopy(net.abstract())
        refined_net = net.refine(sequence_length=10)
        for j in range(N):
            input_values = {i:(0.5-random.random()) * 2 for i in range(len(net.layers[0].nodes))}
            orig_net_output = sorted(orig_net.evaluate(input_values).items(), key=lambda x: x[0])
            orig_net_speedy_output = orig_net.speedy_evaluate(input_values)
            assert is_evaluation_result_equal(orig_net_output, orig_net_speedy_output)
            # assert all(np.array([x[1] for x in orig_net_output]) - np.array(orig_net_speedy_output) < EPSILON)
            abstract_net_output = sorted(abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
            abstract_net_speedy_output = abstract_net.speedy_evaluate(input_values)
            assert is_evaluation_result_equal(abstract_net_output, abstract_net_speedy_output)
            # assert all(np.array([x[1] for x in abstract_net_output]) - np.array(abstract_net_speedy_output) < EPSILON)
            refined_net_output = sorted(refined_net.evaluate(input_values).items(), key=lambda x: x[0])
            refined_net_speedy_output = refined_net.speedy_evaluate(input_values)
            assert is_evaluation_result_equal(refined_net_output, refined_net_speedy_output)
            # assert all(np.array([x[1] for x in refined_net_output]) - np.array(refined_net_speedy_output) < EPSILON)

            assert(len(orig_net_output) == len(abstract_net_output))
            assert(len(refined_net_output) == len(abstract_net_output))
            for k in range(len(orig_net_output)):
                if (orig_net_output[k][1] - refined_net_output[k][1]) > EPSILON or \
                        (refined_net_output[k][1] - abstract_net_output[k][1]) > EPSILON:
                    msg = "test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_3"
                    embed_ipython(debug_message=msg, verbose=True)
                    assert False


def generate_random_layers_sizes(layers_sizes=None):
    """
    generate random layers_sizes
    if layers_sizes is empty, choose random number of layers between 3-8
    for each layer with None value, generate random size
    @layers_sizes map from layer index to layer size or None
    """
    MIN_LAYERS = 3
    MAX_LAYERS = 8
    MIN_NODES = 2
    MAX_NODES = 50
    if layers_sizes is None:
        num_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    else:
        num_layers = len(layers_sizes)
    for i in range(num_layers):
        layers_sizes[i] = layers_sizes.get(i, None)
        if layers_sizes[i] is None:
            layers_sizes[i] = random.randint(MIN_NODES, MAX_NODES)
    return layers_sizes


def creat_random_network(layers_sizes):
    """
    create net with layers sizes a.t to @layers_sizes and random edges weights
    @layers_sizes map from layer index to layer size
    """
    #TODO IMPLEMENT
    """
    layers_sizes = generate_random_layers_sizes(layers_sizes)
    edges = []
    for layer_index, layer_size in layers_sizes.items():
        for layer_index, layer_size in layers_sizes.items():
            edges.append(Edge(src=))
    nodes = []
    for edge in edges:
        ### TODO implement
        nodes.append(node)
    """
    pass


def create_2_2_1_rand_weihts_net():
    e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=((0.5-random.random()) * 2))
    e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=((0.5-random.random()) * 2))
    e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=((0.5-random.random()) * 2))
    e_01_11 = Edge(src="x_0_1", dest="x_1_1", weight=((0.5-random.random()) * 2))
    e_10_y = Edge(src="x_1_0", dest="y", weight=((0.5-random.random()) * 2))
    e_11_y = Edge(src="x_1_1", dest="y", weight=((0.5-random.random()) * 2))

    # nodes
    x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11])
    x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11])
    x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10], out_edges=[e_10_y])
    x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11], out_edges=[e_11_y])
    y = ARNode(name="y", ar_type=None, in_edges=[e_10_y, e_11_y], out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1])
    h1_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h1_layer, output_layer])
    return net


def create_2_2_2_1_rand_weihts_net():
    e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=((0.5-random.random()) * 2))
    e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=((0.5-random.random()) * 2))
    e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=((0.5-random.random()) * 2))
    e_01_11 = Edge(src="x_0_1", dest="x_1_1", weight=((0.5-random.random()) * 2))
    e_10_20 = Edge(src="x_1_0", dest="x_2_0", weight=((0.5-random.random()) * 2))
    e_10_21 = Edge(src="x_1_0", dest="x_2_1", weight=((0.5-random.random()) * 2))
    e_11_20 = Edge(src="x_1_1", dest="x_2_0", weight=((0.5-random.random()) * 2))
    e_11_21 = Edge(src="x_1_1", dest="x_2_1", weight=((0.5-random.random()) * 2))
    e_20_y = Edge(src="x_2_0", dest="y", weight=((0.5-random.random()) * 2))
    e_21_y = Edge(src="x_2_1", dest="y", weight=((0.5-random.random()) * 2))

    # nodes
    x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11])
    x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11])
    x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10], out_edges=[e_00_10, e_00_11])
    x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11], out_edges=[e_01_10, e_01_11])
    x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10], out_edges=[e_10_20, e_10_21])
    x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11], out_edges=[e_11_20, e_11_21])
    x_2_0 = ARNode(name="x_2_0", ar_type=None, in_edges=[e_10_20, e_11_20], out_edges=[e_20_y])
    x_2_1 = ARNode(name="x_2_1", ar_type=None, in_edges=[e_10_21, e_11_21], out_edges=[e_21_y])
    y = ARNode(name="y", ar_type=None, in_edges=[e_20_y, e_21_y], out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1])
    h1_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1])
    h2_layer = Layer(type_name="hidden", nodes=[x_2_0, x_2_1])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h1_layer, h2_layer, output_layer])
    return net


def create_3_3_3_1_rand_weihts_net():
    e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=(0.5-random.random()) * 2)
    e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=(0.5-random.random()) * 2)
    e_00_12 = Edge(src="x_0_0", dest="x_1_2", weight=(0.5-random.random()) * 2)
    e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=(0.5-random.random()) * 2)
    e_01_11 = Edge(src="x_0_1", dest="x_1_1", weight=(0.5-random.random()) * 2)
    e_01_12 = Edge(src="x_0_1", dest="x_1_2", weight=(0.5-random.random()) * 2)
    e_02_10 = Edge(src="x_0_2", dest="x_1_0", weight=(0.5-random.random()) * 2)
    e_02_11 = Edge(src="x_0_2", dest="x_1_1", weight=(0.5-random.random()) * 2)
    e_02_12 = Edge(src="x_0_2", dest="x_1_2", weight=(0.5-random.random()) * 2)
    e_10_20 = Edge(src="x_1_0", dest="x_2_0", weight=(0.5-random.random()) * 2)
    e_10_21 = Edge(src="x_1_0", dest="x_2_1", weight=(0.5-random.random()) * 2)
    e_10_22 = Edge(src="x_1_0", dest="x_2_2", weight=(0.5-random.random()) * 2)
    e_11_20 = Edge(src="x_1_1", dest="x_2_0", weight=(0.5-random.random()) * 2)
    e_11_21 = Edge(src="x_1_1", dest="x_2_1", weight=(0.5-random.random()) * 2)
    e_11_22 = Edge(src="x_1_1", dest="x_2_2", weight=(0.5-random.random()) * 2)
    e_12_20 = Edge(src="x_1_2", dest="x_2_0", weight=(0.5-random.random()) * 2)
    e_12_21 = Edge(src="x_1_2", dest="x_2_1", weight=(0.5-random.random()) * 2)
    e_12_22 = Edge(src="x_1_2", dest="x_2_2", weight=(0.5-random.random()) * 2)
    e_20_y = Edge(src="x_2_0", dest="y", weight=(0.5-random.random()) * 2)
    e_21_y = Edge(src="x_2_1", dest="y", weight=(0.5-random.random()) * 2)
    e_22_y = Edge(src="x_2_2", dest="y", weight=(0.5-random.random()) * 2)

    # nodes
    x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11, e_00_12])
    x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11, e_01_12])
    x_0_2 = ARNode(name="x_0_2", ar_type=None, in_edges=[], out_edges=[e_02_10, e_02_11, e_02_12])
    x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10, e_02_10], out_edges=[e_10_20, e_10_21, e_10_22])
    x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11, e_02_11], out_edges=[e_11_20, e_11_21, e_11_22])
    x_1_2 = ARNode(name="x_1_2", ar_type=None, in_edges=[e_00_12, e_01_12, e_02_12], out_edges=[e_12_20, e_12_21, e_12_22])
    x_2_0 = ARNode(name="x_2_0", ar_type=None, in_edges=[e_10_20, e_11_20, e_12_20], out_edges=[e_20_y])
    x_2_1 = ARNode(name="x_2_1", ar_type=None, in_edges=[e_10_21, e_11_21, e_12_21], out_edges=[e_21_y])
    x_2_2 = ARNode(name="x_2_2", ar_type=None, in_edges=[e_10_22, e_11_22, e_12_22], out_edges=[e_22_y])
    y = ARNode(name="y", ar_type=None, in_edges=[e_20_y, e_21_y, e_22_y], out_edges=[])

    # layers
    input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1, x_0_2])
    h1_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1, x_1_2])
    h2_layer = Layer(type_name="hidden", nodes=[x_2_0, x_2_1, x_2_2])
    output_layer = Layer(type_name="output", nodes=[y])

    # net
    net = Net(layers=[input_layer, h1_layer, h2_layer, output_layer])
    return net


def test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_4():
    create_network_func = create_2_2_1_rand_weihts_net
    test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs(create_network_func=create_network_func)


def test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_5():
    create_network_func = create_2_2_2_1_rand_weihts_net
    test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs(create_network_func=create_network_func)


def test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_6():
    create_network_func = create_3_3_3_1_rand_weihts_net
    test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs(create_network_func=create_network_func)


def test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs(create_network_func):
    """
    for 1-50:
        create network with random weights,
        abstract network,
        for 1-100:
            generate random input
            check that abstract output > test_refinement output > original output
    @create_network_func method that generate net with specific layers sizes
    """
    N = 50
    # layers_sizes = {0:2, 1:2, 2:1}
    for i in range(100):
        # net = creat_random_network(layers_sizes)
        net = create_network_func()
        orig_net = copy.deepcopy(net)
        abstract_net = copy.deepcopy(net.abstract())
        refined_net = net.refine()
        for j in range(N):
            input_values = {i: (0.5-random.random()) * 2 for i in range(len(net.layers[0].nodes))}
            orig_net_output = sorted(orig_net.evaluate(input_values).items(), key=lambda x: x[0])
            speedy_orig_net_output = orig_net.speedy_evaluate(input_values)
            assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
            abstract_net_output = sorted(abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
            speedy_abstract_net_output = abstract_net.speedy_evaluate(input_values)
            assert is_evaluation_result_equal(abstract_net_output, speedy_abstract_net_output)
            refined_net_output = sorted(refined_net.evaluate(input_values).items(), key=lambda x: x[0])
            speedy_refined_net_output = refined_net.speedy_evaluate(input_values)
            assert is_evaluation_result_equal(refined_net_output, speedy_refined_net_output)
            assert(len(orig_net_output) == len(abstract_net_output))
            assert(len(refined_net_output) == len(abstract_net_output))
            for k in range(len(orig_net_output)):
                if (orig_net_output[k][1] - refined_net_output[k][1]) > EPSILON or \
                        (refined_net_output[k][1] - abstract_net_output[k][1]) > EPSILON:
                    import IPython
                    IPython.embed()
                    print(i,j)
                    assert False


def test_heuristic_abstract_smaller_than_complete_abstract_outputs_3():
    """
    test to validate that complete test_abstraction outputs >= heuristic test_abstraction outputs >= original net outputs:

    for every ACASXU network, generate complete/heristic test_abstraction, then evaluate 100 inputs and
    assert that the original net output is smaller than the heuristic abstracted net output
    and that the heuristic abstracted net output is smaller than the complete abstract net output
    """
    N = 100
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    test_property_1 = get_test_property_acas()
    for i, filename in enumerate(os.listdir(nnet_dir)):  # eg ACASXU_run2a_1_1_batch_2000.nnet
        # if i != 6:
        #     continue
        debug_print("{} {}".format(i, filename))
        nnet_filename = os.path.join(nnet_dir, filename)
        net = net_from_nnet_file(nnet_filename)
        orig_net = copy.deepcopy(net)
        complete_abstract_net = copy.deepcopy(net).abstract()
        heuristic_abstract_net = copy.deepcopy(orig_net).heuristic_abstract(test_property_1)
        debug_print("filename={}".format(nnet_filename))
        continue
        assert orig_net.get_general_net_data()["num_nodes"] <= \
               complete_abstract_net.get_general_net_data()["num_nodes"] <= \
               heuristic_abstract_net.get_general_net_data()["num_nodes"]
        for j in range(N):
            input_values = {i: (0.5-random.random()) * 2 for i in range(len(net.layers[0].nodes))}
            input_values = net.get_limited_random_input(test_property_1)
            orig_net_output = sorted(orig_net.evaluate(input_values).items(), key=lambda x: x[0])
            speedy_orig_net_output = orig_net.speedy_evaluate(input_values)
            assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
            complete_abstract_net_output = sorted(complete_abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
            speedy_complete_abstract_net_output = complete_abstract_net.speedy_evaluate(input_values)
            assert is_evaluation_result_equal(complete_abstract_net_output, speedy_complete_abstract_net_output)
            heuristic_abstract_net_output = sorted(heuristic_abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
            speedy_heuristic_abstract_net_output = heuristic_abstract_net.speedy_evaluate(input_values)
            assert is_evaluation_result_equal(heuristic_abstract_net_output, speedy_heuristic_abstract_net_output)

            assert(len(orig_net_output) == len(complete_abstract_net_output))
            assert(len(heuristic_abstract_net_output) == len(complete_abstract_net_output))
            for k in range(len(orig_net_output)):
                if (orig_net_output[k][1] - heuristic_abstract_net_output[k][1]) > EPSILON or \
                        (heuristic_abstract_net_output[k][1] - complete_abstract_net_output[k][1]) > EPSILON:
                    import IPython
                    IPython.embed()
                    assert False


def test_heuristic_abstract_smaller_than_complete_abstract_outputs_4():
    create_network_func = create_2_2_1_rand_weihts_net
    test_property = get_test_property_input_2_output_1()
    test_heuristic_abstract_smaller_than_complete_abstract_outputs(create_network_func=create_network_func,
                                                                   test_property=test_property)


def test_heuristic_abstract_smaller_than_complete_abstract_outputs_5():
    create_network_func = create_2_2_2_1_rand_weihts_net
    test_property = get_test_property_input_2_output_1()
    test_heuristic_abstract_smaller_than_complete_abstract_outputs(create_network_func=create_network_func,
                                                                   test_property=test_property)


def test_heuristic_abstract_smaller_than_complete_abstract_outputs_6():
    create_network_func = create_3_3_3_1_rand_weihts_net
    test_property = get_test_property_input_3_output_1()
    test_heuristic_abstract_smaller_than_complete_abstract_outputs(create_network_func=create_network_func,
                                                                   test_property=test_property)


def test_heuristic_abstract_smaller_than_complete_abstract_outputs(create_network_func, test_property):
    """
    for 1-50:
        create network with random weights,
        generate complete and heristic abstract networks,
        for 1-100:
            generate random input
            check that heuristically abstracted net's output < complete abstracted net's output
    @create_network_func method that generate net with specific layers sizes
    """
    N = 50
    # layers_sizes = {0:2, 1:2, 2:1}
    for i in range(100):
        # net = creat_random_network(layers_sizes)
        net = create_network_func()
        orig_net = copy.deepcopy(net)
        complete_abstract_net = copy.deepcopy(net.abstract())
        # print(i)
        heuristic_abstract_net = copy.deepcopy(orig_net).heuristic_abstract(test_property)
        for j in range(N):
            input_values = {i: (0.5-random.random()) * 2 for i in range(len(net.layers[0].nodes))}
            orig_net_output = sorted(orig_net.evaluate(input_values).items(), key=lambda x: x[0])
            speedy_orig_net_output = orig_net.speedy_evaluate(input_values)
            assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
            complete_abstract_net_output = sorted(complete_abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
            speedy_complete_abstract_net_output = complete_abstract_net.speedy_evaluate(input_values)
            assert is_evaluation_result_equal(complete_abstract_net_output, speedy_complete_abstract_net_output)
            heuristic_net_output = sorted(heuristic_abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
            speedy_heuristic_net_output = heuristic_abstract_net.speedy_evaluate(input_values)
            assert is_evaluation_result_equal(heuristic_net_output, speedy_heuristic_net_output)
            assert(len(orig_net_output) == len(complete_abstract_net_output))
            assert(len(heuristic_net_output) == len(complete_abstract_net_output))
            for k in range(len(orig_net_output)):
                if orig_net_output[k][1] > heuristic_net_output[k][1] or heuristic_net_output[k][1] > complete_abstract_net_output[k][1]:
                    print("orig_net_output[{}][1] > heuristic_net_output[{}][1] or "
                          "heuristic_net_output[{}][1] > complete_abstract_net_output[{}][1]".format(k, k, k, k))
                    print(i, j)
                    assert False


def test_net_from_nnet_file_acas_tiny_nets():
    outputs = [
        {'0': -0.22975206607673998, '1': 0.007867314468542996, '2': 0.875059087916634, '3': 0.10108754874669001, '4': 0.078946838829464},
        {'0': -1.293840174985929, '1': -0.25361091752618714, '2': -1.1841821124083598, '3': -1.9064481569480283, '4': -2.422375480309849},
        {'0': 0.49004888651636336, '1': -1.9145215075374988, '2': 2.2510454229235766, '3': -0.7326971935354969, '4': 1.7257435902500207},
        {'0': -9.613632366207742, '1': -1.7802944197353936, '2': -1.570224805467037, '3': 2.900541518215031, '4': -7.767607497595311},
        {'0': 1.34936633469572, '1': -16.246148925939426, '2': -2.0212639644716854, '3': -17.584466399124917, '4': -1.925012406235469}
    ]
    tiny_nnet_dir = PATH_TO_MARABOU_ACAS_EXAMPLES
    # nnet_filename = os.path.join(tiny_nnet_dir, "ACASXU_run2a_1_1_tiny_5.nnet")
    nets = [fname for fname in os.listdir(tiny_nnet_dir) if "tiny" in fname and fname.endswith(".nnet")]
    for i,nnet_filename in enumerate(sorted(nets)):
        net = net_from_nnet_file(os.path.join(tiny_nnet_dir, nnet_filename))
        input_values = {0: 0.6000, 1: -0.5000, 2: -0.5000, 3: 0.4500, 4: -0.4500}
        output = net.evaluate(input_values)
        speedy_output = net.speedy_evaluate(input_values)
        assert is_evaluation_result_equal(output.items(), speedy_output)
        # the names of nodes are different so work with node index
        for y_index, y_value in outputs[i].items():
            index = [name for name in output.keys() if name.endswith(y_index)][0]
            output_val = output[index]
            assert(output_val == y_value)


def test_net_from_nnet_file_acas_net_1_1():
    correct_output = {'0': -0.022128988677506706, '1': -0.01904528058874152, '2': -0.019140123561458566, '3': -0.019152128000934545, '4': -0.019168840924865056}
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    nnet_filename = os.path.join(nnet_dir, "ACASXU_run2a_1_1_batch_2000.nnet")
    net = net_from_nnet_file(nnet_filename)
    input_values = {0: 0.6000, 1: -0.5000, 2: -0.5000, 3: 0.4500, 4: -0.4500}
    output = net.evaluate(input_values)
    speedy_output = net.speedy_evaluate(input_values)
    assert is_evaluation_result_equal(output.items(), speedy_output)
    for y_index, y_value in correct_output.items():
        index = [name for name in output.keys() if name.endswith(y_index)][0]
        output_val = output[index]
        assert(output_val == y_value)

    assert output
    #y0 =  -0.0221
    #y1 =  -0.0190
    #y2 =  -0.0191
    #y3 =  -0.0192
    #y4 =  -0.0192


if __name__ == "__main__":
    # check preprocess: pos-neg split
    #test_split_pos_neg_1()
    #test_split_pos_neg_2()

    # __eq__
    #test_net_eq_sanity_check()

    # test_abstraction
    # example_1()
    # example_1_1()

    # minimal test_abstraction
    # test_minimal_abstract_01()  # while loop does not end, should specify a better test_property
    # test_minimal_abstract_all_acas_nets()

    # test_abstraction and test_refinement
    # test_abstract_and_refine_result_with_orig_1()
    # test_abstract_and_refine_result_with_orig_2()
    # test_abstract_and_refine_result_with_orig_3()
    # test_abstract_and_refine_result_with_orig_acas()
    # test_abstract_and_refine_result_with_orig_all_acas()
    # example_2()

    # marabou query
    # example_3()

    # net from acasxu nnet file
    # example_4()

    # test_abstraction and test_refinement using net from acasxu nnet file
    # example_5()

    # do experiments of property_1 from Cav17 on one network
    #experiment_1()
    #experiment_2_tiny()
    #experiments()

    # results = []
    # for nnet_filename,is_tiny in [
    #     #("ACASXU_run2a_1_1_tiny_4.nnet", True),
    #     #("ACASXU_run2a_1_1_tiny_5.nnet", True),
    #     # ("ACASXU_run2a_1_9_batch_2000.nnet", False),
    #     # ("ACASXU_run2a_1_8_batch_2000.nnet", False),
    #     # ("ACASXU_run2a_1_7_batch_2000.nnet", False),
    #     # ("ACASXU_run2a_1_1_batch_2000.nnet", False),
    #     # ("ACASXU_run2a_1_2_batch_2000.nnet", False),
    #     # ("ACASXU_run2a_5_3_batch_2000.nnet", False),
    #     ("ACASXU_run2a_1_6_batch_2000.nnet", False),
    #     ]:
    #     if is_tiny:
    #         DIR_PATH = PATH_TO_MARABOU_ACAS_EXAMPLES
    #     else:
    #         DIR_PATH = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    #     fullname = os.path.join(DIR_PATH, nnet_filename)
    #     debug_print("one_experiment({})".format(fullname))
    #     results.append(one_experiment(fullname, is_tiny=is_tiny))
    #     #results.append(one_experiment(, is_tiny=True))
    # for res in results:
    #     debug_print(res)


    # test_evaluate_1()
    # test_evaluate_2()
    # test_evaluate_3()
    # the next test checks that our eval func and marabou eval func outputs hte same values.
    # the test is wrt some net and random input
    # the test fails because the output that marabou returns is [0, ..., 0] because python-cpp api problem
    # you can see the print-outs that approve the equality of the outputs
    # test_evaluate_marabou_equality()

    # -------------------------------------------------------------------------
    # test that abstract output >= test_refinement output >= original output

    # const net 2_2_2_1, N=100 random inputs
    #12 test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_1()

    # # const net 3_3_3_1, N=1000 random inputs
    #11 test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_2()
    #
    # # acas-xu networks, N=100
    #10 test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_3()
    #
    # # random weighted 100 nets, all are 2_2_1, N=100 random inputs
    #9 test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_4()
    #
    # # random weighted 100 nets, all are 2_2_2_1, N=100 random inputs
    #8 test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_5()
    #
    # # random weigted 100 nets, all are 3_3_3_1, N=100 random inputs
    #7 test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_6()
    #
    # # -------------------------------------------------------------------------
    # # test that abstract output >= heuristic output >= original output
    #
    #6 # acas-xu networks, N=100
    test_heuristic_abstract_smaller_than_complete_abstract_outputs_3()
    #
    # # random weigted 100 nets, all are 2_2_1, N=100 random inputs
    #5 test_heuristic_abstract_smaller_than_complete_abstract_outputs_4()
    #
    # # (DID NOT PASSED YET) random weigted 100 nets, all are 2_2_2_1, N=100 random inputs
    #4 test_heuristic_abstract_smaller_than_complete_abstract_outputs_5()
    #
    # # random weigted 100 nets, all are 3_3_3_1, N=100 random inputs
    #3 test_heuristic_abstract_smaller_than_complete_abstract_outputs_6()

    # -------------------------------------------------------------------------
    # test net_from_nnet_file by checking that the same output is accepted

    #2 test_net_from_nnet_file_acas_tiny_nets()
    #1 test_net_from_nnet_file_acas_net_1_1()
