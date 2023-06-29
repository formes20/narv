import os
import import_marabou
from maraboupy import MarabouNetworkNNet
from maraboupy import MarabouNetworkONNX
from core.configuration.consts import (
    PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
)
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Layer import Layer
from core.data_structures.Network import Network
from core.utils.activation_functions import relu


def network_from_nnet_file(nnet_filename: str) -> Network:
    """
    generate Net instance which is equivalent to given nnet formatted network
    @nnet_filename fullpath of nnet file, see maraboupy.MarabouNetworkNNet
    under the root dir of git marabou project
    :return: Network object
    """
    # read nnetfile into Marabou Network
    # It's also feasible for MNIST networks
    acasxu_net = MarabouNetworkNNet.MarabouNetworkNNet(filename=nnet_filename)

    # list of layers, layer include list of nodes, node include list of Edge-s
    # notice the acasxu.weights include weights of input edges
    edges = []  # list of list of list of Edge instances
    # edges[i] is list of list of edges between layer i to layer i+1
    # edges[i][j] is list of out edges from node j in layer i
    # edges[i][j][k] is the k'th out edge of "node" j in "layer" i
    for i in range(len(acasxu_net.layerSizes) - 1):
        # add cell for each layer, includes all edges in that layer
        edges.append([])
        for j in range(len(acasxu_net.weights[i])):
            # add cell for each node, includes all edges in that node
            edges[i].append([])
            for k in range(len(acasxu_net.weights[i][j])):
                # acasxu.weights include input edges, so the edge is from the
                # k'th node in layer i to the j'th node in layer i+1
                src = "x_{}_{}".format(i, k)
                dest = "x_{}_{}".format(i + 1, j)
                weight = acasxu_net.weights[i][j][k]
                edge = Edge(src=src, dest=dest, weight=weight)
                edges[i][j].append(edge)
                # print(i,j,k,mn.weights[i][j][k])

    # validate sizes
    assert (len(acasxu_net.weights) == len(edges))
    for i in range(len(acasxu_net.layerSizes) - 1):
        assert (len(acasxu_net.weights[i]) == len(edges[i]))
        for j in range(len(acasxu_net.weights[i])):
            assert (len(acasxu_net.weights[i][j]) == len(edges[i][j]))
            for k in range(len(acasxu_net.weights[i][j])):
                if acasxu_net.weights[i][j][k] != edges[i][j][k].weight:
                    print("wrong edges: {},{},{}".format(i, j, k))
                    assert False

    nodes = []  # list of list of ARNode instances
    # nodes[i] is list of nodes in layer i
    # nodes[i][j] is ARNode instance of node j in layer i
    name2node_map = {}  # map name to arnode instance, for adding edges later
    for i, layer in enumerate(edges):
        nodes.append([])  # add layer
        for j, node in enumerate(layer):
            for k, edge in enumerate(node):
                src_name = edge.src
                if src_name not in name2node_map.keys():
                    src_arnode = ARNode(name=src_name,
                                        ar_type=None,
                                        in_edges=[],
                                        out_edges=[],
                                        activation_func=relu,
                                        bias=0.0,  # assigned later
                                        upper_bound=0.0,
                                        lower_bound=0.0
                                        )
                    nodes[i].append(src_arnode)
                    name2node_map[src_name] = src_arnode

    # add output layer nodes
    # equal to add the next line
    # layer = acasxu_net.weights[len(acasxu_net.layerSizes)-1]
    nodes.append([])
    for j, node in enumerate(layer):
        for k, edge in enumerate(node):
            dest_name = edge.dest
            if dest_name not in name2node_map.keys():
                dest_arnode = ARNode(name=dest_name,
                                     ar_type=None,
                                     in_edges=[],
                                     out_edges=[],
                                     activation_func=relu,
                                     bias=0.0,  # assigned later
                                     upper_bound=0.0,
                                     lower_bound=0.0

                                     )
                nodes[i + 1].append(dest_arnode)
                name2node_map[dest_name] = dest_arnode

    # after all nodes instances exist, add input and output edges
    for i, layer in enumerate(edges):  # layer is list of list of edges
        for j, node in enumerate(layer):  # node is list of edges
            for k, edge in enumerate(node):  # edge is Edge instance
                # print (i, j, k)
                src_node = name2node_map[edge.src]
                dest_node = name2node_map[edge.dest]
                src_node.out_edges.append(edge)
                dest_node.in_edges.append(edge)

    layers = []
    for i, layer in enumerate(nodes):
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

    net = Network(layers=layers, weights=acasxu_net.weights, biases=acasxu_net.biases, acasxu_net=acasxu_net)

    # for i,biases in enumerate(acasxu_net.biases):
    # layer = net.layers[i+1]
    # for j,node in enumerate(layer.nodes):
    #     node.bias = biases[j]
    return net


def network_from_onnx_file(nnet_filename: str) -> Network:
    """
    generate Net instance which is equivalent to given nnet formatted network
    @nnet_filename fullpath of nnet file, see maraboupy.MarabouNetworkNNet
    under the root dir of git marabou project
    :return: Network object
    """
    # read nnetfile into Marabou Network
    acasxu_net = MarabouNetworkONNX.MarabouNetworkONNX(filename=nnet_filename)

    # list of layers, layer include list of nodes, node include list of Edge-s
    # notice the acasxu.weights include weights of input edges
    edges = []  # list of list of list of Edge instances
    # edges[i] is list of list of edges between layer i to layer i+1
    # edges[i][j] is list of out edges from node j in layer i
    # edges[i][j][k] is the k'th out edge of "node" j in "layer" i
    for i in range(len(acasxu_net.layerSizes) - 1):
        # add cell for each layer, includes all edges in that layer
        edges.append([])
        for j in range(len(acasxu_net.weights[i])):
            # add cell for each node, includes all edges in that node
            edges[i].append([])
            for k in range(len(acasxu_net.weights[i][j])):
                # acasxu.weights include input edges, so the edge is from the
                # k'th node in layer i to the j'th node in layer i+1
                src = "x_{}_{}".format(i, k)
                dest = "x_{}_{}".format(i + 1, j)
                weight = acasxu_net.weights[i][j][k]
                edge = Edge(src=src, dest=dest, weight=weight)
                edges[i][j].append(edge)
                # print(i,j,k,mn.weights[i][j][k])

    # validate sizes
    assert (len(acasxu_net.weights) == len(edges))
    for i in range(len(acasxu_net.layerSizes) - 1):
        assert (len(acasxu_net.weights[i]) == len(edges[i]))
        for j in range(len(acasxu_net.weights[i])):
            assert (len(acasxu_net.weights[i][j]) == len(edges[i][j]))
            for k in range(len(acasxu_net.weights[i][j])):
                if acasxu_net.weights[i][j][k] != edges[i][j][k].weight:
                    print("wrong edges: {},{},{}".format(i, j, k))
                    assert False

    nodes = []  # list of list of ARNode instances
    # nodes[i] is list of nodes in layer i
    # nodes[i][j] is ARNode instance of node j in layer i
    name2node_map = {}  # map name to arnode instance, for adding edges later
    for i, layer in enumerate(edges):
        nodes.append([])  # add layer
        for j, node in enumerate(layer):
            for k, edge in enumerate(node):
                src_name = edge.src
                if src_name not in name2node_map.keys():
                    src_arnode = ARNode(name=src_name,
                                        ar_type=None,
                                        in_edges=[],
                                        out_edges=[],
                                        activation_func=relu,
                                        bias=0.0,  # assigned later
                                        upper_bound=0.0,
                                        lower_bound=0.0
                                        )
                    nodes[i].append(src_arnode)
                    name2node_map[src_name] = src_arnode

    # add output layer nodes
    # equal to add the next line
    # layer = acasxu_net.weights[len(acasxu_net.layerSizes)-1]
    nodes.append([])
    for j, node in enumerate(layer):
        for k, edge in enumerate(node):
            dest_name = edge.dest
            if dest_name not in name2node_map.keys():
                dest_arnode = ARNode(name=dest_name,
                                     ar_type=None,
                                     in_edges=[],
                                     out_edges=[],
                                     activation_func=relu,
                                     bias=0.0,  # assigned later
                                     upper_bound=0.0,
                                     lower_bound=0.0

                                     )
                nodes[i + 1].append(dest_arnode)
                name2node_map[dest_name] = dest_arnode

    # after all nodes instances exist, add input and output edges
    for i, layer in enumerate(edges):  # layer is list of list of edges
        for j, node in enumerate(layer):  # node is list of edges
            for k, edge in enumerate(node):  # edge is Edge instance
                # print (i,j,k)
                src_node = name2node_map[edge.src]
                dest_node = name2node_map[edge.dest]
                src_node.out_edges.append(edge)
                dest_node.in_edges.append(edge)

    layers = []
    for i, layer in enumerate(nodes):
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

    net = Network(layers=layers, weights=acasxu_net.weights, biases=acasxu_net.biases, acasxu_net=acasxu_net)

    # for i,biases in enumerate(acasxu_net.biases):
    # layer = net.layers[i+1]
    # for j,node in enumerate(layer.nodes):
    #     node.bias = biases[j]
    return net


def get_all_acas_nets(indices=None):
    """
    :param indices: list of indices of nnet files, if None return all
    :return: list of Net objects of acas networks in the relevant indices
    """
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    l = []
    for i, filename in enumerate(os.listdir(nnet_dir)):
        if i not in indices:
            continue
        nnet_filename = os.path.join(nnet_dir, filename)
        l.append(network_from_nnet_file(nnet_filename))
    return l
