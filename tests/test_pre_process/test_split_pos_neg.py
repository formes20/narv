#/usr/bin/python3


import os

from core.configuration.consts import (
    PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES, VERBOSE
)
from core.pre_process.pre_process import preprocess, preprocess_split_pos_neg
from core.nnet.read_nnet import network_from_nnet_file
from core.data_structures.Network import Network
from core.data_structures.Layer import Layer
from core.data_structures.ARNode import ARNode
from core.data_structures.Edge import Edge
from core.visualization.visualize_network import visualize_network

# check preprocess: pos-neg split
# test_split_pos_neg_1()
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
    net = Network(layers=[input_layer, h1_layer, h2_layer, output_layer])
    preprocess_split_pos_neg(network=net)
    if VERBOSE:
        print(net)
        visualize_network(network=net, title="test_split_pos_neg_1")


#test_split_pos_neg_2()
def test_split_pos_neg_2():
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    filename = "ACASXU_run2a_1_1_batch_2000.nnet"
    nnet_filename = os.path.join(nnet_dir, filename)
    net = network_from_nnet_file(nnet_filename)
    preprocess_split_pos_neg(network=net)
    if VERBOSE:
        print(net)
        # heavy viaualization, therefore it is omitted
        # visualize_network(network=net, title="test_split_pos_neg_1")
