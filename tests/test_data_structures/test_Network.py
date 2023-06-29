#/usr/bin/python3

import os
import sys
import copy
import random
sys.path.append("/home/artifact/narv")
from import_marabou import dynamically_import_marabou
dynamically_import_marabou()
from maraboupy import MarabouCore

# import pytest

from core.configuration.consts import (
    PATH_TO_MARABOU_ACAS_EXAMPLES, PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES,
    EPSILON, VERBOSE
)
from core.utils.verification_properties_utils import get_test_property_acas, get_test_property_tiny, \
    get_test_property_input_2_output_1, get_test_property_input_3_output_1
from core.pre_process.pre_process import preprocess, preprocess_split_pos_neg
from core.abstraction.naive import abstract_network
from core.abstraction.alg2 import heuristic_abstract
from core.nnet.read_nnet import network_from_nnet_file, get_all_acas_nets
from core.utils.alg2_utils import is_evaluation_result_equal
from core.refinement.refine import refine
from core.data_structures.Network import Network
from core.data_structures.Layer import Layer
from core.data_structures.ARNode import ARNode
from core.data_structures.Edge import Edge
from core.utils.debug_utils import debug_print, embed_ipython
from core.visualization.visualize_network import visualize_network
from core.utils.marabou_query_utils import get_query

# -------------------------------------------------------------------------
# helpers
def create_2_2_1_rand_weights_net():
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
    net = Network(layers=[input_layer, h1_layer, output_layer])
    return net


def create_2_2_2_1_rand_weights_net():
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
    net = Network(layers=[input_layer, h1_layer, h2_layer, output_layer])
    return net


def create_3_3_3_1_rand_weights_net():
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
    net = Network(layers=[input_layer, h1_layer, h2_layer, output_layer])
    return net


# net from acasxu nnet file
def example_4():
    """
    generate Net from acasxu network that is represented by nnet format file
    """
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    # filename = "ACASXU_TF12_run3_DS_Minus15_Online_tau0_pra1_200Epochs.nnet"
    filename = "ACASXU_run2a_1_1_batch_2000.nnet"
    nnet_filename = os.path.join(nnet_dir, filename)
    net = network_from_nnet_file(nnet_filename)
    return net


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
    net = Network(layers=[input_layer, h1_layer, h2_layer, output_layer])
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
    net = Network(layers=[input_layer, h0_layer, h1_layer, h2_layer, output_layer])
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
    net = Network(layers=[input_layer, h0_layer, h1_layer, h2_layer, h3_layer, output_layer])
    return net


# example_2()
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
    net = Network(layers=[input_layer, h1_layer, h2_layer, output_layer])
    #print(net)

    print("-"*80)
    print("do test_abstraction")
    print("-"*80)

    orig_net = copy.deepcopy(net)
    abstracted_net = abstract_network(network=net)
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

# marabou query
# example_3()
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
    net = Network(layers=[input_layer, h1_layer, h2_layer, output_layer])
    #net = example_4()
    net = abstract_network(network=net)
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

    vars1, stats1, result = get_query(network=net, test_property=test_property)
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

# test_abstraction and test_refinement using net from acasxu nnet file
# example_5()
def example_5():
    net = example_4()
    print(net)
    #net.visualize(title="acasxu_net")
    abstracted_net = abstract_network(network=net)
    visualize_network(network=abstracted_net, title="abstracted_net")
    refine_1_net = refine(network=abstracted_net, sequence_length=3)
    refine_1_net.visualize("refine 1")


# -------------------------------------------------------------------------
# __eq__
#test_net_eq_sanity_check()
# def test_net_eq_sanity_check():
#     """checks that == operator of Net works"""
#     l1 = get_all_acas_nets()
#     l2 = [copy.deepcopy(net) for net in l1]
#     assert l1 == l2
#
#     # change in edge src
#     orig_src = l2[10].layers[3].nodes[5].in_edges[1].src
#     l2[10].layers[3].nodes[5].in_edges[1].src = "yosi"
#     assert l1 != l2
#     l2[10].layers[3].nodes[5].in_edges[1].src = orig_src
#
#     # change in edge dest
#     orig_dest = l2[10].layers[3].nodes[5].in_edges[1].dest
#     l2[10].layers[3].nodes[5].in_edges[1].dest = "yosi"
#     assert l1 != l2
#     l2[10].layers[3].nodes[5].in_edges[1].dest = orig_dest
#
#     # change in edge weight
#     orig_weight = l2[10].layers[3].nodes[5].in_edges[1].weight
#     l2[10].layers[3].nodes[5].in_edges[1].weight = random.random()
#     assert l1 != l2
#     l2[10].layers[3].nodes[5].in_edges[1].weight = orig_weight
#
#     # change out edge src
#     orig_src = l2[10].layers[3].nodes[5].out_edges[1].src
#     l2[10].layers[3].nodes[5].out_edges[1].src = "yosi"
#     assert l1 != l2
#     l2[10].layers[3].nodes[5].out_edges[1].src = orig_src
#
#     # change out edge dest
#     orig_dest = l2[10].layers[3].nodes[5].out_edges[1].dest
#     l2[10].layers[3].nodes[5].out_edges[1].dest = "yosi"
#     assert l1 != l2
#     l2[10].layers[3].nodes[5].out_edges[1].dest = orig_dest
#
#     # change out edge weight
#     orig_dest = l2[10].layers[3].nodes[5].out_edges[1].weight
#     l2[10].layers[3].nodes[5].out_edges[1].weight = random.random()
#     assert l1 != l2
#     l2[10].layers[3].nodes[5].out_edges[1].weight = orig_weight
#
#     # change node name
#     orig_name = l2[10].layers[3].nodes[5].name
#     l2[10].layers[3].nodes[5].name = "yosi"
#     assert l1 != l2
#     l2[10].layers[3].nodes[5].name = orig_name
#
#     # change node ar_type
#     orig_ar_type = l2[10].layers[3].nodes[5].ar_type
#     l2[10].layers[3].nodes[5].ar_type = "yosi"
#     assert l1 != l2
#     l2[10].layers[3].nodes[5].ar_type = orig_ar_type
#
#     # change node activation function
#     orig_activation_func = l2[10].layers[3].nodes[5].activation_func
#     l2[10].layers[3].nodes[5].activation_func = lambda x:-x
#     assert l1 != l2
#     l2[10].layers[3].nodes[5].activation_func = orig_activation_func
#
#     # change node bias
#     orig_bias = l2[10].layers[3].nodes[5].bias
#     l2[10].layers[3].nodes[5].bias = random.random()
#     assert l1 != l2
#     l2[10].layers[3].nodes[5].bias = orig_bias
#
#     # change node in_edges - remove one edge
#     orig_in_edges = l2[10].layers[3].nodes[5].in_edges
#     del l2[10].layers[3].nodes[5].in_edges[0]
#     assert l1 != l2
#     l2[10].layers[3].nodes[5].in_edges = orig_in_edges
#
#     # change node out_edges - remove one edge
#     orig_out_edges = l2[10].layers[3].nodes[5].out_edges
#     del l2[10].layers[3].nodes[5].out_edges[0]
#     assert l1 != l2
#     l2[10].layers[3].nodes[5].out_edges = orig_out_edges


# -------------------------------------------------------------------------
# test_abstraction and test_refinement
# test_abstract_and_refine_result_with_orig_1()
# def test_abstract_and_refine_result_with_orig_1():
#     net = example_01()
#     preprocess(network=net)
#     orig_net = copy.deepcopy(net)
#     debug_print("original net")
#     print(orig_net)
#     abstract_network(network=net, do_preprocess=False)
#     debug_print("abstract net")
#     print(net)
#     refine(network=net,
#            sequence_length=orig_net.get_general_net_data()['num_nodes'])
#     debug_print("refined net")
#     print(net)
#     # import IPython
#     # IPython.embed()
#     assert net == orig_net
#
#
# # test_abstract_and_refine_result_with_orig_2()
# def test_abstract_and_refine_result_with_orig_2():
#     """test on 4 layers"""
#     net = example_02()
#     preprocess(network=net)
#     orig_net = copy.deepcopy(net)
#     debug_print("original net")
#     print(orig_net)
#     abstract_network(network=net, do_preprocess=False)
#     if VERBOSE:
#         visualize_network(network=net, title="abstract net")
#     debug_print("abstract net")
#     print(net)
#     refine(network=net,
#            sequence_length=orig_net.get_general_net_data()['num_nodes'])
#     debug_print("refined net")
#     print(net)
#     assert net == orig_net
#
#
# # test_abstract_and_refine_result_with_orig_3()
# def test_abstract_and_refine_result_with_orig_3():
#     """test on 5 layers"""
#     net = example_03()
#     preprocess(network=net)
#     orig_net = copy.deepcopy(net)
#     debug_print("original net")
#     print(orig_net)
#     abstract_network(network=net, do_preprocess=False)
#     debug_print("abstract net")
#     print(net)
#     refine(network=net,
#            sequence_length=orig_net.get_general_net_data()['num_nodes'])
#     debug_print("refined net")
#     print(net)
#     assert net == orig_net

# test_abstract_and_refine_result_with_orig_acas()
# def test_abstract_and_refine_result_with_orig_acas():
#     net = example_4()
#     preprocess(network=net)
#     orig_net = copy.deepcopy(net)
#     abstract_network(network=net, do_preprocess=False)
#     #net.minimal_abstract(test_property=get_test_property_acas(), do_preprocess=False)
#     refine(network=net,
#            sequence_length=orig_net.get_general_net_data()['num_nodes'])
#     assert net == orig_net


# test_abstract_and_refine_result_with_orig_all_acas()
# LONG RUN - didn't verified yet
# def test_abstract_and_refine_result_with_orig_all_acas():
#     nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
#     for i,filename in enumerate(os.listdir(nnet_dir)):
#         print("check {} - filename = {}".format(i, filename ))
#         nnet_filename = os.path.join(nnet_dir, filename)
#         net = network_from_nnet_file(nnet_filename)
#         preprocess(network=net)
#         orig_net = copy.deepcopy(net)
#         abstract_network(network=net, do_preprocess=False)
#         refine(network=net,
#                sequence_length=orig_net.get_general_net_data()['num_nodes'])
#         assert net == orig_net




# -------------------------------------------------------------------------
# tests evaluate / speedy_evaluate
# test_evaluate_1()
# def test_evaluate_1():
#     # edges
#     e_00_10 = Edge(src="x00", dest="x10", weight=1)
#     e_00_11 = Edge(src="x00", dest="x11", weight=2)
#     e_01_10 = Edge(src="x01", dest="x10", weight=3)
#     e_01_11 = Edge(src="x01", dest="x11", weight=4)
#     e_10_20 = Edge(src="x10", dest="x20", weight=-1)
#     e_10_21 = Edge(src="x10", dest="x21", weight=2)
#     e_11_20 = Edge(src="x11", dest="x20", weight=2)
#     e_11_21 = Edge(src="x11", dest="x21", weight=-3)
#     e_20_y = Edge(src="x20", dest="y", weight=-4)
#     e_21_y = Edge(src="x21", dest="y", weight=6)
#
#     # nodes
#     x_0_0 = ARNode(name="x00", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11])
#     x_0_1 = ARNode(name="x01", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11])
#     x_1_0 = ARNode(name="x10", ar_type=None, in_edges=[e_00_10, e_01_10], out_edges=[e_10_20, e_10_21])
#     x_1_1 = ARNode(name="x11", ar_type=None, in_edges=[e_00_11, e_01_11], out_edges=[e_11_20, e_11_21])
#     x_2_0 = ARNode(name="x20", ar_type=None, in_edges=[e_10_20, e_11_20], out_edges=[e_20_y])
#     x_2_1 = ARNode(name="x21", ar_type=None, in_edges=[e_10_21, e_11_21], out_edges=[e_21_y])
#     y = ARNode(name="y", ar_type=None, in_edges=[e_20_y, e_21_y], out_edges=[])
#
#     # layers
#     input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1])
#     h1_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1])
#     h2_layer = Layer(type_name="hidden", nodes=[x_2_0, x_2_1])
#     output_layer = Layer(type_name="output", nodes=[y])
#
#     # net
#     net = Network(layers=[input_layer, h1_layer, h2_layer, output_layer])
#     # net.visualize()
#     eval_output = net.evaluate(input_values={0:2, 1:1})
#     speedy_eval_output = net.speedy_evaluate(input_values={0:2, 1:1})
#     assert is_evaluation_result_equal(eval_output.items(), speedy_eval_output)
#     assert eval_output['y'] == -44


# test_evaluate_2()
# def test_evaluate_2():
#     e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=10)
#     e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=5)
#     e_00_12 = Edge(src="x_0_0", dest="x_1_2", weight=1)
#     e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=2)
#     e_01_11 = Edge(src="x_0_1", dest="x_1_1", weight=4)
#     e_01_12 = Edge(src="x_0_1", dest="x_1_2", weight=7)
#     e_02_10 = Edge(src="x_0_2", dest="x_1_0", weight=-3)
#     e_02_11 = Edge(src="x_0_2", dest="x_1_1", weight=-8)
#     e_02_12 = Edge(src="x_0_2", dest="x_1_2", weight=-7)
#     e_10_20 = Edge(src="x_1_0", dest="x_2_0", weight=5)
#     e_10_21 = Edge(src="x_1_0", dest="x_2_1", weight=4)
#     e_10_22 = Edge(src="x_1_0", dest="x_2_2", weight=9)
#     e_11_20 = Edge(src="x_1_1", dest="x_2_0", weight=-7)
#     e_11_21 = Edge(src="x_1_1", dest="x_2_1", weight=-1)
#     e_11_22 = Edge(src="x_1_1", dest="x_2_2", weight=-2)
#     e_12_20 = Edge(src="x_1_2", dest="x_2_0", weight=-5)
#     e_12_21 = Edge(src="x_1_2", dest="x_2_1", weight=-3)
#     e_12_22 = Edge(src="x_1_2", dest="x_2_2", weight=-6)
#     e_20_y = Edge(src="x_2_0", dest="y", weight=4)
#     e_21_y = Edge(src="x_2_1", dest="y", weight=3)
#     e_22_y = Edge(src="x_2_2", dest="y", weight=-6)
#
#     # nodes
#     x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11, e_00_12])
#     x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11, e_01_12])
#     x_0_2 = ARNode(name="x_0_2", ar_type=None, in_edges=[], out_edges=[e_02_10, e_02_11, e_02_12])
#     x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10, e_02_10], out_edges=[e_10_20, e_10_21, e_10_22])
#     x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11, e_02_11], out_edges=[e_11_20, e_11_21, e_11_22])
#     x_1_2 = ARNode(name="x_1_2", ar_type=None, in_edges=[e_00_12, e_01_12, e_02_12], out_edges=[e_12_20, e_12_21, e_12_22])
#     x_2_0 = ARNode(name="x_2_0", ar_type=None, in_edges=[e_10_20, e_11_20, e_12_20], out_edges=[e_20_y])
#     x_2_1 = ARNode(name="x_2_1", ar_type=None, in_edges=[e_10_21, e_11_21, e_12_21], out_edges=[e_21_y])
#     x_2_2 = ARNode(name="x_2_2", ar_type=None, in_edges=[e_10_22, e_11_22, e_12_22], out_edges=[e_22_y])
#     y = ARNode(name="y", ar_type=None, in_edges=[e_20_y, e_21_y, e_22_y], out_edges=[])
#
#     # layers
#     input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1, x_0_2])
#     h1_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1, x_1_2])
#     h2_layer = Layer(type_name="hidden", nodes=[x_2_0, x_2_1, x_2_2])
#     output_layer = Layer(type_name="output", nodes=[y])
#
#     # net
#     net = Network(layers=[input_layer, h1_layer, h2_layer, output_layer])
#     eval_output = net.evaluate(input_values={0:2, 1:1, 2:-1})
#     speedy_eval_output = net.speedy_evaluate(input_values={0:2, 1:1, 2:-1})
#     assert is_evaluation_result_equal(eval_output.items(), speedy_eval_output)
#     assert eval_output['y'] == -420

# test_evaluate_3()
# def test_evaluate_3():
#     nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
#     filename = "ACASXU_run2a_1_1_batch_2000.nnet"
#     nnet_filename = os.path.join(nnet_dir, filename)
#     net = network_from_nnet_file(nnet_filename)
#     res = net.evaluate({0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
#     speedy_res = net.speedy_evaluate({0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
#     assert is_evaluation_result_equal(res.items(), speedy_res)
#     #for i in range(len(net.layers[-1].nodes)):
#     #    print(i)
#     #    assert res["x_7_{}".format(i)] == 0.0
#     eval_output = net.evaluate(input_values={0: 0.25, 1: 0.3, 2: -0.2, 3: -0.1, 4: 0.15})
#     speedy_eval_output = net.speedy_evaluate(input_values={0: 0.25, 1: 0.3, 2: -0.2, 3: -0.1, 4: 0.15})
#     assert is_evaluation_result_equal(eval_output.items(), speedy_eval_output)
#     # print(eval_output)
#     # print(speedy_eval_output)

# the next test checks that our eval func and marabou eval func outputs hte same values.
# the test is wrt some net and random input
# the test fails because the output that marabou returns is [0, ..., 0] because python-cpp api problem
# you can see the print-outs that approve the equality of the outputs
# test_evaluate_marabou_equality()
# def test_evaluate_marabou_equality():
#     """
#     validates equality between the outputs of marabou/Net evaluations for random inputs.
#     """
#     nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
#     filename = "ACASXU_run2a_1_1_batch_2000.nnet"
#     nnet_filename = os.path.join(nnet_dir, filename)
#     # prepare 2 nets
#     ar_net = network_from_nnet_file(nnet_filename)
#     marabou_net = MarabouCore.load_network(nnet_filename)
#
#     # prepare 2 input (and output) parameters
#     input_vals = [random.random() for i in range(MarabouCore.num_inputs(marabou_net))]
#     ar_input = {i:val for i,val in enumerate(input_vals)}
#     output = [0.0 for i in range(MarabouCore.num_outputs(marabou_net))]
#
#     # run 2 evaluations and validate equality
#     ar_res = ar_net.evaluate(ar_input)
#     speedy_ar_res = ar_net.speedy_evaluate(ar_input)
#     assert is_evaluation_result_equal(ar_res.items(), speedy_ar_res)
#     MarabouCore.evaluate_network(marabou_net, input_vals, output, False, False)
#     print(f"ar_res={ar_res}")
#     print(f"speedy_ar_res ={speedy_ar_res }")
#     print(f"output={output}")
#     print(len(ar_res))
#     print(len(output))
#     assert len(ar_res) == len(output)
#     assert ar_res.values() == output
#     MarabouCore.destroy_network(marabou_net)



# -------------------------------------------------------------------------
# test that abstract output >= test_refinement output >= original output

# const net 2_2_2_1, N=100 random inputs
# test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_1()
# def test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_1():
#     e_11_21 = Edge(src="x11", dest="x21", weight=-1)
#     e_11_22 = Edge(src="x11", dest="x22", weight=2)
#     e_12_21 = Edge(src="x12", dest="x21", weight=-3)
#     e_12_22 = Edge(src="x12", dest="x22", weight=4)
#     e_21_31 = Edge(src="x21", dest="x31", weight=-1)
#     e_21_32 = Edge(src="x21", dest="x32", weight=-2)
#     e_22_31 = Edge(src="x22", dest="x31", weight=2)
#     e_22_32 = Edge(src="x22", dest="x32", weight=-3)
#     e_31_y = Edge(src="x31", dest="y", weight=-4)
#     e_32_y = Edge(src="x32", dest="y", weight=6)
#
#     # nodes
#     x11 = ARNode(name="x11", ar_type=None, in_edges=[],
#                  out_edges=[e_11_21, e_11_22])
#     x12 = ARNode(name="x12", ar_type=None, in_edges=[],
#                  out_edges=[e_12_21, e_12_22])
#     x21 = ARNode(name="x21", ar_type=None,
#                  in_edges=[e_11_21, e_12_21],
#                  out_edges=[e_21_31, e_21_32])
#     x22 = ARNode(name="x22", ar_type=None,
#                  in_edges=[e_11_22, e_12_22],
#                  out_edges=[e_22_31, e_22_32])
#     x31 = ARNode(name="x31", ar_type=None,
#                  in_edges=[e_21_31, e_22_31],
#                  out_edges=[e_31_y])
#     x32 = ARNode(name="x32", ar_type=None,
#                  in_edges=[e_21_32, e_22_32],
#                  out_edges=[e_32_y])
#     y = ARNode(name="y", ar_type=None,
#                in_edges=[e_31_y, e_32_y],
#                out_edges=[])
#
#     # layers
#     input_layer = Layer(type_name="input", nodes=[x11,x12])
#     h1_layer = Layer(type_name="hidden", nodes=[x21,x22])
#     h2_layer = Layer(type_name="hidden", nodes=[x31,x32])
#     output_layer = Layer(type_name="output", nodes=[y])
#
#     # net
#     net = Network(layers=[input_layer, h1_layer, h2_layer, output_layer])
#
#     N = 100
#     orig_net = copy.deepcopy(net)
#     abstract_net = abstract_network(copy.deepcopy(net))
#     refined_net = refine(network=net)
#     for j in range(N):
#         input_values = {i:(0.5-random.random()) * 2 for i in range(len(net.layers[0].nodes))}
#         orig_net_output = sorted(orig_net.evaluate(input_values).items(), key=lambda x: x[0])
#         speedy_orig_net_output = orig_net.speedy_evaluate(input_values)
#         assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
#         abstract_net_output = sorted(abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
#         speedy_abstract_net_output = abstract_net.speedy_evaluate(input_values)
#         assert is_evaluation_result_equal(abstract_net_output, speedy_abstract_net_output)
#         refined_net_output = sorted(refined_net.evaluate(input_values).items(), key=lambda x: x[0])
#         speedy_refined_net_output = refined_net.speedy_evaluate(input_values)
#         assert is_evaluation_result_equal(refined_net_output, speedy_refined_net_output)
#         assert(len(orig_net_output) == len(abstract_net_output))
#         assert(len(refined_net_output) == len(abstract_net_output))
#         for k in range(len(orig_net_output)):
#             assert orig_net_output[k][1] <= refined_net_output[k][1] <= abstract_net_output[k][1]
#
#
# # const net 3_3_3_1, N=1000 random inputs
# # test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_2()
# def test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_2():
#     e_00_10 = Edge(src="x_0_0", dest="x_1_0", weight=10)
#     e_00_11 = Edge(src="x_0_0", dest="x_1_1", weight=5)
#     e_00_12 = Edge(src="x_0_0", dest="x_1_2", weight=1)
#     e_01_10 = Edge(src="x_0_1", dest="x_1_0", weight=2)
#     e_01_11 = Edge(src="x_0_1", dest="x_1_1", weight=4)
#     e_01_12 = Edge(src="x_0_1", dest="x_1_2", weight=7)
#     e_02_10 = Edge(src="x_0_2", dest="x_1_0", weight=-3)
#     e_02_11 = Edge(src="x_0_2", dest="x_1_1", weight=-8)
#     e_02_12 = Edge(src="x_0_2", dest="x_1_2", weight=-7)
#     e_10_20 = Edge(src="x_1_0", dest="x_2_0", weight=5)
#     e_10_21 = Edge(src="x_1_0", dest="x_2_1", weight=4)
#     e_10_22 = Edge(src="x_1_0", dest="x_2_2", weight=9)
#     e_11_20 = Edge(src="x_1_1", dest="x_2_0", weight=-7)
#     e_11_21 = Edge(src="x_1_1", dest="x_2_1", weight=-1)
#     e_11_22 = Edge(src="x_1_1", dest="x_2_2", weight=-2)
#     e_12_20 = Edge(src="x_1_2", dest="x_2_0", weight=-5)
#     e_12_21 = Edge(src="x_1_2", dest="x_2_1", weight=-3)
#     e_12_22 = Edge(src="x_1_2", dest="x_2_2", weight=-6)
#     e_20_y = Edge(src="x_2_0", dest="y", weight=4)
#     e_21_y = Edge(src="x_2_1", dest="y", weight=3)
#     e_22_y = Edge(src="x_2_2", dest="y", weight=-6)
#
#     # nodes
#     x_0_0 = ARNode(name="x_0_0", ar_type=None, in_edges=[], out_edges=[e_00_10, e_00_11, e_00_12])
#     x_0_1 = ARNode(name="x_0_1", ar_type=None, in_edges=[], out_edges=[e_01_10, e_01_11, e_01_12])
#     x_0_2 = ARNode(name="x_0_2", ar_type=None, in_edges=[], out_edges=[e_02_10, e_02_11, e_02_12])
#     x_1_0 = ARNode(name="x_1_0", ar_type=None, in_edges=[e_00_10, e_01_10, e_02_10], out_edges=[e_10_20, e_10_21, e_10_22])
#     x_1_1 = ARNode(name="x_1_1", ar_type=None, in_edges=[e_00_11, e_01_11, e_02_11], out_edges=[e_11_20, e_11_21, e_11_22])
#     x_1_2 = ARNode(name="x_1_2", ar_type=None, in_edges=[e_00_12, e_01_12, e_02_12], out_edges=[e_12_20, e_12_21, e_12_22])
#     x_2_0 = ARNode(name="x_2_0", ar_type=None, in_edges=[e_10_20, e_11_20, e_12_20], out_edges=[e_20_y])
#     x_2_1 = ARNode(name="x_2_1", ar_type=None, in_edges=[e_10_21, e_11_21, e_12_21], out_edges=[e_21_y])
#     x_2_2 = ARNode(name="x_2_2", ar_type=None, in_edges=[e_10_22, e_11_22, e_12_22], out_edges=[e_22_y])
#     y = ARNode(name="y", ar_type=None, in_edges=[e_20_y, e_21_y, e_22_y], out_edges=[])
#
#     # layers
#     input_layer = Layer(type_name="input", nodes=[x_0_0, x_0_1, x_0_2])
#     h1_layer = Layer(type_name="hidden", nodes=[x_1_0, x_1_1, x_1_2])
#     h2_layer = Layer(type_name="hidden", nodes=[x_2_0, x_2_1, x_2_2])
#     output_layer = Layer(type_name="output", nodes=[y])
#
#     # net
#     net = Network(layers=[input_layer, h1_layer, h2_layer, output_layer])
#     if VERBOSE:
#         visualize_network(network=net, title="orig")
#     N = 1000
#     orig_net = copy.deepcopy(net)
#     abstract_net = copy.deepcopy(abstract_network(network=net))
#     print("before refine")
#     refined_net = refine(network=net)
#     print("after refine")
#     for j in range(N):
#         input_values = {i:(0.5-random.random()) * 2 for i in range(len(net.layers[0].nodes))}
#         orig_net_output = sorted(orig_net.evaluate(input_values).items(), key=lambda x: x[0])
#         speedy_orig_net_output = orig_net.speedy_evaluate(input_values)
#         assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
#         abstract_net_output = sorted(abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
#         speedy_abstract_net_output = abstract_net.speedy_evaluate(input_values)
#         assert is_evaluation_result_equal(abstract_net_output, speedy_abstract_net_output)
#         refined_net_output = sorted(refined_net.evaluate(input_values).items(), key=lambda x: x[0])
#         speedy_refined_net_output = refined_net.speedy_evaluate(input_values)
#         assert is_evaluation_result_equal(refined_net_output, speedy_refined_net_output)
#         assert(len(orig_net_output) == len(abstract_net_output))
#         assert(len(refined_net_output) == len(abstract_net_output))
#         for k in range(len(orig_net_output)):
#             if (orig_net_output[k][1] > refined_net_output[k][1]
#                 and abs(orig_net_output[k][1] - refined_net_output[k][1]) > EPSILON)\
#                     or (refined_net_output[k][1] > abstract_net_output[k][1]
#                         and abs(refined_net_output[k][1] - abstract_net_output[k][1]) > EPSILON):
#                 print ("error: orig is bigger than test_refinement's output or test_refinement is bigger than abstract output")
#                 print("input_values: {}".format(input_values.items()))
#                 print("orig_net_output[k][1] = {}".format(orig_net_output[k][1]))
#                 print("refined_net_output[k][1] = {}".format(refined_net_output[k][1]))
#                 print("abstract_net_output[k][1] ({}) = {}".format(abstract_net_output[k][1]))
#                 assert False
#
# # # acas-xu networks, N=100
# # test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_3()
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
        net = network_from_nnet_file(nnet_filename)
        orig_net = copy.deepcopy(net)
        abstract_net = abstract_network(copy.deepcopy(net))
        refined_net = refine(network=net, sequence_length=10)
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
#
#
# # # random weighted 100 nets, all are 2_2_1, N=100 random inputs
# # test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_4()
# def test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_4():
#     create_network_func = create_2_2_1_rand_weights_net
#     check_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs(create_network_func=create_network_func)
#
#
# # # random weighted 100 nets, all are 2_2_2_1, N=100 random inputs
# # test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_5()
# def test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_5():
#     create_network_func = create_2_2_2_1_rand_weights_net
#     check_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs(create_network_func=create_network_func)
#
# # # random weigted 100 nets, all are 3_3_3_1, N=100 random inputs
# # test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_6()
# def test_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs_6():
#     create_network_func = create_3_3_3_1_rand_weights_net
#     check_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs(create_network_func=create_network_func)


# def check_original_outputs_smaller_than_refinement_outputs_smaller_than_abstract_outputs(create_network_func):
#     """
#     for 1-50:
#         create network with random weights,
#         abstract network,
#         for 1-100:
#             generate random input
#             check that abstract output > test_refinement output > original output
#     @create_network_func method that generate net with specific layers sizes
#     """
#     N = 50
#     # layers_sizes = {0:2, 1:2, 2:1}
#     for i in range(100):
#         # net = creat_random_network(layers_sizes)
#         net = create_network_func()
#         orig_net = copy.deepcopy(net)
#         abstract_net = abstract_network(copy.deepcopy(net))
#         refined_net = refine(network=copy.deepcopy(abstract_net))
#         for j in range(N):
#             input_values = {i: (0.5-random.random()) * 2 for i in range(len(net.layers[0].nodes))}
#             orig_net_output = sorted(orig_net.evaluate(input_values).items(), key=lambda x: x[0])
#             speedy_orig_net_output = orig_net.speedy_evaluate(input_values)
#             assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
#             abstract_net_output = sorted(abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
#             speedy_abstract_net_output = abstract_net.speedy_evaluate(input_values)
#             assert is_evaluation_result_equal(abstract_net_output, speedy_abstract_net_output)
#             refined_net_output = sorted(refined_net.evaluate(input_values).items(), key=lambda x: x[0])
#             speedy_refined_net_output = refined_net.speedy_evaluate(input_values)
#             assert is_evaluation_result_equal(refined_net_output, speedy_refined_net_output)
#             assert(len(orig_net_output) == len(abstract_net_output))
#             assert(len(refined_net_output) == len(abstract_net_output))
#             for k in range(len(orig_net_output)):
#                 if (orig_net_output[k][1] - refined_net_output[k][1]) > EPSILON or \
#                         (refined_net_output[k][1] - abstract_net_output[k][1]) > EPSILON:
#                     import IPython
#                     IPython.embed()
#                     print(i,j)
#                     assert False

# -------------------------------------------------------------------------
# test that abstract output >= heuristic output >= original output
# def check_heuristic_abstract_smaller_than_complete_abstract_outputs(create_network_func, test_property):
#     """
#     for 1-50:
#         create network with random weights,
#         generate complete and heristic abstract networks,
#         for 1-100:
#             generate random input
#             check that heuristically abstracted net's output < complete abstracted net's output
#     @create_network_func method that generate net with specific layers sizes
#     """
#     N = 50
#     # layers_sizes = {0:2, 1:2, 2:1}
#     for i in range(100):
#         # net = create_random_network(layers_sizes)
#         net = create_network_func()
#         orig_net = copy.deepcopy(net)
#         complete_abstract_net = abstract_network(network=copy.deepcopy(orig_net))
#         # print(i)
#         heuristic_abstract_net = heuristic_abstract(network=copy.deepcopy(orig_net), test_property=test_property)
#         for j in range(N):
#             input_values = {i: (0.5-random.random()) * 2 for i in range(len(net.layers[0].nodes))}
#             orig_net_output = sorted(orig_net.evaluate(input_values).items(), key=lambda x: x[0])
#             speedy_orig_net_output = orig_net.speedy_evaluate(input_values)
#             assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
#             complete_abstract_net_output = sorted(complete_abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
#             speedy_complete_abstract_net_output = complete_abstract_net.speedy_evaluate(input_values)
#             assert is_evaluation_result_equal(complete_abstract_net_output, speedy_complete_abstract_net_output)
#             heuristic_net_output = sorted(heuristic_abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
#             speedy_heuristic_net_output = heuristic_abstract_net.speedy_evaluate(input_values)
#             assert is_evaluation_result_equal(heuristic_net_output, speedy_heuristic_net_output)
#             assert(len(orig_net_output) == len(complete_abstract_net_output))
#             assert(len(heuristic_net_output) == len(complete_abstract_net_output))
#             for k in range(len(orig_net_output)):
#                 if orig_net_output[k][1] > heuristic_net_output[k][1] or heuristic_net_output[k][1] > complete_abstract_net_output[k][1]:
#                     print("orig_net_output[{}][1] > heuristic_net_output[{}][1] or "
#                           "heuristic_net_output[{}][1] > complete_abstract_net_output[{}][1]".format(k, k, k, k))
#                     print(i, j)
#                     assert False

# def test_heuristic_abstract_smaller_than_complete_abstract_outputs_3():
#     """
#     test to validate that complete test_abstraction outputs >= heuristic test_abstraction outputs >= original net outputs:
#
#     for every ACASXU network, generate complete/heristic test_abstraction, then evaluate 100 inputs and
#     assert that the original net output is smaller than the heuristic abstracted net output
#     and that the heuristic abstracted net output is smaller than the complete abstract net output
#     """
#     N = 100
#     nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
#     test_property_1 = get_test_property_acas()
#     for i, filename in enumerate(os.listdir(nnet_dir)):  # eg ACASXU_run2a_1_1_batch_2000.nnet
#         # if i != 6:
#         #     continue
#         debug_print("{} {}".format(i, filename))
#         nnet_filename = os.path.join(nnet_dir, filename)
#         net = network_from_nnet_file(nnet_filename)
#         orig_net = copy.deepcopy(net)
#         complete_abstract_net = abstract_network(network=copy.deepcopy(net))
#         heuristic_abstract_net = heuristic_abstract(
#             network=copy.deepcopy(orig_net), test_property=test_property_1)
#         debug_print("filename={}".format(nnet_filename))
#         continue
#         assert orig_net.get_general_net_data()["num_nodes"] <= \
#                complete_abstract_net.get_general_net_data()["num_nodes"] <= \
#                heuristic_abstract_net.get_general_net_data()["num_nodes"]
#         for j in range(N):
#             input_values = {i: (0.5-random.random()) * 2 for i in range(len(net.layers[0].nodes))}
#             input_values = net.get_limited_random_input(test_property_1)
#             orig_net_output = sorted(orig_net.evaluate(input_values).items(), key=lambda x: x[0])
#             speedy_orig_net_output = orig_net.speedy_evaluate(input_values)
#             assert is_evaluation_result_equal(orig_net_output, speedy_orig_net_output)
#             complete_abstract_net_output = sorted(complete_abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
#             speedy_complete_abstract_net_output = complete_abstract_net.speedy_evaluate(input_values)
#             assert is_evaluation_result_equal(complete_abstract_net_output, speedy_complete_abstract_net_output)
#             heuristic_abstract_net_output = sorted(heuristic_abstract_net.evaluate(input_values).items(), key=lambda x: x[0])
#             speedy_heuristic_abstract_net_output = heuristic_abstract_net.speedy_evaluate(input_values)
#             assert is_evaluation_result_equal(heuristic_abstract_net_output, speedy_heuristic_abstract_net_output)
#
#             assert(len(orig_net_output) == len(complete_abstract_net_output))
#             assert(len(heuristic_abstract_net_output) == len(complete_abstract_net_output))
#             for k in range(len(orig_net_output)):
#                 if (orig_net_output[k][1] - heuristic_abstract_net_output[k][1]) > EPSILON or \
#                         (heuristic_abstract_net_output[k][1] - complete_abstract_net_output[k][1]) > EPSILON:
#                     import IPython
#                     IPython.embed()
#                     assert False
#
#
# def test_heuristic_abstract_smaller_than_complete_abstract_outputs_4():
#     create_network_func = create_2_2_1_rand_weights_net
#     test_property = get_test_property_input_2_output_1()
#     check_heuristic_abstract_smaller_than_complete_abstract_outputs(create_network_func=create_network_func,
#                                                                    test_property=test_property)
#
#
# def test_heuristic_abstract_smaller_than_complete_abstract_outputs_5():
#     create_network_func = create_2_2_2_1_rand_weights_net
#     test_property = get_test_property_input_2_output_1()
#     check_heuristic_abstract_smaller_than_complete_abstract_outputs(create_network_func=create_network_func,
#                                                                    test_property=test_property)
#
#
# def test_heuristic_abstract_smaller_than_complete_abstract_outputs_6():
#     create_network_func = create_3_3_3_1_rand_weights_net
#     test_property = get_test_property_input_3_output_1()
#     check_heuristic_abstract_smaller_than_complete_abstract_outputs(create_network_func=create_network_func,
#                                                                    test_property=test_property)
#
#
# # -------------------------------------------------------------------------
# # test net_from_nnet_file by checking that the same output is accepted
# def test_network_from_nnet_file_acas_tiny_nets():
#     outputs = [
#         {
#             '0': -0.22975206607673998,
#             '1': 0.007867314468542996,
#             '2': 0.875059087916634,
#             '3': 0.10108754874669001,
#             '4': 0.078946838829464
#         },
#         {
#             '0': -1.293840174985929,
#             '1': -0.25361091752618714,
#             '2': -1.1841821124083598,
#             '3': -1.9064481569480283,
#             '4': -2.422375480309849
#         },
#         {
#             '0': 0.49004888651636336,
#             '1': -1.9145215075374988,
#             '2': 2.2510454229235766,
#             '3': -0.7326971935354969,
#             '4': 1.7257435902500207
#         },
#         {
#             '0': -9.613632366207742,
#             '1': -1.7802944197353936,
#             '2': -1.570224805467037,
#             '3': 2.900541518215031,
#             '4': -7.767607497595311
#         },
#         {
#             '0': 1.34936633469572,
#             '1': -16.246148925939426,
#             '2': -2.0212639644716854,
#             '3': -17.584466399124917,
#             '4': -1.925012406235469
#         }
#     ]
#     tiny_nnet_dir = PATH_TO_MARABOU_ACAS_EXAMPLES
#     # nnet_filename = os.path.join(tiny_nnet_dir, "ACASXU_run2a_1_1_tiny_5.nnet")
#     nets = [fname for fname in os.listdir(tiny_nnet_dir) if "tiny" in fname and fname.endswith(".nnet")]
#     for i, nnet_filename in enumerate(sorted(nets)):
#         net = network_from_nnet_file(os.path.join(tiny_nnet_dir, nnet_filename))
#         input_values = {0: 0.6000, 1: -0.5000, 2: -0.5000, 3: 0.4500, 4: -0.4500}
#         output = net.evaluate(input_values)
#         speedy_output = net.speedy_evaluate(input_values)
#         assert is_evaluation_result_equal(output.items(), speedy_output)
#         # the names of nodes are different so work with node index
#         for y_index, y_value in outputs[i].items():
#             index = [name for name in output.keys() if name.endswith(y_index)][0]
#             output_val = output[index]
#             assert(output_val == y_value)
#
#
# def test_network_from_nnet_file_acas_net_1_1():
#     correct_output = {'0': -0.022128988677506706,
#                       '1': -0.01904528058874152,
#                       '2': -0.019140123561458566,
#                       '3': -0.019152128000934545,
#                       '4': -0.019168840924865056}
#     nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
#     nnet_filename = os.path.join(nnet_dir, "ACASXU_run2a_1_1_batch_2000.nnet")
#     net = network_from_nnet_file(nnet_filename)
#     input_values = {0: 0.6000, 1: -0.5000, 2: -0.5000, 3: 0.4500, 4: -0.4500}
#     output = net.evaluate(input_values)
#     speedy_output = net.speedy_evaluate(input_values)
#     assert is_evaluation_result_equal(output.items(), speedy_output)
#     for y_index, y_value in correct_output.items():
#         index = [name for name in output.keys() if name.endswith(y_index)][0]
#         output_val = output[index]
#         assert(output_val == y_value)
#
#     assert output
