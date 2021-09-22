from core.abstraction.step import union_couple_of_nodes
from core.configuration.consts import VERBOSE, FIRST_ABSTRACT_LAYER
from core.data_structures.Network import Network
from core.pre_process.pre_process import preprocess
from core.pre_process.after_preprocess import after_preprocess
from core.utils.debug_utils import debug_print
from core.utils.abstraction_utils import finish_abstraction
from core.visualization.visualize_network import visualize_network
from core.pre_process.pre_process_mine import do_process_before,do_process_after
#from core.utils.alg2_utils import has_violation, get_limited_random_inputs
from core.abstraction.step import union_couple_of_nodes
from core.utils.abstraction_utils import finish_abstraction
from core.configuration.consts import (
    VERBOSE, FIRST_ABSTRACT_LAYER, INT_MIN, INT_MAX
)
from core.utils.verification_properties_utils import TEST_PROPERTY_ACAS
from core.utils.alg2_utils import has_violation, get_limited_random_inputs
from core.utils.cal_contribution import calculate_contribution
from core.utils.cal_layer_average_in_weight import cal_average_in_weight
from core.utils.combine_influ_of2nodes import combin_influnce
from core.data_structures.Abstract_action import Abstract_action 
from core.utils.find_relation import find_relation
from core.abstraction.after_abstraction import after_abstraction
import copy
import time
import _pickle as cPickle

def propagation_net(network: Network):
    for layer_index in range(len(network.layers)-2,FIRST_ABSTRACT_LAYER-1,-1):
        for node_index in range(len(network.layers[layer_index].nodes)-1,-1,-1):
            node = network.layers[layer_index].nodes[node_index]
            if node.deleted:
                for out_edge in node.out_edges:
                    out_node = network.name2node_map[out_edge.dest]
                    if out_node.deleted:
                        assert out_edge.weight == 0
                    if not out_node.deleted:
                        out_node.bias += node.bias*out_edge.weight
                network.remove_node(node, layer_index)
    network.biases = network.generate_biases()
    network.weights = network.generate_weights()
