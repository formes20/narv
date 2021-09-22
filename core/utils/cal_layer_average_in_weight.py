from core.data_structures.Network import Network
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Layer import Layer
import copy 

def cal_average_in_weight(network:Network, layer_index:float) -> float:
    sum_of_nodes_in_weight = 0
    for node in network.layers[layer_index].nodes:
        sum_of_nodes_in_weight += node.sum_in_edges
    return sum_of_nodes_in_weight/len(network.layers[layer_index].nodes)