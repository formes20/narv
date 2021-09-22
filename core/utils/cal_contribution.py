from core.data_structures.Network import Network
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Layer import Layer
import copy

def calculate_contribution (network:Network, layer_index: float) -> list:
    contri_list = []
    for node in network.layers[layer_index].nodes:
        contribution = node.cal_contri()
        contri_list.append([node.name,contribution])
    
    return copy.deepcopy(contri_list)