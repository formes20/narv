from typing import AnyStr, List, Dict
from core.data_structures.Network import Network
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Layer import Layer

def calcu_bounds(network:Network, name2node_map) -> None:
    for i in range(1, len(network.layers)-1):
        network.layers[i].calculate_bounds(name2node_map)