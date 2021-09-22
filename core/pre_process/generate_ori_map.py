from core.data_structures.Network import Network
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Layer import Layer

def generate_ori_net_map(network:Network) -> None:
    for i in range(len(network.layers)):
        network.layers[i].generate_ori_nodename2weight()

def genarate_symb_map(network:Network, name2node_map) -> None:
    for i in range(1,len(network.layers)):
        network.layers[i].generate_symb_map(name2node_map)

        #ce shi len(network.layers)