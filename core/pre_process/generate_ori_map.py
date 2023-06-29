from core.data_structures.Network import Network


def generate_ori_net_map(network: Network) -> None:
    for i in range(len(network.layers)):
        network.layers[i].generate_ori_nodename2weight()


def genarate_symb_map(network: Network, name2node_map) -> None:
    for i in range(1, len(network.layers)):
        network.layers[i].generate_symb_map(name2node_map)
