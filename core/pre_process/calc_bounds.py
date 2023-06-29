from core.data_structures.Network import Network


def calcu_bounds(network: Network, name2node_map) -> None:
    for i in range(1, len(network.layers) - 1):
        network.layers[i].calculate_bounds(name2node_map)
