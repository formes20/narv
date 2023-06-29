from core.data_structures.Network import Network
from .read_input_bound import read_bounds_from_property
from .calc_bounds import calcu_bounds
from .generate_ori_map import genarate_symb_map, generate_ori_net_map
from .del_dec_nodes import delete_dec_nodes


def do_process_before(network: Network, test_property: str, NAIVE_BOUND_CALCULATION: bool = False) -> None:
    network.generate_name2node_map()
    generate_ori_net_map(network)
    # print(network)
    read_bounds_from_property(network, test_property)
    if NAIVE_BOUND_CALCULATION:
        calcu_bounds(network, network.name2node_map)
    else:
        genarate_symb_map(network, network.name2node_map)
    # print(network)


def do_process_after(network: Network) -> None:
    delete_dec_nodes(network)
