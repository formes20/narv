import itertools
from typing import AnyStr, List, Dict
from collections import defaultdict
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.utils.activation_functions import relu
from core.utils.abstraction_utils import choose_weight_func
from core.utils.dict_merge import fun_dict_merge


class Layer:
    """
    This class represents a layer in a neural network that supports abstraction
    and refinement steps in the process of verification. A layer has list of
    ARNodes and layer type (input, hidden, output), and include some functions
    to manipulate the nodes in it, e.g abstract, split_pos_neg, split_inc_dec
    """

    def __init__(self, type_name: AnyStr = "hidden", nodes: List = []):
        self.nodes = nodes
        self.type_name = type_name  # type_name is one of hidden/input/output

    def __eq__(self, other) -> bool:
        if len(self.nodes) != len(other.nodes):
            return False
        other_nodes_sorted = sorted(other.nodes, key=lambda node: node.name)
        for i, node in enumerate(sorted(self.nodes, key=lambda node: node.name)):
            if node != other_nodes_sorted[i]:
                print("self.nodes[{}] ({}) != other.nodes[{}] ({})".format(i, node, i, other_nodes_sorted[i]))
                return False
        return True

    def evaluate(self, cur_values: List, nodes2variables: Dict, next,
                 variables2nodes: Dict, variable2layer_index: Dict) -> List:
        """
        return the next layer values, given cur_values as self inputs
        """
        cur_var2val = {nodes2variables[node.name] for node in self.nodes}
        next_var2val = {nodes2variables[node.name] for node in next.nodes}
        out_values = []
        for i, val in enumerate(cur_values):
            for out_edge in self.nodes[i].out_edges:
                pass
        return out_values

    def split_pos_neg(self, name2node_map: Dict) -> None:
        """
        split nodes in layer to pos/neg nodes
        :param name2node_map: Dict map from name to relevant ARNode
        """
        if self.type_name == "output":
            # all nodes are increasing nodes
            # new_nodes = []
            for node in self.nodes:
                node.in_edges = []
        else:
            assert self.type_name == "hidden"
            new_nodes = []
            for node in self.nodes:
                # add the 2 nodes (that are splitted from node) to new_nodes
                splitted_nodes = node.split_pos_neg(name2node_map)
                new_nodes.extend(splitted_nodes)
                for sn in splitted_nodes:
                    name2node_map[sn.name] = sn
                del (name2node_map[node.name])
            self.nodes = new_nodes

    def split_inc_dec(self, name2node_map: Dict) -> None:
        """
        split nodes in layer to inc/dec nodes
        :param name2node_map: Dict map from name to relevant ARNode
        """
        if self.type_name == "output":
            # all nodes are increasing nodes
            new_nodes = []
            for node in self.nodes:
                new_node = ARNode(name=node.name + "_inc",
                                  ar_type="inc",
                                  activation_func=node.activation_func,
                                  in_edges=[],
                                  out_edges=[],
                                  bias=node.bias,
                                  upper_bound=node.upper_bound,
                                  lower_bound=node.lower_bound
                                  )
                new_nodes.append(new_node)
                name2node_map[new_node.name] = new_node
                del (name2node_map[node.name])
            self.nodes = new_nodes
        else:
            # split to increasing and decreasing nodes
            new_nodes = []
            for node in self.nodes:
                # add the 2 nodes (that are splitted from node) to new_nodes
                splitted_nodes = node.split_inc_dec(name2node_map)
                new_nodes.extend(splitted_nodes)
                for sn in splitted_nodes:
                    name2node_map[sn.name] = sn
                del (name2node_map[node.name])
            self.nodes = new_nodes

    def get_couples_of_same_ar_type(self) -> List:
        layer_couples = list(itertools.combinations(self.nodes, 2))
        layer_couples = [couple for couple in layer_couples if
                         couple[0].ar_type == couple[1].ar_type and
                         ((couple[0].out_edges[0].weight >= 0) == (couple[1].out_edges[0].weight >= 0))]
        return layer_couples

    def get_ar_type2nodes(self) -> List:
        """
        :return: dict from ar_types to list of all nodes with that ar_type
        """
        node_type2nodes = defaultdict(list)
        for node in self.nodes:
            node_type2nodes[node.ar_type].append(node)
        return node_type2nodes

    def get_same_type_nodes(self, is_pos: bool, is_inc: bool) -> List:
        node_names = []
        for node in self.nodes:
            if pos_neg:
                if is_inc:
                    if node.ar_type == "inc" and node.out_edges[0].weight >= 0:
                        node_names.append(node.name)
                else:
                    if node.ar_type == "inc" and node.out_edges[0] < 0:
                        node_names.append(node.name)
            else:
                if is_inc:
                    if node.ar_type == "dec" and node.out_edges[0].weight >= 0:
                        node_names.append(node.name)
                else:
                    if node.ar_type == "dec" and node.out_edges[0] < 0:
                        node_names.append(node.name)
        return node_names

    def abstract(self, name2node_map: Dict, next_layer_part2union: Dict) -> Dict:
        if self.type_name == "output":
            # output layer and its nodes' names are not changed
            # return trivial next_layer_part2union
            return {node.name: node.name for node in self.nodes}

        node_inc_pos = ARNode(name="",
                              ar_type="inc",
                              activation_func=relu,
                              in_edges=[],
                              out_edges=[],
                              bias=0.0
                              )
        node_inc_neg = ARNode(name="",
                              ar_type="inc",
                              activation_func=relu,
                              in_edges=[],
                              out_edges=[],
                              bias=0.0
                              )
        node_dec_pos = ARNode(name="",
                              ar_type="dec",
                              activation_func=relu,
                              in_edges=[],
                              out_edges=[],
                              bias=0.0
                              )
        node_dec_neg = ARNode(name="",
                              ar_type="dec",
                              activation_func=relu,
                              in_edges=[],
                              out_edges=[],
                              bias=0.0
                              )

        current_layer_part2union = {}

        # name of union node include its inner nodes' names with "+" in between
        # e.g name of union node of x11_inc,x12_inc is "x11_inc+x12_inc"
        for node in self.nodes:
            if node.ar_type == "inc":
                if node.out_edges[0].weight >= 0:
                    union_node = node_inc_pos
                else:
                    union_node = node_inc_neg
            if node.ar_type == "dec":
                if node.out_edges[0].weight >= 0:
                    union_node = node_dec_pos
                else:
                    union_node = node_dec_neg
            union_node.name += ("+" if union_node.name else "") + node.name
            current_layer_part2union[node.name] = union_node
            # union_node.out_edges.extend(node.out_edges)

        # update current_layer_part2union to include updated names
        # "updated" means after listing all parts in the for loop above
        for part_node, union_node in current_layer_part2union.items():
            current_layer_part2union[part_node] = union_node.name

        abstract_layer_nodes = []
        for node in [node_inc_pos, node_inc_neg, node_dec_pos, node_dec_neg]:
            if node.name:
                abstract_layer_nodes.append(node)

        dest_part2dest = {}
        dest2part_weights = {
            # notice: handle duplicated dests by using dict
            dest: {} for dest in next_layer_part2union.values()
        }
        dest_names = set([])

        for node in abstract_layer_nodes:
            # node_positivity = node.get_positivity()
            node.new_out_edges = []

            # get parts
            node_parts = node.name.split("+")

            # get dests
            current_dest_names = set([])
            for part in node_parts:
                for edge in name2node_map[part].out_edges:
                    dest = next_layer_part2union[edge.dest]
                    current_dest_names.add(dest)
                    dest_names.add(dest)

            # get dest_part2dest
            for dest in current_dest_names:
                dest_node_parts = name2node_map[dest].name.split("+")
                for dest_part in dest_node_parts:
                    dest_part2dest[dest_part] = dest

            # get a dictionary {dest: min/max weight from specific part to it}
            for part in node_parts:
                for edge in name2node_map[part].out_edges:
                    dest = dest_part2dest[edge.dest]
                    dest_node = name2node_map[dest]
                    max_min_func, cur_weight = choose_weight_func(node, dest_node)
                    cur_weight = dest2part_weights[dest].get(part, cur_weight)
                    dest2part_weights[dest][part] = max_min_func(cur_weight, edge.weight)
            # choose the minimal/maximal weight as the weight from node to dest
            dest_weights = {}
            for dest, part_weights in dest2part_weights.items():
                if dest not in current_dest_names:
                    continue
                part_weights = {p: ws for p, ws in part_weights.items()
                                if p in node_parts
                                }
                dest_weights[dest] = sum(part_weights.values())

                # define the node bias to be the min/max bias among all parts
                part_nodes = [name2node_map[np] for np in node_parts]
                parts_biases = [part_node.bias for part_node in part_nodes]
                node.bias = max_min_func(parts_biases)

            # update node.out_edges and dest.in_edges
            for dest, weight in dest_weights.items():
                edge = Edge(src=node.name, dest=dest, weight=weight)
                node.new_out_edges.append(edge)
                dest_node = name2node_map[dest]
                try:
                    dest_node.new_in_edges.append(edge)
                except AttributeError:
                    dest_node.new_in_edges = [edge]

            # update the out edges of the node
            node.out_edges = node.new_out_edges
            del node.new_out_edges

        # update the in edges of the dest node
        for dest in dest_names:
            # union_dest is the node which include dest
            union_dest = name2node_map[dest]
            union_dest.in_edges = union_dest.new_in_edges
            del union_dest.new_in_edges
        self.nodes = abstract_layer_nodes
        return current_layer_part2union

    @staticmethod
    def get_loss(part: AnyStr, node: ARNode,
                 part2node_map: Dict[AnyStr, ARNode]) -> float:
        total_loss = 0
        for part_edge in part.out_edges:
            # get the dest to get its node to get the edge to decrease
            part_dest_node = part2node_map[part_edge.dest]
            # NOTE: can improve performance by pre-calculating node_dest2edge_map
            # node_edge = node_dest2edge_map[(node,part_dest_node)]
            for node_edge in node.out_edges:
                if node_edge.dest == part_dest_node:
                    break
            total_loss += abs(part_edge.weight - node_edge.weight)
        return total_loss

    def __str__(self) -> str:
        return self.type_name + "\n\t" + "\n\t".join(node.__str__() for node in self.nodes)

    def generate_ori_nodename2weight(self) -> None:

        # if self.type_name == "input":

        #    for node in self.nodes:
        #        node.ori_nodename2weight = {node.name:1}
        # else:
        for node in self.nodes:
            node.generate_ori_nodename2weight()
            # print(node)

    def generate_symb_map(self, name2node_map: Dict) -> None:
        for node in self.nodes:
            node.ori_nodename2weight = node.find_ori_symbolic(name2node_map)
            # print(node)
            node.calc_bounds(name2node_map)
            # print(node)

    def calculate_bounds(self, name2node_map: Dict) -> None:
        for node in self.nodes:
            node.calc_bounds(name2node_map)

    def generate_in_edge_weight_sum(self) -> None:
        for node in self.nodes:
            node.sum_in_edges_weights()
