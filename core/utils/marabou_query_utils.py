#!/usr/bin/env python3

from typing import Tuple, Sized

from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions


from core.configuration.consts import (
    VERBOSE, INT_MAX, SAT_EXIT_CODE, UNSAT_EXIT_CODE, INPUT_LOWER_BOUND,
    INPUT_UPPER_BOUND, FIRST_ABSTRACT_LAYER
)
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Layer import Layer
from core.utils.verification_properties_utils import get_winner_and_runner_up
from core.utils.activation_functions import identity


def initiate_query() -> str:
    input_query = "\nfrom maraboupy import MarabouCore\n\n"
    input_query += "inputQuery = MarabouCore.InputQuery()\n\n"
    return input_query


def get_num_vars(variables2nodes) -> str:
    set_num_vars = "inputQuery.setNumberOfVariables({})\n\n"
    set_num_vars = set_num_vars.format(len(variables2nodes.keys()))
    return set_num_vars


def finish_query() -> str:
    end_query = "vars1, stats1 = MarabouCore.solve(inputQuery, \"\", 0)\n\n"
    end_query += "\n".join(["print(vars1, stats1)",
                            "if len(vars1)>0:",
                            "\tprint('SAT')",
                            "\texit({})".format(SAT_EXIT_CODE),
                            "else:",
                            "\tprint('UNSAT')",
                            "\texit({})".format(UNSAT_EXIT_CODE)])
    return end_query + "\n"



# def handle_acas_xu_conjunction(net:Sized, test_property) -> Tuple:
#     """
#     The purpose of this function is to support the properties of ACAS XU which
#     include a conjunction of conditions regarding the output
#     This is done by adding two last layers that encodes the conjunction formula
#     using logic formula: X AND Y = NOT(NOT(X) OR NOT(Y)), therefore negate the
#     first equation, sum the results to output node and check the negated
#     property ("Lower" instead "Upper" in acas_xu case) on the output node
#     because the current abstraction code supports abstraction when the property
#     is "Lower", we both negate the weights of the output node in-edges from 1
#     to -1 and change the "Upper" to "Lower" with the negated number (e.g. -4
#     instead of 4)
#     :param net: Network
#     :param test_property: Dict that represents a test_property in ACAS XU form
#     :return: Tuple of updated network and test_property
#     """

#     # first, add the last hidden layer
#     new_last_hidden_nodes = []
#     for i, formula in enumerate(test_property["output"]):
#         # new_test_property_bounds.append(formula[1])
#         node_name = f"{len(net.layers)}_{i}"
#         formula_out_node = ARNode(name=node_name, ar_type="",
#                                   in_edges=[], out_edges=[])
#         formula_left_size = formula[0] # e.g ([(1,0), (-1,1)], {"Upper": 0})
#         used_indices = []
#         for formula_part in formula_left_size:
#             node_index = formula_part[1]
#             used_indices.append(node_index)
#             out_edge_weight = formula_part[0]
#             edge = Edge(src=net.layers[-1].nodes[node_index].name,
#                         dest=formula_out_node.name,
#                         weight=out_edge_weight)
#             # print(f"edge={edge}")
#             formula_out_node.in_edges.append(edge)
#             net.layers[-1].nodes[node_index].out_edges.append(edge)
#         unused_indices = list(set(range(len(net.layers[-1].nodes))).difference(used_indices))
#         for j in unused_indices:
#             edge = Edge(src=net.layers[-1].nodes[j].name,
#                         dest=formula_out_node.name,
#                         weight=0.0)
#             # print(f"0 edge={edge}")
#             formula_out_node.in_edges.append(edge)
#             net.layers[-1].nodes[node_index].out_edges.append(edge)

#         new_last_hidden_nodes.append(formula_out_node)
#         # new_output_layer_nodes.append(formula_out_node)

#     net.layers[-1].type_name = "hidden"
#     new_last_hidden_layer = Layer(
#         nodes=new_last_hidden_nodes, type_name="hidden")
#     net.layers.append(new_last_hidden_layer)

#     # then, append the output layer
#     output_node_name = f"{len(net.layers)}_0"
#     output_node = ARNode(name=output_node_name, activation_func=identity,
#                          ar_type="", in_edges=[], out_edges=[])
#     for node in new_last_hidden_nodes:
#         # edge weights = -1 to implement property negation
#         edge = Edge(src=node.name, dest=output_node.name, weight=-1.0)
#         node.out_edges.append(edge)
#         output_node.in_edges.append(edge)

#     new_output_layer = Layer(nodes=[output_node], type_name="output")
#     net.layers.append(new_output_layer)

#     test_property["output"] = [
#         (0, {"Lower": 0})
#     ]

#     net.generate_name2node_map()
#     net.weights = net.generate_weights()
#     net.biases = net.generate_biases()
#     # from core.visualization.visualize_network import visualize_network
#     # visualize_network(network=net)
#     # print(net)
#     return net, test_property

# def handle_adversarial(net, test_property) -> Tuple:
#     minimum = 100
#     for index, bound in test_property["output"]:
#         if bound["Lower"] < minimum:
#             minimum = bound["Lower"]
#             minimum_ind = index
#     minimum_node = net.layers[-1].nodes[minimum_ind]
#     new_last_hidden_nodes = []
#     #print(list(set(range(len(net.layers[-1].nodes))).difference([minimum_ind])))
#     for i,j in enumerate(list(set(range(len(net.layers[-1].nodes))).difference([minimum_ind]))):
#         #print("j"+str(j))
#         node_name = f"x_{len(net.layers)-1}_{i}"
#         new_node = ARNode(name=node_name, ar_type="",
#                                   in_edges=[], out_edges=[])
#         another_node = net.layers[-1].nodes[j]
#         minimum_in_edges = sorted(minimum_node.in_edges, key=lambda x:x.src)
#         another_in_edges = sorted(another_node.in_edges, key=lambda x:x.src)
#         assert(len(minimum_in_edges) == len(another_in_edges))
#         for k, m_edge in enumerate(minimum_in_edges):
#             new_node.in_edges.append(
#                 Edge(
#                     src=m_edge.src, dest=new_node.name,
#                     weight=m_edge.weight - another_in_edges[k].weight
#                 )
#             )
#         for node in net.layers[-2].nodes:
#             node.out_edges = [edge for edge in new_node.in_edges if edge.src == node.name]
#         new_last_hidden_nodes.append(new_node)
#     net.layers[-1].type_name = "hidden"
#     net.layers[-1].nodes = new_last_hidden_nodes
#     output_node_name = f"x_{len(net.layers)}_0"    
#     output_node = ARNode(name=output_node_name, activation_func=identity,
#                          ar_type="", in_edges=[], out_edges=[])
#     for node in new_last_hidden_nodes:
#         edge = Edge(src=node.name, dest=output_node.name, weight=1.0)
#         node.out_edges.append(edge)
#         output_node.in_edges.append(edge)
#     new_output_layer = Layer(nodes=[output_node], type_name="output")
#     net.layers.append(new_output_layer)
#     # print(net.layers[-1])
#     # print("###############################")
#     # print(net.layers[-2])
#     test_property["output"] = [
#         (0, {"Lower": 0.0001})
#     ]
#     # print(test_property["output"])
#     net.generate_name2node_map()
#     net.weights = net.generate_weights()
#     net.biases = net.generate_biases()
#     return net, test_property

# FOLLOWING IS FOR MNIST BENCHMARK
def handle_adversarial(net, test_property) -> Tuple:
    minimum = 100
    for index, bound in test_property["output"]:
        if bound["Lower"] < minimum:
            minimum = bound["Lower"]
            minimum_ind = index
    minimum_node = net.layers[-1].nodes[minimum_ind]
    new_last_hidden_nodes = []
    print("labeled as {}".format(minimum_ind))
    #print(list(set(range(len(net.layers[-1].nodes))).difference([minimum_ind])))
    for node in net.layers[-2].nodes:
        node.out_edges = []
    for i,j in enumerate(list(set(range(len(net.layers[-1].nodes))).difference([minimum_ind]))):
        #print("j"+str(j))
        node_name = f"x_{len(net.layers)-1}_{i}"
        new_node = ARNode(name=node_name, ar_type="",
                                  in_edges=[], out_edges=[])
        another_node = net.layers[-1].nodes[j]
        minimum_in_edges = sorted(minimum_node.in_edges, key=lambda x:x.src)
        another_in_edges = sorted(another_node.in_edges, key=lambda x:x.src)
        assert(len(minimum_in_edges) == len(another_in_edges))
        for k, m_edge in enumerate(minimum_in_edges):
            new_node.in_edges.append(
                Edge(
                    src = m_edge.src, dest=new_node.name,
                    weight = (another_in_edges[k].weight - m_edge.weight)
                )
            )
        for node in net.layers[-2].nodes:
            for edge in new_node.in_edges:
                if edge.src == node.name:
                    node.out_edges.append(edge)
            #print(len(node.out_edges))
        new_last_hidden_nodes.append(new_node)
    net.layers[-1].type_name = "hidden"
    net.layers[-1].nodes = new_last_hidden_nodes
    output_node_name = f"x_{len(net.layers)}_0"    
    output_node = ARNode(name=output_node_name, activation_func=identity,
                         ar_type="", in_edges=[], out_edges=[])
    for node in new_last_hidden_nodes:
        edge = Edge(src=node.name, dest=output_node.name, weight=1.0)
        node.out_edges.append(edge)
        output_node.in_edges.append(edge)
    new_output_layer = Layer(nodes=[output_node], type_name="output")
    net.layers.append(new_output_layer)
    # print(net.layers[-1])
    # print("###############################")
    # print(net.layers[-2])
    test_property["output"] = [
        (0, {"Lower": 0.001})
    ]
    # print(test_property["output"])
    net.generate_name2node_map()
    net.weights = net.generate_weights()
    net.biases = net.generate_biases()
    return net, test_property


# def handle_adversarial(net, test_property) -> Tuple:
#     """
#     replace network last output layer to include one node that contain the
#     difference of the two noded winner and runner_up, which are the nodes with
#     1st and 2nd minimal values. it is equal to additional layer with one node
#     with the difference of the winner and runner_up nodes in the original output
#     layer.

#     # past implementation - theoretically equal to cuurent implementation
#     # change net to net with additional layer which is the new output layer, that
#     # include on node that include the difference of the two nodes whose values
#     # are the minimal values wrt the "center-point" of the adversarial propery
#     # test_property, (which is actually represents that in an some n-dimensional
#     # circle whose center is some n-dimensional point X the winner(minimal) wins
#     # (is less than) runner(second minimal)

#     :param net: original network to check the property on
#     :param test_property: is actually represents that in an some n-dimensional
#     circle whose center is some n-dimensional point X the winner(minimal) wins
#     (is less than) runner(second minimal)
#     # :return: new network with additional layer (the new output layer), that
#     # include on node that gets the difference of the two nodes whose values
#     # are the minimal values wrt the "center-point" (denoted with X above) of the
#     # adversarial propery
#     # test_property
#     :return: Tuple, updated network and updated property
#     """
#     winner, w_index, runner_up, r_index = get_winner_and_runner_up(
#         adversarial_test_property=test_property)
#     # print(f'w_index={w_index}, r_index={r_index}')
#     net_winner_node = net.layers[-1].nodes[w_index]
#     net_runner_up_node = net.layers[-1].nodes[r_index]
#     output_name = f"x_{len(net.layers)-1}_0"
#     output_node = ARNode(name=output_name, ar_type="", in_edges=[],
#                          out_edges=[], activation_func=identity)
#     # output-node in-edges are the difference of runner_up and winner in-edges
#     w_in_edges = sorted(net_winner_node.in_edges, key=lambda x:x.src)
#     r_in_edges = sorted(net_runner_up_node.in_edges, key=lambda x:x.src)
#     assert(len(w_in_edges) == len(r_in_edges))
#     for i, w_edge in enumerate(w_in_edges):
#         output_node.in_edges.append(
#             Edge(
#                 src=w_edge.src, dest=output_node.name,
#                 weight=-(r_in_edges[i].weight - w_edge.weight)
#             )
#         )

#     # fix last_hidden nodes' out edges (remove irrelevant, change dest name)
#     for node in net.layers[-2].nodes:
#         node.out_edges = [edge for edge in output_node.in_edges if edge.src == node.name]
#     net.layers[-1].nodes = [output_node]

#     # test_property["prev_output"] = test_property["output"]
#     test_property["output"] = [
#         (0, {"Lower": 0})
#     ]
#     net.generate_name2node_map()
#     net.weights = net.generate_weights()
#     net.biases = net.generate_biases()
#     return net, test_property

def reduce_property_to_basic_form(network, test_property) -> Tuple:
    if "type" not in test_property.keys():
        err_msg = f"'type' not in test_property.keys(): {test_property.keys()}"
        raise Exception(err_msg)

    property_type = test_property["type"]
    print(f"reduce_property_to_basic_form(): property_type={property_type}")
    if property_type == "basic":
        pass
    elif property_type == "adversarial":
        network, test_property = handle_adversarial(
            net=network, test_property=test_property
        )
    elif property_type == "acas_xu_conjunction":
        network, test_property = handle_acas_xu_conjunction(
            net=network, test_property=test_property
        )
    # elif property_type == "local_robustness":
    #     network, test_property = handle_robustness(
    #         net=network, test_property=test_property
    #     )
    else:
        err_msg = f"property type is not supported: {test_property['type']}"
        raise Exception(err_msg)
    return network, test_property


def get_query(network, test_property, verbose:bool=VERBOSE) -> Tuple:
    """
    @test_property is a property to check in the network, of the form:
    {
        layer_name:
        [
            (variable_name, {"Lower": l_value, "Upper": u_value}),
            (variable_name, {"Lower": l_value, "Upper": u_value})
            ...
            (variable_name, {"Lower": l_value, "Upper": u_value})
        ],
        ...
    }
    e.g:
    {
        "input":
            [
                (0, {"Lower": 0, "Upper": 1}),
                (1, "Lower", 2),
                (2, "Upper", -1),
            ],
        "output":
            [
                (0, {"Lower": 0, "Upper": 1}),
                (1, "Lower", -4),
                (2, "Upper", 1.6),
            ]
    }
    @return Marabou query - does @test_property holds in @network?
    """
    property_type = test_property["type"]
    inputQuery = MarabouCore.InputQuery()
    # large
    large = INT_MAX
    nodes2variables, variables2nodes = network.get_variables(
        property_type=property_type
    )
    # setNumberOfVariables
    inputQuery.setNumberOfVariables(len(variables2nodes.keys()))
    # bounds
    out_layer_var_index = len(nodes2variables) - len(network.layers[-1].nodes)

    # "fix" test_property: add [-large,large] bounds to missing variables
    for i in range(len(network.layers[0].nodes)):
        if i not in [x[0] for x in test_property["input"]]:
            print("var {} is missing in test_property".format(i))
            missing_var_bounds = (i, {"Lower": -large,
                                      "Upper": large})
            test_property["input"].append(missing_var_bounds)
    # for i in range(len(self.layers[-1].nodes)):
    #    if i not in [x[0] for x in test_property["output"]]:
    #        missing_var_bounds = (i, {"Lower": -large, "Upper": large})
    #        test_property["output"].append(missing_var_bounds)

    for key, value in test_property.items():
        if key == "type":
            continue
        else:
            layer_name, bounds_list = key, value
        if layer_name not in ["input", "output"]:
            raise Exception(f"invalid test property key: {layer_name}")
        for (var_index, var_bounds_dict) in bounds_list:
            lower_bound = var_bounds_dict.get("Lower", -large)
            upper_bound = var_bounds_dict.get("Upper", large)
            if layer_name == "output":
                var_index = out_layer_var_index + var_index
            inputQuery.setLowerBound(var_index, lower_bound)
            if verbose:
                print("setLowerBound({}, {})".format(var_index, lower_bound))
            inputQuery.setUpperBound(var_index, upper_bound)
            if verbose:
                print("setUpperBound({}, {})".format(var_index, upper_bound))

    # print(f"nodes2variables={nodes2variables}")
    # # add lower and upper bounds to all hidden nodes
    # for layer in network.layers[1:-1]:
    #     for node in layer.nodes:
    #         var_b = nodes2variables[node.name + "_b"]
    #         var_f = nodes2variables[node.name + "_f"]
    #         inputQuery.setLowerBound(var_b, -large)
    #         inputQuery.setUpperBound(var_b, large)
    #         inputQuery.setLowerBound(var_f, -large)
    #         inputQuery.setUpperBound(var_f, large)

    # mark input and output variables
    for node_index, node in enumerate(network.layers[0].nodes):
        inputQuery.markInputVariable(node_index, node_index)
    for node_index, node in enumerate(network.layers[-1].nodes):
        inputQuery.markOutputVariable(out_layer_var_index + node_index, node_index)

    # equations
    i = 0
    for layer_index, layer in enumerate(network.layers):
        if layer.type_name == "input":
            continue
        for node in layer.nodes:
            equation = MarabouCore.Equation()
            # test_abstraction does not occur on layers
            # [0, ..., FIRST_ABSTRACT_LAYER], therefore after test_abstraction,
            # in_edges of layers [0, ..., FIRST_ABSTRACT_LAYER] nodes may
            # include multiple edges with same dest (if two dests were unioned,
            # the union in_edges include both edges), therefore the weights into
            # each union node have to be summed
            # example:
            # assume that start edges are: x00--(1)-->x10, x00--(2)-->x11
            # assume that during test_abstraction x10 and x11 became x1u (union)
            # so current edges: x00--(1)-->x1u, x00--(2)-->x1u
            # should avoid two "addAddend"s with variable x00 by summing
            # x00--(1+2)-->x1u
            if layer_index <= FIRST_ABSTRACT_LAYER:
                # print(f"layer_index == {FIRST_ABSTRACT_LAYER}")
                edge2weight = {}
                dest_seen = {}
                # eg: x1u.in_edges = [x00--(1)-->x1u, x00--(2)-->x1u]
                # generate {(x00,x1u): 3}
                for in_edge in node.in_edges:
                    if layer_index == 1:
                        # "_f" suffix does not exist in input layer node names
                        src_variable = nodes2variables[in_edge.src]
                    # !!!!!!!!!!!!!
                    # TODO: handle adversarial  ?  acasxu_conjunction  ?
                    # !!!!!!!!!!!!!
                    else:
                        src_variable = nodes2variables[in_edge.src + "_f"]
                    dest_variable = nodes2variables[in_edge.dest + "_b"]
                    key = (src_variable, dest_variable)
                    var_weight = edge2weight.get(key, 0.0)
                    new_weight = var_weight + in_edge.weight
                    edge2weight[(src_variable, dest_variable)] = new_weight
                for ((src_var, dest_var), weight) in edge2weight.items():
                    equation.addAddend(weight, src_var)
                    if verbose:
                        print("eq {}: addAddend({}, {})".format(i,
                                                                weight,
                                                                src_var))
                    if not dest_seen.get(dest_var, False):
                        equation.addAddend(-1, dest_var)
                        dest_seen[dest_var] = True
                        if verbose:
                            print("eq {}: addAddend({}, {})".format(i,
                                                                    -1,
                                                                    dest_var))
            else:
                if verbose:
                    print(f"layer_index({layer_index})!={FIRST_ABSTRACT_LAYER}")
                in_edge = None
                for in_edge in node.in_edges:
                    # in acas_xu_conjunction, the output layer became the third
                    # layer from end, and relu is not applied to its outputs
                    if property_type == "acas_xu_conjunction" and \
                            layer_index == len(network.layers)-2:
                        src_variable = nodes2variables[in_edge.src + "_b"]
                    # in adverserial robustness, the in-edges of the output
                    # layer come from the last hidden which was the previous
                    # output layer, and relu is not applied to its outputs
                    # elif property_type == "adversarial" and \
                    #         layer_index == len(network.layers) -1:
                    #     src_variable = nodes2variables[in_edge.src + "_b"]
                    else:
                        src_variable = nodes2variables[in_edge.src + "_f"]
                    equation.addAddend(in_edge.weight, src_variable)
                    if verbose:
                        print("eq {}: addAddend({}, {})".format(i,
                                                                in_edge.weight,
                                                                src_variable))
                dest_variable = nodes2variables.get(in_edge.dest + "_b", None)
                if dest_variable is None:
                    dest_variable = nodes2variables[in_edge.dest]
                equation.addAddend(-1, dest_variable)
                if verbose:
                    print("eq {}: addAddend(-1, {})".format(i, dest_variable))
            if verbose:
                print("eq {}: setScalar({})".format(i, -node.bias))
            equation.setScalar(-node.bias)
            inputQuery.addEquation(equation)
            if verbose:
                print("eq {}: addEquation".format(i))
            i += 1
    # relu constraints
    for layer_index, layer in enumerate(network.layers):
        if layer.type_name != "hidden":
            # print(f'layer.type_name({layer.type_name}) != "hidden"')
            # print(f'len(layer.nodes)={len(layer.nodes)}')
            continue
        # in acas_xu_conjunction, there are no relus in two last hidden layers
        # (the added last hidden layer and the original output layer)
        if property_type == "acas_xu_conjunction" and \
                layer_index == len(network.layers)-3:
            print(f'layer without relus, layer_index={layer_index}')
            # print(f'len(layer.nodes)={len(layer.nodes)}')
            continue
        # in adversarial robustness, there are no relus in the last hidden layer
        # (which is the original output layer)
        # if property_type == "adversarial" and \
        #         layer_index == len(network.layers)-2:
        #     print(f'old output without relus, layer_index={layer_index}')
        #     print(f'len(layer.nodes)={len(layer.nodes)}')
        #     continue
        for node in layer.nodes:
            node_b_index = nodes2variables.get(node.name + "_b", None)
            node_f_index = nodes2variables.get(node.name + "_f", None)
            if (node_b_index is None) or (node_f_index is None):
                continue
            # print("add Relu constraint: {}\t{}".format(node_b_index, node_f_index))
            MarabouCore.addReluConstraint(inputQuery,
                                          node_b_index,
                                          node_f_index)
            if verbose:
                print(f"add Relu constraint({node_f_index}=R({node_b_index}))")

    # solve query
    # print("bbb")
    # import IPython
    # IPython.embed()
    options = createOptions()
    MarabouCore.saveQuery(inputQuery, "/tmp/query.log")
    vars1, stats1 = MarabouCore.solve(inputQuery, options, "")
    result = 'SAT' if len(vars1) > 0 else 'UNSAT'
    return vars1, stats1, result


def get_query_str(self, test_property) -> str:
    """
    @test_property is a property to check in the network, of the form:
    {
        layer_name:
        [
            (variable_name, {"Lower": l_value, "Upper": u_value}),
            (variable_name, {"Lower": l_value, "Upper": u_value})
            ...
            (variable_name, {"Lower": l_value, "Upper": u_value})
        ],
        ...
    }
    e.g:
    {
        "input":
            [
                (0, {"Lower": 0, "Upper": 1}),
                (1, "Lower", 2),
                (2, "Upper", -1),
            ],
        "output":
            [
                (0, {"Lower": 0, "Upper": 1}),
                (1, "Lower", -4),
                (2, "Upper", 1.6),
            ]
    }
    @return Marabou query - is test_property holds in the network?
    """
    query = ""
    query += self.initiate_query()
    query += self.get_large()
    nodes2variables, variables2nodes = self.get_variables()
    query += self.get_num_vars(variables2nodes)
    query += self.get_bounds(nodes2variables, test_property)
    query += self.get_equations(nodes2variables)
    query += self.get_relu_constraints(nodes2variables)
    query += self.finish_query()
    return query


def get_bounds(self, nodes2variables, test_property) -> str:
    """
    returns the part in marabou query which is related to variables bounds.
    for details about @test_property, see get_query() method documentation.
    """
    bounds = ""
    out_layer_var_index = len(nodes2variables) - len(self.layers[-1].nodes)
    for layer_name, bounds_list in test_property.items():
        for (var_index, var_bounds_dict) in bounds_list:
            lower_bound = var_bounds_dict.get("Lower", "-large")
            upper_bound = var_bounds_dict.get("Upper", "large")
            if layer_name == "output":
                var_index = out_layer_var_index + var_index
            lower = "inputQuery.setLowerBound({}, {})\n"
            lower = lower.format(var_index, lower_bound)
            upper = "inputQuery.setUpperBound({}, {})\n"
            upper = upper.format(var_index, upper_bound)
            bounds += lower
            bounds += upper
        bounds += "\n"
    return bounds


def get_equations(self, nodes2variables) -> str:
    equations = ""
    eq_index = 0
    for layer in self.layers:
        if layer.type_name == "input":
            continue
        for node in layer.nodes:
            equations += self.get_equation(node, nodes2variables, eq_index)
            eq_index += 1
    return equations


def get_equation(node, nodes2variables, eq_index) -> str:
    # print(nodes2variables.items())
    equation = "equation{} = MarabouCore.Equation()\n".format(eq_index)
    in_edge = None
    for in_edge in node.in_edges:
        # default in case of input layer where there is no "f" in node name
        node_variable = nodes2variables.get(in_edge.src + "f", None)
        if node_variable is None:
            node_variable = nodes2variables.get(in_edge.src)
        equation_part = "equation{}.addAddend({}, {})\n"
        equation_part = equation_part.format(eq_index,
                                             in_edge.weight,
                                             node_variable)
        equation += equation_part
    # default in case of input layer where there is no "b" in node name
    node_variable = nodes2variables.get(in_edge.dest + "b", None)
    if node_variable is None:
        node_variable = nodes2variables.get(in_edge.dest)
    equation_part = "equation{}.addAddend({}, {})\n"
    equation_part = equation_part.format(eq_index, -1, node_variable)
    equation += equation_part

    equation_part = "equation{}.setScalar(0)\n".format(eq_index)
    equation_part += "inputQuery.addEquation(equation{})\n".format(eq_index)
    equation += equation_part
    equation += "\n"
    return equation


def get_relu_constraints(network, nodes2variables) -> str:
    """
    return string that includes the relu constraints in marabou query
    one constraint is derived from each node in every hidden layer
    """
    # NOTE: can use @network for node2layer_type_name map
    constrains = ""
    relu_con = None
    node_b_index = None
    node_f_index = None
    for layer in network.layers:
        if layer.type_name != "hidden":
            continue
        for node in layer.nodes:
            node_b_index = nodes2variables.get(node.name + "b", None)
            node_f_index = nodes2variables.get(node.name + "f", None)
            if (node_b_index is None) or (node_f_index is None):
                continue
            relu_con = "MarabouCore.addReluConstraint(inputQuery, {}, {})\n"
        constrains += relu_con.format(node_b_index, node_f_index)
        # MarabouCore.addReluConstraint(inputQuery,3,4)
    constrains += "\n"
    return constrains
