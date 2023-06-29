import sys
import os
import json
import copy
import time
import pandas as pd
# import timeout_decorator

from import_marabou import dynamically_import_marabou
from core.configuration import consts
from experiments.consts import BEST_CEGARABOU_METHODS
from core.utils.debug_utils import debug_print
from core.utils.verification_properties_utils import is_satisfying_assignment
from core.abstraction.global_abstraction import global_abstraction_based_on_contribution
from core.abstraction.kmeans_abstraction import kmeans_abstraction_based_on_contribution
import _pickle as cPickle
from core.utils.propagation import propagation_net
from core.utils.network2rlv import network2rlv
from core.nnet.read_nnet import network_from_nnet_file
from core.abstraction.naive import abstract_network
from core.abstraction.alg2 import heuristic_abstract_alg2
# from core.abstraction.clustering_abstract import heuristic_abstract_clustering
from core.utils.marabou_query_utils import reduce_property_to_basic_form, get_query
from core.refinement.refine import refine, global_refine

from inputs.delta_004 import TEST_PROPERTY_ACAS

def generate_results_filename(
        nnet_filename, property_id, mechanism, refinement_type,
        abstraction_type, refinement_sequence, abstraction_sequence
):
    return "__".join(["experiment",
                      "NN_{}".format(nnet_filename),
                      "PID_{}".format(property_id),
                      "M_{}".format(mechanism),
                      "R_{}".format(refinement_type),
                      "A_{}".format(abstraction_type),
                      "RS_{}".format(refinement_sequence),
                      "AS_{}".format(abstraction_sequence),
                      "DATETIME_{}".format(consts.cur_time_str)
                      ])

# @timeout_decorator.timeout(72000)
def one_experiment(
        nnet_filename, refinement_type, abstraction_type, mechanism,
        refinement_sequence, abstraction_sequence, results_directory,
        property_id=consts.PROPERTY_ID, verbose=consts.VERBOSE
):
    test_property = TEST_PROPERTY_ACAS.get(property_id, None)
    dynamically_import_marabou(query_type=test_property["type"])

    fullname = "./networks/ACASXu_{}.nnet".format(nnet_filename)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    results_filename = generate_results_filename(nnet_filename=nnet_filename,
                                                 property_id=property_id,
                                                 mechanism=mechanism,
                                                 refinement_type=refinement_type,
                                                 abstraction_type=abstraction_type,
                                                 refinement_sequence=refinement_sequence,
                                                 abstraction_sequence=abstraction_sequence)

    # for i in range(len(test_property["output"])):
    #     test_property["output"][i][1]["Lower"] = lower_bound
    # net  = network_from_nnet_file(fullname)

    net = network_from_nnet_file(fullname)

    print(f"size={len(net.layers)}")
    net, test_property = reduce_property_to_basic_form(network=net, test_property=test_property)
    # network2rlv(net, test_property, "test.rlv")
    counter_example = []
    sat = False

    # if mechanism is vanilla marabou
    if mechanism == "marabou":
        print("query using vanilla Marabou")
        print("net.get_general_net_data()")
        print(net.get_general_net_data())

        t0 = time.time()
        vars1, stats1, query_result = get_query(
            network=net,
            test_property=test_property,
            verbose=consts.VERBOSE
        )
        t1 = time.time()
        # time to check property on net with marabou
        marabou_time = t1 - t0
        if verbose:
            print(f"query time = {marabou_time}")
        # if vars1:
        #     o_net = copy.deepcopy(net)
        #     net_output = o_net.speedy_evaluate(vars1)
        #     print(net_output[0])

        res = [
            ("net_name", nnet_filename),
            ("property_id", property_id),
            ("query_result", query_result),
            ("orig_query_time", marabou_time),
            ("net_data", json.dumps(net.get_general_net_data())),
        ]
        # generate dataframe from result
        df = pd.DataFrame.from_dict({x[0]: [x[1]] for x in res})
        df.to_json(os.path.join(results_directory, "df_" + results_filename))
        with open(os.path.join(results_directory, results_filename), "w") as fw:
            fw.write("-" * 80)
            fw.write("parameters:")
            fw.write("-" * 80)
            fw.write("\n")
            # for arg in vars(args):
            #     fw.write("{}: {}\n".format(arg, getattr(args, arg)))
            fw.write("+" * 80)
            fw.write("results:")
            fw.write("+" * 80)
            fw.write("\n")
            for (k, v) in res:
                fw.write("{}: {}\n".format(k, v))
        return res

    # otherwise mechanism is marabou_with_ar
    orig_net = copy.deepcopy(net)
    print("query using Marabou with AR")
    t2 = time.time()
    # do_process_before(net,property_id)
    random_input = [1]
    if abstraction_type == "global":
        net_before, random_input, processed_net, sat, actions = \
            global_abstraction_based_on_contribution(network=net, test_property=test_property)
        # print(net_before)
        net_before_p = cPickle.loads(cPickle.dumps(net_before, -1))
        # print('net_after-propagation')
        propagation_net(net_before)
        net = net_before
        # print(net_before)
    elif abstraction_type == "kmeans":
        net_before, random_input, processed_net, sat, actions = \
            kmeans_abstraction_based_on_contribution(network=net, test_property=test_property)
        # print(net_before)
        net_before_p = cPickle.loads(cPickle.dumps(net_before, -1))
        # print('net_after-propagation')
        propagation_net(net_before)
        net = net_before
    elif abstraction_type == "complete":
        net, processed_net = abstract_network(net)
    elif abstraction_type == "heuristic_alg2":
        net, sat = heuristic_abstract_alg2(
            network=net,
            test_property=test_property,
            sequence_length=abstraction_sequence
        )
    else:
        raise NotImplementedError("unknown abstraction")

    # print(net)

    cegar_flag = False
    if (not sat) and random_input:
        cegar_flag = True
        abstraction_time = time.time() - t2
        num_of_refine_steps = 0
        ar_times = []
        ar_sizes = []
        refine_sequence_times = []
        spurious_examples = []
        while True:  # CEGAR / CETAR method
            t4 = time.time()

            # print("net.get_general_net_data()")
            print(net.get_general_net_data())

            vars1, stats1, query_result = get_query(
                network=net, test_property=test_property,
                verbose=consts.VERBOSE
            )
            debug_print(f'query_result={query_result}')
            t5 = time.time()
            ar_times.append(t5 - t4)
            ar_sizes.append(net.get_general_net_data()["num_nodes"])
            # if verbose:
            print("query time after A and {} R steps is {}".format(num_of_refine_steps, t5 - t4))
            debug_print(net.get_general_net_data())
            if query_result == "UNSAT":
                # if always y'<3.99 then also always y<3.99
                if verbose:
                    print("UNSAT (finish)")
                break
            if query_result == "SAT":
                if verbose:
                    print("SAT (have to check example on original net)")
                    print(vars1)
                # print(vars1)
                # debug_print(f"vars1 = {vars1}')
                # st = time.time()
                # orig_net_output = orig_net.evaluate(vars1)
                # print("evaluate: {}".format(time.time() - st))
                # st = time.time()
                orig_net_output = orig_net.speedy_evaluate(vars1)
                cur_net_output = net.speedy_evaluate(vars1)
                before_output = net_before_p.speedy_evaluate(vars1)
                print("current_output_value")
                print(cur_net_output)
                print(before_output)
                if cur_net_output < 0.0001 and not actions and not net.deleted_name2node:
                    query_result = "UNSAT"
                    print("false nagative")
                    break
                # print(f"orig_net_output={orig_net_output}")
                # print(f"orig_net.name2node_map={orig_net.name2node_map}")
                # print("speedy evaluate: {}".format(time.time() - st))
                nodes2variables, variables2nodes = orig_net.get_variables()
                # we got y'>3.99, check if also y'>3.99 for the same input
                if is_satisfying_assignment(network=orig_net,
                                            test_property=test_property,
                                            output=orig_net_output,
                                            variables2nodes=variables2nodes):

                    if verbose:
                        print("property holds also in orig - SAT (finish)")
                    counter_example = vars1
                    break  # also counter example for orig_net
                else:
                    spurious_examples.append(vars1)
                    t_cur_refine_start = time.time()
                    if verbose:
                        print("property doesn't holds in orig - spurious example")
                    num_of_refine_steps += 1
                    if verbose:
                        print("refine step #{}".format(num_of_refine_steps))
                    # refine until all spurious examples are satisfied
                    # since all spurious examples are satisfied in the original
                    # network, the loop stops until net will be fully refined
                    # print(vars1)
                    example = vars1
                    if abstraction_type != "heuristic_alg2":
                        ori_var2val = processed_net.evaluate(example)
                    refinement_sequences_counter = 0
                    while True:
                        refinement_sequences_counter += 1
                        # print(f"refinement_sequences_counter={refinement_sequences_counter}")
                        if refinement_type == "cegar":
                            debug_print("cegar")
                            net = refine(network=net,
                                         sequence_length=refinement_sequence,
                                         example=vars1)
                        # else:
                        #     debug_print("weight_based")
                        #     net = refine(network=net, sequence_length=refinement_sequence)
                        elif refinement_type == "global":
                            debug_print("global")
                            net = global_refine(network=net_before_p, processed_net=processed_net,
                                                ori_var2val=ori_var2val, actions=actions, example=vars1)

                        # after refining, check if the current spurious example is
                        # already not a counter example (i.e. not satisfied in the
                        # refined network). stop if not satisfied, continue if yes
                        net_output = net.speedy_evaluate(vars1)
                        # print(f"net_output={net_output}")
                        # print(f"net.name2node_map={net.name2node_map}")
                        nodes2variables, variables2nodes = net.get_variables()
                        if not is_satisfying_assignment(
                                network=net,
                                test_property=test_property,
                                output=net_output,
                                variables2nodes=variables2nodes):
                            net_before_p = cPickle.loads(cPickle.dumps(net, -1))
                            propagation_net(net)
                            break
                    # print(net)
                    t_cur_refine_end = time.time()
                    refine_sequence_times.append(t_cur_refine_end - t_cur_refine_start)
    elif sat:
        query_result = "SAT"
        abstraction_time = 0
        ar_times = [0.0]
        ar_sizes = [net.get_general_net_data()["num_nodes"]]
        refine_sequence_times = []
        num_of_refine_steps = 0
    else:
        query_result = "UNSAT"
        abstraction_time = 0
        ar_times = [0.0]
        ar_sizes = [net.get_general_net_data()["num_nodes"]]
        refine_sequence_times = []
        num_of_refine_steps = 0
    t3 = time.time()

    # time to check property on net with marabou using CEGAR
    total_ar_time = t3 - t2
    if verbose:
        print("ar query time = {}".format(total_ar_time))

    # time to check property on the last network in CEGAR
    last_net_ar_time = t3 - t4 if cegar_flag else t3
    if verbose:
        print("last ar net query time = {}".format(last_net_ar_time))

    res = [
        ("net_name", nnet_filename),
        ("property_id", property_id),
        ("abstraction_time", abstraction_time),
        ("query_result", query_result),
        ("num_of_refine_steps", num_of_refine_steps),
        ("total_ar_query_time", total_ar_time),
        ("ar_times", json.dumps(ar_times)),
        ("ar_sizes", json.dumps(ar_sizes)),
        ("refine_sequence_times", json.dumps(refine_sequence_times)),
        ("last_net_data", json.dumps(net.get_general_net_data())),
        ("counter-example", counter_example)
        # ("last_query_time", last_net_ar_time)
    ]
    # generate dataframe from result
    df = pd.DataFrame.from_dict({x[0]: [x[1]] for x in res})
    df.to_json(os.path.join(results_directory, "df_" + results_filename))
    # write result to output file
    with open(os.path.join(results_directory, results_filename), "w") as fw:
        fw.write("-" * 80)
        fw.write("parameters:")
        fw.write("-" * 80)
        fw.write("\n")
        # for arg in vars(args):
        #     fw.write("{}: {}\n".format(arg, getattr(args, arg)))
        fw.write("+" * 80)
        fw.write("results:")
        fw.write("+" * 80)
        fw.write("\n")
        for (k, v) in res:
            fw.write("{}: {}\n".format(k, v))
    return res


if __name__ == '__main__':
    one_exp_res = one_experiment(
        nnet_filename="1_2",
        property_id="adversarial_3",
        mechanism="marabou",
        refinement_type="global",
        abstraction_type="global",
        results_directory="./results-tmp/",
        refinement_sequence="",
        abstraction_sequence=""
    )
    debug_print(one_exp_res)