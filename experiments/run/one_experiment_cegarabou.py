"""
run one experiment - query cegarabou engine:
calculate if property (p1 or p2) is sat/unsat in a net which is represented by a given .nnet formatted file
write to result file the result of the query and data on the calculation process - times, sizes, etc. .

usage: python3 -f <nnet_filename> -a <abstraction_type> -r <test_refinement type> -e <epsilon value> -l <lower_bound> -o? -p?
example of usage: python3 -f ACASXU_run2a_1_8_batch_2000.nnet -a heuristic -r cegar -e 1e-5 -l 25000 -o -p -s 100
"""

# external imports
import os
import json
import copy
import time
import argparse
import pandas as pd

# internal imports
from core.pre_process.pre_process import preprocess
from core.abstraction.naive import abstract_network
from core.abstraction.alg2 import heuristic_abstract
from core.nnet.read_nnet import network_from_nnet_file
from core.utils.verification_properties_utils import (
    get_test_property_acas, get_test_property_tiny, is_satisfying_assignment
)
from core.configuration import consts
from core.utils.parse_utils import str2bool
from core.utils.debug_utils import debug_print
from core.utils.marabou_query_utils import get_query
from core.refinement.refine import refine


def generate_results_filename(nnet_filename, is_tiny, refinement_type,
                              abstraction_type, epsilon, lower_bound,
                              property_id, preprocess_orig_net,
                              refinement_sequence_length,
                              abstraction_sequence_length):
    # get commit number from git
    commit = None
    # the rest of the name is the parameters
    return "__".join(["experiment",
                      "F_{}".format(nnet_filename),
                      "T_{}".format(is_tiny),
                      "R_{}".format(refinement_type),
                      "A_{}".format(abstraction_type),
                      "E_{}".format(epsilon),
                      "L_{}".format(lower_bound),
                      "P_{}".format(preprocess_orig_net),
                      "RS_{}".format(refinement_sequence_length),
                      "AS_{}".format(abstraction_sequence_length),
                      "PID_{}".format(property_id),
                      "COMMIT_{}".format(commit),
                      "DATETIME_{}".format(consts.cur_time_str)
                      ])


def one_experiment(nnet_filename, is_tiny, refinement_type, abstraction_type,
                   epsilon, lower_bound, preprocess_orig_net,
                   refinement_sequence_length, abstraction_sequence_length,
                   results_directory, property_id=consts.PROPERTY_ID,
                   verbose=consts.VERBOSE):
    if verbose:
        debug_print("one_experiment_cegarabou({})".format(json.dumps([nnet_filename, is_tiny, refinement_type,
                                                                      abstraction_type, epsilon, lower_bound,
                                                                      property_id, preprocess_orig_net])))

    if is_tiny:
        example_nets_dir_path = consts.PATH_TO_MARABOU_ACAS_EXAMPLES
    else:
        example_nets_dir_path = consts.PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    fullname = os.path.join(example_nets_dir_path, nnet_filename)

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    results_filename = generate_results_filename(nnet_filename=nnet_filename,
                                                 is_tiny=is_tiny,
                                                 refinement_type=refinement_type,
                                                 abstraction_type=abstraction_type,
                                                 epsilon=epsilon,
                                                 lower_bound=lower_bound,
                                                 property_id=property_id,
                                                 preprocess_orig_net=preprocess_orig_net,
                                                 abstraction_sequence_length=abstraction_sequence_length,
                                                 refinement_sequence_length=refinement_sequence_length)
    test_property = get_test_property_tiny() if is_tiny else get_test_property_acas(property_id)
    # for i in range(len(test_property["output"])):
    #     test_property["output"][i][1]["Lower"] = lower_bound
    net = network_from_nnet_file(fullname)
    print(f"size={len(net.layers)}")

    orig_net = copy.deepcopy(net)
    if args.preprocess_orig_net:
        preprocess(orig_net)

    if verbose:
        print("query using AR")
    t2 = time.time()
    if abstraction_type == "complete":
        print("complete")
        net = abstract_network(net)
    else:
        print("heuristic")
        net = heuristic_abstract(network=net, test_property=test_property,
                                 sequence_length=abstraction_sequence_length)
    abstraction_time = time.time() - t2
    num_of_refine_steps = 0
    ar_times = []
    ar_sizes = []
    refine_sequence_times = []
    while True:  # CEGAR / CETAR method
        t4 = time.time()
        vars1, stats1, query_result = get_query(
            network=net, test_property=test_property,
            verbose=consts.VERBOSE
        )
        debug_print(f'query_result={query_result}')
        t5 = time.time()
        ar_times.append(t5 - t4)
        ar_sizes.append(net.get_general_net_data()["num_nodes"])
        if verbose:
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
            debug_print(f'vars1={vars1}')
            st = time.time()
            orig_net_output = orig_net.evaluate(vars1)
            print("evaluate: {}".format(time.time() - st))
            st = time.time()
            orig_net.speedy_evaluate(vars1)
            print("speedy evaluate: {}".format(time.time() - st))
            nodes2variables, variables2nodes = orig_net.get_variables()
            # we got y'>3.99, check if also y'>3.99 for the same input
            if is_satisfying_assignment(network=orig_net,
                                        test_property=test_property,
                                        output=orig_net_output,
                                        variables2nodes=variables2nodes):
                if verbose:
                    print("property holds also in orig - SAT (finish)")
                break  # also counter example for orig_net
            else:
                t_cur_refine_start = time.time()
                if verbose:
                    print("property doesn't holds in orig - spurious example")
                num_of_refine_steps += 1
                if verbose:
                    print("refine step #{}".format(num_of_refine_steps))
                if refinement_type == "cegar":
                    debug_print("cegar")
                    net = refine(network=net,
                                 sequence_length=refinement_sequence_length,
                                 example=vars1)
                else:
                    debug_print("cetar")
                    net = refine(network=net,
                                 sequence_length=refinement_sequence_length)
                t_cur_refine_end = time.time()
                refine_sequence_times.append(t_cur_refine_end - t_cur_refine_start)

    t3 = time.time()

    # time to check property on net with marabou using CEGAR
    total_ar_time = t3 - t2
    if verbose:
        print("ar query time = {}".format(total_ar_time))

    # time to check property on the last network in CEGAR
    last_net_ar_time = t3 - t4
    if verbose:
        print("last ar net query time = {}".format(last_net_ar_time))

    res = [
        ("net name", nnet_filename),
        ("property_id", property_id),
        ("abstraction_time", abstraction_time),
        ("query_result", query_result),
        ("num_of_refine_steps", num_of_refine_steps),
        ("total_ar_query_time", total_ar_time),
        ("ar_times", json.dumps(ar_times)),
        ("ar_sizes", json.dumps(ar_sizes)),
        ("refine_sequence_times", json.dumps(refine_sequence_times)),
        ("last_net_data", json.dumps(net.get_general_net_data())),
        ("last_query_time", last_net_ar_time)
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
        for arg in vars(args):
            fw.write("{}: {}\n".format(arg, getattr(args, arg)))
        fw.write("+" * 80)
        fw.write("results:")
        fw.write("+" * 80)
        fw.write("\n")
        for (k, v) in res:
            fw.write("{}: {}\n".format(k, v))
    return res


def parse_args():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-nn", "--net_number",
                        dest="net_number",
                        default="4_3")
    parser.add_argument("-t", "--is_tiny",
                        dest="is_tiny",
                        default=False,
                        action="store_true")
    abstraction_type_default = "complete" if consts.COMPLETE_ABSTRACTION \
        else "heuristic"
    parser.add_argument("-a", "--abstraction_type",
                        dest="abstraction_type",
                        default=abstraction_type_default,
                        choices=["complete", "heuristic"])
    refinement_type_default = "cegar" if consts.DO_CEGAR else "cetar"
    parser.add_argument("-r", "--refinement_type",
                        dest="refinement_type",
                        default=refinement_type_default,
                        choices=["cegar", "cetar"])
    parser.add_argument("-p", "--preprocess_orig_net",
                        dest="preprocess_orig_net",
                        default=consts.COMPARE_TO_PREPROCESSED_NET,
                        type=str2bool)
    parser.add_argument("-pid", "--property_id",
                        dest="property_id",
                        default=consts.PROPERTY_ID,
                        type=str)
    parser.add_argument("-e", "--epsilon",
                        dest="epsilon",
                        default=consts.EPSILON,
                        type=float)
    parser.add_argument("-l", "--lower_bound",
                        dest="lower_bound",
                        default=25000,
                        type=int)
    parser.add_argument("-as", "--abstraction_sequence_length",
                        dest="abstraction_sequence_length",
                        default=100,
                        type=int)
    parser.add_argument("-rs", "--refinement_sequence_length",
                        dest="refinement_sequence_length",
                        default=100,
                        type=int)
    parser.add_argument("-d", "--results_directory",
                        dest="results_directory",
                        default=consts.results_directory)
    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        default=consts.VERBOSE,
                        action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # run experiment
    nnet_general_filename = "ACASXU_run2a_1_1_tiny_{}.nnet" if args.is_tiny \
        else "ACASXU_run2a_{}_batch_2000.nnet"
    one_exp_res = one_experiment(
        nnet_filename=nnet_general_filename.format(args.net_number),
        property_id=args.property_id,
        is_tiny=args.is_tiny,
        refinement_type=args.refinement_type,
        abstraction_type=args.abstraction_type,
        epsilon=args.epsilon,
        lower_bound=args.lower_bound,
        preprocess_orig_net=args.preprocess_orig_net,
        abstraction_sequence_length=args.abstraction_sequence_length,
        refinement_sequence_length=args.refinement_sequence_length,
        results_directory=args.results_directory)
    debug_print(one_exp_res)
