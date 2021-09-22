#/usr/bin/python3

"""
run one experiment - query marabou engine:
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
from core.nnet.read_nnet import network_from_nnet_file
from core.utils.verification_properties_utils import (
    get_test_property_acas, get_test_property_tiny, get_winner_and_runner_up
)
from core.utils.debug_utils import debug_print
from core.utils.parse_utils import str2bool
from core.utils.marabou_query_utils import get_query
from core.pre_process.pre_process import preprocess
from core.configuration import consts
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Layer import Layer


def generate_results_filename(nnet_filename, is_tiny, lower_bound,
                              preprocess_orig_net, property_id,
                              is_adversarial_property):
    # get commit number from git
    commit = None
    # the rest of the name is the parameters
    return "__".join(["experiment",
                      "F_{}".format(nnet_filename),
                      "T_{}".format(is_tiny),
                      "L_{}".format(lower_bound),
                      "P_{}".format(preprocess_orig_net),
                      "PID_{}".format(property_id),
                      "ADV_{}".format(is_adversarial_property),
                      "COMMIT_{}".format(commit),
                      "DATETIME_{}___TMP_SUFFIX".format(consts.cur_time_str)
                      ])


def one_experiment(nnet_filename, is_tiny, lower_bound, preprocess_orig_net,
                   results_directory, property_id=consts.PROPERTY_ID,
                   is_adversarial_property=False, verbose=consts.VERBOSE):
    if verbose:
        debug_print("one_experiment_marabou({})".format(
            json.dumps([nnet_filename, is_tiny,  lower_bound,
                        preprocess_orig_net, property_id, results_directory])))
    if is_tiny:
        example_nets_dir_path = consts.PATH_TO_MARABOU_ACAS_EXAMPLES
    else:
        example_nets_dir_path = consts.PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    fullname = os.path.join(example_nets_dir_path, nnet_filename)

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    results_filename = generate_results_filename(
        nnet_filename=nnet_filename, is_tiny=is_tiny, lower_bound=lower_bound,
        preprocess_orig_net=preprocess_orig_net, property_id=property_id,
        is_adversarial_property=is_adversarial_property
    )
    test_property = get_test_property_tiny() if is_tiny else get_test_property_acas(property_id)
    # for i in range(len(test_property["output"])):
    #     test_property["output"][i][1]["Lower"] = lower_bound
    net = network_from_nnet_file(fullname)

    orig_net = copy.deepcopy(net)
    if args.preprocess_orig_net:
        preprocess(orig_net)

    # query original net
    if verbose:
        print("query orig_net")
    t0 = time.time()
    if verbose:
        debug_print("orig_net.get_general_net_data(): {}".format(orig_net.get_general_net_data()))
    vars1, stats1, query_result = get_query(
        network=orig_net,
        test_property=test_property,
        is_adversarial_property=consts.IS_ADVERSARIAL,
        verbose=consts.VERBOSE
    )
    t1 = time.time()
    # time to check property on net with marabou
    marabou_time = t1 - t0
    if verbose:
        print("orig_net query time ={}".format(marabou_time))

    res = [
        ("net name", nnet_filename),
        ("property_id", property_id),
        ("query_result", query_result),
        ("orig_query_time", marabou_time),
        ("net_data", json.dumps(orig_net.get_general_net_data())),
    ]
    # generate dataframe from result
    df = pd.DataFrame.from_dict({x[0]: [x[1]] for x in res})
    df.to_json(os.path.join(results_directory, "df_" + results_filename))
    with open(os.path.join(results_directory, results_filename), "w") as fw:
        fw.write("-"*80)
        fw.write("parameters:")
        fw.write("-"*80)
        fw.write("\n")
        for arg in vars(args):
            fw.write("{}: {}\n".format(arg, getattr(args, arg)))
        fw.write("+"*80)
        fw.write("results:")
        fw.write("+"*80)
        fw.write("\n")
        for (k,v) in res:
            fw.write("{}: {}\n".format(k,v))
    return res


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-nn", "--net_number", dest="net_number", default="1_1")
    parser.add_argument("-t", "--is_tiny", dest="is_tiny", default=False, action="store_true")
    parser.add_argument("-p", "--preprocess_orig_net", dest="preprocess_orig_net",
                        default=consts.COMPARE_TO_PREPROCESSED_NET, type=str2bool)
    parser.add_argument("-pid", "--property_id", dest="property_id",
                        default=consts.PROPERTY_ID, type=str)
    parser.add_argument("-adv", "--is_adversarial_property",
                        dest="is_adversarial_property",
                        default=consts.IS_ADVERSARIAL, type=bool)
    parser.add_argument("-l", "--lower_bound", dest="lower_bound", default=25000, type=int)
    parser.add_argument("-d", "--results_directory", dest="results_directory", default=consts.results_directory)
    parser.add_argument("-v", "--verbose", dest="verbose", default=consts.VERBOSE, action="store_true")
    args = parser.parse_args()

    # run experiment
    nnet_general_filename = "ACASXU_run2a_1_1_tiny_{}.nnet" if args.is_tiny else "ACASXU_run2a_{}_batch_2000.nnet"
    res_one_exp = one_experiment(
        nnet_filename=nnet_general_filename.format(args.net_number),
        is_tiny=args.is_tiny,
        property_id=args.property_id,
        preprocess_orig_net=args.preprocess_orig_net,
        lower_bound=args.lower_bound,
        results_directory=args.results_directory,
        is_adversarial_property=args.is_adversarial_property,
        verbose=args.verbose
    )
    debug_print(res_one_exp)


