"""
analyze one experiment - print and plot interesting details and graphs
"""

import os
import json
import pandas as pd
import argparse
from experiments.consts import (
    BEST_CEGARABOU_METHODS, EXP_PARAMS, map_to_formal_names, map_cat_to_formal_name,
    USED_RS_1, USED_RS_2, USED_AS_1, USED_AS_2, BEST_RS, BEST_AS,
    BEST_CEGARABOU_METHODS, EXP_PARAMS
)
from core.configuration import consts


def read_marabou_df(exp_result_dirname):
    marabou_df_all = "/home/yizhak/Research/Code/CEGAR_NN/AR_results/marabou/{}/outfile_df2json/df_all"
    df_path = marabou_df_all.format(exp_result_dirname)
    print(f"marabou_df_path={df_path}")
    marabou_df = pd.read_json(
        # path_or_buf="/home/yizhak/Research/Code/CEGAR_NN/AR_results/marabou/{}/outfile_df2json/df_all".format(exp_result_dirname))
        # medium no preprocess
        path_or_buf=df_path)
    # long no preprocess
    # path_or_buf="/home/yizhak/Research/Code/CEGAR_NN/AR_results/marabou/2019-10-22/outfile_df2json/df_all".format(exp_result_dirname))
    # marabou_df = pd.read_json(
    #     path_or_buf="/tmp/union_exp_dfs/marabou/df_all")
    return marabou_df


def read_cegarabou_df(
        exp_result_dirname, best_cegarabou_methods={}, exp_params=EXP_PARAMS
):
    cegarabo_df_results_dir = "/home/yizhak/Research/Code/CEGAR_NN/" \
                              "AR_results/cegarabou/{}/outfile_df2json/df_all"
    print(f"df_path={cegarabo_df_results_dir.format(exp_result_dirname)}")
    cegarabou_df = pd.read_json(path_or_buf=cegarabo_df_results_dir.format(exp_result_dirname))

    best_cegarabou_methods.update(exp_params)
    for k, v in best_cegarabou_methods.items():
        if type(v) == list:
            dfs = [cegarabou_df[cegarabou_df.filenames.str.contains("__{}_{}_".format(k, v1))] for v1 in v]
            cegarabou_df = pd.concat(dfs)
        else:
            cegarabou_df = cegarabou_df[cegarabou_df.filenames.str.contains("__{}_{}_".format(k, v))]

    return cegarabou_df


def read_experiment_df(
        exp_result_dirname, exp_method="cegarabou",
        best_cegarabou_methods={}, exp_params={}
) -> pd.DataFrame:
    results_dir = "/home/yizhak/Research/Code/CEGAR_NN/AR_results/"
    df_results_path = os.path.join(results_dir, exp_method, exp_result_dirname, "outfile_df2json/df_all")
    print(f"df_results_path={df_results_path}")
    df_results = pd.read_json(path_or_buf=df_results_path)

    if exp_method == "cegarabou":
        best_cegarabou_methods.update(exp_params)
        for k, v in best_cegarabou_methods.items():
            if type(v) == list:
                dfs = [df_results[df_results.filenames.str.contains("__{}_{}_".format(k, v1))] for v1 in v]
                df_results = pd.concat(dfs)
            else:
                df_results = df_results[df_results.filenames.str.contains("__{}_{}_".format(k, v))]

    return df_results


def read_dfs(exp_result_dirname):
    marabou_df = read_marabou_df(exp_result_dirname=exp_result_dirname)
    cegarabou_df = read_cegarabou_df(exp_result_dirname=exp_result_dirname)
    return marabou_df, cegarabou_df


def print_num_of_refine_sequences(df):
    print(f"num_of_refine_steps mean={df.num_of_refine_steps.mean()}")
    print(f"num_of_refine_steps median={df.num_of_refine_steps.median()}")


def print_abstraction_time(df):
    print(f"abstraction_time mean={df.abstraction_time.mean()}")
    print(f"abstraction_time median={df.abstraction_time.median()}")
    print(f"abstraction_percent mean={(df.abstraction_time / df.total_ar_query_time).mean()}")
    print(f"abstraction_percent median={(df.abstraction_time / df.total_ar_query_time).median()}")


def print_refinement_time(df):
    # load json into list
    df.refine_sequence_times = \
        df.refine_sequence_times.apply(lambda l: json.loads(l))
    # extract sum of all refinement steps
    df["sum_refine_sequence_times"] = \
        df.refine_sequence_times.apply(lambda l: sum(l))

    print(f"sum_refine_sequence_times.mean="
          f"{(df.sum_refine_sequence_times * df.num_of_refine_steps / df.num_of_refine_steps.sum()).mean()}")
    print(f"sum_refine_sequence_times.median="
          f"{(df.sum_refine_sequence_times * df.num_of_refine_steps / df.num_of_refine_steps.sum()).median()}")
    print(f"sum_refine_sequence_times_percent mean={(df.sum_refine_sequence_times / df.total_ar_query_time).mean()}")
    print(
        f"sum_refine_sequence_times_percent median={(df.sum_refine_sequence_times / df.total_ar_query_time).median()}")


def print_sum_ar_times(df):
    pass


def analyze(exp_result_dir):
    # exp_result_dir = "2020-06_11_marabou_vs_cegarabou_alg2_wrt_acasxu_properties"
    cegarabou_df = read_cegarabou_df(exp_result_dirname=exp_result_dir)
    print_abstraction_time(df=cegarabou_df)
    print_num_of_refine_sequences(df=cegarabou_df)
    print_refinement_time(df=cegarabou_df)
    print_sum_ar_times()


def parse_args():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--results_directory",
                        dest="results_directory",
                        default=consts.results_directory)
    args = parser.parse_args()
    return args.results_directory


if __name__ == '__main__':
    results_directory = parse_args()
    analyze(exp_result_dir=results_directory)
