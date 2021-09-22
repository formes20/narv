
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math
from itertools import combinations

from core.configuration import consts
from experiments.collect.analyze import (
    print_abstraction_time, print_refinement_time, read_marabou_df,
    print_num_of_refine_sequences, read_cegarabou_df, read_dfs
)
from experiments.consts import (
    map_to_formal_names, map_cat_to_formal_name,
    USED_RS_1, USED_RS_2, USED_AS_1, USED_AS_2,
    BEST_CEGARABOU_METHODS, EXP_PARAMS, FONT
)

# long
# exp_result_dirname = "2020-05-07_CAV_2020_Fig10"
# total_num_experiments = 2400
# exp_result_dirname = "2020-06_08_best_abstraction_wrt_adversarial_properties"
# total_num_experiments = 2700 # 45 nets, 20 acasxu properties, 3 abstraction methods
# exp_result_dirname = "2020-06_07_best_abstraction_wrt_acasxu_properties"
total_num_experiments = 540 # 45 nets, 4 acasxu properties, 3 abstraction methods
# exp_result_dirname = "naive_vs_alg2_wrt_acasxu"
# total_num_experiments = 360 # sum of experimetns
# exp_result_dirname = "2020-06_07_best_abstraction_wrt_basic_properties"
# total_num_experiments = 270 # 45 nets, 2 basic properties, 3 abstraction methods
# exp_result_dirname = "2020-05-11_CAV_2020_Fig7"
# total_num_experiments = 1440
# exp_result_dirname = "2020-05-06_CAV_2020_Fig9"
# total_num_experiments = 90
# exp_result_dirname = "2019-10-06"
# total_num_experiments = 1440
# exp_result_dirname = "2019-10-11"
# meduim
# exp_result_dirname = "2019-10-23"

# ----- </constants in use> -----


def get_last_vs_orig_df(marabou_df, cegarabou_df, on_col="net name"):
    last_vs_orig_df = pd.merge(marabou_df[[on_col, "orig_query_time"]],
                               cegarabou_df[[on_col, "last_query_time"]],
                               on=on_col)
    return last_vs_orig_df


def show_df_col_vs_col(df, col_x="orig_query_time", col_y="last_query_time", color="b",
                       min_x_limit=None, max_x_limit=None, min_y_limit=None, max_y_limit=None,
                       xscale="log", yscale="log", title="", set_timeouts_line=False):
    # df.plot.scatter(x=col_x, y=col_y, c=color, title=title)

    # if any(l is not None for l in [min_y_limit, max_y_limit, min_y_limit, max_y_limit]):
    #     min_x_limit = min(df[col_x]) if min_x_limit is None else min_x_limit
    #     max_x_limit = max(df[col_x]) if max_x_limit is None else max_x_limit
    #     min_y_limit = min(df[col_y]) if min_y_limit is None else min_y_limit
    #     max_y_limit = max(df[col_y]) if max_y_limit is None else max_y_limit
    #     f, ax = plt.subplots(figsize=(6, 6))
    #     ax.scatter(x=df[col_x], y=df[col_y], c=color)
    #     ax.set(xlim=(min_x_limit, max_x_limit), ylim=(min_y_limit, max_y_limit), title=title)

    min_x_limit = np.nanmin(df[col_x]) if min_x_limit is None else min_x_limit
    max_x_limit = np.nanmax(df[col_x]) if max_x_limit is None else max_x_limit
    min_y_limit = np.nanmin(df[col_y]) if min_y_limit is None else min_y_limit
    max_y_limit = np.nanmax(df[col_y]) if max_y_limit is None else max_y_limit

    max_val = max(max_x_limit, max_y_limit)
    timeout_val = 1.1 * max_val
    df[col_x] = df[col_x].fillna(timeout_val)
    df[col_y] = df[col_y].fillna(timeout_val)

    df_finished = df[(df[col_x] != timeout_val) & (df[col_y] != timeout_val)]
    df_timeout = df[(df[col_x] == timeout_val) | (df[col_y] == timeout_val)]
    assert df.shape[0] == df_timeout.shape[0] + df_finished.shape[0]
    # add to df_timeout all experiments that were not finished in any method
    none_series = pd.Series([None for _ in range(total_num_experiments)])
    df_not_finished_experiments = pd.DataFrame.from_dict({
        col_name: none_series for col_name in df.columns
    })
    df_not_finished_experiments[col_x] = pd.Series([timeout_val] * total_num_experiments)
    df_not_finished_experiments[col_y] = pd.Series([timeout_val] * total_num_experiments)
    df_timeout = pd.concat([df_timeout, df_not_finished_experiments], ignore_index=True)

    df_extract_sizes = pd.concat([df_timeout, df_finished], ignore_index=True)
    point_sizes = df_extract_sizes.groupby([col_x, col_y]).size()
    # point_sizes[1] = 10**4
    # for p in point_sizes:
    #     print(p)
    # mean = point_sizes.mean()
    # wanted_mean = mean  # consts.EXP_POINT_MEAN_SIZE
    # finished_sizes = np.array([x + (wanted_mean - mean) for x in point_sizes])
    # std = finished_sizes.std()
    # variance = std ** 2
    # point_sizes = [(math.e**(((x-mean)**2) / (2*variance)))/(2*std*(2*math.pi)**0.5) for x in point_sizes]
    # point_sizes = [20*4**(float(i)-min(point_sizes))/(max(point_sizes)-min(point_sizes)) for i in point_sizes]
    point_sizes = [20*4**x for x in list(np.array(point_sizes) / np.linalg.norm(point_sizes))]

    # finished_sizes = df_finished.groupby([col_x, col_y]).size()
    # mean = finished_sizes.mean()
    # wanted_mean = mean  # consts.EXP_POINT_MEAN_SIZE
    # finished_sizes = np.array([x + (wanted_mean - mean) for x in finished_sizes])
    # std = finished_sizes.std()
    # variance = std ** 2
    # finished_sizes = [(math.e**(((x-mean)**2) / (2*variance)))/(2*std*(2*math.pi)**0.5) for x in finished_sizes]
    # norm_sizes = [(float(i)-min(sizes))/(max(sizes)-min(sizes)) for i in sizes]
    # finished_sizes = list(np.array(finished_sizes) / np.linalg.norm(finished_sizes) * 100)
    # finished_points = plt.scatter(x=df_finished[col_x], y=df_finished[col_y], c=color, s=point_sizes, clip_on=False)
    finished_points = plt.scatter(x=df_finished[col_x], y=df_finished[col_y], c=color, s=100, clip_on=False)

    # timeout_sizes = df_timeout.groupby([col_x, col_y]).size()
    # mean = timeout_sizes.mean()
    # wanted_mean = mean  # consts.EXP_POINT_MEAN_SIZE
    # timeout_sizes = np.array([x + (wanted_mean - mean) for x in timeout_sizes])
    # std = timeout_sizes.std()
    # variance = std ** 2
    # timeout_sizes = [(math.e**(((x-mean)**2) / (2*variance)))/(2*std*(2*math.pi)**0.5)  for x in timeout_sizes]
    # timeout_sizes = list(np.array(timeout_sizes) / np.linalg.norm(timeout_sizes) * 100)
    # timeout_points = plt.scatter(x=df_timeout[col_x], y=df_timeout[col_y], c='r', s=point_sizes, marker="x", clip_on=False)
    timeout_points = plt.scatter(x=df_timeout[col_x], y=df_timeout[col_y], c='r', s=100, marker="x", clip_on=False)

    # add y=x line
    max_val = math.ceil(max(df[col_x].max(), df[col_y].max(), timeout_val))
    min_val = math.floor(min(df[col_x].min(), df[col_y].min(), 0))
    if set_timeouts_line:
        # add timeouts
        vertical_timeout_line = plt.axvline(x=timeout_val, ymin=min_val, ymax=timeout_val, color='y', linestyle='-')
        horizontal_timeout_line = plt.axhline(y=timeout_val, xmin=min_val, xmax=timeout_val, color='y', linestyle='-')
        # vertical_timeout_line = plt.plot((timeout_val, timeout_val), (0, timeout_val), 'c--', label="timeout line")
        # horizontal_timeout_line = plt.plot((0, timeout_val), (timeout_val, timeout_val), 'c--', label="timeout line")
    else:
        helper_line = plt.axvline(x=timeout_val, ymin=0, ymax=timeout_val, color='g', linestyle='--')
    # y_equals_x_line = plt.scatter(x=list(range(int(min_val), int(max_val))), y=list(range(int(min_val), int(max_val))), c='g', marker="_", label="y=x")
    y_equals_x_line = plt.plot(list(range(int(min_val), int(max_val))), 'g--', label="y=x")

    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)

    plt.xlim((min_val, max_val))
    plt.ylim((min_val, max_val))

    # add legend map
    if set_timeouts_line:
        objs = (vertical_timeout_line, y_equals_x_line, timeout_points, finished_points)
        names = ("timeout line", "y=x", "timeout experiment", "finished experiment")
    else:
        objs = (helper_line, y_equals_x_line, timeout_points, finished_points)
        names = ("y=x", "y=x", "timeout experiment", "finished experiment")
    plt.legend(
        objs,
        names,
        scatterpoints=1,
        ncol=1,
        fontsize=20
    )


    # f, ax = plt.subplots(figsize=(6, 6))
    # ax.scatter(x=df[col_x], y=df[col_y], c=color)
    # ax.set(xlim=(min_x_limit, timeout_val), ylim=(min_y_limit, timeout_val), title=title)
    #
    # # set diagonal line on y=x
    # def on_change(axes):
    #     # When this function is called it checks the current
    #     # values of xlim and ylim and modifies diag_line
    #     # accordingly.
    #     x_lims = ax.get_xlim()
    #     y_lims = ax.get_ylim()
    #     diag_line.set_data(x_lims, y_lims)
    #
    # # Connect two callbacks to your axis instance.
    # # These will call the function "on_change" whenever
    # # xlim or ylim is changed.
    # ax.callbacks.connect('xlim_changed', on_change)
    # ax.callbacks.connect('ylim_changed', on_change)
    #
    # lim = max(ax.get_xlim(), ax.get_ylim())
    # diag_line, = ax.plot(lim, lim, ls="--", c=".3")
    # diag_line.set_label("y=x")
    # ax.legend()

    # set x,y labels
    plt.xlabel(col_x, fontdict=FONT)
    plt.ylabel(col_y, fontdict=FONT)
    # plt.grid()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()


def show_orig_vs_last_time():
    marabou_df, cegarabou_df = read_dfs()
    df = get_last_vs_orig_df(marabou_df, cegarabou_df, on_col="net name")
    col_x = "orig_query_time"
    col_y = "last_query_time"
    show_df_col_vs_col(df, col_x=col_x, col_y=col_y, color="b",
                       min_x_limit=0, max_x_limit=200, min_y_limit=0, max_y_limit=200,
                       xscale="log", yscale="log", title="scatter {} vs {}".format(col_x, col_y))


def show_complete_vs_heuristic_last_time(exp_result_dirname):
    cegarabou_df = read_cegarabou_df(exp_result_dirname)
    complete_df = cegarabou_df[cegarabou_df.filenames.str.contains("__A_complete_")]
    heuristic_df = cegarabou_df[cegarabou_df.filenames.str.contains("__A_heuristic_")]
    print(complete_df.shape)
    print(heuristic_df.shape)
    print(complete_df.shape[0] + heuristic_df.shape[0] == cegarabou_df.shape[0])

    replace_func = lambda filename: filename.replace("__A_complete_", "__A_heuristic_")
    complete_df["replace_filenames"] = complete_df.filenames.apply(replace_func)
    complete_heuristic_df = complete_df.merge(right=heuristic_df,
                                              left_on="replace_filenames",
                                              right_on="filenames",
                                              suffixes=["_complete", "_heuristic"])
    col_x = "last_query_time_complete"
    col_y = "last_query_time_heuristic"

    show_df_col_vs_col(complete_heuristic_df, col_x=col_x, col_y=col_y, color="b",
                       min_x_limit=0, max_x_limit=1000, min_y_limit=0, max_y_limit=1000,
                       xscale="log", yscale="log", title="scatter {} vs {}".format(col_x, col_y))


def show_col_vs_col_wrt_category(df, left_col, right_col, category,
                                 cat_aggregation_func=lambda x: x,
                                 aggregated_name=None,
                                 xscale=None,
                                 yscale=None,
                                 min_x_limit=None,
                                 max_x_limit=None,
                                 min_y_limit=None,
                                 max_y_limit=None,
                                 title="",
                                 set_timeouts_line=False):
    if aggregated_name is None:
        aggregated_name = "Aggregated {}".format(category.title())
    df[aggregated_name] = df[category].apply(cat_aggregation_func)
    left_df = df[df.filenames.str.contains(left_col)]
    right_df = df[df.filenames.str.contains(right_col)]
    # assert (left_df.shape[0] + right_df.shape[0] == df.shape[0])

    # remove datetime from filename to enable merge according to the experiment's parameters
    for cur_df in [left_df, right_df]:
        cur_df.filenames = cur_df.filenames.apply(lambda f: f.rsplit("DATETIME")[0])

    replace_func = lambda filename: filename.replace(left_col, right_col)
    left_df["replace_filenames"] = left_df.filenames.apply(replace_func)

    left_col_formal_name = left_col.replace(left_col, map_to_formal_names.get(left_col, left_col))
    right_col_formal_name = right_col.replace(right_col, map_to_formal_names.get(right_col, right_col))
    left_right_df = left_df.merge(right=right_df,
                                  left_on="replace_filenames",
                                  right_on="filenames",
                                  suffixes=[":\n{}".format(left_col_formal_name),
                                            ":\n{}".format(right_col_formal_name)],
                                  how="outer")

    col_x = "{}:\n{}".format(aggregated_name, left_col_formal_name)
    col_y = "{}:\n{}".format(aggregated_name, right_col_formal_name)

    show_df_col_vs_col(left_right_df, col_x=col_x, col_y=col_y, color="b",
                       xscale=xscale, yscale=yscale,
                       min_x_limit=min_x_limit, max_x_limit=max_x_limit,
                       min_y_limit=min_y_limit, max_y_limit=max_y_limit,
                       title=title if title else "scatter {} vs {}".format(col_x, col_y),
                       set_timeouts_line=set_timeouts_line)


def show_cegar_vs_cetar_wrt_last_time(exp_result_dirname):
    cegarabou_df = read_cegarabou_df(exp_result_dirname)
    print(cegarabou_df.columns)

    cegar_df = cegarabou_df[cegarabou_df.filenames.str.contains("__R_cegar__")]
    cetar_df = cegarabou_df[cegarabou_df.filenames.str.contains("__R_cetar__")]
    print(cegar_df.shape)
    print(cetar_df.shape)
    assert(cegar_df.shape[0] + cetar_df.shape[0] == cegarabou_df.shape[0])

    replace_func = lambda filename: filename.replace("__R_cegar__", "__R_cetar__")
    cegar_df["replace_filenames"] = cegar_df.filenames.apply(replace_func)
    cegar_cetar_df = cegar_df.merge(right=cetar_df,
                                    left_on="replace_filenames",
                                    right_on="filenames",
                                    suffixes=["_cegar", "_cetar"])
    col_x = "last_query_time_cegar"
    col_y = "last_query_time_cetar"

    show_df_col_vs_col(cegar_cetar_df, col_x=col_x, col_y=col_y, color="b",
                       min_x_limit=0, max_x_limit=100, min_y_limit=0, max_y_limit=75,
                       title="scatter {} vs {}".format(col_x, col_y), set_timeouts_line=True)


def show_cegar_vs_cetar_wrt_num_of_queries(exp_result_dirname, xscale=None, yscale=None):
    cegarabou_methods = copy.deepcopy(BEST_CEGARABOU_METHODS)
    if "R" in cegarabou_methods:
        cegarabou_methods.pop("R")
    cegarabou_df = read_cegarabou_df(exp_result_dirname=exp_result_dirname,
                                     best_cegarabou_methods=cegarabou_methods,
                                     exp_params=EXP_PARAMS)
    refinement_sequence_length = 100
    cegarabou_df["num_of_refine_steps"] = cegarabou_df["num_of_refine_steps"].apply(lambda x:
                                                                                    refinement_sequence_length * x)
    show_col_vs_col_wrt_category(df=cegarabou_df,
                                 left_col="_R_cegar_",
                                 right_col="_R_cetar_",
                                 category="num_of_refine_steps",
                                 aggregated_name="#Refinement Steps",
                                 xscale=xscale,
                                 yscale=yscale,
                                 min_x_limit=0,
                                 max_x_limit=None,
                                 min_y_limit=0,
                                 max_y_limit=None)


def get_sum_times_func():
    """
    :return: function that sum list of times that were stored as a list in cell in dataframe
    e.g func get "[1,2,3]", loads it into [1,2,3] and sum it into 6
    the function is used in the metthod "apply()" of pandas sequence
    """
    return lambda query_times: sum(json.loads(query_times))


def show_as_vs_as_wrt_sum_of_query_times(exp_result_dirname, as_1=USED_AS_1, as_2=USED_AS_2, xscale=None, yscale=None):
    cegarabou_methods = copy.deepcopy(BEST_CEGARABOU_METHODS)
    if "AS" in cegarabou_methods:
        cegarabou_methods.pop("AS")
    cegarabou_df = read_cegarabou_df(exp_result_dirname=exp_result_dirname, best_cegarabou_methods=cegarabou_methods)
    refinement_sequence_length = 100
    cegarabou_df["num_of_refine_steps"] = cegarabou_df["num_of_refine_steps"].apply(lambda x:
                                                                                    refinement_sequence_length * x)
    show_col_vs_col_wrt_category(df=cegarabou_df,
                                 left_col="_AS_{}_".format(as_1),
                                 right_col="_AS_{}_".format(as_2),
                                 category="ar_times",
                                 cat_aggregation_func=get_sum_times_func(),
                                 aggregated_name="Sum of Query Times",
                                 xscale=xscale,
                                 yscale=yscale,
                                 min_x_limit=0,
                                 max_x_limit=None,
                                 min_y_limit=0,
                                 max_y_limit=None)


def show_rs_vs_rs_wrt_sum_of_query_times(exp_result_dirname, rs_1=USED_RS_1, rs_2=USED_RS_2, xscale=None, yscale=None):
    cegarabou_methods = copy.deepcopy(BEST_CEGARABOU_METHODS)
    if "RS" in cegarabou_methods:
        cegarabou_methods.pop("RS")
    cegarabou_df = read_cegarabou_df(exp_result_dirname=exp_result_dirname, best_cegarabou_methods=cegarabou_methods)
    refinement_sequence_length = 100
    cegarabou_df["num_of_refine_steps"] = cegarabou_df["num_of_refine_steps"].apply(lambda x:
                                                                                    refinement_sequence_length * x)
    show_col_vs_col_wrt_category(df=cegarabou_df,
                                 left_col="_RS_{}_".format(rs_1),
                                 right_col="_RS_{}_".format(rs_2),
                                 category="ar_times",
                                 cat_aggregation_func=get_sum_times_func(),
                                 aggregated_name="Sum of Query Times",
                                 xscale=xscale,
                                 yscale=yscale,
                                 min_x_limit=0,
                                 max_x_limit=None,
                                 min_y_limit=0,
                                 max_y_limit=None)


def show_as_vs_as_wrt_num_of_queries(exp_result_dirname, as_1=USED_AS_1, as_2=USED_AS_2, xscale=None, yscale=None):
    cegarabou_methods = copy.deepcopy(BEST_CEGARABOU_METHODS)
    if "AS" in cegarabou_methods:
        cegarabou_methods.pop("AS")
    cegarabou_df = read_cegarabou_df(exp_result_dirname=exp_result_dirname, best_cegarabou_methods=cegarabou_methods)
    refinement_sequence_length = 100
    cegarabou_df["num_of_refine_steps"] = cegarabou_df["num_of_refine_steps"].apply(lambda x:
                                                                                    refinement_sequence_length * x)
    show_col_vs_col_wrt_category(df=cegarabou_df,
                                 left_col="_AS_{}_".format(as_1),
                                 right_col="_AS_{}_".format(as_2),
                                 category="num_of_refine_steps",
                                 aggregated_name="#Refinement Steps",
                                 xscale=xscale,
                                 yscale=yscale,
                                 min_x_limit=0,
                                 max_x_limit=None,
                                 min_y_limit=0,
                                 max_y_limit=None)


def show_rs_vs_rs_wrt_num_of_queries(exp_result_dirname, rs_1=USED_RS_1, rs_2=USED_RS_2, xscale=None, yscale=None):
    cegarabou_methods = copy.deepcopy(BEST_CEGARABOU_METHODS)
    if "RS" in cegarabou_methods:
        cegarabou_methods.pop("RS")
    cegarabou_df = read_cegarabou_df(exp_result_dirname=exp_result_dirname, best_cegarabou_methods=cegarabou_methods)
    refinement_sequence_length = 100
    cegarabou_df["num_of_refine_steps"] = cegarabou_df["num_of_refine_steps"].apply(lambda x:
                                                                                    refinement_sequence_length * x)
    show_col_vs_col_wrt_category(df=cegarabou_df,
                                 left_col="_RS_{}_".format(rs_1),
                                 right_col="_RS_{}_".format(rs_2),
                                 category="num_of_refine_steps",
                                 aggregated_name="#Refinement Steps",
                                 xscale=xscale,
                                 yscale=yscale,
                                 min_x_limit=0,
                                 max_x_limit=None,
                                 min_y_limit=0,
                                 max_y_limit=None)


def show_cegar_vs_cetar_wrt_sum_of_query_times(exp_result_dirname, xscale=None, yscale=None):
    cegarabou_methods = copy.deepcopy(BEST_CEGARABOU_METHODS)
    if "R" in cegarabou_methods:
        cegarabou_methods.pop("R")
    cegarabou_df = read_cegarabou_df(exp_result_dirname=exp_result_dirname, best_cegarabou_methods=cegarabou_methods,
                                     exp_params=EXP_PARAMS)
    refinement_sequence_length = 100
    cegarabou_df["num_of_refine_steps"] = cegarabou_df["num_of_refine_steps"].apply(lambda x:
                                                                                    refinement_sequence_length * x)
    show_col_vs_col_wrt_category(df=cegarabou_df,
                                 left_col="_R_cegar_",
                                 right_col="_R_cetar_",
                                 category="ar_times",
                                 cat_aggregation_func=get_sum_times_func(),
                                 aggregated_name="Sum of Query Times",
                                 xscale=xscale,
                                 yscale=yscale,
                                 min_x_limit=0,
                                 max_x_limit=None,
                                 min_y_limit=0,
                                 max_y_limit=None)


def show_complete_vs_heuristic_wrt_num_of_queries(exp_result_dirname, xscale=None, yscale=None):
    cegarabou_methods = copy.deepcopy(BEST_CEGARABOU_METHODS)
    if "A" in cegarabou_methods:
        cegarabou_methods.pop("A")
    cegarabou_df = read_cegarabou_df(exp_result_dirname=exp_result_dirname, best_cegarabou_methods=cegarabou_methods)
    cegarabou_df["refinement_sequence_length"] = cegarabou_df.filenames.apply(lambda s: int(s.split("__RS_")[1].split("__AS_")[0]))
    cegarabou_df["num_of_refine_steps"] = cegarabou_df["num_of_refine_steps"] * cegarabou_df["refinement_sequence_length"]
    show_col_vs_col_wrt_category(df=cegarabou_df,
                                 left_col="_A_complete_",
                                 right_col="_A_heuristic_",
                                 category="num_of_refine_steps",
                                 aggregated_name="#Refinement Steps",
                                 xscale=xscale,
                                 yscale=yscale,
                                 min_x_limit=0,
                                 max_x_limit=None,
                                 min_y_limit=0,
                                 max_y_limit=None)




def show_abstraction1_vs_abstraction2_wrt_sum_of_query_times(
        exp_result_dirname,
        abstraction_1="_A_complete_",
        abstraction_2="_A_heuristic_alg2_",
        xscale=None,
        yscale=None
):
    cegarabou_methods = copy.deepcopy(BEST_CEGARABOU_METHODS)
    if "A" in cegarabou_methods:
        cegarabou_methods.pop("A")
    cegarabou_df = read_cegarabou_df(exp_result_dirname=exp_result_dirname, best_cegarabou_methods=cegarabou_methods)
    cegarabou_df["refinement_sequence_length"] = cegarabou_df.filenames.apply(
        lambda s: int(s.split("__RS_")[1].split("__AS_")[0]))
    print_num_of_refine_sequences(df=cegarabou_df)
    cegarabou_df["num_of_refine_steps"] = cegarabou_df["num_of_refine_steps"] * cegarabou_df[
        "refinement_sequence_length"]
    print(f"cegarabou_df.columns={cegarabou_df.columns}")
    print_abstraction_time(df=cegarabou_df)
    print_refinement_time(df=cegarabou_df)
    show_col_vs_col_wrt_category(df=cegarabou_df,
                                 left_col=abstraction_1,
                                 right_col=abstraction_2,
                                 category="ar_times",
                                 cat_aggregation_func=get_sum_times_func(),
                                 aggregated_name="Sum of Query Times",
                                 xscale=xscale,
                                 yscale=yscale,
                                 min_x_limit=0,
                                 max_x_limit=None,
                                 min_y_limit=0,
                                 max_y_limit=None)


def show_marabou_vs_cegarabou_wrt_last_query_time():
    cat = "last_query_time"
    cat_aggregation_func = lambda x: x
    # show_marabou_vs_cegarabou_wrt_cat(cat,
    #                                   cat_aggregation_func,
    #                                   exp_params={"L": 100},
    #                                   best_cegarabou_methods={"R": "cegar", "A": "heuristic", "RS": 100, "AS": 100})
    show_marabou_vs_cegarabou_wrt_cat(cat, cat_aggregation_func)


def show_marabou_vs_cegarabou_wrt_sum_ar_times():
    cat = "ar_times"
    cat_aggregation_func = get_sum_times_func()
    show_marabou_vs_cegarabou_wrt_cat(cat, cat_aggregation_func)


def show_marabou_vs_cegarabou_wrt_cat(exp_result_dirname,
                                      cat,
                                      cat_aggregation_func,
                                      exp_params=EXP_PARAMS,
                                      best_cegarabou_methods=BEST_CEGARABOU_METHODS,
                                      color="b",
                                      set_timeouts_line=True,
                                      xscale=None,
                                      yscale=None,
                                      min_x_limit=None,
                                      min_y_limit=None,
                                      max_x_limit=None,
                                      max_y_limit=None):
    marabou_df, cegarabou_df = read_dfs(exp_result_dirname=exp_result_dirname)
    # experiment__F_ACASXU_run2a_5_9_batch_2000.nnet__T_False__L_500__P_True__COMMIT_None__DATETIME_2019-10-13_02:54:02
    # marabou_df["L"] = marabou_df.filenames.apply(lambda f: int(f.split("__L_")[-1].split("__P_")[0]))
    # cegarabou_df["L"] = cegarabou_df.filenames.apply(lambda f: int(f.split("__L_")[-1].split("__P_")[0]))

    # filter only experiments with relevant parameters
    for k, v in exp_params.items():
        if type(v) == list:
            dfs = [marabou_df[marabou_df.filenames.str.contains("__{}_{}_".format(k, v1))] for v1 in v]
            marabou_df = pd.concat(dfs)
        else:
            marabou_df = marabou_df[marabou_df.filenames.str.contains("__{}_{}_".format(k, v))]
    best_cegarabou_methods.update(exp_params)
    for k,v in best_cegarabou_methods.items():
        if type(v) == list:
            dfs = [cegarabou_df[cegarabou_df.filenames.str.contains("__{}_{}_".format(k, v1))] for v1 in v]
            cegarabou_df = pd.concat(dfs)
        else:
            cegarabou_df = cegarabou_df[cegarabou_df.filenames.str.contains("__{}_{}_".format(k, v))]
    if cegarabou_df.shape[0] == 0:
        print ("cegarabou_df.shape[0] == 0, check params")
        assert False
    if marabou_df.shape[0] <= 0:
        print ("marabou_df.shape[0] == 0, check params")
        assert False
    if cegarabou_df.shape[0] != marabou_df.shape[0]:
        print("notice that there are timeouts in at least one method:")
        print("#cegarabou finished experiments".format(cegarabou_df.shape[0]))
        print("#marabou finished experiments".format(marabou_df.shape[0]))

    print("#experiments: {}".format(marabou_df.shape[0]))
    print(marabou_df.columns)
    print(cegarabou_df.columns)
    cegarabou_df[cat] = cegarabou_df[cat].apply(cat_aggregation_func)
    df = marabou_df.merge(
        right=cegarabou_df,
        left_on="net name",
        right_on="net name",
        suffixes=["_marabou", "_cegarabou"],
        how="outer")
    print(df.columns)
    col_x = 'Marabou'
    col_y = "{}: Marabou with Abstraction".format(map_cat_to_formal_name[cat])
    df.rename(columns={'orig_query_time': col_x, cat: col_y}, inplace=True)
    print(df.columns)

    min_x_limit = np.nanmin(df[col_x]) if min_x_limit is None else min_x_limit
    max_x_limit = np.nanmax(df[col_x]) if max_x_limit is None else max_x_limit
    min_y_limit = np.nanmin(df[col_y]) if min_y_limit is None else min_y_limit
    max_y_limit = np.nanmax(df[col_y]) if max_y_limit is None else max_y_limit

    max_val = max(max_x_limit, max_y_limit)
    timeout_val = 1.1 * max_val
    df[col_x] = df[col_x].fillna(timeout_val)
    df[col_y] = df[col_y].fillna(timeout_val)

    df_finished = df[(df[col_x] != timeout_val) & (df[col_y] != timeout_val)]
    df_timeout = df[(df[col_x] == timeout_val) | (df[col_y] == timeout_val)]
    assert df.shape[0] == df_timeout.shape[0] + df_finished.shape[0]

    finished_sizes = df_finished.groupby([col_x, col_y]).size()
    # norm_sizes = [(float(i)-min(sizes))/(max(sizes)-min(sizes)) for i in sizes]
    finished_sizes = list(np.array(finished_sizes) / np.linalg.norm(finished_sizes) * 100)
    finished_points = plt.scatter(x=df_finished[col_x],
                                  y=df_finished[col_y],
                                  c=color,
                                  s=finished_sizes,
                                  marker="o")

    timeout_sizes = df_timeout.groupby([col_x, col_y]).size()
    timeout_sizes = list(np.array(timeout_sizes) / np.linalg.norm(timeout_sizes) * 300)
    timeout_points = plt.scatter(x=df_timeout[col_x],
                                 y=df_timeout[col_y],
                                 c='r',
                                 s=timeout_sizes,
                                 marker="x")

    if set_timeouts_line:
        # add timeouts
        vertical_timeout_line = plt.axvline(x=timeout_val, ymin=0, ymax=timeout_val, color='c', linestyle='-')
        horizontal_timeout_line = plt.axhline(y=timeout_val, xmin=0, xmax=timeout_val, color='c', linestyle='-')
        # vertical_timeout_line = plt.plot((timeout_val, timeout_val), (0, timeout_val), 'c--', label="timeout line")
        # horizontal_timeout_line = plt.plot((0, timeout_val), (timeout_val, timeout_val), 'c--', label="timeout line")

    # add y=x line
    max_val = math.ceil(max(df[col_x].max(), df[col_y].max(), timeout_val))
    min_val = math.floor(min(df[col_x].min(), df[col_y].min(), 0))
    y_equals_x_line = plt.plot(list(range(int(min_val), int(max_val))), 'g--', label="y=x")

    # plt.scatter(x=df[col_x], y=df[col_y], c="b", s=df.groupby([col_x, col_y]).size())
    if xscale is not None:
        plt.xscale("log")
    if yscale is not None:
        plt.yscale("log")
    plt.xlabel(col_x, fontdict=FONT)
    plt.ylabel(col_y, fontdict=FONT)

    plt.legend(
        (vertical_timeout_line, y_equals_x_line, timeout_points, finished_points),
        ("timeout line", "y=x", "timeout experiment", "finished experiment"),
        scatterpoints=1,
        ncol=1,
        fontsize=20
    )

    # plt.grid()
    plt.show()


def parse_args():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--results_directory",
                        dest="exp_results_dir",
                        default=consts.results_directory)
    args = parser.parse_args()
    return args.exp_results_dir


def main():
    # cegarabou vs marabou
    # show_orig_vs_last_time()
    # show_marabou_vs_cegarabou_wrt_last_query_time()
    # show_marabou_vs_cegarabou_wrt_sum_ar_times()

    # not in use
    # show_cegar_vs_cetar_last_time()
    # show_complete_vs_heuristic_last_time()

    exp_result_dirname = parse_args()
    print(f"main() - exp_result_dirname={exp_result_dirname}")
    # cegarabou parameters
    # show_complete_vs_heuristic_wrt_num_of_queries(exp_result_dirname)
    abstractions = ["_A_complete_", "_A_heuristic_alg2_"]  # , "_A_heuristic_random_"]
    for abstraction_1, abstraction_2 in combinations(abstractions,2):
        show_abstraction1_vs_abstraction2_wrt_sum_of_query_times(
            exp_result_dirname=exp_result_dirname,
            abstraction_1=abstraction_1,
            abstraction_2=abstraction_2,
            xscale="symlog",
            yscale="symlog"
        )
    # show_cegar_vs_cetar_wrt_num_of_queries(exp_result_dirname)
    # show_cegar_vs_cetar_wrt_sum_of_query_times(exp_result_dirname, xscale="symlog", yscale="symlog")
    # show_as_vs_as_wrt_num_of_queries(exp_result_dirname)
    # show_as_vs_as_wrt_sum_of_query_times(exp_result_dirname, xscale="symlog", yscale="symlog")
    # show_rs_vs_rs_wrt_num_of_queries(exp_result_dirname)
    # show_rs_vs_rs_wrt_sum_of_query_times(exp_result_dirname, xscale="symlog", yscale="symlog")


if __name__ == '__main__':
    main()
