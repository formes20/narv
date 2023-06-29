import math
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from experiments.collect.analyze import read_experiment_df
from experiments.collect.analyze import \
    print_abstraction_time, print_refinement_time, print_num_of_refine_sequences

TIMEOUT_VAL = 5 * 3600
FONT = {
    # 'family': 'serif',
    'color': 'black',
    # 'weight': 'bold',
    'size': 20,
}


def row2exp_name(row):
    """
    :param row: row in experiments result pandas dataframe
    :return: name of experiment for the result that is represented @row
    e.g 4_4-adversarial_13
    """
    net_name = row[1]['net name']
    net_name = net_name.partition('ACASXU_run2a_')[-1]
    net_name = net_name.rpartition('_batch_2000.nnet')[0]
    exp_name = f"{net_name}-{row[1]['property_id']}"
    print(f"exp_name={exp_name}")
    return exp_name


def main(exp_dirname):
    # # extract marabou_before_pp results, generates dict from net name to run time
    # marabou_json_file_path = f"/home/yizhak/Research/Code/CEGAR_NN/AR_results/" \
    #     f"marabou/{exp_dirname}/outfile_df2json/df_all"
    # print(f"read {marabou_json_file_path} df")
    # df_results_marabou_before_pp = pd.read_json(marabou_json_file_path)
    df_results_marabou_before_pp = read_experiment_df(
        exp_method="marabou", exp_result_dirname=exp_dirname
    )
    # df_results_marabou_before_pp = read_experiment_df(
    #     exp_method="marabou", exp_result_dirname="2020-06_11_marabou_vs_cegarabou_alg2_wrt_adversarial_properties"
    # )

    marabou_exp_name2time = {}
    for row in df_results_marabou_before_pp.iterrows():
        marabou_exp_name2time[row2exp_name(row)] = float(row[1]["orig_query_time"])
    d_marabou_before_pp = marabou_exp_name2time
    print(f'len(d_marabou_before_pp) = {len(d_marabou_before_pp)}')

    # extract best cegarabou results
    df_results_cegarabou = read_experiment_df(exp_method="cegarabou", exp_result_dirname=exp_dirname)

    # verify that all experiments are with best configuration:
    # (abstraction=complete, refinement=cegar, a_steps=100, r_steps=50)
    df_results_cegarabou_complete = df_results_cegarabou[df_results_cegarabou.filenames.str.contains("_A_complete_")]
    df_results_cegarabou_complete_cegar = df_results_cegarabou[df_results_cegarabou.filenames.str.contains("_R_cegar_")]
    df_results_cegarabou_complete_cegar_as100 = df_results_cegarabou[
        df_results_cegarabou.filenames.str.contains("_AS_100_")]
    df_results_cegarabou_complete_cegar_as100_rs50 = df_results_cegarabou[
        df_results_cegarabou.filenames.str.contains("_RS_50_")]
    # assert df_results_cegara  1bou_complete_cegar_as100_rs50.shape[0] == df_results_cegarabou.shape[0]

    df_results_cegarabou = df_results_cegarabou[
        ["net_name", "total_ar_query_time", "property_id", "ar_times"]]
    print(f'df_results_cegarabou.shape = {df_results_cegarabou.shape} ')

    print_num_of_refine_sequences(df_results_cegarabou_complete_cegar_as100_rs50)
    print_abstraction_time(df_results_cegarabou_complete_cegar_as100_rs50)
    print_refinement_time(df_results_cegarabou_complete_cegar_as100_rs50)

    # print the average size (number of nodes) in the last net used during the query
    for method, df, last_net_data_col_name in [
        ("marabou", df_results_marabou_before_pp, "net_data"),
        ("cegarabou", df_results_cegarabou_complete_cegar_as100_rs50, "last_net_data")
    ]:
        df.num_nodes = df[last_net_data_col_name].apply(
            lambda x: int(json.loads(x)["num_nodes"])
        )
        print(f"using method {method}:", end="\t")
        print(f"avg_last_net_size={df.num_nodes.mean()}")
        print(f"median_last_net_size={df.num_nodes.median()}")

    # generate dict from short name to sum ar times
    cegarabou_sum_ar_times_exp_name2time = {}
    for row in df_results_cegarabou.iterrows():
        cegarabou_sum_ar_times_exp_name2time[row2exp_name(row)] = \
            sum(float(x) for x in row[1]["ar_times"][1:-1].split(", "))
    d_cegarabou_sum_ar_times = cegarabou_sum_ar_times_exp_name2time

    # generate dict from short name to last query time
    cegarabou_last_query_time_exp_name2time = {}
    for row in df_results_cegarabou.iterrows():
        cegarabou_last_query_time_exp_name2time[row2exp_name(row)] = \
            float(row[1]["ar_times"][1:-1].split(", ")[-1])
    d_cegarabou_last_query_time = cegarabou_last_query_time_exp_name2time

    # generate dict from short name to total AR query time
    d_cegarabou_total_ar_time_exp_name2time = {}
    for row in df_results_cegarabou.iterrows():
        d_cegarabou_total_ar_time_exp_name2time[row2exp_name(row)] = \
            row[1]["total_ar_query_time"]
    d_cegarabou_total_ar_time = d_cegarabou_total_ar_time_exp_name2time

    # plot graphs of cegarabou (last time/sum times) vs marabou
    for title, d_cegarabou in [
        # ("Last Query Time", d_cegarabou_last_query_time),
        # ("Sum Query Times", d_cegarabou_sum_ar_times),
        ("Total AR Time", d_cegarabou_total_ar_time)
    ]:
        print("-" * 80)
        print(title)
        print("-" * 80)

        # cegarabo_vs_marabou_before_pp
        print("-" * 40)
        print("cegarabo_vs_marabou_before_pp")
        print("-" * 40)
        cegarabo_vs_marabou_before_pp = []
        marabou_finished_cegarabou_not = 0
        cegarabou_finished_marabou_not = 0
        for k, v in d_cegarabou.items():
            if k in d_marabou_before_pp.keys():
                item = (v, d_marabou_before_pp[k])
            else:
                cegarabou_finished_marabou_not += 1
                item = (v, TIMEOUT_VAL)
            cegarabo_vs_marabou_before_pp.append(item)
        for k, v in d_marabou_before_pp.items():
            if k in d_cegarabou.keys():
                continue  # the item is already in the list
            else:
                marabou_finished_cegarabou_not += 1
                item = (TIMEOUT_VAL, v)
                cegarabo_vs_marabou_before_pp.append(item)
        print(f'marabou_finished_cegarabou_not={marabou_finished_cegarabou_not}')
        print(f'cegarabou_finished_marabou_not={cegarabou_finished_marabou_not}')

        finishes = [(x, y) for (x, y) in cegarabo_vs_marabou_before_pp if x != TIMEOUT_VAL and y != TIMEOUT_VAL]
        timeouts = [(x, y) for (x, y) in cegarabo_vs_marabou_before_pp if x == TIMEOUT_VAL or y == TIMEOUT_VAL]

        cegarabou_avg = np.mean([x for (x, y) in cegarabo_vs_marabou_before_pp])
        marabou_avg = np.mean([y for (x, y) in cegarabo_vs_marabou_before_pp])
        cegarabou_median = np.median([x for (x, y) in cegarabo_vs_marabou_before_pp])
        marabou_median = np.median([y for (x, y) in cegarabo_vs_marabou_before_pp])
        print(f"cegarabou_avg={cegarabou_avg}")
        print(f"marabou_avg={marabou_avg}")
        print(f"cegarabou_median={cegarabou_median}")
        print(f"marabou_median={marabou_median}")

        NUM_OF_ALL_TIMEOUTS = 900 - len(finishes) - len(timeouts)  # NOTE: update to relevant number
        print(f'#timeouts={NUM_OF_ALL_TIMEOUTS}')
        all_timeouts = [(TIMEOUT_VAL, TIMEOUT_VAL) for j in range(NUM_OF_ALL_TIMEOUTS)]
        x_finishes = [x for (x, y) in finishes]
        y_finishes = [y for (x, y) in finishes]
        x_timeouts = [x for (x, y) in timeouts]
        y_timeouts = [y for (x, y) in timeouts]
        x_all_timeouts = [x for (x, y) in all_timeouts]
        y_all_timeouts = [y for (x, y) in all_timeouts]

        finished_points = plt.scatter(s=100, x=x_finishes, y=y_finishes, color="b", marker="o", clip_on=False)
        timeout_points = plt.scatter(s=100, x=x_timeouts, y=y_timeouts, color="r", marker="x", clip_on=False)
        all_timeout_points = plt.scatter(s=100, x=x_all_timeouts, y=y_all_timeouts, color="r", marker="x",
                                         clip_on=False)
        # plt.scatter(x=cegarabou_results, y=marabou_before_pp_results)
        # timeouts lines
        vertical_timeout_line = plt.axvline(x=TIMEOUT_VAL, ymin=0, ymax=TIMEOUT_VAL, color='g', linestyle='--')
        helper_line = plt.axvline(x=TIMEOUT_VAL, ymin=0, ymax=TIMEOUT_VAL, color='k', linestyle='-')
        helper_line2 = plt.axvline(x=TIMEOUT_VAL, ymin=0, ymax=TIMEOUT_VAL, color='k', linestyle='--')
        horizontal_timeout_line = plt.axhline(y=TIMEOUT_VAL, xmin=0, xmax=TIMEOUT_VAL, color='y', linestyle='--')
        helper_line3 = plt.axhline(y=TIMEOUT_VAL, xmin=0, xmax=TIMEOUT_VAL, color='k', linestyle='-')
        # y=x line
        min_finished = math.floor(min(min(x_finishes), min(y_finishes)))
        max_finished = math.ceil(max(max(x_finishes), max(y_finishes)))
        min_timeouts = math.floor(min(min(x_timeouts), min(y_timeouts)))
        max_timeouts = math.ceil(max(max(x_timeouts), max(y_timeouts)))
        min_val = 0.01  # min(min_finished, min_timeouts, 0)
        max_val = TIMEOUT_VAL  # max(max_finished, max_timeouts, TIMEOUT_VAL)
        y_equals_x_line = plt.plot(list(range(int(min_val), max_val)), 'g--', label="y=x")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim((min_val, max_val))
        plt.ylim((min_val, max_val))
        plt.xlabel(title + ":\nMarabou with Abstraction", fontdict=FONT)
        plt.ylabel(title + ":\nMarabou", fontdict=FONT)
        # plt.legend(
        #     (y_equals_x_line, finished_points),
        #     ("y=x", "finished experiment"),
        #     scatterpoints=1,
        #     ncol=1,
        #     fontsize=20
        # )
        plt.legend(
            (vertical_timeout_line, y_equals_x_line, timeout_points, finished_points),
            ("y=x", "y=x", "timeout experiment", "finished experiment"),
            scatterpoints=1,
            ncol=1,
            fontsize=20
        )
        plt.show()


def parse_args():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--exp_dirname", dest="exp_dirname",
        default="2020-06_12_marabou_vs_cegarabou_naive_wrt_adversarial_properties"
    )
    args = parser.parse_args()
    return args.exp_dirname


if __name__ == '__main__':
    # exp_dirname = "2020-06_10_marabou_vs_cegarabou_alg2_wrt_basic_properties"
    # exp_dirname = "2020-06_11_marabou_vs_cegarabou_alg2_wrt_acasxu_properties"
    # exp_dirname = "2020-06_11_marabou_vs_cegarabou_naive_wrt_acasxu_properties"
    # exp_dirname = "2020-06_11_marabou_vs_cegarabou_alg2_wrt_adversarial_properties"
    # exp_dirname = "2020-06_12_marabou_vs_cegarabou_naive_wrt_adversarial_properties"
    # exp_dirname = "2020-05-15_support_acasxu_1-2-3-4"
    # exp_dirname = "2020-05-11_CAV_2020_Fig8"
    # exp_dirname = "2020-05-10_CAV_2020_Fig9"
    exp_dirname = parse_args()
    main(exp_dirname=exp_dirname)
