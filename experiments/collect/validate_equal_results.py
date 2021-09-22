
import argparse
import pandas as pd
from core.configuration import consts
from experiments.collect.analyze import read_cegarabou_df, read_marabou_df


def validate_equal_results_marabou_cegarabou(exp_name="2020-01-24"):
    df_marabou = read_marabou_df(exp_result_dirname=exp_name)
    df_cegarabou = read_cegarabou_df(exp_result_dirname=exp_name)
    verify_equal_query_results(marabou_df=df_marabou, cegarabou_df=df_cegarabou)

    # # cegarabou df
    # df1 = pd.read_json(f"/home/yizhak/Research/Code/CEGAR_NN/AR_results/cegarabou/{exp_name}/outfile_df2json/df_all")
    # df1 = df1[['net name', 'property_id', 'query_result']]
    #
    # # marabou df
    # df2 = pd.read_json(f"/home/yizhak/Research/Code/CEGAR_NN/AR_results/marabou/{exp_name}/outfile_df2json/df_all")
    # df2 = df2[['net name', 'property_id', 'query_result']]
    #
    # # list of name,pid,result of df1 to be used as keys in df2
    # df1_name_pid_res = []
    # for row in df1.iterrows():
    #     # print(row[1]["net_name"])
    #     df1_name_pid_res.append((row[1]["net name"], row[1]["property_id"], row[1]["query_result"]))
    #
    # # validate that for same name+pid the result is equal in cegarabou and marabou
    # all_equal = True
    # for name, pid, res in df1_name_pid_res:
    #     res2 = df2[(df2["net name"] == name) & (df2["property_id"] == pid)]
    #     if res2.shape[0] > 0:
    #         assert res2.shape[0] == 1
    #         res2 = res2.reset_index()
    #         # print(res2.iloc[0]["query_result"])
    #
    #         if res2.iloc[0]["query_result"] != res:
    #             print(name, pid, res, res2.iloc[0]["query_result"])
    #             all_equal = False
    # assert all_equal


def compare_query_results(result_dir_name="2020-03-24", verbose=True):
    try:
        df_marabou = pd.read_json(f"/home/yizhak/Research/Code/CEGAR_NN/AR_results/marabou/{result_dir_name}/outfile_df2json/df_all")
    except ValueError:
        print("no results in marabou")
        return
    try:
        df_cegarabou = pd.read_json(f"/home/yizhak/Research/Code/CEGAR_NN/AR_results/cegarabou/{result_dir_name}/outfile_df2json/df_all")
    except ValueError:
        print("no results in cegarabou")
        return
    marabou_net_name_property_id_2_query_result = {}
    for row in df_marabou.iterrows():
        marabou_net_name_property_id_2_query_result[f"{row[1]['net name']}_{row[1]['property_id']}"] = row[1]['query_result']
    cegarabou_net_name_property_id_2_query_result = {}
    for row in df_cegarabou.iterrows():
        cegarabou_net_name_property_id_2_query_result[f"{row[1]['net name']}_{row[1]['property_id']}"] = row[1]['query_result']
    print(f"number of marabou query results: {len(marabou_net_name_property_id_2_query_result)}")
    print(f"number of cegarabou query results: {len(cegarabou_net_name_property_id_2_query_result)}")
    # compare results
    joint_keys = set(cegarabou_net_name_property_id_2_query_result.keys()).intersection(marabou_net_name_property_id_2_query_result.keys())
    print(f"number of joint keys (both methods have query_result): {len(joint_keys)}")
    diffs_counter = 0
    equals_counter = 0
    for jk in joint_keys:
        if cegarabou_net_name_property_id_2_query_result[jk] != marabou_net_name_property_id_2_query_result[jk]:
            if verbose:
                (f"different query_result on {jk}: cegarabou: {cegarabou_net_name_property_id_2_query_result[jk]}, marabou: {marabou_net_name_property_id_2_query_result[jk]}")
            diffs_counter += 1
        else:
            equals_counter += 1
    print(f"number of diffs: {diffs_counter}")
    print(f"number of equals: {equals_counter}")


def verify_equal_query_results(marabou_df, cegarabou_df):
    marabou_results_df = marabou_df[["net_name", "property_id", "query_result"]]
    cegarabou_results_df = cegarabou_df[["net_name", "property_id", "query_result"]]

    # generate df for validation (include the results of the two methods for each <net_id, property_id> couple)
    validate_df = pd.merge(
        marabou_results_df, cegarabou_results_df, how='left',
        left_on=["net_name", "property_id"], right_on=["net_name", "property_id"],
        suffixes=["_marabou", "_cagarabou"]
    )

    # remove experiments that were not finished in both methods
    validate_df.dropna(subset=["query_result_cagarabou", "query_result_marabou"], inplace=True)

    # get all rows where same query_result was obtained
    correct_results_df = validate_df[validate_df.query_result_cagarabou == validate_df.query_result_marabou]

    # print "valid" only if for all experiments the same result was obtained
    print(
        "VALID" if correct_results_df.shape[0] == validate_df.shape[0] else "INVALID",
        correct_results_df.shape[0],
        validate_df.shape[0]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--results_dir_name", dest="results_dir_name",
                        default="2020-01-24")
    parser.add_argument("-v", "--verbose", dest="verbose",
                        default=consts.VERBOSE, action="store_true")
    args = parser.parse_args()
    return args.results_dir_name, args.verbose

if __name__ == '__main__':
    result_dir_name, verbose = parse_args()
    validate_equal_results_marabou_cegarabou(exp_name=result_dir_name)
    # compare_query_results(result_dir_name=result_dir_name, verbose=verbose)
