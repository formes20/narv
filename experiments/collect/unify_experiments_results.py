import os
import argparse
import pandas as pd

# internal imports
from core.configuration import consts


TIMEOUT = 36000

FILTER_DICT = {
    "total_ar_query_time": lambda timeout: timeout < TIMEOUT
}


def unify_experiments_results(results_dirname):
    df_all = None
    filenames = []
    for result_file in [f for f in os.listdir(results_dirname) if f != "df_all"]:
        if "df_all" not in result_file:
            filenames.append(result_file)
        if df_all is None:
            df_all = pd.read_json(path_or_buf=os.path.join(results_dirname, result_file))
        else:
            df_one = pd.read_json(path_or_buf=os.path.join(results_dirname, result_file))
            df_all = df_all.append(df_one, ignore_index=True)
    if filenames:
        df_all = df_all.join(pd.DataFrame({"filenames": filenames}, index=df_all.index))
    # for col, filter_col_func in FILTER_DICT.items():
    #     df_all = df_all[df_all[col].apply(filter_col_func)]
    df_all.to_json(os.path.join(results_dirname, "df_all"))
    print(df_all.shape)
    return df_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dirname", dest="results_dirname", default=consts.results_directory, help="results dir")
    return parser.parse_args()


def main():
    args = parse_args()
    unify_experiments_results(args.results_dirname)


if __name__ == '__main__':
    main()
