#!/usr/bin/env python3

# REQUIRED IMPORTS FOR CONSTANTS INITIALIZATION
import sys
import os
from datetime import datetime

# CONSTANTS
RUN_ON_CLUSTER = False if os.path.exists("/home/yizhak") else True

if RUN_ON_CLUSTER:
    MARABOU_DIR = "/cs/usr/yizhak333/Research/Marabou_Adv"
    MARABOUPY_DIR = os.path.join(MARABOU_DIR, "maraboupy")
    PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES = "/cs/usr/yizhak333/Research/MarabouApplications/acas/nnet/"
    # directory to save result graph and log file of each experiment
    CEGAR_DIR = "/cs/usr/yizhak333/Research/CEGAR_NN"
    sbatch_files_dir = "/cs/labs/guykatz/yizhak333/AR_sbatch_files/"
    results_directory = os.path.join(CEGAR_DIR, "experiments/")
else:
    PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES = "/home/yizhak/Research/Code/MarabouApplications/acas/nnet/"
    MARABOU_DIR = "/home/yizhak/Research/Code/Marabou_Adv"
    MARABOUPY_DIR = os.path.join(MARABOU_DIR, "maraboupy")
    CEGAR_DIR = "/home/yizhak/Research/Code/CEGAR_NN"
    sbatch_files_dir = os.path.join(CEGAR_DIR, "AR_sbatch_files/")
    results_directory = os.path.join(CEGAR_DIR, "experiments/dev_exp_results_archive")

PATH_TO_MARABOU_ACAS_EXAMPLES = f'{MARABOU_DIR}/src/input_parsers/acas_example/'
properties_directory = os.path.join(CEGAR_DIR, "AR_properties/")

PROPERY_FILNAME_ACAS = os.path.join(os.path.dirname(properties_directory), "property1.txt")

VERBOSE = False
INT_MAX = sys.maxsize
INT_MIN = -sys.maxsize-1
VISUAL_WEIGHT_CONST = 3
SAT_EXIT_CODE = 1
UNSAT_EXIT_CODE = 2

INPUT_LOWER_BOUND = -0.5
INPUT_UPPER_BOUND = 0.5

DO_CEGAR = True
DO_ABSTRACTION_TO_FIRST_HIDDEN_LAYER = False
if DO_ABSTRACTION_TO_FIRST_HIDDEN_LAYER:
    FIRST_ABSTRACT_LAYER = 1
    FIRST_INC_DEC_LAYER = 0
    FIRST_POS_NEG_LAYER = 0
else:
    FIRST_ABSTRACT_LAYER = 2
    FIRST_INC_DEC_LAYER = 1
    FIRST_POS_NEG_LAYER = 1

COMPLETE_ABSTRACTION = True

LAYER_INTERVAL = 8
NODE_INTERVAL = 2
LAYER_TYPE_NAME2COLOR = {
    "input": "r",
    "hidden": "b",
    "output": "g"
}

# read sorting_color_key func documentation
# sorting_color_map is a dict to use when sorting color values,
# such that the "red" node (input) precede blue nodes (hidden) etc.,
# and every layer get color according to its type
SORTING_COLOR_MAP = {
    "white": 0,
    "r": 1,
    "b": 2,
    "g": 3
}

EPSILON = 10 ** -5
DELTA = 0.1

PROPERTY_ID = "property_3"
IS_ADVERSARIAL = True

COMPARE_TO_PREPROCESSED_NET = False

cur_time_str = "_".join(str(datetime.now()).rpartition(".")[0].split(" "))
timeout_str = "10:00:00"
program = "python3 /cs/usr/yizhak333/Research/CEGAR_NN/src/experiments/run/one_experiment_QUERY_METHOD.py"
EXP_POINT_MEAN_SIZE = 100

ar_type2sign = {
    "inc": "+",
    "dec": "-"
}
