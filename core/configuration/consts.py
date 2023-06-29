# REQUIRED IMPORTS FOR CONSTANTS INITIALIZATION
import sys
import os
from datetime import datetime

# CONSTANTS
CODE_DIR = "/home/artifact/"
PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES = os.path.join(CODE_DIR, "Marabou/resources/nnet/acasxu")
PATH_MINIST_EXAMPLES = "/oval/"
CEGAR_DIR = os.path.join(CODE_DIR, "CEGAR_NN")
# results_directory = os.path.join(CEGAR_DIR, "experiments/dev_exp_results_archive")
results_directory = os.path.join(CODE_DIR, "narv/results/")


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
DELTA = 0.01

PROPERTY_ID = "property_3"

COMPARE_TO_PREPROCESSED_NET = False

cur_time_str = "_".join(str(datetime.now()).rpartition(".")[0].split(" "))

ar_type2sign = {
    "inc": "+",
    "dec": "-"
}

