#/usr/bin/python3

# external imports
import os
import sys
sys.path.append("/cs/usr/yizhak333/Research/CEGAR_NN/src/")
import argparse

# internal imports
from core.configuration import consts
from experiments.consts import BEST_CEGARABOU_METHODS

# general_experiment_content includes the general content of sbatch file, the specific parameters that are changed
# between specific experiments are QUERY_METHOD, FILENAME, CALLING_PARAMETERS
general_experiment_content = "\n".join([
    "#!/bin/bash",
    "#",
    "#SBATCH --job-name=yizhak_test",
    "#SBATCH --cpus-per-task=2",
    "#SBATCH --mem-per-cpu=8G",
    "#SBATCH --output=OUTPUT_DIR/FILENAME.output",
    "#SBATCH --partition=long",
    "#SBATCH --time=TIMEOUT",
    "#SBATCH --signal=B:SIGUSR1@300",
    "",
    "export PYTHONPATH=$PYTHONPATH:/cs/usr/yizhak333/Research/Marabou",  # TODO: replace with PROGRAM_PREREQUISITS
    # "export PYTHONPATH=$PYTHONPATH:/cs/usr/yizhak333/Research/Marabou",  # TODO: replace with PROGRAM_PREREQUISITS
    "export PYTHONPATH=$PYTHONPATH:/cs/usr/yizhak333/Research/CEGAR_NN/src/",  # TODO: replace with PROGRAM_PREREQUISITS
    "PROGRAM CALLING_PARAMETERS"
])

net_numbers = [
    "1_1", "1_2", "1_3", "1_4", "1_5", "1_6", "1_7", "1_8", "1_9",
    "2_1", "2_2", "2_3", "2_4", "2_5", "2_6", "2_7", "2_8", "2_9",
    "3_1", "3_2", "3_3", "3_4", "3_5", "3_6", "3_7", "3_8", "3_9",
    "4_1", "4_2", "4_3", "4_4", "4_5", "4_6", "4_7", "4_8", "4_9",
    "5_1", "5_2", "5_3", "5_4", "5_5", "5_6", "5_7", "5_8", "5_9"
]
property_ids = [
    "basic_1", "basic_2",
    "property_1", "property_2", "property_3", "property_4",
    "adversarial_0", "adversarial_1", "adversarial_2", "adversarial_3",
    "adversarial_4", "adversarial_5", "adversarial_6", "adversarial_7",
    "adversarial_8", "adversarial_9", "adversarial_10", "adversarial_11",
    "adversarial_12", "adversarial_13", "adversarial_14", "adversarial_15",
    "adversarial_16", "adversarial_17", "adversarial_18", "adversarial_19"
]
#lower_bounds = [100, 500]
refinements = [BEST_CEGARABOU_METHODS["R"]] # ["cegar", "weight_based"]
abstractions = ["naive", "alg2"]  # [BEST_CEGARABOU_METHODS["A"]]  # ["naive", "alg2", "random"]
refinement_sequences = [BEST_CEGARABOU_METHODS["RS"]] # [50, 100]
abstraction_sequences = [BEST_CEGARABOU_METHODS["AS"]] # [100, 250]
preprocess_options = [False]

# experiments_data is dict of the form
# {
#   experiment_dir ->
#   {
#       "program" -> program_path,
#       "timeout": program_timeout,
#       "params_names": [(param shortcut in argparse, param name in argparse), ...],
#       "params_possible_values":
#       {
#           parameter name in argparse -> [possible values]
#       }
#   }
#
# },
experiments_all_data = {
    "marabou": {
        "dirname": os.path.join(consts.sbatch_files_dir, "marabou"),  # path to directory where all .sbatch files are generated
        "output_dir": os.path.join(consts.sbatch_files_dir, "marabou"),  # path to directory where all .output files are generated
        "params_possible_values": {
            "NET_NUMBER": net_numbers,
            "PROPERTY_ID": property_ids,
            # "LOWER_BOUND": lower_bounds,
            # "PREPROCESS": preprocess_options,
            "MECHANISM": ["marabou"]
        },
        "params_names": [
            ("nn", "NET_NUMBER"),
            ("p", "PROPERTY_ID"),
            # ("p", "PREPROCESS"),
            ("m", "MECHANISM"),
            # ("l", "LOWER_BOUND")
        ],
        "program": consts.program,  #.replace("QUERY_METHOD", "marabou"),
        "timeout": consts.timeout_str
    },
    "cegarabou": {
        "dirname": os.path.join(consts.sbatch_files_dir, "cegarabou"),  # path to directory where all .sbatch files are generated
        "output_dir": os.path.join(consts.sbatch_files_dir, "cegarabou"),  # path to directory where all .output files are generated
        "params_possible_values": {
            # { parameter place_holder -> parameter possible values }
            # "LOWER_BOUND": lower_bounds,
            "ABSTRACTION": abstractions,
            "REFINEMENT": refinements,
            "NET_NUMBER": net_numbers,
            "PROPERTY_ID": property_ids,
            "REFINEMENT_SEQUENCE": refinement_sequences,
            "ABSTRACTION_SEQUENCE": abstraction_sequences,
            # "PREPROCESS": preprocess_options,
            "MECHANISM": ["marabou_with_ar"]
        },
        "params_names": [
            # [ (parameter name, parameter place_holder), (...) ]
            ("nn", "NET_NUMBER"),
            ("p", "PROPERTY_ID"),
            # ("p", "PREPROCESS"),
            ("m", "MECHANISM"),
            ("as", "ABSTRACTION_SEQUENCE"),
            ("rs", "REFINEMENT_SEQUENCE"),
            ("a", "ABSTRACTION"),
            ("r", "REFINEMENT"),
            # ("l", "LOWER_BOUND")
        ],
        "program": consts.program,  #.replace("QUERY_METHOD", "cegarabou"),
        "timeout": consts.timeout_str
    }
}

# general_params is a dictionary of the form: { query method  --> { parameter name --> parameter place_holder } }


def recursive_generation_of_experiments(experiment_filename_str, experiment_content_string, param_shortcuts,
                                        param_names, param_name2values, dirname, calling_parameters="", verbose=False):
    """
    recursive function to generate experiments filenames.
    an experiment file has specific filename and content a.t. experiment's parameters
    :param experiment_filename_str: filename of the experiment
    :param experiment_content_string: content of the experiment
    :param param_shortcuts: shortcut for param names, e.g "nn" for "NET_NUMBER"
    :param param_names: name of the parameter
    :param param_name2values: map from parameter's name to list of the possible values of the parameter, e.g
                              {"NET_NUMBER": ["1_3", "2_1"]}
    :param calling_parameters: string that include the signature of calling parameters, e.g "-nn NET_NUMBER"
    :param dirname: path to directory where all .sbatch files are generated
    :param verbose: verbosity flag
    """
    if verbose:
        print(("\n"+"+"*80+"\n").join([
            "experiment_filename_str={}".format(experiment_filename_str),
            "experiment_content_string={}".format(experiment_content_string),
            "param_shortcuts={}".format(param_shortcuts),
            "param_names={}".format(param_names),
            "param_name2values={}".format(param_name2values),
            ])
        )
    if not param_names:
        with open(os.path.join(dirname, experiment_filename_str + ".sbatch"), "w") as fw:
            experiment_content_string = experiment_content_string.replace("FILENAME", experiment_filename_str)
            experiment_content_string = experiment_content_string.replace("CALLING_PARAMETERS", calling_parameters)
            fw.write(experiment_content_string)
        return
    param_name = param_names[0]
    param_shortcut = param_shortcuts[0]
    for value in param_name2values[param_name]:
        updated_filename = "_".join([experiment_filename_str, param_shortcut, str(value)])
        updated_calling_parameters = calling_parameters + " -{} {}".format(param_shortcut, str(value))
        recursive_generation_of_experiments(experiment_filename_str=updated_filename,
                                            experiment_content_string=experiment_content_string,
                                            param_shortcuts=param_shortcuts[1:],
                                            param_names=param_names[1:],
                                            param_name2values=param_name2values,
                                            dirname=dirname,
                                            calling_parameters=updated_calling_parameters,
                                            verbose=verbose)


def generate_sbatch_files(experiments_data, verbose=False):
    """
    prepare sbatch files in a directories a.t the content of @experiment_json.
    each directory contain sbatch files for experiments
    :param: experiment_data: dictionary with data about all experiments
    """
    for experiment_name, experiment_data in experiments_data.items():
        dirname = experiment_data["dirname"]
        output_dir = experiment_data["output_dir"]
        timeout = experiment_data["timeout"]
        program = experiment_data["program"]

        if verbose:
            print("-"*80)
            print(dirname)
            print(output_dir)
            print(timeout)
            print(program)
            print("-"*80)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        experiment_content = general_experiment_content.replace("OUTPUT_DIR", output_dir)
        experiment_content = experiment_content.replace("TIMEOUT", timeout)
        experiment_content = experiment_content.replace("PROGRAM", program)
        recursive_generation_of_experiments(experiment_filename_str="exp",
                                            experiment_content_string=experiment_content,
                                            param_shortcuts=[p[0] for p in experiment_data["params_names"]],
                                            param_names=[p[1] for p in experiment_data["params_names"]],
                                            param_name2values=experiment_data["params_possible_values"],
                                            dirname=dirname,
                                            calling_parameters="",
                                            verbose=verbose)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False, help="verbosity flag")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_sbatch_files(experiments_data=experiments_all_data, verbose=args.verbose)


if __name__ == '__main__':
    main()
