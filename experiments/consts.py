# ----- <constants in use> -----
FONT = {
    # 'family': 'serif',
    'color':  'black',
    # 'weight': 'bold',
    'size': 20,
}

RS_1 = 1
RS_10 = 10
RS_50 = 50
RS_100 = 100
AS_100 = 100
AS_250 = 250

# USED are used to cegarabou parameters comparison
USED_RS_1 = RS_50
USED_RS_2 = RS_100
USED_AS_1 = AS_100
USED_AS_2 = AS_250

# BEST are used to marabou-cegarabou comparison
BEST_RS = USED_RS_1
BEST_AS = USED_AS_1


BEST_CEGARABOU_METHODS = {
    "R": "cegar",
    "A": "alg2",
    "RS": BEST_RS,
    "AS": BEST_AS
}

EXP_PARAMS = {
    "L": [100, 500]
}

map_to_formal_names = {
    "_A_complete_": "Abstraction to Saturation",
    "_A_heuristic_alg2_": "Indicator Guided Abstraction",
    "_A_heuristic_random": "Random Partial Abstraction",
    "_R_cegar_": "Counterexample-Guided",
    "_R_cetar_": "Weight-Based",
}

map_cat_to_formal_name = {
    "ar_times": "Sum of Query times",
    "last_query_time": "Last Query Time",
}
