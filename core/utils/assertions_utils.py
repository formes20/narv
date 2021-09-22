#!/usr/bin/env python3

import numpy as np
from core.configuration.consts import EPSILON


def is_evaluation_result_equal(output, speedy_output) -> bool:
    # print(output)
    # print(speedy_output)
    return all(np.array([x[1] for x in output]) - np.array(speedy_output) < EPSILON)
