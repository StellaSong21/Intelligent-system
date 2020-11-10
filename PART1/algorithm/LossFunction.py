import numpy as np


def squared_error(target, res):
    return np.power((target - res), 2)


def cross_entropy(target, res):
    return -target * np.log(res)
