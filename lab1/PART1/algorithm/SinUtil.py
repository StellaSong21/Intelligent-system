import numpy as np


def get_set(length):
    input = np.random.rand(length) * 2 * np.pi - 1 * np.pi
    target = np.sin(input)
    output = np.zeros(length, dtype='float')
    loss = np.zeros(length, dtype='float')
    return input, target, output, loss
