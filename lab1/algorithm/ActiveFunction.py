import numpy as np


class Function:
    def __init__(self):
        pass

    def activate(self, x):
        return x

    def deactivate(self, x):
        return x


class Sigmoid(Function):
    def __init__(self):
        Function.__init__(self)
        pass

    def activate(self, x):
        return .5 * (1 + np.tanh(.5 * x))
        # return 1.0 / (1.0 + np.exp(-x))

    def deactivate(self, x):
        fx = self.activate(x)
        return fx * (1 - fx)

    def __str__(self):
        return "Sigmoid"


class Tanh(Function):
    def __init__(self):
        Function.__init__(self)
        pass

    def activate(self, x):
        return np.tanh(x)

    def deactivate(self, x):
        return 1 - (np.tanh(x)) ** 2

    def __str__(self):
        return "Tanh"


class ReLU(Function):
    def __init__(self):
        Function.__init__(self)
        pass

    def activate(self, x):
        return (np.abs(x) + x) / 2

    def deactivate(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def __str__(self):
        return "ReLU"
