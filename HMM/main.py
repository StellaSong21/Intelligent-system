import numpy as np

'''
B I E S
'''
MIN = -3.14E+100


class HMMModel:
    def __init__(self):
        self.PI = self.init_PI()
        print(self.PI)
        self.A = self.init_A()
        print(self.A)
        self.B = None
        pass

    def init_PI(self):
        return np.array([-0.69314718055994531, MIN, MIN, -0.69314718055994530])

    def init_A(self):
        return np.array([[MIN, -1.8657478800299427, MIN, -0.16815881486551582],
                        [MIN, -1.2823397422535623, MIN, -0.32488219676260977],
                        [-0.6917693770168262, MIN, MIN, -0.6945268850651212],
                        [-0.5667411557594416, MIN, MIN, -0.8378756655713817]])

    def init_B(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

    def viterbi(self):
        pass

    def EM(self):
        pass

    def update_A(self):
        pass

    def update_B(self):
        pass


if __name__ == '__main__':
    hmm = HMMModel()
    pass
