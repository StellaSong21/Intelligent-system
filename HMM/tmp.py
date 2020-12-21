import math
import numpy as np

from util import DataUtil as dutil

# 1. DEBUG
# 2. 更改接口
# 3. 防止过拟合

np.set_printoptions(threshold=np.inf)

'''
B I E S
'''
MIN = -1000000
precision = 1E-110


class HMMModel:
    def __init__(self):
        filepaths = ['../DATASET/dataset1/train.utf8', '../DATASET/dataset2/train.utf8']
        self.states = ['B', 'I', 'E', 'S']
        self.charset, self.observes, self.indexes = dutil.stat_charset(filepaths)
        # self.PI = self.init_PI()
        # self.A = self.init_A()
        # self.B = self.init_B()
        #
        # np.save('./PI.npy', self.PI)
        # np.save('./A.npy', self.A)
        # np.save('./B.npy', self.B)

        # alpha = self.cal_alpha(self.observes[0])
        # beta = self.cal_beta(self.observes[0])

        # np.save('./alpha.npy', alpha)
        # np.save('./beta.npy', beta)

        self.PI, self.A, self.B = np.load('./PI.npy'), np.load('./A.npy'), np.load('./B.npy')
        # alpha, beta = np.load('./alpha.npy'), np.load('./beta.npy')

        # gamma = self.cal_gamma(self.observes[0], alpha, beta)
        # xi = self.cal_xi(self.observes[0], alpha, beta)

        # np.save('./gamma.npy', gamma)
        # np.save('./xi.npy', xi)

        # gamma, xi = np.load('./gamma.npy'), np.load('./xi.npy')

        # PI, A, B = self.cal_lambda(self.observes[0], self.indexes[0], gamma, xi)

        self.EM(self.observes[0], self.indexes[0], 50)
        path = self.viterbi(self.observes[0])
        print(path)

    def EM(self, observe, index, iteration):
        if len(observe) <= 1:
            return

        alpha = np.zeros((len(observe), len(self.states)), dtype=float)
        old_error = 0.0
        for iter in range(iteration):
            old_alpha = np.copy(alpha)
            print(self.PI)
            alpha, beta = self.cal_alpha(observe), self.cal_beta(observe)
            gamma, xi = self.cal_gamma(observe, alpha, beta), self.cal_xi(observe, alpha, beta)
            self.PI, self.A, self.B = self.cal_lambda(observe, index, gamma, xi)
            error = self.difference(alpha, old_alpha)
            print('error = ', error)
            if error < precision or abs(error - old_error) < precision:
                print('old_error', old_error)
                print('error', error)
                return
            old_error = error
        pass

    def viterbi(self, observe):
        observe_len = len(observe)
        states_len = len(self.states)
        path = np.zeros((observe_len, states_len), dtype=int)
        deltas = np.zeros((observe_len, states_len), dtype=float)

        deltas[0, :] = self.PI + self.B[:, observe[0]]
        path[0, :] = np.arange(states_len)

        for t in range(1, observe_len, 1):
            for i in range(states_len):
                tmp = deltas[t - 1, :] + self.A[:, i]
                deltas[t][i], path[t][i] = np.max(tmp) + self.B[i][observe[t]], np.argmax(tmp)
                if path[t][i] in [1, 2]:
                    print(path[t][i])
        state = np.zeros(observe_len, dtype=int)
        state[observe_len - 1] = 3 if deltas[observe_len - 1][3] >= deltas[observe_len - 1][2] else 2
        for i in range(observe_len - 2, -1, -1):
            state[i] = path[i + 1][state[i + 1]]
        return state

    def difference(self, alpha, old_alpha):
        if old_alpha is None:
            return MIN
        minus = np.max(np.abs(old_alpha - alpha))
        return minus if minus > MIN else MIN

    def init_PI(self):
        return np.array([-0.69314718055994531, MIN, MIN, -0.69314718055994530])

    def init_A(self):
        return np.array([[MIN, -1.8657478800299427, MIN, -0.16815881486551582],
                         # [MIN, -1.2823397422535623, MIN, -0.32488219676260977],
                         [-0.5, -1.2823397422535623, -0.5, -0.32488219676260977],
                         [-0.6917693770168262, MIN, MIN, -0.6945268850651212],
                         [-0.5667411557594416, MIN, MIN, -0.8378756655713817]])

    def init_B(self):
        size = len(self.charset[0])
        init_value = math.log(1.0 / size)
        return np.full((len(self.states), size), fill_value=init_value)

    def cal_alpha(self, observe):
        alpha = np.zeros((len(observe), len(self.states)), dtype=float)
        alpha[0] = self.PI + self.B[:, observe[0]]
        for t in range(len(observe))[1:]:
            for i in range(len(self.states)):
                fp = alpha[t - 1, :] + self.A[:, i]
                alpha[t][i] = self.log_sum(fp) + self.B[i][observe[t]]
        return alpha

    # 等同于
    # def cal_alpha(self, observe):
    #     alpha = np.zeros((len(observe), len(self.states)), dtype=float)
    #     for i in range(len(self.states)):
    #         alpha[0][i] = self.PI[i] + self.B[i][observe[0]]
    #     for t in range(len(observe))[1:]:
    #         for i in range(len(self.states)):
    #             fp = np.zeros(len(self.states), dtype=float)
    #             for j in range(len(self.states)):
    #                 fp[j] = alpha[t - 1][j] + self.A[j][i]
    #             alpha[t][i] = self.sum_log(fp) + self.B[i][observe[t]]
    #     return alpha

    def cal_beta(self, observe):
        observe_len = len(observe)
        beta = np.zeros((len(observe), len(self.states)), dtype=float)
        beta[observe_len - 1, :] = 1
        for t in range(observe_len - 2, -1, -1):
            for i in range(len(self.states)):
                # TODO beta[t + 1, :] + self.A[i, :] * self.B[:, observe[t + 1]] ?
                bp = beta[t + 1, :] + self.A[i, :] + self.B[:, observe[t + 1]]
                beta[t][i] = self.log_sum(bp)
        return beta

    def cal_gamma(self, observe, alpha, beta):
        gamma = alpha + beta
        for t in range(len(observe)):
            gamma[t, :] = gamma[t, :] - self.log_sum(gamma[t])
        return gamma

    def cal_xi(self, observe, alpha, beta):
        observe_len = len(observe)
        states_len = len(self.states)
        xi = np.zeros((observe_len - 1, states_len, states_len), dtype=float)

        for t in range(observe_len - 1):
            for i in range(states_len):
                for j in range(states_len):
                    xi[t][i][j] = alpha[observe[t]][i] + self.A[i][j] + self.B[j][observe[t + 1]] + \
                                  beta[observe[t + 1]][j]
            xi[t, :, :] -= self.log_sum(xi[t].reshape(-1))
        return xi

    def cal_lambda(self, observe, index, gamma, xi):
        PI = self.update_PI(gamma)
        A = self.update_A(gamma, xi)
        B = self.update_B(observe, index, gamma)
        return PI, A, B

    def update_PI(self, gamma):
        PI = gamma[0]
        return PI

    def update_A(self, gamma, xi):
        states_len = len(self.states)
        observe_len = len(gamma)
        A = np.zeros((states_len, states_len), dtype=float)
        for i in range(states_len):
            for j in range(states_len):
                A[i][j] = self.log_sum(xi[:, i, j]) - self.log_sum(gamma[np.arange(observe_len - 1), :][:, i])
        return A

    def update_B(self, observe, index, gamma):
        states_len = len(self.states)
        B = np.copy(self.B)
        gamma_k = np.zeros(len(self.charset[0]), dtype=float)
        for j in range(states_len):
            for k in range(len(self.charset[0])):
                tmp = index.get(self.charset[0][k], [])
                if len(tmp) == 0:
                    gamma_k[k] = MIN
                else:
                    gamma_k[k] = self.log_sum(gamma[tmp, j])
                pass
            B[j] = gamma_k - self.log_sum(gamma[:, j])
        return B

    def log_sum(self, arr):
        max_value = np.max(arr)
        return math.log(np.sum(np.exp(arr - max_value))) + max_value


if __name__ == '__main__':
    hmm = HMMModel()
    pass
