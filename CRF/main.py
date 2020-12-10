import util.DataUtil as dutil
import util.TemplateUtil as tutil
import numpy as np

'''
1. alpha 初始化，长度d？
2. 训练误差如何计算？softmax？
3. 训练的中止条件，如何防止过拟合？
'''


class CRFModel:
    def __init__(self):
        datapaths = ['../DATASET/dataset1/train.utf8', '../DATASET/dataset2/train.utf8']
        tempaths = ['../DATASET/dataset1/template.utf8', '../DATASET/dataset2/template.utf8']
        self.states = ['B', 'I', 'E', 'S']
        self.charset, self.observes, self.indexes = dutil.stat_charset(datapaths)
        templates = tutil.get_templates(tempaths)
        self.alpha = np.zeros(len(templates), dtype=int)
        print(templates)
        pass

    def train(self):
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


if __name__ == '__main__':
    crf = CRFModel()
