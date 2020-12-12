from CRF.util import DataUtil as dutil
from CRF.util import TemplateUtil as tutil
import numpy as np

'''
1. alpha 初始化，长度d？
2. 训练误差如何计算？softmax？
3. 训练的中止条件，如何防止过拟合？
4. 如何处理未出现的字符？
'''


class CRFModel:
    def __init__(self):
        datapaths = ['../DATASET/dataset1/train.utf8', '../DATASET/dataset2/train.utf8']
        tempaths = ['../DATASET/dataset1/template.utf8', '../DATASET/dataset2/template.utf8']
        self.states = ['B', 'I', 'E', 'S']
        self.charset, self.sequences, self.tags, self.space = dutil.stat_charset1(datapaths)
        self.templates = tutil.get_templates(tempaths)
        print(self.templates)
        self.temp_len = len(self.templates[0]) + len(self.templates[1])
        self.u_lambda = [[] for i in range(len(self.templates[0]))]
        self.b_mu = [[] for i in range(len(self.templates[1]))]
        print(self.u_lambda)
        print(self.b_mu)
        # N * M * T
        pass

    # T: 训练次数
    # self.templates
    def train(self, T, sequence, tags, templates):
        for i in range(T):
            for t in range(len(sequence)):
                # 不处理换段
                if sequence[t] == ' ':
                    continue

                # Unigram
                for template in templates[0]:
                    for j in range(len(template)):
                        pass
                    pass

                # Bigram
                for template in templates[1]:
                    for j in range(len(template)):
                        pass
                    pass
            pass
        pass

    def update(self, sequence, output, target, templates):
        sequence_len = len(sequence)

        for index in range(sequence_len):
            if output[index] == ' ':
                continue

            for u_templates in templates[0]:
                for k in range(len(u_templates)):
                    for i in range(len(u_templates[k])):
                        if index + i < 0 or index + i >= sequence_len:
                            # TODO：更新对应模板，(字符，U|B，第几个模板)
                            self.alpha[sequence[index]][0][k] = 0
                            continue
                        pass
                    pass

            for b_templates in templates[1]:
                for b_temp in range(len(b_templates)):
                    pass
        pass

    def viterbi(self, sequence):
        sequence_len = len(sequence)

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
