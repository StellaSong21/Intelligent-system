from CRF.util import DataUtil as dutil
from CRF.util import TemplateUtil as tutil
import numpy as np
import time

'''
1. alpha 初始化，长度d？
2. 训练误差如何计算？softmax？
3. 训练的中止条件，如何防止过拟合？
4. 如何处理未出现的字符？
'''


class CRFModel(object):
    def __init__(self):
        super(CRFModel, self).__init__()
        datapaths = ['../DATASET/dataset0/train.utf8', '../DATASET/dataset0/train.utf8']
        tempaths = ['../DATASET/dataset0/template.utf8']
        self.states = ['B', 'I', 'E', 'S']
        self.charset, self.sequences, self.tags, self.space = dutil.stat_charset1(datapaths)
        print(self.tags[0])
        self.templates = tutil.get_templates(tempaths)
        print(self.templates)
        print(self.space[0])
        print(len(self.sequences[0]))
        print(self.sequences[0][-1])
        self.temp_len = len(self.templates[0]) + len(self.templates[1])
        self.u_lambda = [dict() for i in range(len(self.templates[0]))]
        self.b_mu = [dict() for i in range(len(self.templates[1]))]
        self.train(10, self.sequences[0], self.tags[0], self.space[0])
        state_tmp = self.viterbi(self.sequences[1], self.space[1])
        print(state_tmp)
        pass

    # T: 训练次数
    # self.templates
    def train(self, T, sequence, tags, space):
        for i in range(T):
            print(i)
            start = time.time()
            states = self.viterbi(sequence, space)
            mid = time.time()
            print(i, ", viterbi time: ", mid - start)
            self.update(sequence, states, tags, space)
            end = time.time()
            print(i, ", update time: ", end - mid)
            print(i, ", time: ", end - start)
        pass

    def update(self, sequence, output, target, space):
        for i in range(len(space) - 1):
            self.sub_update(sequence[space[i] + 1:space[i + 1]], output[space[i] + 1: space[i + 1]],
                            target[space[i] + 1:space[i + 1]])

    def sub_update(self, sequence, output, target):
        sequence_len = len(sequence)
        for index in range(sequence_len):
            # 更新 unigram
            for utmpl_index in range(len(self.templates[0])):
                y = [sequence[index + i] if (index + i) in range(0, sequence_len) else None
                     for i in self.templates[0][utmpl_index]]
                t = y.copy()
                y.insert(0, output[index])
                t.insert(0, target[index])
                self.u_lambda[utmpl_index][tuple(y)] = self.u_lambda[utmpl_index].get(tuple(y), 0) - 1
                self.u_lambda[utmpl_index][tuple(t)] = self.u_lambda[utmpl_index].get(tuple(t), 0) + 1
                pass

            # 更新 bigram
            for btmpl_index in range(len(self.templates[1])):
                y = [output[index - 1] if index > 0 else None, output[index]]
                t = [target[index - 1] if index > 0 else None, target[index]]
                tmp = [sequence[index + i] if (index + i) in range(0, sequence_len) else None
                       for i in self.templates[1][btmpl_index]]
                y.extend(tmp)
                t.extend(tmp)
                self.b_mu[btmpl_index][tuple(y)] = self.b_mu[btmpl_index].get(tuple(y), 0) - 1
                self.b_mu[btmpl_index][tuple(t)] = self.b_mu[btmpl_index].get(tuple(t), 0) + 1
                pass

    def viterbi(self, sequence, space):
        states = []
        for i in range(len(space) - 1):
            states.extend(self.sub_viterbi(sequence[space[i] + 1: space[i + 1]]))
            states.append(' ')
        states.pop()
        return states

    def sub_viterbi(self, sequence):
        sequence_len = len(sequence)
        states_len = len(self.states)
        utmpl_len = len(self.templates[0])
        btmpl_len = len(self.templates[1])

        states = ['0' for i in range(sequence_len)]
        score = np.zeros((sequence_len, states_len, states_len), dtype=int)
        u_score = np.zeros(len(self.states), dtype=int)
        b_score = np.zeros((states_len, states_len), dtype=int)

        for index in range(sequence_len):
            for tmpl_index in range(utmpl_len):
                u_score = self.u_score(sequence, index, tmpl_index)
            for tmpl_index in range(btmpl_len):
                b_score = self.b_score(sequence, index, tmpl_index)
            score[index] = b_score + u_score

        pos = list(np.unravel_index(np.argmax(score[-1], axis=None), score[-1].shape))
        states[sequence_len - 1] = self.states[pos[0]]
        sj_index = pos[1]

        for index in range(sequence_len - 2, -1, -1):
            states[index] = self.states[sj_index]
            sj_index = np.argmax(score[index][sj_index])

        return states

    def u_score(self, sequence, index, template_index):
        """
        :param sequence: 观察序列
        :param index: 当前字符在观察序列中的下标
        :param template_index: 当前用的模板下标
        :return: 按 self.states 的顺序返回每一种状态的分数
        """
        state_score = np.zeros(len(self.states), dtype=int)
        sequence_len = len(sequence)
        for s in range(len(self.states)):
            list = [sequence[index + i] if (index + i) in range(0, sequence_len) else None
                    for i in self.templates[0][template_index]]
            list.insert(0, self.states[s])
            state_score[s] = self.u_lambda[template_index].get(tuple(list), 0)
        return state_score

    def b_score(self, sequence, index, template_index):
        """
        :param sequence: 观察序列
        :param index: 当前字符在观察序列中的下标
        :param template_index: 当前用的模板下标
        :return: 返回 [si-1, si] 的分数
        """
        sequence_len = len(sequence)
        states_len = len(self.states)
        state_score = np.zeros((states_len, states_len), dtype=int)
        for si in range(len(self.states)):
            for sj in range(len(self.states)):
                list = [self.states[si], self.states[sj]]
                list.extend([sequence[index + i] if (index + i) in range(0, sequence_len) else None
                             for i in self.templates[1][template_index]])
                state_score[sj][si] = self.b_mu[template_index].get(tuple(list), 0)
        return state_score


if __name__ == '__main__':
    crf = CRFModel()
