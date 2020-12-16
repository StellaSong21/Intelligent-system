from CRF.util import DataUtil as dutil
from CRF.util import TemplateUtil as tutil
import numpy as np
import time
import matplotlib.pyplot as plt

'''
3. 训练的中止条件，如何防止过拟合？
4. 如何处理未出现的字符？
5. 取平均参数值
6. 比较不同模板
7. 分析数据
'''


class CRFModel(object):
    def __init__(self):
        super(CRFModel, self).__init__()
        datapaths = ['../DATASET/dataset1/train.utf8', '../DATASET/dataset0/train.utf8']
        tempaths = ['../DATASET/templates/template0.utf8']
        self.states = ['B', 'I', 'E', 'S']
        self.charset, self.sequences, self.tags = dutil.get_train_set(datapaths)
        self.templates = tutil.get_templates(tempaths)
        self.alphas = []
        self.alpha = [[dict() for i in range(len(self.templates[0]))], [dict() for i in range(len(self.templates[1]))]]
        self.train(100, self.sequences[0], self.tags[0], self.sequences[1], self.tags[1])
        pass

    # T: 训练次数
    # self.templates
    def train(self, T, sequence, tags, test_sequence=None, test_tags=None):

        ####################### 测试部分 #######################
        test_loss = np.zeros(T, dtype=float)
        #######################################################

        for i in range(T):
            start = time.time()
            states = self.viterbi(sequence)
            self.update(sequence, states, tags)
            end = time.time()
            print(i, ", time: ", end - start)

            ####################### 测试部分 #######################
            start = time.time()
            test_states = self.viterbi(test_sequence)
            test_loss[i] = precision(test_states, test_tags)
            end = time.time()
            print(i, ", time: ", end - start)
            print(test_loss[i])
            #######################################################

        ####################### 测试部分 #######################
        # 创建画板
        fig = plt.figure()

        # 创建画纸
        ax1 = fig.add_subplot(1, 1, 1)

        # test result
        ax1.set_title('Accuracy/Epoch')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('accuracy')
        ax1.plot(range(T), test_loss, '-')
        plt.show()
        #######################################################

        pass

    def update(self, sequence, output, target):
        self.sub_update(sequence, output, target)

    def sub_update(self, sequence, output, target):
        sequence_len = len(sequence)
        for index in range(sequence_len):
            # 更新 unigram
            for utmpl_index in range(len(self.templates[0])):
                y = self.get_key([output[index]], sequence, index, self.templates[0][utmpl_index])
                t = self.get_key([target[index]], sequence, index, self.templates[0][utmpl_index])
                self.alpha[0][utmpl_index][y] = self.alpha[0][utmpl_index].get(y, 0) - 1
                self.alpha[0][utmpl_index][t] = self.alpha[0][utmpl_index].get(t, 0) + 1
                pass

            # 更新 bigram
            for btmpl_index in range(len(self.templates[1])):
                y = self.get_key([output[index - 1] if index > 0 else None, output[index]],
                                 sequence, index, self.templates[1][btmpl_index])
                t = self.get_key([target[index - 1] if index > 0 else None, target[index]],
                                 sequence, index, self.templates[1][btmpl_index])
                self.alpha[1][btmpl_index][y] = self.alpha[1][btmpl_index].get(y, 0) - 1
                self.alpha[1][btmpl_index][t] = self.alpha[1][btmpl_index].get(t, 0) + 1
                pass

    def viterbi(self, sequence):
        states = []
        states.extend(self.sub_viterbi(sequence))
        return states

    def sub_viterbi(self, sequence):
        sequence_len = len(sequence)
        states_len = len(self.states)
        utmpl_len = len(self.templates[0])
        btmpl_len = len(self.templates[1])

        states = ['0' for i in range(sequence_len)]

        score = np.zeros((sequence_len, states_len), dtype=int)
        path = np.zeros((sequence_len, states_len), dtype=int)

        # 1. 初始化
        for tmpl_index in range(utmpl_len):
            score[0] += self.get_score_tmp(sequence, 0, 0, tmpl_index, None)
        for tmpl_index in range(btmpl_len):
            score[0] += self.get_score_tmp(sequence, 0, 1, tmpl_index, None)

        # 2. 递推
        for index in range(1, sequence_len, 1):
            state_score = self.get_score(sequence, index)
            state_score += score[index - 1].reshape(1, -1).T
            for sj in range(states_len):
                score[index][sj], path[index][sj] = np.max(state_score[:, sj]), np.argmax(state_score[:, sj])

        # 3. 中止
        sj = np.argmax(score[-1])
        states[-1] = self.states[int(sj)]

        # 4. 回溯
        for index in range(sequence_len - 1, 0, -1):
            si = path[index][sj]
            states[index - 1] = self.states[si]
            sj = si
            pass

        return states

    def get_score(self, sequence, index):
        states_len = len(self.states)

        # unigram
        u_score = np.zeros(states_len, dtype=int)
        for tmpl_index in range(len(self.templates[0])):
            u_score += self.get_score_tmp(sequence, index, 0, tmpl_index)

        # bigram
        b_score = np.zeros((states_len, states_len), dtype=int)  # (上一个状态，当前状态)
        for tmpl_index in range(len(self.templates[1])):
            for si in range(states_len):
                b_score[si] += self.get_score_tmp(sequence, index, 1, tmpl_index, self.states[si])

        # unigram + bigram
        state_score = u_score + b_score
        return state_score

    def get_score_tmp(self, sequence, index, ub, template_index, last_tag=None):
        """
        :param sequence: 观察序列
        :param index: 当前字符在观察序列中的下标
        :param template_index: 当前用的模板下标
        :return: 按 self.states 的顺序返回每一种状态的分数
        """
        state_score = np.zeros(len(self.states), dtype=int)
        for sj in range(len(self.states)):
            states = [self.states[sj]]
            if ub == 1:
                states.insert(0, last_tag)
            key = self.get_key(states, sequence, index, self.templates[ub][template_index])
            state_score[sj] = self.alpha[ub][template_index].get(key, 0)
        return state_score

    def get_key(self, states, sequence, index, template):
        list = states.copy()
        sequence_len = len(sequence)
        tmp = [sequence[index + i] if (index + i) in range(0, sequence_len) else None
               for i in template]
        list.extend(tmp)
        return tuple(list)


def precision(output, target):
    total = 0
    correct = 0
    i = 0
    while i < len(output):
        if output[i] == 'S':
            total += 1
            correct += 1 if target[i] == 'S' else 0
        elif output[i] == 'B':
            j = i
            i += 1
            while i < len(output) and output[i] == 'I':
                i += 1
            if i < len(output) and output[i] == 'E':
                total += 1
                correct += 1 if target[j:i + 1] == output[j:i + 1] else 0
            else:
                i -= 1
        i += 1
    return 1.0 * correct / total if total > 0 else 0.0


if __name__ == '__main__':
    crf = CRFModel()
