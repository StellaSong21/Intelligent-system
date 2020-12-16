from CRF.util import DataUtil as dutil
from CRF.util import TemplateUtil as tutil
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import os

'''
3. 训练的中止条件，如何防止过拟合？
4. 如何处理未出现的字符？
5. 取平均参数值
6. 比较不同模板
7. 分析数据
'''


class CRFModel(object):
    def __init__(self, datapaths, test_tempaths, T=100, train=0.9):
        super(CRFModel, self).__init__()
        self.states = ['B', 'I', 'E', 'S']
        self.charset, self.sequences, self.tagss = dutil.get_train_set(datapaths)
        self.templatess = tutil.test_get_templates(test_tempaths)
        self.accuracy = np.zeros((len(self.templatess), 2, T), dtype=float)
        for t in range(len(self.templatess)):
            templates = self.templatess[t]
            self.alphas = []
            train_set = []
            train_tags = []
            test_set = []
            test_tags = []
            self.alpha = [[dict() for i in range(len(templates[0]))], [dict() for i in range(len(templates[1]))]]
            for sequence, tags in zip(self.sequences, self.tagss):
                sequence_len = len(sequence)
                tags_len = len(tags)
                train_set.append(sequence[0:int(sequence_len * train)])
                test_set.append(sequence[int(sequence_len * train):])
                train_tags.append(tags[0:int(tags_len * train)])
                test_tags.append(tags[int(tags_len * train):])
            dirname = os.path.splitext(os.path.basename(test_tempaths[t]))[0]
            save_path = [os.path.join('../record', dirname, 'normal'), os.path.join('../record', dirname, 'average')]
            for path in save_path:
                if not os.path.exists(path):
                    os.makedirs(path)
            self.train(T, train_set, train_tags, t, templates, save_path, test_set, test_tags)
        print(self.accuracy)

        pickle.dump(self.accuracy, open('../record/accuracy.pickle', 'wb'))
        self.accuracy = pickle.load(open('../record/accuracy.pickle', 'rb'))

        ####################### 测试部分 #######################
        # 创建画板
        fig = plt.figure()

        # 创建画纸
        ax1 = fig.add_subplot(1, 1, 1)

        # test result
        ax1.set_title('Accuracy/Epoch')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('accuracy')
        for t in range(len(self.templatess)):
            for a in range(2):
                ax1.plot(range(1, T + 1, 1), self.accuracy[t][a], '-',
                         label=os.path.splitext(os.path.basename(test_tempaths[t]))[0]
                               + '/' + 'normal' if a == 1 else 'average')
        plt.legend()
        plt.show()
        #######################################################
        pass

    def train(self, T, sequences, tagss, t, templates, save_path, test_sequences=None, test_tagss=None):
        for i in range(T):
            start = time.time()
            for sequence, tags in zip(sequences, tagss):
                states = self.viterbi(self.alpha, sequence, templates)
                self.update(sequence, states, tags, templates)
            self.alphas.append(self.alpha)
            alphas = [self.alpha, self.avg_alpha(self.alphas, i + 1, templates)]

            # TODO 保存 alpha 和 avg_alpha
            pickle.dump(alphas[0], open(os.path.join(save_path[0], str(i + 1) + '.pickle'), 'wb'))
            pickle.dump(alphas[1], open(os.path.join(save_path[1], str(i + 1) + '.pickle'), 'wb'))
            alphas[0] = pickle.load(open(os.path.join(save_path[0], str(i + 1) + '.pickle'), 'rb'))
            alphas[1] = pickle.load(open(os.path.join(save_path[1], str(i + 1) + '.pickle'), 'rb'))

            end = time.time()
            print(i, ", time: ", end - start)

            ####################### 测试部分 #######################
            start = time.time()
            # alphas = [self.alpha, self.avg_alpha(self.alphas, i + 1, templates)]
            for a in range(2):
                alpha = alphas[a]
                correct = np.zeros(len(test_tagss), dtype=int)
                total = np.zeros(len(test_tagss), dtype=int)
                for j in range(len(test_tagss)):
                    test_sequence = test_sequences[j]
                    test_tags = test_tagss[j]
                    test_states = self.viterbi(alpha, test_sequence, templates)
                    correct[j], total[j] = precision(test_states, test_tags)
                accuracy = 1.0 * np.sum(correct) / np.sum(total) if np.sum(total) > 0 else 0.0
                self.accuracy[t][a][i] = accuracy
                print(accuracy)
            end = time.time()
            print(i, ", time: ", end - start)
            #######################################################

    def avg_alpha(self, alphas, T, templates):
        alpha = [[dict() for i in range(len(templates[0]))], [dict() for i in range(len(templates[1]))]]
        for ub in range(len(templates)):
            for t in range(len(templates[ub])):
                sum = Counter(alphas[0][ub][t])
                for epoch in range(1, T, 1):
                    sum += Counter(alphas[epoch][ub][t])
                sum_dict = dict(sum)
                for key in sum_dict:
                    sum_dict[key] = sum_dict[key] / T
                alpha[ub][t] = sum_dict
        return alpha

    def update(self, sequence, output, target, templates):
        self.sub_update(sequence, output, target, templates)

    def sub_update(self, sequence, output, target, templates):
        sequence_len = len(sequence)
        for index in range(sequence_len):
            # 更新 unigram
            for utmpl_index in range(len(templates[0])):
                y = self.get_key([output[index]], sequence, index, templates[0][utmpl_index])
                t = self.get_key([target[index]], sequence, index, templates[0][utmpl_index])
                self.alpha[0][utmpl_index][y] = self.alpha[0][utmpl_index].get(y, 0) - 1
                self.alpha[0][utmpl_index][t] = self.alpha[0][utmpl_index].get(t, 0) + 1
                pass

            # 更新 bigram
            for btmpl_index in range(len(templates[1])):
                y = self.get_key([output[index - 1] if index > 0 else None, output[index]],
                                 sequence, index, templates[1][btmpl_index])
                t = self.get_key([target[index - 1] if index > 0 else None, target[index]],
                                 sequence, index, templates[1][btmpl_index])
                self.alpha[1][btmpl_index][y] = self.alpha[1][btmpl_index].get(y, 0) - 1
                self.alpha[1][btmpl_index][t] = self.alpha[1][btmpl_index].get(t, 0) + 1
                pass

    def viterbi(self, alpha, sequence, templates):
        states = []
        states.extend(self.sub_viterbi(alpha, sequence, templates))
        return states

    def sub_viterbi(self, alpha, sequence, templates):
        sequence_len = len(sequence)
        states_len = len(self.states)
        utmpl_len = len(templates[0])
        btmpl_len = len(templates[1])

        states = ['0' for i in range(sequence_len)]

        score = np.zeros((sequence_len, states_len), dtype=float)
        path = np.zeros((sequence_len, states_len), dtype=int)

        # 1. 初始化
        for tmpl_index in range(utmpl_len):
            score[0] += self.get_score_tmp(alpha, sequence, 0, templates, 0, tmpl_index, None)
        for tmpl_index in range(btmpl_len):
            score[0] += self.get_score_tmp(alpha, sequence, 0, templates, 1, tmpl_index, None)

        # 2. 递推
        for index in range(1, sequence_len, 1):
            state_score = self.get_score(alpha, sequence, index, templates)
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

    def get_score(self, alpha, sequence, index, templates):
        states_len = len(self.states)

        # unigram
        u_score = np.zeros(states_len, dtype=float)
        for tmpl_index in range(len(templates[0])):
            u_score += self.get_score_tmp(alpha, sequence, index, templates, 0, tmpl_index)

        # bigram
        b_score = np.zeros((states_len, states_len), dtype=float)  # (上一个状态，当前状态)
        for tmpl_index in range(len(templates[1])):
            for si in range(states_len):
                b_score[si] += self.get_score_tmp(alpha, sequence, index, templates, 1, tmpl_index, self.states[si])

        # unigram + bigram
        state_score = u_score + b_score
        return state_score

    def get_score_tmp(self, alpha, sequence, index, templates, ub, template_index, last_tag=None):
        """
        :param sequence: 观察序列
        :param index: 当前字符在观察序列中的下标
        :param template_index: 当前用的模板下标
        :return: 按 self.states 的顺序返回每一种状态的分数
        """
        state_score = np.zeros(len(self.states), dtype=float)
        for sj in range(len(self.states)):
            states = [self.states[sj]]
            if ub == 1:
                states.insert(0, last_tag)
            key = self.get_key(states, sequence, index, templates[ub][template_index])
            state_score[sj] = alpha[ub][template_index].get(key, 0)
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
    return correct, total


if __name__ == '__main__':
    crf = CRFModel(['../../DATASET/dataset1/train.utf8', '../../DATASET/dataset2/train.utf8'],
                   ['../../DATASET/templates/template1.utf8', '../../DATASET/templates/template2.utf8',
                    '../../DATASET/templates/template3.utf8', '../../DATASET/templates/template4.utf8',
                    '../../DATASET/templates/template5.utf8'],
                   T=100)
