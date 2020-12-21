from util import DataUtil as dutil
from util import TemplateUtil as tutil
import numpy as np
import pickle


class CRFModel(object):
    def __init__(self, alpha_path, tempaths):
        super(CRFModel, self).__init__()
        self.states = ['B', 'I', 'E', 'S']
        self.templates = tutil.get_templates(tempaths)
        self.alpha = pickle.load(open(alpha_path, 'rb'))
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

        score = np.zeros((sequence_len, states_len), dtype=float)
        path = np.zeros((sequence_len, states_len), dtype=int)

        # 1. 初始化
        for tmpl_index in range(utmpl_len):
            score[0] += self.get_score_tmp(self.alpha, sequence, 0, self.templates, 0, tmpl_index, None)
        for tmpl_index in range(btmpl_len):
            score[0] += self.get_score_tmp(self.alpha, sequence, 0, self.templates, 1, tmpl_index, None)

        # 2. 递推
        for index in range(1, sequence_len, 1):
            state_score = self.get_score(self.alpha, sequence, index, self.templates)
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


if __name__ == '__main__':
    crf = CRFModel('./record/template4/normal/67.pickle', ['../DATASET/templates/template4.utf8'])
    output = crf.viterbi(list('《东方书林之旅》告诉你什么叫出版策划——书海弄潮儿'))
    print(dutil.precision(output, list('SBEBESSSBESBESBEBEBESSBIE')))
