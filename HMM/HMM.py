import math
import numpy as np

from util import DataUtil as dutil

np.set_printoptions(threshold=np.inf)

MIN = -1000000


class HMMModel:
    def __init__(self, filepaths):
        self.states = ['B', 'I', 'E', 'S']
        self.charset, dict_start, dict_trans, dict_emit = dutil.HMM_stat(filepaths)
        self.PI = self.init_PI(dict_start)
        self.A = self.init_A(dict_trans)
        self.B = self.init_B(self.charset, dict_start, dict_emit)

    def init_PI(self, dict_start):
        return np.array([math.log(1.0 * dict_start['B'] / (dict_start['B'] + dict_start['S'])),
                         MIN, MIN,
                         math.log(1.0 * dict_start['S'] / (dict_start['B'] + dict_start['S']))])

    def init_A(self, dict_trans):
        return np.array([[MIN, math.log(1.0 * dict_trans['B']['I'] / (dict_trans['B']['I'] + dict_trans['B']['E'])),
                          MIN, math.log(1.0 * dict_trans['B']['E'] / (dict_trans['B']['I'] + dict_trans['B']['E']))],
                         [MIN, math.log(1.0 * dict_trans['I']['I'] / (dict_trans['I']['I'] + dict_trans['I']['E'])),
                          math.log(1.0 * dict_trans['I']['E'] / (dict_trans['I']['I'] + dict_trans['I']['E'])), MIN],
                         [math.log(1.0 * dict_trans['E']['B'] / (dict_trans['E']['B'] + dict_trans['E']['S'])), MIN,
                          MIN, math.log(1.0 * dict_trans['E']['S'] / (dict_trans['E']['B'] + dict_trans['E']['S']))],
                         [math.log(1.0 * dict_trans['S']['B'] / (dict_trans['S']['B'] + dict_trans['S']['S'])), MIN,
                          MIN, math.log(1.0 * dict_trans['S']['S'] / (dict_trans['S']['B'] + dict_trans['S']['S']))]])

    def init_B(self, charset, dict_start, dict_emit):
        charset_size = len(charset) + 1
        B = np.zeros((len(self.states), charset_size), dtype=float)
        for char in charset:
            for s in range(len(self.states)):
                count = dict_emit[self.states[s]].get(char, 0)
                B[s][charset[char]] = MIN if count == 0 else math.log(1.0 * count / dict_start[self.states[s]])
        for s in range(len(self.states)):
            B[s][-1] = MIN
        return B

    def viterbi(self, observe):
        observe_len = len(observe)
        observe = [(self.charset[observe[i]] if observe[i] in self.charset else -1) for i in range(observe_len)]

        states_len = len(self.states)

        path = np.zeros((observe_len, states_len), dtype=int)
        deltas = np.zeros((observe_len, states_len), dtype=float)

        deltas[0, :] = self.PI + self.B[:, observe[0]]
        path[0, :] = np.arange(states_len)

        for t in range(1, observe_len, 1):
            for i in range(states_len):
                tmp = deltas[t - 1, :] + self.A[:, i]
                deltas[t][i], path[t][i] = np.max(tmp) + self.B[i][observe[t]], np.argmax(tmp)
        state = np.zeros(observe_len, dtype=int)
        state[observe_len - 1] = 3 if deltas[observe_len - 1][3] >= deltas[observe_len - 1][2] else 2
        for i in range(observe_len - 2, -1, -1):
            state[i] = path[i + 1][state[i + 1]]
        return [self.states[i] for i in state]


if __name__ == '__main__':
    hmm = HMMModel(['../DATASET/dataset1/train.utf8', '../DATASET/dataset2/train.utf8'])
    output = hmm.viterbi(list(
        '方鸣认为，策划是对图书进行整体包装，不单单是指书稿内容和装帧设计，还包括市场等。'
        '他认为，图书也像人穿衣一样，不仅料子要好，还要做工精致。只有追求精致、精道、精当、精美，才能反映出书的内在价值。譬如《东方书林之旅》，首次采用黄色胶板纸作为内瓤，柔和的米黄色给人一种读书人的儒雅感觉，油然生出一种淡淡的书香气；外封以高级乌光铜板纸精制而成，追求高品质；'
        '以优良的牛皮纸制作内封，通过“牛皮纸情结”，使那些从小就习惯用牛皮纸包书皮的学子们产生怀旧感和亲近感。'))
    print(dutil.precision(output, list(
        'SSSBEBESSSSSSBIIESSSSSBESSBIESSSBIESSSBIESSSBIESSSBIESSSBIESSSSBEBEBEBESSSSSSSBEBESSSSBESSSSSSSBESSBEBEBESBIEBESSBESSSSSBEBESBEBESSSSSSBESBEBESBEBESBEBESBEBESBEBESBEBESBESSBESBESBESBESBESBEBEBEBEBESSSBEBESSSSBESSSBEBESBEBESBESBESSBEBEBESBEBEBEBEBEBEBES')))
    pass
