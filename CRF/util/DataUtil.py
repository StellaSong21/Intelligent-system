"""
1. π 矩阵：统计开始概率，即 B 和 S 的概率，(统计S和B分别出现的次数)
2. A 矩阵：B->[M|E], M->[M|E], E->[S|B], S->[B|S]，(统计B->M和B->E的次数)
3. B 矩阵：B->[a|b|c|...], M->[a|b|c|...], E->[a|b|c|...], S->[a|b|c|...]，统计a->B|M|E|S的次数，字的排序
4. 数据平滑处理，+1 或者 古德-图灵
5. 统计数据持久化
{'中':{'B':0,'I':0,'E':0,'S':0}}
{'B':0, 'I':0, 'E':0, 'S':0}
{'B':{'I':0,'E':0}, ...}
6. 标点符号等非中文的处理
"""
import math


# 有监督学习，直接统计词频
def statics(filepath):
    dict_start = {'B': 0, 'I': 0, 'E': 0, 'S': 0}
    dict_trans = {'B': {'I': 0, 'E': 0},
                  'I': {'I': 0, 'E': 0},
                  'E': {'B': 0, 'S': 0},
                  'S': {'S': 0, 'B': 0}}
    dict_emit = {'B': {}, 'I': {}, 'E': {}, 'S': {}}
    file = open(filepath, encoding='utf8')
    last_state = None
    for line in file:
        x = line.split()
        if len(x) == 0:
            continue
        # dict_start
        dict_start[x[1]] += 1
        # dict_trans
        if last_state is not None:
            dict_trans[last_state][x[1]] += 1
        last_state = x[1]
        # dict_emit
        y = dict_emit.get(x[1])
        y[x[0]] = y[x[0]] + 1 if x[0] in y else 1
    print(dict_start)
    print('B: ', math.log(dict_start['B'] / (dict_start['B'] + dict_start['S'])))
    print('S: ', math.log(dict_start['S'] / (dict_start['B'] + dict_start['S'])))
    print(dict_trans)
    print('B->I: ', math.log(dict_trans['B']['I'] / (dict_trans['B']['I'] + dict_trans['B']['E'])))
    print('B->E: ', math.log(dict_trans['B']['E'] / (dict_trans['B']['I'] + dict_trans['B']['E'])))
    print('I->I: ', math.log(dict_trans['I']['I'] / (dict_trans['I']['I'] + dict_trans['I']['E'])))
    print('I->E: ', math.log(dict_trans['I']['E'] / (dict_trans['I']['I'] + dict_trans['I']['E'])))
    print('E->B: ', math.log(dict_trans['E']['B'] / (dict_trans['E']['B'] + dict_trans['E']['S'])))
    print('E->S: ', math.log(dict_trans['E']['S'] / (dict_trans['E']['B'] + dict_trans['E']['S'])))
    print('S->B: ', math.log(dict_trans['S']['B'] / (dict_trans['S']['B'] + dict_trans['S']['S'])))
    print('S->S: ', math.log(dict_trans['S']['S'] / (dict_trans['S']['B'] + dict_trans['S']['S'])))
    # print(dict_emit)
    return dict_start, dict_trans, dict_emit
    # return [π, A, B]
    # 可以直接返回次数，因为概率过小，不知道有没有必要


'''
1. 统计字符集后index查找顺序
2. 统计过程中统计输出顺序list，但是需要边统计边确定是否在之前已经统计过
'''


def stat_charset(filepaths, encoding='utf8'):
    i = 0
    charset = [dict(), dict()]  # 1:'中', '中':1
    observes = []
    indexes = []
    for filepath in filepaths:
        j = 0
        observe_tmp = []
        index_tmp = dict()
        file = open(filepath, encoding=encoding)
        for line in file:
            x = line.split()
            if len(x) == 0:
                continue
            if x[0] in charset[1]:
                observe_tmp.append(charset[1][x[0]])
            else:
                observe_tmp.append(i)
                charset[0][i] = x[0]
                charset[1][x[0]] = i
                i += 1
            pass
            if x[0] in index_tmp:
                index_tmp[x[0]].append(j)
            else:
                index_tmp[x[0]] = [j]
            j += 1
            pass
        observes.append(observe_tmp)
        indexes.append(index_tmp)
    return charset, observes, indexes


# 返回字符集、观察集
# 字符集，set类型；观察集，list
# 考虑空格为分段，那么每一段都可以从头开始分析，因此此处在观察集和标签集中保留了' '和' '
def stat_charset1(filepaths, encoding='utf8'):
    space = []
    charset = set()
    observes = []
    tags = []
    for filepath in filepaths:
        i = 0
        space_tmp = [-1]
        observe_tmp = []
        tag_tmp = []
        file = open(filepath, encoding=encoding)
        for line in file:
            x = line.split()
            if len(x) == 0:
                space_tmp.append(i)
                observe_tmp.append(' ')
                tag_tmp.append(' ')
                continue
            charset.add(x[0])
            observe_tmp.append(x[0])
            tag_tmp.append(x[1])
            i += 1
            pass
        space_tmp.append(i)
        observes.append(observe_tmp)
        tags.append(tag_tmp)
        space.append(space_tmp)
    return charset, observes, tags, space


def take_second(elem):
    return elem[1]


def test_list_sort():
    random = [(2, 2), (3, 4), (4, 1), (1, 3)]
    random.sort(key=take_second)
    print('排序列表：', random)


import numpy as np


def sum_log(arr):
    max_value = np.max(arr)
    return math.log(np.sum(np.exp(arr - max_value))) + max_value


if __name__ == '__main__':
    # charset1, observes = stat_charset(['../DATASET/dataset1/train.utf8', '../DATASET/dataset2/train.utf8'])
    # print(charset1)
    # print(observes[0][:100])

    # arr = np.arange(12 * 4).reshape((12, 4))
    # tmp = arr.reshape((12, -1))
    # print(arr)
    # for i in range(4):
    #     print(arr[:, i])
    #     print(sum_log(arr[:, i]) == sum_log(tmp[:, i]))

    print(math.log(0.0000000001))
    pass
