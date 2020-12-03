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
    print(dict_emit)
    return dict_start, dict_trans, dict_emit
    # return [π, A, B]
    # 可以直接返回次数，因为概率过小，不知道有没有必要


def take_second(elem):
    return elem[1]


def test_list_sort():
    random = [(2, 2), (3, 4), (4, 1), (1, 3)]
    random.sort(key=take_second)
    print('排序列表：', random)


if __name__ == '__main__':
    statics('../DATASET/dataset1/train.utf8')
    pass
