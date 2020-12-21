# """
# 1. π 矩阵：统计开始概率，即 B 和 S 的概率，(统计S和B分别出现的次数)
# 2. A 矩阵：B->[M|E], M->[M|E], E->[S|B], S->[B|S]，(统计B->M和B->E的次数)
# 3. B 矩阵：B->[a|b|c|...], M->[a|b|c|...], E->[a|b|c|...], S->[a|b|c|...]，统计a->B|M|E|S的次数，字的排序
# 4. 数据平滑处理，+1 或者 古德-图灵
# 5. 统计数据持久化
# {'中':{'B':0,'I':0,'E':0,'S':0}}
# {'B':0, 'I':0, 'E':0, 'S':0}
# {'B':{'I':0,'E':0}, ...}
# 6. 标点符号等非中文的处理
# """
# import math
#
# '''
# 1. 统计字符集后index查找顺序
# 2. 统计过程中统计输出顺序list，但是需要边统计边确定是否在之前已经统计过
# '''
#
#
# def stat_charset(filepaths, encoding='utf8'):
#     charset = dict()
#     training_data = []
#     for filepath in filepaths:
#         sub_sequence = []
#         sub_tags = []
#         file = open(filepath, encoding=encoding)
#         for line in file:
#             x = line.split()
#             if len(x) == 0:
#                 if len(sub_sequence) > 0:
#                     training_data.append((sub_sequence, sub_tags))
#                 sub_sequence = []
#                 sub_tags = []
#                 continue
#             if x[0] not in charset:
#                 charset[x[0]] = len(charset)
#             sub_sequence.append(x[0])
#             sub_tags.append(x[1])
#             pass
#     return charset, training_data
#
#
# # 返回字符集、观察集
# # 字符集，set类型；观察集，list
# # 考虑空格为分段，那么每一段都可以从头开始分析，因此此处在观察集和标签集中保留了' '和' '
# def stat_charset1(filepaths, encoding='utf8'):
#     space = []
#     charset = set()
#     observes = []
#     tags = []
#     for filepath in filepaths:
#         i = -1
#         space_tmp = [0]
#         observe_tmp = []
#         tag_tmp = []
#         file = open(filepath, encoding=encoding)
#         for line in file:
#             i += 1
#             x = line.split()
#             if len(x) == 0:
#                 space_tmp.append(i)
#                 observe_tmp.append(' ')
#                 tag_tmp.append(' ')
#                 continue
#             charset.add(x[0])
#             observe_tmp.append(x[0])
#             tag_tmp.append(x[1])
#             pass
#         space_tmp.append(i)
#         observes.append(observe_tmp)
#         tags.append(tag_tmp)
#         space.append(space_tmp)
#     return charset, observes, tags, space
#
#
# def take_second(elem):
#     return elem[1]
#
#
# def test_list_sort():
#     random = [(2, 2), (3, 4), (4, 1), (1, 3)]
#     random.sort(key=take_second)
#     print('排序列表：', random)
#
#
# import numpy as np
#
#
# def sum_log(arr):
#     max_value = np.max(arr)
#     return math.log(np.sum(np.exp(arr - max_value))) + max_value
#
#
# if __name__ == '__main__':
#     charset1, training_data = stat_charset(['../../DATASET/dataset1/train.utf8', '../../DATASET/dataset2/train.utf8'])
#     print(training_data)
