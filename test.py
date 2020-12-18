import numpy as np
import json as json
from collections import Counter
import pickle

'''
def avg_alpha( alphas, T):
    sum = Counter(alphas[0])
    for i in range(1, T, 1):
        sum += Counter(alphas[i])
    sum_dict = dict(sum)
    for key in sum_dict:
        sum_dict[key] = sum_dict[key]/2
    return sum_dict

if __name__ == '__main__':
    a_dict = {('S', '中'): 111, 2: [23, 1, 4], '23': {1: 2, 'd': 'sad'}}
    file = open('pickle_example.pickle', 'wb')
    pickle.dump(a_dict, file)

    with open('pickle_example.pickle', 'rb') as file:
        a_dict1 = pickle.load(file)

    print(a_dict1)

    friends = [{"name": "王虎", "name1": "张二", "name2": "姚晨"}, {}]
    print(json.dumps(friends, ensure_ascii=False, indent=3))
    json.dump(friends, open(r'./test.json', 'w'), ensure_ascii=False, indent=3)
    friend = json.load(open(r'./test.json'))
    print(friend)

    a = np.array([[1, 2], [3, 4]])
    b = np.array([1, 2])
    print(a + b)

    a = np.array([[2, 4, 6, 1], [1, 5, 2, 9]])
    print(np.argmax(a))
    print(np.argmax(a, axis=0))  # 竖着比较，返回行号
    print(np.argmax(a, axis=1))  # 横着比较，返回列号

    path = [None for i in range(4)]
    print(path)

    score = np.arange(16).reshape(4, 4)
    print(np.argmax(score[:, 1]))
    # TODO 如何取下标
    sj = np.argmax(score[-1])
    states = ['B', 'I', 'E', 'S']
    print(sj)
    print(states[int(sj)])
    # state_score = np.arange(16).reshape(4, 4)
    # print(state_score)
    # print(score)
    # print(score.shape)
    # score = score.reshape(1, -1).T
    # print(state_score + score)
    # tmp = state_score + score
    # print(np.max(tmp[:, 1]), np.argmax(tmp[:, 1]))

    u_score = np.arange(4)
    print(u_score)
    b_score = np.arange(16).reshape(4, -1)
    print(b_score)
    # print(u_score + b_score)

    score = np.arange(4)
    state_score = np.arange(16).reshape(4, -1)
    state_score += score.reshape(1, -1).T
    print(state_score)
    print(np.max(state_score[:, 1]), np.argmax(state_score[:, 1]))

    x = {'a': 1, 'b': 2, 'c': 3}
    y = {'c': 4, 'd': 5}
    z = avg_alpha([x,y], 2)
    print(z)
    # X, Y = Counter(x), Counter(y)
    # z = dict(X + Y)
    # print(z)
    # for key in z:
    #     z[key] = z[key] / 2
    # print(z)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # 输入单词用一个维度为3的向量表示, 隐藏层的一个维度3，仅有一层的神经元，
# 记住就是神经元，这个时候神经层的详细结构还没确定，仅仅是说这个网络可以接受[seq_len,batch_size,3]的数据输入
# print(lstm.all_weights)

inputs = [torch.randn(1, 3) for _ in range(5)]
# 构造一个由5个单单词组成的句子 构造出来的形状是 [5,1,3]也就是明确告诉网络结构我一个句子由5个单词组成，
# 每个单词由一个1X3的向量组成，就是这个样子[1,2,3]
# 同时确定了网络结构，每个批次只输入一个句子，其中第二维的batch_size很容易迷惑人
# 对整个这层来说，是一个批次输入多少个句子，具体但每个神经元，就是一次性喂给神经元多少个单词。
print('Inputs:', inputs)

# 初始化隐藏状态
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
# print('Hidden:', hidden)
for i in inputs:
    # print('i:', i)
    # 将序列的元素逐个输入到LSTM，这里的View是把输入放到第三维，看起来有点古怪，
    # 回头看看上面的关于LSTM输入的描述，这是固定的格式，以后无论你什么形式的数据，
    # 都必须放到这个维度。就是在原Tensor的基础之上增加一个序列维和MiniBatch维，
    # 这里可能还会有迷惑，前面的1是什么意思啊，就是一次把这个输入处理完，
    # 在输入的过程中不会输出中间结果，这里注意输入的数据的形状一定要和LSTM定义的输入形状一致。
    # 经过每步操作,hidden 的值包含了隐藏状态的信息
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    print('out1:', out)
    print('hidden2:', hidden)
# 另外, 我们还可以一次对整个序列进行训练. LSTM 返回的第一个值表示所有时刻的隐状态值,
# 第二个值表示最近的隐状态值 (因此下面的 "out"的最后一个值和 "hidden" 的值是一样的).
# 之所以这样设计, 是为了通过 "out" 的值来获取所有的隐状态值, 而用 "hidden" 的值来
# 进行序列的反向传播运算, 具体方式就是将它作为参数传入后面的 LSTM 网络.

# 增加额外的第二个维度
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
print('inputs:', inputs)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print('out2', out)
print('hidden3', hidden)
