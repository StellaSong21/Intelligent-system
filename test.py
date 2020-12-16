import numpy as np
import json as json
from collections import Counter
import pickle

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

