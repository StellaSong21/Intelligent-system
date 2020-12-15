import numpy as np
import json as json

if __name__ == '__main__':
    friends = {"name": "王虎", "name1": "张二", "name2": "姚晨"}
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
    state_score +=  score.reshape(1, -1).T
    print(state_score)
    print(np.max(state_score[:,1]), np.argmax(state_score[:,1]))
