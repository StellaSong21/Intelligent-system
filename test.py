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
    print(a+b)

    c = None
    list = [c, 'a']
    print(list)