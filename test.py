import numpy as np

if __name__ == '__main__':
    c = np.zeros((100, 2, 2), dtype=int)
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    c[0] = a + b
    print(np.argmax(b))
    pos = np.unravel_index(np.argmax(a), a.shape)
    print(pos)
    print(pos[1])
    for i in range(0):
        print('ss')
