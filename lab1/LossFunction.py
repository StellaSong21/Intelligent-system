import numpy as np

def MSE(x, y):
    return np.power((x - y), 2)

if __name__ == '__main__':
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    print(MSE(x, y))