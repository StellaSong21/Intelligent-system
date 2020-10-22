import numpy as np
import math
from matplotlib import pylab
import BPNetwork

bp = BPNetwork.BPNetwork()


# 拟合sinx函数的训练集
def get_train(n):
    train_set = np.random.rand(n) * 2 * np.pi - 1 * np.pi
    train_res = np.sin(train_set)
    return train_set, train_res


# 拟合sinx函数的测试集
def get_test(n):
    test_set = np.random.rand(n) * 2 * np.pi - 1 * np.pi
    test_res = np.sin(test_set)
    return test_set, test_res


'''
反向传播算法对训练集进行训练
repeat：
    向前传播，计算output_cells
    反向传播，计算δ, 更新weight和bias
'''


def back_propagate_train(input, expects, learn=0.05, limit=10000):
    for j in range(limit):
        print(j)
        for i in range(len(input)):
            bp.forward_propagate(input[i])
            bp.calculate_delta(expects[i])
            bp.update_w(learn)
            bp.update_b(learn)


'''
测试
'''
if __name__ == '__main__':
    '''
    初始化神经网络的结构
    神经网络层数的选择：
    根据经验公式：
    h = (n+m)
    '''
    bp.setup(1, 1, [10, 10])
    # 初始化学习率，训练次数
    learn = 0.05
    times = 40
    train_set, train_res = get_train(3000)

    back_propagate_train(train_set, train_res, learn, times)

    test_set, test_res = get_test(300)
    average_loss, predicate_res = bp.get_average_loss(test_set, test_res)
    print(average_loss)

    # 画图
    pylab.plt.scatter(train_set, train_res, marker='x', color='g', label='train set')

    x = np.arange(-1 * np.pi, np.pi, 0.01)
    x = x.reshape((len(x), 1))
    y = np.sin(x)
    pylab.plot(x, y, label='standard sinx')

    pylab.plt.scatter(test_set, predicate_res, label='predicate sinx, learn = ' + str(learn) + ' times = ' + str(times),
               linestyle='--',
               color='r')

    pylab.plt.legend(loc='best')
    pylab.plt.show()


