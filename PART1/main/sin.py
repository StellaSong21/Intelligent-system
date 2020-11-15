import numpy as np
from PART1.algorithm import SinUtil as util, LossFunction as lf, BPNetwork as BPN, ActiveFunction as af
import matplotlib.pyplot as plt

dir = '../record/sin'

if __name__ == '__main__':
    '''
    初始化神经网络的结构
    神经网络层数的选择：
    根据经验公式：
    h = (n+m)
    '''
    bp = BPN.BPNetwork([1, 50, 1],
                       learn_w=0.05, learn_b=0.05,
                       active_func=af.Sigmoid(), loss_func=lf.squared_error,
                       softmax=False, sin=True)

    train_loop = 150

    train_loss = np.zeros(train_loop, dtype='float')
    dev_loss = np.zeros(train_loop, dtype='float')

    train_len = 1000
    train_input, train_target, train_output, train_each_loss = util.get_set(train_len)

    test_len = 300
    test_input, test_target, test_output, test_each_loss = util.get_set(test_len)

    for i in range(train_loop):
        for j in range(train_len):
            train_each_loss[j] = bp.train(train_input[j], train_target[j])[1]
            pass
        train_loss[i] = np.mean(train_each_loss)

    bp.save(dir)
    bp.load(dir)

    for i in range(test_len):
        test_output[i], test_each_loss[i] = bp.query(test_input[i], test_target[i])
        pass
    test_loss = np.mean(test_each_loss)

    print("损失率：", test_loss)

    # 创建画板
    fig = plt.figure()

    # 创建画纸
    ax1 = fig.add_subplot(1, 1, 1)

    # test result
    ax1.set_title('Fit Result')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.plot(test_input, test_target, 'x', label='Expect')
    ax1.plot(test_input, test_output, 'o', label='Test')
    ax1.legend()
    plt.show()
