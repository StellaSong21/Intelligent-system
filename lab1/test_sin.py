import numpy as np
from algorithm import LossFunction as lf, ActiveFunction as af, BPNetwork as BPN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    初始化神经网络的结构
    神经网络层数的选择：
    根据经验公式：
    h = (n+m)
    '''
    bp = BPN.BPNetwork([1, 100, 100, 1], active_func=af.Sigmoid(), loss_func=lf.squared_error, softmax=False, sin=True)
    trainNum = 4000
    train_each_loss = np.zeros(trainNum, dtype='float')
    trainInput = np.random.rand(trainNum) * 2 * np.pi - 1 * np.pi
    trainTarget = np.sin(trainInput)

    train_loop = 40
    train_loss = np.zeros(train_loop, dtype='float')

    devNum = 300
    dev_each_loss = np.zeros(devNum, dtype='float')
    devInput = np.random.rand(devNum) * 2 * np.pi - 1 * np.pi
    devTarget = np.sin(devInput)

    dev_loop = 5
    dev_loss = np.zeros(int(train_loop / dev_loop) + 1, dtype='float')

    for iter in range(train_loop):
        for i in range(trainNum):
            train_each_loss[i] = bp.train(trainInput[i], trainTarget[i])[1]
            # bp.check()
            pass
        train_loss[iter] = np.mean(train_each_loss)

        if iter % dev_loop == dev_loop - 1:
            for j in range(devNum):
                dev_each_loss[j] = bp.query(devInput[j], devTarget[j])[1]
                pass
            dev_loss[int(iter / dev_loop) + 1] = np.mean(dev_each_loss)
            pass
        pass

    bp.save('./record/BPNetwork/create')
    bpp = BPN.BPNetwork([1, 100, 100, 1], active_func=af.Sigmoid(), loss_func=lf.squared_error, softmax=False, sin=True)
    bpp.load('./record/BPNetwork/create')

    # test
    testNum = 3000
    testInput = np.random.rand(testNum) * 2 * np.pi - 1 * np.pi
    testTarget = np.sin(testInput)
    testOutput = np.zeros(testNum, dtype='float')
    test_each_loss = np.zeros(testNum, dtype='float')
    for i in range(testNum):
        testOutput[i], test_each_loss[i] = bpp.query(testInput[i], testTarget[i])
        pass
    test_loss = np.mean(test_each_loss)
    print(test_loss)

    # 创建画板
    fig = plt.figure()

    # 创建画纸
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # test result
    ax1.set_title('Fit Result')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.plot(testInput, testTarget, 'x', label='Expect')
    ax1.plot(testInput, testOutput, 'o', label='Test')
    ax1.legend()

    # train loss
    ax2.set_title('Epoch Loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.plot(range(train_loop), train_loss, '-', label='train')
    ax2.plot(range(dev_loop, len(dev_loss) * dev_loop, dev_loop), dev_loss[1:], '-', label='dev', alpha=0.8)
    ax2.legend()

    plt.show()
