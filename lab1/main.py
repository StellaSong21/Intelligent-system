import numpy as np
import BPNetwork as BPN
import ActiveFunction as af
import LossFunction as lf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    初始化神经网络的结构
    神经网络层数的选择：
    根据经验公式：
    h = (n+m)
    '''
    bp = BPN.BPNetwork([1, 100, 100, 100, 100, 1], active_func=af.Sigmoid(), loss_func=lf.MSE, softmax=False, sin=True)
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
            train_each_loss[i] = bp.calculate(trainInput[i], trainTarget[i])
            # bp.check()
            pass
        train_loss[iter] = np.mean(train_each_loss)

        if iter % dev_loop == dev_loop - 1:
            for j in range(devNum):
                dev_each_loss[j] = bp.calculate(devInput[j], devTarget[j], False)
                pass
            dev_loss[int(iter / dev_loop) + 1] = np.mean(dev_each_loss)
            pass
        pass

    print(train_loss)
    print(dev_loss)

    # 创建画板
    fig = plt.figure()
    # 创建画纸
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.set_title('Fit Result')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.plot(trainInput, trainTarget, 'x', label='Expect')
    ax1.legend()

    ax2.set_title('Epoch Loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.plot(range(train_loop), train_loss, '-', label='train')
    ax2.plot(range(dev_loop, len(dev_loss) * dev_loop, dev_loop), dev_loss[1:], '-', label='dev', alpha=0.8)
    ax2.legend()
    plt.show()