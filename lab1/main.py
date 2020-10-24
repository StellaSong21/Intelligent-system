import numpy as np
from matplotlib import pylab
import BPNetwork as BPN
import ActiveFunction as af
import LossFunction as lf

if __name__ == '__main__':
    '''
    初始化神经网络的结构
    神经网络层数的选择：
    根据经验公式：
    h = (n+m)
    '''
    bp = BPN.BPNetwork([1, 100, 1], active_func=af.Sigmoid(), loss_func=lf.MSE, softmax=False, sin=True)
    trainNum = 100
    loop = 40
    loss_arr = np.zeros(trainNum, dtype='float')
    train_loss = np.zeros(loop, dtype='float')
    trainInput = np.random.rand(trainNum) * 2 * np.pi - 1 * np.pi
    trainTarget = np.sin(trainInput)
    for iter in range(loop):
        for num in range(trainNum):
            err = bp.train(trainInput[num], trainTarget[num])
            loss_arr[num] = err
            # bp.check()
            pass
        print(loss_arr)
        train_loss[iter] = np.mean(loss_arr)
        pass

    print(train_loss)

