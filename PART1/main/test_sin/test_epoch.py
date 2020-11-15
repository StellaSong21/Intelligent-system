import numpy as np
from PART1.algorithm import SinUtil as util, LossFunction as lf, BPNetwork as BPN, ActiveFunction as af
import matplotlib.pyplot as plt
import os

learn_w = 0.05
learn_b = 0.05

dir = '../../record/test_sin/epoch/'

if __name__ == '__main__':

    train_loop = 150
    train_loss = np.zeros(train_loop, dtype='float')
    dev_loss = np.zeros(train_loop, dtype='float')

    train_len = 1000
    train_input, train_target, train_output, train_each_loss = util.get_set(train_len)

    dev_len = 300
    dev_input, dev_target, dev_output, dev_each_loss = util.get_set(dev_len)

    bp = BPN.BPNetwork([1, 50, 1],
                       learn_w=learn_w, learn_b=learn_w,
                       weight=0, bias=-0.2,
                       active_func=af.Sigmoid(), loss_func=lf.squared_error,
                       softmax=False, sin=True)
    for j in range(train_loop):
        for k in range(train_len):
            train_each_loss[k] = bp.train(train_input[k], train_target[k])[1]
            pass
        train_loss[j] = np.mean(train_each_loss)

        bp.save(os.path.join(dir, str(j + 1)))
        bp.load(os.path.join(dir, str(j + 1)))

        for m in range(dev_len):
            dev_output[m], dev_each_loss[m] = bp.query(dev_input[m], dev_target[m])
        dev_loss[j] = np.mean(dev_each_loss)
    pass

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Epoch Loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    ax1.plot(range(train_loop), dev_loss, '-', label='loss')
    ax1.plot(range(train_loop), [0.01] * train_loop, '-', label='loss=0.01')
    ax1.legend()
    plt.show()
