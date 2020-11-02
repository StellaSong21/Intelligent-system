import numpy as np
from PART1.algorithm import SinUtil as util, LossFunction as lf, BPNetwork as BPN, ActiveFunction as af
import matplotlib.pyplot as plt
import os

'''
模型存储位置：record/test_learn_rate/...
'''

layers_list = [1, 100, 1]
learn_w = [0.1, 0.05, 0.02, 0.01, 0.005]
dir = 'record/test_learn_rate/'

if __name__ == '__main__':
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Loss')
    ax1.set_xlabel('train_num')
    ax1.set_ylabel('loss')

    train_loop = 40
    train_loss = np.zeros(train_loop, dtype='float')

    train_len = 1000
    train_input, train_target, train_output, train_each_loss = util.get_set(train_len)

    dev_len = 300
    dev_input, dev_target, dev_output, dev_each_loss = util.get_set(dev_len)

    for i in range(len(learn_w)):
        alpha = learn_w[i]
        bp = BPN.BPNetwork(layers_list,
                           learn_w=alpha, learn_b=alpha,
                           weight=0, bias=-0.2,
                           active_func=af.Sigmoid(), loss_func=lf.squared_error,
                           softmax=False, sin=True)
        for j in range(train_loop):
            for k in range(train_len):
                train_each_loss[k] = bp.train(train_input[k], train_target[k])[1]
                pass
            train_loss[j] = np.mean(train_each_loss)
        pass

        bp.save(os.path.join(dir, str(alpha)))
        bp.load(os.path.join(dir, str(alpha)))

        for j in range(dev_len):
            dev_output[j], dev_each_loss[j] = bp.query(dev_input[j], dev_target[j])
        dev_loss = np.mean(dev_each_loss)

        ax1.plot(range(train_loop), train_loss, '-', label='alpha=' + str(alpha) + ", dev_loss=" + str(dev_loss))

    ax1.legend()
    plt.show()
