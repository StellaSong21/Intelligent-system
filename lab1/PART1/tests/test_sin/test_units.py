import numpy as np
from PART1.algorithm import SinUtil as util, LossFunction as lf, BPNetwork as BPN, ActiveFunction as af
import matplotlib.pyplot as plt
import os

'''
模型存储位置：record/test_hidden/...
'''

hidden = [[10], [20], [50], [70], [100], [120], [150], [200]]
learn_w = 0.05
learn_b = 0.05

dir = 'record/test_hidden/'

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

    for i in range(len(hidden)):
        layers_list = np.column_stack(([1], hidden[i], [1])).reshape(-1)
        bp = BPN.BPNetwork(layers_list,
                           learn_w=learn_w, learn_b=learn_w,
                           weight=0, bias=-0.2,
                           active_func=af.Sigmoid(), loss_func=lf.squared_error,
                           softmax=False, sin=True)
        for j in range(train_loop):
            for k in range(train_len):
                train_each_loss[k] = bp.train(train_input[k], train_target[k])[1]
                pass
            train_loss[j] = np.mean(train_each_loss)
        pass

        bp.save(os.path.join(dir, str(hidden[i])))
        bp.load(os.path.join(dir, str(hidden[i])))

        for j in range(dev_len):
            dev_output[j], dev_each_loss[j] = bp.query(dev_input[j], dev_target[j])
        dev_loss = np.mean(dev_each_loss)

        ax1.plot(range(train_loop), train_loss, '-', label='hidden=' + str(hidden[i]) + ", dev_loss=" + str(dev_loss))

    ax1.legend()
    plt.show()
