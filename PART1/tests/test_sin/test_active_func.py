import numpy as np
from PART1.algorithm import SinUtil as util, LossFunction as lf, BPNetwork as BPN, ActiveFunction as af
import matplotlib.pyplot as plt
import os

'''
模型存储位置：record/active_func/...
'''

learn_w = 0.05
learn_b = 0.05
active_func = [af.Sigmoid(), af.Tanh(), af.ReLU()]

dir = 'record/active_func/'

if __name__ == '__main__':
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Loss')
    ax1.set_xlabel('train_num')
    ax1.set_ylabel('loss')

    train_loop = 1000
    train_loss = np.zeros(train_loop, dtype='float')

    train_len = 1000
    train_input, train_target, train_output, train_each_loss = util.get_set(train_len)

    dev_len = 300
    dev_input, dev_target, dev_output, dev_each_loss = util.get_set(dev_len)

    for i in range(len(active_func)):
        bp = BPN.BPNetwork([1, 50, 1],
                           learn_w=learn_w, learn_b=learn_w,
                           weight=0, bias=-0.2,
                           active_func=active_func[i], loss_func=lf.squared_error,
                           softmax=False, sin=True)
        for j in range(train_loop):
            for k in range(train_len):
                train_each_loss[k] = bp.train(train_input[k], train_target[k])[1]
                pass
            train_loss[j] = np.mean(train_each_loss)
        pass

        bp.save(os.path.join(dir, str(active_func[i])))
        bp.load(os.path.join(dir, str(active_func[i])))

        for j in range(dev_len):
            dev_output[j], dev_each_loss[j] = bp.query(dev_input[j], dev_target[j])
        dev_loss = np.mean(dev_each_loss)

        ax1.plot(range(train_loop), train_loss, '-',
                 label='active_func=' + str(active_func[i]) + ", dev_loss=" + str(dev_loss))

    ax1.legend()
    plt.show()
