import numpy as np
from PART1.algorithm import ImgUtil as util, LossFunction as lf, BPNetwork as BPN, ActiveFunction as af
from tqdm import tqdm
import pandas as pd
import random, os
import matplotlib.pyplot as plt

'''
模型存储位置：record/test_units/...
'''

pic_size = 28 * 28
total_class = 12

hidden = [[10], [50], [100], [150], [200], [250], [300], [400], [500], [650]]

learn_w = 0.01
learn_b = 0.005

dir = 'record/test_units/'

if __name__ == '__main__':
    train_set, dev_set = util.k_mixture(path='../../../DATASET/train', total_class=10, dev_count=1)
    train_loop = 10
    hit_rate = np.zeros(len(hidden))
    loss_arr = np.zeros(train_loop)

    # 创建画板
    fig = plt.figure()

    # 创建画纸
    ax1 = fig.add_subplot(1, 1, 1)

    # test result
    ax1.set_title('Loss')
    ax1.set_xlabel('train_num')
    ax1.set_ylabel('loss')

    for i in range(len(hidden)):
        layers_list = np.column_stack(([pic_size], hidden[i], [total_class])).reshape(-1)

        bp = BPN.BPNetwork(layers_list,
                           active_func=af.Sigmoid(), loss_func=lf.cross_entropy,
                           softmax=True, sin=False)

        try:
            with tqdm(total=len(train_set) * train_loop) as pbar:
                for num in range(train_loop):
                    bp.set_learn_rate(pow(0.8, num / 2) * learn_w,
                                      pow(0.8, num / 2) * learn_b)
                    random.shuffle(train_set)
                    loss_arr_in = []
                    for item in train_set:
                        input = util.read_img_as_list(item[1])
                        target = util.get_target_list(item[0], total_class)
                        data, loss = bp.train(input, target)
                        loss_arr_in.append(loss)
                        pbar.update(1)
                    loss_arr[num] = np.mean(loss_arr_in)
        except KeyboardInterrupt:
            pbar.close()
            raise

        bp.save(os.path.join(dir, str(hidden[i])))
        bp.load(os.path.join(dir, str(hidden[i])))

        score = 0
        dev_size = len(dev_set)
        result = np.zeros(dev_size, dtype='int')

        for j in range(dev_size):
            item = dev_set[j]
            input = util.read_img_as_list(item[1])
            target = util.get_target_list(item[0], total_class)
            res = bp.query(input, target)[0]
            result[j] = np.argmax(res) + 1
            if result[j] == item[0]:
                score += 1
            pbar.update(1)
            pass
        dataFrame = pd.DataFrame(
            {'target': [x[0] for x in dev_set], 'path': [x[1] for x in dev_set], 'result': result})
        dataFrame.to_csv(os.path.join(dir, str(hidden[i]), str(hidden[i]) + ".csv"), sep=",", index=False)
        pass

        if len(dev_set) != 0:
            hit_rate[i] = 1.0 * score / len(dev_set)

        ax1.plot(range(train_loop), loss_arr, '-', label='hidden=' + str(hidden[i]) + ', hit rate=' + str(hit_rate[i]))

    for k in range(len(hidden)):
        print("hidden = ", str(hidden[k]), "hit rate = ", hit_rate[k])

    ax1.legend()
    plt.show()
