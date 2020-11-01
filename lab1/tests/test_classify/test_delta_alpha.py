import numpy as np
from algorithm import BPNetwork as BPN, LossFunction as lf, ActiveFunction as af, ImgUtil as util
from tqdm import tqdm
import pandas as pd
import random, os
import matplotlib.pyplot as plt

'''
模型存储位置：record/test_delta_alpha/...
'''

pic_size = 28 * 28
total_class = 12

layers_list = [pic_size, 250, total_class]
learn_w = 0.01
learn_b = 0.005
delta_alpha = ['pow(0.6,(num)/2)', 'pow(0.6,(num)/5)', 'pow(0.6,(num)/10)',
               'pow(0.8,(num)/2)', 'pow(0.8,(num)/5)', 'pow(0.8,(num)/10)',
               '0.8']
dir = 'record/test_delta_alpha/'

if __name__ == '__main__':
    train_set, dev_set = util.k_mixture(path='../../DATASET/train', total_class=10, dev_count=1)
    train_loop = 20
    hit_rate = np.zeros(len(delta_alpha))
    loss_arr = np.zeros(train_loop)

    # 创建画板
    fig = plt.figure()

    # 创建画纸
    ax1 = fig.add_subplot(1, 1, 1)

    # test result
    ax1.set_title('Loss')
    ax1.set_xlabel('train_num')
    ax1.set_ylabel('loss')

    for i in range(len(delta_alpha)):
        bp = BPN.BPNetwork(layers_list,
                           active_func=af.Sigmoid(), loss_func=lf.cross_entropy,
                           softmax=True, sin=False)

        try:
            with tqdm(total=len(train_set) * train_loop) as pbar:
                for num in range(train_loop):
                    bp.set_learn_rate(eval(delta_alpha[i]) * learn_w,
                                      eval(delta_alpha[i]) * learn_b)
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

        bp.save(os.path.join(dir, str(i)))
        bp.load(os.path.join(dir, str(i)))

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
        dataFrame.to_csv(os.path.join(dir, str(i), str(i) + ".csv"), sep=",", index=False)
        pass

        if len(dev_set) != 0:
            hit_rate[i] = 1.0 * score / len(dev_set)

        ax1.plot(range(train_loop), loss_arr, '-', label='delta=' + delta_alpha[i] + ', hit rate=' + str(hit_rate[i]))

    for k in range(len(delta_alpha)):
        print("delta_alpha = ", delta_alpha[k], "hit rate = ", hit_rate[k])

    ax1.legend()
    plt.show()
