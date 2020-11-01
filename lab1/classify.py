import numpy as np
from algorithm import LossFunction as lf, ActiveFunction as af, BPNetwork as BPN
from tqdm import tqdm
import pandas as pd
from algorithm import ImgUtil as util
import random, os

pic_size = 28 * 28
total_class = 12
learn_w = 0.01
learn_b = 0.005
dir = './record/classify'

if __name__ == '__main__':
    bp = BPN.BPNetwork([pic_size, 250, total_class],
                       learn_w=learn_w, learn_b=learn_b,
                       weight=0, bias=-0.1,
                       active_func=af.Sigmoid(), loss_func=lf.cross_entropy,
                       softmax=True, sin=False)

    train_set, test_set = util.k_mixture(path='./DATASET/train', total_class=1, dev_count=0)

    train_loop = 30

    try:
        with tqdm(total=len(train_set) * train_loop) as pbar:
            for num in range(train_loop):
                random.shuffle(train_set)
                bp.set_learn_rate(pow(0.8, num / 2) * learn_w,
                                  pow(0.8, num / 2) * learn_b)
                for item in train_set:
                    train_list = util.read_img_as_list(item[1])
                    target = util.get_target_list(item[0], total_class)
                    data, loss = bp.train(train_list, target)
                    pbar.update(1)
                    pass
                # bp.save(os.path.join(dir, str(num)))
                pass
            pass
    except KeyboardInterrupt:
        pbar.close()
        raise

    bp.save(dir)
