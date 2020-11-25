from tqdm import tqdm
from PART1.algorithm import ImgUtil as util, LossFunction as lf, BPNetwork as BPN, ActiveFunction as af
import numpy as np

pic_size = 28 * 28
total_class = 12
learn_w = 0.01
learn_b = 0.005
dir = '../record/classify'

if __name__ == '__main__':
    bp = BPN.BPNetwork([pic_size, 250, total_class],
                       learn_w=learn_w, learn_b=learn_b,
                       weight=0, bias=-0.1,
                       active_func=af.Sigmoid(), loss_func=lf.cross_entropy,
                       softmax=True, sin=False)
    bp.load(dir)
    test_set = util.get_test_set('../DATASET/test')

    result = np.zeros(len(test_set), dtype=int)

    try:
        with tqdm(total=len(test_set)) as pbar:
            for i in range(len(test_set)):
                test_list = util.read_img_as_list(test_set[i])
                # print(test_list)
                data = bp.test(test_list)
                result[i] = np.argmax(data) + 1
                pbar.update(1)
                pass
            pass
        pass
    except KeyboardInterrupt:
        pbar.close()
        raise

    util.write_result(result)
