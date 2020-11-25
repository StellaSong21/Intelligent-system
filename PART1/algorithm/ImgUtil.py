import numpy as np
import matplotlib.image as mplimg
import os
import random


def get_target_list(target, total_class=12):
    p_distribution = np.zeros(total_class) + 0.01
    p_distribution[target - 1] = 0.99
    return p_distribution


# 将图片转化为对应的[二维]数组形式
def img_to_array(img, pic_size=28 * 28):
    R = img[:, :, 0]
    arr = ((np.array(R).reshape(pic_size) / 255.0 * 0.99) + 0.01)[None]
    return arr


# 读取图片信息
def read_img_as_list(path, pic_size=28 * 28):
    img = mplimg.imread(path)
    return img_to_array(img, pic_size)


# k 折数据划分将数据集划分为 train set 和 dev set
def k_mixture(path='./DATASET/train', total_class=10, dev_count=2):
    dir = os.listdir(path)
    train_set = []
    dev_set = []
    count = 0
    for subdir in dir:
        target = int(subdir)
        cur_path = os.path.join(path, subdir)
        img_list = os.listdir(cur_path)
        for img in img_list:
            img_path = os.path.join(cur_path, img)
            if count % total_class < dev_count:
                dev_set.append((target, img_path))
            else:
                train_set.append((target, img_path))
                pass
            count += 1
            pass
        pass
    random.shuffle(train_set)
    random.shuffle(dev_set)
    return train_set, dev_set


# 取其中2000个作为 dev set
def len_mixture(path='./DATASET/train', dev_segment=2000):
    dir = os.listdir(path)
    total_set = []
    for subdir in dir:
        target = int(subdir)
        cur_path = os.path.join(path, subdir)
        img_list = os.listdir(cur_path)
        for img in img_list:
            img_path = os.path.join(cur_path, img)
            total_set.append((target, img_path))
            pass
        pass
    random.shuffle(total_set)
    dev_set = total_set[0:dev_segment]
    train_set = total_set[dev_segment:]
    return train_set, dev_set


def get_test_set(path, start=1, end=1800):
    set = []
    for item in range(start, end + 1):
        file = str(item) + ".bmp"
        cur_path = os.path.join(path, file)
        set.append(cur_path)
        pass
    return set


def write_result(result, file='../record/interview/res.txt'):
    f = open(file, 'w')
    for res in result:
        f.write(str(res) + '\n')
        pass
    f.close()
    pass
