import os
import shutil

if __name__ == '__main__':
    imgs = './test'
    root = './tests'
    os.mkdir(root)
    for i in range(12):
        if not os.path.isdir(os.path.join(root, str(i + 1))):
            os.mkdir(os.path.join(root, str(i + 1)))
    img = os.listdir(imgs)
    img.sort(key=lambda x : int(x[:-4]))
    gold_file = open('gold.txt', 'r').readlines()
    for image, gold in zip(img[:1800], gold_file):
        shutil.copy(os.path.join(imgs, image), os.path.join(root, gold.strip()))
