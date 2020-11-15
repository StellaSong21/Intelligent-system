import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from PART2.algorithm import Util
from PART2.algorithm import CNN
from PART2.algorithm import ResNet

if __name__ == '__main__':
    CNN_trainset = torchvision.datasets.ImageFolder('../DATASET/train',
                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                                  transforms.Normalize(
                                                                                      [0.485, 0.456, 0.406],
                                                                                      [0.229, 0.224, 0.225])]))
    CNN_testset = torchvision.datasets.ImageFolder('../DATASET/tests',
                                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                                 transforms.Normalize(
                                                                                     [0.485, 0.456, 0.406],
                                                                                     [0.229, 0.224, 0.225])]))

    # 训练集
    ResNet_trainset = torchvision.datasets.ImageFolder('../DATASET/train',
                                                       transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                                     transforms.ToTensor(),
                                                                                     transforms.RandomHorizontalFlip(),
                                                                                     transforms.Normalize(
                                                                                         [0.485, 0.456, 0.406],
                                                                                         [0.229, 0.224, 0.225])]))

    # 测试集
    ResNet_testset = torchvision.datasets.ImageFolder('../DATASET/tests',
                                                      transform=transforms.Compose([transforms.Resize(256),
                                                                                    transforms.CenterCrop(224),
                                                                                    transforms.ToTensor(),
                                                                                    transforms.Normalize(
                                                                                        [0.485, 0.456, 0.406],
                                                                                        [0.229, 0.224, 0.225])]))

    labels = ['SGD, 4', 'Momentum, 8', 'RMSProp, 80', 'Adam, 100', 'ResNet, 4']
    nets = [CNN.CNN(), CNN.CNN(), CNN.CNN(), CNN.CNN(), ResNet.ResNetModel(12, [2, 2, 2, 2], False)]
    batch_size = [4, 8, 80, 100, 4]
    root = ['../record/CNN/CNN_optm4/SGD', '../record/CNN/CNN_optm8/Momentum',
            '../record/CNN/CNN_optm80/RMSProp', '../record/CNN/CNN_optm100/Adam', '../record/ResNet/18']
    trainsets = [CNN_trainset, CNN_trainset, CNN_trainset, CNN_trainset, ResNet_trainset]
    testsets = [CNN_testset, CNN_testset, CNN_testset, CNN_testset, ResNet_testset]
    trainloaders = []
    testloaders = []

    for i in range(len(labels)):
        trainloaders.append(DataLoader(trainsets[i], batch_size=batch_size[i],
                                       shuffle=True, num_workers=0))
        testloaders.append(DataLoader(testsets[i], batch_size=batch_size[i],
                                      shuffle=True, num_workers=0))

    # 损失函数 [pytorch 中交叉熵包括了 softmax 层]
    criterion = nn.CrossEntropyLoss()

    # 训练参数
    count = 30

    loss = np.zeros(count, dtype=float)
    acc = np.zeros(count, dtype=float)

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.set_title('Loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    ax2.set_title('Accuracy')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('acc')

    # 训练过程
    for label, path, net, testset, testloader in zip(labels, root, nets, testsets, testloaders):
        for epoch in range(count):
            net.load_state_dict(torch.load(os.path.join(path, str(epoch + 1) + '.pt')))
            net.eval()
            test_loss, test_acc = Util.calculate(testset, testloader, net, criterion)
            loss[epoch], acc[epoch] = test_loss, test_acc
            print("[%d/%d] Loss: %.5f, Acc: %.2f%%" % (epoch + 1, count, test_loss, 100 * test_acc))

        ax1.plot(range(count + 1)[1:], loss, label=label)
        ax2.plot(range(count + 1)[1:], acc, label=label)

    ax1.legend()
    ax2.legend()

    plt.show()

    print('Finished Training')
