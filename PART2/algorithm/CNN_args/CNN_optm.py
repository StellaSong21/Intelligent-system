import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from PART2.algorithm import Util
from PART2.algorithm import CNN

if __name__ == '__main__':
    # 存储模型的路径
    path = '../../record/CNN/CNN_optm'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 训练集
    trainset = torchvision.datasets.ImageFolder('../../DATASET/train',
                                                transform=transform)
    trainloader = DataLoader(trainset, batch_size=4,
                             shuffle=True, num_workers=0)

    # 测试集
    testset = torchvision.datasets.ImageFolder('../../DATASET/tests',
                                               transform=transform)
    testloader = DataLoader(testset, batch_size=4,
                            shuffle=True, num_workers=0)
    # 卷积神经网络
    labels = ['SGD', 'Momentum', 'RMSProp', 'Adam']
    net_SGD = CNN.CNN()
    net_Momentum = CNN.CNN()
    net_RMSProp = CNN.CNN()
    net_Adam = CNN.CNN()
    nets = [net_SGD, net_Momentum, net_RMSProp, net_Adam]
    # 损失函数 [pytorch 中交叉熵包括了 softmax 层]
    criterion = nn.CrossEntropyLoss()
    # 优化器，调整参数
    LR = 0.001
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)  # 是SGD的改进，加了动量效果
    opt_RMSProp = torch.optim.RMSprop(net_RMSProp.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSProp, opt_Adam]

    # 训练参数
    count = 50

    if not os.path.exists(path):
        os.mkdir(path)

    for label in labels:
        if not os.path.exists(os.path.join(path, label)):
            os.mkdir(os.path.join(path, label))

    # 训练过程
    for label, net, optimizer in zip(labels, nets, optimizers):
        for epoch in range(count):
            train_loss, train_acc = Util.calculate(trainset, trainloader, net, criterion, True, optimizer)
            torch.save(net.state_dict(), os.path.join(path, label, str(epoch + 1) + '.pt'))
            net.load_state_dict(torch.load(os.path.join(path, label, str(epoch + 1) + '.pt')))
            net.eval()
            test_loss, test_acc = Util.calculate(testset, testloader, net, criterion)
            print("[%d/%d] Loss: %.5f, Acc: %.2f%%" % (epoch + 1, count, test_loss, 100 * test_acc))

    print('Finished Training')
