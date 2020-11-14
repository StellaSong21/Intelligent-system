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
    path = '../../record/CNN/CNN_lr'
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
    net = CNN.CNN()
    # 损失函数 [pytorch 中交叉熵包括了 softmax 层]
    criterion = nn.CrossEntropyLoss()
    # 优化器，调整参数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)

    # 等间隔调整
    StepLR_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)
    # 按需间隔调整
    MultiStepLR_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1,
                                                                 last_epoch=-1)
    # 指数衰减调整
    ExponentialLR_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)
    # 余弦退火调整
    CosineAnnealingLR_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0,
                                                                             last_epoch=-1)
    # 自适应调整学习率
    ReduceLROnPlateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                                             patience=10,
                                                                             verbose=False, threshold=0.0001,
                                                                             threshold_mode='rel',
                                                                             cooldown=0, min_lr=0, eps=1e-08)
    schedulers = [StepLR_scheduler, MultiStepLR_scheduler, ExponentialLR_scheduler,
                  CosineAnnealingLR_scheduler, ReduceLROnPlateau_scheduler]
    labels = ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau']

    # 训练参数
    count = 50

    if not os.path.exists(path):
        os.mkdir(path)

    for label in labels:
        if not os.path.exists(os.path.join(path, label)):
            os.mkdir(os.path.join(path, label))

    # 训练过程
    for label, scheduler in zip(labels, schedulers):
        for epoch in range(count):
            # if label is not 'ReduceLROnPlateau':
            #     scheduler.step()
            train_loss, train_acc = Util.calculate(trainset, trainloader, net, criterion, True, optimizer)
            if label is 'ReduceLROnPlateau':
                scheduler.step(train_loss)
            else:
                scheduler.step()
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
            torch.save(net.state_dict(), os.path.join(path, label, str(epoch + 1) + '.pt'))
            net.load_state_dict(torch.load(os.path.join(path, label, str(epoch + 1) + '.pt')))
            net.eval()
            test_loss, test_acc = Util.calculate(testset, testloader, net, criterion)
            print("[%d/%d] Loss: %.5f, Acc: %.2f%%" % (epoch + 1, count, test_loss, 100 * test_acc))

    print('Finished Training')
