import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


class ResNetModel(nn.Module):
    """
    实现通用的ResNet模块，可根据需要定义
    """

    def __init__(self, num_classes=12, layer_num=None):
        super(ResNetModel, self).__init__()

        # conv1
        if layer_num is None:
            layer_num = []
        self.pre = nn.Sequential(
            # in 224*224*3
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),  # 输入通道3，输出通道64，卷积核7*7*64，步长2,根据以上计算出padding=3
            # out 112*112*64
            # 数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(64),  # 输入通道C = 64
            nn.ReLU(inplace=True),  # inplace=True, 进行覆盖操作
            # out 112*112*64
            nn.MaxPool2d(3, 2, 1),  # 池化核3*3，步长2,计算得出padding=1;
            # out 56*56*64
        )

        self.residualBlocks1 = self.add_layers(64, 256, layer_num[0], 64)
        self.residualBlocks2 = self.add_layers(128, 512, layer_num[1], 256, 2)
        self.residualBlocks3 = self.add_layers(256, 1024, layer_num[2], 512, 2)
        self.residualBlocks4 = self.add_layers(512, 2048, layer_num[3], 1024, 2)

        self.fc = nn.Linear(2048, num_classes)

    def add_layers(self, inchannel, outchannel, nums, pre_channel=64, stride=1):
        layers = []

        # 传递每个block的shortcut，shortcut可以根据是否传递pre_channel进行推断

        # 添加首层,首层需要传递上一批 blocks 的 channel
        layers.append(BottleNeckBlock(inchannel, outchannel, pre_channel, stride))
        for i in range(1, nums):  # 添加n-1个剩余blocks，正常通道转换，不传递pre_channel
            layers.append(BottleNeckBlock(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.residualBlocks1(x)
        x = self.residualBlocks2(x)
        x = self.residualBlocks3(x)
        x = self.residualBlocks4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class BottleNeckBlock(nn.Module):
    '''
    定义resnet50的瓶颈结构
    '''

    def __init__(self, inchannel, outchannel, pre_channel=None, stride=1, shortcut=None):
        super(BottleNeckBlock, self).__init__()
        # 首个bottleneck需要承接上一批blocks的输出channel
        if pre_channel is None:  # 为空则表示不是首个bottleneck，
            pre_channel = outchannel  # 正常通道转换


        else:  # 传递了pre_channel,表示为首个block，需要shortcut
            shortcut = nn.Sequential(
                nn.Conv2d(pre_channel, outchannel, 1, stride, 0, bias=False),
                nn.BatchNorm2d(outchannel)
            )

        self.left = nn.Sequential(
            # 1*1,inchannel
            nn.Conv2d(pre_channel, inchannel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            # 3*3,inchannel
            nn.Conv2d(inchannel, inchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            # 1*1,outchannel
            nn.Conv2d(inchannel, outchannel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        return F.relu(out + residual)


def calculate(dataset, dataloader, net, criterion, train=False, optimizer=None):
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data

        if train:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        correct_num = torch.sum(torch.eq(labels, predict))
        running_acc += correct_num.item()

    running_loss /= len(dataset)
    running_acc /= len(dataset)

    return running_loss, running_acc


import time

if __name__ == '__main__':
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224,
                                                          0.225])])

    # 训练集
    trainset = torchvision.datasets.ImageFolder('../DATASET/testss',
                                                transform=transform)
    trainloader = DataLoader(trainset, batch_size=4,
                             shuffle=True, num_workers=0)

    # 测试集
    testset = torchvision.datasets.ImageFolder('../DATASET/testss',
                                               transform=transforms.Compose([transforms.Resize(256),
                                                                             transforms.CenterCrop(224),
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                                                  [0.229, 0.224,
                                                                                                   0.225])]))
    testloader = DataLoader(testset, batch_size=4,
                            shuffle=True, num_workers=0)
    # 卷积神经网络
    net = ResNetModel(12, [3, 4, 6, 3])
    # 损失函数 [pytorch 中交叉熵包括了 softmax 层]
    criterion = nn.CrossEntropyLoss()
    # 优化器，调整参数
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # 训练参数
    count = 50  # 训练次数

    # 训练过程
    for epoch in range(count):
        start_time = time.time()
        train_loss, train_acc = calculate(trainset, trainloader, net, criterion, True, optimizer)
        test_loss, test_acc = calculate(testset, testloader, net, criterion)
        end_time = time.time()
        print("[%d/%d] Loss: %.5f, Acc: %.2f%%, time: %.2fs" % (
        epoch + 1, count, test_loss, 100 * test_acc, (end_time - start_time)))

    print('Finished Training')
