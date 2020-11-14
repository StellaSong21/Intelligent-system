import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from PART2.algorithm import Util


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        assert self.fc1.in_features == self.num_flat_features(x)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    # 存储模型的路径
    path = '../record/CNN/CNN0'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 训练集
    trainset = torchvision.datasets.ImageFolder('../DATASET/train',
                                                transform=transform)
    trainloader = DataLoader(trainset, batch_size=4,
                             shuffle=True, num_workers=0)

    # 测试集
    testset = torchvision.datasets.ImageFolder('../DATASET/tests',
                                               transform=transform)
    testloader = DataLoader(testset, batch_size=4,
                            shuffle=True, num_workers=0)
    # 卷积神经网络
    net = CNN()
    # 损失函数 [pytorch 中交叉熵包括了 softmax 层]
    criterion = nn.CrossEntropyLoss()
    # 优化器，调整参数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)

    # 训练参数
    count = 50

    # 训练过程
    for epoch in range(count):
        train_loss, train_acc = Util.calculate(trainset, trainloader, net, criterion, True, optimizer)
        torch.save(net.state_dict(), os.path.join(path, str(epoch + 1) + '.pt'))
        net.load_state_dict(torch.load(os.path.join(path, str(epoch + 1) + '.pt')))
        net.eval()
        test_loss, test_acc = Util.calculate(testset, testloader, net, criterion)
        print("[%d/%d] Loss: %.5f, Acc: %.2f%%" % (epoch + 1, count, test_loss, 100 * test_acc))

    print('Finished Training')
