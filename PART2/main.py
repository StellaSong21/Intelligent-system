import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from PART2.algorithm import CNN

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder('./DATASET/train',
                                            transform=transform)

trainloader = DataLoader(trainset, batch_size=4,
                         shuffle=True, num_workers=0)

net = CNN.CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')

testset = torchvision.datasets.ImageFolder('./DATASET/train',
                                           transform=transform)

testloader = DataLoader(testset, batch_size=25,
                        shuffle=True, num_workers=0)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


classes = ('1', '2', '3', '4',
           '5', '6', '7', '8', '9', '10', '11', '12')

dataiter = iter(testloader)
images, labels = dataiter.next()  #
imshow(torchvision.utils.make_grid(images, nrow=5))  # nrow是每行显示的图片数量，缺省值为8
print('GroundTruth: '
      , " ".join('%5s' % classes[labels[j]] for j in range(25)))  # 打印前25个GT（test集里图片的标签）
outputs = net(images)
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(25)))
# 打印前25个预测值
