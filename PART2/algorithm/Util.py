import torch
import os

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
