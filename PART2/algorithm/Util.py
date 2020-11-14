import torch


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
