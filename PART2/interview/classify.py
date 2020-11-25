import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from PART2.algorithm import Util
from PART2.algorithm import ResNet

if __name__ == '__main__':
    # 获取val图片已得到类别class_names
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])

    # 得到分类的种类名称
    ResNet_testset = torchvision.datasets.ImageFolder('../DATASET/train')
    class_names = ResNet_testset.classes

    net = ResNet.ResNetModel(12, [2, 2, 2, 2], False)
    net.load_state_dict(torch.load('../record/ResNet/18/22.pt')) # 18
    net.eval()
    test_set = Util.get_test_set('../DATASET/test')
    result = np.zeros(len(test_set), dtype=int)
    for i in range(len(test_set)):
        img = Image.open(test_set[i]).convert('RGB')
        inputx = transform(img).unsqueeze(0)
        output = net(inputx)
        _, indices = torch.sort(output, descending=True)
        # percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        # print([(class_names[idx], percentage[idx].item()) for idx in indices[0][:12]])
        result[i] = class_names[indices[0][0].item()]
    Util.write_result(result)

    # ResNet_testset = torchvision.datasets.ImageFolder('../DATASET/tests',
    #                                                   transform=transforms.Compose([transforms.Resize(256),
    #                                                                                 transforms.CenterCrop(224),
    #                                                                                 transforms.ToTensor(),
    #                                                                                 transforms.Normalize(
    #                                                                                     [0.485, 0.456, 0.406],
    #                                                                                     [0.229, 0.224, 0.225])]))
    #
    # ResNet_testloader = DataLoader(ResNet_testset, batch_size=4,
    #                                shuffle=False, num_workers=0)
    #
    # net = ResNet.ResNetModel(12, [2, 2, 2, 2], False)
    #
    # net.load_state_dict(torch.load('../record/ResNet/18/8.pt'))
    # net.eval()
    # criterion = nn.CrossEntropyLoss()
    # test_loss, test_acc = Util.calculate(ResNet_testset, ResNet_testloader, net, criterion)
    # print("Loss: %.5f, Acc: %.2f%%" % (test_loss, 100 * test_acc))
