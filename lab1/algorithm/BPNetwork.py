import numpy as np
from algorithm import LossFunction as lf, ActiveFunction as af
import os


class ActiveLayer:
    def __init__(self, function=af.Function()):
        self.activate = function.activate
        self.deactivate = function.deactivate
        pass

    # input
    def forward(self, input):
        return self.activate(input)

    def backward(self, input, err):
        return np.multiply(err, self.deactivate(input))


class WeightLayer:
    def __init__(self, input_n, output_m, learn_w=0.05, learn_b=0.02, weight=-1.0, bias=-0.2):
        self.input_n = input_n  # 输入个数
        self.output_m = output_m  # 输出个数
        self.learn_w = learn_w  # weight 学习率
        self.learn_b = learn_b  # bias 学习率
        self.weights = np.random.randn(input_n, output_m) * weight
        self.biases = np.random.randn(output_m) * bias

    def forward(self, input):
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, err):
        next_err = np.dot(err, self.weights.T)
        self.weights = self.weights + self.learn_w * np.dot(input.T, err)
        self.biases = self.biases + self.learn_b * err
        return next_err


class SoftmaxLayer:
    def __init__(self):
        pass

    def forward(self, input):
        exp_x = np.exp(input - np.max(input))
        return exp_x * 1.0 / np.sum(exp_x)

    def backward(self, input, err):
        return err


class SinNormalizeLayer:
    def __init__(self, low=-1.0, up=1.0):
        self.up = up
        self.low = low
        pass

    def forward(self, input):
        t1 = input > self.up
        t2 = input < self.low
        ch = t1 + 2 * t2
        return np.choose(ch, (input, self.up, self.low))

    def backward(self, input, targets):
        return targets


class BPNetwork:
    def __init__(self, layer_list,
                 learn_w=0.01, learn_b=0.005,
                 weight=0, bias=-0.1,
                 active_func=af.Function(), loss_func=lf.squared_error,
                 softmax=False, sin=False):
        self.learn_w = learn_w
        self.learn_b = learn_b
        layers_count = len(layer_list)
        self.loss_func = loss_func
        self.layers = []
        for i in range(layers_count - 2):
            weight_layer = WeightLayer(input_n=layer_list[i], output_m=layer_list[i + 1],
                                       learn_w=learn_w, learn_b=learn_b,
                                       weight=weight, bias=bias)
            active_layer = ActiveLayer(active_func)
            self.layers.append(weight_layer)
            self.layers.append(active_layer)
            pass
        weight_layer = WeightLayer(input_n=layer_list[layers_count - 2], output_m=layer_list[layers_count - 1],
                                   learn_w=learn_w, learn_b=learn_b,
                                   weight=np.choose(weight == 0,
                                                    (weight, -1.0 / np.sqrt(layer_list[layers_count - 2]))),
                                   bias=bias)
        self.layers.append(weight_layer)
        if softmax:
            self.layers.append(SoftmaxLayer())
            pass
        if sin:
            self.layers.append(SinNormalizeLayer(-1, 1))
            pass

    def train(self, input, target):
        data = input
        H = [input]
        # forward
        for layer in self.layers:
            data = layer.forward(data)
            H.append(data)
            pass
        # 误差
        loss = np.mean(self.loss_func(data, target))
        # backward
        err = target - data
        for i in range(len(self.layers) - 1, -1, -1):
            h = H[i]
            err = self.layers[i].backward(h, err)
            pass
        return data, loss

    def query(self, input, target):
        data = input
        for layer in self.layers:
            data = layer.forward(data)
            pass
        return data, np.mean(self.loss_func(data, target))

    def test(self, input):
        data = input
        for layer in self.layers:
            data = layer.forward(data)
            pass
        return data

    def set_learn_rate(self, learn_w, learn_b):
        for layer in self.layers:
            layer.learn_w = learn_w
            layer.learn_b = learn_b
        pass

    def set_learn_w(self, learn_w):
        for layer in self.layers:
            layer.learn_w = learn_w
        pass

    def set_learn_b(self, learn_b):
        for layer in self.layers:
            layer.learn_b = learn_b
        pass

    def append(self, layer):
        self.layers.append(layer)
        pass

    def save(self, path='../record/BPNetwork'):
        if not (os.path.exists(path)):
            os.makedirs(path)

        # 储存神经网络
        num = 0
        for layer in self.layers:
            if not (isinstance(layer, WeightLayer)):
                continue
            file_name = "w" + str(num) + ".npy"
            file = os.path.join(path, file_name)
            np.save(file, layer.weights)

            file_name = "b" + str(num) + ".npy"
            file = os.path.join(path, file_name)
            np.save(file, layer.biases)
            num += 1
            pass
        pass

    def load(self, path="../record/BPNetwork"):
        # 读取神经网络
        num = 0
        for layer in self.layers:
            if not (isinstance(layer, WeightLayer)):
                continue
            file_name = "w" + str(num) + ".npy"
            file = os.path.join(path, file_name)
            layer.weights = np.load(file)

            file_name = "b" + str(num) + ".npy"
            file = os.path.join(path, file_name)
            layer.biases = np.load(file)
            num += 1
            pass
        pass
