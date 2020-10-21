import random
import numpy as np

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


# 初始化为 m * n 的矩阵
def init_weight(m, n):
    w = [0.0] * m
    for i in range(m):
        w[i] = [0.0] * n
        for j in range(n):
            w[i][j] = rand(-1, 1)
    return w


def init_bias(m):
    b = [0.0] * m
    for i in range(m):
        b[i] = rand(-1, 1)
    return b


# tanh激活函数
def tanh(x):
    return np.tanh(x)


# tanh函数的梯度函数
def tanh_derivative(x):
    return 1 - np.tanh(x) * np.tanh(x)


def active_function(x, deriv=False):
    if deriv:
        return 1 - np.tanh(x) * np.tanh(x)  # tanh 函数的导数
    return np.tanh(x)


def get_loss(expect, output_cell):
    error = 0.0
    for o in range(len(output_cell)):
        error += 0.5 * (expect[o] - output_cell[o]) ** 2
    return error


class BPNetwork:
    def __init__(self):
        self.input_n = 0  # 输入层神经元个数
        self.input_cells = []  # 输入层的输入数据，包括 bias
        self.output_m = 0  # 输出层神经元个数
        self.output_cells = []  # 输出层的输出数据

        self.input_w = []  # 输入层到第一个隐藏层的 weight
        self.output_w = []  # 最后一个隐藏层到输出层的 weight

        self.hidden_ns = []  # 隐藏层的个数=len(hidden_ns)，每层神经元的个数=hidden_ns[i]
        self.hidden_ws = []  # 隐藏层间的 weight 数组
        self.hidden_bs = []  # 上一层到当前隐藏层的 bias，长度为隐藏层的个数
        self.hidden_res = []  # 隐藏层的输出数组

        self.output_b = []  # 输出层的 bias
        self.output_deltas = []  # 输出层的 delta 调整值，相当于 ∂Error / ∂(output_w)
        self.hidden_deltases = []  # 隐藏层的 delta 调整值

    def setup(self, input_n, output_m, hidden_ns):
        # 初始化输出
        self.input_n = input_n
        self.input_cells = [0.0] * self.input_n

        # 初始化输出
        self.output_m = output_m
        self.output_cells = [0.0] * self.output_m

        # 隐藏层的个数
        hidden_layers = len(hidden_ns)

        # 初始化隐藏层
        self.hidden_ns = hidden_ns.copy()

        # 初始化输入层到第一个隐藏层的 weight
        self.input_w = init_weight(input_n, hidden_ns[0])

        # 初始化最后一个隐藏层到输出层的 weight
        self.output_w = init_weight(hidden_ns[hidden_layers - 1], output_m)

        # 初始化隐藏层间的 weight 数组
        self.hidden_ws = [0.0] * (hidden_layers - 1)
        for i in range(hidden_layers - 1):
            self.hidden_ws[i] = init_weight(hidden_ns[i], hidden_ns[i + 1])

        # 初始化 bias 数组
        self.hidden_bs = [0.0] * hidden_layers
        for i in range(hidden_layers):
            self.hidden_bs[i] = init_bias(hidden_ns[i])

        # 初始化隐藏层的输出数组
        self.hidden_res = [0.0] * hidden_layers

        # 初始化输出层的 bias
        self.output_b = init_bias(output_m)

    # 向前传播，对每一层求值
    def forward_propagate(self, input_val):
        # 根据 input 初始化输入层的输入数据
        self.input_cells = input_val.copy()

        # 输入层
        self.hidden_res[0] = [0.0] * self.hidden_ns[0]
        for h in range(self.hidden_ns[0]):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_w[i][h]
            self.hidden_res[0][h] = active_function(total + self.hidden_bs[0][h])

        # 隐藏层
        hidden_layers = len(self.hidden_ns)
        for k in range(hidden_layers - 1):
            self.hidden_res[k + 1] = [0.0] * self.hidden_ns[k + 1]
            for h in range(self.hidden_ns[k + 1]):
                total = 0.0
                for i in range(self.hidden_ns[k]):
                    total += self.hidden_res[k][i] * self.hidden_ws[k][i][h]
                self.hidden_res[k + 1][h] = active_function(total + self.hidden_bs[k + 1][h])

        # 输出层
        for h in range(self.output_m):
            total = 0.0
            for i in range(self.hidden_ns[hidden_layers - 1]):
                total += self.hidden_res[hidden_layers - 1][i] * self.output_w[i][h]
            self.output_cells[h] = active_function(total + self.output_b[h])

        return self.output_cells

    def calculate_delta(self, expect):
        # 输出层 delta
        self.output_deltas = [0.0] * self.output_m
        for o in range(self.output_m):
            error = expect[o] - self.output_cells[o]
            self.output_deltas[o] = active_function(self.output_cells[o], True) * error

        # 隐藏层 deltas
        hidden_layers = len(self.hidden_ns)
        tmp_deltas = self.output_deltas
        tmp_w = self.output_w
        self.hidden_deltases = [0.0] * hidden_layers
        k = hidden_layers - 1
        while k >= 0:
            self.hidden_deltases[k] = [0.0] * (self.hidden_ns[k])
            for o in range(self.hidden_ns[k]):
                error = 0.0
                for i in range(len(tmp_deltas)):
                    error += tmp_deltas[i] * tmp_w[o][i]
                self.hidden_deltases[k][o] = active_function(self.hidden_res[k][o], True) * error
            k = k - 1           # k -= 1
            if k >= 0:
                tmp_deltas = self.hidden_deltases[k + 1]
                tmp_w = self.hidden_ws[k]

    def update_w(self, learn):
        # 更新最后一个隐藏层到输出层的 output_w
        k = len(self.hidden_ns) - 1
        for i in range(self.hidden_ns[k]):
            for o in range(self.output_m):
                change = self.output_deltas[o] * self.hidden_res[k][i]
                self.output_w[i][o] += change * learn

        # 更新隐藏层之间的 hidden_ws
        while k > 0:
            for i in range(self.hidden_ns[k - 1]):
                for o in range(self.hidden_ns[k]):
                    change = self.hidden_deltases[k][o] * self.hidden_res[k - 1][i]
                    self.hidden_ws[k - 1][i][o] += change * learn
            k = k - 1

        # 更新输出层到第一个隐藏层之间的 weight
        for i in range(self.input_n):
            for o in range(self.hidden_ns[k]):
                change = self.hidden_deltases[k][o] * self.input_cells[i]
                self.input_w[i][o] += change * learn

    def update_b(self, learn):
        # 更新隐藏层的 bias
        k = len(self.hidden_ns) - 1
        while k >= 0:
            for i in range(self.hidden_ns[k]):
                self.hidden_bs[k][i] = self.hidden_bs[k][i] + learn * self.hidden_deltases[k][i]
            k = k - 1

        # 更新输出层的bias
        for i in range(self.output_m):
            self.output_b[i] = self.output_b[i] + learn * self.output_deltas[i]

    # 计算output的平均损失值
    def get_average_loss(self, datas, expects):
        error = 0
        predicate_res = []
        for i in range(len(datas)):
            predicate_res.append(self.forward_propagate(datas[i]))
            error += get_loss(expects[i], self.output_cells)
        error = error / len(datas)
        return error, predicate_res
