import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
torch.manual_seed(515)

class SparseLinear2(nn.Module):
    def __init__(self, in_features, out_features, bias=True, connect_prob=None, ifPositive=True):
        super(SparseLinear2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.connection_probability = connect_prob
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.ifPositive = ifPositive

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self.mask = self.weight.data.clone().float().fill_(0)
        self.apply_mask()

    def apply_mask(self):
        if self.connection_probability <= 1.0:
            self.mask.bernoulli_(self.connection_probability)
        self.mask = self.mask.to(torch.float32)
        self.weight.data *= self.mask

    def uniform_init(self, init_w):
        # 均匀分布初始化weight并维护稀疏性
        nn.init.uniform_(self.weight, 0, init_w)
        self.weight.data *= self.mask

    def forward(self, x):
        if self.mask.device != self.weight.device:
            self.mask = self.mask.to(self.weight.device)
        if self.ifPositive:
            # 约束连接权重永远为正
            self.weight.data.clamp_(0, 1)
        return F.linear(x, self.weight * self.mask, self.bias)

if __name__ == '__main__':
    # 检查SparseLinear层
    model = SparseLinear2(10, 5, bias=False, connect_prob=0.1,ifPositive=True)
    model.uniform_init(init_w=0.5)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 检查权重是否稀疏
    print("Sparse weights:", model.weight)

    # 生成数据
    inputs = torch.randn(32, 10)  # 假设批量大小为32
    targets = torch.randn(32, 5)  # 假设目标维度与输出维度相同

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 检查权重是否稀疏
    print("Sparse weights:", model.weight)

    before_update = model.weight.data.clone()

    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()

    after_update = model.weight.data

    # 比较变化权重
    unchanged_weights = (before_update - after_update).abs() < 1e-6

    # 检查未变化的权重是否与mask的0元素对应
    assert torch.all(unchanged_weights[model.mask == 0]), "Some unconnected weights were updated!"

    print("Test passed: Unconnected weights were not updated.")