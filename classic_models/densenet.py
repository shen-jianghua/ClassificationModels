import re
from typing import Any, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor


class _DenseLayer(nn.Module):
    def __init__(self, input_c: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False):
        super(_DenseLayer, self).__init__()
        # 1. Dense layer的结构参照了ResNetV2的结构，BN->ReLU->Conv
        # 2. 与ResNet的bottleneck稍有不同的是，此处仅做两次conv（1*1conv,3*3conv)，不需要第三次1*1conv将channel拉升回去
        # 3. 由于Dense block中Tensor的channel数是随着Dense layer不断增加的,所以Dense layer设计的就很”窄“（channel数很小，
        # 固定为growth_rate)，每层仅学习很少一部分的特征
        # 4. add.module 等效于 self.norm1 = nn.Batchnorm2d(num_input_features)
        self.add_module("norm1", nn.BatchNorm2d(input_c))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(in_channels=input_c,  out_channels=bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        # 每一Dense layer结束后dropout丢弃的feature maps的比例
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]):
        concat_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))
        return bottleneck_output

    @staticmethod
    def any_requires_grad(inputs: List[Tensor]):
        for tensor in inputs:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, inputs: List[Tensor]):
        def closure(*inp):
            return self.bn_function(inp)

        return cp.checkpoint(closure, *inputs)

    def forward(self, inputs: Tensor):
        if isinstance(inputs, Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("memory efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,  p=self.drop_rate, training=self.training)

        return new_features


# Dense block其实就是多个Dense layer的叠加，需要注意的就是两个layer连接处的input_features值是逐渐增加growth_rate
class _DenseBlock(nn.ModuleDict): 
    def __init__(self, num_layers: int, input_c: int, bn_size: int, growth_rate: int,  drop_rate: float,  memory_efficient: bool = False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            # 由于一个DenseBlock中，每经过一个layer，宽度（channel）就会堆叠增加growth_rate，所以仅需要改变num_input_features即可
            layer = _DenseLayer(input_c + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate,  memory_efficient=memory_efficient)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


# Transition layer（过渡层）, 两个Dense block间加入的Transition层起到了两个作用：
# 防止features数无限增大，进一步压缩数据; 下采样，降低feature map的分辨率
class _Transition(nn.Sequential):
    def __init__(self,
                 input_c: int,
                 output_c: int):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(input_c))
        self.add_module("relu", nn.ReLU(inplace=True))
        # 作用1：即使每一层Dense layer都采取了很小的growth_rate，但是堆叠之后channel数难免会越来越大,
        # 所以需要在每一个Dense block之后接transition层用1*1conv将channel再拉回到一个相对较低的值（一般为输入的一半）
        self.add_module("conv", nn.Conv2d(input_c, output_c, kernel_size=1, stride=1, bias=False))
        # 作用2：用average pooling改变图像分辨率，下采样
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densenet-BC model class for imagenet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient
    """

    def __init__(self,
                 growth_rate: int = 32,
                 block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
                 num_init_features: int = 64,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 num_classes: int = 1000,
                 memory_efficient: bool = False):
        super(DenseNet, self).__init__()

        # first conv+bn+relu+pool
        # 和ResNet一样，先通过7*7的卷积，将分辨率从224*224->112*112
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # each dense block
        num_features = num_init_features
        # 读取每个Dense block层数的设定
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                input_c=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # 第四个Dense block后不再连接Transition层
            if i != len(block_config) - 1:
                # 此处可以看到，默认过渡层将channel变为原来输入的一半
                trans = _Transition(input_c=num_features,
                                    output_c=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # finnal batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # fc layer
        self.classifier = nn.Linear(num_features, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):        # 函数来判断一个对象是否是一个已知的类型，是Python的一个内置函数
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121(num_classes):
    # Top-1 error: 25.35%
    # 'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 24, 16),
                    num_init_features=64,
                    num_classes=num_classes)


def densenet169(num_classes):
    # Top-1 error: 24.00%
    # 'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 32, 32),
                    num_init_features=64,
                    num_classes=num_classes)


def densenet201(num_classes):
    # Top-1 error: 22.80%
    # 'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 48, 32),
                    num_init_features=64,
                    num_classes=num_classes)


def densenet161(num_classes):
    # Top-1 error: 22.35%
    # 'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth'
    return DenseNet(growth_rate=48,
                    block_config=(6, 12, 36, 24),
                    num_init_features=96,
                    num_classes=num_classes)

