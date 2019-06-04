# -*- coding: utf-8 -*-

# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
# 

#  Editor      : PyCharm
#  Time        : 5/27/19 1:03 AM
#  Author      : Flyman
#  Email       : fly_cjb@163.com
#  File        : utils.py
#  Description : 

#
# ================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module
from mobilenet_v3 import MobileNetV3_Large, MobileNetV3_Small

class hsigmoid(Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


def load_model(model, model_path, pt_version: float =1.1):
    # 这里要做个处理,pytorch1.1跟1.0会不太一样
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location='cpu')['state_dict']
    if pt_version == 1.1:
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    elif pt_version == 1.0:
        pass
    else:
        raise ValueError("pt_verison must be either 1.0 or 1.1")
    end_layer_name = list(pretrained_dict.keys())[-2:]
    # 选择不加载最后线性层权重即可
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (k not in end_layer_name)}
    # 用新的键值对去更新
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('成功加载模型！')
    return model


if __name__ == '__main__':
    model_path = 'weights/mbv3_large.old.pth.tar'
    model = load_model(model=MobileNetV3_Large(num_classes=100),
                       model_path=model_path,
                       pt_version=1.1)
    # 验证结果
    model_weights = model.state_dict()
    print(model_weights.keys())
    print(model_weights['conv1.weight'])
    print(model_weights['linear4.weight'].shape)
    print(model_weights['linear4.bias'].shape)