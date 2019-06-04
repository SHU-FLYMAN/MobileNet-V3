[TOC]

> Fork自：<A PyTorch implementation of MobileNetV3> https://github.com/xiaolai-sqlai/mobilenetv3

完全Fork自 [Here](https://github.com/xiaolai-sqlai/mobilenetv3) ，所不同的是做了一些很小很小的改进，使大家更容易拿过来训练，顺便加了一些个人简单的解读。

# 改进

- [x] 修复一些兼容性bug，增加支持Pytorch1.1
- [x] 增加了新的加载数据的方式
- [ ] 训练

其实主要是键值对的问题，所有增加的内容在`utils.load_model`函数里面。

## 分类

如果想要用ImageNet初始化权重去分类话，可以像这样构建模型：

```python
from mobilenet_v3 import MobileNetV3_Large

model_path = 'weights/mbv3_large.old.pth.tar'
model = load_model(model=MobileNetV3_Large(num_classes=100),
                   model_path=model_path,
                   pt_version=1.1)
```

我们验证一下：

```python
# 验证结果
model_weights = model.state_dict()
print(model_weights['conv1.weight'])
print(model_weights['linear4.weight'].shape)
print(model_weights['linear4.bias'].shape)
-------
# 权重不是全部接近0的小数
torch.Size([100, 1280])
torch.Size([100])
```

## 其他

如果你想要训练VOC等目标检测数据集，你可以继承`MobileNetV3_Large`类，然后去掉最后的线性层，之后拼接上你所需要的相应的Head即可。

# 论文解读

在现代深度学习算法研究中，通用的骨干网+特定任务网络head成为一种标准的设计模式，比如说：

- VGG + 检测Head
- inception + 分割Head

在移动端部署深度卷积网络，无论什么视觉任务，选择高精度的计算量少和参数少的骨干网是必经之路。而谷歌最近开源的MobileNet-V3，无论在精度或者说速度上都达到了state of art，是移动端卷积模型的首选。现在我们来解读一下MobileNetv3的设计思想以及网络结构。

# 两个版本

其实MobileNet-V3 没有引入新的 Block，Searching已经道尽该网络的设计哲学：神经架构搜索。研究人员公布了 MobileNetV3 有两个版本，MobileNetV3-Small 与 MobileNetV3-Large 分别对应对计算和存储要求低和高的版本。

![](https://image.jiqizhixin.com/uploads/editor/a91d5e94-b203-4f36-bc48-0d7d71a863b6/640.png)

下图是ImageNet分类精度、MADD计算量、模型大小的比较，MobileNetV3依然是最优秀的。

![](https://image.jiqizhixin.com/uploads/editor/11110a96-aa59-46e2-ad51-a3c26837f62a/640.png)

# 创新点

## 01 高效的网络构建模块

1. MobileNetV1   >>>   **深度可分离卷积**              (Depthwise Separable Convolutions)
2. MobileNetV2   >>>   **线性瓶颈的倒残差结构**   (The Inverted Residual With Linear Bottleneck)
3. MnasNet          >>>   **轻量级注意力模型**           (Squeeze and Excitation)

## 02 搜索

实际上这部分就不要看了，没钱！一个ImageNet都训练不下来。大概就是：

在网络结构搜索中，作者结合两种技术：资源受限的NAS（platform-aware NAS）与NetAdapt，前者用于在计算和参数量受限的前提下搜索网络的各个模块，所以称之为模块级的搜索（Block-wise Search） ，后者用于对各个模块确定之后网络层的微调。

这两项技术分别来自论文：

> M. Tan, B. Chen, R. Pang, V. Vasudevan, and Q. V. Le. Mnasnet: Platform-aware neural architecture search for mobile. CoRR, abs/1807.11626, 2018. 

> T. Yang, A. G. Howard, B. Chen, X. Zhang, A. Go, M. Sandler, V. Sze, and H. Adam. Netadapt: Platform-aware neural network adaptation for mobile applications. In ECCV, 2018

前者相当于整体结构搜索，后者相当于局部搜索，两者互为补充。

## 03 网络改进

作者们发现MobileNetV2 网络端部最后阶段的计算量很大，重新设计了这一部分，如下图：

![](https://image.jiqizhixin.com/uploads/editor/983d390e-7cfc-415a-878d-41ec8305b125/640.png)

## 04 损失函数改进

作者发现一种新出的激活函数$swish x$，能够有效改进网络精度：

![](https://image.jiqizhixin.com/uploads/editor/4051d112-0602-4118-91b1-bb24f80fbae4/640.png)

但是发现计算了太大，作者做了一个数值近似，效果也差不多：

![](https://image.jiqizhixin.com/uploads/editor/25228464-d093-4e03-b0b8-71fb39446345/640.png)

实际上效果确实差不多：

![](https://image.jiqizhixin.com/uploads/editor/192df9f0-2cb2-4912-8c2a-0a9e82acabe1/640.png)

# 最终模型

## 01 Large

![](https://image.jiqizhixin.com/uploads/editor/afef587c-6c4a-4044-8085-15145e2517e4/640.png)

## 02 Small

![](https://image.jiqizhixin.com/uploads/editor/add5c5ac-ffa8-4115-9dbe-3eadfaa74035/640.png)

# 实验结果

![](https://image.jiqizhixin.com/uploads/editor/926af99f-aa36-4480-bd9c-deac85a312c8/640.png)

还有其他一些指标，总而言之，就是又好又快！

