---
layout: post
title: xavier initialization介绍
description: 
modified: 2018-01-30
tags: [PNN]
---

Xavier Glorot在2010年的《Understanding the difficulty of training deep feedforward neural networks》提出了xavier glorot intialization，该方法在tensorflow中直接有集成。

# 介绍

在2006年前，深度多层神经网络不能很成功地进行训练，自那以后，许多方法被成功训练，实验结果表明层度更深的架构要比层度浅的要好。取得的所有这些实验结果，都使用了新的初始化（intialization）或训练机制。我们的目标是，更好理解为什么深度神经网络进行随机初始化(random initialzation)的标准梯度下降，效果会很差；以及更好理解最近的成功算法以及更好地帮助你设计算法。我们首先观察下非线性激活函数的影响。我们发现，**logistic sigmoid激活函数对于进行随机初始化的深度网络是不合适的**，因为它的均值(mean value)会特别推动top hidden layer进入饱和态(saturation)。令人吃惊的是，我们发现饱和态单元(saturated units)可以自己摆脱饱和态（即使很慢），并解释了在训练神经网络时有时会看到的停滞态（plateaus）。我们发现，一种具有更少的饱和态的新非线性函数会更有意义。最后，在训练期间，我们研究了activations和gradients会随着layers的不同而不同，并认为：当每个layer相关的jacobian的奇异值(singular values)与1相差很大时，训练可能更困难。基于该思想，我们提出了一种新的intialization scheme，它可以带来更快的收敛。

# 1.DNN介绍

略

我们的分析受在多layers、跨多个训练迭代上监控activations（观看hidden units饱和态）、gradients的实验所启发。也评估了多种activation函数、intialzation过程上的效果。

# 2.实验设置和数据集

代码在这部分有介绍：http://www.iro.umontreal.
ca/˜lisa/twiki/bin/view.cgi/Public/
DeepGradientsAISTATS2010

## 2.1 Shapeset-3x2上的在线学习

最近的深度结构研究(bengio 2009)表明，非常大的训练集或在线学习上，非监督预训练上的初始化会产生大幅提升，随着训练样本数的增加，并不会有vanish。online setting也很有意思，因为它主要关注optimization issues，而非在小样本正则化的影响，因此我们决定在我们的实验中包含一个人工合成的图片数据集，并从中抽样多个样本，来测试online学习。

我们称该数据集为Shapeset-3x2 dataset，如图1所示。shapeset-3x2包含了1或2个二维物体(objects)，每个都从3个shape类目（三角形、平行四边形、椭圆形）中获取，并使用随机形态参数（相对长度／角度）、缩放、旋转、翻转和灰度化进行放置。

我们注意到，识别图片中只有一个shape的很简单。因此，我们决定抽样带有两个物体（objects）的图片，限制条件是，第二个物体不能与第一个重合超过50%的区域，来避免整体隐藏掉。该任务是预计多个物体（比如：三角形+椭圆、并行四边形+并行四边形），不需要去区分foreground shape和background shape。因此我们定义了九种配置类。

该任务相当困难，因为我们需要发现在旋转、翻转、缩放、颜色、封闭体和相对位置。同时，我们需要抽取变量因子来预测哪个object shapes。

图片的size固定在32x32,以便更有效的进行深度dense网络。

## 2.2 有限数据集

- MNIST digits数据集：50000训练图片、10000验证图片、10000测试图片、每个展示了一个关于10个数字的28x28灰度图片。

- CIFAR-10: 50000训练样本、10000验证图片、10000测试图片。10个类，每个对应于图片上的一个物体：飞机、汽车、鸟、等等。

- Small-ImageNet：

## 2.3 实验设置

我们使用1到5个hidden layers、每层1000个hidden units、output ayer上一个softmax logistic regression来优化前馈神经网络. cost function是负log似然，$$-logP(y \mid x)$$，其中(x,y)是(输入图片，目标分类)对。该网络在size=10的minibatchs上使用随机BP，例如： $$\frac{\partial -logP(y \mid x)}{\partial \theta}$$的平均g，可以通过10个连续training pairs(x,y)计算，并用于更新在该方向上的参数$$\theta$$，$$\theta \leftarrow \theta - \epsilon g$$。learning rate $$\epsilon$$是一个超参数，它可以基于验证集error来进行最优化。

我们会使用在hidden layers上的多种类型的非线性激活函数：sigmoid，tanh(x)，以及softsign $$w/(1+\mid x\mid$$。其中softsign与tanh很相似，但它的尾部是二次多项式，而非指数，比如：它的渐近线更缓慢。

作为对比，我们会为每个模型搜索最好的参数（learning rate和depth）。注意，对于shapeset-3x2, 最好的depth总是5, 而对于sigmoid，最好的深度为4.

我们会将biases初始化为0, 在每一层的weights $$W_{ij}$$，有如下公共使用的启发式方法：

$$
W_{ij} ~ U[\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}]
$$

其中，U[-a, a]是在区间(-a,a)上的均匀分布(uniform distribution)，n是前一layer的size（列数目为W）。

# 3.激活函数的效果、训练期间的饱和态

我们希望避免两个事情，它们可以通过激活函数的演化来揭露：激活函数的过饱和（梯度将不会很好地进行传播），过度线性单元（overly linear units；他们有时会计算一些不感兴趣的东西）。

## 3.1 sigmoid实验

## 3.2 tanh实验

## 3.3 softsign实验

# 4.研究梯度和它们的传播

## 4.1 cost function的效果

## 4.2 初始化时的梯度

### 4.2.1 一种新的归一化初始化理论思考

我们在每一层的inputs biases上研究了BP梯度，或者梯度的cost function。Bradley (2009)发现BP的梯度会随着output layer向input layer方法传递时会更小，在初始化之后。他研究了在每层使用线性激活函数的网络，发现BP梯度的variance会随着我们在网络上进行backwards而减小。我们也通过研究线性状态开始。

对于使用在0处有单位导数(比如：$$f'(0)=1$$)的对称激活函数（symmetric activation function）f的一个dense的人工神经网络（artificial neural network），如果我们将layer i的activation vector写为$$z^i$$，layer i上激活函数的参数向量(argument vector)写为$$s^i$$，我们有：$$s^i = z^i W^i + b^i$$，$$z^{i+1}=f(s^i)$$。从这些定义中我们可以获得如下：

$$
$$

...(2) 

...(3)

方差(variances)各种对应input、output、权重进行随机初始化。考虑到这样的假设：我们在初始化时会在线性状态（linear regime），权重会独立初始化，输入特征方差相同（=Var[x]）。接着，我们可以说，$$n_i$$是layer i的size，x为input：

...(4)

...(5)

我们将在layer i'上的所有权重的共享标量方差写为$$Var[W^{i'}]$$。接着对于 d layers的一个网络：

...(6)
...(7)

从一个前向传播的角度看，我们可以像以下方式来保持信息流：

...(8)

从一个后向传播的角色，相似的，有：

...(9)

这两个条件(conditions)转换为：

...(10)

...(11)

作为这两个限制条件间的折中，我们希望有：

...(12)

当所有layers具有相同的宽度时，需要注意：如何同时满足这两个限制(constraints)。如果我们为权重(weights)具有相同的初始化，我们可以得到以下有意思的特性：

...(13)

...(14)

我们可以看到，在权重上梯度的方便对于所有layers是相同的，但BP梯度的方差可能仍会随着更深的网络而消失（vanish）或爆炸（explode）。

我们使用等式(1)的标准初始化，它会产生如下特性的方差：

$$
n Var[W] = \frac{1}{3}
$$

...(15)

其中，n是layer size（假设所有layers都具有相同的size）。这会造成BP梯度的方差依赖于该layer（并递减）。

当初始化深度网络时，由于通过多层的乘法效应（multiplicative effect），归一化因子（normalization factor）可能非常重要，我们建议，以下的初始化过程可以近似满足我们的目标：保持激活函数的方差（activation variances）和BP梯度方差会随着网络上下移动。我们称之为归一化初始化（normalized initialzation）：

$$
W \sim U [ - \frac{\sqrt{6}}{\sqrt{n_j + n_{j+1}}}, \frac{\sqrt{6}}{\sqrt{n_j + n_{j+1}}} ]
$$

...(16)

### 4.2.2 梯度传播研究

为了经验上验证以上的理论思想，我们绘制了一些关于activation values、weight gradients的归一化直方图，在初始化处的BP梯度，使用两种不同的初始化方法。shapeset-3x2的实验结果如图所示（图6,7,8），其它数据集也可以获得相似的结果。

我们会监控与layer i相关的Jacobian矩阵的奇异值：

$$
J^i = \frac{\partial z^{i+1}}{\partial z^i}
$$

...(17)

当连续层（consecutive layers）具有相同的维度时，对应于极小量平均比例(average ratio of infinitesimal volumes)的平均奇异值从$$z^i$$映射到$$z^{i+1}$$，而平均激活方差的ratio也会从$$z^i$$映射到$$z^{i+1}$$。有了我们的归一化初始化，该ratio会在0.8周围; 而使用标准初始化，它会降到0.5.


图6

### 4.3 学习期间的梯度BP

在这样的网络中，学习的动态性是很复杂的，我们想开发更好的工具来分析和跟踪它。特别的，我们在我们的理论分析中不能使用简单的方差计算；因为权重值不再独立于activation values，也会违反线性假设。

正如Bradley(2009)年首次提到的，我们观察到(如图7所示)，在训练的开始处，在标准初始化（等式1）后，BP梯度的方差会随着传播的下降而变更小。然而，我们发现，该趋势在学习期间逆转的非常快。使用我们的归一化初始化，我们不会看到这样的减小的BP梯度（如图7底部所示）。

令人吃惊的是，当BP gradients变得更小时，跨layers的weights gradients仍是常数，如图8所示。然而，这可以通过我们上述的理论分析（等式14）进行解释。如图9所示，令人感兴趣的是，关于标准初始化和归一化初始化的权重梯度上的这些观察会在训练期间变化（这里是一个tanh网络）。实际上，梯度初始时粗略具有相同的幅度，随着训练的进度会各自不同（越低层具有更大的梯度），特别是当使用标准初始化时。注意，这可能是归一化初始化的一个优点，因为它在不同的layers上具有非常不同幅值的梯度，可以产生病态性（ill-conditioning），以及更慢的训练。

最终，我们观察到，使用归一化初始化的softsign网络与tanh网络共享相似，我们可以对比两者的activations的演进看到（图3底部和图10）。

# 5.error曲线和结论

我们关心的最终考虑点是，使用不同策略的训练的成功，这可以使用error curves来展示，它可以展示test error随着训练的过程和渐近而演进。图11展示了在shapeset-3x2上在线学习训练的曲线，其中表1给出了对所有数据集的最终的test error。作为baseline，我们在10w个shapeset样本上使用优化的RBF SVM模型，并获得了59.47%的test error，而在相同的集合上，使用一个深度为5的tanh网络、并使用归一化初始化可以获得50.47%的test error。

结果表明，activation和intialization的选择的效果。作为参考，我们在图11中。


# 参考

- 1.[https://arxiv.org/pdf/1506.01497.pdf](https://arxiv.org/pdf/1506.01497.pdf)
- 2.[https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer)
