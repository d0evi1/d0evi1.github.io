---
layout: post
title: distillation介绍
description: 
modified: 2017-12-11
tags: 
---

hinton在google提出了《Distilling the Knowledge in a Neural Network》 knowledge distillation。

# 1.介绍

# 2.Distillation

许多networks通常会使用一个"softmax" output layer来生成类概率（class probabilities），它会对logits $$z_i$$进行转化，为每个class计算一个probability, $$q_i$$，并与其它logits进行相互比较$$z_i$$。

$$
q_i = \frac{exp(z_i / T)}{\sum_j exp(z_j / T)}
$$

...(1)

其中，T是一个temperature，通常设置为1. 将T设置得更高会生成一个在classes上的softer probability分布。

在distillation的最简形式中，通过在一个transfer set上训练模型，并使用一个对每个case在transfer set中的一个soft target分布（使用cumbersome model并在softmax中使用一个高temperature），knowledge会被转移到distilled model中。当训练该distilled model时，会使用相同高的temperature，但在它被训练后，它会使用一个temperature为1。

当correct labels被所有或部分transfer set知道时，该方法可以极大提升，也可以训练该distilled model来生成correct labels。这样做的一种方式是，使用correct labels来修正soft targets，但我们发现一种更好的方式是：简单使用一个对两个不同objective functions的weighted average。第一个objective function是使用soft targets的cross entropy，该cross entropy使用distilled model的softmax上的相同logits，但temperature=1。我们发现，获得的最好结果，会在objective function上使用一个相当低的weight。由于通过soft targets生成的梯度幅度缩放为 $$1/T^2$$，当同时使用hard和osft targets时，乘上$$T^2$$很重要。当实验使用meta-parameters时，如果用于distillation的temperature发生变化，这可以确保hard和soft targets的相对贡献仍然大致不变。

## 2.1 匹配logits是distillation的一个特征

在transfer set中每个case，对于distilled model的每个logit $$z_i$$，贡献了一个cross entropy gradient, $$dC/ dz_i$$。如果cumbersome model具有logits $$v_i$$，它会生成soft target probabilities $$p_i$$，并且transfer training会在temperature T上完成，我们给出了该gradient：

$$
\frac{\partial C}{\partial z_i} = \frac{1}{T} (q_i - p_i) = \frac{1}{T} (\frac{e^{z_i/T}}{\sum_j e^{z_j/T}} - \frac{e^{v_i/T}}{\sum_j e^{v_j/T}})
$$

...(2)

如果对于logits的幅值，temperature高，我们可以近似：

$$
\frac{\partial C}{\partial z_i}  \approx \frac{1}{T}(\frac{1+z_i/T}{N+\sum_j z_j/T} - \frac{1+v_i/T}{N + \sum_j v_j/T}
$$

...(3)

如果我们假设：对于每个transfer case，logits已经是独立零均值的，以便$$\sum_j z_j = \sum_j v_j = 0$$，等式3简化为：

$$
\frac{\partial C}{\partial z_i} \approx \frac{1}{NT^2} (z_i - v_i)
$$

...(4)

因此，在高temperature限制下，distilliation等价于最小化$$1/2(z_i - v_i)^2$$，提供的logits对于每个transfer case都是独立零均值的。在更低的temperatures上时，distilliation会花费更少的attention来matching logits，以便比平均有更多的negative。这有潜在优势，因为这些logits对于用于训练cumbersome modelcost function几乎完整无限制，因此他们可能非常有noisy。另一方面，非常负的logits可能会传达关于由cumbersome model获取knowledge的有用信息。这些效果占据着哪些是一个经验性问题。我们展示了当distilled model太小，而不能获取cumbersome model中的所有知识，intermediate temperatures会运行最好，会强烈建议忽略掉大的negative logits，这样会有用。

。。。


# 参考


- 1.[https://arxiv.org/pdf/1503.02531.pdf](https://arxiv.org/pdf/1503.02531.pdf)