---
layout: post
title: ali COLD介绍
description: 
modified: 2020-09-10
tags: 
---

《Feedback Control of Real-Time Display Advertising》在PID中引入了feedback control。

## 3.4 点击最大化

假设：feedback controller是一个有效工作，用来分发广告主的KPI目标，在该节中，我们会演示feedback control机制可以被当成一个model-free的click maximisation framework，它可以嵌入到任意bidding策略，并执行在不同channels上，通过设置smart reference values来进行竞价预算分配。

当一个广告主指定目标受众时（通常会组合上广告曝光上下文分类）来进行它指定的campaign，来自独立channels（比如：不同的广告交易平台(ad exchanges)、不同的用户regions、不同的用户PC/model设备等）的满足目标规则（target rules）的曝光（impressions）。通常，DSP会集合许多ad exchanges，并分发来自所有这些广告交易平台(ad exchanges)的所需ad曝光（只要曝光能满足target rule），尽管市场价格会大有不同。图3展示了这些，对于相同的campaign，不同的广告交易平台(ad exchanges)会有不同的eCPC。如【34】中所说，在其它channels上（比如：user regions和devices上）也会有所不同。

开销差异提供给广告主一个机会，可以基于eCPC来最优化它的campaign效果。为了证实这点，假设一个DSP被集在到两个广告交易平台A和B。对于在该DSP中的一个campaign，如果来自A的eCPC要比B的高，这意味着来自平台B的库存要比平台A的费用更高效，接着会重新分配一些预算给A和B，这将潜在减小该campaign的整体eCPC。实际上，预算重分配（budget reallocation）可以通过对平台A减小竞价、并增加平台B的竞价来完成。这里，我们正式提出一个用于计算每个交易平台的均衡eCPC模型，它可以被用作最优化reference eCPC来进行feedback control，并在给定预算约束下生成一个最大数目的点击。

数学上，假设对于一个给定的ad campaign，存在n个交易平台（可以是其它channels），比如：1, 2, ..., n，它们对于一个target rule具有ad volume。在我们的公式里，我们关注最优化点击，而转化率公式可以相似被获取到。假设：

- $$\epsilon_i$$：是在交易平台i上的eCPC，
- $$c_i(\epsilon_i)$$是campaign在交易平台i上调整竞价使得eCPC为$$\epsilon_i$$，所对应的在campaign的lifetime中获得的点击数

对于广告主，他们希望在给定campaign预算B下，最大化campaign-level的点击数：

$$
max_{\epsilon_1,\epsilon_2,\cdots,\epsilon_n} \sum_i c_i(\epsilon_i) \\
s.t. \sum_i c_i(\epsilon_i) \epsilon_i = B
$$

...(6)(7)

它的拉格朗日项为：

$$
L(\epsilon_1,\epsilon_2,\cdots,\epsilon_n, \alpha) = \sum_i  c_i(\epsilon_i)  - \alpha ( c_i(\epsilon_i) \epsilon_i - B)
$$

...(8)

其中，$$\alpha$$是Lagrangian乘子。接着我们采用它在$$\epsilon_i$$上的梯度，并假设它为0

$$
\frac{\partial L(\epsilon_1,\epsilon_2,\cdots,\epsilon_n, \alpha)}{\partial \epsilon_i} = c_i^'(\epsilon_i) - \alpha(c_i^' (\epsilon_i) \epsilon_i + c_i(\epsilon_i)) = 0 \\
\frac{1}{\alpha} = \frac{c_i^' (\epsilon_i) \epsilon_i + c_i(\epsilon_i)}{c_i^'(\epsilon_i)} = \epsilon_i + \frac{c_i(\epsilon_i)}{c_i^'(\epsilon_i)}
$$

...(9) (10)

其中，对于每个交易平台i都适用于该等式。这样，我们可以使用$$\alpha$$来桥接任意两个平台i和j的等式：

$$
\frac{1}{\alpha} = \epsilon_i + \frac{c_i(\epsilon_i)}{c_i^'(\epsilon_i)} = \epsilon_j + \frac{c_j(\epsilon_j)}{c_j^'(\epsilon_j)}
$$

...(11)

因此，最优解条件给定如下：

$$
\frac{1}{\alpha} = \epsilon_1 + \frac{c_i(\epsilon_1)}{c_1^'(\epsilon_1)} = \epsilon_2 + \frac{c_2(\epsilon_2)}{c_2^'(\epsilon_2)} = ... = \epsilon_n + \frac{c_n(\epsilon_n)}{c_n^'(\epsilon_n)} \\
\sum_i c_i(\epsilon_i) \epsilon_i = B
$$

...(12)(13)

有了足够的数据样本，我们可以发现，$$c_i(\epsilon_i)$$通常是一个concave和smooth函数。一些示例如图4所示。基于该观察，可以将$$c_i(\epsilon_i)$$定义成一个通用多项式：

$$
c_i(\epsilon_i) = c_i^* a_i  (\frac{\epsilon_i}{\epsilon_i^*})^{b_i}
$$

...(14)

其中，

- $$\epsilon_i^*$$是在交易平台i在训练数据周期期间，该campaignad库存的历史平均eCPC
- $$c_i^*$$：相应的点击数（click number）

这两个因子可以直接从训练数据中获得。参数$$a_i$$和$$b_i$$可以进行调参来拟合训练数据。

等式(14)转成(12)：

$$
\frac{1}{\alpha} = \epsilon_i + \frac{c_i(\epsilon_i)}{c_i^'(\epsilon_i)} = \epsilon_i +  ... = (1+\frac{1}{b_i} \epsilon_i
$$

...(15)

我们将等式(12)重写：

$$
\frac{1}{\alpha} = (1+\frac{1}{b_1}) \epsilon_1 =  (1+\frac{1}{b_2}) \epsilon_2 = ... =  (1+\frac{1}{b_n}) \epsilon_n \\
\epsilon_i = \frac{b_i}{\alpha (b_i+1)}
$$

...(16) (17)

有意思的是，等式(17)中的equilibrium不在相同交易平台的eCPCs的state中。作为替代，当在平台间重新分配任意预算量时，不会做出更多的总点击；例如，在一个双平台的情况下，当来自一个平台的点击增加等于另一个的下降时，会达到平衡。更特别的，对于广告交易平台i，我们从等式(17)观察到，如果它的点击函数$$c_i(\epsilon_i)$$相当平，例如：在特定区域，点击数会随着eCPC的增加而缓慢增加，接着学到的$$b_i$$会很小。这意味着因子$$\frac{b_i}{b_i + 1}$$也会很小；接着等式(17)中，我们可以看到在广告交易平台i中最优的eCPC应相当小。

将等式(14)和等式(17)代入等式(7)中：

$$
\sum_i \frac{c_i^* a_i}{\epsilon_i^{* b_i}} (\frac{b_i}{b_i + 1})^{b_i + 1} (\frac{1}{\alpha})^{b_i + 1} = B
$$

...(18)

出于简洁，我们将每个ad交易平台i的参数 $$\frac{c_i^* a_i}{\epsilon_i^{* b_i}} (\frac{b_i}{b_i + 1})^{b_i + 1} $$ 表示为$$\delta_i$$。这给出了一个更简洁的形式：

$$
\sum_i \delta_i (\frac{1}{\alpha})^{b_i + 1}  = B
$$

...(19)

等式(19)对于$$\alpha$$没有封闭解（closed form slove）。然而，由于$$b_i$$非负，$$\sum_i \delta_i (\frac{1}{\alpha})^{b_i + 1} $$随着$$\frac{1}{\alpha}$$单调增，你可以轻易获得$$\alpha$$的解，通过使用一个数值求解：比如：SGD或Newton法。最终，基于求得的$$\alpha$$，我们可以发现对于每个ad交易平台i的最优的eCPC $$\epsilon_i$$。实际上，这些eCPCs是我们希望campaign对于相应的交易平台达到reference value。。。

作为特例，如果我们将campaign的整体容量看成一个channel，该方法可以被直接用于一个通用的bid optimisation tool。它会使用campaign的历史数据来决定最优化的eCPC，接着通过控制eCPC来执行click optimisation来将最优的eCPC设置为reference。注意，该multi-channel click最大化框架可以灵活地合并到任意竞价策略中。

# 4.实证研究

## 4.1 

## 4.2 

## 4.3 控制难度

## 4.4 PID setting：静态 vs. 动态references

## 4.5 click maximisation的reference setting

## 4.6 PID参数调整





# 参考

- 1.[https://arxiv.org/pdf/2007.16122.pdf](https://arxiv.org/pdf/2007.16122.pdf)