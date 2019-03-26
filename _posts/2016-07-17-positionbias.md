---
layout: post
title: coec介绍
description: 
modified: 2015-11-17
tags: 
---

microsoft在《Position-Normalized Click Prediction in Search Advertising》对coec做了介绍。

# 1.介绍

竞价排名搜索广告的ctr预估系统的目标是：根据其它上下文知识（比如：用户信息），对一个query-ad pair估计CTR。ctr预估对于ad排序、位置分配（allcation），定价（pricing），以及回报（payoff）很关键。通过估计得到的CTR作为query-ad相关度的一种衡量，并且与其它非相关因子相互独立。然而实际上，有许多外围因子影响着基于相关度的ctr系统，通常会在观察到的点击数据（click-through data）上扮演着重要角色。一个经典的示例是：**广告展现位置（ad presentation position）**。这些外在因子必须小心处理，否则会导致一个次优的ctr预估，许多真实世界的系统都会存在这种缺陷。

我们提出了一个概率因子模型（probabilistic factor model）作为一个总的原则性方法来研究这些效应。该模型很简单并且是线性的，在广告领域会做出经验性的调整。对于纠正在搜索算法上的**位置偏差(positional bias)**有大量研究，许多研究都是：检测模型（examination model）[12]，cascade model[5], 动态贝叶斯网络模型(DBN)[3]，而对于搜索广告领域却很少。我们的方法采用了与examination model相似的因子分解假设，也就是说：在item上的点击概率，是一个关于位置先验概率和一个基于相关度的与位置无关的概率的乘积。再者，我们会专门研究广告领域的位置概念，通过合并其它广告特有的重要信号，例如：query-ad keyword match type和广告总数。

来自搜索算法的其它模型(比如：cascade和DBN模型)，通常会假设：**一个item的估计得到的ctr（estimated CTR）是与展示在搜索结果页的items的相关度(relevance)相互独立的**。这些更复杂的假设对于搜索算法结果更合适，其中用户对于结果链接之一上的点击行为具有一个高概率。然而对于广告，在广告上的点击概率总是相当低，通常是一个百分比。因此，效果越高（非点击）的广告是相互接近的因子的乘积。

# 2.因子模型

假设：

- i表示一个query-ad pair
- j表示ad的位置
- c表示点击次数 
- v表示曝光次数

观察到的CTR是一个条件概率 $$p(click \mid i,j)$$。对于在竞价搜索广告中的经验假设，我们做出如下简化：

- 1.一个ad的点击是与它的位置相互独立的 (假设：位置可以物理上进行检查examining)。
- 2.给定一个ad的位置，检查（examining）一个广告是与它的内容或相关度是相互独立的

正式的，位置依赖的CTR(position-dependent CTR)可以表示为：

$$
p(click | i,j) = p(click | exam, i) p(exam | j)
$$

...(1)

其中：

- 第一个因子 $$p(click \mid exam, i)$$：可以简单表示为$$p_i$$，它是一个位置归一化的CTR（position-normalized CTR），可以表示ad的相关度
- 第二个因子 $$p(exam \mid j)$$，可以简单表示为$$q_j$$，体现了位置偏差（ positional bias）。

有了该CTR因子分解（factorization），我们可以处理关于点击行为的两种自然随机模型，接着在部署模型上通过一个先验进行平滑。【1,9】

# 3. Binomial模型

很自然的，假设点击数遵循一个二项分布：

$$
c_{ij} \sim Binomial(v_{ij}, p_i q_j),  \forall i,j
$$
。。。

# 4.POISSON模型

如果尝试次数n足够大，成功概率p (success probability)足够小，那么$$Binomial(n,p) \rightarrow Poisson(np)$$。由于广告(ad)就是这样一个领域，我们可以导出Poisson模型，它会生成一个相似的且足够有效的更新。该生成模型是：

$$
c_{ij} \sim Poisson(v_{ij} p_i q_j), \forall i,j
$$

...(9)

...

# 5.GAMMA-POISSON模型

对于empirical和regularization的目的，我们在Poisson模型中在位置因子(positional factor)上引入了一个gamma先验：

$$
q_j \sim Gamma(\alpha, \beta), \forall j
$$

...(16)

经验上，观察CTR(observed CTR)几何上会随位置的降低而递减【11】，展示出与gamma信号有一个很好的拟合。实例上，次一点的位置（inferior positions，比如：side bar的底部位置）可能会遭受严峻的数据稀疏性问题，尤其是点击；因此，正则化(regularizing)或平滑(smoothing)这些噪声估计会产生更好的泛化。gamma分布是一个常见的选择，因为它是Poisson的一个共轭先验。

。。。

# 6.点击模型

**一个点击模型或CTR预测模型的目标是：为给定的一个query-ad pair，估计一个位置无偏CTR(positional-unbiased CTR)**，例如：相关度CTR(relevance CTR) $$p(click \mid exam,i)$$或$$p_i$$。上述描述的位置归一化点击模型(positional
normalized click model)会做这样的处理，同时也会发生位置先验概率 $$p(exam \mid j)$$或 $$q_j$$。factor模型的另一个观点是：在ad位置上做过平滑的kNN模型；当特征空间只包含了query-ad pairs，k=1。对于有足够历史点击数据的query-ad pairs这是可信的，因子模型可以合理执行。而对于一个冷启动问题原则性处理方法是，将queries和ads的一元特征（unigram features）添加到特征空间中，当在预测时遇到新的pairs时对CTR预估做backing-off。

位置归一化点击模型也可以被独立应用，并联合其它点击模型来估计relevance-only CTR。更严厉的，我们会所设位置因子与其它相关度因子相互独立。在模型训练时，需要通过它的位置先验$$v_{ij} q_j$$来归一化每个ad曝光。在预测时，CTR预测器会从位置归一化的训练数据中进行学习，并生成完全的relevance-only CTR。

# 7.实验

## 7.1 使用人造数据仿真

我们首先在一个通过概率模型（例如：给定一个sound模型）生成的人造数据集上仿造了Gamma-Poisson模型。通过仔细设计模型参数，人造数据可以十分模仿真实的搜索广告数据。尽管模仿数据不能完全反映真实系统，但它至少有两个优点：

- 1.当从真实噪声中进行抽象时，允许快速研究大量参数
- 2.通过该数据曝露的真实分布来验证学到的模型，对于真实数据很重要

数据生成如下：

- 1. $$\foreach$$ 位置 $$position j \in [1,...,m]$$，生成一个$$q_j \sim Gamma(\alpha,\beta)$$，以降序对q排序，通过$$1/q_1$$对q进行缩放
- 2.


# 参考

- 1.[http://wan.poly.edu/KDD2012/docs/p795.pdf](http://wan.poly.edu/KDD2012/docs/p795.pdf)
