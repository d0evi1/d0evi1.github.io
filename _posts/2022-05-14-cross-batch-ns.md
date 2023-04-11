---
layout: post
title: cross-batch negative sampling介绍
description: 
modified: 2022-05-10
tags: 
---


huawei在2021《Cross-Batch Negative Sampling for Training Two-Tower Recommenders》中提出了一种cross-batch negative sampling的方法：


# 摘要

双塔结构被广泛用于学习item和user representations，它对于大规模推荐系统来说是很重要的。许多two-tower models会使用多样的in-batch negative sampling的策略，**这些策略的效果天然依赖于mini-batch的size**。然而，使用大batch size的双塔模型训练是低效的，它需要为item和user contents准备一个**大内存容量**，并且在feature encoding上消耗大量时间。有意思的是，我们发现，neural encoders在训练过程中在热启（warm up）之后对于相同的input可以输出相对稳定的features。基于该事实，我们提出了一个有效的sampling策略：称为“Cross-Batch Negative Sampling (CBNS)”，**它可以利用来自最近mini-batches的encoded item embeddings来增强模型训练**。理论分析和实际评估演示了CBNS的有效性。

# 3.模型框架

## 3.1 问题公式

我们考虑对于large-scale和content-aware推荐系统的公共设定。我们具有两个集合：

- $$U = \lbrace U_i \rbrace_i^{N_U}$$
- $$I = \lbrace I_j \rbrace_i^{N_I}$$

其中：

- $$U_i \in U$$和$$I_j \in I$$是features（例如：IDs, logs和types）的预处理vectors集合

在用户为中心的场景，给定一个带features的user，目标是：检索一个感兴趣items的子集。通常，我们通过设置两个encoders（例如：“tower”）：

$$f_u: U \rightarrow R^d, g_v: I \rightarrow R^d$$

之后我们会通过一个scoring function估计user-item pairs的相关度：

$$s(U,I) = f_u(U)^T g_v(I) \triangleq u^T v$$

其中：

- u,v分别表示来自$$f_u, g_v$$的user、item的encoded embeddings

## 3.2 基础方法

通常，大规模检索被看成是一个带一个（uniformly）sampled softmax的极端分类问题（extreme classification）：

$$
p(I | U; \theta) = \frac{e^{u^T v}}{e^{U^T v} + \sum\limits_{I^- \in N} e^{U^T v^-}}
$$

...(1)

其中：

- $$\theta$$表示模型参数
- N是sampled negative set
- 上标“-”表示负样本

该模型会使用cross-entropy loss（等价为log-likelihood）进行训练：

$$
L_{CE} = - \frac{1}{|B|} \sum\limits_{i \in [|B|]} log p(I_i | U_i; \theta)
$$

...(2)


<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/669e4ea943060b43daaf5329584fe4511fac84c2f4914bf1d2bd0b391d3b21744c2c137a19bb5727af0aab6b82888827?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 双塔模型的采样策略

为了提升双塔模型的训练效率，一个常用的抽样策略是in-batch negative sampling，如图1(a)所示。具体的，**它会将相同batch中的其它items看成是负样本（negatives），负样本分布q遵循基于item frequency的unigram分布**。根据sampled softmax机制，我们修改等式为：

$$
P_{In-batch} (I | U; \theta) = \frac{e^{s'(U,I;q)}}{e^{s'(U,I;q)} + \sum_{I^- \in B \ \lbrace I \rbrace} e^{s'(U, I^-;q)}} \\
s'(U, I;q) = s(U,I) - log q(I) = u^T v - log q(I) 
$$

...(3)(4)

其中：

- logq(I) 是对sampling bias的一个correction。

In-batch negative sampling会避免额外additional negative samples到item tower中，从而节约计算开销。不幸的是，**in-batch items的数目batch size线性有界的，因而，在GPU上的受限batch size会限制模型表现**。

## 3.3 Cross Batch Negative Sampling

### 3.3.1 Nueral model的embedding稳定性（embedding stability of neural model）

由于encoder会在训练中**持续更新**，来自过往mini-batches的item embeddings通常会被认为是过期并且丢弃。然而，因为embedding stability of neural model，我们会识别这样的信息，并且被复用成一个在当前mini-batch的valid negatives。我们会通过估计item encoder $$g_v$$的feature drift【26】来研究该现象，feature drift定义如下：

$$
D(I, t; \Delta t) \triangleq \sum\limits_{I \in I} \| g_v(I; \theta_g^t) - g_v(I; \theta_g^{t - \Delta t}) \|_2
$$

...(5)

其中：

- $$\theta_g$$是$$g_v$$的参数
- $$t, \Delta_t$$分别表示训练迭代数和训练迭代间隔（例如：mini-batch）

我们会从头到尾使用in-batch negative softmax loss来训练一个Youtube DNN，并计算具有不同间隔$$\lbrace 1,5,10 \rbrace$$的feature drift。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/75432a7ac6f16b0d4041f4a44d5f2b1122bbb17770a9f8be5de0e30b2f8765ec6dbab6354c99499ae5ede5d7aa7a53c6?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2  YoutubeDNN的Feature drift w.r.t. Δts, 数据集：Amazon-Books dataset

如图2所示，features会在早期激烈变化。**随着learning rate的减小，在$$4 \times 10^4$$次迭代时features会变得相对稳定，使得它可以合理复用它们作为合法负样本（valid negatives）。我们将这样的现象称为“embedding stability”**。我们进一步以公理3.1方式展示：embedding stability会提供一个关于scoring function的gradients error上界，因此， stable embeddings可以提供合法信息进行训练。

**引理3.1 假设：$$\| \hat{v}_j - v_j \|_2^2 < \epsilon$$，scoring function的output logit是$$\hat{o}_{ij} \triangleq u_i^T \hat{v}_j$$**并且user encoder $$f_u$$满足Lipschitz continuous condition，接着：gradient w.r.t user $$u_i$$的偏差为：

$$
|| \frac{\partial \hat{o}_{ij}}{\partial \theta} - \frac{\partial o_{ij}}{\partial \theta} ||_2^2 < C \epsilon
$$

...(6)

其中：C是Lipschitz常数。

证明：近似梯度error可以被计算为：

$$
\| \frac{\partial \hat{o}_{ij}}{\partial \theta} - \frac{\partial o_{ij}}{\partial \theta} \|_2^2 = \| \frac{\partial \hat{o}_{ij}}{\partial \theta} - \frac{\partial o_{ij}}{\partial \theta} ) \frac{\partial u_i}{\partial \theta} \|_2^2 = \| (\hat{v}_j - v_j) \frac{\partial u_i}{\partial \theta} \|_2^2 \\
\leq \| \hat{v}_j - v_j \|_2^2 \|\frac{\partial u_i}{\partial \theta} \|_2^2 \leq \| \frac{\partial f_u(U_i;\theta^t)}{\partial \theta} \|_2^2 \epsilon \leq C \epsilon
$$

...(7)(8)

经验上，$$C \leq 1$$持有。因而，gradient error可以被embedding stability控制。

### 3.3.2 对于Cross Batch Features使用FIFO Memory Bank

由于embeddings在早期变化相对剧烈，我们会使用naive in-batch negative sampling对item encoder进行warm up到$$4 \times 10^4$$次迭代，它会帮助模型逼近一个局部最优解，并生成stable embeddings。接着，我们开始使用一个FIFO memory bank $$M = \lbrace (v_i, q(I_i)) \rbrace_{i=1}^M$$训练推荐系统，其中$$q(I_i)$$表示在unigram分布q下item $$I_i$$的抽样概率，其中M是memory size。cross-batch negative sampling（CBNS）配合FIFO memory bank如图1(b)所示，CBNS的softmax的output被公式化为：

$$
p_{CBNS}(I | U; \Theta) = \frac{e^{s'(U,I;q)}}{e^{e'(U,I;q) + \sum_{I^- \in M U B \\lbrace I \rbrace} e^{s'(U,I^-;q)}}}
$$

...(9)

在每次迭代的结尾，我们会enqueue该embeddings 以及 对应当前mini-batch的抽样概率，并将早期的数据进行dequeue。注意，我们的memory bank会随embeddings更新，无需任何额外计算。另外，memory bank的size相对较大，因为它对于这些embeddings不需要更多memory开销。

# 4.实验与结果



[https://dl.acm.org/doi/pdf/10.1145/3404835.3463032](https://dl.acm.org/doi/pdf/10.1145/3404835.3463032)