---
layout: post
title: T2Diff介绍
description: 
modified: 2025-1-7
tags: 
---

kuaishou团队在《Unleashing the Potential of Two-Tower Models: Diffusion-Based Cross-Interaction for Large-Scale Matching》提出了使用diffusion的方法来做双塔模型：

# 摘要

双塔模型在工业规模的匹配阶段被广泛采用，覆盖了众多应用领域，例如内容推荐、广告系统和搜索引擎。该模型通过分离user和item表示，有效地处理大规模候选item筛选。然而，这种解耦网络也导致了对user和item表示之间潜在信息交互的忽视。当前最先进的（SOTA）方法包括添加一个**浅层全连接层（例如，COLD）**，但其性能受限，且只能用于排序阶段。出于性能考虑，另一种方法尝试通过**将历史正向交互信息视为输入特征（例如，DAT），从另一塔中捕获这些信息**。后来的研究表明，这种方法获得的收益仍然有限，因为缺乏对下一次user意图的指导。为了解决上述挑战，我们在匹配范式中提出了一个“跨交互解耦架构”。该user塔架构利用扩散模块重建下一次正向意图表示，并采用混合注意力模块促进全面的跨交互。在生成下一次正向意图的过程中，我们通过显式提取user行为序列中的时间漂移，进一步提高了其重建的准确性。在两个真实世界数据集和一个工业数据集上的实验表明，我们的方法显著优于SOTA双塔模型，并且我们的扩散方法在重建item表示方面优于其他生成模型。

# 1 引言

推荐系统旨在通过推荐user感兴趣的内容，提升user体验和商业价值，从而促进user参与度和满意度。在工业场景中，如图1(a)所示，两阶段推荐系统被广泛用于在严格延迟要求下为user提供个性化内容。第一阶段称为**匹配阶段**，从大规模语料库中筛选出候选集。第二阶段称为**排序阶段**[1, 11]，从中选择user可能感兴趣的最终结果。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/a5cfd8c1fc5d94fc2d74f78a3da79eea080416dc975a1ada9b2a99cf841cfb6e8a814f9b85ec7476adfc6e37cf55a362?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 现实世界中的两阶段推荐系统。(a) 两阶段架构包括匹配阶段和排序阶段，匹配阶段对大量item进行评分，而排序阶段则对较小的子集进一步优化评分。(b) 匹配和排序方法在准确性和效率上的直观展示，其中所提出的匹配方法源自排序方法，并优化为一种交叉交互架构。

匹配阶段是推荐系统的关键环节，它需要从数十亿规模的语料库中快速过滤掉不相关的候选内容。由于匹配模型对高精度和低延迟的要求，**双塔模型**[13, 23, 33, 35]成为候选匹配的主流范式，并支持高效的top-k检索[26]。双塔模型由两个独立的塔组成，一个塔处理查询（user、上下文）的所有信息，另一个塔处理候选内容的信息。两个塔的输出是低维嵌入，随后通过相乘对候选内容进行评分。

由于双塔模型是独立训练的，它们无法充分利用user和item特征之间的交叉特征或交互信息，直到最后阶段才进行交互，这被称为**“晚期交互”**[17]。最近关于获取交互信号的研究主要分为两种方法。一种方法通过在双塔架构中添加一个浅层全连接层，将其转换为单塔结构（例如COLD[32]和FSCD[22]），但效率仍然受限，且仅适用于排序阶段。另一种方法尝试通过从另一个塔中捕捉历史正向交互信息的向量来增强每个塔的嵌入输入（例如DAT[35]），但最近研究表明，由于缺乏对user下一个正向意图的指导，其增益仍然有限[18]。当前的最先进方法难以在模型效果和推理效率之间取得平衡。图1(b)从推理效率和预测准确性的角度描述了上述模型。

为了解决效率与准确性之间的权衡问题，我们提出了一种生成式交叉交互解耦架构的匹配范式，名为**释放双塔模型潜力：基于扩散（diffusion）的大规模匹配交叉交互（T2Diff）**。T2Diff通过扩散模块恢复目标item的指导，提取user-item交叉特征，突破了双塔架构的限制。考虑到匹配阶段大规模语料库带来的性能问题，我们没有采用单塔结构，而是**通过生成式方法，在user塔中通过扩散模型重建item塔中包含的user正向交互**。为了充分建模user和item特征之间的交互，我们引入了一个**混合注意力模块**，以增强从另一个塔中获取的user正向交互。该混合注意力模块通过与item信息和user历史行为序列的交互，更准确地提取user表示。

本文的主要贡献如下：

- 我们提出了一种新的匹配范式**T2Diff**，它是一种生成式交叉交互解耦架构，强调信息交互，释放了双塔模型的潜力，同时实现了高精度和低延迟。
- T2Diff引入了两项关键创新：
    - 1）通过基于扩散的模型生成user的下一个正向意图；
    - 2）通过**混合注意力机制**[29, 38]在模型架构的基础层面促进更复杂和丰富的user-item特征交互，从而解决“晚期交互”的挑战。
- T2Diff不仅在两个真实世界数据集和一个工业数据集上优于基线模型，还展现了出色的推理效率。​

## 2 相关工作

### 基于嵌入的检索（Embedding-based Retrieval, EBR）
EBR 是一种使用嵌入表示user和item的技术，将检索问题转化为嵌入空间中的最近邻（NN）搜索问题[5, 15]。EBR 模型广泛应用于匹配阶段[12]，根据user的历史行为从大规模语料库中筛选候选列表。通常，EBR 模型由两个并行的深度神经网络组成，分别学习user和item的编码，这种架构也被称为**双塔模型**[13, 33, 34]。这种架构具有高吞吐量和低延迟的优势，但在捕捉user和item表示之间的交互信号方面能力有限。为了解决这一问题，DAT[35] 引入了一种自适应模仿机制，为每个user和item定制增强向量，以弥补交互信号的不足。然而，后续研究[18]表明，仅引入增强向量作为输入特征的增益有限。因此，T2Diff 利用**混合注意力模块**提取高阶特征交互和user历史行为，并结合扩散模块生成的目标表示。

### 基于会话的推荐与兴趣漂移
Feng 等人[3]观察到，user在单个会话内的行为表现出高度同质性，但在不同会话之间往往会发生兴趣漂移。Zhou 等人[37]发现，当预测与兴趣漂移趋势一致时，点击率（CTR）预测的准确性显著提高。

### 生成模型在序列推荐中的应用
尽管传统的序列模型（如 SASRec[16]、Mamba4Rec[20]）已经表现出令人满意的性能，但生成模型的出现为这一领域开辟了新的方向。变分自编码器（VAEs）[2, 8, 31]被用于学习item和user的潜在空间表示，并从中生成新序列。然而，这类生成模型可能会过度简化数据分布，导致信息丢失和表示准确性下降。扩散模型在许多领域取得了显著成功，包括推荐系统[10, 19, 30, 39]、自然语言处理[8, 14, 21]和计算机视觉[9, 24, 25]。DiffuRec[19]首次尝试将扩散模型应用于序列推荐（SR），并利用其分布生成和多样性表示的能力，采用单一嵌入捕捉user的多种兴趣。在计算机视觉中应用的 VAEs 和扩散模型[8, 14, 21]通常依赖于 Kullback-Leibler 散度损失（KL-loss）来衡量学习到的潜在分布与先验分布（通常是高斯分布）之间的差异，而 DiffuRec 在重建目标item的过程中选择了交叉熵损失。为了稳定且准确地恢复item表示，T2Diff 采用了基于 Kullback-Leibler 散度损失（KL-loss）的扩散模块。该模块能够以低延迟准确重建目标item，为在双塔结构中捕捉交叉信息提供了坚实的基础。

## 3 预备知识

在本节中，我们简要介绍扩散模型作为预备知识。

### 3.1 扩散模型

扩散模型可以分为两个阶段：**扩散过程**和**反向过程**。扩散模型的基本原理是通过在扩散过程中逐步添加高斯噪声来破坏训练数据，然后在反向过程中通过逆向去噪过程学习恢复数据。

#### 扩散过程
在扩散过程中，扩散模型通过马尔可夫链（即 $ x_0 \rightarrow x_1 \rightarrow \dots \rightarrow x_T $）逐步向原始表示 $ x_0 $ 添加高斯噪声，定义如下：

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I) \tag{1}
$$

其中：

- $ \mathcal{N}(x; \mu, \sigma^2) $ 是均值为 $ \mu $、方差为 $ \sigma^2 $ 的高斯分布。
- $ \beta_t $ 表示添加的高斯噪声的幅度，
- $ \beta_t $ 值越大，引入的噪声越多。
- $ I $ 是单位矩阵。

我们可以通过一种可处理的方式从输入数据 $ x_0 $ 推导到 $ x_T $，后验概率可以定义为：

$$
q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1}) \tag{2}
$$

根据 DDPM[9]，通过重参数化技巧，我们发现后验 $ q(x_r \mid x_0) $ 服从高斯分布。令 $ \alpha_r = 1 - \beta_r $ 且 $ \bar{\alpha}_r = \prod_{i=1}^r \alpha_i $，则公式 (2) 可以改写为：

$$
q(x_r | x_0) = \mathcal{N}(x_r; \sqrt{\bar{\alpha}_r} x_0, (1 - \alpha_r) I) \tag{3}
$$

#### 反向过程
在反向过程中，我们从标准高斯表示 $ x_T $ 逐步去噪，并以迭代方式逼近真实表示 $ x_0 $（即 $ x_T \rightarrow x_{T-1} \rightarrow \dots \rightarrow x_0 $）。特别地，给定当前恢复的表示 $ x_t $ 和原始表示 $ x_0 $，下一个表示 $ x_{t-1} $ 可以计算如下：

$$
p(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I) \tag{4}
$$

其中：

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t (1 - \bar{\alpha}_{t-1})}}{1 - \bar{\alpha}_t} x_t \tag{5}
$$

$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \tag{6}
$$

然而，在反向过程中，原始表示 $ x_0 $ 通常是未知的，因此需要深度神经网络来估计 $ x_0 $。反向过程通过最小化以下变分下界（VLB）进行优化：

$$
\mathcal{L}_{VLB} = \mathbb{E}_{q(x_1 | x_0)} [\log p_\theta(x_0 | x_1)] - D_{KL}(q(x_T | x_0) || p_\theta(x_T)) - \sum_{t=2}^T \mathbb{E}_{q(x_t | x_0)} [D_{KL}(q(x_{t-1} | x_t, x_0) || p_\theta(x_{t-1} | x_t))] \tag{7}
$$

其中，$ p_\theta(x_t) = \mathcal{N}(x_t; 0, I) $，$ D_{KL}(\cdot) $ 是 KL 散度。在 $ \mathcal{L}_{VLB} $ 中，除了 $ L_0 $ 之外，每个 KL 散度项都涉及两个高斯分布的比较，因此这些项可以以闭式解析计算。$ L_T $ 项在训练过程中是常数，对优化没有影响，因为分布 $ q $ 没有可训练的参数，且 $ x_T $ 只是高斯噪声。对于建模 $ L_0 $，Ho 等人[9] 使用了一个从 $ \mathcal{N} $ 派生的离散解码器。根据[9]，$ \mathcal{L}_{VLB} $ 可以简化为一个高斯噪声学习过程，表示为：

$$
\mathcal{L}_{simple} = \mathbb{E}_{t \in [1,T], x_0, \epsilon_t} \left[ ||\epsilon_t - \epsilon_\theta(x_t, t)||^2 \right] \tag{8}
$$

其中，$ \epsilon \sim \mathcal{N}(0, I) $ 是从标准高斯分布中采样的噪声，$ \epsilon_\theta(\cdot) $ 表示一个可以通过深度神经网络学习的估计器。

## 4 方法

在本节中，我们首先介绍与 **T2Diff** 相关的符号和背景，然后详细描述模型的框架。如图2(a)所示，我们的模型由**扩散模块**和**混合注意力模块**组成。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/737bedf89f7da5f77c140f8a63878ab529f383859e23b5304455ad7e6b618afaad9d7c19c4fec60515b9765c7f17db55?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2

### 4.1 符号与问题定义

假设我们有一个user集合 $ \mathcal{U} $ 和一个item集合 $ \mathcal{M} $。我们收集每个user的行为序列，并将其表示为 $ X_{sequence} \in \mathcal{M} $。对于user $ u \in \mathcal{U} $ 的每个行为，我们记为 $ x_j^u $，其中 $ j $ 表示行为序列中的第 $ j $ 个item。对于每个user，假设我们有 $ n $ 个历史行为，则索引 $ j \in \{1, 2, \dots, n+1\} $，且 $ X_{sequence} = [x_1, x_2, \dots, x_n] $。基于[3]中提出的概念，我们希望通过根据每个行为之间的时间间隔将行为序列划分为两个部分，从而实现对user行为序列的更精细建模。具体来说，我们将有序的行为序列划分为**当前会话**和**历史行为**，其中当前会话包含最近的 $ k $ 个交互行为，记为 $ X_{session} = [x_{n-k+1}, \dots, x_n] $，而历史行为记为 $ X_{history} = [x_1, x_2, \dots, x_{n-k}] $。我们认为，user在最近会话中的行为在时间上是连续的，反映了user最近的意图。最后，最重要的是，我们通过引入从真实行为 $ x_{n+1} $ 预测的下一个正向行为 $ \hat{x}_{n+1} $，释放了双塔模型的潜力。

基于嵌入的检索（EBR）方法通过两个独立的深度神经网络将user和item特征编码为嵌入。item $ \mathcal{M} $ 与user $ \mathcal{U} $ 的相关性基于user嵌入 $ e_u $ 和item嵌入 $ e_i $ 之间的距离（最常见的是内积）。

我们提出的 **T2Diff** 包含两个主要部分：  
1. **扩散模块**：在训练阶段识别相邻行为之间的兴趣漂移，并在推理阶段重新引入下一个行为。  
2. **基于会话的混合注意力模块**：通过自注意力模块提取最近会话中的当前兴趣，并通过目标注意力机制获取历史兴趣。这两个组件的结合实现了user行为序列与下一个行为之间的全面交叉交互。

## 4.3 混合注意力模块

为了克服双塔模型中的“晚期交互”问题，我们提出了一种**混合注意力机制**，通过将多层user表示与扩散模块（第4.2节）重建的user最近正向item表示相结合，促进复杂的特征交互。在短视频推荐领域，user消费行为表现出时间连续性。我们认为最近会话中包含了user的近期正向意图，为了增强历史序列与下一个正向item表示之间的交叉交互，我们将 $ X_{session} $ 和 $ \hat{x}_{n+1} $ 沿时间维度连接。在我们的方法中，我们部署了Transformer架构[29]的编码器组件和平均池化，以生成当前兴趣嵌入 $ h_s $，用于“早期交互”。

$$
h_s = \text{avg}(\text{Transformer}(\text{concat}([X_{session}, \hat{x}_{n+1}]))) \tag{19}
$$

为了进一步利用交叉交互的优势，我们遵循[38]，使用 $ h_s $ 作为指导，从user的历史行为 $ X_{history} $ 中提取相似信息。在激活单元中，历史行为嵌入 $ X_{history} $、当前兴趣嵌入 $ h_s $ 以及它们的外积作为输入，生成注意力权重 $ A_{history} $，如图3所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b9ab05c19afe6683e6ce04f25fdcd9bfbe5e7d439fbbbd463dfa3e77ffccc45a80bc0ffdef0a34818058456ced84ab23?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3

最终，$ h_t $ 和 $ h_s $ 共同决定user嵌入 $ e_u $。

$$
a_j = \frac{\text{FFN}(\text{concat}([x_j, x_j - h_s, x_j * h_s, h_s]))}{\sum_{i=1}^{n-k} \text{FFN}(\text{concat}([x_i, x_i - h_s, x_i * h_s, h_s]))} \tag{20}
$$

$$
h_l = f(h_s, [x_1, x_2, \dots, x_{n-k}]) = \sum_{j=1}^{n-k} a_j x_j \tag{21}
$$

$$
e_u = \text{FFN}(\text{concat}([h_l, h_s])) \tag{22}
$$

其中，$ a_j $ 是 $ A_{history} $ 的第 $ j $ 个元素。考虑到会话内的时间依赖性和跨会话行为模式的相关性，我们引入了目标行为与历史行为之间的时间滞后作为关键特征。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/97ca7a41cb2322fe095f40dc16a8dd044dd37ff38fb8aac00b23a2657056c1f2b5662d681f4c7f816ed8756ea3b10368?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=a1.jpg&amp;size=750">

算法 1

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/a6e2354e1bc306053664997a0dfcd92ac6e8bb5cf7b34ec544656c15b8c57b4aac484be23b2680d8d9c01f298c241cbb?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=a2.jpg&amp;size=750">

算法 2

## 4.4 模型优化

在每一步扩散过程中，我们直接从 $ z_r $ 推导 $ \hat{z}_0 $，其中 $ \hat{z}_0 $ 和 $ z_0 $ 都表示通过重参数化得到的分布的均值。因此，公式7中 $ \mathcal{L}_{VLB} $ 的简化版本可以改写为 $ \mathcal{L}_{KL} $，如下所示：

$$
\mathcal{L}_{KL} = \mathbb{E}_{r \in [1,T], x_0, \mu_r} \left[ ||\mu_r - \mu_\theta(z_r, r)||^2 \right] \tag{23}
$$

其中，$ \mu_r $ 和 $ z_r $ 分别表示在扩散过程第 $ r $ 步中添加的噪声和添加噪声后的结果，$ \mu_\theta $ 表示具有参数 $ \theta $ 的估计器。

在 $ \mathcal{L}_{KL} $ 的帮助下，我们可以减少 $ z_0 $ 和 $ \hat{z}_0 $ 之间的差异，并通过梯度下降更新估计器中的参数。扩散模块的扩散过程如算法1所示。

遵循推荐系统中损失函数的一般原则，我们使用softmax损失 $ \mathcal{L}_{TOWER} $ 使user嵌入 $ e_u $ 接近目标item嵌入 $ e_i $，同时远离其他不相关的item嵌入 $ e_{m \in \mathcal{M}} $，其定义为：

$$
\mathcal{L}_{TOWER} = -\log \frac{\exp(e_u \cdot e_i)}{\sum_{m \in \mathcal{M}} \exp(e_u \cdot e_m)} \tag{24}
$$

在损失函数 $ \mathcal{L}_{TOWER} $ 的驱动下，稀疏嵌入表经过充分训练，从而为扩散过程训练奠定了坚实的基础。总损失可以表示为：

$$
\mathcal{L}_{TOTAL} = \mathcal{L}_{TOWER} + \lambda \mathcal{L}_{KL} \tag{25}
$$

其中，$ \lambda $ 是一个超参数，通常设置为1或10。由于扩散模块中估计器的优化方向与传统推荐系统不一致，这容易导致梯度相互抵消的情况，因此我们采用**停止梯度机制**来隔离扩散模块的梯度更新，有效提高了估计器和塔参数的优化效率，如图2(a)底部所示。

# 5.实验

略

# 

[https://arxiv.org/pdf/2502.20687](https://arxiv.org/pdf/2502.20687)