---
layout: post
title: Contrastive Search介绍
description: 
modified: 2023-08-23
tags: 
---

Yixuan Su等人在《A Contrastive Framework for Neural Text Generation》中提出了Contrastive Search的方法：

# 摘要

文本生成（Text generation）对许多自然语言处理应用非常重要。然而，自然语言模型中基于maximization-based decoding 方法（例如：Beam Search）经常导致退化的解决方案——**生成的文本不自然并且包含不良重复**。现有的一些方法通过抽样引入随机性、或修改训练目标以减少某些tokens的概率（例如，不可能性训练）。但是，它们经常导致缺乏连贯性的解决方案。在这项工作中，我们展示了模型退化的一个潜在原因是token representations的非各向同性分布。我们提出了一个对比解决方案：

- （i）**SimCTG**：一种对比训练目标，用于校准模型的表示空间；
- （ii）一种解码方法——**对比搜索（contrastive search）**：以在生成的文本中鼓励多样性同时保持连贯性。

对两种语言的三个基准测试进行的广泛实验和分析表明，我们提出的方法在人工和自动指标评估下显着优于当前最先进的文本生成方法。

# 1.介绍

使用Transformer进行开放式神经网络文本生成（Open-ended neural text generation）[52]是各种自然语言应用中不可或缺的组成部分，例如故事生成[11，43]、上下文文本补全[36]和对话系统[48]。然而，**使用最大似然估计（MLE）训练语言模型并解码最可能的序列**的传统方法往往不够的[14，54]。具体而言，**这种建模方法通常会导致退化问题**，即从语言模型生成的文本往往会在不同级别（例如token级别、短语级别和句子级别）上变得乏味并包含不良重复[8]。为了缓解这个问题，先前的解决方案通过**从低可能性的词汇表中进行抽样**来修改解码策略[11，14]。虽然减少了生成的重复，但这些抽样方法引入了另一个关键问题（语义不一致）——抽样文本往往会**偏离或甚至与人类编写的前缀定义的原始语义相矛盾**[3]。另一种方法通过**使用unlikelihood training来修改模型的输出词汇表分布**来解决退化问题[54]。

图1

在本工作中，我们认为神经语言模型的退化源于token representations的非各向同性分布（anisotropic distribution），即它们的representations存在于整个空间的一个狭窄子集中[10，9，44]。在图1（a）中，我们展示了GPT-2生成的token表示（来自Transformer的输出层）的余弦相似度矩阵。我们看到，**句子内token之间的余弦相似度超过0.95，这意味着这些表示彼此接近**。这种高相似性是不可取的，因为它可能会自然地导致模型在不同步骤生成重复token。在理想情况下，token representations应该遵循各向同性分布，即token相似性矩阵应该是稀疏的，并且不同token的representations应该具有区分性，如图1（b）所示。此外，在解码过程中，生成的文本的标记相似性矩阵的稀疏性应该被保留以避免模型退化。

基于上述动机，我们提出了SimCTG（神经文本生成的简单对比框架），以鼓励模型学习具有区分性和各向同性的token表示。我们还提出了一种新的解码策略，以补充SimCTG，即对比搜索（contrastive search）。对比搜索（contrastive search）的核心意图是：

- （i）在每个解码步骤中，应该从模型预测的最可能候选集合中选择output，以更好地保持生成文本与人类编写的前缀之间的语义连贯性；
- （ii）应该**保留生成文本的token相似性矩阵的稀疏性**以避免退化。

我们在三个广泛使用的基准测试上进行了全面的实验。我们展示了我们的方法适用于不同的任务和不同的语言（§4和§5），以及不同的模型大小（§4.3和附录D）。具体而言，实验结果验证了SimCTG通过困惑度(perplexity)和token预测准确性的评估来提高语言模型的内在质量（§4.2和附录D）。此外，我们证明了所提出的对比搜索(contrastive search)在人工和自动评估中都要优于SOTA的解码方法（§4和§5）。此外，我们提供了深入的分析，以更好地了解我们提出的方法的内部运作机制（§6）。

# 2.背景

语言建模的目标是学习一个变长文本序列 $ x = \leftbrace x1，…，x_{\mid x \mid} \rightbrace$ 上的概率分布$p_{\theta}(x)$，其中$\theta$表示模型参数。通常，使用极大似然估计（MLE）目标来训练语言模型，该目标定义为：

$$
L_{MLE} = − \frac{1}{|x|} \sum\limits_{i=1}^{|x|} log p_{\theta}(x_i | x_{<i})
$$

...(1)

然而，正如许多最近的研究所观察到的[10，9，44]，使用极大似然估计目标进行训练，往往会产生模型表示（representations）的非各向同性分布（特别是对于基于Transformer的模型），这会削弱模型的能力。

## 2.2 开放式文本生成（Open-ended Text Generation）

在本工作中，我们专注于研究开放式文本生成任务，因为它在各种应用中具有广泛的适用性，例如故事生成[11，43]、上下文文本补全[36]、诗歌生成[23]和对话系统[48]。形式上，给定人类编写的前缀（即context）x，该任务是从语言模型中解码出一个连续的$\hat{x}$，生成的文本为：

$$
{x1，..，x_{|x|}，\hat{x}_{|x|+1}，\cdots，\hat{x}_{|x|+|\hat{x}|}}
$$

通常，有两类方法用于解码，即（1）确定性方法（Deteriminstic methods）和（2）随机方法（Stochastic methods）。

- **Deteriminstic方法**。两种广泛使用的Deteriminstic方法是贪心搜索(greedy search)和束搜索(beam search)，旨在基于模型的概率分布$p_θ$选择具有最高概率的文本延续(text continuation)。然而，**仅仅最大化输出概率往往会导致生成的文本变得乏味[22]并且出现退化问题[11，14]**。
- **Stochastic方法**：为了解决Deteriminstic解码的问题，已经提出了几种从$p_θ$中进行采样的方法。为了避免从分布的不可靠尾部进行采样，Fan等人[11]提出了top-k采样，该方法从最大化$\sum_{v \in V^{(k)}} p_θ(v\mid x)$的词汇子集V(k)中抽取样本。这里，$\mid V(k) \mid=k$，x是前缀上下文。与之不同的是，当前SOTA的核心采样(nucleus sampling)[14]从具有总概率大于阈值$p \in [0，1]$的最小词汇子集U中抽取样本；即，U是最小的词汇子集，使得$\sum_{v \in U} p_θ(v \mid x)≥p$。虽然采样方法有助于缓解模型退化，但这些方法中的内在随机性可能会导致采样文本的语义含义与人类编写的前缀发生分歧甚至矛盾[3]。


# 3.方法

在本节中，我们首先介绍如何将对比学习应用于校准语言模型的表示空间。然后，我们介绍我们提出的对比搜索（contrastive search decoding）解码算法。

## 3.1 Contrastive Training

我们的目标是：鼓励语言模型学习具有区分性（discriminative）和各向同性（isotropic）的token representations。为此，我们在语言模型的训练中引入了对比目标$L_{CL}$。具体而言，给定一个变长序列$ x = \lbrace x1，\cdots，x_{\mid x \mid} \rbrace $，$L_{CL}$定义为：

$$
L_{CL} = \frac{1}{|x| \times (|x|−1)} \sum\limits_{i=1} \sum\limits_{j=1,j \neq i}^{|x|} max \lbrace 0, \rho − s(h_{x_i} , h_{x_i}) + s(h_{x_i} , h_{x_j}) \rbrace
$$

...(2)

其中:

- $\rho \in [−1，1]$ 是预定义的边界
- $h_{x_i}$：是模型生成的token $x_i$的表示
- 相似性函数s：计算标记表示之间的余弦相似度，如下所示：

$$
s(h_{x_i}, h_{x_j}) = \frac{h_{x_i}^T h_{x_j}} { \| h_{x_i} \| \cdot \| h_{x_j} \|}
$$

...(3)

直观地说，通过使用$L_{CL}$进行训练，模型学习将不同token的表示之间的距离拉开。因此，可以获得具有区分性和各向同性的模型表示空间。总体训练目标$L_{SimCTG}$定义为: 

$$
L_{SimCTG} = L_{MLE} + L_{CL}
$$

...(4)

其中，最大似然估计（MLE）目标$L_{MLE}$的定义如公式（1）所示。请注意，当$L_{CL}$中的边界ρ等于0时，$L_{SimCTG}$会退化为普通的MLE目标$L_{MLE}$。

## 3.2 Contrastive Search

我们提出了一种新的解码(decoding)方法，对比搜索(contrastive search)。在每个解码步骤中，对比搜索(contrastive search)的关键思想是：

- （i）生成的输出（generated output）应该从模型预测的最可能的候选集合中选择；
- （ii）生成的输出（generated output）应该足够具有区分性，以便与前文（previous context）相关。通过这种方式，生成的文本可以更好地保持与前缀的语义连贯性，同时避免模型退化（model degeneration）。

形式上，给定在timestep t时的上文 $x_{<t}$，输出$x_t$的选择遵循以下过程：

$$
x_t = \underset{v \in V^{(k)}}{argmax} \lbrace (1-\alpha) \times \underbrace{p_{\theta}(v | x_{<t>})}_{model \ confidence} - alpha \times \underbrace{(max \lbrace s(h_v, h_{x_j}): 1 \leq j \leq t-1 \rbrace)}_{degeneration \ penalty} \rbrace
$$

...(5)

其中：

- $V(k)$是模型的概率分布$p_\theta(\cdot \mid x_{<t})$中前k个预测的集合，k通常设置为3∼10。

在公式（5）中, 

- 第一项“模型置信度(model confidence)”是模型预测的候选项v的概率
- 第二项“退化惩罚(degeneration penalty)”衡量候选项v相对于前文$x_{<t}$的区分性，s在公式（3）中定义。具体而言，它被定义为候选项v的表示与$x_{<t}$中所有token的表示之间的最大余弦相似度。在这里，候选项的表示$h_v$是由模型给出的，给定x<t和v的连接。

直观地说，较大的退化惩罚意味着候选项与上下文更相似，因此更有可能导致模型退化。超参数$\alpha \in [0，1]$调节了这两个组成部分的重要性。当α=0时，对比搜索退化为贪心搜索方法。

# 其它

略

# 

- 1.[https://arxiv.org/pdf/2202.06417.pdf](https://arxiv.org/pdf/2202.06417.pdf)