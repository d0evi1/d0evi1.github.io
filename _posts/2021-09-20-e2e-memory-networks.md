---
layout: post
title: end2end Memory Networks介绍
description: 
modified: 2021-09-19
tags: memory-network
---


facebook在《MEMORY NETWORKS》中提出了memory networks，在另一文《End-To-End Memory Networks》提出了end2end的实现：

# 摘要

我们会介绍一种使用recurrent attention model的neural network，它构建在一个可能很大的外部memory之上。该架构与Memory network的形式一样，但与其中的model不同的是：**它的训练是end-to-end的，因而在训练期间几乎是无监督学习**，这使得它很容易应用到实际环境中。它可以看成是关于RNNsearch[2]的一个扩展：**其中在执行每次output symbol时会执行多个计算steps（hops）**。该模型的灵活性允许我们将它应用到像QA等不同的任务以及语言建模上。对比起之前的Memory Networks的研究，它几乎是无监督的。我们在Penn TreeBank和Text8 datasets上验证了我们的方法要好于RNNs和LSTMs。

# 1.介绍

在AI 研究领域，在QA或补全任务上，构建模型会涉及多个计算步骤，模型可以描述序列数据中的long-term dependencies。


最近的一些工作，在建模时会使用显式存储和attention；维持这样的一个storeage对于解决这样的挑战提供了一种方法。在[23]中，存储会通过一个continuous representation给出；从该存储进行读取和写入，可以通过neural networks的actions进行建模。

在本工作中，我们提出了一种新的RNN结构，其中：**在输出一个symbol之前，recurrence会从一个可能很大的external memory中读取多次**。我们的模型可以被看成是在[23]中的Memory Network的一个continuous form。本工作中的模型通过BP训练并不容易，需要在该network的每个layer上进行监督。模型的连续性意味着：它可以从input-output pairs通过end-to-end的方式进行训练，因此很容易应用到许多任务上，例如：语言建模或者真实的QA任务。我们的模型可以看成是RNNsearch[2]的一个版本，它在每个output symbol上具有多个计算步骤。我们会通过实验展示：在long-term memory上的多跳对于我们的模型的效果来说很重要，训练memory representation可以以可扩展的方式进行集成到end-to-end neural network model上。

# 2.方法

我们的模型会采用：

- 一个关于inputs $$x_1, \cdots, x_n$$的离散集合，它们会被存储到memory中
- 一个query q
- 输出一个answer a

**$$x_i, q, a$$的每一个都包含了来自具有V个词的字典的symbols**。该模型会将所有x写到memory中，直到达到一个确定的buffer size，接着我们会为x和q寻找一个连续的representation。这会允许在训练期间，error signal的BP通过多个memory accesses回到input。

## 2.1 Single Layer

我们在single layer的case中开始描述我们的模型，它会实现一个单个memory hop操作。我们接着展示了它可以被stack来给出在memory上的多跳。

**Input memory representation**

假设我们给定一个input set $$x_1, \cdots, x_i$$被存到memory中。$$\lbrace x_i \rbrace$$的整个集合会被转到d维memory vectors $$\lbrace m_i \rbrace$$中，它通过在一个连续空间上嵌入每个$$x_i$$计算得到，在最简单的case中，使用一个embedding matrix A（其中：size=$$d \times V$$）。**query q也会被嵌入来获得一个internal state u**。在embedding space中，我们会计算在u和每个memory $$m_i$$间的match程度，通过以下公式对内积采用softmax得到：

$$
p_i = Softmax(u^T m_i)
$$

...(1)

其中，$$Softmax(z_i) = \frac{ e^{z_i} } {\sum_j e^{z_j}}$$。在该方式中，p是在inputs上的一个概率向量（probability vector）。

**Output memory representation**

每个$$x_i$$都具有一个相应的output vector $$c_i$$。memory o的response vector是一个在transformed input $$c_i$$与来自input的probability vector进行加权求和:

$$
o = \sum_i p_i c_i
$$

...(2)

由于从input到output的函数是smooth的，我们可以轻易地计算gradients以及BP。其它最近提出的memory或attention形式也采用该方法【2】【8】【9】。

**生成最终的prediction**

在single layer case中，output vector o和input embedding u接着通过一个最终的weight matrix W（size为$$V \times d$$）和一个softmax来生成predicted label：

$$
\hat{a} = Softmax(W(o+u))
$$

...(3)

整体模型如图1(a)所示。在训练期间，所有三个embedding matrics A, B, C，以及W都通过最小化$$\hat{a}$$和 true label a间的一个标准的cross-entropy loss进行联合学习。训练会使用SGD进行执行。

图1

## 2.2 Multiple Layers

我们现在扩展我们的模型来处理K跳的操作。memory layers会以如下方式进行stack：

- 在第一个之上的layers的input，是从layer k的output $$o^k$$和input $$u^k$$的求和（后续会有不同组合）：

$$
u^{k+1} = u^k + o^k
$$

...(4)

- 每个layer会具有它自己的embedding matrics $$A^k, C^k$$，用于嵌入到inputs $$\lbrace x_i \rbrace$$中。然而，如下所示，他们会被限制以便减轻训练、减少参数数目.

- 在network的顶层，W的input也会将top memory layer的input和output进行组合：

$$
\hat{a} = Softmax(W u^{K+1}) = Softmax(W(o^K + u^K))
$$

我们探索了在模型中两种类型的weight tying机制：

- 1.Adjacent：一个layer的output embedding是下一个的input embedding，例如：$$A^{k+1} = C^k$$。我们也会限制：a) answer prediction matrix会与最终的output embedding相似，例如：$$W^T = C^K$$， b) question embedding会与第一层的input embedding相匹配，例如：$$B = A^1$$
- 2.Layer-wise（RNN-like）：input和output embeddings对于不同的layers是相同的，例如：$$A^1 = A^2 = \cdots = A^K$$以及$$C^1 = C^2 = \cdots = C^K$$。我们已经发现：添加一个线性映射H到在hops间的u的update上是有用的；也就是说：$$u^{k+1} = H u^k + o^k$$。该mapping会随着剩余参数学习，并在我们的实验上用于layer-wise weight tying。

图1(b)展示了一个3-layer版本。总体上，它与[23]中的Memory Network相似，除了每一layer中的hard max操作已经使用了一个来自softmax的continuous weighting替换外。

注意，如果我们使用layer-wise weight tying scheme，我们的模型可以被转成一个传统的RNN，其中我们会将RNN的outputs分割成internal和external outputs。触发一个internal output可以与考虑一个memory相对应，触发一个external output对应于预测一个label。从RNN的角度，图1(b)和等式(4)的u是一个hidden state，该模型会使用A生成一个internal output p（图1(a)中的attention weights）。该模型接着会使用C来吸收p，并更新hidden state等。这里，与一个标准RNN不同的是，我们会在K hops期间，显式的基于在memory中储存的outputs作为条件，我们会采用soft的方式来保存这些outputs，而非对它们采样。这样，我们的模型会在生成一个output之前做出一些计算step，这意味着被“外部世界”见过。

# 其它 

略

- 1.[https://arxiv.org/pdf/1503.08895.pdf](https://arxiv.org/pdf/1503.08895.pdf)