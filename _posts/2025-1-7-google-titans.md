---
layout: post
title: google Titans介绍
description: 
modified: 2025-1-7
tags: 
---

google在《Titans: Learning to Memorize at Test Time》提出了区别于Transformer的的一种新架构:Titans。我们来看一下它的实现，是否有前景：

# 摘要

在过去的十多年里，关于如何有效利用循环模型（recurrent models）和注意力机制（attentions）的研究已经非常广泛。**循环模型的目标是将数据压缩到一个固定大小的内存中（称为hidden state），而注意力机制则允许模型关注整个上下文窗口，捕捉所有token之间的直接依赖关系**。然而，这种更精确的依赖关系建模带来了二次方的计算成本(quadratic cost)，限制了模型只能处理固定长度的上下文。我们提出了一种新的**神经长期记忆模块（neural long-term memory module）**，该模块能够学习记忆历史上下文，并帮助注意力机制在利用过去信息的同时关注当前上下文。我们展示了这种神经记忆模块**具有快速并行化训练的优势，同时保持了快速的推理能力**。从记忆的角度来看，我们认为注意力机制由于其有限的上下文但精确的依赖关系建模，起到了短期记忆的作用；而神经记忆（neural memory）由于其能够记忆数据的能力，起到了长期、更持久的记忆作用。基于这两个模块，**我们引入了一系列新的架构，称为Titans**，并提出了三种变体，以探讨如何有效地将记忆融入这一架构中。我们在语言建模、常识推理、基因组学和时间序列任务上的实验结果表明，Titans比Transformers和最近的现代线性循环模型更有效。此外，与基线模型相比，Titans能够有效地扩展到超过200万的上下文窗口大小，并在“大海捞针”任务中表现出更高的准确性。

# 1.介绍

Transformers，一种纯基于注意力机制的架构（Vaswani 等人，2017），已被牢牢确立为序列建模中的最先进模型，主要归功于其上下文学习能力和大规模学习能力（Kaplan 等人，2020）。Transformers 的核心构建模块——注意力模块——充当关联记忆模块（Bietti 等人，2024），它们学习存储key-value关联性（associations），并通过计算query（即搜索信号）和key（即上下文）之间的成对相似性来检索这些关联。因此，从设计上看，Transformer 的输出完全取决于当前上下文窗口中token的直接依赖关系。然而，这种精确的依赖关系建模带来了**与上下文长度相关的二次方时间和内存复杂度**。在复杂的现实任务中（例如语言建模（N. F. Liu 等人，2024）、视频理解（C.-Y. Wu 等人，2019）、长期时间序列预测（H. Zhou 等人，2021）），上下文窗口可能变得非常大，这使得 Transformers 在这些下游任务中的适用性面临挑战。

为了克服 Transformers 的可扩展性问题，最近的研究旨在设计不同变体的线性 Transformers（Kacham、Mirrokni 和 P. Zhong，2024；Katharopoulos 等人，2020；S. Yang、B. Wang、Shen 等人，2024），其中**注意力机制中的 softmax 被核函数取代（详见 §2.1），从而显著降低了内存消耗**。尽管线性 Transformers 具有高效性并能够扩展到更长的上下文，但与 Transformers 相比，**它们的性能并不具有竞争力，因为核技巧使模型变成了线性循环网络**，其中数据被压缩为矩阵值状态（Katharopoulos 等人，2020）。然而，这带来了关于线性循环（或线性 Transformers）模型的一个矛盾事实：一方面，我们使用这些线性模型来增强可扩展性和效率（线性与二次方复杂度），其优势在非常长的上下文中显现；另一方面，非常长的上下文无法被适当地压缩到一个小的向量值或矩阵值状态中（S. Wang，2024）。


此外，除了效率问题外，大多数现有架构——从 Hopfield 网络（Hopfield，1982）到 LSTM（Jürgen Schmidhuber 和 Hochreiter，1997）以及 Transformers（Vaswani 等人，2017）——**在处理泛化、长度外推和/或推理（Anil 等人，2022；Qin、Y. Zhong 和 Deng，2024）时都面临挑战，**而这些是许多复杂现实任务中不可分割的部分。尽管这些架构从人类大脑中汲取了灵感，但它们都缺少以下关键部分：

- （1）学习过程中的关键组件——例如短期记忆、长期记忆、元记忆、关注当前上下文等（Cowan，2008）；
- （2）这些组件如何作为可以独立运行的互联系统；以及/或
- （3）从数据中主动学习并记忆过去历史的抽象能力。

我们认为，在一个有效的学习范式中，类似于人类大脑，存在独立但相互关联的模块，每个模块都负责学习过程中至关重要的组件。

### 记忆视角

记忆（memory）是一种基本的心理过程，也是人类学习中不可分割的组成部分（Terry，2017）。**如果没有一个正常运作的记忆系统，人类和动物将只能局限于基本的反射和刻板行为**。因此，记忆一直是机器学习文献中许多开创性研究的灵感来源；例如，Hopfield 网络（Hopfield，1982）、LSTM（Jürgen Schmidhuber 和 Hochreiter，1997）以及 Transformers（Vaswani 等人，2017）。

从神经心理学文献中对记忆和学习的常见定义中汲取灵感（Okano、Hirano 和 Balaban，2000），大多数现有架构将记忆视为由输入引起的神经更新，并将学习定义为在给定目标的情况下获取有效且有用记忆的过程。从这个角度来看，循环神经网络（RNN）（Williams 和 Zipser，1989）可以被定义为具有**向量值记忆模块 M（也称为hidden state）**的模型，其主要步骤包括：

在时间 𝑡 给定新输入 $𝑥_𝑡$ 时，模型

- （1）使用函数 $𝑓(M_{𝑡−1}, 𝑥_𝑡) $ 更新记忆（带有压缩）；
- （2）使用函数 $𝑔(M_𝑡, 𝑥_𝑡)$ 检索输入的相应记忆（详见 §2.1）。

类似地，Transformers 可以被视为具有增长记忆和两个相似步骤的架构。即，**key和value矩阵对充当模型的记忆**，模型：

- （1）通过将key和value附加到记忆中来更新记忆（无压缩），
- （2）通过查找query向量与key向量的相似性来检索query向量的相应记忆，然后将其用于加权value向量以生成输出。

这种视角可以帮助我们更好地理解现有范式、它们的关键差异，并设计更有效的架构。例如，Transformers（Vaswani 等人，2017）和线性 Transformers（Katharopoulos 等人，2020）之间的主要区别在于记忆结构以及记忆更新步骤，其中：**线性 Transformers 将历史数据压缩为固定大小的矩阵值记忆，而 Transformers 则保留所有历史数据（在上下文长度内）而不进行任何压缩**。虽然线性 Transformers 和线性 RNN（包括状态空间模型）都在记忆更新步骤中压缩信息，但关键区别在于记忆的结构，其中线性 RNN（相对于线性 Transformers）使用向量值记忆（相对于矩阵值记忆）。因此，这种视角促使我们提出以下问题：

- （Q1）什么是**良好的记忆结构**？
- （Q2）什么是**适当的记忆更新机制**？
- （Q3）什么是**良好的记忆检索过程**？

重新审视我们对人类记忆的理解，它既不是一个单一的过程，也不服务于单一的功能（Cowan，2008）。事实上，记忆是一个系统的联合体——例如**短期记忆、工作记忆（working memory）和长期记忆**——每个系统服务于不同的功能，具有不同的神经结构，并且每个系统都能够独立运行（Willingham，1997）。这一事实促使我们提出：

- （Q4）如何设计一个包含**不同互联记忆模块的高效架构**。

最后，存储记忆是一个神经过程，需要对过去的抽象进行编码和存储。假设一个单一向量或矩阵（其参数以线性方式编码数据）足以存储长期历史可能过于简化。

- （Q5）是否需要**深度记忆模块**来有效存储/记住遥远的过去？

### 贡献与路线图

在本文中，我们旨在通过设计一个长期神经记忆模块来回答上述五个问题，该模块能够在测试时高效且有效地学习记忆。基于其设计，我们讨论了如何将其融入架构中。

**神经记忆（§3）**。我们提出了一种（深度）神经长期记忆模块，它（作为元上下文模型）学习如何在测试时将数据记忆/存储到其参数中。受人类长期记忆系统（Mandler，2014）的启发，

我们设计了这个记忆模块，使得违反预期的事件（即令人惊讶的事件： surprising）更容易被记住。为此，我们通过神经网络在关联记忆损失中对输入的梯度来衡量输入的“惊讶度（surprise）”（详见 §3.1）。为了更好地处理有限的内存，我们提出了一种衰减机制，该机制考虑了内存大小与数据惊讶度的比例，从而实现更好的内存管理。我们展示了这种衰减机制实际上是现代循环模型中遗忘机制的泛化（Dao 和 Gu，2024；Gu 和 Dao，2024；S. Yang、Kautz 和 Hatamizadeh，2024）。有趣的是，我们发现这种机制等同于使用小批量梯度下降、动量和权重衰减来优化元神经网络。基于张量化小批量梯度下降以使用更多矩阵乘法操作（Yu Sun 等人，2024），我们提出了一种快速且可并行化的算法来训练我们的深度神经长期记忆模块。

### Titans 架构（§4）

在设计完长期神经记忆模块后，一个重要的问题是：**如何高效且有效地将记忆融入深度学习架构中**。我们提出了 **Titans**，这是一个由三个超头部（hyper-heads）组成的深度模型家族：

- （1）**核心模块**：该模块包含短期记忆，负责数据处理的主要流程（我们使用有限窗口大小的注意力机制）；
- （2）**长期记忆模块**：这一分支是我们的神经长期记忆模块，负责存储/记住遥远的过去；
- （3）**持久记忆模块**：这是一组可学习但与数据无关的参数，用于编码任务相关知识。

最后，作为概念验证，我们提出了 Titans 的三种变体，其中我们将记忆分别融入为：

- （i）一个上下文（context）
- （ii）层（layer）
- （iii）一个门控分支（gated branch）

### 实验结果（§5）
我们在语言建模、常识推理、记忆密集型任务、“大海捞针”任务、时间序列预测和 DNA 建模任务上进行了实验评估。我们观察到，Titans 架构在所有现代循环模型及其混合变体（结合滑动窗口注意力机制）的综合基准测试中均表现优异。此外，**Titans 在相同上下文窗口下优于 Transformers，并在使用整个上下文的 Transformers 中表现出竞争力。这些结果是在 Titans 能够扩展到超过 200 万上下文窗口大小的情况下实现的，而 Transformers 则无法做到这一点**。

## 2 预备知识

在本节中，我们将讨论本文中使用的符号和一些背景概念。我们令：

- $ x \in \mathbb{R}^{N \times d_{\text{in}}} $ 表示输入
- $ \mathbf{M} $ 表示神经网络（神经记忆模块：neural memory）
- $ \mathbf{Q} $、$ \mathbf{K} $、$ \mathbf{V} $ 分别表示注意力机制中的query、key和value
- $ M $ 表示注意力掩码（attention mask）
-  $ S^{(i)} $ 表示：在对序列进行分段时，使用第 $ i $ 段。

在本文中，我们简化符号并使用下标来指代矩阵、向量或段中的特定元素。例如，我们令：

-  $ S^{(i)}_j $ 表示第 $ i $ 段中的第 $ j $ 个 token。

唯一的例外是下标为 $ t $ 的情况，我们保留它来表示时间上的递归或神经网络在时间 $ t $ 的状态。

给定：神经网络 $ \mathbf{N} $ 和数据样本 $ x $，我们使用：

- $ \mathbf{N}(x) $（或 $ \mathbf{N}^*(x) $）表示带权重调整（或不带权重调整）的前向传播

此外，我们简化符号并使用：

- $ \mathbf{N}^{(k)} $ 表示神经网络的第 $ k $ 层

接下来，我们首先讨论注意力机制及其高效变体的背景，然后回顾现代线性 RNN，最后讨论这些架构的记忆视角，这促使我们设计了 Titans。

### 2.1 背景

**注意力机制**。Transformers（Vaswani 等人，2017）作为许多深度学习模型的实际骨干，基于注意力机制。给定：

- 输入 $ x \in \mathbb{R}^{N \times d_{\text{in}}} $

因果注意力机制基于输入依赖的key、value和query矩阵计算输出 $ y \in \mathbb{R}^{N \times d_{\text{in}}} $：

$$
\mathbf{Q} = x \mathbf{W}_Q, \quad \mathbf{K} = x \mathbf{W}_K, \quad \mathbf{V} = x \mathbf{W}_V, \quad (1)
$$

$$
y_i = \frac{\sum_{j=1}^i \exp\left(\frac{\mathbf{Q}_i^\top \mathbf{K}_j}{\sqrt{d_{\text{in}}}}\right) \mathbf{V}_j}{\sum_{\ell=1}^i \exp\left(\frac{\mathbf{Q}_i^\top \mathbf{K}_\ell}{\sqrt{d_{\text{in}}}}\right)}, \quad (2)
$$

其中：

-  $ W_Q, W_K, W_V \in R^{d_{in} \times d_{in}} $ 是可学习参数

尽管 Transformers 在召回能力上表现出色，但它们至少需要 $ N \times d $ 次操作来计算输出，导致内存消耗较大且对较长序列的吞吐量较低。

**高效注意力机制**。为了提高软注意力机制在长序列上的内存消耗和吞吐量，许多研究集中在注意力机制的 I/O 感知实现（Dao 2024；Dao, D. Fu 等人，2022），通过稀疏化注意力矩阵（B. Chen 等人，2021；Choromanski 等人，2021；Dai 等人，2019）、近似 softmax（Arora 等人，2024）或开发基于核的（线性）注意力机制（Aksenov 等人，2024；Kacham, Mirrokni 和 P. Zhong，2024；Schlag, Irie 和 Jürgen Schmidhuber，2021；S. Yang, B. Wang, Shen 等人，2024）来设计更高效的注意力机制。在本部分，我们重点关注后者，即线性注意力机制，其中标准注意力中的 softmax 被替换为替代核函数 $ \phi(\cdot, \cdot) $，使得 $ \phi(x, y) = \phi(x) \phi(y) $。因此，注意力可以写成：

$$
y_i = \frac{\sum_{j=1}^i \phi(\mathbf{Q}_i^\top \mathbf{K}_j) \mathbf{V}_j}{\sum_{\ell=1}^i \phi(\mathbf{Q}_i^\top \mathbf{K}_\ell)} = \frac{\phi(\mathbf{Q}_i)^\top \sum_{j=1}^i \phi(\mathbf{K}_j) \mathbf{V}_j}{\phi(\mathbf{Q}_i)^\top \sum_{\ell=1}^i \phi(\mathbf{K}_\ell)}, \quad (3)
$$

由于：

- 项 $ \sum_{j=1}^i \phi(K_j), \sum_{\ell=1}^i \phi(K_\ell) $ 在每一步中重复使用，因此吞吐量更高。

当选择核函数为单位矩阵时（Yutao Sun 等人，2023），上述公式可以写成递归形式：

$$
\mathbf{M}_t = \mathbf{M}_{t-1} + \mathbf{K}_t^\top \mathbf{V}_t, \quad (4)
$$

$$
y_t = \mathbf{Q}_t \mathbf{M}_t, \quad (5)
$$

这使得线性注意力机制能够高效推理。

**现代线性模型及其记忆视角**。如前所述，可以将学习定义为获取有效且有用记忆的过程。基于此，可以将循环神经网络（RNN）的hidden state视为记忆单元，模型旨在将信息压缩到其中。因此，在一般形式的循环神经网络中，hidden state可以被视为记忆单元，递归过程可以分为记忆单元的读和写操作。即，令：

- $ x \in \mathbb{R}^{N \times d_{\text{in}}} $ 为输入
- $ \mathbf{M} \in \mathbb{R}^d $ 为记忆单元
- $ y \in \mathbb{R}^{d_{\text{in}}} $ 为输出

则循环神经网络的一般形式定义为：

$$
\mathbf{M}_t = f(\mathbf{M}_{t-1}, x_t), \quad \text{写操作} \quad (6)
$$

$$
y_t = g(\mathbf{M}_t, x_t), \quad \text{读操作} \quad (7)
$$

其中：

- $ f(\cdot, \cdot) $ 是读操作，
- $ g(\cdot, \cdot) $ 是写操作。

注意，这里的 $ \mathbf{M}_t $ 下标表示记忆在时间 $ t $ 的状态。

从这一视角来看，线性 Transformers 的递归公式（见公式 4）等同于将键和值 $ (\mathbf{K}_t, \mathbf{V}_t) $ 加性地压缩并写入矩阵值记忆单元 $ \mathbf{M}_t $ 中。因此，**在处理长上下文数据时，这种加性特性会导致内存溢出，显著损害模型性能**。为了解决这一问题，研究集中在两个有前景的方向上：

- （1）**添加遗忘机制**：一些研究提出了线性模型的自适应（数据依赖）遗忘门机制，可以在需要时擦除记忆。例如，GLA（S. Yang, B. Wang, Shen 等人，2024）、LRU（Orvieto 等人，2023）、Griffin（De 等人，2024）、xLSTM（Beck 等人，2024）和 Mamba2（Dao 和 Gu，2024）等模型，后者还与离散化的传统状态空间模型（Gu 和 Dao，2024）相关联。
- （2）**改进写操作**：为了克服传统循环模型中记忆写操作的加性特性，Widrow 和 Hoff（1988）提出了 Delta 规则，在添加记忆（即键值对）之前，模型首先移除其过去的值。为了增强可并行化训练和扩展性，S. Yang, B. Wang, Yu Zhang 等人（2024）提出了一种快速并行化算法。最后，最近 S. Yang, Kautz 和 Hatamizadeh（2024）通过添加遗忘门改进了 DeltaNets。

**记忆模块**。记忆一直是神经网络设计的核心部分之一（Graves, Wayne 和 Danihelka，2014；JH Schmidhuber，1992；Jürgen Schmidhuber 和 Hochreiter，1997；J. Zhang 等人，2024）。将线性层视为键值（关联）记忆系统（key-value (associative) memory system）的思想可以追溯到快速权重程序（fast weight programs），其中动态快速程序被纳入循环神经网络中作为可写记忆（JH Schmidhuber，1992）。Hebbian（Hebb，2005）和 delta（Prados 和 Kak，1989）学习规则是快速权重程序中最流行的学习规则，已在各种研究中广泛探索（Irie, Schlag 等人，2021；Munkhdalai, Sordoni 等人，2019；Munkhdalai 和 H. Yu，2017；Schlag, Irie 和 Jürgen Schmidhuber，2021；JH Schmidhuber，1992；S. Yang, Kautz 和 Hatamizadeh，2024；S. Yang, B. Wang, Yu Zhang 等人，2024）。然而，所有这些模型都基于瞬时惊讶度，忽略了序列中的 token 流（见第 3.1 节），并且大多数模型缺乏遗忘门，导致内存管理不佳。

我们在附录 C 中进一步讨论了我们的架构与最近模型的联系。其他相关工作在附录 A 中讨论。

## 3 测试时的记忆学习

为了克服长期记忆的不足，并使模型能够学习、遗忘和检索信息，本节提出了一种神经长期记忆模块，这是一种在测试时学习记忆的元模型。

- 在 3.1 节中，我们首先讨论神经记忆的动机和设计。
- 在 3.2 节中，我们讨论如何通过快速且可并行化的训练使我们的架构设计受益。
- 在 3.3 节中，我们通过持久记忆模块增强我们的架构，其中使用可学习但与数据无关的参数来学习任务的元信息。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/f0bc800bcb932d47cbb6b7a900a42e721e299f33e0b4ad93d21647782585d6905b27255c9aac50b4bb6dce1188db6ff4?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 关于如何并行训练神经记忆并使用矩阵乘法（matmuls）的图示说明。

### 3.1 长期记忆

为了设计一个神经长期记忆模块，我们需要一个能够**将过去历史的抽象编码到其参数中**的模型。一个例子是大语言模型（LLMs），它们被证明能够记忆训练数据（Leybzon 和 Kervadec，2024；Schwarzschild 等人，2024；Staab 等人，2024）。因此，一个简单的想法是：训练一个神经网络并期望它记忆其训练数据。然而，**记忆化几乎总是被认为是神经网络中的不良现象，因为它限制了模型的泛化能力（Bayat 等人，2024），引发隐私问题（Staab 等人，2024）**，并导致测试时性能不佳。此外，训练数据的记忆化在测试时可能没有帮助，因为数据可能是分布外的。我们认为，我们需要一个在测试时学习如何记忆/遗忘数据的在线元模型。在这种设置中，模型学习的是一个能够记忆的函数，但它不会过度拟合训练数据，从而在测试时实现更好的泛化。

**学习过程与惊讶度度量(Surprise Metric)**。训练长期记忆的关键思想是：将其训练视为一个在线学习问题，我们的目标是将过去的信息 $ x_1, \ldots, x_{t-1} $ 压缩到长期神经记忆模块 $ \mathbf{M}_t $ 的参数中。如前所述，**违反预期的事件（即令人惊讶的事件）对人类来说更容易被记住**（Mandler，2014）。受此启发，模型惊讶度的一个简单定义可以是：其相对于输入的梯度。**梯度越大，输入数据与过去数据的差异越大**。因此，使用这种惊讶度评分，我们可以更新记忆如下：

$$
\mathbf{M}_t = \mathbf{M}_{t-1} - \theta_t \nabla \ell(\mathbf{M}_{t-1}; x_t) \quad \text{（惊讶度）} \quad (8)
$$

然而，这种惊讶度度量可能会导致错过在重大惊讶时刻之后的重要信息。也就是说，梯度在几次惊讶步骤后可能变得非常小，导致陷入平坦区域（即局部最小值），并错过序列某些部分的信息。从人类记忆的角度来看，一个事件可能不会在长时间内持续让我们感到惊讶，尽管它是值得记忆的。原因是初始时刻足够令人惊讶，足以在长时间内吸引我们的注意力，从而记住整个时间段。为了改进上述惊讶度度量（公式 8），我们将惊讶度度量分为：

- （1）**过往惊讶度（past surprise）**，衡量最近过去的惊讶程度；
- （2）**瞬时惊讶度（momentary surprise）**，衡量输入数据的惊讶程度：

$$
\mathbf{M}_t = \mathbf{M}_{t-1} + S_t, \quad (9)
$$

$$
S_t = \eta_t S_{t-1} \quad \text{（过去惊讶度）} - \theta_t \nabla \ell(\mathbf{M}_{t-1}; x_t) \quad \text{（瞬时惊讶度）} \quad (10)
$$

有趣的是，这个公式类似于带有动量的梯度下降，其中：

- $ S_t $ 是动量项

因此，这里的动量充当了跨时间（序列长度）的惊讶度记忆。在这个公式中：

- 项 $ \eta_t $ 是一个数据依赖的惊讶度衰减（$ x_t $ 的函数），**控制惊讶度随时间衰减的程度**
- 项 $ \theta_t $ 则控制瞬时惊讶度应以数据依赖的方式纳入最终惊讶度度量的多少

这种数据依赖性在这个设计中尤为重要：虽然前一个 token 的惊讶度可能需要影响下一个 token 的惊讶度，但这只有在所有 token 都相关且处于同一上下文中时才有效。因此，数据依赖的 $ \eta $ 可以控制记忆是否需要：

- （1）通过设置 $ \eta_t \to 0 $ 忽略上一次的惊讶度（可能由于上下文的变化），
- （2）通过设置 $ \eta_t \to 1 $ 完全纳入上一次的惊讶度（可能因为 token 与其最近的过去 token 高度相关）。

**目标**。我们上述的惊讶度度量基于损失函数 $ \ell(\cdot; \cdot) $，这是我们的记忆模块在测试时学习的目标。也就是说，我们的记忆模块是一个元模型，它基于损失函数 $ \ell(\cdot; \cdot) $ 学习一个函数。

在本节中，我们重点讨论**关联记忆**，其目标是将过去的数据存储为k-V对。给定输入 $ x_t $，类似于 Transformers（Vaswani 等人，2017），我们使用两个线性层将 $ x_t $ 投影为key和value：

$$
\mathbf{k}_t = x_t \mathbf{W}_K, \quad \mathbf{v}_t = x_t \mathbf{W}_V, \quad (11)
$$

其中 $ W_K $ 和 $ W_V \in R^{d_{in} \times d_{in}} $。接下来，我们希望记忆模块能够学习键和值之间的关联。为此，我们定义损失函数如下：

$$
\ell(\mathbf{M}_{t-1}; x_t) = \|\mathbf{M}_{t-1}(\mathbf{k}_t) - \mathbf{v}_t\|_2^2, \quad (12)
$$

通过在元模型（记忆）的内循环中优化上述损失函数，模型学习如何在测试时记忆键和值之间的映射。需要注意的是，类似于元学习模型（Nichol，2018；Zintgraf 等人，2019），记忆的训练是在内循环中进行的，因此参数 $ \mathbf{W}_K $ 和 $ \mathbf{W}_V $ 是上述损失函数中的超参数。因此，在内循环中，我们优化记忆模块 $ \mathbf{M} $ 的权重，而在外循环中，我们优化整个架构的其他参数。

### 遗忘机制
当处理非常长的序列（例如数百万个 token）时，**管理哪些过去信息应该被遗忘至关重要**——即使使用深度或非常大的矩阵值记忆。为此，我们使用一种**自适应遗忘机制**，允许记忆遗忘不再需要的信息，从而更好地管理记忆的有限容量。具体来说，给定下一个 token $ x_t $，我们修改更新规则如下：

$$
\mathbf{M}_t = (1 - \alpha_t) \mathbf{M}_{t-1} + S_t, \quad (13)
$$

$$
S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(\mathbf{M}_{t-1}; x_t), \quad (14)
$$

其中：

- $ \alpha_t \in [0, 1] $ 是一个门控机制，灵活控制记忆；即决定应该遗忘多少信息。例如：

- 通过设置 $ \alpha_t \to 0 $，可以在不影响过去抽象的情况下更新记忆；
- 通过设置 $ \alpha_t \to 1 $，可以清除整个记忆

在本节后面，我们将展示这种权重衰减机制与现代 RNN 中的门控机制密切相关（Dao 和 Gu，2024；Orvieto 等人，2023）。

### 记忆架构

在本文中，我们专注于使用具有 $ L_M \geq 1 $ 层的简单多层感知机（MLP）作为长期记忆的架构。选择这种架构的主要原因是，我们希望集中精力更好地激励长期记忆的设计及其融入架构的方式。然而，我们的公式和架构设计为设计在数据记忆方面更有效和高效的神经架构开辟了新的研究方向。最近，有一些有前景的工作致力于设计此类架构（Berges 等人，2024；Cetin 等人，2024；J. Zhang 等人，2024），将这些架构融入我们的框架（即用此类架构替换简单的 MLP）可能是一个有趣的未来工作方向。

当使用向量值或矩阵值记忆（De 等人，2024；Orvieto 等人，2023；S. Yang, B. Wang, Shen 等人，2024）时，记忆模块会压缩过去的数据并将其拟合到一条线上。也就是说，从元学习或在线学习的角度来看（Yu Sun 等人，2024），使用矩阵值记忆 $ \mathbf{M} = \mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}} $ 等同于优化 $ \ell(\mathbf{W}_{t-1}; x_t) = \|\mathbf{W}_{t-1} \mathbf{k}_t - \mathbf{v}_t\|_2^2 $，这是一个在线线性回归目标，因此最优解假设历史数据的潜在依赖关系是线性的。另一方面，我们认为深度记忆模块（即 $ L_M \geq 2 $ 层）在实践中更有效。这与理论结果一致，即至少具有两层的 MLP 严格比线性模型更具表达能力（Hornik, Stinchcombe, and White, 1989）。在第 5.5 节中，我们展示了深度记忆模块在实际应用中的有效性。

---

### 记忆检索

在上面，我们讨论了如何设计和训练一个在测试时学习记忆的长期记忆模块。一个关键的问题是：如何从记忆中检索信息？我们简单地使用不更新权重的前向传播（即推理）来检索与查询对应的记忆。形式上，给定输入 $ x_t $，我们使用线性层 $ \mathbf{W}_Q $ 投影输入，即 $ \mathbf{q}_t = x_t \mathbf{W}_Q $，并通过以下方式从记忆中检索相应的（或有用的）信息 $ y_t $：

$$
y_t = \mathbf{M}^*(\mathbf{q}_t). \quad (15)
$$

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/0c621ebc9b99450603f059ad66182055d64db73c7152ff85ed07ce909b1959efece6b3cab8645cfbe69c4a71f75bd6b0?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2：记忆作为上下文（MAC）架构。该架构包括三个分支：（1）核心分支，（2）上下文（长期）记忆分支，以及（3）持久记忆分支。核心分支将相应的长期记忆和持久记忆与输入序列连接起来。接下来，注意力机制在序列上执行，并决定哪些信息应存储在长期记忆中。在测试时，与上下文记忆对应的参数仍在学习，与核心分支对应的参数负责上下文学习，而持久记忆的参数负责存储任务知识，因此是固定的。

### 3.2 如何并行化长期记忆的训练

如上所述，我们的长期记忆模块的设计等同于通过优化关联记忆损失函数 $ \ell(\mathbf{M}_{t-1}; x_t) = \|\mathbf{M}_{t-1}(\mathbf{k}_t) - \mathbf{v}_t\|_2^2 $ 来训练一个元模型，使用带有动量和权重衰减的梯度下降。因此，理论上，长期记忆模块的训练需要 $ O(N) $ 的浮点运算（FLOPs），其中 $ N $ 是序列长度。然而，在实践中，我们需要并行化训练过程，并充分利用硬件加速器（例如 TPU、GPU），因此需要将过程张量化并使用更多的矩阵乘法（matmuls）。

接下来，我们展示如何通过小批量梯度下降、数据依赖的学习率和权重衰减来重新表述内循环中的权重计算，使其仅使用矩阵乘法和求和。我们基于 Yu Sun 等人（2024）的工作，该工作表明，使用小批量梯度下降（具有恒定学习率）优化的模型的前向传播可以通过矩阵乘法计算。我们可以将序列分割为大小为 $ b \geq 1 $ 的块，并将小批量梯度下降表示为：

$$
\mathbf{M}_t = (1 - \alpha_t) \mathbf{M}_{t-1} - \theta_t \nabla \ell(\mathbf{M}_{t-1}; x_t) = \beta_t \mathbf{M}_0 - \sum_{i=1}^t \theta_i \frac{\beta_t}{\beta_i} \nabla \ell(\mathbf{M}_{t'}; x_i), \quad (16)
$$

其中 $ t' = t - \text{mod}(t, b) $，且 $ \beta_i = \prod_{j=1}^i (1 - \alpha_j) $。为了简化，我们专注于第一个块，即 $ t = b $，因此 $ t' = 0 $。此外，我们解释当 $ \mathbf{M}_t = \mathbf{W}_t $ 是线性时的情况。对于具有 $ N_p \geq 2 $ 层的 MLP，过程类似。使用我们的损失函数，我们有：

$$
\nabla \ell(\mathbf{W}_0; x_t) = (\mathbf{W}_0 x_t - x_t) x_t^\top \Rightarrow \sum_{i=1}^b \theta_i \frac{\beta_b}{\beta_i} \nabla \ell(\mathbf{W}_0; x_i) = \Theta_b \mathbf{B}_b (\mathbf{W}_0 \mathbf{X} - \mathbf{X}) \mathbf{X}^\top, \quad (17)
$$

其中 $ \Theta_b = \text{diag}(\theta_1, \theta_2, \ldots, \theta_b) $，且 $ \mathbf{B}_b $ 类似地定义在 $ \frac{\beta_b}{\beta_i} $ 上。需要注意的是，我们不需要存储所有 $ \Theta_{kb} $ 和 $ \mathbf{B}_{kb} $（$ k = 1, \ldots, N/b $），而是为每个块存储这些矩阵，从而减少内存使用。接下来，我们扩展这种表示，以便还可以纳入动量项。在带有动量的小批量梯度下降中，如果我们看动量项，我们有：

$$
S_t = \eta_t S_{t-1} - \theta_t u_t, \quad (18)
$$

其中 $ u_t = \nabla \ell(\mathbf{M}_{t'}; x_t) $。需要注意的是，我们可以同时计算所有 $ u_t $，因此公式 (18) 是一个线性递归，其中 $ u_t $ 是输入，$ S_t $ 是隐藏状态，$ \eta_t $ 是输入依赖的转移值。因此，我们可以使用并行关联扫描（J. T. Smith, Warrington, and Linderman, 2023）来计算该块中的 $ S_t $。

### 参数作为块的函数
与其让参数 $ \alpha_t $、$ \theta_t $ 和 $ \eta_t $ 依赖于输入（即 token $ x_t $ 的函数），我们可以让它们成为块的函数。尽管这会降低表达能力，但这种表述可以帮助使训练更快。在这种情况下，我们在每个块中对 $ \alpha $、$ \theta $ 和 $ \eta $ 使用相同的值。因此，在公式 (17) 中，我们可以使用单个标量存储 $ \Theta $。类似地，我们可以使公式 (18) 更快。也就是说，当 $ \eta $ 和 $ \theta $ 在每个块内可学习但时间不变时，该方程变为线性时不变系统（LTI），可以通过全局卷积计算（Gu, Goel, and Re, 2022）。在我们的实验中，我们将这些参数作为 token 的函数。然而，这种简化（即作为块的函数）可能是未来工作的兴趣点，以便以更高效的方式训练更大的模型。

### 3.3 持久记忆

我们的长期记忆也可以被视为一种上下文记忆，这意味着输出完全依赖于上下文。因此，除了长期记忆外，我们还使用一组可学习但与输入无关的参数来充当任务相关的记忆。这种类型的记忆在文献中被称为持久记忆或元记忆（X. Dong 等人，2024；Sukhbaatar, Grave 等人，2019）。给定 $ N_p \geq 1 $，我们使用可学习参数 $ P = [p_1, p_2, \ldots, p_{N_p}] $ 并将其附加到序列的开头：即，给定上下文窗口大小为 $ N $，我们将输入修改为：

$$
x_{\text{new}} = [p_1, p_2, \ldots, p_{N_p}] \parallel x, \quad (19)
$$

其中 $ \parallel $ 表示连接操作。接下来，我们从三个角度讨论持久记忆的动机：

---

#### 记忆视角
如前所述，我们的神经长期记忆是一种上下文记忆，其中所有参数都依赖于输入。然而，一个有效的记忆系统还需要与输入无关的参数来存储任务知识的抽象。也就是说，掌握一个任务需要记忆如何完成该任务的知识，而这些参数负责存储此类知识。

---

#### 前馈网络视角
在 Transformer 架构中，注意力模块之后有全连接层，这些层被证明类似于注意力权重，但具有与数据无关的参数。即，Sukhbaatar, Grave 等人（2019）表明，将全连接层中的 ReLU 替换为 Softmax 可以产生类似注意力的权重，其中权重与数据无关：

$$
FFN(x) = W_V \text{Softmax}(W_K x), \quad (20)
$$

实际上，当 $ W_K $ 和 $ W_V $ 与输入无关时，它们的作用类似于注意力模块中的 $ K $ 和 $ V $ 矩阵。持久记忆权重预计具有相同的功能，这意味着在序列的开头部分使用它们会导致具有与输入无关的注意力权重（Sukhbaatar, Grave 等人，2019）。

---

#### 技术视角
带有因果掩码的注意力机制对序列中的初始 token 具有隐式偏差，因此注意力权重几乎总是对初始 token 高度活跃，从而导致性能下降。从技术角度来看，序列开头的这些可学习参数可以通过更有效地重新分配注意力权重来缓解这种影响（Han 等人，2024；Xiao 等人，2024）。

---

**总结**：
- **持久记忆的作用**：存储任务知识的抽象，与输入无关。
- **前馈网络的类比**：持久记忆权重类似于注意力机制中的 $ K $ 和 $ V $ 矩阵，但具有与数据无关的特性。
- **技术优势**：通过在序列开头引入可学习参数，持久记忆可以缓解注意力机制对初始 token 的偏差，从而提升模型性能。

持久记忆的引入为模型提供了任务知识的存储能力，并通过优化注意力权重的分配进一步提升了模型的性能。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/d96cb43ad141c18aa7d8d3c412509b64a3c961382266904dfaa6a45a4ca8d7f16a03e20e38a78ad30b09b0103a4f4e6b?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3

### 4 如何融入记忆？

一个重要但尚未解答的问题是：如何有效且高效地将设计的神经记忆融入深度学习架构中？如前所述，从记忆的角度来看，Transformers 中的键值对矩阵可以解释为关联记忆块。由于其依赖关系的精确建模以及有限的上下文窗口，我们将其解释为短期记忆模块，专注于当前上下文窗口大小。另一方面，我们的神经记忆能够从数据中持续学习并将其存储在其权重中，可以扮演长期记忆的角色。在本节中，我们旨在通过提出 Titans 的三种不同变体来回答上述问题。在后续实验中，我们将展示每种变体的优缺点，以及在超长上下文中的效率与有效性之间的权衡。

---

### 4.1 记忆作为上下文（Memory as a Context, MAC）

在第一种架构设计中（见图 2），我们将记忆视为当前信息的上下文。即，给定一个长序列 $ x \in \mathbb{R}^{N \times d_{\text{in}}} $，我们首先将序列分割为固定大小的段 $ S^{(i)} $（$ i = 1, \ldots, N/C $）。给定传入的段 $ S^{(t)} $，我们将其视为当前上下文，并将其过去的段视为历史信息。因此，设 $ \mathbf{M}_{t-1} $ 为段 $ S^{(t)} $ 之前的长期记忆状态，我们使用输入上下文作为查询，从长期记忆 $ \mathbf{M}_{t-1} $ 中检索相应的信息。即，我们检索与 $ S^{(t)} $ 对应的过去信息：

$$
h_t = \mathbf{M}^*_{t-1}(\mathbf{q}_t), \quad (21)
$$

其中 $ \mathbf{q}_t = S^{(t)} \mathbf{W}_Q $。接下来，我们将此历史信息与持久记忆参数一起作为注意力模块的输入序列：

$$
\tilde{S}^{(t)} = [p_1, p_2, \ldots, p_{N_p}] \parallel h_t \parallel S^{(t)}, \quad (22)
$$

$$
y_t = \text{Attn}(\tilde{S}^{(t)}). \quad (23)
$$

整个序列的注意力图结构如图 3a 所示。然后，我们使用 $ y_t $ 更新长期记忆模块以用于下一个段和最终输出：

$$
\mathbf{M}_t = \mathbf{M}_{t-1}(y_t), \quad (24)
$$

$$
o_t = y_t \otimes \mathbf{M}^*_t(y_t). \quad (25)
$$

需要注意的是，在上述过程中，我们通过前向传播更新 $ \mathbf{M}_{t-1} $ 的权重。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/c1709441b29d0e40245933eba1486c7d57b850ae79581d04c0c1dacd706709f2d431e9f9fa5ecbb3349789ec71835704?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=4.jpg&amp;size=750">

图4

该架构具有两个关键优势：
1. **注意力机制**：通过同时拥有历史和当前上下文，能够决定在当前数据下是否需要长期记忆信息。
2. **记忆管理**：注意力模块帮助长期记忆仅存储当前上下文中有用的信息，避免记忆溢出。
3. **测试时的行为**：
   - 持久记忆参数固定，因为它们编码了任务知识，不应更改。
   - 注意力模块权重是上下文学习者。
   - 长期记忆模块在测试时仍在学习（记忆）信息，即神经记忆的权重在测试时也会更新，以编码过去的抽象。

---

### 4.2 门控记忆（Gated Memory）

在第二种变体中（见图 4），我们在一个分支中直接使用输入数据更新长期记忆，在另一个分支中使用滑动窗口注意力（SWA）：

$$
\tilde{x} = [p_1, p_2, \ldots, p_{N_p}] \parallel x, \quad (26)
$$

$$
y = \text{SW-Attn}^*(\tilde{x}), \quad (27)
$$

$$
o = y \otimes \mathbf{M}(\tilde{x}), \quad (28)
$$

其中 $ \text{SW-Attn}^* $ 是带有前缀的滑动窗口注意力（见图 3b）。需要注意的是，与之前的设计不同，我们不对输入数据进行分段。此外，我们简化符号并使用 $ \mathbf{M}(x) $ 表示序列中所有 token 递归后的记忆最终输出。在上述等式中，$ \otimes $ 可以是任何非线性门控。在我们的实验中，我们使用可学习的向量值权重对输出 $ y $ 和 $ \mathbf{M}(\tilde{x}) $ 进行归一化，然后应用非线性函数 $ \sigma(\cdot) $。

该设计的整体注意力掩码如图 3b 所示。在这种设计中，滑动窗口注意力充当精确的短期记忆，而神经记忆模块则充当模型的衰减记忆。这种架构设计也可以被视为一种多头架构，其中头的结构不同（X. Dong 等人，2024）。

### 4.3 记忆作为层（Memory as a Layer, MAL）

在第三种变体中，我们将神经记忆作为深度神经网络的一层（见图 5）。这种架构设计在文献中更为常见，其中混合模型将循环模型与完整或滑动窗口注意力机制堆叠在一起。给定输入 $ x $，我们有：

$$
\tilde{x} = [p_1, p_2, \ldots, p_{N_p}] \parallel x, \quad (29)
$$

$$
y = \mathbf{M}(\tilde{x}), \quad (30)
$$

$$
o = \text{SW-Attn}(y), \quad (31)
$$

其中 $ \text{SW-Attn} $ 是滑动窗口注意力。这种设计的主要缺点是模型的能力受限于每一层，因此无法充分利用注意力和神经记忆模块的互补数据处理能力。在我们的实验中，为了评估这种设计中的记忆，我们使用了类似于 H3（D. Y. Fu 等人，2023）的架构，其中我们将序列模型替换为我们的神经记忆模块（LMM）。

---

#### 无注意力的记忆
尽管上述讨论中我们将 MAL 视为 LMM 和注意力机制的顺序组合，但 MAL 的一个简单变体是将 LMM 视为没有任何注意力机制的序列模型。从记忆的角度来看，如第 1 节所述，我们期望记忆系统的每个部分都能独立工作，即使其他组件受到干扰。因此，即使没有短期记忆（即注意力机制），长期记忆模块仍然应该是一个强大的模型。我们在实验中称这种变体为 LMM 或 Titans（LMM）。我们在附录 C 中提供了关于 Titans 与其他现代循环模型联系的更多讨论。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/c770139a7caba343aa9698629816c844dc5899746a3052e22f2b6c2070356bd09cdc3465ee070af2675416850f71a2e8?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=5.jpg&amp;size=750">

图5

---

### 4.4 架构细节

为了简洁和清晰，我们避免讨论实现细节，例如使用残差连接、线性层门控和归一化。在所有块中，我们使用残差连接。在我们的实现中，我们使用 SiLU(.) 激活函数（Elfwing, Uchibe, and Doya, 2018）作为计算查询、键和值的非线性激活，并使用 $ \ell_2 $-范数对查询和键进行归一化。

---

#### 卷积
遵循最近的现代线性循环模型（Gu 和 Dao，2024；S. Yang, Kautz, and Hatamizadeh，2024），我们在每个查询、键和值投影之后加入一维深度可分离卷积层。虽然这些一维卷积对性能的影响不大，但它们已被证明可以提升性能，并且在计算上也很高效。

---

#### 门控
我们还遵循最近的架构，在最终输出投影之前使用归一化和线性层门控（Mehta 等人，2023）。

---

### 定理 4.1

与 Transformers、对角线性循环模型和 DeltaNet 不同，这些模型都受限于 $ \text{TC}^0 $（Merrill, Petty, and Sabharwal, 2024），Titans 能够解决超出 $ \text{TC}^0 $ 的问题，这意味着 Titans 在状态跟踪任务中理论上比 Transformers 和大多数现代线性循环模型更具表达能力。

# 

[https://arxiv.org/pdf/2501.00663v1](https://arxiv.org/pdf/2501.00663v1)