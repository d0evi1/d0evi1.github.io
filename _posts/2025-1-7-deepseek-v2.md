---
layout: post
title: DeepSeekV2介绍
description: 
modified: 2025-1-7
tags: 
---

Deepseek AI在《DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model》详细介绍了V2版的实现：

# 1.介绍

在过去的几年中，大型语言模型（LLMs）（Anthropic, 2023; Google, 2023; OpenAI, 2022, 2023）经历了快速发展，为**通用人工智能（AGI）**的曙光提供了初步的展望。通常，随着参数数量的增加，LLM的智能水平会显著提升，使其能够在各种任务中展现出**涌现（emergent）**能力（Wei et al., 2022）。然而，这种改进的代价是：**更大的训练计算资源需求以及推理吞吐量的潜在下降**。这些限制对LLM的广泛采用和利用构成了重大挑战。为了解决这一问题，我们提出了**DeepSeek-V2**，一个强大的开源混合专家（MoE）语言模型，其特点是通过创新的Transformer架构实现经济的训练和高效的推理。该模型总参数量为236B，每个token激活21B参数，并支持128K token的上下文长度。

我们在Transformer框架（Vaswani et al., 2017）中优化了注意力模块和前馈网络（FFNs），分别提出了**多头隐注意力（MLA）**和**DeepSeekMoE**。

1. **在注意力机制方面**，多头注意力（MHA）（Vaswani et al., 2017）的**键值（KV）缓存对LLM的推理效率构成了显著障碍**。为了解决这一问题，研究者们探索了多种方法，包括分组查询注意力（GQA）（Ainslie et al., 2023）和多查询注意力（MQA）（Shazeer, 2019）。然而，这些方法在减少KV缓存的同时往往会牺牲性能。为了兼顾两者，我们引入了MLA，这是一种配备**低秩键值联合压缩（low-rank key-value joint compression）**的注意力机制。实验表明，MLA在性能上优于MHA，同时显著减少了推理过程中的KV缓存，从而提升了推理效率。

2. **在前馈网络（FFNs）方面**，我们采用了DeepSeekMoE架构（Dai et al., 2024），该架构通过细粒度的专家分割和共享专家隔离，实现了更高的专家专业化潜力。与传统的MoE架构（如GShard（Lepikhin et al., 2021））相比，DeepSeekMoE展现出了显著优势，使我们能够以较低的成本训练强大的模型。在训练过程中，我们采用专家并行策略，并设计了补充机制来控制通信开销并确保负载均衡。

通过结合这两种技术，DeepSeek-V2在性能（图1(a)）、训练成本和推理吞吐量（图1(b)）方面均表现出色。**我们构建了一个高质量、多来源的预训练语料库，包含8.1T token**。与DeepSeek 67B（我们之前的版本）（DeepSeek-AI, 2024）使用的语料库相比，该语料库的数据量更大，尤其是中文数据，且数据质量更高。

- 首先我们在完整的预训练语料库上对DeepSeek-V2进行预训练。
- 然后，我们收集了**1.5M个对话会话**，涵盖数学、代码、写作、推理、安全等多个领域，用于对DeepSeek-V2 Chat（SFT）进行监督微调。
- 最后，我们遵循DeepSeekMath（Shao et al., 2024）的方法，采用**组相对策略优化（GRPO）**进一步对齐模型与人类偏好，生成DeepSeek-V2 Chat（RL）。

我们在广泛的中英文基准测试中评估了DeepSeek-V2，并将其与代表性的开源模型进行了比较。评估结果表明，即使仅激活21B参数，DeepSeek-V2仍然在开源模型中表现出顶级性能，成为最强的开源MoE语言模型。

- 图1(a)显示，在MMLU上，DeepSeek-V2仅以少量激活参数就达到了顶级性能。
- 如图1(b)所示，与DeepSeek 67B相比，DeepSeek-V2节省了42.5%的训练成本，减少了93.3%的KV缓存，并将最大生成吞吐量提升至5.76倍。

我们还在开放式基准测试中评估了DeepSeek-V2 Chat（SFT）和DeepSeek-V2 Chat（RL）。值得注意的是，DeepSeek-V2 Chat（RL）在AlpacaEval 2.0（Dubois et al., 2024）上达到了38.9的长度控制胜率，在MT-Bench（Zheng et al., 2023）上获得了8.97的综合评分，在AlignBench（Liu et al., 2023）上获得了7.91的综合评分。英文开放式对话评估表明，DeepSeek-V2 Chat（RL）在开源聊天模型中具有顶级性能。此外，AlignBench的评估表明，在中文方面，DeepSeek-V2 Chat（RL）超越了所有开源模型，甚至击败了大多数闭源模型。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/c03d34cef444d7a19a24d47b6d0570b9d6090d848e2fae434ca836680cc6df78f879859ad5fe5a92649295610441d1e4?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 (a)不同开源模型中MMLU精度与激活参数的对比。(b) DeepSeek-67B（Dense）和DeepSeek-v2的训练成本和推理效率。

为了促进对MLA和DeepSeekMoE的进一步研究和开发，我们还向开源社区发布了**DeepSeek-V2-Lite**，这是一个配备MLA和DeepSeekMoE的小型模型。其总参数量为15.7B，每个token激活2.4B参数。关于DeepSeek-V2-Lite的详细描述见附录B。

在本文的其余部分，我们首先详细描述了DeepSeek-V2的模型架构（第2节）。随后，我们介绍了预训练工作，包括训练数据构建、超参数设置、基础设施、长上下文扩展以及模型性能和效率的评估（第3节）。接着，我们展示了对齐工作（alignment），包括监督微调（SFT）、强化学习（RL）、评估结果及其他讨论（第4节）。最后，我们总结了结论，探讨了DeepSeek-V2的当前局限性，并展望了未来的工作（第5节）。

## 2. 架构

总体而言，DeepSeek-V2仍然基于Transformer架构（Vaswani et al., 2017），其中每个Transformer块由一个注意力模块和一个前馈网络（FFN）组成。然而，对于注意力模块和FFN，我们设计并采用了创新的架构。对于注意力模块，我们设计了**多头隐注意力（MLA）**，利用低秩键值联合压缩来消除推理时键值（KV）缓存的瓶颈，从而支持高效推理。对于FFN，我们采用了**DeepSeekMoE架构**（Dai et al., 2024），这是一种高性能的MoE架构，能够以较低的成本训练强大的模型。图2展示了DeepSeek-V2的架构示意图，本节将详细介绍MLA和DeepSeekMoE的细节。对于其他微小细节（例如层归一化和FFN中的激活函数），除非特别说明，DeepSeek-V2遵循DeepSeek 67B（DeepSeek-AI, 2024）的设置。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/20621c28bc61d4dff59b1ce6eccf40ad260eda493f06d6d6326ad4269ee4f269cc19d8bead0dc52849da02579356f108?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2 DeepSeek-V2架构示意图。MLA通过显著减少生成的KV缓存来确保高效的推理，DeepSeekMoE通过稀疏架构以经济的成本训练强模型

### 2.1 多头隐注意力：提升推理效率

传统的Transformer模型通常采用多头注意力（MHA）（Vaswani et al., 2017），但**在生成过程中，其庞大的键值（KV）缓存会成为限制推理效率的瓶颈**。为了减少KV缓存，研究者提出了**多查询注意力（MQA）**（Shazeer, 2019）和**分组查询注意力（GQA）**（Ainslie et al., 2023）。这些方法需要更少的KV缓存，但其性能无法与MHA媲美（我们在附录D.1中提供了MHA、GQA和MQA的消融实验）。

对于DeepSeek-V2，我们设计了一种创新的注意力机制，称为**多头隐注意力（MLA）**。MLA配备了低秩键值联合压缩，不仅性能优于MHA，而且所需的KV缓存显著减少。以下我们将介绍其架构，并在附录D.2中提供MLA与MHA的对比。

#### 2.1.1 预备知识：标准多头注意力

我们首先介绍标准MHA机制作为背景。设：

- $d$为嵌入维度
- $n_h$为注意力头（attention heads）的数量
- $d_h$为每个头的维度
- $h_t \in \mathbb{R}^d$为第$t$个token在注意力层的输入

标准MHA首先通过三个矩阵$W_Q$、$W_K$、$W_V \in \mathbb{R}^{d_h n_h \times d}$分别生成$q_t$、$k_t$、$v_t \in \mathbb{R}^{d_h n_h}$：

$$
q_t = W^Q h_t, \quad (1) \\
k_t = W^K h_t, \quad (2) \\
v_t = W^V h_t. \quad (3)
$$


然后，$q_t$、$k_t$、$v_t$ 将被切分为 $n_h$ 个头以进行多头注意力计算：

$$
[q_{t,1}; q_{t,2}; \dots; q_{t,n_h}] = q_t, \quad (4) \\
[k_{t,1}; k_{t,2}; \dots; k_{t,n_h}] = k_t, \quad (5) \\
[v_{t,1}; v_{t,2}; \dots; v_{t,n_h}] = v_t, \quad (6) \\
o_{t,i} = \sum_{j=1}^t \text{Softmax}\ _j \left( \frac{q_{t,i}^T k_{j,i}}{\sqrt{d_h}} \right) v_{j,i}, \quad (7) \\
u_t = W_O [o_{t,1}; o_{t,2}; \dots; o_{t,n_h}], \quad (8)
$$

其中:

- $q_{t,i}$、$k_{t,i}$、$v_{t,i} \in \mathbb{R}^{d_h}$ 分别表示第 $i$ 个注意力头的查询、键和值；
- $W_O \in \mathbb{R}^{d \times d_h n_h}$ 表示输出投影矩阵。

**在推理过程中，所有键和值都需要被缓存以加速推理**，因此MHA需要为每个token缓存 $2 n_h d_h l$ 个元素（$l$ 为层数）。在模型部署中，这种庞大的KV缓存是一个巨大的瓶颈，限制了最大batch-size和序列长度。

### 2.1.2 低秩键值联合压缩

MLA的核心是：通过低秩联合压缩键和值来**减少KV缓存**：

$$
c^{KV}_t = W^{DKV} h_t, \quad (9) \\
k^C_t = W^{UK} c^{KV}_t, \quad (10) \\
v^C_t = W^{UV} c^{KV}_t, \quad (11)
$$

其中:

- $c^{KV}_t \in \mathbb{R}^{d_c}$ 是键和值的压缩隐向量；
- $d_c (\ll d_h n_h)$ 表示**KV压缩维度**；
- $W^{DKV} \in \mathbb{R}^{d_c \times d}$ 是**下投影矩阵(down-projection matrix)**；
- $W^{UK}$ 和 $W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$ 分别是键和值的**上投影矩阵(up-projection matrices)**。

在推理过程中，MLA只需缓存 $c_t^{KV}$，因此其KV缓存仅为 $d_c l$ 个元素。此外，在推理过程中，由于 $W^{UK}$ 可以被吸收到 $W^Q$ 中，$W^{UV}$ 可以被吸收到 $W_O$中，我们甚至**不需要显式计算键和值来进行注意力计算**。图3直观地展示了MLA中的KV联合压缩如何减少KV缓存。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/76af75a1498979d22d9a3f26179570dc77f16ed2b43156b6ffa8ab4249b85fb36c81869e60fb29b9e72a2864801a1bb7?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3 多头注意（MHA）、分组查询注意（GQA）、多查询注意（MQA）和多头潜在注意（MLA）的简化说明。通过将键和值联合压缩成一个隐向量，MLA显著降低了推理过程中的KV缓存

此外，为了减少训练期间的激活内存，我们还**对Query进行了低秩压缩**，尽管这并不能减少KV缓存：

$$
c^Q_t = W^{DQ} h_t, \quad (12) \\
q^C_t = W^{UQ} c^Q_t, \quad (13)
$$

其中:

- $c^Q_t \in \mathbb{R}^{d'_c}$ 是查询的压缩隐向量；
- $d'_c (\ll d_h n_h)$ 表示查询压缩维度；
- $W^{DQ} \in R^{d'_c \times d}$ 和 $W_{UQ} \in R^{d_h n_h \times d'_c}$ 分别是查询的下投影和上投影矩阵。

### 2.1.3 解耦的旋转位置嵌入

我们计划为DeepSeek-V2使用旋转位置嵌入（RoPE）（Su et al., 2024），这与DeepSeek 67B（DeepSeek-AI, 2024）一致。然而，RoPE与低秩KV压缩不兼容。具体来说，RoPE对键和查询都是位置敏感的。如果我们对键 $k^C_t$ 应用RoPE，公式10中的 $W_{UK}$ 将与一个位置敏感的RoPE矩阵耦合。这样，$W_{UK}$ 在推理过程中无法再被吸收到 $W_Q$ 中，因为与当前生成token相关的RoPE矩阵会位于 $W_Q$ 和 $W_{UK}$ 之间，而矩阵乘法不满足交换律。因此，我们必须在推理过程中重新计算所有前缀token的键，这将显著降低推理效率。

作为解决方案，我们提出了**解耦RoPE策略**，该策略使用额外的多头查询 $q^R_{t,i} \in \mathbb{R}^{d^R_h}$ 和一个共享键 $k^R_t \in \mathbb{R}^{d^R_h}$ 来承载RoPE，其中 $d^R_h$ 表示解耦查询和键的每头维度。配备解耦RoPE策略后，MLA执行以下计算：

$$
[q^R_{t,1}; q^R_{t,2}; \dots; q^R_{t,n_h}] = q^R_t = \text{RoPE}(W^{QR} c^Q_t), \quad (14) \\
k^R_t = \text{RoPE}(W^{KR} h_t), \quad (15) \\
q_{t,i} = [q^C_{t,i}; q^R_{t,i}], \quad (16) \\
k_{t,i} = [k^C_{t,i}; k^R_t], \quad (17) \\
o_{t,i} = \sum_{j=1}^t \text{Softmax}_j \left( \frac{q_{t,i}^T k_{j,i}}{\sqrt{d_h + d^R_h}} \right) v^C_{j,i}, \quad (18) \\
u_t = W^O [o_{t,1}; o_{t,2}; \dots; o_{t,n_h}], \quad (19)
$$

其中：

- $W^{QR} \in \mathbb{R}^{d^R_h n_h \times d'_c}$ 和 $W^{KR} \in \mathbb{R}^{d^R_h \times d}$ 是生成解耦查询和键的矩阵；
- $\text{RoPE}(\cdot)$ 表示应用RoPE矩阵的操作；$[\cdot; \cdot]$ 表示拼接操作。

在推理过程中，解耦键也需要被缓存。因此，DeepSeek-V2需要的总KV缓存为 $(d_c + d^R_h) l$ 个元素。

为了展示MLA的完整计算过程，我们在附录C中整理并提供了其完整公式。

### 2.1.4 KV缓存对比

我们在表1中展示了不同注意力机制下每个token的KV缓存对比。MLA仅需要少量的KV缓存，相当于仅2.25组的GQA，但其性能优于MHA。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/9b829fe40840e3fe1639d19537c8a84f82d135f484bd36278c15f34e88342d1c61a4f8a6cc4234cb3aba12b85de1e176?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t1.jpg&amp;size=750">

表1
---

### 2.2 DeepSeekMoE：以经济成本训练强大模型

#### 2.2.1 基本架构

对于FFN，我们采用了DeepSeekMoE架构（Dai et al., 2024）。DeepSeekMoE有两个关键思想：将专家分割为更细粒度以实现更高的专家专业化和更准确的知识获取，以及隔离一些共享专家以减少路由专家之间的知识冗余。在激活专家参数和总专家参数数量相同的情况下，DeepSeekMoE能够大幅超越传统MoE架构（如GShard（Lepikhin et al., 2021））。

设 $u_t$ 为第 $t$ 个token的FFN输入，我们计算FFN输出 $h'_t$ 如下：

$$
h'_t = u_t + \sum_{i=1}^{N_s} \text{FFN}^{(s)}_i (u_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}^{(r)}_i (u_t), \quad (20) \\
g_{i,t} = \begin{cases} 
s_{i,t}, & s_{i,t} \in \text{Topk}(\{s_{j,t} | 1 \leq j \leq N_r\}, K_r), \\
0, & \text{否则},
\end{cases} \quad (21) \\
s_{i,t} = \text{Softmax}_i (u_t^T e_i), \quad (22)
$$

其中：

- $N_s$ 和 $N_r$ 分别表示共享专家和路由专家的数量；
- $\text{FFN}^{(s)}_i (\cdot)$ 和 $\text{FFN}^{(r)}_i (\cdot)$ 分别表示第 $i$ 个共享专家和第 $i$ 个路由专家；
- $K_r$ 表示激活的路由专家数量；
- $g_{i,t}$ 是第 $i$ 个专家的门控值；
- $s_{i,t}$ 是token与专家的亲和度；
- $e_i$ 是第 $i$ 个路由专家在该层的中心点；
- $\text{Topk}(\cdot, K)$ 表示从第 $t$ 个token与所有路由专家的亲和度分数中选取最高的 $K$ 个分数。

#### 2.2.2 设备限制路由

我们设计了一种**设备限制路由机制**，以限制MoE相关的通信成本。当采用专家并行时，路由专家将分布在多个设备上。对于每个token，其MoE相关的通信频率与其目标专家覆盖的设备数量成正比。由于DeepSeekMoE中的细粒度专家分割，激活的专家数量可能较大，因此如果采用专家并行，MoE相关的通信成本会更高。

对于DeepSeek-V2，除了简单的top-K选择路由专家外，我们还确保每个token的目标专家最多分布在 $M$ 个设备上。具体来说，对于每个token，我们首先选择 $M$ 个设备，这些设备中的专家具有最高的亲和度分数。然后，我们在这些设备上的专家中进行top-K选择。在实践中，我们发现当 $M \geq 3$ 时，设备限制路由能够实现与无限制top-K路由大致相当的良好性能。

### 2.2.3 负载均衡的辅助损失

在自动学习的路由策略中，我们考虑了负载均衡问题。首先，负载不均衡会增加路由崩溃的风险（Shazeer et al., 2017），导致某些专家无法被充分训练和利用。其次，当采用专家并行时，负载不均衡会降低计算效率。在DeepSeek-V2的训练过程中，我们设计了三种辅助损失，分别用于控制专家级负载均衡（$L_{\text{ExpBal}}$）、设备级负载均衡（$L_{\text{DevBal}}$）和通信均衡（$L_{\text{CommBal}}$）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/67e90c989f026352287945936ded88c1f21359ffd5cdaf07aa77e97f6064349a69df132d3182ba62744bc6d20e7d75c1?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t2.jpg&amp;size=750">

表2

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b12e7352413202916f84660bd22e3ee2a658a37f363fb8bb272f0f1f4cd454f272de622400784b17d4bdaab1d69727d8?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t3.jpg&amp;size=750">

表3

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/db05798415f1aead541d45d299a794ae79099fd062661c823e690b1db066018b1b3a8c0f5201f7fa409749a1faf5bb66?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t4.jpg&amp;size=750">

表4

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/3cf4bd0d5dad06ffe5452d09d16131755845a5bdd83c456d690c8f4715d16c613170a188fec740b96cf6e1950eee58d3?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t5.jpg&amp;size=750">

表5

#### 专家级均衡损失

我们使用专家级均衡损失（Fedus et al., 2021; Lepikhin et al., 2021）来减轻路由崩溃的风险：

$$
L_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N_r} f_i P_i, \quad (23) \\
f_i = \frac{N_r}{K_r T} \sum_{t=1}^T \mathbb{1}(\text{Token } t \text{ 选择专家 } i), \quad (24) \\
P_i = \frac{1}{T} \sum_{t=1}^T s_{i,t}, \quad (25)
$$

其中，$\alpha_1$ 是一个超参数，称为专家级均衡因子；$\mathbb{1}(\cdot)$ 是指示函数；$T$ 表示序列中的token数量。

#### 设备级均衡损失

除了专家级均衡损失外，我们还设计了设备级均衡损失，以确保不同设备之间的计算负载均衡。在DeepSeek-V2的训练过程中，我们将所有路由专家划分为 $D$ 个组 $\{E_1, E_2, \dots, E_D\}$，并将每个组部署在单个设备上。设备级均衡损失计算如下：

$$
L_{\text{DevBal}} = \alpha_2 \sum_{i=1}^D f'_i P'_i, \quad (26) \\
f'_i = \frac{1}{|E_i|} \sum_{j \in E_i} f_j, \quad (27) \\
P'_i = \sum_{j \in E_i} P_j, \quad (28)
$$

其中：

- $\alpha_2$ 是一个超参数，称为设备级均衡因子。

#### 通信均衡损失

最后，我们引入了通信均衡损失，以确保每个设备的通信负载均衡。尽管设备限制路由机制保证了每个设备的发送通信是有界的，但如果某个设备接收的token比其他设备多，实际通信效率也会受到影响。为了缓解这一问题，我们设计了通信均衡损失如下：

$$
L_{\text{CommBal}} = \alpha_3 \sum_{i=1}^D f''_i P''_i, \quad (29) \\
f''_i = \frac{D}{M T} \sum_{t=1}^T \mathbb{1}(\text{Token } t \text{ 被发送到设备 } i), \quad (30) \\
P''_i = \sum_{j \in E_i} P_j, \quad (31)
$$

其中：

- $\alpha_3$ 是一个超参数，称为通信均衡因子。

设备限制路由机制的原则是确保每个设备最多向其他设备传输 $M T$ 个隐藏状态。同时，通信均衡损失用于鼓励每个设备从其他设备接收大约 $M T$ 个隐藏状态。通信均衡损失保证了设备之间的信息交换均衡，从而提高了通信效率。

### 2.2.4 Token丢弃策略

尽管均衡损失旨在鼓励负载均衡，但必须承认它们无法保证严格的负载均衡。为了进一步减轻因负载不均衡导致的计算浪费，我们在训练期间引入了设备级的token丢弃策略。该方法首先计算每个设备的平均计算预算，这意味着每个设备的容量因子为1.0。然后，受Riquelme et al. (2021)启发，我们在每个设备上丢弃具有最低亲和度分数的token，直到达到计算预算。此外，我们确保属于大约10%训练序列的token永远不会被丢弃。通过这种方式，我们可以根据效率需求灵活决定在推理期间是否丢弃token，并始终确保训练和推理之间的一致性。

## 3. 预训练

### 3.1 实验设置

#### 3.1.1 数据构建

在保持与DeepSeek 67B（DeepSeek-AI, 2024）相同的数据处理阶段的基础上，我们扩展了数据量并提升了数据质量。为了扩大预训练语料库，我们探索了互联网数据的潜力并优化了清理流程，从而恢复了大量被错误删除的数据。此外，我们加入了更多的中文数据，旨在更好地利用中文互联网上的语料库。除了数据量，我们还关注数据质量。我们从各种来源丰富了预训练语料库的高质量数据，同时改进了基于质量的过滤算法。改进后的算法确保大量无益数据被移除，而有价值的数据则大部分被保留。此外，我们从预训练语料库中过滤掉了争议性内容，以减轻特定区域文化引入的数据偏差。关于该过滤策略影响的详细讨论见附录E。

我们采用了与DeepSeek 67B相同的分词器，该分词器基于字节级字节对编码（BBPE）算法构建，词汇量为100K。我们的分词预训练语料库包含8.1T token，其中中文token比英文多约12%。

#### 3.1.2 超参数

**模型超参数**：我们将Transformer层数设置为60，隐藏维度设置为5120。所有可学习参数均以标准差0.006随机初始化。在MLA中，我们将注意力头数$n_h$设置为128，每头维度$d_h$设置为128。KV压缩维度$d_c$设置为512，查询压缩维度$d'_c$设置为1536。对于解耦查询和键，我们将每头维度$d^R_h$设置为64。根据Dai et al. (2024)，我们将除第一层外的所有FFN替换为MoE层。每个MoE层包含2个共享专家和160个路由专家，其中每个专家的中间隐藏维度为1536。在路由专家中，每个token激活6个专家。此外，低秩压缩和细粒度专家分割会影响层的输出规模。因此，在实践中，我们在压缩隐向量后使用额外的RMS Norm层，并在宽度瓶颈（即压缩隐向量和路由专家的中间隐藏状态）处乘以额外的缩放因子以确保训练稳定。在此配置下，DeepSeek-V2总参数量为236B，每个token激活21B参数。

**训练超参数**：我们使用AdamW优化器（Loshchilov and Hutter, 2017），超参数设置为$\beta_1 = 0.9$、$\beta_2 = 0.95$、$\text{weight\_decay} = 0.1$。学习率采用预热和阶梯衰减策略（DeepSeek-AI, 2024）。初始时，学习率在前2K步从0线性增加到最大值。随后，在训练约60%的token后，学习率乘以0.316，在训练约90%的token后再次乘以0.316。最大学习率设置为$2.4 \times 10^{-4}$，梯度裁剪范数设置为1.0。我们还使用了批量大小调度策略，在前225B token的训练中，批量大小从2304逐步增加到9216，之后保持9216。我们将最大序列长度设置为4K，并在8.1T token上训练DeepSeek-V2。我们利用流水线并行将模型的不同层部署在不同设备上，每层的路由专家均匀分布在8个设备上（$D = 8$）。对于设备限制路由，每个token最多发送到3个设备（$M = 3$）。对于均衡损失，我们设置$\alpha_1 = 0.003$、$\alpha_2 = 0.05$、$\alpha_3 = 0.02$。我们在训练期间使用token丢弃策略以加速训练，但在评估时不丢弃任何token。

#### 3.1.3 基础设施

DeepSeek-V2基于HAI-LLM框架（High-flyer, 2023）进行训练，这是我们工程师开发的高效轻量级训练框架。它采用了16路零气泡流水线并行（Qi et al., 2023）、8路专家并行（Lepikhin et al., 2021）和ZeRO-1数据并行（Rajbhandari et al., 2020）。由于DeepSeek-V2激活参数相对较少，并且部分算子被重新计算以节省激活内存，因此可以在不需要张量并行的情况下进行训练，从而减少通信开销。此外，为了进一步提高训练效率，我们将共享专家的计算与专家并行的all-to-all通信重叠。我们还为通信、路由算法和跨专家的融合线性计算定制了更快的CUDA内核。此外，MLA还基于改进版的FlashAttention-2（Dao, 2023）进行了优化。

我们在配备NVIDIA H800 GPU的集群上进行所有实验。H800集群中的每个节点包含8个GPU，节点内通过NVLink和NVSwitch连接。节点间使用InfiniBand互连以促进通信。

#### 3.1.4 长上下文扩展

在DeepSeek-V2的初始预训练后，我们使用YaRN（Peng et al., 2023）将默认上下文窗口长度从4K扩展到128K。YaRN特别应用于解耦共享键$k^R_t$，因为它负责承载RoPE（Su et al., 2024）。对于YaRN，我们将缩放因子$s$设置为40，$\alpha$设置为1，$\beta$设置为32，目标最大上下文长度设置为160K。在这些设置下，我们可以预期模型在128K的上下文长度下表现良好。与原始YaRN略有不同，由于我们独特的注意力机制，我们调整了长度缩放因子以调节注意力熵。因子$\sqrt{t}$计算为$\sqrt{t} = 0.0707 \ln s + 1$，旨在最小化困惑度。

我们额外训练了1000步，序列长度为32K，批量大小为576个序列。尽管训练仅在32K的序列长度下进行，但模型在128K的上下文长度下仍表现出色。如图4所示，在“Needle In A Haystack”（NIAH）测试中，DeepSeek-V2在所有上下文窗口长度（最高128K）下均表现良好。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/7b9f89f053fe1da8e72a44d74eab50f50db47bafebdc6866202e9f7a89ed7588c4b13357754c7beb57242ddb1c120719?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=4.jpg&amp;size=750">

图4

### 3.2. 评估

#### 3.2.1. 评估基准

DeepSeek-V2 是在双语语料库上进行预训练的，因此我们在英语和中文的一系列基准上对其进行了评估。我们的评估基于集成在 HAI-LLM 框架中的内部评估框架。包含的基准分类如下，其中带下划线的基准为中文：

- **多学科选择题数据集** 包括 MMLU (Hendrycks et al., 2020)、C-Eval (Huang et al., 2023) 和 CMMLU (Li et al., 2023)。
- **语言理解和推理数据集** 包括 HellaSwag (Zellers et al., 2019)、PIQA (Bisk et al., 2020)、ARC (Clark et al., 2018) 和 BigBench Hard (BBH) (Suzgun et al., 2022)。
- **闭卷问答数据集** 包括 TriviaQA (Joshi et al., 2017) 和 NaturalQuestions (Kwiatkowski et al., 2019)。
- **阅读理解数据集** 包括 RACE (Lai et al., 2017)、DROP (Dua et al., 2019)、C3 (Sun et al., 2019) 和 CMRC (Cui et al., 2019)。
- **指代消解数据集** 包括 WinoGrande (Sakaguchi et al., 2019) 和 CLUEWSC (Xu et al., 2020)。
- **语言建模数据集** 包括 Pile (Gao et al., 2020)。
- **中文理解与文化数据集** 包括 CHID (Zheng et al., 2019) 和 CCPM (Li et al., 2021)。
- **数学数据集** 包括 GSM8K (Cobbe et al., 2021)、MATH (Hendrycks et al., 2021) 和 CMath (Wei et al., 2023)。
- **代码数据集** 包括 HumanEval (Chen et al., 2021)、MBPP (Austin et al., 2021) 和 CRUXEval (Gu et al., 2024)。
- **标准化考试** 包括 AGIEval (Zhong et al., 2023)。注意，AGIEval 包含英语和中文子集。

根据我们之前的工作 (DeepSeek-AI, 2024)，我们对以下数据集采用基于困惑度（perplexity）的评估：HellaSwag、PIQA、WinoGrande、RACE-Middle、RACE-High、MMLU、ARC-Easy、ARC-Challenge、CHID、C-Eval、CMMLU、C3 和 CCPM；对以下数据集采用基于生成的评估：TriviaQA、NaturalQuestions、DROP、MATH、GSM8K、HumanEval、MBPP、CRUXEval、BBH、AGIEval、CLUEWSC、CMRC 和 CMath。此外，我们对 Pile-test 进行基于语言建模的评估，并使用 Bits-Per-Byte (BPB) 作为指标，以确保使用不同分词器的模型之间的公平比较。

为了直观地了解这些基准，我们在附录 G 中提供了每个基准的评估格式。

#### 3.2.2. 评估结果

在表 2 中，我们将 DeepSeek-V2 与几个代表性的开源模型进行了比较，包括 DeepSeek 67B (DeepSeek-AI, 2024)（我们之前的版本）、Qwen1.5 72B (Bai et al., 2023)、LLaMA3 70B (AI@Meta, 2024) 和 Mixtral 8x22B (Mistral, 2024)。我们使用内部评估框架评估了所有这些模型，并确保它们共享相同的评估设置。总体而言，DeepSeek-V2 仅激活了 21B 参数，但在几乎所有基准上都显著优于 DeepSeek 67B，并在开源模型中达到了顶级性能。

进一步，我们详细比较了 DeepSeek-V2 与其他开源模型的表现：

1. **与 Qwen1.5 72B 的比较**：Qwen1.5 72B 是另一个支持中文和英文的模型。DeepSeek-V2 在大多数英语、代码和数学基准上表现出压倒性优势。在中文基准上，Qwen1.5 72B 在多学科选择题任务上表现更好，而 DeepSeek-V2 在其他任务上表现相当或更好。需要注意的是，对于 CHID 基准，Qwen1.5 72B 的分词器在我们的评估框架中会遇到错误，因此我们未记录 Qwen1.5 72B 的 CHID 分数。

2. **与 Mixtral 8x22B 的比较**：DeepSeek-V2 在英语基准上表现相当或更好，除了与英语常识知识密切相关的 TriviaQA、NaturalQuestions 和 HellaSwag。值得注意的是，DeepSeek-V2 在 MMLU 上优于 Mixtral 8x22B。在代码和数学基准上，DeepSeek-V2 与 Mixtral 8x22B 表现相当。由于 Mixtral 8x22B 并未专门针对中文数据进行训练，其中文能力远不及 DeepSeek-V2。

3. **与 LLaMA3 70B 的比较**：DeepSeek-V2 的训练数据量不到 LLaMA3 70B 的四分之一。因此，我们承认 DeepSeek-V2 在基础英语能力上仍与 LLaMA3 70B 存在轻微差距。然而，即使训练数据和激活参数少得多，DeepSeek-V2 在代码和数学能力上仍与 LLaMA3 70B 相当。此外，作为双语模型，DeepSeek-V2 在中文基准上显著优于 LLaMA3 70B。

最后，值得一提的是，某些先前的研究 (Hu et al., 2024) 在预训练阶段引入了 SFT 数据，而 DeepSeek-V2 在预训练期间从未接触过 SFT 数据。

#### 3.2.3. 训练和推理效率

**训练成本**：由于 DeepSeek-V2 为每个 token 激活的参数较少，且所需的 FLOPs 少于 DeepSeek 67B，理论上训练 DeepSeek-V2 比训练 DeepSeek 67B 更经济。尽管训练 MoE 模型会引入额外的通信开销，但通过我们的操作符和通信优化，DeepSeek-V2 的训练可以达到相对较高的模型 FLOPs 利用率（MFU）。在实际的 H800 集群训练中，每训练一万亿 token，DeepSeek 67B 需要 300.6K GPU 小时，而 DeepSeek-V2 仅需 172.8K GPU 小时，即稀疏的 DeepSeek-V2 比密集的 DeepSeek 67B 节省了 42.5% 的训练成本。

**推理效率**：为了高效部署 DeepSeek-V2 提供服务，我们首先将其参数转换为 FP8 精度。此外，我们还对 DeepSeek-V2 进行了 KV 缓存量化 (Hooper et al., 2024; Zhao et al., 2023)，进一步将其 KV 缓存中的每个元素平均压缩到 6 位。得益于 MLA 和这些优化，实际部署的 DeepSeek-V2 所需的 KV 缓存显著少于 DeepSeek 67B，因此可以支持更大的批量大小。我们基于实际部署的 DeepSeek 67B 服务的提示和生成长度分布，评估了 DeepSeek-V2 的生成吞吐量。在单个 8 H800 GPU 节点上，DeepSeek-V2 的生成吞吐量超过每秒 50K token，是 DeepSeek 67B 最大生成吞吐量的 5.76 倍。此外，DeepSeek-V2 的提示输入吞吐量超过每秒 100K token。

### 4. 对齐

#### 4.1. 监督微调（SFT）

基于我们之前的研究（DeepSeek-AI, 2024），我们构建了包含 150 万条实例的指令微调数据集，其中 120 万条用于提升模型的有用性，30 万条用于提升安全性。与初始版本相比，我们提高了数据质量，以减少幻觉响应并增强写作能力。我们对 DeepSeek-V2 进行了 2 轮微调，学习率设置为 $5 \times 10^{-6}$。

对于 DeepSeek-V2 Chat（SFT）的评估，我们主要采用基于生成的基准测试，除了几个具有代表性的选择题任务（如 MMLU 和 ARC）。我们还对 DeepSeek-V2 Chat（SFT）进行了指令跟随评估（IFEval）（Zhou et al., 2023），使用提示级别的宽松准确率作为指标。此外，我们使用 2023 年 9 月 1 日至 2024 年 4 月 1 日的 LiveCodeBench（Jain et al., 2024）问题来评估聊天模型。除了标准基准测试外，我们还在开放式对话基准测试上进一步评估了模型，包括 MT-Bench（Zheng et al., 2023）、AlpacaEval 2.0（Dubois et al., 2024）和 AlignBench（Liu et al., 2023）。为了进行比较，我们还在我们的评估框架和设置中评估了 Qwen1.5 72B Chat、LLaMA-3-70B Instruct 和 Mistral-8x22B Instruct。对于 DeepSeek 67B Chat，我们直接参考了之前发布的评估结果。

#### 4.2. 强化学习（RL）

为了进一步释放 DeepSeek-V2 的潜力并使其与人类偏好对齐，我们进行了强化学习（RL）以调整其偏好。

**强化学习算法**：为了节省 RL 的训练成本，我们采用了组相对策略优化（GRPO）（Shao et al., 2024），该方法省略了通常与策略模型大小相同的评论模型，而是从组分数中估计基线。具体来说，对于每个问题 $q$，GRPO 从旧策略 $\pi_{\theta_{\text{old}}}$ 中采样一组输出 $\{o_1, o_2, \dots, o_G\}$，然后通过最大化以下目标来优化策略模型 $\pi_{\theta}$：

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}\left[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)\right] \frac{1}{G} \sum_{i=1}^G \min\left(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i\right) - \beta D_{\text{KL}}(\pi_{\theta} || \pi_{\text{ref}}),
$$

其中：

- $\epsilon$ 和 $\beta$ 是超参数，
- $A_i$ 是优势值，通过每组输出对应的奖励 $\{r_1, r_2, \dots, r_G\}$ 计算：

$$
A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})}.
$$

**训练策略**：在初步实验中，我们发现针对推理数据（如代码和数学提示）的 RL 训练表现出与通用数据训练不同的独特特性。例如，模型的数学和编码能力可以在更长的训练步骤中持续提升。因此，我们采用了两阶段 RL 训练策略：首先进行推理对齐，然后进行人类偏好对齐。在第一阶段，我们训练了一个用于代码和数学推理任务的奖励模型 $RM_{\text{reasoning}}$，并使用其反馈优化策略模型：

$$
r_i = RM_{\text{reasoning}}(o_i).
$$

在第二阶段，我们采用多奖励框架，从有用性奖励模型 $RM_{\text{helpful}}$、安全性奖励模型 $RM_{\text{safety}}$ 和基于规则的奖励模型 $RM_{\text{rule}}$ 中获取奖励。最终响应 $o_i$ 的奖励为：

$$
r_i = c_1 \cdot RM_{\text{helpful}}(o_i) + c_2 \cdot RM_{\text{safety}}(o_i) + c_3 \cdot RM_{\text{rule}}(o_i),
$$

其中：

- $c_1, c_2, c_3$ 是对应的系数。

为了获得可靠的奖励模型，我们精心收集了偏好数据，并进行了严格的质量过滤和比例调整。我们基于编译器反馈获取代码偏好数据，基于真实标签获取数学偏好数据。对于奖励模型训练，我们使用 DeepSeek-V2 Chat（SFT）初始化奖励模型，并使用点对或对对的损失进行训练。在实验中，我们观察到 RL 训练能够充分挖掘和激活模型的潜力，使其能够从可能的响应中选择正确且令人满意的答案。

**训练效率优化**：在极大模型上进行 RL 训练对训练框架提出了高要求。我们实施了以下工程优化：
- （1）采用混合引擎，分别针对训练和推理采用不同的并行策略以提高 GPU 利用率；
- （2）利用 vLLM（Kwon et al., 2023）作为推理后端，加速推理速度；
- （3）精心设计模型卸载到 CPU 和加载回 GPU 的调度策略，以实现训练速度和内存消耗的近最优平衡。

#### 4.3. 评估结果

**标准基准测试评估**：我们首先在标准基准测试上评估了 DeepSeek-V2 Chat（SFT）和 DeepSeek-V2 Chat（RL）。值得注意的是，DeepSeek-V2 Chat（SFT）在 GSM8K、MATH 和 HumanEval 评估中相比其基础版本有显著提升，这归因于我们的 SFT 数据中包含大量数学和代码相关内容。此外，DeepSeek-V2 Chat（RL）进一步提升了数学和代码基准测试的表现。

与其他模型的比较中，DeepSeek-V2 Chat（SFT）在几乎所有英语、数学和代码基准测试上均优于 Qwen1.5 72B Chat。在中文基准测试上，DeepSeek-V2 Chat（SFT）在多学科选择题任务上略低于 Qwen1.5 72B Chat，与其基础版本的表现一致。与最先进的开源 MoE 模型 Mixtral 8x22B Instruct 相比，DeepSeek-V2 Chat（SFT）在大多数基准测试上表现更好，除了 NaturalQuestions 和 IFEval。与最先进的开源模型 LLaMA3 70B Chat 相比，DeepSeek-V2 Chat（SFT）在代码和数学相关基准测试上表现相似，LLaMA3 70B Chat 在 MMLU 和 IFEval 上表现更好，而 DeepSeek-V2 Chat（SFT）在中文任务上表现更强。最终，DeepSeek-V2 Chat（RL）在数学和编码任务上相比 DeepSeek-V2 Chat（SFT）进一步提升了性能。

**开放式生成评估**：我们在开放式对话基准测试上进一步评估了模型。对于英语开放式对话生成，我们使用 MT-Bench 和 AlpacaEval 2.0 作为基准测试。评估结果显示，DeepSeek-V2 Chat（RL）相比 DeepSeek-V2 Chat（SFT）有显著优势，展示了 RL 训练在实现更好对齐方面的有效性。与其他开源模型相比，DeepSeek-V2 Chat（RL）在 MT-Bench 和 AlpacaEval 2.0 上均优于 Mistral 8x22B Instruct 和 Qwen1.5 72B Chat。与 LLaMA3 70B Instruct 相比，DeepSeek-V2 Chat（RL）在 MT-Bench 上表现相当，在 AlpacaEval 2.0 上显著优于后者。

对于中文开放式生成能力，我们基于 AlignBench 进行了评估。结果显示，DeepSeek-V2 Chat（RL）相比 DeepSeek-V2 Chat（SFT）有轻微优势。值得注意的是，DeepSeek-V2 Chat（SFT）显著优于所有开源中文模型，在中文推理和语言任务上大幅领先第二好的开源模型 Qwen1.5 72B Chat。此外，DeepSeek-V2 Chat（SFT）和 DeepSeek-V2 Chat（RL）均优于 GPT-4-0613 和 ERNIEBot 4.0，巩固了我们的模型在支持中文的顶级 LLM 中的地位。

#### 4.4. 讨论

**SFT 数据量**：关于是否需要大量 SFT 数据的讨论一直存在争议。先前的工作（Young et al., 2024; Zhou et al., 2024）认为少于 10K 条 SFT 数据足以产生令人满意的结果。然而，在我们的实验中，如果使用少于 10K 条数据，我们在 IFEval 基准测试上观察到显著的性能下降。可能的解释是，语言模型需要一定量的数据来发展特定技能。尽管随着模型规模的增加，所需数据量可能会减少，但不能完全消除。我们的观察强调了为 LLM 提供足够数据以具备所需能力的关键性。此外，SFT 数据的质量也至关重要，尤其是在涉及写作或开放式问题的任务中。

**强化学习的对齐税**：在人类偏好对齐过程中，我们观察到开放式生成基准测试的显著性能提升，无论是 AI 还是人类评估者的评分。然而，我们也注意到“对齐税”现象（Ouyang et al., 2022），即对齐过程可能会对某些标准基准测试（如 BBH）的性能产生负面影响。为了缓解对齐税，我们在 RL 阶段在数据处理和训练策略改进上付出了巨大努力，最终在标准基准测试和开放式基准测试之间实现了可接受的权衡。

**在线强化学习**：在我们的偏好对齐实验中，我们发现在线方法显著优于离线方法。因此，我们投入了大量精力实现了一个在线 RL 框架来对齐 DeepSeek-V2。关于在线或离线偏好对齐的结论可能因不同情境而异，我们将在未来工作中进行更深入的比较和分析。

### 5. 结论、局限性与未来工作

本文介绍了 DeepSeek-V2，一个支持 128K 上下文长度的大型 MoE 语言模型。除了强大的性能外，它还以经济高效的训练和推理为特点，得益于其创新的架构（包括 MLA 和 DeepSeekMoE）。在实际应用中，相比 DeepSeek 67B，DeepSeek-V2 在显著提升性能的同时，节省了 42.5% 的训练成本，减少了 93.3% 的 KV 缓存，并将最大生成吞吐量提升至 5.76 倍。评估结果表明，仅激活 21B 参数的 DeepSeek-V2 在开源模型中达到了顶级性能，成为最强的开源 MoE 模型。

DeepSeek-V2 及其聊天版本具有其他 LLM 常见的局限性，包括预训练后缺乏持续知识更新、可能生成未经核实的信息（如未经验证的建议）以及可能产生幻觉。此外，由于我们的数据主要由中文和英文内容组成，模型在其他语言上的表现可能有限。在中文和英文以外的场景中，应谨慎使用。

DeepSeek 将持续投资于开源大模型，致力于逐步接近通用人工智能的目标。我们正在探索进一步扩展 MoE 模型的方法，同时保持经济高效的训练和推理成本。我们的下一步目标是在即将发布的版本中实现与 GPT-4 相当的性能。我们的对齐团队不断努力提升模型，旨在开发一个不仅有用而且诚实、安全的模型，最终目标是使模型的价值与人类价值观对齐，同时尽量减少人类监督的需求。通过优先考虑伦理和负责任的发展，我们致力于为社会创造积极和有益的影响。目前，DeepSeek-V2 仅支持文本模态。在未来计划中，我们打算使模型支持多模态，增强其在不同场景中的多功能性和实用性。


# 

[https://arxiv.org/pdf/2412.19437](https://arxiv.org/pdf/2412.19437)