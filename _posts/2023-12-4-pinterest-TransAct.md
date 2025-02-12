---
layout: post
title: pinterest的TransAct介绍
description: 
modified: 2023-12-4
tags: 
---

pinterest在《TransAct: Transformer-based Realtime User Action Model for
Recommendation at Pinterest》提出了它们的排序模型。

# 摘要

用于预测下一个动作的编码用户活动的序列模型，已成为构建大规模个性化推荐系统的一种流行设计选择。传统的序列推荐方法要么利用实时用户行为的端到端学习，要么以离线批处理生成的方式单独学习用户表示。本文：

- （1）介绍了Pinterest的Homefeed排序架构，这是我们的个性化推荐产品和最大的参与度表面；
- （2）提出了TransAct，一个从用户实时活动中提取**短期偏好的序列模型**；
- （3）描述了我们的混合排序方法，该方法结合了通过TransAct的**端到端序列建模**和**批生成的用户embedding**

混合方法使我们能够结合直接在实时用户活动上学习的反应性优势和批用户表示的成本效益，这些用户表示是在更长时间内学习的。

我们描述了消融研究的结果，在产品化过程中面临的挑战，以及在线A/B实验的结果，这些验证了我们混合排序模型的有效性。我们进一步证明了TransAct在其他表面如上下文推荐和搜索方面的有效性。我们的模型已部署在Pinterest的Homefeed、相关Pin、通知和搜索中。

# 1 引言

近年来在线内容的激增为用户导航创造了压倒性的信息量。为了解决这个问题，推荐系统被用于各种行业，帮助用户从大量选择中找到相关item，包括产品、图片、视频和音乐。通过提供个性化推荐，企业和组织可以更好地服务用户，并保持他们对平台的参与。因此，推荐系统对企业至关重要，因为它们通过提高**参与度（engagement）、销售和收入**来推动增长。

作为最大的内容分享和社交媒体平台之一，Pinterest拥有数十亿个Pin，拥有丰富的上下文和视觉信息，为超过4亿用户提供灵感。

访问Pinterest时，用户会立即看到图1所示的Homefeed页面，这是主要的灵感来源，**占平台总用户参与度的大部分**。Homefeed页面由一个3阶段的推荐系统驱动，该系统根据用户兴趣和活动检索、排序和混合内容。在检索阶段，我们根据用户兴趣、关注的板块等多种因素，从Pinterest上创建的数十亿个Pin中筛选出数千个。然后我们使用pointwise排序模型通过预测它们对用户的个性化相关性来对候选Pin进行排序。最后，使用混合层调整排序结果以满足业务需求。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/1923a061e83d77a63bb9a3d72633a0680f6995910caecdd5fedaa3fb9117f1c90896cfdad7da8f2da46cb4a8e8021e00?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1  Pinterest Homefeed页

实时推荐至关重要，因为它为用户提供快速和最新的推荐，改善了他们的整体体验和满意度。集成实时数据，如最近的用户行为，可以更准确地推荐，并增加用户发现相关item的可能性[4, 21]。更长的用户行为序列可以改善用户表示，从而提高推荐性能。然而，使用长序列进行排序对基础设施提出了挑战，因为它们需要大量的计算资源，并可能导致延迟增加。为了解决这一挑战，

- 一些方法已经**利用哈希和最近邻搜索在长用户序列中[21]**。
- 其他工作将用户过去的行为编码为用户嵌入[20]，以表示长期用户兴趣。用户嵌入特征通常作为batch特征生成（例如，每天生成），在多个应用中以低延迟服务是成本效益的。

现有序列推荐的局限性在于：它们要么只使用实时用户行为，要么只使用从长期用户行为历史中学习的batch用户表示。

我们引入了一种新颖的实时-批量混合（realtime-batch hybrid）的排序方法，结合了实时用户行为信号和batch用户表示。为了捕捉用户的实时行为，我们提出了TransAct：

- 一个基于transformer的新模块，旨在编码**最近的用户行为序列并理解用户的即时偏好**。
- **对于在较长时间内发生用户行为，我们将它们转换为批用户表示[20]**。

通过结合TransAct的表现力和批用户embedding，这种混合排序模型**为用户提供了他们最近行为的实时反馈，同时也考虑了他们的长期兴趣**。实时组件和batch组件互补推荐准确性。这导致Homefeed页面上用户体验的整体改善。

本文的主要贡献总结如下：

- 我们描述了**Pinnability**，Pinterest的Homefeed生产排序系统的架构。Homefeed个性化推荐产品占Pinterest总用户参与度的大部分。
- 我们提出了**TransAct**：一个基于transformer的实时用户行为序列模型，有效地从用户最近的行动中捕获用户的短期兴趣。我们证明了将TransAct与每天生成的用户表示[20]结合到混合模型中，在Pinnability中实现了最佳性能。这种设计选择通过全面的消融研究得到了证明。我们的代码实现是公开可用的1。
- 我们描述了在Pinnability中实施的**服务优化**，使得在引入TransAct到Pinnability模型时，计算复杂度增加了65倍成为可能。具体来说，优化是为了使我们以前的基于CPU的模型能够在GPU上服务。
- 我们描述了使用TransAct在现实世界推荐系统上的**在线A/B实验**。我们展示了在线环境中的一些实际问题，如推荐多样性下降和参与度衰减，并提出了解决这些问题的方案。

本文的其余部分组织如下：第2节回顾相关工作。第3节描述了TransAct的设计及其生产细节。实验结果在第4节报告。我们在第5节讨论了一些超出实验的发现。最后，我们在第6节总结我们的工作。

# 2 相关工作

## 2.1 推荐系统

协同过滤（CF）[12, 18, 24]基于这样的假设进行推荐：用户会偏好其他相似用户喜欢的物品。它使用用户行为历史来计算用户和物品之间的相似性，并基于相似性推荐物品。这种方法受到用户-物品矩阵稀疏性的困扰，并且无法处理从未与任何物品互动过的用户。另一方面，因子分解机[22, 23]能够处理稀疏矩阵。

最近，深度学习（DL）已被用于点击率（CTR）预测任务。例如：

- 谷歌使用Wide & Deep [5]模型进行应用推荐。wide组件通过捕获特征之间的交互来实现记忆，而deep组件通过使用前馈网络学习分类特征的嵌入来帮助泛化。
- DeepFM [7]通过自动学习低阶和高阶特征交互进行了改进。
- DCN [34]及其升级版本DCN v2 [35]都旨在自动建模显式特征交叉。

上述推荐系统在捕捉用户的短期兴趣方面表现不佳，因为只利用了用户的静态特征。这些方法也倾向于忽略用户行为历史中的序列关系，导致用户偏好的表示不足。

## 2.2 序列推荐

为了解决这个问题，序列推荐在学术界和工业界都得到了广泛研究。序列推荐系统使用用户的行为历史作为输入，并应用推荐算法向用户推荐适当的物品。序列推荐模型能够在长时间内捕捉用户的长期偏好，类似于传统推荐方法。此外，它们还有一个额外的好处，**即能够考虑到用户兴趣的演变**，从而实现更高质量的推荐。

序列推荐通常被视为下一个物品预测任务，目标是基于用户过去的行为序列预测用户的下一个行动。我们在编码用户过去的行为到dense表示方面受到了先前序列推荐方法[4]的启发。一些早期的序列推荐系统使用机器学习技术，如马尔可夫链[8]和基于会话的K最近邻（KNN）[11]来模拟用户行为历史中交互的时间依赖性。这些模型因为仅通过组合不同会话的信息而无法完全捕捉用户的长期模式而受到批评。最近，深度学习技术，如循环神经网络（RNN）[25]在自然语言处理中取得了巨大成功，并在序列推荐中变得越来越流行。因此，许多基于DL的序列模型[6, 9, 30, 42]使用RNNs取得了出色的性能。卷积神经网络（CNNs）[40]广泛用于处理时间序列数据和图像数据。在序列推荐的背景下，基于CNN的模型可以有效地学习用户最近交互的一组物品内的依赖性，并相应地进行推荐[31, 32]。

注意力机制起源于神经机器翻译任务，该任务模拟输入句子的不同部分对输出词的重要性[2]。自注意力是一种已知的机制，用于衡量输入序列的不同部分的重要性[33]。已经有更多推荐系统使用注意力[43]和自注意力[4, 13, 16, 27, 39]。

许多先前的工作[13, 16, 27]仅使用公共数据集进行离线评估。然而，在线环境更具挑战性和不可预测性。由于问题表述的差异，我们的方法与这些工作不直接可比。我们的方法类似于点击率（CTR）预测任务。深度兴趣网络（DIN）使用注意力机制在CTR预测任务中模拟用户过去行为的依赖性。阿里巴巴的行为序列transformer（BST）[4]是DIN的改进版本，与我们的工作密切相关。他们提出使用transformer从用户行为中捕捉用户兴趣，强调行为顺序的重要性。然而，我们发现位置信息并没有增加太多价值。我们发现其他设计，如更好的早期融合和行为类型嵌入，在处理序列特征时是有效的。

# 3 方法论

在本节中，我们介绍了TransAct，我们的实时-batch混合排序模型。我们将从Pinterest Homefeed排序模型Pinnability的概述开始，然后描述如何使用TransAct在Pinnability中编码实时用户行为序列特征以进行排序任务。

## 3.1 预备知识：Homefeed排序模型

在Homefeed排序中，我们将推荐任务建模为pointwise多任务预测问题，可以定义如下：**给定用户 $ u $ 和Pin $ p $，我们构建一个函数来预测用户 $ u $ 对候选Pin $ p $ 执行不同动作的概率**。不同动作的集合包含正面和负面动作，例如点击、保存和隐藏。

我们构建了Pinnability，Pinterest的Homefeed排序模型，来解决上述问题。高层架构是Wide and Deep学习（WDL）模型[5]。Pinnability模型利用各种类型的输入信号，如用户信号、Pin信号和上下文信号。这些输入可以以不同的格式出现，包括类别型、数值型和嵌入型特征。

- 我们使用嵌入层（embedding layer）将类别型特征投影到dense特征，并在数值型特征上执行batch归一化
- 然后，我们应用一个全秩DCN V2[35]的特征交叉来显式建模特征交互
- 最后，我们使用具有一组输出动作头 $ H = \lbrace h_1, h_2, ..., h_k \rbrace $ 的全连接层来预测用户对候选Pin $ p $ 的动作。每个头对应一个动作

如图2所示，我们的模型是一个实时-batch混合模型，通过实时（TransAct）和batch（PinnerFormer）方法编码用户行为历史特征，并针对排序任务[37]进行优化。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/4d43f7e4c6a42696407e6f715e0e52a5e62b187b5776fe95fcb15e57010aab015a11bc9d66bf80d24ffa30ebbfb064ae?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2 Pinterest Homefeed ranking model (Pinnability)

每个训练样本是 (x, y)，其中:

- x 表示一组特征，
- $ y \in \lbrace 0, 1 \rbrace^{ \mid H \mid} $，y中的每个条目对应于H中动作头的label。

Pinnability的loss函数是一个加权交叉熵损失，旨在优化多标签分类任务。我们将损失函数公式化为：

$$ 
L = w_u \sum_{h \in H} \left\{ -w_h [y_h \log f(x)_h + (1 - y_h) (1 - \log f(x)_h)] \right\} 
$$

...(1)

其中：

- $ f(x) \in (0, 1)^H $，$ f(x)_h $ 是头 $ h $ 的输出概率。
- $ y_h \in \lbrace 0, 1\rbrace $ 是头 $ h $ 的真实标签。
- 权重 $ w_h $ 应用于每个头的输出 $ f(x)_h $ 的交叉熵。
- $ w_h $ 是根据真实标签 y 和标签权重矩阵 $ M \in \mathbb{R}^{\mid H \mid \times \mid H \mid} $ 计算的：

$$ 
w_h = \sum_{a \in H} M_{h,a} \times y_a 
$$

标签权重矩阵 $ M $ 作为每个动作对每个头损失项贡献的控制因素。注意，如果 M 是对角矩阵，方程（1）简化为标准的多头二元交叉熵损失。但是**选择根据经验确定的标签权重 M 可以显著提高性能**。

此外，每个训练样本都由用户依赖的权重 $ w_u $ 加权，这由用户属性决定，如用户状态、性别和位置。我们通过乘以用户状态权重、用户性别权重和用户位置权重来计算 $ w_u $：

$$
w_u = w_{\text{state}} \times w_{\text{location}} \times w_{\text{gender}}
$$

这些权重根据特定业务需求进行调整。

## 3.2 实时用户行为序列特征

用户过去的行为历史自然是一个可变长度特征——不同用户在平台上的过去行为数量不同。
尽管更长的用户行为序列通常意味着更准确的用户兴趣表示，但**实际上，包含所有用户行为是不可行的**。因为获取用户行为特征和执行排序模型推理所需的时间也可能大幅增长，这反过来又会影响用户体验和系统效率。考虑到基础设施成本和延迟要求，**我们选择包含每个用户最近的100个行为序列。对于少于100个行为的用户，我们用0填充特征到100的长度。用户行为序列特征按时间戳降序排序，即第一个条目是最近的行为**。

用户行为序列中的所有行为都是Pin级别的行为。对于每个行为，我们使用三个主要特征：

- 行为的时间戳
- 行为类型
- Pin的32维PinSage嵌入[38]：**PinSage是一个紧凑的embedding，编码了Pin的内容信息**。

## 3.3 我们的方法：TransAct

与静态特征不同，实时用户行为序列特征 $ S(u) = [a_1, a_2, ..., a_n] $ 是使用一个名为TransAct的专用子模块处理的。TransAct从用户的历史行为中提取序列模式，并预测 $ (u, p) $ 相关性分数。

### 3.3.1 特征编码

用户参与的Pin的相关性，可以通过用户行为历史中对它们采取的行动类型来确定。例如：

- 通常认为用户保存到自己看板的Pin比仅查看的Pin更相关。
- 如果Pin被用户隐藏，相关性应该非常低。

为了纳入这一重要信息，我们使用可训练的嵌入表将行动类型投影到低维向量。用户行为类型序列随后被投影到用户行为嵌入矩阵 $ W_{\text{actions}} \in \mathbb{R}^{\mid S \mid \times d_{\text{action}}} $，其中 $ d_{\text{action}} $ 是行动类型嵌入的维度。

如前所述，用户行为序列中的Pin内容由PinSage嵌入[38]表示。因此，用户行为序列中所有Pin的内容是一个矩阵 $ W_{\text{pins}} \in \mathbb{R}^{\mid S \mid \times d_{\text{PinSage}}} $。最终编码的用户行为序列特征是$CONCAT (W_{actions} \ , W_{pins}) \in \mathbb{R}^{\mid S \mid \times (d_{PinSage} + d_{action})} $。

### 3.3.2 早期融合（early fusion）

直接在排序模型中使用用户行为序列特征的一个独特优势是，我们可以**显式地建模候选Pin和用户参与的Pin之间的交叉**。早期融合（early fusion）在推荐任务中指的是在推荐模型的早期阶段合并用户和物品特征。通过实验，我们发现早期融合是提高排序性能的重要因素。评估了两种早期融合方法：

- append：将候选Pin的PinSage embedding附加到用户行为序列作为序列的最后一项，类似于BST[4]。**使用零向量作为候选Pin的虚拟动作类型**。
- concat：对于用户行为序列中的每个动作，将候选Pin的PinSage embedding与用户行为特征连接起来。

我们根据离线实验结果**选择concat作为我们的早期融合方法**。早期融合的结果序列特征是：一个2维矩阵 $ U \in \mathbb{R}^{\mid S \mid \times d} $，其中 $ d = (d_{\ action} \ + 2d_{\ PinSage}) $。

### 3.3.3 序列聚合模型

准备好用户行为序列特征 $ U $ 后，下一个挑战是：**有效地聚合用户行为序列中的所有信息以表示用户的短期偏好**。工业中用于序列建模的一些流行模型架构包括CNN[40]、RNN[25]和最近的transformer[33]等。我们尝试了不同的序列聚合架构，并选择了基于transformer的架构。我们采用了标准transformer编码器，有2个编码器层和一个头。前馈网络的隐藏维度表示为 $ d_{\text{hidden}} $。这里不使用位置编码，因为我们的离线实验表明位置信息是无效的。

### 3.3.4 随机时间窗口掩码

在用户的所有最近行为上训练可能会导致**兔子洞效应（rabbit hole effect）**，即模型推荐与用户最近参与内容相似的内容。这会损害用户Homefeed的多样性，对长期用户留存有害。为了解决这个问题，我们使用用户行为序列的时间戳构建transformer编码器的时间窗口掩码。该掩码在自注意力机制应用之前过滤掉输入序列中的某些位置。在每次前向传递中，从0到24小时均匀采样一个随机时间窗口 $ T $。**在 $ (t_{\text{request}} - T, t_{\text{request}}) $ 内的所有行为都被掩码，其中 $ t_{\text{request}} $ 代表接收排序请求的时间戳**。重要的是要注意，随机时间窗口掩码仅在训练期间应用，而在推理时不使用掩码。

### 3.3.5 transformer输出压缩

transformer编码器的输出是一个矩阵：$ O = (o_0 : o_{\mid S \mid -1}) \in \mathbb{R}^{\mid S \mid \times d} $。我们只取前K列（$ o_0 : o_{K-1} $），将它们与最大池化向量 $ \text{MAXPOOL}(O) \in \mathbb{R}^d $ 连接起来，然后将其展平为一个向量 $ \mathbf{z} \in \mathbb{R}^{(K+1) \times d} $。前 $ K $ 列输出捕获了用户最近的兴趣，而 $ \text{MAXPOOL}(O) $ 表示用户对 $ S(u) $ 的长期偏好。由于输出足够紧凑，它可以很容易地使用DCN v2[35]特征交叉层集成到Pinnability框架中。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/1142f09da6e3bdb2d8497ad46297b72a030c4de4c8f28bde2f850680483ca0f08393b6ccfbb29f7b69e60d7f7f3725b9?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3 TransAct架构是一种可以插入到任何类似架构中的子模块，比如Pinnability。

## 3.4 模型产品化

### 3.4.1 模型重新训练

对于推荐系统来说，重新训练很重要，因为它允许系统不断适应用户行为和偏好随时间的变化。没有重新训练，推荐系统的性能会随着用户行为和偏好的变化而降低，导致推荐准确性下降[26]。当我们在排序中使用实时特征时，这一点尤其正确。模型对时间更敏感，需要频繁重新训练。否则，模型可能在几天内变得过时，导致预测准确性降低。我们每周从头开始重新训练Pinnability两次。我们发现这种重新训练频率对于确保一致的参与率和保持可管理的训练成本至关重要。我们将在第4.4.3节深入探讨重新训练的重要性。

## 3.4.2 GPU服务
带有TransAct的Pinnability在浮点运算方面比其前身复杂65倍。如果没有模型推理的突破，我们的模型服务成本和延迟将增加相同的规模。GPU模型推理允许我们以中性的延迟和服务成本提供带有TransAct的Pinnability。

在GPU上提供Pinnability的主要挑战是CUDA内核启动开销。在GPU上启动操作的CPU成本非常高，但它通常被延长的GPU计算时间所掩盖。然而，这对于Pinnability GPU模型服务有两个问题。首先，Pinnability和推荐模型通常处理数百个特征，这意味着有大量的CUDA内核。其次，在线服务期间的batch大小很小，因此每个CUDA内核需要的计算量很少。有了大量小CUDA内核，启动开销比实际计算更昂贵。我们通过以下优化解决了技术挑战：

- 合并CUDA内核。一个有效的方法是尽可能合并操作。我们利用标准深度学习编译器，如nvFuser7，但通常发现需要人为干预许多剩余操作。一个例子是我们的嵌入表查找模块，它由两个计算步骤组成：原始ID到表索引查找和表索引到嵌入查找。由于特征数量众多，这个过程需要重复数百次。我们通过利用cuCollections8支持GPU上的原始ID哈希表，并实现自定义的合并嵌入查找模块，将多个特征的查找合并为一次查找，从而将与稀疏特征相关的数百个操作减少为一个。

- 合并内存拷贝。每次推理时，数百个特征被作为单独的张量从CPU复制到GPU内存。调度数百个张量拷贝的开销成为瓶颈。为了减少张量拷贝操作的数量，我们在将它们从CPU传输到GPU之前，将多个张量合并为一个连续的缓冲区。这种方法减少了单独传输数百个张量的调度开销，改为传输一个张量。

- 形成更大的批次。对于基于CPU的推理，更小的批次更受欢迎，以增加并行性和减少延迟。然而，对于基于GPU的推理，更大的批次更有效[29]。这导致我们重新评估我们的分布式系统设置。最初，我们使用scatter-gather架构将请求分割成小批次，并在多个叶节点上并行运行它们以获得更好的延迟。然而，这种设置与基于GPU的推理不兼容。相反，我们直接使用原始请求中的更大批次。为了补偿缓存容量的损失，我们实现了一个使用DRAM和SSD的混合缓存。

- 利用CUDA图。我们依靠CUDA图9来完全消除剩余的小操作开销。CUDA图将模型推理过程捕获为操作的静态图，而不是单独调度的操作，允许计算作为一个单独的单元执行，没有任何内核启动开销。

## 3.4.3 实时特征处理

当用户采取行动时，基于Flink10的实时特征处理应用程序会消费前端事件生成的用户行为Kafka11流。它验证每个行动记录，检测并合并重复项，并管理多个数据源的任何时间差异。然后，应用程序将特征具体化并存储在Rockstore[3]中。在服务时间，每个Homefeed日志/服务请求触发处理器将序列特征转换为模型可以使用的格式。

# 4.实验

略

# 附录

[https://arxiv.org/pdf/2306.00248](https://arxiv.org/pdf/2306.00248)