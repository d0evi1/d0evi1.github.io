---
layout: post
title: md embedding介绍
description: 
modified: 2020-07-01
tags: 
---

facebook在2019的《Mixed Dimension Embedding with Application to
Memory-Efficient Recommendation Systems》，提出了一种mixed dimension embedding，并且在dlrm中做了实现，我们来看下具体的概念。

# 2.背景

我们主要关注explicit CF和CTR预估这两块。在explicit CF中，特定items的user ratings会被直接观测到，因此，它可以被看成是一个矩阵补全问题（matrix complition problem）。在解决矩阵补全问题时，embedding-based的方法（比如：MF、NCF）是流行的高效解法。它的核心是，使用一个convex relaxation来发现最小的nuclear norm solution。它需要求解一个半正定的program，时耗是$$O(n^4)$$，因而不能扩展到real-world应用中。作为替代，**embedding/factorization方法具有明显缺点：显式需要一个embedding dimension**，但实际上，这可以使用cross-validation或其它超参数tuning技术求解。在CTR预测任务中，我们会预测一个click的概率，它可以被看成是只有二元rating的context-based CF。最近该领域开发了许多模型。这些state-of-the-art的模型会具有许多相似的特征，他们无一例外都使用**内存密集型(memory-intensive) embedding layers**来简化模型其余部分。

在现代ML中，embeddings通常会用来表示categorical features。embeddings vectors会从数据中进行挖掘，由vectors表示的categorical概念间的特定语义关系可以通过spatial或geometric关系、以及vectors的属性来编码。因而，**large embeddings**是推荐系统的天然选择，它需要模型来理解users和items间的关系。

目前开发了许多技术来减小由embedding layers消耗的内存量。他们可以被划分成两类：

- (i) 压缩算法（compression algorithms）
- (ii) 压缩结构 （compressed architectures）

在标准训练之前，压缩算法通常会涉及到对模型进行一些额外处理。它们可以离线（只涉及到post-training处理）或在线执行（压缩处理会交太、或者部分变更训练处理）。简单的offline压缩算法包括：post-training quantization、pruning或low-rank SVD。模型蒸馏技术（比如：compositional coding）和neural binarization也是一种复杂的离线压缩方法，其中：autoencoders会被训练成mimic uncompressed、pre-trained embedding layers。在线压缩算法包括：quantization-aware training、gradual pruning、以及periodic regularization。我们注意到，许多这些压缩算法对于embedding layers是不唯一的，在模型压缩文献中被广泛使用。

另一方面，我们也可以压缩结构，它尝试使用更少的参数来构建关于可比静态质量(comparable statistical quality)的embedding representations。**压缩结构的优点是，inference时不只可以减小内存需要，在training时也可以**。该方法遵循hashing-based和tensor factorization方法，它们可以以多种方式通过re-using参数来减小在一个embedding layer上使用的参数数目。我们的方法与这些技术不同，我们基于embedding popularity来对embedding vectors的维度进行非统一（non-uniform）reduction。原则上，我们提出的技术与大多数其它压缩算法或压缩结构的方法可以混合使用。这是未来的一个方向。

最终，我们注意到，non-uniform和deterministic sampling在矩阵补全文献中有提出【37】，同时也包括：纠正popularity来提升statistical recovery performance，或者在non-uniform sampling下为completion提供理论保证。据我们所知，我们是第一个利用popularity来在大规模embedding layers中实际减小参数数的。


# 4.Mixed Dimension Embedding Layer

我们开始定义MD embedding layer，并讨论如何将它应用于CF和CTR预测任务上。

假设一个mixed dimension embedding layer $\bar{E}$ 包含了k+1个blocks，它可以通过2k+1个matrices来定义：

$$
\bar{E} = (\bar{E}^{(0)},\bar{E}^{(1)}, \cdots, \bar{E}^{(k)}, P^{(1)}, \cdots, P^{(k)})
$$

...(4)

其中，

- $\bar{E}^{(i)} \in R^{n_i \times d_i}$
- $P^{(i)} \in R^{d_i \times d_0}$，其中$P^{(0)} \in R^{d_0 \times d_0}$是显式定义

假设这些blocks的维度是固定的。接着，对于一个MD embedding layer的forward propagation，会采用一个范围在(1, $$n=\sum_{i=0}^k n_i$$)的index x，并产生一个如算法1所定义的embedding vector $$e_x$$。在该算法中涉及到的steps是可微的，因此我们会通过该layer执行backward propagation，并在训练期间更新matrices $$\bar{E}^{(i)}$$和 $$P^{(i)}$$。我们注意到算法1可以被泛化成支持multi-hot lookups，其中对应于一些z query indices的embedding vectors会被取到（fetched），并通过一个可微操作符（比如：add, multiply, concatenation）进行recude。


<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/d652fa9bc5c07a2274d796078002cf2ed0b9a232de315ccc4cebcaf0b23aebfd809583963a772d6114e6cdfe9e97c15a?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=a1.jpg&amp;size=750">

算法一

注意，我们会为除了第一个embedding matrix之外的所有embeddings返回投影embeddings（projected embeddings），所有的embedding vectors $$e_j$$会具有相同的base dimension $$d:= d_0$$。因此，**基于一个mixed dimension embedding layer的模型应根据$$\bar{d}$$来确定size**。我们在图2中展示了mixed dimension embedding layer的矩阵结构，它具有两个blocks，其中，由uniform或mixed dimension matrices的参数预算（parameter budget（总区域））是相同的，但内存分配不同。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/a848f955fc4ecc887841af6c9772ed233a885aa47f3e4dc86c732876bfae0b8121c37bdbe29797ae8d99e4a288f32f89?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2 Uniform和Mixed Dimension Embedding layers的matrics结构


接着，我们看下如何来在mixed dimension结构中寻找block结构。这包括了：在mixed dimension embedding layer中分配给每个block的行数（row count）$$n_i$$、以及维度（dimension）$$d_i$$。我们使用流行度信息（popularity information）来界定（sizing）mixed dimension embedding layer的大小（例如：访问一个特定feature的频率f；假设这里在training和test样本间大部分一致）。我们注意到，你可以使用一个与importance相关但不同的概念：它指的是一个特定的feature通常是如何统计信息给target variable的inference的。Importance可以通过domain experts或在训练时通过data-driven的方式来决定。

## 4.1 Mixed Dimensions的Blocking Scheme

从一个uniform dimension embedding layer转成一个mixed dimension layer，存在一些延续性。通过使用合理的re-indexing，多个embedding matrics可以被stack成单个block matrix，或者单个embedding matrix可以通过row-wise的形式划分(partitioned)成多个block matrices。**partitioned blocking scheme的关键是，将n个total embedding rows映射成blocks**，其中，block-level row counts通过$$(n_0, \cdots, n_k)$$和offset vector $$t \in N^{k+1}$$给出，有：

$$
t_i := \sum_{j=0}^{i-1} n_j
$$

**Blocking for CF**

在CF任务中，我们只有两类features——分别对应于users和items，对应的embedding matrices分别为：

- $$W \in R^{n \times d}$$
- $$V \in R^{m \times d}$$

为了对mixed dimension embedding layer定大小，我们会使用mixed dimensions，它使用单独的embedding matrices来对它们进行划分。

- 首先，我们基于row-wise frequency来对rows进行排序（sort）和re-index：满足$$i < i' \rightarrow f_i \geq f_{i'}$$（即：索引i越小，频率越大）。
- 接着，我们将每个embedding matrix划分成k+1个blocks，以便每个block内的total popularity（AUC）是常数，如算法2所示。

对于一个给定的frequency f，k等分（k-equipartition）是唯一的，并且很容易计算。在我们的实验中可以看到，对于观察到由mixed dimensions带来的效果，以及这之外的递减效应（diminishing effect），在(8,16)范围内的任意地方设置k是足够的。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/1332f15bd41d50ce2ae0ea3841e878837005a1c6e4eedb507ebb21511617eceabace26fc24d8402098bf6f70ea0beb9a?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=alg2.jpg&amp;size=750">

算法2

**Blocking for CTR prediction**

在CTR预测任务中，我们有一些categorical features，以及k+1个相对应的embedding matrics： $$E^{(i)} \in R^{n_i \times d}$$。对于ctr prediction应用，为了界定MD embedding layer的size大小，我们会跨不同的embedding matrices上使用mixed dimensions来对它们进行stacking。因此，该问题结构定义了blocks数；在每个original embedding中的vectors数目定义了在md embedding layer中相应block的行数（row counts）$$n_i$$。

## 4.2 popularity-based mixed dimensions

假设在md embedding layer $$\bar{E}$$的每个block中的vectors数目$$n_i$$是已经固定的。因此，它只分配了维度$$d:=(d_0, \cdots, d_k)$$来完全指定它。


我们提出了一个popularity-based scheme来在block-level上操作，它基于这样一个heuristic：每个embedding应分配一个维度，它与popularity的一些分数幂（fractional power）成比例。注意，这里我们会将block-level probability p与row-wise frequency f进行区别。给定f，我们会定义$$a_i = \sum_{j=t_i}^{t_{i+1}} f_j$$作为在区间$$[t_i, t_{i+1}]$$间的frequency curve的面积，总的$$\tau = \sum_{j=0}^n f_j$$。接着，我们假设：block-level probability vector $$p \in R^{k+1}$$通过它的elements $$p_i=a_i/\tau$$来定义。我们将算法3中的popularity-based scheme进行公式化，使用一个额外的超参数temperature $$a > 0$$。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/9893f13df3b771cf8176302bd687f0fb125c36ac909750e4c1c271acc69f3ac19065c72171730782ead4b27b46aedb05?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=alg3.jpg&amp;size=750">

算法3

提出的技术需要知道probability vector p，它管理着feature popularity。当这样的分布是未知的，我们会很容易使用来自数据样本的经验分布来替代它。

可选的，我们可以将$$\lambda \leftarrow B(\sum_i p_i^{a-1})^{-1}$$设置成：将 embedding layer sizeing的大小限制到一个total budget B上。更进一步，为了获得crisp sizing，可以使用round d，可能是2的最近幂，在应用算法3后。

注意，我们最终会具有所有的tools来对MD embedding layers进行size，同时使用算法1-3对它们进行forward和backward propagation。


# 参考

- 1.[https://arxiv.org/pdf/1909.11810.pdf](https://arxiv.org/pdf/1909.11810.pdf)