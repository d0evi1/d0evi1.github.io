---
layout: post
title: PDAOM loss介绍
description: 
modified: 2023-04-18
tags: 
---


meituan在《Enhancing Personalized Ranking With Differentiable Group AUC Optimization》中提出了PDAOM loss：


# 抽要

AUC是评估classifier效果的一个常用指标。然而，大多数分类器使用cross entropy训练，它不会直接最优化AUC metric，这在training和evaluation间会存在一个gap。这里提出的PDAOM loss：一个**最大化violation的个性化可微AUC最优化方法（Personalized and Differentiable AUC Optimizationy method with Maximum violation）**, 当训练一个二分类器时可以直接应用，并使用gradient-based方法进行最优化。特别的，我们会使用通过user ID group在一起的sub-batches内的不同的（neg, postive）pair样本，来构建了pairwise exponential loss，目标是指导分类器关注：**在独立用户视角下，很难区分的正负样本对间的关系**。对比起pairwise exponential loss的原始形式，**提出的PDAOM loss不仅会提升在离线评估中的AUC和GAUC metrics，也会减少训练目标的计算复杂度**。再者，在“猜你喜欢”的feed推荐上，PDAOM loss的在线评估可以获得在点击数上 1.4%的提升，在订单数上获得0.65%的提升，这在online life service推荐系统上是个显著的提升。

# 1.介绍

二分排序(Bipartite ranking)在过去受到了大量关注，在工业应用中被广泛使用。它的目标是：**学习一个模型，使得正样本的排序高于负样本**。不失一般性，我们以推荐系统为例，并详述二分排序。根据一个用户的历史行为统计，推荐系统会提供一个关于items的有序列表，其中，感兴趣的items会出现在不感兴趣的items之前。达到该目标的关键思想是：为每个item预估CTR。用户浏览过但没有点击的items会被标记为负样本，点击过的items会被标记为正样本。接着，CTR预估模型可以被训练成一个二分类器，并使用 cross entropy进行最优化。这种方式下，每个样本会被独立对待，并且在训练期间，正负样本间的限制关系不会被引入。另一个关注点是：对比起用户看过的items，clicked items只占一小部分。因而，模型的效果基本上使用AUC metric进行评估，其中数据分布是imbalanced的。**AUC会measures：对于一个随机抽样的正例的score，它要比一个随机抽样的负例score具有更高分的概率**。然而，在训练期间cross entropy目标，不会完全与evaluation期间的目标完全对齐。实际上，一个常见现象是，当训练loss减少时AUC metric不会增加，在工业界推荐数据集上训练的一个示例如图1所示。它会启发我们，在训练期间直接对AUC metric进行最优化。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/038a25bd4c11d9f7f628fb50ee02f3fd4989fb48524db51935d213c71b73030f46ab9147663f8951cf106cf697b809f0?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 loss曲线和AUC metric随training steps的变化，其中AUC metric不总是随loss的减少而增加

另一个大问题是：**推荐系统经常面对“长尾”现象，例如：一小部分商品会占据着大量的销售额**。图2展示了来自Meituan电商的统计数据。我们根据它们的订单质量将商品分为100 bins，并绘制出top 30 bins。我们可以看到，**top 1 bin的商品贡献了37%的订单，top 20 bins的商品贡献了80%的订单**。**如果我们使用这样不均衡的数据来训练一个CTR预估模型，该模型会趋向于分配更高得分给热门商品，即使一些用户可能不喜欢这些items，这在个性化预估上会降低模型效果**。Group AUC【19】是一个合理的metric，用于评估一个ranking model的个性化推荐能力。它会通过user ID进行分组，计算各个sets中的AUC，并每个set的结果进行平均。经验上，对比起AUC metric，离线的GAUC metric会与在线效果更一致些，进一步启发我们在训练ranking model时将GAUC metric加入到objective中。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/51d0bc987f73347f8be231d74b4b868d1bcdfa7dd80c9afacfa4e2046598d575f1c9e3ace472d2d390cf079e7873abd8?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2 在美团电商中的长尾现象。top bins中的商品贡献了大比例的订单

然而，有两个原因阻止我们直接最优化AUC metric。

- 一方面，**AUC metric会通过一个binary indicator函数的总和来计算，它是不可微的**。因而，gradient-based optimization方法不能应用于该问题中。
- 另一方面，**AUC的公式会考虑上每个postive/negative样本对，会导致时间复杂度上升到$$O(N^+ N^-)$$**，它在实际工业推荐场景中是不可接受的，通常会是数十亿阶。

该问题上做了大量研究：

- 对于前者：对于原始AUC公式，最近工作尝试替代可微替代目标函数（differentiable surrogate objective function）。【18】提出了使用它的convex surrogate（例如：hinge loss function）来替代indicator function。【2】设计了一个regression-based算法，它使用pairwise squared objective function，用于measure在不同分类的两个实例间的ranking errors。【3】研究了基于最优化pairwise surrogate objective functions的AUC一致性
- 对于后者：mini-batch 最优化策略可以使得处理大规模数据集。为了将AUC最大化方法应用于data-intensive场景，我们会研究以mini-batch方式最优化它。

特别的，我们提出了PDAOM loss，它关注于**难区分的正负样本对**，而非将所有组合考虑在内。该trick不仅会提升offline效果，也会减小最优化的复杂度。

# 2.相关工作

略

# 3.方法

## A.前提

AUC的原始定义与ROC curve有关。假设分类器会生成一个连续值，用于表示输入样本是postive的概率，**接着需要一个决策阈值来决定：样本是该标记成postive还是negative**。对于每个阈值来说，我们可以获得一个true-postive rate和false-postive rate的pair。通过将该阈值从0到1进行遍历，我们绘制出获得的rates的pairs，就生成了ROC曲线。因而，通过对ROC曲线的下面面积计算均值的AUC metric是很复杂的。AUC的一个等价形式是：归一化的Wilcoxon-Mann-Whitney (WMW)统计：

$$
AUC = \frac{\sum\limits_{i=0}^{m-1} \sum\limits_{j=0}^{n-1} \mathbb{1} (f(x_i^+) > f(x_j^-))}{mn}
$$

...(1)

其中：

$$\lbrace x_i^+ \rbrace$$和$$\lbrace x_j^- \rbrace$$分别是postive和negative样本的集合。由于indicator function是不可微的，WMW统计不能使用gradient-based算法进行最优化。它启发我们寻找一种可微的surrogate objective function，用于替代WMW statistic。也就是说，获得surrogate function的最优解等价于最大化AUC metric。我们将使用将surrogate的objective function公式为：

$$
\underset{x^+ \sim P^+, \\ x^- \sim P^-}{E} (\phi(f(x^+) - f(x^-)))
$$

...(2)

其中：

- $$\phi$$是surrogate function，一些常用的示例如表1所示。
- $$P^+, P^-$$分别表示正负样本分布

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ebe48202159d7bb9fba6a936ebee37ecc2ae4c94c3e84f7f8ef1be9c5c25422c3771a87fded7035dc121529d1e34d115?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t1.jpg&amp;size=750">

表1 常用的surrogate function，当成AUC的indicator

在本paper中，我们出于两个原因会**使用pairwise exponential loss 作为surrogate**。

- 首先，【3】中已经证明，pairwise exponential loss与AUC一致。
- 第二，我们会对比在我们的先决实验列出的surrogates，并发现pairwise exponential loss要胜过离线评估（如表4所示）

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ade73257d7c724c997c3254525432d93be0d6d63ec86a10db3c384b7550fca3b88a5f3175534fa9e5afec336ba0bbf4e?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t4.jpg&amp;size=750">

表4 

## B. AUC Optimization with Maximum Violation

上述提到的surrogate objective function存在两个主要缺点：

- 一方面，这样的objective会等价地关注于每个pair，因而，分类器会花费大量精力在建模**易区分的正负样本对关系上**。
- 另一方面，对于一个batch数据，它包含了$$N^+$$的正样本和$$N^-$$的负样本，处理每个pair的复杂度是$$O(N^+ N^-)$$，它对于一个大batch-size来说是很耗时的。

聚焦于上面的问题，我们提出构建**难样本对**，并让模型关注于难样本对，而非所有样本对。一个难样本对指的是：**模型很难区分正/负样本labels的实例。因而，对于这样的正负样本对，输出scores是很接近的，它使得决策边界不好判断**。考虑：

$$
\overset{E}{x^+ \sim P^+, x^- \sim P^-} (\phi(f(x^+)) - f(x^-)) \leq \overset{max}{x^+ \sim P^+, x^- \sim P^-} ( \phi(f(x^+) - f(x^-)))
$$

...(3)

（3）的一个可行解是：设置 $$\underset{max}{x^+ \sim P^+, x^- \sim P^-}(\phi(f(x^+) - f(x^-)))$$作为objective function。计算该最大值只依赖于那些很可能有violate relation的正负样本对。在该方式下，来自easy negatives的loss累积不会影响模型的更新。尽管这样的转换会导致模型关注于确定决策边界，复杂度仍然是$$O(N^+ N^-)$$。由于$$f(x^+) - f(x^-) \in [-1, 1]$$，surrogate function $$\phi$$会在该区间内单调递减。相等的，$$$max_{x^+ \sim P^+, x^- \sim P^-} (\phi(f(x^+) - f(x^-)))$$可以简化为：

$$
\phi (min_{x^+ \sim P^+, x^- \sim P^-}(f(x^+) - f(x^-))) = \phi( min_{x^+ \sim P^+} f(x^+) - max_{x^- \sim P^-} f(x^-))
$$

...(4)

理想的，的一个正样本的最低分期望会高于在一个batch内负样本的最高分。我们将DAOM loss定义为：

$$
L_{DAOM} = \phi( \overset{min}{x^+ \sim P^+} f(x^+) - \overset{max}{x^- \sim P^-} f(x^-))
$$

...(5)

注意，我们只会选择具有最高score的正样本，以及具有最低score的负样本，构建pair的复杂度会减小到 $$O(N^+ + N^-)$$。

## 通过GAUC最优化来增强个性化排序

以上章节详述了如何构建在一个batch内的paired samples，它不会满足个性化推荐的需求。实际上，我们会发现，GAUC[19] metric与在线效果更一致些。相应的，一个天然的想法是，当最优化模型时，将GAUC metric添加到objective中。考虑GAUC指标的原始计算，样本会首先被分成多个groups。在本context中，groups会被通过user ID进行划分。接着，AUC metric会分别在每个group中计算，GAUC metric会通过将所有groups的AUC metrics进行加权平均得到。weight与曝光或点击次数成比例，这里我们对所有用户将weight设置为1。我们会在训练阶段模拟GAUC的计算。当准备训练数据时，我们会根据每个样本的user ID对样本进行排序，以便一个用户的样本会出现在相同的batch中。一个batch的数据可能会包含许多不同的user IDs，我们将batch划分成sub-batches，在sub-batch中的user ID是相同的。接着我们应用DAOM loss到每个sub-batch中，并将个性化DAOM loss定义为：

$$
L_{PDAOM} = \sum\limits_{u \in U} \phi( min\limits_{x^+ \sum P_u^+} f(x^+) - max\limits_{x^- \sim P_u^-} f(x^-))
$$

...(6)

其中，U表示由user ID分组的sub batches。在训练一个二分类器的条件下，提出的PDAOM loss与cross entropy一起来形成最终的objective function：

$$
L = -y log(f(x)) - (1-y) log(1 - f(x)) + \lambda \sum\limits_{u \in U} \phi( min\limits_{x^+ \sim P_u^+} f(x^+) - max\limits_{x^- \sim P_u^-} f(x^-))
$$

...(7)

其中：y是label，$$\lambda$$会对cross entropy和PDAOM loss的weight进行balance。

# 实验

略

- 1.[https://arxiv.org/pdf/2304.09176.pdf](https://arxiv.org/pdf/2304.09176.pdf)