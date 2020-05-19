---
layout: post
title: polularity bias与miscalibration介绍
description: 
modified: 2019-10-23
tags: 
---

在《The Impact of Popularity Bias on Fairness and Calibration in
Recommendation》paper中，提出了polularity bias与miscalibration之间具有一定的关联：

# 摘要

最近，在fairness-aware的推荐系统上收获了许多关注，包括：在不同users或groups上提供一致performance的fairness。如果推荐不能公平地表示某一个确定user group的品味，而其它user groups则与他们的偏好一致时，则一个推荐系统可以被认为是unfair的。另外，我们使用一个被称为“miscalibration”的指标来measure一个推荐算法响应用户的真实偏好程度，我们会考虑多个算法在miscalibration上的不同程度。在推荐上一个知名类型的bias是polularity bias，在推荐中有少量流行items被过度呈现（over-represented），而其它items的majority不会获得大量曝光。我们推测，popularity bias是导致在推荐中的miscalibration的一个重要因素。我们的实验结果使用两个真实数据集，展示了不同user groups间algorithmic polularity bias和他们在popular items上的兴趣程度的强相关性。另外，我们展示了，一个group受algorithmic polularity bias的影响越多，他们的推荐是miscalibrated也越多。最后，我们展示了具有更大popularity bias趋势的算法会具有更大的miscalibration。

# 1.介绍

在推荐生成中近期关注的一个热点是：fairness。推荐的fairness在推荐的不同domain、不同users或user groups的特性（比如：protected vs. unprotected）、以及系统设计者的目标下具有不同的含义。例如，[12]中将fairness定义成不同user groups间accuracy一致性。在他们的实验中，观察到，特定groups（比如：女性）会比男性获得更低的accuracy结果。

用来衡量推荐质量的一个metrics是：calibration，它可以衡量推荐分发与用户评分过的items的一致性。例如，如果一个用户对70%的action movies以及30%的romance movies评过分，那么，该用户在推荐中也期望看到相似的pattern[27]。如果该ratio与用户profile不同，我们则称推荐是miscalibrated。Miscalibration自身不能被当成unfair，因为它只能简单意味着推荐不够个性化。然而，如果不同的users或user groups在它们的推荐中都具有不同程度的miscalibration，这可能意味着一个user group的unfair treatment。例如，[28]中定义了一些fairness metrics，它关注于不同user groups间estimation error的一致性效果。

协同推荐系统的一个显著限制是popularity bias：popular items会被高频推荐，在一些cases中，推荐甚至超过它们的popularity，而大多数其它items不能获得合适比例的关注。我们将algorithmic popularity bias定义成：一个算法扩大了在不同items上已存在的popularity差异。我们通过popularity lift指标来measure该增强效应（amplification），它表示在input和output间平均item polularity的差异。比如，关于popularity bias可能会有疑惑，可能有以下原因：long-tail items（non-popular）对于生成一个用户偏好的完整理解来说很重要。另外，long-tail推荐也可以被理解成是一个社会福利（social good）；存在popularity bias的market会缺乏机会来发现更多obscure的产品，从而被大量大品牌和知名artists占据。这样的market会越来越同质化（homogeneous），为创新提供更少的机会。

在本paper中，我们推荐：popularity bias是导致推荐列表miscalibration的一个重要因素。

。。。

# 2.相关工作



# 3.Popularity bias和miscalibration

## 3.1 Miscalibration

miscalibration用来measure用户真实偏好与推荐算法间差异程度。前面提到，如果在所有用户上都存在miscalibration，可能意味着个性化算法的failure。当不同user groups具有不同程度的miscalibration时，意味着对特特定user groups存在unfair treatment。

。。。

为了measure 推荐的miscalibration，我们使用[27]中的metric。假设u是一个user，i是一个item。对于每个item i，存在一个features集合C来描述它。例如，一首歌可能是pop、jazz，或一个电影的genres可以是action、romance、comedy等。我们使用c来表示这些独立categories之一。我们假设每个user会对一或多个items进行评分，这意味着会对属于这些items的features c感兴趣。对于每个user u的两个分布：一个是u评分过的所有items间的categories c的分布，另一个是u的所有推荐items间categories c的分布：

- $$p_u(c \| u)$$：在过去用户u评分过的items集合$$\Gamma$$上的feature c的分布为：

$$
p_u(c | u) = \frac{\sum\limits_{i \in \Gamma w_{u,i} p(c|i)}}{\sum\limits_{i \in \Gamma} w_{u,i}}
$$

...(1)

其中$$w_{u,i}$$是item i的权重，表示user u评分的频率。本paper中可以将w设置为1. 更关注不同分布上的差异，而非user profile的时序方面.

- $$q_u(c \| u)$$：推荐给user u的items list的feature c上的分布：

$$
q_u(c|u) = \frac{\sum\limits_{i \in \wedge} w_r(i) p(c|i)}{ \sum\limits_{i \in  \wedge} w_r(i)}
$$

...(2)

$$\wedge$$表示推荐items集合。item i的weight则表示推荐中的rank r(i)。比如：MRR或nDCG。此外我们将$$w_r$$设置为1, 确保$$q_u$$和$$p_u$$可比。

在两个分布间的dissimilarity程度用来计算推荐中的miscalibration。常见的比方法有：统计假设检验。本文则使用KL散度来作为miscalibration metric。在我们的数据中，许多profiles没有ratings，这会导致在$$p_u$$上出现零值，同样具有推荐列表只关注特定features，对于一些users也在$$q_u$$上也会出现零值。对于没有observations的情况KL散度是undefined。作为替代，我们使用Hellinger distance H，它适用于存在许多0的情况。miscalibration的定义如下：

$$
MC_u(p_u, q_u) = H(p_u, q_u) = \frac{|| \sqrt{p_u} - \sqrt{q_u}||_2}{\sqrt{2}}
$$

...(3)

通过定义发现，H distance满足三角不等式。$$\sqrt{2}$$可以确保$$H(p_u, q_u) \leq 1$$。

对于每个group G的整体的miscalibration metric $$MC_G$$可以通过在group G中所有users u的$$MC_u(p,q)$$的平均来获得。例如：

$$
MC_G(p, q) = \frac{\sum_{u \in G} MC_u(p_u, q_u)}{|G|}
$$

....(4)

### fairness

和[27]相似，本文将unfair定义为：对于不同user groups具有不同程度miscalibration时，则存在unfair。存在许多方式来定义一个user group，可以基于以下features：gender、age、occupation（职业）、education等。相似的，我们也可以基于它们的兴趣相似程度来定义user group。例如：对某些category上的兴趣度。

## 3.2 rating data中的Popularity Bias

推荐算法受popularity bias影响。也就是说，少量items会被推荐给多个users，而大多数其它items不会获得大量曝光。该bias可能是因为rating data的天然特性会倾向于popular items，因为该bias会存在algorithmic amplification。图1-b展示了两个user A和B的rated items的百分比。我们可以看到，user A的推荐被popularity bias高度影响，而user B在它们推荐中则不存在popularity bias的增强效应。

在许多领域，rating data会倾向于popular items——许多流行items会获得大量ratings，而其余items则具有更少的ratings。图2展示了item popularity的长尾分布。在其它datasets中也有相似的分布。流行的items之所以流行是有原因的，algorithmic popularity bias通常会将该bias放大到一个更大的程度。

并非每个用户都在流行items上具有不同程度的兴趣[4,22]。也会存在用户只能非流行、利基（niche）的items感兴趣。推荐算法也需解决这些用户的需求。图3展示了在不同user profiles上rated items的average popularity。用户首先会基于items的average popularity进行sort，接着绘出data。在movielens上，在图的最右侧和右侧，表示存在少量user，它们具有大量average item popularity，中间部分大量用户的分布具有0.1到0.15之间的average item popularity。在Yahoo Movies中，具有少量users具有low-popularity profiles，否则，distribution是相似的。这些图表明，用户在popular items上具有不同程度的偏向。

由于原始rating data上的imbalance，通常在许多情况下，算法会增强该bias，从而过度推出popular items，使得它们具有更大的机会被更多users进行评分。这样的推荐循环会导致rich-get-richer、poor-get-poorer的恶性循环。然而，并非每个推荐算法对popularity bias具有相同的放大能力（amplication power）。下一部分描述了，测量推荐算法所传播的popularity bias的程度。经验上，会根据popularity bias增强来评估不同算法的performance。

# 4.方法

略.

# 参考

- 1.[https://arxiv.org/pdf/1910.05755.pdf](https://arxiv.org/pdf/1910.05755.pdf)
- 2.[https://www.youtube.com/watch?v=lWFvGdZGMzk](https://www.youtube.com/watch?v=lWFvGdZGMzk)