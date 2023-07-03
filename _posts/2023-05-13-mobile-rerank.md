---
layout: post
title: ks端重排中的beam search介绍
description: 
modified: 2023-05-13
tags: 
---

kuaishou在《Real-time Short Video Recommendation on Mobile Devices》中介绍了它们的beam search方式：

# 5.实时triggered context-aware reranking

一旦用户观看完一个视频，并生成新的实时ranking信号，我们可以相应更新我们的client-side model predictions，并触发一个新的re-ranking process。给定更新后的model predictions，存在许多方式来决定展示给用户的视频列表。最广泛使用的是point-wise ranking，它会贪婪地通过得分递减方式来对视频排序，然而，point-wise ranking会忽略在候选间的相互影响，因而它不是最优的。

理想的，我们希望：发现候选集合C的最优排列P，它会导致最大化ListReward(LR)，定义成：

$$
LR(P) = \sum\limits_{i=1}^{|P|} s_i(\alpha \cdot p(effective\_view_i | c(i)) + \beta \cdot p(like_i | c(i)))
$$

...(4)

其中：

$$
s_i = \begin{cases}
\prod\limits_{j=1}^{i-1} p(has\_next_j | c(j)), && i geq 2 \\
1, && i = 1
\end{cases}
$$

...(5)

是累积has_next概率直到位置i，它会作为一个discounting factor来合并future reward。

- $$p(has\_next_i \mid c(i)), p(effective\_view_i \mid c(i)), p(like_i \mid c(i))$$：分别是在$$v_i$$上has_next、effective_view、like上的predictions
- $$c(i)$$：在等式(1)中定义的ranking context 
- $$\alpha$$和$$\beta$$是不同rewards的weights

然而，直接搜索最优的排列，需要评估每个可能list的reward，它是代价非常高的，由于它是阶乘复杂度$$O(m!)$$（m是candidate set的size）。Beam search是对该问题的常用近似解，它会将时间复杂度减少到$$O(km^2)$$，其中k是beam size。然而，二次时间复杂度对于部署在生产环境上仍然过高。幸运的是，不同于server-side reranking的case，我们只需要延迟决定用户在该设备上可能同时看到的下n个视频（图1(a)中所示在我们的沉浸式场景n=1）。在我们的离线实验中（表5），我们观察到，在不同beams间的ListReward的相对不同之处是，在每个search step会随着搜索step的增长而同步递减。因而，我们提出一个新的beam search策略，来选择一个adaptive search step $$n \leq 1 \ll m$$，并进一步减小搜索时间复杂度到$$O(klm)$$。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/14d27f02b4ee48fb1a5b3ce458f6bb7e4c6206ff1852753f8bf6b5447c3977ff07beb495908fafab6fa799a78b08a5ea?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图4: adaptive beam search过程演示，其中：候选数n=4，beam size k=2, stability阈值t=0.95，在每个candidate、或candidate list上的数目表示相应的reward

为了实现adaptive beam search，我们定义了一个 stability：在当前的beam search step上将最小的ListReward除以最大的ListReward：

$$
stability(score\_list) = \frac{min(score\_list)}{max(score\_list)}
$$

...(6)

一旦stability超出了一个给定threshold t，beam search过程会终止，以便节省不必要的计算，由于我们可能期望：在剩余search steps中不会存在大的区别。adaptive beam search过程如图4所示。

算法如算法1所示，它会在导出的TFLite的执行图中被实现。


<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/2bbabc4e43692ede01b174ca7f00d984b0b6f7d621cc13c2c8d6c6fc32b6ecbaed0f2539b511fb4b3169d008163f195b?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

算法1


- 1.[https://arxiv.org/pdf/2208.09577.pdf](https://arxiv.org/pdf/2208.09577.pdf)