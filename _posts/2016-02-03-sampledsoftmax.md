---
layout: post
title: sampled softmax介绍
description: 
modified: 2016-02-03
tags: 
---

一种更快的训练softmax分类器的方式。

# 介绍

假设我们有一个单标签问题（single-label）。每个训练样本$$(x_i, \lbrace t_i \rbrace)$$包含了一个context以及一个target class。我们将$$P(y \mid x)$$作为：给定context x下，一个target class y的概率。

我们可以训练一个函数F(x,y)来生成softmax logits——也就是说，给定context，该class相对log概率：

$$
F(x,y) \leftarrow log(P(y|x)) + K(x)
$$

其中，K(x)是一个特有函数，它不依赖于y。

在full softmax训练中，对于每个训练样本$$(x_i,\lbrace t_i \rbrace)$$，我们会为在$$y \in L$$中的所有类计算logits $$F(x_i,y)$$。如果类L很大，计算很会昂贵。

在"Sampled Softmax"中，对于每个训练样本$$(x_i, \lbrace t_i \rbrace)$$，我们会根据一个选择抽样函数$$Q(y \mid x)$$来选择一个关于“sampled” classese的小集合$$S_i \subset L$$。每个类$$ y \in L $$被包含在$$S_i$$中，它与概率$$Q(y \mid x_i)$$完全独立。

$$
P(S_i = S|x_i) = \prod_{y \in S} Q(y|x_i) \prod_{y \in (L-S)} (1-Q(y|x_i))
$$

我们创建一个候选集合$$C_i$$，它包含了关于target class和sampled classes的union：

$$
C_i = S_i \cup \lbrace t_i \rbrace
$$

我们的训练任务是为了指出，在给定集合$$C_i$$上，在$$C_i$$中哪个类是target class。

对于每个类$$y \in C_i$$，给定我们的先验$$x_i$$和$$C_i$$，我们希望计算target class y的后验概率。

使用Bayes' rule：

$$
P(t_i=y|x_i,C_i) = P(t_i=y,C_i|x_i) / P(C_i|x_i) \\
=P(t_i=y|x_i) P(C_i|t_i=y,x_i) / P(C_i|x_i) \\
=P(y|x_i)P(C_i|t_i=y,x_i) / P(C_i|x_i)
$$

接着，为了计算$$P(C_i \mid t_i=y,x_i)$$，我们注意到为了让它发生，$$S_i$$可能或不可能包含y，必须包含$$C_i$$所有其它元素，必须不包含在$$C_i$$任意classes。因此：

$$
P(t_i=y|x_i, C_i) = P(y|x_i) \prod_{y \in C_i - \lbrace y \rbrace} Q({y'}|x_i) \prod_{y \in (L-C_i)} (1-Q({y'}|x_i)) / P(C_i | x_i) \\
= \frac{P(y|x_i)}{Q(y|x_i)} \prod_{{y'} \in C_i} Q({y'}|x_i) \prod_{{y'} \in (L-C_i)} (1-Q({y'}|x_i))/P(C_i|x_i) \\
=\frac{P(y|x_i)}{Q(y|x_i)} / K(x_i,C_i)
$$

其中，$$K(x_i,C_i)$$是一个函数，它与y无关。因而：
这些是relative logits，应feed给一个softmax classifier，来预测在$$C_i$$中的哪个candidates是正样本（true）。

因此，我们尝试训练函数F(x,y)来逼近$$log(P(y \mid x))$$，它会采用在我们的网络中表示F(x,y)的layer，减去$$log(Q(y \mid x))$$，然后将结果传给一个softmax classifier来预测哪个candidate是true样本。

$$
training softmax input=F(x,y) - log(Q(y|x))
$$

从该classifer对梯度进行BP，可以训练任何我们想到的F。

# 参考

[https://www.tensorflow.org/extras/candidate_sampling.pdf](https://www.tensorflow.org/extras/candidate_sampling.pdf)

