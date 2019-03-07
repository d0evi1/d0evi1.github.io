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

在full softmax训练中，对于每个训练样本$$(x_i,\lbrace t_i \rbrace)$$，我们会为在$$y \in L$$中的所有类计算logits F(x_i,y)。如果类L很大，计算很会昂贵。

在"Sampled Softmax"中，对于每个训练样本$$(x_i, \lbrace t_i \rbrace)$$，我们会根据一个选择抽样函数$$Q(y \mid x)$$来选择一个关于“sampled” classese的小集合$$S_i \subset L$$。每个类$$ y \in L $$被包含在$$S_i$$中，它与概率$$Q(y \mid x_i)$$完全独立。



# 参考

[https://www.tensorflow.org/extras/candidate_sampling.pdf](https://www.tensorflow.org/extras/candidate_sampling.pdf)

