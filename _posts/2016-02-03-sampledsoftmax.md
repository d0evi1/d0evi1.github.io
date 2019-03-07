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

# 参考

[https://www.tensorflow.org/extras/candidate_sampling.pdf](https://www.tensorflow.org/extras/candidate_sampling.pdf)

