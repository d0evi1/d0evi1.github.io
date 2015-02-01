---
layout: page
title: gensim 
tagline: gensim 
---
{% include JB/setup %}



#介绍

gemsim是一个免费python库，设计目的是，从文档中有效地自动抽取语义主题。

gensim可以处理原始的，非结构化的文本（"plain text"）。gensim中的算法，包括：LSA(Latent Semantic Analysis), LDA(Latent Dirichlet Allocation), RP (Random Projections), 通过在一个训练文档语料库中，检查词汇统计联合出现模式, 可以用来发掘文档语义结构. 这些算法属于非监督学习，这意味着无需人工输入－－你只需提供一个语料库即可。

## 特性

- 内存独立- 对于训练语料来说，没必要在任何时间将整个语料都驻留在RAM中
- 有效实现了许多流行的向量空间算法－包括tf-idf，分布式LSA, 分布式LDA 以及 RP；并且很容易添加新算法
- 对流行的数据格式进行了IO封装和转换
- 在其语义表达中，可以相似查询

gensim的创建的目的是，由于缺乏简单的（java很复杂）实现主题建模的可扩展软件框架. 。。。

gensim的设计原则：

- [1].简单的接口，学习曲线低。对于原型实现很方便
- [2].根据输入的语料的size来说，内存各自独立；基于流的算法操作，一次访问一个文档.

更多：document similarity server(http://pypi.python.org/pypi/simserver).

## 核心概念

gensim的整个package会涉及三个概念：corpus(d0evil.github.io/gensim/corpus.html), vector(), model. 
