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

