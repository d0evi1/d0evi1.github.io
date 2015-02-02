---
layout: page
title: gensim教程 
tagline: 介绍
---
{% include JB/setup %}

#1.介绍

gemsim是一个免费python库，设计目的是，从文档中有效地自动抽取语义主题。

gensim可以处理原始的，非结构化的文本（"plain text"）。gensim中的算法，包括：LSA(Latent Semantic Analysis), LDA(Latent Dirichlet Allocation), RP (Random Projections), 通过在一个训练文档语料库中，检查词汇统计联合出现模式, 可以用来发掘文档语义结构. 这些算法属于非监督学习，这意味着无需人工输入－－你只需提供一个语料库即可。

#2.特性

- 内存独立- 对于训练语料来说，没必要在任何时间将整个语料都驻留在RAM中
- 有效实现了许多流行的向量空间算法－包括tf-idf，分布式LSA, 分布式LDA 以及 RP；并且很容易添加新算法
- 对流行的数据格式进行了IO封装和转换
- 在其语义表达中，可以相似查询

gensim的创建的目的是，由于缺乏简单的（java很复杂）实现主题建模的可扩展软件框架. 。。。

gensim的设计原则：

- [1].简单的接口，学习曲线低。对于原型实现很方便
- [2].根据输入的语料的size来说，内存各自独立；基于流的算法操作，一次访问一个文档.

更多：[document similarity server](http://pypi.python.org/pypi/simserver).

#3.核心概念

gensim的整个package会涉及三个概念：[corpus](d0evi1.github.io/gensim/corpus.html), [vector](d0evi1.github.io/gensim/vector.html), [model](d0evi1.github.io/gensim/model.html).

## a.语库(corpus)
文档集合.该集合用来自动推出文档结构，以及它们的主题等。出于这个原因，这个集合被称为：训练语料。

## b.向量(vector) 
在向量空间模型(VSM)中，每个文档被表示成一个特征数组。例如，一个单一特征可以被表示成一个问答对(question-answer pair):

- [1].在文档中单词"splonge"出现的次数？ 0个
- [2].文档中包含了多少句子？ 2个
- [3].文档中使用了多少字体? 5种

这里的问题可以表示成整型id (比如：1,2,3等), 因此，上面的文档可以表示成：(1, 0.0), (2, 2.0), (3, 5.0). 如果我们事先知道所有的问题，我们可以显式地写成这样：(0.0, 2.0, 5.0). 这个回答序列可以认为是一个多维矩阵（3维）. 对于实际目的，只有问题.

对于每个文档来说问题是类似的.两个向量（分别表示两个文档），我们希望可以下类似的结论：“如果两个向量中的数是相似的，那么，原始的文档也相似”。

## c.稀疏矩阵(Sparse vector)

[todo]

## d.模型(model)

[todo]
