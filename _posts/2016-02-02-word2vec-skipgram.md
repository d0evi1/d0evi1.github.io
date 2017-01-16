---
layout: post
title: word2vec中的Skip-gram
description: 
modified: 2016-02-02
tags: [word2vec+Huffman]
---

本文主要译自Tomas Mikolov、Jeffrey Dean等人的<Distributed Representations of Words and Phrases
and their Compositionality>.

# Abstract

最近介绍的continuous Skip-gram model是一个很有效的方法，用于学习高质量的分布式向量表示，它可以捕获大量精准的语法结构和语义词关系。在该paper中，我们介绍了许多种扩展，用来改进向量的质量和训练速度。通过对高频词进行subsampling，我们可以获得极大的速度提升，也可以学到更常规的词表示。我们也描述了在hirearchical softmax之外的一种简单的备选方法：negative sampling。

词向量表示的一个限制是，它不同于词顺序(word order)，它们不能表示常用短语。例如，"Canada"和"Air"的意思，不能简单的组合在一起来获取"Air Canada"。受该示例的启发，我们描述了另一种方法来寻找文本中的短语，并展示了如何在上百万的句子中学习到好的向量表示。

# 介绍

词在向量空间上的分布式表示(distributed representations)，通过将相似的词语进行群聚，可以帮助学习算法在nlp任务中完成更好的性能。词向量表示的最早应用可以追溯到1986年Rumelhart, Hinton等人提的(详见paper 13). 该方法用于统计语言建模中，并取得了可喜的成功。接下来，应用于自动语音识别和机器翻译(14,7)，以及其它更广泛的NLP任务(2,20,15,3,18,19,9)

最近，Mikolov（8）提出了Skip-gram模型，它是一个高效的方法，可以从大量非结构化文本数据中学到高质量的词向量表示。不同于大多数之前用于词向量学习所使用的神经网络结构，skip-gram模型(图1)不会涉太到稠密矩阵乘法(dense matrix multiplications)。这使得学习极其有效率：一个优化版的单机实现，一天可以训练超过10亿个词。

使用神经网络的词向量表示计算非常有意思，因为学习得到的向量显式地编码了许多语言学规律和模式。更令人吃惊的是，许多这些模式可以被表示成线性变换(linear translations)。例如，比起其它向量，向量计算vec("Madrid")-vec("Spain")+vec("France")与vec("Paris")的结果更接近(9,8)。

本文中，我们描述了一些原始skip-gram模型的扩展。我们展示了高频词的subsampling，在训练期间可以带来极大的提升（2x-10x的性能提升），并改进了低频词的向量表示的精度。另外，我们提出了一种Noise Contrastive Estimation (NCE) (4)的变种，来训练skip-gram模型，对比于复杂的hierachical softmax，它的训练更快，并可以为高频词得到更好的向量表示。

词向量表示受限于它不能表示常用短语，因为它们不由独立的单词组成。例如, “Boston Globe”实际是个报纸，因而它不是由“Boston”和"Globe"组合起来的意思。因此，使用向量来表示整个短语，会使得skip-gram模型更有表现力。因此，通过语向量来构成有意义的句子的其它技术（比如：递归autoencoders 17)，可以受益于使用短语向量，而非词向量。

从基于词的模型扩展成基于短语的模型相当简单。首先，我们使用数据驱动的方法标识了大量的短语，接着我们在训练中将这些短语看成是独自的tokens。为了评估短语向量的质量，我们开发了一个类比原因任务(analogical reasoning tasks)测试集，它同时包含了词和短语。我们测试集中的一个典型的类比对(analogy pair)：“Montreal”:“Montreal Canadiens”::“Toronto”:“Toronto Maple Leafs”。如果与vec("Montreal Canadiens")-vec("Montreal")+vec("Toronto")最接近的向量是：vec("Toronto Maple Leafs")，那么我们可以认为回答是正确的。

译者注1：

- Montreal: 蒙特利尔(城市)
- Montreal Canadiens: 蒙特利尔加拿大人(冰球队)
- Toronto: 多伦多(城市)
- Toronto Maple Leafs: 多伦多枫叶队(冰球队)

译者注2:

英文是基于空格做tokenized的. 常出现这个问题。

最后，我们再描述skip-gram模型的另一个有趣特性。我们发现，向量加法经常产生很有意思的结果，例如：vec("Russia")+vec("river")的结果，与vec("Volga River")接近。而vec("Germany")+vec("capital")的结果，与vec("Berlin")接近。这种组成暗示着，语言中一些不明显的程度，可以通过使用基本的词向量表示的数据操作来获取。

# 2.Skip-gram模型

Skip-gram模型的训练目标是，为预测一个句子或一个文档中某个词的周围词汇，找到有用的词向量表示。更正式地，通过给定训练词汇w1,w2,w3,...,wT, Skip-gram模型的目标是，最大化平均log概率：


<img src="http://www.forkosh.com/mathtex.cgi?\frac{1}{T}\sum_{t=1}{T}\sum_{-c\leqj\leqc,j\neq0}logp(w_{t+j}|w_t)">  (1)

其中，c是训练上下文的size(wt是中心词)。c越大，会产生更多的训练样本，并产生更高的准确度，训练时间也更长。最基本的skip-gram公式使用softmax函数来计算p(wt+j|wt): 

<img src="http://www.forkosh.com/mathtex.cgi?p(wO|wI)=\frac{exp(v'_{wo}^Tv_{wI}}{\sum_{w=1}{W}exp(v'_{w}^Tv_{wI}}">  (1)

其中，vw和v'w表示w的输入向量和输出向量。W则是词汇表中的词汇数。该公式在实际中不直接采用，因为计算logp(wo|wi)的梯度与W成正比，经常很大(10^5-10^7次方)





# 参考

－ 1.[Domain adaptation for large-scale sentiment classi-
fication: A deep learning approach](http://svn.ucc.asn.au:8080/oxinabox/Uni%20Notes/honours/refTesting/glorot2011domain.pdf)


当数据集包含上百万的词时，该模型的表现优于基于分类的3-gram，但比paper 6中的NPLM表现差。这种层次化NPLM模型，比普通的NPLM快2个数量级。这种方法的主要限制，主要是用于构建word树的过程。该树可以从WordNet IS-A分类体系开始，通过结合人工和数据驱动处理，并将它转换成一个二叉树。我们的目标是，将该过程替换成从训练数据中自动构建树，不需要任何专家知识。我们也探索了使用树（里面的词汇至少出现一次）的性能优点。
公式四：<img src="http://www.forkosh.com/mathtex.cgi?P(d_{i}=1|q_{i},w_{1:n-1})=\delta(\hat{r}^Tq_{i}+b_{i})">


## 参考

- 1.[A Scalable Hierarchical Distributed Language Model](http://www.cs.toronto.edu/~amnih/papers/hlbl_final.pdf)
