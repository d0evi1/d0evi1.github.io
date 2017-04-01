---
layout: post
title: paragraph2vec介绍
description: 
modified: 2016-03-05
tags: [sentence2vec]
---

paragraph2vec在[1]有详细介绍，我们先来看下具体的概念：

## 1.PV-DM:(Paragraph Vector:Distributed Memory model) 
 
学习段落向量(paragraph vector)的方法，受词向量(word vector)方法的启发。词向量会被用于预测句子中的下一个词。因此，尽管实际上，词向量的初始化是随机的，它们仍可以捕获语义，作为预测任务的间接结果。我们在paragraph vector中使用相类似的方式。从段落中抽样获得多个上下文，paragraph vector也同样可以用来预测下一个词。

在我们的Paragraph Vector框架中(见图2), 每个段落（paragraph）都被映射到一个唯一的vector中，表示成矩阵D中的某一列；每个词(word)都映射到一个某一个向量中，表示成矩阵W中的某一列。对paragraph vector和word vector求平均，或者级联(concatenated)起来，以预测在上下文中的下一个词。在该试验中，我们使用级联(concatenation)作为组合向量的方法。

<img src="http://pic.yupoo.com/wangdren23/Gl206Ip6/medish.jpg">

图2: 学习paragraph vector的框架。该框架与word2vec的框架相似；唯一的区别了，会将额外的paragraph token通过矩阵D映射到一个vector中。在该模型中，级联或对该向量求平均，再带上一个三个词的上下文，用来预测第4个词。paragraph vector表示从当前上下文缺失的信息，可以看成是paragraph主题的记忆单元。

更正式的，在模型中与词向量框架的唯一变化是，h是从W和D中构建的。

paragraph的token可以认为是另一个词。它扮演的角色是，作为一个记忆单元，记住当前上下文--或者paragraph的主题。出于该原因，我们经常称该模型为Paragraph Vector分布式记忆模型（PV-DM）。

上下文是固定长度的，从沿paragraph滑动的一个滑动窗口中采样。所有相同paragraph生成的上下文，共享着paragraph vector。不同的paragraphs间，共享着相同的词向量矩阵W，比如，单词"powerful"的向量，对于所有paragraphs是相同的。

paragraph vectors和word vectors都使用SGD进行训练，梯度通过backpropagation算法求得。在SGD的每一步，你可以从一个随机paragraph中抽样一个固定长度的上下文，计算error的梯度，更新模型参数。

在预测阶段，对于一个新的paragraph，需要执行一个推断步骤(inference)来计算paragraph vector。这也可以通过梯度下降法获取。在该步骤时，对于模型其余部分的参数，word vectors W以及softmax weights，是固定的。

假设在语料中有N个段落（paragraph），词汇表中有M个词，我们希望学到paragraph vectors，每个paragraph都被映射到p维上，每个word被映射到q维上，接着该模型具有总共N x p + M x q 个参数（将softmax参数排除在外）。尽管当N很大时，参数的数目会很大，在训练期间的更新通常是稀疏的，并且很有效。

在训练之后，paragraph vectors可以当成是该paragraph的特征(例如：代替bow或作为bow的额外附加)。我们可以将这些features直接输入到常用的机器学习技术（LR, SVM或者K-means）中。

总之，算法本身有两个关键步骤：1) 在training阶段：在已知的paragraphs上，获取词向量W，softmax的权重(U,b)以及paragraph向量D. 2)在inference阶段，保持W,U,b固定不变，通过增加D中的更多列，在D上进行梯度下降，为新的paragraph（未曾见过的）获取paragraph vectors D。我们使用D来做预测关于更多的特定labels。

**paragraph vectors的优点**：paragraph vectors的一个重要优点是，它们可以从未标记的数据（unlabeld data）中学到，在没有足够多带标记的数据（labeled data）上仍工作良好。

Paragraph vectors也抛出了一些BOW模型所具有的核心缺点。首先，它们继承了词向量的一个重要特性：词的语义（sematics）。在该空间中，比起"Paris"， "powerful"与"strong"更接近。Paragraph vector的第二个优点是：它们会考虑词顺序，至少在某个小上下文上，相同方式下，n-gram模型则有一个大的n。另一个重要点，因为n-gram模型保留着一部分paragraph的信息，包括词顺序。也就是说，我们的模型可能优于一个bag-of-n-gram模型，因为一个bag-of-n-gram模型可能创建出一个高维表示，这很难泛化。

## 2.PV-DBOW: (无词序的Paragraph Vector: Distributed BOW)

上面的方法使用的是，在一个文本窗口中，paragraph vector的串联模式，以及词向量来预测下一个词。另一种方法则是忽略掉输入中的上下文词汇，强制模型去预测从paragraph中随机抽样出的词作为输出。在实际上，这意味着，在SGD的每次迭代中，我们可以抽样一个文本窗口，接着从该文本窗口中抽样一个随机词汇，去构建这样一个分类器任务来获取Paragraph Vector。该技术如图3所示。我们将该版本称为：PV-DBOW (Distributed
Bag of Words version of Paragraph Vector)

<img src="http://pic.yupoo.com/wangdren23/Gl3kipIb/medish.jpg">

图3: PV-DBOW.在该版本中，训练该paramgraph vector以预测在一个小窗口中的词汇.

除了概念简单，该模型存储的数据更少。我们只需要存储softmax的权重，而PV-DM则需要存储softmax权得以及词向量。该模型与word2vec中的skip-gram模型相类似。

在我们的试验中，每个paragraph vector是一个两种向量的组合：一个标准PV-DM模型由学到，另一个PV-DBOW模型学到的。对于大多数任务PV-DM单独工作也能达到很好的效果（state-of-art），如果与PV-DBOW组合在一起使用，在许多不同任务上可以更一致，强烈推荐使用组合方式。

## 3.实现

gensim的models.doc2vec实现了该模型。

{% highlight python %}

class gensim.models.doc2vec.Doc2Vec(documents=None, 
	dm_mean=None, 
	dm=1, 
	dbow_words=0, 
	dm_concat=0, 
	dm_tag_count=1, 
	docvecs=None, 
	docvecs_mapfile=None, 
	comment=None, 
	trim_rule=None, 
	**kwargs)

{% endhighlight %}

它的基类是gensim中的: gensim.models.word2vec.Word2Vec

- documents：一个元素为TaggedDocument的list，对于更大的语料可以使用磁盘/网络。如果不提供documents，则模型会未初始化。
- dm: 缺省为1. dm=1,表示使用PV-DM。否则使用PV-DBOW.
- size: 特征向量的维度(基类中)
- window: 要预测的词与上下文间的最大距离，用于文档中的预测
- alpha: 初始的learning-rate（随着训练的进行，会线性降至0）
- seed: 用于随机数字生成器。注意，对于一个完整的确定可再生的运行过程，你必须将该模型限制到单个worker线程上， 以便消除OS线程调度引起的时序抖动。(在python 3中，不同解释器加载之间可再生也需要使用PYTHONHASHSEED环境变量来控制hash随机化)
- min_count: 忽略总频率低于该值的所有词
- max_vocab_size: 在词汇表构建时的最大RAM限制; 如果有许多单个的词超过该值，会对频率低的进行剪枝。每1000w的词类型，需要大概1GB的RAM。缺省设为None，即无限制。
- sample: 配置的阀值，更高频的词会随机下采样(random downsampled)。缺省为0(off), 有用的值为1e-5.
- workers: 使用多个worker线程来训练模型（多核机器更快）
- iter: 在语料上的迭代次数(epoches)。缺省值从Word2Vec继承下来，为5. 但对于'Paragraph Vector'来说，10或20更常用。
- hs: 如果为1, 表示使用hierarchical sampling来进行模型训练，否则为0. 缺省为1
- negative: 如果>0, 会使用negative sampling，int值表示应抽样“noise words”多少次。（通常设在5-20间）
- dm_mean: 如果为0(缺省情况), 会使用上下文的词向量的求和(sum)。如果为1,则使用求平均（mean）。如果dm以非级联(non-concatenative)的模式，才会使用它。
- dm_concat: 如果为1,则使用上下文向量的级联方式(concatenation)，而非(sum/average)方式；缺省为0(off)。注意，级联(concatenation)会导致一个大的多的模型，输入不再是一个词向量（被抽样出或者算术结合）的size，而是使用该tag(s)的size和上下文中的所有词捆在一起。
- dm_tag_count: 当使用dm_concat模式时，每个文档所期望常数个文档标签；缺省为1
- dbow_words: 如果设置为1, 则会训练word-vectors(以skip-gram的方式)，同时训练DBOW的doc-vector；缺省为0(只训练doc-vectors训练会更快）
- trim_rule: 词汇表剪枝规则，指定了特定的词是否应保留在词汇表中，是否被削剪掉，或者使用缺省方式处理（如果词的count<min_count，直接抛弃）. 可以置为None(即使用min_count)，或者使用一个回调，使用参数(word,count,min_count)，返回下述值：util.RULE_DISCARD, util.RULE_KEEP or util.RULE_DEFAULT. 注意：如果给定该规则，会使用它在build_vocab()期间来剪枝词汇表，不会被当成是模型的一部分进行保存。

相应的示例代码，可以参见: 

- [doc2vec-IMDB](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb)  
- [test-doc2vec](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/test/test_doc2vec.py)


# 参考

- [Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)
- [gensim.models.doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html)