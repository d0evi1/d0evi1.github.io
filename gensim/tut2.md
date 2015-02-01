---
layout: page
title: 主题和变换 
---
{% include JB/setup %}

同样的，记得打开日志：

    >>> import logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#变换接口

在先前的教程中，我们创建了一个文档语料，将它表示成向量流。接下来，我们将使用gensim并使用它的语料：

    >>> from gensim import corpora, models, similarities
    >>> dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
    >>> corpus = corpora.MmCorpus('/tmp/deerwester.mm')
    >>> print(corpus)
    MmCorpus(9 documents, 12 features, 28 non-zero entries)

在本篇教程中，我们将看到，如何将一个向量表示转换成另一个。这个过程有两个目的：

- 1.为了得到语料背向的结构，发现词之间的关系，并使用一种新的或者更加语义化的方式来使用它们描述文档。
- 2.为了让文档表示更紧凑。这个更有效（新的表示法消耗更低资源）且效果好（降低噪声等）

## 创建一个变换

变换这个概念其实是个python对象，可以通过一个训练语料初始化：

    >>> tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

我们使用课程1中的老语料库来初始化变换模型。不同的变换需要不同的初始化参数；在tfidf中，"训练"包含了将提供的语料库做一次彻底的扫描，计算所有特征的文档频率。训练其它的模型，比如LSA 或者 LDA，则需要涉及到更多的时间。

注意：

变换总是在两个指定的向量空间之间进行。相似的向量空间（＝特征id的相似集合）必须用来训练向面的向量变换。如果使用相似的输入特征空间失败，比如使用一个不同的字符串预处理，使用不同的特征id，或者使用tfidf向量期待的词袋输入向量，将导致在变量执行时特征不匹配，或者在随后的垃圾回收以及运行时异常。

###变换向量



