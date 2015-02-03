---
layout: page
title: gensim教程
tagline: 总述 
---
{% include JB/setup %}

教程的组织由一系列示例组成，包含一系列gensim的特性.

示例分成以下几部分：

- [语料和矢量空间](http://d0evi1.github.com/gensim/tut1)
    - String转成向量
    - 语料流
    - 语料格式
    - 兼容NumPy和SciPy
- [主题和变换](http://d0evi1.github.com/gensim/tut2)
    - 变换接口
    - 提供的转换
- [相似查询](http://d0evi1.github.com/gensim/tut3)
    - 相似接口
    - next?
- [英文Wikipedia示例](http://d0evi1.github.com/gensim/wiki)
    - 准备语料
    - LSA
    - LDA
- [分布式计算](http://d0evi1.github.com/gensim/distributed)
    - 为什么要分布式计算?
    - 先决条件
    - 核心概念
    - 提供的分布式算法 

# 1.先决条件

所有的示例都可以通过Python解析器shell直接copy运行.  IPython的cpaste命令可以拷贝代码段，包含前导的>>>字符.

gensim使用Python的标准logging模块来记录不同优先级的日志；如果想激活日志，运行：

    >>> import logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 2.快速示例

首先，让我们import gensim包，并且创建一个包含9个文档和12个特征的小语料库:

    >>> from gensim import corpora, models, similarities
    >>>
    >>> corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
    >>>           [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
    >>>           [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
    >>>           [(0, 1.0), (4, 2.0), (7, 1.0)],
    >>>           [(3, 1.0), (5, 1.0), (6, 1.0)],
    >>>           [(9, 1.0)],
    >>>           [(9, 1.0), (10, 1.0)],
    >>>           [(9, 1.0), (10, 1.0), (11, 1.0)],
    >>>           [(8, 1.0), (10, 1.0), (11, 1.0)]]

返回的Corpus是一个对象，用来表示稀疏矩阵. 如果你对矩阵空间模型(VSM)不熟，我们将在下一篇文章[语料和向量空间]()中介绍raw string, corpora以及sparse vectors间的区别.

如果你对VSM很熟，你可能了解，你解析文档并且将它们转换成向量的方式，会对以后的应用的质量有很大影响.

注意：

在本例中，整个语料库会在内存中以python list的方式存储。然后，corpus接口只用来表示，一个语料必须通过构成文档进行迭代。对于很大的corpora，它的优点是可以将语料保存在磁盘上，按序访问文档，一次一个。以这样方式实现的所有的操作和变换，可以让它们按语料的size彼此内存独立.

接下来，让我们初始化一个变换：

    >>> tfidf = models.TfidfModel(corpus)

一个变换，可以将文档从一个向量表示变换成另外的一个:
    
    >>> vec = [(0, 1), (4, 1)]
    >>> print(tfidf[vec])
    [(0, 0.8075244), (4, 0.5898342)]

这里，我们使用tf-idf变换，可以将文档表示成词袋数，并使用权重将常用的单词数降权. 它可以将结果向量放大成单元长度.

详细的变换包含在 [主题和变换]() 一节.

整个语料可以通过tfidf进行变换和索引，相似查询：
    
    >>> index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)

我们将待查询的矢量vec，对在语料库中的每个文档进行相似查询相似.
    
    >>> sims = index[tfidf[vec]]
    >>> print(list(enumerate(sims)))
    [(0, 0.4662244), (1, 0.19139354), (2, 0.24600551), (3, 0.82094586), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0)]

如何读取这个输出？文档号0（第一个文档）具有的相似分为：0.466=46.6%，第二个文档具有相似分:19.1% 等。

这样，根据tfidf文档表示以及余弦相似度计算，我们可以查询vec的最相似的文档是 no.3文档，它的相似度为：82.1%。注意，在tfidf表示中，vec和任何文档不公共的特性，具有的相似分数为0. 更多细节，查看: [相似查询]()


