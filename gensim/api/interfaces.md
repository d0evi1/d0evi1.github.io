---
layout: page
title: 接口 － 核心gensim接口 
---
{% include JB/setup %}

该模块包含了整个gensim包中常用的基本接口。

该接口都是一些抽象基类（例如：提供了一些选项功能，因此该接口可以被子类继承）

---------------------------------------------------------

class gensim.interfaces.CorpusABC

    Base: gensim.utils.SaveLoad

    语料接口（抽象基类）。一个语料可以是一个简单的迭代对象，每次迭代都yield一个文档：

    >>> for doc in corpus:
    >>>     # do something with the doc...

    一个文档可以是一个顺序的(fieldId, fieldValue)的2-tuples:

    >>> for attr_id, attr_value in doc:
    >>>     # do something with the attribute

    注意：尽管提供了一个缺省的len()方法，它非常低效（会执行一个线性扫描来判断长度）。 语料的size是必须的，事先需要知道（或者至少不要做变更，以便可以cache），len()方法可以被覆盖。

    参见 gensim.corpora.svmlightcorpus 模块来见一个示例。

    可以使用save方法保存corpus（通过继承 utils.SaveLoad），可以只存储在内存对象（二进制，pickled方式），流状态，并且不是文档自身。参见静态函数 save_corpus来序列化实际的流内容。

classmethod load(fname, mmap=None)

    加载之前可存的文件对象。

    同上。

-----------------------------------------------------------------------

save(*args, **kwargs)

static save_corpus(fname, corpus, id2word=None, metadata=False)

    保存一个存在的corpus语料到磁盘上。

    也支持其它一些格式来保存字典（feature_id -> word  映射），通过可选参数 id2word 来完成。

    >>> MmCorpus.save_corpus('file.mm', corpus)

    一些语料类也支持每个文档的索引，因而磁盘上的文档可以以O(1)复杂度被访问（参见：corpora.IndexedCorpus基类）。这种情况下，save_corpus 通过 serialize函数 内部自动被调用，save_corpus将在同时保存索引，因此，你可以像这样来保存语料：

    >>> MmCorpus.serialize('file.mm', corpus) # stores index as well, allowing random access to individual documents

    调用serialize()函数比调用save_corpus()要好.

-------------------------------------------------------------

class gensim.interfaces.SimilarityABC(corpus)

    基类：gensim.utils.SaveLoad

    通过语料，用于相似查询的抽象接口。

    在所有的实例中，查询相似的文档。

    对于每个相似查询中，输入是一个文档，输出则是它与整个语料库的相似度。

    相似度查询的实现通过调用 self[query_document]来完成。

    这是一个很方便的wrapper类，整个语料中的每个文档的相似度，通过self的yield进行迭代。（例如：查询整个文档语料）

-------------------------------------------------------------

get_similarities(doc)

classmethod load(fname, mmap=None)

    加载一个之前保存的文件对象。

    同上。

-------------------------------------------------------------

save(fname, separately=None, sep_limit=10485760, ignore=frozenset([]))

    保存文件对象.

    同上。

-------------------------------------------------------------

class gensim.interfaces.TransformationABC

    基类：gensim.utils.SaveLoad

    变换接口。一个“变换”指的是，任何对象它接受一个稀疏文档，通过字典[]符，返回另一个稀疏矩阵：

    >>> transformed_doc = transformation[doc]

    或者：

    >>> transformed_corpus = transformation[corpus]

    参见：gensim.models.tfidfmodel 模块的变换示例。

classmethod load(fname, mmap=None)

    加载之前保存的文件对象。

    。。。

------------------------------------------------------------

save(fname, separately=None, sep_limit=10485760, ignore=frozenset([]))

    保存文件对象.

    同上。

------------------------------------------------------------

class gensim.interfaces.TransformedCorpus(obj, corpus, chunksize=None)

    基类： gensim.interfaces.CorpusABC

    classmethod load(fname, mmap=None)

    加载之前的文件对象。

    同上.

------------------------------------------------------------

save(*args, **kwargs)

static save_corpus(fname, corpus, id2word=None, metadata=False)

    保存一个语料库到磁盘中。

    也支持保存字典(feature_id -> word 的映射)，通过可选参数id2word来提供。

>>> MmCorpus.save_corpus('file.mm', corpus)


    一些语料也支持索引来表示每个文档的起点，因此文档磁盘可以以O(1)时间复杂度进行访问（参见：corpora.IndexedCorpus基类）。在这个示例中，可以通过serialize函数内部自动调用save_corpus，它在save_corpus之外，还会同时保存索引，因此，你可以这样存储：

    >>> MmCorpus.serialize('file.mm', corpus) # stores index as well, allowing random access to individual documents

    优先调用serialize()函数.


[英文原版](http://radimrehurek.com/gensim/interfaces.html)
