---
layout: page
title: similarities.docsim 文档相似查询 
---
{% include JB/setup %}

该模块包含了在VSM集合中进行相似计算的查询函数和类。

主类为Similarity，它将从一个给定的文档集合上创建索引。一旦索引被创建好，你就可以执行类似这样的快速查询：“Tell me how similar is this query document to each document in the index?”. 返回结果是一个数组，大小与初始化文档集一样大，也就是说，每个索引文档都有一个float值。可选的，你可以为该查询只请求top-N个最近似的索引文档。

你可以通过Similarity.add_documents() 来添加新的文档到索引中。

# 1. 工作机制？

Similarity将索引(index)划分成许多更细粒度的子索引(sub-indexs)，称为"shards"，它们是基于磁盘存储的。如果你的整个索引满足内存（比如 成千上万的文档需1GB RAM），你也可以直接使用MatrixSimilarity或者SparseMatrixSimilarity。它们很简单，但是扩展性很差（整个索引会保存在RAM中）

一旦索引被初始化，你就可以查询文档的相似度了：

    >>> index = Similarity('/tmp/tst', corpus, num_features=12) # build the index
    >>> similarities = index[query] # get similarities between the query and all index documents

如果你有更多的查询文档，你可以一次批处理全提交：

    >>> for similarities in index[batch_of_documents]: # the batch is simply an iterable of documents (=gensim corpus)
    >>>     ...

该批处理查询（称为：chunked）的好处是：性能更高。如果想在你的机器上获得加速，你可以运行：python -m gensim.test.simspeed 。

当你需要比较索引与索引之间的相似文档时（比如：查询=索引文档），可以有一个特别的语法。这种特别的语法在内部使用更快的批查询：

    >>> for similarities in index: # return all similarities of the 1st index document, then 2nd...
    >>>     ...

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

class gensim.similarities.docsim.MatrixSimilarity(corpus, num_best=None, dtype=<type 'numpy.float32'>, num_features=None, chunksize=256)

  通过存储索引在内存中，来计算语料库文档中的相似度。这个相似度计算通过两个向量之间的余弦相似度（cosine）来计算。

  如果你的输入语料包含了dense vector(比如在LSI空间上的文档)，并且符合内存空间大小，可以使用它。

  这个矩阵内部以dense numpy array的形式进行存储。必须满足整个矩阵都符合主内存，否则可以使用Similarity进行替代。

  可以参考该模块中的Similarity和SparseMatrixSimilarity。

  num_features 是语料中的特征总数（可以通过扫描语料库来自动完成）。可以参见Similarity类的其它描述。

----------------------------------------------------------------

get_similarities(query)

  作为一个numpy array返回在语料库中的所有文档的sparse vector query的相似度。

  如果query是一个文档集，返回一个2D数组，query中的每个文档都在语料的所有文档中（批查询，每个文档更快处理）都有一个相似度。

  **不要直接使用该函数，可以使用self[query]语法作为替代**

classmethod load(fname, mmap=None)

  加载一个之间保存的文件对象 (可以参见 save)。

  如果对象被保存成独立的大数组，通过设置mmap='r'，你就可以通过mmap（共享内存）方式加载它们。缺省不使用mmap，加载大对象时会当成普通对象处理。

save(fname, separately=None, sep_limit=10485760, ignore=frozenset([]))

  将对象保存成文件 (可以参见 load)。

  如果separatedly是None，将自动检测对象中保存的numpy/scipy的大稀疏矩阵，并将它们以独立的方式进行保存。这可以避免很多内存问题，并且允许将大数组通过mmap进行有效加载。

  你可以人工设置separately，它必须是以不同的文件存储的一列属性名。这种情况下将处动执行check。

  ignore是一个非序列化（比如：文件句柄，cache等）的属性列名。随后的load()这些属性将被设置成None。

------------------------------------------------------------------

class gensim.similarities.docsim.Shard(fname, index)

  一个proxy类，它使用一个Similarity索引来表示单个共享实例。

  基本上，它封装了(Sparse)MatrixSimilarity，因而，它可以从磁盘上进行mmap，在相请的请求查询。

------------------------------------------------------------------

get_document_id(pos)

  在合适的pos位置返回索引vector。

  vector与底层的索引是相似类型（）
