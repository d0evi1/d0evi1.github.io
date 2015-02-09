---
layout: page
title: similarities.docsim 文档相似查询 
---
{% include JB/setup %}

该模块包含了在向量空间模型（VSM）集合中进行相似计算的查询函数和类。

主类为Similarity，它会在一个给定的文档集合上创建索引。一旦索引被创建好，你就可以执行类似这样的快速查询：“Tell me how similar is this query document to each document in the index?”. 结果返回是一个数组，其大小与初始化文档集一样大，也就是说，每个索引文档都有一个float值。可选的，你可以为该查询只请求top-N个最近似的索引文档。

你可以通过Similarity.add_documents() 来添加新的文档到索引中。

# 1. 工作机制？

Similarity将索引(index)划分成许多更细粒度的子索引(sub-indexs)，称为"shards"，它们是基于磁盘存储的。如果你的整个索引空间内存刚好容得下（比如 成千上万的文档需1GB RAM），那么你可以直接使用MatrixSimilarity或者SparseMatrixSimilarity。它们很简单，但是扩展性很差（因为整个索引会保存在RAM中）

一旦索引被初始化，你就可以查询文档的相似度了：

    >>> index = Similarity('/tmp/tst', corpus, num_features=12) # build the index
    >>> similarities = index[query] # get similarities between the query and all index documents

如果你有更多的查询文档，你可以一次批处理全提交：

    >>> for similarities in index[batch_of_documents]: # the batch is simply an iterable of documents (=gensim corpus)
    >>>     ...

这种批处理的查询方式，（称为：chunked）其好处是：性能更高。如果想在你的机器上获得加速，你可以运行：python -m gensim.test.simspeed 。

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

    返回值为：pos位置的索引vector。

    返回的这个vector与底层索引是相同的类型（比如：MatrixSimilarity的dense，或者SparseMatrixSimilarity的scipy.sparse）


classmethod load(fname, mmap=None)
    
    加载一个前之保存的文件对象（参见 save）。

    如果对象被独自保存成大的数组，你可以通过mmap的方式(mmap='r')进行加载。缺省不使用mmap。

--------------------------------------------------------------------

save(fname, separately=None, sep_limit=10485760, ignore=frozenset([]))

    将对象保存成文件（参见 load）。

    ... 同上.

---------------------------------------------------------------------

class gensim.similarities.docsim.Similarity(output_prefix, corpus, num_features, num_best=None, chunksize=256, shardsize=32768)

    在一个静态文档语料库中，计算一个动态查询的余弦相似度。

    通过将索引共享成更小的子索引（shared），每个子索引可以放在合适的内存中（参见SparseMatrixSimilarity类），这种方法有效地提高的可扩展性。shared可以存储在磁盘文件上，并通过mmap按需读回。

    corpus: 从该corpus构建索引。索引通过add_documents方法进行扩展。注意：为了更快的进行BLAS有麻烦俄天，通过转换成一个矩阵，文档（内部的，透明的）被划分成共享文档shardsize的多份shard。每个shard以这样的形式在磁盘上进行存储：output_prefix.shard_number（你需要有写权限）。如果你没有指定一个输出前缀，将会使用一个随机的文件名。

    shardsize: 应注意：因为一个大小为(shardsize x chunksize)的矩阵，刚好对于主内存合适。

    num_features: 为语料库的特征数（比如：字典大小，或者lsi的主题数等）

    num_best: 未定。


    >>> index = Similarity('/path/to/index', corpus, num_features=400) # if corpus has 7 documents...
    >>> index[query] # ... then result will have 7 floats
    [0.0, 0.0, 0.2, 0.13, 0.8, 0.0, 0.1]


    如果指定了num_best，只返回num_best个最相似的查询，其余的文档的相似度接近为0。如果输入vector本身有0值特征（=），返回的list也将为空。

    >>> index.num_best = 3
    >>> index[query] # return at most "num_best" of `(index_of_document, similarity)` tuples
    [(4, 0.8), (2, 0.13), (3, 0.13)]


    你可以通过num_best动态改写，在查询前，通过设置 self.num_best = 10 即可做到。

----------------------------------------------------------------

add_documents(corpus)

    使用新的文档来扩展index。

    在内部实现中，documents是buffered是，接着以self.shardsize的大小分割到磁盘。

----------------------------------------------------------------

check_moved()

    更新共享内存位置，服务器目录将移到文件系统上.

----------------------------------------------------------------

close_shard()

    强迫关闭最新的shard （变换成一个matrix，并存入磁盘）。自上一次调用后，如果没有增加新的文档，将不会做任何事情。

    注意：shard会关闭，即使它还没有完全满（它的size比self.shardsize小）。如果文档通过add_documents()被添加，那么不完整的shard将被再次载入。

-----------------------------------------------------------------

destory()

    删除self.output_prefix下的所有文件。在调用该方法后，任何对象都不再可用，需要注意！

------------------------------------------------------------------

iter_chunks(chunksize=None)

    迭代中将文档的chunk作为索引yield返回，每个size<=chunksize。

    chunk会以原始的形式返回（矩阵 或者 稀疏矩阵的切片slice）。chunk的size可能小于请求；这完全取决于真实长度的结果确认，使用chunk.shape[0]。


classmethod load(fname, mmap=None)

    从文件加载之前保存的对象。

    如果对象通过大数组方式独立保存，可以通过mmap来加载这些数组.

-----------------------------------------------------------------

query_shards(query)

    对于self.shards中的每个shard，作为一个序列，返回使用shard[query]的结果。

    如果设置了PARALLEL_SHARDS，shard会被并行的查询，使用多处理器模块。

-----------------------------------------------------------------

save(fname=None, *args, **kwargs)

    将过picking将对象保存为在构造函数中指定的文件名的文件。

-----------------------------------------------------------------

similarity_by_id(docpos)

    返回给定文档的相似度。docpos：要查询文档在index所在的位置。

-----------------------------------------------------------------

vector_by_id(docpos)

    在指定docpos位置处理返回所索引vector。

-----------------------------------------------------------------

class gensim.similarities.docsim.SparseMatrixSimilarity(corpus, num_features=None, num_terms=None, num_docs=None, num_nnz=None, num_best=None, chunksize=500, dtype=<type 'numpy.float32'>)

    在内存中保存的sparse index矩阵，计算相似度。两vector间的相似度计算使用余弦相似度计算。

    如果你的输入语料中包含稀疏矩阵（比如：词袋形式的文档），并且符合内存大小，可以考虑使用它。

    该矩阵在内部使用scipy.sparse.csr矩阵。除非 整个矩阵满足主内存的大小，使用Similarity作为替代。

-----------------------------------------------------------------

get_similarities(query)

    返回值：返回一个numpy数组，稀疏矩阵 query 及语料中所有文档的相似度。

    如果query是一个文档集合，返回一个2D数组，query中的每个文档与语料中所有文档的相似度（＝批查询，比一个一个查询更快）。

    不要直接使用这个函数，使用self[query]语法作为替代。

classmethod load(fname, mmap=None)

    加载之前保存的文件对象.

    同上. 

-----------------------------------------------------------------

save(fname, separately=None, sep_limit=10485760, ignore=frozenset([]))

    将对象保存成文件.

    同上. 






