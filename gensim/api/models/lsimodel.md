---
layout: page
title: models.lsimodel - 隐含语义分析 
---
{% include JB/setup %}

python 隐含语义分析LSA（或者LSI）模块。

通过奇异值分解（SVD）的方式进行实现。SVD可以在任何时间添加新值进行更新（实时，增量，有效内丰训练）

该模块实际上包含了许多算法，用来分解大语料，允许构建LSI模块：

- 语料比RAM大：只需要常量内存，对于语料的size来说 (尽管依赖特征集的size)
- 语料是流化的：文档只能按顺序访问，非随机访问
- 语料不能被暂时存储：每个文档只能被查看一次，必须被立即处理（one-pass算法）
- 对于非常大的语料，可以通过使用集群进行分布式运算

English Wikipedia的性能（2G的语料position，在最终的TF-IDF矩阵中，3.2M文档，100k特征，0.5G的非零实体），请求 top 400 LSI因素：

<table>
    <tr>
        <th>算法</th>
        <th>单机串行</th>
        <th>分布式</th>
    </tr>
    <tr>
        <td>one-pass merge算法</td>
        <td>5h 14m</td>
        <td>1h 41m</td>
    </tr>

    <tr>
        <td>multi-pass stochastic算法（2阶迭代）</td>
        <td>5h 39m</td>
        <td>N/A</td>
    </tr>
</table>


serial = Core 2 Duo MacBook Pro 2.53Ghz, 4GB RAM, libVec

distributed = cluster of four logical nodes on three physical machines, each with dual core Xeon 2.0GHz, 4GB RAM, ATLAS

stochastic算法可以是分布式的，但是花费大多数时间在读取/解压输入磁盘文件上。额外的网络传输是因为数据分布在集群节点上，所以看起来变慢了。

-------------------------------------------------------------

class gensim.models.lsimodel.LsiModel(corpus=None, num_topics=200, id2word=None, chunksize=20000, decay=1.0, distributed=False, onepass=True, power_iters=2, extra_samples=100)

    Bases:  gensim.interfaces.TransformationABC

    该类允许构建和维护一个LSI的模型。

    main方法有：

    － 1.构造函数：用来初始化潜语义空间
    － 2.[]方法：它返回任何输入文件在潜语义空间的表示
    － 3.add_documents(): 使用新的文档进行增量模型更新

左奇异矩阵被存储在lsi.projection.u中，奇异值在lsi.projection.s。右奇异矩阵可以在lsi[training_corpus]的输出中被重新构造。

模型持久化可以通过它的load/save方法来完成.

[https://github.com/piskvorky/gensim/wiki/Recipes-&-FAQ#q4-how-do-you-output-the-u-s-vt-matrices-of-lsi](https://github.com/piskvorky/gensim/wiki/Recipes-&-FAQ#q4-how-do-you-output-the-u-s-vt-matrices-of-lsi)

num_topics: 请求因子的数目 (潜维度)

在模型被训练后，你可以估计主题，使用 topics = self[document] 字典。你也可以使用self.add_documents 来添加新的训练文档，因此，训练可以在任何时间被stop和resume，LSI变换可以在任何时候进行。

如果你指定了一个corpus，它可以被用来训练模型。方法 add_documents 以及相应的参数chunksize和decay描述。

关闭onepass，强制使用multi-pass stochastic算法。

power_iters和extra_samples影响了stochastic multi-pass算法的精准度，它可以在内部使用(onepass=True)， 或者前端算法(onepass=False)。增加迭代数可以改次精确度，但是性能会降低。

打开distributed可以使用分布式计算。

示例：

>>> lsi = LsiModel(corpus, num_topics=10)
>>> print(lsi[doc_tfidf]) # project some document into LSI space
>>> lsi.add_documents(corpus2) # update LSI on additional documents
>>> print(lsi[doc_tfidf])



[3] [http://nlp.fi.muni.cz/~xrehurek/nips/rehurek_nips.pdf](http://nlp.fi.muni.cz/~xrehurek/nips/rehurek_nips.pdf)

----------------------------------------------------------

add_documents(corpus, chunksize=None, decay=None)

   更新奇异值分解，来说明新文档语料。

   一次训练只处理chunksize大于的文档chunk。chunksize的大小，需要在增速（更大的chunksize） vs 低内存footprint（更小的chunksize）之间权衡。如果分布式模式是打开的，每个chunk都被发送到一个不同的worker/computer。

   若设置decay < 1.0，因为.

