---
layout: page
title: models.lsimodel - 隐含语义分析 
---
{% include JB/setup %}

python 隐含语义分析LSA（或者LSI）模块。

通过奇异值分解（SVD）的方式进行实现。SVD可以在任何时间添加新值进行更新（实时、增量、高效利用内存的训练）

该模块实际上包含了许多算法，用来分解大语料，允许构建LSI模块：

- 语料比RAM大：对于语料的size来说 (尽管依赖特征集的size)，只需要常量的内存空间
- 语料是流式的：文档只能按顺序访问，非随机访问
- 语料不能被临时存储：每个文档只能被查看一次，必须被立即处理（one-pass算法）
- 对于非常大的语料，可以通过使用集群进行分布式运算

English Wikipedia的性能测试（2G的语料position，3.2M文档，100k特征，在最终的TF-IDF矩阵中0.5G的非零实体），请求 top 400 LSI因子：

<table border="1">
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
        <td>N/A[注]</td>
    </tr>
</table>

注：

serial = Core 2 Duo MacBook Pro 2.53Ghz, 4GB RAM, libVec

distributed = cluster of four logical nodes on three physical machines, each with dual core Xeon 2.0GHz, 4GB RAM, ATLAS

[注]stochastic算法可以设成分布式，此由于数据分布在集群节点上，时大量的时间花费在读取/解压输入磁盘文件、额外的网络传输，所以看起来变慢了。

-------------------------------------------------------------

class gensim.models.lsimodel.LsiModel(corpus=None, num_topics=200, id2word=None, chunksize=20000, decay=1.0, distributed=False, onepass=True, power_iters=2, extra_samples=100)

    Bases:  [gensim.interfaces.TransformationABC](http://radimrehurek.com/gensim/interfaces.html#gensim.interfaces.TransformationABC)

    该类允许构建和维护一个LSI的模型。

    主要的方法有：

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

   更新奇异值分解，重视新文档语料。

   一次训练只处理chunksize大于的文档chunk。chunksize的大小，需要在增速（更大的chunksize） vs 低内存footprint（更小的chunksize）之间权衡。如果分布式模式是打开的，每个chunk都被发送到一个不同的worker/computer。

   若设置decay < 1.0，会造成在输入文档流中偏向新数据，通过给定少量的emphasis给老数据。这将允许LSA逐渐“忘记”老的文档，并将更多偏向给新数据。

classmethod load(fname, *args, **kwargs)

    加载之前保存的对象文件。

    可以被还原成mmap大数组.

----------------------------------------------------------

print_debug(num_topics=5, num_words=10)

    打印日志给num_topics个主题以最重要的信息。

    不同于print_topics()，对于一个指定的主题来说，words信息很重要。这可以产生一个可读性良好的主题描述。

----------------------------------------------------------

print_topic(topicno, topn=10)

    返回对应主题号单个主题字符串。可以参见show_topic()的参数。

见：

>>> lsimodel.print_topic(10, topn=5)
'-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + -0.174 * "functor" + -0.168 * "operator"'

----------------------------------------------------------

print_topics(num_topics=5, num_words=10)

    show_topics()的别名函数，可以打印top5主题的日志。

----------------------------------------------------------

save(fname, *args, **kwargs)

    将model保存成文件。

    大的内部array将被保存成独立的文件，使用fname作为前缀.

-----------------------------------------------------------

show_topic(topicno, topn=10)

    返回一个指定的主题（=左奇异矩阵）字符串， 0 <= topicno < self.num_topics。

    只返回topn的话语，它会最接近该主题的方向（正或者负）

    >>> lsimodel.show_topic(10, topn=5)
    [(-0.340, "category"), (0.298, "$M$"), (0.183, "algebra"), (-0.174, "functor"), (-0.168, "operator")]


-----------------------------------------------------------

show_topics(num_topics=-1, num_words=10, log=False, formatted=True)

    返回num_topics 的最大主题（缺省返回所有）。对于每个主题，可以展示给num_words最重要的word信息（缺省10个word）

    如果formatted=True,该主题将返回一个列表；若为False，则返回一个(weight,word)的2-tuples的list。

    如果log为True，也会输出该结果到日志上。

-----------------------------------------------------------

class gensim.models.lsimodel.Projection(m, k, docs=None, use_svdlibc=False, power_iters=2, extra_dims=100)

    Bases:  gensim.utils.SaveLoad

    从一个corpus文档中构造一个(U,S)的projection。该projection可以接着通过self.merge()将另一个Projection进行merge时进行更新动作。

    该类负责“核心数学计算”，通过高层的LsiModel类操作，可以处理包含处理corpora的接口，将大的corpora分割成chunks，或将它们进行merge等。

-----------------------------------------------------------

empty_like()

classmethod load(fname, mmap=None)

    加载之前保存的一个文件对象。

-----------------------------------------------------------

merge(other, decay=1.0)

    merge另一个projection。

    other: 它的内容会在处理过程中销毁，如果你在后面还需要用到它，可以考虑传给该函数一个other的copy。

-----------------------------------------------------------

save(fname, separately=None, sep_limit=10485760, ignore=frozenset([]))

    保存文件对象.

    同其它。

-----------------------------------------------------------

gensim.models.lsimodel.ascarray(a, name='')

-----------------------------------------------------------

gensim.models.lsimodel.asfarray(a, name='')

-----------------------------------------------------------

gensim.models.lsimodel.clip_spectrum(s, k, discard=0.001)

    给定本征值s，它将返回：需要保存的因子个数，从而避免假值（tiny, numerically instable）。

    它将忽略spectrum的尾部，relative combined mass < min(discard, 1/k).

    返回值被k截断（=不会返回多于k个值）

-----------------------------------------------------------

gensim.models.lsimodel.print_debug(id2token, u, s, topics, num_words=10, num_neg=None)

-----------------------------------------------------------

gensim.models.lsimodel.stochastic_svd(corpus, rank, num_terms, chunksize=20000, extra_dims=None, power_iters=0, dtype=<type 'numpy.float64'>, eps=1e-06)

    在一个sparse输入上，运行SVD。

    (U,S): 返回值。输入数据流语料上的左SV和奇异值。语料本身可能比RAM大（通过vector迭代）。

    这将返回比请求的top rank因子还要少，因为输入本身的rank就很低。extra_dims（过采样）和指定的power_iters(power次迭代)参数将影响分解。

    该算法使用2+power_iters传给输入数据。这种情况下，你只能承担一个single的输入，在LsiModel中将onepass=True，并且直接避免使用该函数。

    该分解算法基于：Halko, Martinsson, Tropp. Finding structure with randomness, 2009

    如果corpus使用的是一个scipy.sparse矩阵，那么假设整个corpus满足主内存大小，并选择一个不同的代码路径（效率更高）.
