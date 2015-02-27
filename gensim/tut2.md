---
layout: page
title: 主题和变换 
---
{% include JB/setup %}

同样的，记得打开日志：

    >>> import logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 1.变换接口

在先前的教程中，我们创建了一个文档语料，将它表示成向量流。接下来，我们将使用gensim并使用它的语料：

    >>> from gensim import corpora, models, similarities
    >>> dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
    >>> corpus = corpora.MmCorpus('/tmp/deerwester.mm')
    >>> print(corpus)
    MmCorpus(9 documents, 12 features, 28 non-zero entries)

在本篇教程中，我们将看到，如何将一个向量表示转换成另一个。这个过程有两个目的：

- 1.为了得到语料背向的结构，发现词之间的关系，并使用一种新的或者更加语义化的方式来使用它们描述文档。
- 2.为了让文档表示更紧凑。这个更有效（新的表示法消耗更低资源）且效果好（降低噪声等）

# 2.创建一个变换

变换这个概念其实是个python对象，可以通过一个训练语料初始化：

    >>> tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

我们使用课程1中的老语料库来初始化变换模型。不同的变换需要不同的初始化参数；在tfidf中，"训练"包含了将提供的语料库做一次彻底的扫描，计算所有特征的文档频率。训练其它的模型，比如LSA 或者 LDA，则需要涉及到更多的时间。

注意：

变换总是在两个指定的向量空间之间进行。相似的向量空间（＝特征id的相似集合）必须用来训练向面的向量变换。如果使用相似的输入特征空间失败，比如使用一个不同的字符串预处理，使用不同的特征id，或者使用tfidf向量期待的词袋输入向量，将导致在变量执行时特征不匹配，或者在随后的垃圾回收以及运行时异常。

### 2.1 变换向量

从这一步开始，tfidf被认为是一个只读对象，可以被用来进行任何向量转换，从旧的表示（词袋数统计）转化成新的表示（tfidf实数权重）

    >>> doc_bow = [(0, 1), (1, 1)]
    >>> print(tfidf[doc_bow]) # step 2 -- use the model to transform vectors
    [(0, 0.70710678), (1, 0.70710678)]

或者将这个转换应用到整个语料库：

    >>> corpus_tfidf = tfidf[corpus]
    >>> for doc in corpus_tfidf:
    ...     print(doc)
    [(0, 0.57735026918962573), (1, 0.57735026918962573), (2, 0.57735026918962573)]
    [(0, 0.44424552527467476), (3, 0.44424552527467476), (4, 0.44424552527467476), (5, 0.32448702061385548), (6, 0.44424552527467476), (7, 0.32448702061385548)]
    [(2, 0.5710059809418182), (5, 0.41707573620227772), (7, 0.41707573620227772), (8, 0.5710059809418182)]
    [(1, 0.49182558987264147), (5, 0.71848116070837686), (8, 0.49182558987264147)]
    [(3, 0.62825804686700459), (6, 0.62825804686700459), (7, 0.45889394536615247)]
    [(9, 1.0)]
    [(9, 0.70710678118654746), (10, 0.70710678118654746)]
    [(9, 0.50804290089167492), (10, 0.50804290089167492), (11, 0.69554641952003704)]
    [(4, 0.62825804686700459), (10, 0.45889394536615247), (11, 0.62825804686700459)]

在这个特例中，我们将之前用来训练的相同的语料进行转换，但是。一旦转换模型被初始化后，它可以被用在任何向量上（当然，它们来自相同的向量空间），即使它们在训练语料中根本没有使用到。这可以通过LSA的folding-in，或者通过LDA的topic-inference来处理。

注意：

调用model[corpus]只能在旧的文档流中创建一个wrapper－－实际的转换是即时的，在迭代每个文档时完成。当调用 corpus_transfromed = model[corpus] 时，我们不能将整个语料库进行转换，因为它将结果存储到内存中，而这与gensim的内存独立对象相矛盾。如果你对corpus_transformed对象迭代多次，那化变换的代价很高，将结果语料序列化到磁盘，并继续使用它。

变换可以被序列化，一个接一个，以链的方式进行：

    >>> lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
    >>> corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

这里我们tfidf语料通过LSI (Latent Sematic Indexing)变换成一个隐含语义的2D空间(2D，通过设置num_topics=2来完成)。现在，你可以想像，两个隐含维度分别表示什么？让我们看下 models.LsiModel.print_topics(): 

    >>> lsi.print_topics(2)
    topic #0(1.594): -0.703*"trees" + -0.538*"graph" + -0.402*"minors" + -0.187*"survey" + -0.061*"system" + -0.060*"response" + -0.060*"time" + -0.058*"user" + -0.049*"computer" + -0.035*"interface"
    topic #1(1.476): -0.460*"system" + -0.373*"user" + -0.332*"eps" + -0.328*"interface" + -0.320*"response" + -0.320*"time" + -0.293*"computer" + -0.280*"human" + -0.171*"survey" + 0.161*"trees"

（主题通过日志打印出来－－可以通过日志进行查看）

根据LSI, "trees", "graph", "minors"都是相关的词汇（在第一个主题上。。），而第二个主题则更关注其它词汇。正如我们所料，前五个文档与主题2更相关，而其余的则与主题1更相关：

    >>> for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    ...     print(doc)
    [(0, -0.066), (1, 0.520)] # "Human machine interface for lab abc computer applications"
    [(0, -0.197), (1, 0.761)] # "A survey of user opinion of computer system response time"
    [(0, -0.090), (1, 0.724)] # "The EPS user interface management system"
    [(0, -0.076), (1, 0.632)] # "System and human system engineering testing of EPS"
    [(0, -0.102), (1, 0.574)] # "Relation of user perceived response time to error measurement"
    [(0, -0.703), (1, -0.161)] # "The generation of random binary unordered trees"
    [(0, -0.877), (1, -0.168)] # "The intersection graph of paths in trees"
    [(0, -0.910), (1, -0.141)] # "Graph minors IV Widths of trees and well quasi ordering"
    [(0, -0.617), (1, 0.054)] # "Graph minors A survey"

模型的持久化通过save()和load()函数来完成：

    >>> lsi.save('/tmp/model.lsi') # same for tfidf, lda, ...
    >>> lsi = models.LsiModel.load('/tmp/model.lsi')

接下来的问题是：文档间的相似程度有多精确？是否有方法来计算相似度，比如给定一个输入文档，我们可以根据相似度获取其它相关的文档集合？ 相似查询将在[下一节]()中讲到。

## 2.2 可靠的变换

gensim实现了许多流行的VSM算法：

- TF-IDF：在初始化训练语料库时，将它们认为是词袋（整型值）。在变换期间，输入一个向量，将返回另一个相同维度的向量，在训练语料中很少的某些特征，它们的值将会被放大。因而，它会将整型值向量转换成实数型的向量，并且保留完整的维度。它可以将结果向量归一化成（欧氏）单位长度。
    
    

    >>> model = tfidfmodel.TfidfModel(bow_corpus, normalize=True)
    
- LSI(或LSA): 将词袋或者基于tfidf权重的空间转换成一个更低维的隐含语义空间。对于上面的示例，我们使用了2个隐含维度，但是在实数语料上，推荐使用“黄金标准”：200-500的目标维度。
    
    >>> model = lsimodel.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)
    
LSI训练是唯一的，因为我们可以在任何时候继续训练，只需提供更多的训练文档。它可以通过底层模型来完成增量更新，这个过程称为“在线训练”。因为这个特性，输入文档流可以是无限大－－只需要喂给LSI新文档，再同时使用计算变换模型即可。

    >>> model.add_documents(another_tfidf_corpus) # now LSI has been trained on tfidf_corpus + another_tfidf_corpus
    >>> lsi_vec = model[tfidf_vec] # convert some new document into the LSI space, without affecting the model
    >>> ...
    >>> model.add_documents(more_documents) # tfidf_corpus + another_tfidf_corpus + more_documents
    >>> lsi_vec = model[tfidf_vec]
    >>> ...

更多关于"如何在无限流中使用LSI"，查看 gensim.models.lsimodel 文档。如果你希望获取更多，你可以通过调参来影响LSI算法的速度 vs. 内存 vs. 精确度.

gensim使用一个小说在线增量流分布式训练算法，可以在这查看[[5]](http://radimrehurek.com/gensim/tut2.html#id10)。gensim也执行一个stochastic multi-pass algorithm from Halko et al. ...

- RP：目换是为了减少向量空间维度。 这是个非常有效的方法（内存和CPU友好），逼近文档之里的tfidf距离，通过使用一个小的随机数。推荐目标维度在成千上万，依赖于数据集。

    >>> model = rpmodel.RpModel(tfidf_corpus, num_topics=500)

- LDA: 另一个可以将词袋数转换到主题低维空间的变换。LDA是LSA（也称为PCA）的扩展。因此，LDA的主题可以被解释成词的概率分布。类似于LSA，这个分布自动从自动从一个训练语料中进行infer。文档被解析成一个 主题的softmix混合（同样类似于LSA）。

    >>> model = ldamodel.LdaModel(bow_corpus, id2word=dictionary, num_topics=100)

gensim使用在线LDA参数估计的一个快速实现版本，可以修改运行在分布式模式的计算机集群上。

- HDP：一种非参数贝叶斯方法（注意缺少请求的主题数）：

    >>> model = hdpmodel.HdpModel(bow_corpus, id2word=dictionary)

gensim使用一个快速的在线实现版本[[3]](http://radimrehurek.com/gensim/tut2.html#id8). HDP模型是一个gensim的新添加特性，在学术界仍在探索。

添加新的VSM变换（比如：不同的权重模式）是相当琐碎的；可以查看API等。

需要重复的是，所有唯一的，增量的实现，不必在主内存中完整训练整个语料。我们正在改进[分布式计算](http://d0evi1.github.io/gensim/distributed)，来改进cpu性能。如果你想贡献（测试、用例、或编码），可以联系我们。

下一节：[相似查询](http://d0evi1.github.io/gensim/tut3).

[英文原版](http://radimrehurek.com/gensim/tut2.html)
