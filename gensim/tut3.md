---
layout: page
title: 相似查询 
---

同样的，如查希望查看日志，请打开：

    >>> import logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#相似接口

在关于<语料和向量空间>以及<主题和变换>的教程中，我们提到过，在向量空间模型中创建一个语料，以及如何在向量空间做变换。一个公共的原因是，我们希望在文档间发现它们的相似度，或者一个指定文档与其它文档间的相似度（比如：一个用户查询 vs. 文档索引）

为了展示gensim是如何来完成的，我们考虑下前面示例中提到的相同的语料(它来自于Deerwester et al.’s “Indexing by Latent Semantic Analysis” seminal 1990 article)：

    >>> from gensim import corpora, models, similarities
    >>> dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
    >>> corpus = corpora.MmCorpus('/tmp/deerwester.mm') # comes from the first tutorial, "From strings to vectors"
    >>> print(corpus)
    MmCorpus(9 documents, 12 features, 28 non-zero entries)

为了按照Deerwester的示例，我们首先使用小语料库来定义一个2维的LSI空间：

    >>> lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

现在，假设用户输入查询：“Human computer interaction”。 我们希望将我们的9个语料库，以查询的相关程度进行排序。不像现代搜索引擎那样，这里，我们只关心单方面可能的相似度－－文本语义相关（根据词）。没有链接，没有随机漫步静态rank（random-walk static ranks），只是通过boolean型的关键词匹配做的一个语义扩展：

    >>> doc = "Human computer interaction"
    >>> vec_bow = dictionary.doc2bow(doc.lower().split())
    >>> vec_lsi = lsi[vec_bow] # convert the query to LSI space
    >>> print(vec_lsi)
    [(0, -0.461821), (1, 0.070028)]

另外，我们将考虑 余弦相似度 来决定两个向量之间的相似程度。余弦相似度是一个在向量空间模型中的标准计算方法，但是向量表示了概率分布，不同的相似度计算可能更合适。

### 初始化查询结构

为了准备相似查询，我们需要输入所有我们希望与查询相比较的文档。在我们的case中，使用与LSI训练相同的9个文档语料，将它转换成2维 LSA空间。但是这只是偶然发生的，我们必须通过一个不同的语料被索引到一块去。

    >>> index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it

警告：

    当整个向量集合与内存合适时，类similarities.MatrixSimilarity 是合适的。如果一个上千W的文档语料，它在256维的LSI空间需要2GB RAM内存，可以使用这个类。
    如果没有2GB的空闲RAM，你需要使用similarities.Similarity 类。这个类在固定内存中操作， 通过在磁盘中划分多个文件索引进行共享。它在内部使用 similarities.MatrixSimilarity  和similarities.SparseMatrixSimilarity，因此，它仍然很快，尽管有点复杂。

索引持久化通过标准的save()和load()函数来完成：

    >>> index.save('/tmp/deerwester.index')
    >>> index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')

对于所有的相似索引类(similarities.Similarity, similarities.MatrixSimilarity和similarities.SparseMatrixSimilarity)来说，这是合适的。接下来，索引可以是任何一个对象。使用similarities.Similarity这个最扩展的版本，它也可以支持向索引中添加更多的文档。

##执行查询

为了在9个索引文档中，获取查询文档的相似度：

    >>> sims = index[vec_lsi] # perform a similarity query against the corpus
    >>> print(list(enumerate(sims))) # print (document_number, document_similarity) 2-tuples
    [(0, 0.99809301), (1, 0.93748635), (2, 0.99844527), (3, 0.9865886), (4, 0.90755945),
    (5, -0.12416792), (6, -0.1063926), (7, -0.098794639), (8, 0.05004178)]

通过余弦相似度进行计算，返回范围在<-1, 1>之间。（越大表示越相似），第一个文档的分数为：0.99809301等。

使用一些python标准函数，我们按相似度进行降序排列，从而获取查询“Human computer interaction”最后的答案。

    >>> sims = sorted(enumerate(sims), key=lambda item: -item[1])
    >>> print(sims) # print sorted (document number, similarity score) 2-tuples
    [(2, 0.99844527), # The EPS user interface management system
    (0, 0.99809301), # Human machine interface for lab abc computer applications
    (3, 0.9865886), # System and human system engineering testing of EPS
    (1, 0.93748635), # A survey of user opinion of computer system response time
    (4, 0.90755945), # Relation of user perceived response time to error measurement
    (8, 0.050041795), # Graph minors A survey
    (7, -0.098794639), # Graph minors IV Widths of trees and well quasi ordering
    (6, -0.1063926), # The intersection graph of paths in trees
    (5, -0.12416792)] # The generation of random binary unordered trees

（注意相应的注释）

我们需要注意文档no.2（“The EPS user interface management system”）以及文档no.4（“Relation of user perceived response time to error measurement”）将不会返回一个标准的boolean的全文搜索，因为它们与"Human computer interaction"没有公共部分。然后，在使用LSI后，我们发现，它们都具有很高的相似度（no.2实际上最相似），这与我们希望共享"computer-human"相关主题查询的目换一致。实际上，这种语义归纳也是为什么我们要应用变换来处理主题建模的原因。

##下一步

看到这样，那么恭喜你，你已经完成了教程。你已经理解了gensim的工作机制，如果想了解更多细节，可以查看API文档，或者查看Wiki，或者cheout出gensim的[分布式计算](http://d0evi1.github.io/gensim/distributed)。

gensim是一个相当成熟的包，它已经成功地用在了许多行业和公司，在快速原型以及生产环境中都已经在使用。当然这并不意味着一切很完美：

  - 仍然有些部分的实现可以做的更高效（例如，使用C），或者充分利用并行计算(利用多核)
  - 新的算法一直都有在发布，可以在gensim上讨论
  - 需要你的回馈（不仅仅是代码）：包括贡献 思想、bug 或者其它等。。

gensim的目的不是成为一个通用的跨越NLP/机器学习等子领域的全领域框架。它的目标只是帮助NLP实践者在大数据集上进行主题建模算法，对新研究者来说可以进行新算法的快速原型实践。

[英文文档](http://radimrehurek.com/gensim/tut3.html)
