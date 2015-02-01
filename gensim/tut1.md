---
layout: page
title: 语料与向量空间
---
{% include JB/setup %}


开始时，如果想设置日志，别忘记设置：

    >>> import logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


String到Vector的转换

这次，我们的文档由字符串表示：

    >>> from gensim import corpora, models, similarities
    >>>
    >>> documents = ["Human machine interface for lab abc computer applications",
    >>>              "A survey of user opinion of computer system response time",
    >>>              "The EPS user interface management system",
    >>>              "System and human system engineering testing of EPS",
    >>>              "Relation of user perceived response time to error measurement",
    >>>              "The generation of random binary unordered trees",
    >>>              "The intersection graph of paths in trees",
    >>>              "Graph minors IV Widths of trees and well quasi ordering",
    >>>              "Graph minors A survey"]

这是一个小语料，由9个文档组成，每个文档都由一句话组成.

首先，我们先对文档进行分割，移除常见词语(使用工具stoplist)，以及移除在语料库中只出现一次的词:

    >>> # remove common words and tokenize
    >>> stoplist = set('for a of the and to in'.split())
    >>> texts = [[word for word in document.lower().split() if word not in stoplist]
    >>>          for document in documents]
    >>>
    >>> # remove words that appear only once
    >>> all_tokens = sum(texts, [])
    >>> tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    >>> texts = [[word for word in text if word not in tokens_once]
    >>>          for text in texts]
    >>>
    >>> print(texts)
    
    [['human', 'interface', 'computer'],
     ['survey', 'user', 'computer', 'system', 'response', 'time'],
     ['eps', 'user', 'interface', 'system'],
     ['system', 'human', 'system', 'eps'],
     ['user', 'response', 'time'],
     ['trees'],
     ['graph', 'trees'],
     ['graph', 'minors', 'trees'],
     ['graph', 'minors', 'survey']]

处理文档的方式：这里，使用空格进行分割，每个单词转成小写。实际上，我使用这个特别的过程来模仿Deerwester et al.`s 的LSA文章中提到的实验.

这里处理文档的方式很多样，并且每个应用、甚至每种语言都不一样，以致于我决定不通过任何接口进行限定。相反的，一个文档可以通过抽取它的特征来表示，而非通过它的"surface"字符串格式：如何获取特征完全取决于你。下面我将描述一个公共方法（称为：词袋），但是请记住，不同的应用域有着不同的特征，或者， ...

为了将文档转换成向量，我们使用一个文档表示法（词袋）。在这种表示中，每个文档可以通过一个向量进行表示，这个向量的元素表示了一个问答对，以这样的形式：

    "How many times does the word system appear in the document? Once."

通过整型id来表示问题的优点很明显。问题及id之间的映射，可以称为字典：

    >>> dictionary = corpora.Dictionary(texts)
    >>> dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference
    >>> print(dictionary)
    Dictionary(12 unique tokens)

这里，我们通过 gensim.corpora.dictionary.Dictionary 分配了一个唯一的整型id给所有在语料库中出现过的词. 通过扫描整个文本，收集词汇数与相应的统计。在最后，我们将看到，在处理的语料中，含有12个不同的词，这意味着，每个文档将由12个数字表示（比如：通过12维向量），我们可以查询词与id之间的映射关系：
    
    >>> print(dictionary.token2id)
    {'minors': 11, 'graph': 10, 'system': 5, 'trees': 9, 'eps': 8, 'computer': 0,
    'survey': 4, 'user': 7, 'human': 1, 'time': 6, 'interface': 2, 'response': 3}

为了将切割过的文档转换成向量：

    >>> new_doc = "Human computer interaction"
    >>> new_vec = dictionary.doc2bow(new_doc.lower().split())
    >>> print(new_vec) # the word "interaction" does not appear in the dictionary and is ignored
    [(0, 1), (1, 1)]

函数doc2bow() 可以统计出每个不同的词汇的出现次数，将该词汇转换成它的整型id，并返回一个稀疏矩阵. 这个矩阵为 [(0, 1), (1, 1)]，可以理解成：在文档"Human computer interaction"中，词"computer"的id为0, 以及词"human"的id为1出现过一次；另十个字典词汇出现0次。

    >>> corpus = [dictionary.doc2bow(text) for text in texts]
    >>> corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus) # store to disk, for later use
    >>> print(corpus)
    [(0, 1), (1, 1), (2, 1)]
    [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]
    [(2, 1), (5, 1), (7, 1), (8, 1)]
    [(1, 1), (5, 2), (8, 1)]
    [(3, 1), (6, 1), (7, 1)]
    [(9, 1)]
    [(9, 1), (10, 1)]
    [(9, 1), (10, 1), (11, 1)]
    [(4, 1), (10, 1), (11, 1)]

对于“How many times does the word graph appear in the document?”这个问题，它的id=10的向量特征表示十分清楚。第六个文档的答案为0,其余三个为1. 实际上，我们可以通过快速示例看到类似的语料向量。

##语料流-一次一个文档

注意，上面的语料完全在内存中以python list的形式存在. 在这个简单的示例中，它
