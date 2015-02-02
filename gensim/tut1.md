---
layout: page
title: 语料与向量空间
---
{% include JB/setup %}

开始时，如果想设置日志，别忘记设置：

    >>> import logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 1.将String映射到Vector

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

这里处理文档的方式有许多，每个应用、甚至每种语言可能都不一样，以致于我决定不通过任何接口进行限定。相反的，一个文档可以通过抽取它的特征来表示，而非通过它的"surface"字符串格式：如何获取特征完全取决于你。下面我将描述一个公共方法（称为：词袋），但是请记住，不同的应用领域的调用有着不同的特征，不管怎么样，garbage in, garbage out...

为了将文档转换成向量，我们使用一个文档表示法（词袋）。在这种表示中，每个文档可以通过一个向量进行表示，这个向量的元素表示了一个问答对，以这样的形式：

    "How many times does the word system appear in the document? Once."

通过整型id来表示answer的优点很明显。answer及id之间的映射，可以称为字典：

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

#2.语料流-一次一个文档

注意，上面的语料完全在内存中以python list的形式存在. 在这个简单的示例中，可能关系不大。我们假设语料中有几百万的文档。想把它们所有都保存在RAM中做不到。相反的，我们可以假设，文档存储在磁盘中的文件，每行一个文档。gensim只需要一个语料，一次必须返回一个文档向量:

    >>> class MyCorpus(object):
    >>>     def __iter__(self):
    >>>         for line in open('mycorpus.txt'):
    >>>             # assume there's one document per line, tokens separated by whitespace
    >>>             yield dictionary.doc2bow(line.lower().split())
    
Corpus是一个对象。我们没有定义任何方法来打印它，因此它只打印内存对象的地址。不是非常有用。为了查看相应的矢量，可以迭代corpus对象来打印每个文档向量(一次一个) ：

    >>> for vector in corpus_memory_friendly: # load one vector into memory at a time
    ...     print(vector)
    [(0, 1), (1, 1), (2, 1)]
    [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]
    [(2, 1), (5, 1), (7, 1), (8, 1)]
    [(1, 1), (5, 2), (8, 1)]
    [(3, 1), (6, 1), (7, 1)]
    [(9, 1)]
    [(9, 1), (10, 1)]
    [(9, 1), (10, 1), (11, 1)]
    [(4, 1), (10, 1), (11, 1)]

尽管输出与python list对象相似，corpus会友好地占用更多内存，因为至多一个向量在内存RAM中驻留一次。你的corpus可以如你想像的大。

相似的，我们可以不用将所有文本载入到内存中来构成这个字典：

    >>> # collect statistics about all tokens
    >>> dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))
    >>> # remove stop words and words that appear only once
    >>> stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
    >>>             if stopword in dictionary.token2id]
    >>> once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    >>> dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
    >>> dictionary.compactify() # remove gaps in id sequence after words that were removed
    >>> print(dictionary)
    Dictionary(12 unique tokens)

当然，我们如何处理这样的语料，是另一个问题；如何统计这些不同的词汇的频率可以十分有用。我们需要使用这个转换，我们可以使用它来计算任何有用的文档vs.文档相似度。转换将在下一篇教程中介绍。在此之前，我们将关注下语料的持久化。

#3.语料格式

存在许多文件格式，用来将一个向量空量语料序列化到磁盘上。gensim通过流接口来实现：文档通过一个延迟加载的方式读取（或存储），而非将整个语料读到内存中。

一个常用的文档格式是：mm(Market Matrix)格式. 为了将语料库保存成mm格式：

    >>> from gensim import corpora
    >>> # create a toy corpus of 2 documents, as a plain Python list
    >>> corpus = [[(1, 0.5)], []]  # make one document empty, for the heck of it
    >>>
    >>> corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)

其它的格式包括：joachim的SVMlight格式， Blei的LDA-C格式，以及GibbsLDA++格式.

    >>> corpora.SvmLightCorpus.serialize('/tmp/corpus.svmlight', corpus)
    >>> corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
    >>> corpora.LowCorpus.serialize('/tmp/corpus.low', corpus)

相反的，当我们从一个MM文件中加载一个语料时：

    >>> corpus = corpora.MmCorpus('/tmp/corpus.mm')

语料对象是流，因此通常你不能直接打印它：

    >>> print(corpus)
    MmCorpus(2 documents, 2 features, 1 non-zero entries)

相反的，为了查看语料中的内容：

    >>> # one way of printing a corpus: load it entirely into memory
    >>> print(list(corpus)) # calling list() will convert any sequence to a plain Python list
    [[(1, 0.5)], []]

或者

    >>> # another way of doing it: print one document at a time, making use of the streaming interface
    >>> for doc in corpus:
    ...     print(doc)
    [(1, 0.5)]
    []

第二种方式明显更内存占用友好些，对于测试和开发来说，则使用list(corpus)更方便.

为了将相似的MM文档流保存成Blei的LDA-C格式：

    >>> corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)

这种方式下，gensim可以当成是一个内存I/O格式转换器：只需要加载一个使用一种格式的文档流，就可以保存成另外一种。添加新的格式相当容易，可以checkout [SVMlight语料的代码](https://github.com/piskvorky/gensim/blob/develop/gensim/corpora/svmlightcorpus.py).

#4.兼容NumPy和SciPy

gensim也包含了有效的工具函数，来帮助转换numpy矩阵：

    >>> corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
    >>> numpy_matrix = gensim.matutils.corpus2dense(corpus)

以及scipy.sparse矩阵的from/to函数：

    >>> corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
    >>> scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)

下一篇将介绍：[主题和变换](http://d0evi1.github.com/gensim/tut2)


