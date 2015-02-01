---
layout: page
title: 教程 
---
{% include JB/setup %}

#语料与向量空间

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


