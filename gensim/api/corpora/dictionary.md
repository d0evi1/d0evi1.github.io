---
layout: page
title: corpora.dictionary - 构建 word <-> id 映射 
---
{% include JB/setup %}

该模块实现了字典（Dictionary）的概念－－一个在word与整型id之间的映射。

字典可以通过corpus创建，接着根据文档频率进行prune调整（比如：通过Dictionary.filter_extremes()移除公共或非公共词汇），从磁盘进行save/load（通过Dictionary.save()或者Dictionary.load()方法），或者和其它字典进行合并（Dictionary.merge_with()）等。

-----------------------------------------------------------

class gensim.corpora.dictionary.Dictionary(documents=None, prune_at=2000000)

    基类：gensim.utils.SaveLoad, _abcoll.Mapping

    字典封装了在归一化词汇（word）与整型id之间的映射关系。

    主要函数有 doc2bow，它将许多词汇转换成词袋（bag-of-words）模型表示：一个2-tuples列表（word_id, word_frequency）。

    如果给定了documents，使用它们进行字典初始化（参见：add_documents()）

-----------------------------------------------------------

add_documents(documents, prune_at=2000000)

    从文档中更新字典。每个文档是一个token列表 = token化和归一化的字符串（utf8 或者 unicode）。

    有一个便利的wrapper类，在每个文档上调用doc2bow时，使用allow_update=True，可以对频率不高的word进行调整，保持唯一单词总量<=prune_at。在非常大的输入时，可以保存在内存中。若要禁止使用pruning，可以将prune_at=None.

    >>> print(Dictionary(["máma mele maso".split(), "ema má máma".split()]))
    Dictionary(5 unique tokens)

-----------------------------------------------------------

compactify()

    为所有的词汇分派新的id。

    它可以让id更加紧凑，通过filter_tokens()移除一些token、以及id序列间的间隙来实现。调用该方法可以移除间隙。

-----------------------------------------------------------

doc2bow(document, allow_update=False, return_missing=False)

    转文档（一些词汇列表）转换成词袋模型 = 一列2-tuples（token_id, token_count）。每个词汇都假设是一个token化和归一化的字符串（unicode 或者 utf8编码）。文档中的词汇不需要再做进一步预处理，在调用该方法前应tokenization, stemming等。

    如果设置了allow_update，接着在处理中更新字典：为每个新的词汇创建id。同时，更新文档频率－－对于在文档中出现的每个词汇，对文档频率加1（self.dfs）。

    如果没有设置allow_update，则该函数是const，只读。

-----------------------------------------------------------

filter_extremes(no_below=5, no_above=0.5, keep_n=100000)

    过滤掉token，它们出现在：

    - 1.小于no_below（绝对数）— 的文档
    - 2.大于no_above的文档
    - 3.在(1)和(2)之后，保持首个keep_n个最频繁的token（如果为None，保持所有的）

    在pruning之后，将词汇id间隙进行压缩。

    注意：由于间隙压缩，在之前和之后调用该函数，相同的词汇可以有不同的词汇id。

-----------------------------------------------------------

filter_tokens(bad_ids=None, good_ids=None)

    从所有的字典映射中，移除选中的bad_ids的token，或者，保存在映射中选中的 good_ids，移除其余的。

    bad_ids和good_ids都是被移除的词汇id集合。

static from_corpus(corpus, id2word=None)

    从一个存在的语料中创建一个字典（Dictionary）。如果你只有一个词汇-文档的BOW矩阵（由语料来表示）这将会很有用，注：非原始文本语料。

    该函数这将扫描词汇-文档数矩阵中的所有词汇id，接着构建和返回一个将word_id-> id2word[word_id]映射的字典（Dictionary）。

static from_documents(documents)

-----------------------------------------------------------

get(key, default=None)

-----------------------------------------------------------

items()

-----------------------------------------------------------

iteritems()

-----------------------------------------------------------

iterkeys()

-----------------------------------------------------------

itervalues()

-----------------------------------------------------------

keys()

    返回所有token id的列表。

classmethod load(fname, mmap=None)

    加载之前保存的文件对象。

    同上。

static load_from_text(fname)

    加载一个之前保存文件的字典，镜像函数：save_as_text.

-----------------------------------------------------------

merge_with(other)

    将另一个字典合并到该字典中，将一些相同的token映射到相同的id上，新的token对应新的id。这个函数的目的是为了将两个语料进地合并。

    other: 可以是任何 id=>word 映射。（一个dict，或者一个Dictionary对象，...） 

    当作为 result[doc_from_other_corpus]进行访问时，返回一个变换对象，它使用other 字典将语料中的文档，使用新的字典转换成另一个文档。

    示例：

    >>> dict1 = Dictionary(some_documents)
    >>> dict2 = Dictionary(other_documents)  # ids not compatible with dict1!
    >>> dict2_to_dict1 = dict1.merge_with(dict2)
    >>> # now we can merge corpora from the two incompatible dictionaries into one
    >>> merged_corpus = itertools.chain(some_corpus_from_dict1, dict2_to_dict1[some_corpus_from_dict2])


-----------------------------------------------------------

save(fname, separately=None, sep_limit=10485760, ignore=frozenset([]))

    将对象保存成文件。

    如果separately为None，自动检测大的 numpy/scipy.sparse对象数组，将它们以独立文件方式进行存储。这将避免pickle内存错误，允许mmap映射加载大数组。

    ...

-----------------------------------------------------------

save_as_text(fname, sort_by_word=True)

    将一个文本文件保存成Dictionary，格式为： id[TAB]word_utf8[TAB]document frequency[NEWLINE]. 按词汇进行排序，或者通过词频进行排序。 

    注意：使用文本格式时进行语料内省。使用save/load来存储二进制格式（pickle）来改进性能。

-----------------------------------------------------------

values()

[英文原版](http://radimrehurek.com/gensim/corpora/dictionary.html)














    


