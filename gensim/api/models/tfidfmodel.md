---
layout: page
title: similarities.docsim 文档相似查询 
---
{% include JB/setup %}

class gensim.models.tfidfmodel.TfidfModel(corpus=None, id2word=None, dictionary=None, wlocal=<function identity at 0x114b0d758>, wglobal=<function df2idf at 0x1152e6488>, normalize=True)

基类：[gensim.interfaces.TransformationABC](http://radimrehurek.com/gensim/interfaces.html)

该类的对象，实现了从词－文档之间的协矩阵（integer）转换成一个locally/globally权重的TF_IDF矩阵（正，float型）。

主要的函数有：

- 1.构造函数：用于计算在训练语料中的所有term的逆文档数 idf。
- 2.[]方法，它将一个简单的count表示转换成tfidf空间.

调用：

    >>> tfidf = TfidfModel(corpus)
    >>> print(tfidf[some_doc])
    >>> tfidf.save('/tmp/foo.tfidf_model')

模型的持久化，可以通过它的load/save方法来完成。

计算tf-idf通过一个本地组件(tf)和一个全局组件（idf）进行相乘，并将结果文档归一化到单位长度上。在一个包含D文档的语料中，对于在文档j中的未归一化的term i的权值计算公式如下：

    weight_{i,j} = frequency_{i,j} * log_2(D / document_freq_{i})

或者，更进一步说：

    weight_{i,j} = wlocal(frequency_{i,j}) * wglobal(document_freq_{i}, D)

因此，你可以插入自己定制实现的wlocal函数或者wglobal函数。

缺省的wlocal是指定的（其它选项有：math.sqrt，math.log1p，...），缺省的wglobal是 log_2(total_docs/doc_freq)，通过上面的公式给定。

normalize: 表示最终被转换的vector是如何归一化的。normalize=True意味着设置成单位长度（缺省方式）；若设为False，意味着不进行归一化。你可以将normalize设置成你自己实现的函数，并接受和返回一个稀疏矩阵。

如果指定了dictionary，它必须是一个corpora.Dictionary对象，并且它将直接用来构建 idf映射关系（如果指定，那么将忽略corpus）

---------------------------------------------------------------

initialize(corpus)

    计算逆文档权值，它可以被用来修改term的文档频率

classmethod load(fname, mmap=None)

    从文件中加载一个之前保存的对象。

    同其它。。。
    
---------------------------------------------------------------

save(fname, separately=None, sep_limit=10485760, ignore=frozenset([]))

    将对象保存到文件中.

    同其它。。。

---------------------------------------------------------------

gensim.models.tfidfmodel.df2idf(docfreq, totaldocs, log_base=2.0, add=0.0)

    计算缺省的逆文档频率，其文档频率为doc_freq：

    idf = add + log(totaldocs / doc_freq)


gensim.models.tfidfmodel.precompute_idfs(wglobal, dfs, total_docs)

    为所有term进行预计算idf。


[英文文档](http://radimrehurek.com/gensim/models/tfidfmodel.html)


