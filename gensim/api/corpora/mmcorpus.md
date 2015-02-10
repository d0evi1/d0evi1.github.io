---
layout: page
title: corpora.mmcorpus - Matrix Market格式的语料 
---
{% include JB/setup %}

Matrix Market格式的语料

---------------------------------------------------------------

class gensim.corpora.mmcorpus.MmCorpus(fname)

    基类：gensim.matutils.MmReader, gensim.corpora.indexedcorpus.IndexedCorpus

    在Matrix Market格式存储的语料.

---------------------------------------------------------------

docbyoffset(offset)

    返回在文件offset处的偏移(单位：字节)

classmethod load(fname, mmap=None)

    加载之前保存的文件对象。

    同上。。。

---------------------------------------------------------------

save(*args, **kwargs)

static save_corpus(fname, corpus, id2word=None, progress_cnt=1000, metadata=False)

    将语料以Matrix Market格式保存成磁盘。

    该函数可以被MmCorpus.serialize()自动调用；如果不想直接调用，可以调用serialize()作为替代。

classmethod serialize(serializer, fname, corpus, id2word=None, index_fname=None, progress_cnt=None, labels=None, metadata=False)

    通过文档流corpus进行迭代，将文件保存成fname，并记录每个文档的对应偏移。将产生的索引结构保存成文件 index_fname （或者 不要设置 fname.index）

    这依靠底层的语料类 serializer 提供（除了标准迭代外）：

    - save_corpus 方法，将返回一个字节偏移量的序列，每个保存的文档都有一个.
    - docbyoffset(offset)方法，它将返回一个在持久化存储中位于offset字节处的文档。

示例：

    >>> MmCorpus.serialize('test.mm', corpus)
    >>> mm = MmCorpus('test.mm') # `mm` document stream now has random access
    >>> print(mm[42]) # retrieve document no. 42, etc.


skip_headers(input_file)

    跳过文件头，直接定位到第一个文档。

[英文正文](http://radimrehurek.com/gensim/corpora/mmcorpus.html)

