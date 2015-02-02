---
layout: page
title: wiki语料实验 
---
{% include JB/setup %}

为了测试gensim的性能，我们在它之上运行了wikipedia英文版本.

该页描述了如何获取和处理Wikipedia，任何人都可以获取结果。当然前提是你安装了合适的gensim。

# 1.准备语料

- 1.首先，下载wikipedia英文语料[enwiki-latest-pages-articles.xml.bz2](http://download.wikimedia.org/enwiki/)，该文件8G,包含了English Wikipedia的所有文章.
- 2.将文章转换成plain text(需处理wiki markup)，并将结果存储在TF-IDF的稀疏矩阵. 在python中，可以很容易做到，我们不必解压整个文件到磁盘。gensim相应的运行脚本如下：

    $ python -m gensim.scripts.make_wiki

注意：

预处理阶段，两个8.2GB压缩的wiki文档dump（一个用于抽取出dictionary，另一个用于创建和存储稀疏矩阵），这个过程在我的笔记本电脑上处理了9个小时，因此你可以喝杯咖啡。

# 2.LSA

首先，我们载入语料迭代器和dictionary，通过上面的步骤：

    >>> import logging, gensim, bz2
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    >>> # load id->word mapping (the dictionary), one of the results of step 2 above
    >>> id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
    >>> # load corpus iterator
    >>> mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
    >>> # mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output (recommended)

    >>> print(mm)
    MmCorpus(3931787 documents, 100000 features, 756379027 non-zero entries)

我们将看到，我们的语料包含了3.9M的文档，100k的特征(不同的标记)，以及0.76G在TF-IDF稀疏矩阵中的非零实体. wikipedia语料总共包含了22.4亿的符号标记.

现在，我们准备计算English Wikipedia的LSA：

    >>> # extract 400 LSI topics; use the default one-pass algorithm
    >>> lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=400)

    >>> # print the most contributing words (both positively and negatively) for each of the first ten topics
    >>> lsi.print_topics(10)
    topic #0(332.762): 0.425*"utc" + 0.299*"talk" + 0.293*"page" + 0.226*"article" + 0.224*"delete" + 0.216*"discussion" + 0.205*"deletion" + 0.198*"should" + 0.146*"debate" + 0.132*"be"
    topic #1(201.852): 0.282*"link" + 0.209*"he" + 0.145*"com" + 0.139*"his" + -0.137*"page" + -0.118*"delete" + 0.114*"blacklist" + -0.108*"deletion" + -0.105*"discussion" + 0.100*"diff"
    topic #2(191.991): -0.565*"link" + -0.241*"com" + -0.238*"blacklist" + -0.202*"diff" + -0.193*"additions" + -0.182*"users" + -0.158*"coibot" + -0.136*"user" + 0.133*"he" + -0.130*"resolves"
    topic #3(141.284): -0.476*"image" + -0.255*"copyright" + -0.245*"fair" + -0.225*"use" + -0.173*"album" + -0.163*"cover" + -0.155*"resolution" + -0.141*"licensing" + 0.137*"he" + -0.121*"copies"
    topic #4(130.909): 0.264*"population" + 0.246*"age" + 0.243*"median" + 0.213*"income" + 0.195*"census" + -0.189*"he" + 0.184*"households" + 0.175*"were" + 0.167*"females" + 0.166*"males"
    topic #5(120.397): 0.304*"diff" + 0.278*"utc" + 0.213*"you" + -0.171*"additions" + 0.165*"talk" + -0.159*"image" + 0.159*"undo" + 0.155*"www" + -0.152*"page" + 0.148*"contribs"
    topic #6(115.414): -0.362*"diff" + -0.203*"www" + 0.197*"you" + -0.180*"undo" + -0.180*"kategori" + 0.164*"users" + 0.157*"additions" + -0.150*"contribs" + -0.139*"he" + -0.136*"image"
    topic #7(111.440): 0.429*"kategori" + 0.276*"categoria" + 0.251*"category" + 0.207*"kategorija" + 0.198*"kategorie" + -0.188*"diff" + 0.163*"категория" + 0.153*"categoría" + 0.139*"kategoria" + 0.133*"categorie"
    topic #8(109.907): 0.385*"album" + 0.224*"song" + 0.209*"chart" + 0.204*"band" + 0.169*"released" + 0.151*"music" + 0.142*"diff" + 0.141*"vocals" + 0.138*"she" + 0.132*"guitar"
    topic #9(102.599): -0.237*"league" + -0.214*"he" + -0.180*"season" + -0.174*"football" + -0.166*"team" + 0.159*"station" + -0.137*"played" + -0.131*"cup" + 0.131*"she" + -0.128*"utc"


