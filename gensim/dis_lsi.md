---
layout: page
title: 分布式LSA 
---
{% include JB/setup %}

# 1.建立一个集群

我们将通过示例展示如何运行分布式LSA。我们有5台机器，它们都在同一个网段(=通过网络广播可达)。我们首先在每台机器上安装gensim和Pyro:

    $ sudo easy_install gensim[distributed]
    $ export PYRO_SERIALIZERS_ACCEPTED=pickle
    $ export PYRO_SERIALIZER=pickle

接着在其中一台机器上，运行Pyro的名字服务器：
    
    $ python -m Pyro4.naming -n 0.0.0.0 &

我们的示例集群包含高内存的多核机器。在另4台机器上分别运行两个worker脚本，创建8个逻辑worker节点：

    $ python -m gensim.models.lsi_worker &

接着，执行gensim的lsi_worker.py脚本（在每台机器上运行2次）。让gensim知道它可以在其中的一台机器上并行运行两个job，以使计算加速，当然每个机器上也会占用更多内存。

接着，选择一台机器充当job调度顺，负责worker的同步，在它之上运行LSA dispatcher。在我们的示例中，我们将使用第5台机器来充当dispatcher：

    $ python -m gensim.models.lsi_dispatcher &

总之，dispatcher和其它任一worker节点都相似。dispatcher不会做更多的CPU时间，但是会选择一台内存丰富的机器。

当集群建立和运行时，准备开始接受job。如果为了移除worker，可以终止它的lsi_worker进程。为了添加另一个worker，可以运行另一个lsi_worker（它不会影响正在运行的计算，添加和删除都不是动态的）。但如果你终止了lsi_dispatcher，你将不会运行计算，直到你再次运行它（worker进程可以被重用）.


## 1.1 多机部署(译者添加)


由于该版本的代码只支持broadcast域的分布式节点，如果你的局域网不支持，可以通过修改gensim的代码来实现。如果你的Pyro名字服务器运行在：
    
    python -m Pyro4.naming -n 10.177.128.143 &


### 1.1.1 utils.py修改
    
在site-packages下找到gensim，将gensim/utils.py中的代码的getNS()函数进行修改：

    return Pyro4.locateNS()

的两个地方，全修改为：

    return Pyro4.locateNS("10.177.128.143", 9090)

同时，将get_my_ip()函数下的:
    
    ns = Pyro4.naming.locateNS()

也替换成：

    ns = Pyro4.naming.locateNS("10.177.128.143", 9090)

即可。

### 1.1.2 lsi_dispatcher.py 修改

gensim使用pyro的PYRONAME方式进行对象查询，因此，需要将initialize()函数中的：

    self.callback = Pyro4.Proxy('PYRONAME:gensim.lsi_dispatcher')

修改为：

    self.callback = Pyro4.Proxy('PYRONAME:gensim.lsi_dispatcher@10.177.128.143')

### 1.1.3 lsimodel.py 修改

对__init__()函数处，修改：

    dispatcher = Pyro4.Proxy('PYRONAME:gensim.lda_dispatcher')

修改为：

    dispatcher = Pyro4.Proxy('PYRONAME:gensim.lda_dispatcher@10.177.128.143')


这样，一切就ok了。你可以在不同的机器上进行部署。

通过运行命令，可以查看该pyro名字服务器的连接ip和端口.

    pyro4-nsc list Pyro

通过运行下面的命令，可以查看该名字服务器上有哪些服务连接上.
    
    pyro4-nsc list

# 2.运行LSA

可以测试我们建的集群，运行分布式LSA计算。在5台机器上的一台打开python shell，尝试：

    >>> from gensim import corpora, models, utils
    >>> import logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    >>> corpus = corpora.MmCorpus('/tmp/deerwester.mm') # load a corpus of nine documents, from the Tutorials
    >>> id2word = corpora.Dictionary.load('/tmp/deerwester.dict')

    >>> lsi = models.LsiModel(corpus, id2word=id2word, num_topics=200, chunksize=1, distributed=True) # run distributed LSA on nine documents

这里，使用[语料和向量空间](http://d0evi1.github.io/gensim/tut1)中的corpus和feature-token映射。如果你想在python session中打开log，可以看到这样的一行：

    2010-08-09 23:44:25,746 : INFO : using distributed version with 8 workers

这意味着所有都ok。你可以确认日志来自于你的worker和dispatcher进程－－这对于定位问题来说特别有用。为了确认LSA结果，让我们打印两个隐含主题：

    >>> lsi.print_topics(num_topics=2, num_words=5)
    topic #0(3.341): 0.644*"system" + 0.404*"user" + 0.301*"eps" + 0.265*"time" + 0.265*"response"
    topic #1(2.542): 0.623*"graph" + 0.490*"trees" + 0.451*"minors" + 0.274*"survey" + -0.167*"system"

成功了! 但是包含9个文档的语料对于这么强大的集群来说没啥挑战.. 实际上，我们必须减少job的size（通过上面的chunksize参数指定）以便一次一个文档，否则所有文档将会被单个worker一次给处理了。

再看看我们运行100w的文档时的情况：

    >>> # inflate the corpus to 1M documents, by repeating its documents over&over
    >>> corpus1m = utils.RepeatCorpus(corpus, 1000000)
    >>> # run distributed LSA on 1 million documents
    >>> lsi1m = models.LsiModel(corpus1m, id2word=id2word, num_topics=200, chunksize=10000, distributed=True)

    >>> lsi1m.print_topics(num_topics=2, num_words=5)
    topic #0(1113.628): 0.644*"system" + 0.404*"user" + 0.301*"eps" + 0.265*"time" + 0.265*"response"
    topic #1(847.233): 0.623*"graph" + 0.490*"trees" + 0.451*"minors" + 0.274*"survey" + -0.167*"system"

1M 的LSA日志如下：

    2010-08-10 02:46:35,087 : INFO : using distributed version with 8 workers
    2010-08-10 02:46:35,087 : INFO : updating SVD with new documents
    2010-08-10 02:46:35,202 : INFO : dispatched documents up to #10000
    2010-08-10 02:46:35,296 : INFO : dispatched documents up to #20000
    ...
    2010-08-10 02:46:46,524 : INFO : dispatched documents up to #990000
    2010-08-10 02:46:46,694 : INFO : dispatched documents up to #1000000
    2010-08-10 02:46:46,694 : INFO : reached the end of input; now waiting for all remaining jobs to finish
    2010-08-10 02:46:47,195 : INFO : all jobs finished, downloading final projection
    2010-08-10 02:46:47,200 : INFO : decomposition complete

由于我们的100w语料中只有很小的词汇表size，以及很简单的结构。LSA的计算只需要12秒。为了压测下我们的集群，我们可以测试下wikipedia英文语料的LSA.

# 3.Wikipedia的分布式LSA

首先，下载和准备wikipedia语料，加载语料迭代器：

    >>> import logging, gensim, bz2
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    >>> # load id->word mapping (the dictionary)
    >>> id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
    >>> # load corpus iterator
    >>> mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
    >>> # mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output

    >>> print(mm)
    MmCorpus(3199665 documents, 100000 features, 495547400 non-zero entries)

现在，我们准备运行Wikipedia的分布式LSA：

    >>> # extract 400 LSI topics, using a cluster of nodes
    >>> lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=400, chunksize=20000, distributed=True)

    >>> # print the most contributing words (both positively and negatively) for each of the first ten topics
    >>> lsi.print_topics(10)
    2010-11-03 16:08:27,602 : INFO : topic #0(200.990): -0.475*"delete" + -0.383*"deletion" + -0.275*"debate" + -0.223*"comments" + -0.220*"edits" + -0.213*"modify" + -0.208*"appropriate" + -0.194*"subsequent" + -0.155*"wp" + -0.117*"notability"
    2010-11-03 16:08:27,626 : INFO : topic #1(143.129): -0.320*"diff" + -0.305*"link" + -0.199*"image" + -0.171*"www" + -0.162*"user" + 0.149*"delete" + -0.147*"undo" + -0.144*"contribs" + -0.122*"album" + 0.113*"deletion"
    2010-11-03 16:08:27,651 : INFO : topic #2(135.665): -0.437*"diff" + -0.400*"link" + -0.202*"undo" + -0.192*"user" + -0.182*"www" + -0.176*"contribs" + 0.168*"image" + -0.109*"added" + 0.106*"album" + 0.097*"copyright"
    2010-11-03 16:08:27,677 : INFO : topic #3(125.027): -0.354*"image" + 0.239*"age" + 0.218*"median" + -0.213*"copyright" + 0.204*"population" + -0.195*"fair" + 0.195*"income" + 0.167*"census" + 0.165*"km" + 0.162*"households"
    2010-11-03 16:08:27,701 : INFO : topic #4(116.927): -0.307*"image" + 0.195*"players" + -0.184*"median" + -0.184*"copyright" + -0.181*"age" + -0.167*"fair" + -0.162*"income" + -0.151*"population" + -0.136*"households" + -0.134*"census"
    2010-11-03 16:08:27,728 : INFO : topic #5(100.326): 0.501*"players" + 0.318*"football" + 0.284*"league" + 0.193*"footballers" + 0.142*"image" + 0.133*"season" + 0.119*"cup" + 0.113*"club" + 0.110*"baseball" + 0.103*"f"
    2010-11-03 16:08:27,754 : INFO : topic #6(92.298): -0.411*"album" + -0.275*"albums" + -0.217*"band" + -0.214*"song" + -0.184*"chart" + -0.163*"songs" + -0.160*"singles" + -0.149*"vocals" + -0.139*"guitar" + -0.129*"track"
    2010-11-03 16:08:27,780 : INFO : topic #7(83.811): -0.248*"wikipedia" + -0.182*"keep" + 0.180*"delete" + -0.167*"articles" + -0.152*"your" + -0.150*"my" + 0.144*"film" + -0.130*"we" + -0.123*"think" + -0.120*"user"
    2010-11-03 16:08:27,807 : INFO : topic #8(78.981): 0.588*"film" + 0.460*"films" + -0.130*"album" + -0.127*"station" + 0.121*"television" + 0.115*"poster" + 0.112*"directed" + 0.110*"actors" + -0.096*"railway" + 0.086*"movie"
    2010-11-03 16:08:27,834 : INFO : topic #9(78.620): 0.502*"kategori" + 0.282*"categoria" + 0.248*"kategorija" + 0.234*"kategorie" + 0.172*"категория" + 0.165*"categoría" + 0.161*"kategoria" + 0.148*"categorie" + 0.126*"kategória" + 0.121*"catégorie"

串行模式下，使用one-pass算法在Wikipedia上创建LSI模型需要花费5.25h(OS X, C2D 2.53GHz, 4GB RAM with libVec). 而4 worker的分布式模式（Linux, dual-core Xeons of 2Ghz, 4GB RAM with ATLAS），整个过程降至1小时41分钟。更多的详情，可以讲[我的论文](http://nlp.fi.muni.cz/~xrehurek/nips/rehurek_nips.pdf)。

[英文原版](http://radimrehurek.com/gensim/dist_lsi.html)

