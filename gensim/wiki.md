---
layout: page
title: wiki语料实验 
---
{% include JB/setup %}

为了测试gensim的性能，我们在它之上运行了wikipedia英文语料.

该页描述了如何获取和处理Wikipedia语料，并且任何人都可以获取结果。当然，前提是你安装了合适的gensim。

# 1.准备语料

- 1.首先，下载wikipedia英文语料[enwiki-latest-pages-articles.xml.bz2](http://download.wikimedia.org/enwiki/)，该文件大小8G，包含了English Wikipedia的所有文章。
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

创建wiki的LSI模型，大概花费4小时左右。占满所有I/O时，每分钟处理16000文档。

注意：

如果你需要结果更快，可以看教程：[分布式计算](). 注意，gensim中的BLAS库透明地使用多核，因此，相同的数据可以在多核环境上获得更快的处理，无需进行分布式设置。

我们看到，整个处理时间，从原始wikipedia的XML，到dump出TF-IDF语料的整个过程，大概花费9小时。

gensim使用的算法，只需要一次查看一个输入文档，因此，很适合这样的场景：文档作为非重复流，或者存储/迭代整个语料的代价太高。

# 3.LDA

和上面的LSA类似，首先载入语料迭代器和dictionary:

    >>> import logging, gensim, bz2
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    >>> # load id->word mapping (the dictionary), one of the results of step 2 above
    >>> id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
    >>> # load corpus iterator
    >>> mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
    >>> # mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output

    >>> print(mm)
    MmCorpus(3931787 documents, 100000 features, 756379027 non-zero entries)

我们将运行Online LDA，该算法的过程是：将文档分块，更新LDA模型，获得另一个块，再更新模型，等。Online LDA与batch LDA截然不同(后者会一次传入即处理整个语料)，接着更新模型，接着传另一个，再更新... 不同之处是，给定一个合理的不变的文档流(没有许多主题漂移)，online版本更新小块的方式更好，因此模型估计聚集更快。作为结果，我们可能只需传单个语料：如果语料中有300w文章，那么我们每1w篇更新一次，这意味着我们需要300次更新，足够精确的主题估计：

    >>> # extract 100 LDA topics, using 1 pass and updating once every 1 chunk (10,000 documents)
    >>> lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)
    using serial LDA version on this node
    running online LDA training, 100 topics, 1 passes over the supplied corpus of 3931787 documents, updating model once every 10000 documents
    ...

不像LSA, LDA的主题更容易解释：

    >>> # print the most contributing words for 20 randomly selected topics
    >>> lda.print_topics(20)
    topic #0: 0.009*river + 0.008*lake + 0.006*island + 0.005*mountain + 0.004*area + 0.004*park + 0.004*antarctic + 0.004*south + 0.004*mountains + 0.004*dam
    topic #1: 0.026*relay + 0.026*athletics + 0.025*metres + 0.023*freestyle + 0.022*hurdles + 0.020*ret + 0.017*divisão + 0.017*athletes + 0.016*bundesliga + 0.014*medals
    topic #2: 0.002*were + 0.002*he + 0.002*court + 0.002*his + 0.002*had + 0.002*law + 0.002*government + 0.002*police + 0.002*patrolling + 0.002*their
    topic #3: 0.040*courcelles + 0.035*centimeters + 0.023*mattythewhite + 0.021*wine + 0.019*stamps + 0.018*oko + 0.017*perennial + 0.014*stubs + 0.012*ovate + 0.011*greyish
    topic #4: 0.039*al + 0.029*sysop + 0.019*iran + 0.015*pakistan + 0.014*ali + 0.013*arab + 0.010*islamic + 0.010*arabic + 0.010*saudi + 0.010*muhammad
    topic #5: 0.020*copyrighted + 0.020*northamerica + 0.014*uncopyrighted + 0.007*rihanna + 0.005*cloudz + 0.005*knowles + 0.004*gaga + 0.004*zombie + 0.004*wigan + 0.003*maccabi
    topic #6: 0.061*israel + 0.056*israeli + 0.030*sockpuppet + 0.025*jerusalem + 0.025*tel + 0.023*aviv + 0.022*palestinian + 0.019*ifk + 0.016*palestine + 0.014*hebrew
    topic #7: 0.015*melbourne + 0.014*rovers + 0.013*vfl + 0.012*australian + 0.012*wanderers + 0.011*afl + 0.008*dinamo + 0.008*queensland + 0.008*tracklist + 0.008*brisbane
    topic #8: 0.011*film + 0.007*her + 0.007*she + 0.004*he + 0.004*series + 0.004*his + 0.004*episode + 0.003*films + 0.003*television + 0.003*best
    topic #9: 0.019*wrestling + 0.013*château + 0.013*ligue + 0.012*discus + 0.012*estonian + 0.009*uci + 0.008*hockeyarchives + 0.008*wwe + 0.008*estonia + 0.007*reign
    topic #10: 0.078*edits + 0.059*notability + 0.035*archived + 0.025*clearer + 0.022*speedy + 0.021*deleted + 0.016*hook + 0.015*checkuser + 0.014*ron + 0.011*nominator
    topic #11: 0.013*admins + 0.009*acid + 0.009*molniya + 0.009*chemical + 0.007*ch + 0.007*chemistry + 0.007*compound + 0.007*anemone + 0.006*mg + 0.006*reaction
    topic #12: 0.018*india + 0.013*indian + 0.010*tamil + 0.009*singh + 0.008*film + 0.008*temple + 0.006*kumar + 0.006*hindi + 0.006*delhi + 0.005*bengal
    topic #13: 0.047*bwebs + 0.024*malta + 0.020*hobart + 0.019*basa + 0.019*columella + 0.019*huon + 0.018*tasmania + 0.016*popups + 0.014*tasmanian + 0.014*modèle
    topic #14: 0.014*jewish + 0.011*rabbi + 0.008*bgwhite + 0.008*lebanese + 0.007*lebanon + 0.006*homs + 0.005*beirut + 0.004*jews + 0.004*hebrew + 0.004*caligari
    topic #15: 0.025*german + 0.020*der + 0.017*von + 0.015*und + 0.014*berlin + 0.012*germany + 0.012*die + 0.010*des + 0.008*kategorie + 0.007*cross
    topic #16: 0.003*can + 0.003*system + 0.003*power + 0.003*are + 0.003*energy + 0.002*data + 0.002*be + 0.002*used + 0.002*or + 0.002*using
    topic #17: 0.049*indonesia + 0.042*indonesian + 0.031*malaysia + 0.024*singapore + 0.022*greek + 0.021*jakarta + 0.016*greece + 0.015*dord + 0.014*athens + 0.011*malaysian
    topic #18: 0.031*stakes + 0.029*webs + 0.018*futsal + 0.014*whitish + 0.013*hyun + 0.012*thoroughbred + 0.012*dnf + 0.012*jockey + 0.011*medalists + 0.011*racehorse
    topic #19: 0.119*oblast + 0.034*uploaded + 0.034*uploads + 0.033*nordland + 0.025*selsoviet + 0.023*raion + 0.022*krai + 0.018*okrug + 0.015*hålogaland + 0.015*russiae + 0.020*manga + 0.017*dragon + 0.012*theme + 0.011*dvd + 0.011*super + 0.011*hunter + 0.009*ash + 0.009*dream + 0.009*angel

创建Wikipedia的LDA模型在我的笔记本电脑上需要6小时左右。如果你更快的速度，可以在集群上运行[分布式LDA]().

注意，LDA和LSA运行的区别：我们使用LSA来抽取400个主题，LDA只有100个主题(因此，速度上的区别更明显). gensim中的LSA实现是真正意义上的online：如果输入流即时发生的变更，LSA将重新做过增量更新。相比而言，LDA不是真正意义上的online，在模型上的后续更新的影响逐渐减弱。如果在输入文档流中存在主题漂移，LDA将会混乱，并且将它调整到新状态时增长的缓慢。

简单的说，注意，如果使用LDA，增量添加新的文档到模型中。使用batch LDA是可以的且不受影响，整个训练语料库可以事先知道。

如果运行batch LDA(非online方式)，训练LdaModel：

    >>> # extract 100 LDA topics, using 20 full passes, no online updates
    >>> lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=0, passes=20)

通常，一个训练模型可以被用来将新的，未知的文档（plain bag-of-words count vectors）变换成LDA主题分布：

    >>> doc_lda = lda[doc_bow]

[1] 我的笔记本：MacBook Pro, Intel Core i7 2.3GHz, 16GB DDR3 RAM, OS X with libVec.
[2] 这里最关心的是性能，但是它关心LSA概念. 我不是Wikipedia专家，不知道wiki的内部机制，Brian Mingus说过相应的结果：

。。。

[英文文档](http://radimrehurek.com/gensim/wiki.html)
