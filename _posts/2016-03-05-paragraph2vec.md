---
layout: post
title: paragraph2vec介绍
description: 
modified: 2016-03-05
tags: [sentence2vec]
---

我们都清楚word2vec，这是Deep NLP最基本的任务。对于词有词向量，对于句子，或者是段落，也一样可以生成相应的向量（注意：两者的向量空间是不一样，不能合在一个空间中）。paragraph2vec在[1]有详细介绍，我们先来看下具体的概念：

## 1.PV-DM:(Paragraph Vector:Distributed Memory model) 
 
受词向量(word vector)方法的启发，我们也学习到段落向量(paragraph vector)。词向量会被用于预测句子中的下一个词。因此，尽管实际上词向量的初始化是随机的，它们仍可以捕获语义，作为预测任务的间接结果。我们在paragraph vector中使用相类似的方式。在给定从段落中抽样获取多个上下文的情况下，也可使用paragraph vector来预测下一个词。

在我们的Paragraph Vector框架中(见图2), 每个段落（paragraph）都被映射到一个唯一的vector中，表示成矩阵D中的某一列；每个词(word)都映射到一个某一个向量中，表示成矩阵W中的某一列。对paragraph vector和word vector求平均，或者级联(concatenated)起来，以预测在上下文中的下一个词。在该试验中，我们使用级联(concatenation)作为组合向量的方法。

<img src="http://pic.yupoo.com/wangdren23/Gl206Ip6/medish.jpg">

图2: 学习paragraph vector的框架。该框架与word2vec的框架相似；唯一的区别是：会有额外的paragraph token通过矩阵D映射到一个vector中。在该模型中，对该向量以及再带上一个三个词的上下文，对它们进行级联或者求平均，用来预测第4个词。paragraph vector表示从当前上下文缺失的信息，可以看成是关于该段落(paragraph)的主题(topic)的记忆单元。

更正式的，在模型中与词向量框架的唯一变化是，h是从W和D中构建的。

**paragraph的token可以认为是另一个词**。它扮演的角色是，作为一个记忆单元，可以记住当前上下文所缺失的东西--或者段落（paragraph）的主题。出于该原因，我们经常称该模型为Paragraph Vector分布式记忆模型（PV-DM：Distributed Memory Model of Paragraph Vectors）。

上下文是固定长度的，从沿段落（paragraph）滑动的一个滑动窗口中采样。**所有在相同段落（paragraph）上生成的上下文，共享着相同的paragraph vector。在不同的段落（paragraphs）间，则共享着相同的词向量矩阵W**，比如，单词"powerful"的向量，对于所有段落（paragraphs）是相同的。

Paragraph Vectors和Word Vectors都使用SGD进行训练，梯度通过backpropagation算法求得。在SGD的每一步，你可以从一个随机paragraph中抽样一个固定长度的上下文，计算error的梯度，更新模型参数。

在预测阶段，对于一个全新的段落（paragraph），需要执行一个推断步骤(inference)来计算paragraph vector。这也可以通过梯度下降法获取。在该步骤时，对于模型其余部分的参数，word vectors:W以及softmax的权重，是固定的。

假设在语料中有N个段落（paragraph），词汇表中有M个词，我们希望学到paragraph vectors，每个paragraph都被映射到p维上，每个word被映射到q维上，接着该模型具有总共**N x p + M x q** 个参数（将softmax参数排除在外）。尽管当N很大时，参数的数目会很大，在训练期间的更新通常是稀疏的，并且很有效。

在训练之后，paragraph vectors可以当成是该段落（paragraph）的特征（例如：代替bow或作为bow的额外附加）。我们可以将这些features直接输入到常用的机器学习技术（LR, SVM或者K-means）中。

总之，算法本身有两个关键步骤：

- 1) 在训练（training）阶段：在已知的段落（paragraphs）上，获取词向量W，softmax的权重(U,b)以及paragraph vector: D. 
- 2) 在推断（inference）阶段：保持W,U,b固定不变，通过增加D中的更多列，在D上进行梯度下降，为新未曾见过的的段落(paragraph)获取paragraph vectors: D。我们使用D来做预测关于更多的特定labels。

**paragraph vectors的优点**：paragraph vectors的一个重要优点是，它们可以从未标记的数据（unlabeled data）中学到，在没有足够多带标记的数据（labeled data）上仍工作良好。

Paragraph vectors也处理了一些BOW模型所具有的主要缺点。首先，它们继承了词向量的一个重要特性：词的语义（semantics）。在该空间中，对比"Paris"与"strong"，"powerful"与"strong"更接近。Paragraph vector的第二个优点是：它们会考虑词顺序（至少在某个小上下文上会考虑），与n-gram模型(有一个大的n)的方式相同。这十分重要，因为n-gram模型保留着一部分段落(paragraph)的信息，包括词顺序。也就是说，我们的模型可能优于一个bag-of-n-gram模型，因为一个bag-of-n-gram模型可以创建出一个高维表示，这很难泛化。

## 2.PV-DBOW: (无词序的Paragraph Vector: Distributed BOW)

上面的方法会将paragraph vector和词向量串联起来来预测一个文本窗口中的下一个词。接下来的另一种方法则是忽略掉输入中的上下文词汇，强制模型去预测从段落(paragraph)中随机抽样出的词作为输出。在实际上，这意味着，在SGD的每次迭代中，我们可以抽样一个文本窗口，接着从该文本窗口中抽样一个随机词汇，去构建这样一个分类器任务来获取Paragraph Vector。该技术如图3所示。我们将该版本称为：PV-DBOW (Distributed
Bag of Words version of Paragraph Vector)

<img src="http://pic.yupoo.com/wangdren23/Gl3kipIb/medish.jpg">

图3: PV-DBOW.在该版本中，训练该paramgraph vector以预测在一个小窗口中的词汇.

除了概念简单之外，该模型存储的数据也更少。**我们只需要存储softmax的权重，而PV-DM则需要存储softmax权重以及词向量**。该模型与word2vec中的skip-gram模型相类似。

在我们的试验中，每个paragraph vector是一个两种向量的组合：一个向量由标准PV-DM模型学到，另一个向量由PV-DBOW模型学到的。对于大多数任务PV-DM单独工作也能达到很好的效果（state-of-art），**如果与PV-DBOW组合在一起使用，在许多不同任务上可以更一致，强烈推荐使用组合方式**。

## 3.实验

我们会对paragraph vectors的表现进行实验。

对于语义分析，我们使用两个数据集：Stanford sentiment
treebank dataset 以及 IMDB dataset。这些数据集中的文档在长度上区别很大：Stanford数据集是单句，而IMDB则包含着多句。

我们也在一个信息检索任务上测试我们的方法，目标是：给定一个query，一个文档是否该被索引出。

### 3.1 基于sentiment-treebank数据集的Sentiment Analysis

数据集：该数据集首先在2005年提出，随后在2013进行扩展，是sentiment analysis的一个benchmark。它包含了11855个句子，从烂蕃茄（Rotten Tomatoes）的电影评论中获取。

该数据集包含了三个集合：8544个句子用于训练(training)，2210个句子用于测试(test)，1101个句子用于验证(validation)。

数据集中的每个句子都有一个label，表示极性的正负程度，从0.0到1.0.label由亚马逊的众包（Amazon Mechanical Turk）人工标注完成。

该数据集对于句子有详细的label，子句（subphrases）同样也需要。为了达成该目标，Socker et al.(2013b)使用Stanford Parser(Klein & Manning,2003)来将每个句子解析成子句(subphrases)。子句接着以相同的方式被人工标注。目前该数据集总共有239232个带标记的句子。数据集下载地址：[https://nlp.stanford.edu/sentiment/](https://nlp.stanford.edu/sentiment/)

任务以及Baselines: 在(Socker et al.,2013b)中，作者提出了两种benchmarking的方法。首先，可以考虑5-way细粒度分类任务，对应的label为：{Very Negative, Negative, Neutral, Positive, Very Positive}或一个2-way的粗粒度分类：{Negative, Positive}。另外，可以分为：是对整句，或者子句的划分。本工作主要针对完整句子的labeling.

**在该数据集中，Socher应用许多方法，并发现Recursive Neutral Tensor Network比BOW模型更好！** 这里仍有争议，因为电影评论通常很短，语义合成性（compositionality）在决定评论极性正负时扮演着重要角色。对于这个小训练集，对词语间的相似度也同样很重要。

试验约定：我们按照(Socher et al.2013b)所描述的实验约定。为了充分利用带标记数据，在我们的模型中，每个子句，都被当成是一个独立的句子，我们将为训练集中所有的子句学到它的向量表示。

在学习到训练句子和它们的子句的向量表示之后，我们将它们输入到一个logistic regression中来学习电影评分的预测。

在测试时，我们确定每个词的向量表示，使用梯度下降学到句子的向量表示。一旦学到测试句子的向量表示，我们将它们输入到logistic regression中来预测电影评分。

在我们的试验中，我们使用验证集对window size交叉验证，可选的window size为8。该分类器的向量表示是两个向量的串联：一个来自PV-DBOW，另一个来自PV-DM。在PV-DBOW中，学到的段落向量表示为400维。在PV-DM中，学到的词向量和段落向量表示均为400维。为了预测第8个房屋中，我们将paragraph vectors和7个word vectors相串联。**我们将特征字符“,.!?”这些看成是一个普通词。如果该段落（paragraph）少于9个词，我们会预补上（pre-pad）一个特殊的NULL符号（NULL word symbol）。**

结果：如表1所示。我们上报了不同方式的错误率。该表中高度部分是BOW或者是bag-of-n-gram模型(NB, SVM, NiNB)的效果很差。对词向量求平均（以BOW的方式）不会提升结果。因为BOW模型不会考虑句子是如何构成的（比如：词顺序），因此在识别许多复杂语义现象时（例如：反讽:sarcasm）会失败。结果也展示了更多的高级方法（比如：Socher.2013b的RNN），它需要parsing以及会对语义合成性做理解，效果更好。

<img src="http://pic.yupoo.com/wangdren23/Gll3HMbd/medium.jpg">

我们的方法比所有的这些baselines都要好，尽管实际上不需要parsing。在粗粒度分类任务上，我们的方法在error-rates上有2.4%的提升。相对提升16%!

### 3.2 多句：IMDB数据集的Sentiment Analysis

前面的技术只应用在单句上，而非带有多句的段落或者文档上。例如：RNN会基于在每个句子上进行parsing，而对于多个句子上的表示的组合是不清楚的。这种技术只限制在句子上，而不能用在段落或者文档上。

我们的方法不需要parsing，它可以对于一个包含多句的长文档生成一个表示。这个优点使人们的方法比其它方法更通用。下面的试验在IMDB数据集上展示了该优点。

数据集：IMDB数据集，首先由Maas et al., 2011提出作为sentiment analysis的一个benchmark. 该数据集包含来自IMDB的10W的电影评论。该数据集的一个关键点是，每个电影评论都有多句话组成。

10w的电影评论被分成三个数据集：2.5W带标记的训练实例，2.5W带标记的测试实例，5W未标记的训练实例。有两类label: 正向（Positive），负向（Negative）。这些label对于训练集和测试集是均衡的(balanced)。数据集下载：[http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)

实验约定：我们会使用7.5W的训练文档（2.5W已经标注的实例，5W未标注的实例）来学到word vectors和paragraph vectors。对于2.5W已标注实例的paragraph vectors，接着会输入(feed)到一个单层的、含50个单元神经网络中，以及一个logistic分类器来预测语义。

在测试时，给定一个测试语句，我们再次固定网络的其余部分，通过梯度下降学到测试评论中段落向量（paragraph vectors）。当学到向量时，我们将它们输入到神经网络中来预测评论的sentiment。

我们的paragraph vector模型的超参数，和先前的任务相同。特别的，我们交叉验证了window size，最优的window size为10个词。输入到分类器的向量表示，是两个向量的串联，一个是PV-DBOW，另一个是PV-DM。在PV-DBOW中，学到的向量表示具有400维。在PV-DM中，为words和documents学到的向量表示都有400维。为了预测第10个词，我们将paragraph vectors和word vectors相串联。特殊词：",.!?"被看成是一个普通词来对街。如果文档比9个词少。我们会使用一个特殊的NULL词符号进行以预补足（pre-pad）。

结果：Paragraph Vectors的结果和其它baselines如表2所示。对于长文档，BOW模型执行很好，很难在它们之上使用词向量进行提升。最大的提升发生在2012年(Dahl et al.2012)，它将一个RBM模型与BOW相组合。两种模型的组合在error-rates上有1.5%的提升。

另一个大提升来自(Wang & Manning,2012)。他们使用了许多变种，在bigram特征上使用NBSVM，效果最好，在error-rates上有2%的提升。

在该paper上描述的方法，超出了10%的error-rate提升。它达到了7.42%，比上面最好的模型有1.3%的绝对提升(相对提升有15%)

<img src="http://pic.yupoo.com/wangdren23/GlluSRb1/medium.jpg">

表2: IMDB的Paragraph vector的效果对比.

### 3.3 使用PV的IR

我们在IR任务中，使用固定长度的paragraph表示。

这里，我们有一个段落数据集，给定100W的最流行搜索，返回有10个结果。这些段落的线一个都被称为片段“snippet”，它是一个网页的内容摘要，以及一个网页是如何匹配query的。

从这些collection中，我们派生了一个新的数据集作为测试集的paragraph向量表示。两个段落（paragraph）是对于相同的query的结果，第三个段落(paragraph）是从该collection的其它部分随机抽样得到的paragraph（作为一个不同的query得到的结果返回）。我们的目标是，确认这三个paragraph中哪些是相同query返回的结果。为了达到该目的，我们使用paragraph vectors，并计算paragraphs间的距离(distance)。也就是说：相同查询的段落对的距离的距离小，以及不同查询的段落对(paragraphs pairs)间的距离大。

这里有关于三个段落的一个样本，第一个段落更接近于第二个段落（比第三个）：

- 段落1: calls from ( 000 ) 000 - 0000 . 3913
calls reported from this number . according to 4 reports the identity of this caller is american airlines .
- 段落2: do you want to find out who called you
from +1 000 - 000 - 0000 , +1 0000000000 or ( 000
) 000 - 0000 ? see reports and share information you
have about this caller
- 段落3: allina health clinic patients for your
convenience , you can pay your allina health clinic
bill online . pay your clinic bill now , question and
answers...

该三联体(triplets)被划分为三个数据集：80%训练，10%验证，10%测试。任何方法都需要在训练集上学习，而超参数的选择则在验证集上选择。

我们对4种方法做benchmark，并计算段落的特征：bag-of-words, bag-of-bigrams, 对词向量求平均，对Paragraph Vector求平均。为了提升bag-of-bigrams，我们也学到了一个加权的martix：前2个的距离最小化，第1个和第3个段落的距离最大化（两个losses间的加权因子是个hyperparameter）

当每个方法中，两个段落的距离越来越小，第一个和第三个段落的距离越来越大时，我们记录了对应的时间数。如果方法不生成期望的结果，会出来一个error。

Paragraph Vector的结果和其它baseline如表3所示。在该任务中，我们发现，TF-IDF的加权效果比raw counts要好，因此，我们只上报了TF-IDF weighting方法。

结果展示了Paragraph Vector工作良好，在error-rate给出了一个32%的相对提升。实际上，paragraph-vector的方法好于bag-of-words以及bag-of-bigrams。

<img src="http://pic.yupoo.com/wangdren23/GllFt7Cy/medium.jpg">

### 3.4 一些进一步观察

- PV-DM比PV-DBOW的一致性要好。单独使用PV-DM达到的效果与本paper中的许多结果相接近（见表2）。例如，在IMDB上，PV-DM只达到了7.63%。PV-DM与PV-DBOW合一起更好一些（7.42%），因而推荐使用。
- 在PV-DM中使用串联(concatenation)，通常比求和(sum)更好。
- 对window-size进行cross-validate更好。许多应用的window size在：5-12之间.
- Paragraph Vector的计算开销大，但可以在测试时并行完成。平均的，我们的实现花了30分钟来计算IMDB测试集的paragraph vector，使用16-core机器(2.5W文档，每个文档平均230个词)

## 4.实现

# 4.1 gensim实现

gensim的models.doc2vec实现了该模型。

{% highlight python %}

class gensim.models.doc2vec.Doc2Vec(documents=None, 
	dm_mean=None, 
	dm=1, 
	dbow_words=0, 
	dm_concat=0, 
	dm_tag_count=1, 
	docvecs=None, 
	docvecs_mapfile=None, 
	comment=None, 
	trim_rule=None, 
	**kwargs)

{% endhighlight %}

它的基类是gensim中的: gensim.models.word2vec.Word2Vec

- documents：一个元素为TaggedDocument的list，对于更大的语料可以使用磁盘/网络。如果不提供documents，则模型会未初始化。
- dm: 缺省为1. dm=1,表示使用PV-DM。否则使用PV-DBOW.
- size: 特征向量的维度(基类中)
- window: 要预测的词与上下文间的最大距离，用于文档中的预测
- alpha: 初始的learning-rate（随着训练的进行，会线性降至0）
- seed: 用于随机数字生成器。注意，对于一个完整的确定可再生的运行过程，你必须将该模型限制到单个worker线程上， 以便消除OS线程调度引起的时序抖动。(在python 3中，不同解释器加载之间可再生也需要使用PYTHONHASHSEED环境变量来控制hash随机化)
- min_count: 忽略总频率低于该值的所有词
- max_vocab_size: 在词汇表构建时的最大RAM限制; 如果有许多单个的词超过该值，会对频率低的进行剪枝。每1000w的词类型，需要大概1GB的RAM。缺省设为None，即无限制。
- sample: 配置的阀值，更高频的词会随机下采样(random downsampled)。缺省为0(off), 有用的值为1e-5.
- workers: 使用多个worker线程来训练模型（多核机器更快）
- iter: 在语料上的迭代次数(epoches)。缺省值从Word2Vec继承下来，为5. 但对于'Paragraph Vector'来说，10或20更常用。
- hs: 如果为1, 表示使用hierarchical sampling来进行模型训练，否则为0. 缺省为1
- negative: 如果>0, 会使用negative sampling，int值表示应抽样“noise words”多少次。（通常设在5-20间）
- dm_mean: 如果为0(缺省情况), 会使用上下文的词向量的求和(sum)。如果为1,则使用求平均（mean）。如果dm以非级联(non-concatenative)的模式，才会使用它。
- dm_concat: 如果为1,则使用上下文向量的级联方式(concatenation)，而非(sum/average)方式；缺省为0(off)。注意，级联(concatenation)会导致一个大的多的模型，输入不再是一个词向量（被抽样出或者算术结合）的size，而是使用该tag(s)的size和上下文中的所有词捆在一起。
- dm_tag_count: 当使用dm_concat模式时，每个文档所期望常数个文档标签；缺省为1
- dbow_words: 如果设置为1, 则会训练word-vectors(以skip-gram的方式)，同时训练DBOW的doc-vector；缺省为0(只训练doc-vectors训练会更快）
- trim_rule: 词汇表剪枝规则，指定了特定的词是否应保留在词汇表中，是否被削剪掉，或者使用缺省方式处理（如果词的count<min_count，直接抛弃）. 可以置为None(即使用min_count)，或者使用一个回调，使用参数(word,count,min_count)，返回下述值：util.RULE_DISCARD, util.RULE_KEEP or util.RULE_DEFAULT. 注意：如果给定该规则，会使用它在build_vocab()期间来剪枝词汇表，不会被当成是模型的一部分进行保存。

另几个比较重要的函数：

- delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

抛弃在训练时和评分时用到的参数。如果你确认模型训练完了，就可以使用它。keep_doctags_vectors=False，不会保存doctags向量，这种情况下，不可以使用most_similar进行相似度判断。keep_inference=False表示，你不希望保存用于infer_vector的参数.


相应的示例代码，可以参见: 

- [doc2vec-IMDB](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb)  
- [test-doc2vec](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/test/test_doc2vec.py)


## 二、Tomas Mikolov的c实现

Tomas Mikolov在[https://groups.google.com/forum/#!msg/word2vec-toolkit/Q49FIrNOQRo/J6KG8mUj45sJ](https://groups.google.com/forum/#!msg/word2vec-toolkit/Q49FIrNOQRo/J6KG8mUj45sJ)处提供了他的sentence2vec的实现。

- cbow=0: 表示PV-DBOW.


## 三、其它实现

[https://github.com/zseymour/phrase2vec](https://github.com/zseymour/phrase2vec)





# 参考

- [Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)
- [gensim.models.doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html)