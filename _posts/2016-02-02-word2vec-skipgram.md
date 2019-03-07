---
layout: post
title: word2vec中的Skip-gram
description: 
modified: 2016-02-02
tags: [word2vec+Huffman]
---

本文主要译自Tomas Mikolov、Jeffrey Dean等人的<Distributed Representations of Words and Phrases
and their Compositionality>.

# Abstract

最近介绍的continuous Skip-gram model是一个很有效的方法，用于学习高质量的分布式向量表示，它可以捕获大量精准的语法结构和语义词关系。在该paper中，我们介绍了许多种扩展，用来改进向量的质量和训练速度。通过对高频词进行subsampling，我们可以获得极大的速度提升，也可以学到更常规的词表示。我们也描述了在hirearchical softmax之外的一种简单的备选方法：negative sampling。

词向量表示的一个限制是，它不区分词顺序(word order)，它们不能表示常用短语。例如，"Canada"和"Air"的意思，不能简单的组合在一起来获取"Air Canada"（加拿大航空）。受该示例的启发，我们描述了另一种方法来寻找文本中的短语，并展示了如何在上百万的句子中学习到好的向量表示。

# 介绍

词在向量空间上的分布式表示(distributed representations)，通过将相似的词语进行群聚，可以帮助学习算法在nlp任务中完成更好的性能。词向量表示的最早应用可以追溯到1986年Rumelhart, Hinton等人提的(详见paper 13). 该方法用于统计语言建模中，并取得了可喜的成功。接下来，应用于自动语音识别和机器翻译(14,7)，以及其它更广泛的NLP任务(2,20,15,3,18,19,9)

最近，Mikolov（8）提出了Skip-gram模型，它是一个高效的方法，可以从大量非结构化文本数据中学到高质量的词向量表示。不同于以往用于词向量学习所使用的大多数神经网络结构，skip-gram模型(图1)不会涉及到稠密矩阵乘法(dense matrix multiplications)。这使得学习过程极为高效：一个优化版的单机实现，一天可以训练超过10亿个词。

使用神经网络的词向量表示计算非常有意思，因为通过学习得到的向量可以显式地对许多语言学规律和模式进行编码。更令人吃惊的是，许多这些模式可以被表示成线性变换(linear translations)。例如，比起其它向量，向量计算vec("Madrid")-vec("Spain")+vec("France")与vec("Paris")的结果更接近(9,8)。

本文中，我们描述了一些原始skip-gram模型的扩展。我们展示了对高频词进行subsampling，可以在训练期间带来极大提升（2x-10x的性能提升），并且同时能改进低频词的向量表示的精度。另外，我们提出了一种Noise Contrastive Estimation (NCE) (4)的变种，来训练skip-gram模型，对比于复杂的hierachical softmax，它的训练更快，并可以为高频词得到更好的向量表示。

词向量表示受限于它不能表示常用短语，因为它们不由独立的单词组成。例如, “Boston Globe”（波士顿环球报）实际是个报纸，因而它不是由“Boston”和"Globe"组合起来的意思。**因此，使用向量来表示整个短语，会使得skip-gram模型更有表现力**。因此，通过短语向量来构成有意义的句子的其它技术（比如：递归autoencoders 17)，可以受益于使用短语向量，而非词向量。

从基于词的模型扩展成基于短语的模型相当简单。**首先，我们使用数据驱动的方式标注了大量短语，接着我们在训练中将这些短语看成是独自的tokens**。为了评估短语向量的质量，我们开发了一个类比推理任务(analogical reasoning tasks)测试集，它同时包含了词和短语。我们的测试集中的一个典型的类比对(analogy pair)如下：

“Montreal”:“Montreal Canadiens”::“Toronto”:“Toronto Maple Leafs”

如果与vec("Montreal Canadiens")-vec("Montreal")+vec("Toronto")最接近的向量是：vec("Toronto Maple Leafs")，那么我们可以认为回答是正确的。

译者注1：

- Montreal: 蒙特利尔(城市)
- Montreal Canadiens: 蒙特利尔加拿大人(冰球队)
- Toronto: 多伦多(城市)
- Toronto Maple Leafs: 多伦多枫叶队(冰球队)

译者注2:

英文是基于空格做tokenized的. 常出现这个问题。

最后，我们再描述skip-gram模型的另一个有趣特性。我们发现，向量加法经常产生很有意思的结果，例如：vec("Russia")+vec("river")的结果，与vec("Volga River")接近。而vec("Germany")+vec("capital")的结果，与vec("Berlin")接近。这种组成暗示着，语言中一些不明显的程度，可以通过使用基本的词向量表示的数据操作来获取。

# 2.Skip-gram模型

Skip-gram模型的训练目标是，为预测一个句子或一个文档中某个词的周围词汇，找到有用的词向量表示。更正式地，给定训练词汇$$w_1,w_2,w_3,...,w_T$$, Skip-gram模型的学习目标是，最大化平均log概率：

$$
\frac{1}{T} \sum_{t=1}^{T} \sum_{-c\leq{j}\leq{c},j\neq0}^{} log p(w_{t+j} | w_t)
$$ 

...  (1)

其中，c是训练上下文的size($$w_t$$是中心词)。c越大，会产生更多的训练样本，并产生更高的准确度，训练时间也更长。最基本的skip-gram公式使用softmax函数来计算 \$ p(w_{t+j} \| w_t) \$: 

$$
p(w_O | w_I) = \frac{ exp({v'}_{w_O}^T * v_{w_I})}{\sum_{w=1}^{W} exp({v'}_{w}^T * v_{w_I})}
$$

... (2)

其中，vw和v'w表示w的输入向量和输出向量。W则是词汇表中的词汇数。该公式在实际中不直接采用，因为计算\$ \nabla
{logp(w_{O} \| w_I)} \$与W成正比，经常很大($$10^5-10^7$$次方)

## 2.1 Hierarchical Softmax

略，详见另一篇。

## 2.2 Negative Sampling

Hierarchical Softmax外的另一可选方法是Noise Contrastive Estimation(NCE)，它由Gutmann and Hyvarinen（4）提出，将由Mnih and Teh(11)用于语言建模中。NCE假设，一个好的模型应该能够通过logistic regression从噪声中区分不同的数据。这与Collobert and Weston（2）的hinge loss相类似，他通过将含噪声的数据进行排序来训练模型。

而NCE可以近似最大化softmax的log概率，Skip-gram模型只关注学习高质量的向量表示，因此，我们可以自由地简化NCE，只要向量表示仍能维持它的质量。我们定义了Negative sampling(NEG)的目标函数：

$$
log \sigma{({v'}_{w_O}^T v_{w_I})} + \sum_{i=1}^k E_{w_i} 
\sim P_n(w)[log \sigma{(-{v'}_{w_i}^T v_{w_I})} ]
$$

在Skip-gram目标函数中，每个$$ P(w_O \mid w_I) $$项都被替换掉。该任务是为了区分目标词wo，以及从使用logistic回归的噪声分布$$P_n(w)$$得到的词。其中每个数据样本存在k个negative样本。我们的试验中，对于小的训练数据集，k的值范围(5-20)是合适的；而对于大的数据集，k可以小到2-5。Negative sampling和NCE的最主要区分是，NCE同时需要样本和噪声分布的数值概率，而Negative sampling只使用样本。NCE逼近最大化softmax的log概率时，该特性对于我们的应用不是很重要。

NCE和NEG都有噪声分布$$P_n(w)$$作为自由参数。对于$$P_n(w)$$我们采用了一些不同选择，在每个任务上使用NCE和NEG，我们尝试包含语言建模（该paper没提及），发现unigram分布U(w)提升到3/4幂（如：\$ U(w)^{2/4}/Z \$）时，胜过unigram和uniform分布很多！

## 2.3 高频词的subsampling

# 3.结果

该部分我们评估了Hierarchical Softmax(HS), Noise Contrastive Estimation, Negative Sampling和训练词汇的subsampling。我们使用由Mikolov引入的analogical reasoning task进行评估(8)。该任务包含了类似这样的类比：s“Germany” : “Berlin” :: “France” : ?。通过找到这样的一个向量x，使得在cosine距离上，vec(x)接近于vec("Berlin")-vec("Germany")+vec("France")。如果x是"Paris"，则该特定示例被认为是回答正确的。该任务有两个宽泛的类别：

- syntactic analogies：句法结果的类比(比如： “quick” : “quickly” :: “slow” : “slowly”)
- semantic analogies：语义类比（比如：国家与城市的关系）

对于训练Skip-gram模型来说，我们已经使用了一个大数据集，它包括许多新文章（内部Google数据集，超过10亿单词）。我们抛弃了训练集中在词汇表中出现次数不足5次的词汇，这样产生的词汇表大小为：692k。在词类比测试中，多种Skip-gram模型的性能如表1。在analogical reasoning task上，该表展示了Negative Sampling的结果比Hierarchical Softmax效果要好，并且它比Noise Contrasitive Estimation的效果也略好。高频词的subsampling提升了好几倍的训练速度，并使得词向量表示更加精准。

仍有争议的是，skip-gram模型使它的向量更适合linear analogical reasoning，但Mikolov的结果(8)也展示了在训练数据量极剧增加时，由标准的sigmoidal RNN(非线性)可以在该任务上获得极大的提升，建议，对于词向量的线性结果，非线性模型同样可以有很好的表现。

# 4.学习短语

在前面的讨论中，许多短语具有特定的意义，它不是单个词的含义的简单组合。为了学习到短语的向量表示，我们首先发现，在一些不常见的上下文中，有些词经常共现。例如，“New York Times”和"Toronto Maple Leafs"在训练数据中，被替换成唯一的token，但另一个bigram:"this is"则保留不做更改。

<img src="http://pic.yupoo.com/wangdren23/G9Kx4Djd/medish.jpg">

表2：短语的analogical reasoning task（完整测试集：3218个示例）。目标是使用前三2上计算第4个短语。在该测试集上最好的模型的准确率为72%

这种方法，我们可以产生许多合理的短语，同时也不需要极大增加词汇的size；理论上，我们可以使用所有n-gram来训练Skip-gram模型，但这样很耗费内存。在文本上标识短语方面，之前已经有许多技术提出。然而，对比比较这些方法超出了该paper范围。我们决定使用一种简单的基于数据驱动的方法，短语的形成基于unigram和bigram的数目，使用：

$$
score(w_i,w_j)= \frac{count(w_i w_j- \delta} ){count(w_i) * count(w_j)}
$$  

...(6)

其中，delta被用于一个打折系数(discounting coefficient)，它可以阻止产生过多的包含许多不常见词的短语。bigram的score如果比选择的阀值要大，那么则认为该短语成立。通常，我们会不断降低阀值，运行2-4遍的训练数据，以允许形成包含更多词的更长短语。我们使用一个新的关于短语的analogical reasoning task，来评估短语表示的质量。该数据集在网上是公开的。[下载](http://2code.google.com/p/word2vec/source/browse/trunk/questions-phrases.txt)

## 4.1 短语的Skip-Gram结果

我们使用在前面的试验中相同的新闻数据，我们首先基于训练语料来构建短语，接着我们训练了多个Skip-gram模型，它们使用不同的超参数。在这之前，我们使用了300维的向量，上下文size=5。该设置可以在短语数据集上达到很好的效果，我们快速比较Negative Sampling和Hierarchical Softmax，是否采用高频token的subsampling。结果如表3所示：

<img src="http://pic.yupoo.com/wangdren23/G9KQD812/medish.jpg">

表3：Skip-gram在短语类比数据集上的准确率。这些模型在10亿词的新闻数据集上进行训练

为了最大化短语类比任务的准确率，我们增加了训练数据量，使用了另一个包含330亿词汇的数据集。我们使用hierarchical softmax，1000维，上下文为整个句子。模型上的结果，准确率将达到72%。当我们将训练数据集减小到60亿的词汇量时，得到更低的准确率66%，这意味着，数据量的大小是十分重要的。

为了更深理解，不同模型学到的词向量表示的不同，我们人工检查了不同模型下的不常用短语的最近邻词。如表4，我们展示了这样的一个比较样例。前面的结果是一致的，它展示了可以学到的短语最佳向量表示模型是：hierarchical softmax和subsampling。

<img src="http://pic.yupoo.com/wangdren23/G9KXydvz/medium.jpg">

表4：两个模型下，给定短语，与它们最接近的其它条目

# 5.加法组合

我们展示了由Skip-gram模型学到的词和短语向量表示，它们展示出一种线性结构，这使得使用向量运算来执行精准的analogical reasoing成为可能。有意思的是，我们发现，Skip-gram表示法展示出了另一种线性结构，它可以将词向量进行element-wise加法组成。该现象见表5.

<img src="http://pic.yupoo.com/wangdren23/G9L4mp2b/medium.jpg">

表5：使用element-wise加法的向量组合。使用最好的skip-gram模型得到的， 与该向量和接近的4个接近的tokens

向量的加法属性可以通过对训练目标进行检查来解释。该词向量与softmax非线性的输入存在线性关系。训练出的词向量用来预测句子周围的词，这些向量可以被看成是，用来表示一个词在特定上下文出现中的分布。这些值与由输出层计算的概率的对数(logP)相关，两个词向量的和（sum）与两个上下文分布的乘积（product）相关联。这里的该乘积是AND函数：两个词向量中都分配了高概率的词，也会得到高概率，否则会得到低概率。因而，如果“Vloga River”在相同的句子中，与"Russian"和"river"出现的很频繁，那么两个词向量的和将产生这样的特征向量，它们与"Vloga River"很接近。

# 6.目前的词向量表示的比较

之前，有许多作者在基于词向量的神经网络领域工作，并发表了许多模型，可以用于进一步使用和比较：最著名的有Collober和Weston(2), Turian(17)，以及Mnih和Hinton的(10). 我们从网上下载了这些词向量, [下载地址](http://metaoptimize.com/projects/wordreprs/)。Mikolov(8)已经在词类比任务上评估了这些词向量表示，其中，Skip-gram模型达到了最好的性能和效果。

<img src="http://pic.yupoo.com/wangdren23/G9LnNdqi/medish.jpg">

表6：各种模型比较，空意味着词不在词汇表里.

为了更深地理解学到的向量质量的不同之处，我们提供了表6的比较。这些示例中，Skip-gram模型在一个大的语料上进行训练，可以看到，效果比其它模型好。部分原因是因为模型训练的词料词汇超过300亿个词，是其它数据集的3个数量级。有意思的是，尽管训练集更大，Skip-gram的训练时间复杂度比前面的模型还要短。



# 参考

－ 1.[Domain adaptation for large-scale sentiment classi-
fication: A deep learning approach](http://svn.ucc.asn.au:8080/oxinabox/Uni%20Notes/honours/refTesting/glorot2011domain.pdf)

- 1 Yoshua Bengio, R´ejean Ducharme, Pascal Vincent, and Christian Janvin. A neural probabilistic language
model. The Journal of Machine Learning Research, 3:1137–1155, 2003.
- [2] Ronan Collobert and Jason Weston. A unified architecture for natural language processing: deep neural
networks with multitask learning. In Proceedings of the 25th international conference on Machine
learning, pages 160–167. ACM, 2008.
- [3] Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Domain adaptation for large-scale sentiment classi-
fication: A deep learning approach. In ICML, 513–520, 2011.
- [4] Michael U Gutmann and Aapo Hyv¨arinen. Noise-contrastive estimation of unnormalized statistical models,
with applications to natural image statistics. The Journal of Machine Learning Research, 13:307–361,
2012.
- [5] Tomas Mikolov, Stefan Kombrink, Lukas Burget, Jan Cernocky, and Sanjeev Khudanpur. Extensions of
recurrent neural network language model. In Acoustics, Speech and Signal Processing (ICASSP), 2011
IEEE International Conference on, pages 5528–5531. IEEE, 2011.
- [6] Tomas Mikolov, Anoop Deoras, Daniel Povey, Lukas Burget and Jan Cernocky. Strategies for Training
Large Scale Neural Network Language Models. In Proc. Automatic Speech Recognition and Understanding,
2011.
- [7] Tomas Mikolov. Statistical Language Models Based on Neural Networks. PhD thesis, PhD Thesis, Brno
University of Technology, 2012.
- [8] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations
in vector space. ICLR Workshop, 2013.
- [9] Tomas Mikolov, Wen-tau Yih and Geoffrey Zweig. Linguistic Regularities in Continuous Space Word
Representations. In Proceedings of NAACL HLT, 2013.
- [10] Andriy Mnih and Geoffrey E Hinton. A scalable hierarchical distributed language model. Advances in
neural information processing systems, 21:1081–1088, 2009.
- [11] Andriy Mnih and Yee Whye Teh. A fast and simple algorithm for training neural probabilistic language
models. arXiv preprint arXiv:1206.6426, 2012.
- [12] Frederic Morin and Yoshua Bengio. Hierarchical probabilistic neural network language model. In Proceedings
of the international workshop on artificial intelligence and statistics, pages 246–252, 2005.
- [13] David E Rumelhart, Geoffrey E Hintont, and Ronald J Williams. Learning representations by backpropagating
errors. Nature, 323(6088):533–536, 1986.
- [14] Holger Schwenk. Continuous space language models. Computer Speech and Language, vol. 21, 2007.
- [15] Richard Socher, Cliff C. Lin, Andrew Y. Ng, and Christopher D. Manning. Parsing natural scenes and
natural language with recursive neural networks. In Proceedings of the 26th International Conference on
Machine Learning (ICML), volume 2, 2011.
- [16] Richard Socher, Brody Huval, Christopher D. Manning, and Andrew Y. Ng. Semantic Compositionality
Through Recursive Matrix-Vector Spaces. In Proceedings of the 2012 Conference on Empirical Methods
in Natural Language Processing (EMNLP), 2012.
- [17] Joseph Turian, Lev Ratinov, and Yoshua Bengio. Word representations: a simple and general method for
semi-supervised learning. In Proceedings of the 48th Annual Meeting of the Association for Computational
Linguistics, pages 384–394. Association for Computational Linguistics, 2010.
- [18] Peter D. Turney and Patrick Pantel. From frequency to meaning: Vector space models of semantics. In
Journal of Artificial Intelligence Research, 37:141-188, 2010.
- [19] Peter D. Turney. Distributional semantics beyond words: Supervised learning of analogy and paraphrase.
In Transactions of the Association for Computational Linguistics (TACL), 353–366, 2013.
- [20] Jason Weston, Samy Bengio, and Nicolas Usunier. Wsabie: Scaling up to large vocabulary image annotation.
In Proceedings of the Twenty-Second international joint conference on Artificial Intelligence-Volume
Volume Three, pages 2764–2770. AAAI Press, 2011.