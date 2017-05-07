---
layout: page
title: spark中的LDA
tagline: 介绍
---
{% include JB/setup %}

# 前言

spark mllib主要是根据paper: [Online Learning for Latent Dirichlet Allocation](http://www.cs.columbia.edu/~blei/papers/HoffmanBleiBach2010b.pdf)开发完成的。

LDA是一个主题模型，它可以从文本文档中推断出相应的主题。LDA被认为是一种聚类算法，因为：

- 主题(Topics)与簇中心相关，文档(documents)与数据集中的样本（rows）相关
- 主题与文档都在一个特征空间中存在，其中特征向量是词数的向量(BOW)。
- LDA不使用传统的距离方式来估计簇，LDA使用的函数基于一个关于文档是如何生成的统计模型。

LDA通过 setOptimizer 函数支持不同的推断算法。EMLDAOptimizer 在似然函数上使用 expectation-maximization 算法来学习聚类，并生成合理结果。而OnlineLDAOptimizer则使用迭代的mini-batch抽样来进行在线变分推断（online variational inference），通常对内存更友好。

LDA的输入为：一个文档集合表示的词频向量，使用下列参数(使用builder模式设置)：

- k: 主题数(比如：簇中心数)
- optimizer: 用于学习LDA模型的Optimizer："em", "online"，分别对应EMLDAOptimizer 和 OnlineLDAOptimizer。
- docConcentration: 文档-主题(doc-topic)分布的先验Dirichlet参数。值越大，推断的分布越平滑。
- topicConcentration: 主题-词语(topic-term)分布的先验Dirichlet参数。值越大，推断的分布越平滑。
- maxIterations: 迭代次数的限制
- checkpointInterval: 如果使用checkpointing(在Spark配置中设置)，该参数将指定要创建的checkpoints的频次，如果maxIterations很大，使用checkpointing可以帮助减少在磁盘上shuffle文件的大小，有助于失败恢复。

所有的spark.mllib的LDA模型都支持：

- describeTopics: 返回的主题，是一个关于最重要的term以及对应权重组成的数组。
- topicsMatrix: 返回一个 vocabSize x k 的矩阵，每一列即是一个topic。

注意：LDA仍然是一个正在活跃发展的实验特性。某些特性只在两种优化器/由优化器生成的模型中的其中之一提供。目前，分布式模型可以转化为本地模型，反过来不行。

下面将独立地描述optimizer/model pair。

## EM(Expectation Maximization)

具体实现详见[EMLDAOptimizer](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.clustering.EMLDAOptimizer) 和 [DistributedLDAModel](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.clustering.DistributedLDAModel)

提供给LDA的参数有:

- docConcentration: 只支持对称的先验，因此在提供的k维向量中所有值必须都相等。所有值也必须大于1.0。提供Vector(-1)会导致缺省行为 (均匀K维向量：每个值为(50/k)+1)。
- topicConcentration: 只支持对称的先验，所有值也必须大于1.0。提供-1, 会导致默认值为：0.1 + 1。
- maxIterations: EM迭代的最大次数。

注意：做足够多次迭代是重要的。在早期的迭代中，EM经常会有一些无用的topics，在经过更多次的迭代后，这些topics会有显著提升。通常至少需要不少于20次迭代，50-100次的迭代通常是合理的，具体依赖你的数据集。

EMLDAOptimizer 会产生一个 DistributedLDAModel, 它不只存储要推断的主题，也存储所有的训练语料，以及训练语料中每个文档的主题分布。一个DistributedLDAModel 支持：

- topTopicsPerDocument: 训练语料中每个文档的topN个主题和对应的权重
- topDocumentsPerTopic: 每个主题下的前topN个文档和这些文档中对应的主题权重
- logPrior: 给定超参数docConcentration 和 topicConcentration的情况下，要估计的主题(topics)和文档-主题(doc-topic)分布的对数概率
- logLikelihood: 给定要推断的主题和文档-主题分布的情况下，训练语料的对数似然

## 在线变分贝叶斯(Online Variational Bayes)

由 [OnlineLDAOptimizer](http://spark.apache.org/docs/latest/api/scala/org/apache/spark/mllib/clustering/OnlineLDAOptimizer.html) 和 [LocalLDAModel](http://spark.apache.org/docs/latest/api/scala/org/apache/spark/mllib/clustering/LocalLDAModel.html)来实现。

提供给LDA的参数有：

- docConcentration: 通过输入一个每维度值都等于Dirichlet参数的向量使用不对称的先验，值应该大于等于0 。提供 Vector(-1)为缺省行为( 均匀k维向量，值为(1.0/k) )
- topicConcentration: 只支持对称的先验，值必须大于等于0。提供值-1会使用默认值 (1.0/k)
- maxIterations: 提交的minibatches的最大次数

此外，OnlineLDAOptimizer 接收下列参数：

- miniBatchFraction: 每次迭代使用的语料库抽样比例。缺省为0.05,通常要满足：maxIterations * miniBatchFraction >= 1
- optimizeDocConcentration: 如果设置为true，每次minibatch 之后会对超参数docConcentration (也称为alpha) 执行最大似然估计，在返回的LocalLDAModel使用优化过的docConcentration。
- tau0和kappa: 用作学习率衰减，用 (τ0+iter)^−k 计算，其中iter为当前的迭代次数

OnlineLDAOptimizer 生成一个 LocalLDAModel，它只存储了推断过的主题。一个LocalLDAModel支持：

- logLikelihood(documents): 对于给定的推断的主题，计算所提供的documents的下界
- logPerplexity(documents): 对于给定的推断的主题，计算所提供的documents的困惑度的上界

示例：

下例使用LDA来infer三个主题。输入希望的簇数目。接着输出主题，通过在词上的概率分布表示。

{% highlight scala %}

import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA}
import org.apache.spark.mllib.linalg.Vectors

// Load and parse the data
val data = sc.textFile("data/mllib/sample_lda_data.txt")
val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble)))
// Index documents with unique IDs
val corpus = parsedData.zipWithIndex.map(_.swap).cache()

// Cluster the documents into three topics using LDA
val ldaModel = new LDA().setK(3).run(corpus)

// Output topics. Each is a distribution over words (matching word count vectors)
println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + " words):")
val topics = ldaModel.topicsMatrix
for (topic <- Range(0, 3)) {
  print("Topic " + topic + ":")
  for (word <- Range(0, ldaModel.vocabSize)) { print(" " + topics(word, topic)); }
  println()
}

// Save and load model.
ldaModel.save(sc, "target/org/apache/spark/LatentDirichletAllocationExample/LDAModel")
val sameModel = DistributedLDAModel.load(sc,
  "target/org/apache/spark/LatentDirichletAllocationExample/LDAModel")


{% endhighlight %}

关于更详细的说明，详见代码。这是spark mllib的一个通病（文档不够细，示例不够全）。


参考：

1.[http://spark.apache.org/docs/latest/mllib-clustering.html#latent-dirichlet-allocation-lda](http://spark.apache.org/docs/latest/mllib-clustering.html#latent-dirichlet-allocation-lda)
