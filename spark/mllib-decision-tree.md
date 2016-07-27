---
layout: page
title: spark decision tree
tagline: 介绍
---
{% include JB/setup %}

# 前言

决策树和它们的ensembles是机器学习中很流行的分类和回归方法。决策树被广泛使用，因为它们很容易解释，处理类别型feature很方便，可以扩展到多分类，不需要进行feature归一化（feature scaling），并且可以捕获到非线性和feature交叉。树的ensumble算法：比如RF和boosting，是它们中的顶级代表。

spark.mllib支持决策树，可用于二分类、多分类以及回归，可以同时使用连续型feature、以及类别型feature。它的实现将数据划分成rows，允许上百万的数据进行分布式训练。

树的ensemble方法会另外介绍。

# 1. 基本算法

决策树是贪婪算法，它会对feature空间执行递归二分。树会为每个叶子划分预测相同的label。每个划分会被贪婪地选择，通过在一个可能的划分集合上选择最好的分割（best split），目标是：最大化树节点的信息增益。换句话说，在每个树节点的划分选择，都会设置：

<img src="http://www.forkosh.com/mathtex.cgi?argmax_x IG(D,s)"> 

其中IG(D,s)是当在数据集D上的一个划分s的信息增益

# 2.节点不纯度与信息增益

节点不纯度(node impurity)用来衡量同一个节点下对应label的同质（homogeneity）情况。当前版本的实现提供了两种impurity分类计算(Gini和entropy)，以及一个回归版本的impurity计算（variance）。

（待插入表）

信息增益，指的是父节点不纯度与两个子节点不纯度的加权和之间的不同。假设在size为N的数据集D上的一个划分s为：<img src="http://www.forkosh.com/mathtex.cgi?D_{left}"> 和<img src="http://www.forkosh.com/mathtex.cgi?D_{right}"> ，其size分别为：<img src="http://www.forkosh.com/mathtex.cgi?N_{left}">和<img src="http://www.forkosh.com/mathtex.cgi?N_{right}">。各自的，信息增益为：

<img src="http://www.forkosh.com/mathtex.cgi?IG(D,s)=Impurity(D)-\frac{N_{left}}{N}Impurity(D_{left})-\frac{N_{right}}{N}Impurity(D_{right})">

# 3.分割候选集

连续feature

对于在单机实现上的小数据集，对于连续feature的split候选集通常是feature的唯一值。一些实现会对feature值进行排序，并接着使用排好序的唯一值作为split侯选集来加快树的计算。

对于大的分布式数据集，对feature值进行排序很昂贵。这种实现会计算一个split侯选集的近似，通过在数据的一个抽样上执行一个分位数计算。排过序的split会创建"bins"，并且最大数目的bin可以通过使用maxBins参数来设定。

注意：bins的数目不能大于实例N的数目（一个特例是：因为缺省的的maxBins值为32）。树算法会在条件不满足时自动减小bins的数目。

# 4.类别型feature

对于类别型feature, 它有M个可能的值（类别），可能出现<img src="http://www.forkosh.com/mathtex.cgi?2^{M-1}-1">的split候选集。对于二分类和回归，我们可以减小split候选集的数目到M-1，通过对类别型feature值的平均label进行排序。例如：对于一个二分类问题，一个类目型feature有三个类别：A,B,C，相应的label的比例为：0.2, 0.6, 0.4，类别feature的顺序为：A, C, B。两个split候选集为：A|C,B和A,C|B，符号|表示split。

在多分类问题上，所有的<img src="http://www.forkosh.com/mathtex.cgi?2^{M-1}-1">种split都有可能。当<img src="http://www.forkosh.com/mathtex.cgi?2^{M-1}-1">比maxBins参数大时，我们会使用一个启发式方法。M种类别feature值通过impurity进行排序，可以考虑的split候选集为:M-1种。

# 5.stopping-rule

当下面的条件满足时，在一个节点上的递归树的构建会停止：

- 1.节点深度等于maxDepth训练参数
- 2.没有split候选集，将导致IG>minInfoGain
- 3.没有split候选集，将产生子节点，每个都有至少minInstancesPerNode个训练实例

# 6.用法tips

新用户可以主要考虑maxDepth参数，以及特定参数问题这一部分。

## 6.1 特定参数问题

下面的参数描述了你想解决的问题。它们可以被指定，但不需要进行调参。

- algo: Classification 或 Regression
- numClasses： 分类的数目（只针对分类问题）
- categoricalFeaturesInfo: 指定了哪个feature是类别型的，以及每个这样的feature有多少个类别值。它给出了一个map[feature索引, feature类别数目]。任何不在该map中的feature都会被当成是连续值。

比如：

- Map(0->2, 4->10)，这个就指定了feature 0有两个类目（0和1），feature4具有10个类别值（值为：0,1,2,3,...,9）。注意：feature索引是从0开始的：feature 0和feature 4分别指的是feature向量中第1个和第5个feature。
- 注意：你不必须指定categoricalFeaturesInfo。算法仍会运行，并得到合理值。然而，如果类别feature被合理指定，执行的性能会更好。

## 6.2 stopping原则

这些参数决定了树什么时候停止构建（添加新节点）。当调整这些参数时，必须仔细对测试数据进行交叉验证以避免overfitting。

- maxDepth: 树的最大深度。树越深，计算效果会越好（可能会得到更高的accuracy），但也可能训练开销越大，更容易导致overfit。
- minInstancesPerNode：对于一个节点的进一步划分，它的每个子节点必须接受至少该数目的训练实例。这也经常在RF中使用，因为它们经常比单个树训练更深。
- minInfoGain：对于一个节点的进一步划分，split必须提升至少比它多。

## 6.3 可调参数

这些参数是可调的。注意：在测试数据上进行交叉验证，以避免overfitting。

**maxBins: bins的数目，离散化连续feature时使用。**

- 增加maxBins可以让算法考虑更多的split候选集，并做出更好的split决策，然而它也会增加计算量和通信开销。
- 注意：对于任意类别feature，maxBins参数必须至少是类别M的最大数。


**maxMemoryInMB: 用于收集足够统计的内存量。**

- 缺省选择256MB，以允许决策算法在大多数情况下正常工作。通过增加maxMemoryInMB，可以导致更快的训练（如果内存允许的话），减少数据间的传输。然而，随着maxMemoryInMB的增大，训练时间有可能会降低，因为每次迭代的通信量必须与maxMemoryInMB成比例。
- 实现细节：对于更快的处理，决策树算法会收集关于将各组节点分成split的统计信息（而非一次一个节点）。在单个group上的节点数目处理，由所需内存决定（区别于每个feature）。maxMemoryInMB参数指定了每个worker可以用于统计的内存限制，以M字节为单位。
- subsamplingRate: 该参数决定了多少部分的训练数据用于进行决策树的学习。该参数与ensembles方法的训练有关（RF和GBT），它可以用于对原始数据进行子抽样。对于训练单个决策树来说，该参数没啥用，因为训练实例的数目通常不是主要约束条件。


- impurity: 不纯度用来衡量在候选的split之间做选择。该参数必须与algo参数匹配。

## 6.4 caching与checkpointing

MLlib 1.2 增加了许多新特性，用来按比例放大到更大（更深）的树和树的ensembles。当maxDepth设置大值时，它对于调整节点ID的caching和checkpointing很管用。当在RF中的numTrees设置的够大时，这些参数也同样很有用。

useNodeIdCache: 如果设置为true，该算法会避免在每次迭代时传当前模型(tree或trees)给executors。

- 对于深的树、RF来说很有用（加速在workers上的计算），减小每次迭代的通信开销
- 实现细节：缺省的，该算法会将当前模型与executors进行通信，以便executors能将训练实例与树节点相匹配。当打开该设置时，该算法将替代缓存这部分信息。

**Node ID缓存会生成一个RDD sequence(每次迭代1个)。**这个long lineage会导致性能问题，但如果在RDD内部进行checkpointing可以减缓这个问题。注意：checkpointing只能用在当设置了useNodeIdCache时。

- checkpointDir: checkpointing 节点id缓存RDD的目录。
- checkpointInterval: checkpointing node ID cache RDDs的频率。设置过低会导致额外写到HDFS上的开销；设置过大会导致另一问题：如果executors，那么RDD就必须重新计算。

## 6.5 规模

计算规模与训练实例的数目、feature的数目、maxBins参数线性近似。通信规模与feature的数目、maxBins的数目线性近似。

该算法实现可以同时读取稀疏和dense数据。然而，它对于sparse输入是没有优化的。

# 示例

分类代码：

impurity=gini, maxTreeDepth=5。计算测试误差accuracy。

examples/src/main/scala/org/apache/spark/examples/mllib/DecisionTreeClassificationExample.scala

回归代码：

impurity=variance，maxTreeDepth=5。计算MSE。

examples/src/main/scala/org/apache/spark/examples/mllib/DecisionTreeRegressionExample.scala

参考：

1.[http://spark.apache.org/docs/latest/mllib-decision-tree.html](http://spark.apache.org/docs/latest/mllib-decision-tree.html)
