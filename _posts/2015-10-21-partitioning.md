---
layout: post
title: Label Partitioning介绍
description: 
modified: 2015-10-21
tags: [ctr]
---

在google 发表的paper: 《Label Partitioning For Sublinear Ranking》中，有过介绍：

# 一、介绍

许多任务的目标是：**对一个巨大的items、documents 或者labels进行排序，返回给其中少量的top K给用户**。例如，推荐系统任务，比如：通过协同过滤，需要对产品（比如：电影或音乐）的一个大集合，根据给定的user profile进行排序。对于注解任务（annotation），比如：对图片进行关键词注解，需要通过给定的图片像素，给出的可能注解的一个大集合进行排序。最后，在信息检索中，文档的大集合（文本、图片or视频）会基于用户提供的query进行排序。该paper会涉及到实体(items, documents, 等)，被当作labels进行排序，所有上述的问题都看成是**标签排序问题（label ranking problem）**。在机器学习界中，提出了许多强大的算法应用于该领域。这些方法通常会通过对每个标签（label）依次进行打分（scoring）后根据可能性进行排序，可以使用比如SVM, 神经网络，决策树，其它流行方法等。我们将这些方法称为**标签打分器（label scorers）**。由于对标签打分是独立进行的，许多这些方法的开销与label的数量上是成线性关系的。因而，不幸的是，当标签数目为上百万或者更多时变得不实际，在serving time时会很慢。

本paper的目标是：当面临着在现实世界中具有海量的labels的情况时，让这些方法变得实用。这里并没有提出一种新方法来替换你喜欢的方法，**我们提出了一个"wrapper"方法**，当想继续维持（maintaining）或者提升(improve) accuracy时，这种算法能让这些方法更容易驾驭。(**注意，我们的方法会改善测试时间，而非训练时间，作为一个wrapper方法，在训练时实际不会更快**)

该算法**首先会将输入空间进行划分**，因而，任意给定的样本可以被映射到一个分区（partition）或者某分区集合(set of partitions)中。在每个分区中，只有标签的一个子集可以由给定的label scorer进行打分。我们提出该算法，用于优化输入分区，以及标签如何分配给分区。两种算法会考虑选择label scorer，来优化整体的precision @ k。我们展示了如何不需考虑这些因素，比如，label scorer的分区独立性，会导致更差的性能表现。这是因为当标签分区时（label partitioning），对于给定输入，最可能被纠正（根据ground truth）的是labels的子集，原始label scorer实际上表现也不错。我们的算法提供了一个优雅的方式来捕获这些期望。

本paper主要：

- 引入通过label partitioning，在一个base scorer上进行加速的概念
- 对于输入划分(input partitioning)，我们提供了一个算法来优化期望的预测（precision@K）
- 对于标签分配（label assignment），我们提供了一个算法来优化期望的预测（precision@K）
- 应用在现实世界中的海量数据集，来展示该方法

# 二、前置工作

有许多算法可以用于对标签进行打分和排序，它们与label set的size成线性时间关系。因为它们的打分操作会依次对每个label进行。例如，one-vs-rest方法，可以用于为每个label训练一个模型。这种模型本身可以是任何方法：线性SVM，kernel SVM，神经网络，决策树，或者其它方法。对于图片标注任务，也可以以这种方法进行。对于协同过滤，一个items的集合可以被排序，好多人提出了许多方法应用于该任务，也是通常依次为每个item进行打分，例如：item-based CF，latent ranking模型(Weimer et al.2007)，或SVD-based系统。最终，在IR领域，会对一个文档集合进行排序，SVM和神经网络，以及LambdaRank和RankNet是流行的选择。在这种情况下，不同于注解任务通常只会训练单个模型，它对输入特征和要排序的文档有一个联合表示，这样可以区别于one-vs-test训练法。然而，文档仍然会在线性时间上独立打分。本paper的目标是，提供一个wrapper方法来加速这些系统。

有许多算法用来加速，这些算法取决于对输入空间进行**hashing**，比如通过**局部敏感哈希**(LSH:  locality-sensitive hashing)，或者通过**构建一棵树**来完成。本文则使用分区的方法来加速label scorer。对于该原因，该方法可以相当不同，因为我们不需要将样本存储在分区上（来找到最近邻），我们也不需要对样本进行划分，而是对label进行划分，这样，分区的数目会更小。

在sublinear classification schemes上，近期有许多方法。我们的方法主要关注点在ranking上，而非classification上。例如：label embedding trees（bengio et al.,2010）可以将label划分用来正确分类样本，(Deng et al.,2011)提出一种相似的改进版算法。其它方法如DAGs，filter tree, fast ECOC，也主要关注在快速分类上。尽管如此，我们的算法也可以运行图片标注任务。

# 3.Label Partitioning

给定一个数据集: pairs \$(x_i, y_i), i=1, ..., m \$. 在每个pair中，\$ x_i \$是输入，\$ y_i \$是labels的集合（通常是可能的labels D的一个子集）。**我们的目标是：给定一个新的样本 \$ x^{*} \$, 为整个labels集合D进行排序，并输出top k给用户，它们包含了最可能相关的结果**。注意，我们提到的集合D是一个"labels"的集合，但我们可以很容易地将它们看成是一个关于文档的集合（例如：我们对文本文档进行ranking），或者是一个items的集合（比如：协同过滤里要推荐的items）。在所有情况下，我们感兴趣的问题是：D非常大，如果算法随label集合的size规模成线性比例，那么该算法在预测阶段并不合适使用。

假设用户已经训练了一个label scorer: \$f(x,y)\$， 对于一个给定的输入和单个label，它可以返回一个real-valued型的分值(score)。在D中对这些labels进行ranking，可以对所有\$ y \in D\$，通过简单计算f(x,y)进行排序来执行。**这对于D很大的情况是不实际的**。再者，在计算完所有的f(x,y)后，你仍会另外做sorting计算，或者做topK的计算（比如：使用一个heap）。

我们的目标是：给定一个线性时间(或更差)的label scorer: f(x,y)，能让它在预测时更快（并保持或提升accuracy）。我们提出的方法：label partitioning，有两部分构成：

- (i)**输入分区（input partititoner）**: 对于一个给定的样本，将它映射到输入空间的一或多个分区上
- (ii)**标签分配（label assignment）**: 它会为每个分区分配labels的一个子集

对于一个给定的样本，label scorer只会使用在相对应分区的labels子集，因此它的计算更快。

在预测时，对这些labels进行ranking的过程如下：

- 1.给定一个测试输入x，input partitioner会将x映射到partitions的某一个集合中： \$ p=g(x) \$
- 2.我们检索每个被分配到分区 \$ p_j \$上的标签集合(label sets)：$$ L = \bigcup_{j=1}^{|p|} \mathscr{L}_{p_j} $$，其中 $$ \mathscr{L}_{p_j} \subseteq D $$是分配给分区 \$ p_j \$的标签子集。
- 3.使用label scorer函数\$ f(x,y) \$对满足\$ y \in L \$的labels进行打分，并对它们进行排序来产生我们最终的结果

**在预测阶段ranking的开销，已经被附加在将输入分配到对应分区（通过计算\$ p=g(x) \$来得到）上的开销；以及在相对应的分区上计算每个label（计算: \$ f(x,y), y \in L \$）**。通过使用快速的input partitioner，就不用再取决于label set的size大小了（比如：使用hashing或者tree-based lookup）。提供给scorer的labels set的大小是确定的，相对小很多（例如：\$ \|L\| << \|D\| \$），我们可以确保整个预测过程在\$ \|D\| \$上是**亚线性(sublinear)**的。

## 3.1 输入分区（Input Partitioner）

我们将如何选择一个输入分区（input partitioner）的问题看成是：\$ g(x) \rightarrow p \subseteq \mathcal{P} \$，它将一个输入点x映射到一个分区p的集合中，其中P是可能的分区：\$ \mathcal{P} = \lbrace 1,...,P \rbrace \$。g总是映射到单个整数上，因而，每个输入只会映射到单个分区，但这不是必须的。

有许多文献适合我们的input partitioning任务。例如：可以使用最近邻算法作为input partitioner，比如，对输入x做**hashing**（Indyk & Motwani, 1998)，或者**tree-based clustering和assignment** (e.g. hierarchical k-means (Duda
et al., 1995)，或者**KD-trees** (Bentley, 1975)，这些方法都可行，我们只需关注label assignment即可。然而，注意，这些方法可以对我们的数据有效地执行**完全非监督式划分分区（fully unsupervised partitioning）**，但不会对我们的任务的唯一需求考虑进去：即**我们希望在加速的同时还要保持accuracy**。为了达到该目标，我们将输入空间进行分区：让**具有相似相关标签（relevant labels：它们通过label scorer进行高度排序）的相应样本在同一个分区内**。

我们提出了一种**层次化分区（hierarchical partitioner）**的方法，对于：

- 一个标签打分函数（label scorer）：\$f(x,y)\$
- 一个训练集：\$(x_i,y_i), i=\lbrace 1,...,m \rbrace \$，（注：x为input，y为label）
- 之前定义的label集合D

它尝试优化目标：precision@k。对于一个给定的训练样本\$(x_i,y_i)\$以及label scorer，我们定义了：

accuracy的measure（比如：precision@k）为：

$$
\hat{l}(f(x_i),y_i)
$$

以及最小化loss为：

$$
l(f(x_i),y_i)=1-\hat{l}(f(x_i),y_i)
$$

注意，上述的f(x)是对所有labels的得分向量（f(x)与f(x,y)不同）：

$$
f(x)=f_{D}(x)=(f(x,D_1),...,f(x,D_{|D|})))
$$

其中\$ D_i \$是整个label set上的第i个label。然而，为了衡量label partitioner的loss，而非label scorer，我们需要考虑\$l(f_{g(x_i)}(x_i), y_i)\$，该值为ranking时\$x_i\$对应的分区上的label set的loss。比如：\$ f_{g(x)}(x)=(f(x,L_1),...,f(x,L_{\|L\|)})) \$

对于一个给定的分区，我们定义它的整个loss为：

$$
\sum_{i=1}^{m}l(f_{g(x_i)}(x_i),y_i)
$$

**不幸的是，当训练输入分区（input partitioner）时，L(label assignments)是未知的，它会让上述的目标函数不可解(infeasible)**。然而，该模型发生的errors可以分解成一些成分（components）。对于任意给定的样本，如果发生以下情况，它的precision@k会收到一个较低值或是0:

- 在一个分区里，相关的标签不在该集合中
- 原始的label scorer在排第一位的分值就低

当我们不知道label assignment时，我们将会把每个分区上labels的数目限制在一个相对小的数（\$ \|L_j\|<<\|D\| \$）。实际上，我们会将考虑两点来定义标签分区（label partitioner）：

- 对于共享着高度相关标签的样本，应被映射到相同的分区上
- 当学习一个partitioner时，对于label scorer表现好的样本，应被优先(prioritized)处理

基于此，我们提出了方法来进行输入分区（input partitioning）。让我们看下这种情况：假如定义了分区中心(partition centroids) \$c_i, i=1,...,P\$，某种划分，它使用最接近分配的分区：

$$
g(x)=argmin_{i=\lbrace 1,...,P \rbrace} \| x-c_i \|
$$

这可以很容易地泛化到层次化的情况中（hierarchical case），通过递归选择子中心(child centroids)来完成，通常在hierarchical k-means和其它方法中使用。

**加权层次化分区（Weighted Hierarchical Partitioner）** ，这是一种来确保输入分区（input partitioner）对于那些使用给定label scorer表现较好的样本（根据precision）进行优先处理的简单方法。采用的作法是，对每个训练样本进行加权：

$$
\sum_{i=1}^{m}\sum_{j=1}^{P} \hat{l}(f(x_i),y_i)\|x_i-c_j\|^{2}
$$

实际上，一个基于该目标函数的层次化分区（hierarchical partitioner），可以通过一个“加权(weighted)”版本的 hierarchical k-means来完成。在我们的实验中，我们简单地执行一个"hard"版本：我们只在训练样本集合 \$  \lbrace (x_i,y_i): \hat{l}(f(x_i),y_i) \geq \rho  \rbrace \$上运行k-means，取ρ = 1。

注意，我们没有使用 \$ l(f_{g(x_i)}(x_i), y_i) \$, 而是使用\$ l(f(x_i),y_i) \$，但它是未知的。然而，如果\$ y_i \in L_{g(x_i)}\$，则：\$ l(f_{g(x_i)}(x_i), y_i) \leq l(f_D(x_i),y_i) \$，否则，\$ l(f_{g(x_i)}(x_i), y_i)=1\$。也就是说，我们使用的proxy loss，上界逼近真实值，因为比起完整的集合，我们只有很少的label，因而precision不能降低——除非真实label不在分区中。为了阻止后面的情况，我们必须确保具有相似label的样本在同一个分区中，我们可以通过学习一个合适的metrics来完成。

**加权嵌入式分区（Weighted Embedded Partitioners）**, 在上述构建加权层次式分区器（weighted hierarchical partitioner）时，我们可以更进一步，引入约束（constraint）：共享着高度相关labels的样本会被映射到同一个分区（partitioner）上。编码这些constraint可以通过一种metric learning阶段来完成(Weinberger et al., 2006).。

接着，你可以学习一个input partitioner，通过使用上面的weighted hierarchical partitioner目标函数，在要学的"embedding"空间上处理：

$$
\sum_{i=1}^{m} \sum_{j=1}{P} \hat{l}(f(x_i),y_i)||Mx_i-c_j||^2
$$

然而，一些label scorer已经学到了一个latent "embedding" space。例如，SVD和LSI等模型，以及一些神经网络模型（Bai et al., 2009). 在这样的case中，你可以在隐空间(latent space)上直接执行input partitioning，而非在输入空间上；例如：如果label scorer模型的形式是：\$ f(x,y)= \Phi_{x}(x)^T \Phi_{y}(y) \$，那么partitioning可以在空间 \$ \Phi_x(x) \$上执行。这同样可以节省计算两个embeddings（一个用于label partitioning，一个用于label scorer）的时间，在特征空间中的进一步分区则为label scorer调整。

## 3.2 Label Assignment

本节将来看下如何选择一个L（label assignment）。

- 训练集\$ (x_i,y_i), i=1,...,m \$，label set为：D
- input partitioner: g(x)，使用之前的方式构建
- 线性时间label scorer: f(x,y)

**我们希望学到label assignment: \$ L_j \subseteq D \$，第j个分区对应的label set**。我们提出的label assignment方法会应用到每个分区中。首先，来考虑下优化precision@1的情况，这种简化版的case中，每个样本只有一个相关的label。这里我们使用索引t来索引训练样本，相关的label为\$ y_t \$。我们定义：\$ \alpha \in \lbrace 0,1 \rbrace^{\|D\|}\$，其中\$ \alpha_{i} \$决定着一个label \$ D_i \$是否会被分配到该分区上（\$ \alpha_{i}=1 \$），或不分配（\$ \alpha_{i}=0 \$）。这里的\$ \alpha_{i} \$就是我们希望优化的变量。接下去，我们通过给定的label scorer对rankings进行编码：

- \$ R_{t,i} \$是对于样本t的label i的rank分值：

$$
R_{t,i}= 1 + \sum_{j \neq i}\delta(f(x_t,D_j)>f(x_t,D_i))
$$

- \$ R_{t,y_t} \$是样本t的true label的rank分值

我们接着将需要优化的目标函数写出来：

$$
max_{\alpha} \sum_{t} \alpha_{y_t}(1 - max_{R_{t,i}<R_{t,y_t}} \alpha_i)
$$

...(1)

服从：

$$
\alpha_{i} \in {0,1} 
$$

...(2)

$$
| \alpha | = C 
$$

...(3)

其中，C是分配给该分区的label数。对于一个给定的样本t，为了最大化precision@1,需满足两个条件：

- (1) true label必须被分配给该分区 
- (2) true label必须是所有被分配labels上排序分值最高的

我们可以看到，等式1可以精确计算precision@1，因为项\$ \alpha_{y_t} \$和\$ (1-max_{R_{t,i}<R_{t,y_t}} \alpha_{i}) \$ 会对这两个条件各自进行measure。我们的目标函数会统计训练样本数precision@1。

有意思的是，注意，label partitioning的性质意味着：

- (i) 如果训练样本t在原始的label scorer上标记不正确，但由于高度不相关的label不会被分配到该分区上，会被label partitioner正确标注
- (ii) 原始的label scorer可以正确标注样本，但由于相关的label没有被分配到该分区上，会被label partitioner标注不正确

该优化问题，尽可能地将多个相关的label放在同一分区中，并且尽可能消除尽可能混淆的labels（高排序值但不正确），如果通过移除它们，更多的样本会被正确标注。如图1所示：

<img src="http://pic.yupoo.com/wangdren23/GUC322fl/medish.jpg">

图1: 如何从D中选择2个labels的label assignment问题，只考虑它的precision@1。这里的\$ R_i \$是样本排序后的labels（粗体为true labels）。当选择为sky时，会正确预测样本1和2；而对于样本3-5，sky比true labels的排序还要高。最优的选择是car和house，它们在样本3-5中可以被正确预测，**因为所有有更高排序但不相关labels（higher-ranked irrelevant labels）会被抛弃掉**。这种选择问题就是我们在label assignment任务中要面临的挑战。

不幸的是，等式2的二元限制（binary constraint）致使等式(1)的最优化变得很难，但我们可以将约束放松些：

$$
max_{\alpha} \sum_{t} \alpha_{t_t} (1 - max_{R_{t,i} < R_{t, y_t}} \alpha_i) ,  0 \leq \alpha_i \leq 1
$$

...(4)

\$ \alpha \$的值不再离散（discrete），我们不会使用等式（3)的约束，但在训练后会对连续值\$ \alpha_{i}\$做排序，会采用最大的C label作为分区的成员。

我们将上述泛化成pricision@k（k>1）的情况。如果至少一个“不相关(violating)”的label排在相关label之上，我们必须统计排在相关label之上的violations的数目。回到未放松约束的最优化问题上，我们有：

$$
max_{\alpha} \sum_{t} \alpha_{y_t} (1 - \Phi( \sum_{R_{t,i} < R_{t,y_t}} \alpha_{i}))
$$

...(5)

服从：

$$
\alpha_i \in \lbrace 0, 1 \rbrace, |\alpha| = C
$$

...(6)

这里对于precision@k的优化，如果 r<k，我们可以简单地取\$ \Phi(r) = 0 \$，否则取1。

我们已经讨论了具有一个相关标签的情况，但在许多情况下，样本具有多个相关标签的情况是很常见的，它可以使得loss的计算变得稍微更具挑战性些。我们回到precision@1的情况。在这种情况下，原始的目标函数（等式（1））将返回为：

$$
max_{\alpha}^{} \sum_{y \in y_t} a_y (1 - max_{R_{t,i} < R_{t,y}} \alpha_i)
$$

...(7)

服从：

$$
\alpha_{i} \in \lbrace 0, 1 \rbrace, |\alpha|=C
$$

...(8)

这里，\$ y_t \$包含着许多相关标签 \$ y \in y_t \$，如果它们中的所有都是排在前面的（top-ranked），那么会得到一个precision@1为1,这样我们可以取 \$ max_{y \in y_t}\$

我们可以结合等式(5)和等式(7)来形成一个关于precision@k的cost function，用于multi-label的训练样本上。为了更适合优化，我们使用一个sigmoid来将在等式(7)中的约束\$max_{y \in y_t}\$放松到一个均值和近似值 \$ \Phi(r) \$：

$$
\Phi(r) = \frac{1}{1+e^{(k-r)}}
$$

我们的目标接着变成：

$$
max_{\alpha} \sum_{t} \frac{1}{|y_t|} \sum_{y \in y_t} \alpha_y(1-\Phi(\sum_{R_{t,i}<R_{t,y)}} \alpha_i))
$$

...(9)

服从：

$$
0 \leq \alpha_i \leq 1
$$

...(10)

对于单个样本，期等的目标是一个相关label出现在top k中。然而，当penalty不会影响真实排序位置的情况下不成立（例如：我们原始的cost等价于在位置k+1的排序，或者在位置\$\|D\|\$的位置）。早前我们希望那些label scorer的执行很差的样本降低其重要性。为了达到该目的，我们引入了一个带加权项（term weighting）的样本，通过使用原始label scorer得到的相关label排序的反序来实现，等式(4)和等式(9)变为：

$$
max_{\alpha} \sum_{t} \frac{\alpha_{y_t}}{w(R_{t,y_t})}(1 - max_{R_{t,i} < R{t,y_t}} \alpha_i)
$$

$$
max_{\alpha} \sum_{t} \frac{1}{|y_t|} \sum_{y \in y_t} \frac{a_y}{w(R_{t,y})} (1 - \Phi( \sum_{R_{t,i} < R_{t,y}} \alpha_i))
$$

这里我们作了简化：\$w(R_{t,y}) = (R_{t,y})^{\lambda}, \lambda \geq 0 \$，在我们的试验中，设置\$\lambda=1\$（该值越高会抑制具有更低排序的相关label的样本）。这些等式表示了label assignment objective的放宽条件版本的最终形式，可以使用SGA（随机梯度上升：A: ascent）进行优化。

**最优化注意事项(Optimization Considerations)** 我们考虑这样的情况，被选中的输入分区g(x)，表示每个输入x映射到单个分区上。每个分区的label assignment问题是独立的，这允许它们可以并行的求解（例如：使用MapReduce框架）。为了进一步减小训练时间，对于每个分区我们在完整label set上的一个子集上进行优化（例如：选择 \$ \hat{D} \subseteq D, C < \|\hat{D}\| < \|D\| \$）。对于每个分区，我们选择\$ \hat{D} \$：它是在该分区的训练样本中使用原始label scorer进行排序的最高排序的相关label。在所有的实验中，我们设置\$ \| \hat{D} \| = 2C \$。注意，在我们的实验中，我们发现设置成\$ \| \hat{D} \| = 2C \$后减少参数集的size，影响可忽略不计。原因是，任何分区中在任何训练样本中，在D中大部分labels不会作为相关labels出现。因为这样的labels不会接受任何正值的梯度更新。

**统计Heuristic baseline** 通过比较我们提出的label assignment的最优化，在我们的实验中，我们也考虑了一个更简单的Heuristic：只考虑等式(1)的第一项，例如：\$ max_{\alpha} \sum_{t} \alpha_{t_t}\$。这种情况下，最优化可以简化为：只需统计在分区中的每个true label的出现次数，并让C保持为最多的labels。这种基于统计的assignment提供了一个很好的baseline，来对比我们提出的优化。



# 4.实验

## 4.1 图像注解

首先使用ImageNet数据集来测试图片注解任务。ImageNet是一个很大的图片数据集，它将人口验证通过的图片与WordNet的概念相绑定。我们使用Spring 2010的版本，它具有9M的images，我们使用：10%用于validation, 10%用于test，80%用于training。该任务会对15589个可能的labels做rank，它们的范围从animals(“white
admiral butterfly”)到objects(“refracting telescope”).

...【略】


## 4.2 视频推荐

从一个大型在线视频社区给用户推荐视频。上百万最流行的视频被认为是集合D，我们的目标是，对一个给定用户排序这些视频，并提供给该用户相关的视频。训练数据的格式中每个训练pair都基于一个匿名用户。对于每个用户，**输入\$x_i\$是表示他的偏好的一个特征集合**。这些特征通过聚合每个用户所感兴趣的所有视频的主题来生成。这些主题集合接着被聚类成各特征列。有2069个这样的聚类特征列（clusters）来表示用户，其中任何时候有10个聚类特征列是表示活跃的（意思：每个用户大致都有10个以上的特征）。**label \$y_i\$是已知相关视频的一个集合**。该数据集包含了1亿的样本，每个样本具有2069个输入特征，平均接近有10个相关视频。我们设置另外50w的样本用于validation，1M样本用于test。

我们的baseline label scorer \$W_{SABIE}\$在P@10上对比于Naive Bayes，它给出了108%的提升。因而，baseline已经足够强了。我们接着使用hierarchical k-means，它具有10000个分区，以及许多种label assignment set sizes，结果如表2所示。我们的方法可以提速990x倍，而在label scorer上的P@10提升13%。该结果和我们见到的一样重要：我们使用的label scorer是一个线性模型，其中label partitioner在某种程度上是“非线性”的：它可以在输入空间的不同分区上更改label sets——这可以纠正原始scorer的错误（在某种程度上，这有点像个re-ranker）。注意基于最优化的label partitioner比counting heuristic效果要好。

<img src="http://pic.yupoo.com/wangdren23/GULjf8CZ/medish.jpg">

表2

我们的label partitioner被用于视频推荐系统中，用来尝试提升一个比较强的baseline ML系统。在我们的上述实验中使用的是precision，但precision只是一个online metrics，而在观看期视频的ctr作为衡量更好。当在实际系统中评估label partitioner时，它可以在ctr和观看时长（接近2%）上获得极大的提升。注意，我们不会将它与原始的label scorer做比较，那种情况下使用它是不可行的。

# 5.结论

我们提出了一种“wrapper”方法来加速label scoring rankers。它使用一种新的优化法：通过学习一个input partitioning和label assignment，来胜过其它baseline。该结果与原始的label scorer效果相似（或者效果更好），同时运行更快。这使得该技术被用于现实的视频推荐系统中。最终，我们我们觉得提出的label assignment是解决该问题的好方法，input partitioners间的巨大性能差距意味着，将来还有重大问题需要解决。

## 参考

[Label Partitioning For Sublinear Ranking](http://www.thespermwhale.com/jaseweston/papers/label_partitioner.pdf)


