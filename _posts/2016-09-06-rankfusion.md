---
layout: post
title: rank fusion介绍
description: 
modified: 2016-09-06
tags: 
---

《Web Metasearch: Rank vs. Score Based
Rank Aggregation Methods》提出了rank fusion的问题。

# 1.介绍

rank fusion的问题是，给定多个judges的独立ranking偏好（一个ranking是一个items集合的线性顺序），计算一个“一致的（consensus）”ranking。该ranking fusion问题出现在许多情况下，一个著名的场景是metasearch：metasearch会处理由多个search engines对于一个给定query所返回的多个result lists组合，其中，在result list中的每个item对于各自的search engine是有序的，并且有一个相关分值。

搜索引擎们会帮助人们定位与用户信息相关的信息，但仍有一些不足：

- i) 检索web数据是一个耗时的任务。由于web内容变化很快，每个search engine必须在覆盖率(coverage)间做个trade-off，例如，web文档的数据会与整个web以及更新频率（例如，在完整数据库的后续re-indexing间发生的时间）有关。
- ii) 有许多信息源
- iii) 对于一些搜索引擎，广告商付费越多，在search results上的rank会越高，这被称为：“pay-for-placement”，这在precesion上会有平均损失
- iv) search engines会有作弊（spamming）

由于上述这些限制，必须引入metasearch engines来提升检索效果。

ranking fusion的理想场景是，每个judge（search engine）给出对所有alternative items一个完整顺序。不幸的是，在metasearch中这不太现实，有两个原因：

- i) search engines的coverage是不同的
- ii) search engines会限制访问top 100 or 1000个ranked items

因此，rank aggregation必须可以处理具有有限数目个entires的每个ranking。当然，如果rankings间没有重叠项，。。。设计ranking fusion的一个挑战是，当在每个ranking中的top 几百个／上午个items间存在有限个重合项（但non-trivial）。

。。。

# 2.ranking fusion方法

对于rank-ordered lists进行合并，存在许多方法。最基础的，他们会使用来自ranked lists中已经提供的信息。在大多数情史中，策略依赖于以下信息：

- i) 顺序排序(irdinal rank)：rank list中的一个item分配一个顺序
- ii) 得分(score)：在rank list中分配给每个item的得分

在score-based方法中，items会根据rank lists中的scores进地rank，或者对这些scores做一些转换（transformation）[1,4,8,9,12,13,17,18]；而在rank-based方法中，items会根据rank lists中的rank重新排序，或者对这些ranks做一些转换[1,7,12,19]。另一种正交去重的rank fusion方法则依赖于训练数据（比如：Bayes-fuse方法[1]，或者线性组合方法[17]，或者偏好rank组合方法[8]）。基它一些方法基于ranked items的内容。

## 2.1 前提

首先看一些基本定义。假设：

- U是一个items集合，称为“universe”。
- $$\tau$$是一个rank list，它与U有关，是关于子集$$S \subseteq U $$的一个排序，例如：$$\tau = [x_1 \geq x_2 \geq \cdots \geq x_k$$，对于每个$$x_i \in S$$，$$\geq$$是在S上的顺序关系.

如果 $$i \in U$$出现在$$\tau$$中，写成$$i \in \tau$$，$$\tau(i)$$表示i的position或rank。我们假设$$U = \lbrace 1,2,\cdots, \mid U \mid \rbrace $$，通过对每个$$i \in U$$分配一个唯一的id，没有对普适性有损失。


。。。

## 2.2 Fusion方法

本节中，我们将讲述分析的ranking fusion方法。考虑一个包含了n个rankings的集合$$R=\lbrace \tau_1, \cdots, \tau_n \rbrace$$。假设：

- U表示在$$\tau_1, \cdots, \tau_n$$中的items的union，例如：$$U = \cup_{\tau \in R, i \in \tau} \lbrace i \rbrace $$。
- $$\hat{\tau}$$表示ranking（被称为fused ranking或fused rank list），它是一个rank fusion方法应用到在R中的rank lists之后得到的结果。

为了彻底说明$$\hat{\tau}$$，它足够决定分值$$s^{\tau}(i)$$（称为：fused score）(对于每个item $$i \in U$$)，因为$$\hat{\tau}$$会根据$$s^{\hat{\tau}}$$的值递减进行排序。我们认为：

- 如果两个fused ranking $$\hat{\tau}_1, \hat{\tau}_2$$是**相等的（equal）**，则：$$\hat{\tau}_1 = \hat{\tau}_2$$
- 如果$$\hat{\tau}_1$$和$$\hat{\tau}_2$$是**等价的(equivalent)**，那么对于$$i \in U$$，有$$\hat{\tau}_1(i) = \hat{\tau}_2(i)$$（他们具有相同的顺序）

当然，equality意味着equivalence，但反过来不成立。


### 2.2.1 线性组合

Fox[9]提出的ranking fusion方法基于对每个item的归一化得分(normalised score)采用不加权（unweighted）的min, max或sum。另外，Lee[12]解决了rank取代score的case。两种方法如下：

$$
CombSUM: s^{\hat{\tau}}(i) = \sum\limits_{\tau \in R} w^{\tau}(i) \\
CombMNZ: s^{\hat{\tau}}(i) = h(i, R) \cdot \sum\limits_{\tau \in R} w^{\tau}(i)
$$

从[9,12]的测试结果看，CombMNZ被认为是最好的ranking fusion方法，它的效果要稍好于CombSUM。根据Lee的实验，CombMNZ基于该事实：“不同搜索引擎返回(return)相似的相关文档集合，但检索(retrieve)不同的不相关文档集合”。确实，CombMNZ组合函数会对常见文档进行大的加权。

。。。

#### 2.2.1.1 Borda Count

Borda的方法[2,14]是一个基于ranks的voting方法，例如，一个candidate出现在每个voter的ranked list中，则会为相应的ranks分配一个weight。计算上非常简单，因为实现是线性的。Borda Count (BC)方法在rank fusion问题中被引入，并以如下方式运转。每个voter会对一个关于c个candidates的固定集合以偏好顺序进行rank。对于每个voter，会给top ranked candidate为c分，第二个ranked candidate会得到c-1分，以此类推。在unranked candidates间则均匀分配剩余得分。candidates会以总分(total points)的递减顺序进行rank。正式的，该方法与以下过程等价：对于每个item $$i \in U$$以及rank list $$\tau \in R$$，Borda归一化权重为$$w^{\tau}(i)$$（定义1）。fused rank list $$\hat{\tau}$$根据Borda score $$s^{\hat{\tau}}$$的顺序排序，其中在$$\hat{\tau}$$中的一个item $$i \in U$$被定义为：

$$
s^{\hat{\tau}}(i) = \sum\limits_{\tau \in R} w^{\tau}(i)
$$

...(6)

相应的，BC等价于CombSUM方法组合上Borda rank normalisation，（例如：$$\sum.b.0$$）。Aslam[1]也考虑过一种Weighted Borda Count，其中，在对归一化Borda weights求和的部分，会使用这些weights的一个线性组合。

### 2.2.2 Markovchain-based方法

[7]中提出一种关于rank fusion的有意思的方法，它基于Markov chains实现。一个系统的一个(齐次：homogeneous)Markov chain可以通过一个状态集$$S=\lbrace 1,2, \cdots, \mid S \mid \rbrace$$以及一个$$\mid S \mid \times \mid S \mid $$的非负随机矩阵M（例如：每行的求和为1）来指定。该系统从S中的某个状态开始，并在每个step时会从一个state转移到另一个state。转移(transition)通过矩阵M来指导：在每个step上，如果**系统从状态i移到状态j的概率为$$M_{ij}$$**。如果给定当前状态作为概率分布，下一状态的概率分布通过表示当前状态的该vector与M相乘得到。总之，系统的起始状态（start state）根据在S上的一些分布x被选中。在m个steps后，该系统的状态会根据$$xM^m$$进行分布。在一些条件下，不需要考虑**起始分布x**，该系统最终会达到一个唯一确定点（状态分布不再变化）。该分布称为“**稳态分布(stationary distribution)**”。该分布可以通过M的左主特征向量（principal left eigenvector）y给出，例如：$$yM = \lambda y$$。实际上，一个简单的power-iteration算法可以快速获得关于y的一个合理近似。y中的entries定义了在S上的一个天然顺序。我们称这样的顺序为**M的马尔可夫排序（Markov chain ordering）**。

对rank fusion问题使用Markov chains如下所示。**状态集合(State Set) S对应于包含待排序(rank)的所有candidates的list**（例如：在$$R=\lbrace \tau_1, \cdots, \tau_2 \rbrace$$中的所有items的集合）。**在M中的转移概率在某种程度上依赖于$$\tau_1, \cdots, \tau_n$$，待估计的$$\hat{\tau}$$是在M上的Markov chain ordering**。下面，[7]提出了一些Markov chains(MC):

- $$MC_1$$: 如果当前state为item i，那么，下一state从对应rank >= item i的所有items j的multiset中均匀选中，例如，从multiset $$Q_i^{C_1} = \cup_{k=1}^n \lbrace j: \tau_k(j) \leq \tau_k(i) \rbrace $$中均匀选中下一state。

- $$MC_2$$：如果当前state为item i，下一state的挑选过程如下：首先从所有包含i的$$\tau_1, \cdots, \tau_n$$中均匀选中一个ranking $$\tau$$，接着从集合$$Q_{\tau,i}^{C_2} = \lbrace \tau(j) \leq \tau(i) \rbrace $$中均匀选中一个item j

- $$MC_3$$：如果当前state为item i，下一state的挑选过程如下：首先从包含i的所有$$\tau_1, \cdots, \tau_n$$中均匀选择一个ranking $$\tau$$，接着均匀选择通过$$\tau$$排序的一个item j。如果$$\tau(j) < \tau(i)$$那么跳到j，否则留在i；

- $$MC_4$$：如果当前state为item i，下一state挑选过程如下：首先从S中均匀选中一个item j。如果$$\tau(j) < \tau(i)$$对于会同时对i和j进行排序的lists $$\tau \in R$$中的大多数都成立，那么跳到j，否则留在i。

注意，Markov chain方法确实只依赖于ranks间的比较，即不考虑scores也不考虑hits。下面是一个示例。

示例1.考虑以下在S={1,2,3}中的三个items的rankings $$R = \lbrace \tau_1, \tau_2, \tau_3 \rbrace $$.

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/8fe86a664e89e5b0946628cc88367ad4539979af81b0f7a96fe43d1515f380abc6e40ded7fa0cddafc6edb876e286b46?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1

它可以展示成关于$$MC_1, MC_2, MC_3, MC_4$$的转移矩阵，分别为：$$M^1, M^2, M^3, M^4$$。

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/9fcac4fcc571e19208c5a25982dbc39bb817b7926719802639aa42657c575a4b8bd6ef92f525f1cb1ba628afcba96bac?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2

下面，我们会为matrix entries展示一些示例计算。记住$$M_{ij}^k$$是给定当前state（item i）到下一state（item j）的概率。

- $$M_{13}^1$$是2/6. 确实，$$Q_1^{C_1}$$是$$\lbrace  1,1,3,1,2,3 \rbrace$$(rank<=item 1的case：1; 1,3; 1,2,3)。因此，在$$Q_1^{C_1}$$中均匀选中某一元素的概率为1/6, 选中item 3的概率为2/6.

- $$M_{21}^2$$是5/18. 均匀选择一个包含item 2的rank list的概率为1/3. 如果$$\tau_1$$被选中，那么$$Q_{\tau_1,2}^{C_2}=\lbrace 2,1 \rbrace$$。因此，选中item 1的概率为1/2. 相似的，有$$Q_{\tau_2,2}^{C_2}=\lbrace 2,1,3 \rbrace$$，以及$$Q_{\tau_3,2}^{C_2}=\lbrace 2, 3 \rbrace$$。因此，$$M_{21}^2 = \frac{1}{3} \cdot \frac{1}{2} + \frac{1}{3} \cdot \frac{1}{3} + \frac{1}{3} \cdot 0 = \frac{5}{18}$$

- $$M_{23}^3$$是2/9. 均匀选中一个包含item 2的概率为1/3. 在一个rank list范围内均匀选择一个item的概率也为1/3. 由于$$\tau_1(3) \nless \tau_1(2), \tau_2(3) < \tau_2(2), \tau_3(3) < \tau_3(2)$$, 因此$$M_{23}^3 = \frac{1}{3} \cdot 0 + \frac{1}{3} \cdot \frac{1}{3} + \frac{1}{3} \cdot \frac{1}{3} = \frac{2}{9}$$。

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/769ccb287bbab5e5cdbce670c737edc61b6d1bf9d255a1ae57042d27382db0fccc88309666f9622aafcbacba81fa2b38?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=3.jpg&amp;size=750">

图3

- $$M_{22}^4$$是1/3. 均匀选中在S中一个item的概率为1/3. 另外，考虑图3中的表：在上表中的每个entry $$a_{ij}$$是满足$$\tau \in R, \tau(j) < \tau(i)$$的lists的count数（例如，有多少rankings满足：item j的rank比item i要好）。由于存在三个lists，因此majority的阀值是2. $$M_{22}^4$$指的是：给定item 2, 在下一step之后我们仍停留在item 2上的概率。由于$$a_{21}, a_{22}, a_{23}$$分别是2､ 0､ 2, 三种情况中有两种会从item 2进行转移，另一种仍会停留在item 2上。相应的$$M_{22}^4$$是1/3.

最终，rank set R的fused rank list $$\hat{\tau}_k$$是在$$M^k, k=1, \cdots, 4$$上的Markov chain ordering。可以看到所有4种情况都是：$$\hat{\tau}=[3 \geq 2 \geq 1]$$。

# 3.实验

## 3.1 数据集

使用Text Retrieval Conference(TREC)数据集。它提供了具有许多rank lists的大的、标准的数据集，准备进行fused。通常，每年会提供一个大的文档数据base S和一个包含 50个querie的list。在ad-hoc和web信息检索大会上，每个系统x会



# 参考

- 1.[http://www.umbertostraccia.it/cs/download/papers/SAC03/SAC03.pdf](http://www.umbertostraccia.it/cs/download/papers/SAC03/SAC03.pdf)
