---
layout: post
title: Learning to Blend Rankings介绍
description: 
modified: 2017-10-11
tags: 
---

yahoo在2010年提出的《Learning to Blend Rankings: a Monotonic Transformation to Blend Rankings from Heterogeneous Domains》。

# 介绍

给定一个关于items的集合 $$X=\lbrace x_1, \cdots, x_n \rbrace$$，X的一个ranking是一个关于$$[n]=\lbrace 1, \cdots, n \rbrace$$的排列。在l2r的领域中有大量研究[1,2,5,8,12]。然而，在许多应用中，我们需要将多个**异构领域(heterogeneous domains)**的关于items的rankings集成到一个关于在所有sets中所有items的单个ranking中，比如多种垂直搜索引擎(vertical search engine)：视频搜索、图片搜索、博客搜索等。例如，一个items集合可以是来自Web的文档集合，而另一个可以是来自一个垂直搜索引擎（比如：Blog或News搜索）的文档集合。将来自多个异构领域的rank lists进行合并是一个非平凡问题（non-trivial topic），因为：

- 1) 这些异构集合可以共享一些文档，但很可能也有许多非公共文档
- 2) 异构领域通常具有不同的features和feature-to-relevance相关性(correlations)

以问答网站（Yahoo! Answer）为例，尽管对于普通网站的text matching和click features可以被用于该domain的ranking中，使用这些独一无二的页面结构和用户反馈开发的features，比如：在Yahoo! Ansers中的点赞率(thumbs up ratings)和反馈总数(feedbacks)，在自有domain中的ranking上有大的用处。但Yahoo! Answers和普通网页文档共享的features在两个domains中的相关度上可能有非常不同的相关性。因此，需要使用一种跨domain的统一ranking function，以便在每个私有domain内更好地对文档进行排序，因此需要新的技术：将来自异构domains的文档融合(blend)到单个ranking list中。

我们想强调的是，该问题通常与rank aggregation问题[4,9]是相当不同的，RA问题需要在items的异构集合(homogeneous set)上将不同的rankings进行merge。

我们将来自多个异构域的rank lists进行集成(integration)定义为**一个blending问题**，将以如下方式将learning to blend rankings的问题进行公式化：

- a) 我们具有异构类型的items。每种类型的items在相应domain内都有一个rank order
- b) blending的训练数据的形式为：items sets和它相关的rankings的pairs，pair中的第一个属于items的某一类型(type)，第二者属于items的另一类型(type)。

最优组合排序(optimal combined rankings)是learning to blend的ground truth，可以以如下两个steps生成：

- 1) 为这些rankings中的每个item分配相关度标签，比如：Perfect, Excellent, Good, Fair, Bad (简写为：P/E/G/F/B)
- 2) 根据这些标签将ranking lists进行merge sort

以这种方式进行Blending可以最大化Discounted Cumulative Gain(DCG)，并且**可以为这些rankings保留原排序（ordering）**。

给定训练数据——组合排序（combined ranking）和在私有domain中的rankings，我们希望学到一种**单调递增的转换（在私有domain上的ranking score）**，使得当使用关于(item sets, 相关rankings)的一个新pair时，我们可以使用转换后的ranking scores来生成一个combined ranking。

在本paper中，我们将该问题公式化成**一个二次规划问题(quadratic programming problem)，并学习一个线性单调转换，使得在每个domain中的排序（rank order）保留，以及转换后的分值是可比的**。

# 2.问题公式化

为了设计一个blending转换，我们假设：训练数据是一个包含了pairs$$\lbrace q_i \rbrace_{i=1}^Q$$的集合。在该工作中，我们主要关注以下场景：每个私有的ranking的order会在blending后保留。具有该constraint的Blending非常像**归并排序（merge sorting）**。

出于简洁性，假设我们只有两个rankings。考虑$$q_i$$，我们有：

$$
R_1^i = \lbrace \langle d_{11}^i, r_{11}^i \rangle, \langle d_{12}^i, r_{12}^i \rangle, \cdots, \langle d_{1M}^i, r_{1M}^i \rangle \rbrace \\
R_2^i = \lbrace \langle d_{21}^i, r_{21}^i \rangle, \langle d_{22}^i, r_{22}^i \rangle, \cdots, \langle d_{2N}^i, r_{2N}^i \rangle \rbrace
$$

其中，M和N分别是第一个set和第二个set的items数目，$$d_{1m}^i$$和$$d_{2n}^i$$是items。

在每个domain中的rank order为$$R_1^i$$和$$R_2^i$$，关于rankings的格式我们考虑两种situations：

- 1) 对于一个items集合，只有items的ranking
- 2) 对于一个items集合，每个item都有一个score，items的ranking通过items的scores引出，例如：ranking是通过对items的scores进行sorting获得的

给定一个关于item sets和它相关rankings的pair，我们可以区分三种cases：

- 两个sets都是situation 1)。我们需要学习一个transformation：它可以将一个set中的ranks与另一set的ranks相关联
- 一个set是situation 1)，另一个set是situation 2)。我们需要学习一个transformation：它可以将一个set中的ranks与另一个set中的scores相关联
- **两个sets都是situation 2)。我们需要学习一个transformation：它可以将两个sets中的scores进行校正(calibrate)**

对于在situation 1)中的一个ranking $$r_{1m}^i$$或$$r_{2n}^i$$是它的rank的负数，而在situation 2)中是相应的score。因此，三种cases可以使用一个公式进行表述。

对于$$R_1^i$$和 $$R_2^i$$，我们有：

$$
r_{11}^i \geq r_{12}^i \geq \cdots \geq r_{1M}^i \\
r_{21}^i \geq r_{22}^i \geq \cdots \geq r_{2N}^i
$$

对应于$$R_1^i$$和$$R_2^i$$，我们也具有共M+N items的combined ranking（**出于简洁，这里假设两个list间没有重叠items**）：

$$
R^i = \lbrace \cdots, d_{1m}^i, \cdots, d_{2n}^i, \cdots \rbrace
$$

根据需要，来自两个list的items的原list顺序会被保留。

相应的，我们定义了$$\lbrace (m,n) 1 \leq m \leq M, m \leq n \leq N \rbrace $$两个子集：$$S_i^+$$对应于$$d_{1m}^i$$的rank高于$$d_{2n}^i$$的cases，$$S_i^-$$对应于$$d_{1m}^i$$的rank低于$$d_{2n}^i$$的cases，并定义了：

$$
S^+= \bigcup\limits_{q_i \in Q} S_i^+, S^- = \bigcup_{q_i \in Q} S_i^-
$$

核心问题是，如何从训练数据中自动化学习一个blending transformation。我们提出在$$R_2^i$$中对$$r_{2n}^i \ (n=1, \cdots, N)$$使用一个单调递增函数$$f(\cdot)$$，使得该blending可以基于$$r_{1m}^i$$和$$f(r_{2n}^i)$$。通过这样做，来自每个单独的ranking list的顺序(order)可以被自动保留。$$f(\cdot)$$的学习会最大程度地遵循editorial blending ranking。（假设我们具有X个rankings，$$X \geq 2$$。可以选择其中之一做为参照点，其余$$X-1$$个transformations会被学到）

# 3.算法

## 3.1 我们的算法

我们将transformation learning问题公式化成一个二次规划问题。

$$
min \sum\limits_{k=1}^K \zeta_k^2
$$

服从：

$$
r_{1m}^i \geq f(r_{2n}^i) - \zeta_k  \ \ \ (m,n) \in S_i^+, \\
f(r_{2n}^i) \geq r_{1m}^i - \zeta_k \ \ \ (m,n) \in S_i^-, \\
\zeta_k \geq 0
$$

其中：

- K是来自$$S_i^+$$和$$S_i^-, i=1, \cdots, Q$$ 两者的items的总数目。

如果假设$$f(\cdot)$$是线性的，并且形式为：$$f(x)=\alpha x + \beta$$，上述问题将变为：

$$
\underset {\alpha, \beta, \zeta_k}{min} \sum\limits_{k=1}^K \zeta_k^2 + \lambda_1 \alpha^2 + \lambda_2 \beta^2
$$

...(1)

满足：

$$
r_{1m}^i \geq \alpha r_{2n}^i + \beta - \zeta_k \ \ \ (m,n) \in S_i^+, \\
\alpha r_{2n}^i + \beta \geq r_{1m}^i - \zeta_k \ \ \ (m,n) \in S_i^-, \\
\zeta_k \geq 0, \alpha \geq 0
$$

通过求解以上QP问题，我们可以获取该线性变换（linear transformation）的一个$$(\alpha, \beta)$$ （相同的$$(\alpha, \beta)$$可以被应用到所有queries上）

如果query的分类信息足够，我们也可以为每个query length、或者每种类型的queries学习一个$$(\alpha, \beta)$$。在等式(1)中的constraints会给定相同的权重，它可以进行调整来为更重要的constraints提供更高的weights。其它非线性单调变换在将来的工作中会进行探索。

等式(1)演示了两个domains的思想。**该算法可以轻易地扩展到blend超过两个的rankings上**。给定来自X domains的ranking lists，选择其中一个作为参照点，其余X-1的转换$$(\alpha_1, \beta_1), \cdots, (\alpha_{X-1}, \beta_{X-1})$$。该QP问题的constraints会涉及到来自任意两个domains的item sets的所有pairs，例如：该问题将变为：

$$
\underset{\alpha_v, \beta_v, \zeta_k}{min} \sum_{k=1}^{K} \zeta_k^2 + \lambda_{1,1} \alpha_1^2 + \lambda_{1,2} \beta_1^2 + \cdots + \lambda_{X-1,1} \alpha_{X-1}^2 + \lambda_{X-1,2} \beta_{X-1}^2
$$

符合：

$$
\alpha_u r_{um}^i + \beta_u \geq \alpha_v r_{vn}^i + \beta_v - \zeta_k \ \ \ (m,n) \in \underset{u, v_i}{S^+}, \\
\alpha_v r_{vn}^i + \beta_v \geq \alpha_u r_{um}^i + \beta_u - \zeta_k \ \ \ (m,n) \in \underset{u, v_i}{S^-}, \\
r_{1m}^i \geq \alpha_u r_{um}^i + \beta_u - \zeta_k \ \ \ (m,n) \in \underset{1,u_i}{S^+}, \\
\alpha_u r_{um}^i + \beta_u \geq r_{1m}^i - \zeta_k \ \ \ (m,n) \in \underset{1, u_i}{S^-}, \\
\zeta_k \geq 0, \alpha_v \geq 0, u, v = 1, \cdots, X-1
$$

# 4.实验

## 4.1 数据

我们对提出的算法进行了评估，所使用数据为：使用web搜索结果与Yahoo！Answers domain产生的垂直搜索结果进行blending。1300个queries从一个商业搜索引擎的query logs中抽样得到，800个queries被用于训练，500个用于validation。对于每个query，我们具有两个集合的文档：普通web文档、Yahoo! Answers文档。每个文档会被5种label之一进行标记(label)：Perfect、Excellent、Good、Fair和Bad，以相关度的递减序排列。我们在每个domain上具有预生成的ranking functions，并于rank score $$r_{1m}^i$$或$$r_{2n}^i$$可以通过在每个domain上对相应domain的文档使用ranking function生成。给定$$R_1^i$$和$$R_2^i$$，QP问题的constrains可以通过应用merge-sort到两个rank lists上进行构建，并在web文档和Answers文档间保留paired score perference。

## 4.2 实验

为了评估提出的算法，我们只关注$$f(\cdot)$$是线性变换的简单case，例如：$$f(x)=\alpha x + \beta$$。使用800个queries来学习transformation和500个queries用于validation。

**Baseline方法**

我们对比的该baseline是Naive blending方法，其中$$r_{1m}^i$$和$$r_{2n}^i$$的scores直接拿来比较进行排序。

**评估metrics**

我们上报了广泛使用的相度度指标：Discounted Cumulaive Gain(DCG)。对于一个N个文档的ranked list（N被设置成10, 或者实验中的1），我们使用以下的DCG变种：

$$
DCG_N = \sum\limits_{i=1}^N \frac{G_i}{log_2(i+1)}
$$

其中$$G_i$$表示在position i上分配的label的weights（比如：10表示Perfect match，7表示Excellent match, 3表示Good match等），相关度越高，DCG的值越高。我们使用DCG来表示：在testing queries的集合上的DCG值的平均。

在我们的应用中，目标是将Yahoo! Ansers的文档blend到web rank list中。我们上报了DCG1和DCG10, 如表1中的web rank list和blended list。我们的方法可以观察到有1.18% DCG10和0.9% DCG1增益。两者在统计上都是很大的提升。在我们的应用中，$$\lambda_1, \lambda_2$$的选择不会极大影响我们的实验，我们在实验中使用$$\lambda_1=1, \lambda_2=10$$。该Naive blending方法不会达到任何DCG的提升。这表明，从异构域的rank scores不能直接比较，需要一个blending算法。

也需要计算pair-wise error rate，（例如：item sets的pairs百分比，不能被正确rank）。换句话说，该error rate会衡量在QP问题中有多少constraints不能被满瞳。表2上报了error rate。学到的线性变换给出了一个35%的error rate。因此，我们会研究optimal DCG，螃蟹烧开吃测评发票merge-sort策略获取。

**Blending的上界**

merge-sort的思想是，两个ranking lists可以被认为是最好的DCG10(它可以通过blending获取)，例如：一个blending算法可以达到的上界。我们的测试数据中，最好的DCG 10是7.06. 因此，还有提升的空间。第5节会讨论将来的研究方法。

# 5.相关工作

最近几年，ranking问题被多次表示成一个监督机器学习问题。这些l2r方法可以组合不同类型的features来训练ranking functions。ranking的问题可以被看成是从pair-wise的偏好数据中学习一个ranking function。该思想是，最小化在训练数据中的矛盾对的数目。例如，RankSVM会使用SVM来学习ranking function。RankNet则使用神经网络来梯度下降来获取一个ranking function。RankBoost则使用boosting从一个弱ranking functions集合中来构建一个高效的ranking function。。。

受[10]的启发，我们的算法将一个pairwise ranking问题看成是一个二次规划问题。

# 6.略

# 参考

- 1.[http://yichang-cs.com/yahoo/cikm10_blending.pdf](http://yichang-cs.com/yahoo/cikm10_blending.pdf)