---
layout: post
title: alpha-NDCG介绍
description: 
modified: 2019-05-08
tags: 
---


Charles L.等2008在《Novelty and Diversity in Information Retrieval Evaluation》中提出了alpha-NDCG：

# 4.评估框架

probability ranking principle (PRP)构成了信息检索研究的基石。principle如下：

**如果一个IR系统对每个query的response是：关于documents的一个ranking，它的顺序按相关度递减，该系统对用户的整体效率应最大化。**


PRP通常被解释成一个新检索算法：估计每个文档的相关度概率，并进行排序。我们采用一个不同的视角，开始将PRP定义解释成：通过IR系统来最优化一个objective function。

假设：

- q：是一个query. 该query是隐式（implicit）并且是确定的（fixed）。
- u：为一个想根据q获取信息的用户
- d：为一个与u交互可能相关/不相关的 document
- R：是一个关于相关性的二元随机变量

为了应用PRP，我们估计：

$$
P(R=1 | u, d)
$$

在归纳和问答社区常指到“信息点（information nuggets）”。我们：

- 将**用户信息**建模成一个nuggets集合$$u \subseteq N$$, 其中$$N = \lbrace n_1, \cdots, n_m \rbrace$$表示可能的nuggets空间。
- 将**一个文档中出现的信息**会被建模成一个 nuggets集合：$$d \subseteq N$$。

我们解释了一个nugget的概念，将它的通用含义扩展到包含关于一个document的任意二元属性。由于在归纳和问答中很常用，一个nugget可以表示一个信息片段。在QA示例中，一个nugget可以表示成一个答案。然而，一个nugget可以表示其它二元属性，比如：主题。我们也使用nugget来表示某特殊网站一部分的一个页面、一个关于不间断电力供应的特定事实、一个跟踪包裹的表格、或大学主页等。

**如果一个特定document它包含了用户所需信息的至少一个nugget，那么则是相关的**:

$$
P(R = 1|u, d) = P(\exists n_i \ such \ that \ n_i \in u \cup d)
$$

对于一个特定的nugget $$n_i$$：

- $$P(n_i \in u)$$表示用户信息包含$$n_i$$的概率，
- $$P(n_i \in d)$$表示document包含了$$n_i$$的概率

这些概率会被估计，用户信息需要独立于文档，文档需要独立于用户信息。相互间唯一的连接是：nuggets集合。

传统上，对于u和d的特定样本，相应的概率会被估计为0或1；也就是说：$$P(n_i \in u) = 1$$表示：$$n_i$$满足u的认知；否则不满足。相似的，$$P(n_i \in d) = 1$$表示：$$n_i$$可以在d中找到，否则不是。这种传统建模过分强调了这些待评估质量的确定性。采用一个更宽松的视角，可以更好建模真实情况。人工评估者在judgements上有可能是不一致的。来自隐式user feedback的要关评估可能不总是精准的。如果一个分类器被应用到人工羊honr，我们可以利分分类器本身提供的概率。

# 5.  CUMULATIVE GAIN MEASURES

我们接着应用之前章节中的结果，使用nDCG来计算gain vectors。在过去一些年，当有评估分级相关度值(graded relevance values)时，nDCG已经作为标准evaluation measure。由于graded relevance values随前面的框架提出，使用nDCG看起来很合适。

计算nDCG的第一步是，生成一个 gain vector。当我们直接从等式(6)直接计算一个 gain vector时，简化该等式如下：

。。。

丢掉常数$$\gamma \alpha$$，它对于相对值没有影响，我们定义了gain vector G的第k个元素：

$$
G[k] = \sum\limits_{i=1}^m J(d_k, i)(1 - \alpha)^{r_{i, k-1}}
$$

...(7)

对于我们的QA示例，如果我们设置$$\alpha = \frac{1}{2}$$，表2中列出的document会给出：

$$
G = <2, \frac{1}{2}, \frac{1}{4}, 0, 2, \frac{1}{2}, 1, \frac{1}{4}, \cdots >
$$

注意，如果我们设置:$$\alpha = 0$$，并且使用单个nugget来表示主题，等式7的gain vector表示标准的二元相关性。

计算nDCG中的第二步是：计算cumulative gain vector：

$$
CG[k] = \sum\limits_{j=1}^k G[j]
$$

对于我们的QA示例有：

$$
CG = <2, 2\frac{1}{2}, 2\frac{3}{4}, 2\frac{3}{4}, 4\frac{3}{4}, 5\frac{1}{4}, 6\frac{1}{4}, 6\frac{1}{2}, \cdots >
$$

在计算CG后，会应用一个 discount到每个rank上来惩罚在ranking中较低的documents，来反映用户访问到它们需要额外的努力才行。一个常用的discount是$$log_2(1+k)$$，尽管其它discount functions也是可能的，并且有可能更好的反映user effort[20]。我们定义DCG如下：

$$
DCG[k] = \sum\limits_{j=1}^k G[j] / (log_2(1 + j))
$$

对于我们的QA示例：

$$
DCG = <2, 2.315, 2.440, \cdots>
$$

最后一步会通过使用一个“ideal” gain vector来归一化discounted cumulative gain vector。然而，CG和DCG也会被用来直接作为 evaluation measures。在我们的研究中，[3]中表明CG/DCG要比nDCG的用户满意度要好。然而，我们在结果中包含了normalization，后续会更进一步探索。


## 5.1 计算Ideal Gain

理想顺序是：能最大化所有国evels上的cumulative gain。在第3.2节中，我们提出，对表表2的documents的理想顺序背后的直觉。对于这些 documents, 理想的顺序是：a-e-g-b-f-c-h-i-j。相关的ideal gain vector为：

$$
G' = <2, 2, 1, 1/2, 1/2, 1/4, 1/4, \cdots>
$$

ideal gain vector是：

$$
CG' = <2, 4, 5, 5\frac{1}{2}, 6, 6 \frac{1}{2}, 6\frac{1}{2}, \cdots>
$$

ideal discounted cumulative gain vector是：

$$
DCG' = <2, 3.262, 3.762, \cdots>
$$

理论上，ideal gain vector的计算是NP-complete。给定等式7的gain定义，最小化顶点覆盖（minimal vertex covering）可能会减小到计算一个 ideal gain vector。为了转换vertex covering，我们会将每个 vertext映射到一个 document。每个edge对应于一个 nugget，每个nugget会出现在两个 documents中。使用$$\alpha=1$$时的ideal gain vector会提供最小的vertex covering。

实际上，我们发现，使用一个greedy方法来计算ideal gain vector足够了。在每个step上，我们会选择具有最高gain value的document，断开任意的连接（ties）。如果我们从而遇到ties，该方法会计算ideal gain vector。如果ties出现，gain vector可能不是最优的。

## 5.2 $$\alpha$$-nDCG

计算nDCG的最终step是：我们会通过ideal discounted cumulative gain vector来归一化cumulative gain:

$$
nDCG[k] = \frac{DCG[k]}{DCG'[k]}
$$

对于我们的QA示例：

$$
nDCG = <1, 0.710, 0.649, \cdots>
$$

在IR evaluation measures中很常见，nDCG会在一个queries集合上计算，对于单个queries会对nDCG值采用算术平均。nDCG通常会在多个检索depths上上报，与precision和recall类似。

我们的nDCG会通过在等式7中定义的gain value来对novelty进行rewards。否则，它会遵循nDCG的一个标准定义。为了区分nDCG的版本，我们将它称为$$\alpha-nDCG$$，在计算gain vector时会强调参数$$\alpha$$的角色。当$$\alpha=0$$时，$$\alpha-nDCG$$ measure对应于标准nDCG，匹配的nuggets数目会用到graded relevance value。


- 1.[https://plg.uwaterloo.ca/~gvcormac/novelty.pdf](https://plg.uwaterloo.ca/~gvcormac/novelty.pdf)