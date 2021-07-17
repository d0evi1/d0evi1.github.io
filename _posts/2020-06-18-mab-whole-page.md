---
layout: post
title: whole page优化介绍
description: 
modified: 2020-06-18
tags: 
---

# 摘要

MAB framework（Multi-Armed Bandit）已经被成功应用到许多web应用中。然而，许多会涉及到内容推荐的复杂real-world应用不能满足传统的MAB setting。为了解决该问题，我们考虑一个ordered combinatorial semi-bandit问题，其中，learner会从一个包含K actions的base set中推荐S个actions，并在S（超过M）个不同的位置展示这些结果。目标是：在事后根据最可能子集和位置来最大化累积回报（cumulative reward）。通过采用minimum-cost maximum-flow network（最小费用最大流网络），基于thompson sampling的算法常用于(contextual)组合问题中，以解决这个组合难题。它可以潜在与whole-page推荐以及任意概率模型一起工作，来说明我们方法的高效性，我们主要关注Gaussian process optimization以及一个contextual setting，其中ctr使用lr进行预测。我们演示了该算法在synthetic Gaussian process问题上的效果，以及在Yahoo!首页的Today Module的大规模新闻推荐数据集进行了验证。

# 1.介绍

我们使用新闻推荐作为示例。图1是一个portal website的snapshot。在该view中，存在6个slots来展示新闻推荐。这些slots在positions、container sizes以及visual appearance上都不同。一些研究表明：用户不会顺序扫描webpages[Liu 2015, Lagun 2014]。如何作出whole-page推荐，也就是说，从一个更大池子中选择6篇文章，并将它们放置在webpage上，这是一个在ranking之外的组合问题（combinatorial problem）。我们的目标是，寻找一个最优的布局配置（layout configuration）来最大化期望的总CTR。这个主题也有相关工作（Wang 2016），然而搜索引擎以real-time方式工作，它们的模型在batch data上训练（而非online learning的方式），因而忽略了exploration/exploitation tradeoff。

一些已经存在的工作使用multi-plays来解决bandits，例如：

- subset regret problem（Kale 2010..）
- batch mode bandit optimization with delayed feedbackes(Desautels 2014)
- ranked bandits(Radlinski 2008)

这类learning问题也被看成是一个combinatorial bandit/semi-bandit （Gai 2013）。然而，我们示例中的复杂combinatorial setting难度超出了这些方法。

为了建模这种场景，我们考虑如下的rodered combinatorial bandit问题。给定最优的context信息，而非选择一个arm，learner会选择一个关于S个actions的subset，并从M个可能位置上在S个不同位置对它们进行展示。我们的新颖性有：

- 1.我们的方法不会求助于其它方法来提供近似解。相反，我们会将该问题通过mcmf network进行公式化，并有效提供精确解（exact solutions）
- 2.据我们所知，我们的模型会处理通用的layout信息，其中positions的数目可以大于选中arms的subset数，例如：S < M.
- 3.我们会使用Thompson sampling作为bandit实例。Thompson sampling的一个优点是，不管随机reward function有多复杂，它在计算上很容易从先验分布进行抽样，并在所抽样参数下选择具有最高reward的action。因此，它可以潜在与任意probabilisitic user click模型一起工作，例如：Cascade Model和Examination Hypothesis。


图1

# 2.问题设定

由于position和layout bias，很合理假设：对于每篇文章和每个position，存在与(content, position) pair相关联的一个ctr，它指定了用户对于在一个特定position上展示的内容的点击概率。在一个序列rounds（$$t=1,2,\cdots, T$$）上，learner需要从关于K个actions的一个base set A中选中S actions来在S（小于M）个不同positions上展示，并接受到一个reward：它是在选中subset中关于(action,position) pair的rewards的总和。对于每个展示的(content, position) pair，所有回报（payoff）会被展示。该feedback model被称为“semi-bandits”（Audiber, 2013）。我们可以根据该方法来展示建模关于selected arms中subset的positions，我们称该setting为“ordered combinatorial semi-bandits”。

**Ordered(Contextual) Combinatorial Bandits**

在每个round t，learner会使用一个（optional）context vector $$x_t$$展示。为了考虑layout信息，会为每个(action, position) pair (k, m)构建一个feature vector $$a_{k,m}$$。该learner会从A中选择S个actions来在S(小于M)个不同positions进行展示。因此，一个合法的combinatorial subset是一个从S个不同actions到S个不同positions的映射；或者更简单地，它是一个one-to-one映射 $$\pi_t : \lbrace 1,2,\cdots, S\rbrace \rightarrow (A, \lbrace 1,2,\cdots, M \rbrace)$$。我们接着将每个$$\pi_t$$看成是一个super arm。该learner接着为每个选中的(action, position) pair接收 reward $$r_{\pi_t(s)} (t)$$。round t的总reward是每个position $$\sum\limits_{s=1}^{S} r_{\pi_t(s)}(t)$$的rewards总和。目标是：随时间最大化expected cumulative rewards $$E[\sum_{t=1}^T \sum_{s=1}^S r_{\pi_t(s)}(t)]$$。

contextual conbinatorial bandits的一个重要特例是，context-free setting：它对于所有t来说，context $$x_t$$是个常量。通过将S, K设置成特殊值，许多已经存在的方法可以被看成是我们的combinatorial bandits setting的特例。例如：S=1等价成传统的contextual K-armed bandits。如果我们将K=1设置成dummy variable，并将N个positions看成是actions，我们的combinatorial bandit问题会退化成为unordered subset regrets问题（Kale 2010）。bandit ordered slate问题以及ranked bandits可以看成是S=M的特例。我们的setting不局限于l2r，足够通用可以对whole-page presentation进行optimize。

# 3.Thompson Sampling

在(contextual) K-armed bandit问题中，在每个round会提供一个最优的context信息x。learner接着会选择一个action $$a \ in A$$并观察一个reward r。对于contextual bandit问题，Thompson Sampling在Bayesian setting中是最容易的。每个过往observation包含了一个三元组$$(x_i, a_i, r_i)$$，以及reward的likelihood function，通过参数形式$$Pr(r \mid a, x, \theta)$$在参数集$$\Theta$$上进行建模。给定一些已知的在$$\Theta$$上的先验分布，这些参数的后验分布基于过往observations通过Bayes rule给出。在每个time step t上，learner会从后验中抽取$$\hat{\theta}^t$$，并基于抽取的$$\hat{\theta}^t$$选择具有最大expected reward的action，如算法1所示。
 

# 4.Orderd Combinatorial Semi-Bandits算法

由于Thompson sampling对于复杂reward functions的灵活性，我们的主要算法思想使用它来进行ordered semi-bandits。

**在每个round t，ordered combinatorial semi-bandit问题涉及到：从一个K actions的集合A中选中S个actions，并在S个不同positions上进行展示，并收到一个reward（所选subset的reward和）**。

一个naive方法是，将每个复杂的组合看成是一个super arm，并使用一个传统的bandit算法，它会在所有super arms上枚举values。由于super arms的数目会快速膨胀，该方法具有实际和计算限制。

假设每个context x以及(action, position) pair $$a_{k,m}$$的reward的likelihood function以$$Pr(r \mid x,a,\theta)$$的参数形式建模。下面三部分会开发thompson sampling的特殊变种，它们可以有效找到最优mapping $$\pi_t^*: \lbrace 1,2, \cdots, S \rbrace \rightarrow (A, \lbrace 1,2,\cdots, M \rbrace) $$，使得：

$$
\pi_t^* \in argmax_{\pi_t} \sum_{s=1}^S E[r \mid a_{\pi_t(s), x_t, \hat{\theta}^t]
$$

...(1)

## 将action selection看成是一个约束优化（constrained optimization）

为了找到最佳的super arm $$\pi_t^*$$，等式(1)中没有enumeration，我们首先定义：每个(action, position) pair的的expected reward为 $$E[r \mid a_{k,m}, x_t, \hat{\theta}^t]$$ 。其中：对于在position $$p_m$$上展示action $$a_k$$，给定context $$x_t$$以及采样参数$$\hat{\theta}^t$$。。。。我们也定义了指示变量$$f_{k,m}$$来表示action $$a_k$$是否会在position $$p_m$$上被展示，$$f_{k,m} \in \lbrace 0, 1 \rbrace$$。我们接着将一个合法的super arm转成数学约束。首先，由于每个action会至多被展示一次，它对应于constraint $$\sum_m f_{k,m} \leq 1, \forall k$$。第二，在同一个position上不能放置两个action，因而我们有$$\sum_k f_{k,m} \leq 1, \forall m=1, \cdots, M$$。最终，会选中S个actions，它等价于$$\sum_k \sum_m f_{k,m} = S$$。在等式(1)中对super arms进行最大化，可以被表示成如下的integer programming：

$$
\overset{f}{max} \sum\limits_{k=1}^K \sum\limits_{m=1}^M f_{k,m} e_{k,m} 
$$

服从：

$$
...
$$

...(2)

总之，integer programming问题不能被高效求解。然而，在下一节所示，给定的公式可以被解释成一个network flow，它遵循多项式时间解。

## Network flow

integer optimization问题(2)可以被看成是一个minimum-cost maximum-flow公式，它的edge costs为$$-e_{k,m}$$，如图2所述。决策变量$$f_{k,m}$$表示flow的量，沿着一个bipartite graph的edges进行转移，具有expected rewards $$e_{k,m}$$。另外，S表示network flow的total size。另外，与biparite graph相连的edges的flow capacity为1，它表示这些edges可以适应一个flow，至多1个unit。另外，我们可以将constraints的最后集合使用连续等价$$f_{k,m} \in [0,1]$$进行relaxing，将(2)的integer programming公式变更成一个linear programming。

**定理1**：一个有向网络（directed network）的node-arc incidence matrix是完全unimodular。

这里，我知道问题(2)在linear programming relaxation中constraints的集合可以被表示成标准形式：$$Ax = b, x \geq 0$$，它具有一个完全unimodular的constraint matrix A。由于一个graph的incidence matrix具有线性独立行（linearly independent rows），S是一个integer，我们知道linear programming relaation(2)的 super arm selection问题会导致一个integer optimal solution $$f^* \in \lbrace 0,1 \rbrace^{K \times M}$$。另外，linear programming问题可以使用interior-point方法在多项式时间求解，因此我们可以有效求解super arm selection问题。请注意，对于min-cost network flow问题的专有算法可以比linear programming方法具有更好的运行时间。然而，这样的专有方法通常不会允许引入额外的constrainints。对于这些原因，我们在实验中使用一个linear programming solver。

## Thompson sampling进行combinatorial semi-bandits



# 参考


- [Efficient Ordered Combinatorial Semi-Bandits for Whole-page Recommendation](http://www.yichang-cs.com/yahoo/AAAI17_SemiBandits.pdf)
