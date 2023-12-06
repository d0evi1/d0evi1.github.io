---
layout: post
title: google RCR Calibrated Ranking介绍
description: 
modified: 2023-11-5
tags: 
---

google在《Regression Compatible Listwise Objectives for Calibrated Ranking with Binary Relevance》中提出了一种RCR方法。

# 摘要

由于学习排序（ Learning-to-Rank：LTR）方法主要旨在提高ranking质量，因此它们的输出分数在设计上并没有进行比例校准（ scale-calibrated）。这从根本上限制了LTR在**分数敏感应用（score-sensitive applications）**中的使用。虽然一个会结合了回归（regression）和排名目标（ranking objective）的简单多目标方法，可以有效地学习**比例校准分数（scale-calibrated scores）**，但我们认为这两个目标不一定兼容，这使得它们之间的权衡不够理想。在本文中，我们提出了一种实用的**回归兼容排名（RCR：regression
compatible ranking）方法**，实现了更好的权衡，其中排名和回归组件被证明是相互对齐的。虽然同样的思想适用于具有二元（binary）和分级相关性（graded relevance）的排序，但我们在本文中主要关注binary label。我们在几个公共LTR基准测试上评估了所提出的方法，并表明它在回归和排名指标方面始终实现了最佳或有竞争力的结果，并在多目标优化的背景下显著改进了帕累托边界（Pareto frontiers）。此外，我们在YouTube搜索上评估了所提出的方法，并发现它不仅提高了生产环境pCTR模型的ranking质量，还提高了点击预测的准确性。所提出的方法已成功部署在YouTube生产系统中。

# 1.介绍

学习排序（LTR：Learning-to-Rank）旨在从训练数据中构建一个排序器（ranker），以便它可以正确地对未见过的对象进行排序。因此，需要ranker在ranking指标（如NDCG）上表现良好。通常情况下，以排序为中心的pairwise或listwise方法（例如RankNet [3]或ListNet [29]）比采用pointwise公式的回归方法实现更好的排序质量。

另一方面，这些应用中的现代系统具有多个阶段，**下游阶段会消费前面阶段的预测结果。通常希望ranking分数得到很好的校准，并且分布保持稳定**。以在线广告为例，需要对pCTR（预测点击率）模型进行良好的校准，因为它会影响下游拍卖和定价模型[6、16、30]，尽管广告的最终排序对效果来说最为重要。这表明我们希望ranker不仅在排序指标上表现良好，而且在回归指标上也能够将ranker输出分数校准到某个外部尺度上。流行的回归指标：包括用于分级相关性标签（graded relevance labels）的MSE、和用于二元相关性标签（binary relevance labels）的LogLoss。

毫不奇怪，能力强的ranking方法在回归指标上会表现不佳，因为它们的loss函数对于rank-preserving的分数变换是不变的，并且倾向于学习未经比例校准的分数以适应回归目标。此外，这些方法在训练过程中容易出现不稳定，因为所学习的分数可能在连续训练或重新训练中无限发散[30]。这些因素严重限制了它们在分数敏感应用中的使用。因此，我们别无选择，只能退回到regression-only的方法，即使它们在面向用户的排序指标方面不是最优的。

已经证明，标准的多目标方法可以有效地学习用于ranking的比例校准分数（scale-calibrated scores）[16、25、30、31]。然而，我们认为在这种标准的多目标设置中，regression和ranking目标本质上是相互冲突的，因此最佳权衡可能对其中之一都不理想。在本文中，我们提出了一种实用的回归兼容排序（RCR： regression compatible ranking）方法，其中ranking和regression组件被证明是可以相互对齐的。虽然同样的思想适用于具有二元排序和分级相关性排序，但我们在本文中主要关注二元标签（binary label）。在实证方面，我们在几个公共LTR数据集上进行了实验，并表明所提出的方法在regression和ranking指标方面实现了最佳或竞争结果，并在多目标优化的背景下显著改进了帕累托边界。此外，我们在YouTube搜索上评估了所提出的方法，并发现它不仅提高了生产pCTR模型的ranking能力，还提高了点击预测的准确性。所提出的方法已经在YouTube生产系统中得到了完全部署。

# 3.背景

学习排序（LTR）关注的问题是：给定一个上下文，学习一个模型来对一个对象列表进行排序。在本文中，我们使用“query”表示上下文，“document”表示对象。在所谓的“得分和排序(score-and-sort)”设置中，学习一个ranker来为每个doc评分，并通过根据分数对docs进行排序来形成最终的ranked list。

更正式地说，设 $𝑞 \in 𝑄$ 为一个query，$𝑥 \in X$ 为一个doc，则得分函数定义为 $𝑠(𝑞, 𝑥; 𝜽) : 𝑄 × X → R$，其中： 𝑄 是query空间，X 是doc空间，𝜽 是得分函数𝑠的参数。一个典型的LTR数据集𝐷由表示为元组(𝑞, 𝑥, 𝑦) ∈ 𝐷的示例组成，其中𝑞，𝑥和𝑦分别为查询，文档和标签。设$q = {𝑞| (𝑞, 𝑥, 𝑦) ∈ 𝐷}$为由𝐷引导的查询集。设$L_{query(𝜽; 𝑞)}$为与单个查询$𝑞 ∈ 𝑄$相关联的损失函数。根据$L_query$的定义方式，LTR技术可以大致分为三类： pointwise, pairwise和listwise.。

在pointwise方法中，query loss $L_{query}$表示为共享相同query的doc的loss之和。例如，在逻辑回归排名（即使用二元相关性标签的排名）中，每个文档的Sigmoid交叉熵损失（用SigmoidCE表示）定义为：

$$
SigmoidCE(𝑠, 𝑦) = −𝑦 log 𝜎(𝑠) − (1 − 𝑦) log(1 − 𝜎(𝑠))
$$

...(1)

其中：

- $𝑠 = 𝑠(𝑞, 𝑥; 𝜽)$：是query-doc pair（𝑞，𝑥）的预测分数
- $𝜎(𝑠) = (1 + exp(−𝑠))−1$：是Sigmoid函数

在文献[30]中表明，SigmoidCE在比例校准方面是可行的，因为当$𝜎(𝑠) → E[𝑦|𝑞, 𝑥]$时，它会达到全局最小值。

在pairwise方法中，query loss $L_{query}$表示为共享相同query的所有doc-doc pair的损失之和。基本的RankNet方法使用pairwise Logistic loss（用PairwiseLogistic表示）[3]：

$$
PairwiseLogistic(𝑠1, 𝑠2, 𝑦1, 𝑦2) = − I(𝑦2 > 𝑦1) log 𝜎(𝑠2 − 𝑠1)
$$

...(2)

其中:

- 𝑠1和𝑠2是文档𝑥1和𝑥2的预测分数
- I是指示函数
- 𝜎是Sigmoid函数

当$𝜎(𝑠2 − 𝑠1) → E[I(𝑦2 >𝑦1) \mid 𝑞, 𝑥1, 𝑥2]$时，PairwiseLogistic会达到全局最小值，这表明loss函数主要考虑pairwise分数差异，这也被称为平移不变性质（translation-invariant）[30]。

在listwise方法中，query loss $L_{query}$归因于共享相同查询的整个文档列表。流行的ListNet方法使用基于Softmax的Cross Entropy loss（用SoftmaxCE表示）来表示listwise loss[29]：

$$
SoftmaxCE(𝑠_{1:𝑁} , 𝑦_{1:𝑁}) = - \frac{1}{C} \sum\limits_{i=1}^N y_i log \frac{exp(s_i)}{\sum\limits_{j=1}^N exp(s_j)}
$$

...(3)

其中：

- 𝑁是list size
- $𝑠_𝑖$是预测分数
- $𝐶 = ∑_{𝑗=1}^N 𝑦_𝑗$

在【29】中全局最小值将在以下来实现：

$$
\frac{exp(s_i)}{\sum_{j=1}^N exp(s_j)} \rightarrow \frac{E[y_i | q, x_i]}{\sum\limits_{j=1}^N E[y_j | q, x_j]}
$$

...(4)

与PairwiseLogistic类似，SoftmaxCE损失是平移不变（translation-invariant）的，并且可能会根据回归指标给出任意更差的分数。


# 4.REGRESSION COMPATIBLE RANKING

在本节中，我们首先介绍动机，然后正式提出回归兼容排序（RCR）方法。

## 4.1 动机

文献表明，标准的多目标方法可以有效地学习用于排名的比例校准分数[16、25、30]。以逻辑回归排名为例，Yan等人将多目标损失定义为SigmoidCE和SoftmaxCE损失的加权和：

$$
L_{query}^{MultiObj} (\theta; q) = (1-\alpha) \cdot \sum\limits_{i=1}^N SigmoidCE(s_i, y_i) + \alpha \cdot SoftmaxCE(s_{1:N}, y_{1:N})
$$

...(5)

其中𝛼 ∈ [0, 1]是权衡权重。为简单起见，我们将这种方法称为SigmoidCE + SoftmaxCE。可以看出，SigmoidCE + SoftmaxCE不再是平移不变的（translation-invariant），并且已被证明对于校准排序（calibrated ranking）是有效的。让我们更深入地了解按照这种简单的多目标公式学习的分数是什么。

给定query 𝑞，设$𝑃_𝑖 = E[𝑦_𝑖 \mid 𝑞, 𝑥_𝑖]$为进一步条件化于文档𝑥𝑖的基本事实点击概率。回想一下，当𝜎(𝑠𝑖) → 𝑃𝑖时，SigmoidCE会达到全局最小值，这意味着我们有以下SigmoidCE的点对点学习目标：

$$
𝑠_𝑖 \rightarrow log 𝑃_𝑖 − log(1 − 𝑃_𝑖)
$$

...(6)

另一方面，当以下公式成立时，SoftmaxCE达到全局最小值：

$$
\frac{exp(𝑠_𝑖)}{\sum\limits_𝑗=1^𝑁 exp(𝑠_𝑗) \rightarrow 𝑃_𝑖 \sum\limits_{𝑗=1}^N 𝑃_𝑗
$$

...(7)

或者等价于：

$$
𝑠_𝑖 \rightarrow log 𝑃_𝑖 − log \sum\limits_{𝑗=1}^N 𝑃_𝑗 + log \sum\limits_{𝑗=1}^N exp(𝑠_𝑗)
$$

...(8)

其中log-∑︁-exp项是未知常数，对最终的SoftmaxCE损失的值或梯度没有影响。

在随机梯度下降的背景下，方程（6）和（8）表明，从SigmoidCE和SoftmaxCE组件生成的梯度将分别将分数推向显著不同的目标。这揭示了标准多目标设置中的两个损失本质上是相互冲突的，将无法找到对两者都理想的解决方案。我们如何解决这个冲突呢？

注意到由于𝜎(𝑠𝑖)在点对点上趋近于𝑃𝑖，如果我们将方程（8）右侧的基本事实概率𝑃𝑖替换为经验逼近𝜎(𝑠𝑖)并删除常数项，我们正在构建一些虚拟的logits：

$$
𝑠_𝑖' \leftarrow log 𝜎(𝑠_𝑖) − log \sum\limits_{𝑗=1}^N 𝜎(𝑠_𝑗)
$$

...(9)

如果我们进一步在新的 $logits 𝑠_i′$上应用SoftmaxCE loss，我们正在建立以下新的列表学习目标：

$$
\frac{exp(𝑠_𝑖')}{\sum\limits_{𝑗=1}^N exp(s_𝑗')} \rightarrow \frac{𝑃_𝑖}{\sum\limits_{𝑗=1}^N 𝑃_𝑗}
$$

...(10)

它等价于：

$$
\frac{𝜎(𝑠_𝑖)}{\sum\limits_{𝑗=1}^N 𝜎(𝑠_𝑗)} \rightarrow \frac{𝑃_𝑖}{\sum\limits_{𝑗=1}^N 𝑃_𝑗}
$$

...(11)

很容易看到，等式（6）自动蕴含等式（11），这意味着，作为点wise回归和列表wise排序目标，它们在实现全局最小值方面是良好对齐的。

## 4.2 主方法

受上述启发性示例的启发，我们首先定义一种新的列表交叉熵损失（ListCE），如下所示。

定义1：设𝑁为列表大小，𝑠1:𝑁为预测分数，𝑦1:𝑁为标签。设𝑇(𝑠)：R → R+为分数上的非降变换。使用变换𝑇的列表交叉熵损失定义为：

$$
ListCE(𝑇 , 𝑠_{1:𝑁}, 𝑦_{1:𝑁}) = − \frac{1}{𝐶} \sum\limits_{𝑖=1}^N 𝑦_𝑖 log \frac{𝑇(𝑠_𝑖)}{\sum\limits_{𝑗=1}^N 𝑇(𝑠_𝑗)}
$$

...(12)

其中，$𝐶 = \sum\limits_{𝑗=1}^N 𝑦_𝑗$是一个归一化因子。

在本文的范围内，我们可以交替使用带有变换𝑇的列表交叉熵损失ListCE(𝑇 )，或者在没有歧义的情况下使用ListCE。我们立即得到以下命题。 

命题1：ListCE(exp)简化为SoftmaxCE。 

命题2：当满足以下条件时，ListCE(𝑇)可以达到全局最小值：

$$
\frac{𝑇(𝑠_𝑖)}{\sum\limits_{𝑗=1}^N 𝑇 (𝑠_𝑗) \rightarrow \frac{E[𝑦_𝑖 |𝑞, 𝑥_𝑖]}{\sum\limits_{𝑗=1}^N E[𝑦_𝑗 |𝑞, 𝑥_𝑗]
$$

...(13)

证明。设$\bar{𝑦} = E[𝑦|𝑞, 𝑥]$为查询-文档对(𝑞, 𝑥)的期望标签。在$(𝑥, 𝑦) \in 𝐷$上应用ListCE损失等价于在期望上将其应用于(𝑥,𝑦)。给定变换𝑇和预测分数𝑠1:𝑁，其中$𝑝𝑖 = \frac{𝑇(𝑠𝑖)/Í𝑁 𝑗=1 𝑇 (𝑠𝑗)$，我们有：

$$
ListCE(𝑇 , 𝑠_{1:𝑁}, 𝑦_{1:𝑁}) = \frac{1} {\sum\limits_{𝑗=1}^N 𝑦_𝑗} \sum\limits_{i=1}^N \bar{y_i} log 𝑝_i
$$

...(14)

满足：$\sum_{i=1}^N p_i = 1$.

接着构建以下的Lagrangian的公式化：

$$
L (𝑝_{1:𝑁}, 𝜆) = \frac{1}{\sum\limits_{𝑗=1}^N \bar{𝑦_𝑗} \sum\limits_{i=1}^N \bar{𝑦_𝑖} log 𝑝_𝑖 + \lambda ( \sum\limits_{i=1}^N 𝑝_𝑖 1)
$$

...(15)

找出等式（14）的极值，接着等价于等式（15）的驻点，它满足：

$$
\frac{\partial L (𝑝_{1:𝑁}, 𝜆)}{\partial 𝑝_𝑖} = \frac{\bar{𝑦_𝑖}{𝑝_𝑖 \sum\limits_{𝑗=1}^N \bar{𝑦_j}} + \lambda = 0
$$

...(16)

并且：

$$
\frac{\partial L (𝑝_{1:𝑁}, \lambda)}{\partial \lambda} = \sum\limits_{𝑖=1}^N 𝑝_𝑖 1 = 0
$$

...(17)

注意，等式（16）和（17）给出一个在N+1 unknowns上的关于N+1的系统。很容易看到，相等的解决方案是：

$$
p_i = \frac{\bar{y_i}}{\sum_{j=1}^N \bar{y_j}}
$$

...(18)

并且$$\lambda=1$$。

这意味着唯的的全局极值在：

$$
\frac{𝑇 (𝑠_𝑖)}{\sum_{𝑗=1}^N 𝑇(𝑠_𝑗)} \rightarrow \frac{E[𝑦_𝑖 |𝑞, 𝑥_𝑖]}{\sum\limits_{𝑗=1}^N E[𝑦_𝑗 |𝑞, 𝑥_𝑗]} 
$$

...(19)

很容易验证这个唯一的全局极值归因于全局最小值，这证明了命题。

在逻辑回归排序（logistic-regression ranking）中，所有标签都是二元化的或在[0,1]范围内。一个自然的点对点目标是SigmoidCE损失。使用SigmoidCE作为点对点组件，然后需要使用Sigmoid函数作为变换，以便可以同时进行优化而不产生冲突。 定义2：适用于逻辑回归排名任务（即使用二元相关标签进行排名）中单个查询的回归兼容排名（RCR）损失定义为：

$$
L_{query}^{Compatible} (\theta; 𝑞) = (1 − 𝛼) \cdot \sum\limits_{𝑖=1}^N SigmoidCE(𝑠_𝑖 , 𝑦_𝑖) + \alpha \cdot ListCE(\sigma, 𝑠_{1:𝑁}, 𝑦_{1:𝑁})
$$

...(20)

其中：

- $$\sigma$$是sigmoid funciton

为简单起见，我们将这种方法称为SigmoidCE + ListCE(𝜎)。我们有以下命题： 

- 命题3：当𝜎(𝑠𝑖) → E[𝑦𝑖|𝑞, 𝑥𝑖]时，SigmoidCE + ListCE(𝜎)可以达到全局最小值。 
- 证明：SigmoidCE组件在$𝜎(𝑠_𝑖) \rightarrow E[𝑦_𝑖 \mid 𝑞, 𝑥_𝑖]$时可以达到全局最小值，这意味着：

$$
\frac{𝜎(𝑠_𝑖)}{\sum\limits_{𝑗=1}^N \sigma(𝑠_𝑗)} \rightarrow \frac{E[𝑦_𝑖 |𝑞, 𝑥_𝑖]}{\sum\limits_{𝑗=1}^N E[𝑦_𝑗 |𝑞, 𝑥_𝑗]}
$$

...(21)

它会最小化ListCE(𝜎)在它的全局最小值上。

# 

略

- [https://arxiv.org/pdf/2211.01494.pdf](https://arxiv.org/pdf/2211.01494.pdf)