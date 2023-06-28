---
layout: post
title: prerank rethinking介绍
description: 
modified: 2023-06-17
tags: 
---

# 介绍

在preranking阶段除了SSB问题外，我们也假设：ranking和preranking的目标是不同的。ranking的专业是选择top items，会对preranking outputs内的顺序进行reranking，并决定最终的输出。**而preranking阶段的主要目标是：返回一个最优的无序集合（unordered set），而非一个关于items的有序列表**。基于以上分析，在taobao search上做了在线和离线实验，重新思考了preranking的角色，并重新定义了preranking的两个目标：

- **高质量集合（High quality set）**：通过解决在preranking candidates上的SSB问题，来提升output set的质量
- **高质量排序（High quality rank）**：为了获得ranking在output set上的一致性，确保高质量items可以在ranking中获得更高得分，并被曝光给用户。

然而，同时最大化两个目标是不可能的。第一个目标最重要，是一个preranking模型必须要追求的目标。第二个目标，只要确保output set集合的质量不会下降就能满足它。换句话说，**当模型达到帕累托边界（Pareto frontier），通常在整个output set的质量和它的内部排序（inside rank）间存在一个“跷跷板效应（Seesaw Effect）”**。在不涉及更多在线计算的情况下，当prerank更关注于整个set时，它的output set内的排序会变差。相似的，当拟合ranking并提升它的内部AUC时，整个output set的质量会变差。这也是为什么AUC与在线业务指标不一致的原因。我们会在第4节中详述。

已经存在的离线评估指标（比如：AUC）可以对preranking能力（第二个目标）进行measure。然而，**AUC会衡量一个有序item list的质量，不适合于评估输出的无序集合的质量**。没有一个metric可以有效评估第一个目标。尽管在工业界存在一些研究者，尝试提升output set的质量，他们没有提供一个可靠的离线评估metric来衡量该提升。实际上，大多数公共策略是通过在线A/B testing进行衡量效果提升。然而，在线评估开销巨大，并且时间成本高，因为它通常要花费许多周来获得一个可信的结果。在本文中，**我们提出了一个新的evaluation metric，称为：全场景Hitrate（ASH：All-Scenario Hitrate），用来评估preranking模型的outputs的质量**。通过对在ASH与在线业务指标间的关系进行系统分析，我们会验证该新离线指标的效果。为了达到我们的目标，我们进一步**提出一个基于全场景的多目标学习框架（ASMOL：all-scenario-based multiobjective learning framework），它会显著提升ASH**。令人吃惊的是，当输出上千items时，新的preranking模型效果要好于ranking model。该现象进一步验证了preranking阶段应关注于：**输出更高质量集合，而不是盲目拟合ranking**。在ASH上有提升，会与在线提升相一致，它会进一步验证了：ASH是一个更有效的离线指标，并在taobao search上获得一个1.2%的GMV提升。

总结有三：

- preranking的两个目标的rethinking。
- 提出了一个ASMOL
- 展示了ASH与在线业务指标间的相关性

# 2.相关工作

。。。

# 3.preranking的评估指标

考虑到第1节中提的两个目标，我们首先分析已存在的AUC指标，以及第第3.2/3.3节中的hitrate@k。再者，我们会引入一个新的离线指标，称为ASH@k。在第3.4节中，我们会分析在taobao search中每个stage的当前能力，并使用我们的新指标来演示：如何使用新metric来分析一个multi-stage的电商搜索系统。

## 3.1 问题公式

在本节中，我们首先将preranking问题和数学概念进行公式化。假设：

- $$U = \lbrace u_1, \cdots, u_{\mid U \mid}\rbrace$$：表示用户与它的features一起的集合。User features主要包含了用户行为信息（比如：它的点击items、收藏items、购买items、或者加购物车的items）
- $$Q = \lbrace q_1, \cdots, q_{\mid Q \mid} \rbrace$$：表示搜索queries以及它的相应分段（segmentation）的集合。
- $$P=\lbrace p_1, \cdots, p_{\mid P \mid}\rbrace$$：表示products（items）以及它的features的集合。Item features主要包含了item ID，item统计信息，items卖家等。

$$\mid U \mid, \mid Q \mid, \mid P \mid$$分别是users、queries、items的去重数。

当一个user u提交一个query q时，我们将在matching output set中的每个item $$p_t$$与user u和query q组合成一个三元组$$(u, q, p_t)$$。preranking models会输出在每个三元组上的scores，通常会从matching的output set上根据scores选择topk个items。正式的，给定一个三元组$$(u, q, p_t)$$，ranking model会预估以下的score z：

$$
z = F(\phi(u, q), \psi(p))
$$

...(1)

其中：

- $$F(\cdot)$$是score funciton
- $$\phi(\cdot)$$：user embedding function
- $$\psi(\cdot)$$：item embedding function

在本文中，我们会遵循vector-product-based的模型框架，并采用cosine相似度操作$$F(\cdot)$$。

## 3.2 ranking stage的一致性

考虑ranking系统在工业界的离线指标，AUC是最流行的。以taobao search为例，AUC会通过曝光进行计算。由于taobao search的目标是有效提升交易，ranking stage主要考虑购买作为曝光上的正向行为，并关注一个被购买item的likelihood要高于其它items的序。因此，taobao search中的items的最大数目被设置为：**每次请求10个，我们使用购买AUC（Purchase AUC）at 10（PAUC@10）来衡量排序模型的能力**。作为结果，PAUC@10通常也会被用在preranking中（与ranking中的相似），可以衡量在线排序系统的一致性。

## 3.3 output set的质量

最近的preranking工作很少使用任意metric来评估整个output set的质量。评估output set的一个评估指标是：hitrate@k（或：recall@k），它会被广泛用于matching stage中。hitrate@k表示：**模型对target items（点击或购买）是否在candidate set的top k中**。正式的，一个$$(u, q)$$ pair，它的hitrate@k被定义如下：

$$
hitrate@k = \frac{\sum\limits_{i=1}^k 1(p_i \in T)}{|T|}
$$

...(2)

其中：

- $$\lbrace p_1, \cdots, p_k \rbrace$$：表示了由preranking model返回的top k items，
- T：表示包含了$$\mid T \mid$$个items的target-item set
- $$1(p_i \in T)$$：当$$p_i$$是target set T时为1，否则为0

当在matching stage中使用该metric时，通常用于衡量一个matching模型的output set的质量（例如：在图1中的Matching 1，非在线的所有matching models的整个output set）。作为对比，**当我们通过hitrate@k来评估pre-ranking实验时，离线metrics的结论会与在线业务指标的结论相矛盾**。在进一步分析后，我们发现，在hitrate中选择k是一个non-trivial问题。为了准确评估在preranking stage的output item set的质量，k被假设：等于preranking output set的size $$\mid R \mid$$。然而，由于在preranking output set中的items在在线serving期间可以被曝光和被购买，所有正样本（target items）都会在preranking的在线output set中。**这会造成当$$k=\mid R \mid$$时，在线preranking model的$$hitrate@k \equiv 1$$**。作为结果，离线hitrate@k可以只会衡量在离线模型输出集合与在线输出集合间的不同之处，而非quality。常规的preranking方法，使用$$k << \mid R \mid$$来避免以上问题。**$$k << \mid R \mid$$的缺点是很明显的，因为它不能表示整个preranking output set的质量**。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/5b0eb710ba51654422b83ae24641f735a1adad4bfbe21ca27bc56b1da57a933036cdfcb4f3b9bc68c164ed902eaa63b3?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 在taobao search中的multi-stage ranking系统

在本工作中，我们一个新的有效评估指标，称为ASH@k。为了创建一个真正表示preranking output set的质量的metric，我们**引入来自taobao其它场景（比如：推荐、购物车、广告等）的更多正样本（例如：购买样本）**。由于来自其它场景的一些正样本不会存在于preranking在线输出中，他们可以表示与场景无关的用户偏好。在本case中，即使$$k = \mid R \mid$$，hitrate@k也不会等于1。由于我们更关心taobao search的交易，我们只会使用来自非搜索场景的购买正样本。为了区分在不同正样本hitrate间的不同，我们称：

- ISPH@k：In-Scenario Purchase Hitrate@k，只在**搜索场景**出现的购买样本的hitrate@k
- ASPH@k：All-Scenario Purchase Hitrate@k，即在**其它场景**的购买正样本hitrate@k

接着，我们详述了如何引入来自其它场景的正样本。在评估中的一个正样本是一个关于user, query, item的triple：$$(u_i, q_j, p_t)$$。**然而，在大多数非搜索场景（比如：推荐）不存在相应的query**。为了构建搜索的评估样本，我们需要绑定一个非搜索购买$$(u_i, p_t)$$到一个相应user发起的请求query $$u_i, q_j$$上。假设：

- $$A_u^i$$：表示user $$u_i$$在taobao场景上的购买的target-item set
- $$Q_u$$：表示在taobao搜索中的所有queries user搜索

一个直觉方法是：相同的user在queries和购买items间构建一个Cartesian Product，并使用所有三元组$$(u_i, q_j, p_t)$$作为正样本，其中：$$q_j \in Q_u$$以及$$p_t \in A_u^i$$。然而，它会引入一些不相关的query-item pairs。例如，一个用户可能在taobao search中搜索“iPhone”，并在其它推荐场景购买一些水果。该样本中：将“iPhone”作为一个query，同时将"apple(fruit)"作为一个item，将它作为在taobao search中的一条正样本是不合适的。**为了过滤不相关的样本，我们只能保证相关样本：它的相关分$$(q_j, p_t)$$在上面的边界**。我们称：

- $$q_k$$为对于全场景pair$$(u_i, p_t)$$的一个“相关query”；
- $$p_t$$是一个全场景"相关item"，可以被绑定到in-scenario pair $$(u_i, q_j)$$

再者，我们也会移除重复样本。在该triple中的每个$$(u, p)$$ pair是唯一的，因此，即使$$u_i$$购买了一个$$p_t$$超过一次，我们只会对使用$$(u_i, p_t)$$的triple仅评估一次。同时，如果我们可以发现：在用户购买行为之前，超过一个相关query，则构建不同的triples，比如：$$(u_i, q_1, p_t), (u_i, q_2, p_t), (u_i, q_j, p_t)$$，我们只会保留用于评估的最新q的triple。正式的，与等式2相似，对于每个$$(u_i, q_k)$$ pair，ASPH@k被定义成：

$$
ASPH@k = \frac{\sum_{i=1}^k 1(p_i \in A_u^i)}{| A_u^i |}
$$

...(3)

其中：$$A_u^i$$：表示包含了$$u_i$$在其它场景购买的$$\mid A_u^i \mid$$个items的target-item set，被绑定到query。

## 3.4 在taobao search中的ASPH

我们展示了在pre-ranking model的pre-generation、提出的pre-ranking model、以及ranking model的离线指标，如图2所示。**为了公平对比在pre-ranking stage中的模型能力，所有其它模型都会在该pre-ranking candidates上进行评估**。对于pre-generation pre-ranking model，它会使用与ranking model的相同样本，从图中的$$10^5$$到$$10^1$$可以看出，它的模型能力会弱于ranking model。通过对比，**当k变大时，提出的preranking model在ASPH@k和ISPH@k上会极大优于ranking**。该现象表明：当输出成千上万个items时，提出的preranking模型能力可以胜过ranking。

同时，**在图2中，在ASPH@k的结果和ISPH@k的结果间存在一个巨大差异**。从ISPH@k metric的视角来看，当k小于3000时，ranking model要胜过preranking model，而从ASPH@k指标的视角，只有当k小于2000时，ranking才会胜过pre-ranking model。在第3.3节所述，我们会argue：ISPH@k得分会表示在offline和online sets间的差异，没必要表示offline集合的质量。由于ranking model的得分决定了最终曝光的items，当使用ISPH@k作为评估指标时，ranking model会具有一个巨大优点。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/e3b71bc83edf2f23c462426490558f2431120066c3a61e8417c62952ad1c410bb98d23659397ea1e8ca4ff30a0d599ca?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2 在taobao search中的hitrate@k。下图是对上图的放大

为了进一步验证ASPH@k的效果，我们执行一个在线A/B test，它的pre-ranking output分别为2500和3000个items。如果ISPH@k的评估是有效的，那么输出3000个items的preranking的在线业务指标会更高。如果ASPH@k是有效的，那么该结论是相反的。在线结果表明，对比起3000个items的模型，输出2500的preranking具有0.3%的在线30天A/B的交易GMV提升。**该实验验证了：ASPH@k是要比ISPH@k在离线评估上是一个更可靠的指标**。再者，该实验也表明了preranking可以处理ranking所不能处理的能力，因为reranking output set的size并不是越大越好。相应的，一个preranking应发展它在更高质量outputs上的优点，而不是盲目模拟ranking。

# 4.preranking的优化

尽管模拟ranking会使得一个preranking与ranking更一致，**但它对整体preranking的output set的质量贡献很少**。本节将讨论我们的用来达到preranking目标的最优化技术，包括训练数据构建、全场景labels、loss functions和distillation。我们首先引入ASMOL preranking，接着分析每个部分，最后通过表明它的效果。

## 4.1 多目标学习的整体框架

为了同时提升pre-ranking output set的质量以及与ranking的一致性，我们设计了一个新的we design a novel AllScenario-based Multi-Objective框架。一方面，我们将训练样本从曝光扩展到具有全场景label的整个空间训练样本集上，它的目标是提升pre-ranking output set的质量。另一方面，我们设计了一个distillation loss来提升在ranking stage上的一致性。图4和图3分别展示了已有存在的pre-ranking models和我们的模型。

图4展示了一个传统preranking的常见架构。在传统preranking model【8】中的训练样本，饮食了一个user、一个query、以及在曝光中的一个item。CTR任务和CVR任务的labels是场景内点击（in-scenario click）和场景内购买（in-scenario purchase）。该preranking系统包含了预估CTR和CVR的两个不同模型，并且最终效果分是CTR*CVR。由于在taobao搜索中前面的generation pre-ranking model会遵循该框架，分别输出CTR score和CVR score，我们称图4中的框架为basline。

作为对比，如图3所示，我们的ASMOL会在query-level训练样本上使用多个目标和多个postive items进行训练。特别的，在每个训练样本中，存在user features、query features和三种类型的items。所有的items会使用相同的可训练参数共享相同的DNN。用于处理每个user、query和item的input features和models结构如附录A所示，它与已存在的preranking models很不同。每个样本包含了在一个query请求中的所有items，以及从在线log系统中抽样得到的ranking和preranking candidates。三种训练样本间的关系如图5所示，定义如下：

- 曝光（Exposures, EX, 也称为impressions）：在一次query请求中曝光的N个items，包括由用户点击或购买的items。这些曝光的items会进行排序，并在所有候选的top，最具最大的概率来满足用户。
- Ranking候选（RC：Ranking Candidates）：由online preranking模型输出的Items，并作为ranking的候选集，它还没有被曝光给用户。我们会通过一个在线日志系统为每个query从上千个RC中抽样M个items。
- Pre-Ranking候选（PRC：Pre-Ranking Candidates）：由preranking系统产生但没有输出的items。我们会通过一个在线日志系统从成千上万个items中抽样L条items。

由于Ranking Candidates和Preranking Candidates在所有目标中都是负例（negatives）。这两种类型的负样本的目标是：在当前preranking系统中解决SSB问题。再者，受【7】的启发，preranking候选会被训练成easy样本，并且Ranking Candidates是相对hard的样本。我们在第4.2节展示了不同样本的必要性。

再者，对于在一条样本中的所有items，我们也会采用三种binary labels，分别对应于三种最优化目标：

- 全场景购买label（All-Scenario Purchase Label (ASPL)）：该label表示：用户是否在任意场景购买了该item。如果用户在当前query（taobao search）购买了该item，或者在其它场景购买了该item，可以被标记成与当前query的一个相关item，purchase label为1。否则，该purchase label为0。
- 全场景点击label（All-Scenario Click Label (ASCL)）: 与ASPL相似，该label表示用户是否在任意场景点击了该item。如果用户在购买它前点击了它，但是点击和购买行为又发生在不同场景导致了冲突的labels，当ASPL为1时，我们会将item的click label设置为1。
- Adaptive Exposure Label (AEL)：该label表示item是否在该taobao search请求中被曝光，它由ranking系统决定。对于曝光来说，该labels为1；其它items为0。另外，当ASCL为1时，我们会将item的曝光label置为1.

为了达到preranking的两个目标，我们会创建一个loss function，它会组合一个ranking loss和一个distillation loss：

$$
L = L_{rank} + L_{distill}
$$

...(4)

ranking loss $$L_{rank}$$会同时使用三种不同的labels。我们会创建一个具有三个任务的多目标loss function：Purchase、Click、Exposure。我们会为每个任务采用一个新的list-wise loss，并使得正样本的logit要大于负样本。我们的多目标学习框架，目标是通过最大化下面不同类型的正样本得分，来学习下面的importance序：购买items > 点击但未购买items > 未点击曝光 > ranking candidates (RC) 和 pre-ranking candidates (PRC)。我们在4.3节会演示了多目标最优化的必要性。该list-wise loss会在第4.4节讨论。

再者，我们会添加一个auxiliary distillation loss $$L_{distill}$$从来自ranking model（它具有更多features和可训练参数）来学习模型。令人吃惊的是，我们发现：简单将所有训练样本进行distilling并不是最好的方案。ranking model不总是个好teacher，特别是在还没被曝光的样本上。我们会在第4.5节分析该现象。

## 4.2 整个空间的训练样本

如表1所示，当训练样本包含了曝光、ranking candidates、preranking candidates时，ASPH@3000和在线GMV会被提升。如果preranking模型只使用曝光作为输入，通过从在线log系统中抽样的case分析，我们会发现：它的output set的质量很糟糕。由于仅抽样曝光的（exposure-sample-only）preranking model在训练期间不会看到非曝光样本，在评估时当在preranking candidates中给出样本时它会感到困惑。在该方式下，大多数候选的得分并不令人信服，会导致在output set中出现许多低质量items。

我们进一步探索了不同样本的比例的影响。一些研究者指出，通常一个不同类型负样本【7】的最优比例和一个关于负样本和正样本的最优比例【29】。在我们的多目标框架下，以click目标为例，ASCL等于1的items是正样本，其它是负样本。来自曝光的负样本、ranking candidates到preranking candidates，会变得easy。与click目标相近，购买目标只包含了三种类型的负样本。然而，对于exposure目标，所有的exposure样本是正样本，ranking candidates和pre-ranking candidates是负样本。作为结果，我们发现：即使我们移除了一定比例的非点击曝光，它也会伤害exposure目标，使得ASPH@3000极具下降。同时，从在线log系统中抽样的RC的数目和PRC的数目并不会更大更好。图6详述了：y轴表示，基于在表1中展示的实验“ASMOL w/o RC&PRC”，添加不同candidates数目的离线指标gaps。为最最大化ASPH@3000，我们设置RC的数目和PRC的数目分别为10和40。除此之外，我们可以看到，当RC和PRC小于10和40时，可以看到很明显的“Seesaw Effect”，因为该模型已经达到的Pareto frontier。

另外，由于preranking会使用RC，而ranking则不会使用PRC，这使得preranking不同于ranking。如图2所示，当输出数千items，它天然使得preranking的效果要好于ranking。online A/B test结果也会展示，**当ASPH@3000和PAUC@10结论冲突时，ASPH@3000更可信，它与preranking的主要目标和在线业务指标更一致**。

## 4.3 在多目标学习中的全场景labels

我们会进行一个消融实验，通过在多目标中移除每个label的相关loss来进行。实验结果如表2所示。当我们移移adaptive exposure label（AEL）时，全场景点击label（ASCL）和全场景购买label(ASPL)会通过分别移除它们相应的loss，评估指标的结论表明：模型效果会急剧下降。exposure label会帮助preranking model学习下流ranking系统的ranking模式，并且我们的结果验证了它的效果。同时，ASCL和ASPL要比AEL具有更少的in-scenario bias，可以给preranking model更精准关于用户偏好的信息。实验结果表明：我们三个losses的组合是合理并且有效的，对于提升preranking models的在线和离线指标有用。

我们也做了许多实验来证验ASL与ISL的对比效果，如表2所示。"->"表示label的变更。例如："ASL->ISL"表明：ASPL和ASCL同时变更为ISPL和ISCL。我们使用表2中的最后三行来对比在ASMOL间的结果，可以天然下结论：使用ASL会在ASH@k和在线业务指标上更一致。同时，AUC可能会因为“Seesaw Effect”下降。

## 4.4 multi-positive label的List-wise loss

我们设计了一个多目标loss来同步将exposure、click和purchase在一个模型内进行组合。这三个最优化目标可以进行jointly训练：

$$
L_{rank} = \alpha_{ex} L_{exposure} + \alpha_{cl} L_{click} + \alpha_{pur} L_{purchase}
$$

...(5)

对于每个任务，我们使用list-wise ranking loss。例如，对于购买任务，ranking loss可以被公式化为：

$$
L_{purchase} = \sum\limits_{i \in D} - log \frac{exp z_i}{\sum_{j \in S} exp_{z_j}}
$$

...(6)

其中：

- z是logit
- S是full training样本集合
- D是购买任务的正样本集合，包含了exposures、RC和PRC。

等式(6)对于购买任务工作良好，因为通常至多一次query只有一个购买。然而，multi-postive label在点击和曝光任务中是很常见的。等式6对于具有multiple postive样本的任务不再合适。对于一个multi-postive任务，等式（7）的 vanilla Softmax会导致最优化陷入到在多个正样本间的对比，而非正负样本间的对比。受CircleLoss的启发，对于多个postive labels的任务我们会轻微修改等式（6）：

$$
L_{exposure} = \sum\limits_{i \in \epsilon} - log \frac{exp z_i}{\sum_{j \in \lbrace S \\epsilon,i} \rbrace exp_z_j}
$$

...(7)

其中，$$\epsilon$$表示exposure任务的训练样本集合。当存在只有一个正样本时，等式(7)会退化成等式（6）。我们会在附录B详述等式（7）。为了验证在等式（7）中修改的有效性，我们会开展实验对比这两个loss functions，并在table 3中展示该结果。该实验令人吃惊的表明：在所有指标上，multi-label Softmax的效果要好于vanilla Softmax。我们会根据Zheng[29]来调整为每个任务$$\alpha_{ex}, \alpha_{cl}, \alpha_{pur}$$的weights。


表3

## 4.5 ranking stage的distillation

使用一个已ready的具有更多features和参数的大型ranking模型来蒸馏训练pre-ranking model是很天然的。在taobao search ranking stage中，存在两个独立模型：对应于CTR和CVR预估。在preranking stage，我们使用calibrated CTR和CVR predictions作为teachers来蒸馏我们的preranking model。为了充分利用CTR和CVR模型，preranking distillation loss会是CTR distill、CTCVR distill的组合：

$$
L_{distill} = \alpha_{cl} L_{CTR} + \alpha_{pur} L_{CTCVR}
$$

...(8)

CTR distillation任务可以进行如下公式化：

$$
L_{CTR} = \sum\limits_{i \in D} - p_{CTR} log \frac{exp z_i}{\sum_{j \in D} exp z_j}
$$

...(9)

其中：

- z是preranking model的logit
- D是要distilled 样本集合
- teacher $$p_{CTR}$$是ranking CTR model的prediction

相似的，CTCVR distillation的teacher是$$p_{CTR} * p_{CVR}$$。$$p_{CTR} = p(click \mid expose)$$和$$p_{CTCVR} = p(click & purchase \mid expose)$$是在给定相同条件p(expose)=1下的条件概率。由于$$p_{CVR}=p(purchase \mid click)$$不同于$$p_{CTR}$$，我们会使用$$p_{CTCVR}$$作为soft label来与$$p_{CTR}$$相一致。CTR distillation的loss weight和CTCVR distillation会分别遵循click task和purchase task的weights。

再者，我们观察到，定义



- 1.[https://arxiv.org/pdf/2305.13647.pdf](https://arxiv.org/pdf/2305.13647.pdf)