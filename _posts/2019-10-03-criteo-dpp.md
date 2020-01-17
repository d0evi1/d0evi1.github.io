---
layout: post
title: tensorized DPP介绍
description: 
modified: 2019-10-03
tags: 
---

criteo也开放了它们的dpp方法:《Tensorized Determinantal Point Processes for Recommendation》, 我们来看下：

# 摘要

DPP在机器学习中的关注度越来越高，因为它可以在组合集合上提供一种优雅的参数化模型。特别的，**在DPP中的所需的参数数目只与ground truth(例如：item catalog)的size成平方关系**，而items的数目增长是指数式的。最近一些研究表明，DPPs对于商品推荐和（basket completion）任务 来说是很高效的模型，因为他们可以同时在一个集合中解释diversity和quality。我们提出了一种增强的DPP模型：tensorized DPP，它特别适合于basket completion任务。我们利用来自张量分解（tensor factorization）的思想，以便将模型进行定制用在next-item basket completion任务上，其中next item会在该模型的一个额外维度中被捕获。我们在多个真实数据集上评估了该模型，并找出：tensorized DPP在许多settings中，比许多SOTA模型提供了更好的predictive quality。

# 1.介绍

在averge shooping basket中items数的增加，对于在线零售商来说是一个主要关注点。该问题存在许多处理策略。而本工作主要关注于：算法会生成一个items集合，它们能适合补全用户的当前shopping basket。

Basket analysis和completion是机器学习中非常老的任务。许多年来，关联规则挖掘（association rule mining）是SOTA的。尽管该算法具有不同变种，主要的准则是涉及到：通过统计在过往observations中的共现，来计算购买一个额外商品的条件概率。由于计算开销和健壮性，现代方法更喜欢i2i CF，或者使用LR基于二分类购买得分来预测一个用户是否会构买一个item。

标准CF方法必须被扩展到能正确捕获商品间的diversity。**在basket completion中，需要插入一定形式的diversity，因为推荐过于相似的items给用户并不好**。实践者经常通过添加constraints到items推荐集合中来缓和该问题。例如，当使用类目信息时，在裤子被添加到basket时可以强制推荐相匹配的鞋子，而如果按天然的共同出售(co-sale) patterns会导致其它裤子的推荐。在这种情况中，diversity推荐的出现不会通过学习算法直接驱动，但可以通过side information和专家知识。Ref【28】提出了一种有效的Bayesian方法来学习类目的权重，当类目已知时。

然而，不依赖额外信息直接学习合适的diversity更令人关注。**不使用side information，直接从数据中的diversity的naive learning，会得到一个高的计算开销**，因为可能集合的数目会随类目中items数目而指数增长。该issue是no-trivial的，即使当我们只想往已存在集合中添加一个item时，而当我们想添加超过一个item来达到最终推荐set的diversity时会更难。

【9, 10】使用基于DPPs的模型来解决该组合问题。DPPs是一个来自量子物理学的优雅的关于排斥(repulsion)的概率模型，在机器学习上被广泛使用[17]。它允许抽样一个diverse的点集，相似度（similarity）和流行度(popularity)会使用一个称为“kernel”半正定矩阵进行编码。关于marginalization和conditioning DPPs有很多高效算法提供。从实用角度，学习DPP kernel是个挑战，因为相关的likelihood是non-convex的，从items的observed sets中学到它是NP-hard的。

对于basket completion问题，天然地会考虑：那些转化成售买的baskets的sets。在该setting中，**DPP通过一个size为$$p \times p$$的kernel matrix进行参数化，其中p是catalog(item目录表)的size**。因而，参数的数目会随着p的二次方进行增长，计算复杂度、预测、抽样会随着p的三次方增长。由于学习一个full-rank的DPP是很难的，[10]提出了通过对kernel限制到low rank来对DPP正则化（regularization）。该regularization会在不伤害预测效果下提升generalization，并可以提供更diversity的推荐。在许多settings中，预测质量也会被提升，使得DPP对于建模baskets问题是一个理想的工具。再者，对比起full-rank DPP，**low-rank假设也提供了更好的runtime效果**。

另外，由于DPP的定义，正如在Model部分所描述的，low-rank假设对于kernel来说，意味着任意可能的baskets会比那些概率为0的选中rank要具有更好的items。该方法对于大的baskets来说不可能，一些其它DPP kernel的正则化可能更合适。另外，由于DPP kernel的对称性，可以建模有序（ordered corrections）。然而，这些被添加到shooping basket中的items的order会在basket completion任务中扮演重要角色。

主要贡献：

- 在kernel上修改了constraints来支持大的baskets；也就是说，对于大于kernel rank的sets来说，我们会阻止返回概率0
- 我们通过在DPP kernel的行列式上添加一个logistic function，来修改在所有baskets上的概率。我们将训练过程适配成处理这种非线性，并在许多real-world basket数据集上评估了我们的模型
- 通过使用tensor factorization，我们提出了一种新方式来对在目录中的集合间的kernel进行正则化。该方法也会导致增强预测质量
- 我们展示了这种新模型，称之为"tensorfized DPP"，允许我们可以捕获ordered basket completion。也就是说，我们可以利用该信息，忽略掉items被添加到basket的顺序，来提升预测质量

另外，我们展示了这些思想的组合来提升预测质量，tensorized DPP建模的效果要好于SOTA模型一大截。


# 2.相关工作

# 3.模型

DPPs最初用来建模具有排斥效应(replusive effect)的粒子间的分布。最近，在利用这种排斥行为上的兴趣，已经导致DPP在机器学习界受到大量关注。数学上，离散DPPs是在离散点集上的分布，在我们的case中，点就是items，模型会为观察到的给定items集合分配一个概率。假设I表示一个items集合，L是与DPP相关的kernel matrix（它的entries会在items间对polularity和similarity进行编码）。观察到的set I的概率与主子矩阵（principal submatrix）L的行列式成正比：$$I: P(I) \propto del L_I$$。因而，如果p表示在item catalog中的items数目，DPP是在$$2^p$$上的概率measure（），而它只包含了$$p^2$$的参数。kernel L会对items间的polularities和similarities进行编码，而对角条目$$L_{ii}$$表示item i的流行度，off-diagonal entry $$L_{ij} = L_{ji}$$表示item i和item j间的相似度。行列式从几何角度可以被看成是体积（volume），因此更diverse的sets趋向于具有更大的行列式。例如，选择items i和j的概率可以通过以下计算：

$$
P[\lbrace i,j \rbrace] \propto \begin{vmatrix}
    L_{ii} & L_{ij} \\
    L_{ji} & L_{jj} \\
    \end{vmatrix} = L_{ii} L_{jj} - L_{ij}^2
$$

...(1)

等式(1)中我们可以看到：如果i和j更相似，他们被抽样在一起的可能性越低。entries $$L_{ij}$$因此会决定kernel的排斥行为。例如，如果使用图片描述符来决定相似度，那么DPP会选择那些有区别的图片。另一方面，如果entries $$L_{ij}$$使用之前观察到的sets学到，比如：电商购物篮[10]，那么，“similarity” $$L_{ij}$$会低些。由于共同购买的items可能具有某些diversity，DPPs对于建模包含购买items的baskets是一种天然选择。在搜索引擎场景、或者文档归纳应用中，kernel可以使用特征描述述 $$\phi_i \in R^D$$（例如：文本中的tf-idf）、以及一个关于每个item i的相关得分$$q_i \in R^+$$，比如：$$L_{ij} = q_i \phi_i^T q_j$$（它会喜欢相关items ($$q_i$$值大)，阻止相似items组成的lists）。

## 3.1 Logistic DPP

我们的目标是，寻找一个最可能一起购买的items集合。我们将该问题看成是一个分类问题，目标是预测：**一个items的特定集合会生成一个转化（conversion），即：所有items都将被一起购买，这可以表示成$$Y \in \lbrace 0, 1 \rbrace$$**。我们将class label Y建模成一个Bernoulli随机变量，它具有参数$$\phi(I)$$，其中$$I$$是items集合，$$\phi$$是如下定义的函数：

$$
p(y | I) = \phi(I)^y (1- \phi(I))^{1-y}
$$

...(2)

我们使用一个DPP来建模函数$$\phi$$。

我们假设：存在一个隐空间，在该空间内diverse items很可能会被一起购买。与[10]相似，我们假设：在kernel matrix $$L \in R^{p \times p}$$上存在一个low-rank constraint，我们进行如下分解：

$$
L = VV^T + D^2
$$

...(3)

其中，$$V \in R^{p \times r}$$是一个隐矩阵，其中每个row vector i会编会item i的r个latent factors。D是一个对角阵（diagonal matrix），$$\|V_i \|$$，表示每个item的intrinsic quality或popularity。在D上的平方指数确保了，我们总是具有一个合理的半正定kernel。我们接着定义：

$$
\phi(I) \propto det(V_{I_{,:}} V_{I_{,:}}^T + D^2)  \geq 0 
$$

注意，没有对角项，r的选择会限制observable set的cardinality，由于$$\mid I \mid > r$$暗示着当$$D \equiv 0$$时$$\phi(I)=0$$。使用该term会确保，任意set的后续概率都是正的，但对于基数(cardinality)高于r的sets，它的cross-effects会更低。我们也看到，具有相似latent vectors的items，对比起具有不同latent vectors的items，被抽到的可能性会更小，由于相似vectors会生成一个具有更小体积（volume）的超平行体（parallelotope）。为了对概率归一化，并鼓励vectors间的分离，我们会在$$\phi$$上使用一个logistic function：

$$
\phi(I) = P(y = 1 | I) & \doteq  1 - exp(-w det L_I) \\
				& \doteq \delta(w del L_I)
$$
...(5)

通常，logistic function的形式是：$$1/(1 + exp(-w det L_I))$$。然而，在我们的case中，行列式总是正的，因为L是半正定的，这会导致$$P(y=1 \mid I)$$总是大于0.5 。通过构建，我们的公式允许我们获得一个介于0和1之间的概率。最终，$$w \in R$$是一个scaling参数，可以通过cross-validation被选中，这确保了指数不会爆炸，因于对角参数会近似为1.

**Learning**。为了学习matrix V，我们确保了历史数据 $$\lbrace I_m, y_m \rbrace_{1 \leq m \leq M}$$，其中，$$I_m$$是items集合，$$y_m$$是label set，如果该set购买则为1, 否则为0。该训练数据允许我们通过最大化数据的log似然来学习矩阵V和D。为了这样做，我们首先对所有y写出点击概率：

$$
P(y | I) = \sigma(w det L_I)^y (1-\sigma(w det L_I))^{1-y}
$$

...(6)

$$f(V,D)$$的log似然接着被写成：

$$
f(V,D) = log \prod\limits_{m=1}^m P(y_m | I_m) - \frac{a_0}{2} \sum\limits_{i=1}^{p} a_i ( \| V_i \|^2 + \| D_i \|^2) \\
= \sum\limits_{m=1}^M log P(y_m | I_m) - \frac{a_0}{2} \sum\limits_{i=1}^{p} a_i ( \| V_i \|^2 + \| D_i \|^2)
$$

根据[10]，$$a_i$$是一个item正则权重，它与item流行度成反比。矩阵V和D可以使用SGA来最大化log似然进行学习。GA的一个step需要计算一个对称矩阵($$L_i$$，其中I是gradient step的相应item set)的行列式，它可以使用 optimized CW-like algorithm算法来达到，复杂度为：$$O(f^3)$$或$$O(f^{2.373})$$，其中，f对应于在I中的items数目。用于学习所使用的最优化算法如算法1所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/dc66c9d80dbbc0a23e9e42cb601d032b713cad057df59da75416aef665e839afc61dc96db4754248c8ed622916effae0?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=a1.jpg&amp;size=750">

算法1

## 3.3 Tensorized DPP

我们现在提出了对之前模型的一个修改版本，它更适合basket completion任务。为了这样做，对于basket completion场景，我们加强logistic DPP，其中我们对概率建模：用户将基于已经出现在shooping basket中的items来购买一个指定额外的item。我们使用一个tensor来表示它，目标是预测用户是否会基于basket来购买一个给定的candidate target item。该tensor的每个slice对应于一个candidate target item。在该setting中，对于在catalog p中的item （减去basket中已经存在的items），会存在越来越多的问题待解决。为每个待推荐的item学习一个kernel，每个item会与其它所有items相独立，在实际上是不可能的，会存在稀疏性问题。每个item只在baskets中的一小部分出现，因而每个kernel只会接受一小部分数据来学习。然而，所有items间并不完全相互独立。为了解决稀疏性问题，受RESCAL分解的启发，我们使用一个low-rank tensor。我们使用一个cubic tensor $$K \in R^{p \times p \times p }$$，其中K的每个slice $$\tau$$(标为：$$K_{\tau}$$)是candidate item (low-rank) kernel。通过假设：tensor K是low-rank的，我们可以实现在每个item间学到参数的共享，以如下等式所示：

$$
K_{\tau} = V R_{\tau}^2 V^T + D^2
$$

...(7)

其中，$$V \in R^{p \times r}$$是item latent factors，它对所有candidates items是共用的，$$R_{\tau} \in R^{r \times r}$$是一个candidate item指定的matrix，会建模每个candidate item间的latent components的交叉。为了对candidate items与已经在basket中的items间的自由度进行balance，我们进一步假设：$$R_{\tau}$$是一个对角矩阵。因此，$$R_{\tau}$$的对角向量会建模每个candidate item的latent factors，item的latent factors可以被看成是在每个latent factor上的产品的相关度。正如在matrix D的case，在$$R_{\tau}$$上的平方指数（squared exponent）可以确保我们总是有一个合理的kernel。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/cd6c888b5f291c25ea9371bb18711d1c728944a80f25b6138c6067954d4a358df5f27673ab8de376583d273d0a7bbabc?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1

图1展示了factorization的一个图示。candidate item $$\tau$$的概率与已经在basket中的items set I是相关的：

$$
P(y_{\tau} = 1 | I) = \sigma (w det K_{\tau, I} = 1 - exp(-w det K_{\tau,I})
$$

...(8)

因此，$$g(V,D,R) \doteq g$$的log似然为：

$$
g = \sum\limits_{m=1}^M log P(y_{\tau} | I_m) - \frac{a_0}{2} a_i (\| V_i \|^2 + \| D_i \|^2 + \| R^i \|^2)
$$

其中，每个observation m与一个candidate item有关，$$I_m$$是与一个observation相关的basket items的set。由于之前的描述，矩阵V, D，以及$$(R_{\tau})_{\tau \in \lbrace 1, \cdots, p\rbrace}$$通过使用SGA最大化log似然学到。正如logistic DPP模型，gradient ascent的一个step需要计算对称矩阵 $$L_I$$的逆和行列式，会产生$$O(f^{2.373})$$的复杂度（I中items数目为f）。算法2描述了该算法。关于最优化算法的细节详见附录。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/8cc8fb24c587fe9ec0d5fc3ecef6bfae1621e3885f97fbe377be4a88d6da10b5fe417457a3040924bfdca39c2f093865?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=a2.jpg&amp;size=750">

算法2

**泛化到高阶交叉**。在basket completion应用中，尝试同时推荐多个items挺有意思的。这可以使用一个贪婪方法来完成。也就是说，我们首先使用一个初始产品（initial product）来补充basket，并将augmented basket看成是一个新的basket，接着补充它。一种更直接的方法是，更适合捕获items间的高阶交叉，这可以泛化等式(7)。我们提出了一种高阶版本的模型，将来会对该模型进行效果评估。假设：d是要推荐的items数目，$$\tau = [\tau_1, \cdots, \tau_d] \in [p]^d$$。我们接着可以将kernel $$K_{\tau}$$定义为：

$$
K_{\tau} = V \prod\limits_{k=1}^d R_{(d), \tau_d}^2 V^T + D^2
$$

...(9)

其中，每个$$R_{(d), \tau_d} \in R^{r \times r}$$是一个对角矩阵。

## 3.3 预测

如前所述，从一个DPP中抽样可能是一个很难的问题，提出了许多解法[6,12]。尽管，在所有可能sets间抽样最好的set是个NP-hard问题，我们的目标是，寻找最好的item来补全basket。在这样的应用中，可以有效使用greedy方法，特别是我们的模型具有low-rank结构。另外，[10]提出了一种有效的方法来进行basket completion，涉及到对DPP进行conditioning，这在我们的logistic DPP模型有使用。


# 4.实验
略

# 5.实验结果

略

# 参考

- 1.[https://dl.acm.org/doi/pdf/10.1145/3292500.3330952?download=true](https://dl.acm.org/doi/pdf/10.1145/3292500.3330952?download=true)