---
layout: post
title: youtube dpp介绍
description: 
modified: 2019-01-30
tags: 
---

youtube也开放了它们的diversity方法:《Practical Diversified Recommendations on YouTube with Determinantal Point Processes》, 我们来看下：

# 介绍

在线推荐服务通常以feed一个有序的item列表的方式来呈现内容给用户浏览。比如：youtube mobile主页、Facebook news feed。挑选和排序k个items的集合的目标是，最大化该set的效用(utility)。通常times recommenders会基于item的质量的排序来这样做——为每个item i分配一个pointwise的质量分$$q_i$$，并通过对该score进行排序(sorting)。然而，这通常是次优的，因为**pointwise estimator会忽略items间的关联性。例如，给定一个已经在页面上被展示过的篮球视频，再展示另一个篮球视频可能会收效更小**。相似视频会趋向于具有相近的质量分，这个问题会加剧。不幸的是，即使**我们构建一个好的set-wise estimator，对每种可能的ranked list的排列进行打分的代价高昂**。

本paper中，我们使用一种称为DPP的机器学习模型，它是一种**排斥(repulsion)的概率模型**，用于对推荐items进行diversity。DPP的一个关键点是：它可以有效地对一整个items list进行有效打分，而非每个进行单独打分，使得可以更好地解释item关联性。

在成熟的推荐系统中实现一个DPP-based的解决方案是non-trivial的。首先，DPP的训练方法在一些通用的推荐系统中[3,12,14,20,21,26,27]非常不同。第二，对已经存在的推荐系统集成DPP optimization是很复杂的。一种选择是，使用set-wise的推荐重组整个基础，但这会抛弃在已经存在的pointwise estimators上的大量投入。**作为替代，我们使用DPP在已经存在的基础顶层作为last-layer model**。这允许许多底层的系统组件可以独立演进。更特别的，对于一个大规模推荐系统，我们使用两个inputs来构建一个DPP：

- 1) 从一个DNN来构建pointwise estimators[9]，它会给我们一个关于item质量分$$q_i$$的高精度估计
- 2) **pairwise item distances $$D_{ij}$$会以一个稀疏语义embedding space中计算**（比如：[19]）

从这些输入中，我们可以构建一个DPP，并将它应用到在feed上的top n items上。我们的方法可以让研究团队继续独立开发$$q_i$$和$$D_{ij}$$的estimators、以及开发一个set-wise scoring系统。因此，我们可以达到diversification的目标，并能在大规模预测系统中利用上现有投入的系统。**在Youtube上的经验结果表明，会增加short-term和long-term的用户满意度**。

# 2.相关工作

当前推荐研究主要关注：**提升pointwise estimate $$q_i$$（quanlity），即：一个用户对于一个特定item有多喜欢**。

该研究线开始于20年前的UCF和ICF，接着是MF。在我们的系统中，我们从DNN中获得这些pointwise estimates，其中：一个用户的偏好特征可以组合上item features来一起估计用户有多喜欢该内容。

在这些改进的过程中，对于推荐结果的新颖性（novelty）和多样性（diversity）也得到了很大的研究【16，24，29，39，41，43，45】。相似的，在信息检索系统上，关于diversitication已经取得了较大的研究。【6，8，10，11，15，33，35，40，42】。考虑到所有这些文献，研究者已经提出了许多diversification概念。这里我们关于内容多样性总结并对比了两种不同的视角。

## 2.1 帮助探索的多样化

首先，多样化（diversification）有时被看成是一种帮助探索（exploration）的方法；它展示给用户更多样化的内容，可以：

- (A)帮助它们发现新的兴趣主题
- (B)帮助推荐系统发现更多与用户相关的东西

为了发现用户兴趣，信息检索有一个分支研究是，使用分类(taxonomy)来解决用户意图上的二义性(ambiguity)。例如，[2]中的IA-Select使用一个taxonomy来发现一个ambiguous query，接着最大化用户选择至少一个返回结果的概率。。。。


## 2.2 实用服务的多样化

关于多样化的的一种不同视角是，**多样性直接控制着utility服务——通过合适的多样化曝光，可以最大化feed的utility**。从该角度看，diversity更与交互关联，**增加多样性意味着：使用用户更可能同时喜欢的items替换掉冗余(redundant)的视频曝光**。这些新视频通常具有更低的个体得分（individual scores），但会**产生一个更好的总体页面收益**。

简洁的说，**一种达到多样性的方式是：避免冗余项，它对于推荐系统特别重要**。例如，在2005 Ziegler[45]中，使用一种贪婪算法利用books的taxonomy来最小化推荐items间的相似度。输出(output)接着使用一个利用一个多样化因子的非多样化的结果列表(non-diversitified result list)进行合并。在另一个信息检索的研究中，Carbonell和Goldstein提出了最大间隔相关度（MMR:maxinal marginal relevance）的方法。该方法涉及迭代式地一次选择一个item。一个item的score与它的相关度减去一个penalty项（用于衡量之前与选中items的相似度）成比例。其它关于redundancy的显式概念在【32】有研究，它使用一个关pairwise相似度上的decay函数。最近，Nassif【30】描述了一种使用次模优化的方式来对音乐推荐多样化。Lin[25]描述了一种使用次模函数来执行文档归纳的多样性。[38]描述了一种次模最大化的方式来选择items序列，[37]描述了使用次模多样性来基于category来进行top items re-rank。我们的目标与本质上相当相似，但使用了一个不同的优化技术。另外，我们不会将item idversity作为一个优先目标；我们的目标是：通过多性化信息提供给整个推荐系统，来尝试增加正向的用户交互数。你可以想像，这里表述的模型上的迭代用于表示一个个性化的diversity概念。被推荐内容的feed也是该方案的一个context，因为用户通常并不会寻找一个特定的item，在一个session过程中与多个items交互。

冗余的概念可以进一步划分成两个独立的相关性概念：替代（substitutes）和补足（complements）。这些概念已经被许多推荐系统所采用。在一个电商推荐应用中，用户做出一个购买决策之前，提供在考虑中的candidates的substitutes可能会更有用；而在用户做出购买行为之后，可以提供补全(complements)的商品。

## 2.3 相关工作

总之，许多研究者在我们的工作之前已经开始研究：如何在推荐和搜索结果中提升diversity。一些研究者同时处理许多这些diversity概念。例如，Vargas[39]解决了覆盖度与冗余性，以及推荐列表的size。我们关心的是在实践中在一个大规模推荐系统中能运行良好的技术。diversity的概念足够灵活，它可以随时间演化。因此，我们不会选择纠缠taxonomic或topic-coverage方法，因为他们需要一些关于diversity的显式表示（例如：在用户意图或topic ocverage上的一个显式猜测）。

相反的，我们提出了一种使用DPP（determinantal point processes）方法。**DPP是一种set-wise的推荐模型**，它只需要提供两种显式的、天然的element：一个item对于某个用户有多好，以及items pair间有多相似。

# 3.背景

## 3.1 Youtube主页feed

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/adef073ee62bc2466603e91f4827eb97b40f86ac6d3bfd654162091031191d272871ba402c91de2da3b8ab3e975e9ab1?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1 基础的serving scheme

在Youtube mobile主页feed上生成视频推荐的整体架构如图1所示。该系统由三个阶段组成：

- (1) 候选生成（candidata generation）：feed items从一个大的catalogue中选中
- (2) 排序（ranking）：它会对feed items进行排序
- (3) 机制（policy）：它会强加一些商业需求（比如：需要在页面的某些特定位置出现一些内容）

第(1)和(2)阶段都会大量使用DNN。

candidate generation受用户在系统中之前行为的影响。**ranking阶段则趋向于对相似的视频给出相近的utility预测，这会经常导致feeds具有重复的内容以及非常相似的视频**。

为了消除冗余（redundancy）问题。首先，我们引入了受[32,45]的启发法到policy layer，**比如：对于任意用户的feed，单个up主的内容不超过n个items**。而该规则有时很有效，我们的经验：这种方式与底层推荐系统的交互相当少。

- **由于candidate generation和ranking layers不知道该启发法（heuristic），他们会浪费掉那些不会被呈现items的空间，做出次优的预测**。
- 再者，由于**前两个layers会随时间演进，我们需要重新调整heuristics的参数**——该任务代价相当高昂，因此实践中不会这么做去很频繁地维持该规则效果。

最终，实际上，多种heuristics类型间的交互，会生成一种很难理解的推荐算法。另外从结果看：系统是次优的，很难演进。

## 3.2 定义

为了更精准些，我们假设：

一个用户与在一个给定feed中的items中所观察到的交互表示成一个二元向量y：

$$y=[0,1,0,1,1,0,0,\cdots]$$

其中：可以理解的是，用户通常不会检查整个feed流，但会从较低数目的索引开始。

我们的目标是，最大化用户交互数：

$$
G'=\sum\limits_{u \sim Users} \sum\limits_{i \sim Items} y_{ui}
$$

...(1)

为了训练来自之前交互的模型，我们尝试选择模型参数来最大化对feed items进行reranking的累积增益:

$$
G = \sum\limits_{u \sim Users} \sum\limits_{i \sim Items} \frac{y_{ui}}{j}
$$

...(2)

其中：**j是模型分配给一个item的新rank**。

该quantity会随着rank我们对交互进行的越高而增加。(**实践中，我们会最小化$$j y_{ui}$$，而非最大化$$\frac{y_{ui}}{j}$$，但两个表达式具有相似的optima**) 在下面的讨论中，我们出于简洁性会抛弃u下标，尽管所有值都应假设对于每个user biasis是不同的

我们进一步假设，使用一些黑盒估计y的quality：

$$
q_i \approx P(y_i = 1 | \ features \ of \ item \ i)
$$

...(3)

明显的ranking policy是根据q对items进行sort。注意，尽管$$q_i$$是一个只有单个item的函数。如果存在许多具有与$$q_i$$相近值的相似items，它们会在排序(rank)时会相互挨着，这会导致用户放弃继续下刷feed。我们的最终目标是：最大化feed的总utility，我们可以调用两个items，等同于当：

$$
P(y_i=1, y_j=1) < P(y_i=1) P(y_j=1)
$$

...(4)

换句话说，**当一起出现时，它们是负相关的——说明其中之一是冗余的**。如果在feed中存在相似的items，那么通过q进行sorting不再是最优的policy。

假设我们提供了黑盒item distances：

$$
D_{ij} = distance(item \ i, item \ j) \in [0, \infty)
$$

...(5)

这些距离被假设成是“无标定的（uncalibrated）”，换句话说，他们无需直接与等式(4)相关。**例如，如果问题中的items是新闻文章，D可以是一个在每篇文章中tokenized words的Jaccard distance**。现在的目标是，基于q、D、y生成一个ranking policy, 比起通过q简单排序，它可以达到一个关于G的更小值。这可以很理想地与现有基础设施集成和演进。

## 3.3 设计需要

如果item similarity（如等式4定义）存在于dataset中，并且dataset足够大，那么我们的目标可以通过多种不同的方法来达成。我们喜欢这样的方法：

- 1）能很好满足已存在的逻辑框架：基于已观测的事件来构建机器学习estimators
- 2）可以优雅地在复杂性上随时间进行扩展
- 3）不需要巨大变更就可以应用到已存在的系统和专家意见上

启发法【45】可能会有效但并不理想。例如：假设强制一个规则：在n个相邻的items内，任意两个items必须满足$$D_{ij} < \tau$$。会引起多个问题：

- 1) **该规则运作与q独立**。这意味着，高得分的items会与低得分的items具有相同条件。在应用该策略后，对q的accuracy进行独立提升会失去效果
- 2）参数n和$$\tau$$可以通过grid search进行暴力搜索，但**额外的复杂性变得相当高**，因为训练时间随参数数目指数级增长。
- 3）在一定程度上包含q之外，如何扩展该规则并随时间做出增量提升，这一点并不明显。

一个重要点是：**该启发法会隐式地将冗余问题（redundancy problem）看成是一个与最大化utility具有不同的目标**。事实上，它建议：该hypothesis会提升diversity，并可能减少utility（至少在短期内），因为它会丢掉那些具有高分q的items。相反的，我们提出的方法会考虑：items的pairs上的utility（通过等式4描述的anti-correlation），因而，使用utility本身能更好地调整的特定items。

当然，基于上述的anti-correlation会定义一个启发法是可能的，比如“在相同的feed中不允许这样的两个items：$$\frac{P(y_i=1, y_j=1)}{P(y_i=1)P(y_j=1)}$$在x以下”。然而，**如上所述，该规则不能说明q，可能需要频繁地对参数x进行re-tuning**，并且即使有常规的调整，对于精准捕获我们期望的行为也不够灵活，我们会引入DPPs到系统中，作为多样性推荐的方式。

我们会在policy layer之前插入DPPs，但在point-wise scoring layer之后（如图2）所示。这允许我们以一个十分复杂的pointwise scorer进行研究，并确保遵守商业策略（business policies）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/0083b98e81ea6f2ec616cac8d0755ba7192a6fdb924bb7a96a097a9f5d2eca8810f19cf6aa772a4d70f5d315f3309af6?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2 新的serving schema

# 4.方法

## 4.1 DPP总览

我们首先回顾下DPPs（deternminantal point processes）的总览。在一个集合$$S=\lbrace 1, 2, \cdots, N \rbrace$$（例如：在一个用户的Youtube移动首页feed中的N个视频的集合）上的一个point process P是一个概率分布（S的所有子集）。也就是说，$$\forall S \subseteq S$$, P会分配一些概率P(S)，并且$$\sum_{S \subseteq S} = 1$$。DPPs表示一个概率分布族（family），它的参数可调，以便一个subset的概率P(S)与在S中的items的quality以及这些items的diversity的一个组合measure成比例。这样，发现set $$max_{S:\mid S \mid = k} P(S)$$是从一个N个items的更大池中选择关于k个items的high-quality和diverse的subset的一种方式。

如第2节所述，存在许多合理的measures，可以解释：item quality和diversity，比如：MMR方法（maximal marginal relevance）。使用DPPs的优点有两块：

- 1） DPPs在推荐任务中可以胜过MMR
- 2）一个DPP是一个概率模型（probalilistic model）

后一点意味着，我们可以利用概率operations算法的优点，比如：边缘化（marginalization）、调节（conditioning）、采样（sampling）。这些operations与构建一个系统的目标对齐得很好，可以很优雅地随时间在复杂度上可扩展。

我们现在描述，如何我们使用一个DPP来建模用户行为。对于一个有N items的feed，长度为N的binary vector y，表示用户与哪个视频交互。假设：Y表示这些items的index set（例如：对于y=[0, 1, 0, 0, 1, 1]，我们有$$Y = \lbrace 2, 5, 6 \rbrace$$）。接着我们假设，一个用户u的行为是通过一个具有概率分布P的DPP建模，以如下方式：$$Y ~ P_u$$。也就是说，互交的视频集合Y，表示由一个user-specific DPP定义的概率分布中抽取。

尽管一个DPP定义了一个在指数数目集合（所有$$2^N$$的子集有$$S=\lbrace 1,2, \cdots, N \rbrace$$）上的概率分布，它可以通过一个N X N的半正定kernel matrix进行密集参数化（compactly），我们称它为L。更具体的，一个DPP的概率可以写成一个关于L子矩阵的行列式：

$$
P(Y) = \frac{det(L_Y)}{\sum_{Y' \subseteq S} det(L_{Y^'})}
$$

...(6)

其中：

$$L_{Y}$$是L限制了只有它的行、列，通过Y进行index（例如：$$Y=\lbrace 2,5,6 \rbrace$$，对应的矩阵$$L_Y$$是size 3X3）。注意，等式(6)的分母简化为一个规范术语(normalizing term)，它可以被写成和有效计算成一个简单的行列式：

$$
\sum_{Y \subseteq S} det(L_Y) = det(L + I)
$$

...(7)

其中，I是单位矩阵。

为了看到$$det(L_Y)$$如何定义一个关于items集合的quality和diversity的balanced measure，它可以帮助以如下方式理解L的entries：

- 1）一个对角entry $$L_{ii}$$是一个关于item i的quanlity的measurement
- 2）一个非对角（off-diagonal）元素$$L_{ij}$$：是一个关于item i和item j间的相似度的归一化measurement

有了这些直觉，我们考虑一个$$\mid Y \mid = 2$$的case。如果$$Y=\lbrace 1,2 \rbrace$$，接着：

$$
L_y =  \left[
\begin
  L_{11}&L_{12}\\
  L_{21}&L_{22}
\end
\right] 
$$

该submatrix的行列式为：$$det(L_Y) = L_{11}L_{22} - L_{12}L_{21}$$。因此，它是一个item quanlities减去归一化item相似度（scaled item similarities）的乘积。该行列式表达式对于更大子矩阵来说更复杂，但直觉上是相似的。

在以下的章节，我们讨论在L从系统输入的多种构建方式，比如：pointwise item quanlity scores，q，第3.2节描述。

## 4.2 Kernel参数化

当前部署如图2所示，diversification发现在pipeline的相对靠后，因此一个典型的输入set size是：N=100.  对于这些N个视频中的每一个，我们具有两个主要的输入特征（input features）：

- 一个个性化quanlity score q
- 一个sparse embedding $$\phi$$，从视频的主题内容中提取出

这些features完全由独立的子系统生成。通过将我们的diversification系统叠加到它们的top上，我们可以利用这些子系统的持续提升。

对于DPPs初始引入，我们首先使用一个相对简单的参数，关于$$N \times N$$的DPP kernel matrix L：

$$
L_{ii} = q_i^2 \\
L_{ij} = \alpha q_i q_j exp(-\frac{D_{ij}}{2\sigma^2}), for i \neq j
$$

...(9) (10)

每个$$D_{ij}$$通过$$\phi_i$$和$$\phi_j$$计算得到；第5节描述了准确的embedding $$\phi$$



## 4.3 训练方法

我们的训练集包含了将近4w的样本，它们从Youtube mobile主页feed上收集的某天数据中抽样得到。**每个训练样本是单个homepage的feed曝光：一个用户的单个实例，对应于用户访问了youtube mobile主页，并被呈现出一个关于推荐视频的有序列表**。

对于每个这样的曝光，我们有一个关于用户喜欢哪些视频的记录，我们表示为 set Y。我们注意到，使用这样的数据来训练模型存在一个partial-label bias，因为我们只观察到用户与那些被选中呈现给他们的视频的交互，而非随机均匀选中的视频。通常，我们会使用与过去训练pointwise模型相同类型的方式来解决该问题，比如：使用一个e-greedy exploration策略。

对于前面章节中描述的basic kernel，存在两个参数：$$\alpha$$和$$\sigma$$，因此我们可以做一个grid search来找来能使等式(2)中的累积增益最大化的值。图3展示了$$\alpha$$和$$\sigma$$的多种选择所获得的累积增益。颜色越暗，结果越糟糕。有意思的是，你可以观察到，在右上角象限中的灾难性悬崖（catastrophic clif），以及随后的高原。必须对训练样本使用DPP kernels来变为增加non-PSD。记住，随着$$\alpha$$增长，L的非对角阵也会增长，这会增加一个non-PSD L的机率。由于非对角阵一定程度上会随$$\sigma$$增加，对于许多训练样本来说，大的$$\alpha, \sigma$$组合会导致non-PSD矩阵。直觉上，看起来整个右上角会具有低累积增益值，而非：低的值会集中在观察带上。然而，记住，我们会将任意non-PSD矩阵投影回PSD空间上。该投影分别对于$$\alpha$$和$$\sigma$$来说都是非线性的，因此，在投影后的矩阵的quanlity，不会期望与我们关于这些参数的直觉强相关。整体上，我们发现，具有最高的累积增益会在$$\sigma$$的中间区间、以及$$\alpha$$的上半区间达到。由这些参数产生的L kernels更可能是PSD，因此，只有一个偶然的训练样本的kernel需要投影。

## 4.4 Deep Gramian Kernels

正如之前所讨论，通过启发法使用DPP的一个主要好处是，DPP允许我们构建一个在复杂度上可以随时间优雅扩展的系统。启发法的复杂度扩展性很差，因为必须在参数上做grid search来调参，因此，对于训练一个启发法的runtime，与参数数目成指数关系。在本节中，使用DPP，我们可以超越grid search，使用许多参数来高效训练一个模型。

可以以多种方式来学习DPP kernel matrices。这些工作通常是为了最大化训练数据的log似然。更具体的，假设：

- L的参数是一些长度为r的vector w
- 我们具有M个训练样本，每个包含了：1）一个关于N个items的集合 2) 用户与之交互的这些items的子集Y

假设：L(w)是N x N的kernel matrix，通过参数w进行索引。接着训练数据的log似然是：

$$
LogLike(w) = \sum\limits_{j=1}^M log(P_{L(w)}(Y_j)) \\
 = \sum\limits_{j=1}^M [log(det(L(w)_{Y_j})) - log(det(L(w) + I))]
$$

其中，$$Y_j$$是来自与用户交互的训练样本j的items的子集。使用log似然作为一个目标函数的能力，允许我们使用比grid search更复杂的方法（并且更有效）来学习DPP参数。

我们然后通过使用在LogLike上的gradient descent，开始探索学习一个kernel，它具有许多参数，比如：前面提过的$$\alpha$$和$$\simga$$。我们仍会使用输入$$\phi$$ embeddings来区别视频内容。对于个性化视频的quality scores来说（非scalar score $$q_i$$），我们可以从已经存在的基础设施中获得quanlity scores $$q_i$$的整个vector，因此我们使用该vector来更通用地做出我们的模型。（vector $$q_i$$的每个entry一定程度上会捕获：对于一个用户做出一个好的视频选择），我们从input data中学到的full kernel $$L(\phi, q)$$可以通过下面方式进行表示：

$$
L_{i,j} = f(q_i) g(\phi_i)^T g(\phi_i)^T g(\phi_j) f(q_j) + \sigma 1_{i,j}
$$

...(13)

其中，f和g是neural network中的独立stacks。（$$\sigma$$可以简化为一个正则参数，我们可以固定在某个小值上）注意，quantity $$f(q_i)$$是一个scalar，其中$$g(\phi_i)$$是一个vector。计算f的neural network相当浅层，而g的network则更穿梭，在空间中有效的re-embeded $$\phi$$，会更能描述视频的utility correlation（如图4）。我们可以注意，不同于早前讨论的basic kernel parameterization，其中$$\alpha$$的大值会产生non-PSD L，这种更复杂的参数化实际会保证总是无需投影即可生成PSD矩阵。这遵循事实：L的该特定构造会使它是一个Gramian矩阵，并且所有这样的矩阵都是PSD的。

为了学习neural network的所有参数来计算f和g，我们会使用tensorflow来根据等式(11）进行最优化LogLike。产生的deep DPP models在线上实验会有utility提升（如表1的Deep DPPs所示）。然而，对比非多样性的baseline，这样的更深模型会大体上对ranking进行改变，**二级业务指标会被极大影响，需要进行额外调参**。

## 4.5 DPP的高效ranking

在本节中，我们描述了在serving时如何使用4.3节/4.4节学到的DPP参数。也就是说，当一个用户访问Youtube移动端主页时，DPP是如何决定哪些videos会在推荐feed的top展示的？对于任意给定的用户，Youtube系统基础设施的底层会将个性化质量得分(personalized quality scores) q和N个视频集合的视频embedding vectors $$\phi$$发送给DPP layer。我们会根据scores、embeddings、以及之前学到的参数来构建一个DPP kernel L。我们接着将window size $$k << N$$固定，并请求DPP来选取一个关于k个视频的高概率集合。我们将这些视频放置一feed的顶部，接着再次询问DPP来从剩余N-k个未见过的视频中选选一个关于k个视频的高概率集合。这些视频会变为在feed中的next k个视频。我们会重复该过程，直到我们对N个视频的整个feed排好序(ordered)。

使用stride size=k来构建数据的子窗口(sub-windows)背后的思想是，两个相似items间的排斥（repulsion）会随它们在feed中的距离越近而增加。也就是说，相似的video 1和video 100不如相似的video 1和video 2带给用户更差的体验。实际上，对一个包含了N=上百个视频的feed进行排序(ordering)，我们会使用k为十几个视频的sub-windows。

当我们“请求DPP来获取一个关于k个视频的高概率集合”时，我们实际要做的是，请求size-k 集合Y，它们会具有用户与这k个items的每一个交互的最高概率。这对应于以下的最大化问题：

$$
\underset{Y:|Y|=k}{max} det(L_Y)
$$

...(14)

如[18]所示，该最大化是NP-hard的。在实际中，尽管一个标准的次模最大化贪婪算法[31]看起来可以近似较好地求解该问题。该贪婪算法从$$Y=\emptyset$$（空集）开始，接着运行k次迭代，每次迭代添加一个video到Y中。在第i轮迭代选中的video是video v，当被添加到当前选中集合中时它会产生最大的行列式值（determinant value）:

$$
max_{v \in remaining videos} det(L_{Y \cup v})
$$

...(15)

除了简洁外，该贪婪算法的一个额外优点是，如果我们跟踪贪婪选中视频的order，会发现在相对应的用户feed的size-k window中，给出的视频的天然顺序(natural order)。

<>

算法1

算法1总结了本节的ranking算法。在后续章节你会看到，该ranking会帮助我们发现更容易消费的内容。


# 5.实验结果

首先，我们描述了一些基本的比较baseline。在最终达到DPPs之前，我们会尝试三种diversification的heuristics：

- 模糊去重（Fuzzy deduping）：不允许任意视频i，它与已在feed中的video j间的距离在一个阀值$$\tau$$之下：$$D_{ij} < \tau$$
- 滑动窗口（sliding window）：允许每m个items最多有n个在distance阀值$$\tau$$之下
- 平滑分罚项（smooth score penalty）：当正在为position n+1进行选择video v时，对quanlity scores进行re-scale来解释对于已经被选中的videos 1到n间的相似度

$$
q_{new,v} = q_{original, v} * e^{-b (\phi_v \cdot \phi_{previous})} 
$$

...(16)

其中：

$$
\phi_{previous} = \sum\limits_{k=0}^n a^{n-k-1} \phi_k
$$

...(17)

其中，q是我们排序的quality score，a和b是free parameters，$$\phi$$是embedding vector。

如表1所示，所有这些尝试都会产生一个less useful的移动端主页feed，通过对从主页开始的long sessions的用户数进行measure。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/5a209941bb06b37edc2ac9654fa721c1cdff990a1fb7917d96d29a705888a3767faf7227c45e1c3ec7454d0d1cc7ee9d?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=t1.jpg&amp;size=750">

表1

当使用DPPs进行实验时，我们首先使用4.2节描述的kernel L，并评估多个embeddings和distance functions（dense和sparse audio embeddings, frame embeddings，thumbnail image embeddings, document text embedding等）。我们发现，使用Jaccard distance来计算等式10中的$$D_{ij}$$会很有效，并应用到包含item tokens的sparse vectors $$\phi$$上。（例如，Saturday Night Live video "Olive Grarden-SNL"具有tokens "snl"、"olive garden"、"saturday night"、"night live"以及"sketch"等）。在youtube移动主页推荐上的真实线上实验可以看到，对于我们的用户有显著的提升。如表1所示，在satisfied homepage watchers metric上有+0.63%的提升，我们也可以看到在overall watch time上有+0.52%的提升，它对于baseline来说是个相当大的jump。由于在mobile上的成功，通过DPPs进行多样化已经被部署到所有surfaces上，包括：TV、desktop、Live streams。（注意：deep Gramian DPPs每看起来在"satisfied homepage watchers"指标上提升非常大，它还没有被部署）。正如之前所述，这些deeper models对比起非多样化的baseline，在ranking的变更上足够大，二级商业指标开始受到影响，需要额外调参）

有意思的是，对于一些参数选择，我们可以看到在homepage上直接交互的损失（losses），但从整个丫点看我们可以有一个整体的收益（overall win）。图5展示了来自homepage的view time上的百分比提升。这意味着，用户会发现内容足够吸引人，它会导致从homepage开始的一个更长的session。另外，我们也观察到，会在相关视频栏（related videos panel：它位于当前正播放视频的一侧panel，提供其它相关视频）上增加activity，包括：CTR、播放数（number of views）、总播放时间（amount of total view time），尽管事实上我们只影响了在homepage feed上的视频。这种可累积性（cumulatively），意味着对比以往方式，用户会更多他们喜欢的视频。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/9e041572b21d7fbe6d7f7dc9f132f4c11daea44979057befc0c04df9a7d5c826d4e1fbd530bfb3741a2d1cb61f08f0f2?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=5.jpg&amp;size=750" width="400">

图5

另外，我们也能从多样性的用户feeds中观察到一个long-term的"learning effect"【17】。也就是说，随着时间的延伸，多样性会导致用户更愿意返回并享受我们提供的服务。在们通过运行两个long-term holdback实验的集合来对该effect进行评估。在第一个holdback条件中，用户不会获得DPP-diversified feeds，但该部分用户流量子集会随每天变动（这些用户通常会曝fkhtgcdiversified feed，除了在他们在该holdback set中结束的很少几天rare day）。在第二个holdback条件中，一个consistent的用户集合不会看到DPP-diversified feeds。我们接着观察，DPP多样性是否会产生一个在用户体验上的long-term提升：当对比control groups时，通过观察在两个holdbacks间的差异来得到。如图6所示，通过这两个holdback groups，用户从homepage上观看至少一个视频的数目会增加：使用diversified feeds曝光的用户更容易找到他们在youtube主页上感兴趣的视频。因此，我们可以看到，diversified feeds会导致在立即项（immediate term）上增加用户满意度，该effect会随时间变得更显著。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/9803043e98bcbff32c741a96e667a05818df93f3694e40baf6cf36ce1e06bd262c0a2a48967b536ae8bd211e80f93451?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=6.jpg&amp;size=750">

图6


# 参考

- 1.[http://jgillenw.com/cikm2018.pdf](http://jgillenw.com/cikm2018.pdf)