---
layout: post
title: youtube dpp介绍
description: 
modified: 2019-01-30
tags: 
---

youtube也开放了它们的diversity方法:《Practical Diversified Recommendations on YouTube with Determinantal Point Processes》, 我们来看下：

# 介绍

在线推荐服务通常以feed一个有序的item列表的方式来呈现内容给用户浏览。比如：youtube mobile主页、Facebook news feed。挑选和排序k个items的集合的目标是，最大化该set的效用(utility)。通常times recommenders会基于item的质量的排序来这样做——为每个item i分配一个pointwise的质量分$$q_i$$，并通过对该score进行排序(sorting)。然而，这通常是次优的，因为pointwise estimator会忽略items间的关联性。例如，给定一个已经在页面上被展示过的篮球视频，再展示另一个篮球视频可能会收效更小。相似视频会趋向于具有相近的质量分，这个问题会加剧。不幸的是，即使我们构建一个好的set-wise estimator，对每种可能的ranked list的排列进行打分的代价高昂。

本paper中，我们使用一种称为DPP的机器学习模型，它是一种排斥(repulsion)的概率模型，用于对推荐items进行diversity。DPP的一个关键点是，它可以有效地对一整个items list进行有效打分，而非每个进行单独打分，使得可以更好地解释item关联性。

在成熟的推荐系统中实现一个DPP-based的解决方案是non-trivial的。首先，DPP的训练方法在一些通用的推荐系统中[3,12,14,20,21,26,27]非常不同。第二，对已经存在的推荐系统集成DPP optimization是很复杂的。一种选择是，使用set-wise的推荐重组整个基础，但这会抛弃在已经存在的pointwise estimators上的大量投入。作为替代，我们使用DPP在已经存在的基础顶层作为last-layer model。这允许许多底层的系统组件可以独立演进。更特别的，对于一个大规模推荐系统，我们使用两个inputs来构建一个DPP：

- 1) 从一个DNN来构建pointwise estimators[9]，它会给我们一个关于item质量分$$q_i$$的高精度估计
- 2) pairwise item distances $$D_{ij}$$会以一个稀疏语义embedding space中计算（比如：[19]）

从这些输入中，我们可以构建一个DPP，并将它应用到在feed上的top n items上。我们的方法可以便研究团队继续开发$$q_i$$和$$D_{ij}$$的estimators、以及开发一个set-wise scoring系统。因此，我们可以达到diversification的目标，并能在大规模预测系统中利用上现有投入的系统。在Youtube上的经验结果表明，会增加short-term和long-term的用户满意度。

# 2.相关工作

当前推荐研究主要关注：提升pointwise estimate $$q_i$$，即：一个用户对于一个特定item有多喜欢。该研究线开始于20年前的UCF和ICF，接着是MF。在我们的系统中，我们从DNN中获得这些pointwise estimates，其中：一个用户的偏好特征可以组合上item features来一起估计用户有多喜欢该内容。

在这些改进的过程中，对于推荐结果的新颖性（novelty）和多样性（diversity）也得到了很大的研究【16，24，29，39，41，43，45】。相似的，在信息检索系统上，关于diversitication已经取得了较大的研究。【6，8，10，11，15，33，35，40，42】。考虑到所有这些文献，研究者已经提出了许多diversification概念。这里我们关于内容多样性总结并对比了两种不同的视角。

## 2.1 帮助探索的多样化

首先，多样化（diversification）有时被看成是一种帮助探索（exploration）的方法；它展示给用户更多样化的内容，可以：

- (A)帮助它们发现新的兴趣主题
- (B)帮助推荐系统发现更多与用户相关的东西

为了发现用户兴趣，信息检索有一分支的研究是，使用分类(taxonomy)来解决用户意图上的二义性(ambiguity)。例如，[2]中的IA-Select使用一个taxonomy来发现一个ambiguous query，接着最大化用户选择至少一个返回结果的概率。。。。


## 2.2 实用服务的多样化

关于多样化的的一种不同视角是，多样性直接控制着实用(utility)服务——通过合适的多样化曝光，可以最大化feed的utility。从该角度看，diversity更与交互关联，增加多样性意味着：使用用户更可能同时喜欢的items替换掉冗余(redundant)的视频曝光。这些新视频通常具有更低的个体得分（individual scores），但会产生一个更好的总体页面收益。

简洁的，一种达到多样性是，避免冗余项，它对于推荐系统特别重要。例如，在2005 Ziegler[45]中，使用一种贪婪算法利用books的taxonomy来最小化推荐items间的相似度。输出(output)接着使用一个利用一个多样化因子的非多样化的结果列表(non-diversitified result list)进行合并。在另一个信息检索的研究中，Carbonell和Goldstein提出了最大间隔相关度（MMR:maxinal marginal relevance）的方法。该方法涉及迭代式地一次选择一个item。一个item的score与它的相关度减去一个penalty项（用于衡量之前与选中items的相似度）成比例。其它关于redundancy的显式概念在【32】有研究，它使用一个关pairwise相似度上的decay函数。最近，Nassif【30】描述了一种使用次模优化的方式来对音乐推荐多样化。Lin[25]描述了一种使用次模函数来执行文档归纳的多样性。[38]描述了一种次模最大化的方式来选择items序列，[37]描述了使用次模多样性来基于category来进行top items re-rank。我们的目标与本质上相当相似，但使用了一个不同的优化技术。另外，我们不会将item idversity作为一个优先目标；我们的目标是：通过多性化信息提供给整个推荐系统，来尝试增加正向的用户交互数。你可以想像，这里表述的模型上的迭代用于表示一个个性化的diversity概念。被推荐内容的feed也是该方案的一个context，因为用户通常并不会寻找一个特定的item，在一个session过程中与多个items交互。

冗余的概念可以进一步划分成两个独立的相关性概念：替代（substitutes）和补足（complements）。这些概念已经被许多推荐系统所采用。在一个电商推荐应用中，用户做出一个购买决策之前，提供在考虑中的candidates的substitutes可能会更有用；而在用户做出购买行为之后，可以提供补全(complements)的商品。

## 2.3 相关工作

总之，许多研究者在我们之前已经研究了，如何在推荐和搜索结果中提升diversity。一些研究者同时处理许多这些diversity概念。例如，Vargas[39]解决了覆盖度与冗余性，以及推荐列表的size。我们关心的是在实践中在一个大规模推荐系统中能运行良好的技术。diversity的概念足够灵活，它可以随时间演化。因此，我们不会选择纠缠taxonomic或topic-coverage方法，因为他们需要一些关于diversity的显式表示（例如：在用户意图或topic ocverage上的一个显式猜测）。

相反的，我们提出了一种使用DPP（determinantal point processes）方法。DPP是一种set-wise的推荐模型，它只需要提供两种显式的、天然的element：一个item对于某个用户有多好，以及items pair间有多相似。

# 3.背景

## 3.1 Youtube主页feed

在Youtube mobile主页feed上生成视频推荐的整体架构如图1所示。该系统由三个阶段组成：

- (1) candidata generation，feed items从一个大的catalogue中选中
- (2) ranking，它会对feed items进行排序
- (3) policy，它会强制商业需求（比如：需要在页面的某些特定位置出现一些内容）

第(1)和(2)阶段都会大量使用DNN。


图1

candidate generation受用户在系统中之前行为的影响。ranking阶段则趋向于对相似的视频给出相近的utility预测，这会经常导致feeds具有重复的内容以及非常相似的视频。

为了消除redundancy问题。首先，我们引入了受[32,45]的启发法到policy layer，比如：对于任意用户的feed，单个的uploader可以贡献不超过n个items。而该规则有时很有效，我们的体验是它与底层推荐系统的交互相当贫乏(poorly)。由于candidate generation和ranking layers不知道该启发法（heuristic），他们会浪费掉那些不会被呈现items的空间，做出次优的预测。再者，由于前两个layers会随时间演进，我们需要重新调整heuristics的参数——该任务相当昂贵，因此实践中不会这么做去很频繁地维持该规则效果。最终，实际上，多种heuristics类型间的交互，提出了一种很难理解的推荐算法。结果是，系统是次优的，很难演进。

## 3.2 定义

为了更精准，假设一个用户与在一个给定feed中的items中所观察到的交互表示成一个二元向量y（比如：$$y=[0,1,0,1,1,0,0,\cdots]$$），其中可以理解的是，用户通常不会查看整个feed，但会以较低数目的索引开始。我们的目标是，最大化用户交互数：

$$
G'=\sum\limits_{u \sim Users} \sum\limits_{i \sim Items} y_{ui}
$$

...(1)

为了训练来自之前交互的模型，我们尝试选择模型参数来最大化对feed items进行reranking的累积增益:

$$
G = \sum\limits_{u \sim Users} \sum\limits_{i \sim Items} \frac{y_{ui}}{j}
$$

...(2)

其中，j是模型分配给一个item的新rank。该quantity会随着rank我们对交互进行的越高而增加。(实践中，我们会最小化$$j y_{ui}$$，而非最大化$$\frac{y_{ui}}{j}$$，但两个表达式具有相似的optima) 在下面的讨论中，我们出于简洁性会抛弃u下标，尽管所有值都应假设对于每个user biasis是不同的

我们进一步假设，使用一些黑盒估计y的quality：

$$
q_i \approx P(y_i = 1 | features \ of \ item i)
$$

...(3)

明显的ranking policy是根据q对items进行sort。注意，尽管$$q_i$$是一个只有单个item的函数。如果存在许多相似的items具有与$$q_i$$相近的值，它会在排序(rank)时会相互挨着，这会导致用户放弃继续feed。我们的最终目标是，最大化feed的总utility，我们可以调用两个items，等同于当：

$$
P(y_i=1, y_j=1) < P(y_i=1) P(y_j=1)
$$

...(4)

换句话说，当一起出现时，它们是负相关的——说明其中之一是冗余的。如果在feed中存在相似的items，那么通过q进行sorting不再是最优的policy。

假设我们提供了黑盒item distances：

$$
D_{ij} = distance(item \ i, item \ j) \in [0, \infty)
$$

...(5)

这些距离被假设成是“无标定的（uncalibrated）”，换句话说，他们无需直接与等式(4)相关。例如，如果问题中的items是新闻文章，D可以是一个在每篇文章中tokenized words的Jaccard distance。现在的目标是，基于q、D、y生成一个ranking policy, 比起通过q简单排序，它可以达到一个关于G的更小值。这可以很理想地与现有基础设施集成和演进。

## 3.3 设计需要

# 4.方法

## 4.1 DPP总览

## 4.2 Kernel参数化

## 4.3 训练方法

我们的训练集包含了将近4w的样本，它们从Youtube mobile主页feed上收集的某天数据中抽样得到。每个训练样本是单个主页feed曝光：单个实例，一个用户在youtube mobile主页上的，并被呈现一个关于推荐视频的有序列表。对于每个这样的曝光，我们有一个关于用户喜欢哪些视频的记录，我们表示为 set Y。我们注意到，使用这样的数据来训练模型存在一个partial-label bias，因为我们只观察到用户与那些被选中呈现给他们的视频的交互，而非随机均匀选中的视频。通常，我们会使用与过去训练pointwise模型相同类型的方式来解决该问题，比如：使用一个e-greedy exploration策略。

对于前面章节中描述的basic kernel，存在两个参数：$$\alpha$$和$$\sigma$$，因此我们可以做一个grid search来找来能使等式(2)中的累积增益最大化的值。图3展示了$$\alpha$$和$$\sigma$$的多种选择所获得的累积增益。颜色越暗，结果越糟糕。有意思的是，你可以观察到。。。

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

其中，$$Y_j$$是来自与用户交互的训练样本j的items的子集。使用log似然作为一个目标函数的能力，

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