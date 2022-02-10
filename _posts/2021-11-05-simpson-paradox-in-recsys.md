---
layout: post
title: 离线评估中的Simpson Paradox
description: 
modified: 2021-11-05
tags: 
---


JADIDINEJAD在《The Simpson’s Paradox in the Offline Evaluation of
Recommendation Systems》中提出推荐系统中的Simpson’s Paradox：


# 1.介绍

推荐系统通常会以online（A/B testing或interleaving）或offline方式进行评估。然而，由于online evaluation[6,14]部署研究系统的难度，推荐系统的离线评估仍然是大范围使用的evaluation方式。实际上，很难以离线方式通过使用历史用户交互日志可靠地评估一个推荐模型的效果【7，16，43】。该问题是因为存在混淆因子（confounders），例如：那些可以影响items的曝光（exposure）以及结果（outcomes）（比如：ratings）的变量。**在当一个平台尝试建模用户行为时，如果没有解释在曝光的推荐items的选择偏差（selection bias），这时会出现confounding问题，从而导致不可预料的结果**。在这种情况下，**很难区别用户交互是来自于用户的真实偏好、还是受部署推荐系统（deployed recommender system）的影响**。

通常，推荐系统的离线评估会具有两个阶段：

- 1) 收集来自一个deployed system的用户反馈集合
- 2) 使用这样的feedback来经验性评估和比较不同的推荐模型

第一阶段可以服从不同类型的confounders，即可以是由用户关于item的选择行为来初始化，或者通过受deployed推荐系统的动作影响来初始化。例如，除了其它交互机制（比如：搜索）之外，**用户更可能会与被曝光的items进行交互**。在这种方式下，在一个新推荐模型的离线评估中使用历史交互，而这样的交互从deployed推荐系统中获取得到，**形成一个closed loop feedback**，例如：deployed recommender system存在对收集到的feedback的具有一个直接影响，它可以被用来进行对其它推荐模型的离线评估。因而，新的推荐模型会根据它趋向于模拟由deployed model收集到的交互有多像来进行评估，而非有多满足用户的真实偏好。另一方面，在一个**open loop（随机化）场景**中，deployed model是一个随机推荐模型，例如：为users曝光随机的items。因此，在deployed model与新的推荐model间的feedback loop会打破，deployed model不会对收集到的feedback dataset具有影响，相应的，它对于在基于收集到的feedback dataset上对任意新模型的离线评估都没有影响。然而，为users曝光随机items是天然不实际的，用户体验会降级。**因此，基于closed loop feedback对推荐系统进行训练和评估会是一个严重问题**。

Simpson’s paradox是统计学中的一个现象，当多个不同分组的观察数据中出现的一个显著趋势，会在这些分组组合在一起时消失或者逆转【29】。**在推荐场景中，当曝光（例如：推荐items）和结果（例如：用户的隐式和显式反馈）相关联时，并且曝光和结果会受一个第三方变量强烈影响时，会发生Simpson’s paradox**。在统计学上，如果观察到的feedback是Simpson’s paradox的一个产物，根据confounding variable对feedback进行分层，可以消除悖论。**我们会讨论：在推荐系统的情况下，该confounding variable是交互数据被收集的deployed model（或系统），a.k.a：closed loop feedback【17】**。在本paper中，我们的核心目标是，对于closed loop feedback在推荐系统离线评估上提供一个in-depth研究，并提供了一个健壮的解来解决该问题。特别的，我们会讨论：从一个deployed model收集到的feedback datasets会偏向于deployed model的特性，并导致证实Simpson’s paradox的结论。我们通过研究在推荐系统离线评估上的confounding variable（例如：deployed model's的特性），可以观察到显著趋势；当从经典离线setting中上报observations时，该趋势接着会消失或逆转。另外，我们提出了一种新的评估方法，它可以解决Simpson’s paradox，以便产生一种更合理的推荐系统离线评估方法。

为了更好地理解该问题的微妙之处，考虑一个deployed推荐模型，它会提升一个指定分组的items（例如：流行的items）——**对比起只有少量交互的长尾items，存在一些少量的头部items，它们会被广泛曝光给用户并获得大量交互**。当我们基于从前面deployed model收集到的feedback来评估一个新的推荐模型时，如果没有解释不同的items曝光的有多频繁，评估过程会被deployed model的特性所混淆，例如：**任意模型的效果会具有这样一个趋势，展示与已经存在的deployed model相似的属性，这很可能会引起过估计（over-estimated）**。在这种情况下，我们可能会选择部署一个匹配已deployed model的特性的新模型，从实际用户角度看，它会比另一个模型更低效。在本paper中，我们通过研究该问题在标准离线评估中做出结论的结果，并提出一种新的方法来解决该问题。特别的，本paper的贡献有两块：

- 我们提出了一种in-depth分析
- 为了解决该问题，提出了一个新的propensity-based stratified evaluation方法

# 2.相关工作

我们的工作主要受三块相关工作的启发：

- 在l2r中解决bias（2.1节）
- 算法混淆（algorithmic confounding）或closed loop feedback（2.2节）
- 在counterfactual learning和evaluation中的工作（2.3）

... 

# 3.离线评估法

本节中，我们会将当前offline evaluation方法进行总结，称为标准holdout evaluation和counterfactual evaluation。

## 3.1 Holdout Evaluation

...

## 3.2 Counterfactual Evaluation

...

# 4.介绍

辛普森悖论（Simpson’s paradox）是统计学中的一种观察现象：**在观察数据集的许多不同groups中都出现的一个显著趋势，当将这些groups组合在一起时会消失甚至反转**。该topic在许多文献上被广泛讨论。在该现象中会出现一个明显的悖论，当聚合数据时会支持这么一个结论：它与在数据聚合前的相同的分层数据的结论相反。**当两个变量间的关系被研究时，如果这些变量会被一个协变量（confounding variable）所强烈影响时，就会发生辛普森悖论**。当该数据根据混杂变量（confounding variable）进行分层时，该悖论会展示出相悖的结论。在这种情况下，使用一个显著性检验（significance test）可以识别出在一个指定层做出的错误结论；然而，如第7节所示，显著性检验不可能识别出这样的统计趋势（trends）。在推荐系统的评估场景，会在所有用户上进行testing，这里讨论的悖论通常会涉及到user-item feedback生成过程。在另一方面，当因果关系（causal relations）在统计建模中被合理解决时，辛普森悖论可被解决。

在本节中，为了演示辛普森悖论，我们会从一个paper[8]中呈现一个真实示例，它会对比肾结石（kidney stone disease）的两种治疗方法（treatments）的成功率。这里的目标是：**基于观察找出哪个treatment更高效**。【8】会随机抽样350个病人，它们会接受每个治疗，并上报如表1所示的成功率。一个合理的结论是：treatment B要比treatment A更高效（83% vs. 78%的康复率）。另一方面，悖论是，当考虑上结石大小时，**比如：treatment A对于小size（93% vs. 87%），大size（73% vs. 69%）两者都要有效，但最终的成功率会反转**。[8]会讨论treatment (A vs. B) 以及结果（成功 vs. 失败）会与一个第三个混杂变量（confounding variable：这里的结石大小）有关。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/f612b059e405ddc741835a8dafee4c3a6f7f692900a804b8866dce7b42ece08a6cf2b6cb68d125bbcdabd62d03b04e49?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

表1  a) 肾结石示例。每个条目表示：恢复数/总病人数，成功率见括号  b) 推荐系统中离线评估的辛普森悖论。每个条目表示了受检模型在相应分层上的效果。*表示对比其它模型的一个显著差异（paired t-test, p<0.05）


假设 1: 医生趋向于对不严重病例（例如：小结石）选择treatment B，对于更严重病例（例如：大结石）使用treatment A进行治疗

表1验证了以上的假设，例如：大多数接受treatment A的病人都是大结石病例（group 3中350个随机病人中有263个），而大多数接受treatment B的病人都是小结石病例（Group 2的350中的270）。**因此，当从接受A或B治疗的病人中被随机挑选样本时，抽样过程并不纯随机，例如：样本会倾向于：严重病例进行treatment A进行measuring，而轻症病例进行treatment B进行measuring**。这就是因果分析中著名的现象：辛普森悖论。

表1b展示了在推荐系统的离线评估中的辛普森悖论示例。我们对两个推荐模型的有效性进行评估，如表1b中的A、B模型。两个模型会在相同的dataset上使用相同的evaluation metric进行评估。根据一个paired t-test，在检查的数据集上的标准离线评估表明：模型A要明显好于模型B。然而，将待检测数据集划分成两个层（Q1和Q2）表明：模型B对于待测数据集的99%（Q1）要更好，而模型A在1%（Q2）上面要更好。当Q1和Q2进行聚合时，模型B在99%待测数据集上的的统计优势会消失。在以下部分，我们会呈现：辛普森悖论是如何影响推荐系统的离线评估的，并讨论这样的悖论的原因，以及提出一个新的评估方法来解决它。

# 5.基于倾向的分层评估（PROPENSITY-BASED STRATIFIED EVALUATION)

当为一个推荐系统的离线评估创建一个数据集时，用户反馈不仅会从与推荐items的交互上会通过推荐系统被收集，也会通过其它其它形式（比如：当浏览item的目录时发生的交互、或者点了sponsored items的链接）进行收集。对于区分用户的不同反馈源来说并不简单，因为没有公共数据集提供这样的数据来确定用户反馈的source。因此，在本paper中，我们的研究主要关注于用户反馈的主源，称为deployed system。为了演示在推荐系统中的辛普森悖论，我们需要一个因果假设，它与第4节中的假设1相似。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/d2a0fdf4409da08dda40e6ac67c7fff584e4bbdd5022516ecc6cf34e5338021682d83f89c1118f9f135ea2455dfbe20f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图1


图1a展示了一个典型推荐系统的信息流，其中用户反馈会通过deployed system进行收集。部署好的推荐系统组件（通过RecSys表示）会为target user（例如：通过推荐items的一个ranked list）过滤出items进行曝光（exposure: e）。另一方面，用户在items上记录的偏好（例如：ratings或clicks）（用r表示）会被作为交互数据来训练 或 评估推荐模型的下一次生成（next-generation）。因而，由于用户点击是从RecSys曝光的items获得的，模型本身会影响数据的genreation，它们会用于训练和评估它。图1a中的系统是一个动态系统，其中，系统进行简单联想推理（associative
reasoning ）的原因是很难的，因为每个组件会互相影响。图1b表明了在这样一个闭合循环反馈场景下的因果关系图。实线表示了在原因和效果间一个explicit/observed关系，而虚线表示了一个implicit/unobserved关系。如上所示，在推荐系统的case中，主要的混合变量是，来自交互数据的deployed model会被收集。我们的目标是，基于来自deployed model收集到的封闭循环反馈（r））评估一个推荐模型（Y）的效果会影响主干扰因子（main confounder），例如：deployed model的特性。在该情况下，很难区分: 来源于用户真实偏好影响的的用户交互，或者受deployed recommendation model影响的用户交互。因此，在该场景下，用户反馈通常会通过deployed system进行收集，我们会假定，基于闭循环反馈数据集的推荐模型离线评估，会受以下deployed recommendation model的强烈影响：

假设2: 闭环循环反馈（closed loop feedback）会从一个deployed recommendation model来收集到，并倾向于deployed model的特性（characteristics）（例如：曝光items），**deployed model的特性**会在推荐模型的离线评估中作为一个混淆因子（confounding factor）。

deployed recommendation model的核心问题是：尝试建模潜在的用户偏好，这使得它很难在没有解释算法混淆的情况下对用户行为（或者使用用户数据）做出断言。另一方面，如果图1a中的RecSys组件是一个random模型，那么在User和RecSys组件间的连接会被移除，RecSys组件会在收集到的feedback dataset上没有影响，这种情况 称为“开环循环反馈（open loop feedback）”，对比起在collection中的任意其它item，没有items会接收到或多或少的期望曝光。

如图1b所示，如果deployed model的特性（e）可以被标识和评估，离线评估（Y）会与confounder独立。因此，为了验证在推荐系统中（假设2）closed loop feedback的影响，我们必须量化deployed model的特性。结尾处，根据5,47，我们会如下定义propensity score：

**定义5.1  propensity score $$p_{u,i}$$是deployed model（如图1a中的RecSys描述）对于expose item $$i \in I$$的**曝光给user $$u \in U$$的趋势

propensity $$p_{u,i}$$是在一个闭环反馈场景下，deployed model将item i曝光给user u的概率。该值会量化来自一个无偏开环曝光（unbiased
open loop exposure）场景的系统偏差（deviation），其中：随机items会被曝光给该用户，deployed model对收集到的feedback没有影响。propensity score $$p_{u,i}$$允许我们基于观察到的闭环feedback来设计并分析推荐模型的离线评估，因此它会模拟一些开环场景的特殊特性。

分层（Stratification）是用来标识和估计因果效应的知名方法，会在每个层调查因果效应之前，首先标识出潜在层（underlying strata）。总意图是，在confounding variable上进行分层，并研究在每个层上的潜在结果。因此，衡量不考虑confounding variable的潜在结果是可能的。在这种情况下，在confounding variable上的边缘化（marginalisation）可以作为一个组合估计（combined estimate）被使用。如前所述，在推荐系统中的假设condounding variable是deployed model的特征（假设2）。定义5.1会将该变量量化成倾向（propensities）。在这种情况下，在propensity scores上的分层会允许我们分析deployed model特性在推荐模型的离线评估上的该效应。

出于简洁性，假设我们具有一个单一的categorical confounding variable X。如果我们基于X的可能值将观察到的结果进行分层，那么潜在结果（Y）的期望值如下所示：

$$
E(Y) = \sum\limits_x E(Y | X=x) P(X=x)
$$

...(4)

其中，$$E(Y \mid X = x)$$是在给定分层x下对于observed结果的条件期望，$$P(X=x)$$是x的边缘分布。例如，在肾结石的示例中，condounding variable是结石大小，它基于我们的分层（$$X = \lbrace small, large \rbrace$$）以及潜在结果是treatment效果。我们可以基于等式(4)计算每个treatment的期望值。例如：treatment A的期望值可以按如下方式计算：

$$
E(A) = E(A | X = small) P(X=small) + E(A | X = large) P(X = large)
$$

...(5)

基于表1a和等式(5)中的数字，treatment A的期望值（resp. B）分别计算为：0.832和0.782，例如：E(A) > E(B)，它可以更好地估计treatments的实验效果。相似的，对于表1b的推荐示例，模型A和模型B的期望值分别被计算为：0.343和0.351，例如：E(B) > E(A)。如上所示，在推荐系统中，主要的假设混淆变量是deployed model（假设2）。该变量可以量化成propensity scores（定义5.1）。Propensity是一个连续变量。在本paper中，我们会通过将propensity scores进行排序和分片成将它转换成一个categorical variable，然后转成一个预定义好数目的分层，例如：表1b中的Q1和Q2分层。在表1a和表1b的两个case，都基于假设混淆变量并使用等式（4）对等验证分层进行边缘化，会解决Simpson’s paradox。例如，在表1b中，模型B会被认为superior model，因为它对于99%的user-item feedback在效果上要好。这个重要的趋势会被提出的分层评估进行捕获，而在标准离线评估中，当将Q1和Q2分层聚合在一起时，结论会完全逆转。在下节中，我们会研究simpson paradox的效应，以及提出的propensity-based分层评估的好处。特别的，我们研究了以下问题：

**研究问题1：在闭环feedback场景下，推荐系统的离线评估有多大程度是受deployed model特性所影响的**

然而，我们的目标是评估deployed model特性在推荐模型离线评估中的donfounding effect，如图1b所示。就这一点而言，相似的原理图示例如第4节所示，我们对于将基于deployed model的特征的observed closed-loop feedback进行分层这一点比较感兴趣。这样的分层分析使得我们可以去评估：在推荐系统的标准离线评估中辛普森悖论的存在。从而，我们会研究许多不同推荐模型的相关关系的显著趋势，其中：在层的大多数上观察到的一个趋势，在标准离线评估中会消失或者逆转。

**研究问题2:当进行一个可比离线评估时，propensity-based stratified evaluation是否可以帮助我们更好地估计实际的模型效果**

在上述研究问题中，我们的目标是：评估在推荐模型的离线评估中，提出的提出的propensity-based stratified evaluation是否有效。如等式（4）所示，我们可以利用在confounding variable分布上的边缘化（marginalisation）作为潜在结果的一个组合估计。在该研究问题上，我们会评估：该估计是如何与open loop(randomised) evaluation相关的，其中deployed model是一个随机推荐模型（例如：random items会被曝光给该user）。特别的，给定一个特别的评估方法（open loop、closed loop等），我们可以去measure基于标准的rank-based评估指标（nDCG）多种推荐模型的效果，并记录这些模型的所用metrics的相对排序。我们接着研究在examined models的相对排序间的相关性，并对比从一个open loop（随机化）设定下获得的rankings，以及使用propensity-based evaluation这两者的评估。我们会利用适当的显著性检测，如第6节所述，来决定：在对比标准的离线holdout evaluation时，由d propensity-based stratified evaluation评估的模型效果要比open loop(随机) evaluation更好。

# 6.实验设定

下面，我们会通过实验来解决两个前面的研究问题。图2表示了实验的总结构。每个评估方法（X, Y和Z）会会基于它们的相对表现对多个检查的推荐模型进行排序。我们会使用Kendall’s $$\tau$$ rank相关系数来对受检模型在每个评估方法上（Y or Z）的相对顺序间的联系，对比ground truth进行measure，例如：open loop（randomiszed） evaluation(X)。这样的相关值会在图2中被描述成$$\tau_{XY}$$和$$\tau_{XZ}$$。另外，我们会使用Steiger’s方法来测试在两个评估方法间的差异显著性，例如：baseline evaluation method(Y)以及提出的evaluation method(Z)。我们会对比propensity-based stratified evaluation、标准offline holdout和counterfactual evaluation方法作为baseline两者。以下，我们描述了我们实验设定的详情，包含：使用的数据集和评估指标、受检推荐模型、以及如何在实验中估计propensities。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/6fce44f0628acab3b00bb27f1f7a78f6978137ec692742583afd8d2411a031d4321c49d49825445e4864609c9813c15f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=3.jpg&amp;size=750">

图2 

## 6.1 

## 6.2 

## 6.3 估计Propensity Scores

propensity score $$p_{u,i}$$会被定义成：deployed model将item i曝光给user u的趋势（定义5.1）。由于它对于deployed model来曝光每个item给user是不实际的，我们必须为多个user/item pairs 估计propensity scores $$p_{u,i}$$。[47]提出了一种简单方法来估计propensity scores，它基于如下的简单假设：

假设1: propensity score是用户独立的（user independent），例如：$$p_{u,i} = p_{*,i}$$。该假设是为了解决：在公开数据集中的辅助用户信息的缺失。

user independent propensity score $$p_{*,i}$$可以使用一个2-step的生成过程（generative process）来估计：

$$
p_{*,i} = p_{*,i}^{select} * p_{*,i}^{interact \mid select}
$$

...(6)

其中：$$p_{*,i}^{select}$$是先验概率（prior probability），item i通过deployed model被推荐出来，$$p_{*,i}^{interact \mid select}$$是user 与推荐出来的item i进行交互下的条件概率。基于这样的温和假设，我们可以估计user independent propensity score $$p_{*,i}$$如下：

$$
\hat{p}_{*,i} \propto (n_i^*)^{\frac{\gamma + 1}{2}}
$$

...(7)

其中，$$n_i^*$$是item i被交互的总次数，$$\gamma$$是一个参数，它影响着在items上具有不同流行度的propensity分布。power-law参数$$\gamma$$会影响着在items上的propensity分布，具体取决于dataset。根据之前的研究，对于每个dataset，我们会使用极大似然来估计$$\gamma$$参数。

# 7.实验结果与分析

我们在第5节提出使用一个propensity-based的直接评估方法，它会在推荐系统的离线评估中，考虑上deployed model的混杂角色（ confounding role）。在第6节中，我们提出使用Kendall's $$\tau$$相关系数来量化在propensity-based stratified evaluation方法以及open loop evaluation间多个目标推荐模型的相对顺序间的相似度。下面，我们会会分别根据在第5节的两个研究问题开展实验，并关注和验证在标准离线评估的中 Simpson’s paradox，以及使用提出的propensity-based ed stratified evaluation方法的效果（7.2）。

## 7.1 RQ1:  研究Simpson’s Paradox

RQ1会研究：在推荐系统的离线评估中，基于假设2中提到的一个closed loop feedback dataset来研究deployed model的confounding effect。在第6.3节中，我们表述了一个简单的统计方法来表示deployed model的角色作为propensity scores。在下面，我们会使用estimated propensity score $$\hat{p}_{*,i}$$来将数据集划分成两个相同大小的层（ strata），称为Q1层和Q2层。

Q1和Q2 strata来分别表示用户与长尾items和头部items的交叉，我们会根据等式（7），基于每个item交互的总次数来估计propensity scores。

首先，我们会在closed loop dataset（MovieLens和Netflix）和open loop dataset（Yahoo!和Coat）上举例说明Simpson’s paradox。表2和表3会对比评估方法的效果（称为：holdout、IPS、提出的stratified评估法）。出于简洁，以下，我们会关注MovieLens dataset，分析两个模型（BPR、WMF）以及一个metric（nDCG）。我们观察到：在使用holdout和IPF评估法时，BPR要比WMF都好。然而，在同样的test dataset上进行分层分析（Q1和Q2层）表明：对于Q1分层，WMF要比BPR好很多；对于Q2分层，BPR则是支配模型。另外，我们观察到：Q1和Q2分层会分别覆盖99%和1%的user-item interactions。实际上，对于99%的test dataset（例如：Q1分层），WMF要比BPR好很多，但当我们结合Q2分层上1%的feedback时趋势会逆转，如holdout evaluation表示所示。因此，实际问题是：当使用holdout evaluation方法进行评估时认为的，BPR是否会比WMF执行的更好？分层分析表明：通过我们提出的分层评估法，WMF会被认为是更优的模型。我们注意到，BPR只会在1%的feedback dataset上是更优的模型，holdout和IPS evaluation方法两都都会受在test dataset上少量user-item interactions的影响。例如：在Q2分层中的1%的user-item interactions。该分层对应在MovieLens dataset的3499个items中只有4个items。在表3中，在MMMF和MF models间，在Netflix dataset中观察到相同的pattern。

在MovieLens和Netflix datasets中，我们不会访问open loop feedback，例如：feedback会通过一个随机曝光items进行收集。结果是，我们不能measure在一个open loop场景下的模型实验效果，对比起相应的closed loop场景。如第6.1节，我们对于在Coat dataset中的所有users，都有一些open loop feedback；而在Yahoo! dataset上只有一部分users有。表4和表5会表示在Yahoo! dataset中模型的两个pairs的效果对比。特别的，在表4中，我们会对比BPR和MF，它使用nDCG评估指标，并观察到：使用经典holdout evaluation方法，BPR要明显好于MF。另一方法，IPS评估法更偏向于MF，但，基于paired t-test(p < 0.05)上的差异没有显著。然而，基于估计倾向（Q1和Q2分层）的分层分析表明：对于Q1分层，MF要好于BPR；它可以覆盖在test dataset上92%的feedback；而BPR对于Q2分层则是更好的模型，它在closed loop test dataset上覆盖了8%的feedback。确实，对于test dataset中的大多数feedback和items（92%和99.5%），MF要胜过BPR。因此，我们会争论：通过任意的评估法（我们提出的分层评估法，或者open loop evaluation），MF应该被认为是一个更好的模型。当我们基于Q1和Q2分层（例如：holdout evaluation）的聚合数据进行评估模型时，在Yahoo! dataset上，1000个总items中只有少量5个items它对应于只有8%的总user-item interactions，会在受检模型的离线评估中，扮演着一个混杂因子的角色。当考虑上Coat dataset（表5）时，当评估GMF和SVD推荐模型的效果时，我们也会观察到相同的现象。表2、表3、表5中观察到的现象可以通过从一个closed loop场景收集到的datasets（MovieLens、Netflix、Yahoo!和Coat）被解释。因此，收集到的closed loop dataset会倾向于deployed model的特性（如表2、表4和表5中的Q2分层所捕获的）。

接着，为了完整回答RQ1，我们会评估以上观察的总结，通过验证在所有104个模型上的评估中 Simpson’s paradox的盛行。图3表明了，所有受检模型（）在Yahoo!和Coat datasets上在open loop evaluation和closed loop evaluation间的相关性。我们观察到，模型会表现出在Q1和Q2分层上的一个imbalanced效果，例如：Q1分层的nDCG score与Q2分层不成比例（$$nDCG_{Q2} >> nDCG_{Q1}$$）。另一方法，在Q1分层和open loop evaluation上Kendall's $$\tau$$相关性，对比起Q2分层要更高。特别的，对于Coat dataset（图3b），基于Q2分层的closed loop evaluation对比open loop evaluation具有一个负向correlation。因此，在holdout evaluation中通过组合这两个异构分层，如图3a和3b所示，不能说明Q1分层覆盖feedback的大多数（在Yahoo!和Coat datasets上分别是92%和93%的total feedback），会导致不可预料的后果。特别的，在open loop evaluation和holdout evaluation间的Kendall's $$\tau$$相关性，会比Q1分层和open loop evaluation间的对应相关性要低很多，例如：对于Yahoo!和Coat datasets，分别为：0.62 < 0.72 和 0.20 < 0.51. 这回答了RQ1: 推荐系统的offline holdout evaluation会受deployed model特性的影响（在Q2分层捕获），会导致对Simpson's paradox的一个证实。

总之，在本节中，我们强调了在推荐系统的标准离线评估中，基于于closed loop和open loop datasets，证实了Simpson's paradox的存在。接着，我们会研究提出的propensity-based分层评估法来解决该问题。

## 7.2 RQ2: 评估Propensity-based Stratified Evaluation方法

在本节中，我们会研究提出的propensity-based分层评估法会导致：对比IPS和holdout evaluation方法，结论与从open loop（随机化）evaluation获取的结果对齐的范围。确实，我们的目标是：研究每种评估法对应于open loop evaluation的范围。相反的，我们会考虑open loop evaluation，它与在线评估（A/B testing或interleaving）更相似，其中：评估过程不会受任意混淆因子的影响。如第6.2节所示，我们会利用在evaluation方法与open loop evaluation间Kendall's $$\tau$$的rank相关系数。

表6展示了，在评估方法和ground truth（例如：open loop evaluation）间的Kendall's $$\tau$$的rank相关系数。在分析该表时，我们会观察到，对于Yahoo! dataset，在holdout evaluation和open loop evaluation间的$$\tau$$相关系数是medium-to-strong（$$0.599 \leq \tau  \leq 0.729$$）；而对于Coat dataset， weaker($$-0.40 \leq \tau  \leq 0.327$$)，对于更大的截止点，两个数据集上的相关度都会轻微上升。确实，我们注意到，Coat dataset吃饭休息发货呢MovieLens/Netflix/Yahoo！会有一个不同的数据生成过程。另外，受检用户和items的数目会低于在Yahoo！dataset中数目。该发现与我们之前的研究一致：基于标准rank-based metrics的离线评估不必与在线评估的结果正相关，研究者们会质疑离线评估的合法性。

接着，我们会考虑在counterfactual evaluation(IPS)和open loop evaluation方法间的Kendall's $$\tau$$相关性。表6表明，IPS评估方法效果要比holdout评估要稍微好些，例如：对于两个数据集中nCDG cut-offs的绝大多数，IPS的相关值要大于holdout evalution方法。然而，根据Steiger's test，这些相关度差异没有任何在统计上是显著的。因此，我们会下结论，在评估期间，对比起holdout evaluation方法，counterfactual(IPS) evaluation方法不会更好地表示无偏用户偏好。这是因为：在IPS评估方法中，每个feedback是基于它的propensity进行加权独立（weighted individually）的。feedback sampling process（例如：从closed loop feedback中收集到的过程），并不是一个随机过程（random process）。因此，抽样的feedback会倾斜，相应的estimated propensity scores是不均的。在这种情况下，基于disproportionate propensities对feedback进行reweighting会导致在使用IPS评估模型效果评估上的高variance。

我们现在考虑表6中分层评估的相关性。这表明：在两个数据集中，对于nDCG cut-offs的绝大多数，特别是更深的cut-offs，分层评估的效果要好于holdout和IPS评估方法。例如：考虑将nDCG作为evaluation metric，我们会观察到，提出的分层评估方法效果要显著好于holdout evaluation和IPS evaluation，例如：对于Yahoo! dataset，0.710 > 0.622，0.710 > 0.644；对于Coat dataset，有0.283 > 0.202和0.283 > 0.225. 总体上，我们发现，对于两个datasets中更深的nDCG cut-offs，我们的propensity-based evaluation方法，效果要显著好于holdout和counterfactual evaluation。这可以回答RQ2.

然而，holdout evaluation的效果，IPS evaluation和提出的propensity-based分层评估方法，对于shallow nDCG cut-offs并不显著（对于Yahoo! dataset和Coat dataset来说，分别是：$$10 \leq k \leq 20 和 20 \leq k \leq 30$$）。对于deeper rank cut-offs的增加的sensitivity，支持了之前的研究：它对于推荐系统评估上deeper cut-offs的sparsity和discriminative pwoer来说更健壮。【43】发现，在feedback sampling process期间，由于只有少量items会曝光给该user，使用deeper cut-offs会允许研究者执行更健壮和discriminative evaluations。由【25】中，deeper rank cut-offs的使用对于训练listwise learning2rank技术来说可以做出相似的观察。

在feedback sub-populations (表6中的Q1和Q2分层)的进一步实验表明：受检模型的离线评估，它只基于long-tail feedback（Q1分层）更好地与open loop evaluation相关。特别的，对于Coat dataset，基于Q2分层的受检模型的离线评估，会与open loop evaluation具有一个负相关性。**这支持了[9]的工作，它发现：非常少量的头部流行items会对推荐系统的rank-based evaluation进行倾斜。他们建议：当评估推荐模型的效果时，排除非常流行的items**。

而表6演示了提出的分层评估方法对于两个sub-populations（Q1和Q2分层）的效果，我们接着检查了所使用分层数目的影响。特别的，在图4中，演示了提出的分层评估对于Coat和Yahoo! datasets的分层数目的影响。水平轴（X={1,2, ..., 10}）会表示分层的数目，而竖直轴表示在提出的分层数目上与open loop（随机化）评估间对于104个受检模型的相关性。当我们只具有一个分层时（例如：在图4a和图4b中的X=1），提出的分层评估法对应于表6中的holdout evaluation。我们观察到，对于在两个数据集中的分层数目，对比起holdout评估（例如：X=1），提出的分层评估方法可以更好地与open loop（随机）评估相关。然而，在提出的分层评估和open loop（随机）评估间的相关度上，分层数目具有一个边缘效应（marginal effect），例如：在提出的分层评估（2<=X <=10）对于Coats和Yahoo! datasets间的平均相关度（mean correlations）是：$$0.266 \pm 0.021$$以及$$0.706 \pm 0.026 $$，对比在holdout evaluation(X=1)中分别是0.202和0.622相关度。另外，对于$$2 \leq X \leq 10$$，大多数cases（coats和Yahoo!中分别是5/9和7/9）表示显著高于holdout evaluation的相关度。注意，每个dataset中的分层数目，可以使用分层原则（stratification principle）【24】来决定，它倾向于使用在每个层（stratum）内的层（strata），用户的反馈尽可能会相似。尽管不同数据集具有不同级别的closed loop effect，它取决于deployed model的特性，我们的实验表明：没有关于deployed model的进一步信息，基于estimated propensities将closed loop feedback分层为sub-populations，允许研究者们说明来自收集到的feedback dataset的closed loop effect。

# 8. 结论

略

# 参考


- 1.[https://arxiv.org/pdf/2104.08912.pdf](https://arxiv.org/pdf/2104.08912.pdf)