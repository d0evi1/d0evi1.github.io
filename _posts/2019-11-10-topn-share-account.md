---
layout: post
title: 共享账号topN推荐介绍
description: 
modified: 2019-11-10
tags: 
---

# 介绍

共享账号的推荐，很早就有人发paper，我们看下先人的一些探索《Top-N Recommendation for Shared Accounts》：

# 介绍

一般的推荐系统假设：每个用户账号(user account)表示单个用户(user)。然而，**多个用户(user)经常共享单个账号(account)**。一个示例是：一个家庭中的所有人会共享同一个在线视频账号、同一个在线音乐流账号、同一个在线电商账号、同一个商店的购物卡。

当多个用户共享账号时会引出三个问题。

首先，**dominance problem**。当所有推荐项（recommendations）只与共享账号的部分用户有关，至少一个用户不会获得任何相关推荐项，此时会引起“主导问题（dominance problem）”。我们会说：只有少量用户会主宰该账号。另外，考虑一个经常购物的家庭。他们通常会在购买家庭用品时偶尔也会为孩子购买玩具。现在，很有可能所有推荐项会基于更多的家庭用品来生成，该推荐系统对孩子来说是基本没啥用。

第二，**generality problem**。当推荐项不仅与共享账号下所有用户都有些相关时，并且不会与任一个用户相关时，会引发“generality problem”。当多个用户的不同品味兴趣被合到同一个账户上时，推荐系统更可能会推荐被大多数人喜欢的**通用items（general items）**，忽略掉私人的品味。

第三，如果推荐系统应该为共享账号的每个用户生成相关的推荐，每个用户如何知道哪个推荐是为你推的呢？我们称这为**“presentation problem”**。

如果上下文信息（比如：time、location、购买意图、item内容、session logs等）有提供，context-aware推荐系统是SOTA的解决方案，它可以将accounts划分成多个用户，并检测在推荐时活跃用户的身份。

然而，通常对于划分账号来说没有这么多上下文信息可提供。第一个示例是：许多公司不会保存过往的上下文信息日志，同时也不保存timestamps。第二个示例是：家庭会在一个超市里进行一起购买：他们只有一个信用卡账户，当访问商店时会将购物进行放在一起。在这种情况下，context信息对于每个家庭成员来说是相同的，不能用来将家庭账号划分成各自成员。因此，我们会引入：在缺失上下文信息下的共享账号的top-N推荐，上面的三个共享账号问题会在不使用任意上下文信息的情况下进行解决。

正式的，我们先考虑使用二元、positive-only的偏好反馈的CF的setting。我们将可提供数据表示成一个偏好矩阵，其中：行表示用户，列表示items。偏好矩阵中的每个值是1或0，其中1表示一个已知偏好，0表示未知。[12]称这种setting为one-class CF（OCCF），但它也被称为基于binary、postive-only偏好数据的top-N推荐。这种数据通常与implicit feedback有关。然而，它也是explicit feedback的结果。这不同于社交网站的例子（explicit）。其它相关的应用有：照片标签、文档的words、一个顾客购买的物品。

尽管在缺失上下文信息下共享账号的top-N推荐很重要，我们没有先前研究解决过该挑战。我们通过提出一个解决方案来解决所有item-based topN CF推荐系统的该case，基于binary、postive-only feedback来生成推荐。在这种情况下，我们覆盖了大多数应用，因为ICF很流行。大多数作者将这种流行性归结为：多个特性的组合（简洁、稳定、高效、准确），可以直觉性解释它们的推荐，并且能立即解释新进来的feedback。

我们展示了新的item-based top-N CF推荐系统，它允许我们在O(nlogn)下计算推荐分（非指数时间内）。

主要贡献如下：

- 正式介绍缺失上下文信息下的共享账号top-N推荐
- 提出解决方案
- 一个必要特性
- 实验表明：多个数据集上，可以检测共享账号下独立用户的偏好

# 2.问题定义

假设：U是用户集合，A是账号集合，I是items集合。更进一下，假设：$$U(a) \subseteq U$$是具有共享账号a的用户的集合，例如：账号a的用户集合，假设$$a(u) \in A$$是用户u属于的账号。注意：在该问题设定中，每个用户会属于一个账号。

首先，考虑用户评分矩阵$$T \in \lbrace 0, 1 \rbrace^{|U| \times |I|}$$。其中：

- $$T_{ui}=1$$表示存在用户$$u \in U$$对于item $$i \in I$$的偏好
- $$T_{ui}=0$$表示没有这样的偏好

我们给定：一个相关的推荐系统$$R_{ref}$$会产生在给定T下的期望推荐。因此，我们会说：通过相关推荐系统$$R_{ref}(T)$$在**用户评分矩阵（user-rating-matrix）**T上计算，如果i在u的top-N推荐中，则一个item i与一个user u相关。

不幸的是，在我们的问题设定中，T是未知的。因此，我们会给出**账号评分评阵（account-rating-matrix）** $$R \in \lbrace 0, 1 \rbrace^{|A| \times |I|}$$。其中：

- $$R_{ai}=1$$表示账号$$a \in A$$对于$$i \in I$$是存在一个已知偏好的
- $$R_{ai}=0$$表示不存在这样的信息

现在，缺失上下文信息的共享账号的top-N推荐，会设计一个共享账号推荐系统$$R_{sa}(R)$$，它基于账号评分评阵（account-rating-matrix）R，计算每个账号a的top $$N_a$$的推荐，如下：

- top $$N_a$$包含了在账号a的用户集合下每个用户的top-N items，有：$$N = \frac{N_a}{\|U(a)\|}$$。实际上，目标是，通过最大化具有至少一个item的用户数来避免dominance问题和generality问题。
- 账户a的用户集合中的某一用户，在top-$$Na$$中的哪个items是对他有意义的，这是很清楚的，例如：presentation problem会得到解决

注意：在上面的定义中，共享账号的推荐系统不会获得共享每个账号的用户数目作为输入。更进一步，关于共享一个账号的用户的共享兴趣，不会做出任何假设。他们可以具有完全不同的兴趣，或者部分重叠的兴趣，或者完全重合的兴趣。

最后，注意该问题定义与一个典型的group推荐问题【10】正交。

- 首先，在group推荐中，在共享账号下的用户的个体profiles通常都是已知的。而在这里，他们是未知的。
- 第二：在group推荐中，通常会假设，推荐内容会被在共享账号中的所有用户所消费；而在这里，共享账号的每个用户可以区别对它有意义的推荐，并单独消费这些推荐。

# 3.相关推荐系统

通常，推荐系统会发现，一个用户u的top-N推荐，通过首次为每个候选推荐i计算推荐得分 s(u,i)，接着选择最高分排序的top-N推荐来获得。

对于binary、postive-only feedback，大多数流行的推荐系统的其中之一是，item-based CF推荐系统【2】。这些item-based推荐系统根源于这样的意图：好的推荐会与target user偏好的目标items相似，其中在两个items间的相似性会通过使用在用户喜欢的items的各自集合间的任意相似度measure进行衡量。因此，对于一个target user u，这种推荐系统：

- 首先会发现：KNN(j)，这是对j的k个最相似items，其中：每个喜欢的item j（T_{uj}=1）通过使用一个相似measure sim(j,i)进行衡量。
- 接着，用户u对于一个候选推荐i的item-based推荐得分给定如下：

$$
S_{IB}(u, i) = s_{IB}(I(u), i) = \sum_{j \in I(u)} sim(j, i) \dot | KNN(j) \cap \lbrace i \rbrace  
$$ 

...(1)

其中，$$I(u) = \lbrace j \in I  \| T_{uj} = 1 \rbrace$$，这是u喜欢的items集合。

sim(j,i)的一个常见选择中，cosine相似度。更进一步，归一化相似度得分可以提升表现【2】。sim(j,i)的定义如下：

$$
sim(j,i) = \frac{cos(j,i)}{ \sum_{l \in KNN(j)} cos(j, l)}
$$

我们可以使用这种推荐系统作为相关推荐系统$$R_{ref}$$。

# 4.相关推荐系统的共享账号问题

简单将相关推荐系统（reference recommender system）应用到account-rating-matrix R上会导致次优结果，因为RRS会有三个共享账号问题。我们使用两个示例进行说明。在两个示例中，我们考虑：

- 两个用户$$u_a$$和$$u_b$$会共享账号s。
- 用户$$u_a$$对于items $$a_1$$和$$a_2$$具有一个已知偏好
- 用户$$u_b$$对于items $$b_1$$, $$b_2$$, $$b_3$$具有一个已知偏好
- 存在5个候候选推荐：$$r_a^1, r_a^2, r_b^1, r_b^2, r_g$$，其中$$r_a^1$$和$$r_a^2$$是对于$$u_a$$来说的好推荐；$$r_b^1$$和$$r_b^2$$是对于$$u_b$$来说的好推荐。$$r_g$$是一个完全通用的推荐，两个用户都是中性

表1和2总结了一些立即计算。

。。。

表1 

表2

两个示例都适合说明RRS存在的presentation问题。例如，考虑表1的第一行。推荐得分$$s(s, r_a^1) = 11$$是以下三项$$sim(a_1, r_a^)=5, sim(a_2, r_a^1) = 5, s(b_1, r_a^1) = 1$$的总和。因此，它可以通过$$a_1, a_2, b_1$$来进行解释。然而这是一个坏的解释，因为由于$$b_1$$的缺失，$$u_a$$会难于区分解释，$$u_b$$会错误下结论成这对他是有意义的。

# 5.解决genreality问题

前面章节表明，由于item-based RRS不会区分：一个得分是少量大相似得分的求和，还是许多小相似分得分的求程。因此会出现generality问题。因而，我们的第一步是采用item-based的推荐得分（等式1）到length-adjusted item-based推荐得分上：

$$
S_{LIB}(u, i) = S_{LIB}(I(u), i) = \frac{1}{|I(u)|^p} \cdot S_{IB}(I(u), i)
$$

...(2)

其中，超参数$$p \in [0,1]$$。尽管这种adjustment不会立即解决genreality问题，它会提供一种方式，来区分在少量大相似得分的求和，而是许多小得分求和的问题。通过选择p > 0，我们可以创建一个bias，它偏向于少量大相似得分的求程。p值越大，该bias就越大。

由于因子$$\frac{1}{\| I(u) \|^P}$$对于所有候选推荐i来说是相同的，对于用户u根据$$S_{LIB}$$和$$S_{IB}$$的top N items是相同的。然而，当我们计算两个不同用户间的得分时，$$S_{LIB}$$也会解释用户喜欢的items总量。

为了避免generality问题，我们理想地希望推荐这么一个item，如果一个item i与共享账号a中的某个用户高度相关。因此，我们会计算item i与每个个体用户$$u \in U(a)$$的推荐得分。正式的，我们希望根据它的推荐得分对所有item i进行排序：

$$
max_{u \in U(a)} S_{LIB} (I(u), i)
$$

不幸的是，我们不能计算理想推荐得分，因为$$U(a)$$和I(u)是未知的。相似的，我们只知道$$I(a) = \lbrace j \in I \I R_{aj} = 1 \rbrace$$, 它表示账号a喜欢的items集合。

然而，我们可以使用它的上界对理想的推荐得分进行近似：

$$
max_{S \in 2^{I(a)}} S_{LIB} (S, i) \geq max_{u \in U(a)} S_{LIB}(I(u), i)
$$

其中，$$2^{I(a)}$$是$$I(a)$$的幂（power），例如，该集合包含了所在I(a)的可能子集。提出的近似是一个理想得分的上界，因为I(u)的每个集合对于$$u \in U(a)$$来说是$$2^{I(a)}$$的一个元素。该近似是基于假设：所有I(a)的子集，它们对应于用户更可能生成最高的推荐得分，而非将它们随机地放在一起。

相应的，我们提出使用disambiguating item-based(DAMIB)推荐系统，关于账号a对于item i的推荐得分如下：

$$
s_{DAMIB}(a, i) = max_{S \in 2^{I(a)}} S_{LIB}(S, i)
$$

...(3)

每个得分$$s_{DAMIB}(a, i)$$对应于一个最优的子集$$S_i^* \subseteq I(a)$$：

$$
S_i^* = argmax_{S \in 2^{I(a)}} S_{LIB}(S, i)
$$

...(4)

因而，$$s_{DAMIB}(a, i) = s_{LIB}(S_i^*, i)$$。这样，DAMIB推荐系统不仅会计算推荐得分，也会发现账号a对于i的最大化length-adjuested item-based推荐得分的最优子集$$S_i^*$$。

换句话说，DAMIB推荐系统会基于直觉和task-specific准则，显式地将共享账号a划分成子集$$S_i^*$$，每个$$S_i^*$$会对于候选推荐项i之一来最大化$$S_{LIB}$$。当$$S_{LIB}(S_i^*, i)$$很高时，我们希望，$$S_i^*$$能很好地对应一个个体用户。当$$S_{LIB}(S_i^*, i)$$很低时，在共享账号上没有用户对于i是一个强推荐，我们希望$$S_i^*$$是一个随机子集。

这样，我们会避免error prone任务：来估计在共享账号上的用户数目、并基于一个通用的聚类准则，显式地将账号a划分成它的个体用户。

更进一步，由于子集可以是潜在重合，DAMIB推荐系统不会关注在共享账号上的用户的已知偏好是否强烈、轻微、或者根本不重合。

最终，注意，对于p=0来说，它总是有：$$s_{DAMIB} = s_{LIB} = s_{IB}$$。因此，item-based推荐系统是DAMIB推荐系统的一个特例。

# 6.高效计算

可以发现，在等式(3)中的最大化，需要以指数时间复杂度来直接计算$$S_{LIB}$$，称为$$2^{\I I(a) \|}$$。相应的，直接计算$$s_{DAMIB}$$是很难的。

幸运的是，我们表明：$$S_{LIB}$$会允许我们来以O(nlogn)来计算$$s_{DAMIB}$$，其中：$$n = \| I(a) \|$$。该特性由定理6.1给出。

定理6.1 ： 假设a是一个账号，它喜欢items I(a)的集合。i是一个候选推荐。如果我们将所有items $$j,l \in I(a)$$进行排序，$$rank(j) < rank(l) \Leftrightarrow sim(j,i) > sim(l,i)$$，这样，子集$$S_i^* \subseteq I(a)$$会在所有$$S \in 2^{I(a)}$$上最大化$$S_{LIB}(S, i)$$，它是ranking的一个prefix。

证明：略。。。

# 7.解决Dominance问题

DAMIB推荐系统允许我们检测，当domainance问题发现时。这是因为，由DAMIB提供的每个推荐项i会伴随着一个清晰的解释，形式为：最优子集$$S_i^*  \subseteq  I(a)$$。因此，如果union $$U_{i \in top\-N{a}} S_i^*$$只是I(a)的一个小子集时，我们知道，这个小子集会在账号a的top $$N_a$$推荐中占统治地位。

求解dominance问题，可以选择在算法1中的ALG=DAMIB，我们称为COVER。例如，我们为共享账号推荐的最终算法是DAMIB-COVER，其中DAMIB-COVER(a) = COVER(a, DAMIB)。

该DAMIB-COVAER算法会使用DAMIB得分来找出$$N_a$$的最高得分候选推荐，如果它的解释$$S_c^*$$与更高排序的候选的解释不够有区分性，我们就从top $$N_a$$中移除一个候选推荐c。$$D(S_c^*, C(a)) \geq \theta_D$$的解释区分性条件，会衡量一个候选($$S_c^*$$)以及更高排序候选(C(a))是否足够不同。

关于explanation-diffference条件的可能的启发定义是：$$S_c^* \ C(a) | \geq 0.5 \cdot | S_c^* | $$。。。


# 8.求解presentation问题

对于一个共享账号a，使用DAMIB-COVEAR生成top-Na推荐项是不够的，因为共享账号的用户不知道哪个推荐项属于哪个用户。这被称为presentation问题。

我们的解决方案是，将每个推荐项 $$i \in top-N_a$$、与它的解释(explanation)$$S_i^*$$表示在一起。我们期望：对于在top-Na中的items i的绝大多数，explanation $$S_i^*$$是u（共享账号a的某一用户）的偏好I(u)的一个子集。我们会在第9节进行验证该假设。

因此，我们将该推荐表示为item r推荐给喜欢s1, s2, s3的用户。接着，一个会用户认为s1, s2, s3是他的偏好，并知道r是推荐给他的。

# 9.实验评估

所有数所集都是公开提供的。另外，我们的源代码和数据集链接在：https://bitbucket.org/BlindReview/rsa 上有提供。该网站包含了脚本来自动化运营每个实验。另外，所有结果可以复现。

## 9.1 数据集

我们使用一个包含了真实的共享账号信息的数据集。CAMRa 2011数据集，它包含了家庭成员信息，关于用户对电影评分的一个子集。这样，我们可以使用该数据集构建真实共享账号。不幸的是，owner不希望分发该数据集，我们没有其它包含共享信息的数据集。然而，从CAMRa 2011数据集上，我们可以学到，大多数家庭账号包含了两个用户（290个家庭有272个），一些包含了三个（14/290），四个（4/290）。因此，我们会根据Zhang[15]的方法，来创建“人工伪造（synthetic）”的共享账号，随机将用户分成2个、3个、4个的组。尽管该方法并不完美，[15]表明‘synthetic’的共享账号会与CAMRa 2011数据集的真实共享账号特性相似。

我们在4个数据集中对解决方案进行评估：Yahoo!Music[13]、Movielens1M[4]，Book-Crossing[16]以及Wiki10+[17]数据集上。

YahooMusic: 14382个用户在1000首歌上进行评分：1-5分。由于我们会考虑binary、postive-only数据的问题设定。我们将4-5的评分转成1，忽略其它评分。平均的，一个用户具有8.7个偏好。

Movielens1M：包含了6038个用户在3533电影上，评分：1-5分。另外，我们会将4-5的评分转化为1. 平均一个用户具有95.3个偏好。

Book-Crossing：包含了两种信息。首先，用户在books上的评分：1-10分。与前2个数据集类似，我们会将评分8, 9, 10分转成偏好并忽略所有的其它评分。第二，也存在二元偏好，我们会添加到偏好列表中。总之，存在87835个用户，300695本书，每个user具有平均11个偏好。

wiki10+：包含了99162个tags，分配给20751个wikipedia文章。在本case中，我们考虑文章的标签推荐，文章看成是“users”的角色，tags看成是"items"的角色。如果一个文章a被打上至少一次的tag t标签，我们认为文章a具有tag t 的一个"perference"。在该context下，**一个共享账号是一个大文章**，它具有广泛主题，包含了多个更小在子主题下的“articles”。平均，每个article具有22.1个tags。

由于空间限制，我们只展示了在Yahoo!Music dataset上的数值结果。然而，对于其它三个数据集，可以在https://bitbucket.org/BlindReview/rsa上下结论，并导致相同的结论。

## 9.2 对比算法

我们比较了新算法：DAMIB-COVER，以及另两种算法。第一个对比算法是IB，是item-based RRS在account-rating-matrix上的应用，它忽略了shared account问题的存在。这是我们的baseline。第二个对比算法是IB-COVER，它被定义成IB-COVER(a) = COVER(a, IB). IB-COVER是与【14】的算法相似。

## 9.3 效果

首先，考虑到共享一个账号a（它具有$$\| U(a) \|$$个其它用户）的其中一个用户的recall。对于它的shared account来说，这是个体的top-5推荐项也出现在top-Na推荐项中的百分比，具有$$N_a = 5 \cdot |U(a)| $$。正式的，我们会定义user u的recall如下：

$$
rec(u) = \frac{|top\-5(u) \cap top\-N_a(a)}{5}
$$

理想的，在共享账号中所有用户的recall是1，这意味着，对于共享账号的top-Na是个体top-5的union，共有$$\|U(a)\|$$个用户共享账号。

现在，为了研究有多少用户会有共享账号的问题，我们会衡量用户没有获得任何相关推荐的比例，例如：不会发现单个的top-5个体推荐项在共享账号的top-Na推荐项。我们将该数目表示成$$rec_0^u$$，召回为0的用户占比。正式的，我们定义如下：

$$
rec_0^\mathcal{U} = \frac{u \in \mathcal{U} | rec(u) = 0 }{| \mathcal{U}  |}
$$

一个具有共享账号问题的用户的示例如表3所示。该表展示了两个来自Movielens1M数据集的两个真实用户，它们各自已知的偏好I(u)，以及item-based个体top-5推荐。它们item-based个体top-5推荐项看起来合理给出它们的已知偏好，这两个用户是同一家庭的一部分并且共享账号看起来是不现实的。相应的，表3也展示了对于'synthetic'账号的推荐项，它们会在两个cases中（ $$R_{sa} = IB$$以及$$R_{sa} = DAMIB-COVER$$）被两个用户共享。在case $$R_{sa} = IB, rec(562) = 0$$中，例如：user 56`2不会获得单个推荐，它会存在共享一个账号的问题。`在case $$R_{sa} = DAMIB-COVER$$中，$$rec(562) = 0.6$$，例如：user 562会获得3个好的推荐，不存在严重的问题。很明显，只有一个示例，我们需要查询数据集中的所有用户来对比不同算法。

图1展示了Yahoo!Music数据集的$$rec_0^u$$。最近邻的数目k，是一个item-based RRS的参数。存在多种方式来选择k。在这些方法中，样本是以一个offline实验的accuracy，推荐的主观质量调整；在线A/B的accuracy，计算效率，等。因此，我们会呈现关于RRS的变化带来的结果，例如：ICF RS会与其它k种选择不同。相应的，图1的每个plot会展示一个不同k的结果。

对于每种k的选择，以及单独的top-5推荐，对应于4个实验：一个账号被1、2、3、4个用户共享。注意，一个账号如果由一个用户共享，这种实际并不是共享。每个水平轴表示共享该账号的用户数目，每个垂直轴表示产生的$$rec_0^u$$。4种不同的标记表明，4种不同的共享账号推荐系统$$R_{sa}$$：

- baseline 算法IB
- IB-COVER
- 提出的DAMIB-COVER算法的两个变种

这两个变种会根据参数选择的不同有所区别：p=0.5和p=0.75。由于我们会随机重复每个实验5次，每个plot包含了5x4=20个相似的标记。然而，由于低传播、大多数标记会在相互的top上被绘制，形成dense的maker云。更进一步，由于95%的置信区间，意味着5个数据集的maker-clounds会更窄，我们没有进行绘制。相应的，两个maker-clouds会进行单独可视化，也会在5%的显著级别上具有显著不同。

我们从图1中做出4个观察：

- 首先，我们观察到，baseline的效果并不好。当他们相互共享账号时，接近19%的用户获得不相关推荐。这表明，共享账号会对于推荐系统引起极大问题
- 第二，我们提出的解法，DAMIB-COVER算法，可以提大提升$$rec_0^u$$。在一些情况下，提交是剧烈的。一个示例是$$\| U(a) | = 2$$，个体top-5会使用k=200进行生成。在本case中，当使用baseline IR算法时，12%的用户不会获得相关推荐。通过使用DAMIB-COVER(p=0.75)，该数目会减小到因子4（$$rec_0^U = 0.03$$）。
- 第三，一些IB-COVER已经在IB上进行了提升。存在多个cases，其中DAMIB-COVER会进一步在IB-COVER上提升。另外，DAMIB-COVER会胜过IB-COVER会变得在评估presentation问题时更清晰。
- 最后，当$$\| U(a) \| = 1$$，例如，当账号并不共享时，$$rec_0^u=0$$，是为baseline算法IB进行定义。然而，我们观察到，对于IB-COVER以及DAMIB算法的变种，$$rec_0^u$$可以足够低。因而，当账号不共享时，提出的DAMIB算法不会失败。

## 9.4 有限的trade-off

为了强调这一点：当没有账号共享时，DAMIB-COVER算法仍在传统设定下表现良好，我们也讨论了DAMIB-COVER的结果会在一个【2】使用的更确定的实验设定下。为了避免所有的困惑：该实验设定不会做关于共享账号的任何事。在该实验设定下，每个用户的一个偏好会被随机选中作为该用户的测试偏好$$h_u$$。如果一个用户只有一个偏好，不会有测试偏好被选中。剩余偏好会被在training matrix R中表示成1（由于没有账号会被共享，它在该case中是完全相同的）。R的所有其它条目都为0。我们将$$U_t$$定义成具有一个测试偏好的用户集合。对于每个user $$u \in U_t$$，每个算法会基于R对items $$\lbrace i \in I \| R_{ui} = 0 \rbrace$$进行排序。根据[2]，我们会使用hitrate@5来评估每个ranking。hitrate@5的给定如下：

$$
HR@5 = \frac{1}{U_t} \sum_{u \in U_t} | \lbrace h_u \rbrace \cap top5(u) | 
$$

其中，top5(u)是用户u的5个最高的ranked items。因而，HR@5给出了测试偏好在top5 推荐中的测试用户百分比。Yahoo!Music数据集的实验结果如图2所示。除了早前讨论的算法外，图2也包含了baseline算法POP的结果，非个性化算法会根据它们的流行度对所有items进行排序；例如：用户在training set中的用户会偏向于该item的数目。在该case中，我们会重复实验5次，使用一个不同的随机分。另外，由于低传播性，5个数据点通常会绘制在互相的顶部。图2展示了DAMIB-COVER和IB在HR@5上非常相似。因此，以HR@5作为全局accuracy几乎没有trade-off。

## 9.5 Presentation

在第8节中，我们提出解通过将每个推荐项与它的explanation放在一起进行presenting，来解决presentation问题。如果这样，共享账号中的一个用户可以将一个explanation看成是他的偏好的一个子集，该用户会区分该推荐项，因此可以知道推荐项对它是有意义的。对于这种解决方案，很重要的是推荐项是可辨别的，例如：它的explanation是共享账号中某一用户的偏好子集。对于一个共享账号a，我们可以使用explanation $$S_i^*$$来测量一个推荐项i的可辨识性：

$$
ldent(S_i^*) = max_{u \in U(a)} \frac{| S_i^* \cap I(u)}{ |S_i^*| }
$$

理想的，ident(S_i^*) = 1，例如，explanation中的每个item是某个用户一个偏好。在最坏的情况下，$$ident(S_i^*) = 1/ \| U(a) \|$$，例如：explanation包含了一个在共享账号中所有用户的一个等量偏好，因此没有用户可以使用推荐进行区分。

图3展示了对于多个共享账号推荐系统$$R_{sa}$$，在Yahoo!Music数据集上 $$\| U(a) \| = 2$$的top10推荐的identifiability的直方图。从图3a所示，我们可以学到，如果某个用户简单地将item-based reference算法应用到Yahoo的共享账号数据集上，会发生presentation问题：很少的推荐项可以被在共享账号中的某一用户所区分出，例如：$$ident(S_i^*) = 1$$只有10%的explanations。图3b展示了使用IB-COVER来替代IB，不会对情况带来改善。然而，图3c展示了使用DAMIB-COVER可以弹性增加推荐项的区分性，例如：$$ident(S_i^*) = 1$$有近似60%的explanations。因而，DAMIB的explanations会优于iitem-based explanations。



# 参考

- 1.[https://uploads-ssl.webflow.com/5ee118733109492e193e63d2/5f068f4e6b396d7536c1bdbc_190823%20-%20Research%20Paper%20-%20Top-N%20Recommendation%20for%20Shared%20Accounts.pdf](https://uploads-ssl.webflow.com/5ee118733109492e193e63d2/5f068f4e6b396d7536c1bdbc_190823%20-%20Research%20Paper%20-%20Top-N%20Recommendation%20for%20Shared%20Accounts.pdf)