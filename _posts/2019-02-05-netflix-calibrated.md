---
layout: post
title: netflix Calibrated推荐介绍
description: 
modified: 2019-01-30
tags: 
---

netflix在recsys 2018的paper《Calibrated Recommendations》提出了Calibrated的概念, 我们来看下：

# 抽要

当一个用户观看了70个爱情片(romance)和30个动作片(action)时，那么很合理的期望结果是：电影推荐的个性化列表中由70%的爱情片和30%的动作片组成。**这是个很重要的特性，称为“校准(calibration)”**，最近在机器学习的关于公平性（fairness）的背景下重新获得关注。在items推荐列表中，calibration可以保证：一个用户的多个（过往）兴趣领域，受它对应的占比影响。Calibration特别重要，因为推荐系统在离线环境下通常对accuracy(比如：ranking metrics)进行最优化，会**很容易导致这样的现象：一个用户的兴趣过少，推荐会被用户的主兴趣"占满"**——这可以通过“校正推荐（calibrated recommendations）”来阻止。为了这个目的，我们会描述用于量化calibration程度（degree）的指标，以及一种简单有效的re-ranking算法来对推荐系统的输出进行后处理（post-processing）。

# 1.介绍

推荐系统在许多不同应用领域提供了个性化的用户体验，包括：电商、社交网络、音乐视频流。

在本paper中，我们展示了：**根据accuracy（例如：ranking指标）训练的推荐系统，很容易为一个用户生成集中在主要兴趣领域（main areas）上的items**——当用户的兴趣领域越少时，items会趋向于未被充分表示（underrepresented）或者缺失（absent）。随着时间流逝，**这样的不平衡推荐会让用户的兴趣领域越来越窄**——这与"回音室(echo chambers)效应"或"过滤气泡(filter bubbles)效应"相似。该问题也会在以下情况中存在：一些用户共享相同的账号，其中：使用相同账号的少量活跃用户的兴趣会在推荐中“挤出”。我们会在第2节的一些思维实验(thought experiments)、以及第6节的真实数据实验中展示该效果。

Calibration在机器学习中是一个通用概念，最近在机器学习算法关于公平性(fairness)中开始流行起来。**如果关于多个分类的预测比例与实际数据点的比例相一致，那么这个分类算法被称为"calibrated"**。相类似的，在本paper中，calibrated recommendations的目标是：影响在推荐列表中一个用户的多种兴趣，以及它们合适的比例。为了这个目的，我们在第3节描绘了calibration degree的量化指标。在第4节，我们提出了一个算法，目标函数使它更接近calibrated，来对一个给定推荐的ranked list进行post-processing。等等

为了方便，我们会使用进行如下释义：

- **与items交互的用户**：观看了电影的用户
- **items类目(categories)**：genres

# 2.动机

在本节中，我们描述了一种思维实验，它演示了会造成推荐items列表变得不平衡（unbalanced）的核心机制。我们会使用三个steps开发，从最极端情况开始。

我们会考虑常用的离线环境（offline setting），其中数据集由历史的user-item交互组成，它们被分割成trainset和testset（例如：基于时间、或者随机划分）；评估目标(evaluation objective)是：预测在testset中哪些items与用户交互时会达到最佳的accuracy，通常会根据ranking metrics进行量化。该setting的优点是很容易实现，并且可应用到用于CF的公开数据集上。

在我们的示例中，假设一个电影用户在离线训练数据中播放了70部爱情片和30部动作片：我们的objective是生成一个包含10个推荐电影的列表，可以让预测该用户的test-movies的概率最大化（例如：在offline test data中被该用户播放的held-out movies）。这会最大化推荐accuracy。出于简洁性，我们假设：**两个genres是完全互斥的（例如：一个电影可以是动作片、或者是爱情片，但不能是动作爱情片）**

## 2.1 分类不平衡（class imbalance）

在第一个以及最极端情况下，假设：我们只知道用户在genres上的偏好，但我们没有在每个genre内的单个电影的额外信息。在缺少任何额外信息的情况下，该问题与机器学习中的不平衡分类问题（imbalanced classification）相类似：通过预测majority class的label，总是会获得最好的预测accuray。**在一个二分类问题中，已知有70%的数据点的label为+1的，而剩余30%数据点的label为-1，在缺少其它额外信息的情况下，为所有数据点预测label为+1总是最好的——因为会有70%的数据点是正确的**。相反的，如果我们随机使用概率70%和30%（出现在数据中的概率）来预测label +1和label -1，我们期望的预测labels只有0.7 ·
 70% + 0.3 · 30% = 58%的数据点会是正确的。

回到我们的推荐示例：在缺少其它额外信息情况下，如果我们推荐100%的爱情片给用户（一部动作片也没有），在test data上会获得最好的accuracy。

我们的极端假设是：没有额外信息提供。在真实世界中，会有更多数据提供——然而，数据总是有限或带噪声的，因而，该效应在某种程度上仍会出现。注意，该问题与任何特定的机器学习模型无关。在第6节的真实数据中，我们会展示不平衡推荐的风险：**当根据accuracy对推荐系统最优化时，用户只有很少兴趣的genres很容易被“挤出”，而用户兴趣的主要领域(main areas)则会被放大**。

该问题的另一个视角是，从有偏推荐（biases recommendations）的角度：在理想情况下，提供的数据是没有任何偏差的(biases)，**在有限数据上的朝着accuracy进行的训练会引入在推荐列表中的一个bias，例如，它会偏向(bias)于用户的主兴趣方向**。

相反的，做出更平衡（balanced）或校正（calibrated）的推荐的objective，会减少推荐的accuracy，这并不令人吃惊。

## 2.2 变化的电影概率

该节开发了一种略微更复杂些的思维实验（thought experiment）：我们假设：

每个电影i具有一个不同的概率$$p(i \mid g)$$，它表示用户u决定播放genre g电影的概率。

之前的示例中，我们已知：$$p(g_r \mid u)$$=0.7 （r: romance movies即爱情片），$$p(g_a \mid u)=0.3$$ （a: action movies即动作片）。假设两个电影集合genres是相互排斥的，用户u播放在genre g上的电影i的概率可以通过以下得到：$$p(i \mid u) = p(i \mid g) \cdot p(g \mid u)$$。为了得到最佳预测accuracy，我们已经找到具有被该用户播放的最高概率$$p(i \mid u)$$的10部电影i。我们考虑下：最可能播放的动作片$$i_{g_a,1}$$（例如：在动作片中排序第1的），以及最可能播放的第10个爱情片$$i_{g_r,10}$$，我们会获得：

$$
\frac{p(i_{g_r,10} | u)}{p(i_{g_a,1} | u)} = \underbrace{\frac{p(i_{g_r,10} | g_r)}{p(i_{g_a,1} | g_a)}}_{\approx 1/2.1} \cdot \underbrace{\frac{p(g_r | u)}{p(g_a | u)}}_{=\frac{0.7}{0.3} \approx 2.33} \approx \frac{2.33}{2.1} > 1
$$

...(1)

其中，值2.1通过MovieLens 20 Million数据集[13]确定。**在该示例的变种中，第10部爱情片要比最好的动作片具有一个更大的播放概率**。因此，根据accuracy，待推荐的最优的10个titles可以全是爱情片title（没有一部动作片）。

## 2.3 LDA

该示例受LDA的启发。LDA描述了一个用户会以一个2-step方式来选择一个电影：用户首先选择一个genre(topic)，然后在该选中genre中选择一个电影(word)。提到LDA有三个原因。

首先，如果我们假设，真实用户选择一部电影会遵循2-step过程，那么LDA模型是合适的模型。当该LDA被训练时，它可以捕获每个用户兴趣的正确平衡(correct balance)，以及正确的比例。因而，当遵循该生成过程时，会得到平衡的推荐，推荐列表会通过一次添加一个title的方式迭代式生成：首先，为用户u学到的genre分布$$p(g \mid u)$$中抽样一个genre g，接着根据genre g从学到的分布$$p(i \mid g)$$中抽样一个电影i。与根据$$p(i \mid u)$$进行ranking的电影相对比，Sampling出来的电影会产生更低的accuracy，其中: $$p(i \mid u) = \sum_g p(i \mid g) \cdot p(g \mid u)$$。原因是，具有较小概率值$$p(i \mid u)$$的电影i，会在接近推荐列表的top位置的被抽样到。相反的，ranking是deterministic的，并能保证：用户u喜欢具有最大概率$$p(i \mid u)$$的电影i，会在推荐列表的top，很明显：如果学到的概率$$p(i \mid u)$$被正确估计，那么可以在test data上达到最佳的accuracy。

略 

# 3.Calibration指标

在本节中，我们描述了关于推荐电影列表的量化calibration degree的指标。我们考虑两个分布，两者都基于每个电影i的genres g分布$$p(g \mid i)$$，假设如下：

- $$p(g \mid u)$$：用户u在过去播放过的电影集合H在genres g上的分布:

$$
p(g | u) = \frac{\sum\limits_{i \in H} w_{u,i} \cdot p(g | i)}{\sum\limits_{i \in H} w_{u,i}}
$$

...(2)

其中，$$w_{u,i}$$是电影i的weight，例如：用户u最近播放有多近。等式(7)有一个正则版本。

- $$q(g \mid u)$$：推荐给user u的电影列表I在genres g上的分布：

$$
q(g | u) = \frac{\sum\limits_{i \in I} w_{r(i)} \cdot p(g | i)}{\sum\limits_{i \in I} w_{r(i)}}
$$

...(3)

其中，I是推荐电影的集合。电影i的weight会因为它在推荐中的rank r(i)被表示为$$w_{r(i)}$$。可能选项包括在ranking指标中所使用的weighting schemes，比如：MRR和nDCG.

**有许多方法来决定这两个分布$$q(g \mid u)$$和$$p(g \mid u)$$是否相似**。为了说明这样的分布从有限数据中（由N个推荐电影和M个被该用户播放电影组合）估计得到，使用零假设：两个分布是相同的。这通常将一个独立检验转化成在两个随机变量上的多项分布：genres g，以及一个影响两个电影集合(I和H)的变量。给定：N或M可能实际很小，这对于exact tests是必需的（像多项检验和fisher test）。这些tests在实际上是不可计算的。一种计算高效的方法是：渐近检验（asymptotic tests），比如：G-test或$$x^2$$-test。

我们不会计算p值，我们会忽略有限数据的大小N和M的影响，直接计算分布$$p(g \mid u)$$和$$q(g \mid u)$$。为了该目的，我们会使用KL散度作为calibration metric $$C_{KL}(p, q)$$：

$$
C_{KL}(p, q) = KL(p || \hat{q}) = \sum\limits_{g} log \frac{p(g | u)}{\hat{q}(g | u)}
$$

...(4)

其中，我们会使用$$p(g \mid u)$$作为target分布。**如果$$q(g \mid u)$$与它相似，$$C_{KL}(p, q)$$会具有小值**。给定，对于一个genre g，如果$$q(g \mid u)=0$$并且$$p(g \mid u) > 0$$，则KL散度会背离（diverge），我们会使用下式替代：

$$
\hat{q}(g | u ) = (1-\alpha) \cdot q(g | u) + \alpha \cdot p(g | u)
$$

...(5)

其中，$$\alpha > 0$$，$$\alpha$$值小，以便$$q \approx \hat{q}$$。在我们的实验中，我们会使用$$\alpha = 0.01$$。KL散度具有许多属性，正是在推荐中calibration degree的量化所需：

- (1) 完美校正(perfect calibration)时，KL为0：$$p(g \mid u)  = \hat{q}(g \mid u)$$
- (2) 当$$p(g \mid u)$$很小时，对于在$$p(g \mid u)$$和$$\hat{q}(g \mid u)$$间的微小差异很敏感。例如，如果一个用户播放的genre只有2%的时间，推荐该genre 1%在KL散度上会被认为是一个较大的差异。比起（一个genre被播放50%，但推荐只有49%）的case差异要更大。
- (3) 它喜欢更平均、并且更不极端的分布：如表1所示，如果一个用户播放一个genre 30%的时间，推荐31%该genre 会被认为要比29%要好。

这些属性确保了该用户很少播放的genres也可以在推荐列表中相应的比例被影响。作为KL散度的替代，你也可以使用其它f-散度，比如：在p和q间的Hellinger距离，$$C_H(p,q) = H(p,q) = \| \sqrt{p} - \sqrt{q} \|_2 / 2$$，其中$$\| \cdot \|_2$$表示概率向量(跨geners)的2-norm。Hellinger距离在零值上是定义良好的；它也对p和q间的小差异敏感，并且当p很小时，degree会小于KL散度。

整体calibration metric C可以通过跨所有users进行$$C(p, q)$$平均。

# 4.Calibration方法

推荐的calibration是一个与list相关的特性（list-property）。由于许多推荐系统以用一种pointwise/pariwise的方式进行训练，在训练中可能不包括calibration。因而建议：对推荐系统的预测列表以post-processing方式进行re-rank，这也是机器学习中一种calibrating常用方法。为了决定N个推荐电影的最优集合$$I^*$$，我们会使用最大间隔相关度（maximum marginal relevance）：

$$
I^* = \underset{I,|I|=N}{argmax} \lbrace (1-\lambda) \cdot s(I) - \lambda \cdot C_{KL} (p, q(I)) \rbrace
$$

...(6)

其中，$$\lambda \in [0, 1]$$决定着两项间的trade-off:

- (1) s(I)：$$s(i)$$表示电影$$i \in I$$被推荐系统预测的scores ，其中：$$s(I) = \sum_{i \in I} s(i)$$。注意，你可以为每个电影的score使用一个单调转换。
- (2) $$C_{KL}$$：calibration metric(等式4)，我们已经显式表示了在推荐电影I上的q依赖，它会在等式(6)进行优化

同时注意，更好的calibration会引起一个更低的calibration score，因此我们在最大化问题中必须使用它的负值。

在关注accuracy的推荐与calibration间的trade-off，可以通过等式(6)的$$\lambda$$进行控制。我们会考虑calibration作为推荐列表的一个重要属性，如第5节所示，它会需要一个相当大的值$$\lambda$$。

**寻找N个推荐电影的最优集合$$I^*$$是一个组合优化问题，它是NP-hard的**。在附录中，我们会描述该最优化问题的贪婪最优化（greedy optimization）等价于一个代理次模函数（surrogate submodular）函数的贪婪最优化。次模函数的贪婪最优化可以达到一个$$(1-1/e)$$的最优化保证，其中e是欧拉数。**贪婪最优化会从empty set开始，迭代式地每次添加一个电影i：在step n，当我们已经具有n-1个电影组成的集合$$I_{n-1}$$，对于集合$$I_{n-1} \cup \lbrace i \rbrace $$可以最大化等式(6)的电影i被添加进行来获得$$I_n$$**。该贪婪方法具有额外优点。

- 首先，它会生成一个关于电影的有序列表（ordered/ranked list）。
- 第二，该贪婪方法在每个step产生的list在相同size的lists间是$$(1-1/e)$$最优的。

即使我们可以生成一个关于N部电影的ranked list，用户可能只会看到前n部（n<N）的推荐，比如：剩余电影只会在下滑后在视见区（view-port）变得可见。除此之外，用户可能会自顶向下扫描关于N部电影的list。在两种情况下，次模函数的greedy optimization会自动保证推荐列表中每个sub-list的前n部电影(n<N)是(1-1/e)最优的。

注意，该方法允许一个电影i根据可能的多个genres g进行加权，如等式(2)和(3)中所用的$$p(g \mid i)$$。再者，如果你根据多个不同的categories（例如：genres、subgenres、languages、movie-vs.-TV-show, etc）对推荐列表进行calibrate，会为每个category添加一个独立的calibration项 $$C_{KL}^{(category)}$$，并使用期望的weight/importance $$\lambda^{(category)}$$。生成的多个次模函数的和(sum)仍是一个次模函数，因而最优化问题仍然有效。

# 5.相关概念

Calibration在机器学习中常被使用，主要在分类中，通常发现简单的post-processing方法很有效。在最近几年，calibration再获关注，特别是在机器学习的fairness中。

在推荐系统文献中，除了accuracy外还有许多指标（详见[21]），其中diversity与calibration比较接近。

## 5.1 Diversity

Diversity在许多papers中有定义，例如：最小冗余（minimal redundancy）或推荐items间的相似度，可以帮助避免推荐中100%都是爱情片：假设只有两种电影，最diverse的推荐为50%的爱情片和50%的动作片。如果有额外的电影类型，推荐的diversity可以通过推荐用户没观看过的其它genres来增加，比如：儿童片或记录片。Diversity不会保证将动作片的比例从0%增加到30%，从而影响用户的兴趣度。如果在accuracy和diversity之间的trade-off被选定，你可以获得well-calibrated推荐。**这在实际中很难达到，因为该trade-off对于每个用户是不同的**。这表明，diversity的目标并不使用合适比例来直接影响一个用户的多种兴趣。这与calibrated推荐有一个主要的不同之处。

**第二个关键不同点是：diversity可以帮助用户逃脱可能的filter bubble现象，因为它可能包括用户未曾播放过的genres。而calibrated recommendations并没有提供这个重要特性**。这驱使我们对calibrated推荐进行一个简单扩展，以便从用户过往兴趣之外的genres的电影可以被添加到推荐列表中：假设$$p_0(g)$$表示一个先验分布，对于所有genres g会使用正值，从而提升在推荐中的diversity——两个明显选择是：均匀分布（uniform distribution）、或所有用户在genre分布上的平均。这种diversity-promoting先验$$p_0(g)$$以及calibration target $$p(g \mid u)$$的加权平均：

$$
\bar{p}(g|u) = \beta \cdot p_0(g) + (1-\beta) \cdot p(g | u)
$$

...(7)

其中，参数$$\beta \in [0, 1]$$，决定了在diversity和calibration间的trade-off。这种extended calibration probability $$\bar{p}(g \mid u)$$可以被用于替代$$p(g \mid u)$$。

在许多paper中，**如果一个list只有少量的冗余度或者 在items相似度低，就认为是diverse的**。已经提出的大多数方法会生成这样的diverse推荐，比如：[4,15,31,32]，包括DPP(行列式点过程)【8，11】，次模最优化【1，2，19】。

**第二条研究线是：在还未选择任意n-1个items ranked/displayed上（比如：一个浏览模型），对用户从推荐列表中选择第n个item的概率进行建模**。该思想会产生ranking metric（称为：ERR），也被用于生成一个更diverse ranked list的方法中。

只有少量paper解决了该重要的issue：推荐会以适当比例影响用户的多种兴趣[9,25,26]，我们会在下面讨论。

比例性的思想首先在[9]中关于搜索结果多样化中提出。在[9]中，提出的指标，称为DP，本质上是一个在分布$$p(g \mid u)$$和$$g(g \mid u)$$间的平方差的修改版本。当它满足calibration metrics的性质1时，它不会表现出其它两个性质：如表1所示，对于target proportions为：60%:40%，当两个genres中具有7:3会接收更不平衡的推荐，但会与均匀5:5的情况一样，得到相同的DP=1。假设：两者都脱离6:4的理想推荐（将某一电影放到另一个genre中），根据性质（3），5:5可以比7:3接收一个更好的calibration score。性质（2）也不会满足，因为当10部电影被评建时，对于1部电影是如何与target分布相背离的程度，DP=1——理想上，该得分对于target distribution 70%:30%会更糟糕，因为它比60%:40%更极端。注意，KL散度会满足表1的性质。在[9]中，生成一个proportional list的算法会使用用于在选举(election)之后坐位安排(seat assignment)的过程，因此，每个party的坐位会与它们收来的投票数(votes)成比例。他们为该过程(procedure)开发了一个概率化版本来解决items属于多个类目的问题，并发现该方法的效果要好于在实验中的原始实验。在完美比例不能达到的情况下，会发现具有某些偏差(deviations)的一个近似解，它们的算法必须将偏差(deviaitons)看成与现有metric不同，因为他们在概念上是无关的。关于该近似解是否服从在calibrated recommendations中所期望的属性是不明显的。

在[25]中，个性化多样性(personalized diversification)从次模化（submodularity）的角度解决。而他们在[25]中提出的一个次模目标函数（等式(2)），由一个log-sum项组成，与我们附录中的等式(8)相似，它与[25]中未描述的KL散度有关。在[25]中仍未讲明的是，该次模函数的实际目标是，推荐多个与它们的weights（例如：[25]中的CTR）成比例的item-categories。

[26]中提出的metric叫BinomDiv，是精心制作的，并且满足性质(2)和(3)：例如：关于表1中的target proportions 60%:40%，7:3更极端的推荐是，比更平衡的5:5得到一个更差的分值。这对于proportionality是很重要的性质。它们的指标不满足性质1，然而，即使更放松，采用在perfect calibration情况下的相同固定值（替代0）：如果$$p(g \mid u)=q(g \mid u)$$，该指标可以采用不同值，取决于推荐列表的长度、以及genres $$p(g \mid u)$$的分布，见表1. 这有两个缺点：首先，metric的一个给定值，不能为提供一种该推荐是如何calibrate的感觉——对于一个特定用户，它只允许你根据不同推荐列表做出相对比较。第二，假如每个用户趋向于具有不同分布的兴趣/类目(interests/genres)，该指标不能简单地跨用户平均的方式来获得一个聚合指标。为了评估，该指标会转化成一个z-score。我们也发现：当推荐电影的数目超过数百时，指标计算会遭受数值下溢（numerical underflow）——这在许多应用中会引起问题，比如：top10推荐，同时也有推荐数百items的场景，比如：Netflix主页。除此之外，我们注意到，增加一个先验（prior）的思想在[26]有提及。该算法会基于最大间隔相关度（maximum marginal relevance）[6]。这些指标可能不是次模的（submodular），然而，他们可能不存在一个最优保证。

## 5.2 公平性（Fairness）

在机器学习领域中，fairness的重要性越来越大，例如：[33]。Fairness是避免对在总体（polulation）中特定人或人群的歧视，例如，基于gender、race、age、等。它通常与总体中个人的scores或class labels有关。


在文献中，提出了许多公平性准则（fairness criteria），包括：calibration、等可能性(equalized odds)、机会均等（equal opportunity）、统计平等（statistical parity）。[12]中提出了一种post-processing方法，使用equalized odds作为fairness-metric。[28]提出了将fairness集成到training objective中。

在CF的内容中，[29]讨论了在user-base中的少量亚种族（sub-populations，例如：人口不均衡），以及更低活跃的亚种族（例如：提供更少评分的人）可能会收到更偏的推荐。除此之外，[29]还关注在rating prediction和RMSE，替代隐式反馈数据和ranking metrics的更相关场景。

在该paper中，我们会考虑fairness的一个完整概念：除了考虑人的fairness外，我们会考虑一个用户多种兴趣(various interests)上的公平性（fairness），目标是根据它相应的兴趣比例进行影响。在本节剩余部分，我们会描述，为什么calibration criteria对于fairness的非标准概念特别有用。

如[16]所示，calibration和equalized odds/equal opportunity不会被同时满足（精确，非近似）——除了两个特例：当机器学习模型做出perfect predictions（它们会被公平对待），或者当不同分组的用户具有相同的base rate时，例如：相同比例的positive classification-labels，它通常不会在真实中hold。假设一个用户通常使用不同比例播放genres（比如：70%爱情片，30%动作片），这两种genres（在fairness文献中被称为"groups"）的base rate很明显不同，对于在这两个genres中的电影的平均预测得分也不同。因此，fiarness criteria equalized odds、equal opportunity以及statistical parity不能立即应用到我们的context中。这驱使我们在推荐中将calibration作为一种合适的fairness criteria。

# 参考

- 1.[https://dl.acm.org/citation.cfm?id=3240372](https://dl.acm.org/citation.cfm?id=3240372)