---
layout: post
title: feedback loops介绍
description: 
modified: 2019-10-28
tags: 
---

deepmind在19年发了篇关于feedback loops的paper：《Degenerate Feedback Loops in Recommender Systems》，我们可以来看下它的paper介绍：

# 摘要

**在生产环境中的推荐系统大量使用机器学习。这些系统的决策会影响user beliefs和preferences，从而影响学习系统接受到的feedback——这会创建一个feedback loop**。这个现象被称为“echo chambers”或“filter bubbles”。本paper提供了一个新的理论分析，来检查user dynamics的角色、以及推荐系统的行为，从而从filter bubble效应中解脱出来。另外，我们提供了实际解决方案来减小系统的退化（degeneracy）。

# 介绍

推荐系统广泛被用于提供个性化商品和信息流。这些系统会采用user的个人特征以及过往行为来生成一个符合用户个人偏好的items list。虽然商业上很成功，**但这样的系统会产生一个关于窄曝光（narrowing exposure）的自我增强的模式，从而影响用户兴趣，这样的问题称为“echo chamber”和“filte bubble”**。大量研究都致力于对曝光给用户的items set上使用favor diversity来解决。然而，echo chamber和filter bubble效应的当前理解很有限，实验分析表明会有冲突结果。

在本paper中，我们将echo chamber定义为这样的效应：**通过重复曝光一个特定item或item类目，一个用户的兴趣被正向地（positively reinforced）、或负向地（negatively）增强**。这是Sunstein(2009)的定义概括，其中，该术语通常指的是：对相似政治意见的over-exposure和limited-exposure，会增强个人的已存在信念（beliefs）。Pariser(2011)引入了filter bubble的定义，来描述推荐系统会选择有限内容来服务用户。我们提供了一个理论方法来允许我们单独考虑echo chamber和filter bubble效应。**我们将用户兴趣看成是一个动态系统（dynamical system），并将兴趣看成是系统的退化点（degeneracy points）**。我们考虑不同模型的动态性，并确定系统随时间degenerate的充分条件集合。我们接着使用该分析来理解推荐系统所扮演的角色。最终我们展示了：在一个使用模拟数据和多个经典bandit算法的仿真学习中，在user dynamics和推荐系统actions间的相互作用。结果表明，推荐系统设计的许多缺陷（pitfalls）和缓和策略。

# 相关工作

。。。

# 3.模型

我们考虑一个推荐系统，它会与用户随时间一直交互。在每个timestep t上，recommender系统会从一个有限的item set M中提供l个items给一个user。总之，该系统的目标是：将可能感兴趣的items呈现（present）给用户。我们假设：

- **在timestep t上，user在一个item $a \in M$上的兴趣，可以通过函数$$\mu_t: M \rightarrow R$$来描述。**
- 如果用户对该item很感兴趣，那么$$\mu_t(a)$$很大（positive）；反之为很小（negative）


给定一个推荐(recommendation) $$a_t = (a_t^1, \cdots, a_t^l) \in M^l$$，用户基于它当前的兴趣 $$\mu_t(a_t^1), \cdots, \mu_t(a_t^l)$$提供一些feedback $$c_t$$。该交互具有多个effects：在推荐系统传统文献中，feedback $$c_t$$被用于更新用于获取推荐$$a_t$$推荐系统的internal model $$\theta_t$$，接着新模型$$\theta_{t+1}$$会依赖$$\theta_t, a_t, c_t$$。实际上，$$\theta_t$$通常会预测user feedback的分布来决定哪个items $$a_t$$应被呈现给该用户。在本paper中，我们关注另一个效应（effect），并显式考虑用户与推荐系统的交互可能会在下一次交互时以不同的items来变更他的兴趣，这样，该兴趣$$\mu_{t+1}$$会依赖于$$\mu_t, a_t, c_t$$。交互的完整模型如图1所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/2ed24fc42044cfe9e6ca014cf8e512a329d2f37f05bdf544c1a4362c6ef08c51aa5da0b7c7ec70621d93eb0b6922d066?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750" width="400">

图1

我们对用户兴趣的演化研究很感兴趣。这样的一个演化示例是，兴趣会通过用户与所推荐items的交互进行增强，也就是说：

- 如果用户在timestep t上对一个item a进行点击，那么$$\mu_{t+1}(a) > \mu_t(a)$$；
- 如果a被展示但未被点击，则$$\mu_{t+1}(a) < \mu_t(a)$$. （这里, $$c_t \in \lbrace 0, 1 \rbrace^l$$可以被定义成所对应items的关于点击的indicator vector）

为了分析echo chamber或filter bubble效应，**我们对于用户兴趣在什么时候发生非常剧烈变化非常感兴趣**。在我们的模型中，这会转成$$\mu_t(a)$$采用任意不同于初始兴趣$$\mu_0(a)$$的值：大的positive values表示用户会对item a变得非常感兴趣，而大的negative values表示用户不喜欢a。正式的，**对于一个有限item set M，我们会问到：L2-norm $$\| \mu_t - \mu_0 \|_2 = (\sum_{a \in M} (\mu_t(a) - \mu_0(a))^2)^{1/2}$$是否可以变得任意大？**如果满足下面条件，几乎必然用户的兴趣序列$$\mu_t$$被称为“弱退化（weakly  degenerate）”(证明见附录)：

$$
\underset{t \rightarrow \infty} {lim} sup || \mu_t - \mu_0 ||_2 = \infty, 
$$

...(1)

关于degenracy的"stronger"概念，需要满足：一旦$$\mu_t$$远离$$\mu_0$$，它就是个stronger degeneracy。序列$$\mu_t$$是一个较强的degenrate，（ almost surely）需满足：

$$
\underset{t \rightarrow \infty} {lim} || \mu_t - \mu_0 ||_2 = \infty
$$

...(2)

下一节中，我们将展示，在$$\mu_t$$的动态演进上的充分条件较缓和时，degeracy发生的强弱。

对于一个无限item set M的情况，存在多种方式来扩展上述定义。出于简化，我们只考虑使用$$ \| \mu_t - \mu_0 \|_2 $$来替代等式(1)和等式(2)中的$$sup_{a \in M} \mid \mu_t(a) - \mu_0(a) \mid $$，当M是有限时，它等于原始定义。

# 3. 用户兴趣动态性——Echo Chamber

由于items通常以多样的类目(t diverse categorie)的方式呈现，我们简化猜想：**它们间是相互独立的**。通过对于所有t (例如：$$M = \lbrace a \rbrace $$)，设置$$l=1$$和$$a_t^1 = a$$，我们可以移除推荐系统的影响，并单独考虑用户的动态性( Dynamics)。这使得我们分析echo chamber effect：如果item a被无限次推到，兴趣 $$\mu_t(a)$$会发生什么？

由于a是固定的，为了简化概念，我们将$$\mu_t(a)$$替换成$$\mu_t$$。给定$$a_t$$，**根据图1，$$\mu_{t+1}$$是一个关于$$\mu_t$$的函数**（可能随机，由于$$\mu_{t+1}$$依赖于$$c_t$$和$$\mu_t$$，$$c_t$$依赖于$$\mu_t$$）。我们考虑当漂移量（drift） $$\mu_{t+1} - \mu_t$$是一个非线性随机函数时的通用case；对于drift的deterministic模型在附录B中。

**非线性随机模型（Nonlinear Stochastic Model）**

$$\mu_{t+1} = \mu_t + f(\mu_t, \xi_t)$$

其中：

- 我们假设$$\mu_0 \in R$$是固定的
- $$(\xi_t)_{t=1}^{\infty}$$是一个关于独立均匀分布的随机变量的无限序列，它引入噪声到系统中（例如：$$\mu_{t+1}$$是一个$$\mu_t$$的随机函数）
- 函数 $$f: R \times [0, 1]$$是假设是可测的，但其它任意。通过$$U([0, 1])$$来定义[0,1]上的均匀分布，假设：

$$
\bar{f}(\mu) = E_{\xi \sim U([0,1])} [f(\mu, \xi)]
$$

是当$$\mu_t = \mu$$的期望增量$$\mu_{t+1} - \mu_t$$。我们也定义了：

$$
F(\mu, x) = P_{\xi \sim U([0,1])} (f(\mu, \xi) \leq x)
$$

是increment的累积分布。$$\mu_t$$的渐近行为依赖于f，但在mild假设下，系统会微弱（定理1）／较强（定理2）退化。

**定理1（弱退化weak degeneracy）**。假设F对于所有$$\mu \in R$$在$$(\mu, 0)$$上是连续的，存在一个$$\mu_o \in R$$，使得：

- 1) 对于所有$$\mu \geq \mu_o$$，有$$F(\mu, 0) < 1$$
- 2) 对于所有$$\mu \geq \mu_o$$，有$$F(\mu, 0) > 0$$

那么序列$$\mu_t$$是weakly degenerate，比如：$$\underset{t \rightarrow \infty}{lim} sup \mid \mu_t \mid = \infty$$

该假设保证了在任意闭区间内，存在一个常数概率，当分别从$$\mu_o$$的左侧/右侧开始时，该random walk会逃离区间左侧/右侧。在stronger condition下，可以保证random walk的分散性（divergence）。

**定理2 (强退化: strong degeneracy)**

假设定理1恒定，另外存在$$c \in R$$使得 $$\| \mu_{t+1} - \mu_t \| \leq c$$，存在一个$$ \epsilon > 0$$，使得对于所有足够大的$$\mu$$，有$$f(\mu) > \epsilon$$，对于所有足够小的$$\mu$$有$$f(\mu) \leq - \epsilon$$。接着，$$limit_{t \rightarrow \infty} \mu_t = \infty$$或者$$limit_{t \rightarrow \infty} \mu_t = -\infty$$。

直觉上，如果用户兴趣具有一些关于drift的非零概率，weak degeneracy在一个随机环境中发生。而strong degeneracy会持有额外的$$\mu_{t+1} - \mu_t$$被限制，对于$$\mu_t$$足够大/小，而增量$$\mu_{t+1} - \mu_t$$具有positive/negative drift，它大于一个constant。

定理1和2表明，用户兴趣会在非常温和的条件下退化（degenerate），特别的是在我们的模似实验中。在这样的cases中，如果一个item（或一个item category）是被展示有限多次时，degeneracy可以被避免，否则你只能寄希望于控制$$\mu_t$$退化的有多快了（例如：趋向于$$\infty$$）。

# 4.系统设计角色——filter bubble

在之前的部分讨论了对于不同user interest dynamics的degeneracy的条件。在本节中，会检查另一面：**推荐系统动作对于创建filter bubbles上的影响**。我们通常不知道现实世界用户兴趣的动态性。然而，我们考虑echo chamber/filter bubble的相关场景：其中在一些items上的用户兴趣具有退化动态性(degenerative dynamics)，并检查了如何设计一个推荐系统来减缓degeneracy过程。**我们会考虑三个维度：model accuracy，曝光量（exploration amount），growing candidate pool**。

## 4.1 Model accuracy

推荐系统设计者的一个常见目标是，增加internal model $$\theta_t$$的预测accuracy。然而，模型accuracy如何与greedy optimal $$a_t$$组合一起来影响degeneration的速度？对于exact predictions的极端示例，**例如：$$\theta_t = \mu_t$$，我们会将这样的预测模型为“oracle model”**。我们会在前提假设法（surfacing assumption）下讨论，oracle模型与greedily optimal action selection组合来生成快速的degeneracy。

为了具体分析该问题，对于$$\mu_t(a)$$的$$a \in M$$，我们会关注degenerate线性动态模型，例如：$$\mu_{t+1}(a) = (1+k) \mu_t(a) + b$$。接着，我们可以为$$\mu_t(a)$$求解，对于$$\mid 1+k(a) \mid > 1$$来获得：

$$
\mu_t(a) = (\mu_0(a) + \frac{b(a)}{k(a)} (1 + k(a))^t - \frac{b(a)}{k(a)}
$$

**Sufacing Assuption**:

假设$$[m] = \lbrace 1,2, \cdots, m \rbrace $$是size=m的candidate set。如果一个items子集 $$S \subset [m]$$会产生positive degenerate  （例如，对于所有$$a \in S$$, $$\mu_t(a) \rightarrow +\infty $$），那么我们假设：**存在一个时间$$\tau > 0$$，对于所有$$ t \geq \tau$$，S会占据根据$$\mu_t$$值生成的top $$\mid S \mid$$的items**。该$$\mu_t$$由指数函数 $$\mid 1 + k(a) \mid $$的base value进行排序。

**如果给定足够的曝光（exposure）时间，surfacing assumption可以确保很快地将items surface退化到top list上**。它可以被泛化到关于$$\mu_t$$的非线性随机动态性上，从而提供：来自S的items具有一个稳定的随时间退化速度$$\mid \mu_t(a) - \mu_0(a) \mid /t$$的排序。

在通用的surfacing assumption下，在时间$$\tau$$之后，退化（degeneration）的最快方式是：根据$$\mu_t$$或oracle model的$$\theta_t$$来服务top l items。即使该assuption有一定程度的冲突，oracle model仍会产生非常高效的退化（degeneracy），通过根据$$\mu_t$$来选择top l items，由于较高的$$\mu_t$$，他不会接受到positive feedback，从而增加$$\mu_{t+1}$$、并增强过往选择。

**实际上，推荐系统模型是不准确的（inaccurate）。我们可以将inaccurate models看成是具有不同级别的noises(添加到$$\theta_t$$)的oracle model**。

## 4.2 探索量（Amount of Exploration）

考虑一种$$\epsilon$$-random exploration，其中$$a_t$$总是从一个有限candidate pool [m]中（它在$$\theta_t$$上具有一个uniform $$\epsilon$$ noise），根据$$\theta_t^{'} = \theta_t + U([-\epsilon,  \epsilon])$$，选择出top l items。

**给定相同的模型序列$$\theta_t$$，$$\epsilon$$越大，系统退化（degenerate）越慢**。然而，实际上，$$\theta_t$$从observations上学到，在一个oracle model上添加的random expoloration可能会加速退化（degeneration）：**random exploration可以随时间展示最正向的degenerating items，使得suracing assumption更可能为true（在图5中，我们在仿真实验中展示了该现象）**。另外，如果user interests具有degenerative dynamics，随机均匀推荐items会导致degeration，虽然相当慢。

**接着我们如何确保推荐系统不会使得user interests退化（degenerate）？**一种方法是，限制一个item服务给user的次数（times）， 越多次会使得用户兴趣动态性退化。实际上，很难检测哪个items与dynamics退化相关，然而，如果所有items都只服务一个有限次数，这通常会阻止degeneration，这也暗示着需要一个不断增长的candidate items池子。

## 4.3 Growing Candidate Pool M

有了一个不断增长的（growing） candidate pool，在每个timestep时，会有一个关于new items的额外集合可提供给该user。从而，function $$\mu_t$$的domain会随时间t的增加而扩展（expands）。**线性增加新items通常是一个避免退化的必要条件**，因为在一个有限/任意次线性（sublinearly）增长的candidate pool上，通过鸽巢原理(pigeon hole principle)，必须存在至少一个item，在最坏情况下它会退化（degenerate）（在定理2描述的通用条件下）。然而，有了一个至少线性增长的candidate pool M，系统会潜在利用任意item的最大服务次数来阻止degeneration。

# 5.模拟实验

本节考虑一个$$\mu_t$$简单的degenerative dynamics，并检查在5个不同的推荐系统模型中的degeneration速度。我们进一步演示了，增加new items到candidate pool中会有一个对抗system degeneracy的有效解法。

我们为图1中一个推荐系统和一个用户间的交互创建了一个simulation。初始size=$$m_0$$，在timestep t上的size=$$m_t$$。**在timestep t时，一个推荐系统会从$$m_t$$个items中，根据内部模型$$\theta_t$$来选择top l 个items  $$a_t = (a_t^1, \cdots, a_t^l)$$服务给一个user**。

该user会独立考虑l items的每一个，并选择点击一个子集（也可能不点击，为空），从而生成一个size=l的binary vector $$c_t$$，其中$$c_t(a_t^i)$$会给出在item $$a_t^i$$上的user feedback，根据$$c_t(a_t^i) \sim Bernoulli(\phi(\mu_t(a_t^i)))$$，其中$$\phi$$是sigmoid function $$\phi(x) = 1/(1 + e^{-x})$$。

接着该系统会基于过往行为（past actions）、feedbacks和当前模型参数$$\theta_t$$来更新模型$$\theta_{t+1}$$。我们假设，用户兴趣通过$$\theta(a')$$增加/减少，如果item $$a'$$会收到/未收到一个点击，例如：

$$
\mu_{t+1}(a_t^i) - \mu_t(a_t^i) = \begin{cases}
\delta(a_t^i)  & \text{if  $ c_t(a_t^i) = 1 $ } \\
-\delta(a_t^i) & \text{otherwise}
\end{cases}
$$

...(3)

其中，function $$\delta$$会从candidate set映射到R上。从定理2中，我们知道对于所有item，有$$\mu_t \rightarrow \pm \infty$$。在该实验中，我们设置l=5,并从一个均匀随机分布$$U([-0.01, 0.01])$$中抽样drift $$\delta$$。对于所有items，用户的初始兴趣$$\mu_0$$是独立的，它们从一个均匀随机分布U([-1,1])中抽样得到。

内部推荐系统模型根据以下5个算法更新：

- **Random model**
- **Oracle**：
- **UCB Multi-armed bandit**
- **Thompson Multi-armed bandit**

## 6.1 Echo Chamber & Filter Bubble effect

我们通过在一个fixed size $$m_t = m = 100$$、time horizon T=5000的candidate pool上运行仿真，来检查echo chamber和filter bubble效应。

在图2中，我们展示了user interest $$\mu_t$$（左列）的degenreation、以及每个item的serving rate（右列），因为每个推荐模型会随时间演化。一个item的serving rate展示了在report interval内服务的频次。为了清楚地观察该分布，我们根据report time上的z-values对items进行排序。**尽管所有模型会造成user interest degeneration，degeneration的speeds相当不同（Optimal Oracle > Oracle, TS, UCB > Random Model）**。Oracle, TS 和 UCB是基于$$\mu_t$$优化的，因此我们可以看到对于$$\mu_t$$有一个positive的degenerative dynamics。Optimal Oracle会直接在degeneration speed上进行优化，而不会在$$\mu_t$$上，因此我们可以看到在$$\mu_t$$上同时有一个positive和negative degeneration。Random Model也会在两个方向上对$$\mu_t$$进行drifts，但以一个更慢的rate。然而，除了Random Model外，在所服务的top items上和top user interests会非常快速地收窄降到$$l=5$$的最positive的reinfofced items上。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/83440aa88143fa3d473d0d9e07ffc8d011cc2f578a60db119c9b5f9fb94569bbe91668f8b4fd181f168effd2d095916d?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2

**Degenracy Speed**

接着，我们在fixed和growing的两种candidate sets上，为5个推荐系统模型比较了degeneracy speed。由于测量系统degeneracy的L2矩离对于所有五种模型来说是线性对称的，我们可以在有限candidate pools上，对于不同的实验设定比较$$\mid \mu_t - \mu_0 \mid_2 /t $$。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ecd224206082b35a69af5945caafbc1eaf3f1a7f270db51656f31af96d34f6235b35b9a11242766e2d62a493e110a9f5?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=3.jpg&amp;size=750">

图3  5000个time steps上的系统演进，其中在report interval为500。结果是在30次运行上的平均，共享区域表示标准差。在degeneracy speed上，Optimal Oracle > Oracle > TS > UCB > Random

图3展示了5种模型的degeneracy speed，。。。。我们可以看到，Optimal Oracle会产生最快的degnereation，接着是：Oracle, TS，UCB。Random Model最慢。

**Candidate Pool Size**

在图4a中，我们比较了Optimal Oracle、UCB、TS的degenreacy speed $$ \|\| \mu_t - \mu_0 \|\|_2 /t $$，直到5000 time steps，candidate pool sizes $$m=10, 10^2, 10^3, 10^4$$。除了random model，在给定一个大的candidate pool下，我们可以看到UCB会减慢degeneracy最多，因为它会首先强制探索任意unserved item。对于bandit算法，一个越大的candidate pool对于exploration需要一个更长时间。在给定time horizon下，由于candidate pool size增长到10000 UCB的dengeracy speed，从未达到峰值，但事实上会在给定一个更长时间上增长。TS具有越高的degereracy speed，因为在new items上具有更弱的exploration。Optimal Oracle 在给定一个更大pool上加速degeneration，因为比起一个更小的pool，它会潜在选择具有更快degenerative的items。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/d186d3c5f3d86442bc7b3c11a5b691a5b611f6af76e34512ebe346a4005ce29f1573575ee7b63f3645a0697b7c1925b1?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=4.jpg&amp;size=750" width="600">

图4

另外，在图4b中，我们画出了所有5个模型的degeneracy speed，对于T=20000, 对比candidate pool sizes的相同变化。Optimal Oracle 和Oracle的degeneracy speed会随着candidate set的size增长。实际上，具有一个大的candidate pool可以是一个临时解法(temporary solution)来降低系统degeneration。

**Noise Level的效应**

**接着我们展示了internal model inaccuracy在degeneracy speed上的影响**。我们对比使用不同均匀随机噪声量的Oracle model，例如：系统根据noisy internal model $$\theta_t^{'} = \theta_t + U([-\epsilon, \epsilon])$$来对外服务top l items。candidate pool具有fixed size $$m=100$$。在图5中，我们从0到10来区分$$\epsilon$$。**与直觉相反的是（ Counter-intuitively），添加噪声到Oracle中会加速degeneration**，因为对比起由$$\mu_0$$排序的top l items的fixed set，添加噪声会使得具有更快degenerative的items会被偶然选中，更可能满足Surfacing Assumption。给定$$\epsilon > 0$$，我们可以看到，随着noise level的增长，在degeneracy speed上具有一个单调递增的阻尼效应（damping effect）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/32452133dfdcb1a430fc58bfd712b4eb86fabee8579283bf8ad06346406687b52a3772c4c0c6a90531fa5e65d5928356?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=5.jpg&amp;size=750" width="300">

图5 在Oracle model上具有不同noise levels $$\epsilon \in [0, 10]$$的degeneracy speed，直到T=20000. 在Oracle中添加noise会加速degeneration，但随着noise level的增长，degneracy会缓下来

**Growing Candidate Pool**

我们通过计算$$sup_{a \in M} \mid \mu_t(a) - \mu_0(a)\mid /t$$，将degeneracy speed的定义扩展到一个有限的candidate pool上。由于degeneracy speed对于所有5种模型不是渐近线性的（asymptotically linear），我们会在10000个time steps上直接检查sup distance $$sup_{a \in M} \mid \mu_t(a) - \mu_0(a) \mid$$。为了构建在不同growth speed上的growing candidate pools，我们定义了一个增长函数$$m_t = \lfloor m_0 + l t^{\eta}\rfloor$$，通过不同增长参数$$\eta = 0, 0.5, 1, m_0 = 100$$定义。在图6中，我们在10个独立运行上对结果进行平均。对于所有growth rates，Oracle和Optimal Oracle都是degenerate的。Random Model会在sublinear growth $$\eta=0.5$$上停止degeneration，UCB也如此，这归因于之前unserved items上进行强制exploration，尽管它的trajectory具有一个小的上翘（upward tilt）。TS模型会在sublinear growth上degenerates，但在linear growth $$\eta=1$$上停止degernation。对于所有模型，growth rate $$\eta$$越高，他们degerate会越慢，如果他们完全这样做的话。当全部可用时，**线性growing candidate set和continuous random exploration看起来是一个不错的方法**，来对抗$$\mu_t$$的dynamics来阻止degeneracy。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/fc01a847d9320bf6fae01f7ea89c760362115fc35ab2568524c5a2586b0363f9837bff7279b48bef934099fab31d9c04?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=6.jpg&amp;size=750">

图6 5个模型的比较，它们使用growing candidate pools，具有不同的rates $$\eta = 0, 0.5, 1.0$$，degeneracy直到T=10000, 在10个运行结果上平均得到。对于所有growth rates，Oracle和Optimal Oracle都是degenerate的。Random Model和UCB会在sublinear growth上停止generation，而TS model需要linear growth才会停止degeneration。

# 参考

- 1.[https://arxiv.org/pdf/1910.05755.pdf](https://arxiv.org/pdf/1910.05755.pdf)