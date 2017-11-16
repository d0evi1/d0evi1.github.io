---
layout: post
title: TrueSkill排名系统
description: 
modified: 2017-08-28
tags: [trueskill]
---

前一阵子的AlphaGo和围棋很火，当时的AlphaGo在战胜kejie后排名世界第一；最近王者荣耀很火，它的排位赛机制中的内部匹配系统也十分令人诟病。不管是在围棋赛场，还是在多人竞技电子赛场上，排位系统是很重要的。常见的算法有：Elo，TrueSkill™。

Elo在上一篇已经介绍，再来看下TrueSkill算法。更详细情况可见MS的paper。

TrueSkill排名系统是一个为MS的Xbox Live平台开发的**基于实力（skill）的排名系统**。排名系统的目的是，标识和跟踪玩家在一场比赛中的实力，以便能将他们匹配到有竞争的比赛上（王者中有“质量局”一说，估计是这个意思）。TrueSkill排名系统只使用在一场比赛中所有队伍的最终战绩，来更新在游戏中的所有玩家的实力估计（排名）。

# 1.介绍

在竞技游戏和体育中，实力排名（Skill rating）主要有三个功能：首先，它可以让玩家能匹配到实力相当的玩家，产生更有趣、更公平的比赛。第二，排名向玩家和公众公开，以刺激关注度和竞技性。第三，排名可以被用于比赛资格。随着在线游戏的到来，对排名系统的关注度极大地提高，因为上百万玩家每天的在线体验的质量十分重要，危如累卵。

在1959年，Arpad Elo为国际象棋开发了一个基于统计学的排名系统，在1970年FIDE采纳了该排名。**Elo系统背后的核心思想是，将可能的比赛结果建模成关于两个玩家的实力排名s1和s2的函数**。在游戏中，每个玩家i的表现分\$ p_i \sim N(p_i; s_i, \beta^{2}) \$ ，符合正态分布，其中实力(skill)为\$s_i\$，相应的方差为\$\beta^{2}\$。玩家1的获胜概率，由他的表现分\$p_1\$超过对手的表现分\$p_2\$的概率来给定：

$$
P(p_1 > p_2 | s_1, s_2) = \Phi(\frac{s_1-s_2}{\sqrt{2}*\beta})
$$

...(1)

其中\$ \Phi\$表示零均值单位方差的高斯分布的累积密度（查表法取值）。在游戏结束后，实力排名s1和s2会更新，以至于观察到的游戏结果变得更可能，并保持s1+s2=const常数（一人得分，另一人失分）。假如如果玩家1获胜则y=+1; 如果玩家2获胜则y=-1; 如果平局则y=0. 接着，生成的Elo（线性增长）会更新为：\$ s1 \leftarrow s1 + y\Delta, s2 \leftarrow s2 - y \Delta \$, 其中：

$$
\Delta = \alpha \beta \sqrt{\pi} (\frac{y+1}{2} - \Phi(\frac{s1-s2}{\sqrt{2} \beta}))
$$ 

其中，\$\alpha \beta \sqrt{\pi}\$表示K因子， \$ 0 < \alpha < 1\$决定着新事实vs.老估计的权重。大多数当前使用Elo的变种都使用logistic分布(而非高斯分布)，因为它对棋类数据提供了更好的拟合。从统计学的观点看，Elo系统解决了成对竞争数据（paired comparison data）的估计问题，高斯方差对应于Thurstone Case V模型，而logistic方差对应于Brad ley-Terry模型。

在Elo系统中，一个玩家少于固定场次的比赛数（比如20场），那么他的排名将被看作是临时的（provisional）。该问题由Mark Glickman的Bayesian排名系统Glicko提出，该系统引入了将一个选手的实力建模成**高斯置值分布（Gaussian belief distribution：均值为\$ \mu \$， 方差为\$\sigma^2\$）**的思想。

实力排名系统的一个重要新应用是多人在线游戏（multiplayer online games），有利于创建如下的在线游戏体验：**参与的玩家实力相当，令人享受，公平，刺激**。多人在线游戏提供了以下的挑战：

- 1.游戏结果通常涉及到玩家的队伍，而个人玩家的实力排名对将来后续的比赛安排（matchmaking）也是需要的。
- 2.当超过两个玩家或队伍竞赛时，那么比赛结果是关于队伍或玩家的排列组合（permutation），而非仅仅是决出胜者和负者。

[paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2007/01/NIPS2006_0688.pdf)中介绍了一种新的排名系统：TrueSkill，它可以在一个principled Bayesian框架下解决这些挑战。我们将该模型表述成一个**因子图**（factor graph，第2节介绍），使用**近似消息传递**（第3节介绍）来推断每个选手实力的临界置信分布（marginal belief distribution）。在第4节会在由Bungie Studios生成的真实数据（Xbox Halo 2的beta测试期间）上进行实验。

# 2.排名因子图（Factor Graphs）

在一个游戏中，总体有**n个选手 {1, ..., n}**，在同一场比赛中有**k只队伍**参与竞技。队伍分配（team assignments）通过k个非重合的关于玩家总体的子集 \$ A_j \in \lbrace 1, ..., n \rbrace \$，如果 \$ i \neq j\$， \$A_i \bigcap A_j = \emptyset \$。比赛结果 \$ r := (r_1, ..., r_k) \in \lbrace 1, ..., k \rbrace \$，**每个队伍j都会有一个排名\$r_j\$**，其中r=1表示获胜，而\$r_i=r_j\$表示平局的概率。排名从游戏的得分规则中生成。

给定实际**玩家的实力s**，以及**队伍的分配**\$A := \lbrace A_1, ..., A_k \rbrace\$，我们对**游戏结果r**建模其可能概率：\$P(r\|s, A)\$。从贝叶斯规则（Bayes' rule）可知，我们获得其先验分布为：

$$
p(s|r,A) = \frac{P(r|s,A) p(s) }{p(r|A)}
$$

...(2)

我们假设一个因子分解的高斯先验分布为：\$p(s) := \prod_{i=1}^{n} N(s_i; \mu_i, \sigma^2)\$。每个**玩家i在游戏中的表现**为: \$ p_i \sim N(p_i; s_i, \beta^2)\$，以实力\$ s_i \$为中心，具有方差为\$\beta^2\$。**队伍j的表现**\$t_j\$被建模成其队员的表现分的总和：\$ t_j := \sum_{i \in A_j} p_i \$。我们以排名的降序对队伍进行重排序：\$r_{(1)} \leq r_{(2)} \leq ... \leq r_{(k)}\$。忽略平局，**比赛结果r的概率**被建模成：

$$
P(r| \lbrace t1, ..., t_k \rbrace) = P(t_{r_{(1)}} > t_{r_{(2)}} >... > t_{r_{(k)}})
$$

也就是说，表现的顺序决定了比赛结果的顺序。如果允许平局，获胜结果\$r_{(j)} < r_{(j+1)} \$需要满足 \$r_{(j)} < r_{(j+1)} + \epsilon \$，那么平局结果 $r_{(j)} = r_{(j+1)} \$ 需要满足 \$ \|r_{(j)} - r_{(j+1)} \| \leq \epsilon \$，其中\$\epsilon > 0\$是平局临界值，可以从平局的假设概率中计算而来。（见paper注1）

**注意："1与2打平"的传递关系，不能通过关系 \$ \| t_1 - t_2\| \leq \epsilon\$进行准确建模，它是不能传递的。如果\$ \| t_1 - t_2\| \leq \epsilon\$ 和 \$ \| t_2 - t_3\| \leq \epsilon\$，那么该模型会生成一个三个队伍的平局，尽管概率满足\$ \| t_1 - t_3\| \leq \epsilon\$**

在每场游戏后，我们需要能报告实力的评估，因而使用在线学习的scheme涉及到：高斯密度过滤（Gaussian density filtering）。后验分布（posterior）近似是高斯分布，被用于下场比赛的先验分布（prior）。如果实力与期望差很多，可以引入高斯动态因子 \$ N(s_{i,t+1}; s_{i,t}, \gamma^2) \$，它会在后续的先验上产生一个额外的方差成分\$\gamma^2\$。

例如：一个游戏，k=3只队伍，队伍分配为 \$ A_1 = \lbrace 1 \rbrace \$， \$ A_2=\lbrace 2,3 \rbrace \$，\$ A_3 = \lbrace 4 \rbrace \$。进一步假设队伍1是获胜者，而队伍2和队伍3平局，例如：\$ r := (1, 2, 2) \$。我们将产生的联合分布表示为：\$ p(s,p,t \|r,A)\$，因子图如图1所示。

<img src="http://pic.yupoo.com/wangdren23/GTzfRtn1/medish.jpg">

图1: 一个TrueSkill因子图示例。有4种类型的变量：\$s_i\$表示所有选手的实力（skills），\$p_i\$表示所有玩家的表现（player performances），\$ t_i \$表示所有队伍的表现（team performances），\$ d_j \$表示队伍的表现差（team performance differences）。第一行因子对（乘:product）先验进行编码；剩下的因子的乘积表示游戏结果Team 1 > Team 2 = Team 3的似然。**箭头表示最优的消息传递schedule**：首先，所有的轻箭头消息自顶向底进行更新。接着，在队伍表现（差：difference）节点的schedule按数的顺序进行迭代。最终，通过自底向顶更新所有平局箭头消息来计算实力的后验。

**因子图是一个二分图（bi-partite graph），由变量和因子节点组成，如图 1所示，对应于灰色圆圈和黑色方块**。该函数由一个因子图表示————在我们的示例中，联合分布 \$ p(s,p,t \|r,A) \$ ————由所有（潜在）函数的乘积组成，与每个因子相关。因子图的结构给定了因子间的依赖关系，这是有效推断算法的基础。回到贝叶斯规则(Bayes' Rule)上，给定比赛结果r和队伍关系A，最关心的是关于实力的后验分布\$p(s_i \| r,A)\$。\$p(s_i \| r, A)\$从联合分布中（它集成了个人的表现{pi}以及队伍表现{ti}）进行计算。

$$
p(s | r, A) = \int_{-\infty}^{\infty}...\int_{-\infty}^{\infty}dp dt.
$$

<img src="http://pic.yupoo.com/wangdren23/GTzg8fWq/medish.jpg">

图2: 对于平局临界值\$\epsilon\$的不同值的近似临界值的更新规则。对于一个只有两只队伍参加的比赛，参数t表示胜负队伍表现的差值。在胜者列（左），t为负值表示一个意料之外的结果会导致一个较大的更新。在平局列（右），任何队伍表现的完全误差都是令人意外，会导致一个较大的更新。

# 3.近似消息传递(Approximate Message Passing)

在因子图公式中的和积算法（sum-product algorithm）会利用（exploits）图的稀疏连接结构，来通过消息传递（messgage passing）对单变量临界值（single-variable marginals）执行有效推断（ecient inference）。连续变量的消息传递通过下面的方程表示（直接符合分布率）：

$$
p(v_k) = \prod_{f \in F_{v_k}} m_{f \rightarrow v_k}(v_k)
$$

...(3)

$$
m_{f \rightarrow v_j}(v_j) = \int ... \int f(v) \prod_{i \neq j} m_{v_i \rightarrow f}(v_i) dv_{ \backslash j}
$$

...(4)

$$
m_{v_k \rightarrow f}(v_k) = \prod _{\hat{f} \in F_{v_k} \backslash {f}}  m_{\hat{f} \rightarrow v_k} (v_k)
$$

...(5)

其中\$F_{v_k}\$表示连接到变量\$v_k\$的因子集，而 \$ v_{\backslash j} \$则表示向量v除第j个元素外的其它成分。如果因子图是无环的（acyclic），那么消息可以被精确计算和表示，接着每个消息必须被计算一次，临界值 \$ p(v_k) \$可以借助等式(3)的消息进行计算。

从图1可以看到，TrueSkill因子图实际上是无环的，消息的主要部分可以被表示成1维的高斯分布。然而，等式(4)可以看到，从比较因子（\$I(\cdot > \epsilon) \$）a或（\$ I(\cdot \leq \epsilon)\$）到表现差\$d_i\$去的消息2和5并不是高斯分布的——实际上，真实的消息必须是（非高斯分布）因子本身。

根据期望传播算法（EP： Expectation Propagation），我们将这些消息作近似，通过将临界值\$ p(d_i)\$通过变化的矩匹配（moment matching）产生一个高斯分布\$ \hat{p}(d_i) \$，它与\$ p(d_i) \$具有相同的均值和方差。对于高斯分布，矩匹配会最小化KL散度。接着，我们利用(3)和(5)得到：

$$
\hat{p}(d_i) = \hat{m}_{f \rightarrow d_i}(d_i) \cdot m_{d_i \rightarrow f}(d_i) \Leftrightarrow  \hat{m}_{f \rightarrow d_i}(d_i) = \frac{\hat{p}(d_i)}{m_{d_i \rightarrow f}}(d_i)
$$

...(6)

表1给出了所有的更新方程，这些等式对于在TrueSkill因子图中的推断是必要的。top 4的行由标准的高斯积分产生。底部的规则是由上述的矩匹配（moment matching）产生。第4个函数是对一个（双倍）截断式高斯分布的均值和方差的加乘校正项:

$$
V_{I(\cdot > \epsilon)}(t, \epsilon) := \frac{N(t-\epsilon)}{\Phi(t-\epsilon)}, W_{I(\cdot > \epsilon)}(t, \epsilon) := V_{I(\cdot > \epsilon)}(t, \epsilon) \cdot (V_{I(\cdot > \epsilon)}(t, \epsilon) + t - \epsilon)
$$

$$
V_{I(\cdot > \epsilon)}(t, \epsilon) := \frac{N(-\epsilon - t) - N(\epsilon -t)}{\Phi(\epsilon -t) - \Phi(-\epsilon -t) }
$$

$$
V_{I(\cdot > \epsilon)}(t, \epsilon) := V_{I(\cdot > \epsilon)}^2(t, \epsilon) + \frac{(\epsilon -t) \cdot N(\epsilon -t) + ( \epsilon + t) N(\epsilon +t )}{\Phi(\epsilon -t) - \Phi(-\epsilon -t)}
$$

<img src="http://pic.yupoo.com/wangdren23/GTzgutCu/medish.jpg">

表1: 对于缓存的临界值p(x)的更新方程、以及对于一个TrueSkill因子图中所有因子类型的消息\$m_{f \rightarrow x}\$。我们根据标准参数来表示高斯分布 \$ N(\cdot; \mu, \sigma) \$：准确率（precision） \$ \pi := \delta^{-2} \$，准确率调和平均（precision adjusted mean）\$ \tau := \pi \mu \$。以及关于该消息或从(6)获得的临界值的缺失的更新方程

由于消息2和消息5是近似的，我们需要对所有消息在任意两个近似临界\$\hat{p}(d_i)\$的最短路径上进行迭代，直到该近似临界值几乎不再改变。产生的最优化的消息传递schedule可以在图1中发现。（箭头和大写）

# 4.试验和在线服务

## 4.1 Halo 2 Beta Test

为了评估TrueSkill算法的表现，我们在Bungie Studios的游戏Xbox Halo 2的beta测试阶段生成的游戏结果数据集上进行试验。数据集包含了成千上万的游戏结果，包含4种不同的游戏类型：8个玩家自由对抗（自由模型），4v4(小队模式）， 1v1, 8v8(大队模式)。每个因子节点的平局临界\$\epsilon\$通过统计队伍间平局的比例（“期望平局概率”）进行设置，将平局临界\$\epsilon\$与平局的概率相关联：

$$
draw-probability = \Phi(\frac{\epsilon}{\sqrt{n_1+n+2}}\beta) - \Phi(\frac{-\epsilon}{\sqrt{n_1+n_2}\beta}) = 2 \Phi(\frac{\epsilon}{\sqrt{n_1+n_2}\beta}) - 1
$$

其中n1和n2分别是两只队伍的玩家数目，可以与图1中的节点\$ I(\cdot > \ epsilon)\$或 \$ I(\|\cdot\| \leq \epsilon)\$相比较。表现的方差\$ \beta^2 \$和动态的方差 \$ \gamma^2 \$被设置成标准值（见下一节）。我们使用一个高斯表现分布(1)和 \$ \alpha=0.07\$在TrueSkill算法上与Elo作对比；这对应于Elo中的K因子等于24, 该值被认为是一个较好且稳定的动态值。当我们必须处理一个队伍的比赛时，或者超过两个队伍的比赛时，我们使用“决斗（duelling）”技巧：对于每个玩家，计算\$ \Delta \$，对比其它所有玩家，基于玩家的队伍结果、每个其它玩家的队伍结果、并执行一个\$ \Delta \$平均的更新。在最后一节描述的近似消息传递算法相当有效；在所有我们的实验中，排序算法的运行时在简单的Elo更新运行时的两倍以内。

**预测表现(Predictive Performance)** 下表表述了两种算法（列2和列3）的预测error（队伍在游戏之前以错误顺序被预测的比例）。该衡量很难被解释，因为排名（ranking）和比赛安排（matchmarking）会相互影响：依赖于所有玩家的（未知的）真实实力，最小的预测error可以达到最大50%。为了补偿该隐式的、未知的变量，我们在ELO和TrueSkill间安排了一个对比：让每个系统预测尽可能匹配的比赛，将它们应用在其它算法上。该算法会预测更多正确的比赛结果，能更好地匹配。对于TrueSkill，我们使用比赛安排标准（matchmaking criterion），对于Elo，我们使用在Elo得分中的差：\$s_1 - s_2\$。

<img src="http://pic.yupoo.com/wangdren23/GTzgPdZg/medish.jpg">

**匹配质量**

排名系统的一个重要应用是，能够尽可能匹配到相似实力的玩家。为了比较Elo和TrueSkill在该任务上的能力，我们对比赛基于匹配质量（match quality）作划分，将两个系统应用到每场比赛上。如果匹配很好，那么很可能观察到平局。因而，我们可以画出平局的比例（所有可能的平局）在每个系统分配的匹配质量顺序上进行累积。在该图中，右侧可知，对于“自由模式（Free of All）”和1v1模式(Head to Head），TrueSkill比Elo更好。而在“4v4模式（Small Teams）”Elo比TrueSkill更好。这可能是因为额外的队伍表现模型（在该模式下大部分比赛是夺旗赛模式（Capture-the-Flag））的影响。

<img src="http://pic.yupoo.com/wangdren23/GTzhwWrz/medish.jpg">

**胜率（Win Probability）**

一个排名系统的感观质量，可以通过获胜比例来衡量：如果获胜比例高，那么该玩家在该排名系统中分配太弱的对手是错误的（反之亦然）。在第二个试验中，我们处理了Halo 2数据集，但舍弃了那些没有达到一定匹配质量阈值的比赛。对于被选中的比赛，取决于每个玩家所玩的最低数目的比赛数，我们计算了每个玩家的获胜概率，来测量**获胜概率与50%（最优获胜比例）的平均误差**（越小越好）。产生的结果如图所示（在1v1模式下），它展示了TrueSkill模式下，每个参加了比较少比赛的玩家，会获得更公平的匹配（胜率在35%-64%）。

<img src="http://pic.yupoo.com/wangdren23/GTzhx7A3/medish.jpg">

**收敛性能（Convergence Properties）**

最后，我们画出了两个典型的、在自由模式下（Free for All）两个最高排名的玩家的收敛轨迹：（实线：TrueSkill；虚线：Elo）。如上所见，TrueSkill会自动选择正确的learning rate，而Elo只会对目标实力做较慢的收敛。实际上，TrueSkill与信息论极限（information theoretic limit ）更接近： nlog(n)位来编码n个玩家的排名。对于8个玩家的比赛，信息论极限是 \$ log(n) / log(8) \approx 5\$，每个玩家平均5场比赛，而这两位观察到的玩家的收敛约等于10场比赛！

<img src="http://pic.yupoo.com/wangdren23/GTzhxdqw/medish.jpg">

## 4.2 Xbox 360 Live上的TrueSkill

微软的XBox Live主要是在线游戏服务。世界各地的玩家一起玩，他们具有不同的成百上千个头衔。在2005.9, XBox Live具有超过200w的订阅用户，在该服务上花费13亿个小时。新的改进版Xbox 360 Live服务使用TrueSkill算法来提供自动玩家排名（automatic player rating）和比赛安排（matchmaking）。该系统会每天处理成千上万的游戏比赛，使它成为贝叶斯推断（Bayesian inference）的最大应用。

在Xbox Live上，我们使用了一个数值范围，由先验\$ \mu_0=25\$和\$ \delta_0^2=(25/3)^2\$给定，对应于接近99%的正向实力概率。表现的方差由\$\beta^2 = (\sigma_0/2)^2\$给定，动态方差则选择\$ \gamma^2 = (\sigma_0 / 100)^2 \$。一个玩家i的TrueSkill实力，目前被表现为一个保守的实力估计，由低于1%的分位数\$ \mu_i-3\delta_i\$给定。该选择确保了排行榜（一个根据\$\mu-3\delta得到的所有玩家列表\$）的top榜单只可能被那些具有较高实力、更高确定度（certainty）的玩家占据，它们从\$ 0=\mu_0 - 3 \delta_0\$开始逐步建立。对玩家的成对比赛安排（Pairwise matchmaking），使用从相对最高可能的平均概率的平局概率派生而来的匹配质量准则来执行，取极限\$ \epsilon \rightarrow 0 \$，

$$
q_{draw}(\beta^2, \mu_i, \mu_j, \sigma_i, \sigma_j) := \sqrt{\frac{2\beta^2}{2\beta^2 + \sigma_i^2 + \sigma_j^2}} \cdot exp(-\frac{(\mu_i - \mu_j)^2}{2 (2\beta^2 + \sigma_i^2 + \sigma_j^2)})
$$

...(7)

注意，比赛安排过程（matchmaking process）可以被看成是一个逐次实验设计（sequential experimental design）的过程。因为一个匹配质量由它的结果的不可预知性（unpredictability）决定，比赛安排和发现最有益匹配的目标是为了均衡（align）。

另一个吸引人的副产品是，我们有机会在成千上万的玩家上学习TrueSkill的运转。而我们只开始分析大量的结果数据，已经有一些有趣的观察结果。

- 1.比赛随有效实力等级的数目的不同而不同。机遇型游戏（Games of chance）（例如：单场双陆棋比赛或UNO）具有更窄的实力分布，而凭实力取胜的游戏（Games of skill）（例如：半实况的竞速比赛）具有更宽的实力分布。
- 2.比赛安排（Matchmaking）和实力展示（skill display）会对玩家产生一个反馈循环，玩家经常会看它们的实力估计作为表现的奖惩。一些玩家试图通过：不玩、小小选择对手、或者作弊来保护或提升它们的实力排名。
- 3.如果是新玩家，通常会在最初的几场比赛时失利，总的实力分布会偏移到先验分布之下。而当实力会初始化重置后，我们发现更准的比赛安排的效果会消失。

# 5.结论

TrueSkill一是个全局部署Bayesian的实力排名系统，它基于在因子图的近似消息传递。比起Elo系统，它具有许多理论和实际的优点，并在实践中运行良好。

而我们主要关注TrueSkill算法，在因子图框架内可以开发许多更多有趣的模型。特别的，因子图公式可以被应用到受限制的分类模型族（the family of constraint classication models），它包含了更宽范围的多分类和排名算法。另外，作为对个人实体进行排名的替代，你可以使用特征向量来构建一个排名函数，例如，对于网页可以表述成bags-of-words。最终，我们计算运行一个关于棋类游戏的完全时间独立的EP分析，来对获得关于象棋大师的TrueSkill排名。

# 6.实现

trueskill的一个python实现：[http://trueskill.org/](http://trueskill.org/)。

另外，MS还提供了一个在线模拟器，这个可以结合着去理解：[http://boson.research.microsoft.com/trueskill/rankcalculator.aspx](http://boson.research.microsoft.com/trueskill/rankcalculator.aspx)

关于TrueSkill的数学基础，详见：[http://www.moserware.com/assets/computing-your-skill/The%20Math%20Behind%20TrueSkill.pdf](http://www.moserware.com/assets/computing-your-skill/The%20Math%20Behind%20TrueSkill.pdf)

# 参考

- 0.[microsoft TrueSkill paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2007/01/NIPS2006_0688.pdf)
