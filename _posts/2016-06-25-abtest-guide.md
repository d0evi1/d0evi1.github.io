---
layout: post
title: A/B testing 5个问题
description: 
modified: 2016-06-30
tags: 
---

Microsoft在KDD2007时的paper《Practical Guide to Controlled Experiments on the Web:Listen to Your Customers not to the HiPPO》中，对A/B testing系统设计有一些practical guide，可以参考一下：

# 3.controlled experiments

在最简单的受控实验（controlled experiment）中，通常指的是一个A/B test，用户会被随机曝光给两个variants之一：control(A)、treatment(B)，如图2所示。

这里的关键点是：“random”。用户不能“随意（any old which way）”分布，没有因子可以影响该决策。基于收集到的observations，会为每个variant生成一个OEC。

例如，在Checkout示例(2.1节)中，OEC可以是转化率、购买单元数、收益、利润、期望生命周期值、或者它们之间的加权组合。

如果experiment的设计和执行是合理的，在两个variants间唯一一直不同的是：Control和Teatment间的变化，因此，任意在OEC上的差异必然会产生该assignment，这存在因果关系。

## 3.1 术语

- OEC
- Factor
- Variant
- Experimentation Unit
- Null Hypothesis
- Confidence level
- Power
- A/A Test
- Standard Deviation (Std-Dev)
- Standard Error (Std-Err)

## 3.2 Hypothesis Testing和Sample Size

为了评估一个treatment组与Control组是否不同，需要做一个统计试验（statistical test）。如果该test拒绝(reject)零假设（OEC是并没有差异），我们认为：一个Treatment是统计显著(statistically significantly)有差异的。

我们不会回顾statistical test的细节，因为他们在许多统计书籍中有描述。

我们要回顾下影响试验(test)的一些重要的因子：

- 置信等级（confidence level）。通常设置为95%，该level表示：5%的机会是不正确的
- 统计功效（Power）。通常在80-95%间，尽管不是直接控制的。如果零假设是false的（比如：在OEC上有差异）, power表示决定该差异的概率，是统计显著的。（Type II error）
- (Standard Error)。Std-Err越小，test越强。有三种方法减小std-err：
	- 被估计的OEC通常是大样本的一个平均。如3.1所示，一个均值(mean)的Std-Err会随着sample size的均方根按比例递减，因此，增大sample size（这通常意味着运行实验更久），可以减小Std-Err，从而增大power
	- 使用天然具有更小变动性的OEC components（例如：Std-Dev, $$\delta$$），std-err会越小. 例如，转化率（conversion probability 0-100%）通常比购买单元数目（通常是小整数）具有更低的Std-Dev，换句话说，它比收益（revenue: real-valued）具有一个更低的std-dev。
	- 通过过滤掉那些没有曝光给variants的用户，OEC的可变性会越低. 例如，如果你对checkout page做出一个变更，只分析访问该page的用户，由于每个人都增加了噪声(noise)，会增加可变性（variability）
- 4.**the effect**：对于variants来说在OECs上的差异。差异越大越容易检测，因此，great ideas不可能错过。相反的，如果Type I或Type II errors发生，当该effects很小时，他们更可能发生。

以下的公式近似期望的sample size，假设期望的置信level为95%，期望的power是90%(31)：

$$
n = (4r\delta / \Delta)^2
$$

其中：

- n是sample size
- r是variants数目
- $$\delta$$是OEC的std-dev
- $$\Delta$$是OECs之间的最小差异
- 4是因子，表示对于大的n会估计过高25%，但该近似会满足下面示例。

假设有一个电商网站，在实验周期期间，访问该网站的5%用户会以购买行为结束。这样的购买会花费大约75美金。用户的平均花费为3.75美金（95%的人没有消费）。假设：标准差（standard deviation）为30美金。如果你正运行一个A/B test，该test希望检测在收益上有一个5%左右的变化，基于上面的公式：$$4 \cdot 2 \cdot 30 / (3.75 \cdot 0.05))^2$$，你将需要超过1600w用户来达到期望的90% power。

然而，如果只只希望在转化率（conversion rate）上检测一个5%的变化，基于3.b会使用一个可变性更低的OEC。购买（一个常见事件）可以被建模成一个p=0.05(购买概率）的伯努利实验（Bernoulli trial）。一个Bernoulli的Std-Err为$$\sqrt{p(1-p)}$$，因此，基于公式$$(4 \cdot 2 \cdot \sqrt{0.05 \cdot (1-0.05)}/(0.05 \cdot 0.05))^2$$，你需要至少50w用户来达到所期望的power。

由于平方因子的存在，如果目标放宽些，以便你希望在转化上（因子为4）检测一个20%变化，通过乘以一个16的因子来将用户数目下降到30400。

如果对checkout过程做出一个变化，你只需要分析那些启动checkout过程的用户，因为其它用户不能看到任何差异，只会增加噪声。假设10%的用户从checkout开始，并且其中有50%用户完成了它。该用户分片（user segment）是更同质的，因此OEC具有更低的可变性。正如之前使用相同的数子，平均转化率是0.5, std-dev是0.5, 因而基于$$(4 \cdot 2 \cdot \sqrt{0.5 \cdot (1-0.5)} / (0.5 \cdot 0.05))^2$$，你需要25600个用户访问checkout来检测一个5%的变化。由于我们会将没有初始化的90%用户排除，访问该网站的总用户数应该是256000, 这几乎是之前结果的一半，因此实验可以运行一半时间，来生成相同的power。

当运行实验时，提前在OEC上做决定很重要；否则，会增加寻找有机会出现显著结果的风险（与type I error相似）。一些文献中也提出了一些方法，但他们基本上等同于增加95%的置信level，从而减小statistical power。

## 3.3 online settings的扩展

basic controlled experiments的一些扩展在online setting中是可能的。

## 3.3.1 Treatment Ramp-up

### 3.3.2 自动化(Automation)

### 3.3.3 软件迁移(software migrations)



## 3.4 限制

尽管根据因果关系提供的controlled experiments的优点显著，它们也有一些必须理解的限制。

- **1.量化指标，但没有任何解释**。我们可能知道哪个“variant”更好，以及好多少，但不知道“为什么（why）”。例如，在用户研究中，行为(behavior)通常会随用户评论(comments)而增加，因此可用性实验室（usability lab）可以用于增加和完善controlled experiments。
- **2.短期效应(short term) vs. 长期效应（long term effects）**。controlled experiments会在实验期间（通常数周）衡量OEC的effect。而一些paper作者则批评说：这样只会注某一指标意味着short-term focus，我们对此不赞同。long-term目标应该是OEC的一部分。假设我们以搜索广告为例。如果你的OEC是收益（revenue），你可能在一个页面上粘帖数个广告，但我们知道：越多广告会伤害用户体验，因此一个好的OEC应包含对于没有点击的房地产广告的一个惩项（penalty term），或／和 对重复访问和放弃的直接measure。同样的，查看一个延迟的转化指标是明智的，其中从一个用户被曝光某样东西到该用户做出动作这段时间内有个迟延(lag)。这有时被称为潜在转化（latent conversions）。找好好的OEC是很难的，但有什么替代方法吗？这里的关键是，认识到该限制。
- **3.首效应(Primacy effects)和新效应（newness effects）**。有一些负面效应需要认识。如果你在一个web网站上更换了导航，体验的用户可能更低效，直到他们完全适应了新导航，因而Control组会天然具有优势。相反的，如果一个新设计或新特性被引入，一些用户可能会研究它，到处进行点击，从而引入一个“newness” bias。该bias有时会与霍索恩效应（Hawthorne effect）相关。Primacy和Newness这两者意味着：一些实验需要运行多周。可以做个分析：为不同variants上的新用户计算OEC，因为他们不会受两个因素影响。
- **4.必须实现的特性**。一个真实的controlled experiment需要曝光一些用户给一个Treatment，不同于当前站点（Control组）。该prototype可能是个prototype，它可以在很小比例上进行测试，或者不会覆盖所有的边缘情况（edge cases）（例如：实验可能故意排除20%的需要显著测试的浏览器类型）。尽管如些，该特性必须被实现，并曝光给用户足够的量。Jacob(34)正确指出：paper prototyping可以被用于在早期定性反馈和快速改进设计。我们赞同并推荐这样的技术来完善controlled experiments。
- **5.一致性（consistency）**。用户可能注意到：比起它们的朋友和家人，他们获得了一个不同的variant。也有可能是一些相同的用户当使用不同电脑（具有不同cookies）看到多个variants。这相对比较罕见：用户会注意到该差异。
- **6.并行实验（Parallel Experiments）**。我们的实验实际上很少发生很强的交叉（33），我们认为这一点是过度关注了。提升对这一点的关注对于实验者来说足够避免交叉（interact）的测试。成对的统计测试（statistical test）也可以完成，来自动标记这样的交叉（interactions）。
- **7.启动事件和媒体公告**。如果对新特性做出一个大的公告，（比如：该特性通过媒体进行公告），所有用户需要看到它。

第4部分来自另一paper。

# 4.多因子实验(Multi-Variable Testing)

一个包含了超过一个因子（factor）的实验通常被称为：MultiVariable test(MVT)。例如，考虑在MSN主页上在单个实验中测试5个factors。MSN主页截屏如图8所示，为每个factors展示了control。

表二：

在单一测试中，我们可以估计每个factor的效果(main effects)，以及多个factors间的交叉效果（interactive effects）。首先，我们会考虑：MVT vs. one-factor-at-a-time (或者A/B tests)的好处和限制。接着我们会讨论对于online MVTs的三种方法，以前每种方法会利用潜在好处并缓和该限制。

对于测试相同factors的single MVT vs. 多个sequential A/B test，MVT主要有两个好处：

- 1.你可以在一个较短时间内测试许多factors，加速提升。例如，你想测试在网站上的5个变化，你需要每个A/B test运行4周时间来达到你需要的power，它会至少需要5个月来完成A/B tests。然而，你可以使用所有5个factors来运行单个MVT，它只需要1个月就可以达到5个A/B tests相同的power。
- 2.你可以估计factors间的交叉。两个factors的交叉，如果它们的组合效应与两个单独的effects叠加不同。如果这两个factors一起可以增强结果，那么该交叉（interaction）是协同的（synergistic）。如果相反的，他们相互抵触会抑制该效果，该交叉（interaction）是对抗的（antagonistic）。

三个常见限制是：

- 1.一些factors的组合可能会给出一个较差的用户体验。例如，对于一个在线零售商来说，正在测试的两个factors可能是：增加一个商品的图片 or 提供额外的商品详情。当单独进行测试时，两者分别均会提升销售，但当两者在同时完成时，“buy box”会被推到折叠（fold）下面，这会减小销售。这就是一个大的对抗交叉（antagonistic interaction）。该交叉在计划阶段（planning phase）可以被捕获，以便这两个factors不会被同时测试。
- 2.分析和解释是更困难的。对于单个factor的test，你通常具有许多指标来进行Treatment-Control的比较。对于一个MVT，你需要为多个Treatment-Control比较设置相同的matrics（对于每个测试的factor至少要有一个），以及在factors间交叉的分析和解释。确定的是，该信息集越丰富，对于treatments的评估也越复杂。
- 3.它会花费更长时间开始测试。如果你具有希望测试的5个factors，并计划以一次测一个的方式测试它们，你可以启动任意待测的，接着测试其它的。使用一个MVT，你必须在test开始之前，具备所有5个准备好的testing。如果任何一个延误了，会延误test的启动。

我们不认为：在大多数情况中任意的限制（limitations）是严重的限制，但在进行一个MVT之前应意识到它。通常，我们相信，第一个实验确实会是一个A/B test，主要是因为在相同的test中测试多个factor的复杂性。

有三种哲学来处理在线 MVTs.

## 4.1 传统MVT

该方法所用的设计被用于制造业（manufacturing）和其它离线应用。这些设计大多数通常是部分因子（fractional factorial）以及Plackett and Burman designs，它们是完全因子设计（所有factor levels上的组合）的一个特定子集。这些设计由Genichi Taguchi普及，有时被称为“田口设计（Taguchi designs）”。用户必须小心选择一个设计，它可能具有足够估计主效应和所关注的交叉效应。

对于我们的MSN示例，我们展示了一个包含了5个factors的test设计，它使用完全因子（full factorial），部分因子（fractional factorial） or Plackeet-Burman design。

表1.

完全因子会具有factors的所有组合：$$2^5=32$$个user groups。部分因子（factional factorial）则是完全因子的一个部分，它具有$$2^K$$个user groups，每一列与其它4列是正交的（orthogonal）。很明显，许多这样的factions具有8-16个user groups。K=3的部分因子（fractional factorial）如表1所示，-1表示control，1表示treatment。

如果factors是在2 levels上，Plackett–Burman designs可以被构建，user groups的数目可以是4的倍数，比如：4,8, 12, 16, 20等。对于任意这种designs来说，可以被测试的factors数目是user groups的数目减去一。如果user groups的数目是一个2的power，那么该Plackett–Burman design也是一个部分因子（fractional factorial）。

由于使用了部分因子（fractional factorial），对于一个给定数目的user groups，通常会使用许多Plackett–Burman designs。

在实验设计的统计领域，一个主要研究领域是：寻找这样的设计，它可以最小化用于test所需的user groups数目，同时允许你估计主效应和交叉效应，允许小量或没有混杂（confounding）。表1的fractional factorial 可以估计所有5种main effects，但不能很好估计interactions。对于许多实验来说，运行MVT的主要原因之一，估计要测试的factors间的交叉（interaction）。使用该设计，你不能很好地估计任何交叉，因为所有交叉都是与main effects或two-factor interactions相混淆的。即使在分析或数据挖掘上花再大精力，也不能让你单独估计这些interactions。如果你想估计5个factors中的所有two-factor interactions，你需要一个具有16个treatment combinations的 fractional factorial design。表2的Placket–Burman design具有所有two factor interactions，部分会与main effects和其它two-factor interactions相混杂(confound)。这使得two factor interactions的估计充满挑战。

对于online tests来说，对比传统的MVT方法，我们推荐另外两种更好的可选方法。你喜欢的一种将会依赖于：你对估计interactions的评价有多高。

## 4.2 并行运行tests的MVT

在offline testing中，可以使用完全因子（full factorial）的部分，因为使用更多treatment组合通常有个开销，即使experimental units的数目没有增加。这不一定是与网站相关的test。如果我们设置了每个factor来运行一个one-factor实验，并同时运行所有的tests，我们可以简化我们的工作，最终获得一个完全因子（full factorial）。在该模式中，我们同时在相同的用户集合上（用户会被独立随机分配给每个实验）开始和停止所有这些one-factor tests。最终结果是在所有要测试因子上你将具有一个完全因子实验。当然，在完全因子上，你可以估计任何你想要的交叉。该方法的一个边缘优点是，你可以在任何时间关闭任何factor（例如：如果对于某一factor的一个treatment是灾难性的）不会影响其它factors。包含剩余factors的实验不会被影响。

通常会认为：实验的power会随着treament组合(cells)的数目增加而减少。如果该分析是基于每个单独的cell与Control cell相对比得出，那么这个观点可能是true的。然而，如果分析是基于更传统的方式（为每个effect使用所有数据来计算main effects和interactions），power则不变或减小很少。如果sample size是固定的，它不会影响你正测试单个factor或多个factor。。。有两件事件可以减小你的power。一是对于某一factor增加levels（variants）的数目。这对于你想做出的比较来说，会有效减小sample size，不管test是一个MVT或一个A/B test。另一个是分配少于50%的test polulation到treatment（如果它们是two levels）。在一个MVT中，需要与Control具有相同比例的population，这对于treatment来说这特别重要。

## 4.3 Overlapping实验

该方法会简单测试一个actro作为一个one-factor实验，当该factor准备与每个独立随机化的test一起被测试时。它与4.2节不同，这里当treatment准备进行每个实验会开启，而非一次性加载所有factors。在一个敏捷软件开发世界中，频繁运行更多的tests有极大好处，因为他们准备进行部署。如果带这些组合的tests上没有明显的用户体验方面问题，这些tests可以同时同步进行，可以展示给任何用户。如果你想最大化你想测试的ideas的速度，并且你不关心交叉（interactions），你可以采用这种方法。factors间的大交叉实际上比大多数人相信的要少很多，除非是已知的，比如：buy box示例。比起之前提到的traditional方法，这是一种更好的替代方法。在传统方法中有这样的限制：必须所有factors都准备好进行测试时你才能进行test。另外，当你完成时，你不能很她地估计交叉。有了overlapping experiments，如果在任何两个factors有足够的overlap，你可以更快速的测试factors，另外你可以估计这些factors间的交叉。如果你特别关心两个特定factors间的交叉，你可以同时计划测试这些factors。

我们相信，这两种方法比传统的MVT方法更适合online实验。如果你想尽可能快地测试你的想法，并且不关心interactions，可以使用overlapping experiments方法。如果同时运行实验时估计交叉很重要，用户可以被独立随机化到每个test中，并给你一个完全因子实验。

# 5.实验架构

在一个网站上实现一个实验，涉及到两个部分。第一个component是：随机算法(randomization algorithm)，它是一个将users映射到variants的函数。第二个component是分配方法（assignment method），它使用随机算法的output来决定每个用户是否能看到该网站的实验。在实验期间，必须收集observations，数据需要被聚合和分析。

## 5.1 随机算法

寻找一个好的随机算法很重要，因为controlled experiments的统计，会假设：一个实验的每个variant都具有关于users的一个随机抽样。特别的，随机算法必须具有以下4个特性：

- 1.用户必须等可能地看到一个实验的每个variant（假设：50-50 split）。对于任意指定variant必须是无偏的(no bias)。
- 2.对于单个user重复分配（repeat assignments）必须是一致的（consistent）；在对该网站的每次后续访问上，该user必须被分配到相同的variant上。
- 3.当多个实验同时并行运行时，实验之间必须没有关联（correlation）。在一个实验中，一个user被分配（assignment）到一个variant，对于被分配给其它任意实验的一个variant，在概率上没有影响
- 4.该算法必须支持单调递增（monotonic ramp-up），这意味着，如果没有对那些已经被配给Teatment的用户分配进行变更，用户看到一个Treatment的百分比可能会缓慢增加。

## 5.4.1 使用caching的伪随机

当组合使用caching时，可以使用标准伪数字生成器(standard pseudorandom number generator)作为随机算法。一个好的伪数字生成器会满足关于随机算法的第一和第三个要求。

在第一和第三要求上，我们测试了一些流行的随机数字生成器。我们在100w序列化user IDs上测试了5个仿真实验。运行卡方检验（chi-square）来检索交叉（interactions）。我们发现，只要生成器的seed只在server启动时初始化一次，在许多流行语言（例如：C#）内置的随机数字生成器可以运转良好。在每个请求上对随机数字生成器进行Seeding会造成：相邻的请求会使用相同的seed，这会在实验(experiements)间引入显著相关性（noticeable correlations）。特别的，我们发现，Eric peterson建议使用VB的A/B test代码，会在实验间创建很强的two-way interactions。

为了满足第二个要求，该算法必须引入状态（state）：用户的分配必须被缓存，以便他们再次访问该网站。缓存的完成可以是服务端（通过数据库进行存储），也可以是客户端（例如：将user的assignment存到cookie中）。

两种形式都很难扩展到一个具有大量servers的大系统上。server做出随机分配，必须将它的state与其它servers（包括使用了后端算法的那些）进行通信，以便保持分配一致。

使用该方法时，第4个要求（单调递增）特别难实验。不管使用哪种方法来维持状态(state)，，会在一个ramp-up之后，该系统需要将Control users（访问该网站的这些用户）重新分配给一个treatment。我们还没有见过一个使用基于伪随机分配(pseduorandom-based assignment)的系统是支持ramp-up的。

### 5.1.2 Hash和分区

不同于伪随机方法，该方法的完成是无状态的。每个user会被分配一个唯一的id，它可以通过database或cookie来维持。该id会被附加(append)到该experment的name或id上。接着在该组合id上使用hash函数来获得一个integer，它在数值范围内是均匀分布(uniformly distributed)的。该range接着被进行划分，每个variant通过一个partition进行表示。

该方法对于hash函数的选择非常敏感。如果hash函数具有任意的漏斗（funnels）（那些相邻keys的实例会映射到相同的hash code上），那么这会与第一个特性（均匀分布）相冲突。如果hash函数具有characteristics（某个key的变动会产生一个在hash code上可预测的变动），那么实验之间会发生相关。在该技术上，只有很少的hash函数够资格使用。

我们使用许多流行的hash函数、以及一种类似于伪随机数字生成器的方来测试该技术。任意hash函数都会满足第二个要求，但满足第一和第三是很困难的。我们发现，只有加密hash函数MD5生成的数据在实验间没有相关性。SHA256(另一个加密hash）接近，需要一个five-way interaction才会产生一个相关。.NET的string hashing函数则会在一个2-way interaction test上会失败。

## 5.2 Assignment方法

Assignment方法是软件的一部分，它允许正实验的网站对于不同的用户执行不同的代码路径。有许多方法来实验一个assignment方法，优点和缺点各不同。

**流量分割（Traffic splitting）**

该方法会涉及到：一个experiement的每个variant在不同的servers的实验，分为物理（physical）和虚拟（virtual）。该网站会将随机算法嵌入到一个负载均衡器(load balancer)或代理服务器（proxy server）中以便对variants间的流量进行分割。流量不能对已存在代码进行变更来实现一个experiment。然而，对于跨所有正运行实验的variants的每个唯一组合来说，该方法需要为它进行一个并行fleet的setup和config，这使得该方法在处理assignment时代价很大。

另一种可选的方法是：服务器端选择（server-side selection），API调用会嵌入到网站的servers中来调用随机算法、并使得分枝逻辑对于每个variant处理一个不同的用户体验。Server-side selection是相当通用的方法，它在一个网站的任一部分可以支持多个实验，从可视化元素到后端算法。然而，从一个开发者实现代码变更到运行实验，这个过程需要额外工作。

最终的可选方法是：客户端选择（client-side selection），JavaScript调用会嵌入到每个web页上，会进行一个远端服务进行分配，并能动态修改页面来产生合适的用户体验。client-side实验通常要比server-side更容易实现，因为开发者只需要将javascript代码嵌入到每个页面上。然而，该方法会严重限制实验的特性；特别的，关于动态内容实验或者后端特性的实验更难实现。

# 6.学到的课程

理论和实际是不同的。

许多理论技术看起来很适合实际使用，但需要十分小心，以防止污染真实环境。Controlled experiments也不例外。关于运行大量在线实验，我们在三个方面的实际经验分享：

- (i) 分析
- (ii) trust & execution
- (iii) culture & business

略 


参考：

- [Practical Guide to Controlled Experiments on the Web:Listen to Your Customers not to the HiPPO](https://ai.stanford.edu/~ronnyk/2007GuideControlledExperiments.pdf)
- [http://www.robotics.stanford.edu/~ronnyk/2009controlledExperimentsOnTheWebSurvey.pdf](http://www.robotics.stanford.edu/~ronnyk/2009controlledExperimentsOnTheWebSurvey.pdf)