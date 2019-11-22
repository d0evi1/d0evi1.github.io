---
layout: post
title: A/B testing 5个问题
description: 
modified: 2016-06-30
tags: 
---

Microsoft在《Trustworthy Online Controlled Experiments:
Five Puzzling Outcomes Explained 》中对A/B testing中存在的5个谜题做了解释。

# 3.1 搜索引擎的OEC

## 3.1.1 背景

选择一个好的OEC对于整个商业经营很重要。这些指标驱动着go/no-go的决策。在之前工作中，我们强调了长期关注的必要，并建议将lifetime value作为一个指导原则。像日活用户（DAU）这样的指标被一些公司所使用。在《7 pitfalls to Avoid when Running controlled Experiements on the web》中，首个陷阱是：

从商业角度上看，应选择这样一个OEC：它可以通过做一些很明显“错误（wrong）“的事情就很轻易地击败control组实验

当我们尝试为Bing获得一个OEC时，我们会首先关注商业目标。当前有两个顶级的长期目标：查询分享（query share）以及每次搜索的回报（revenue per search）。确实，许多项目被激励来提升它们，但有一个很好的示例：短期目标和长期目标完全背离。

## 3.1.2 Puzzling Outcome

当Bing在实验中有个bug，导致了展示给用户非常差的结果，但两个关键的公司级指标提升显著：每个用户的不同查询数（distinct queries per user）涨了10%，每用户回报（revenue per user）提升了30%! Bing应该如何评估该实验？什么是OEC？

很明显，实验中的长期目标与短期指标是不能对齐的。如果它们可以对齐，我们可以降低质量来提升query share和revenue!

## 3.1.3 解释

从搜索引擎的角度，退化的算法结果（展示给用户的主搜索引擎结果，有时也被称为10个蓝链接(10 blue links)）会强制用户发起更多的查询（这会增加每个用户的queries）以及点击更多的广告（增加回报）。然而，这很明显是短期提升，这与零售商店里提升价格相似：你可以增加短期回报，但客户更偏向于与时间竞赛，因此平均顾客的生命周期价值（lifetime value）将会缩短。

为了理解该问题，我们分解了query share。我们将月查询分享数(monthly query share)定义为：Bing上不同查询数 / 一整个月所有搜索引擎的不同查询数，正如comScore所测量的（distinct意味着：同一用户在半小时内在相同的垂直搜索引擎（比如：网页or图片）上连续重复的queries，被统计成1）。由于在Bing上，我们可以很轻易地测量分子（我们的distinct queries，而非整个makret），该目标是为了增加该component。每个月的distinct queries可以被分解成三个项相乘：

$$
\frac{Users}{Month} \times \frac{Sessions}{User} \times {Distinct \ queries}{Session}
$$

...(1)

其中，在乘积中的第2项和第3项会在月周期上计算，session被定义为：用户开始发起一个query、30分钟后在搜索引擎上没有活动就结束。

如果一个搜索引擎的目标是，允许用户快速发现它们的答案、或者完成它们的任务，那么减小每个任务的distinct queries是一个明确的目标，它会与关于增加收入的商业目标相冲突。因为该指标与每个sesion不同查询数(distinct queries per session)高度相关（），我们推荐：distinct queries不能单独用来作为搜索实验的OEC。

给定如等式(1)所示的不同查询的分解，我们来看下这三个项：

- 1.User per month。在一个controlled experiment中，独立用户数由设计决定。例如，在一个相等的A/B test中，用户数会以近似相同的数目分到两个variants中。（如果在variants中用户的比例很不同，这很可能是个bug）。出于该原因，该项不能成为该实验OEC的一部分。
- 2.每任务查询数（distinct queries per task）应该最小化，但它很难测量。每session不同查询数（distinct queries per session）是一个可用的代理指标(surrogate metric)。这是一个精细指标（subtle metric），然而，由于增加它可能意味着用户需要发起理多查询来完成该任务，但减少它可能表示放弃查询。该指标需要最小化，意味着该任务会被成功完成
- 3.Sessions/user 是在实验中要优化(增加)的关键指标，因为满意的用户会来得更多。这是一个在Bing中OEC的关键成分（key component）。如果我们有好的方式来标识tasks，等式(1)中的分解可以通过task来进行，我们可以优化tasks/user。

在搜索引擎结果页上展示的退化算法结果，会给用户一个明显更差的搜索体验，但会造成用户点击更多广告（它的相对相关度会增加），从而增加短期回报。在没有其它约束下，每用户回报(Pevenue per user)同样不能被用来当成搜索和广告实验的一个OEC。当关注回报指标时，我们希望：在不会对用户满意度指标（比如：sessions/user）造成负面影响的情况下增加它们。

## 3.1.4 学到的

query量的分解，搜索的长期目标，展示了冲突的组成(components)：一些会增加短期指标（sessions/user），另一些会减小短期指标（queries/session）表示任务成功完成。我们的假设是，更好的用户体验会增加users/month，最后一项不能在一个control实验中进行测量。

该分析不仅影响搜索实验，也会影响SEM等（search engine marketing）。当决定广告的bid量时，很自然的会尝试和优化在session中的queries数，它会伴随广告点击。然而，长的sessions表示用户找不到满意的东西（例如：驱动用户下拉更多结果页）

顾客生命周期值（Lifetime customer value）通常应是决定OEC的指导准则。对于controlled experiments，特定的短期指标的选择需要很好地理解商业，同时理解以下这点很重要：长期目标不能总是与短期指标相对齐。

# 3.2 点击跟踪

## 3.2.1 背景

跟踪用户的在线点击和表单提交（比如：搜索(searches）)对于web分析、controlled experiments、以及商业智能很重要。大多数网站使用web beacons（1x1 pixel图片请求）来跟踪用户动作，但等待beacon在点击和提交上的返回会减慢下一动作（例如：展示搜索结果、或者目标页）。一种可能性是，使用一个较短超时(short timeout)，共识是：跟踪机制（停留在用户动作）的用时越多，数据损失越低。从Amazon、Google、Microsoft的研究表明，数百毫秒的小延迟，对于回报和用户体验会有剧烈的副作用，我们发现许多网站允许较长延时以便更可靠的收集点击数据。例如，到2010年3月，多个microsoft网站等待click beacons返回一个2s的timeout，这会在用户点击上引入一个大约400ms的平均延迟。关于该主题的一个白皮书最新已经发布[23]。据我们所知，该issue并没有被大多数网站owners所理解，它们的实现具有很大的click losses。对于广告，点击与支付相绑定，重定向通常被用于避免click loss。然而，对于用户这会引入一个额外的delay，对于tracking clicks来说并不常用。

## 3.2.2 Puzzling Outcome

仅仅添加一小段代码，比如：当用户点击一个搜索结果时，执行额外的JavaScript。需要在那个点被执行该段Javascript代码的原因是，在浏览器被允许处理和打开目标站点之前，目标站点的session-cookie被更新。

这会轻微地减慢用户体验，但实验表明用户会点击更多！为什么呢？

## 3.2.3 解释

用户点击更多的“成功”并不是真实的，准确来说是一个仪表误差（instrumentation difference）。chrome、firfox、safari对于结束从当前页导航走的请求更激进些，关于click beacons的一个不可忽视的比例并不会向server做出[23]。。。

# 3.初始效应(Initial Effect)

## 3.3.1 背景

给定，介绍中提到的评估A/Btesting具有很高的失败率，这非常常见，对于在实验的头几天，可以看到新版本是winner或者可以过早结束。
在该内容中会提到当新特性被引入时的两个效应：始效应（Primacy effects）和Novelty effects。这些都是负面效应，有时会影响实验。 当你在网站上变更了导航（navigation）时会出现Primacy effect，体验的用户可能会变得更低效，直到它们习惯了新导航，因而对于Control组有着天然的优势。相反的，当一个新设计或新特性被引入时，一些用户会研究该新特性，到处进行点击，从而引入一个“Novelty” bias，这会快速消逝，因为该特性并不是真的有用。该bias有时与”霍索恩效应(Hawthorne Effect)”有关，例如：一个短暂的提升。上述提到的实验，其中在MSN主页上的hotmail链接被变更成以独立tab/window方式打开hotmail，会有一个很强的Novelty effect：用户可能很吃惊，并尝试它多次。而Novelty effects会在一个较短时间后消逝，并产生一个更小的效应，长期影响(long-term impact)仍会是正向、无效、或负向的。在这个case中，long-term effect是正向的，该特性会驻留在MSN主页上。Primacy effects和Novelty effects可以通过生成在时间上的delta graph（在Control和Treatment间），以可视化或可分析地方式来进行评价，并评估趋势。如果我们怀疑这样的趋势，我们可以拓展该实验。为了评估正向效果（true effect）,可以做一个分析，其中只对在不同variants实验上的新用户计算OEC，因为他们没有受Primacy和Novelty effect的影响。另一个观点是，排除第一周的实验，因为delta通常会在一周之后变得稳定。我们的结果：大多数情况下疑似的Primacy和Novelty effects并不是真实的，只是统计的加工品（statistical artifact）。

## 3.3.2 Puzzling outcome

在许多实验中，前几天的效果看起来会趋高或者趋低。例如：图3展示了一个真实实验在关键指标上前4天的效果，其中在图中的每个点展示了直到那一天的累积效应（cumulative effect: delta），通过feature owner跟踪得到。

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/a5dc7f0ddf1f1bb78b7a88864a97f8339c76cb6e4544fca245f87eabb59629168e92f08ae4f16617eea3bf59e5a398e8?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图3

该效果表明：在前4天有一个很强的正趋势。实验者（它希望得到一个正向结果）看到了该intial negtive delta，但使用点虚线来线性推断该趋势，并认为在接下来的几天，该效果将以跨过0%并在第6天开始变成正向。这种思维很常见：我的新特性是明显很棒的，但对用户来说需要花费时间来习惯它，例如：在头几天时我们会看到Primacy effect。用户必须开始越来越喜欢该特性，这对吗？当然是错的！许多情况下，这是可预期的。

## 3.3.3 解释

对于许多指标来说，均值的标准差与$$1/\sqrt{n}$$成正比，其中n是用户数。出于简洁性，假设没有重复用户。例如，每个用户在实验期间只访问一次（结果不会变化很多，但使用实际会发生的次线性增长时）。因此，n与天数成正比。如图4所示，当实际效应(actual effect)为0时，对于测量效应（measured effet）的95%置信图。

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/3acf755776d89c41938871e4d4b07b1272293b489914f793c05998ef0e8ec7944b9dda0568642de1495154e8e7fa9105?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图4

前几天是高度可变的，因此在初始几天的效应(effect)可能会或高或低于在两或三周后的效应(effect)。例如，头一天具有67%的可能性在95%置信区间之外在实验尾部下降；第二天具有55%的可能性在该区间外。由于该序列(series)是自相关的(auto-correlated)，存在两个含义：

- 1.相对于之前实验发布的最终结果（它们会运行更长的时间），在初始几天的effects通常看起来过度正向或者负向。
- 2.在前几天期间，累积结果看起来会延伸。例如，假设一个实验在兴趣指标上没有effect。头一天可能是负向的-0.6%，但随着更多数据被累计，该效应(effect)会回归到0均值、95%置信锥的true。feature owners不正确地假设该effect是会延伸的，那将很快会跨过0线。当然，这很少会发生。

图3的graph实际上来自于一个A/A test（control和treatment间没有区别），effect的均值为0. 第一天具有一个负差值(negative delta)（注意：该宽置信区间会与0交叉），随着时间向前，该置信区间会收缩，结果会回归到该均值。确实，如图5所示，该graph会随时间在0附近稳定。

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/172a60697dc11692f8bc55ab1a0a279638c5975751bb292315ff1e756304b676effa49b1057e6fd854b39a23061db5e2?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=3.jpg&amp;size=750">

图5

## 3.3.4 学到的东西

出现"趋势（trends）"是可预期的，因为我们将它看成是一种educational和awareness issue，尽管我们承认后见之明(hindsight)是20/20,我们也会被初始趋势迷惑多次。当你开始涉及实现一个idea、并希望它成功时，确认偏误（confirmation bias）是很强的。当你构建一个假设并希望它趋向正确方向时，初始的负向结果通常进行抑制。

我们运行的实验很少有Primacy effects与intial effects相反的(reverse)，例如：某一特性初始是负向的，直到用户学会它并对它习惯，它才会是正向。由于存在这些效应(effects)，我们不能发现：单个实验在某一方向上具有统计显著的结果，在另一方向上也会统计显著（例如：一个统计显著的负向，变成统计显著的正向）。

大多数实验具有一个稳定效应（常数均值），但高方差意味着需要收集足够数据来得到更好估计；早期结果通常会有误导。由于存在真实的Novelty effects或Primacy effects，对于一个统计显著负效应，最常见的方式是，随时间变得更负向；而对于一个统计显著的正效应，随时间变得更正向。对于在几周之后扩展（extend）那些统计显著负向的实验，是没有多大价值的。

# 3.4 实验长度（Experiment Length）和统计强度(Statistical Power)

## 3.4.1 背景

不同于大多数离线实验，在线实验(online experiments)会持续补充（recruit）用户，而非在实验之前具有一个补充期（recruitment period）。因此，sample size随着实验的运行越长会增加。因此，很自然地期望：运行一个实验越长，会提供关于treatment effect的一个更好的估计，也具有更高的statistical power。注意，对于一些指标，比如：Sessions/User，该均值会随实验运行的越久而增加，也会基于百分比变化计算power。

## 3.4.2 Puzzling Outcome

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/1878cde3b1dd0fc7ce211743e05d4cb93098d406feeedf04dc9e67c1293d5ebb0a8678476ca401f0e1a61896ec1bc060?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=4.jpg&amp;size=750">

图6

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/6a3bd2fbf8f5ead436c5a519d50ab1902b75d668da114f2257754a318b33026a1fbfe330d069a6f54b2f19df7e71d5d1?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=5.jpg&amp;size=750">

图7



# 3.5 延滞效应（carryover effects）

## 3.5.1 背景

一些在线实验平台，包括：Bing、Google、Yahoo，依赖于“bucket system”来安排用户进行实验[3]。该bucket system会将用户随机化到不同的buckets中，接着安排buckets进行实验。这是个灵活的系统，允许对后验实验上对用户简单复用。

## 3.5.2 Puzzling Outcome

一个实验运行后，结果非常惊人。它通常是良好的，因为违反直觉的结果帮助我们理解新的ideas，但指标与在非预期方向的变化移动不相关，该效应(effect)是高度统计显著的。我们以一个更大的sample重新运行该实验来增加statistical power，许多该effect会消失。

## 3.5.3 解释

“bucket system”的一个大缺陷是，它具有carryover effects的缺点，受前一实验影响的相同用户，被用于接下来的实验中。这是已知的，可以运行A/A tests确认carryover effects，但当他们失败时，我们会失去能力直到我们对bucket assignment进行re-randomize。令人吃惊的是，craayover effect的持续时间（duration）。我们分享以下的两个示例。

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b4339fba1c2d952fd40efe0da090e907ad45d14ea8cca48cafb7664a87633e37f82df19303594e91a7443e4f0da45fe0?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=6.jpg&amp;size=750">

图8

在首个示例中，我们在三个阶段（stages）上运行了该实验，其中，我们在47天的A/B实验之前，在user buckets上有一个7-day的A/A实验。在我们完成实验后，我们关掉实验，接着在接下来的三周上继续监控相同的user buckets。图8展示了在treatment和control之间，OEC（Sessions/User）上的日常百分误差(daily percent delta)。灰色bars表示三个stages的划分。

很明显在实验完成之后，在用户上有一个carryover effect。该carryover effect看起来会在实验之后的三周左右消失。

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ee38ac5cbdb991768a47b0c3bca101dd04a8a4b7edd5362663a9703318745396d409533a47cb6797b2caecc98b6b4c93?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=7.jpg&amp;size=750">

图9

在另一个示例中，如图9所示，用户在实验中曝出的一个bug，会引起较差用户体验。carryover effect会持续更长时间。即使在三个月之后，user buckets仍然没有恢复到之前实验的水平。

## 3.5.4 缓解技术

尽管carryover effect本身对于实验者很重要，但对于一个实验平台来说保证质量更重要，而非抵制它。在bucket system中，由于user buckets会会从一个实验进行回收利用到另一个实验上，任意carryover impact可以很容易对后续实验造成bias。

缓解该问题的一种方式是，局部随机化（local randomization）。造成carryover effect的一个根源是：bucket system没有对每个实验进行重新随机化（re-randomize）。整个bucket system会依赖于一个频次不高的bucket re-assignment来对polulation进行随机化，接着该bucket分配在相当长周期内仍不变(constant)。将用户重新随机化到bucket system可以通过变更hashing function来完成，但该bucket system会以一个“bucket line”或者“layer”[3]的方式将所有实验进行成对化（couple），这样，我们必须停止在该bucket line上所有运行的实验，并变更hashing function，这会伤害（capacity）和敏捷性（agility）。另一个可选方案是，使用一个two-level的bucket system，它可以提供局部随机化；也就是说，只在一个buckets子集上进行重新随机化（re-randomizing），但不需要影响其它buckets。

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b79efd14b1d0d5d1257f4e9e0f2f44423e912e3a116ff5f7609c36d22c4c10380bd38dc49ab132829bdfa50716aea436?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=8.jpg&amp;size=750">

图10

上图展示了，局部重随机化是如何通过一个two-level bucket system来完成的。顶层的bucket system定义了包含在实验中的experiement units的一个集合，tratment assignment则在第二层bucket system中进行。对于每个experiment，第二层hashing会使用一个不同的hash seed。这保证了每一实验随机（per-experiment randomization），从而treatment的分配独立于任何历史事件，包括之前实验的carryover effects。上述方案的一个缺点是，我们不能使用一个共享的Control组：每个实验都需要它自己的Control，以便来自某一实验的任意carryover被“混合”到Control和Treatment(s)中。

局部随机化的一个好处是，我们可以运行一个“回顾式(retrospective)”的A/A 实验，无需实际占据总有效时间(calendar time)。通过对hashing function进行变更，在重启A/B实验之前，我们可以在前几天运行A/A实验进行评估。由于局部重新随机化的独立性，如果我们在实验之前对于任意周期回顾式地比较被分配到control和treatment组上的用户，该比较会是一个A/A。如果该A/A表明对于核心指标有影响，比如：p value < 0.2(由于划分的"unlucky"）, 我们需要重新变更hashing key并进行重试。


参考：

- [http://notes.stephenholiday.com/Five-Puzzling-Outcomes.pdf](http://notes.stephenholiday.com/Five-Puzzling-Outcomes.pdf)
