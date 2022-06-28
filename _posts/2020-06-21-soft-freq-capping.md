---
layout: post
title: yahoo soft fequency capping
description: 
modified: 2020-06-21
tags: 
---


yahoo在2019年《Soft Frequency Capping for Improved Ad Click Prediction in Yahoo Gemini Native》提出了Soft Frequency Capping的技术。我们来看下：

# 1.介绍

yahoo的本地广告市场(native ad marketplace)（称为“Gemini native”）会提供给用户相应的广告：通过周围的本地内容进行重组后进行渲染（如图1所示）。对比搜索广告市场，用户意图通常是未知的。5年前启动，现在操控着每天数十亿美金的运行率，Gemini native是Yahoo的主要业务之一。每天超过20亿次曝光，数十万的active ads的库存，该市场会执行实时GSP（generalized second price）竞拍，会考虑：广告定向（ad targeting），预算考虑，频控规则（frequency）和近因原则(recency rule)，以及服务等级协议SLA（或latency）小于80ms的超过99%。

为了对在CPC价格类型特定context下的一个用户对本地广告进行排序，会为每个广告计算一个score（或expected revenue）：通过将广告主竞价 乘以 pCTR来得到。尽管Gemini native会处理其它价格类型，比如：oCPC，本文主要关注CPC价格类型。

pCTR会通过使用周期更新的模型“OFFSET”来计算：它是一个A feature enhanced collaborative-filtering (CF) based event-prediction algorithm。OFFSET是一个one-pass算法，它会为每个新的mini-batch的logged data使用SGD-based的学习方法来更新它的latent factor model。OFFSET的实现使用map-reduce架构，其中每个新的mini-batch的logged data会被预处理，通过许多mappers进行并行解析，模型实例的持续训练会通过多个reducers并行完成。

OFFSET通过他们的特征（age、gender、geo等）来表示它的用户，其中每个feature value（对于性别有：female、male或者unknown）通过一个latent factor vector（LFV）进行表示。一个用户的LFV通通应用一个non-linear function，它允许pairwise feature的依赖。由于OFFSET是一个user-less模型，一个特定用户观看一个特定广告（或frequency feature）的次数不能仅仅通过记录下来的曝光进行捕获。另外，frequency即不是一个user feature，也不是一个ad feature。因此，为了阻止用户重复观看同一个广告，基于硬频率捕获（HFC：hard frequency capping）的规则会在ad ranking过程被用于serving系统中。总之，用户在预定义的周期内观看一个ads超过预定义次数，会从ranked list中移除，不允许参与竞价。

从观测看，展示CTR会随着重复ad观看次数下降（15、17），在本工作中，我们会考虑一种新方法：通过模型将它看成是一种user-ad feature来处理频次。根据该方法，称为（SFC：soft frequency capping），对于每种曝光，frequency feature会通过user-ad pair进行计算，并用于训练一个frequency weight vector作为OFFSET SGD的一部分。在serving时，会根据incoming impression的frequency feature挑选合适的weight，并叠加到OFFSET score上。正如我们所见，frequency weight vector，产生的pCTR会随着frequency递减，表示用户对于重复观察相同的ads会厌倦。提出的方法在离线和在线评估上，对比SFC和HFC，表现出一个令人吃惊的效果提升。特别的，我们在在线实验服务真实用户时，获得一个7.3%的收益提升。SFC会增强OFFSET model，传到真实生产中，它会服务所有的Gemini native流量。我们也提供了关于frequency feature的统计，展示了不同人群在点击趋势。总之，在许多setting中会观察到“user fatigue（用户厌倦）”，因为CTR会随着频率特征的增加而递减。在许多特定的observation reveal中，男性和妇性用户体验相似的ad fatigue patterns，在观察5次广告后，在age group 50-60的群体上比group 20-30的群体的fatigue的两倍。

。。。
略

# 2.背景

## 2.1 Gemini Native

Gemini native是Yahoo主业务之一，。。。。

## 2.2 OFFSET Click-Prediction算法

Gemini native模型的算法是OFFSET：a feature
enhanced collaborative-filtering (CF)-based ad click-prediction algorithm[2]。对于一个给定用户u的pCTR和一个ad a，会有：

$$
pCTR(u, a) = \sigma(s_{u,a}) \in [0, 1]
$$

...(1)

其中：$$\sigma(x) = (1+e^{(-x)})^{-1}$$是sigmoid function，


# 4.FREQUENCY FEATURE

Yahoo users的日志活动会包括native ad impressions，从中我们可以抽取和计算frequency，例如：一个指定用户在一个预定义周期内看到一个特定ad的次数。我们可以计算每个ad featrure的frequency（例如：创意广告、campaign、或advertiser）。因此，在设置后，ad feature $$A_f$$、时间周期$$T_f$$，我们可以提供每个user u以及每个ad a相应的frequency featrue $$f_{u,a}(A_f, T_f)$$（或者简单些：$$f_{u,a}$$）。注意，通过定义，frequency feature是一个非负整数$$f_{u,a} \in N^{+}$$。

示例：假设user u看了三个广告$$a_1, a_2, a_3$$，每个ad $$a_i$$具有ad features：advertiser $$Ad_i$$、campaign $$C_{a_i}$$，creative $$Cr_i$$，另外，假设这是在星期六晚上，刚好在午夜后，user u的Gemini native在最近一周的天活动日期如表1所示（从左到右）。下面给出了不同settings下的frequency feature的一些值：

$$
f_{u,a_1} (camp., last day) = 2; f_{u, a_1} (adver., last day) = 3 \\
f_{u,a_2} (camp., last week) = 5; f_{u, a_2} (adver., last day) = 5  \\
f_{u,a_3} (camp., last  4 days) = 2; f_{u, a_3} (adver., last week) = 5
$$

# 5.统计与观察

在该节，我们提出一些关于frequencey的统计和观察。最重要的，我们展示了，frequency feature是很重要的，它对CTR具有重要影响。我们聚合了30天内的统计。它包括数十亿impression和clicks。我们注意到，当SFC方法包含在OFFSET中时，这里所使用的data会被收集，用于服务所有流量。

**Global view（全局视角）**

在图3中，平均归一化CTR，曝光数 CDF（cumulative density function），一个特定用户关于一个特定广告的观看数v（或frequency）会绘制成关于曲线。注意：v=0意味着，在这些曝光中，用户从未观看过在这之前的广告。 对于v次views的测算，normalized CTR可以通过除以average CTR来计算；对于v=0 （之前无views），可以通过平均CTR进行测算：

$$
CTR_n(v) = \frac{CTR(v)}{CTR(0)}; v=0,1, \cdots, 50
$$

注意，在两个曲线上，最后一点包括了对于v>=50以上所有聚合的measurements。

从图上检查，可以做出许多observations。我们抛开异常点v=25，CTR会随着观看次数（频次）进行单调递减。特别的，在只有单一past view之后，平均CTR会以20%下跌，在7次views之后几乎有50%。这是个很清晰的证据表明：用户被展示相同广告多次后的快速厌倦。然而，CTR下降率会随着views次数递减，并且曝光数会随着frequency递减（忽略最后一个点，它包含了v>=50以上的所有曝光）。特别的，47%的曝光是之前从未看过的广告（v=0），对于见过一次的广告（v=1）只有10%，见过两次的广告（v=2）只有6%。

**性别视角（Gender view）**

在图4中，normalized CTR和曝光CDF会为男性、女生、未知性别的frequency函数进行给制。Gemini native流量对于表2中的每种性别都是共享的。令人吃惊的是，男性用户要比妇性用户多。性别不确定是因为注册用户未标注性别。曝光CDF曲线会提供后续支持，70%的未知用户曝光是之前从未见过的广告（v=0），而男性、女性只有40%这样的曝光。

如图所示，我们注意到frequency在男性、女性用户上具有相同的效应，具有几乎相同的厌倦模式。然而，未知用户行为却很不同，对于广告重复次数具有更高的容忍度。对于这些用户的这样行为的一个合理解释是，这些用户很可能是未注册用户，到达Yahoo时，来自外部搜索或社交媒体网站，比起注册用户来说具有一个相当不同的体验。

图4

**年龄组视角（Age group view）**

类似。

**Yahoo vertical view**

# 6.我们的目标

采用一种soft frequency capping方法，通过将frequency fearure包含到OFFSET模型中。提出的解决方案，对比起HFD可以被优化来提供最佳的offline和online效果。

# 7.SOFT FREQUENCY CAPPING

总览， frequency feature可以是一个特定用户在某个预定义时间周期$$T_f$$（例如：last day、last week、last month）内对某个特定ad feature $$A_f$$（创意creative、广告campaign、广告主advertiser）的。

总之，我们将frequency feature看成是一个user-ad feature，其中我们会学习一个frequency weight vector(s)，对应于一个预定义的weights category参数$$W_c$$，它决定了我们是否具有单个全局的vector 或对于每个campaign or 每个advertiser具有一个独立的vector。

特别的，对于每个incoming train事件 {(u, a, y, t)}，feature value $$f_{a,u}(A_f, T_f)$$会进行分箱操作（binned），乘以合适的frequency weight vector的对应entry，并加上OFFSET score。frequency weight vectors会通过SGD使用user和ad features进行学习，label 为y（click或skip）。在serving time中，frequency weight vectors被用作OFFSET model的一部分来计算pCTR，并用于竞价。

**公式描述**

SFC方法如算法1描述。

为什么要binning？作为binning based方法的另一种选择，我们会使用一个线性回归来进行additive frequency weight：

$$
s_{u,a}^' = s_{um,a} + c_a \cdot g(f_{u,a})
$$

其中，$$c_a$$是一个weight，它可以被全局学到，每个campaign、或每个advertiser均有，$$g(\cdot)$$是一个arbitrary function。使用一个weight vector（每个bin具有一个weight entry）的优点是，我们不必假设：一个特定依赖（例如：$$g(\cdot)$$）会提供最好的效果，我们让模型来决定最好的拟合（fit）。在我们的case中，不存在缺点，因为frequency fearture可以具有非负整数值，可以避免量化错误（ quantizatio errors）。

**期望的影响（Expected impac）**

直觉上，这种方法的影响可以限制分数：对于相同用户对同一个ad出现重复观看。然而，理论上的考虑表明，这样的影响实际会强加给一个广告的首次views。

当我们使用HFC时，predictive model的得分会忽略：frequency会趋向于首次曝光和重复曝光的一个平均CTR。由于重复曝光会具有更低的CTR，这些得分会比首次view具有更低的CTR。添加SFC可以使得pCTR在首次view时具有更高的CTR，之后的views会随着SFC weights进行递减。因此，会接收许多次views的广告（ads）得分不再随这些views减小，从而它们的首次views的点击预测会更准确，之后的views也更准确。

# 8.效果评估

本节我们会进行在线、离线评估。

注意，由于记录的数据会被用于评估模型，很明显地，通过其它方式来产生结果是不可能的。该 警告在papers中很常见。

## 8.1 离线评估

为了评估离线效果，我们训练了两个offset models，一个使用SFC，如第7节所描述，其它没有frequency capping，作为baseline。我们会从头运行这些模型，其中，所有模型参数会被随机初始化，它会在一个月的Gemini native 数据上进行训练，包括了数十亿次曝光。

由于技术限制，我们使和以下的binning vector，它具有26个bins:

$$
B_{26} = [0:1), [1:2), \cdots, [25, \infity)
$$

另外，我们会测试SFC算法参数的多个组合，并找到最好的setup来使用campaign作为ad feature（$$A_f = campaign$$），并在last week（$$T_f=week$$上聚合views），并使用一个global weight vector（$$W_c=global$$），它会消除一些稀疏性。我们使用sAUC和LogLoss metrics来评估离线效果，在应用效果metrics之前，每次曝光会被用来训练系统。OFFSET hyper parameters，比如SGD step size和正则系数，会通过adaptive online tuning机制进行自动决定。

效果指标

**Area-under ROC curve(AUC)**：AUC指定了概率，给出两个随机事件（一正一负，点击或跳过），以及预测的pairwise ranking是正确的。

**Stratified AUC (sAUC)**：每个Yahoo section的AUC的平均加权（通过postive event，例如：点击数），该metric的使用是因为：不同Yahoo sections会具有不同的prior click bias，因此，单独使用section feature对于达到更高AUC values来说被证明是充分的。

**LogLoss**

$$
\sum\limits_{(u,a,y,t) \in T} -y log pCTR(u,a) - (1-y) log(1 - pCTR(u,a))
$$

其中，$$T$$是一个training set，$$y \in \lbrace 0, 1\rbrace$$是postive event indicator（例如：click或skip）。我们注意到，LogLoss metric会用于优化OFFSET model参数以及它的算法超参数。

结果：LogLoss和sAUC的提升会随时间进行绘制，在图7上，对于一个使用binning vector $$B_{26}$$训练的OFFSET model，最好的SFC算法参数是：$$A_f=campaign, T_f=week, W_c=global$$，其中每个点表示了数据3小时价值。从图中看出，SFC model的优点是，所有提升都是正向。特别的，在last week训练上我们在LogLoss上有平均1.02%的提升，在sAUC上有0.83的提升。我们注意到，对于一个成熟算法来说，要达到这样的高accuracy提升，需要持续优化数年，这相当令人印象深刻。

为了达到这部分，我们将产生的全局campaign frequency weight vector如图8所示。weights 会随着frequency单调递减，其中，最后一点会包含所有大于v=25的frequency，它在曲线最下会掉落。后者会造成pCTR在该区域不准确，例如：$$f_{u,a}=25$$的under-prediction以及对$$f_{u,a} >> 25$$的over-prediction。由于我们具有较少的events，并具有更高的frequencies（如图3所示），这表明整体平均效果是under-prediction的。

**统计异常解释**

异常包含了在nCTR的小“jump”，它会“破坏”nCTR随frequency的单调递减性，这会发生在v=24和v=25间。如上所述，当SFC集成到offset中时，会使用binning vector $$B_{26}$$中的$$[25, \infity)$$作为最后一个bin，进行收集统计信息。在之前的paragraph中，这会造成一个整体under-prediction effect在该区域的下降。由于statistics会被收集来进行auction wining events，我们会获得in-spite。

际

# 参考

- 1.[https://dl.acm.org/doi/pdf/10.1145/3357384.3357801](https://dl.acm.org/doi/pdf/10.1145/3357384.3357801)