---
layout: post
title: Smart Pacing流量控制介绍
description: 
modified: 2020-03-25
tags: 
---

2015年yahoo在《Smart Pacing for Effective Online Ad Campaign
Optimization》提出了一种smart pacing的策略。

# 0.摘要

在定向广告中，广告主会在预算范围内在分发约束的情况下最大化竞价效果。**大多数广告主通常喜欢引入分发约束（delivery constraint）来随时间平滑地花费预算，以便触达更广范围的受众，并具有持续性的影响**。对于在线广告，由于许多曝光会通过公开竞拍(public auctions)来进行交易，流动性（liquidity）使得价格更具弹性，在需求方和供给方间的投标景观（bid landscape）会动态变化。因此，对于同时执行平滑步调控制（smooth pacing control）并且最大化竞价效果很具挑战。本文中提出了一种smart pacing方法，它会同时从离线和在线数据中学习每个campaign的分发步调（delivery pace），来达到平滑分发和最优化效果目标。我们也在真实DSP系统中实现了该方法。实验表明，在线广告活动（online ad campaign）和离线模拟都表明我们的方法可以有效提升campaign效果，并能达到分布目标。

# 1.介绍

在线广告是一个数十亿美金的产业，并且在最近几年持续两位数增长。市场见证了搜索广告（search
advertising）、上下文广告（ contextual advertising）、保证展示广告(guaranteed display advertising)、以及最近的基于竞价的广告的出现。**我们主要关注基于竞价的广告（auction-based），它具有最高的流动性，例如：每次ad曝光可以通过一个公开竞价使用一个不同的价格来交易**。在市场中，DSPs（Demand-Side Platforms ）是个关键角色，它扮演着大量广告主的代理，并通过许多direct ad-network 或者RTB（实时竞价）广告交换来获得不同的广告曝光，来管理ad campaigns的整个welfare。一个广告主在一个DSP上的目标可以归为：

- **达到分发和效果目标**：对于品牌活动（branding campaigns），目标通常是花费预算来达到一个广泛受众、同时使得活动效果尽可能好；对于效果活动（performance campaigns），目标通常是达到一个performance目标（比如：eCPC <= 2美元），并同时尽可能花费越多预算。其它campaigns的目标通常在这两个极端之内。

- **执行预算花费计划(budget spending plan)**：广告主通常期望它们的广告会在购买周期内平滑展示，以便达到一个更广的受众，可以有持续性影响，并在其它媒介上（TV和杂志）增加活动。因此，广告主可以定制自己的预算花费计划（budget spending plans）。图1给出了budget spending plan的两个示例：平均步调（even pacing）和基于流量的步调（traffic based pacing）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/d47a9752d0621f63d31144653004e88291a21b5a70dd7c9922f243fe7929d6188623087c4ddabd22979d8e26ef557797?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1 不同的预算花费计划

- **减少创意服务开销（creative serving cost）**：除了通过由DSPs负责的开销外，也存在由第3方创意服务提供商负责的creative serving cost。现在，越来越多的广告活动会以视频或富媒体的形式出现。这种类型的曝光的creative serving cost可以与优质存货开销（premium inventory cost）一样多，因此，广告主总是愿意减少这样的开销，并且来高效和有效地分发曝光给合适的用户。

# 3.问题公式化

我们关注两个campaign类型： 1) 品牌广告（branding campaigns） 2) 效果广告（performance campaigns）。其它campaign类型位于两个之间。这些类型的campaign可以具有它自己唯一的budget spending plan。我们首先将该问题公式化来解决，接着给出我们的解法。

## 3.1 前提

假设Ad是一个ad campaign，B是Ad的预算，G是Ad的效果目标。spending plan是一个随着由K个time slots构成的预算序列，表示在每个time slot上待花费的期望预算数目。我们将AD的spending plan表示为：

$$
B = (B^{(1)}, \cdots, B^{(K)})
$$

其中，$$B^{(t)} >= 0$$并且 $$\sum_{t=1,\cdots,K} B^{(k)} = B$$。假设$$Req_i$$是由一个DSP接受到的第i个ad request。如第2节所述，我们使用概率节流（probabilistic throttling）来进行budget pacing control。我们表示为：

- $$s_i \sim Bern(r_i)$$：该变量表示：在$$Req_i$$上Ad是否参与竞价。其中：$$r_i$$是在$$Req_i$$上的point pacing rate。$$r_i \in [0, 1]$$表示Ad参与$$Req_i$$上竞价的概率。

- $$w_i$$：该变量表示在$$Req_i$$上参与该次竞价时是否赢得该Ad。它会依赖于通过竞价最优化模块（bid optimization module）给出的竞价$$bid_i$$

- $$c_i$$：当Ad服务于$$Req_i$$时的广告主开销。注意：开销包括inventory cost和creative serving cost。

- $$q_i \sim Bern(p_i)$$：该变量表示当Ad服务于$$Req_i$$时，用户是否会执行一些期望的响应（例如：click）。其中$$p_i = Pr(respond \mid Req_i, Ad)$$是这些响应的概率。

- $$C = \sum_i s_i \times w_i \times c_i $$是ad campaign Ad的总开销。

- $$P=C/\sum_i s_i \times w_i \times q_i$$：ad campaign Ad的效果（performance）（例如：当期望响应是点击时的eCPC）

- $$C = (C^{(1)}, \cdots, C^{(k)})$$：在K个time slots上的spending pattern，其中$$C^{(t)}$$是第t个time slot的开销，$$C^{(t)} >= 0$$并且$$\sum_{t=1,\cdots,K} C^{(k)} = C$$

给定一个广告活动Ad，**我们定义：$$\Omega$$是penalty(error) function，它会捕获spending pattern C是如何偏离spending plan B。值越小表示对齐（alignment）越好**。作为示例，我们会将penalty定义如下：

$$
\Omega (C, B) = \sqrt \frac{1}{K} \sum\limits_{t=1}^K (C^{(t)} - B^{(t)})^2
$$

...(1)

## 3.2 在线广告campaign最优化的Smart Pacing问题

广告主会花费预算，执行spending plan，并同时最优化campaign效果。然而，这样的一个抽象的多目标最优化问题，会有多个**帕累托最优解（Pareto optimal solutions）**。**在真实场景中，广告主通常会为不同campaigns对这些目标定制优化级**。对于品牌广告（branding campaigns），广告主通常会将花费预算放在最高优化级，接着是spending plan，而对效果并不很关注。在serving time时（例如：ad request time），由于我们使用概率节流（probabilistic throttling），**我们完全可以控制的唯一东西是$$r_i$$**。因而，对于没有指定效果目标的ad campaigns的smart pacing问题定义为：决定$$r_i$$的值，以便以下的测算是最优的：

$$
\underset{r_i}{min} P \\
s.t. C = B, \Omega (C,B) \leq \epsilon
$$

...(2)

其中，$$\epsilon$$定义了违背spending plan的容忍度。相反的，对于效果广告活动（performance campaigns）具有指定的效果目标，达成效果目标是top优先级。此时坚持spending plan通常是低优先的。我们将**smart pacing for ad campaigns with specific performance goals**的问题定义为：决定$$r_i$$的值，以便以下测算是最优的：

$$
\underset{min}{r_i} \Omega(C, B)  \\
s.t. P <= G, B - C <= \epsilon
$$

...(3)

其中，$$\epsilon$$定义了没有花光所有预算的容忍度。由于市场的动态性，单目标最优化问题很难解决。在工业界已存在被广泛使用的方法，只会捕获效果目标，或者只会捕获预算花完目标。达到效果目标的一个示例是：对retargeting beacon触发ad requests，总是竞价。不幸的是，避免过度消费（overspending）或者欠消费（underspending）是无保障的。对于平滑步调控制（smooth pacing control）的另一个示例是，引入一个全局pacing rate，以便ad requests具有相同的概率被一个campaign竞价。然而，这些已经存在的方法没有一个可以解决我们公式化的smart pacing问题。为了解决该问题，我们研究了流行的campain setups，并做出一些关键观察（可以触发我们的解）：

- CPM campaigns：广告主对于每个曝光会会付定固定数目的钱。对于品牌广告主（branding advertisers），campaign最优化的定义如公式2所示。只要预算可以被花费，并且spending pattern会随plan安排，高响应广告请求会比低响应的具有一个更高的point pacing rate，以便效果可以被最优化。对于效果广告主（performance advertisers，例如：eCPC、eCPA为目标），campaign最优化的定义如公式3所示。很明显，高响应的ad requerest应具有更高的point pacing rate来达到performance目标。

- CPC/CPA campaigns：广告主会基于clicks/actions的绝对数目进行付费。显式效果目标是，保证当代表广告主进行竞价时，DSP不会丢掉钱。因此，这种optimzation的定义为等式(3)。

- CPC/CPA campaigns：。。。

#

# 3.4 解法汇总

受这些观察的启发，我们开发了新的heuristics来求解smart pacing问题。该heuristics尝试找到一个可行解，它会满足如等式2或3定义的所有constraints，接着进一步通过feedback control来最优化目标。我们首先从离线服务日志中构建一个response prediction模型来估计$$p_i = Pr(respond \mid Req_i, Ad)$$，它会帮助我们区分高响应广告请求 和 低响应广告请求。第二，我们会通过将相似的响应请求group在一起来减小solution space，并且在相同group中的请求会共享相同的group pacing rate。使用高responding rates的groups会具有高的pacing rates（比如图2(a)中的蓝色箭头）。第三，我们会开发一个新的control-based的方法来从在线feedback data中学习，并能动态调整group pacing rates来逼近最优解。不失一般性，我们假设campaign setup是具有/没有一个eCPC目标的CPM计费。我们的方法可以应用到其它计费（billing）方法上 ，效果广告或者其它grouping策略，比如：基于$$p_i/c_i$$的grouping。（期望的response per cost）

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ade9127e6a1766a7afcf3c2193044b8661dcd83e128caef301a95b58169624f0e6e290c8afb3816940604de00b27b292?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2

# 4.response预测

我们的解法依赖于一个准确的response prediction模型来预估$$p_i$$。如第2节，有许多文献解决该问题。这里我们简单描述了如何执行该预估。我们会使用在(2,11)中的方法，并基于它做出一些改进。在这种方法中，我们首先利用在数据中的层次结构来收集具有不同间隔的response feedback features。例如，在ad侧，从root开始，接着一层接一层是：advertiser category，advertiser，campaign，最后是ad。在层次结构的不同levels上的历史响应率（historical response rates）可以当作features来使用机器学习模型（LR、gbdt等）来给出一个$$p_i$$的原始预估（raw estimation），称为$$\hat{p}_i$$。接着我们使用属性（比如：用户的age、gender）来构建一个shallow tree。树的每个叶子节点标识一个关于ad requests的不相交集合，它可以进一步划分成具有不同平均响应率的子集。最后，我们会在叶子节点$$Req_i$$内对$$\hat{p}_i$$进行微调，使用一个picewise linear regression来估计最终的$$p_i$$。该scheme会生成一个公平的accurate response prediction。

# 5.control-based的解法

在一个在线环境中，很难达到完全最优解来解决等式（2）和等式（3）的问题。我们采用启发法来减小原始问题的解空间。更特别的，使用第4节中描述的response prediction模型，相似的，responding ad requests会分组到一起，他们会共享相同的group pacing rate。不同分组会具有不同的group pacing rates来影响在高responding ad request groups上的偏好。原始问题（求解每个$$r_i$$的point pacing rate）会简化成求解一个group pacing rates的集合。我们会采用control-based的方法来调节group pacing rates以便online feedback data可以立即用于campaign最优化。换句话说，group pacing rates会通过campaign的生命周期动态调节。出于简洁性，在本文其它地方，pacing rate和group pacing rate会相通，我们会在第l个group的group pacing rate表示为$$r_l$$。

## 5.1 一个Layered Presentation

对于每个ad campaign，我们会维护一个layered数据结构，其中每层对应于一个ad request group。我们会以layerd数据结构来保存每个ad request group的以下信息：平均响应率（通常是：CTR、AR等，它来自response prediction模型）、ad request group的优先级、pacing rate（例如：在ad request group中对一个ad request的竞价概率）、campaign在该ad request group中在最新time slot上的花费。这里的原则是：

- 1) 对应于高响应ad request groups的layers应具有高优先级
- 2) 高优先级layer的pacing rate应会比一个低优先级layer要更小

对于每个campaign，当DSP接收到一个合格的ad request时，它会首先决定：该ad request 会落在哪个ad request group上，指的是相应的layer来获得该pacing rate。DSP接着会代表campaign来竞价，它的概率会等于由一个preceding bid 最优化模块给出的retrieved pacing rate。

## 5.2 Online Pacing Rate调节

我们基于实时反馈，来采用一个control-based方法来调节每层的pacing rate。假设我们具有L个layers，对于每个layer，由response prediction model给出的response rate预估是：$$p=(p_1, \cdots, p_L)$$，这里，如果期望的response是click，那么预估的每层的eCPC是$$e(e_1, \cdots, e_L)$$，其中：$$e_i = \frac{CPM}{ 1000 \times p_i}$$。假设每层的pacing rate在第t-1个time slot上是$$r^{(t-1)} = (r_1^{(t-1), \cdots, r_L^{(t-1)}}$$，那么，每个layer的spending为$$c^{(t-1)} = (c_1^{(t-1), \cdots, c_L^{(t-1)}}$$，对于将要到来的第t个 time slot会基于campaign目标，control-based的方法会预估$$r^{(t)} = (r_1^{(t)}, , \cdots, r_L^{(t)}$$。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/59cd41d4eda859c9e377fb9e7493b93a206d22864c9128127a47f5d07850f2b3759de543baf7f51e2f3985c71a676811?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=3.jpg&amp;size=750">

图3

## 5.2.1 没有performance目标的Campaigns

我们首先描述对于没有指定效果目标的ad campaigns的微调算法。对于这种campaign类型，首要目标是花费预算，并根据budget spending plan进行安排。因而，在每个time slot的end，算法需要决定在下一个time slot中的预算量，并调整layered pacing rates来花费该量。

在下一time slot中的待花费预算，由当前预算花费状态来决定。给定一个ad campaign，假设它的总预算是B，budget spending plan是$$B = (B^{(1)}, \cdots, B^{(K)})$$，在运行m个time slots后，剩余预算变为$$B_m$$。我们需要决定在每个剩余time slots中的花费，表示为$$\hat{C}^{m+1}, \cdots, \hat{C}^{(K)}$$，以便总预算可以花完，penalty最小。

$$
\underset{arg min \Omega}{\hat{C}^{(m+1)}, \cdots, \hat{C}^{(K)}} \\
s.t. \sum\limits_{t=m+1}^k \hat{C}^{(t)} = B_m
$$

...(4)

其中，如果我们采用等式（1）的$$\Omega$$定义，我们具有以下的最优化公式：

$$
\hat{C}^{(t)} = B^{(t)} + \frac{B_m - \sum_{t=m+1}^K B^{(t)}}{K - m}
$$

...(5)

其中，$$t=m+1, \cdots, K$$。我们会触发该细节：由于页面限制，如何估计$$\hat{C}^{(t)}$$。在在线环境中，假设在最新的time slot中的实际花费是$$C^{(t-1)}$$，我们定义$$R=\hat{C}^{(t)} - C^{(t-1)}$$是residual，它可以帮助我们来做出调整决策。

算法1给出了adujstment是如何完成的。假设index L表示最高优先级，index 1表示最低优先级，假设$$l'$$是具有非零pacing rate的最后一层。如果R=0, 则不需要做adjustment。如果R>0，这意味着需要加速分发，pacing rates会以一种自上而下的方式进行调整。从第L层开始，每一层的pacing rate会随层数一层层增加，直到第$$l'$$层。第5行会计算当前层的期望pacing rate，为了offset R。当第$$l' \neq 1$$时并且它的updated pacing rate $$r_{l'}^{(t)} > trial \ rate$$时，我们给第$$l' - 1$$层一个trial rate来准备进一步加速，如果R< 0，这意味着分发会变慢，每一层的pacing rate会以自底向上的方式减小，直接R是offset。第11行会生成当前layer到offset R的期望的pacing rate。假设l是最后要调的layer，$$l \neq 1$$和它的新的pacing rate $$r_l^{(t)} > trial \ rate$$，我们会给出第$$l-1$$层的trail rate来准备进一步加速。图4是一个分发如何变慢 的示例。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/1cc5b3643f8e08d6ee79275e0da8df9bf7ee94de3d364e82d919215976cd38d5adba3c146d427c8d8913403f0ce4a1e9?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=4.jpg&amp;size=750">

图4

我们注意到，在在线环境中，该greedy策略会尝试达到等式（2）的最优解。在每个time slot内，它会努力投资inventories，并在总预算和speding plan约束下具有最好的效果。

### 5.2.2 具有效果目标的Campaigns

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/e4a22c94acba05bee6f4e62b9328f8d927dcce839c5a77b9178158200530a01be14acc31030f849f35d4ea8ae939a7cd?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=a1.jpg&amp;size=750">

算法1

对于指定效果目标的campaigns（例如：eCPC <=2美元），pacing rate adjustment是有点复杂。很难预见在所有未来time slots内的ad request traffic，并且response rate分布可以随时间变化。因此，给定预算花费目标，利用在当前time slot中的所有ad requests，它们满足效果目标，不是等式（3）的最优解。算法2描述了对于这种类型的campaigns如何来完成adjustment。我们采用heuristic来进一步基于效果目标进行adjustment，它会添加到算法1中。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/e3a4edc69f4389e8a600c462e8d58577cb4faa790156da2a025ab777e0db511266ada0049996db6dc60d5c5cab2d8d97?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=a2.jpg&amp;size=750">

算法2


如果在算法1后的期望效果不满足效果目标，pacing rates会从低优先级layers one-by-one的减少，直到期望效果满足目标。第7行会生成current layer的期望pacing rate，并使整体期望eCPC满足目标。在第2行，第4行的函数$$ExpPerf(c^{(t-1), r^{(t-1)}, r^{(t)}, e, i}$$会估计layers $$i, cdots, L$$的期望联合eCPC，如果pacing rates会从$$r^{(t-1)}$$调整到$$r^{(t)}$$，其中，$$e_j$$是layer j的eCPC。

$$
ExpPerf(c^{(t-1)}, r^{(t-1)}, r^{(t)}, e, i) = \frac{\sum\limits_{j=i}^L \frac{c_j^{(t-1) \times r_j^{(t)}}{r_j^{(t-1)}}}}{\sum\limits_{j=i}^L} \frac{c_j^{(t-1)} \times r_j^{(t)}}{ r_j^{(t-1)} \times e_j }}
$$

...(6)

## 5.3 Layers的数目，初始化和Trial Rates

设置layers的数目，intial和trial pacing rates很重要。对于一个新的ad campaign，它没有任何分发数据，我们在DSP中标识出最相似的最已存在ad campaigns，并估计一个合适的全局pacing rate $$r_G$$，我们期望新的campaign可以花完它的预算。接着layers的数目设置为$$L = [\frac{1}{r_G}]$$。我们表示：一个合适数目的layers要比过多数目的layer更重要：

- 1) 如果有许多层，每个layer的分发统计并不重要  
- 2) 从系统角度看，过多数目的layers会使用更多宽带和内存

一旦layers的数目决定后，我们会在第一个time slot上运行全局pacing rate $$r_G$$。我们将该step称为一个**intialization phase**，这里分发数据可以被收集。我们将相同数目的曝光，基于它们的预测response rate来来标识layer分界，将它们分组（group）到期望数目的layers上。在下一time slot上，每一layer的pacing rate会基于在下一time slot的计划预算来重新分配，高响应layers会具有1.0的rates，而低响应layers将会具有0.0的rates。

在adjustment算法中，具有非零pacing rate相互挨着的直接连续的layer，会分配一个trial pacing rate。目标是在该layer收集分发数据，并准备将来加速。该trial rate假设相当低。我们通过保留预算的一个特定比例$$\lambda, e.g. \lambda=1%$$，生成这样一个rate，来在下一time slot中花费。假设trial layer是第l层，下一time slot上的预算是$$\hat{C}^{(t)}$$，我们会在至少一个time slot（初始阶段）上具有该layer的历史花费和pacing rate，trial pacing rate会生成：$$trial \ rate = r_l^{(*)} \times \frac{\lambda \times \hat{C}^{(t)}}{c_l^{(*)}}$$，其中：$$c_l^{(*)}$$和$$r_l^{(*)}$$是第l层的历史花费，以及pacing rate。

快速总结下，我们采用一个基于predicted response rate生成的关于所有ad requests的分层表示，并在每个layer level上执行budget pacing control来达到分发和效果目标。在当前time slot上的预算，以及剩余预算会被考虑来计算在下一time slot上的layered pacing rates。我们也尝试另一种方法来控制一个threshold，以便超过该threshold的predicted response rate的ad requests可以竞价。这种方法的缺点是不能令人满意。主要
原因是，ad requests通常不随response rate平滑分布，因此，在单个threshold上很难实现平滑控制。

# 6.实验评估



# 参考

- 1.[https://arxiv.org/pdf/1506.05851.pdf](https://arxiv.org/pdf/1506.05851.pdf)