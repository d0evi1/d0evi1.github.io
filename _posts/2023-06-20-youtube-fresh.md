---
layout: post
title: youtube多通道新内容推荐
description: 
modified: 2023-06-20
tags: 
---

google在《Fresh Content Needs More Attention: Multi-funnel Fresh Content Recommendation》中提出了multi-funnel新内容推荐：

# 1.介绍

推荐系统非常依赖于：在推荐平台上**将用户与合适的内容进行连接**。许多推荐系统会基于从推荐内容的历史用户交互中收集到的交互日志进行训练。这些交互通常是这些系统会选择将内容曝光给用户的条件，从而创建一个很强的feedback loop，导致“马太效应”。由于缺少初始曝光和交互，新上传内容（特别是那些具有低流行的content providers提供的）想要被推荐系统选中并被展示给合适用户，面临着一个极大的障碍。这常被称为item的cold-start问题。

我们的目标是：**打破feedback loop，并创建一个健康的平台，其中高质量新内容可以浮现，并能像病毒式扩散**。为了达成该目标，我们需要seed这些内容的初始曝光，从而补充推荐系统的缺失知识，以便将这些内容推荐给合适的用户。尽管一些新兴的研究关注于cold-start推荐，如何在工业规模推荐平台上bootstrap海量新内容（例如：每分钟上传到Youtube超过500小时的内容；Soptify每1.4秒上传一个新的track）仍是未被充分探索的。

更具体的，我们的目标是：**构建一个pipeline来展示新和长尾内容，并在推荐平台上增加discoverable corpus**。存在一个窘境：当系统探索完整范围（spectrum）的corpus，会导致提升长期用户体验【8，24】；然而，由于推荐了更少特定的内容，并推荐了更多其它新内容，这通常以会对短期用户体验有影响；我们希望缓和它。

为了在新内容推荐pipeline上**平衡short-term和long-term用户体验**，我们会在两个维度measure它的有效性：

- i) coverage：检查是否该pipeline可以得到更独特（unique）的新内容曝光给用户
- ii) relevance：检查系统是否将用户感兴趣的新内容与用户相连接

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/7d86185fa5454cbdcdf5a4e17b8ceb680cd851a06ad63aacccd0435c54f9214b78cc2bebe8fb5dbbd70ead7c7dba5c30?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 新内容推荐中的coverage vs. relavance

设计**新内容推荐stack**仍然面临着许多未知：

i) **如何定位(position)新内容推荐stack w.r.t 已存在的推荐stack？** 

我们会选择构建一个专有新内容推荐stack，它与其余推荐stacks（仅关注relevance，并且更容易强化feedback loop）相独立

ii) **哪些components需要在该stack中？** 

我们会设计该stack来包含一个提名系统、graduation filter、一个ranking系统来对candidates进行打分和排序

iii) **如何对coverage和relevance进行平衡？** 

一个基于随机推荐新内容的系统，达到最大的coverage，代价是低relevance并影响用户体验。为了达到平衡，我们会构建一个multi-funnel提名系统，它会分流用户请求在一个具有高coverage的模型和高relevenace的模型（详见图1）；

iv) **如何使用少量或无先验的engagement数据来建模内容？** 

我们会依赖一个two-tower DNN，它会使用content features来进行泛化，用于对intial分布进行bootstrap，并且一个序列模型会更新near real-time user feedback来快速寻找高质量内容的audience；

v) **如何对新内容推荐的收益进行measure？** 

我们会采用一个user-corpus co-diverted实验框架来measure提出的系统在增加discoverable corpus上的效果。

我们会以如下方式组织paper：

- 第2节：**专有multi-stage新内容推荐stack**：有nomination, filtering, scoring和ranking组件，有效对cold-start items进行bootstrap；并在提升corpus coverage、discoverable corpus以及content creation上showcase它的价值；
- 第3节：在专有新内容推荐stack上设计一个**multi-funnel提名系统**，它会组合高泛化能力的模型、以及一个near real-time更新的模型来有效对新内容推荐的coverage和relevance间进行平衡。
- 第5节：基于request context，将系统扩展到分流用户请求到这些模型上，以进一步提升新内容推荐

# 2.新内容推荐

**Pipeline setup**

我们首先介绍**专有新内容推荐stack**，它被用来在大规模商业推荐系统平台上的让相对新和长尾的内容露出（surface）。

生产环境的推荐系统具有多个stages，其中：

- 第一个stage包含了多个检索模型来推荐来自整个corpus的candidates
- 第二个stage会对candidates进行打分和排序
- 最后一个stage会根据多样性（diversity）和不同商业目标对选中的candidates进行打包（pack）

由于closed feedback loop，新和长尾items很难被系统发现。**为了让这些内容露出，我们会将一个不固定槽位（floating slot）给到新内容（<=X天的内容）和长尾内容（小于Y个positive用户交互的内容）。其余slots还是会使用生产系统进行填满**。该专用的新推荐系统的pipeline如图2所示：

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/4990aa4ff42a9ded82642885ee60f4b92cfbb77159a736493aacb98ccb968d6a3920dd5100d73bb34fefa141957a166e?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2 一个专有新内容推荐stack


每个组件的描述如下：

**新内容提名器（Fresh content nominator）**

在提名相对新和长尾内容中的一个关键挑战是：用户对这些数据缺少用户交互。为了克服冷启item推荐问题，我们采用一个双塔的content-based推荐模型：

- 1）一个query tower被用于基于消费历史来学习user representation；
- 2）一个candidate tower被用于基于item features来学习item representation。
    
该模型被训练来预估历史数据中的postive user interactions，幸运地，通过item features，它能泛化到那些具有零或非常少用户交互的items。在serving时，我们依赖于一个多尺度量化（multiscale quantization）方法，用于快速近似最大内积搜索（MIPS），**并有效检索top-50的fresh candidates**。

**毕业过滤器（Graduation filter）**

一旦新内容累积了初始交互，他们可以通过主推荐系统被轻易选到，并在其它slots上展示给合适用户。**在专有slot中继续探索这些items，相应的收益会递减，并且这些曝光资源可以被节省下来重新分配给其它潜在的新和长尾内容**。我们会采用一个graduation filter ，它会**实时移除那些已经被用户消费(consume)超过n次的内容**。

**Ranking组件**

一旦被提名的candidates通过了filter，我们会通过两个组件来对它们进行排序：

- 一个实时pre-scorer
- 与主系统共享的ranker

pre-scorer是轻量的（lightweight），可以实时反映用户的early feedback；**top-10的candidates被pre-scorer选中**，接着通过ranker（它是一个high-capacity DNN，具有更好的accuracy但会有更长latency）来评估candidates的相关性并**返回top-1**。特别的，轻量级pre-scorer实现了一个实时Bernoulli multi-arm bandit：每个arm对应于一个内容，每个arm的reward估计了一个新内容的good CTR（例如：在一次点击后基于用户消费至少10s的点击率），并遵循一个Beta后验分布：

$$
r_i \sim Beta(\alpha_0 + x_i, \beta_0 + n_i - x_i)
$$

...(1)

其中：

- $$r_i$$：是对于arm i的reward，具有一个先验Beta分布$$Beta(\alpha_0, \beta_0)$$，其中：参数$$\alpha_0, \beta_0 \in R_{>0}$$，
- $$x_i$$和$$n_i$$：分别表示arm i实时的交互数和曝光数

实际上，我们会通过对**至少100次以上曝光的新items**使用极大似然估计来估计全局先验（global prior）$$\alpha_0$$和$$\beta_0$$。在serving时，我们会采用Thompson Sampling根据等式（1）为每个candidate来生成一个sampled reward，并**返回具有最高rewards的top10内容**，由ranker来决定最终candidate。

该“专用slot（dedicated slot）”能让我们measure多种新内容推荐treatments，它具有更多可控性和灵活性。对比起“full-system”的新内容推荐（例如：允许在当前推荐stacks中进行更多探索(exploration)，并潜在影响每个slot），对“dedicated slot”进行setup具有许多优点：

- i) 可交付（Deliverable）。对于新内容，通过在一个dedicated slot上设置，一旦被上传，它们可以更轻易达到它们的目标受众。这些内容将面对与许多头部和更流行的内容（主推荐系统中存在popularity bias）进行许多竞争。
- ii) 可测量（Measurable）。有了一个“ dedicated slot”和一个更简单的pipeline，你可以更轻易地setup不同的treatments，并通过下面提出的user-corpus diverted实验来精准测量corpus影响。
- iii) 可延长（Extensible）。新内容推荐stack可以很容易地通过扩展单个slot treatment到多个上，并允许忍受未发现的corpus增长。


**user-corpus co-diverted实验**

在传统的A/B testing中，我们通常会采用一个 user-diverted setup，如图3（左）所示，其中，用户会被随机分配给control和treatment groups，并接受来自整个corpus中的相应推荐。你可以对比两个arms间的用户指标：比如CTR和停留时长，并从用户视角来测量treatment的效果。然而，由于两个arms会共享相同的corpus，user-diverted setup不能measure在corpus上的任意treatment效果，因为treatment组泄漏（treatment leakage）。例如，一个新item会通过在实验中的新内容推荐stack来接受曝光，它们也会出现在control arm上。为了克服这样的泄漏，我们会采用一个user-corpus-dieverted A/B testing setup（如图3（右）），其中：我们会首先暂时不考虑control arm上的x% corpus；以及对于treatment arm上的非重合x% corpus。接着用户被按比例分配给不同的arms，这意味着在control arm中的用户只能接受来自control arm corpus中的contents，在treatment arm中的users只能接受来自treatment arm的内容。由于user size与corpus size成比例，例如：在实验期间，x%的users会被曝光给x%的corpus，在实验阶段treatment的效果评估，与推全部署是一致的（当100%的用户被曝光给100% corpus上）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/d91a92f36e8fd4a9599e4c43367fa6b5fc9aa2fb06b2e705965a435c89b8997107d6ed4db2a273cbafbf7b2c92f53f4a?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3

**效果评估指标**

我们使用以下的corpus metrics来评估在推荐个性化新内容方面的效果评估，从覆盖率（coverage）和相关性（relevance）维度：

- 每日去重独立内容数@K（DUIC@K: Daily Unique Impressed Contents）：是一个corpus metric，它会统计每天接收到K次曝光的去重独立内容数。我们集中关注在低界上：例如：相对小的K值，来验证覆盖率（coverage）变化。
- 新内容停留时长（Fresh Content DwellTime）：用于measure用户在曝光的新内容上花费的时间。更长的停留时间表示系统能承受用户在新内容上的偏好越精准，从而达到更高的相关性（relevance）。
- 在Y天内接收X（post bootstrapping）个正向交互的内容数（Discoverable Corpus@X,Ydays）：用于measure新内容推荐的长期收益。通过post bootstrapping，我们不会统计从专有新内容推荐stack中接收到的item的交叉。一个更大的discoverable corpus表示：系统可以发现（uncover）和培育（seed）更多有价值的内容：例如：那些可以吸引晚多交叉的内容，并在专有slot后通过自己能达到病毒式传播。

同时，为了确保新引入的新内容推荐不会牺牲更多短期用户engagement，我们也会考虑user metric：它会衡量在该平台上整体用户停留时间。

## 2.1 新内容的价值

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/5ff2160d98eed2613473fb316a2ab6d90ec4f7913f9b782887d3e5d91f3fb5c0a8f474ea18049f5a1ca882ec44064b28?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=4.jpg&amp;size=750">

图4

# 3.多通道（Multi-funnel）新内容推荐

不同于主推荐系统，专有新内容推荐stack会聚焦于新items，它们没有累积足够多的交互信息。在本节中，我们会描述：如何在该stack中扩展新内容提名，以便达到在提名新内容时的高覆盖和高相关。该技术可以轻易地泛化到pipeline的其它组件上。我们会围绕以下问题讨论：

- 1） RQ1: 那些没有交互或少交互的新内容，如何有效infer它们与用户间的相关度，并将这些内容进行bootstrap？
- 2） RQ2: 在累积了一些初始交互feedback后，如何快速利用有限的实时用户反馈来放大有价值内容？
- 3） RQ3: 如何在内容泛化和实时学习间进行平衡，并减小新内容的用户开销，以便我们在推荐新内容时，可以达到高coverage和高releance？

## 3.1 内容泛化

大多数CF-based推荐模型依赖于一个因子分解的backbone，来通过历史用户交互或ratings行为获得user embeddings和content/item ID embeddings。学到的embeddings接着会被用于infer用户在任意内容上的偏好。由于items存在长尾分布，对于那些新上传的内容来说缺少足够的用户消费labels（engagement labels）来学习一个有信息的ID embedding。实际上，没有合适的treatment，来自新和长尾内容的少量labels通常会被模型忽略，变为训练噪声。这些内容因此很少能曝光给合适的用户。

主要挑战是：在用户与这些新上传内容间的交互labels缺失。一种解法是，使用一个content provider-aware推荐系统，它可以bootstrap由用户熟悉并且订阅的provider生产的新上传内容。为了克服交互labels缺少的问题，我们依赖 content-based features来描述新上传内容如【28等】。这些content features允许模型去泛化：将来自流行内容的充足engagement labels泛化到与这些内容相似的新上传内容上。


<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/c8773da3faec1b7f8c60df5e09fb9ec1e78702fc007ec245c011b2da276a37051d7c70d6193925c4c013cfe3c6990026?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=5.jpg&amp;size=750">

图5

该模型结构会遵循双塔结构（如【51】），它会使用一个user和一个item/content tower，为了分别编码来自users和items的信息。在这些双塔间的dot product被学习用来预估在一个user和一个item间的历史偏好。模型本身仍会遵循popularity bias。为了模型适应新内容推荐，我们做出以下变化：

- 1）我们会在item tower上完全丢掉item ID，以阻止模型记住在单独items上的历史偏好
- 2）我们会排除关于一个item的历史流行度的任意features（例如：曝光数、正向engagements），以减少在item tower上的流行度bias

换句话说，在popular和新上传内容间，在item tower中只有meta features可以泛化。

我们会开展一个在线A/B testing来measure上述变化对于提升新内容推荐的coverage和relevance上的影响。control arm会运行一个双塔结构的提名模型，如第2节所解释，包括在item tower中与一个item相关的所有的meta features。treatment arm会运行完全相同的模型，但会在item tower上排除item ID和popularity features。我们发现，通过移除item ID embeddings和item popularity features，corpus coverage指标：例如：DUIC@1000上涨3.3%，95%置信区间为[3.0%, 3.7%]。新内容停留时长也会增加2.6%。control model可以依赖item/content ID embeddings，以便从流行和确定的内容中记住交互labels，但它对新上传内容效果较差。treatment model则依赖于content features作为刻画用户在其它流行和确定内容上的偏好，从而学习到具有相似features的新内容，从而提升对于新内容推荐的相关性。

**在使用的Content Features**

在control和treatment arm中使用的content features，包含了多个从内容本身得到的具有不同粒度的categorical features，描述了语义topic、类目topic和内容语言。我们也包括了平均评分（rating）来过滤低质量内容。

## 3.2 Real-Time learning

一个提名器非常依赖于泛化的content featuers，它对于具有很少用户交互的新内容的启动来说是很有用的，它缺少记忆能力，因而可以快速反馈用户的intial feedback。这样快速的响应确实是必要的，因为：

- i) 我们通常不会有需要完全描述该内容并影响一个新上传内容的质量的所有的features；
- ii) 对于intial user feedback的提示反映，可以帮助纠正：在低质量或低相关新内容中的早期分布，减小开销，同时快速重新分发并进一步放大高质量和相关新内容给其它具有相似兴趣的受众，以便进一步增强discoverable corpus的增长，以及content provider获得奖励上传更多内容。这就需要近实时提名（near real-time nominator），它可以挖掘数据，因为新交互数据会以流式方式进来。

为了构建这样的一个近实时提名（near real-time nominator），我提出：

- i) 使用near real-time user交互数据来训练
- ii) 带来一个低延迟的个性化检索模型

我们开始最小化在构建该nominator中不同组件的runtime，例如：数据生成、模型训练、模型pushing，以便在一个用户交互发生时，和在被用于serving的模型（具有该交互更新）之间的端到端延迟是数小时内。注意，对比起在主推荐stack中的已存在推荐模型（它的端到端延迟通常会是18-24小时或数天），这是个巨大的时延提升。数据生成job会收集在新和长尾内容上最近15分钟的用户交互，它会被当成是labels用来训练retrieval模型。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/3f3f7d95f7a897fbb197158112986cf42dfb53e38663b11044ae1494a274f0d463af578f94160c7731823c17085574ed?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=6.jpg&amp;size=750">

图6

retrieval模型被训练的目标是：在给定在该平台上的历史交互后，用于预估用户会交互的下一个item。该架构会再次使用一个双塔结构，其中：user/query tower会编码用户的交互序列，item tower会使用简单的ID embeddings以及categorical features，如图6所示。为了减少训练时间，我们会设计一个简单架构的模型。user state会被表示成用户最近交互的content IDs的embeddings的weighted sum，它会与user query features进行concatenated。特别的，我们会使用attention[41]来提升user state representation。对于一个具有最近n个消费内容$$[V_1, V_2, V_i, \cdots, V_n]$$的给定用户，我们会采用对最近n次交互进行weighted sum形式来获得user representation U:

$$
U = \sum\limits_{i=1}^n w_i * Embedding(V_i)
$$

其中：

对于每个内容$$V_i$$的weight $$w_i$$，是在[0,1]内的normalized softmax weight，从item features得到：

$$
w_i = softmax(f(dwelltime(V_i), timegap(V_i), position(V_i)))
$$

其中：

- dwelltime会捕捉用户在item $$V_i$$上的用户消费时间
- timegap会捕捉在交互发生以及当前请求时间之间的time gap。

这些features会被量化，并被编码成离线embeddings。对于在item $$V_i$$上的历史交互，f接着学习将这些embeddings进行mapping到一个scalar上，得到最终weight $$w_i$$。当聚合历史交互时，这样的一个weighted score $$w_i$$会强调内容更接近于具有更长dwelltime的当前request。在我们的实验中，我们发现，变更前面的简单attention设计，使用更复杂的序列模型（比如：RNNs）不会提升效果，出于最小模型复杂度，我们接着保持简单的weighted embedding。模型会从前一checkpoint进行warm-start，并在最近的15分钟日志上进行训练，每个训练round大概一小时。之后在发布到服务器的不同数据中心上提供真实服务。在serving time时，我们再次依赖一个多尺度量化方法，用于快速近似MIPS，并有效检索top-50的新候选内容。

**Category-centric Reweighting**

为了确保快速记住在这些新内容上的早期用户反馈（early user feedback），我们会在realtime nominator的item tower中包含item ID embeddings。在新内容上传时，交互数据在模式上区别很大：一些会在几分钟内累积上千次交互数据，而其它一些则只会在15分钟log上看到少量交互。由于不均衡的交互数据，只依赖于ID embeddings的模型在新上传内容的"头部（head）"内容上会有运行over-indexing的风险。为了克服该问题，我们也会包括一些第3.1节提到的content features来刻画这些items。然而，许多categorical features会落入长尾分布。一些categorical features会很普遍，被应用到大量items上，比如：“music”，而其它更专业和有信息量的，比如：“Lady Gaga Enigma+Jazz&Piano”。我们会通过在整个item corpus上的流行度进行inverse，我们会引入IDF-weighting来调权一个feature，为了泛化同时忽略那些普通存在的features，因此模型会更关注于学习这些更专业的content features。

## 3.3 低通道（Low-funnel） vs. 中通道（Middle-funnel）内容

在我们的目标中：在推荐新内容时要达到高coverage和高relevance存在trade-off。专有新内容推荐stack的corpus，确实可以被进一步归类成两部分：

- i) low-funnel内容：具有非常有限或者零交互
- ii) middle-funnel内容：会通过内容泛化收集到少量初始交互反馈

对于low-funnel内容，实时学习框架会丢掉它的预测能力，泛化这些内容是急需。另一方面，对于middle-funnel内容，早期feedback可以控制real-time nomination系统的训练，允许更好的个性化和相关性。作为尝试使用单个nominator来达到好的泛化和实时学习的替代，我们会为不同通道(funnels)部署不同的nominators来解耦任务（如图7所示）：一个具有好的泛化效果的nominator，定向目标是low-mid funnel；另一个nominator则关注于快速适应用户反馈，定向目标是mid-funnel（具有合理量级的用户反馈来启动）。我们采用服务两或多个推荐系统的思想来同步获得更好的效果，同时具有更少的缺点。另外，我们也会讨论在这样的混合系统中如何决定何时来将一个low-funnel content转移到middle-funel中。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/6e74b6a393c75c2afe6bbcd3006b5a64e9ec531178601d9c7d023e639a479ebf07d564664d60cf7499005fbfe01f2b1f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=7.jpg&amp;size=750">

图7

**Multi-funnel Nomination的query division**

一个naive策略，将两个nominators进行组合，并分别询问提名候选，接着依赖于graduaiton filter、prescorer、ranker来为dedicated slot来选择最终内容。但我们观察到，middle-funnel contents会终止提名slot，因为在ranker中的popularity bias。对于单个用户请求共同激活两个nominators也会有更高的serving开销。为了缓和该问题，我们提出了query division multiplexing：用来随进选择two-tower DNN：具有概率p%来检索每个query的low-funnel candidates（或者real-time nominator具有(100-p)%的概率来检索middle-funnel candidates）。我们在第4.2节中在corpus和user metrics间的tradeoff值开展不同的p值实验。

- 1.[https://arxiv.org/pdf/2306.01720.pdf](https://arxiv.org/pdf/2306.01720.pdf)