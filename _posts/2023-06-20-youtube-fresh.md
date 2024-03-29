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

由于closed feedback loop，新和长尾items很难被系统发现。**为了让这些内容露出，我们会将一个槽位或可浮动槽位（floating slot）给到新内容（<=X天的内容）和长尾内容（小于Y个positive用户交互的内容）。其余slots还是会使用生产系统进行填满**。该专用的新推荐系统的pipeline如图2所示：

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

- i) 可触达（Deliverable）。对于新内容，通过在一个dedicated slot上设置，一旦被上传，它们可以更轻易达到它们的目标受众。否则，在主推荐系统中，由于存在popularity bias，这些内容将面临与许多头部内容和更流行的内容之间的竞争
- ii) 可测量（Measurable）。有了一个“ dedicated slot”和一个更简单的pipeline，你可以更轻易地setup不同的treatments，并通过下面提出的user-corpus diverted实验来精准测量corpus影响
- iii) 可扩展（Extensible）。新内容推荐stack可以很容易地进行扩充：通过将单个slot treatment扩展到多个上，并允许忍受未发现的corpus增长

**user-corpus co-diverted实验**

在传统的A/B testing中，我们通常会采用一个 user-diverted setup，如图3（左）所示，其中，用户会被随机分配给control和treatment groups，并接受来自整个corpus中的相应推荐。你可以对比两个arms间的用户指标（比如CTR和停留时长），并从用户视角来测量treatment的效果。然而，两个arms会共享相同的corpus，由于treatment组泄漏（treatment leakage），user-diverted setup不能measure在corpus上的任何treatment效果。例如，在该实验中通过新内容推荐stack曝光的一个新item，它也会出现在control arm上。

为了克服这样的泄漏，我们会采用一个user-corpus-dieverted A/B testing setup（如图3（右）），其中：**我们会首先挑出x% corpus给control arm；以及挑出另外的非重合x% corpus的给treatment arm。**接着用户按比例被分配给不同的arms，这意味着在control arm中的用户只能接受来自control arm corpus中的内容，在treatment arm中的users只能接受来自treatment arm的内容。由于user size与corpus size成比例，例如：在实验期间，x%的corpus只会曝光给x%的users，在实验阶段中treatment的效果评估，与推全部署是一致的（当100%的用户被曝光给100% corpus上）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/d91a92f36e8fd4a9599e4c43367fa6b5fc9aa2fb06b2e705965a435c89b8997107d6ed4db2a273cbafbf7b2c92f53f4a?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3 User diverted vs. User corpus co-diverted实验图

**效果评估指标**

我们使用以下的corpus metrics来评估在推荐个性化新内容方面的效果评估，从覆盖率（coverage）和相关性（relevance）维度：

- **每日去重曝光内容数@K（DUIC@K: Daily Unique Impressed Contents）**：是一个corpus metric，它会统计每天接收到K次曝光的去重独立内容数。我们集中关注在低界上：例如：相对小的K值，来验证覆盖率（coverage）变化。
- **新内容停留时长（Fresh Content DwellTime）**：用于measure用户在曝光的新内容上花费的时间。更长的停留时间表示系统能承受用户在新内容上的偏好越精准，从而达到更高的相关性（relevance）。
- **在Y天内接收具有X（post bootstrapping）次正向交互的内容数（Discoverable Corpus@X,Ydays）**：用于measure新内容推荐的长期收益。通过post bootstrapping，我们不会统计从专有新内容推荐stack中接收到的items的交互数。一个更大的discoverable corpus表示：系统可以发现（uncover）和扶持（seed）更多有价值的内容：例如：在存在于专有slot之后，那些可以吸引到正向交互的内容，并通过自身能达到病毒式传播。

同时，为了确保新引入的新内容推荐不会牺牲更多短期用户engagement，我们也会考虑user metric：它会衡量在该平台上整体用户停留时间。

## 2.1 新内容的价值

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/5ff2160d98eed2613473fb316a2ab6d90ec4f7913f9b782887d3e5d91f3fb5c0a8f474ea18049f5a1ca882ec44064b28?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=4.jpg&amp;size=750">

图4 有了新内容推荐的dedicated slot，我们会展示：a) 不同曝光阈值下DUIC的提升 b) 在延后毕业Y天内，收到X次延后毕业点击数的提升 c) dedicated新内容推荐系统会从content providers上鼓励更多上传，随实验开展会有一个上升趋势

我们首先会在服务数十亿用户的商业推荐平台上开展user corpus co-diverted线上实验，并在超过一个月周期上measure建立新内容推荐stack的收益。**在这些实验中，control arm的users只会被展示主推荐系统生成的推荐。在treatment arm，会保留一个delicated slot来展示来自新内容推荐stack的推荐，而其它slots则被与control arm相同的主推荐系统推出的内容填充**。我们会做出以下观察：

- **Corpus coverage会提升**。图4(a)会绘制corpus coverage metric——DUIC@K。你可以观察到：不同的K值，会有4%~7.2%间的corpus coverage的一致增长。例如，在treatment arm中，由于delicated新内容推荐stack，存在超过7.2%的独特内容每天会接受到超过1000次曝光（对比起control arm）。正如预期，在更低的K值上更能更证明增长。

- **用户会探索更大的corpus**。在图(b)中，我们会制了discoverable corpus指标，它会measures在Y天内接到X post bootstraping正向交互的内容数的提升。另外，你可以在X的范围（从100到10k）、Y（从3到28天）天观察到一致提升。换句话说，有了在delicated新内容stacks中扶持的initial exposure与交互，更多数目的独特内容会被主推荐系统推出来，并因此被用户发现。**该提升也消除了新内容推荐stack不仅能增加corpus coverage，也能bootstrap更有价值内容**。有了更大的discoverable corpus，更多用户会发现围绕他们特定兴趣偏好中心的内容，从而带来最好的用户体验和一个更健康的平台。尽管一个新内容从上传到被主推荐系统选中并获得探索需要一定的时间，我们发现该数目趋向于在7天之后，即使对于那些高端X值。因而，在我们的真实实验中，我们使用discoverable corpus@X,7days作为main discoverable corpus metric。

- **content providers也会被鼓励从而上传更多内容**。图4(c)绘制了在使用dedicated新内容推荐stack上treatment arm上传的内容的增长，对比起control arm。通过一个月的实验周期，可以观察到一个一致的提升。另外，我们注意到随着实验继续有一个上升趋势。通过在dedicated slot关于新内容推荐，content providers会被鼓励上传更多内容作为他们的新上传items，获得更多曝光和收益。

- **用户会消费更多新内容，并在短期用户engagement上具有一个最小影响**。图5(a)演示了一个新内容数在7天正向交互上获得了+2.52%的极大增长。在图5(b)上，我们发现，平台上的整体用户停留时间会发生-0.12%的下降。然而，如图5(c)所示，对于小的content providers（少于多少订阅），用户停留时间会获得5.5%的增长。该trade-off会考虑上关于一个更大discoverable corpus、以及更多活跃content providers的long-term收益，如上所示。


<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/c8773da3faec1b7f8c60df5e09fb9ec1e78702fc007ec245c011b2da276a37051d7c70d6193925c4c013cfe3c6990026?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=5.jpg&amp;size=750">

图5 a) 新内容7天的正向交互数变化(% change) b) 平台上整体用户停留时长（% change） c) 小content provider的用户停留时长（% change）


对于treatment的dedicated slot和user corpus co-diverted实验框架的建立，我们会进一步解决新内容推荐stack的效率增长。

# 3.多通道（Multi-funnel）新内容推荐

不同于主推荐系统，专有新内容推荐stack会聚焦于新items，它们没有累积足够多的交互信息。在本节中，我们会描述：如何在该stack中扩展新内容提名，以便达到在提名新内容时的高覆盖和高相关。该技术可以轻易地泛化到pipeline的其它组件上。我们会围绕以下问题讨论：

- 1） RQ1: 那些**没有交互或少交互的新内容**，如何有效infer它们与用户间的相关度，并将这些内容进行bootstrap？
- 2） RQ2: 在**累积了一些初始交互feedback后**，如何快速利用有限的实时用户反馈来放大有价值内容？
- 3） RQ3: 如何在内容泛化和实时学习间进行平衡，并减小新内容的用户开销，以便我们在推荐新内容时，可以达到高coverage和高releance？

## 3.1 内容泛化

大多数CF-based推荐模型依赖于一个因子分解的backbone，来通过历史用户交互或ratings行为获得user embeddings和content/item ID embeddings。学到的embeddings接着会被用于infer用户在任意内容上的偏好。**由于items存在长尾分布，对于那些新上传的内容来说缺少足够的用户消费labels（engagement labels）来学习一个有信息的ID embedding**。实际上，没有合适的treatment，来自新和长尾内容的少量labels通常会被模型忽略，变为训练噪声。这些内容因此很少能曝光给合适的用户。

主要挑战是：在用户与这些新上传内容间的交互labels缺失。一种解法是：使用一个content provider-aware推荐系统，它可以bootstrap由用户熟悉并且订阅的provider生产的新上传内容。**为了克服交互labels缺少的问题，我们依赖content-based features来描述新上传内容如【28等】**。这些content features允许模型去泛化：将来自流行内容的充足engagement labels泛化到与这些内容相似的新上传内容上。


该模型结构会遵循**双塔结构**（如【51】），它会使用一个user和一个item/content tower，为了分别编码来自users和items的信息。在这些双塔间的dot product被学习用来预估在一个user和一个item间的历史偏好。模型本身仍会遵循popularity bias。**为了模型适应新内容推荐，我们做出以下变化**：

- 1）我们会在item tower上完全丢掉item ID，以阻止模型记住在单独items上的历史偏好
- 2）我们会排除关于一个item的历史流行度的任意features（例如：曝光数、正向engagements），以减少在item tower上的流行度bias

换句话说，在popular和新上传内容间，在item tower中只有meta features可以泛化。

**我们会开展一个在线A/B testing来measure上述变化对于提升新内容推荐的coverage和relevance上的影响**。

- control arm会运行一个双塔结构的提名模型，如第2节所解释，包括在item tower中与一个item相关的所有的meta features。
- treatment arm会运行完全相同的模型，但会在item tower上排除item ID和popularity features。

我们发现，通过移除item ID embeddings和item popularity features，corpus coverage指标：例如：DUIC@1000上涨3.3%，95%置信区间为[3.0%, 3.7%]。新内容停留时长也会增加2.6%。

- control model可以依赖item/content ID embeddings，以便从流行和确定的内容中记住交互labels，但它对新上传内容效果较差。
- treatment model则依赖于content features作为刻画用户在其它流行和确定内容上的偏好，从而学习到具有相似features的新内容，从而提升对于新内容推荐的相关性。

**在使用的Content Features**

在control和treatment arm中使用的content features，包含了多个从内容本身得到的具有不同粒度的categorical features，描述了语义topic、类目topic和内容语言。我们也包括了平均评分（rating）来过滤低质量内容。

## 3.2 Real-Time learning

提名器非常依赖于泛化的content features，它对于具有很少用户交互的新内容的启动来说是很有用的。由于缺少记忆能力，因而**它可以快速反馈用户的intial feedback**。这样快速的响应确实是必要的，因为：

- i) 我们通常没有能完整刻画该新内容、并能影响一个新内容的质量的所有features；
- ii) 对intial user feedback做出快速反映，可以帮助在早期分布中纠正低质量或低相关新内容，减小分发代价，同时快速重新分发并进一步放大高质量和相关新内容给其它具有相似兴趣的受众，以便进一步放大discoverable corpus的增长，并让content provider获得奖励上传更多内容。

这就需要**近实时提名（near real-time nominator）**，它可以挖掘数据，因为新交互数据会以流式方式进来。为了构建这样的一个近实时提名（near real-time nominator），这里提出：

- i) 使用near real-time user交互数据来训练
- ii) 带来一个低延迟的个性化检索模型

我们尝试让组成该nominator中的**不同组件runtime（例如：数据生成、模型训练、模型pushing）进行最小化**，以便从一个用户交互发生时，到该交互更新被用于serving模型之间的端到端延迟是**数小时内**。注意，对比起在主推荐stack中的已存在推荐模型（它的端到端延迟通常会是18-24小时或数天），这是个巨大的时延提升。数据生成job会收集在**新和长尾内容上最近15分钟的用户交互**，它会被当成是labels用来训练retrieval模型。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/3f3f7d95f7a897fbb197158112986cf42dfb53e38663b11044ae1494a274f0d463af578f94160c7731823c17085574ed?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=6.jpg&amp;size=750">

图6  用于推荐的Real-Time Sequence Model

**retrieval模型被训练的目标是：在给定用户在该平台上的历史交互后，预估用户会交互的下一个item**。该架构会再次使用一个双塔结构，其中：user/query tower会编码用户的交互序列，item tower会使用简单的ID embeddings以及categorical features，如图6所示。为了减少训练时间，我们会设计一个简单架构的模型。user state会被表示成用户最近交互的content IDs的embeddings的weighted sum，它会与user query features进行concatenated。特别的，我们会使用attention[41]来提升user state representation。对于一个具有最近n个消费内容$$[V_1, V_2, V_i, \cdots, V_n]$$的给定用户，我们会采用对最近n次交互进行weighted sum形式来获得user representation U:

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

这些features会被量化，并被编码成离线embeddings。**对于在item $$V_i$$上的历史交互，f接着学习将这些embeddings进行mapping到一个scalar上，得到最终weight $$w_i$$**。当聚合历史交互时，这样的一个weighted score $$w_i$$会强调内容更接近于具有更长dwelltime的当前request。在我们的实验中，我们发现，变更前面的简单attention设计，使用更复杂的序列模型（比如：RNNs）不会提升效果，出于最小模型复杂度，我们接着保持简单的weighted embedding。**模型会从前一checkpoint进行warm-start，并在最近的15分钟日志上进行训练，每个训练round大概一小时**。之后在发布到服务器的不同数据中心上提供真实服务。在serving time时，我们再次依赖一个多尺度量化方法，用于快速近似MIPS，并有效检索top-50的新候选内容。

**Category-centric Reweighting**

为了确保快速记住在这些新内容上的早期用户反馈（early user feedback），我们会在realtime nominator的item tower中包含item ID embeddings。在新内容上传时，交互数据在模式上区别很大：

- 一些会在几分钟内累积上千次交互数据
- 其它一些则只会在15分钟log上看到少量交互

**由于不均衡的交互数据，只依赖于ID embeddings的模型在新上传内容的"头部（head）"内容上会有over-indexing的风险**。为了克服该问题，我们也会包括一些第3.1节提到的content features来刻画这些items。然而：

- 许多categorical features会落入长尾分布
- 而另一些categorical features则会很普遍，被应用到大量items上，比如：“music”，
- 其它一些更专业和有信息量的，比如：“Lady Gaga Enigma+Jazz&Piano”

我们会通过在整个item corpus上的流行度进行inverse，我们会引入**IDF-weighting来调权一个feature**，为了泛化同时忽略那些普通存在的features，因此模型会更关注于学习这些更专业的content features。

## 3.3 低通道（Low-funnel） vs. 中通道（Middle-funnel）内容

在我们的目标中：在推荐新内容时要达到高coverage和高relevance存在trade-off。专有新内容推荐stack的corpus，确实可以被进一步归类成两部分：

- i) low-funnel内容：具有非常有限或者零交互
- ii) middle-funnel内容：会通过内容泛化收集到少量初始交互反馈

对于low-funnel内容，实时学习框架会丢掉它的预测能力，泛化这些内容是急需。另一方面，对于middle-funnel内容，更快的feedback可以控制real-time nomination系统的训练，允许更好的个性化和相关性。作为使用单个nominator来同时达到好的泛化和实时学习的尝试替代，我们会为不同通道(funnels)部署不同的nominators来解耦任务（如图7所示）：

- 一个具有好的泛化效果的nominator，目标针对low-mid funnel；
- 另一个nominator则关注于快速适应用户反馈，目标针对mid-funnel（具有合理量级的用户反馈来启动）

我们采用serving两或多个推荐系统的思想来同时获得更好的效果，并具有更少的缺点。另外，**我们也会讨论在这样的混合系统中如何决定何时来将一个low-funnel content转移到middle-funel中**。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/6e74b6a393c75c2afe6bbcd3006b5a64e9ec531178601d9c7d023e639a479ebf07d564664d60cf7499005fbfe01f2b1f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=7.jpg&amp;size=750">

图7 一个多通道提名系统

**Multi-funnel Nomination的query division**

一个naive策略，将两个nominators进行组合，并分别询问提名候选，接着依赖于graduaiton filter、prescorer、ranker来为dedicated slot来选择最终内容。但我们观察到，由于在ranker中的popularity bias，最终middle-funnel contents会以主宰slot。对于单个用户请求共同激活两个nominators也会有更高的serving开销。

为了缓和该问题，我们提出了query division multiplexing：用来随机选择two-tower DNN：具有概率p%来检索每个query的low-funnel candidates，或者具有(100-p)%的概率来检索来自middle-funnel的real-time nominator的candidates。我们在第4.2节中在corpus和user metrics间的tradeoff值开展不同的p值实验。

# 4.实验

在本节中，我们通过线上真实流量实验研究了在专有新内容推荐stack上的multi-funnel设计的效果提升。

## 4.1 Setup

我们测试了在第2节中介绍的专有新内容推荐stack的multi-funnel设计。特别的，我们对比了dedicated新内容推荐的以下方法：

- i) single-funnel提名器：使用单个推荐模型来提名新内容candidates。我们将single-funnel提名系统表示成S-two-tower，另一个使用real-time sequence model的称为S-real-time；
- ii) Multi-funnel提名器：会采用two-tower DNN来推荐那些低于$$n_{low}$$次正向用户交互的low-funnel content，real-time model则推荐那些在graduation threshold阈值下的middle-funnel content。这两个nominators会通过qeruy multiplexing进行组合，其中：two-tower DNN被用于p%随机用户请求，而real-time model则被用于剩余的(100-p)%.

我们设置：p为80，$$n_{low}$$为200，如第4.2节。我们会跑1%的在线user diverted实验来measure用户指标，5%的user corpus co-diverted实验来measure相应的corpus影响。

## 4.2 效果与分析

**multi-funnel nomination的影响**

对比multi-funnel nomination vs. single-funnel nomination在corpus以及user metrics，我们会做出以下观测：

- DUIC. 在图8(左)中，我们发现：对比起S-two-tower，S-real-time在low end上具有更低的DUIC。它在1000次曝光阈值上展示了1.79%的降级，这意味着：real-time nominator在推荐较少交互数据的low-funnel contents上要比two-tower DNN模型效率低。通过组合two-tower DNN、real-time nominaor，如图8(右)所示，我们观察到，low end上的DUIC会在multi-funnel nomination setup中得到极大提升，在DUIC@1000上有0.65%的提升。这意味着，对比起single-funnel setup，multi-funnel推荐可以提升新内容覆盖。

- Discoverable Corpus。


- user metrics。

**funnel transition cap的影响**

为了决定当一个新内容从low-funnel转移到middle-funnel时，我们要评估在不同的interaciton caps下two-tower DNN的泛化性的corpus效果。注意，当我们设置interaction cap为100时，它意味着，我们会限制该模型能index的corpus只会是那些具有最大100次交互的新内容。由于low-funnel推荐的主要目的是：提升corpus voerage，我们会主要关注不同caps的DUIC, 如表2所示。当cap设置为200时，DUIC@1000会达到它的最大值。设置该cap为100会达到相似的效果，但进一步降低cap会导致更差的指标。我们分析得到：当cap太低时，更多低质量内容会被强制提名，并在之后的ranking stage中由于更低的relevance而被过滤。事实上，我们会观察到，当cap从400降到100时，接收到非零曝光的unique contents的数目会有2.9%的下降。同时，需要满足来自low-funnel nominator的初始交互的特定量级之后，才能给real-time model提供学习信号。这意味着一个未来方向是：在middle funnel nominator和主推荐系统（比如：ranker）两者均需要提升泛化性，以便multi-funnel transition可以朝着low-funnel的方向移去。

**不同mix概率p%的影响** 

我们测试了不同的multi-plexing概率：p%


# 5.contextual流量分配

新内容推荐对于长期用户体验是有益的，它会造成短期用户engagement变得不那么popular或推荐不熟悉的内容。来到在线平台的用户通常在活跃级别上是不同的，会随着消费消息兴趣上的不同而非常不同。通常存在着一批核心用户（core users），它们会定期有规律的访问平台，其它则是临时用户（casual user），或新用户（emerging users），或者倾向于偶尔访问平台的用户。活跃级别的不同可能导致在用户分组上不同的内容消费模型。并且将用户进行grouping的方式可以在【9】中找到。

在初始探索（initial exploraition）中，我们会采用good CTR，基于用户在该点击后至少花费了10s钟，作为直接用户指标来评估推荐系统的短期效果。在图11中，我们发现，不同用户group的good CTR在由不同模型推荐出的候选上非常不同。例如，对比起real-time nominator，low-funnel模型（two-tower DNN）对于casual users来说会达到相似CTR，而对于core user则具有很大的gap。这意味着这些模型不仅在item corpus上具有不同的strength，（例如：low-funnel vs. middle-funnel），他们在处理不同user groups上也具有不同的strength。

该分析会激发一个潜在方法，可以进一步提升对于multi-funnel的query multiplexing的效果。对于core users来说，在泛化模型上的relevance loss对比起更低活跃级别的用户要更大。我们不会在相同概率下使用不同活跃级别来multiplexing用户，我们会进一步基于users/queries来将contextualize流量分配。我们会随机选择q%的核心用户，并使用来自real-time nominator、并利用它的短期用户engagement增益产生的nominations进行服务。其它用户则总是使用two-tower DNN来最大化corpus覆盖的nominations进行服务。如表3所示，通过使用real-time nominator以及使用不同的概率来服务核心用户，我们可以使用context-aware hybrid来进一步提升推荐效率。例如，当我们使用real-time nominator来服务40%的核心用户时，我们可以获得极大的dwell time以及good CTR提升，并在corpus coverage上有中立变更。更多综合的multiplexing策略在以后会再研究。


# 

- 1.[https://arxiv.org/pdf/2306.01720.pdf](https://arxiv.org/pdf/2306.01720.pdf)