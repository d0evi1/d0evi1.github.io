---
layout: post
title: baidu Query-Ad Matching算法介绍
description: 
modified: 2020-06-05
tags: 
---

baidu在《MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu’s Sponsored Search》介绍了它们的query-ad matching策略。

# 摘要

为了构建一个高效的竞价搜索引擎（sponsored search engine），baidu使用一个3-layer的漏斗（funnel-shaped）结构，来从数十亿候选广告中筛选(screen)出数百万的广告并进行排序(sort)，同时需要考虑低响应时延以及计算资源限制。给定一个user query，top matching layer负责提供语义相关的ad candidates给next layer，而底层的ranking layer则会关注这些ad的商业指标（比如：CPM、ROI等等）。在matching和ranking目标(objectives)间的明显分别会产生一个更低的商业回报。Mobius项目旨在解决该问题。我们尝试训练matching layer时，除了考虑query-ad相关性外，会考虑上CPM作为一个额外的optimization目标，通过从数十亿个query-ad pairs上直接预测CTR来完成。特别的，该paper会详述：当离线训练neural click networks，如何使用active learning来克服在matching layer上click history的低效（insufficiency），以及如何使用SOTA ANN search技术来更高效地检索ads（这里的ANN表示approximate NNS）。

# 1.介绍

baidu每天会处理数十亿的在线用户，来处理它们的多种queries。我们都知道，广告对于所在主流商业搜索引擎来说都是主要收入来源。本paper中，主要解释在baidu search ads系统上的最新尝试和开发。如图2所示，在搜索广告中它扮演着重要角色，会与user query相关的该广告会吸引点击，当它们的广告被点击时，广告主会支付广告费。baidu竞价搜索系统的目标是，在在线用户、广告主、付费搜索平台间形成一个闭环。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/a903296f47b05bca3758b42aa5a814a16b08275a6799d325cba9dffd44eb535e406fd66c35269a3c9e287896c8a44e6d?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750" width="300">

图2

通常，付费搜索引擎会通过一个two-step process来展示广告。第一个step会检索给定一个query的相关广告，下一step会基于预测user engagement来对这些ads进行rank。由于在baidu中竞价搜索引擎的商用性，我们采用一个3-layer漏斗形结构的系统。如图3所示，top matching layer负责提供与一个user query以及user的profile相关的ad候选池到next layer。为了覆盖更多的语义相关广告，此处会大量使用expansion【1,3,4】以及NLP技术【2】。底层的ranking layer会更关注商业指标，比如：cost per mile（CPM=CTR x Bid），回报（ROI），等。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/e771fffce5f57854b25cf594e1b0b3900dfa8c0ebe50f697087f88f411c81beeaa097c1ec09cd992359d5cfe5b00537e?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750" width="300">

图3

然而，在matching和ranking objectives间的不同，出于多种原因会导致一个更低的商业回报。给定一个user query，我们必须采用复杂模型，并花费大量计算资源在对数百或数千个ad候选进行ranking。可能令人失望的是，ranking模型上报分析得到：许多相关ads并没有提供高CPM，从而不会被展示。为了解决该问题，Baidu Search Ads建立“Mobius”项目，它的目标是建立关于付费搜索引擎的next generation query-ad matching system。之项目旨在统一不同的 learning objectives，包括：query-ad相关度以及许多其它商业指标，并符合：低延迟、计算资源的限制，以及对用户体验的较小不良影响。

本paper中，我们介绍了Mobius-V1: 它会让matching layer除query-ad相关度外，采用CPM作为一个额外的optimization objective。Mobius-V1能为数十亿user query&ad pair精准和快速预测CTR。为了达到该目标，我们必须解决以下主要问题：

- 1.**低效的点击历史（insufficient click history）**：由ranking layer所采用的原始的neural click模型会通过高频ads和user queries进行训练。它趋向于估计一个具有更高CTR的query-ad pair来展示，尽管它们具有低相关性。
- 2.**高计算/存储开销**：Mobius期望预测关于数十亿user query&ad pairs的多个指标（包括：相关度、CTR、ROI等）。它天然会面对对计算资源更大开销的挑战。

为了解决上述问题，我们受active learning【34,41】的思想启发，首先设计了一个“teacher-student” framework来加大训练数据，为我们的大规模neural click model来为数十亿user query&ad pair预测CTR。特别的，一个离线数据生成器会负责要建人造query-ad pairs来给出数十亿的user queries和ad candidates。这些query-ad pairs会被一个teacher agent进行judge，它派生自原始的matching layer，并擅长于衡量关于一个query-ad pair的语义相关度。它可以在人造query-ad pairs上帮助检测bad cases（比如：高CTR但低相关度）。我们的neural click model，作为一个student，通过额外bad cases被教会（teach）来提升在长尾queries和ads上的泛化能力。为了节省计算资源，并满足低响应时延的要求，我们进一步采用大量最近的SOTA ANN search，以及MIPS（Maximum Inner Product Search）技术来高效索引和检索大量ads。

# 2.Baidu付费搜索的愿景（vision）

长期以来，漏斗结构是付费搜索引擎的一个经典架构。主要组件包含了：query-ad matching和ad ranking。query-ad matching通常是一个轻量级模块，它会measure一个user query与数十亿ads间的语义相关度。作为对比，ad ranking模块则关注商业指标（比如：CPM、ROI等），并使用复杂的neural模型来对数百个ad候选进行排序后展示。这种解耦结构是个明智选择，可以节约大量计算开销。另外，它可以帮助科学研究和软件工作上作为两个模块，分配给不同的团队可以最大化目标的提升。

百度的付费搜索使用一个3-layer的漏斗结构，如图3所示。top matching layer的最优化目标是最大化在所有query-ad pairs间的平均相关度：

$$
O_{matching} = max \frac{1}{N} \sum\limits_{i=1}^n Relevance(query_i, ad_i)
$$

...(1)

然而，根据我们在baidu付费搜索引擎表现上的长期分析，我们发现，在matching和ranking objectives间的区别/差异会导致更低的CPM（关键指标）。当在ranking layer的模型上报出：许多由matching layer提供的相关ads在搜索结果中并没有被展示，因为它们在估计时没有更高的CPM，这很令人不满意。

随着计算资源的快速增长，baidu搜索ads团队（凤巢）最近确立了Mobius项目，它的目标是baidu付费搜索的下一代query-ad matching系统。该项目的蓝图如图4所示：是统一多个学习目标，包括：query-ad相关度，以及许多其它商业指标，将它们统一到单个模块中，遵循：更低的响应延迟、有限的计算开销、以及对用户体验上较小的影响。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/af26ee3102518402bf2386dd65dc65b8980d5608fe6b4cd9e38fc4679ff5cd200da57fb551f0b8a7f91f16b274e56d54?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=3.jpg&amp;size=750">

图4

该paper主要是Mobius的第一版，它会首先尝试teach matching layer考虑在query-ad relevance之外，将CPM作为一个额外的最优化目标。这里我们将目标公式化：

$$
O_{Mobius-V1} = max \sum\limits_{i=1}^n CTR(user_i, query_i, ad_i) \times Bid_i \\
s.t. \frac{1}{n} Relevance(query_i, ad_i) \geq threshold
$$

...(2)

这里，如何为数十亿的(user-queries, ad候选) pairs精准预测CTR是个挑战。

# 3. 系统

## 3.1 Active-learned CTR模型

baidu付费搜索引擎使用DNN来进行CTR模型预测（G size）具有超过6年的历史。最近Mobius-V1采用一个新的架构。构建Mobius-V1的一种简单方法是，复用在ranking layer中的original CTR模型。它是一个大规模和稀疏的DNN，擅长emmorization。然而，在CTR预测上对于长尾部的user queries和ads它也会有一个严重的 bias。如图5所示，在搜索日志中，同一用户有两个queries："Tesla Model 3"和"White Rose"。对于过去使用的漏斗架构，在query "tesla Model 3"和ad "Mercedes-Benz"间的相关性会在matching layer中有保证。接着，在ranking layer中的neural click模型会趋向于在query-ad pair预测一个更高的CTR。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/789379bc5aca62d3489fa6395909439a0a56d85badb1c19aa163ef8ce0962b2b2cfd57d0f46a4a128a19cc6ed2ae5ba7?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=4.jpg&amp;size=750" width="300">

图5

根据我们在query log上的分析，ads和user queries具有长尾效应以及冷启动问题。因此，我们不能直接利用原始的neural click model来为数十亿长尾的user queries和ads精准预测CTR。解决这个问题的关键是：**如何teach我们的模型学会：将"低相关但高CTR(low relevance but high CTR)"的query-ad pairs认为是bad cases**。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ce95440ceabc018e06f9686528a7b27f4a55ab892676a281d430ba8f371e235aba4e52d0b9e13fe681eda1d3f4567978?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=a1.jpg&amp;size=750" width="300">

算法1

为了解决这个问题，我们提出了使用在matching layer中的原始relevance judger作为teacher，来让我们的neural click model知道“low relevance” query-ad pairs。我们的neural click model，作为student，会以一个active learning的方式从有争议的bad cases上获得关于relevance的额外知识。图6通过一个流图展示了这种方式，算法1展示了使用active learning来teaching我们的neural click model的伪码。总之，active learning的迭代过程具有两个阶段：data augmentation和CTR model learning。特别的，我们会详细描述这两个phase。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/f63aea2487fc4db78773d0b83c2b51d229cf5f15f4aa82ba71f98adcee0651a6136b9b43ec81b0b84008a7c5a51da10f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=5.jpg&amp;size=750">

图6

数据扩量（data augmentation）的阶段会将一个来自query logs的click history的batch加载到一个data augmenter开始。data augmenter每次会接收query-ad pairs，它会将它们split成两个sets：一个query set和一个ad set。接着，我们会对两个sets使用一个cross join操作（\otimes）以便构建更多的user query & ad pairs。假设在click history的batch中存在m个queries和n个ads，那么data augmenter可以帮助生成$$m \times n$$个synthetic query-ad pairs。在列出所有可能的query-ad pairs后，relevance judger会参与其中，并负责对这些pairs的relevance进行评分。由于我们希望发现低相关度的query-ad pairs，会设置一个threold来保留这些pairs作为candidate teaching materials。这些low relevance query-ad pairs，会作为teaching materials首次被feed到我们的neural click model，接着每个pair会通过前一迭代的updated model进行预测CTR来进行分配。为了teach我们的3-classes(例如：click、unclick和bad)的neural click model学会认识“low relevance but high CTR”的query-ad pairs，我们可能直觉上会设置另一个threshold，以便过滤出大多数low CTR query-ad pairs。然而，我们考虑另一个选项来对augmented data的exploration和exploitation进行balance。我们会采用一个data sampler，它会选择并标记augmented data，被称为这些synthetic query-ad pairs的predicted CTRs。一旦一个query-ad pair被抽样成一个bad case作为我们的neural click network，这些pair会被标记为一个额外的category，例如：bad。

在学习我们的CTR模型的阶段，click/unclick history以及labeled bad cases两者都被添加到augmented buffer中作为训练数据。我们的neural click network是一个large-scale和multi-layer sparse DNN，它由两个subnets组成，例如：user query DNN和ad DNN。如图6所示，在左侧的user query DNN，会使用丰富的user profiles和queries作为inputs，而右侧的ad DNN会将ad embeddings作为features。两个subsets会生成一个具有96维的distributed representation，每个都会被划分成三个vectors(32 x 3)。我们对user query DNN和ad DNN间的vectors 的这三个pairs使用3次inner product操作，并采用一个softmax layer进行CTR预估。

总之，我们贡献了一种learning范式来离线训练我们的neural click model。为了提升在CTR预测上对于长尾上数十亿query-ad pairs的泛化能力，neural click model(student)可以actively query到relevence model (teacher)来进行labels。这种迭代式监督学习被称为active learning。

## 3.2 Fast Ads Retrieval

在baidu付费广告搜索中，我们使用如图6的DNN（例如：user query DNN和ad DNN）来各自获取queries和ads的embeddings。给定一个query embedding，Mobius必须从数十亿ad candidates中检索最相关并且最高CPM值的ads，如等式(2)。当然，为每个query穷举式地计算它不实际，尽管brute-force search理论上可以发现我们要找的所有ads（例如：100% ad recall）。online services通常具有严格的latency限制，ad retrieval必须在一个短时间内完成。因此，我们采用ANN search技术来加速retrieval过程，如图7所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/98f71495af72713d58b8ed9e1fd59a1ebae37f2e4fa8e0f02e030ee7cd5c4a01e125b531c0dbfb13c974b994c3df9ced?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=6.jpg&amp;size=750" width="400">

图7

### 3.2.1 ANN Search

如图6所示，mapping function会结合user vectors和ad vectors来计算cosine相似度，接着该cosine值传给softmax layer来生成最终的CTR。这种方式下，cosine值和CTR是线性相关的。在模型被学到之后，它很明显是正相关或负相关的。如果它是负相关，我们可以很轻易地将它转换成正相关，需要对ad vector加个负号就行。这样，我们可以将CTR预测问题reduce成一个cosine ranking问题，它是一个典型的ANN search setting。

ANN search的目标是：对于一个给定的query object，从一个大语料中检索到“最相似”的对象集合，只需要扫描corpus中的一小部分对象就行。这是个基础问题，已经在CS界广泛研究。通常，ANN的最流行算法是基于space-partitioning的思想，包括：tree-based方法、random hashing方法、quantiztion-based方法、random partition tree方法等。对于这类问题，我们发现，random partition tree方法相当高效。random partition tree方法的一个已知实现是："ANNOY"。

### 3.2.2 MIPS （Maximum Inner Product Search）

在上面的解决方案中，business-related weight信息在user vector和ad vector matching之后被考虑。实际上，这个weight在ads ranking很重要。为了解释在ranking之前的这个weight信息，我们通过一个weighted cosine问题公式化成fast ranking过程，如下：

$$
cos(x, y) \times w = \frac{x^T y \times x}{||x|| ||y||} = (\frac{x}{||x||})^T (\frac{y \times w}{||y||})
$$

...(3)

其中：

- w是business related weight
- x是user-query embedding
- y是ad vector

注意，weighted cosine会造成一个inner product seraching问题，通常被称为MIPS。

### 3.2.3 向量压缩（Vector Compression）

为数十亿ads存储一个高维浮点feature vector会花费大量的disk space，如果这些features需要在内存中进行fast ranking时会造成很多问题。一个常见的解法是，将浮点feature vectors压缩成random binary（或integer） hash codes，或者quantized codes。压缩过程可以减小检索召回到一个范围内，但它会带来具大的存储优势。对于当前实现，我们会采用一外quantization-based方法（像K-Means）来将index vectors进行聚类，而非对index中的所有ad vectors进行ranking。当一个query到来时，我们首先会寻找query vector所分配的cluster，接着获取来自index属于相同cluster的ads。PQ的思路是，将vectors分割成许多subvectors，每个split独立进行cluster。在我们的CTR模型中，我们会将query embeddings和ad embeddings同时split成三个subvectors。接着每个vector被分配到一个关于cluster centroids的triplet上。例如，对于一个billion-scale ads的multi-index，可以使用$$10^9$$可能的cluster centroids。在Mobious-V1中，我们使用OPQ（Optimized Product Quantization）变种。

# 4.实验

略



# 参考

- 1.[https://dl.acm.org/doi/pdf/10.1145/3292500.3330651](https://dl.acm.org/doi/pdf/10.1145/3292500.3330651)