---
layout: post
title: kuaishou D2Q介绍
description: 
modified: 2023-05-06
tags: 
---

kuaishou在《Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation》提出了D2Q的模型。

# 0.摘要

观看时间预测仍然是通过视频推荐加强用户参与度的关键因素。鉴于在线视频的日益流行，这一点变得越来越重要。然而，观看时间的预测不仅取决于用户与视频的匹配程度，而且常常被视频本身的时长所误导。**为了提高观看时间，推荐系统总是偏向于时长较长的视频**。
在这种不平衡数据上训练的模型面临**偏见放大**的风险，这误导平台过度推荐时长较长的视频，却忽视了用户的根本兴趣。
本文首次研究了视频推荐中观看时间预测的时长偏差问题。我们采用了因果图来阐明时长是一个混杂因素，它同时影响**视频曝光和观看时间预测**——第一个效果对视频造成偏见问题，应该被消除，而第二个效果对观看时间的影响来自视频的内在特性，应该被保留。为了去除不希望的偏见同时利用自然效果，我们提出了一个时长去混杂分位数基础（D2Q）的观看时间预测框架，它允许在工业生产系统中进行扩展性应用。通过广泛的离线评估和实时实验，我们展示了这一去混杂时长框架的有效性，显著优于现有的最先进基线。我们已经在快手应用上全面推出了我们的方法，这显著提高了实时视频消费量，因为观看时间预测更加准确。

# 1.介绍

在线视频消费的兴起促使人们不断努力优化基于互联网的视频点播（VOD）系统的推荐系统，如YouTube以及流媒体播放器，如TikTok、Instagram Reels和快手（见图1的演示）。因此，**一个主要目标是提高用户观看视频的时间，即所谓的预期观看时间[7]**。观看时间是每个视频观看中存在的密集信号，它关系到平台上的每个用户和视频，并代表了用户注意力的稀缺资源，这是公司竞争的关键。因此，在用户到达时，准确估计候选视频的观看时间至关重要。**准确的预测使平台能够推荐可能具有较大观看时间的视频以提高用户参与度，这直接推动了关键的生产指标——日活跃用户（DAU）——从而推动收入增长**。

图1

观看时间主要受两个因素的影响。众所周知，它主要由**用户对视频的兴趣程度**决定，并且当完全没有兴趣匹配时，观看时间可以为零[12, 29]。同时，**视频本身的时长（即视频的长度）**也在决定用户在视频上花费多长时间方面起着重要作用。图2显示用户观看时间与视频时长正相关。因此，标准的观看时间预测模型通常使用时长以及其他视频特征作为特征输入来进行预测[7, 8]。

图2

然而，这种做法不幸地在许多推荐系统中导致了偏见问题。图3展示了由于平台最大化用户观看时间的目标，推荐系统逐渐偏向于时长较长的视频。结果，时长较长的视频可能会被过度曝光，以至于用户真正的兴趣在推荐中被低估。更严重的是，在这种不平衡的数据上训练的模型会由于系统的反馈循环而放大时长偏见[27]，这不利于理想推荐中的多样性和个性化。

图3

尽管非常普遍，但与由item受欢迎程度或推荐位置引起的许多其他bias相比，时长bias的探索要少得多[1, 2, 18, 31, 38–40]。为了最大化用户观看时间，推荐系统可能会学习到物理时长和观看时间之间的虚假关联；因此，**即使时长较长的视频可能无法很好地匹配用户兴趣，它们也更有可能被展示**。另一方面，时长较长的视频通常由于现有的不平衡曝光而拥有更大的样本量，这可能会主导模型学习，使模型性能在不同时长上有所不同。

本文首次研究了观看时间预测中的时长偏见。我们使用了一个有向无环图（称为因果图[20]）来表征观看时间预测中关于时长的因果关系，由图4(a)建模。具体来说，时长作为一个混杂因素[20]，同时影响观看时间预测和视频曝光。时长对观看时间的第一个效应表明，用户倾向于花更多时间观看本质上物理时长更长的视频，这是一个自然效应，应该被观看时间预测模型捕捉。然而，时长对视频的第二个效应是一个bias项，困扰了许多观看时间预测模型。**这种效应表明，时长影响视频印象的可能性，这代表了模型对时长较长视频的不公平偏好，应该被消除**。与以往只使用时长作为观看时间预测特征的工作相比，这种对时长效应的明确建模使我们能够去除不希望的偏见，同时保留真正的影响。

为了处理时长偏差，我们遵循**后门调整[21]**的原则，并对观看时间预测的因果图进行干预，以**去除时长对视频曝光的不良影响**，如图4(b)所示。我们注意到，时长对观看时间的影响被保留，因为这种关系是内在的，应该在预测中被利用。在操作上，**我们将训练数据按照时长等分为若干部分；对于每个时长组，我们学习一个回归模型来预测组内观看时间的分位数，其中label由原始观看时间值和相应组内观看时间的经验累积分布确定**。这种分位数预测使得模型参数可以在不同时长组之间共享，带来可扩展性的好处。总的来说，我们总结我们的贡献如下：

- 观看时间预测中时长偏差的因果表述。我们采用因果图来形式化观看时间预测中被忽视但普遍存在的问题——时长偏差。我们指出时长是一个混杂因素，它同时影响观看时间预测和视频曝光，其中前者是内在的，应该被保留，而后者是偏差，应该被消除。
- 可扩展性的时长调整。在后门调整的指导下，我们根据时长划分数据，并为每个时长组拟合一个观看时间预测模型，以去除视频曝光上的时长偏差。我们根据时长修改观看时间标签，允许跨组参数共享，并实现可扩展性。
- 广泛的离线评估。我们在快手应用收集的数据上进行了一系列离线评估，以展示我们的模型相对于现有基线的优势。我们进一步对时长组数进行了消融研究，发现随着组数的增加，我们的模型性能首先提高（得益于时长去偏差），然后下降（由于组内样本量减少导致的估计误差增加）。
- 在实时实验中的好处。我们进一步在实时实验中实施了我们的方法，以促进快手平台上的视频推荐，表明通过去除不希望的时长偏差，我们的方法提高了观看时间预测的准确性，并与现有策略相比，有助于优化实时视频消费。

# 3.Watch-Time prediction的因果模型

我们的目标是：当推荐一个视频给某用户时，预估该用户在的watch time。我们会通过一个因果关系图（causal graph）进行公式化：它会将user、video、duration、watch-time、以及推荐系统在watch-time prediction和视频曝光上关于duration的混杂效应（confounding effect），如图4(a)所示：


<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/e25509291ef28f0ce491abd8dcd1149b636e14733fbb4a8eaac0b213279b26f5b75d01bab9101400bb63055145123144?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=4.jpg&amp;size=750">

图4 watch-time prediction的因果关系图：U-user、V-video、D-duration、W-watch time。图(a)建模了在视频曝光和watch-time prediction上的confounding effect。图(b)使用backdoor adjustment来 deconfound duration，并移除它在视频上的effect。

- U：表示user representation，包含了：用户人口统计学（user demographics）、即时上下文（instantaneous context）、历史交互等
- V：表示video representation，包含了：video topics等
- **D：表示video duration，例如：视频长度**
- W：表示用户花费在观看视频上的时间
- $$\lbrace U, V \rbrace \rightarrow W$$：会捕获在watch-time上的interest effect，它可以衡量用户对该视频有多感兴趣
- $$D \rightarrow W$$：会**捕获在watch time上的duration effect**，它会建议：当两个视频与用户兴趣相匹配时，**更长的视频会接受到更长的watch time**
- $$D \rightarrow V$$：表示**duration会影响视频的曝光**。推荐系统经常会对具有更长duration的视频有不平等的偏好；这样的bias会通过feedback loop会放大，如图3所示。另外，duration会影响模型训练，因为：i) sample size随duration的不同而不同，具有长duration的视频通常**具有更大的sample size**，这意味着 prediction模型具有更好的performance； ii) **在标准模型（比如：WLR）中，具有不同duraiton的videos会接受到不同sample weights，（它会影响在模型训练时的梯度分配）**。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/6dec8ff2f8a99cdd0a4c4b63e552d0fe2930b4757a9ee367024a3e1f0899cd6f76196df58a1647cc674aa1aa0327c5af?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3 在11个月上，kuaishou APP每个video duration相应的视频曝光变化。bins以duration的升序进行排序。bin的高度表示在该周期内曝光的差异。出于置信原因，绝对值会被忽略。平台的目标是提升watch time，**曝光会偏向于那些具有长duration的视频**。

明显地，在图4(a)中的causal graph表明：duration是一个会通过两条路径（$$D \rightarrow W, D \rightarrow V \rightarrow W$$）影响watch-time的混淆因子。第一条path会建议：duration具有一个与watch time的直接因果关系，它可以通过watch-time prediction被捕获，因为用户趋向于花费更多时间在长视频（对比起短视频）上。**然而，第二条path会暗示着：video exposure不希望被它的duration所影响，因而，视频分布会偏向于长视频；如果没有缓解，由于推荐系统的feedback loop，predictions会面临着bias amplification的风险**。


# 4.Duration Bias的后门调整（backdoor adujstment）

在本节中，我们会根据backdoor adjustment的原则来对duration进行解混淆（deconfound），其中：**我们会移除来自duration的bias，但会保留来自在watch time上duration的效应**。我们提出了一个可扩展的watch-time prediction框架：时长解混淆&基于分位的方法（Duration-Deconfounded and Quantile-based (D2Q)），主要内容有：

- i) 将数据基于duration进行划分来消除duration bias
- ii) 拟合watch-time 分位，而非原始值；来保证参数可以距多个groups进行共享以便扩展

我们将我们的training和inference过程分别归纳在算法1和算法2中。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/57dce8a85043b8a497d335ec6f5dbe709e617ca6e1f5f92ee8f6d6cb3ce208574dbad8e0f4d4ac6971d5ad1a355be51e?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=a1.jpg&amp;size=750">

算法1

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/859749bfde7c9b608d508dd518c4c4fc189d26e631d4ab0da5f6d57c36660f6a1c1c8fd936ac928c37074c2ce59a1c92?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=a2.jpg&amp;size=750">

算法2

## 4.1 解混淆Duration

根据do计算，通过移除edge：D -> V，我们会block掉在视频曝光上的duration effect，如图4(b)所示。我们将watch-time prediction模型看成是 $$E[W \mid do(U,V)]$$，并且有：

$$
\begin{align}
E[W \mid do(U,V)] & \overset{i}{=} E_{G_1} [W | U,V] \\
& \overset{ii}{=} \sum\limits_d P_{G_1} (D = d | U, V) E_{G_1} [W | U,V,D = d] \\
& \overset{iii}{=} \sum\limits_d P(D=d) E[W| U,V,D = d]
\end{align}
$$

...(1)

其中：

- (i)是总期望；
- (ii)是因为D独立于$$\lbrace U,V \rbrace$$，干预会移除在graph $$G_1$$中的边$$D \rightarrow V$$；
- (iii)是因为：这样的干预不会改变W在条件{U,V,D}上的W分布，D的间隔分布仍会相同

等式（1）阐明了deconfound duration的设计：**你可以独立估计$$P(D)$$和$$E[W \mid U,V,D]$$，接着将他们组合在一起来构建最终的estimation**。在本paper中，我们提出**将duration分布P(D)离散化成不相交的groups**，并拟合group-wise watch-time预估模型$$E[W \mid U,V,D]$$来完成估计。

## 4.2 基于Duration分位数的Data-Splitting

我们现在会展示一个使用duration deconfounded来估计watch-time的通用框架，如图4(b)所描述。更高层的思路是：**将数据基于duration进行划分，并构建group-wise watch-time estimation以便在视频曝光上对duration进行debiase**。

特别的，为了阻止 边D -> V，我们基于duration分位数将训练样本进行划分成M个相等的部分，它可以将分布P(D)离散化成不相交的部分。假设：$$\lbrace D_k \rbrace_{k=1}^M$$是这些duration groups。继续(1)中的派生，我们通过下面近似来估计deconfounded model $$E[W \mid do(U,V)]$$：

$$

\begin{align}
E[W \mid do(U,V)] & = \sum\limits_d P(D = d) E[W | U,V,D = d] \\
& \approx \sum\limits_{k=1}^M 1 \lbrace d\in D_k \rbrace E[W | U,V,D \in D_k] \\
& = \sum\limits_{k=1}^M 1\lbrace d \in D_k \rbrace f_k (U, V)
\end{align}
$$

...(2)

这里我们提供了一个关于“为什么这样的基于duration的数据划分过程，可以解缓图4(a)中边D->V的bias问题”的直觉解释。**在标准的watch-time预估模型（如：WLR）中，具有长watch-time weights的样本会在梯度更新中采样更多，因而预估模型经常在短watch-time的样本上表现很差**。Watch-time是与duration高度相关的，如图2所示。通过基于duration进行数据划分，并将模型以group-wise方式拟合，我们可以在模型训练期间，缓和那些具有长watch-time的样本、以及具有短watch-time的样本的inference。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/3d37cd784136be40da62045d7a83f6289f84ef1468209a5631948dd77067b2ea4cac9afcb5538d984295c9f579b36ef6?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2 根据duration在视频上的60分位的watch time。阴影区（spanned area）表示watch-time的99.99%置信区间

然而，这样的data-splitting方法会抛出另一个问题。如果对于每个duration group $$D_k$$我们拟合一个单独的watch-time prediction模型$$f_k$$（如图5(a)所示），model size会变得更大，这在真实生产系统中是不实际的。但如果我们允许在duration groups间进行参数共享，使用原始watch-time labels进行拟合等价于没有data-splitting的学习，这在duration deconfounding上会失败。下面部分会解释：如何通过**将原始watch-time labels转换成duration-dependent watch-time labels来解决该窘境**，并允许我们同时移险duration bias，并维持模型参数的单个集合来获得可扩展性。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ae269d16eae3cb49eed1137bb4f04b905d548e9324c33ebb811d25d7097f983a1ddbe4267deb35009ecbb45a01e22027?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=5.jpg&amp;size=750">

图5 对于在每个duration group预计watch-time的不同模型结构，例如：$$\hat{\phi}_k(h(u,v))$$。

- 图(a)会拟合独立的模型来预估每个duration group的watch time。“dense input”指的是：视频历史统计数（例如：historical show count, empirical watchtime）等。“ID input”指的是ID features（例如：user id, video id）和categorical features（例如：video类目、用户性别）
- 图(b)会拟合跨所有duration groups的单一模型，其中watch-time分位数的labels会通过在相应的duration group中的watch-time empirical distribution计算得到
- 图(c)会进一步使用网络结构中的duration信息，并相应地提升watch-time estimation

## 4.3 每个Duration Group估计Watch-time

接着，我们描述了，如何使用来自所有duration groups的数据来拟合单个watch-time prediction模型。回顾下我们的设计有两部分：

- i) duration debiasing
- ii) 参数共享

问题的关键是：**将watch-time label转换成duration-dependent，通过根据不同duration group拟合watch-time分位数来实现，而非原始values**。我们会引入Duration-Deconfounded
Quantile-based (D2Q) watch time预估框架。

$${\widehat{\phi}}_k (w)$$：表示在duration group $$D_k$$中关于watch-time的经验累计分布（empirical cumulative distribution）。

给定一个user-video pair (u,v)，D2Q方法会在相应的duration group中预估它的watch-time分位数，接着使用$$\hat{\phi_k}$$将它映射到watch time的值域中（value domain）。也就是说：

$$
f_k(u, v) = \widehat{\phi}_k^{-1} (h(u, v))
$$

...(3)

其中：h是一个watch-time分位数预估模型，它会拟合在所有duration groups上的数据：

$$
h = \underset{h'}{argmin} \sum\limits_{\lbrace (u_i, v_i, w_i)\rbrace_{i=1}^n} (h'(u_i, v_i) - \widehat{\phi}_{k_i}(w_i))^2
$$

...(4)

其中：

- $k_i$是样本i的duration group，以便$d_i \in D_{k_i}$。

你可以应用**任意现成的regression模型来拟合分位数预估模型h，并维护在所有duration groups间单个模型参数集**。

接着，在inference阶段，当一个新的user-video pair $(u_0, v_0)$到达时，模型会首先发现：视频$v_0$会属于哪个duration group $D_{k_0}$，接着将watch-time  quantile预估$h(u_0, v_0)$映射到watch-time值$\widehat{\phi}_{k_0}^{-1}(h(u_0, v_0))$上。我们会在算法1和算法2上总结learning和inference过程。

在该方式下，D2Q会拟合那些是duration-dependent的labels。我们注意到：video duration会是model input的一部分，会将来自不同duration groups的不同样本进行输入，如图5(b)所示。另外，来自不同duration groups的样本会共享关于watch-time quantile的相同的label，但具有不同的特性——一个模型在跨groups学习watch-time quantile时会失败。**为了完全利用duration information，你可以在模型结构中额外包含一个duration adjustment tower（比如：ResNet）**，我们在图5(c)中将它称为Res-D2Q。第5节演示了Res-D2Q会在D2Q之上进一步提升watch-time prediction accuracy。

对应于duration的watch-time labels的转换，允许在跨duration groups间同时进行 deconfounding duration bias和 parameter sharing。然而，随着duration groups的数目增加，group sample size会抖动，每个duration group的watch-time的经验累计分布（he empirical cumulative distributio）也会逐渐偏离它的真实分布。因此，由于 deconfounding duration的好处，模型效果应首先使用duration-based data-spliting来进行提升；接着，随着f duration groups数目的增长，经验时长分布（empirical watch-time distribution）的estimation error会主宰着模型效果，使它变得很糟。第5节会经验性地使用一系列实验来调整效果变化。

# 5.实验结果

在这一部分，我们提供实证证据来展示我们的方法在真实世界数据和实时实验中的效果。广泛的离线评估表明，我们的方法通过提供更准确的观看时间预测，超越了现有的基线，以至于预测值所引导的排名顺序更接近理想的排名。我们注意到，随着平台的目标是提高用户观看时间，与真实的观看时间值相比，排名在实际推荐视频时通常更受重视。此外，通过将我们的方法整合到短视频平台的推荐系统中，我们发现与替代方案相比，它有效地改善了实时视频消费，这得益于它能够基于优化的观看时间预测生成更好的候选视频排名。

## 5.1 离线评估

我们首先在从真实应用中收集的离线数据上评估我们的方法和其他基线。
特别是，我们感兴趣的是：

- （i）去混杂duration对观看时间预测有何贡献？；
- （ii）duration组的数量如何影响我们模型的性能？

### 5.1.1 数据。

我们使用了从快手App的在线推荐系统收集的生产数据。由于全屏Feed推荐的特性，收集样本中的每个视频都已向用户展示，并与用户观看时间相关联（如果用户立即滚动到下一个视频，则可能接近零）。具体来说，对于图4所示的因果图，我们有：

- 用户表示U：用户即时上下文（如地点、时间和设备）、静止上下文（如果可用，如人口统计信息）以及编码他/她兴趣的历史互动。
- 视频表示V：视频主题信息、相应的视频创作者信息以及与其它用户的先前交互。
- durationD：视频的长度。
- 观看时间W：用户观看的时间。

所有评估的算法共享相同的输入特征。总共，我们有 1,211,885,691 个样本用于训练，134,653,965 个样本用于测试，统计数据总结在表1中。

### 5.1.2 方法。

我们关注以下方法：

- VR（价值回归）。这种方法通过最小化预测值与实际观看时间之间的均方误差损失，直接预测观看时间值。
- WLR（加权逻辑回归）[7]。这种方法拟合一个加权逻辑回归模型，并使用学习到的赔率作为预测的观看时间。由于我们的情况中没有不感兴趣的视频，我们根据观看时间是否超过经验观看时间分布的 \( q60 \) 分位数来确定二元标签，该分布是在所有训练样本上计算的。按照[7]，正样本按观看时间加权，负样本接收单位权重。附录A详细说明了这种方法。
- D2Q（我们的）。如第4.3节所述，这种方法（i）基于duration分割数据；（ii）拟合一个回归模型——其架构如图5(b)所示——通过均方误差损失估计观看时间分位数。然后，预测的分位数被映射到观看时间值域——基于组内的经验观看时间分布——以输出最终的观看时间估计。
- Res-D2Q（我们的）。这种方法通过改进D2Q并按照ResNet的设计将duration纳入模型网络层，进一步利用duration信息。模型架构如图5(c)所示。

所有算法共享相同的网络架构，除了基于分类的算法WLR和分位数预测算法D2Q和Res-D2Q，我们通过Sigmoid函数重新调整输出，使其在[0, 1]范围内；对于Res-D2Q，我们在最后一层添加了一个残差多层感知器（MLP）进行duration调整，以帮助模型区分来自不同duration组的样本。附录B详细说明了网络架构的细节。对于我们这两种去混杂duration的算法D2Q和Res-D2Q，我们改变duration组的数量，范围在[1, 10, 20, 30, 50, 100]之间，以研究其对模型性能的影响。

### 5.1.3 指标。

我们考虑以下性能指标：

- MAE（平均绝对误差），它衡量预测值与真实值之间的平均绝对误差

$$
 MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|, \quad (5)
$$

其中：$ y_i$ 和 $ \hat{y}_i$ 分别是样本i的真实值和预测值。

- XAUC，这是对 AUC（Area Under the Curve）的扩展到密集值。对于一对样本，如果两个视频的预测观看时间值的顺序与真实情况相同，我们给分1，反之则给分0。我们从测试集中均匀地抽取这样的样本对，并将这些分数平均作为XAUC。直观上，XAUC衡量预测观看时间所引起的排名与理想排序的一致性。更大的XAUC值表明模型性能更好。

- XGAUC，这是按用户计算 XAUC 然后按用户样本大小成比例加权平均分数。更大的 XGAUC 值表明模型性能更好。

我们指出，像 XAUC 和 XGAUC 这样的与排名顺序相关的指标在实际应用中通常比 MAE 测量的绝对值精度更受重视，因为平台是根据预测值的排名生成推荐。

### 5.1.4 结果-I：整体性能。

表2显示了不同方法在不同duration组数下的性能。请注意，VR（视频排名）和WLR（加权线性回归）没有数据分割，因此我们将它们的结果呈现在组数等于一的行中；当只有一个组时，Res-D2Q（调整后的D2Q）等同于D2Q，因为所有样本共享相同的duration调整，因此我们在那里省略了Res-D2Q的结果。我们的方法D2Q和Res-D2Q在30个duration组的情况下，在所有指标XAUC、XGAUC和MAE上达到了性能的峰值。特别是，通过在模型架构中进一步利用duration信息，Res-D2Q能更好地区分不同duration组的样本，因此在大多数情况下优于D2Q。当没有数据分割时，D2Q（直接拟合观看时间分位数）与LR（拟合观看时间值）的性能相当。然而，一旦数据根据duration进行分割，D2Q在任何实验的duration组数下都比LR生成更准确的预测，这证实了我们按duration分割数据以消除duration影响的有效性。

### 5.1.5 结果-II：duration组数的影响。

图6绘制了我们的方法D2Q和Res-D2Q在不同duration组数下的XGAUC值。当没有数据分割时，这两种方法彼此等效。一旦数据被分割以消除duration的影响，通过改进的网络架构和duration信息，Res-D2Q优于D2Q。随着duration组数的增加，性能首先提高，这是通过数据分割消除duration影响的优点，然后随着样本大小的减少，由于经验观看时间分布估计误差的增加，性能开始下降。这种观察与第4.3节中的讨论一致。

略

# 

- 1.[https://arxiv.org/pdf/2306.01720.pdf](https://arxiv.org/pdf/2306.01720.pdf)