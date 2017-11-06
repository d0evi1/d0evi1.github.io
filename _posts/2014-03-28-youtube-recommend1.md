---
layout: post
title: 2010 youtube推荐系统 
modified: 2014-03-28
tags: [大数据]
---

本文是youtube在2010年提出的系统，现在回过头去看看它在当时是如何实现youtube的推荐系统的。

# 系统设计

推荐系统的整个设计都围绕以下的目标：我们希望相应的推荐(recommendations)是**适度最新的（recent）、新鲜的（fresh），多样化的（diverse），并且与用户最近的动作有相关性（relevant）**。另外，重要的一点是，**用户能理解为什么某个视频会被推荐给他们**。

推荐的视频集合通过使用一个用户的个性化行为（观看，收藏，喜欢）来生成作为**种子**，然后通过基于视频图的co-visitation对集合进行**扩展**。该视频集接着使用多个信号为相关性(relevance)和多样性(diversity)进行**排序**。

从工程的角度看，我们希望系统的单个组件相互解耦，允许它们进行独立的调试，以便容错，降低系统的整体复杂度。

## 1.输入数据

在个性化视频推荐的生成期间，我们会考虑多个数据源。总之，有两大类的数据要考虑：1)内容数据（content data），比如原始视频流和视频元信息（标题，简介等）,2)用户行为数据（user activity data），可以进一步分类为显式和隐式。显式行为包括：对一个视频进行打分（rating），进行收藏/喜欢，或者订阅了某个上传者。隐式行为包括：观看、交互，比如：用户开始观看一个视频，用户观察了某个视频的大部分（长播放行为：long watch）

在所有的case中，数据的处理noisy很多：视频元信息可以不存在，不完整，过期，或者不正确；用户数据只捕获了在网站端的一部分用户行为，只能间接衡量一个用户的参与度（engagement）和高兴度（happiness），比如：**用户完整地观看一个视频，不足够说明她确实喜欢这个视频**。视频的长度和用户的参考程度，受信号质量的影响。再者，隐式行为数据是异步生成的，可能不完整，比如：在我们接受到一个long-watch通知前，用户可能已经关闭了浏览器。

## 2.相关视频

推荐系统的一个构建块是：构造一个映射（mapping），将一个视频\$ v_i \$映射到一个相似（similar）或相关（related）的视频集合\$ R_i \$上。在该上下文上，我们定义了相似视频来作为：在观看了给定的种子视频v(seed video)后，一个用户会喜欢继续观看这些视频集。为了计算该mapping，我们使用了有名的技术：关联规则挖掘（association rule mining）、或者 co-visitation counts。让我们来看下：用户在网站上观看行为的session。对于给定的时间周期（24小时），我们会统计每对视频对(video pair）：\$(v_i,v_j)\$，在session内被同时观看(co-watch)的次数。将该co-visitation count置为\$ c_{ij} \$，我们定义了视频\$ v_j \$基于视频\$ v_i\$的相关得分：

$$
r(v_i,v_j)= \frac{c_{ij}}{f(v_i,v_j)}
$$

...(1)

其中，\$c_i\$和\$ c_j \$是所有session上视频\$v_i\$和\$ v_j\$各自的总共现次数。\$ f(v_i,v_j)\$是一个归一化函数，它会考虑种子视频和候选视频的“全局流行度(global popularity)“。一种最简单的归一化函数是，简单地除以视频的全局流行度的乘积：\$f(v_i,v_j)=c_i \cdot c_j \$。也可以选择另一种归一化函数。见paper[6]。当使用候选的简单乘积进行归一化时，\$c_i\$对于所有的候选相关视频是相似的，可以在我们的设置(setting)中忽略。这本质上会在流行视频中支持更低流行度的视频。

对于一个给定的种子视频\$v_i\$，我们接着选取相关视频\$R_i\$的集合，通过它们的得分\$r(v_i,v_j)\$进行排序，选取topN个候选视频。注意：除了只选取topN个视频外，我们也加入一个最低的得分阀值（minimum score threshold）。因而，对于许多视频，我们不能计算一个可靠的相关视频集合，因为它们整体的观看量（view count：与其它视频的co-visitation counts）很低。

注意，这是一个简化版的描述。实际上，还存在额外的问题需要去解决——表示偏差（presentation bias），噪声观看数据（noisy watch data），等————在co-visitation counts之外的额外数据源，也可以被使用：视频播放的sequence和time stamp、视频元信息等等。

相关视频可以被看成是：在视频集上引导成一个直连图（directed graph）：对于每个视频对\$(v_i,v_j)\$，从\$v_i\$到\$v_j\$上有一条边（edge） \$ e_{ij} \$，如果\$v_j \in R_i \$，该边的权重由等式(1)给定。

## 3 生成推荐候选

为了计算个性化推荐，我们将相关视频的关联规则，以及一个用户在网站上的个人行为相结合：它包括被观看的视频(超过某个固定阀值)，以及显式收藏(favorited)、喜欢(liked)、打分(rated)、添加到播放列表(added to playlists)的视频。我们将这些视频集合称为种子集（seed set）。

对于给定的种子集合S，为了获取候选推荐，我们将它沿着相关视频图的边进行扩展：**对于种子集合里的每个视频\$v_i\$，会考虑它的相关视频\$ R_i \$**。我们将这些相关视频集合的**合集（union）**表示为\$C_1\$：

$$
C_1(S) = \bigcup_{v_i \in S} R_i
$$

...(2)

在许多情况下，计算\$C_1\$对于生成一个候选推荐集合是足够的，它足够大并且足够多样化来生成有趣的推荐。然而，**实际上任何视频的相关视频的范围都趋近于狭窄（narrow），经常会突显(highlighting)出那些与种子视频相似的视频**。这会导致相当狭窄的推荐（narrow recommendation），它确实会让推荐内容与用户兴趣接近，但对于推荐新视频给用户时会很失败。

为了扩大推荐的范围，我们通过在相关视频图上采用一个**有限的传递闭包（limited transitive closure）**，来扩展候选集。**\$C_n\$被定义成这样的视频集合，它从种子集合中的任意视频在距离n内可达**：

$$
C_n(S) = \bigcup_{v_i \in C_{n-1}} R_i
$$

...(3)

其中\$ C_0=S \$是该递归定义中的base case（注意：它会为\$C_1\$产生一个与等式(2)的同等定义）。最终的候选集合\$C_{final}\$接着被定义成：

$$
C_{final}=(\bigcup_{i=0}^{N} C_i) \\ S
$$

...(4)

由于相关视频图的高分枝因子（high branching factor），我们发现，在一个较小的距离上扩展，可以产生一个更宽更多样的推荐集合，即使用户只有一个小的种子集合。注意：候选集中的每个视频，与种子集合中一或多个视频有关联。为了进行ranking，我们继续跟踪这些种子与候选的关联，并为推荐给用户提供解释。

## 4.Ranking

在生成阶段，已经产生了候选视频，它们使用许多信号进行打分和排序。这些信息可以根据ranking的三个不同阶段归类成三组：

- 1)视频质量(video quality) 
- 2)用户特征(user specificity)  
- 3)多样化

视频质量信号，是指在不考虑用户的情况下，判断视频被欣赏的似然（likelihood）。这些信息包括：观看量（view count: 一个视频被观看的总次数），视频的评分，评论，收藏，分享行为，上传时间等。

用户特征信号，用于增强一个视频与某个用户特有的品味和偏好相匹配。我们会考虑种子视频在用户观看历史中的属性，比如：观看量（view count）、观看时长（time of watch）。

通过使用这些信号的一个线性组合，我们这些候选视频生成了一个排序列表。因为，我们只会展示少量的推荐（4到60之间），我们必须选择列表的一个子集。这里不用选择最相关的视频，我们会在相关性和跨类目多样性上做一个平衡优化。因为一个用户通常在不同时刻会在多个不同的主题上有兴趣，如果视频相互间太相似在该阶段会被移除，以便增加多样性。一种简单的方式是，限制每个种子视频相关推荐数目，或者限制相似频道（channel/uploader）的推荐个性。更综合的方式是基于主题聚类，或是进行内容分析。

## 5.UI

推荐的表示在整个用户体验上是很重要的一环。图1展示了推荐是如何在youtube主页上进行表示的。有少量新特性需要注意：首先，所有的推荐视频使用一个缩略图（thumbnail）、标题、视频时间、流行度进行展示。这与主页上的其它部分相类似，可以帮助用户快速决定是否对一个视频有兴趣。再者，我们添加了一个带有种子视频链接（它触发了推荐）的解释（explanation）。最后，我们给出了用户控制，可以看到在主题上有多少个推荐。

<img src="http://pic.yupoo.com/wangdren23/GAIut6at/medish.jpg">

在ranking那一节，我们计算了一个推荐的排序列表，但在serving time只会展示其中一个子集。这允许在每次用户到达网站时提供新的、之前未看过的推荐，即使底层的推荐没有被重新计算。

## 6.系统实现

我们选择一种面向批处理的预计算方式（batch-oriented pre-computation approach），而非按需计算。这样做的优点是：推荐生成阶段访问大量数据会使用大量CPU资源，而在serving时预生成的推荐项可以有极低的访问延时。该方法最大的缺点（downside）是，在生成(generating)和(serving)一个特定的推荐数据集间的delay。我们缓和了这个现象，通过对推荐生成进行pipelining，每天更新数据集多次。

youtube推荐系统的实际的实现被分为三个主要部分：1)数据收集 2)推荐生成 3)推荐serving。

之前提到的原始数据信号存放在YouTube的log中。这些log会被处理，提取信号，按每用户为基础保存到BigTable中。当前处理数百万的用户和上百亿的行为事件，总的footprint为TB级别。

推荐的生成通过MapReduce计算完成，它会在user/video graph上进行walk through，来累积和计分推荐项。

生成的数据集的size相当小（GB），可以很容易通过webserver进行serving。完成一个推荐的请求时间几科由网络传输时间决定。

# 评估

通过A/B testing进行在线评估，真实流量分成不同的组，其中一组作为control或baseline，其它组给新特性、新数据、或新UI。两组接着进行对比。为了评估推荐质量，我们使用不同metrics的组合。主要的metrics包括CTR，long CTR（只统计点击导致观看一个视频的大部分观看行为），session length，首次long watch的时间（time until first long watch），推荐覆盖率（）。我们使用这些metrics来跟踪系统的效果。

<img src="http://pic.yupoo.com/wangdren23/GAHYXhpA/medish.jpg">


[The YouTube Video Recommendation System](https://pdfs.semanticscholar.org/e7d5/3f538f5239739d1f943c81d17e4a167c65c6.pdf)