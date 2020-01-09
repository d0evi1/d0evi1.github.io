---
layout: post
title: NNLM介绍
description: 
modified: 2016-01-10
tags: [word2vec]
---


SM McNee等人在2006年《Being Accurate is Not Enough:
How Accuracy Metrics have hurt
Recommender Systems》中提出accurate是不够的：

# 1.介绍

假设你正使用一个关于旅游的推荐系统。**假设它给出的所有推荐都是你已经旅行过的地方**，即使该系统在ranking上做的很好（所有的地方都根据你的访问偏好排序），这仍是一个很差的推荐系统。你会使用这样的系统吗？

不幸的是，我们当前就是这样测试我们的推荐系统的。**在标准方法中，旅游推荐器会对推荐新地方进行惩罚（而非对已经访问过的地方进行惩罚）**。当前的accuracy指标（比如：MAE），会将算法的预测与用户对item的评分进行比较来衡量推荐算法效果。使用这些指标的最常用技术是：留一法（leave-n-out方法）。本质上：**我们会对推荐用户已经访问过的地方进行奖励（reward），而非对用户将要访问的新地方进行奖励**。

# 2.Similarity

通常，**推荐列表中会包含相似的items**。

Accuracy指标不会看到该问题，因为它们被设计用来判断单个item预测的accuracy；**他们不会判断整个推荐列表的内容**。不幸的是，用户是与这样的列表进行交互的。所有推荐是在当前推荐列表、以及之前用户已经看过的列表的上下文中做出的。该推荐列表应作为一个整体进行评判，而非作为各个单独items的一个集合。

解决该问题的一个方法是：[Ziegler 2005]：推荐列表的Intra-List Similarity Metric和Topic Diversification。返回的列表可以进行变更（增加/减小diversity）。结果表明，**变更后的列表在accuracy上更差，但用户更喜欢变更后的列表**。

依赖于用户的意图，出现在该list中的items组合，对用户对推荐的满意度的影响要超过item accuracy的变更。

# 3.惊喜性（Serendipity）

推荐系统中的Serendipity指的是：**接收到一个意料之外的item推荐**。**与Serendipity相关的心理反应很难在指标上捕获**。但如果去掉该构成，该概念的unexpectedness部分——（即：收到推荐的新颖性novelty）——仍然很难measure。该概念的反面即：ratability(可估的），则很容易measure。可以使用leave-n-out方法进行measure。

我们可以将一个item的ratability定义为：**在已知user profile的情况下，该item作为用户将消费的next item的概率**。从机器学习的角度，具有最高ratability的item会成为next item。因此，推荐器算法在accuracy metrics打分良好。

隐式假设是：一个用户总是会对具有最高ratability的items感兴趣。而**该假设在分类问题中是true的，但在推荐系统(recomenders)中并不总是true的**。用户通常会对那些不可预料的items的推荐进行judge。例如，一个在线音乐商店的推荐系统，会使用一个User-User CF算法。最常见的推荐是： the Beatle乐队的“White Album”。从accuracy的角度，这些推荐是十分准确的（dead-on）：大多数用户非常喜欢该专辑。**但从有用性(usefulness)的角度，该推荐是完全失败的：每个用户已经拥有了“White Album”、或者明确表示不买它**。尽管它的估计分非常高，但“White Album”推荐几乎从不会被用户播放，因为它们几乎没价值。

之前的许多研究【[McNee 2002, Torres 2004】表明：推荐算法会生成相互间不同品质的推荐列表。用户偏向于来自不同的推荐算法的列表。用户会选择不同词汇（words）来描述推荐（比如：User-based CF被认为是生成“novel”的推荐）

这建议我们需要其它方法来对推荐算法进行分类。**而没有用户的反馈，“serendipity metric”可能很难被创建**，评判许多算法的其它指标会提供关于推荐算法不同之处的一个更详细的画面。

# 3.用户体验和期望

用户满意度并不总是与推荐的accuracy相关。还有许多重要的因素必须考虑。

在推荐器中，**新用户与老用户间有不同的需求**。新用户可能会从一个生成高概率items的算法中受益，因为在采用推荐器之前他们必须与推荐器确立信任和交往。之前的工作表明，对于新用户的算法选择，会极大影响用户体验和推荐系统生成它们的accuracy。

我们之前的工作表明，**不同的语言和文化背影会影响着用户满意度[Torres 2004]**。一个在母语用户环境下的推荐器要比使用其它语言的推荐器更受喜欢，即使被推荐的items本身是使用另一种语言的（比如：一个葡萄牙语的研究paper推荐器会推荐使用英语的papers）

## 4.往前看

accuracy metrics可以极大帮助推荐系统；他们给我们提供了一种可比较算法的方式，并能创建健壮的实验设计。我们不应停止使用它们。但我们也不能只使用它们来判断推荐器。现在，我们需要思考如何与推荐系统的用户更贴近。他们不关心使用一个具有在某指标上更高得分的算法，他们想要一个有意义的推荐。有许多方法可以做到。

首先，我们需要评判：当用户看到推荐时的质量：推荐列表。为了这样做，我们需要创建一些在推荐列表上的指标，而非出现在一个list上的items。已经存在一些指标，比如：Intra-List Similarity metric，但我们需要更多指标来理解这些lists的其它方面。

第二，我们需要理解在推荐算法间的不同之处，并能以ratability之外的方式来进行measure。用户可以告诉在推荐算法间的不同。例如，当我们更改在MovieLens电影推荐器上的算法，我们会接收到许多来自用户的emails，它们会有疑惑：为什么MovieLens的推荐会变得如引“守旧（conservative）”。在MAE measures上的得分良好，但对用户来说却很明显不同。

最后，用户回复推荐会超过一段时间，从新用户到已体验用户会增长。每当他们来到系统时，他们具有一些理由过来：他们有一个诉求（purpose）。我们需要判断我们为每个用户生成的推荐是否能满足它们的需要。到到我们认识与用户的这种关系之前，recommenders会继续生成不匹配的推荐。

最后，recommenders的存在可以帮助用户。作为一个社区，我们已经创建了许多算法和方法来研究recommender算法。现在也开始从用户为中心的角度来研究recommenders，而不再仅仅是accurate的角度。

参考：

[https://dl.acm.org/citation.cfm?id=1125659](https://dl.acm.org/citation.cfm?id=1125659)
