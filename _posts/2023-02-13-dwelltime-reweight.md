---
layout: post
title:  Dwell Time Modeling介绍
description: 
modified: 2022-06-04
tags: 
---


wechat在《Reweighting Clicks with Dwell Time in Recommendation》中提出了一种基于停留时长加权的建模：

# 2.模型设计与分析

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/faf185aa65c3b51289631f1a4ad89fcd104cf779b168e2fe151ea869e1d49d17883f103a9a65b59c0e71d2578b67e756?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 

## 2.1 Dwell Time Modeling

用户真正需要什么样的推荐？最近研究表明：对于CTR，停留时长（dweill time）更会影响用户的真实满意度。然而，直接对原始的dwell time进行最优化会导致模型过度增强具有long total duration的items，使得重度用户和long items会主宰模型训练。

我们相信：用户使用推荐系统的中心诉求是获得信息。因此，我们会返回：在dwell time、信息增益、用户偏好间的关系本质，并做出以下猜想：

- **(A1) 对于不同的items和users交互，具有相同dwell time，则正信号是相当的**。因为他们通常表示会具有相同的time cost，对每个人公平
- **(A2) 用户需要一个最小的dwell time，以便从items获得信息**。太短的dwell time表示着非常少的收益
- **(A3) 当前dwell time足够长时，信息增益（information gain）会逐渐随着dwell time的增加而递减**。

基于这些，我们在click reweighting中使用一个normalized dwell time function作为一个更好的监督信号来定义有效阅读（valid read）.

## 2.2 有效阅读选择（valid read selection）

有效阅读是高质量点击行为，可以更好反映用户的真实偏好，它可以通过dwell time来天然选择。对于dwell time的一个更深理解，我们会绘制**点击数随不同log dwell time的趋势**。图2左可以看到：

- 1) 总体上，我们可以假设：**log dwell time具有一个近似高斯分布**，例如： $$lnT=\mu + \sigma \epsilon$$，其中：T是一个random dwell time，$$\epsilon \sim N(0,1)$$。
- 2) 我们会将$$[\mu-\sigma, \mu + \sigma]$$看成是主要的dwell time区间
- 3) 接近19%的点击行为要短于15s dwell time，并且接近15%的点击行为要长于200s dwell time

根据上述假设A2和A3，具有过短和过长的dwell time的点击行为在click reweighting上会被降级。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ad72d479603dc8f4ff46ea0f301b27e709f1dde356b560bf4a08495e394ebf6b630b57ad306090bec4a4623d3addfb43?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2 log dwell time在我们系统中的趋势（左）以及normalized dwell time的趋势（右）

简单做法是：直接设置一个**共享的dwell time阈值**来收集有效阅读。然而，**简单依赖阈值来定义有效阅读，会不可避免地忽略掉关于轻度用户与short items的大量行为信息**。因此，我们定义了**三种类型的user-item clicks作为我们的合法阅读行为**：

- T1: dwell time要比长于$$x_l$$秒
- T2: 该用户在最近一周近至少点击7个items
- T3: dwell time会长于该item的历史dwell time记录的10%（例如：长于分位数P10）

(1) 第一种类型：根据common-sense阈值$$x_l$$构建了有效阅读的基础规则。**我们将$$lnT$$的$$x_l = exp(\mu - \sigma)$$看成是有效阅读的共享dwell time阈值**，它可以适配于不同的推荐系统。在我们的系统中，$$exp(\mu - \sigma)$$接近15s。19%的点击行为会被T1过滤掉。出于简单性，对于所有的users和items，我们根据time costs的绝对值直接采用一个共享的DT阈值，对于不同的user或item groups设置定制的dwell time阈值也很方便。

（2）第二种类型：会在轻度用户上打个补丁，**将所有轻度用户的点击行为看成是在训练中的监督信号，因为他们的行为很稀少**。我们希望避免长尾轻度用户（偏向于扫描浏览而非深度阅读）的重要信息丢失。

(3) 第三种类型：**会在一个指定item上考虑相对dwell time，在相同item上所有历史点击间，取回的click具有一个相对符合条件的dwell time（top 90%）**。通过该方法，我们的有效阅读会考虑上具有天然短长度、少dwell time的items（例如：news或short videos）. 为了避免噪声，我们进一步将清除所有具有5s dwell time的点击，以确保有效阅读的最低能力。在我们的实践中，T1、T2、T3类型分别具有89.9%、2.9%、7.2%的总有效阅读。只有有效阅读会被用于训练中的监督信号。

## 2.3 归一化dwell time函数

有效阅读选择会作为一个pre-filter使用。然而，我们仍然面临着在click reweighting中如何精准定义不同dwell time值的挑战。直觉上，**相同的dwell time提升，当current dwell time更短时对于一个click的quality会具有更大的贡献（例如：[1s -> 15s]要比[601s->615s]更大）**。太长的dwell time会带来疲乏，对用户体验有害。因而，大量工作采用log dwell time的MSE作为训练loss作为dwell time的预估【2，16，27】。

不同于常规模型，我们会将有效阅读定义成高质量的supervised label，并且希望提升有效阅读的数目的比例。因此，我们的dwell time function会拥有以下两种特性C1和C2，分别对应于以上的A2和A3:

- C1: **设计好的dwell time function曲线在early stage应该很陡**，此时具有大梯度（特别是接近有效阅读阈值 $$exp(\mu - \sigma)$$的地方），这会指导模型很好区分有效阅读 vs. 无效点击
- C2: **dwell time funciton曲线在dwell time过长时会比较平，避免过长的items得到太多rewards**，导致伤害轻度用户对短items的交叉。

根据以下规则，我们基于原始的dwell time T，使用一个sigmoid function，设计了normalized dwell time $$T_N$$：

$$
T_N = \frac{A}{1 + exp(- \frac{T-offset}{\tau})} - B
$$

...(1)

图2（右）展示了$$T_N$$的趋势。对比起log dwell time，$$T_N$$会使用设置好的rates单调增加，其中：offset和$$\tau$$本质上是满足C1和C2的参数。

- offset：决定了具有最大梯度的dwell time point。对于C1，我们会设置：$$offset = exp(\mu - \sigma)$$来使得normalized dwell time在有效阅读/无效阅读边界上具有最大的梯度，它会基于supervised training与有效阅读很好的一起协作。
- $$\tau$$：定义了dwell time曲线的sharpness。对于C2，我们定义了一个upper阈值 $$x_h$$作为$$exp(\mu + \sigma)$$，假设：比$$x_h$$更大的dwell time T对$$T_N$$没啥贡献（例如：$$T_N$$提升$$x_h \rightarrow T$$要小于最小精度，例如：在系统中为1e-5）。$$\tau$$被设置成满合$$x_h$$的上述假设。
- A和B：是超参数，可以将$$T_N$$归一化成$$[0, T_{max}]$$，**其中：$$T_{max}$$是当前在线dwell time模型的最大dwell time值**。我们将normalized dwell time范围保持不变，减少可能的不匹配问题。

最终，基于上述讨论，我们设置：$$offset=15, \tau = 20, A = 2.319, B = 0.744$$来满足C1和C2 。 我们也对这些参数做了grid search，发现当前setting可以达到最好的在线效果。

## 2.4 Click Reweighting

有效阅读和normalized dwell time被设置成过滤噪声，选出符合点击的分位数，以便更好的进行学习。在click reweighting中，我们采用一个multi-task learning框架来进行有效阅读预估（valid read prediction）以及**加权有效阅读预估（weighted valid read prediction）**。特别的，我们会进行一个共享bottom来跨任务共享原始的user/item features。

对于valid read tower，我们会采用一个3- layer MLP，它会采用原始user/item features $$f_u, f_{d_i}$$作为inputs，并输出用户u在item $$d_i$$上的预估点击概率$$P_{u, d_i}$$。接着，有效阅读loss $$L_v$$定义如下：

$$
L_v = - \sum\limits_{(u,d_i) \in S_p} log P_{u,d_i} + \sum\limits_{(u,d_j) \in S_n} log (1 - P_{u,d_j})
$$

...(2)

其中：

- $$S_p$$和$$S_n$$表示正样本集（有效阅读）和负样本集（无效点击和未点击）。

相似的对于weighted valid read tower，**我们直接使用normalized dwell time $$T_N^{u, d_i}$$作为每个$$(u, d_i)$$的weight**。另一个3-layer MLP会被用来输出预估点击概率$$P_{u,d_i}'$$。加权有效阅读tower接着会在loss $$L_w$$下被训练：

$$
L_w = \sum\limits_{(u,d_j) \in S_n} T_N^{(u,d_j)} log(1 - P_{u,d_j}') - \sum\limits_{(u,d_i) \in S_p} T_N^{(u,d_i)} log P_{u,d_i}'
$$

...(3)

$$L_v$$和$$L_w$$是线性组合成最终loss：$$L = L_v + L_w$$。在线部署中，两个towers的求和的预估得分会被用于在线排序。终合考虑original和DT weighted有效阅读预估任务是有益的。再者，我们进一步探索了MTL框架（MMoE和PLE），在线提升并不大。可能是因为dwell time与点击高度相关。出于简单，我们直接使用MLP作为shard bottom。

- 1.[https://arxiv.53yu.com/pdf/2209.09000.pdf](https://arxiv.53yu.com/pdf/2209.09000.pdf)