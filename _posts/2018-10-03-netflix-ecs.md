---
layout: post
title: ALE atari介绍
description: 
modified: 2018-09-03
tags: 
---

netflix在《The Netflix Recommender System: Algorithms, Business Value, and Innovation》中，提到了一个指标：ECS（EFFECTIVE CATALOG SIZE）。我们来看下它的实现：

# EFFECTIVE CATALOG SIZE

假设我们在视频库（catalog）中具有N个items，它们根据在线观看时长（hours streamed）从最流行到最不流行进行排序，表示成$$v_1, \cdots, v_N$$。假设 vector $$p=[p_1, \cdots, p_N]$$表示概率质量函数( probability mass function (p.m.f.))，对应于来自在catalog中按流行度排序的视频的时间流的share，也就是说，$$p_i$$是所有（hours streamed）的share，它来自于第i个最流行的流视频 $$v_i$$。注意，对于$$i=1, \cdots, N-1$$以及$$\sum_{i=1}^N p_i=1$$来说，$$p_i \geq p_{i+1}$$。我们寻找这样一个metric：它是关于p作为参数、输出在范围[1, N]内的一个函数，在某种程度上告诉我们，需要有多少视频来解释一个典型的hour streamed。如果最流行视频$$v_1$$占据着大多数hours streamed，该metric应返回一个略高于1的值；如果catalog中的所有视频具有相同的流量，则返回一个N值。这样的一个metric称为effective catalog size（ECS），它的定义如下：

$$
ECS(p) = 2(\sum\limits_{i=1}^N p_i i) - 1
$$

...(1)

等式(1)会简单计算在p.m.f.  p下视频索引（video index）的平均，并将它重新缩放（rescale）到合理区间上。很容易确认，对于所有的i，当$$p_1=1$$时，ECS具有一个最小值1；当$$p_i = 1/N$$时具有一个最大值N。

ECS可以被应用到任意p.m.f.上。我们可以计算一个索引（refenerce）开始，对于该p.m.f的ECS只会考虑最流行的k个视频的hours，随着我们从1到N递增k。特别的，我们定义了$$p(k) = \alpha [p_1, \cdots, p_k]$$，其中，$$\alpha = 1/(\sum\limits_{i=1}^k p_i)$$是一个归一化常数，并绘制了ECS(p(k))来区分不同的k，得到如图4所示的黑线。该线位于identity line(没显示)之下，因为并不是所有视频都具有相同的流行度。在同一图中的红线是使用ECS等式到一个不同的p.m.f q(k)上的结果，k从1到N。p.m.f. q(k)是来自每个关于k的PVR rank的share of hours，或者来自top k PVR ranks的所有streamed hours之外的。为了形成q(k)，对于我们的每个会员（members），我们采用k个最高ranked PVR videos，来寻找由这些member-video pairs生成的所有streaming hours，并定义了它的第i个entry作为这些来自PVR rank i的streaming hours的share。注意，尽管对于每个member q(k)和p(k)一样只包含了k个videos，跨members的一个抽样，更多videos（可能为N）会出现，因为PVR是个性化的。PVR rank对应于跨所有播放（plays）的中位数rank（median rank），effective catalog size是4倍于unpresonalized effective catalog size。

# 

effective catalog size(ECS)是一个这样的metric，它描述在我们的catalog中，跨items的扩展观看（spread viewing）的程度。如果大多数viewing来自于单个视频，它会接近于1。如果所有视频会生成相同量的viewing，ECS会接近于在catalog中的视频数。否则，它介于两者之间。ECS的描述见上一节。

如果没有个性化，所有用户（members）会接收到相同的视频推荐。图4左侧的黑线表明，没有个性化的ECS是如何随着数据中视频数的增长而增长的，从最流行的视频开始，随着x轴向右移动添加下一个流行(next popular)的视频。另一方面，相同图中的红色，展示了ECS是如何增长的，它是一个关于用来进个性化的PVR ranks数目的函数（而非一个关于包含视频数的函数）。尽管是否进行个性化的catalog exploration的量不同之处很显著，但它还不够令人信服。毕竟，我们可以通过对每个session提供完全随机的推荐来进行扩展观看（spread viewing）。

更重要的，个性化允许我们极大增加推荐的成功率。达到该目标的一个metric是take-rate：产生一个播放所提供的推荐比例。图4右侧展示了take-rate，一个是关于视频流行度的函数，另一个是video PVR rank的函数。我们从推荐中获得的在take-rate上的提升是大幅度的。但是，更重要的是，当推荐被正确生产和使用时，会产生在产品（比如：streaming hours）上整体engagement上的大幅提升，以及更低的订阅取消率。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/88ee88b07b6c03258bb4a5d3c9f0831ed773df69b932dab3dc8cda970c00cef9a091af0af7bef48a1666d889c942f140?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图4


# 参考

- 1.[https://dl.acm.org/doi/pdf/10.1145/2843948](https://dl.acm.org/doi/pdf/10.1145/2843948)