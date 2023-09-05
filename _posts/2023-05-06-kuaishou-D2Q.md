---
layout: post
title: kuaishou D2Q介绍
description: 
modified: 2023-05-06
tags: 
---

kuaishou在《Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation》提出了D2Q的模型。

# 摘要

# Watch-Time prediction的因果模型

我们的目标是：当推荐一个视频给某用户时，预估该用户在的watch time。我们会通过一个因果关系图（causal graph）进行公式化：它会将user、video、duration、watch-time、以及推荐系统在watch-time prediction和视频曝光上关于duration的混杂效应（confounding effect），如图4(a)所示：


<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/e25509291ef28f0ce491abd8dcd1149b636e14733fbb4a8eaac0b213279b26f5b75d01bab9101400bb63055145123144?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=4.jpg&amp;size=750">

图4 watch-time prediction的因果关系图：U-user、V-video、D-duration、W-watch time。图(a)建模了在视频曝光和watch-time prediction上的confounding effect。图(b)使用backdoor adjustment来 deconfound duration，并移除它在视频上的effect。

- U：表示user representation，包含了：用户人口统计学（user demographics）、即时上下文（instantaneous context）、历史交互等
- V：表示video representation，包含了：video topics等
- **D：表示video duration，例如：视频长度**
- W：表示用户花费在观看视频上的时间
- $$\lbrace U, V \rbrace \rightarrow W$$：会捕获在watch-time上的interest effect，它可以衡量用户对该视频有多感兴趣
- $$D \rightarrow W$$：会**捕获在watch time上的duration effect**，它会建议：当两个视频与用户兴趣相匹配时，**更长的视频会接受到更长的watch time**
- $$D \rightarrow V$$：表示**duration会影响视频的曝光**。推荐系统经常会对具有更长duration的视频有不平等的偏好；这样的bias会通过feedback loop会放大，如图3所示。另外，duration会影响模型训练，因为：i) sample size随duration的不同而不同，具有长在uration的视频通常具有更大的sample size，这意味着 prediction模型具有更好的performance； ii) 在标准模型（比如：WLR）中，具有不同duraiton的videos会接受到不同sample weights，（它会影响在模型训练时的梯度分配）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/6dec8ff2f8a99cdd0a4c4b63e552d0fe2930b4757a9ee367024a3e1f0899cd6f76196df58a1647cc674aa1aa0327c5af?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3 在11个月上，kuaishou APP每个video durtion相应的视频曝光变化。bins以duration的升序进行排序。bin的高度表示在该周期内曝光的差异。出于置信原因，绝对值会被忽略。平台的目标是提升watch time，**曝光会偏向于那些具有长duration的视频**。

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
E[W \mid do(U,V)] & overset{i}{=} E_{G_1} [W | U,V] \\
& overset{ii}{=} \sum\limits_d P_{G_1} (D = d | U, V) E_{G_1} [W | U,V,D = d] \\
& overset{iii}{=} \sum\limits_d P(D=d) E[W| U,V,D = d]
$$

...(1)

其中(i)是总期望；(ii)是因为D独立于$$\lbrace U,V \rbrace$$，干预会移除在graph $$G_1$$中的边$$D \rightarrow V$$；(iii)是因为：这样的干预不会改变W在条件{U,V,D}上的W分布，D的间隔分布仍会相同。

等式（1）阐明了deconfound duration的设计：你可以独立估计$$P(D)$$和$$E[W \mid U,V,D]$$，接着将他们组合在一起来构建最终的estimation。在本paper中，我们提出将duration分布P(D)离散化成不相交的groups，并拟合group-wise watch-time预估模型$$E[W \mid U,V,D]$$来完成估计。

## 4.2 基于Duration分位数的Data-Splitting

我们现在呈现出一个通用框架使用duration deconfounded来估计watch-time，如图4(b)所描述。更高层的思路是：将数据基于duration进行划分，并构建group-wise watch-time estimation以便在视频曝光上对duration进行debiase。

特别的，为了阻止 边D -> V，我们基于duration分位数将训练样本进行划分成M个相等的部分，它可以将分布P(D)离散化成不相交的部分。假设：$$\lbrace D_k \rbrace_{k=1}^M$$是这些duration groups。继续(1)中的派生，我们通过下面近似来估计deconfounded model $$E[W \mid do(U,V)]$$：

$$
E[W \mid do(U,V)] = \sum\limits_d P(D = d) E[W | U,V,D = d] \approx \sum\limits_{k=1}^M 1 \lbrace d\in D_k \rbrace E[W | U,V,D \in D_k] = \sum\limits_{k=1}^M 1\lbrace d \in D_k \rbrace f_k (U, V)
$$

...(2)

这里我们提供了一个关于“为什么这样的基于duration的数据划分过程，可以解缓图4(a)中边D->V的bias问题”的直觉解释。在标准的watch-time预估模型（如：WLR）中，具有长watch-time weights的样本会在梯度更新中采样更多，因而预估模型经常在短watch-time的样本上表现很差。Watch-time是与duration高度与duration相关的，如图2所示。通过基于duration进行数据划分，并将模型以group-wise方式拟合，我们可以在模型训练期间，缓和那些具有长watch-time的样本、以及具有短watch-time的样本的interference。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/3d37cd784136be40da62045d7a83f6289f84ef1468209a5631948dd77067b2ea4cac9afcb5538d984295c9f579b36ef6?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2

然而，这样的data-splitting方法会抛出另一个问题。如果对于每个duration group $$D_k$$我们拟合一个单独的watch-time prediction模型$$f_k$$（如图5(a)所示），model size会变得更大，这在真实生产系统中是不实际的。但如果我们允许在duration groups间进行参数共享，使用原始watch-time labels进行拟合等价于没有data-splitting的学习，这在duration deconfounding上会失败。下面部分会解释：如何通过将原始watch-time labels转换成duration-dependent watch-time labels来解决该窘境，并允许我们同时移险duration bias，并维持模型参数的单个集合来获得可扩展性。

## 4.3 每个Duration Group估计Watch-time

接着，我们描述了，如何使用来自所有duration groups的数据来拟合单个watch-time prediction模型。回顾下我们的设计有二部分：

- i) duration debiasing
- ii) 参数共享

问题的关键是：将watch-time label转换成duration-dependent，通过根据不同duration group拟合watch-time分位数来实现，而非原始values。我们会引入Duration-Deconfounded
Quantile-based (D2Q) watch time预估框架。

$$\hat{\phi}_k (w)$$表示在duration group $$D_k$$中关于watch-time的期望累计分布（empirical cumulative distribution）。给定一个user-video pair (u,v)，D2Q方法会在相应的duration group中预估它的watch-time分位数，接着使用$$\hat{\phi_k}$$将它映射到watch time的值域中（value domain）。也就是说：

$$
f_k(u, v) = \hat{\phi_k^{-1}} (h(u, v))
$$

...(3)

其中，h是一个watch-time分位数预估模型，它会拟合在所有duraiton groups上的数据：

$$
h = underset{h'}{argmin} \sum\limits_{\lbrace (u_i, v_i, w_i)\rbrace_{i=1}^n} (h'(u_i, v_i) - \hat{\phi}_{k_i}(w_i))^2
$$

...(4)

其中：

- $$k_i$$是样本i的duration group，以便$$d_i \in D_{k_i}$$。

你可以应用任意现成的regression模型来拟合分位数预估模型h，并维护在所有duration groups间单个模型参数集。接着，在inference阶段，当一个新的user-video pair $$(u_0, v_0)$$到达时，模型会首先发现：视频$$v_0$$会属于哪个duration group $$D_{k_0}$$，接着将watch-time  quantile预估$$h(u_0, v_0)$$映射到watch-time值$$\hat{\phi_{k_0}^{-1}}(h(u_0, v_0))$$上。我们会在算法1和算法2上总结learning和inference过程。

在该方式下，D2Q会拟合那些是duration-dependent的labels。我们注意到：video duration会是model input的一部分，会将来自不同duration groups的不同样本进行输入，如图5(b)所示。另外，来自不同duration groups的样本会共享关于watch-time quantile的相同的label，但具有不同的特性——一个模型在跨groups学习watch-time quantile时会失败。为了完全利用duration information，你可以在模型结构中额外包含一个duration adjustment tower（比如：ResNet），我们在图5(c)中将它称为Res-D2Q。第5节演示了Res-D2Q会在D2Q之上进一步提升watch-time prediction accuracy。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ae269d16eae3cb49eed1137bb4f04b905d548e9324c33ebb811d25d7097f983a1ddbe4267deb35009ecbb45a01e22027?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=5.jpg&amp;size=750">

图5

对应于duration的watch-time labels的转换，允许在跨duration groups间同时进行 deconfounding duration bias和 parameter sharing。然而，随着duration groups的数目增加，group sample size会抖动，每个duration group的watch-time的期望累计分布（he empirical cumulative distributio）也会逐渐偏离它的真实分布。因此，由于 deconfounding duration的好处，模型效果应首先使用duration-based data-spliting来进行提升；接着，随着f duration groups数目的增长，
empirical watch-time distribution的estimation error会主宰着模型效果，使它变得很糟。第5节会经验性地使用一系列实验来调整效果变化。

# 5.实验

略






# 

- 1.[https://arxiv.org/pdf/2306.01720.pdf](https://arxiv.org/pdf/2306.01720.pdf)