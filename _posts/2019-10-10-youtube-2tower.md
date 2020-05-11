---
layout: post
title: youtube 双塔模型
description: 
modified: 2019-10-11
tags: 
---


youtube在2019发布了它的双塔模型《Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations》：

# 介绍

在许多服务上（视频推荐、app推荐、在线广告定向），推荐系统帮助用户发现感兴趣内容。在许多情况下，这些系统在一个低延时的条件下，会将数十亿用户与一个相当大的内容语料（数百万到数十亿）相连接。常用的方法是retrieval-and-ranking策略，这是一个two-stage系统。首先，一个可扩展的retrieval模型会从一个大语料中检索出一小部分相关items，接着一个成熟的ranking模型会对这些retrieved items**基于一或多个目标(objectives: 比如clicks或user-ratings)进行rerank**。在本文中，主要关注retrieval system。

给定一个{user, context, item}三元组，构建一个可扩展的retrieval模型的一个常用方法是：

- 1) 分别为{user,context}和{item}各自学习query和item representations
- 2) 在query和item representations间使用一个simple scoring function（比如：dot product）来得到对该query合适的推荐

**context通常表示具有动态特性的variables，比如：天时长(time of day)，用户所用设备（devices）**。representation learning问题通常有以下两个挑战：

- 1) items的corpus对于工业界规模的app来说相当大
- 2) 从用户反馈收集得到的训练数据对于某些items相当稀疏

**这会造成模型预测对于长尾内容（long-tail content）具有很大variance**。对于这种cold-start问题，真实世界系统需要适应数据分布的变化来更好面对**新鲜内容（fresh content）**。

受Netflix prize的启发，MF-based modeling被广泛用在构建retrieval systems中学习query和item的latent factors。在MF框架下，大量推荐研究在学习大规模corpus上解决了许多挑战。常见的思路是，利用query和item的content features。在item id外，content features很难被定义成大量用于描述items的features。例如，一个video的content features可以是从video frames中抽取的视觉features或音频features。MF-based模型通常只能捕获features的二阶交叉，因而，在表示具有许多格式的features collection时具有有限阶（power）。

在最近几年，受deep learning的影响，大量工作采用DNNs来推荐。Deep representations很适合编码在低维embedding space上的复杂的user states和item content features。在本paper中，采用two-tower DNNs来构建retrieval模型。**图1提供了two-tower模型构建的图示，左和右分别表示{user, context}和{item}**。two-tower DNN从multi-class classification NN（一个MLP模型）泛化而来[19]，其中，图1的right tower被简化成一个具有item embeddings的single layer。因而，**two-tower模型结构可以建模当labels具有structures或content features的情形**。MLP模型通常使用许多来自一个fixed的item语料表中sampled negatives进行训练。相反的，使用了deep item tower后，由于item content features以及共享的网络参数，对于计算所有item embeddings来说，在许多negatives上抽样和训练通常是无效的。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/727bab11fbdbc3698fb29496fd211f65663b710033b639c3e514edf43885bf90be605e072ef20ed3c277c5d73aa4f912?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750" width="300">

图1

我们考虑batch softmax optimization，其中item probability会在一个random batch上所有items上计算得到。然而，在我们的实验中所示，batch softmax具有sampling bias倾向，在没有任何纠正的情况下，可能会严重限制模型效果。importance sampling和相应的bias reduction在MLP模型[4,5]中有研究。受这些工作的启发，我们提出了使用estimated item frequency的batch softmax来纠正sampling bias。对比于MLP模型，其中output item vocabulary是固定的（stationary），我们会根据vocabualary和分布随着时间变化来target streaming data。我们提出了一种新算法通过gradient descent来概述（sketch）和估计（estimate） item freqency。另外，我们使用bias-corrected modeling，并将它扩展到在youtube推荐上构建个性化retrieval system。我们也引入了一个sequential training trategy，用来吸收streaming data，与indexing和serving组件一起工作。

主要4个contributions：

- Streaming Frequency Estimation。
- Model Framework
- Youtube recommendation
- offline和Live实现

# 2.相关工作

## 2.1 content-aware&Neural Recommenders

**对于提升泛化(generalization)和解决cold-start问题来说，使用users和items的content features很关键**。一些研究【23】在经典MF框架上采用content features。例如，generalized MF模型（比如：SVDFeatuer和FM），可以被用来采用item content features。这些模型能捕获bi-linear，比如：second-order的特征交叉。在最近几年，DNNs对于提升推荐的accuracy很有效。对比于传统因子分解方法，DNNs由于具有高度非线性，可以很有效地博获复杂的特征交叉。He [21]直接采用CF、NCF架构来建模user-item interactions。在NCF结构中，user和items embeddings被concatenated并被传入一个multi-layer NN来获得最终预测。我们的工作与NCF有两方法区别：

- 1) 我们利用一个two-tower NN来建模user-item interactions，以便可以在sub-linear时间内实现在大语料items的inference。
- 2) 学习NCF依赖于point-wise loss（比如：squared或log loss），而我们会引入multi-class softmax loss以及显式的model item frequency。

在其它work中，Deep RNN(比如：LSTM)被用于采用时序信息和推荐的历史事件，例如：[12,14]。除了单独的user和item representations外，另一部分设计NN的工作主要关注于学习rank systems。最近，multi-task learning是主要技术，对于复杂推荐器上优化多目标【27,28】。Cheng[9]引入了一个wide-n-deep framework来对wide linear models和deep NN进行jointly training。

## 2.2 Extreme classification

在设计用于预测具有大规模输出空间的labels的模型时，softmax是一个常用函数。从语言模型到推荐模型的大量研究，都关注于训练softmax多分类模型。当classes的数目相当大时，大量采用的技术是：抽样classes的一个subset。Bengio[5]表明：**一个好的sampling distribution应该与模型的output distribution相适配**。为了避免计算sampling distribution的并发症，许多现实模型都采用一个简单分布（比如：unigram或uniform）作为替代。最近，Blanc[7]设计了一个有效的adaptive kernel based的sampling方法。**尽管sampled softmax在许多领域很成功，但不能应用在具有content features的label的case中**。这种case中的Adaptive sampling仍然是一个开放问题。许多works表明，具有tree-based的label结构（比如：hierarchical softmax），对于构建大规模分类模型很有用，可以极大减小inference time。这些方法通常需要一个预定义的基于特定categorical attributes的tree structure。因此，他们不适用于包含大量input features的情况。

## 2.3 two-tower模型

构建具有two tower的NN在NLP中最近很流行，比如： 建模句子相似度(sentence similarities)，response suggestions，text-based IR等。我们的工作主要有，在大规模推荐系统上构建two-tower模型的有效性验证。对比于许多语言任务，我们的任务在更大corpus size上，这在Youtube这样的场景下很常见。通过真实实验发现，显式建模item frequency对于在该setting中提升retrieval accuracy很重要。然而，该问题并没有很好地解决。

# 3.模型框架

考虑推荐问题的一个常见设定，我们具有queries和items的一个集合。queries和items通过feature vectors $$\lbrace x_i \rbrace_{i=1}^{N}$$和$$\lbrace y_i \rbrace_{j=1}^M$$表示。这里，$$x_i \in X, y_i \in Y$$，是多种features的混合（比如：sparse IDs和dense features），可以在一个非常高维的空间中。这里的目标是：为给定一个query检索一个items的subset。在个性化场景中，**我们假设：user和context在$$x_i$$中被完全捕获**。注意，我们从有限数目的queries和items开始来解释该情形。我们的模型框架没有这样的假设。

我们的目标是构建具有两个参数化embedding functions的模型：

$$
u: X \times R^d \rightarrow R^k, v: Y \times R^d \rightarrow R^k
$$

将模型参数$$\theta \in R^d$$、query和candidates的features映射到一个k维的embedding space上。如图1所示，我们关注于的u, v通过两个DNN表示的case。模型的output是两个embeddings的inner product，命名为：

$$
s(x,y) = <u(x,\theta), v(y,\theta)>
$$

目标是，从一个具有T个样本的训练集中学习模型参数$$\theta$$：

$$
\mathscr{T} := \lbrace (x_i, y_i, R_i) \rbrace_{i=1}^T
$$

其中，$$(x_i, y_i)$$表示query $$x_i$$和item $$y_i$$的query，**$$r_i \in R $$是每个pair相关的reward**。

相应的，retrieval问题可以被看成是一个具有continuous reward的multi-class分类问题。**在分类任务中，每个label的重要性等价，对于所有postive pairs $$r_i=1$$**。**在recommenders中，$$r_i$$可以被扩展成：对于一个特定candidate捕获到的user engagement的不同程度**。例如，在新闻推荐中，$$r_i$$可以是一个用户花费在特定某个文章上的时间。给定一个query x，对于从M个items $$\lbrace y_i \rbrace_{j=1}^M$$选择候选y的概率分布，常用的选择是基于softmax function，例如：

$$
P(y|x; \theta) = \frac{e^{s(x,y)}}{\sum_{j \in [M]} e^{s(x,y_j)}}
$$

...(1)

接着进一步加入rewards $$r_i$$，我们考虑上下面的weighted log-likelihood作为loss function：

$$
L_T(\theta) := - \frac{1}{T} \sum\limits_{i \in [T]} r_i \cdot log(P(y_i | x_i; \theta) 
$$

...(2)

当M非常大时，在计算partition function时很难包括所有的candidate examples，例如：等式(1)中的分母。我们主要关注处理streaming data。因此，与负样本(negatives)从一个固定corpus中抽样得到的case训练MLP模型不同，对于从相同batch中的所有queries来说，我们只考虑使用in-batch items[22]作为负样本（negatives）。更确切地说，给定一个关于B pairs $$\lbrace (x_i, y_I, r_i) \rbrace_{i=1}^B$$的mini-batch，对于每个$$i \in [B]$$，该batch softmax是：

$$
P_B (y_i | x_i; \theta) = \frac{e^{s(x_i,y_i)}}{ \sum\limits_{i \in [B]} e^{s(x_i, y_i)}}
$$

...(3)

在我们的目标应用中，**in-batch items通常从一个power-law分布中抽样得到。因此，等式(3)在full softmax上会引入了一个大的bias：流行的items通常会过度被当成negatives，因为概率高**。受在sampled softmax model[5]中logQ correction的启发，我们将每个logit $$s(x_i, y_i)$$通过下式进行纠正：

$$
s^c(x_i, y_i) = s(x_i, y_j) - log(p_j)
$$

这里，$$p_j$$表示在一个random batch中item j的sampling概率。

有了该correction，我们有：

$$
P_B^c (y_i | x_i; \theta) = \frac{e^{s^c(x_i,y_i)}}{e^{s^c(x_i,y_i)} + \sum_{j \in [B],j \neq i} e^{s^c(x_i,y_i)}}
$$

接着将上述term插入到等式(2)，产生：

$$
L_B(\theta) := -\frac{1}{B} \sum\limits_{i \in [B]} r_i \cdot log(P_B^c(y_i \| x_i; \theta)) 
$$

...(4)

它是batch loss function。使用learning rate $$\gamma$$运行SGD会产生如下的参数更新：

$$
\theta \leftarrow \theta - \gamma \cdot \nabla_B (\theta)
$$

...(5)

注意，$$L_B$$不需要一个关于queries和candidates的固定集合。相应的，等式(5)可以被应用到streaming training data上，它的分布随时间变化。我们提出的方法，详见算法1.

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/3ba1b0f321a94538aa1521f6f591d5c22135d1a0bf8cce2ae0414bc947395fadce7c8834076213b345e45b7fbc62e99e?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=a1.jpg&amp;size=750" width="300">

算法1

最近邻搜索（NN Search）：一旦embedding function u, v被学到，inference包含两个step:

- 1) 计算query embedding：$$u(x,\theta)$$
- 2) 在item embeddings（通过embedding function v预计算好）上执行最近邻搜索

另外，我们的模型框架提供了选项，可以在inference时选择任意items。不再计算在所有items上的dot product，低时耗retrieval通常基于一个基于hashing技术高效相似度搜索系统，特别的，高维embeddings的compact representations通过quantization、以及end-to-end learning和coarse和PQ来构建。

归一化（Normalization）和温度（Temperature）。经验上，我们发现，添加embedding normalization，比如：$$u(x,\theta) \leftarrow u(x,\theta) / \|\| u(x,\theta) \|\|_2, u(y,\theta) \leftarrow v(y,\theta) / \|\| v(y,\theta) \|\|_2$$，可以提升模型的trainability，从而产生更好的retrieval quanlity。另外，一个tempreature $$\tau$$被添加到每个logit上来对predictions进行削尖(sharpen)：

$$
s(x,y) = <u(x,\theta), v(y,\theta)> / \tau>
$$

实际上，$$\tau$$是一个超参数，用于调节最大化检索指标（比如：recall或precision）。

# 4.Streaming Frequancy估计

在本节中，我们详细介绍在算法1中所使用的streaming frequency estimation。

考虑到关于random batches的一个stream，其中每个batch包含了一个items集合。该问题为：估计在一个batch中每个item y的hitting的概率。一个重要的设计准则是：当存在多个training jobs（例如：workers）时，具有一个完全分布式的估计来支持dstributed training。

在单机或分布式训练时，一个唯一的global step，它表示trainer消费的data batches的数目，与每个sampled batch相关。在一个分布式设定中，global step通常通过parameter servers在多个workers间同步。

...

# 5. Youtube的Neural检索系统

我们在Youtube中使用提出的模型框架。该产品会基于在某个用户观看的某个video上生成视频推荐。推荐系统包含两个stages：nomination(或：retrieval)、ranking。在nomination stage，我们具有多个nominators，每个nomiator都会基于一个user和一个seed video生成成百上千的视频推荐。这些videos会按顺序打分，并在下游的一个NN ranking模型中进行rerank。在本节中，我们关注在retrieval stage中一个额外nominator。

## 5.1 模型总览

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/2517fd8e613fbedc1ec7552fc200ad97fee57cee6bac535e8cd5ca91e91dbc514bfc1520bcdba7f94ea74fa477b341f7?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2

我们构建的youtube NN模型包含了query和candidates。图2演示了总的模型结构。在任意时间点上，用户正观看的某个video，（例如：seed video），提供了一个关于用户当前兴趣的一个很强信号。因此，我们会利用 关于seed video的features一个大集合以及用户观看历史。candidate tower的构建用来从candidate video features中学习。

**training label**。视频点击（video clicks）被用于正样本（positive labels）。另外，对于每个click，我们构建了一个reward $$r_i$$来表示关于该video的不同程度的user engagement。另一方面，$$r_i=1$$表示观看了整个视频。reward被用于example weight，如等式(4)所示。

**VIdeo Features**。video features在categorical和dense features中同时被用到。categorical features的样本包含了：Video Id和Channel Id。对于这两个entities的每个来说，会创建一个embedding layer来将categorical feature映射到一个dense vector上。通常，我们会处理两种categorical features。一些features（例如：Video Id）在每个video上具有一个categorical value，因此，我们具有一个embedding vector来表示它们。另外，一个feature（比如：Video topics）可以是一个关于categorical values的sparse vector，最终的embedding表示在sparse vector中的values的任一个的embeddings的加权求和。为了处理out-of-vocabulary entities，我们会将它们随机分配到一个固定的hash buckets集合中，并为每一个学习一个embedding。Hash buckets对于模型很重要，可以捕获在Youtube中的新实体（new entities），特别是5.2节所使用的sequential training。

**User Features**。我们使用一个user的观看历史来捕获在seed video外的user兴趣。一个示例是，用户最近观看过的k个video ids的一个sequence。我们将观看历史看成是一个bag of words (BOW)，通过video id embeddings的平均来表示它。在query tower中，user和seed video features在input layer进行融合（fuse），接着传入到一个feed forward NN中。

对于相同类型的IDs，embedding可以在相关的features间共享。例如，video id embeddings的相同集合被用于：seed video、candidate video以及用户之前观看过的video。我们也做了不共享embedding的实验，但没有观看大大的模型效果提升。

## 5.2 Sequential training

我们的模型在tensorflow上实验，使用分布式GD在多个workers和parameter servers上训练。在Youtube中，新的training data每天都会生成，training datasets会每天重新组织。该模型训练会以如下方式使用上sequential结构。trainer会从最老的training examples开始顺序消费数据，直到最近天的训练数据，它会等待下一天的训练数据到达。这种方式下，模型可以赶得上最新的数据分布偏移（shift）。训练数据本质上由trainer以streaming方式消费。我们使用算法2 (或算法3）来估计item frequency。等式(6)的在线更新使得模型可以适应新的frequency分布。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/dbbb897078383d5eea4b24667956b8568231674a1e49aa3f297d1cb1245ecf8d12e950f1f1cdf4cb6dde1d241ee2e98f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=a2.jpg&amp;size=750" width="300">

算法2

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/00c1a0d088aa1a844f197fa1f43409c17aad3521a346c5fe9bdb9fe150e51e1533ccaff74b250aa96b3e668d3048f2fc?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=a3.jpg&amp;size=750" width="300">
算法3

## 5.3 Indexing和模型serving

在retrieval系统中的index pipeline会为online serving周期性地创建一个tensorflow savemodel。index pipeline会以三个stages构建：candidate example generation、embedding inference、embedding indexing，如图3所示。在第1个stage，会基于特定准则从youtube corpus中选中的videos集合。它们的features被fetched、以及被添加到candidate examples中。在第二个stage，图2的right tower用来计算来自candidate examples的embeddings。在第三个stage，我们会基于tree和quantized hashing技术来训练一个tensorflow-based embedding index model。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/d7be1b9cd400be30193f66347d928625020471c88551b0834daaf7272e02e9d5371c7ce4649ec133e79ff6321c21c3ee?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=3.jpg&amp;size=750" width="300">

图3

# 6.实验

本节中，我们展示了item frequency estimation的模型框架的有效性。

## 6.1 Frequency估计的仿真

为了评估算法2&3的有效性。我们开始一个仿真研究，我们首先使用每个提出的算法来拟合一个固定的item分布，接着在一个特定step后变更分布。为了更精准，在我们的setting中，我们使用一个关于M items的固定set，每个item根据概率$$q_i \propto i^2$$（其中：$$i \in [M], \sum_i q_i = 1$$）进行独立抽样。

。。。略


# 参考

- 1.[https://dl.acm.org/doi/pdf/10.1145/3298689.3346996](https://dl.acm.org/doi/pdf/10.1145/3298689.3346996)