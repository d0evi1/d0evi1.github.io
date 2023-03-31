---
layout: post
title: Mixed Negative Sampling介绍
description: 
modified: 2020-10-21
tags: 
---

google（google play）《Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations》中提出了Mixed Negative Sampling。

# 3.模型框架

在本节中，我们首先提出：对于retrieval任务的large-corpus推荐系统中的一个数学表示。我们接着提出一个two-tower DNN的建模方法，并描述了如何使用in-batch negative sampling来训练模型。最后，我们介绍了Mixed Negative Sampling (MNS) 技术来解决batch negatives的选择偏差（selection bias）。

## 3.1 问题公式

在推荐系统中的检索任务，目标是：给定一个query，从整个item corpus中，快速选择数百到上千的候选内容。特别的，一个query可以是一个文本、一个item（比如：一个app）、一个user，或者它们的一个混合形式。这里，queries和items可以被表示成feature vectors，用于捕获多样的信息。我们将检索问题看成是一个多分类问题，从一个large corpus(classes) C中选择一个item的likelihood可以被公式化成一个softmax probability：

$$
P(y | x) = \frac{e^{\epsilon(x,y)}}{\sum_{j \in C} e^{\epsilon (x, y_i)}}
$$

...(1)

其中：

- $$\epsilon(x,y)$$表示由retrieval model提供的logits，feature vectors x和y分别表示query和item。

## 3.2 建模方法

我们采用一个two-tower DNN模型结构来计算logits $$\epsilon(x,y)$$。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/a722d6b90f78dcc3e66ef1f65570841b9068aa42521a6a52c54693833f8be64a9fb4c6c8deac795fdd03eeb98dac090d?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 双塔结构

如图1所示，left tower和right tower会分别学习query和item的latent representations。正式的，我们通过函数$$u(x; \theta)$$和$$v(y; \theta)$$来分别表示两个towers，它们会将query features x和item features y映射到一个共享的embedding space上。这里$$\theta$$表示所有的模型参数。该模型会输出query和item embeddings的内积作为等式(1)中的logits：

$$
\epsilon(x, y) = <u(x;\theta), v(y;\theta)>
$$

出于简单，我们会：

- u表示成一个给定query x的的embedding
- $$v_j$$表示从corpus C中的item j的embedding

对于一个$$\lbrace query(x), item(y_l, postive \ label) \rbrace$$pair的cross-entropy loss变为：

$$
L = - log(P(y_l | x)) = - log(\frac{e^{<u, v_l>}}{\sum_{j \in C} e^{<u, v_j>}})
$$

...(3)

对等式(2)根据参数$$\theta$$做梯度，给出：

$$
\nabla_{\theta} (- log P(y_l | x))  \\
    = - \Delta_{\theta}(<u, v_l>) + \sum\limits \frac{e^{<u,v_j>}}{\sum_{j \in C} e^{<u, v_j>}} \nabla_{\theta}(<u, v_j>)   \\
    = - \nabla_{\theta}(<u, v_l>) + \sum\limits_{j \in C} P(y_j | x) \Delta_{\theta}(<u, v_j>)
$$

第二项表示：$$\nabla_{\theta}(<u, v_j>)$$是对于$$P(\cdot \mid x)$$（指的是target分布）的期望（expectation）。通常在大的corpus上对所有items计算第二项是不实际的。因此，我们会通过使用importance sampling的方式抽样少量items来逼近该期望（expectation）。

特别的，我们会从corpus中使用一个预定义分布Q来抽样一个items子集$$C'$$，其中$$Q_j$$是item j的抽样概率（sampling probability），并用来估计等式(3)中的第二项：

$$
E_P [\nabla_{\theta}(<u, v_j>)] \approx_{j \in C'} \frac{w_j}{\sum_{j' \in C'} w_{j'}} \nabla_{\theta}(<u, v_j>)
$$

...(4)

其中，$$w_j = e^{<u, v_j> - log(Q_j)}$$会包含用于sampled softmax中的logQ correction。

双塔DNN模型的一个常用sampling策略是：batch negative sampling。特别的，batch negative sampling会将在相同training batch中的其它items看成是抽样负样本（sampled negatives），因此sampling分布Q会遵循基于item频次的unigram分布。它可以避免feed额外的负样本到右塔，从而节约计算成本。图2展示了在一个training batch上的计算过程。给定在一个batch中有B个{query, iitem} pair，B个queries和B个items的features会分别经过该模型的左塔和右塔。产生$$B X K$$（K是embedding维度）的embedding matrix U和V。接着，logits matrix可以被计算为$$L = U V^T$$。由于batch negative sampling会极大提升训练速度，我们会在下节中讨论它的问题，并提出一个可选的sampling策略。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/53f1cba4f42001ccac8f1ffbda08eac47c0cb9b39a307a991471b8ea67ad751e5dacee16d1243eaa4a8821b85a8ed9b4?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2

## 3.3 Mixed Negative Sampling

控制梯度估计的bias和variance对于模型质量来说很重要。有两种方式来减小bias和variance：

- (1) 增加sample size
- (2) 减少在Q分布和target分布P间的差异

**在训练两塔DNN模型的情况下，batch negative sampling会隐式设置采样分布Q为unigram item frequency分布**。它在推荐场景下有两个问题：

- (1) 选择偏差：例如，一个没有任何用户反馈的item不会包含在训练数据中作为一个candidate app。因而，它不会被batch negative sampling抽样成一个负样本。因此，该模型不能有效区分具有稀疏反馈的items 与 其它items。
- (2) 缺乏调整抽样分布的灵活性：隐式采样分布Q由训练样本决定，不能被进一步调整。Q与target分布P偏离，会导致巨大bias。

我们提出了Mixed Negative Sampling (MNS) 来解决该问题。它会从另一个数据流中均匀抽样$$B'$$个items。我们将这额外数据流称为index data，它是一个由来自整个corpus的items组成的集合。这些items对于每个batch来说会当成额外负样本使用。图3展示了一个training batch的计算过程。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/8b4f492171d4f0bd098566cd72198c136ba650f35dc028cf84c62a6811548cc283bf352d24b23e29db3944a9bfc981b8?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3

除了$$B \times K$$的query embedding matrix $$U_B$$外，$$B \times K$$ candidate item embedding matrix $$V_B$$，均匀从index data中抽样的$$B'$$个candidate items会被feed到右塔来生成一个$$B' \times K$$个candidate item embedding matrix $$V_B'$$。我们将$$V_B$$和$$V_B'$$进行拼接来获得一个$$(B + B') \times K$$个candidate item embedding matrix V。事实上，我们具有$$B \times (B + B')$$个logits matrix $$L = U V^T$$。label matrix因此变成$$B \times (B + B')$$，具有一个所有都为0的$$B \times B'$$的matrix会append到原始的$$B \times B$$的对角矩阵上。相应的，一个training batch的loss function：

$$
L_B \approx - \frac{1}{B} \sum\limits_{i \in [B]} log(\frac{e^{<u_i, v_i>}}{e^{<u_i, v_i>} + \sum\limits_{j \in [B+B'], j \neq i} w_{ij}})
$$

...(5)

其中：

$$w_{ij} = e^{<u_i, v_j> - log(Q_j^*)}$$，$$u_i$$是U的第i行，$$v_j$$表示V的第j行。这里的抽样分布$$Q^*$$变成是一个关于基于unigram sampling的item frequency和uniform sampling的混合形式，有一个关于batch size B和$$B'$$间的ratio。

MNS会解决以上与batch softmax有关的两个问题：

- (1) 减少选择偏差（selection bias）：在corpus中的所有items具有一个机会作为负样本，以便retrieval model可以更好分辩新鲜和长尾的items
- (2) 在控制sampling分布时使得更灵活：有效采样分布Q是一个来自training data基于unigram分布的item freqeuncy和来自index data的uniform分布的mixture。对于一个固定的batch size B，我们可以通过调节$$B'$$来实验不同的Q。这里的$$B'$$可以作为一个超参数进行调节。



# 参考

- 1.[https://dl.acm.org/doi/pdf/10.1145/3366424.3386195](https://dl.acm.org/doi/pdf/10.1145/3366424.3386195)2022-02-13-dwelltime-reweight.md