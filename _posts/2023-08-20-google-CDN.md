---
layout: post
title: google CDN介绍
description: 
modified: 2023-08-20
tags: 
---

# 摘要

工作界推荐系统通常会存在高度倾斜的长尾item分布，一小部分的items会接受到大量的用户反馈。这种倾斜会伤害推荐系统质量，特别是：那些具有较少用户反馈的item。学术界的许多研究，很难部署到真实生产环境中，并且提升很小。这些方法的一个挑战是：通常伤害整体效果；另外，训练和服务通常是复杂和昂贵的。

在本工作中，我们的目标是：提升长尾item推荐，并维持整体效果具有更少的训练和服务开销。我们首先发现：用户偏好的预估在长尾分布下会是有偏的。这种bias来自于training和serving数据间的两个差异：

- 1）item分布
- 2）用户对于某一给定item的偏好

大多数已经存在的方法，主要尝试减少来自item分布角度上的bias，忽略了对于给定某一item的用户偏好差异。这会导致一个严重的遗忘问题，并导致次优的效果。

为了解决该问题，我们设计了一个新的CDN（Cross Decoupling Network）来减少这两个不同点。特别的，CDN会：

- (i) 通过一个MoE结构（mixture-of-expert）来解耦记忆（memorization）和泛化（generalization）的学习过程
- （ii）通过一个正则双边分支网络（regularized bilateral branch network）来解耦来自不同分布的用户样本

最终，一个新的adapter会被引入进来对decoupled vectors进行聚合，并且将training attention进行柔和的转移到长尾items上。大量实验结果表明：CDN要好于SOTA方法。我们也展示了在google大规模推荐系统中的有效性。

# 1.介绍

。。。

# 2.在推荐中的长尾与动机

。。。

# 3.CDN（Cross Decoupling Network）

基于上述分析，我们提出了一个可扩展的cross decoupling network（CDN）来解决在item和user侧的两个差异。主要结构如图2所示。

- 在item侧，我们提出：对头部item和长尾item的represation learning的memorization和generalization进行解耦。为了这么做，我们会使用一个gated MoE结构。在我们的MoE版本中，我们会将memorization相关的features输入到expert子网络中来关注memorization。相似的，我们会将content相关的features输入到expert子网络中来关注generalization。一个gate（通常：是一个learnable function）会被引入进来描述：该模型需要放置多少weight到一个item representation的memorization和generalization上。增强的item representation learning可以将item分布差异进行融合。

- 在user侧，我们可以通过一个regularized bilateral branch network来将user sampling策略进行解耦，来减少用户偏好差异。该网络包含了：一个主分支用于通用的用户偏好学习，一个正则分支来补偿在长尾items上的用户反馈的稀疏性。在两个branch间的一个共享塔会用来扩展到生产环境中。

最终，我们会将user和item learning进行交叉组合，使用一个$$\gamma$$-adapter来学习用户在长尾分布中的头部和尾部items上的多样偏好。

## 3.1 Item Memorization 和 Generalization Decoupling

我们引入memorization features 和generalization features的概念，接着，描述了通过一个gated MoE结构来解耦它们的方法。

### 3.1.1 用于memorization和generalization的features

工业界推荐系统通常会考虑成百上千个features作为model inputs。除了使用相同方式编码这些features之外，我们考虑将这些features进行划分成两组：memorization features和generalization features。

Memorization features.

他们会帮助记住在训练数据中user和item间的交叉（协同信号），比如：item ID。正式的，这些features通常是categorical features，满足：

- 唯一性（Uniqueness）：对于它的feature space V，存在 $$ f_{in}$$满足 $$ f_{in}$$是一个injective function，并且有：$$f_{in}: I \rightarrow V$$
- 独立性（Independence）：对于$$\forall v_1, v_2 \in V$$，$$v_1$$的变化不会影响到$$v_2$$

在生产环境推荐系统中，这些features通常由embeddings表示。这些embedding参数可能只会通过对应的item来被更新（唯一性），并且不会与其它items的任何信息共享（独立性）。因而，它们只会记住对于一个特定item的信息，不会泛化到其它已经存在或未见过的items上。同时，由于唯一性，这些features也展示了一个长尾分布。因此，对于那些对应于头部items的features来说，它们的embedding更新通常会生成一个显著的记忆效果。而对于那些尾部items的features，它们的embeddings可能会有噪音，因为缺少梯度更新。

**Generalization features**

泛化features，可以学到在user偏好与item features间的相关性，并且可以泛化到其它items上。这些features即可以跨多个不同items共享（例如：item类别、标签等），或者是continuous features。因而，可以泛化到其它已存在或未见过的items上，对于提升尾部item的representation learning来说很重要。

### 3.1.2 Item representation learning

我们采用带有一个frequency-based gating的MoE结构来解耦memorization features和generation features。该结图如图2的左侧所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/6689790a233478be2bb2610dbb73e776d96084b5832fc102f72a3c2f1e3f85d0755af7fa28d6bd0e0ab50a0d615e1133?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2

也就是说，对于一个训练样本(u, i)，item embedding可以表示成：

$$
y = \sum\limits_{k=1}^{n_1} G(i)_i E_k^{mm} (i_{mm}) + \sum\limits_{k=n_1 + 1}^{n_1 + n_2} G(i)_k E_k^{gen}(i_{gen})
$$

...(3)

其中：

- $$E_k^{mm}(\cdot)$$：表示memorization-focused expert，它会将所有memorization features $$i_{mm}$$（例如：item ID）的embeddings进行concat作为input；
- $$E_k^{gen}(\cdot)$$：表示generalization-focused expert，它会将所有generalization features $$i_{gen}$$（例如：item类别）的embeddings进行concat作为input
- $$G(\cdot)$$：是gating function，其中：$$G(i)_k$$表示第k个element，$$\sum\limits_{k=1}^{n_1+n_2} G(i)=1$$

这里的gating很重要，可以对头部items和尾部items的memorization和generalization进行动态平衡。直觉上，gate可以将item frequency作为input，并且通过一个non-linear layer对它进行transform：$$g(i) = softmax(W_i_{freq}$$，其中，W是一个可学习的weight matrix。它可以将来自其它features作为input，我们发现：item popularity作为输入效果很好。

这种机制可以以一个简单、优雅的方式来发现长尾分布的items间的差异，用来增强item representation learning。通过将memorization和generalization进行解耦，头部items可以达到更好的memorization能力、尾部items也可以同时得到更多的泛化。如【12】所示，增强的item representation可以补偿在$$P(u \mid i)$$和$$\hat{p}(u \mid i)$$间的条件分布的一致性。另外，通过使用 frequency-based gates的experts对memorization和generazation进行解耦，当learning attention偏向于尾部items时，我们可以缓和遗忘问题（forgetting issue）。也就是说，有了decoupling，当training attention偏向于尾部items，来自尾部items的gradients（知识）会主要更新在generalization-focused expert中的模型参数，从而保持着来自 head items的well-learned
memorization expert。

## 3.2 User Sample Decoupling

如图2的右侧user所示，受【13，20，29】的启发，我们提出了一个regularized bilateral branch network ，它包含了两个分支：

- “main” branch：它会在原始的高度倾斜的长尾分布$$\Omiga_m$$上进行训练；
- “regularizer” branch：它会在一个相对平衡的数据分布$$\Omiga_r$$上进行训练

$$\Omiga_m$$包含了来自头部items和尾部items的所有user feedback，$$\Omiga_r$$则包含了对于尾部items的所有user feedback

其中，对头部items的user feedback进行down-sampling，以使得它与最流行的尾部items一样的频次。在两个branch间会使用一个共享的tower来增加扩展性（scalability）。该方法可以温和地对尾部items的用户偏好的学习进行上加权（up-weight）。因此，这会纠正对尾部items的用户偏好的欠估计（under-estimation），并能缓和用户偏好估计的popularity bias。

在每个step中，一个训练样本：$$(u_m, i_m) \in \Omega_m$$、以及$$(u_r, i_r) \in \Omega_r$$会分别feed到main branch、以及 regularizer branch中。接着 user representation vectors会通过如下进行计算：

$$
x_m = h_m(f(u_m)), x_r = h_r(f(u_r))
$$

...(4)

其中：

- $$f(\cdot)$$是一个由两个branch共享的sub-network
- $$h_m(\cdot)$$和$$h_r(\cdot)$$是 branch-specific sub-networks

共享的network会帮助对来自两个分布的学到知识进行交流，并且能大大减少计算复杂度。branch-specific subnetwork可以为每个数据分布（头部和尾部items）学习唯一的知识。因此，$$\Omega_m$$和$$\Omega_r$$可以被联合学习用来逼近$$\hat{p}(i)$$到$$p(i)$$，并减少先验视角的一致性。

main branch的目标是，学习高质量的user representations，并维持着原始分布的特性，是支持进一步学习regularizer
branch的基石。如【13】所示，在原始分布上的训练可以学习最好、最泛化的representations。regularizer branch被设计是用来：

- (1) 添加尾部信息到模型中，并缓和在尾部items上的高IF影响；
- (2) 通过一个regularized adapter来阻止尾部items的过拟合（over-fitting）

在应用到生产环境时，两个branches可以同时训练。因此，不需要额外的训练开销。注意，在inference时，只会使用main branch，因此没有额外的serving开销。

## 3.3 Cross Learning

为了桥接在head items和tail items上的gap，我们会通过一个$$\gamma$$-adapter将来自user侧和item侧的信息进行解耦学习。

$$\gamma$$-adapter的设计是用来将学到的representations进行融合，并能柔和地朝着尾部items的学习进行偏移。特别的，

- 对于$$x_m$$和$$x_r$$，它们是从main branch和regularizer branch中学习到的user representations
- 对于$$y_m$$和$$Y_R$$，它对应于学到的item representations

predicted logit可以被公式化为：

$$
s(i_m, i_r) = \alpha_t y_m^T x_m + (1 - \alpha_t) y_r^T x_r
$$

...(5)

其中:

- $$\alpha_t$$是$$\gamma$$-adapter，它是一个关于training epoch t的函数：

$$
\alpha_t = 1 - (\frac{t}{ \gamma \times T})^2, \gamma > 1
$$

...(6)

这里，T是epochs的总数目，$$\gamma$$是regularizer rate。我们看到：$$\alpha_t$$会随着训练过程衰减（随着t递增），这会让模型学习从原始分布朝着平衡后的数据分布偏移。在这种方式下，我们会首先学习通用模式，接着渐近地朝着tail items进行偏移来提升它们的效果。这种顺序对于获得一个高质量representation learning来说很重要，它可以进一步促进regularizier branch的学习，如【32】所示。约束条件$$\gamma > 1$$在推荐setting中也很重要，可以缓和forgetting issue：它可以确保通过训练主要关注点仍在main branch。这对于具有不同imbalanced factor IF的长尾分布来说是一个希望的feature，当IF很高时，更偏好于一个更大的$$\gamma$$。事实上，我们会经验性的发现：𝛾-adapter可以极大有益于在高度倾斜的长尾分布上的学习。

有了logit $$s(i_m, i_r)$$，我们可以通过一个softmax来计算：user u对于不同items的偏好概率：

$$
p(i | u ) = \frac{e^{s(i_m, i_r)}}{\sum_{j \in I} e^{s(j_m, j_r)}}
$$

...(7)

在工作界应用中，出于可扩展性，常使用batch softmax。接着loss function可以被公式化为：

$$
L = - \sum\limits_{u \in U, i \in I} \alpha_t \hat{d}(u_m, i_m) log p(i|u) + (1 - \alpha_t) \hat{d}(u_r, i_r) log p(i|u)
$$

...(8)

其中：

- $$\hat{d}(u_m, i_m)$$以及$$\hat{d}(u_r, i_r)$$分别是来自main branch和regularizer branch的user feedback。他们可以帮助学习用到对于items的高偏好得分。

对于inference，为了预估一个user对于一个item的偏好，我们只会使用main branch，并计算preference score：

$$
s(u, i) = y_m^T x_m
$$

...(9)

以便获得在softmax中的logits。

regularizer branch functions则作为一个regularizer用于训练。在prediction时，test data是长尾的，添加regularizer branch会引入在分布中的其它层的不匹配。

**Training和serving开销**：对比起标准的双塔模型，CDN对于训练来说具有很小的额外开销。在serving时，在双塔setting中，user side只会使用main branch，item side则具有相同数目的参数/FLOPS。在training时，在user侧的额外开销只有在regularizer branch的head上，它是可忽略的。

讨论：一个直接问题是：为什么我们要从不同角度去解耦user和item side呢？在本工作中，我们考虑来自item side的长尾分布（例如：长尾item分布），它会将users看成是在长尾分布中的样本。如果我们希望考虑来自user side的长尾分布，那么一个直接的方法是：在users和items间切换decoupling方法，如【29】。然而，我们会讨论long-tail用户分布是否也可以不同建模，因为user侧的IF通常要小于item side。另外，两个sides的长尾分布高度相关，可以在每个side上影响IF。这是个nontrivial问题，我们保留该问题，等后续进一步探索。

# 

- 1.[https://browse.arxiv.org/pdf/2210.14309.pdf](https://browse.arxiv.org/pdf/2210.14309.pdf)