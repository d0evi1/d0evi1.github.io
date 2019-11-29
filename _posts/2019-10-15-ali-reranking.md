---
layout: post
title: ali reranking模型介绍
description: 
modified: 2019-10-15
tags: 
---

# 介绍

alibaba在《Personalized Re-ranking for Recommendation》介绍了一种reranking模型。

# 摘要

ranking是推荐系统的核心问题，通常，一个ranking函数会从labeled dataset中学到，并会为每个单独item产生一个ranking score。然而，它可能是次优的(sub-optimal)，因为scoring function被应用在每个独立item上，没有显式考虑item间的相互影响，以及用户偏好／意图间的不同。因此，这里提出了一种个性化的re-ranking模型。通过直接使用一个已经存在的ranking feature vectors，提出的re-ranking模型可以很轻易地部署成在任意ranking算法之后跟着的一个follow-up模块。它会通过采用一个transformer结构来对在推荐列表中的所有items信息进行有效编码，来直接优化整个推荐列表。特别的，transformer使用一种self-attention机制，它直接建模在整个list中任意items pair间的关系。我们证实，通过引入pre-trained embedding来为不同用户学习个性化编码函数。在offline和online的实验结果上均有较大提升。

# 1.介绍

通常，在推荐系统中的ranking不会考虑在list列表中其它items（特别是挨着的items）的影响。尽管pairwise和listwise l2r方法尝试解决该问题，但它们只关注充分利用labels（比如：click-through data）来优化loss function，并没有显式建模在feature space中items间的相互影响。

一些工作[1,34,37]尝试显式建模items间的相互影响，重新定义由之前ranking算法给出的intial list，这被称为“re-ranking”。构建该scoring function的主要思路是：将intra-item patterns编码成feature space。state-of-the-arts的方法有：RNN-based（比如：GlobalRerank[37]和DLCM[1]）。它们会将初始列表（intial list）按顺序feed给RNN-based结构，并在每个timestep上输出编码后的vector。然而，RNN-based方法对建模在list中items间的交叉的能力有限。之前编码的item的feature信息会沿着编码距离退化（degrade）。同时，由于并行化，transformer的编码过程比RNN-based方法更高效。

除了items间的交叉外，交叉的个性化编码函数可以被考虑用于re-ranking。对推荐系统进行re-ranking是user-specific的，决取于用户的偏好。对于一个对价格敏感的用户，在re-ranking模型中，对“price”特征间进行交叉更重要。常见的global encoding function可能不是最优的，因为它会忽略每个用户在特征分布间的不同。例如，当用户关注价格对比时，具有不同价格的相似items趋向于在list中更聚集。当用户没有明显的购买意图时，推荐列表中的items趋向于更分散。因此，我们在transformer结构中引入一种个性化模块来表示关于item interactions用户偏好和意图。在我们的个性化re-ranking模型中，可以同时捕获：在推荐列表中的items、以及用户间的交叉。

# 2.相关工作

我们的工作主要是，重新定义由base ranker给出的initial ranking list。在这些base rankers间，l2r是一种广泛使用的方法。l2r方法可以根据loss function分为三类：point-wise、pairwise、listwise。所有这些方法可以学习一个global scoring function，对于一个特定feature的权重会被全局学到。然而，这些features的weights应可以意识到：不仅items间的交叉、以及user和items间的交叉。

【1-3,37】的工作主要是re-ranking方法。它们使用整个intial list作为input，并以不同方式建模在items间的复杂依赖。[1]使用unidirectional GRU来将整个list的信息编码到每个item的表示。[37]使用LSTM、[3]使用pointer network，不仅编码整个list信息，也会由decoder生成ranked list。对于这些使用GRU or LSTM的方法来编码items间的依赖，encoder的能力通过encoding distance进行限制。在我们的paper中，我们使用transfomer-like encoder，它基于self-attention机制以O(1) distance来建模任意两个items间的交叉。另外，对于那些使用decoder来顺序生成ordered list的方法，它们不适合online ranking系统，因为需要有严格的延迟。由于sequential decoder使用在time t-1上选中的item作为input来在time t上选择item，它不能并列行，需要n倍的inferences，其中n是output list的长度。[2]提出了一种groupwise scoring function，它可以对scoring进行并列化，但它的计算开销很高，因为它会枚举在list中所有可能的item组合。

# 3.re-ranking模型

在本节中，我们首先给出了一些关于l2r以及re-ranking的先验知识。接着对问题公式化来求解。概念如表1所示。

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/745f2bbfc533a504da2918aa23c746219a7043a30d0eaebb201dd99b75d3ebd3f72a695bc02c2dddfaa5c05c6ebdbb3c?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.png&amp;size=750">

表1: 

l2r方法在ranking中被广泛使用，为推荐和信息检索生成一个ordered list。l2r方法会基于items的feature vector学习一个global scoring function。有了该global function，l2r方法会通过在candidate set中的每个item。该global function通常通过对以下loss L最小化得到：

$$
L = \sum\limits_{r \in R} l( \lbrace y_i, P(y_i | x_i;\theta) | i \in I_r \rbrace ) 
$$

...(1)

其中：

- R是对于推荐所有用户请求的集合。
- $$I_r$$是对于请求$$r \in R$$的items的candidate set
- $$x_i$$表示的是item i的feature space
- $$y_i$$是在item i上的label (例如：click or not)
- $$P(y_i \mid x_i; \theta)$$是由ranking model给出的对于参数$$\theta$$的预测点击概率
- l是通过$$y_i$$和$$P(y_i \mid x_i; \theta)$$计算得到的loss

然而，对于学习一个好的scoring function来说，$$x_i$$是不足够的。我们发现推荐系统的ranking应考虑以下额外信息：

- (a)item-pairs间的相互影响 [8,35]
- (b)users和items间的交叉(interactions)

在item-pairs间的相互影响，可以通过使用已经存在的LTR模型为请求r从inital list $$S_r=[i_1, i_2, \cdots, i_n]$$直接学到。[1][37][2][3]提出了方法来更好利用item-pairs间的相互信息。然而，很少研究去关注users和items间的interactions。**item-pairs的相互影响，对于不同用户来说是不同的**。在本paper中，我们引入了一种个性化矩阵(personlized matrix) PV来学习user-specific encoding function，它可以建模item-pairs间的个性化相互影响。模型的loss function可以被公式化成等式(2)。

$$
L = \sum\limits_{r \in R} l( \lbrace y_i, P(y_i \mid X, PV; \hat{\theta}) | i \in S_r \rbrace )
$$

...(2)

其中：

- $$S_r$$是由之前ranking model给出的inital list
- $$\hat{\theta}$$是我们的re-ranking model的参数
- X是在list中所有items的feature matrix

# 4.个性化re-ranking模型

在本节中，我们首先给出了关于PRM(Personalized Re-ranking Model)的总览。接着详细介绍每一部分。

## 4.1 模型架构

PRM的结构如图1所示。模型包含三个部分：

- input layer
- encoding layer
- output layer

它会将由之前的ranking模型生成的关于items的intial list作为input，并输出一个re-ranked list。详细结构分挨个介绍。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/fac8a30c2472fa5cd21deea300e1af6caf051926f224a6c030c85a49642f0f8b587c62fed6f546caa36443cdf4382428?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.png&amp;size=750">

## 4.2 Input Layer

input layer的目的是，为在initial list中的所有items准备representations，并将它feed给encoding layer。首先，我们具有一个固定长度的intial sequential list $$S=[i_1, i_2, \cdots, i_n]$$，它由之前的ranking方法给出。与之前ranking方法相同，我们具有一个raw feature matrix $$X \in R^{n \times d_{feature}}$$，在X中的每一行表示每个$$i \in S$$的item对应的raw feature vector $$x_i$$。

**Personalized Vector(PV)**

对两个items的feature vectors进行encoding，可以建模它们之间的相互影响，但这些影响进行扩展将会影响那些未知用户。因而需要学习user-specific encoding function。尽管整个intial list的representation可以部分影响用户的偏好，但对于一个强大的personlized encoding function来说它是不够的。如图1(b)所示，我们将raw feature matrix $$X \in R^{n \times d_{feature}}$与一个个性化矩阵$$PV \in R^{n \times d_{pv}}$$进行concat来获取中间（intermediate）的embedding matrix $$E' \in R^{n \times (d_{feature} + d_{pv})}$$，如等式(3)。PV通过一个pre-trained model生成，它会在下一节中介绍。PV的performance增益可以由evaluation部分被介绍。

$$
E' = \left[
\begin{array}
  x_{i_1}; pv_{i_1} \\
  x_{i_2}; pv_{i_2} \\
  \cdots \\
  x_{i_n}; pv_{i_n}
\end{array}
\right]
$$

...(3)

**position embedding(PE)**

为了利用在intial list中的顺序信息，我们将一个position embedding $$PE \in R^{n \times (d_{feature}+d_{pv})}$$注入到input embedding中。接着，该embedding矩阵可以用等式(4)计算。本paper中会使用一个可学习的PE，它的效果要比[28]中固定的position embedding要略微好些。

$$
E'' = \left[
\begin{array}
  x_{i_1}; pv_{i_1} \\
  x_{i_2}; pv_{i_2} \\
  \cdots \\
  x_{i_n}; pv_{i_n}
\end{array}
\right] + \left[
\begin{array}
  pe_{i_1} \\
  pe_{i_2} \\
  \cdots \\
  pe_{i_n}
\end{array}
\right]
$$

...(4)

最后，我们使用一个简单的feed-forward网络来将feature matrix $$E'' \in R^{n \times (d_{feature} + d_{pv}}$$转成$$E \in R^{n \times d}$$，其中：d是encoding layer中每个input vector中潜在维度(latent dimensionality)。E可以通过等式(5)公式化：

$$
E = EW^E + b^E
$$

...(5)

其中，$$W^E \in R^{(d_{feature} + d_{pv}) \times d}$$是投影矩阵，$$b^E$$是d维向量。

## 4.3 Encoding Layer

如图1(a)所示，encoding layer的目标是，将item-pairs间的相互影响、以及其它额外信息进行集成，这些额外信息包含：用户偏好、intial list S的ranking顺序。为了达到该目标，我们采用Transfomer-like encoder，因于Transformer已经在许多NLP任务中被证明是有效的，特别是在机器翻译中。Transformer中的self-attention机制特别适合我们的re-ranking任务，因为它可以直接建模任意两个items间的相互影响，忽略掉两者间的距离。没有了距离衰减(distance decay)，Transfomer可以捕获更多在intial list中离得较远的items间的交叉。如图1(b)所示，我们的encoding模块包含了$$N_x$$个关于Transformer encoder的块（blocks）。每个块（block）（如图1(a)所示）包含了一个attention layer和一个Feed-Forward Network(FFN) layer。

**Attention Layer**

attention函数如等式(6)所示：

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d}})V
$$

...(6)

其中矩阵Q, K, V各自表示queries、keys和values。d是matrix K的维度，为了避免内积的大值。softmax被用于将内积值转化成为value vector V添加权重。在我们的paper中，我们使用self-attention，其中：Q, K和V从相同的矩阵进行投影。

为了建模更复杂的相互影响，我们使用multi-head attention，如等式(7)所示：

$$
S' = MH(E) = Concat(head_1, \cdots, head_h) W^O \\
head_i = Attention(EW^Q, EW^K, EW^V)
$$

...(7)

其中，$$W^Q, W^K, W^V \in R^{d \times d}$$。$$W^O \in R^{hd \times d_{model}}$$是投影矩阵。h是headers的数目。h的不同值间的影响会在下一节被研究。

**FFN(Feed-forward Network)**

该position-wise FFN的函数主要是为了使用在input vectors不同维度间的非线性（non-linearity）和交叉（interacitons）来增强模型。

**对Encoding Layer进行Stacking**

这里，我们使用attention模块，后面跟着position-wise FFN作为一块(block)Transformer encoder。通过对多个blocks进行stacking，我们可以得到更复杂和高阶的相互信息（mutual information）。

## 4.4 Output Layer

output layer的函数主要为每个item $$i = i_1, \cdots, i_n$$生成一个score。（如图1(b)所示Score(i)）我们在softmax layer之后使用一个linear layer。softmax layer的output是每个item的点击概率，被标记为：$$P(y_i \mid X, PV; \hat{\theta})$$。我们使用$$P(y_i \mid X, PV, \hat{\theta})$$作为$$Score(i)$$来在one-step中对items进行re-rank。Score(i)的公式为：

$$
Score(i) = P(y_i \mid X, PV; \hat{\theta}) = softmax(F^{(N_x)}W^F + b^F), i \in S_r
$$

...(8)

其中：

- $$F^{(N_x)}$$是Transformer encoder的$$N_x$$个blocks的output
- $$W^F$$是可学习的投影矩阵
- $$b^F$$是bias term
- n是在intial list中的items数目

在训练过程中，我们使用click-through data作为label并最小化等式(9)的loss function：

$$
L = - \sum\limits_{r \in R} \sum\limits_{i \in S_r} y_i log(P(y_i | X, PV; \hat{\theta})
$$

...(9)

## 4.5 个性化模块

在本节中，我们会引入该方法来计算个性化矩阵PV，它表示user和items间的interactions。使用PRM来学习PV的最简单办法是，通过re-ranking loss以end-to-end的方式进行学习。在re-ranking任务中学到的task-specific representation缺少用户的一般偏好。因此，我们可以利用一个pre-trained NN来产生用户个性化embeddings PV，它接着被用做PRM模型的额外features。pre-trained NN可以从平台的所有click-through logs上学到。图1(c)展示了pre-trained模型的结构。sigmoid layer会输出：在给定所有行为历史$$(H_u)$$和用户的side information时，关于item i、user u的点击概率$$(P(y_i \mid H_u, u; \theta')$$。用户的side information包括：gender、age和purchasing level等。模型的loss通过一个point-wise cross-entropy函数来计算，如等式(10)所示：

$$
L = \sum\limits_{i \in D} (y_i log( P(y_i | H_u, u; \theta')) + (1-y_i) log(1-P(y_i | H_u,u;\theta')
$$

...(10)

其中：

- D是user u在平台上展示的items set。
- $$\theta'$$是pre-trained model的参数矩阵
- $$y_i$$是item i的label

受[13]的启发，我们在sigmoid layer之前采用hidden vector作为personlized vector $$pv_i$$(如图1c所示)，feed到我们的PRM模型中。

图1c展示了pre-trained模型的可能架构，其它模型如：FM, FFM, DeepFM, DCN, FNN和PNN也可以做为生成PV的替代方法。

# 5.实验




# 参考

- 1.[https://dl.acm.org/citation.cfm?id=3346997](https://dl.acm.org/citation.cfm?id=3346997)