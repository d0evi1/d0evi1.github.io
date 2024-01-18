---
layout: post
title: zero attention介绍
description: 
modified: 2022-06-10
tags: attention
---

Amazon search在《A Zero Attention Model for Personalized Product Search》中提出了zero attention的方法：

# 摘要

商品搜索（Product search）是人们在电子商务网站上发现和购买产品的最受欢迎的方法之一。由于个人偏好往往对每个客户的购买决策有重要影响，因此直觉上个性化（personalization）应该对商品搜索引擎非常有益。尽管以前的研究中人工实验显示：**购买历史（purchase histories）对于识别每个商品搜索会话（session）的个体意图是有用的，但个性化对于实际产品搜索的影响仍然大多未知**。在本文中，我们会将个性化商品搜索问题进行公式化，并从电商搜索引擎的搜索日志中进行了大规模实验。初步分析的结果表明，个性化的潜力取决于query特征(characteristics)、查询之间的交互（interactions）、以及用户购买历史（histories）。基于这些观察，我们提出了一个零注意力模型(Zero Attention Model)，用于商品搜索，通过一种新的注意力机制来自动确定何时以及如何对用户-查询对（user-query pair）进行个性化。商品搜索日志上的实证结果表明，所提出的模型不仅显著优于最先进的个性化商品搜索模型，而且提供了有关每个商品搜索会话中个性化潜力的重要信息。

# 4.ZERO ATTENTION模型

在本节中，我们提出了一种零注意力模型（ZAM）用于个性化商品搜索。 ZAM是在基于embedding的生成(generative)框架下设计的。它通过使用零注意力策略（Zero Attention Strategy）构建user profiles来进行查询相关的个性化，从而使其能够自动决定在不同的搜索场景中何时以及如何进行关注（attend）。理论分析表明，**我们提出的注意力策略（attention strategy）可以创建一个动态阈值，它可以根据查询和用户购买历史记录来控制个性化的权重**。

## 4.1 基于嵌入的生成框架 

隐语义模型（Latent semantic models）已被证明对于商品搜索和推荐非常有效[17, 39]。在不同类型的隐语义模型中，神经嵌入模型（neural embedding models）已经在许多基准商品搜索数据集上实现了最先进的性能[2, 16]。具体而言，Ai等人[2]提出了一种基于embedding的生成框架，可以通过最大化观察到的用户购买的可能性来共同学习query、user和item的embedding表示。

假设：

- q为用户u提交的query
- i为query q的候选项集$I_q$中的一个item
- $\alpha$为embedding向量的大小

在基于embedding的生成框架[22, 26]中，给定query q，用户u是否购买i的概率可以建模为：

$$
P(i|u,q) = \frac{exp(i · M_{uq})}{ Σ_{i' \in I_q} exp(i' \cdot M_{uq}) }
$$
... (3) 

其中:

- $i \in R^{\alpha} $是i的嵌入表示
- $M_{uq}$ 是user-query pair (u,q)的联合模型（joint model）

商品会根据$P(i \mid u,q)$进行排序，以便最大化每个搜索会话（session）中用户购买的概率。根据$M_{uq}$的定义，我们可以有多种基于embedding的商品搜索检索模型。在这里，我们介绍两个代表性模型：查询嵌入模型（ Query Embedding Model）、分层嵌入模型（Hierarchical Embedding Model）。

**查询嵌入模型**

查询嵌入模型（QEM）是一种基于embedding的生成模型，用于非个性化的产品搜索[2]。它将$M_{uq}$定义为：

$$
M_{uq} = q 
$$

...(4)

其中：

- $q \in R_{\alpha}$是查询q的embedding表示

由于查询通常事先不知道，在请求时必须在商品搜索中计算q。以前的研究[2, 39]探索了几种直接从查询词构建query embedding的方法。其中一种最先进的范例是使用非线性投影函数$\phi$对查询词进行编码来计算query embedding，定义为:

$$
q = \phi (\lbrace w_q |w_q \in q \rbrace) = tanh(W_ϕ · \frac{Σ_{w_q \in q} w_q}{|q|}+ b_{\phi})
$$

... (5) 

其中：

- $w_q \in R_{\alpha}$：是q中单词$w_q$的embeding
- |q|是查询的长度，
- $W_{\phi} \in R^{\alpha × \alpha}$和$b_{\phi} \in R^{\alpha}$是在训练过程中学习的两个参数

在QEM中，item embedding是从它们关联的文本数据中学习的。假设：$T_i$是与item i相关联的单词集合（例如标题）。Ai等人[2]提出通过优化观察到i时观察到$T_i$的似然来学习i，其中i的embedding表示为:

$$
P(T_i|i) = \prod_{w \in T_i} \frac{exp(w \cdot i)}{ \sum_{w' \in V} exp(w' \cdot i)}
$$

... (6) 

其中：

- $w \in R^a$是单词w的embedding
- V是所有可能单词的词汇表

注意，单独方式学习i，而不是通过平均单词嵌入来表示它是很重要的，因为：用户购买可能会受到除文本之外的其他信息的影响[2]。

**分层嵌入模型**

与QEM类似，HEM [2]也使用encoding函数$\phi$计算query embedding，并使用相关文本$T_i$计算item embedding。然而，与QEM不同的是，HEM将$M_{uq}$在公式（3）中定义为：

$$
Muq = q + u
$$

... (7) 

其中：

- u是用户u的嵌入表示

这样，HEM在产品搜索的项目排名中考虑了query意图和用户偏好。

在HEM中，用户u的embedding表示是通过优化观察到的用户文本$T_u$的似然$P(T_u \mid u)$获得的:

$$
P(T_u |u) = \prod_{w \in T_u} \frac{exp(w \cdot u)}{ \sum_{w' \in V} exp(w' \cdot u)}
$$

...(8)


其中:

- $T_u$可以是任何由用户u编写或输入相关的文本，例如产品评论或用户购买的项目描述。

## 4.2 Attention-based Personalization

正如Ai等人[2]所讨论的那样，HEM是基于用户偏好与查询意图在搜索中是独立的假设构建的。然而，在实践中，这并不是真实的情况。例如，**喜欢Johnson的婴儿产品的客户在搜索“男士洗发水”时可能不想购买Johnson的婴儿洗发水**。为了解决这个问题，产品搜索中更好的个性化范例是考虑查询和用户购买历史之间的关系。具体而言，我们在用户购买历史记录上应用一个attention函数来构建用于商品搜索的用户embedding。设$I_u$是用户u在查询q之前购买的项目集合，则我们可以计算u的embedding表示为（attention对应的Q: q, K: i, V: i）：

$$
u = \prod_{i \in I_u} \frac{exp(f(q,i))} { \sum_{i' \in I_u}  exp(f(q,i'))} i 
$$

... (9) 
 
其中:

- f(q,i)是一个attenion函数，用于确定每个item i相对于当前query q的注意力权重。类似于attention模型的先前研究[8, 41]，我们将f(q,i)定义为:

$$
f(q,i) = i · tanh(W_f · q + bf) ·Wh
$$

...(10) 

其中：

- $W_h \in R_{\beta}，W_f \in R_{\alpha \times \beta \times \alpha}，b_f \in R_{α \times β} $，β是一个超参数，用于控制注意网络中隐藏单元的数量。

给定基于注意力的用户嵌入u，我们可以使用公式（3）和公式（7）中描述的相同方法进一步进行个性化产品搜索。我们将这个模型称为基于注意力的嵌入模型（AEM）。

与HEM不同，AEM通过查询相关的用户个人资料进行个性化。每个用户的嵌入表示是根据他们的查询构建的，以便模型能够更好地捕捉当前搜索环境中相关的用户偏好。这在用户购买了许多与当前查询无关的产品时尤其有益。Guo等人[15]提出了另一个基于注意力的产品搜索模型，其思想与AEM类似。不幸的是，**对于注意力权重的计算，他们的模型假设用户购买历史记录中的每个项目都应与用户提交的至少一个搜索查询相关联，而这在我们的数据集中并不正确**。因此，在本文中我们忽略了它。

然而，有一个重要的问题限制了HEM和AEM的能力。如第3节所示，**不同的查询具有不同的个性化潜力**。尽管在查询相关的用户个人资料方面做出了努力，但在公式（9）中的注意机制要求AEM至少关注用户购买历史记录中的一个项目，这意味着它总是进行个性化。Ai等人[2]探索了一种朴素的解决方案，即在公式（7）中添加一个超参数来控制$M_{uq}$中用户嵌入u的权重，但这仅仅是在一些查询上进行个性化的收益与在其他查询上的损失之间进行权衡。**为了真正解决这个问题，我们需要一个模型，在产品搜索中可以自动确定何时以及如何进行个性化**。

## 4.3  Zero Attention Strategy

个性化的实用性取决于查询和用户的购买历史。例如，钓鱼查询通常会导致：无论是什么样的客户偏好，都会购买同一件物品。**同样，第一次在某个特定类别购物的客户，可能没有相关历史可供个性化基础**。

为解决这个问题，我们提出了一种零注意策略，**通过在注意过程中引入一个零向量（a Zero Vector）来放松现有注意机制的约束**。因此，我们提出了一种零注意模型（ZAM），它根据搜索查询和用户的购买历史在产品搜索中进行差异化个性化。图2显示了ZAM的结构。


图2

与AEM类似，ZAM基于相关单词来学习item embeddings，并使用query embeddings和user embeddings进行检索。ZAM和AEM之间的主要区别在于，**ZAM允许attention网络关注(attend)一个零向量（Zero Vector），而不仅仅是关注用户以前的购买记录，这被称为零注意策略**。形式上，$0 \in R_α$为零向量，其中每个元素都为0。然后，在ZAM中，用户u的嵌入表示计算为：

$$
 u = \sum\limits_{i \in I_u} \frac{exp(f(q,i))} {exp(f(q, 0)) + \sum\limits_{i' \in I_u} exp(f(q,i′))} i
$$

...(11)

其中：

- f(q,0)：是0相对于查询q的注意分数。

现在我们展示了这个简单的修改是如何在产品搜索中实现差异化个性化的。

- $x \in R^{\mid I_u \mid}$：是由${f(q,i) \mid i \in I_u}$形成的向量

然后，公式（11）可以重新表述为： 

$$
u = \frac{exp(x)}{exp(f(q, 0)) + exp^+(x)} \cdot I_u
$$

...(12) 

其中：

- $I_u$：是由用户购买历史中所有item embeddings组成的矩阵
- $exp^+(x)$：是exp(x)的逐元素和

在公式（10）中，$exp(f(q,0))=1$，因此公式（12）中的$I_u$因子实际上是x的sigmoid函数。换句话说，引入零注意策略创建了一个激活函数，控制用户购买历史在当前搜索上下文中的影响。$exp^+(x)$的值是用户先前购买的项目在给定query下接收到的**累积注意力**，而exp(f(q,0))实质上是个性化的阈值。尽管我们的公式是恒定的，但是可以通过使用更复杂的函数定义f来使这个阈值依赖于查询。在任何情况下，只有当用户在与当前查询相关的产品上表现出一致和显著的兴趣时，用户嵌入u才应在ZAM中具有强烈的影响力。这使得ZAM能够在不同的搜索场景中进行差异化个性化。

## 4.4 模型优化

与先前的研究[2,39,44]类似，我们通过最大化观察到的用户购买和item信息的对数似然来优化AEM和ZAM。具体而言，我们用它们的标题表示每个item，用它们以前购买的item表示每个用户，用它们的查询词表示每个查询。假设：$T_i$是项目i标题中的单词列表，那么观察到的user-query-item三元组的对数似然可以计算为：

$$
L(T_i,u,i,q) = log P(T_i|i) + logP(i|u,q) + log P(u,q) \\
    \approx \sum\limits_{w_i \in T_i} log \frac{exp(w_i·i)}{\sum_{w′ \in V} exp(w′ \cdot i)} \\
    + log \frac{exp(i \cdot (q + u))}{ \sum_{i′ \in I_q} exp(i′ \cdot (q+u))} 
$$

...(13) 

其中：

- w和i是学习的参数，q用公式（5）计算，u用公式（9）（对于AEM）或公式（11）（对于ZAM）计算，忽略了$log P(u,q)$，因为训练样例从日志中均匀采样。

然而，由于V中的单词数量和$I_q$中的item数量很大，计算$L(T_i,u,i,q)$通常是不可行的。为了有效训练，我们采用负采样策略[1,22,25]来估计公式（13）。具体来说，对于每个softmax函数，我们采样k个负样本来近似它的分母。AEM和ZAM的最终优化函数可以表示为 

$$
L′ = \sum\limits_{(u,q,i)} L(T_i, u, i, q) \approx \sum\limits_{(u,q,i)}\sum\limits_{w_i \in T_i} (log σ(w_i \cdot i) + k \cdot E_{w′∼P_w}[log σ(−w′·i)]) \\
+ log σ(i \cdot (q + u)) + k \cdot E_{i′ \sim P_{I_q}} [log σ(−i′)\cdot(q + u))]
$$

其中:

- $\sgmoid(x)$是sigmoid函数，
- $P_w$和$P_{I_q}$分别是单词w和item i的噪声分布

在我们的实验中，我们将$P_w$定义为提高3/4次方的单词分布[25]，将$P_{I_q}$定义为均匀分布。此外，如图2所示，query和item标题中的单词以及$I_u$和$I_q$中的item共享一个公共embedding空间。我们尝试了不同类型的正则化技术，如L2正则化，但没有一种技术对模型的效果产生了显著影响。这表明，在我们拥有大规模训练数据的情况下，过度拟合不是我们实验中模型优化的问题。因此，为简单起见，我们忽略公式14中的正则化项。

# 其它

略

- 1.[https://arxiv.org/pdf/1908.11322.pdf](https://arxiv.org/pdf/1908.11322.pdf)