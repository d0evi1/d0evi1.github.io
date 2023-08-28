---
layout: post
title: Non-invasive Self-attention介绍
description: 
modified: 2022-06-15
tags: 
---

华为在《Non-invasive Self-attention for Side Information Fusion in Sequential
Recommendation》中提出了Non-invasive Self-attention的方法：

# 抽要

Sequential recommender systems的目标是：根据用户历史行为建模用户的兴趣演进。对比起传统的模型，DNN等已经达到了较高的水准。最近BERT框架的出现，受益于它的self-attention机制，在处理序列数据上非常合适。然而，**original BERT框架只考虑单一输入源：在自然语言中的tokens**。在BERT框架下如何利用众多不同类型的information仍是一个开放的问题。尽管如此，它看起来可以直接利用其它的side information，比如：item category或者tag，来进行更综合的描述与更好的推荐。**在我们的实验中，我们发现naive方法（直接将不同side information的types进行融合到item embeddings中）通常会带来非常少或负向的效果**。因此，提出了NOninVasive self-Attention机制 (NOVA) 来在BERT框架下有效利用side information。NOVA会利用side informaiton来生成更好的attention分布，而非直接更改item embeddings，它会造成信息压倒性（information overwhelming）。我们验证了NOVA-BERT模型，并能达成SOTA效果，计算开销很小。

。。。

# 3.方法

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/927a5e2ec7c88da955a65dd92c33d0a4ddc9ec4184279a1963b86f210d49a3dc2ae8ee4691e82e795a2294fa35352226?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 invasive和non-invasive方法的一个图示。invasive方法会以不可逆的方式来融合所有类型的信息，接着将它们feed到sequential models上。而**对于non-invasive方法，side information只会参与attention matrix计算，item information会保存在一个独立的vector space中**

## 3.1 问题设定

**给定一个用户与系统的历史交互，顺序推荐（sequential recommendation）任务会询问：下一个要交互哪个item？ **

假设：u表示一个用户，它的**历史交互**可以被表示成一个按时间顺序的序列：

$$
S_u = [v_u^{(1)}, v_u^{(2)}, \cdots, v_u^{(n)}]
$$

其中：

- $$v_u^{(j)}$$：表示用户u做出的第j个交互行为

当只有一种类型的actions、并且没有side information时，每个interaction可以被简单表示成一个item ID：

$$
v_u^{(j)} = ID^{(k)}
$$

其中：

- $$ID^{(k)} \in I$$，表示第k个item ID。

$$
I = \lbrace ID^{(1)}, ID^{(2)}, \cdots, ID^{(m)} \rbrace
$$

其中：

- I是所有items的vocabulary。
- m是vocabulary size，表示在问题domain中的item总数

**给定一个user $$S_u$$的历史，系统会预估用户最可能交互的下一个item**：

$$
I_{pred} = ID^{(\hat{k})} \\
\hat{k} =\underset{k}{argmax} \ \ P(v_u^{(n+1)} = ID^{(k)} | S_u)
$$

## 3.2 Side Information

Side information可以是任意提供额外有用信息的东西，它可以被分类成两种类型：item-related或behavior-related。

- **Item-related side information**：是固有的，可以描述item本身，除了item IDs（例如：价格、生产日期、生产商）。 
- **Behavior-related side information**：是由一个user初始化的一个interaction，例如：action的类型（购买、评分）
、发生时间、用户反馈打分。

**每个交互的顺序（例如：原始BERT中的position IDs）可以被看成是一种behavior-related side information**。

如果side information引入进来，那么一个interaction就是：

$$
v_u^{(j)} = (I^{(k)}, b_{u,j}^{(1)}, \cdots, b_{u,j}^{(q)}) \\
I^{(k)} = (ID^{(k)}, f_k^{(1)}, \cdots, f_k^{(p)})
$$

其中：

- $$b_{u,j}^{(\cdot)}$$：表示一个由user u做出的第j个interaction的behavior-related side information。共有q个该类型的side information
- $$I^{(\cdot)}$$：表示一个item，包含了一个ID和一些item-related side information $$f_k^{(\cdot)}$$。共有p个该类型的side information
- 

Item-related side information是静态的，并存储了每个特定item的内在features。因而，vocabulary可以被重写成：

$$
I = \lbrace I^{(1)}, I^{(2)}, \cdots, I^{(m)} \rbrace
$$

该目标仍是预估下一个item的ID：

$$
I_{pred} = ID^{(\hat{k})} \\
\hat{k} = argmax_k P(v_u^{(n+1)} = (I^{(k)}, b_1, b_2, \cdots, b_q) | S_u)
$$

其中：

- $$b_1, b_2, \cdots, b_q$$：是latent behavior-related side information，如果behavior-related side information被考虑的话。注意：该模型仍能预估下一个item，而非 behavior-related side information被假设或忽略。

## 3.3 BERT和Invasive Self-attention

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/aab9f80c5bf844b75c43c3ca4e94c58b0d8f303cb6820e5eb6c283e54b297e4f3a0ce869978988b9ad41e5fefc0204ec?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2 BERT4Rec。Item IDs和positions会被分别编码成vectors，接着添加在一起作为integrated item representations. 在训练期间，item IDs会被随机mask掉（表示成[M]），以便让模型来进行恢复

BERT4Rec(Sun. 2019)是首个利用BERT框架进行sequential推荐并达到SOTA效果的任务。如图2所示，在BERT框架中，items被表示成vectors（embeddings）。在训练期间，一些items会被随机mask，BERT模型会使用multi-head self-attention机制来尝试恢复它们的vector表示以及item IDs：

$$
SA(Q,K,V) = \sigma(\frac{QK^T}{\sqrt{d_k}}) V
$$

其中：

- $$\sigma$$是softmax function
- $$d_k$$是一个scale factor
- Q, K, V分别表示query、key、value的组件

BERT会遵循一个encoder-decoder的设计，来为在input序列中的每个item生成一个contextual representation。BERT会采用一个embedding layer来保存m个vectors，每个对应于在vocabulary中的一个item。

为了利用side information，像conventional方法会使用独立的embedding layers来将side information编码到vectors中，接着使用一个fusion function F将它们fuse到ID embeddings中。这种invasive类型的方法会将side information注入到原始embeddings中，来生成一个mixed representation：

$$
E_{u,j} = F(\epsilon_{id}(ID), \\
\epsilon_{f1}(f^{(1)}), \cdots, \epsilon_{f_p}(f^{(p)}), \\
\epsilon_{b_1}(b_{u,j}^{(1)}), \cdots, \epsilon_{b_q}(b_{u,j}^{(q)}))
$$

其中：

- $$E_{u,j}$$：是user u对第j个interaction的integrated embedding
- $$\epsilon$$是将objects编码成vectors的embedding layer

该integrated embeddings的序列会被feed到模型中作为user history的input。

BERT框架会通过使用self-attention机制的layer来更新representations layer：

$$
R_{i+1} = BERT\_Layer(R_i) \\
R_1 = (E_{u,1}, E_{u,2}, \cdots, E_{u,n})
$$

在原始BERT和Transformer中，self-attention操作是一个位置不变函数（positional invariant funciton）。因此，一个position embedding会被添加到每个item embedding中来显式编码position信息。position mebeddings也可以被看成是一种behavior-related side information（例如：一个interaction的顺序）。从该视角看，original BERT也可以将positional information看成是唯一的side information，使用加法作为fusion function F。

图2 BERT4Rec

## 3.4 Non-invasive Self-attention (NOVA)

如果我们考虑end-to-end的BERT框架，它是一个具有stacked self-attention layers的auto-encoder。该identical embedding map会被用于encoding item IDs和decoding restored vector representations两者之上。因此，我们会讨论：invasive方法在混合embedding空间的缺点，因为item IDs会使用其它side information进行不可逆的混合。将来自IDs的信息与其它side information进行混合，使得对于模型编码item IDs来说带来不必要的困难。

相应的，我们提出了一种新的方法，称为noninvasive self-attention (NOVA)，来维持embedding space的一致性，从而利用side information建模sequences会更有效。该思想会修改self-attention机制，并仔细控制着self-attention组件(称为：query Q、key K、value V)的信息源。在在3.3节中定义的integrated embedding E之外，NOVA也会为pure ID embeddings保留一个分枝：

$$
E_{u,j}^{(ID)} = \epsilon_{id}(ID)
$$

因而，对于NOVA，用户历史会包含两个representations集合，pure ID embeddings和integrated embeddings：

$$
R_u^{(ID)} = (E_{u,1}^{(ID)}, E_{u,2}^{(ID)}, \cdots, E_{u,n}^{(ID)}) \\
R_u = (E_{u,1}, E_{u,2}, \cdots, E_{u,n})
$$

NOVA会计算来自 integrated embeddings R的Q、K，以及来自item ID embeddings $$E^{(ID)}$$的V。在实际中，我们会以tensor形式处理整个序列（。。。）。NOVA可以公式化为：

$$
NOVA(R,R^{(ID)}) = \sigma(\frac{QK^T}{\sqrt d_k}) V
$$

接着，Q,K,V可以通过线变变换进行计算：

$$
Q = RW_Q, K = RW_K, V=R^{(ID)} W_V
$$

对于side information fusing，NOVA与invasive方式间的对比如图3所示。layer by layer，在NOVA layers上的prepresentations会被保存到一个一致的vector space中，它完全由item IDs的context构成，$$E^{(ID)}$$。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/8468e8813b98fa3fb2e831003b0309f03125c186ecd1e6bd6790aad1370c73db78797390c47f088d04ae8887e7d864cb?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3

## 3.5 Fusion操作

NOVA会利用不同于invasive方法的side information，将它看成是一个auxiliary，并将side information进行fuse到具有fusion function F的Keys和Queries中。在本研究中，我们也研究了不同类型的fusion functions和它们的效果。

如上所示，position information也是一种behavior-related side information，并且original BERT会使用直接的addition操作来利用它：

$$
F_{add}(f_1, \cdots, f_m) = \sum\limits_{i=1}^m f_i
$$

再者，我们定义了“concat” fusor来将所有side information进行拼接，后接一个fully connected layer来对维度进行uniform：

$$
F_{concat}(f_1, \cdots, f_m) = FC(f_1 \odot \cdots \odot f_m)
$$

受(Lei 2019)的启发，我们设计了一个具有可训练参数的gating fusor：

$$
F_{gating}(f_1, \cdots, f_m) = \sum\limits_{i=1}^m G^{(i)} f_i \\
G = \sigma(FW^F)
$$

其中：

- F是所有features $$[f_1, \cdots, f_m] \in R^{m \times h}$$的矩阵形式
- $$W^F$$是一个可训练参数$$R^{h \times 1}$$
- h是要融合的feature vectors的维度 $$f_i \in R^h$$

## 3.6 NOVA-BERT

在图4所示，我们实现了我们的NOVA-BERT模型。每个NOVA layer会采用两个inputs，提供的side information以及item representations序列，接着输出相同shape的更新后的representations，它会被feed到下一layer中。对于第一层的input, item representations是纯item ID embeddings。因为我们只会用side information作为辅助来更好学习attention分布，side information不会沿着NOVA layers进行传播（propagate）。对于每个NOVA layer，side information的相同集合被显式提供。

NOVA-BERT会遵循original BERT的结构，除了将self-attention layers替换成NOVA layers。因而，额外的参数和计算开销会被忽略，它们主要由轻量级的fusion funciton引入。

我们相信，有了NOVA-BERT，hidden representations会保持在相同的embedding space中，它会让decoding处理一个同类型的vector search，这有利于prediction。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/5d4f72a9c7320b13091309c147aa02f2f8d218c28c64e0f7e331e890143c8873341c550f99a3109ce16c1714ede717c4?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=4.jpg&amp;size=750">

图4 NOVA-BERT。每个NOVA layer会采用两个inputs：item representations和side information

- 1.[https://arxiv.org/pdf/2103.03578.pdf](https://arxiv.org/pdf/2103.03578.pdf)