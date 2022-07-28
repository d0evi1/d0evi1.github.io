---
layout: post
title: 对偶增强双塔模型(DAT)介绍
description: 
modified: 2021-08-20
tags: 
---

美团在《A Dual Augmented Two-tower Model for Online Large-scale Recommendation》中提出对偶增强双塔模型（DAT）。

# 抽要

许多现代推荐系统具有非常大的item库（corpus），处理大规模检索的工业界方案是，使用two-tower模型来从content features中学习query和item表示。**然而，模型通常存在缺失two towers间的信息交互的问题。另外，不均衡的类目数据也会阻碍模型效果**。在本paper中，我们提出一个新的模型，称为对偶增强双塔模型（Dual Augmented two-tower model: DAT），它会集成一个新的**自适应模仿机制（Adaptive-Mimic-Mechanism）**以及一个**类目对齐Loss（Category Alignment Loss: CAL）**。我们的AMM会为每个query和item定制一个增强向量（augmented vector）来缓和信息交叉的缺失问题。再者，我们通过对不平衡的类目（uneven categories）进行对齐item representation，我们的CAL可以进一步提升效果。在大规模datasets上的离线实验表示了DAT的优越性。另外，在线A/B testings证实：DAT可以进一步提升推荐质量。

# 1.介绍

略

# 2.模型架构

## 2.1 问题声明

我们考虑一个推荐系统，它具有一个query set $$\lbrace u_i \rbrace_{i=1}^N$$以及一个item set $$\lbrace v_j \rbrace_{j=1}^M$$，其中：N是users数目，M是items数目。这里，$$u_i, v_j$$是许多features（例如：IDs和content features）的concatenations，由于稀疏性它可以是非常高维的。query-item feedback可以通过一个matrix $$R \in R^{N \times M}$$进行表示，其中：

- 当query i 给出在item j上的一个postive feedback时，$$R_{ij}=1$$；
- 否则为$$R_{ij}=0$$。

我们的目标是：给定一个特定query，从整个item corpus中有效选择可能的数千个candidate items。

## 2.2 对偶增强双塔模型

我们提出的模型框架如图1所示。DAT模型使用一个增强向量（augmented vector）$$a_u(a_v)$$来从其它tower中捕获信息，并将该vector看成是一个tower的input feature。另外，Category Alignment Loss会将从具有大量数据的category中学到知识并迁移到其它categories中。


<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/676ac9f50b8cc5a8bf504e170388ff78d5a1f6a45193df3f4362f4a245967f812ad8be417e34d83193ba82318a001249?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1 得出的对偶增强双塔模型的网络架构

### 2.2.1 Embedding Layer

与two-tower模型相似的是，在$$u_i$$和$$v_j$$中的每个feature $$f_i \in R$$（例如：一个item ID）会经过一个embedding layer，接着被映射到一个低维dense vector $$e_i \in R^K$$中，其中K是embedding维度。特别的，我们定义了一个embedding matrix $$E \in R^{K \times D}$$，其中：E通过学习得到，D是唯一特征数，embedding vector $$e_i$$是embedding matrix E的第i列。

### 2.2.2 Dual Augmented layer

对于一个特定query和candidate item，**我们会通过它们的IDs来创建相应的增强向量（augmented vectors）$$a_u$$和$$a_v$$**，并将它们与feature embedding vectors进行cancatenate一起来获得**增强后的输入向量（augmented input vectors）** $$z_u, z_v$$。例如，如果query u具有features "uid=253,city=SH,gender=male,..."，item v具有features "iid=149,price=10,class=cate,..."，我们有：

$$
z_u = [e_{253} || e_{sh} || e_{male} || \cdots || a_u] \\
z_v = [e_{149} || e_{p_{10}} || e_{cate} || \cdots || a_v ]
$$

其中:

- “||”表示向量连接操作符（concatenation op)

增强后的输入向量（augmented input vectors） $$z_u$$和$$z_v$$不仅包含了关于当前query和item的信息，**也包含了通过$$a_u$$和$$a_v$$的历史正交叉**。

接着，我们将$$z_u$$和$$z_v$$ feed到two towers上（它们由使用ReLU的FC layers组成），以便达到在由 $$a_u$$和$$a_v$$的two towers间的信息交叉。接着，FC layers的output会穿过一个L2 normalization layer来获得关于query $$p_u$$和item $$p_v$$的**增强后表示（augmented regresentations）**。正式的，two steps的定义如下：

$$
\begin{align}
h_1 & = ReLU(W_1 z + b), \cdots \\
h_L & = ReLU(W_l h_{L-1} + b_l) \\
p & = L2Norm(h_L)
\end{align}
$$

...(1)

其中：

- z表示$$z_u$$和$$z_v$$
- p表示$$p_u$$和$$p_v$$；
- $$W_l$$和$$b_l$$是第l层的weight matrix和bias vector；

two towers具有相同的结构但具有不同的参数。

再者，为了估计augmented vectors $$a_u$$和$$a_v$$，我们设计了一个Adaptive-Mimic Mechanism(AMM)，它会集成一个mimic loss和一个stop gradient策略。mimic loss的目标是，使用augmented vector来拟合在其它tower上的所有正交叉（postive interactions），它属于相应的query或item。对于label=1的每个样本，我们定义了mimic loss作为在augmented vector和query/item embedding $$p_u, p_v$$间MSE：

$$
loss_u = \frac{1}{T} \sum\limits_{(u,v,y) \in T} [y a_u + (1-y)p_v - p_v]^2 \\
loss_v = \frac{1}{T} \sum\limits_{(u,v,y) \in T} [y a_v + (1-y)p_u - p_u]^2
$$

...(2)

其中，T是在training dataset T中的query-item pairs数目，$$y \in \lbrace 0, 1 \rbrace$$是label。我们在接下来的子章节中讨论了标记过程（labeling process）。如上所示，如果label y=1, $$a_v$$和$$a_u$$会靠近query embedding $$p_u$$和item embedding $$p_v$$；如果label $$y=0$$, 则loss等于0. 如图1所示，augmented vectors被用于一个tower中，query/item embeddings会从另一个生成。也就是说，augmented vectors $$a_u$$和$$a_v$$会总结关于一个query或一个item与另一个tower相匹配的高级信息。因为mimic loss是为了更新$$a_u$$和 $$a_v$$，我们应将$$p_u$$和$$p_v$$的值进行freeze。为了这么做，stop gradient策略会用来阻止$$loss_u$$和$$loss_v$$反向梯度传播到$$p_v$$和$$p_u$$。

一旦获得两个augmented vectors $$a_u$$和$$a_v$$，他们会将它们看成了两个towers的input features，来建模在two towers间的信息交叉。最终，模型的output是query embedding和item embedding的inner product：

$$
s(u, v) = <p_u, p_v>
$$

其中，s(u,v)表示由我们的retrieval model提供的score。

### 2.2.3 Category Alignment

在工界业，items的categories会非常分散（例如：foods、hotels、movies等），每个category的items的数目是非常不均。有了这些不均衡的category数据，two-tower model会对于不同的categories表现不一样，在相对较小数目的items上效果要更差。为了解决该问题，我们在训练阶段提出了一个Category Alignment Loss（CAL），它会将具有较大数目的categories学到的知识迁移到其它categories上。特别的，对于每个batch，具有较大量数据的category的item representation $$p_v$$会被用来形成主要的category set：$$S^{major} = \lbrace p_v^{major}\rbrace$$，并于其它categories的$$p_v$$会形成它们各自的category sets：$$S^2, S^3, S^4, \cdots$$，我们定义了category alignment loss作为在major category和其它categories features间的二阶统计（协变量）间的距离：

$$
loss_{CA} = \sum\limits_{i=2}^n || C(S^{major}) - C(S^i)||_F^2
$$

...(3)

其中：

- $$\|\cdot \|_F^2$$表示平方矩阵Frobenius范数（squared matrix Frobenius norm）
- n表示categories数目
- $$C(\cdot)$$：表示 covariance matrix

## 2.3 模型训练

我们会将retrieval问题看成是一个二元分类问题，并采用一个随机负采样框架。特别的，对于在每个postive query-item pair（label=1）中的query，我们会从item corpus中随机采样S个items来创建具有该query的S个negative query-item pairs（label=0），接着添加这些S+1个pairs到training dataset中。对于这些pairs的cross-entropy loss如下：

$$
loss_p = - \frac{1}{T} \sum\limits_{(u,v,y) \in T} (y log \sigma(<p_u, p_v>)) + (1-y) log(1 - \sigma(<p_u, p_v>)) \\
T=D \times (S+1)
$$

...(4)

其中： 

- D是query-item pairs的postive feedback query-item pairs的数目
- T是train pairs的总数目
- $$\sigma(\cdot)$$表示sigmoid function

final loss function的公式如下：

$$
loss = loss_p + \lambda_1 loss_u + \lambda_2 loss_v + \lambda_3 loss_{CA}
$$

...(5)

其中，$$\lambda_1, \lambda_2, \lambda_3$$是可调参数。

# 3.实验

在本节中，我们会进行在线、离线实验来调整DAT设计的合理性。

## 3.1 Datasets

我们会在两个大规模数据集上评估： 一个从meituan的在线系统的日志中抽样得到，另一个来自Amazon[3]。

- Meituan dataset包含了连续11天的数据，前10天用于训练，第11天用于测试
- 我们会将前10天出现过的items组合在一起形成item corpus

Amazon Books dataset则相对较小，我们只保持至少被看过5次的items，以及至少看过5个items的用户。我们留下剩下的item作为testing。详细如表1所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/f8b92ed9a38adb987c357e35400f3e1cef711699059f5658d1a8be9f6c7bf20f003c03d0e35fde39c2805d1a894e0e43?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=t1.jpg&amp;size=750">

表1

## 3.2 实验设定

下面的方法被广泛用于工作界，用于与DAT model的对比：

- WALS: 
- YoutubeDNN:
- FM:
- Two-tower Model:
- MIND: 

我们通过使用distributed Tensorflow建模以使用及Faiss来从大规模item pool中检索top-N items。对于所有模型，embedding维度和batch size被固定到32-256维上。所有模型通过Adam optiizer进行训练。为了确保一个公平对比，所有模型的其它超参数，被单独调参以达到最优结果。对于DAT，每个tower的FC layers的数目被固定在3，维度为256、128、32. augmented vectors $$a_u$$和$$a_v$$被设置为 d=32，而$$\lambda_1, \lambda_2$$被设置为0.5，$$\lambda_3$$被设置为1。为了评估多个模型的offline效果，我们使用HitRate@K和MRR metrics，它们被广泛用于工业界的retrieval。其中K被置为50到100，因为retrieval module需要检索一个相当大数目的candidate items来feed给ranking module。由于大量的测试实例，我们采用一个MRR的归一化版本，factor=10.

## 3.3 离线结果

### 3.3.1 模型对比

表3所示

### 3.3.2 Augmented Vectors的维度

在DAT中的augmented vector在建模信息交叉上会扮演着一个主要角色，为了分析维度的影响，我们研究了在两个datasets上对应不同augmented vectors维度的DAT效果。如图2所示，在Meituan的DAT的效果提升会随着dimension的增加而获得提升，而在Amazon上的DAT效果提升只发生在首个位置，接着会下降。这是因为两个datasets的数据量的不同造成的。另外，忽略维度外，它总是能达到更好效果，这对augmented vector的有效性作出了解释。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/61bdcb95d3b26d83e8a60a34b636264c638880b7fdf1c5e0cf0db26ee19cd8fe108481efd4bde7a694fc80a58b7a5ff7?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2 在两个datasets上，在HR@100和MRR上的效果，随着augmented vectors的维度变化

### 3.4 在线实验

除了离线研究外，我们会通过部署DAT来处理一周的真实流量，系统每天会服务6000w用户。为了做出公平对比，retrieval stage会遵循相同的ranking procedure。在线实验的baseline方法是一个two-tower模型，它是base retrieval算法，会服务online traffic的主要流量。有上百个candidate items通过一个方法进行检索，并feed给ranking stage。图3展示了7个连续天的在线结果。我们的模型效果要胜过baseline一大截，在CTR、GMV上的整体平均提升分别为：4.17%、3.46%。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/98817cc5c3b68c90ea42174709c4411ad5fcced78b0f0300814ab9ecae82871eecd81bf4094551de7156497ce27f991b?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=3.jpg&amp;size=750">

图3 DAT的在线效果和baselines

# 4.结论

略

# 参考


- 1.[https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf)