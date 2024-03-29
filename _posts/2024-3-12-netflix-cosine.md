---
layout: post
title: Netflix关于cosine相似度的讨论
description: 
modified: 2024-3-12
tags: 
---

Netflix团队发了篇paper《Is Cosine-Similarity of Embeddings Really About Similarity?》，对cosine相似度做了相应的研究。

# 摘要

余弦相似度（cosine similarity）是指两个向量间夹角的余弦值，或者等价于：**归一化后的点积**。一种常见的应用是：通过将余弦相似度应用于学习到的低维特征embedding，来量化高维对象之间的语义相似性。**这种方法在实践中可能比未归一化的嵌入向量之间的点积效果更好，但也可能更差**。为了深入了解这一经验观察，我们研究了从正则化线性模型派生的embedding，其中闭式解有助于分析洞察。**我们从理论上推导了余弦相似度如何产生任意且因此无意义的“相似性”**。对于某些线性模型，相似性甚至不是唯一的，而对于其他模型，它们则由正则化隐式控制。我们讨论了超出线性模型的含义：在学习深度模型时采用了不同正则化的组合；当对结果embedding取余弦相似度时，这些正则化具有隐式和非预期的影响，使结果变得不透明且可能是任意的。基于这些见解，我们警告：**不要盲目使用cosine相似度**，并概述了替代方案。

# 1.引言

离散实体通常通过学习的映射嵌入到各种领域的**稠密实值向量(dense real-valued vector)**中。例如，在大语言模型（LLM）中，单词基于其周围上下文进行嵌入，而推荐系统通常根据用户消费的方式学习item（和user）的embedding。这样的embedding有多方面的优点。特别是，它们可以直接作为（冻结或微调的）输入用于其它模型，它们提供了一种数据驱动的（语义）相似性概念，用来表示之前是原子和离散的实体。

虽然“余弦相似度（cosine similarity）”中的相似性指的是：**与距离度量中的较大值表示更接近（较小值则相反）**，但它也已成为衡量感兴趣实体之间语义相似性的非常流行的度量方法。其动机在于，学习到的**embedding vector的范数**并不如**embedding vector间的方向**对齐那么重要。尽管有无数的论文报告了余弦相似度在实际应用中的成功使用，**但也有人发现它在某些情况下不如其它方法，例如学习embedding间的（未归一化的）点积，参见[3, 4, 8]**。

在本文中，我们尝试阐明这些不一致的经验观察。我们发现，学习到的embedding余弦相似度实际上可以产生任意结果。我们发现，根本原因不在于余弦相似度本身，**而在于学习到的embedding具有一定程度自由度，即使它们的（未归一化的）点积是明确定义且唯一，也可以产生任意的余弦相似度**。为了获得更具一般性的见解，我们推导出解析解，这对于线性矩阵分解（MF）模型是可能的——这将在下一节详细概述。在第3节中，我们提出了可能的解决方案。第4节中的实验说明了我们在本文中得出的发现。

# 2.矩阵分解模型

在本文中，我们关注线性模型，因为它们允许**闭式解（closed-form solutions）**，从而可以从理论上理解应用于学习embedding的余弦相似度度量的局限性。给定：

- 一个矩阵$X \in R^{n × p}$
- 包含n个数据点和p个特征（例如，在推荐系统中分别是user和item）

矩阵分解（MF）模型（或等效地在线性自编码器中）的目标是：估计一个低秩矩阵$AB^T \in R^{p×p}$

其中：

- $A, B \in R^{p×k}, k \leq p$

使得乘积$XAB^⊤$是${X:}^1 X \approx XAB^⊤$的好的近似。

给定：

- X是一个user-item矩阵
- B的行：$\overset{\rightarrow}{b_i}$，通常被称为k维的item embedding
- XA的行：$\overset{\rightarrow}{x_u} \cdot A$，可以解释为user embeddings，其中用户u的embedding是该用户消费的item embeddings $\overset{\rightarrow}{a_j}$的总和。

请注意，该模型是根据user和item embeddings之间的（未归一化的）点积定义的：

$$
(XAB^T)_{u,i} = < \overset{\rightarrow}{x_u} \cdot A,  \overset{\rightarrow}{b_i} >
$$

然而，一旦学习了embedd，常见的做法是：考虑它们之间的余弦相似度，例如：

- 两个item间：$cosSim(\overset{\rightarrow}{b_i}, \overset{\rightarrow}{b'_i})$ 
- 两个user间：$cosSim(\overset{\rightarrow}{x_u} \cdot A, \overset{\rightarrow}{x_u'} \cdot A)$
- user与item间：$cosSim(\overset{\rightarrow}{x_u} \cdot A, \overset{\rightarrow}{b_i})$

在下文中，我们将展示这可能导致任意结果，并且它们甚至可能不是唯一的。

## 2.1 训练 

影响余弦相似度metric的实效（utility）的一个关键因素是：当在学习A、B的embedding时使用的正则化方法，如下所述。

考虑以下两种常用的正则化方案（它们都有封闭形式的解，见第2.2节和第2.3节）： 

$$
\underset{A,B}{min} ||X − XAB^⊤||^2_F + λ||AB^⊤||^2_F \\
\underset{A,B}{min} ||X − XAB^⊤||^2_F + λ(||XA||^2_F + ||B||^2_F ) 
$$

... (1) (2)

这两个训练目标在L2范数正则化方面显然有所不同：

在第一个目标中，$\|AB^⊤\|^2_F$ 应用于它们的乘积。在线性模型中，这种L2范数正则化可以证明等同于：**使用去噪学习，即在输入层进行dropout**，例如，见[6]。 此外，实验发现，在保留的测试数据上得到的预测准确率优于第二个目标的准确率[2]。 不仅在MF模型中，而且在深度学习中，通常观察到去噪或dropout（这个目标）比权重衰减（第二个目标）在保留的测试数据上带来更好的结果。 


第二个目标等价于：常规的矩阵分解目标:

$$
{min}_W \| X − P Q^T \|^2_F + λ(\|P\|^2_F + \|Q\|^2_F)
$$

其中：

- X被分解为$P Q^⊤$，且P = XA和Q = B。

这种等价性在 [2]中有所概述。这里的关键是，每个矩阵P和Q分别进行正则化，类似于深度学习中的权重衰减。 

- $\widehat{A}$和$\widehat{B}$：是任一目标的解（solution）
- $R \in R^{k×k}$：任意旋转矩阵

那么众所周知，具有任意旋转矩阵$R \in R^{k×k}$ 的$\widehat{A}R$和$\widehat{B}R$也是解（solution），因为**余弦相似度在这种旋转R下是不变的**，本文的一个关键见解是：

- 第一个（但不是第二个）目标对于A和B的列（即嵌入的不同潜在维度）的重缩放也是不变的：如果$\widehat{A} \widehat{B}^⊤$是第一目标的解，那么$\widehat{A}DD^−1 \widehat{B}^⊤$也是，其中D ∈ R k×k 是任意对角矩阵。

因此，我们可以定义一个新的解决方案（作为D的函数）如下： 

$$
\widehat{A}^{(D)} := \widehat{A}D \\
\widehat{B}^{(D)} := \widehat{B}D^{−1} 
$$

...(3)

反过来，这个对角矩阵D会影响学习到的user和item embedding（即：行）的归一化：

$$
(X\widehat{A}^{(D)})_{(normalized)} = Ω_AX\widehat{A}^{(D)} = Ω_AX\widehat{A}D \\
\widehat{B}^{(D)}_{(normalized)} = Ω_BBˆ(D) = ΩBBDˆ −1，(4)
$$

其中$Ω_A$和$Ω_B$是适当的对角矩阵，用于将每个学习到的嵌入（行）归一化为单位欧几里得范数。注意，一般来说这些矩阵不可交换，因此不同的D选择不能（精确地）通过归一化矩阵$Ω_A$和$Ω_B$来补偿。由于它们依赖于D，我们通过$Ω_A(D)$和$Ω_B(D)$明确表示这一点。因此，嵌入的余弦相似性也取决于这个任意矩阵D。

当人们考虑两个项目之间、两个用户之间或用户和项目之间的余弦相似性时，这三种组合分别为： 

- item-item： 

$$
cosSim(\widehat{B}^(D), \widehat{B}^(D)) = Ω_B(D) \cdot \widehat{B} \cdot D^{−2} \cdot \widehat{B}^T \cdot Ω_B(D)
$$ 

- user-user： 

$$
cosSim(X\widehat{A}^(D), X\widehat{A}^(D)) = Ω_A(D) \cdot X\widehat{A}^ \cdot D^2 \cdot (X\widehat{A})^T \cdot Ω_A(D)
$$

- user-item： 

$$
cosSim(X\widehat{A}^(D), \widehat{B}^(D)) = Ω_A(D) \cdot X\widehat{A} \cdot \widehat{B}^T \cdot Ω_B(D)
$$

显然，所有三种组合的余弦相似性都取决于任意对角矩阵D：虽然它们都间接依赖于D，因为它影响了归一化矩阵$Ω_A(D)$和$Ω_B(D)$，但请注意，（特别受欢迎的）item-item余弦相似性（第一行）还直接依赖于D（user-user余弦相似性也是如此，见第二项）。

# 2.2 First Objective (Eq. 1)详述

当我们考虑全秩MF模型的特殊情况，即k = p时，余弦相似性的任意性在这里变得尤为明显。这可以通过以下两种情况来说明：

第一种：

如果我们选择：
$$
D = dMat(..., 1/(1+λ/σ^2)^i, ...)^(1/2)
$$

那么我们有:
$$
\widehat{A}_{(1)}^{(D)} = \widehat{A}_{(1)} \cdot D \\
= V · dMat(\cdots, \frac{1}{(1+λ/\sigma_i^2)}, \cdots)
$$

和 

$$
\widehat{B}_{(1)}^{(D)} = \widehat{B}_{(1)} \cdot D^{-1} = V
$$

由于奇异向量矩阵V已经是标准化的（关于列和行），归一化$Ω_B = I$因此等于单位矩阵I。因此，关于item-item余弦相似性，我们得到： 

$$
cosSim(\widehat{B}_{(1)}^{(D)}, \widehat{B}_{(1)}^{(D)}) = V V^T = I
$$

这是一个相当奇怪的结果，因为这意味着任何一对（不同的）项目嵌入之间的余弦相似性为零，即一个item只与自己相似，而不与任何其他item相似！ 

另一个显著的结果是关于user-item余弦相似性： 

$$
cosSim(X \widehat{A}_{(1)}^{(D)}, \widehat{B}_{(1)}^{(D)}) = Ω_A \cdot X \cdot V \cdot dMat(\cdots, \frac{1}{1 + λ/\sigma_i^2}, \cdots) · V^T \\
= Ω_A · X · \widehat{A}_{(1)}\widehat{B}_{(1)}^T
$$

因为与（未归一化的）点积相比，唯一的区别在于矩阵$Ω_A$，它归一化了行——因此，当我们考虑基于预测分数为给定用户对项目进行排序时，余弦相似性和（未归一化的）点积会导致完全相同的项目的排序，因为在这种情况下行归一化只是一个无关紧要的常数。

第2种：

- 如果我们选择：

$$
D = dMat(\cdots, \frac{1}{(1+λ/σ_i^2)}, \cdots)^{-\frac{1}{2}}
$$

那么我们类似于前一种情况有：

$$
\widehat{B}_{(1)}^{(D)} = V \cdot dMat(\cdots, \frac{1}{1+λ/σ_i^2}, \cdots)
$$

并且$\widehat{A}_{(1)}^{(D)} = V$是正交的。我们现在得到关于user-user余弦相似性： 

$$
cosSim(X \widehat{A}_{(1)}^{(D)}, X\widehat{A}_{(1)}^{(D)}) = Ω_A · X · X^T · Ω_A
$$

即，现在用户相似性仅仅基于原始数据矩阵X，即没有任何由于学习到的嵌入而产生的平滑。关于user-item余弦相似性，我们现在得到：

$$
cosSim(X\widehat{A}_{(1)}^{(D)}, \widehat{B}_{(1)}^{(D)}) = Ω_A \cdot X \cdot \widehat{A}_{(1)} \cdot \widehat{B}_{(1)}^T \cdot Ω_B
$$

即，现在$Ω_B$归一化了B的行，这是我们在之前选择D时所没有的。同样，item-item余弦相似性

$$
cosSim(\widehat{B}_{(1)}^{(D)}, B_{(1)}^{(D)}) = Ω_B · V · dMat(\cdots, \frac{1}{1 + λ/σ_i^2}, \cdots)^2 \cdot V^T \cdot Ω_B 
$$

与我们之前在D的选择中得到的奇怪结果大不相同。

总的来说，这两种情况表明，对于D的不同选择会导致不同的余弦相似性，即使学习到的模型

$$
\widehat{A}_{(1)}^{(D)} \widehat{B}_{(1)}^{(D)T} = \widehat{A}_{(1)} \widehat{B}_{(1)}^T
$$

对于D是不变的。换句话说，余弦相似性的结果是任意的，对这个模型来说并不是唯一的。

## 2.3 关于第二个目标

（公式2）的细节 

公式2中的训练目标的解决方案在[7]中推导出来，读作

$$
\widehat{A}_{(2)} = V_k \cdot dMat(\cdots, \sqrt{\frac{1}{σ_i} \cdot (1 - λ/σ_i)+}, \cdots)_k \\ 
\widehat{B}_{(2)} = V_k \cdot dMat(\cdots, \sqrt{σ_i \cdot (1 - λ/σ_i)+}, \cdots)_k
$$ 

... (6) 
 
其中：

- $(y)+ = max(0, y)$
- $X =: U \Sigma V^T$: 是训练数据X的SVD
- $\Sigma = dMat(\cdots, σ_i, \cdots)$

注意，如果我们使用MF中常用的符号，其中：$P = XA$和$Q = B$，我们得到：

$$
\widehat{P} = X\widehat{A}_{(2)} = U_k \cdot dMat(\cdots, \sqrt{σ_i \cdot (1 - \frac{λ}{σ_i})+}, \cdots)_k
$$

在这里我们可以看到，在公式6中，对角矩阵：

$$
dMat(..., \sqrt{σ_i \cdot (1 - \frac{λ}{σ_i})+}, \cdots)_k
$$

对于user embedding和item embedding是相同的，这是由于在公式2的训练目标中的L2范数正则化 $\|P\|_F + \|Q\|_F$的对称性所预期的。

与第一个训练目标（见公式1）的关键区别在于，这里的L2范数正则化$\|P\|_F + \|Q\|_F$是分别应用于每个矩阵的，因此这个解决方案是唯一的（如上所述，直到不相关的旋转），即在这种情况没有办法将任意的对角矩阵D引入到第二个目标的解决方案中。因此，应用于这个MF变体的学习嵌入的余弦相似性产生唯一的结果。

虽然这个解决方案是唯一的，但它仍然是一个悬而未决的问题，这个关于用户和项目嵌入的唯一对角矩阵 $dMat(\cdots, \sqrt{σ_i \cdot (1 - λ/σ_i)+}, \cdots)_k$是否在实践中产生最佳可能的语义相似性。然而，如果我们相信这种正则化使得余弦相似性在语义相似性方面有用，我们可以比较两个变体中对角矩阵的形式，即比较公式6和公式5，这表明在第一个变体中任意的对角矩阵D（见上面的部分）类似地可以选择为：
$D = dMat(..., p1/σi, ...)k$

# 3.针对余弦相似性的补救措施和替代方法 

正如我们上面分析的那样，当一个模型针对点积进行训练时，其对余弦相似性的影响可能是模糊的，有时甚至不是唯一的。一个显而易见的解决方案是针对余弦相似性训练模型，层归一化[1]可能会有所帮助。另一种方法是避免使用导致上述问题的嵌入空间，并将其投影回原始空间，然后在那里应用余弦相似性。例如，使用上述模型，并给定原始数据X，可以将$X\widehat{A}\widehat{B}^T$视为其平滑版本，将$X\widehat{A}\widehat{B}^T$的行视为原始空间中的user embedding，然后可以应用余弦相似性。

除此之外，同样重要的是要注意，在余弦相似性中，只有在学习了嵌入之后才应用归一化。与在学习之前或期间应用某种归一化或减少流行度偏差相比，这可能会显著降低结果的（语义）相似性。这可以通过几种方式完成。例如，统计学中的一种默认方法是标准化数据X（使每列均值为零且方差为单位）。深度学习中的常见方法包括使用负采样或逆倾向缩放（IPS）来考虑不同项目的流行度（和用户活动水平）。例如，在word2vec [5]中，通过按照它们在训练数据中的频率（流行度）的β = 3/4次幂的概率采样负样本，训练了一个矩阵分解模型，这在当时产生了令人印象深刻的词相似性。


[https://arxiv.org/pdf/2403.05440v1.pdf](https://arxiv.org/pdf/2403.05440v1.pdf)