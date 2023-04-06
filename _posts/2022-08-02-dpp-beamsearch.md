---
layout: post
title: Determinantal Beam Search介绍
description: 
modified: 2022-08-02
tags: 
---


Clara Meister等在《Determinantal Beam Search》中提出了Determinantal Beam Search：


# 2.Neural Sequence Models

神经网络序列模型（Neural sequence models）是：**给定一个input x，在一个output space Y上的序列y的概率分布$$p(y \mid x)$$**。这里我们将Y定义成来自词汇表中的所有合法句子序列，以BOS开头，以EOS结尾。通常，序列长度由值$$n_{max} \in Z_+$$给定，它会依赖于x。在本文，我们会考虑局部归一化模型（locally normalized models），例如：给定之前已生成的tokens序列 $$y_{<t}$$ ，这里的p表示：是一个在$$\bar{V} = V \cup \lbrace EOS \rbrace $$的概率分布。完整序列$$y = <y_1, y_2, \cdots>$$的概率接着通过概率的chain rule进行计算：

$$
p(y | x) = \prod\limits_{t=1}^{|y|} p(y_t | y_{<t}, x)
$$

...(1)

其中，$$y_{<1}= y_0 = BOS$$。

我们的模型p通常通过一个具有weights $$\theta$$的neural network进行参数化。由于我们不关注底层模型本身，我们忽略掉p在参数$$\theta$$的依赖。

我们将decoding problem定义为：在空间Y上的所有序列间，根据模型$$y(y \mid x)$$搜索具有最高scoring的y，它也被称为**最大后验概率估计（maximum-a-posteriori（MAP）inference）**：

$$
y^{*} = \underset{y \in Y}{argmax} \  log p(y | x)
$$

...(2)

其中：惯例上，会使用p的log变换。

我们进一步将set decoding problem定义为：对于一个指定的基数k，在所有合法的subsets $$\lbrace Y' \subseteq Y \mid \mid Y'\mid=k \rbrace$$上，搜索一个具有最高分的set $$Y^*$$，定义为：

$$
p(Y | x) = \prod\limits_{y \in Y} p(y | x)
$$

...(3)

类似于等式(2)，set-decoding问题接着被定义为：

$$
Y^* = \underset{Y' \subseteq Y, |Y'|=k}{argmax} \ log p(Y' | x)
$$

...(4)

然而，由于需要注意的是，等式（2）和（4）有许多问题：

- 首先，因为Y是一个指数大的空间，并且p通常是非马尔可夫（non-Markovian）性的，我们不能进行有效搜索，更不用说$$Y^k$$。
- 第二，特别是对于语言生成任务，这些可能不是有用的目标

### 目标降级

需要重点注意的是：在 neural sequence models下具有最高概率解，并不总是高质量的（high-quality）；特别是涉及到语言生成的任务，比如：机器翻译等。相应的，启发式搜索方法或者一些替代目标通常会被用于decoding language generators.

## 2.1 Beam Search

用于逼近等式(2)的decoding problem的一种常见启发法是：**在每一timestep t上，以最大化 $$p(y_t \mid y_{<t}, x)$$的方法，顺序选择token $$y_t$$，直到EOS token被生成，或者达到最大序列长度$$n_{max}$$**。该过程被称为greedy search。Beam search是一种经常使用（oft-employed）的greedy search的生成方法，它会返回k个candidates，并探索更多search space。在本工作中，**我们关注于迭代式子集选择(iterative subset selection)的beam search**，它有一个很简洁的算法公式。给定一个初始集合$$Y_0$$，它只包含了BOS token，对于$$t \in \lbrace 1,\cdots,n_{max} \rbrace$$，我们会根据以下递归来选择子序列$$Y_t$$：

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/c0bfb415afc784695875362a0027d594ed6ef9fa3353d29e81c0a482b10bf3c7789bde5a82db844adcde60c4e906caf0?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1

其中，我们会限制，只扩展在beam set中的candidates，它被定义为：

$$
B_t = \lbrace y_{<t} \circ y |  y_{<t} \in Y_{t-1} \ and \ y \in \bar{V} \rbrace
$$

...(6)

其中：

- $$\circ$$会被用于表示string concatenations。

注意：在$$Y_{t-1}$$中的candidates已经以EOS结尾，会直接添加到$$B_t$$上，例如：$$EOS \circ EOS = EOS$$。在该定义下，我们有基数k的constraint：$$\mid B_t \mid \leq \mid \bar{V} \mid \cdot k$$.

## 2.2 Determinanta新公式

对于公式（5），我们接着引入另一种的等价概念，它使用matrics和determinants，它会阐明beam search的直接泛化（generation）。我们定义了一种timestep-dependent diagonal matrix $$D \in R^{\mid B_t \mid \times \mid B_t \mid}$$，其中：我们会采用diagonal entry:

$$
D_{ii} = p(Y_{\leq t}^{(i)} | x)
$$

...(7)

这里：

- $$y_{\leq t}^{(i)}$$：表示在$$B_t$$中的第i个candidate，它根据一个unique mapping：对于每个element $$y_{\leq t} \in B_t$$会唯一映射到一个介于1和$$\mid B_t \mid$$间的integer
- $$D_{Y_t}$$：来表示只包含了对应于$$Y_t$$的elements的相应行和列的submatrix，其中：$$Y_t \subseteq B_t$$

我们将等式(5)重写成：

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/5508c94c502f883862f97ff48e6c52a20e1aaad08246f290a9bc7d2b539c0edd565fa4d15e9e21352c477dea0866a62d?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2

这里的等式遵循对角阵行列式的定义。正式的，等式（8）被称为“子行列式最大化问题（subdeterminant maximization problem）”，该问题是发现一个行列式，它能最大化一个矩阵子集。而等式(8)引入的概念可能是人为的，它允许我们执行后续泛化。

# 3. Determinantal Beam Search

现在，我们会问该工作的基础问题：如果我们使用一个non-diagonal matrix来替换 diagonal matrix D，会发生什么？这种替换会允许我们对在beam中的elements间的交叉（interactions）做出解释。正式的，我们会考虑一个时间独立半正定矩阵（ timestep-dependent positive semi-definite (PSD) matrix）：$$D+w \cdot K$$，其中：对角矩阵（off-diagonal matrix）K表示在candidates间交叉的strength。该非负权重$$w \geq 0$$控制着在decoding过程中交叉（interactions）的重要性。在本case中，beam search递归变为：

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b17a103b4ed72e942553fea3a115398100844e0ad39b1ab8416bc8546132ed99220d4104b1696fbd4f995fc487937b95?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3

很明显，当w=0时我们会恢复成图2的beam search。然而，我们现在会基于在candidate interactions之上来选择子集。也就是说，等式（9）现在具有一个作为“diversity objective function”解释，当K被合适选择时。由于log的存在，当矩阵$$D_Y + w \cdot K_Y$$是PSD时，等式(9)会被良好定义。

## 3.1 K的构建

构建K的最简单方法是：Gram matrix，其中：每个i, j element会通过一个kernel function: $$K: S \times S \rightarrow R$$来计算，它会将空间中的两个items映射到一个实数上。特别的，我们会定义：$$K_{ij}=K(s_i, s_j)$$，其中$$s_i, s_j \in S$$是S的第i和第j个elements。概念上有些混洧，我们会该该kernel function K overload，它会采用一个set S，以便$$K = K(S)$$是在S的elements之上由pairwise计算的kernel matrix。根据Mercer理论，矩阵K=K(S)必须是PSD的，因为矩阵$$D_Y + w \cdot K_Y$$对于任意$$Y \subseteq S$$是PSD的。

## 3.2 与DPP关系

等式(9)是一个DPP。特别的，它是一个在L-ensemble parameterization上的k-DPP，其中：我们有$$L = D + w \cdot K$$。k-DPP的解释，对于为什么等式（8）是一个diverse beam search，给出了一个非常清晰的理解。对角的entries会编码quality，它会告诉我们：在beam上的每个candidate是多么“好”，而非对角entries（off-diagonal entries）则编码了两个elements有多么相似。

## 3.3 计算log-determinants

不幸的是，等式（9）中的argmax计算是一个NP-hard问题。然而，由于子行列式最大化问题（ subdeterminant maximization problem）具有许多应用，业界研究了许多高效算法来近似计算DPP中的log-determinants。Han et.2017使用一个关于log-determinant function的一阶近似。Chen et.2018使用一个贪婪迭代法；通过增量式更新matrix kernel的Cholesky factorization，该算法可以将infernence time减小到$$O(k^2 \mid S \mid)$$，并返回来自set S中的k个candidates。伪代码可以在Chen et.2018中找到，log-space算法的伪代码，可以在App.A中找到。

## 3.4 运行时(runtime)分析

我们会考虑：在给定任意时间，在等式（9）的递归上选择k个candidates的运行时。在每个timestep上，我们会首先构建一个matrix K。该计算高度依赖于被建模的interactions的集合；这样，当我们使用beam size为k时，$$O(c(k))$$是对于K计算的一个runtime上限。一旦我们构建矩阵$$D + w\cdot K$$，我们必须接着选择k个items。在任意timestep上的hypotheses集合至多是$$k \mid \bar{V} \mid$$。如3.3中讨论，我们假设采用近似算法，以最大化等式（9）的方式精准发现size-k的子集具有指数的runtime。使用Chen et.2018的方法，近似MAP inference会具有$$k^3\mid \bar{V} \mid$$的时间，从一个size为$$k \mid \bar{V} \mid$$的集合返回k个items。这样，在该条件下，determinantal
beam search的每轮迭代的runtime会是：$$O(c(k) + k^3 \mid \bar{V} \mid)$$。注意， standard beam search在每轮迭代会运行$$O(k \mid \bar{V} \mid log(k \mid \bar{V} \mid))$$。由于k通常很小（$$leq 20$$），c(k)的影响可以合理做出，在runtime上的实际增加通常是适中的。

# 4.Case Study: Diverse Beam Search

略略



- 1.[https://arxiv.org/pdf/2106.07400.pdf](https://arxiv.org/pdf/2106.07400.pdf)