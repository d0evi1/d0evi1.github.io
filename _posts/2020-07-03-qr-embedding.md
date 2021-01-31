---
layout: post
title: md embedding介绍
description: 
modified: 2020-07-01
tags: 
---

facebook在2019的《Compositional Embeddings Using Complementary Partitions
for Memory-Efficient Recommendation Systems》，提出了一种compositional embedding，并且在dlrm中做了实现，我们来看下具体的概念。



# 1.介绍

DLRMs的设计是为了处理大量categorical (or sparse) features的情况。对于个性化或CTR预估任务，categorical features的示例可以包括：users、posts、pages、以及成百上千的这些features。在每个categorical feature中，categories的集合可能会有许多多样性的含义。例如，社交媒体主页（socal media pages）可能包含的主题范围是：从sports到movies。

为了利用这些categorical信息，DLRMs利用embeddings将每个category映射到在一个embedded空间中的一个唯一的dense representation；见[2,4,5等]。更精确的，给定一个关于categories的集合S以及它的基数 $$\mid S \mid$$，每个categorical实例会被映射到一个在一个embedding table $$W \in R^{\mid S \mid \times D}$$的indexed row vector上，如图1所示。我们不用预先决定embedding的weights，对于生成准确的模型，在neural network的其余部分对embeddings进行jointly training更有效。

每个categorical feature，可有具有数千万可能不同的categories（比如：$$\mid S \mid \approx 10^7 $$），采用的embedding vector的维度为$$D \approx 100$$。在DLRM的training和inference期，由于存在大量的categories，每个table可能需要多个GBs进行存储，因此embedding vectors的数目构成了主要的内存瓶颈。

一种减小内存需求的天然方法是，通过定义一个hash函数（通常是余项函数：remainder function）来减小embedding tables的size，它可以将每个category映射到一个embedding index上，其中embedding size会比categories的数目要更小。然而，该方法会将许多不同的categories映射到相同的embedding vector上，从而导致在信息的丢失以及在模型质量上变差。理想情况下，我们应减小embedding tables的size，并且仍为每个category生成唯一的representation，从而尊重数据的天然多样性。

在本paper中，我们提出了一种方法，它通过对caegory set使用complementary partitions来生成compositional embeddings，来为每个categorical feature生成唯一的embedding。这些compositional embeddings可以与多个更小的embeddings交互来生成一个final embedding。这些complementary partitions可以从categorical data的天然特性中获取，或者人工强制来减小模型复杂度。我们提出了具体的方法来人工定义这些complementary partitions，并演示了在一个modified DCN以及Facebook DLRM networks在Kaggle Criteo Ad Display Chaalenge dataset上是有用的。这些方法很容易实现，可以在training和inference上同时压缩模型，无需其它额外的pre-或post-training处理，比hashing trick能更好地保留模型质量。

## 1.1 主要贡献

主要有：

- quotient-remainder：。。。
- complementary partitions：
- 更好的实验效果：

# 2.商&余数 trick（QUOTIENT-REMAINDER TRICK）

回顾DLRM setup中，每个category会被映射到embedding table中的一个唯一的embedding vector上。数学上，考虑单个categorical feature，假设：$$\epsilon: S \rightarrow \lbrace 0, \cdots, \mid S \mid -1 \rbrace $$表示S的一个枚举(enumeration)（例如：一个categories集合S包括 S={dog, cat, mouse}, 接着S的一个潜在枚举enumeration：$$ \ epsilon (dog)=0, \epsilon (cat)=1, \ epsilon (mouse)=2$$。假设$$W \in R^{\mid S \mid \times D}$$是相应的embedding matrix或table，其中D是embeddings的维度。我们可以使用$$$$e_i \in R^{\mid S \mid}$$$$将每个category（或者说：category $$x\in S$$具有index $$i=e(x)$$）编码成一个one-hot vector，接着将它映射到一个dense embedding vector $$x_{emb} \in R^D $$上：

$$
x_{emb} = W^T e_i
$$

...(1)

另外，该embedding可以被解释成embedding table上的一个单行lookup，例如：$$x_{emb} = W_i,:$$。注意，这会产生一个$$O(\mid S \mid D)$$的内存复杂度来存储embeddings，当$$\mid S \mid$$很大时这会变得非常受限。

减小embedding table的naive方法是，使用一个简单的hash function[17]，比如：remainder function，这称为hashing trick。特别的，给定一个size为$$m \in N$$（其中, $$m \ll \mid S \mid$$）的embedding table，也就是说，$$\sim{W} \in R^{m \times D}$$，你可以定义一个hash matrix $$R \in R^{m \times \mid S \mid}$$：

$$

$$

...(2)

接着，该embedding通过下面执行：

$$
x_{emb} = \sim{W}^T Re_i
$$

...(3)

该过程可以通过算法1进行归纳：

算法1

尽管该方法可以极大减小embedding matrix的size，由于$$m \ll \mid S \mid$$, 从$$O(\mid S \mid D)$$减小到$$O(mD)$$，它可以天然地将多个categories映射到相同的embedding vector，从而产生信息丢失以及模型质量上的下降。一个重要的observation是，该方法不会为每个unique category生成一个unique embedding，从而不会遵循categorical data的天然多样性。

为了克服这一点，我们提出了quotient-remainder trick。出于简洁性，m除以$$\mid S \mid$$。假以"\"表示整除或商（quotient）操作。使用两个complementary functions（integer quotient function和remainder function），我们可以生成两个独立的embedding tables，对于每个category，两个embeddings组合起来可以生成unique embedding。如算法2所示。

算法2

更严格的，我们定义了两个embedding矩阵：$$W_1 \in R^{m \times D}$$和$$W_2 \in R^{(\mid S \mid/m) \times D}$$。接着定义一个额外的hash矩阵$$Q \in R^{(\mid S \mid /m) \times \mid S \mid}$$：

$$
$$

...(4)

接着，我们可以通过以下方式获取我们的embedding：

$$
x_{emb} = W_1^T R e_i \odot W_2^T Q e_i
$$

...(5)

其中，$$\odot$$表示element-wise乘法。该trick会产生一个$$O(\frac{\mid S \mid}{m} D + mD)$$的内存复杂度，它对比起hashing trick在内存上会有微小的增加，但可以生成unique representation。我们在第5节展示了该方法的好处。

# 3.COMPLEMENTARY PARTITIONS

 quotient-remainder trick只是decomposing embeddings的一个更通用框架下的一个示例。注意，在 quotient-remainder trick中，每个操作（quotient或remainder）会将categories集合划分成多个"buckets"，以便在相同"bucket"中的每个index可以被映射到相同的vector上。然而，通过将来自quotient和remainder的两个embeddings组合到一起，可以为每个index生成一个distinct vector。
 
 相似的，我们希望确保在category set中的每个element可以生成它自己的unique representation，即使跨多个partitions。使用基础的set theory，我们可以将该概念公式化成一个称为“complementary partitions”的概念。假设$$[x]_p$$表示通过partition P索引的$$x \in S$$的等价类。
 
 定义1: 。。。
 
 作为一个具体示例，考虑集合 $$S=\lbrace 0,1,2,3,4 \rbrace$$。接着，以下三个set partitions是complementary：
 
    {{0}, {1,3,4},{2}}, {{0,1,3}, {2,4}}, {{0,3},{1,2,4}}
 
特别的，根据这些partitions中至少一个，你可以确认每个element与其它element是不同的。
 
 注意，一个给定partition的每个等价类指定了一个“bucket”，它可以映射到一个embedding vector上。因而，每个partition对应到单个embedding table上。在complementary partitions下，在对来自每个partitions的每个embedding会通过一些操作进行组合之后，每个index会被映射到不同的embedding vector上，如第4节所示。
 
 ## 3.1 Complementary Partitions示例
 
 使用complementary partitions的定义，我们可以抽象quotient-remainder trick，并考虑其它更通用的complementary partitions。这些示例会在附录中提供。出于简化概念，对于一个给定的$$n \in N$$，我们定义了集合：$$\Epsiion(n) = \lbrace 0, 1, \cdots, n-1 \rbrace$$

(1) Naive Complementary Partition：

$$
P = \lbrace \lbrace x \rbrace: x \in S \rbrace 
$$

如果P满足上式，那么P就是一个Complementary Partition。这对应于一个full embedding table，它的维度是：$$\mid S \mid \times D$$。

(2)  Quotient-Remainder Complementary Partitions:

给定$$m \in N$$，有：

$$
P_1 = \lbrace \lbrace x \in S: \epsilon(x)\m=l \rbrace: l \in \epsilon(\lceil |S| /m \rceil) \rbrace &&
P_2 = \lbrace \lbrace x \in S: \epsilon(x) mod m = l \rbrace: l \in \epsilon(m) \rbrace
$$

这些partitions是complementary的。这对应于第2节中的quotient-remainder trick。

(3) Generalized Quotient-Remainder Complementary Partitions:

对于$$i=1, \cdots, k$$，给定$$m_i \in N$$，以便$$\mid S \mid \leq \prod\limits_{i=1}^{k} m_i$$，我们可以递归定义complementary partitions：

$$
P_1 = \lbrace \lbrace x \in S: \epsilon(x) mod m = l \rbrace: l \in \epsilon(m_1) \rbrace &&
P_j = \lbrace \lbrace x \in S: \epsilon(x)\M_j mod m_j = l \rbrace: l \in \epsilon(m_j) \rbrace
$$

其中，对于$$j=2, \cdots, k$$, 有$$M_j = \prod\limits_{i=1}^{j-1} m_i$$。这会泛化qutient-remainder trick。

(4) Chinese Remainder Partitions:

考虑一个pairwise 互质因子分解（coprime factorization），它大于或等于$$\mid S \mid$$，也就是说，对于所有的$$i=1,\cdots,k$$ 以及$$m_i \in N$$, 有  $$\mid S \mid \leq \prod_{i=1}^{k} m_i$$；以及对于所有的$$i \neq j$$，有$$gcd(m_i, m_j)=1$$。接着，对于$$j=1,\cdots,k$$，我们可以定义该complementary partitions：

$$
P_j = \lbrace \lbrace x\inS: \epsilon(x) mod m_j = l \rbrace: l \in \Epsilon(m_j) \rbrace
$$

具体取决于application，我们可以定义更专门的 complementary partitions。回到我们的car示例，你可以基于year、make、type等来定义不同的partitions。假设这些属性的unique specificiation会生成一个unique car，这些partitions确实是complementary的。在以下章节，我们会演示如何来利用这些结构减小内存复杂度。

# 4.使用complementary partitions的compositional embeddings

在第2节对我们的方法进行泛化，我们可以为每个partitions创建一个embedding table，以便每个等价类可以被映射到一个embedding vector上。这些embeddings可以使用一些operation进行组合来生成一个compositional embedding或直接使用一些独立的sparse features（我们称它为feature generation方法）。feature generation方法尽管有效，但可以极大增加参数数目，需要增加额外features，并没有利用其内在结构，即：complementary partitions可以从相同的initial categorical feature中形成。

更严格的，考虑一个关于category set S的complementary partitions的集合：$$P_1,P_2,\cdots,P_k$$。对于每个partition $$P_j$$，我们可以创建一个embedding table $$W_j \in R^{\mid P_j \mid \times D_j} $$，其中，由$$i_j$$索引的每个等价类$$[x]_{P_j}$$被映射到一个embedding vector中，$$D_j \in N$$是embedding table j的embedding维度。假设：$$p_j: S \rightarrow \lbrace 0, \cdots, \mid P_j\mid -1 $$函数可以将每个element $$x \in S$$映射到相应的等价类的embedding index上，比如：$$x \rightarrow i_j $$。

为了泛化我们（operation-based）的compositional embedding，对于给定的category，我们会将来自每个embedding table与对应的所有embeddings交叉，来获得final embedding vector：

$$
$$

...(6)

其中，$$w: R^{D_1} \times \cdots \times R^{D_k} \rightarrow R^D$$是一个operation function。operation function的示例包括：

- 1) Concatenation:
- 2) Addition:
- 3) Element-wise Multiplication:

你可以看到，这些方法会为在简单假设下的每个category生成一个unique embedding。我们可以看到，在以下理论中。出于简洁性，下文的讨论仅限concatenation操作。

**定理1**

该方法会将存取整个embedding table $$O(\mid S \mid D)$$的内存复杂度减小到$$O(\mid P_1 \mid D_1 + \mid P_2 \mid D_2 + \cdots + \mid P_k \mid D_k)$$。假设$$D_1 = D_2 = \cdots = D_k = D$$并且$$\mid P_j \mid $$可以被专门选中，该方法会生成一个最优的内存复杂度$$O(k \mid S \mid ^{1/k} D)$$，这对于存储和使用full embedding table是一个明显提升。该方法在图2中可视化。

## 4.1 Path-Based Compositional Embeddings

生成embeddings的另一个可选方法是，为每个partition定义一个不同的transformations集合（除了第一个embedding table外）。特别的，我们可以使用单个partition来定义一个initial embedding table，接着，将initial embedding通过一个函数组合来传递来获取final embedding vector。

更正式的，给定一个category set S的complementary partitions集合：$$P_1, P_2, \cdots, P_k$$，我们可以定义一个embedding table $$W \in R^{\mid P_1 \mid} \times D_1$$来进行第一次划分（partition），接着为每一个其它partition定义函数集合$$M_j = \lbrace M_{j,i}: R^{D_{j-1}} \rightarrow R^{D_j}: i \in \lbrace 1, \cdots, \mid P_j \mid \rbrace \rbrace$$。在这之前，假设：$$p_j: S \rightarrow \lbrace 1, \cdots, \mid P_j \mid \rbrace $$是这样的函数：它将每个category映射到embedding index对应的等价类上。

为了为category $$x \in S$$获得embedding，我们可以执行以下transformation：

$$
x_{emb} = (M_{k,p_k(x)} \degree \cdots \degree M_{2,p_2(x)}) (W e_{p_1(x)}
$$

...(7)

我们












# 参考

- 1.[https://arxiv.org/pdf/1909.02107.pdf](https://arxiv.org/pdf/1909.02107.pdf)
