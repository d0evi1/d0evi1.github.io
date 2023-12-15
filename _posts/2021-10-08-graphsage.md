---
layout: post
title: GraphSage介绍
description: 
modified: 2021-10-08
tags: 
---

standford在《Inductive Representation Learning on Large Graphs》中提出了GraphSage：

# 介绍

在large graphs中的节点（nodes）的低维vector embedding，已经被证明对于许多预估和图分析任务来说作为feature inputs是很有用的。在node embedding方法的基本思想是：使用降维技术将关于**一个node的图邻居的高维信息**蒸馏（distill）到一个**dense vector embedding**中。这些node embeddings可以接着被feed到下游的机器学习系统中，并用于分类、聚类、连接预测等任务中。

然而，之前的工作主要关注于来自单个fixed graph的embedding nodes，**许多真实应用，会对于未知nodes、或者全新的subgraphs也能快速生成embeddings**。这些归纳能力对于高吞吐、生产环境机器学习系统来说很重要，它会在演进的图上操作、并能总是遇到未知的nodes（例如：在Reddit上的posts、在Youtube上的users和videos）。对于生成node embeddings的归纳法来说，会面临着在具有相同形式features的各种图上的泛化（generalization）：例如：来自一个模式生物的在蛋白质的相互作用图上，训练一个embedding genreator，接着可以很容易使用训练好的模型，来对于新的生物体（organisms）收集来的数据生成node embeddings。

对比起直推式setting（transductive setting），**inductive node embedding问题是特别难的，因为泛化到unseen nodes需要将新观察到的subgraphs“安排（aligning）”到已经通过算法最优化好的node embeddings中**。一个inductive framework必须学习去认识一个node的邻居的结构化属性，它能表明节点的在图中的**局部角色（local role）**，以及**全局位置（global position）**。

对于生成node embeddings的大多数已经存在的方法，是天然直推式的（transductive）。绝大多数方法是直接使用MF-based目标来最优化每个节点的embeddings，不会天然生成unseen data，因为他们会在一个单一fixed graph上做出预估。这些方法可以在一个inductive setting环境中被修改来执行，但这些修改版本的计算开销都很大，需要在做出新预估之前额外迭代好几轮gradient descent。一些最近的方法在图结构上使用卷积操作（convolutional operators）来进行学习，能提供一个embedding方法。因此，GCNs（graph convolutional networks）已经被成功应用到在fixed graph的直推式setting（transductive setting）上。而在本工作中，我们同时将GCNs扩展到归纳式无监督学习（inductive unsupervised learning）任务上，并提出一个framework来生成GCN方法，它使用**trainable aggregation function（而不是简单的convolutions）**.

我们提出了一个关于inductive node embedding的general framework，称为**GraphSage（抽样和聚合：SAmple and aggreGatE）**。不同于基于MF的embedding方法，**我们会利用node features（例如：文本属性、node profile信息、node degrees）来学习一个embedding function，它会泛化到unseen nodes上**。通过在学习算法中包含node features，我们会同时学习每个node邻居的拓朴结构，以及在邻居上的node features分布。当我们关注feature-rich graphs（例如：具有文本属性的引文数据、功能/分子标记的生物学数据），我们的方法也会利用出现在所有graphs（例如：node degrees）中的结构化features，因而，我们的算法也会被应用于没有node features的graphs中。

不同于为每个node训练一个不同的embedding vector的方式，**我们的方法会训练一个关于aggregator functions的集合，它们会从一个node的局部邻居（local neighborhood）（图1）中学习到聚合特征信息（aggregates feature information）**。每个aggregator function会从一个远离一个给定结点的不同跳数、搜索深度的信息进行聚合。**在测试（test）或推断（inference time）时，我们使用已训练系统来为整个unseen nodes通过使用学到的aggregation functions来生成embeddings**。根据之前在node embeddings生成方面的工作，我们设计了一个无监督loss function，它允许GraphSage使用task-specific supervision来进行训练。我们也表明了：GraphSage可以以一个完全监督的方式进行训练。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/2c7b3fe1cd792a1068bd0bd839f5fb073ccdd51cdfaeb519967afaa817db7e62e2caea63bd3369bacdfcadd62032f79e?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 GraphSAGE抽样和聚合方法的可视化演示

我们会在三个node分类benchmarks上评估我们的算法，它们会测试GraphSAGE的能力来在unseen data上生成有用的embeddings。我们会使用两个演进的document graphs，它们基于citation data和Reddit post data（分别预估paper和post类目），以及基于一个一个蛋白质的相互作用的multi-graph生成实验。使用这些benchmarks，我们展示了我们的方法能够有效生成unseen nodes的表示，效果对比baseline有一个大的提升。。。

# 2.相关工作

略

# 3.GraphSAGE

我们方法的关键思想是，我们会学习**如何从一个节点的局部节点（local neighborhood）中聚合特征信息（例如：邻近节点的degrees和文本属性）**。我们首先描述了GraphSAGE的embedding生成算法（例如：forward propagation），它会假设：为GraphSAGE模型参数已经被学习的节点生成embeddings。我们接着描述：GraphSAGE模型参数可以使用标准的SGD和BP技术被学到。

## 3.1 Embedding生成算法（例如：forward propagation）

在本节中，我们描述了embedding生成，或者forward propagation算法（算法1），它会假设：模型已经被训练过，并且参数是固定的。特别的，我们会假设：我们已经学到了关于K个aggregator functions的参数（表示为：$$AGGREGATE_k, \forall k \in \lbrace 1,\cdots,K\rbrace$$），它会被用于在模型的不同layers间、或者“搜索深度”上传播信息。第3.2节描述了我们是如何训练这些参数的。

- $G(V, E)$：图
- $\lbrace x_v, \forall v \in V \rbrace$：输入特征
- K：深度
- $W^k, \forall k \in \lbrace 1, \cdots, K \rbrace$： 权重矩阵
- $\sigma$：非线性激活
- $AGGREGATE_k, \forall k \in \lbrace 1,\cdots, K \rbrace$：可微聚合函数
- $N: v \rightarrow 2^v$：邻居函数（neighborhood function）

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/8d3afb65037d1c67dbf3b89b4b1973f9bd322ac43ccc7e320432466ac6238e5d442763915778694c09baaedb6594c8a7?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=a1.jpg&amp;size=750">

算法1

算法1的背后意图是：

- 在每个迭代中，或者搜索深度上，节点会聚合来自它们的local neighbors的信息；
- 并且随着该过程迭代，nodes随着graph的更进一步触达逐渐获得越来越多的信息。

算法1描述了case中的embedding生成过程，其中：

- $$G = (V, \epsilon)$$：表示整个graph
- $$\lbrace x_v, \forall v \in V \rbrace$$：表示graph中的所有node的**input features**
- K：表示深度
- $$W^k, \forall \lbrace 1,\cdots,K \rbrace$$：表示weight matrics

我们在下面描述了如何生成该过程到minibatch setting中。算法1的外循环中的每个step过程如下，其中：

- k：表示在外循环（或者搜索深度）中的**当前step**
- $$h^k$$：表示**在该step中一个node的representation**

**首先，每个node $$v \in V$$会将在它的立即邻居（immediate neighborhood）$$\lbrace h_u^{k-1}, \forall u \in N(v) \rbrace$$上的nodes 的representations聚合到单个vector $$h_{N(v)}^{k-1}$$中**。注意，该聚合step依赖于：在outer loop的前一迭代生成的representations，并且k=0（"bad case"）representations会被定义成input node features。

**接着，在聚合了邻近的feature vectors之后，GraphSAGE会将该node的当前representation $$h_v^{k-1}$$与聚合的邻近vector $$h_{N(v)}^{k-1}$$进行拼接**（concatenates），该concatenated vector会通过一个具有非线性activation function $$\sigma$$的fully connected layer进行feed，它会将representations转换成在算法的下一step进行使用（例如：$$h_v^k, \forall v \in V$$）。neighbor representations的聚合可以通过多个aggregator结构来完成，并且我们会在第3.3节中讨论不同的结构选择。

为了扩展算法1到minibatch setting中，给定一个关于input nodes的集合，**我们首先对所需要的neighborhood sets（到深度K）进行forward sample**，接着我们在inner loop进行运行，通过替代迭代所有nodes，我们只会计算：必须满足在每个depth上的递归的representations。

**与 Weisfeiler-Lehman Isomorphism Test的关系**

GraphSAGE算法在概念上受testing graph isomorphism的经典算法的启发。在算法1中，我们：

- (i) 设置$$K=\| V \|$$
- (ii) 设置weight矩阵作为identity
- (iii) 使用一个合适的hash function作为一个aggregator（无非线性），

接着算法1是一个关于WL isomorphism test的实例，也被称为“naive vertex refinement”。如果由算法1输出的representations $$\lbrace z_v, \forall v \in V \rbrace$$。该test在一些情况下会失败，但是对于许多图是合法的。GraphSAGE是对WL test的连续近似，其中，我们将hash function替代成trainable neural network aggregators。当然，我们使用GraphSAGE来生成有用的节点表示（node representations）——而非test graph isomorphism。然而，在GraphSAGE和经典的WL test间的连接，为我们的算法设计提供了理论context来学习关于节点邻居的拓朴结构。

**Neighborhood定义**

在本工作中，我们会均匀地抽样一个关于节点邻居的fixed-size集合，而非使用算法1中完整的邻居集合，是便保持每个batch的计算开销是固定的。也就是说，使用过载的概念，在算法1中，我们将$$N(v)$$定义为一个从集合$$\lbrace u \in V:(u,v) \in \epsilon \rbrace$$中fixed-size的均匀抽取，在每个迭代k上我们会从不同的均匀样本中进行抽样。如果没有这种抽样，单个batch的内存和runtime会不可预知，最坏情况下为$$O(\| V \|)$$。作为对比，对于GraphSAGE的每个batch space和时间复杂度被固定在$$O(\prod_{i=1}^K S_i)$$，其中$$S_i, i \in \lbrace \cdots \rbrace$$，K是user-specified常数。实际来说，我们发现我们的方法可以达到具有K=2的高效果，并且$$S_1 \cdot S_2 \leq 500$$。

## 3.2 学习GraphSAGE的参数

为了以一个完全无监督setting方式学习有用的、可预测的表示（representations），我们会通过SGD使用一个**graph-based loss function**给output representations $$z_u, \forall u \in V$$，并且调节weight矩阵 $$W^k, \forall k \in \lbrace 1,\cdots, K \rbrace $$， 以及得到的aggregator functions的参数。**graph-based loss function**会鼓励邻近节点具有相似的表示（representations），从而强制分离的nodes的representations具有高度的不同：

$$
J_G(z_u) = - log(\sigma(z_u^T z_v)) - Q \cdot E_{v_n \sim P_n(v)} log(\sigma(- z_u^T z_{v_n}))
$$

...(1)

其中:

- v：是一个node，它会与u在一个fixed-length的random walk中共现
- $$\sigma$$：是sigmoid function
- $$P_n$$：是一个**negative sampling分布**
- Q：定义了negative samples的数目

重要的是，不同于之前的embedding方法为每个node（通过一个embedding look-up）训练一个唯一的embedding，**我们feed到该loss function的representations $$z_u$$会由在一个node的局部邻居（local neighborhood）中包含的features来生成**。

该无监督setting会模拟以下情况：其中node features会提供到下游的机器学习应用，或者作为一个服务 或者 在一个静态仓库中。在本case中，其中representations只会在一个指定的下游任务中，无监督loss（等式1）可以简单地被一个task-specific目标（比如：cross-entropy loss）替换。

## 3.3 Aggregator结构

在N-D lattices（例如：句子、图片、或3D空间），**一个node的邻居没有自然序：在算法1的aggregator function必须在一个关于vectors的无序集合上操作**。理由的，一个aggregator function必须是对称的（例如：它的inputs排列是不变的），同时是可训练的，并且保持一个高度表达的能力。**aggregation function的对称性确保了我们的neural network model是可训练的，并且可以被用于随意顺序的节点邻居的feature sets上**。我们会检查三个候选aggregator function：

**Mean aggregator** 

我们的第一个候选 aggregator function是mean aggregator，其中：我们简单地对在$$\lbrace h_u^{k-1}, \forall u \in N(v) \rbrace$$中的vectors做elementwise平均。该mean aggregator接近等于在transductive GCN network中的convolutional propagation rule。特别的，我们可以通过在算法1中的第4和第5行使用下面进行替换，派生一个关于GCN方法的inductive variant：

$$
h_v^k \leftarrow \sigma(W \cdot MEAN(\lbrace h_v^{k-1} \rbrace \cup \lbrace h_u^{k-1}, \forall u \in N(v) \rbrace)
$$

...(2)

我们将该修改版称为mean-based aggregator convolutional，因为它是一个关于一个localized spectral convolution的不平滑的线性近似。在该convolutional aggregator和其它aggregators间的一个重要区别是：**在算法1的第5行，不会执行concatenation操作**。例如：convolutional aggregator不会将node的前一layer representation $$h_v^{k-1}$$与he aggregated neighborhood vector $$h_{N(v)}^k$$进行concatenate。该concatenate可以被看成是关于一个关于在不同“search depths”或GraphSAGE算法中“layers”间的"skip connection"的简单形式，并且它会产生在效果上的巨大收益。

**LSTM aggregator**

我们也会检查一个更复杂的基于一个LSTM结构的aggregator。对比起mean aggregator，LSTMs的优点是：具有更强的表达能力。然而，需要注意的是，**LSTMs没有天然的对称性（例如：它们没有排列不变性），因为他们以序列方式处理它们的inputs**。我们采用LSTMs用来操作一个无序集合，通过简单地将LSTMs应用到一个关于节点邻居的随机排列上。

**Pooling aggregator**

**该aggregator具有对称性，同时可训练**。在pooling方法中，每个邻居的vector会独立地通过一个fully-connected neural network进行feed；根据该转换，一个elementwise max-pooling操作会被应用来对跨邻居集合的信息进行聚合：

$$
AGGREGATE_k^{pool} = max(\lbrace \sigma(W_{pool} h_{u_i}^k + b), \forall u_i \in N(v) \rbrace)
$$

...(3)

其中：

- max表示element-wise max操作符
- $$\sigma$$是一个非线性activation function

原则上，该函数会在max pooling之前被使用，可以作为一个随意的deep multi-layer perceptron，但我们关注于简单的single layer结构。该方法受最近研究【29】的一些启发。直觉上，multi-layer perceptron可以被认为是：作为在邻居集合中的每个节点表示的feature计算的一个函数集合，该模型可以有效地捕获邻居集合的不同方面。注意，原则上，任务对称vector function可以被用来替代max operator（例如：一个element-wise mean）。我们发现：关于developments test在max-pooling和mean-pooling间没有大差别，因此后续主要关注max-pooling。

# 实验

略

- 1.[https://arxiv.org/pdf/1706.02216.pdf](https://arxiv.org/pdf/1706.02216.pdf)