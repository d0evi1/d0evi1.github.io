---
layout: post
title: meta DLRM工程实现介绍
description: 
modified: 2023-10-18
tags: 
---

meta在《Software-Hardware Co-design for Fast and Scalable Training of
Deep Learning Recommendation Models》提出了DLRM的工程实现。

# 摘要

深度学习推荐模型（DLRMs）在Meta的许多关键业务服务中得到应用，并且是其数据中心基础设施需求方面最大的AI应用。在本文中，我们介绍了**Neo，这是一个软硬件共同设计的系统，用于大规模DLRMs的高性能分布式训练**。Neo采用了一种新颖的4D并行策略，结合了表格级（table-wise）、行级（row-wise）、列级（col-wise）和数据并行策略，用于训练DLRMs中的大规模embedding操作（embedding operators）。此外，Neo通过包括混合内核融合、软件管理缓存和质量保持压缩在内的多种关键系统优化，实现了极高的性能和内存效率的embedding计算。最后，Neo与ZionEX配对，**ZionEX是一个新的硬件平台，与Neo的4D并行策略共同设计，用于优化大规模DLRM训练的通信**。我们在128个GPU上使用16个ZionEX节点的评估表明，Neo在训练已部署生产的12万亿参数DLRM模型方面，性能超过了现有系统高达40倍。

# 1 引言

深度学习推荐模型（DLRMs）被在线公司广泛使用，包括亚马逊用于其目录中选择商品[35, 37, 58]，Netflix用于展示电影选项[13, 29]，以及谷歌用于显示个性化广告[7, 9, 19]。它们还被标准基准测试组织采用，如MLCommons (MLPerf) [38, 52]。**在Meta，我们已经在排序和点击通过率（CTR）预测中广泛使用推荐模型**，包括新闻推送和搜索服务[15, 17, 42, 47]。DLRMs是数据中心基础设施需求方面最大的AI应用。

与传统的深度神经网络（DNNs）不同，DLRMs主要包含计算密集型操作（例如，卷积和矩阵乘法），DLRMs结合了计算密集型组件和多达数千个数据密集型嵌入操作符，每个操作符都有不同的资源需求和性能特性[43]。因此，与计算机视觉[8, 18, 59]、自然语言处理[5, 10, 61]和强化学习的同类模型相比，**DLRMs通常表现出更低的算术强度和更大的模型大小，实际部署的模型拥有数万亿参数**，如图1所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/816936759462f2bcee405227a84c20c8f57a07349a0d738a34597bd1e15bec5d3c94fd8dd602001ed3c11b512f0a554d?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 在总计算量上比较深度学习模型，以拍拉浮点运算/天(petaflop/s-days)为单位（上部）[45]和模型容量（下部）

现有的**针对DNN的软件和硬件解决方案**在DLRMs上只能实现次优性能和有限的可扩展性，这是由于以下软件/硬件限制造成的。

在软件方面，现有的深度学习框架通常使用数据、模型或流水线并行化来并行化DNN训练[3, 32, 48]。**支持这些策略组合的框架通常为特定的DNN应用而设计[16, 22, 41, 50]**。然而，为计算密集型DNN模型设计和优化的现有并行化策略在DLRMs上实现有限的性能和可扩展性。特别是:

- **数据并行化**要求每个设备保存整个模型的副本，因此不支持拥有数万亿参数的DLRMs[32]。
- 此外，由于其嵌入操作符的数据依赖行为，DLRM不能直接使用**模型并行化或流水线并行化**。具体来说，处理不同的训练样本可能需要根据每个样本的分类输入访问不同的嵌入参数。这种数据依赖行为使得在满足所有样本的数据依赖性的同时，静态地将DLRM的可训练参数划分为不相交的子集变得不可行，这是使用模型和流水线并行化的必要条件。

此外，**当今的DNN框架旨在优化计算密集型（compute-intensive) DNN计算，忽视了对数据密集型（data-intensive）嵌入操作符的关键优化**。具体来说，DLRM包含多达数千个嵌入操作符。这些嵌入操作符的前向处理、反向传播和梯度同步需要在训练迭代中启动数千个CUDA内核，并消耗高达数TB的累积GPU设备内存，引入了显著的运行时开销和内存需求。

在硬件方面，**现代硬件平台，如基于GPU的集群，提供了显著的能力提升，但它们并不是为了匹配DLRMs的性能特性而设计的**。具体来说，DNN训练的硬件平台通常针对集中式节点间通信（例如，参数服务器[3]）和/或AllReduce通信（例如，Horovod[54]和NCCL[1]）进行优化。然而，如第3节所确定的，高性能和可扩展的DLRM训练需要有效硬件支持多种不同的通信模式，包括AllReduce、AlltoAll、ReduceScatter、OneToMany和ManyToOne。

## 1.1 我们的方法

我们提出了Neo，这是一个软硬件共同设计的系统，用于快速且可扩展的DLRM训练，它基于三个关键技术构建。

**4D并行**

**为了在DLRM中快速且可扩展地训练大规模嵌入操作符**，有效地平衡GPU之间的工作负载分配并最小化通信成本至关重要。我们引入了一种4D并行策略，结合了**表格级、行级、列级和数据并行策略**，共同优化嵌入操作符的并行性能。此外，Neo还支持在不同级别的硬件层次结构中以递归方式应用4D并行，以进一步提高负载平衡和硬件效率。

**高性能嵌入计算**

Neo采用了两项新的优化技术，以最小化嵌入操作符的计算成本和内存需求。

- 首先，我们引入了一种**混合内核融合技术**：**将（1）多个嵌入操作符和（2）嵌入计算及其参数更新全部融合在一个CUDA内核中**。这是通过共同设计嵌入操作符的优化算法和软件实现来实现的。
- 其次，为了提供足够的内存容量以支持DLRM训练，Neo使用**软件管理的缓存机制**来利用现代硬件平台的内存层次结构。
- 最后，进一步应用了**多种压缩技术[29, 63]**来最小化内存需求。

**硬件平台设计**

我们介绍了ZionEX，这是一个与Neo的4D并行共同设计的新型硬件平台，用于优化分布式DLRM训练的节点间通信。ZionEX通过使用专用的基于融合以太网的RDMA（RoCE）扩展网络，支持集群中所有GPU的全连接拓扑。这种拓扑设计促进了分布式DLRM训练中性能主导的通信工作负载（例如，AlltoAll和ManyToOne）的高性能数据传输。同时，ZionEX支持RDMA和GPUDirect通信协议，并保留了灵活的节点内GPU织物。这使得在ZionEX上能够进行高性能的DLRM训练，同时确保与现有数据中心基础设施的兼容性，允许ZionEX的广泛部署。

**结果**

我们已经在三个生产环境中部署的不同任务的DLRM上评估了Neo，包括点击通过率预测、排序和参与度，代表了多样化的生产级推荐模型。我们在16个ZionEX节点上的128个A100 GPU上的评估表明，Neo能够处理高达每秒170万次查询，用于训练具有12万亿参数的DLRM，与现有生产中的DLRM训练解决方案相比，速度提升了40倍。消融研究表明，4D并行、高性能嵌入计算和新的ZionEX平台对于实现快速且可扩展的DLRM训练至关重要。

总结来说，我们的贡献是：

- 我们提出了Neo，一个软硬件共同设计的系统，用于快速且可扩展的DLRM训练。Neo在训练具有12万亿参数的大规模DLRM方面，性能超过了现有系统高达40倍。
- 我们提出了4D并行，这是一种结合了表格级、行级、列级和数据并行策略的训练嵌入操作符的方法。
- 我们开发并实现了使用混合内核融合、软件管理缓存和质量保持压缩的高性能嵌入操作符。
- 我们构建了ZionEX，一个与Neo的4D并行共同设计的新型硬件平台，以加速DLRM训练中的多种通信模式。

# 2 背景

DLRMs通常有两种训练模式——离线和在线，每种模式都有不同的要求。离线训练可以被视为预训练，其中候选模型在足够大的历史数据上进行训练，并期望在部署到当前/未见过的样本时能够泛化。一旦部署，DLRMs继续使用它已经服务过的数据进行在线训练。离线训练受到吞吐量限制，符合更传统的“尽可能快地训练尽可能多的数据”的范式，而在线训练对延迟更敏感，重新训练和更新的频率是一个重要因素。**对于在线训练，吞吐量要求较低，因此可能希望使用相对较少的资源。这创造了一个独特的需求，即在能够容忍较低吞吐量的较小规模上训练非常大的模型**。

**本文专注于对训练吞吐量需求更高的离线训练**——每秒处理多达数百万样本（查询），这些样本来自于在合理时间内处理数十PB训练数据。这推动了训练平台的需求，如表1所总结。

**嵌入操作符**

DLRMs与传统深度神经网络之间的一个主要区别是利用类别型特征（如用户、帖子或页面）。生产中使用的DLRMs通常包含多达数千个类别型特征，每个特征对应一个专用的嵌入操作符。**嵌入操作符以一个multi-hot向量作为输入，向量中的每个非零元素触发嵌入表中的完整行检索，其中输入向量的每个索引对应表中的一行**。最后，对于给定的输入向量，所有嵌入行通过element-wise pooling组合，如图2所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/033adb5e7b6f4ad5bb0e9bacf7ef05d85d0d6cf09e6d57120281e0e2bdc5cdbd32c7e959f5c6d9b35d1c6defb115e1ae?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2 一个embedding operator的Workflow

**并行化策略**

传统上，用于生产环境中训练DLRMs的是基于参数服务器（PS）的分布式CPU训练系统。具体来说：

- 一方面，**MLP模块中的dense参数在训练器之间复制以利用数据并行性**。它们的权重使用集中式dense参数服务器通过弹性平均方法SGD进行同步。
- 另一方面，**embedding table中的参数被分割并放置在多个PS上以利用模型并行性**，因为embedding参数的大小简单地阻止了模型复制。

为了最大化训练吞吐量，使用Hogwild!更新嵌入操作符的参数。此外，读者部署在单独的机器层上，为训练器提供训练批次，如图3所示：

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/61e99c9a03351b7b2e8e71ec531a9bce635503542ac1352369ccbb297ffc02ef700375e862bfdd36a733d6af9af2c9b9?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3 基于分离式参数服务器（Disaggregated parameter-server）的系统

**这种基于PS的系统非常适合DLRMs**，允许分别扩展不同的组件，并在训练具有不同训练器、参数服务器和读者配置的不同模型时实现平衡的资源利用。此外，系统中的资源在很大程度上是可替代的，这使得数据中心运营成本较低。

然而，**需要支持具有数万亿参数的DLRMs**，因此大小达到数TB，这对这种方法的可扩展性提出了严重挑战，需要大量增加训练器和参数服务器的数量以满足不断增长的训练需求。这很快变得不可行，**由于在大量worker之间增加的异步更新，导致模型准确性因陈旧性而降低**。为了解决这些问题，我们构建了一个用于大型DLRMs的高性能同步训练解决方案，将分布式扩展与统计质量解耦。

**同步训练系统的高效设计**，使我们使用一种新颖的4D并行组合（第4节）用于内存密集型嵌入表，数据并行性用于计算密集型DNN操作符，并在不同组件之间进行流水线处理。这种混合并行性需要AlltoAll通信来处理嵌入查找结果，以及如果输入是从数据库批量流式的，还需要重新分配嵌入表输入，这通常是情况。与用于梯度同步的AllReduce通信不同，AlltoAll通信由于数据依赖性而位于关键路径上，强调了互连和通信原语的性能。此外，DLRMs通常在非常大的数据量上进行训练，这些数据对应于来自各种应用的大多数非结构化和未标记的交互。典型的数据集大小在几个PB的范围内，需要使用常见的分布式网络存储，如Tectonic文件系统。对于训练，这些数据需要被流式传输进来，给主机网络和主机到设备带宽带来额外的压力。

# 3 概览

图4展示了Neo的概览，这是一个软硬件共同设计的系统，用于快速且可扩展的DLRM训练。本节简要描述了Neo的关键组件。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/95004be3b78920c117f6bd02658dd0351061a2012e34d0295fcf5c46b25636d0e08555e0157c0e181f265971aaa1c230?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=4.jpg&amp;size=750">

图4 Neo 概览。图中的每个框表示一个神经网络组件，而框之间的边表示不同组件之间共享的张量。

首先，**Neo使用数据并行性来训练计算密集型的DNN层（以橙色显示）**，并切换到一个4**D并行策略，该策略结合了表格级、行级、列级和数据并行性，以高效训练内存密集型的嵌入操作符**。

其次，Neo配备了高性能的嵌入操作符实现。这是通过一系列关键的系统优化实现的，包括:

- （1）混合内核融合技术来减少嵌入操作符的计算成本，
- （2）软件管理的缓存机制来利用现代硬件平台的异构内存
- （3）多种质量保持压缩技术来最小化嵌入计算的内存需求

最后，Neo部署在ZionEX上，这是一个与Neo的4D并行共同设计的新型硬件平台，用于优化DLRM训练的节点间通信。

此外，数据I/O是任何训练系统的重要组成部分，特别是随着完全同步训练和加速器的采用。首先，主机到设备的数据传输应该是非阻塞的，并且足够快，不会限制整体训练吞吐量。理想情况下，使用双缓冲或流水线将输入数据传输与训练重叠。其次，尽管将输入数据分布映射到训练器之间的集体通信更快，但这为集体通信的输入和输出数据布局引入了额外的挑战。初步实验表明，这些可能会给关键路径增加显著的延迟。我们将在第7.1节中展示我们如何克服这些实际挑战。

# 4 4D并行策略

DLRM的一个关键组成部分是嵌入操作符，将在第5节中定义。为了实现嵌入操作符的高性能训练，有效地平衡GPU之间的工作负载分布并最小化通信成本至关重要。我们引入了4D并行策略，它结合了表格级、行级、列级和数据并行策略，共同优化嵌入操作符的并行性能。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/7da81de2ded0e5e60740dfd3d265b529271f5d69ebe414e9f2bcdf582d995ad8870a09e65e41693072dc18fee4c3a2e1?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=5.jpg&amp;size=750">

图5 具有不同通信成本、负载均衡和内存需求影响的嵌入表分片方案。为了简化说明，此图中省略了底部的MLP（多层感知机）。

**表格级并行策略**。

最直接的并行方案是：跨GPU分割和并行化多个嵌入表，如图5a所示。表格级并行策略不会进一步分割嵌入表，因此此方案不需要额外处理嵌入表输入索引或聚合的嵌入结果，从而实现最佳的通信效率。然而，表格级并行策略无法处理超过单个GPU内存容量的大型嵌入表，且由于表大小的偏斜，实现的负载平衡通常有限。

**行级并行策略。**

此方案通过行来并行化大型嵌入表，并将不同的表片段分配给不同的训练器。由于嵌入表输入通过行来索引表，它们需要根据行级并行决策进行分桶并分发到相应的训练器，如图5b所示。此外，多个训练器上的部分结果需要被缩减然后分散到所有训练器以进行下游计算。这要求在前向传递中使用ReduceScatter通信模式。此方案能够很好地处理大型表，并带来更好的负载平衡。然而，通信成本与训练器的数量成线性关系。

**列级并行策略。**

列级并行策略沿嵌入维度划分嵌入表（见图5c），并将划分后的表视为具有较小嵌入维度的独立操作符。此方案需要为划分后的表复制输入索引。与表格级并行策略相比，它保持了相同的流程和通信模式（AlltoAll）。列级并行策略的一个关键优势是能够实现更细粒度的并行策略，特别是对于大型表。然而，它仅在大型嵌入维度下表现良好，并增加了输入索引的负载，这些索引必须复制到所有节点的列片段。此外，由于列级划分的表的行分布在不同的训练器上，使用这些表的独立行更新引入了额外的参数，每个行片段一个参数，而不是使用稀疏优化器时整个行只有一个单一值（见第5.1节了解更多细节）。

**数据并行策略。**

**DLRM往往有广泛的表大小范围，而表格级、行级和列级并行策略适用于相对较大且无法复制的嵌入表**。对于较小的表，数据并行策略能够实现更好的性能，因为数据并行策略在前向传递中不涉及任何通信（见图5d）。因此，对于小型嵌入表，Neo将embedding表视为dense参数并在所有训练器上复制它们。对于数据并行嵌入表的聚合嵌入，不再需要AlltoAll。相反，需要AllReduce来同步所有副本。因此，这取决于聚合嵌入的AlltoAll成本与整个表的AllReduce成本之间的权衡。通常，**行数较少的小型嵌入表是数据并行策略的良好候选者**。这些表的输入索引作为数据并行输入传递，不再需要重新分配。

## 4.1 并行化算法

Neo支持在单个嵌入操作符的粒度上应用4D并行策略，以最大化灵活性。实践者可以混合使用上述原语，以确定划分嵌入操作符的最佳策略。此外，Neo还支持在不同级别的硬件层次结构中以递归方式划分嵌入操作符，以进一步提高工作负载平衡和硬件效率。例如，表格级然后行级方案首先将一组表分配给特定节点，然后在该节点内按行划分表。这种层次并行方案通过充分利用快速的GPU互连并减少节点间通信，提高了硬件局部性。

为上述每种并行方案定义了成本函数，可以探索放置算法以最小化worker之间的成本差异。成本函数是通信开销和训练器之间的负载不平衡的组合。通信开销使用消息量作为代表性指标计算，消息量越大对应成本越高。这在很大程度上准确地捕捉了吞吐量成本，对于延迟测量值，作为固定附加成本纳入。我们通过使用每个训练器的嵌入访问大小来估计负载不平衡，这可以近似为每个训练器的嵌入表数量×全局批量大小×每个样本的平均索引数×嵌入维度。这两种成本的组合为我们提供了通信和负载不平衡的合理估计。进一步，我们为每个单独的成本引入了标量权重，可以根据不同的系统规格进行调整，以获得更准确的估计。

我们实现并评估了两种多项式时间启发式算法作为概念验证。第一个是一个简单的贪婪启发式算法，它将可用方案的成本按降序排序，并首先分配最大的片段，每个worker一个。然后，贪婪算法遍历所有剩余的片段，并将最高成本分配给成本总和最小的节点。第二个启发式是最大差分方法（也称为Karmarker-Karp算法[26]）。主要思想是从输入中取出两个最大的数字，并用它们的差替换它们。它直接减少了总和的差异，通常优于贪婪启发式。

## 4.2 流水线

尽管使用GPU作为主要计算资源在模型评估内提供了有限的流水线机会，我们通过流水线化批间数据移动并与计算重叠通信来提高GPU利用率。

当批次𝑖正在被评估时，相同的GPU可以开始使用单独的流接收和分发批次𝑖 + 1。为了最小化干扰，我们将批次𝑖 + 1的输入AlltoAll与批次𝑖的顶部MLP的前向传播重叠，其中不涉及通信。此外，我们将聚合的嵌入AlltoAll与底部MLP的前向传播重叠，以隐藏延迟。

# 5 嵌入优化

优化DLRM的嵌入操作符（见第2节）的运行时性能需要解决两个关键挑战。

- 首先，嵌入操作符的前向处理、反向传播和梯度更新需要在每次训练迭代中启动数千个GPU内核，引入了显著的GPU内核启动开销。
- 其次，一些嵌入操作符可能包含高达数十亿的参数，无法适应单个GPU的设备内存。

我们引入了三种新技术来减少嵌入操作符的计算成本和内存需求。

- 首先，我们引入了一种混合内核融合技术，以最小化CUDA内核启动开销，并允许每个GPU工作器只启动两个内核（即一个用于前向传播，一个用于反向传播和参数更新）。
- 其次，对于嵌入操作符的并行计算，我们提出了列级并行策略和行级并行策略，除了数据和模型并行策略之外。这四个并行维度的组合使Neo能够支持高达数万亿参数的嵌入表。
- 最后，Neo利用一系列内存节省技术，利用ZionEX平台的内存层次结构，确保DLRM有足够的内存容量。

## 5.1 内核融合

Neo使用混合内核融合机制来最小化执行嵌入计算的CUDA内核启动开销。

- 首先，与为每个嵌入表应用单独的嵌入查找不同，Neo将同一GPU上的多个嵌入查找融合到单个CUDA内核中（图6a），这提高了并行性和带宽利用率，并减少了在GPU上启动多个CUDA内核的开销。
- 其次，Neo还将反向传播与稀疏优化器融合，以进一步减少内核启动开销，并避免将梯度具体化到嵌入表中。这种融合的关键挑战是避免来自不同训练样本的梯度更新之间的潜在竞态条件，以及处理诸如AdaGrad[11]、LAMB[66]和Adam[27]等高级优化器中的非线性。例如，**图2中的样本1和2都有助于嵌入向量1和6的梯度。如果不聚合直接将这些梯度发送到非线性稀疏优化器，将导致嵌入表的错误更新**。为了保证正确性的同时最大化性能，Neo通过行对梯度进行排序，以便对同一嵌入行的梯度由单个CUDA线程块处理，如图6b所示。
- 随后在每个CUDA线程块内使用更快但更小的GPU共享内存进行梯度聚合。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ab1aaefb4a3df56b1c3c0a37bbb17873008a26197fbbea1c81d79ef33cee829861cdc148d2979108bc0725273a2c0631?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=6.jpg&amp;size=750">

图6

Neo的嵌入操作符混合融合技术带来了三个性能优势。

- 首先，Neo通过避免为嵌入梯度分配GPU设备内存来减少嵌入操作符的内存需求。
- 其次，通过使用GPU共享内存来保存中间嵌入梯度，最小化了对GPU设备内存的内存访问。
- 最后，内核融合通过高达7倍的性能提升，改善了嵌入计算的整体性能，与原生实现相比。

优化的嵌入操作符实现作为FBGEMM库∗的一部分开源，并与PyTorch集成。

## 5.2 管理内存层次结构

对于具有高达数万亿参数的DLRM，嵌入表太大，无法完全适应单个GPU。**我们利用ZionEX平台的多个内存层次结构，包括HBM、DRAM和SSDs，以及扩展到多个节点以增加聚合容量，确保模型有足够的内存，更快的内存作为后续层的软件缓存**。Neo的层次内存管理策略特别适用于DLRM的在线训练，由于吞吐量要求较低，因此可以使用较少的节点来训练原始的大型模型，如第2节所述。

管理内存层次结构的一种方法是CUDA的统一内存（UVM）[44]，它为不同类型的内存提供单一的内存地址空间，并自动替换和逐出未使用的页面。然而，嵌入操作符中的随机表查找需要在单个嵌入行的粒度上缓存和替换未使用的参数，这使得直接使用UVM对于DLRM来说是不够的。需要额外处理查找以确保性能不受频繁的主机到设备传输的限制。相反，Neo使用定制的32路集合关联软件缓存[64]，使用最近最少使用（LRU）或最少频繁使用（LFU）缓存替换策略，其中关联性与GPU的warp大小相匹配。这使得可以对缓存和替换进行细粒度控制，允许针对目标模型特性进行调整。请注意，UVM受PCIe带宽限制，而Neo的软件缓存可以弥补PCIe和HBM之间的带宽差距（50倍差异）。与UVM相比，软件缓存将DLRM工作负载的端到端性能提高了约15%。

为了进一步减少嵌入操作符的内存需求，Neo还采用了先前工作中引入的多种压缩技术，如逐行稀疏优化器[14, 62]、使用高精度缓存支持的低/混合精度训练[63]和高级分解技术[29]。

逐行稀疏AdaGrad首次在[14]中引入，然后在[62]中进一步阐述。在逐行稀疏AdaGrad中，每个元素的时刻估计应用于整个嵌入行。对于每一行，它是一个单一的缩放因子，通过添加行中梯度的平均平方和来更新。通过这种方式，我们将动量保持为一个1D张量，有H个元素，而不是H×D的2D张量，其中H和D分别是嵌入表中的行数和每行的元素数。

# 6 ZIONEX: 硬件平台设计

我们首先在第6.1节描述了我们之前用于DLRM的硬件平台的局限性。第6.2节介绍了ZionEX，一个为DLRM设计的新型硬件平台。我们还概述了在ZionEX开发中使用的设计原则。

## 6.1 之前的平台：Zion

2019年推出的Zion是我们之前针对DLRM训练的高性能硬件平台的工作。尽管Zion在单节点级别提供了显著改进的能力，但作为一个分布式平台，它未能扩展以满足迅速增长的DLRM训练需求。我们批判性地评估了它的局限性，但其他基于类似设计的平台也存在相同的局限性；我们在第9节讨论了这些平台。

图7a显示了一个Zion节点的架构，它有8个CPU插槽，1.5TB内存，8个GPU和8个网络接口卡（NIC）。它通过（1）将DLRM的计算密集层（例如，MLP）卸载到GPU上，以及（2）利用CPU处理大型嵌入操作符在相对便宜的DRAM上，而不是HBM，以在单个节点上容纳TB级DLRM，提供了强大的异构超级节点设计。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/7541da8fd34f7c2262c0d5aa6bedfd6d6c1a203c5cbdba17342f1530ee601a34b5a4071a1a211eda28c9a5485d164c39?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=7.jpg&amp;size=750">

图7

然而，这种异构设计引入了许多软件设计和性能挑战。例如，平衡CPU和GPU上的工作负载以确保最大重叠至关重要。这需要在CPU和GPU之间进行精细的流水线处理，并使用精确的成本模型将DLRM划分为细粒度任务。此外，DLRM的异构训练还引入了非平凡的运行时开销，例如增加了CPU和GPU之间的数据传输和节点间通信。最后，Zion的一个关键缺失组件是每个NIC直接连接到一个CPU。因此，所有节点间通信（例如，梯度同步和张量转换）都需要CPU干预和额外的GPU-CPU传输。此外，这些NIC连接到常见的数据中心网络基础设施，引入了网络拥塞的开销和干扰，并且受限于使用更数据中心友好的拓扑结构和协议（TCP/IP），这对于分布式训练是次优的。尽管每个Zion节点都配备了8个100Gbps NIC带宽，但实际上我们发现由于网络开销，很难扩展到多个节点。随着DLRM模型大小需求的增加，Zion无法很好地扩展并充分利用强大的硬件资源。

## 6.2 ZionEX

为了解决这些不足，我们引入了ZionEX，我们已经设计它比之前的Zion平台更具可扩展性，并提高了网络能力，同时保留了其灵活性和核心优势，例如OAM外形因素、模块化设计和灵活的节点内加速器结构。通过所有这些改进，ZionEX在支持增加模型复杂性和提高训练性能方面带来了数个数量级更高的能力。这最好通过比较每个平台支持的最大模型复杂性（以FLOPS/样本计）和实现的训练吞吐量来说明，这可以被视为标准化的有效性能。对于ZionEX，实现了1.2 MQPS的吞吐量，模型复杂性为638 MFLOPS/样本（见表3），这转化为766 TFLOPS/s的有效性能，还有额外的余地上升到数PETAFLOPS/s。而Zion在ZionEX上支持的最大模型复杂性不到一半（约250 MFLOPS/样本），吞吐量更低（约0.25 MQPS），因此最大可实现的有效性能降低了10倍以上，仅为63 TFLOPS/s。图7b显示了整体系统架构。我们简要强调了ZionEX的核心技术原则：

可扩展性。Zion和ZionEX都支持DLRM的异构训练，但最显著的区别是ZionEX设计了足够的扩展和扩展网络能力。如图7b所示，ZionEX为每个通过PCIe交换机连接的GPU配备了专用的RDMA over Converged Ethernet (RoCE) NIC，以允许专用的节点间连接（与常见数据中心网络隔离），并重要地支持更高效的RDMA/GPUDirect通信协议。这些ZionEX节点可以通过专用后端网络连接，形成一个分布式可扩展训练的集群。ZionEX的可扩展设计允许扩展后端网络，连接数千个节点，形成一个数据中心规模的AI训练集群。

高性能。作为一个扩展解决方案，我们将整个DLRM卸载到GPU上，充分利用大规模并行性和高内存带宽来加速MLP和嵌入计算。为了传输张量和同步梯度，每个GPU都可以直接通过专用的低延迟高带宽RoCE NIC与不同节点上的GPU通信，不涉及主机CPU。此外，ZionEX还有一个前端NIC连接到每个CPU。数据摄取通过常规前端网络和PCIe进行，不干扰激活或梯度。主机CPU仅用于设置输入批次和组织训练过程。

能力。通过ZionEX，我们确保平台与现有基础设施兼容，并可以在我们的数据中心内广泛部署，不会造成重大中断。这对于能够有效利用平台的能力并使其随时可用于各种应用和用例至关重要。我们通过使ZionEX平台符合标准的Open Rack规范来实现这一点，这涵盖了与其他基础设施组件的兼容性，如电源、冷却、机械和布线。此外，设计平台为模块化，并依赖基于开放标准技术，例如基于以太网的网络结构，用于高性能扩展解决方案。

图7c显示了整体训练平台，以及分离的数据摄取服务。这支持从网络存储（如Tectonic）流式传输输入数据，并以分布式方式执行轻量级数据预处理操作。以便数据摄取不是端到端训练的瓶颈，并确保在向ZionEX训练器提供数据时有足够的吞吐量。

# 7 实现

我们详细描述了上述用于DLRM的高性能可扩展训练的实现。我们使用PyTorch [48]构建了一个高性能训练软件栈，通过ATen库为大多数深度学习操作提供高效的CUDA实现，并通过PyTorch DistributedDataParallel库自动处理参数复制和梯度同步，以及通过重叠反向传播和AllReduce实现。我们已经启用了以下组件以实现高效的DLRM训练。

## 7.1 数据摄取

数据摄取是确保端到端训练性能的关键组件，特别是对于DLRM，它们通常处理的数据量比其他典型的DNN模型大得多。我们观察到，如果未经优化，数据摄取可能会引入显著的延迟，并为流水线带来非平凡的开销。

最初为分布式异步CPU设置设计的我们的读取器和数据预处理模块将每个稀疏特征的偏移量和索引存储在单独的张量中，每个嵌入表一个。因此，具有数百个嵌入表的DLRM可以轻松地在每次迭代中获得数千个输入张量，这转化为从CPU ↔ GPU传输的重大开销，并且是之前Zion平台的主要瓶颈之一，如第2节所述。

为了克服这一实际挑战，我们共同设计了数据预处理模块，使用组合格式，其中使用长度而不是偏移量，并将不同嵌入表的输入简单地连接起来。使用组合格式的好处是两方面的：（1）它通过合并小传输来优化CPU-GPU传输；（2）它可以直接被嵌入内核消耗，无需额外的布局转换。我们进一步通过使用固定内存来优化输入数据传输，以避免额外的复制。

有了组合格式，我们开发了一个模块，根据分片策略高效地分发嵌入表输入。在表格级分片（如图5a所示）的情况下，需要一个AlltoAll来将全局批次分发给每个工作器的本地表。由于索引的大小取决于长度的值，通信实际上是先进行长度的AlltoAll，然后进行索引的AlltoAll。在有𝑊个工作器，𝑇个本地表和𝐵个本地批次大小的设置中，这给我们提供了（𝑊，𝑇，𝐵）顺序的索引，需要进一步排列为（𝑇，𝑊，𝐵）以供嵌入内核消耗。我们已经开发了自定义的GPU内核，用于排列、分桶和复制，以实现表格级、行级和列级分片方案的最大吞吐量。模型检查点也面临类似的挑战，需要足够频繁地能够写出更大的模型，同时不成为训练的开销，如这篇最近的论文[12]所概述的。

## 7.2 通信原语

高性能的集体通信是DLRM训练表现良好和可扩展性的关键。PyTorch提供了进程组（PG）接口，用于集体操作——一个抽象的平台/集体库不敏感的API。DLRM直接（对于Alltoall）或间接（通过DDP对于Allreduce）使用这个API[32]。我们使用NVIDIA的集体通信库（NCCL）作为我们的主要集体通信库，因为它有效地使用RDMA和NVLINK以获得最佳性能。我们将PyTorch NCCL进程组实现扩展到支持使用NCCL Send/Recv原语的Alltoall/Alltoallv集体操作（需要NCCL 2.7.3或更高版本）。

# 8 评估

我们提供了生产模型端到端训练的结果，以及操作级别的性能分解。

## 8.1 实验设置

表2总结了配备8个NVIDIA A100 GPUs的单个ZionEX节点的聚合能力。节点中的8个GPU提供了总共320GB的HBM，聚合内存带宽为12.4TB/s。4个插槽的CPU提供了1.5TB的内存和320GB/s的带宽。在网络能力方面，GPU通过高带宽NVLink进行节点内GPU通信，每个GPU都有一个专用的200Gbps RoCE NIC用于节点间通信。我们在实验中使用了16个ZionEX节点的集群，总HBM容量为5TB。

## 8.2 端到端训练

我们报告了三个在生产中部署的DLRM的结果，这些DLRM用于不同的任务，包括点击通过率（CTR）预测、排序和参与度。表3列出了这些候选模型的高级特性。模型A代表了大型和复杂的DLRM，它们强调了Neo的计算能力和通信带宽，每个样本使用显著更高的FLOPS和大量的嵌入。模型F提出了一个不同的实际挑战，尽管每个样本的FLOPS很低，嵌入表的数量很少，但它有一个巨大的单一表，无法适应单个GPU的设备内存。最后，模型I代表了中等规模的DLRM，它们通过高平均嵌入池大小强调内存带宽。这些目标模型在集群中最多在16个ZionEX节点（128个GPU）上进行训练。模型质量以归一化熵[20]评估，训练吞吐量以每秒查询数（QPS）衡量。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/4a6900b195439fde58d264f1c74c96aa7f653419fc0f38c068f2dd6c196f1752ca123b1b83f6b19f1e874809fa26e3ac?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=8.jpg&amp;size=750">

图8

首先，我们使用模型A来演示训练质量，因为它也可以在分布式CPU平台上训练。如图8所示，尽管使用了更大的批量大小（64K vs. ~150），在ZionEX上同步大批量训练提供了相当或更好的模型质量（两者都使用调整过的超参数）。在相同的配置下，Neo在16个节点上的128个GPU上实现了1.2 MQPS，与我们之前的一代分布式CPU异步训练平台相比，速度提升了40倍，后者使用了45个参数服务器和15个训练器。以前的解决方案在不损害训练质量的情况下无法进一步扩展，而在ZionEX上完全同步训练允许在16个节点之外进行扩展，甚至可以使用更大的批量大小。

## 8.3 扩展性能

图9显示了在保持每个GPU批量大小不变的情况下，使用多达16个节点的模型A和模型I的归一化训练吞吐量。虽然数据并行训练的工作负载随着扩展保持不变，但由于模型并行策略，每个GPU的嵌入表数量随着扩展而减少。出于同样的原因，每个GPU为其本地表处理整个全局小批量，这与扩展成比例增加，并补偿了减少的表，使得这仍然是一个弱扩展实验。要在较小的节点数量上运行，我们减少了嵌入表的基数，并散列输入以适应减少的行数。这个缩小版本的模型有效地减少了模型大小，对性能特性的影响最小/没有影响，因此用于研究扩展性能。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/60cb2087947fb9c5f73d2978f7dad4a8e9c07123a7bf607d545ce8418aa2445958d3780611ecba613cbb562594ee3e9c?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=9.jpg&amp;size=750">

图9

从图中可以看到，在较大的节点数量上，模型A的扩展效率约为50%，模型I约为75%。尽管模型A和模型I在考虑目标本地批量大小时在有效FLOPS和内存需求方面非常接近，但模型A有更大的完全暴露的AlltoAll延迟。这是因为更多的嵌入表增加了AlltoAll负载，并且混合维度使得同时平衡嵌入计算和AlltoAll通信更加困难。因此，模型A在扩展时受到AlltoAll效率降低的影响更大。

为了更好地理解扩展性能，我们在图10中提供了模型A的序列化和暴露训练迭代延迟的分解。比较序列化和暴露延迟，CPU到GPU传输（即HtoD）完全隐藏，暴露的通信延迟远小于序列化AlltoAll和AllReduce延迟的总和。这证明了Neo的流水线优化有效性，可以重叠通信与计算（见第4.2节）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/be30d36ff2fceb504fb46585763c6989753b72a80d798648dba850b85e267517eac462013eb338c45d6e25be18b9246c?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=10.jpg&amp;size=750">

图10

随着节点数量的增加，我们观察到AlltoAll和AllReduce延迟增加。由于大多数AlltoAll通信在关键路径上，增加的AlltoAll成本直接影响暴露的通信和整体训练延迟。虽然AllReduce在多达16个节点上大部分被隐藏，但增加的AllReduce延迟和不变的计算延迟表明，一旦后向传递中的松弛完全被更高节点数量和/或更快的计算用完，AllReduce可能成为瓶颈。

## 8.4 训练吞吐量优化

以模型A作为案例研究，我们详细说明了各种优化及其在实现高达1.5 MQPS（如图11所示）中的贡献。此外，我们使用附录B中描述的性能屋顶线建模方法来建立可实现性能的上限，并确认报告的吞吐量在理论估计的15%以内。模型A在128个GPU上的基线性能低于700 KQPS。进一步的分析揭示了不同GPU之间嵌入查找延迟的巨大差异，表明存在严重的负载不平衡。通过结合表格级、列级和数据并行策略来处理约1000个嵌入表的≈1000s，将它们分配到128个GPU上，从而缓解了这个问题。请注意，尽管列级并行策略引入了额外的输入AlltoAll成本，但更好的负载平衡的好处超过了开销，总体QPS提高了20%。然而，扩展效率仍比理想的线性扩展低约30%。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/23628d252cdf6eac92c7553b637617a158f30bbddf54e57ed8cd6bd4cced7fd4a4042e1eb001248a0bd481d2292099cf?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=11.jpg&amp;size=750">

图11

如前所述，限制扩展效率的两个主要问题是：（1）负载不平衡和（2）增加的AlltoAll延迟。对于模型A，仅使用HBM进一步平衡负载特别具有挑战性，因为模型大小在TF32下接近128个GPU上的5TB聚合HBM容量。在扣除PyTorch框架和NCCL在每个等级上保留的内存后，Neo几乎没有空间探索放置策略。为了缓解这个问题，我们使用较低精度（FP16）的嵌入表，将模型大小减少了2倍。虽然这本身并不直接提供吞吐量好处，但Neo可以利用这个空间更好地平衡。结果，由于改善的负载平衡，训练吞吐量又增加了20%。

接下来，为了解决增加的AlltoAll延迟问题，我们采用了[65]中提出的量化集体通信，这直接减少了通信量。对于模型A，我们验证了在前向AlltoAll中使用FP16和在后向AlltoAll中使用BF16几乎提供了30%的速度提升，而没有训练质量损失。最后，我们将全局批量大小从64K增加到256K。这直接增加了激活大小，有助于更好地饱和GPU和通信带宽，同时与其他所有优化相辅相成。在适当调整优化器/超参数后，我们能够实现与训练质量相当的训练，但需要更全面的实验，因为DLRM的大批量训练研究得不够充分，将成为未来工作的一部分。总的来说，这些技术相比使用64K全局批量大小的TF32训练，训练吞吐量提高了87%。

## 8.5 模型容量限制研究

我们以模型F为例，在原型系统上推动模型容量。与模型A或模型I不同，有效训练模型F提出了2个不同的挑战。首先，模型F有12T参数，使用简单的训练方法，模型F很容易需要高达96TB的内存，远远超过了16个节点集群上的总内存。其次，模型只有几个巨大的嵌入表，每个表有约100B行和256列，每个表需要多节点的GPU和主机内存来训练。

为了将模型适配到16个节点上，我们首先应用逐行稀疏AdaGrad优化器到嵌入表，这将优化器状态从每个元素减少到每个嵌入行。然后我们在嵌入表上使用FP16精度[67]。这两个优化共同将模型内存占用从96TB降低到24TB，刚好适合4TB HBM + 24TB DRAM内存层次结构。在巨大的嵌入表上，我们启用逐行分片将表分布到多个节点，并调整训练流程使用AlltoAll与桶化和ReduceScatter，如图5b所示。在启用UVM并使用HBM作为缓存的情况下，我们能够以高达1.7 MQPS的吞吐量训练模型F，展示了我们HW/SW共同设计解决方案推动超越当前最先进技术的能力。

# 9 相关工作

研究人员提出了各种系统级创新来应对极大模型带来的挑战。

- DeepSpeed [50]在所有节点上完全分割模型参数、梯度和优化器状态，并使用检查点分区和重新物化[21, 28]来动态重建必要的状态，从而大幅减少内存使用。
- GShard [31]通过在张量级别标注并行策略，训练一个巨大的翻译模型，该模型跨加速器进行分割。
- FlexFlow [22]使用自动搜索来发现图中最佳的操作符并行策略。在自动并行化这一方向上，这些最近的论文[39, 60]使用最优合成和强化学习来找到优化的设备放置，以进一步提高并行性，无需手动干预。

然而，**上述这些通用系统并非专门为高度稀疏的推荐模型设计**。为此：

- 阿里巴巴引入了XDL [23]，这是一个为高维稀疏数据设计的工业级训练系统。XDL包含了诸如层次样本压缩、工作流流水线、零拷贝和CPU绑定等优化，以提高模型稀疏部分的训练效率。
- Kraken [62]针对更高效的在线训练，通过解耦键值获取和嵌入、与ML领域知识共同设计的缓存逐出策略、针对模型的稀疏和密集部分的内存高效优化器，以及允许推理服务器和参数服务器独立扩展的非共址部署模型。
- [25]通过无锁嵌入表更新、调整循环平铺来优化基于CPU的DLRM训练，AlltoAll通信原语，以及利用FP32和BFloat16中的位别名优势来减少内存占用的新split-SGD实现。
- 百度的AIBox [70]采取了不同的方法进行水平扩展，专注于在单个节点上适应大型推荐模型的训练。AIBox通过流水线网络、磁盘和CPU/GPU任务隐藏服务延迟，减少模型更新开销，并通过分组哈希方案和多级内存哈希系统提高SSD寿命。

由于**通信性能**已成为集群和数据中心规模分布式训练的主要瓶颈，因此对通信性能的关注越来越多。

- BytePS和ByteScheduler [24, 49]利用空闲CPU和网络资源以及更好的通信调度来提高参数交换效率。然而，在每个作业跨越多个节点的同质训练集群中，寻找和利用空闲网络资源的机会减少，导致这种方法的次优使用。
- SwitchML和ATP [30, 53]利用可编程网络交换机在数据中心环境中执行网络内聚合，以减少跨机架带宽。
- [6, 36]发现并利用数据中心网络的局部性，并通过学习和最优合成形成优化和动态的聚合路由。
- 这些论文[33, 34]通过使用各种量化方案来减少通信量，以解决通信开销问题。


# 参考

- [https://arxiv.org/pdf/2104.05158](https://arxiv.org/pdf/2104.05158)