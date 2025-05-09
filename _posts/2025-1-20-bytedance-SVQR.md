---
layout: post
title: 字节SVQR
description: 
modified: 2025-1-20
tags: 
---

字节在《Real-time Indexing for Large-scale Recommendation by Streaming Vector Quantization Retriever》提出了流式向量量化检索器。我们来看一下它的实现：

# 摘要

检索器作为推荐系统中最重要的阶段之一，负责在严格的延迟限制下高效地为后续阶段选择可能的正样本。因此，大规模系统通常会使用一个简单的排序模型，依靠近似计算和索引来粗略地缩小候选规模。考虑到简单模型缺乏生成精确预测的能力，大多数现有方法主要集中于引入复杂的排序模型。然而，索引有效性的另一个基本问题仍未解决，这也成为了复杂化的瓶颈。在本文中，我们提出了一种新颖的索引结构：流式向量量化模型（Streaming Vector Quantization model），作为新一代的检索范式。**流式VQ能够实时为item附加索引，赋予其即时性**。此外，通过对可能变体的细致验证，它还实现了索引平衡和可修复性等额外优势，使其能够像现有方法一样支持复杂的排序模型。作为一种轻量级且易于实现的架构，流式VQ已在抖音和抖音极速版中部署，并取代了所有主要的检索器，带来了显著的用户参与度提升。

### 1 引言

在现代推荐系统中，我们不断面临爆炸性增长的语料库，因此由检索、预排序和排序阶段组成的级联框架变得非常普遍。在这些阶段中，检索器的任务是从整个语料库中区分候选样本，但给定的时间却最少。例如，在抖音中，检索器需要从数十亿条内容中筛选出数千个候选样本，而后续阶段只需将候选规模缩小10倍。

然而，扫描所有候选样本会带来极高的计算开销，因此检索阶段不得不依赖于索引结构和近似计算。具体来说，诸如**乘积量化（Product Quantization, PQ [8]）**和**分层可导航小世界（Hierarchical Navigable Small World, HNSW [11]）**等索引方法被提出。PQ通过创建“索引”或“聚类”来表示属于它们的全部内容。当一个聚类被选中时，其所有内容都会被检索出来。同时，用户侧和内容侧的信息被解耦为两个独立的表示，用户表示用于搜索相关聚类。这导致了一种“双塔”架构 [2, 7]，其中每个塔由一个多层感知机（MLP）实现。由于其显著降低计算开销的能力，这种方法在许多工业场景中得到了广泛应用。在下文中，我们将其称为“HNSW双塔”。

尽管HNSW双塔架构简单，但它存在两个缺点：

- （1）其**索引结构需要定期重建**，在此期间内容表示和内容索引分配是**固定的**。然而，在一个充满活力的平台上，新内容每秒都在提交，聚类语义也会因新兴趋势而变化，而这些变化在建模中被忽略了。此外，这种构建过程与推荐目标并不一致。
- （2）**双塔模型很少提供用户-内容交互，因此生成的预测较弱**。不幸的是，在大规模应用中，复杂的模型（如MLP）会带来难以承受的计算开销。

许多现有方法都聚焦于这些问题，并开发了新的索引结构。然而，这些方法主要设计用于支持复杂模型，而忽略了索引本身的关键问题。根据我们的实践经验，**索引即时性**和**索引平衡性**与模型复杂性同样重要。如果索引结构严重失衡，热门内容会集中在少数几个索引中，导致模型难以区分它们。例如，在深度检索（Deep Retrieval, DR [4]）中，我们从路径中收集了500𝐾条内容，而仅排名第一的路径就生成了超过100𝐾个候选样本，这严重降低了检索效果。

在本文中，我们提出了一种新颖的索引结构——**流式向量量化（streaming VQ）模型**，以提升检索器的能力。Streaming VQ具有独特的实时将内容分配到合适聚类的特性，使其能够捕捉新兴趋势。此外，我们还详尽地研究了每种变体，以确定实现索引平衡的最佳解决方案。Streaming VQ使得索引内的内容可区分，因此它能够在保持优异性能的同时生成更紧凑的候选集。尽管它主要关注索引步骤，但它也支持复杂模型和多任务学习。凭借这些创新机制，**streaming VQ在抖音和抖音极速版中超越了所有现有的主流检索器**。事实上，它已经取代了所有主要检索器，带来了显著的用户参与度提升。本文提出的模型的主要优势总结如下：

- **实时索引分配与自修复能力**：内容在训练过程中被实时分配到索引中，并且索引能够自我更新和修复。整个过程无需中断步骤。
- **平衡的索引结构**：Streaming VQ 提供了平衡良好的索引，这有助于高效地选择内容。通过一种合并排序（merge-sort）的改进，所有聚类都有机会参与到推荐过程中。
- **多任务学习的优秀兼容性**：Streaming VQ 展现出与多任务学习的出色兼容性，并且能够支持与其他方法相同的复杂排序模型。
- **易于实现的特性**：最后但同样重要的是，与近期的工作相比，Streaming VQ 以其易于实现的特性脱颖而出。它具有简单清晰的框架，主要基于 VQ-VAE [17] 的现成实现，这使得它能够轻松部署于大规模系统中。

### 2 相关工作

如前所述，由于扫描整个语料库的计算开销过高，各种索引结构被提出以在可接受的误差范围内近似选择候选样本。**乘积量化（Product Quantization, PQ [8]）**就是这样一个例子，它将内容聚集到聚类中。当某些聚类被选中时，属于这些聚类的所有内容都会被检索出来。**可导航小世界（Navigable Small World, NSW [10]）**通过逐步插入节点来构建图，形成节点之间的捷径以加速搜索过程。**分层可导航小世界（Hierarchical Navigable Small World, HNSW [11]）**提供了分层结构，能够快速缩小候选规模，因此被广泛采用，尤其是在大规模场景中。此外，还有一些基于树的方法 [6, 14] 和**局部敏感哈希（Locality Sensitive Hashing, LSH）**方法 [15, 16]，旨在近似选择候选样本。

在建模方面，迄今为止最流行且基础的架构是所谓的“双塔模型”，其主要源自 **DSSM [7]**。双塔模型将用户侧和内容侧的原始特征分别输入到两个独立的多层感知机（MLP）中，并获取相应的表示（嵌入）。用户对某个内容的兴趣通过这两个嵌入的点积来表示。由于它将内容和用户信息解耦，在服务阶段可以预先存储内容嵌入，并通过**近似最近邻（Approximate Nearest Neighbor, ANN）**方法搜索结果。

然而，解耦用户和内容信息会丢弃它们之间的交互，而这种交互只能通过复杂模型（如MLP）来实现。为了解决这个问题，**基于树的深度模型（Tree-based Deep Models, TDM [25], JTM [24], BSAT [26]）**提出了树状结构，以从粗到细的层次化方式搜索候选样本。在TDM中，内容被收集在叶子节点上，而一些虚拟的非叶子节点用于表示其子节点的整体属性。TDM采用复杂的排序模型，并通过注意力模块交叉用户和内容信息。考虑到HNSW本身已经提供了分层结构，**NANN [1]** 直接在HNSW上搜索候选样本，同样使用复杂模型。

另一种方法试图避免ANN算法所需的欧几里得空间假设。**深度检索（Deep Retrieval, DR [4]）**主要由等距层组成，将内容定义为“路径”，并使用束搜索（beam search）逐层缩小候选范围。与TDM和NANN相比，它更关注索引而非排序模型的复杂性。还有一些方法 [9, 12] 使用多索引哈希函数对内容进行编码。

尽管上述方法主要集中在模型复杂性上，**BLISS [5]** 强调了索引平衡的重要性。它通过迭代强制模型将内容映射到桶中，甚至手动将一些内容分配到尾部桶中以保证平衡。

将内容附加到索引本质上是将它们“量化”为可枚举的聚类。因此，可以考虑**向量量化（Vector Quantization, VQ）**方法。从引入可学习聚类的**VQ-VAE [17]** 开始，许多方法 [22, 23] 已经考虑在检索任务中使用它或其变体。在本文中，我们将VQ模型发展为一种以流式方式更新、保持平衡、提供灵活性且轻量级的索引方法，并将其命名为“**流式VQ（streaming VQ）**”。

### 3 流式向量量化模型（Streaming VQ）

通常，检索模型包括**索引步骤和排序步骤**：

- **检索索引**步骤使用近似搜索从初始语料库中逐步缩小候选范围
- **检索排序**步骤则为后续阶段提供有序的结果和更小的候选集

大多数现有方法都遵循这种两步范式。例如，最流行的双塔架构本质上利用HNSW来高效搜索候选样本。在特定操作轮次中，它首先通过排序模型对邻居节点进行排序（排序步骤），然后选择内容并丢弃其他内容（索引步骤）。同样，TDM和NANN模型也依赖于它们自己的索引结构（基于树/HNSW）。DR主要引入了一种可检索的结构，在实践中我们还需要训练一个排序模型来对结果进行排序，并为索引步骤提供用户侧输入嵌入。DR与其他方法的不同之处在于，DR的索引步骤和排序步骤是按时间顺序执行一次，而其他方法中这两个步骤是交替执行的。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/2ffd46a7032aed0fbd3dfa56325154b2118549af5f82890dc813312bf59a85dbf455d6d0479a0e7c81b0c4454e02281f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图 1 提出的流媒体VQ模型的训练框架

本文提出的流式VQ模型也由两个按时间顺序执行的步骤组成。在图1中，我们展示了其完整的训练框架（注意，流式VQ可以扩展到多任务场景，但为了简单起见，我们暂时只考虑预测视频是否会被播放完成的“完成任务”）。在索引步骤中，我们采用双塔架构（原因将在第5.5节讨论），并通过独立的塔生成**内容侧和用户侧**的中间嵌入 **v** 和 **u**（图1中的深蓝色和黄色块）。首先，这两个中间嵌入通过**一个辅助任务**进行优化，该任务采用in-batch Softmax损失函数：

$$
𝐿_{aux} = \sum_{o} -\log \frac{\exp(\mathbf{u}_o^T \mathbf{v}_o)}{\sum_{r} \exp(\mathbf{u}_o^T \mathbf{v}_r)},
$$

其中：

- $o$ 和 $r$ 表示样本索引。

**量化操作出现在内容侧**：我们保持一组可学习的聚类（单任务版本为16𝐾，多任务版本为32𝐾），并分配𝐾个嵌入。当生成 **v** 时，它在聚类集中搜索最近的邻居：

$$
k^*_o = \arg\min_k ||\mathbf{e}^k - \mathbf{v}_o||^2,
$$

...(2)

$$
\mathbf{e}_o = \mathbf{e}^{k^*_o} = Q(\mathbf{v}_o),
$$

...(3)

其中：

- $Q(\cdot)$ 表示量化
-  **e** ：所选聚类嵌入
- **u** ：用户侧嵌入

它们一起优化：

$$
𝐿_{ind} = \sum_{o} -\log \frac{\exp(\mathbf{u}_o^T \mathbf{e}_o)}{\sum_{r} \exp(\mathbf{u}_o^T \mathbf{e}_r)}.
$$

...(4)

搜索到的聚类作为输入内容的“索引”。这种内容-索引分配被写回参数服务器（Parameter Server, PS）。我们遵循标准的指数移动平均（Exponential Moving Average, EMA [17]）更新：**聚类嵌入通过其所属内容的移动平均值进行更新，而内容而非聚类接收聚类的梯度**。EMA过程在图1中用红色箭头表示。

检索排序步骤与检索索引步骤共享相同的特征嵌入，并生成另一组紧凑的用户侧和内容侧中间嵌入。由于在此步骤中，更复杂的模型优于双塔架构，因此可以使用交叉特征和3D用户行为序列特征。我们基于连接嵌入为每个任务预测一个独立的塔（头），并由相应的标签进行监督。详细的模型架构可以在第3.5节中找到。

在服务阶段，我们首先通过以下公式对聚类进行排序：

$$
\mathbf{u}^T \cdot Q(\mathbf{v}).
$$

然后，所选聚类的内容被输入到下一个排序步骤中，并生成最终结果。

以上介绍了所提出方法的基础框架，在本节的剩余部分，我们将详细阐述如何在几个特别关注的方面进行改进，包括索引即时性、可修复性、平衡性、服务技巧，以及如何与复杂模型和多任务学习集成。

#### 3.1 索引即时性

现有检索模型的整体更新周期由候选扫描（检查哪些内容可以被推荐）、索引构建和模型转储组成。其中，主要成本来自索引构建。

对于所有现有检索模型，索引构建是中断的，这导致**索引语义的即时更新被忽略**。例如，**在抖音中，由于我们有数十亿规模的语料库，构建HNSW大约需要1.5-2小时，执行DR中的M步需要1小时**。在此期间，索引保持不变。然而，在一个快速发展的平台上，新兴趋势每天都在出现。这种情况不仅需要实时将新提交的视频分配到适当的索引中，还需要同时更新索引本身。否则，它们无法相互匹配，只会产生不准确的兴趣匹配和较差的性能。相反，我们的模型通过**流式样本进行训练，内容-索引分配会立即决定并实时存储在PS中（键=内容ID，值=聚类ID），无需中断阶段，并且聚类嵌入通过优化目标强制适应内容**。这赋予了它最重要的优势：索引即时性。

现在，在流式VQ中，索引构建变为实时步骤，因此我们已经克服了主要障碍。此外，我们将候选扫描设置为异步，因此整体模型更新周期等于模型转储周期，仅需5-10分钟。

即便如此，仍存在一个潜在问题：内容-索引分配完全由训练样本决定。由于热门内容频繁曝光，它们的分配得到了充分更新。然而，新提交的和不受欢迎的内容获得曝光或更新的机会较少，这进一步恶化了它们的表现。

**为了解决这个问题，我们添加了一个额外的数据流——候选流（candidate stream）——来更新它们**。与称为“曝光流”的训练流不同，候选流只是以等概率逐个输入所有候选内容。如图1（虚线黑色箭头）所示，对于这些样本，我们仅通过前向传播获取并存储内容-索引分配，以确保其与当前聚类集的语义匹配。由于这些样本没有真实标签，因此不计算损失函数或梯度。

### 3.2 索引可修复性

流式更新范式是一把双刃剑：**由于我们放弃了索引重建，整个模型面临性能退化的风险**。这种现象广泛存在于所有检索模型中，但通常通过重建操作来解决。现在对于流式VQ，我们需要在没有重建操作的情况下解决这个问题。

原始的VQ-VAE引入了两个损失函数：一个与 $𝐿_{ind}$ 相同，另一个强调内容-聚类相似性：

$$
𝐿_{sim} = \sum_{o} ||\mathbf{v}_o - \mathbf{e}_o||^2.
$$

...(6)

在计算机视觉领域 [3, 13]，模式很少变化，因此VQ类方法表现良好。然而，在大规模工业推荐场景中，内容自然会发生归属变化，但 $𝐿_{sim}$ 反而会锁定它们。

在我们早期的实现中，我们遵循了与原始VQ-VAE相同的配置，起初在线指标确实有所改善。然而，我们观察到模型退化：性能随着时间的推移逐渐恶化。随后我们意识到，在我们的平台上，由于全局分布漂移，作为内容的概括表示，聚类的语义每天都在变化。内容-索引关系并非静态，相反，内容可能在不同天内属于不同的聚类。不幸的是，$𝐿_{ind}$ 和 $𝐿_{sim}$ 都只描述了内容属于某个聚类的情况。如果它不再适合该聚类，我们不知道它应该属于哪个聚类。这就是性能退化的原因。

通过用 $𝐿_{aux}$ 替换 $𝐿_{sim}$，我们解决了这个问题。由于 $𝐿_{aux}$，内容嵌入可以及时独立地更新，然后 $𝐿_{ind}$ 根据内容表示调整聚类。经过这一修改后，我们成功观察到了持续的改进。我们将其总结为设计检索模型的原则：**内容优先**。内容决定索引，而不是相反。

### 3.3 索引平衡性

推荐模型应能够区分热门内容，并为后续阶段精确选择所需内容。具体来说，对于检索模型，我们希望它们将内容均匀分布在索引中，以便我们只需选择少数索引即可快速缩小候选集。这种特性称为“索引平衡性”。不幸的是，许多现有方法存在流行度偏差，未能提出有效的技术来防止热门内容集中在少数几个顶级索引中。为了缓解这种偏差，BLISS [5] 甚至强制将一些内容分配到尾部聚类。

注意到 $𝐿_{ind}$ 在平均情况下获得最小的量化误差。热门内容占据的曝光量远多于其他内容，因此最小化 $𝐿_{ind}$ 的最直接方法是将它们拆分并分配到尽可能多的聚类中，这自然会实现良好的平衡性。在我们的实现中，流式VQ确实采用了这一策略，并产生了令人惊讶的平衡索引分布（见第5.1节）。

为了进一步提高索引平衡性，我们修改了主要的正则化技术。设 $\mathbf{w}$ 为初步的聚类嵌入，我们在EMA中插入一个流行度项：

$$
\mathbf{w}^{t+1}_k = \alpha \cdot \mathbf{w}^t_k + (1 - \alpha) \cdot (\delta^t)^\beta \cdot \mathbf{v}^t_j,
$$

其中内容 $j$ 属于聚类 $k$，$t$ 表示时间戳，$\delta$ 表示内容出现间隔，如 [21] 中提出的。这里我们添加了一个超参数 $\beta$ 来调整聚类行为，较大的 $\beta$ 会促使聚类更关注不受欢迎的内容。然后，我们还更新记录聚类出现次数的计数器 $c$：

$$
c^{t+1}_k = \alpha \cdot c^t_k + (1 - \alpha) \cdot (\delta^t)^\beta,
$$

最终表示计算为：

$$
\mathbf{e}^{t+1}_k = \frac{\mathbf{w}^{t+1}_k}{c^{t+1}_k}.
$$

我们还在向量量化步骤中提出了“扰动”，即将公式(2)修改为：

$$
k^*_o = \arg\min_k ||\mathbf{e}_k - \mathbf{v}_o||_2 \cdot r,
$$

$$
r = \min\left(\frac{c_k}{\sum_{k'} c_{k'}/K} \cdot s, 1\right),
$$

其中 $r$ 表示折扣系数，$s = 5$ 是一个阈值。这意味着如果整个聚类的曝光量少于平均值的 $1/s$ 倍，则在内容搜索其最近聚类时会被提升。这也有助于构建一个平衡良好的索引结构。

### 3.4 服务阶段的合并排序

内容的表示可能具有两种内在语义：个性化和流行度。我们希望根据内容的个性化而非流行度进行聚类。为此，我们显式地将内容表示解耦为个性化部分（嵌入）和流行度部分（偏差）。数学上，将公式(5)修改为：

$$
\mathbf{u}^T \cdot Q(\mathbf{v}_{emb}) + v_{bias}.
$$

通过这种方式，我们观察到同一聚类内的内容在语义上更加一致。所有训练损失函数也遵循相同的修改。

注意到在公式(11)中，即使同一聚类内的内容具有相同的 $Q(\mathbf{v}_{emb})$，$v_{bias}$ 也可以用于粗略排序。因此，我们提出了一种合并排序解决方案，以有效选择候选内容进入检索排序步骤。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/91045025f5b848ed690dd8382407348fa4386e337f3715d114e9e021c8461b5d082189ee35793e47790009d8651ee2a1?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图 2

如图2(a)所示，$\mathbf{u}^T \cdot Q(\mathbf{v}_{emb})$ 提供了聚类排名，而 $v_{bias}$ 提供了聚类内内容的排名。然后基于这两部分的总和进行合并排序。这确保所有聚类（即使是那些大小超过排序步骤输入的聚类）都有机会为最终结果提供候选内容，因此我们可以收集一个非常紧凑的集合（50𝐾，仅为DR排序步骤输入大小的10%）以超越其他检索器。

具体来说，我们在这里使用最大堆来实现k路合并排序（图2(b)）。聚类的内容首先独立排序并形成列表，这些列表被分成块（大小=8）。然后将这些列表构建成一个堆，由其头部元素初始化。在每次迭代中，我们从堆中弹出顶部元素，但取出其块中的所有元素。然后，来自同一列表的另一个块及其头部元素被添加到堆中。该策略在保持性能质量的同时有效减少了计算开销。更多细节请参见附录A。

### 3.5 模型复杂性

如前所述，在检索索引步骤和检索排序步骤中，我们分别评估16𝐾和50𝐾个聚类/内容。这一规模不再难以承受，因此我们可以使用复杂模型。在图3中，我们展示了索引和排序模型的两种架构：双塔架构和复杂架构。

双塔模型（图3左侧）遵循典型的DSSM [7] 架构。内容侧特征和用户侧特征分别输入到两个独立的塔（即MLP）中，并获得紧凑的嵌入。用户对该内容的兴趣通过这两个嵌入的点积计算。特别地，我们为每个内容添加了一个偏差项，并将其添加到最终得分中，以表示内容的流行度。在排序步骤中使用双塔模型的版本称为“VQ双塔”。

复杂版本（图3右侧）也将内容侧和用户侧特征输入以生成两个中间嵌入。然而，内容侧嵌入被输入到一个多头注意力模块 [19] 中作为查询，以提取非线性用户-内容交互线索，其中用户行为序列被视为键和值。然后，转换后的特征以及其他所有特征（包括交叉特征）被输入到一个深度MLP模型中以输出结果。使用复杂排序模型的版本称为“VQ复杂”。

理论上，这两种架构都可以部署在索引和排序步骤中。然而，在我们的实验中，复杂的索引模型并未带来改进。正如第5.5节所讨论的，复杂模型提供的非线性接口违反了欧几里得假设，并可能将聚类和内容划分到不同的子空间中，从而遗漏一些聚类。因此，我们将索引模型保持为双塔架构。

相反，对于排序步骤，复杂版本优于双塔版本。然而，它也带来了更多的计算开销。考虑到投资回报率（ROI），并非所有目标都部署为复杂版本。详细信息见第5.3节。

作为一个娱乐平台，抖音有许多热门话题和新兴趋势，这些内容集中在用户的近期行为序列中。然而，顶级话题已经被充分估计和分发，因此由热门话题主导的序列特征几乎无益于兴趣建模。为了解决这个问题，我们利用Trinity [20] 提供的统计直方图，过滤掉落在用户前5个次要聚类中的内容（填充更多内容以达到足够长度）。生成的序列倾向于长尾兴趣，并提供更多的语义线索。通过修改后的序列特征，某些目标得到了显著改善（见第5.3节）。

我们还在VQ复杂版本中添加了数十个特征以达到其最佳性能。仅添加特征或增加模型复杂性只能产生适度的结果。然而，通过结合这两种技术，我们获得了显著改进的结果。原因是，随着更多特征的加入，我们的模型能够实现高阶交叉并真正利用复杂性。

### 3.6 多任务流式VQ

尽管前面的讨论是基于单任务框架的，但流式VQ可以扩展到多任务场景。如图1所示，在索引步骤中，用户对每个任务都有独立的表示，但它们共享相同的聚类集。对于每个任务，我们同时计算 $𝐿_{aux}$ 和 $𝐿_{ind}$ 并传播梯度。

对于多任务版本，聚类表示需要针对不同任务进行专门化。具体来说，公式(7)和公式(8)被修改为：

$$
\mathbf{w}^{t+1}_k = \alpha \cdot \mathbf{w}^t_k + (1 - \alpha) \cdot \prod_{p} (1 + h_{jp})^{\eta_p} \cdot (\delta^t)^\beta \cdot \mathbf{v}^t_j,
$$

$$
c^{t+1}_k = \alpha \cdot c^t_k + (1 - \alpha) \cdot \prod_{p} (1 + h_{jp})^{\eta_p} \cdot (\delta^t)^\beta,
$$

其中 $\eta$ 是另一个用于平衡任务的超参数，$h_{jp}$ 是内容 $j$ 在任务 $p$ 中的奖励。例如，如果视频未完成/完成，则 $h_{jp} = 0/1$。对于停留时间目标，它被设计为对数播放时间。注意，整个奖励始终大于1，因此聚类会倾向于产生更高奖励分数的内容。

检索排序步骤为所有任务共享特征嵌入，并训练各自的双塔或复杂模型。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ebdd011462b66edf450e553a1117e539879ca79856e662d308eca3727bc4d7502af061603fbba7a365dd30521fc5d20a?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t1.jpg&amp;size=750">

表 1

---

### 4 检索模型的详细分析

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/d8c508bfec9c50e92ecefcec9ed0ee3b3435f632653bbf7411672a84f18da6a29183438934ba2d9a9aa14509980b8172?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图 3

在这里，我们将流式VQ与其他现有方法进行比较，并说明为什么它有利于大规模工业应用。

表1列出了我们关注的检索模型的7个方面，并逐一讨论：

- **索引是否面向推荐？** 在本文中，“面向推荐”衡量索引构建过程是否针对推荐目标进行了优化。在HNSW中，索引的构建不考虑其分配的任务。类似地，DR由于其M步骤也不是面向推荐的检索器。由于 $𝐿_{aux}$ 和 $𝐿_{ind}$ 都受到推荐目标的监督，流式VQ是面向推荐的。

- **索引步骤中的负采样方法**：HNSW和NANN在索引步骤中不涉及负采样方法。TDM引入了一种随机负采样方法，选择同一层级的另一个节点作为负样本。特别地，DR具有隐式负采样：由于所有节点都通过Softmax归一化，当我们最大化其中一个节点时，其他节点等效地被最小化。然而，这种最小化未考虑样本分布，因此DR仍然严重受到流行度偏差的影响。在我们的实现中，流式VQ在索引步骤中保持双塔架构，因此我们可以直接采用 [21] 中引入的现成in-batch去偏解决方案。

- **流行度去偏**：如上所述，DR无法避免热门内容集中在同一路径中。在我们的系统中，DR索引步骤后总共收集了500𝐾个候选内容，而排名第一的路径提供了100𝐾个。相反，由于第3.3节中提出的所有技术，流式VQ中的热门内容广泛分布在索引中。尽管大多数现有方法都关注复杂性，但我们认为流行度去偏是另一个被忽视但至关重要的问题。

- **构建索引的时间成本**：在抖音中，我们需要1.5-2小时来设置HNSW，并需要1小时来执行DR的M步骤。在流式VQ中，索引在训练过程中实时构建和更新。

- **索引步骤的候选限制**：这意味着我们可以处理多少候选内容作为输入。由于需要存储一些元信息（例如边），它受到单机内存的限制。作为最复杂的结构，HNSW只能存储170𝑀个候选内容。由于我们的语料库规模超过了这一限制，因此会定期随机丢弃一些内容。DR的结构（一个内容可以通过3条路径检索）大大简化，因此我们可以将阈值扩展到250𝑀。当前的流式VQ具有独占结构，因此理论上它可以存储比DR多3倍的候选内容（详细分析见附录B）。我们仅扩展到350𝑀，因为更多的候选内容可能会带来一些过时的信息。

- **排序步骤的节点接触**：这里我们展示了系统中每种方法的实际设置，而不是上限。由于HNSW/TDM/NANN在分层结构中检索候选内容，对于它们来说，排序步骤的节点接触指的是它们计算的总次数，而对于DR/流式VQ，它表示排序列表的大小。为了公平比较，我们将NANN和流式VQ的节点接触次数设置为相同（见第5.4节）。注意，由于流式VQ具有平衡良好的索引结构，并且可以在聚类内精细选择内容，因此即使排序候选规模减少10%，它仍然优于DR。

- **适用的排序模型**：使用复杂的排序模型总是会显著增加计算开销。众所周知，HNSW无法支持复杂架构。在抖音中，由于投资回报率（ROI）较低，DR在排序步骤中也使用双塔模型。其他检索模型使用复杂架构。

# 5 实验

在本节中，我们剖析了流式 VQ 的性能，包括聚类可视化和在线指标。然后，我们解释为什么我们更关注索引结构，而不是开发复杂的排序模型。我们还讨论了是否需要索引复杂化/多层 VQ。

## 5.1 平衡且不受流行度影响的索引

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/648c75bd06d2621053b816e1dfae18040ae1665479082efc90ede0ab2684d914910c3d8a8462b6efb0932cc5f44cca74?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=4.jpg&amp;size=750">

图 4

在图 4 中，我们通过统计直方图（上）和 t-SNE [18]（下）可视化索引分布。在直方图中，我们将聚类按其所属item数量进行聚合。从结果来看，大部分聚类包含的item数 ≤ 25K。考虑到我们有一个十亿规模的语料库和 16K 聚类，在理想的均匀分布下，每个聚类将分配到数万个item。我们得到的结果相当接近这种理想分布。

另一幅图描述了它们在二维空间中的聚合程度，颜色越深表示聚类越大。首先，所有点均匀覆盖整个区域，这意味着它们与其他聚类在语义上是不同的。然后，每个层级的点，尤其是大聚类的点，分散开来，甚至在局部也没有聚集。这表明索引结构能够抵抗流行度的影响。

因此，我们可以得出结论，流式 VQ 确实提供了平衡且不受流行度影响的索引。

## 5.2 工业实验环境

在本文中，所有实验均在我们的大规模工业应用中实施：抖音和抖音 Lite，用于视频推荐。作为一个娱乐平台，我们专注于提升用户参与度，即日活跃用户（DAUs）。由于用户被均匀分配到对照组和实验组，因此无法直接测量 DAUs。我们遵循 Trinity [20] 中的相同指标。我们计算实验期间用户的平均活跃天数作为平均活跃天数（AAD），平均活跃小时数作为平均活跃小时数（AAH），并将观看时间作为辅助指标。

由于检索器是作为单任务模型进行训练的，因此总是存在指标权衡。例如，优化完成目标的检索器可能通过简单地增强短视频的分发来实现，这将导致更多的展示次数（VV），但观看时间会下降。一般来说，一个可接受的上线应该在展示次数和观看时间上保持平衡（例如，增加 0.1% 的观看时间，但减少 0.1% 的展示次数）。一个更有效的检索器应该同时提高观看时间和展示次数。

在检索阶段，我们已经部署了数百个检索器。因此，我们更倾向于那些占据足够展示次数的检索器，这通过展示比例（IR）来衡量。IR 计算这个检索器贡献了多少展示次数，不进行去重。根据我们的经验，IR 是最敏感且最具预测性的指标。一般来说，如果它的 IR 提高了，我们就得到了一个更有效的检索器。

将检索模型升级为流式 VQ 涉及以下目标：停留时间（ST）、完成（FSH）、有效观看（EVR）、活跃停留时间（AST）、个人页面停留时间（PST）、旧候选停留时间（OST）、评论区停留时间（CST）和 Lite 停留时间（LST）。具体来说，停留时间目标衡量用户观看视频的时间，如果他/她观看了超过 2 秒，则记录为正样本。我们根据实际播放时间给正样本分配奖励。AST/PST/CST 描述了相同的信号，但出现在喜欢页面/个人页面/评论区，而不是信息流标签中。OST 和 LST 也建模了停留时间目标，OST 只是将 ST 应用于 1-3 个月前发布的候选item，而 LST 是专门为抖音 Lite 训练的。完成直接描述了视频是否被看完。有效观看是一个综合目标：它首先通过分位数回归预测观看时间等于 60%/70%/80%/90% 持续时间，然后通过加权和融合预测。

## 5.3 在线实验

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/63d368d68aeed16d78542309b40e859e52be28139341812e69128775a13851e41665f3aa829d1e1ff81b0817dad948c1?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t2.jpg&amp;size=750">

在表 2 中，我们展示了在线性能，其中只列出了具有统计显著性的指标。首先，对于每个模型，两种变化（从 HNSW/DR 到 VQ 双塔，以及从 VQ 双塔到 VQ 复杂模型）在 IR 上都提供了显著的改进。正如前面所展示的，这表明了更好的内在有效性，通常指的是索引平衡、即时性等。

所有实验在观看时间、AAD 和 AAH 上都产生了显著的改进，或者至少具有竞争力的表现。我们可以得出结论，流式 VQ 是比 HNSW 和 DR 更好的索引结构（与 NANN 相比，见第 5.4 节），并且 VQ 复杂模型优于 VQ 双塔。然而，令人惊讶的是，仅索引升级就产生了令人信服的 AAD 增益。这表明，尽管大多数现有工作都集中在复杂性上，但索引的有效性同样重要。

对于完成目标，“*” 表示复杂模型的序列特征没有通过 Trinity 进行去偏。通过比较两行相邻的数据，去偏版本在所有指标上都优于另一个版本，这表明长尾行为为全面描述用户的兴趣提供了补充线索。
抖音和抖音 Lite 在 DAUs 方面已经有一个非常高的基线。此外，检索阶段对展示结果的影响已经被 IR 按比例减少。检索模型的变化多年来没有为 AAD 提供显著的好处。然而，通过流式 VQ 替代，我们在几次上线中见证了令人印象深刻的改进。这验证了流式 VQ 作为一种新型检索模型范式的潜力。

## 5.4 索引优先，还是排序优先？

为了更好地理解索引和排序步骤在大规模场景中所起的作用，我们还进行了在线实验，比较了基于 EVR 目标的 NANN [1]（最先进的检索模型）与所提出的方法。为了公平比较，我们确保 NANN 和 VQ 复杂模型具有完全相同的计算复杂性。请注意，NANN 和 VQ 复杂模型也使用了更多的特征。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/58cc7554575718fe539fbf7baed7726cb3592b5dc5d0673e112fc9324e821d34f6e7be8b708b8e259de82a01951d65cf?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t3.jpg&amp;size=750">

表 3

在表 3 中，我们将“HNSW 双塔”作为基线，并列出其他模型的性能。VQ 双塔、NANN 和 VQ 复杂模型依次提供了越来越好的结果，通过观看时间/AAH 来衡量。从这些结果来看，NANN 似乎与两种 VQ 架构具有竞争力。然而，一方面，正如我们在第 5.2 节中所展示的，NANN 比其获得的观看时间失去了更多的 VV，这并不是非常有效。另一方面，在图 5 中，我们可视化了它们的展示分布（与 HNSW 双塔相比的相对差异），这也得出了不同的结论。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/4a9685eb85acbaf8c0b067a8c0f84d661d157775996fb93a1635a2ce070889dd3506cbabdcddf7f928d762cb0e5d2e8f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=5.jpg&amp;size=750">

图 5

为了吸引用户，分发更多热门item（VV>1 亿）一直是一个捷径，因为它们很容易产生更多的观看时间和点赞次数。但一个更有效的系统能够精确匹配兴趣，因此小众话题可以获得更多展示机会。例如，添加更多特征也可以增强系统对不受欢迎item的理解，并改善它们的分发。从这个方面来看，两种 VQ 架构符合我们的期望：VQ 双塔将“1 万 - 10 万”的展示量提高了约 2%，同时将“1 亿 +”减少了 1%。此外，VQ 复杂模型甚至将近乎 5% 的“1 万 - 5 万”展示量提高了，同时将近乎 2% 的“1 亿 +”减少了。然而，NANN 保持了不变的分布，这表明它没有充分利用特征和复杂性。总之，VQ 复杂模型在观看时间和 AAH 上优于 NANN，同时减少了热门item的分发。因此，它是我们应用中的更好模型。

可以得出结论，仅仅复杂化排序模型是不足以充分利用模型结构和特征所提供的所有优势的。这是因为整个模型的性能受到索引步骤的限制。只有拥有先进的索引模型，复杂化才能实现其理想性能。因此，我们建议优先优化索引步骤，特别是在大规模场景中。

## 5.5 索引复杂化

正如第 3 节所展示的，我们也可以在索引步骤中使用复杂的模型。然而，它意外地提供了较差的结果。为了找出原因，我们进一步实施了以下变化：(1) 保持双塔头部，并根据公式（10）附加索引，确定item - 索引分配；(2) 将 e 和 v 输入复杂的模型，如图 3 所示，但不从它那里接收梯度；(3) 除了 e 和 v 之外，共享两个头部的所有其他特征嵌入和 DNN 参数。通过这种方式，我们将item中间嵌入和聚类嵌入强制到相同的语义空间，并尽可能相似。令人惊讶的是，它仍然给出了较差的结果。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/2824285660a7757de52e0dcc68dd6701ae2a7addb3a4d648bbd9c2339c79c08aa5ba3de664c4f697762cc68e06bfacfe?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=6.jpg&amp;size=750">

图 6

为了理解这种现象，想象我们有两个正样本（item）及其聚类（图 6 中的蓝色圆圈，较深的一个表示聚类）。在双塔索引版本中（左），它遵循欧几里得假设，模型只产生近线性界面，因此聚类与其item保持在相同的子空间中。

# 

[https://arxiv.org/pdf/2501.08695](https://arxiv.org/pdf/2501.08695)