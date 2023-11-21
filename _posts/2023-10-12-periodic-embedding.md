---
layout: post
title: Periodic embedding介绍
description: 
modified: 2023-10-12
tags: 
---

Yandex在《On Embeddings for Numerical Features in Tabular Deep Learning》中介绍了一种数值型特征建模embedding的方法：Periodic embedding。

# 摘要

最近，类似Transformer的深度架构在表格数据（tabular data）问题上表现出强大的性能。与传统模型（如MLP）不同，**这些架构将数值特征（numerical features）的标量值映射到高维embedding中，然后将它们混合在主网络（backbone）中**。在该工作中，我们认为数值特征的embedding在tabular DL中没有被充分探索，可以构建更强大的DL模型，并在一些对GBDT友好的基准测试中与GBDT竞争（即，在这些基准测试中，GBDT优于传统的DL模型）。我们首先描述了两种概念上不同的构建embedding模块的方法：

- 第一种**基于标量值的分段线性编码（piecewise linear encoding）**
- 第二种使用**周期性激活（periodic activations）**

然后，我们通过实验证明，与基于传统blocks（如线性层和ReLU激活）的embedding相比，这两种方法可以显著提高性能。重要的是，我们还表明，嵌入数值特征对许多主网络都有好处，不仅仅是对于Transformer。具体而言，在适当的嵌入后，简单的类MLP模型可以与attention-based的架构相媲美。总体而言，我们强调数值特征的embedding是一个重要的设计方向，在tabular DL中具有进一步改进的良好潜力。源代码可在https://github.com/yandex-research/tabular-dl-num-embeddings获得。

# 1.介绍

目前，表格数据（Tabular data）问题是深度学习（DL）研究的最后一道难关。虽然最近在自然语言处理、视觉和语音方面取得了深度模型的突破[12]，但它们在表格领域的成功尚未令人信服。**尽管已经提出了大量的表格DL架构[2、3、13、17、21、24、31、39、40]，但它们与“浅层”决策树集成（如GBDT）之间的性能差距通常仍然显著[13、36]。**

最近的一系列工作[13、24、39]通过成功将Transformer架构[45]调整为表格领域，缩小了这种性能差距。与传统模型（如MLP或ResNet）相比，提出的类Transformer架构具有一些特定方式来处理数据的数值特征。**即：将多个数值特征的scalar values分别映射到高维embedding vectors中，接着通过self-attention模块被混合在一起**。除了transformers，将数值特征映射到向量中也在点击率（CTR）预测问题中以不同形式被应用[8、14、40]。然而，文献大多集中在开发更强大的主网络（backbone），同时保持嵌入模块的设计相对简单。特别是，现有的架构[13、14、24、39、40]使用相当限制的参数映射（例如线性函数）构建数值特征的嵌入，这可能导致次优性能。在这项工作中，我们证明了嵌入步骤对模型的有效性有重大影响，其适当的设计可以显著提高表格DL模型的性能。

具体而言，我们描述了两种不同的构建块（building blocks），适用于数值特征的embeddings构建。

- 第一种是分段线性编码（ piecewise linear encoding）：它为原始标量值产生了可备选的初始表示（intial representations），并且它采用的基于特征分箱（feature binning）的方式，这是一种长期存在的预处理技术[11]。
- 第二种则依赖于周期性激活函数（periodic activation functions）：这是受到隐式神经表示[28、38、42]、NLP[41、45]和CV任务[25]中使用的启发。

第一种方法简单、可解释且不可微，而**第二种方法平均表现更好**。我们观察到，配备我们的embedding方案的DL模型在GBDT友好的基准测试中成功地与GBDT竞争，并在表格DL上实现了新的最先进水平。作为另一个重要发现，我们证明了嵌入数值特征的步骤对于不同的深度架构普遍有益处，不仅仅适用于类Transformer的架构。特别地，我们展示了，在适当的嵌入后，简单的类MLP架构通常可以提供与最先进的attention-based的模型相媲美的性能。总体而言，我们的工作展示了数值特征的嵌入对表格DL性能的巨大影响，并展示了未来研究中探索更高级嵌入方案的潜力。

# 2.相关工作

**表格深度学习（Tabular deep learning）**。在过去几年中，社区提出了大量用于表格数据的深度模型[2、3、13、15、17、21、24、31、39、40、46]。然而，当系统地评估这些模型时，**它们并没有始终优于决策树集成（ensembles of decision trees）**，如GBDT（梯度提升决策树）[7、19、32]，这些决策树集成通常是各种机器学习竞赛的首选[13、36]。此外，一些最近的研究表明，提出的复杂架构并不比经过适当调整的简单模型（如MLP和ResNet）优越[13、18]。在这项工作中，与之前的文献不同，我们的目标不是提出一种新的主网络（backbone）架构。相反，我们专注于更准确地处理数值特征的方法，它可以潜在地与任何模型结合使用，包括传统的MLP和更近期的类Transformer的模型。

**表格DL中的Transformer**。由于Transformer在不同领域取得了巨大成功[10、45]，因此一些最近的工作也将它们的self-attention设计适用于表格DL[13、17、24、39]。与现有的替代方案相比，将self-attention模块应用于表格数据的数值特征，需要将这些特征的标量值映射到高维嵌入向量中。到目前为止，**现有的架构通过相对简单的计算块（computational blocks）执行这种“标量(scalar)”→“向量(vector)”映射，在实践中，这可能限制模型的表达能力**。例如，最近的FT-Transformer架构[13]仅**使用单个线性层**。在我们的实验中，我们证明这种嵌入方案可以提供次优性能，而更高级的方案通常会带来巨大的收益。

**点击率（CTR）预估**。在CTR预估问题中，对象由数值和分类特征表示，这**使得这个领域与表格数据问题高度相关**。在一些工作中，数值特征以某种非平凡的方式处理，但并不是研究的核心[8、40]。最近，Guo等人提出了**更高级的方案AutoDis[14]。然而，它仍然基于线性层和传统的激活函数**，我们发现在我们的评估中这种方法是次优的。

**特征分箱（Feature binning）**。分箱是一种将数值特征（numerical features）转换为类别特征（categorical features）的离散化技术。即，**对于给定的特征，其值范围（value range）被分成若干个bins（intervals），然后原始特征值被替换为对应bin的离散描述符（例如bin indices或one-hot向量）**。我们指出Dougherty等人的工作[11]，它对一些经典的分箱方法进行了概述，并可以作为相关文献的入口。然而，在我们的工作中，我们以不同的方式利用bins。具体而言，**我们使用它们的边缘来构建原始标量值的无损分段线性表示**。结果表明，在一些表格问题（tabular problems）上，这种简单且可解释的表示（representations），可以为深度模型提供实质性的收益。

**周期性激活（Periodic activations）**。最近，周期性激活函数已成为处理类似坐标输入的关键组成部分，这在许多应用中都是必需的。例如，自然语言处理[45]、计算机视觉[25]、隐式神经表示[28, 38, 42]等。在我们的工作中，我们展示了周期性激活函数可用于构建强大的embedding模块，用于解决表格数据问题中的数值特征。与一些前述论文不同的是，在将多维坐标的各个分量（例如，通过linear layers）传递给周期性函数之前，我们发现将每个特征单独嵌入并在主网络（backbone）中混合它们至关重要。

# 3.Embeddings for numerical features

在本节中，我们描述了所谓的“数值特征embedding”的一般框架以及实验比较中使用的主要构建模块。符号表示：对于给定的表格数据监督学习问题，我们将数据集表示为：

$\lbrace(x^j, y^j)\rbrace_{j=1}^n$

其中：

- $y_j \in Y$：表示目标（object）的label
- $x_j = (x^{j(num)}, x^{j(cat)}) \in X$：表示目标（object）的features（num：数值型特征，cat：类别型特征）
- $x_i^{j(num)}$：则表示第 j 个目标（object）的第 i 个数值特征

根据上下文，可以省略 j 索引。数据集被分成三个不相交的部分：$\overline{1, n} = J_{train} \cup J_{val} \cup J_{test}$，其中：“train”部分用于训练，“validation”部分用于early stopping和hyperparameter tuning，“test”部分用于最终评估。

## 3.1 总框架

我们将“数值特征embedding”概念形式化为：

$$
z_i = f_i(x_i^{(num)}) \in R^{d_i}
$$

其中：

- $f_i(x)$：是第i个数值特征的embedding函数
- $z_i$：是第i个数值特征的embedding
- $d_i$：是embedding的维度

重要的是，所提出的框架意味着所有特征的embedding都是独立计算的。请注意，函数 $f_i$ 可以依赖于作为整个模型的一部分或以其他方式训练的参数（例如，在主要优化之前）。在本工作中，我们仅考虑embedding方案，其中：**所有特征的embedding函数具有相同的函数形式。我们【不共享】不同特征的嵌入函数的参数**。

embedding的后续使用取决于模型主网络（backbone）。对于类似 MLP 的架构，它们被拼接（concatenated）成一个flat向量（有关说明，请参见附录 A）。对于基于Transformer的结构，不会执行额外的步骤，embedding会直接传递，因此使用方式通过原始结构来定义。

## 3.2 Piecewise linear encoding

虽然vanilla MLP 被认为是通用逼近器（universal approximator） [9, 16]，但在实践中，由于optimization的特殊性，它在学习能力方面有限制 [34]。然而，Tancik 等人最近的工作 [42] 发现了一种case，即**改变输入空间**可以缓解上述问题。这个观察结果启发我们检查**改变数值特征的原始标量值的representations是否能够提高表格 DL 模型的学习能力**。

此时，我们尝试从简单的“经典”机器学习技术入手。具体而言，我们从one-hot encoding算法中获得灵感，该算法被广泛且成功地用于表示表格数据问题中的类别特征或NLP中的tokens等离散实体。**我们注意到，从参数效率和表达能力之间的权衡角度来看，one-hot表示可以看作是scalar表示的反向解决方案**。为了检查one-hot encoding类似方法是否有助于表格DL模型，我们设计了一种**连续的替代方法**来代替one-hot编码（因为普通的one-hot编码几乎不适用于数值特征）。

形式化地，对于第i个数值特征，我们将其值域分成不相交的$T_i$个区间 $B_i^1, \cdots, B_T^i$，我们称之为箱子：$B_t^i = [b_{t-1}^i, b_t^i)$。分割算法是一个重要的实现细节，我们稍后会讨论。从现在开始，为简单起见，我们省略特征索引i。一旦确定了bins，我们按照以下公式定义encoding scheme，详见公式1：

$$
PLE(x) = [e_1, \cdots, e_T] \in R^T \\
e_t = \begin{case}
0, \\
1, \\
\frac{x-b_t-1}{b_t - b_{t-1}}
\end{case}
$$

...(1)

其中，PLE表示“peicewise linear encoding”。我们在图1中提供可视化。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b97a94509cb470a657385eada6b29e04909e6f4ad0a4a3d105fb2f63b1737d18996901b57a0ec5d75319f917a723ff48?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1

注意：

- PLE为数值特征生成可替代的初始表示（representations），可以被看作是一种预处理策略。这些representations仅计算一次，然后在主要optimization过程中使用，代替原始的标量值。
- 当 T=1 时，PLE表示实际上等效于scalar表示。
- **与分类特征不同，数值特征是有序的**；通过将对应于右边界小于给定feature value的bin的分量设置为1，我们来表示这一点（这种方法类似于如何在序数回归问题中编码标签）
- 方案1也包括了 ($x < b_0$) 和 ($x ≥ b_T$)的case，它们分别产生 ($e_1 ≤ 0$) 和 ($e_T ≥ 1$)。
- 将representation进行**分段线性化（piecewise linear）**的选择本身就是一个值得讨论的问题。我们在第 5.2 小节中分析了一些替代方案。
- PLE 可以看作是特征预处理，这在第 5.3 小节中另外进行了讨论。

关于attention-based的模型的说明。虽然所描述的PLE表示可以直接传递给类似MLP的模型，但attention-based的模型本质上是不受 input embeddings顺序影响的，因此需要一个额外的步骤：**在所获得的encodings中添加关于特征indices的信息**。从技术上讲，我们观察到：只需要在PLE之后放置一个linear layer（不共享特征之间的权重）即可。然而，从概念上讲，该解决方案具有明确的语义解释。也就是说，它等价于：**为每个bin $B_t$分配一个trainable embedding $v_t \in R_d$，并通过以 $e_t$ 为权重将它的bins的embedding进行聚合、并再加上偏移$v_0$，来获得最终feature embedding**。具体地：

$$
f_i(x) = v_0 + \sum\limits_{t=1}^T e_t \cdot v_t = Linear(PLE(x))
$$

在接下来的两个部分中，我们描述了两种简单的算法来构建适合PLE的bins。具体而言，我们依赖于经典的分箱算法 [11]，其中一个算法是无监督的，另一个算法利用标签来构建箱子。

### 3.2.1 从分位数中获取bins

一种自然的baseline方法是：根据相应的单个特征分布的经验分位数，将值范围进行分割来构建PLE的箱子。具体而言，对于第i个特征：

$$
b_t = Q_{\frac{t}{T}} (\lbrace x_i^{j(num)} \rbrace_{j \in J_{train}}) 
$$

其中 Q 是经验分位数函数。

size为0的无意义bins将被删除。在第 D.1 小节中，我们展示了所提出方案在 Gorishniy 等人 [13] 中描述的合成 GBDT 友好数据集上的有用性。

### 3.2.2 构建 target-aware bins

实际上，还有一些利用training labels来构建bins的监督方法 [11]。直观地说，这些目标感知(target-aware)的算法，目标是在产生与可能目标值在相对较窄范围内相对应的bins。我们工作中使用的监督方法在精神上与 Kohavi 和 Sahami [23] 的“C4.5 离散化”算法相同。简而言之，对于每个特征，我们使用目标值（target）作为指导，以贪心方式递归地将值范围（value range）进行分割，这相当于构建一棵决策树（仅使用此特征和目标值进行生长），并将其叶子对应的区域作为PLE的bins（请参见图4中的说明）。此外，我们定义$b_i^0 = min_{j \in J_{train}} x_i^j$ 和 $b_T^i = max_{j \in J_{train}} x_i^j$ 。

## 3.3 周期性激活函数（Periodic activation functions）

回顾第 3.2 小节中提到的 Tancik 等人 [42] 的工作被用作我们开发 PLE 的动机起点。因此，我们也尝试将原始工作本身适应于表格数据问题。我们的变化有两个方面的不同。首先，我们考虑到子节 3.1 中描述的嵌入框架在嵌入过程中禁止混合特征（请参见子节 D.2 进行额外讨论）。其次，我们训练预激活系数而不是保持它们固定。因此，我们的方法与 Li 等人 [25] 非常接近，其中“组”的数量等于数值特征的数量。我们在公式 2 中形式化描述所述方案，

$$
f_i(x) = Periodic(x) = concat[sin(v), cos(v)], v = [2 \pi c_1 x, \cdots, 2 \pi c_k x]
$$

...(2)

其中：

- $c_i$是可训练参数，从N(0, σ)初始化。

我们观察到 σ 是一个重要的超参数。σ 和 k 都使用验证集进行调整。

## 3.4 简单可微层（Simple differentiable layers）

在深度学习的背景下，使用传统的可微分层（例如线性层、ReLU 激活等）对数值特征进行嵌入是一种自然的方法。事实上，这种技术已经在最近提出的attention-based的架构 [13、24、39] 、以及在一些用CTR预测问题的模型 [14、40] 中单独使用。但是，我们也注意到这样的传统模块可以在子节3.2和子节3.3中描述的组件之上使用。在第4节中，我们发现这样的组合通常会导致更好的结果。


# 4.实验

在本节中，我们对第 3 节中讨论的技术进行了实证评估，并将它们与GBDT进行比较，以检查“DL vs GBDT”竞争的现状。

我们使用了来自以前的表格 DL 和 Kaggle 竞赛的 11 个公共数据集。重要的是，我们专注于中大规模的任务，并且我们的基准测试偏向于 GBDT 友好的问题，因为目前，在这些任务上缩小与 GBDT 模型之间的差距是表格 DL 的主要挑战之一。主要数据集属性总结在表 1 中，使用的来源和其他细节在附录 C 中提供。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b1f2279eff361c54ce16147c63f577df02985a8da97a03ccf53aa2486c81874d8b9fb4d74255d7eca2013448ddf5cf3c?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t1.jpg&amp;size=750">

表1

## 4.2 实现细节

我们在超参数调整、训练和评估协议方面主要遵循 Gorishniy 等人 [13]。然而，为了完整起见，我们在附录 E 中列出了所有细节。在下一段中，我们描述了特定于数值特征嵌入的实现细节。

**数值特征嵌入（Embeddings for numerical features）**。如果使用线性层（linear layers），则调整其输出维度。对于所有特征，PLE的超参数是相同的。

- 对于quantile-based PLE，我们会调整分位数的数量
- 对于target-aware PLE，我们调整以下决策树的参数：叶子的最大数量、每个叶子的最小items数、以及在生长树时做出一个split所需要的最小信息增益。
- 对于Periodic module（参见公式 2），我们调整 σ 和 k（这些超参数对于所有特征都是相同的）。

## 4.3 模块名

在实验中，我们考虑了不同的骨干网和嵌入组合。为方便起见，我们使用“骨干网-嵌入”模式来命名模型，其中“骨干网”表示骨干网（例如 MLP、ResNet、Transformer），而“嵌入”表示嵌入类型。请参见表 2，了解所有考虑的嵌入模块。请注意：

- Periodic在公式 2 中定义。 
- $PLE_q$表示quantile-based PLE
- $PLE_t$表示target-aware PLE
- Linear_ 表示无偏线性层（bias-free linear layer），LReLU 表示leaky ReLU，AutoDis 是 Guo 等人 [14] 提出的。 
- “Transformer-L” 等价于 FT-Transformer [13]。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/62db36ed0f0632e6c93db55e027bee3faab4d4ebe4e7cbdf103b88746cba6a5350928ed08ab2219a09a15742c2d7728d?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t2.jpg&amp;size=750">

表2

## 4.4 简单可微embedding模块

我们首先评估由“传统”可微分层（线性层、ReLU 激活等）组成的嵌入模块。结果总结在表 3 中。 主要观点：

-  首先&最重要的，结果表明 MLP 可以从embeddings模块中受益。因此，我们得出结论，在评估embedding模块时，该backbone值得关注。 
- 当应用于MLP时，简单的LR模块可以导致保守、但一致的提升。

有趣的是，“冗余（redundant）”的 MLP-L配置也倾向于优于vanilla MLP。虽然改进并不显著，但这种架构的特殊属性是，linear
embedding模块可以在训练后，与 MLP的第一个线性层融合在一起，从而完全消除了开销。至于 LRLR 和 AutoDis，我们观察到这些重型模块（heavy modules）不值得额外的成本（请参见附录 F 中的结果）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/943275229abc6db6f6e73cc0e30206bb58cce4193bf01772937191c1409ebe3b354b38b83ebb55fcea7aa3f5afc2ceaa?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t3.jpg&amp;size=750">

表3

## 4.5 Piecewise linear encoding

在本节中，我们评估子节 3.2 中描述的编码方案。结果总结在表 4 中。

主要结论： 

- 分段线性编码对两种类型的架构（MLP 和 Transformer）通常有益，并且收益可能很显著（例如，参见 CA 和 AD 数据集）。 
- 在PLE顶部添加可微组件可以改善性能。尽管如此，最昂贵的修改（如 Q-LRLR 和 T-LRLR）不值得这样做（请参见附录 F）。

请注意，基准测试偏向于 GBDT 友好的问题，因此在表 4 中观察到的基于树的箱通常优于基于分位数的箱，可能不适用于更适合 DL 的数据集。因此，我们在这里不对两种方案的相对优势做任何一般性的声明。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/e0a008adcbecbb15ff333fffe588e48dc92813c2217ea95ee0818c491d360d75fa8f85d3f9ed17384eeb4faee26c3eac?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t4.jpg&amp;size=750">

表4

## Periodic activation functions

在本节中，我们评估基于周期性激活函数的嵌入模块，如子节 3.3 中所述。结果报告在表5 中。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/3dda928f6060eaa2f0895a9d0d86d74cfeed7320175f014ccf35c12382d24dd6f32c691512f6760892ea735968ee6d4c?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t5.jpg&amp;size=750">

表5

主要结论：平均而言，MLP-P 优于普通的 MLP。然而，在周期性模块的顶部添加可微分组件应该是默认策略（这与 Li 等人 [25] 的观点一致）。事实上，MLP-PLR 和 MLP-PL 在 MLP-P 的基础上提供了有意义的改进（例如，参见 GE、CA、HO），甚至在 MLP-P 不如 MLP 的情况下“修复”了 MLP-P（OT、FB）。

虽然 MLP-PLR 通常优于 MLP-PL，但我们注意到，在后一种情况下，嵌入模块的最后一个线性层在表达能力上是“冗余”的，并且可以在训练后与骨干网的第一个线性层融合在一起，这理论上可以导致更轻量级的模型。最后，我们观察到 MLP-PLRLR 和 MLP-PLR 之间的差异不足以证明 PLRLR 模块的额外成本（请参见附录 F）。

## 4.7 Comparing DL models and GBDT

在本节中，我们进行了不同方法的大比较，以确定最佳的embedding模块和主网络（backbone），并检查数值特征embedding是否使 DL 模型能够在更多任务上与 GBDT 竞争。重要的是，我们比较 DL 模型的集合与 GBDT 的集合，因为梯度提升本质上是一种集成技术，因此这样的比较将更加公平。请注意，我们只关注最佳指标值，而不考虑效率，因此我们只检查 DL 模型是否在概念上准备好与 GBDT 竞争。

我们考虑三个主网络（backbone）：MLP、ResNet 和 Transformer，因为据报道它们代表了基线 DL 骨干网目前的能力 [13、18、24、39]。请注意，我们不包括也在对象级别上应用注意力的attention-based的模型 [24、35、39]，因为这个非参数组件与我们工作的核心主题不相关。结果总结在表6 中。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/1f3116eddd5c289ef5117eedb69745a535cec0a7f0ea566604679e837aff0e58b59eb2b72d3e04d44d3ef9d3f4021432?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t6.jpg&amp;size=750">

表6

DL 模型的主要结论：

- 对于大多数数据集，数值特征的嵌入可以为三种不同的骨干网提供显著的改进。虽然平均排名不是制定微妙结论的好指标，但我们强调 MLP 和 MLP-PLR 模型之间平均排名的巨大差异。
- 最简单的 LR 嵌入是一个很好的基准解决方案：虽然性能提升并不显著，但它的主要优点是一致性（例如，参见 MLP vs MLP-LR）。
- PLR 模块提供了最好的平均性能。根据经验，我们观察到 σ（见公式 2）是一个重要的超参数，应该进行调整。 
- 分段线性编码（PLE）允许构建表现良好的嵌入（例如，T-LR、Q-LR）。除此之外，PLE 本身也值得关注，因为它具有简单性、可解释性和效率（没有计算昂贵的周期函数）。
- 重要的是，将类似 MLP 的架构与集成学习相结合之后，它们可以在许多任务上与 GBDT 竞争。这是一个重要的结果，因为它表明 DL 模型可以作为集成学习中 GBDT 的替代方案。

“DL vs GBDT” 竞争的主要结论：数值特征的嵌入是一个重要的设计方面，具有极大的潜力来改进 DL 模型，并在 GBDT 友好的任务上缩小与 GBDT 之间的差距。让我们通过几个观察来说明这个说法：

- benchmark最初偏向于GBDT友好的问题，这可以通过比较GBDT解决方案与vanilla DL模型（MLP、ResNet、Transformer-L）来观察。
- 然而，对于绝大多数“主网络&数据集”对，合适的embedding是缩小与GBDT之间差距的唯一方法。例外（相当正式）包括：MI数据集以及以下配对：“ResNet & GE”、“Transformer & FB”、“Transformer & GE”、“Transformer & OT”。 
- 另外，据我们所知，在众所周知的California Housing和Adult数据集上，**DL模型的表现与GBDT相当**，这是第一次。

尽管如此，与GBDT模型相比，对于DL架构，效率仍然可能是一个问题。在任何情况下，tradeoff完全取决于具体的使用情况和需求。

# 5.分析

## 5.1 model size对比

为了量化数字特征嵌入对模型大小的影响，我们在表7中报告了参数量。总的来说，引入数值特征embeddings可能会导致不可忽视的模型大小方面的开销。重要的是，在训练时间和吞吐量方面，size的开销并没有转化为相同的影响。例如，在CH数据集上，MLP-LR的参数计数几乎增加了2000倍，但训练时间仅增加了1.5倍。最后，在实践中，我们发现将MLP和ResNet与embeddings模块结合，产生了仍然比Transformer-based的模型更快的架构。

表7

## 5.2 消融研究

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/824037d03aa37446e799861f91b3bbd70716152e2976a1571f1f7332f07a95db2e38df6790d721cec8939acaff281d01?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t7.jpg&amp;size=750">

表8

在本节中，我们比较了两种基于分箱的编码方案（binning-based encoding：

- 一种是“温度计（thermometer）”（[6]）：会设置值为1，取代掉piecewise linear term；
- 另一种是one-blob encoding的泛化版本（[29]；见第E.1节了解更多细节）。tuning和evaluation协议与第4.2节相同。表8中的结果表明，让基于分箱的编码（binning-based encoding）进行分段线性化（piecewise linear）是一个很好的默认策略。

## 5.3 Piecewise linear encoding作为特征预处理技术

众所周知，标准化（standardization）或分位数变换（quantile transformation）等数据预处理，对于DL模型达到好效果来说至关重要。而且，不同类型的预处理之间的效果可能会有显著差异。同时，PLE-representations仅包含[0,1]中的值，并且它们对于平移（shifting）和缩放（scaling）是不变的，这使得PLE本身成为一种通用特征预处理技术，通用适用于DL模型，无需首先使用传统预处理。

为了说明这一点，在第4节中使用了分位数变换（quantile transformation）来评估数据集。我们使用不同的预处理策略重新评估了MLP、MLP-Q和MLP-T的已调整配置，并在表9中报告了结果（注意，对于具有PLE的模型，标准化等同于无预处理）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ec8337230b64662cc596b0d05b5deb7eb88a53f9d598f5992a1171b0874d18bd69f777dc98168ccaa5c5d6d34ff3579d?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t9.jpg&amp;size=750">

表9

首先，如果没有预处理，vanilla MLP往往变得无法使用。其次，对于vanilla MLP，选择特定类型的预处理（CA、HO、FB、MI）是很重要的，而对于MLP-Q则不那么明显，对于MLP-T则不是（尽管这种特定观察可能是基准测试的特性，而不是MLP-T的特性）。总的来说，结果表明使用PLE模型相对于vanilla MLP对预处理的敏感性较低。这对于实践者来说是一个额外的好处，因为使用PLE后，预处理方面变得不那么重要。

## 5.4 “feature engineering”的角度

乍一看，特征嵌入（feature embeddings）可能类似于特征工程（feature engineering），并且应该适用于所有类型的模型。然而，所提出的embedding schemes是受基于DL-specific方面训练的启发（请参见第3.2节和第3.3节的动机部分）。虽然我们的方法可能会很好地迁移到具有相似训练属性的模型上（例如，对于线性模型，它们是深度模型的特例），但通常并非如此。为了说明这一点，我们尝试采用周期性模块来调整XGBoost的随机系数，同时保持原始特征而不是丢弃它们。调整和评估协议与第4.2节相同。表10中的结果表明，尽管这种技术对于深度学习模型很有用，但它并不能为XGBoost提供任何好处。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/f2db647a8d1f3d10fb00e5da058484c8c338e9de4061f2ba2a97f612e9c77719df7fe7f773caa899ef4d462d22bb638e?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t10.jpg&amp;size=750">

表10

# 

- 1.[https://arxiv.org/pdf/2203.05556.pdf](https://arxiv.org/pdf/2203.05556.pdf)