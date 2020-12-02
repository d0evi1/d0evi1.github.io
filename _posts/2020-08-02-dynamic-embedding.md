---
layout: post
title: dynamic embedding介绍
description: 
modified: 2020-08-02
tags: 
---


# 摘要


深度学习模型的一个限制是：input的sparse features，需要在训练之前定义好一个字典。本文提出了一个理论和实践系统设计来解决该限制，并展示了模型结果在一个更大规模上要更好、更高效。特别的，我们通过将内容从形式上解耦，来分别解决架构演进和内存增长。为了高效处理模型增长，我们提出了一个新的neuron model，称为DynamicCell，它受free energy principle的启发，引入了reaction的概念来排出non-digestive energy，它将gradient descent-based方法看成是它的特例。我们在tensorflow中通过引入一个新的server来实现了DynamicCell，它会接管涉及模型增长的大部分工作。相应的，**它允许任意已经存在的deep learning模型来有效处理任意数目的distinct sparse features（例如：search queries），可以不停增长，无需重新定义模型**。最显著的是，在生产环境中运行超过一年它仍是可靠的，为google smart campaingns的广告主提供高质量的keywords，并达到极大的accuracy增益。

## 1.1 Motivation

为了理解一些已存在的深度学习库的限制，我们考虑一个简单示例：对每天的来自在线新闻上的新闻文章上训练一个skip-gram模型。这里模型的训练实例是相互挨着的一些words，期望的实现是将每个word映射到一个vector space上，它们在语义空间中也相接近。为了实验word2vec算法，我们需要定义一个字典变量，它包含了待学习embeddings的所有的words。**由于在训练前需要一个字典（dictionary），这限制了模型的增长，很难处理从未见过的words或者增加embedding的维度**。

## 1.2 核心

为了更好适应模型增长，我们尝试搜寻一个框架，**它可以将一个neural network layer的input/output看成是满足特定分布的充分统计(sufficient statistics)（即：embeddings），我们提出了一个与free energy principle的概念有关的新neuron model称为DynamicCell**。直觉上，通过对interal state进行调节(regulating)及行动(take actions)，可以最小化它的自由能（free energy）。另外，当input包含了non-digestive energy时，它也会通过reaction将它们排出(discharge)，以维持一个稳定的internal state。我们可以看到对free-energy priciple做小修改，可以让它与传统的gradient descent-based算法来关联。因此，对一个layer的一个input signal可以被连续（continuously）或组合（combinatorially）的方式处理。例如，**当在input端上看到一个新的input feature时，该layer可以为它动态分配一个embedding，并将它发送到upstream layers上以便进一步处理**。

为了实现上述思想，会对tensorflow做出一些修改。特别的，会在tensorflow python API中添加一些新的op集合，来直接将symbolic strings作为input，同时当运行一个模型时，"intercept" forward和backward信号。**这些op接着会访问一个称为“DynaimicEmbeddding Service(DES)”的新的server，来处理模型的content part**。在模型的forward execution期间，这些op会为来自DES的layer input抽取底层的float values（embeddings），并将这们传递给layer output。与backward execution相似，计算的gradients或其它信息，会被传给DES，并基于用户定制的算法来更新interal states。

实际上，DES扮演着扩展Tensorflow的角色，主要有以下几方面影响：

- embedding data的虚拟无限容量（Virtually unlimited capacity）：通过与外部存储系统（比如：Bigtable或Spanner）合作，**可以将模型能力逼近存储的上限**。实际上，我们的系统可以与任意支持kv数据的lookup/update的存储系统相适应
- 灵活的梯度下降更新：**DES可以保持关于一个layer的全局信息**，比如：词频或平均梯度变化，来帮助它决定何时更新一个embedding。Gradient descent update对于每个变量来说不再是均齐过程 (homogeneous process)，每个layer通过采用合适的actions可以维护它自己的“homeostasis”。同时，它也保证了我们的系统与任意gradient descent optimizers（SGD、AdaGrad、Momentum）是后向兼容的。
- 高效：在DES上的计算/内存加载会自动分布到云平台的worker机上。**训练速度与tensorflow workers成正比，模型容量（capacity）由DynamicEmbedding workers的数目决定**。
- 可靠性：有了DES，**tensorflow模型可以变得非常小**，因为大多数数据都被保存到像Bigtable的额外存储中。因此，当训练一个大模型时对于机器失败（由于超过资源限制）变得很有弹性。
- 对迁移学习或多任务学习的支持：通过采用tensorflow的embedding data，**多个模型可以共享相同的layer**，只需要简单使用相同的op name以及DES配置即可。因此，模型会共享一个norm，而非一个option。

我们的DynamicEmbedding系统被证明是在大规模深度学习系统中非常重要的，并且它在多个应用中稳定运行超过一年。**带DynamicEmbedding的tensorflow模型可以和不带该功能的tensorflow运行一样快**，新增的优点是：更大的capacity，更少的编码，更少的数据预处理工作。工程师切换到DynamicEmbedding的主要工作是：**学习新的APIs和配置额外的存储（比如：Bigtable或Spanner），这可以尽可能的简化**。

在过去两年，由于我们的系统上线，我们移植了许多流行的模型，特别是涉及到在训练前需要sparse features的模型，它们可以满足来自input的持续增长。例如，image annotation中使用upgraded Google Inception模型，它可以从来自海量搜索queries的lables中进行学习；用于机器翻译的GNMT的模型，它可以被用于将句子翻译成多数语言描述；我们升级版的Word2vec可以以任意语言快速发现与任意root query相似的的queies。

通过采用DynamicEmbedding, 我们发现，单个不需要任意预处理的模型足够达到令人满意的效果。特别的，对比其它rule-based系统，我们的sparse feature models之一（从网站内容中给出关键词suggesting）可以达到相当高的accurate结果。通过允许系统由**数据自我演化来驱动**，它可以快速胜过其它需要人工调整的系统。

系统总览：

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/1429f60bbfa9e920c386faf26e3acba45e6a8f9aec74a51a43701779a18e92321e60cdae1b4718e2309d7b542745c085?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1

图1展示了我们添加到tensorflow的扩展组件的总览。整体目标是：让存在的tensorflow APIs只处理模型的static part：定义nodes，connections，将数据相互间进行传递，并**将trainable variable的lookup/update/sample操作传到DynamicEmbedding Service(DES)上来允许它们构建和动态增长**。另外，我们也需要定义一个新的python APIs集合，它可以直接将string Tensors作为input，将它们的embeddings作为output。这些tensorflow APIs可以直接访问一个称为**DynamicEmbedding Master(DEM)的组件**，它们会将实际job轮流分发给**DynamicEmbedding Workers(DEWs)**。DEWs负责embedding lookup/update/sample，并与外部云存储（比如：Bigtable或Spanner）进行通信，并基于多种gradient descent算法来更新embedding values。

# 2.数学公式

free energy principle的一个基本思想是，规定：一个生态系统趋向于最小化“surprise”，定义为：

$$
log\frac{1}{P(s | m)}
$$

其中：

- s是一个系统的当前internal和external state；
- m是解释s的一个internal model

我们可以将这样的思想与neural networks相关联，通过**将"surprise"重定义为一个具有contextual inputs与不具体congextual input的state分布间的差异（通过KL-divergence衡量）**，分别表示成：$$P(w \mid c)$$和$$P(w)$$。对于上述原始的公式，我们的新方式可以在一个cell level上实现，不再需要使用一个内部预测模型m来解释state s（它本身可以是一个非常复杂的process）。我们展示了BP算法在embedding space的free-energy最小化的一个通用过程，它会给人工神经网络（artificial neural network：ANN)带来一个新的思路：**一个ANN是关于inter-connected neurons的一个group，它会最小化自己的free energy**。在其余部分，我们会详细解释neural networks的新方法，以及它带来的实际影响，比如：现实中的一个系统设计和提升。

## 2.1 Exponential family, embedding和人工神经网络

使用neural networks来表示sparse features的represent已经在自然语言模型中广泛探索。本质上，在neural network中的layer仅仅只是它的variables对于特定分布$$P(w_1, \cdots, w_n \mid c_1, \cdots, c_m)$$的充分统计。[47]更进一步将这样的思想泛化到许多已经存在的DNN模型中，并派生了embedding space的一个新等式，来解释contextual input到output的相关度。例如，**在NN中的一个layer可以被看成是在embedding空间中$$P(w \mid c)$$分布的一个表示**，其中：**c是layer的contextual input，w是output**。

更进一步假设：

$$
P(w \mid c) \propto exp(\langle\vec{w}, \vec{c}\rangle) 
$$

其中：$$\vec{w}$$和$$\vec{c}$$分别表示w和c的embeddings

接着一个layer可以基于$$\vec{c}$$来简单计算$$\vec{w}$$。

这与传统观念：neurons相互间基于单个动作电位进行通信（action potentials：表示成1D function（二元binary or 连续continuous））来说是个挑战。另外，它偏向于一个更现实的观点：**neurons实际上会与它们的firing patterns【9】相通信，以便单个neuron不会只与单个bit相通信**。【47】采用了probability作为一种描述firing patterns分布的通用语言，并使用embeddings(sufficient statistics)来表示它们的近似形式。

DNN的另一个视角的一个明显优化是：建模能力。如果我们限制AI来定义activation function的组合，不管我们赋予它们什么含义，他们总是会落入解决非常相似形式的问题：

$$
min_{\theta=\lbrace \theta_1, \cdots, \theta_n\rbrace} \sum\limits_{x \in D} L(x, \theta) \equiv f_1(f_2(\cdots f_n(x, \theta_n), \cdots; \theta_2), \cdots, \theta_1), n \in N
$$

...(1)

其中，D表示一整个training data或一个mini-batch。等式(1)的gradients可以通过使用chain rule来对可学习参数集合$$\theta_i$$进行计算，对于每个$$f_i, i=1, \cdots, n$$：

$$
\frac{\partial L(x, \theta)}{\partial \theta_i} = \frac{\partial L(x, \theta)}{\partial f_i} \frac{\partial f_i}{\partial \theta_i} = \frac{\partial L(x, \theta)}{\partial f_1} \frac{f_1}{f_2} \cdots \frac{\partial f_{i-1}}{\partial f_i} \frac{\partial f_i}{\partial \theta_i}
$$

...(2)

从$$f_1$$到$$f_n$$递归计算$$\frac{\partial L(x,\theta)}{\partial f_i}$$和$$\frac{\partial L(x,\theta)}{\partial \theta_i}$$的算法，称为“back-propagation”。定义一个loss function，接着通过back-propagation算法来求解它是人工神经网络的标准做法。

从上面的过程，**如果back-propagation算法一次只运行一个batch，可以看到我们可以更改x或$$\theta_i, i\in \lbrace 1,2,\cdots,n \rbrace$$的维度**。然而，已存在的deep learning库的设计不会将它考虑成一个必要特性。在本节其余部分，我们提出了一个新框架来解释模型增长。


## 2.2 增长需要

一个智能系统的一个基本需要是：能够处理来自感知输入（sensory input）的新信息。当我们在一个neural network中处理一个新的input时，必须将它转化成一个representation，可以由像等式（1）（其中$$x \in R^m$$）的loss function处理。特别的，**如果该input涉及到离散对象（比如：words）时，它必须将它们映射到一个embedding space中**。对于该需求的一个naive解释可以从neural network的视角看：一个discrete input c可以被表示成一个特征向量（one-hot）：$$\vec{c}_{0/1} = [0, \cdots, 1, \cdots, 0]^T$$，接着通过一个linear activation layer，它可以变成$$W \vec{c}_{0/1}=W_i$$，其中$$W_i$$表示real matrix W中的第i列，或等价的，c就是embedding。这样的解释可以说明：这对于使用sparse input values的DNN实现来说是个限制，以及为什么总是需要一个字典（比如：一个字典定义为W）。

实际上，特征向量$$\vec{c}_{0/1}$$的维度（比如：W中的列数）可以增长到任意大，embedding维度（比如：W中的行数）也会相应增长。为了观察embedding dimension为什么增长，我们对neural network layers采用sufficient statistics的视角，一个基本事实是一个embedding的每个dimension都应该被限定。也就是说，假设neural network的一个layer表示了$$P(w \mid c) \propto exp(\langle \vec{w}, \vec{c} \rangle)$$。那么，两个inputs $$c_1$$和$$c_2$$它们相应的分布相互完全分离，它们可以被认为是不同的。假设：$$P_{c_1}(w) \equiv P(w \mid c_1)$$并且$$P_{c_2}(w) \equiv P(w \mid c_2)$$，这可以表示成：

$$
D_{KL} (P_{c_1} \| P_{c_2}) \equiv \int_w P(w \mid c_1) \frac{log P(w | c_1)}{log P(w | c_2)} > \delta
$$

...(3)

其中，$$D_{KL}(P \mid Q)$$表示两个分布P和Q间的KL散度，$$\delta > 0$$是一个threshold。通过**将embedding的形式$$P(w \mid c)$$（例如：$$P(w \mid c) \propto exp(<\vec{w}, \vec{c}>)$$）代入到上面的等式**，我们可以获得：

$$
D_{KL}(P_{c_1} \| P_{c_2} \propto \int_w P(w | c_1) \langle\vec{w}, \vec{c_1} - \vec{c_2}\rangle
$$

...(4)

几何上，它会沿着方向$$\vec{c_1} - \vec{c_2}$$来计算vector $$\vec{w}$$的平均长度。由于$$\vec{c}$$的长度是限定的，当distinct c的数目增长时，让等式(3)的不等式总是成立的唯一方法是：增加$$\vec{c}$$和$$\vec{w}$$的维度。直觉上可以简单地说：为了在一个限定空间中填充更多objects，以便它们相互间隔得足够远，我们必须扩展到更高维度上。

## 2.3 新的neuron model: DynamicCell

现在，我们已经解决了为什么（why）一个AI系统会增长，另一个问题是how：一组neurons只通过input/output signals相互连接，在一起工作来达到整体的稳态？**一个理想的neuron model不应解释单个cell是如何工作的，而是要泛化到groups of cells，甚至groups of organisms**。更好的是，它也能解释在deep learning中广泛使用的已经存在方法（比如：BP算法）的成功。

### 2.3.1 free energy principle的动机

free energy principle是为了**理解大脑的内部运作**而发展起来的，它提供给我们一些线索：关于如何在neural network learning上构建一个更统一的模型。必要的，它假设一个生物系统通过一个马尔可夫毯(Markov blanket：它会将internal state与外部环境相隔离）闭环，通信只通过sensory input和actions发生。生物系统的整体目标是：**不论内部和外部，维持一个稳态(homeostasis)，从而减小内部和外部的free energy(surprises)**。

然而，如果一个组织（organism），通过Markov blanket闭环，可以通过变更internal states来最小化free energy，并且/或者 与环境（environment）交互，如果两者都失败怎么办？例如，当一个人听到关于一个不幸新闻时，他不会有任何反映发生，变更internal state可能只会破坏身体的体内平衡（homeostasis）。**从物理角度，如果信息和energy是内部可变的，那么总的energy是守恒的，non-digestive energy也是维持稳态的一个必要方式**。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b993903e3ff46c10bbb158a49e983a3090be6fff42bcaffa0c6adcf88289ca38380d8c7257850c00792f77591c1b6418?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2 在DynamicCell模型中，我们通过引入reaction到free energy priciple中，构建了生命的基本单元（basic unit of life）(cell)。一个basic activity of life仍会维持它的稳态。另外，它通过变更internal states或actions，会将unexpected input “react”成一种排出过多不能被处理energy的方式。例如：笑与器都意味着分别排出good和bad surprises，它们不会对生存（survival）有贡献。换句话说：life reacts。


因此，我们可以将reaction包含到图2中，来简单改进free energy principle的思想，它会遵循物理中的能量转化定律。在我们的新模型中，每个cell或一个group（称为：organism）可以遵循相似原则：通过变更internal states和/或 actions，来最小化free energy（来自input $$\vec{c}$$的surprise），不能被最小化的过多non-digestive energy会通过reaction抛弃。这里的action signal $$\vec{w}$$被在位于相同Markov blanket中的其它upstream cells接收，只会影响upstream feedback $$\overleftarrow{w}$$。注意，action singal $$\vec{w}$$不同于一个organism采取的与环境交互的物理动作。在我们的模型下，物理动作可以通过upstream singal $$\vec{w}$$进行激活来做有用工作、或者通过downstream singal $$\ overleftarrow {c}$$来排出extra surprises（例如：通过笑或哭）。

### 2.3.2 Formulation

对了对上述思想进行数学上的公式化，我们将[47]重新resort来构建一个新的neuron model。总体上，一个neuron表示分布$$P(w \mid c)$$并且遵循[47]，它的input和output singals可以通过它们的embeddings近似表示，比如：$$P(w \mid c) = \frac{1}{Z(\vec{c})} exp(\langle\vec{w}, \vec{c}\rangle)$$，其中$$\vec{w}$$可能依赖于$$\vec{c}$$，并且$$Z(\vec{c})=\sum_{\vec{w}} exp(\langle\vec{w}, \vec{c}\rangle)$$。给定这样的假设，我们可以将free energy（或surprise）的最小化表示成两部分：internal和external。

**Internal state homeostasis稳态**

一个cell的internal state的稳定性在图2中反应在action state $$\vec{w}$$上。一个cell的长期行为（long-term behavior）可以与它的context c相互独立，因此可以表示成分布$$P_{\vec{w}} = P(w)$$。这里，free energy（或surprise），来自一个给定input c的一个cell的internal state可以被简单表示成：

$$
D_{KL}(P_{\vec{w}} \| P_c) = \sum\limits_x P_{\vec{w}}(x) log \frac{P_{\vec{x}}(x)}{P(x | c)}
$$

...(5)

并且，surprise最小化意味着调整$$P(w \mid c)$$的internal参数，以便$$P(w \mid c) \approx P(w)$$。为了观察surprise minimization是如何在embedding space中实现的，假设我们使用sufficient statistics representation $$P(w \mid c)$$，并将等式(5)重新改写：

$$
D_{KL}(P_{\vec{w}} \| P_c) = - \sum_{x} P_{\vec{w}}\langle\vec{w}, \vec{c}\rangle + log Z(\vec{c}) - H(P_{\vec{w}})
$$

...(6)

其中，$$H(\cdot)$$表示给定分布的entropy，它应该是相对稳定的。为了最小化等式(6)，一个cell需要达到一个这样的state：其中对应到input c的$$D_{KL} (P_{\vec{w}} \mid P_c)$$梯度是0:

$$
\frac{\partial D_{KL}(P_{\vec{w}} \| P_c)}{\partial \vec{c}} \Leftrightarrow - \sum_x P_{\vec{w}}(x) \frac{\partial \langle\vec{w}, \vec{c}\rangle}{\partial \vec{c}} + \frac{\partial log Z(\vec{c})}{\partial \vec{c}} \approx 0 \\
 \Leftrightarrow \langle\vec{w}\rangle P_c \approx \langle\vec{w}\rangle P_{\vec{w}}
$$

...(7)

其中，我们假设：$$\partial \langle\vec{w}, \vec{c}\rangle / \partial {\vec{c}} \approx \vec{w}$$。注意，这与contrastive divergence算法在形式上相似，尽管他们基于完全不同的假设。

**Upsteam state homeostasis稳态**

upstream和downstream的不同之处是，前者的state预期是隐定的。为了对upstream states的稳定性进行measure，你可以将在upstream中信息处理的整个复杂过程看成是一个黑盒，并简单地对来自usual distribution的偏差（deviation）进行measure：

$$
D_{KL} (P_{\vec{w}} \| P_{\vec{w}} = \sum\limits_x P_{\overleftarrow{w}}(x) log \frac{P_{\overleftarrow{w}(x)}(x)}{P(x | w)}
$$

...(8)

其中，$$P_{\overleftarrow{w}}$$表示upstream feedback singal $$\overleftarrow(w)$$的分布（如图2所示）。这与等式(7)相似，我们可以获得该稳定upstream state的condition：

$$
\frac{\partial D_{KL}(P_{\overleftarrow{w}} \| P_{\vec{w}}}{\partial \vec{w}} \Leftrightarrow \langle\vec{w}\rangle P_{\vec{w}} \approx \langle\overleftarrow{w}\rangle P_{\overleftarrow{w}}
$$

...(9)

通过变更$$P(w \mid c)$$的internal state，一个cell可以通过等式(6)和等式(8)进行optimize来最小化整体的surprise。均衡是在internal state和actions间的一个平衡。

**Reaction**

从上面的分析可知，free energy可以通过在满足等式(7)和等式(9)时达到最小化。然而，一个系统的overall state的entropy的天然趋势是增加的，因此，一个封闭的organic系统应期望来自input的一个常量的upcoming surprises。当这些surprises不能通过变更internal states（等式7）或taking actions（等式(9)）最小化时，他们必须抛弃到系统外，例如：通过reaction $$\overleftarrow{c}$$。例如，总和energy的一个选择可以表示成：

$$
\overleftarrow{c} \approx (| \langle \overleftarrow{w} \rangle_{ P_{\overleftarrow{w}}} - \langle\overleftarrow{w} \rlangle_{P_{\vec{w}}} - \langle\vec{w}\rangle_{P_c}|^2) / 2 \geq (\langle \overleftarrow{w} \rangle_{P_{\overleftarrow{w}}} - \langle \overleftarrow{w} \rangle_{P_{\vec{w}}}) \degree (\langle \vec{w} \rangle_{p_{\vec{w}}} - \langle \vec{w} \rangle_{P_c}
$$

...(10)

其中，$$\mid \cdot \mid^2 $$表示element-wise square，$$\degree$$也是一个element-wise product。以下，我们会观察到该形式的选择可以天然地与loss function的梯度下降更新相联系。在定义reaction时存在许多其它可能，本paper不考虑。

**与gradient descent update相联系**

最终，我们来看下，上述过程是如何将常规的使用gradient descent的loss minimization做为它的一个特例的。为了观察到它，我们可以简单将action singal $$\vec{w}$$与一个loss function $$L{\vec{w}}$$相联系，假设$$\vec{w}$$返回loss的评估（例如：$$\vec{w} = L(\vec{w})$$）。从上述关系，在梯度近似时可以将有限差 step设置为1，我们可以得到：

$$
\frac{D_{KL}(P_{\vec{w}} \| P_c)}{\partial \vec{c}} \approx \langle \vec{w} \rangle_{P_{\vec{w}}} - \langle \vec{w} \rangle_{P_c} \approx \frac{\partial{\vec{w}}}{\partial \vec{c}}
$$

...(11)

$$
\frac{D_{KL}} {(P_{\overleftarrow{w}} \| P_{\vec{w}}}){\partial \vec{w}} \approx \langle \ overleftarrow{w} \rangle_{P_{\overleftarrow{w}}}  - \langle \ overleftarrow{w} \rangle_{P_{\vec{w}}} \approx \langle L(\vec{w}) \rangle_{P_{\vec{w}}} - \langle L(\vec{w}) \rangle_{P_{\vec{w}}} \approx \frac{\partial L(\vec{w})}{\partial {\vec{w}}}
$$

...(12)

最终，从等式(10)，我们可以达到熟悉的梯度形式：

$$
\overleftarrow{c} \approx \frac{\partial L(\vec{w})}{\partial \vec{w}} \cdot \frac{\partial \vec{w}}{\partial \vec{c}} = \frac{L}{\vec{c}}
$$

...(13)

**这与认识场景的过程相一致，大脑实际上会做某些形式的back-propagations操作**。

# 3.系统设计

## 3.1 tensorflow API设计



回顾上面，在neural network中的每个layer/neuron被认为是在embedding space中的特定分布$$p(w\mid c)$$（c为input，w为output）。对于在input和output间的中间层（intermediate layers），c已经被表示成一个embedding $$\bar{c} \rightharpoonup$$，我们只需要定义一个函数来计算$$\bar{w}$$。在这样的情况下，我们可以只使用在tensorflow中相同的计算图来进行forward计算（图2中的input和action）和backward执行（在图2中的feedback和reaction），non-gradients baesd update可以通过对tf.gradients做很微小的变化来达到。例如，一个典型的DynamicCell node可以被定义成：

{% highlight python %}
def my_cell_forward(c):
    """returns action w"""

@ops.RegiestorGradient("MyCellForward")
def my_cell_backward(op, w_feedback):
    """returns reaction c_feecback"""

{% endhighlight %}

然而，需要特别注意：当w和c其中之一涉及到sparse features(比如：words)时，由于它可能发生在input或output layer（例如：一个softmax output layer来预测一个word）。已经存在的tensorflow实现总是需要一个字典和string-to-index转换（例如：通过tf.nn.embedding_lookup或tf.math.top_k），它们与我们的哲学（philosophy：用户只需要定义$$P(w \mid c)$$的形式，无需关注它的内容）不兼容。实际上，这些input/output操作是让tensorflow处理ever-growing的关键，它与input/output values相区别，通过将content processing的job转移给Dynamic Embedding service (DES)。另外，为了让tensorflow与DES无缝工作，我们使用单个protocol buffer来编码所有的配置，它在我们的tensorflow APIs中可以表示成input参数de_config。

### 3.1.1 Sparse Input

如上所述，允许tensorflow直接采用任意string作为input，这非常有用。这里我们调用该process来任意string input转换成它的embedding dynamic embedding，使用tensorflow API定义成：

{% highlight python %}

def dynamic_embedding_lookup(keys, de_config, name):
    """returns the embeddings of given keys.""""

{% endhighlight %}

其中，key是任意shape的string tensor，de_config包含了关于embedding的必要信息，包括希望的embedding维度，初始化方法（当key是首次见到时），embedding的存储等。name和config也可以唯一的区分embedding来进行数据共享（data sharing）。

### 3.1.2 Sparse Output

当一个neural network的输出为sparse features时，它通常被用在inference问题上：$$argmax_w P(w \mid c)$$，其中c是来自之前layer的input，表示在neural network中的$$\bar{c}$$。根据第2.1节，如果我们假设$$P(W \MID C) \propto exp(<\bar{w}, \bar{c}>)$$，其中，$$\bar{w}$$是w的embedding，接着$$argmax_w P(w \mid c) = argmax_w <\bar{w}, \bar{c}>$$，它可以简化为：在w的所有值中，离input query $$\vec{c}$$的最近点。实际上，softmax function通常被用在neural network中，它与我们的formulation最相关。为了观察到这一点，假设w的所有可能值集合为W，$$\forall a \in W$$，softmax概率可以被表示为：

$$
P(w=a \mid c) = \frac{exp()}{}
$$

...(14)

如果$$dim(\vec{w})=dim(\vec{c}) +1$$，其中$$dim(\cdot)$$表示一个vector的维度，即落到我们的特例上。

然而，需要计算等式(14)，对于softmax output来说，当在W中的elements的数目非常大时，对于w的所有值计算cross entropy loss非常低效。幸运的是，efficient negative sampling方法已经被很好地研究[21]。在DES中必须支持它。

**Candidate negatie sampling**

为了允许output values具有无限数目，我们根据tf.nn.sampled_softmax_loss来定义一个内部函数实现，它需要返回logit values（）。

{% highlight python %}
_compute_sampled_logits(pos_keys, c, num_samled, de_config, name):
    """returns sampled logits and keys from given positive labels."""
{% endhighlight %}

这里，num_sampled是一个正数，sampling strategy在de_config中定义。

**TopK retrieval**

这里，在训练期间需要candidate negative sampling，在inference期间，我们希望如上计算$$argmax_w P(w \mid c) = argmax_w <\vec{w},\vec{c}>$$，在实际上，它通常来检索到给定input的top-k最近点（例如：在语言inference中beam search）。topK retrieval的interface定义如下：

{% highlight python %}

def top_k(c, k, de_config, name):
    """returns top k closet labels to given activation c."""

{% endhighlight %}

在该场景背后，该函数会调用DynamicEmbedding server来来寻找那些接近$$[\vec{c}, 1]$$的keys。

### 3.1.3 Saving/restoring模型

最终，在model training期间，一个模型需要被周期性保存。由于我们会将大多数数据移出tensorflow的graph外，对于维持在tensorflow与DynamicEmbedding两者保存的checkpoints间的一致性很重要。在API这一侧，每次调用DynamicEmbedding相关API时，相应的embedding data信息，会在一个global variable中保存唯一的(name, de_config)。寻于DynamicEmbedding的checkpoint saving/loading会与tensorflow非常相似：

{% highlight python %}

save_path = save(path, ckpt)
restore(save_path)

{% endhighlight %}

如果用户使用在tensorflow中的automatic training framework，比如：tf.estimator，通过我们的high-level APIs自动处理saving/load模型。但如果他们希望在一个low level的方式，他们需要调用以上的函数和tensorflow相应的I/O函数。

### 3.1.4 使用DynamicEmbedding的Word2Vec模型

总之，对tensorflow API变更以支持DynamicEmbedding非常简单，对于模型构建的job也简单。做为示例，word2vec模型可以使用以下代码行来进行定义：

{% highlight python %}
tokens = tf.placeholder(tf.string, [None, 1])
labels = tf.placeholder(tf.string, [None, 1])
emb_in = dynamic_embedding_lookup(tokens, de_config, 'emb_in')
logits, labels = _compute_sampled_logits(labels, emb_in, 10000, de_config, 'emb_out')
cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits)
loss = tf.reduce_sum(cross_ent)

{% endhighlight %}

注意，一个字典的需求被完全移除。

## 3.2 DynamicEmbedding serving设计

如图1所示，我们的DynamicEmbedding Service(DES)涉及到两部分：DynamicEmbedding Master(DEM)和DynamicEmbedding Workers(DEWs)。前面定义的tensorflow API只会与DEM通信，它涉及到将real work分布到不同的DEWs上。为了同时达到效率和ever-growing模型，在DEWs中的每个worker会对local caching和remote storage进行balance。在该部分，我们会讨论在当前形式下DES的不同方面。

### 3.2.1 Embedding存储

在第2节中讨论的，neurons间的通信会被表示成firing patterns(embedding)的充分统计，它们是floating values的vectors。这些firing patterns本质上是离散的（discrete），可以被表示成string ids。这里，这些embedding data的存储只涉及到(key, value) pairs，并且不吃惊的是，我们会使用protocol buffer来处理data transfer以及为每个embedding like string id, frequency等保存额外信息。

当特定数据被传递到由tensorflow API定义的node中时，它会与DES通信来处理实际job。例如，在运行dynamic_embedding_look op的forward pass期间，一个batch的strings会被传递给tensorflow computation graph的一个node，它接着会询问DEM来处理实际的lookup job。在backward pass期间，feedback信号（例如：对应output的gradients）会被传递给注册的backward node中，它也需要与DEM通信来进行数据更新。

为了允许可扩展的embedding lookup/update，我们设计了一个称为EmbeddingStore的组件，它会专门与google内部的多个storage systems进行通信。每个支持的storage system实现了与基础操作（比如：Lookup(), Update(), Sample(), Import(), Export()）相似的接口，例如，一个InProtoEmbedding实现了EmbeddingStore接口，它通过将整个数据保存到一个protocol buffer格式中，它可以被用来进行local test以及训练小的data set。一个SSTableEmbedding会在training期间将数据加载进DEWs的内存中，并在GFS中将它们保存成immutable且非常大的文件。一个BigtableEmbedding允许数据同时存成local cache和remote、mutable及高度可扩展的Bigtables。因此，从worker failure中快速恢复，使得不必等待，直到所有之前的数据在接受到新请求前被加载。

### 3.2.2 embedding update

在我们的框架中，embedding updates会在forward和backward passes期间同时发生。对于backpropagation算法，updates只发生在$$\partial{L}/\partial{w}$$到达时的backward feddback。为了职证我们的系统与已经存在的gradient descent算法（例如：tf.train.GradientDescentOptimizer或tf.train.AdagradOptimizer）完全兼容，我们需要在DEWs中实现每个算法。幸运的是，我们可以复用tensorflow相似的代码来保证一致性。注意，许多gradient descent算法，比如：Adagrad，会保存关于每个值的全局信息，它们应在gradient updates间一致。在我们的情况中，这意味着我们需要额外信息来存储到embedding中。

**long-short term memory**

当一个学习系统可以处理一段长期的数据时（比如：数月和数年），解决long-short term memory的主题很重要，因为如果特定features只是随时出现，或者在一段较长时间内没有被更新，它对inference accuracy不会有帮助。在另一方面，一些短期输入（momentary input）可能包含需要特殊处理的信息，一个无偏的学习系统（unbiased learning system）yicce处理这些低频数据。接着，我们提出了两个基本技术来管理embedding data的生命周期。

- frequency cutoff:
- Bloom filter:

### 3.2.3 top-k sampling

在模型inference期间，对于高效检索离给定input activation最近的topk个embeddings很重要，其中距离通过点乘（dot product）来测量，如第3.1.2节所示。对于非常大数目的input（例如：[45]），可以很高效和近似地处理。我们会采用在google内部的实现来让在DEWs中的每个worker返回它自己的top k个embeddings给DEM。假设它们有n个DEWs，那么DEM会在$$n \times k$$个candidate vectors间选择top-k个最近点。这样，当$$k << m$$时（其中，m是keys的总数）， accuracy和efficiency会被保障。

### 3.2.4 Candidate sampling

当它们被储存到像Bigtable的远程存储中时，Sampling可以是tricky的。这也是为什么需要metadata，它可以存储必要信息来进行高效采样候选。在很早时，我们支持由两个已存在的tensorflow ops：$$tf.nn.sampled_softmax_loss$$和$$tf.contrib.text.skip_gram_sample$$（基于frequency）所使用的sampling strategies。如果我们希望达到更好的word embedding，则相应地需要计算更高阶的信息比如PMI（互信息概率）或共现次数。因此，这些bookkeeping信息需要在高效采样进行embedding lookup期间被处理。

这里，我们决定重新设计在DES中的candidate sampling，因上以下原因：

- i) 复用tensorflow code很简单，因为每个embedding在一个interger array中都具有一个唯一的索引
- ii) 原始实现不会考虑多个label output，因为实际上它会区别true labels和sampled labels（为了满足限制：所有variables必须在training前被定义，它需要从input中的true labels数目，比如：每个input必须具有明确相同的true labels）。。。

在我们的新设计中，为了满足graph的需求：即graph是固定的，每个input中的true_labels数目会不同，我们会简单地将positive 和negative examples进行合并，并由用户来决定num_samples的值。我们的接着变为：

{% highlight cplusplus %}

class CandidateSampler {
  public:
    struct SampledResult {
      string id;
      bool is_positive;
      float prob;
    }
    
  std::vector<SampledResult> Sample (
      const std::vector<string>& positive_ids, const int num_sampled, const int range) const;
  )
}

{% endhighlight %}

因此，我们的新candidate sampling会解决在tensorflow实现中的限制，从而能更好地处理multi-label learning。

### 3.2.5 分布式计算

分布式很简单，给定每个embedding data，需要一个唯一的string id作为它的lookup key。每当DEM接收到来自tensorflow API的请求时，它会基于它们的ids将数据划分，并将work分布到不同的DEWs（lookup, update, sample，etc）中。这里，每个worker会负责处理总的keys中的一个唯一子集，并进行失败恢复，它还可以标识它负责的keys子集。有了Google's Borg system，在server中的每个worker可以唯一标识它自己的shard number。例如，当存在n个workers时，第i个worker会负责这些embeddings，满足：$$mod(hash(key),n) = i$$。对于高效dandidate sampling来说，DEM应记帐关于每个worker的metadata，并决定每个worker的samples数目。

### 3.2.6 扩展Serving

使用DynamicEmbedding的tensorflow模型，需要一些特例，因为embedding数据需要对大size（>100G）进行高效检索。在本机(local)中很难满足。因此，除了由DynamicEmbedding service提出的最简单serving，我们需要支持更多健壮的设计来处理大的embedding data。为了更健壮的model serving，以下两个optimization需要考虑下。

**Sandbox mode**

**Remote storage lookup with local cache** 

# 4.实验

略


# 参考


- 1.[https://arxiv.org/pdf/2004.08366.pdf](https://arxiv.org/pdf/2004.08366.pdf)