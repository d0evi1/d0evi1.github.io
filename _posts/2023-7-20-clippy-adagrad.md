---
layout: post
title: Clippy Adagrad介绍
description: 
modified: 2023-7-20
tags: 
---

google在《Improving Training Stability for Multitask Ranking Models in Recommender Systems》提出了Clippy Adagrad方法。

# 摘要

推荐系统在许多内容平台上扮演着重要角色。虽然大多数推荐研究都致力于设计更好的模型以提升用户体验，我们发现对于这些模型训练稳定性方面的研究严重不足。随着推荐模型变得更加庞大和复杂，它们更容易遇到**训练不稳定问题，即loss发散，这可能使模型无法使用，浪费大量资源并阻碍模型发展**。在本文中，我们分享了我们在提高YouTube推荐多任务排序模型训练稳定性方面的发现和我们学到的最佳实践。我们展示了导致训练不稳定的模型的一些属性，并对其原因进行了推测。此外，基于**我们在训练不稳定点附近的训练动态观察**，我们假设现有解决方案为何会失败，并提出了一种新算法来减轻现有解决方案的局限性。我们在YouTube生产数据集上的实验表明，与几种常用的基线方法相比，所提出的算法可以显著提高训练稳定性，同时不损害收敛性。我们在 https://github.com/tensorflow/recommenders/ tree/main/tensorflow_recommenders/experimental/optimizers/clippy_adagrad.py 上开源了我们的实现。

# 1 引言

一个优秀的推荐系统对用户体验起着关键作用。它已成为许多网络应用的核心技术，甚至主要用户界面，包括YouTube，世界上最大的在线视频平台之一。因此，可以将许多组件整合到推荐模型中，以捕捉不同模态的上下文并提高推荐质量，包括**音频信号[30]、视频信号[21]、用户历史序列[5, 28]**等。此外，推荐模型的**规模定律（scaling law）[3]**表明，通过在数据丰富的应用中增加模型容量，可以显著提高质量。

随着推荐模型变得更大更复杂，它们更容易遇到**训练不稳定问题[14]，即loss发散（而不是收敛），导致模型“损坏(broken)”并完全无用**。在工业界，提供这样一个“损坏”的模型会导致灾难性的用户体验（见第2.2节）。此外，如果我们不能确保推荐模型的可靠训练，就可能浪费大量资源并阻碍模型发展。因此，我们再怎么强调训练稳定性的重要性也不为过。然而，关于推荐模型的训练稳定性的研究非常少。

一方面，**对于推荐模型为何容易出现训练不稳定问题缺乏基本理解**。特别是，我们观察到，与具有单一目标的检索模型（例如，在大输出空间上的Softmax交叉熵）相比，具有多个目标的排序模型更有可能遇到问题。除了增加模型复杂性，我们发现简单地添加新的输入特征或输出任务也可能导致训练不稳定。为了解决这个问题，**人们大多依赖经验解决方案，有时也依赖运气（当问题随机发生时）**。发展对问题原因的基本理解将使人们能够更自信地导航这个过程。

另一方面，**我们发现缺乏有效的方法来大幅减轻训练不稳定问题**。有一些广泛使用的方法，如激活裁剪[20]、梯度裁剪[7, 24]、学习率预热[12, 14]和层归一化[4]。但在实践中，我们发现这些方法都是权宜之计，不能完全防止我们模型中的训练不稳定。开发一种能有效提高模型训练稳定性的有效方法，通过解决训练问题的担忧，加速模型改进。

本文的重点是要分享从解决YouTube推荐使用的多任务排序模型所经历的训练不稳定问题中学到的经验。

- 在第2节中，我们将展示现实世界推荐系统中不稳定模型训练的影响和后果，强调显著提高模型训练稳定性的重要性和困难。
- 在第3节中介绍了我们模型的一些初步基础知识后，我们将介绍**一些导致更多训练不稳定问题的案例研究**，并提供我们对问题根源的理解。然而，在实践中，我们发现知道根源和拥有有效解决方案之间存在很大差距。一些理应有效的方法在实证上并不奏效。
- 接下来，在第4节中，我们将仔细**检查我们模型的训练动态**，这启发我们提出一种更有效的方法来克服现有方法的局限性。第5节中的YouTube数据集的实证证据揭示了所提出方法在提高模型训练稳定性方面的有效性，特别是当增加模型容量和使用大的学习率以更快地收敛时。我们希望这些发现可以帮助社区更好地理解训练不稳定问题并有效解决它。

# 2 背景和相关工作

## 2.1 症状

训练不稳定性是衡量模型训练不稳定性的一个属性。它有一个常见的症状，即**loss发散（也称为loss爆炸）**。根据我们的观察，我们进一步将loss发散分为两种类型：**微发散(micro-divergence)和完全发散（full divergence）**。当一个模型的loss微发散（见图1中的模型a作为例子），我们可以观察到训练loss突然跳跃和训练指标突然下降，尽管loss可能在训练继续时恢复正常（如例子所示）。**通常，我们不需要太担心这种情况，因为恢复的模型可以与没有遭受loss发散的模型具有同等的质量**。然而，如果一个模型的loss完全发散（见图1中的模型b作为例子），我们可以看到训练loss在几次训练步骤后变得非常高，所有训练指标变得极其糟糕。例如，二元分类AUC（我们整篇论文中主要关注的指标）在图1中下降到0.5，表明模型变得完全无用，实际上给出了随机结果。更糟糕的是，**完全发散的loss无法在训练继续时恢复到发散前的值**。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/0c41d2c32ef0bdc09f5346b13d4c957ce251bd5451b7d59cb05e8d14dfd6182704e46f9e7d5591837c56ba4645776ae7?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 在我们模型中的loss发散示例及其对训练loss（顶部）和AUC（底部）的影响。在这个例子中，模型a的loss发生了微发散然后恢复了，而模型b的loss完全发散了。

## 2.2 动机和挑战

我们从几个方面强调了训练稳定性研究的重要性，特别是工业界推荐系统的研究。首先，一旦发生loss发散问题，它可能影响几乎所有类型的模型开发。这包括但不限于：

- (1) 增加**模型复杂性**：随着更多的建模技术被应用和更多的组件被添加到推荐模型中（以提高其质量），模型遭受loss发散问题的可能性就更大。即使仅仅增加模型容量也可能使模型处于危险状态，尽管当前的规模定律[3]建议在数据丰富的环境下有巨大的好处。
- (2) 添加**更多输入特征或任务**：通常，推荐系统中的排序模型使用许多输入特征进行多项任务[36]。对所有任务的预测组合用来决定候选item的排序。我们发现，添加新的输入特征和添加新任务都可能导致训练不稳定，尽管它们是提高模型质量的常用方法。
- (3) **提高收敛速度**：我们发现，有助于模型收敛的超参数调整（如增加学习率）可以显著增加loss发散的可能性。这迫使模型设计者使用较小的学习率，这导致收敛速度变慢。

第二，由于训练复杂模型需要大量的资源，**loss发散问题阻碍了模型完成训练，浪费了训练资源**。此外，无意中部署一个“损坏（broken）”的模型提供服务也会导致灾难性的用户体验。

因此，我们看到了许多从工程角度缓解这个问题的努力，例如**在提供服务之前确保模型质量**。然而，鉴于工程努力无法防止训练不稳定性的发生，很明显，**从长远来看，大幅提高模型训练稳定性是正确的追求路径**。

在处理模型不稳定性问题时，我们遇到了以下挑战。

- **可复制性（Reproducibility）**：模型在训练期间的任何时候都可能遭受loss发散，然而，只有其中一些可以容易地复现（第3节中有更多讨论）。无法复现坏的情况使得理解模型在loss发散之前发生了什么变得困难。
- **检测（Detection）**：在实践中，频繁地在训练期间评估模型并报告结果成本很高，否则训练可能会显著放缓。由于有时微发散可能发生，**然后非常快地恢复，即使没有牺牲训练速度，也很难检测模型训练期间是否发生了任何微发散**。
- **测量（Measurement）**：很少有研究对模型训练稳定性的定量测量先于训练。要知道（1）建模变更是否会增加loss发散的风险，或者（2）缓解措施是否有助于减少loss发散的风险，人们必须**依赖经验评估**（即，训练模型的多个副本，并检查其中有多少有问​​题），这是耗时且资源密集的。


## 2.3 相关工作

模型训练稳定性一直是一个研究不足的领域，不仅在推荐模型中如此，在通用机器学习中也是如此。幸运的是，随着大型模型的日益增加趋势[8, 11, 29]，稳定模型训练已成为一个新兴的研究领域，并在近年来吸引了更多的关注。

从优化理论的角度来看，Wu等人[32]首次从**学习率和loss曲率的“锐度”（通过lossHessian的最大特征值来衡量）理论**上预测了二次模型的训练不稳定性。对于深度神经网络，Cohen等人[12]，Gilmer等人[14]证实了这一预测仍然足够准确。

在技术方面，有一些方法在语言和视觉模型中得到了广泛应用，如**激活裁剪[20]、梯度裁剪[24]、学习率预热[16]以及各种归一化技术[4, 18]**。此外，You等人[34]提出了一种**新的优化器**，为大批量训练在收敛性和稳定性之间实现了更好的权衡。Brock等人[7]开发了**自适应梯度裁剪**，以提高没有批量归一化[18]的ResNet模型[17]的稳定性。

然而，从经验上，**我们发现这些方法对于完全防止我们的模型训练不稳定性还不够有效（见第5节）。这可能是由于推荐模型的一些独特属性**。正如下一节将讨论的，这些属性可能使多任务排序模型更容易出现训练不稳定性问题。

# 3 理解问题的原因

在本节中，我们首先描述本文要研究的模型及其特性。然后，我们分享了我们对发生在我们模型中的训练不稳定问题根本原因的理解。

## 3.1 模型定义

YouTube的视频推荐系统，首先使用多个候选生成算法检索几百个候选项。接着是一个排序系统，它从这些候选项中生成一个排序列表。本文主要关注YouTube推荐系统中的排序模型。与候选生成模型（即检索模型）不同，后者负责过滤掉大部分不相关item，排序模型旨在提供排序列表，以便对用户最有用的物品显示在顶部。因此，排序模型使用更先进的机器学习技术，使用更昂贵的特征，以获得足够的模型表达能力，学习特征及其与效用的关联。

图2展示了我们希望在整篇论文中研究的排序模型的一般架构。以下是我们排序模型的一些重要特性以及它的训练方式；详情可参考[36]。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/23aa1f149ce7986ef828e8fa1f97f80125b7f12a7bd5a0177b6db99a6af5a779bbd4cf8d28d29b18a31937d303aaa17b?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2 推荐系统中使用的排序模型的一般示意图。该模型具有一个或多个层，这些层由多个任务（软共享或硬共享）共同使用。

- 多任务（multitask）：如图2所示，排序模型有多个任务，预测多个标签。这些预测被组合以形成最终的物品排序列表。不管不同的建模选择[9, 22]，模型中间有一些隐藏层由这些任务共享（要么完全共享，要么软共享）。
- 顺序训练（Sequential training）：模型是顺序训练的，即，**训练是按数据集合的顺序从旧到新完成的**。不同于纯在线学习[6, 25]（它会严格按顺序访问训练数据），**我们定义了一个基于时间的移动窗口，并从这个窗口中的训练数据中随机采样数据batch进行并行训练**。这种训练方案已被广泛使用，并且已知对许多推荐质量方面都有益[2, 23, 36]。
- 优化（Optimization）：众所周知，**大批量训练（Large batch-size training）在梯度上噪声较小，因此optimization更多地受曲率驱动[2, 34]**。我们采用大batch-size和高学习率以实现更快的收敛。我们发现Adagrad[13]在我们的情况下非常有效，尽管优化器有许多进步（例如，Adam[19]，Adafactor[26]）。

## 3.2 根本原因和案例研究

不管loss发散的类型如何，我们认为内在原因可以总结为“**当loss曲率陡峭时，step-size太大**”。一旦**模型在给定状态下满足这两个条件，就容易发生发散**。直观地说，步长应该在陡峭的loss表面上保守（通过lossHessian的最大特征值来测量），以确保loss减少而不是增加。

对于二次模型(quadratic models)，Wu等人[32]从理论上证明了上述论点，并建议
$$
\frac{2}{\eta} > \alpha_*
$$

以使训练稳定，其中：

- $\eta$：是**学习率**
- $𝛼_*$：是**lossHessian的最大特征值**

Cohen等人[12]为证明提供了一个很好的直接例子（见图3）。对于神经网络，这一论点仍然大体成立[12, 14]。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/80dc697a1051cfce682feec0756b19a2e66df5810dbea2b7ffb641ef22e0dc94f822363ac987a0eb7fbcd95e81e2d9ae?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3 来自[12, 图2]。在具有特征值$𝛼_1 = 20$和$𝛼_2 = 1$的二次模型上进行梯度下降。我们可以清楚地观察到，当学习率 $\eta > 2/𝛼_* = 2/𝛼_1 = 0.1$时，开始出现训练不稳定问题。

了解训练不稳定问题的根本原因，使我们能够回答以下研究问题：

- RQ1：**为什么一般的推荐模型比其他领域的模型有更差的训练稳定性？**
- RQ2：在推荐模型中，**为什么排序模型通常比检索模型有更差的训练稳定性？**

我们将这些问题的答案与我们模型的独特属性联系起来。请参阅补充材料第A.1节中的一些实证证据。

- **数据分布变化（RQ1）**：与其他领域的模型相比，推荐模型使用的**数量级更多的输入特征（数百到数千个）**。更糟糕的是，随着顺序训练的进行，这些输入特征（和标签）的分布不断变化。我们认为，**当数据分布发生突变时，可能会产生更陡峭的loss曲率**，这在推荐系统中是常见的。此外，顺序训练的模型永远无法收敛，因为它必须适应新到来的数据点，这些数据点的分布已经改变。因此，需要较大的学习率来使适应足够高效。总结来说，与在固定数据集上训练的其他领域模型相比，训练数据分布的变化对稳定推荐模型的训练提出了更大的挑战。
- **更大的模型大小和复杂性（RQ2）**：与用于候选生成的检索模型相比，排序模型通常具有更大的容量，以准确衡量候选item的作用。随着ML硬件（例如，TPU）的最近发展，**我们能够显著增加模型大小以提高质量**[3]。Gilmer等人[14]的实证研究表明，**增加的模型容量和复杂性是导致loss曲率更陡峭的一个因素**。
- **多目标与单目标（RQ2）**：与通常只有一个目标的检索模型（例如Softmax交叉熵）[33]相比，排序模型通常需要同时优化多个目标[36]。这导致排序模型更容易遭受loss发散。因为如果由于特定任务的不良预测导致出现虚假梯度，这些梯度可以在整个模型中反向传播，导致被多个任务共享的层表现得（略微）异常。但是，由于这些层被不同任务共享，其他任务往往会在之后预测出不规则的值，加强不稳定状态至不可恢复。换句话说，**共享层（以及嵌入）可能是一把双刃剑——它们允许从不同任务中转移学习，但也可能加剧训练不稳定问题，使排序模型比检索模型更脆弱**。

尽管在理解发散问题的根本原因方面最近取得了进展，我们发现我们对问题原因的当前理解与拥有有效解决方案之间仍存在很大差距。我们尝试了许多临时解决方案。一些例子包括：

- (1) 使用**更慢的学习率的热启计划（warmup schedule）**来通过初始模型状态（ initial model state），此时loss曲率陡峭[14]。
- (2) **扩大顺序训练移动窗口**，使训练数据分布变化更平滑。

这些解决方案确实在一段时间内缓解了训练不稳定问题，**但当我们的模型变得更复杂时，loss发散又发生了**。在尝试了多种权宜之计后，我们相信开发一种可以显著提高模型稳定性的更有原则的方法是长期解决方案。

# 4 提高训练稳定性的有效方法

在本节中，我们首先介绍**控制有效步长（当loss曲率陡峭时）的一般方向（梯度裁剪）**，通过介绍这个方向上的一些经典方法，及其符号和表示。尽管这些经典方法在其他领域应用时取得了成功，但我们发现**这些经典方法在我们的模型中应用时效果不够好**。基于对我们模型训练动态的一些观察，我们提出了一种新方法，并解释了为什么它在提高训练稳定性方面可能更有效。

**Adagrad**

我们首先描述Adagrad[13]，这是我们模型中使用的优化器。在Adagrad中，模型参数$𝒘_𝑡$通过以下规则更新：

$$
G_{t} = G_{t-1} + g_{t}^{2} \\
r_{t} = g_{t}{G_{t}^{-1/2}} \\
w_{t+1} = w_{t} - \eta_{t} \cdot r_{t}
$$

...（1）

其中：

- $\eta_𝑡$：表示第𝑡步的学习率
- $𝒈_𝑡$：是模型参数的经验loss的标准化随机梯度，
- $𝑮_𝑡$：被称为“累加器(accumulator)”，**是一个向量，初始化为一个小常数值，通常为0.1**。此外，所有幂运算都是逐元素计算的。

如第3节所述，我们希望采用更有原则的方法来控制loss曲率陡峭时的步长。然而，通过lossHessian的特征值测量的loss曲率在训练期间计算成本非常高。幸运的是，一阶梯度$𝒈_𝑡$可以用作Hessian的替代品（参见[35]）。因此，基于梯度裁剪的算法变得非常流行，用于提高训练稳定性，并在许多大型模型中使用[8, 11, 29]。

**梯度裁剪（Gradient Clipping）**

由Pascanu等人[24]提出，梯度裁剪（GC）在将其应用于模型之前**限制梯度的大小（通过其范数测量）**。换句话说，**当梯度大小变化（loss曲率变得更陡）时，梯度裁剪通过控制“有效步长”来稳定模型训练**。

正式地说，梯度裁剪算法在应用模型更新之前将梯度$𝒈_𝑡$裁剪为：

$$ g \rightarrow
  \begin{cases} 
   \lambda \frac{g}{\|g\|} & \text{if } \|g\| \geq \lambda, \\
   g & \text{else.}
  \end{cases}
$$

或者

$$
 g \rightarrow \sigma \cdot g
$$ 

其中：

$$
\sigma = \min\{\frac{\lambda }{\|g\|}, 1.0\} 
$$

...（2）

裁剪阈值𝜆是一个超参数，控制最大允许的梯度范数$∥𝒈∥$。换句话说，如果在第𝑡步模型梯度$𝒈_𝑡$的大小很大，GC将通过标量裁剪因子$𝜎 \in R^+$重新调整梯度，将其范数限制为𝜆。在实践中，Frobenius范数（或𝐿2范数）$∥.∥_2$是向量范数的常见选择，裁剪通常独立应用于每一层。

**自适应梯度裁剪（Adaptive Gradient Clipping）**

从经验上，尽管GC可以提高模型的训练稳定性，但训练稳定性对裁剪阈值𝜆的选择极其敏感，需要为不同层进行细粒度调整。更糟糕的是，当模型结构、批量大小或学习率发生变化时，阈值𝜆需要重新调整。

为了克服这个负担，Brock等人[7]提出了自适应梯度裁剪（AGC）。**AGC的动机是观察到梯度的范数$∥𝒈_𝑡∥$与模型参数的范数$∥𝒘_𝑡∥$的比率不应该很大，否则训练预计会不稳定**。

具体来说，梯度𝒈通过以下方式裁剪：

$$ 
g \rightarrow
  \begin{cases} 
   \lambda \frac{\|w\|}{\|g\|} g & \text{if } \frac{\|g\|}{\|w\|} \geq \lambda, \\
   g & \text{else.}
  \end{cases}
$$

或者

$$ 
g \rightarrow \sigma \cdot g, 
$$ 

其中：

$$
\sigma = \min\{\lambda \frac{\|w\|}{\|g\|}, 1.0\}  \quad（3）
$$

直观地说，如果在第𝑡步梯度范数 $\| 𝒈_𝑡 \|$大于参数范数 $𝜆·\| 𝒘_𝑡 \|$的一个分数，AGC将通过标量裁剪因子$\sigma  \in R^+$重新调整梯度，将其范数限制为$𝜆 \| 𝒘_t \|$。**AGC可以被视为GC的一个特例，其中裁剪阈值𝜆GC是模型参数的函数$𝜆GC=𝜆 \cdot AGC \| 𝒘 \|$**。所以当使用AGC时，我们不需要为不同的层微调𝜆，这就是“自适应性”的来源。

**4.1 训练动态的观察**

**尽管GC和AGC在各个领域都取得了成功，我们发现当它们应用在我们的模型中时，并不足以防止loss发散**。为了更好地理解GC/AGC的局限性并提出更好的解决方案，我们检查了不使用任何基于梯度裁剪技术的模型训练。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/eeaeffc481cd01f20250dd00e8ff93b843d97739c859fec3fc0572d7a9377b956d1f83e7d22f5a79fe4894f56780447c?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=4.jpg&amp;size=750">

图4 (a) 我们深入研究了模型训练中的三个典型时刻：在步骤a之前，模型健康地训练着。然后在步骤b，模型的损失引起波动，AUC下降。最后在步骤c，损失完全发散，AUC下降到了0.5。 (b) 当检查模型顶层隐藏层的一些统计数据时，我们发现GC和AGC未能提供足够小的裁剪因子。而Clippy的裁剪因子可以比GC和AGC小两个数量级。补充材料的B节有其他层的统计数据。

- 图4a显示了一个特定二元分类任务的训练loss和AUC。为了简化说明，让我们主要看3个最重要的训练步骤：步骤a、步骤b和步骤c。如我们所见，此模型在步骤a之前健康地训练：loss被最小化，AUC迅速提高。然而，在步骤b，模型的训练loss开始发散，AUC开始下降，尽管相对不明显。最后，在步骤c，这个模型完全发散，loss变大，AUC下降到0.5。
- 图4b（左），我们仔细查看了一些顶级共享层的统计数据，以了解loss发散时发生了什么。在模型健康的步骤a之前，梯度范数$∥𝒈∥_2$相当一致。然后在步骤b增长到一个很大的值，表明那一刻loss曲率非常陡。由于我们没有应用任何模型稳定性处理，模型在步骤c完全发散，梯度范数$∥𝒈∥_2$变成了一个小值。**这意味着这一层的所有预激活（应用非线性激活之前的值）已经达到了梯度非常小的状态3，导致loss发散变得不可恢复**。

知道发生了什么，我们构想了GC/AGC在这种情况下将如何反应。

- 图4b（左）绘制了用于确定GC和AGC中裁剪因子的$∥𝒈∥_2$（蓝色）和$\frac{∥𝒈∥_2}{∥𝒘∥_2}$（橙色）的测量值。不出所料，这两种测量在步骤b变得更大。然而，这些测量的变化相对规模是不同的。$\frac{∥𝒈∥_2}{∥𝒘∥_2}$（橙色）比$∥𝒈∥_2$（蓝色）对loss曲率变化更敏感。**这些测量的敏感度差异可能导致不同的裁剪因子𝜎，这是不同方法中梯度的重新调整乘数**。
- 图4b（右）给出了使用$\lambda^{GC}=10^{−1}$和$\lambda^{AGC}=10^{−3}$作为裁剪阈值时GC和AGC的裁剪因子𝜎。通过检查裁剪因子，我们假设：**GC/AGC无效的原因是它们未能在梯度范数突然增加时（即loss曲率变陡），提供足够的梯度约束（即未能提供足够的“有效步长”控制），因为缺乏敏感性**。更具体地说，**这两种方法都依赖于𝐿2范数，这对于只有少数几个坐标中剧烈的梯度变化不敏感，特别是当层宽度很大时**。

## 4.2 提出解决方案：Clippy

为了缓解这一局限性，我们提出了一种名为Clippy的新算法。Clippy对GC/AGC有两个主要变化：首先，它使用$𝐿_∞$范数而不是$𝐿_2$范数，以增加对各个坐标变化的敏感性。其次，它基于更新$𝒓_𝑡=𝒈_𝑡·𝑮_t^{−1/2}$而不是梯度$𝒈_𝑡$进行裁剪，因为更新是模型参数的实际变化，在使用Adagrad优化器时，更新可能与梯度大不相同。

具体来说，Clippy控制

$$
\|r_{t}\|_{\infty} = \|g_{t} \odot G_{t}^{-1/2}\|_{\infty} < \lambda, \quad（4）
$$

当不等式被违反时重新调整更新。从图4b中，我们可以看到当loss发散时，这个测量在步骤b有更剧烈的变化。假设我们使用$𝜆^{Clippy}=10^−1$作为裁剪阈值，**由于测量的更好敏感性，Clippy产生的裁剪因子𝜎比GC/AGC小两个数量级**。换句话说，我们希望Clippy即使在少数几个坐标陡峭的loss曲率时，也能对实际更新施加更大的约束。

正式地，我们在算法1中展示了Clippy。如所见，与我们在等式4中描述的相比，算法的第8行有一些微小但重要的变化。

- **(1) 引入绝对阈值。**在Clippy中，我们使用两个超参数：与GC/AGC相似的相对阈值$𝜆_{rel}$，以及另一个绝对阈值$𝜆_{abs}$。引入绝对阈值$𝜆_{abs}$后，我们可以避免在模型参数为零（例如，初始化为零的偏差）或具有非常小的值时进行激进的裁剪。如4.3.1节所讨论的，这允许Clippy在训练过程中从GC风格切换到AGC风格。
- **(2) 考虑学习率。**在计算裁剪因子时，我们在分母上有学习率$𝜂_𝑡$，以适应不同的学习率计划。如果学习率缓慢增加，这将在初始训练时放宽裁剪阈值，避免在训练的初始阶段收敛速度过慢。

## 4.3 额外讨论

### 4.3.1 与其他方法的关系

Clippy与其他方法有有趣的联系。在基于梯度裁剪的算法中，如果我们用原始梯度（而不是裁剪后的梯度）累积累加器。然后，我们可以有一个通用的Adagrad更新形式，包含上述所有算法

$$
r_{t} = g_{t} \odot G_{t}^{-1/2}, \\
w_{t+1} = w_{t} - (\mu_{t} \sigma_{t}) r_{t}. 
 \quad（5）
$$

也就是说，不同算法用不同的裁剪因子𝜎𝑡来降低学习率𝜂𝑡。不同算法选择裁剪因子的方式总结在下面的表格中。


**Clippy是GC/AGC/LAMB的结合体**

首先，Clippy在训练过程中从GC风格切换到AGC风格。在模型训练初期，当|𝒘| ≈ 0时，$𝜆_{abs}$主导裁剪阈值$𝜆_{rel} \mid 𝒘_𝑡 \mid +𝜆_{abs}$，使得Clippy接近GC。在后续训练中，当$𝜆_{rel} \mid 𝒘 \mid ≫ 𝜆_{abs}$时，Clippy表现得更像AGC。然而，与GC/AGC相比，Clippy依赖于更新而不是梯度。此外，尽管Clippy和LAMB都使用更新，Clippy并没有像LAMB那样完全忽略更新的幅度。最后，Clippy使用𝐿∞范数而不是𝐿2范数，以对少数坐标中的剧烈更新变化更敏感。

### 4.3.2 局部裁剪还是全局裁剪

使用Clippy时，我们对每一层的更新进行裁剪（即局部裁剪），而不是对所有模型参数作为一个整体进行裁剪（即全局裁剪），这与其他方法（如GC/AGC/LAMB）类似。这提供了更细粒度控制的灵活性，但会导致有偏的梯度更新。然而，在大批量设置中，可以证明这种偏差很小[34]。

### 4.3.3 适应其他优化器

通过使用优化器依赖的更新𝒓𝑡，人们可以轻松地将Clippy适应于Adagrad之外的其他优化器。从经验上，我们还观察到，在Adam[19]上应用Clippy时，在不损害收敛性的情况下，训练稳定性有明显好处。但我们将Clippy的理论收敛性分析留给未来的工作。

# 5 实证研究

在本节中进行的实验是基于YouTube生产数据集进行的，实验分为两部分。首先，我们将Clippy与其他基线进行比较，以验证其在提高模型稳定性方面的优势。然后，我们将展示对Clippy的一些进一步分析，以更好地理解它的优势。

## 5.1 实验设置

**5.1.1 模型细节。**除了在第3.1节中已经介绍的所有模型属性外，值得一提的是，我们通过以下方式简化了我们的排序模型：(1) 仅保留最重要的任务子集和输入特征；(2) 使用具有几个共享隐藏层的简单共享底部结构。尽管比生产模型简单得多，我们发现它是一个足够好的测试平台，用于研究训练稳定性问题，因为它允许我们更快地训练模型，更专注于研究视角而不是不相关的建模细节。该模型是使用TensorFlow2 [1]构建的，并在TPU上使用65k的大批量大小进行训练。

**5.1.2 评估协议。**不幸的是，没有可靠的度量标准来量化模型的训练稳定性。为了准确衡量更好的训练稳定性带来的好处，我们改变模型复杂性以及学习率，然后检查模型的离线质量，对于二元分类任务用AUC衡量，对于回归任务用RMSE衡量。可以合理假设，更复杂的模型提供更好的离线质量，但更有可能遭受loss发散问题。因此，如果一种算法能显著提高模型的训练稳定性，我们应该在使用它时观察到更好的离线指标。更具体地说，我们使用数据的前(𝑁−1)天来顺序训练模型，并持续评估模型在最后一天（第𝑁天）数据上的性能（AUC或RMSE）。如果模型在训练期间没有遭受任何loss发散问题，我们应该观察到评估指标不断变好，因为模型正在适应更接近第𝑁天数据的数据分布。而如果模型在训练期间loss发散，无论是完全发散还是持续微发散，评估指标将受到显著影响。

为了探索模型复杂性的影响，我们考虑了表1中总结的各种模型设置。Small和Large都使用简单的前馈网络作为共享底部，分别有两层512和四层4096。Large+DCN是在Large的基础上构建的，通过在输入上添加DCN-v2层[31]，然后是标准的层归一化[4]，进一步增加了复杂性。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/00bc28a541102e6d18683e7f1e7183636e5ebdc04da2d0cead04a95975647650f278e164fd86e22e14e7fca25564c3ed?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t1.jpg&amp;size=750">

表1

### 5.1.3 基线

我们在非嵌入式模型参数上应用Clippy和其他基线，并比较它们的有效性。以下是这些基线和Clippy的更多细节。
- **梯度裁剪（GC）[24]：**我们使用了逐层（局部）梯度裁剪，裁剪阈值从𝜆GC∈{10−1, 10−2, 10−3}中搜索得出。
- **自适应梯度裁剪（AGC）[7]：**我们使用了论文中提供的官方实现，并从𝜆AGC∈{10−2, 10−3, 10−4}中搜索裁剪阈值。
- **LAMB（适应Adagrad）[34]：**LAMB最初是基于Adam[19]提出的，而作者还提供了我们在4.3.1节中介绍的通用裁剪形式。我们选择𝜙(𝑥)=𝑥，如官方实现。由于LAMB使用参数𝐿2范数∥𝒘∥2作为更新幅度，与其他方法不同，我们必须通过𝜇缩放学习率，并搜索𝜇∈{10−1, 10−2, 10−3}。
- **Clippy：**Clippy有两个超参数𝜆abs和𝜆rel，所以调整可能更不平凡，但我们发现简单地设置𝜆rel=0.5和𝜆abs=10−2在我们的实验中就能给出不错的性能。

## 5.2 整体性能

表2展示了Clippy与其他基线在不同模型设置上的总体比较。尽管模型是在六项任务上训练的，但由于空间限制，我们只展示了两个最具代表性任务的指标——一个用AUC（百分比）评估的二元分类任务和另一个用RMSE评估的回归任务。我们不仅使用原始学习率，还尝试加倍学习率，看看是否有任何方法能从中受益。在确定了最佳学习率后，我们在不同的随机种子下重复相同的设置3次，并报告了平均值和标准差。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/044e81ef19bd1402df66e12eba60b95ac3aed19d5caf4ecfa65327107aa7ff6e1108ddeea64c1fb9ed7b40d2559d1c6c?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t2.jpg&amp;size=750">

表2

**查看表2，我们可以看到没有任何训练稳定性处理的简单方法总是会遭受loss发散，即使是在小型模型上也是如此。如果我们大幅降低学习率，它有机会存活下来（见补充材料的A.1节），但我们在这里省略了它的结果，因为它们很差。GC可以在2倍学习率下存活并为小型和大型模型提供良好的结果。但在具有DCN的更复杂模型中，GC只能使用1倍学习率，否则它将遭受loss发散问题（见图5a右侧的蓝线）。AGC在1倍学习率下对小型和大型做了合理的工作，但在2倍学习率下表现变差。在Large+DCN上，AGC使用1倍或2倍学习率都显示出非常高的方差（见图5a中的橙色线），表明AGC在保持训练稳定性方面已经达到了它的极限。LAMB使用1倍学习率成功地训练了模型，没有遭受训练不稳定问题，但收敛性受到了负面影响。在图5a中，我们发现LAMB的结果总是比其他方法差。我们认为这是由于LAMB完全忽略了更新幅度，导致在参数𝐿2范数很小时，初始训练的收敛非常慢。令人惊讶的是，在所有设置中，GC在所有基线中表现最佳，这可能是因为模型相对简单，因此调整GC的裁剪阈值仍然很容易。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/621452b155edaf27d0c7f6bd259faa83090ee7a054a49e6d1d1282222b47e2049927b974facf75204daaecf82a3ebd16?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=5.jpg&amp;size=750">

图5

在表2的最后一列，我们可以看到Clippy在所有模型设置中使用2倍学习率。更重要的是，Clippy没有妥协收敛性，它在小型和大型模型上与GC（即最佳基线）具有可比的结果（见图5b），并且在Large+DCN模型上与GC相比有显著更好的AUC（请注意，在我们模型中0.1%的AUC改进被认为是非常显著的，并且可以导致实时度量增益）和RMSE。

我们想要强调的一个重要发现是，当模型更复杂并且使用更大的学习率训练时，Clippy提供了更大的增益。在图5b中，我们可以看到当使用更复杂的模型和2倍学习率时，Clippy和GC之间的差距正在扩大。所以我们不惊讶Clippy可以在比Large+DCN复杂得多的生产模型中提供帮助。

## 5.3 仔细看看Clippy的裁剪因子

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/85661785075f2ffb3bd2e6e0e94df9eff24fb68fc35f32ef2a34b0e4b4f2088c1ce98d30e8d400cd08a2cc359c22863f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=6.jpg&amp;size=750">

图6

图6显示了在训练Large+DCN模型过程中Clippy在不同层的裁剪因子。如第4节中介绍的，裁剪因子𝜎∈(0.0, 1.0]。较小的裁剪因子表示进行了更多的裁剪以降低学习率。由于裁剪是逐层应用的，我们为几个层绘制了裁剪因子，包括(1)DCN层的权重，在(2)共享底部的(3)顶层和(4)底层隐藏层，以及在(5)二元分类任务和(6)回归任务的输出层。有趣的是，我们看到模型的底层进行了更多的裁剪。我们认为这直观上是有意义的，因为底层通常有较小的参数范数，所以Clippy的裁剪阈值也会较小。另一方面，这可能有助于训练稳定性，因为我们知道底层权重的微小变化可能导致模型输出的很大差异。



# 

# 

[https://arxiv.org/pdf/2302.09178](hhttps://arxiv.org/pdf/2302.09178)