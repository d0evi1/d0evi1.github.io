---
layout: post
title: distillation介绍
description: 
modified: 2017-12-11
tags: 
---

hinton在google提出了《Distilling the Knowledge in a Neural Network》 knowledge distillation。

# 1.介绍

许多昆虫会有一个幼虫状态，可以最优地从环境中抽取能量和营养；也会有一个成虫状态，以便最优地进行飞行和繁衍。在大规模机器学习中，我们通常在训练阶段和部署阶段使用非常相似的模型来，尽管两个阶段有着非常不同的要求：对于像语音识别和目标识别，训练必须从非常大、高度过剩的数据集中抽取结构，**但不能实时地方式运行，它会使用大量计算**。然而，部署大量用户通常在时延（latency）和计算资源上具有非常多的严格要求。在昆虫上的类比，建议我们可以去训练非常大的模型（cumbersome models），它可以轻易地抽取来自数据中的结构。**cumbersome model可以是独立训练模型的ensemble，或者使用像dropout等强正则方式训练的单个非常大的模型**。一旦cumbersome model被训练后，我们可以接着使用一个不同方式的训练，**我们称作“distillation”，可以将来自cumbersome model的knowledge进行transfer到一个小模型中，它会更适合部署**。该策略的版本已经由Rich Caruana和它的同事一起率先尝试。在他们的重要paper中，他们表示可以由一个大模型的ensemble的knowledge进行transfer到一个小模型中。

一个阻止更多研究这种方法的概念块是：我们趋向于识别出（identify）在一个已训练模型中的knowledge（它具有学到的参数值），这使得它很难看到我们是如何变更该模型的形式，但却仍会记住相似的knowledge。从knowledge的一个抽象视角看（这会从任何特殊实例中解释出来），**是从input vectors到output vectors的一个可学习映射（learned mapping）**。对于cumbersome models，可以学习判别许多类，正常的训练目标是，对正确答案（correct answer）最大化平均log概率，但该学习的一个side-effect是：该训练好的模型会分配概率给所有不正确的答案，即使当这些概率非常小时，其中一些会比其它更大。不正确答案的相对概率告诉我们关于cumbersome model是如何趋向于gnerealize的许多信息。例如，一张关于BMW的照片，可能有少量的机会被误认为是一个垃圾车（garbage truck），但这种错误仍然要比被误认为一个胡萝卜的概率要大许多倍。

这通常是可接受的，对于训练使用的objective function会影响该用户的true objective，使得尽可能接近。尽管如此，模型通常会被训练成在训练数据上最优化，当实际objective是泛化给新数据。它会明显更好地训练模型以便更好泛化，但这需要关于correct way的信息来泛化，该信息通常没有提供。当我们从一个大模型中distill知识给小模型时，我们可以训练该小模型来泛化（正如大模型一样）。例如，如果cumbersome model泛化良好，它是一个不同模型的ensemble，一个小模型会以相同的方式训练用来进行泛化，通常会比正常训练的单个小模型在测试数据上要好。

transfer该cumbersome model的泛化能力的一个直接方式是，使用由cumbersome model生成的类概率(class prob)作为“soft targets”来进行训练小模型。**对于transfer阶段，我们可以使用相同的训练集或者一个单独的"transfer" set。当cumbersome model是一个许多更简单模型的ensemble时，我们可以使用一个单一预测分布的算法或者几何平均作为soft targets。当soft targets具有高熵时（high entropy），他们会比hard targets在每个训练case上提供更多信息，并在traning cases间的梯度上具有更小的variance，因此小模型通常会比原始的cumbersome model使用更少数据训练，并使用一个非常更高的learning rate**。

对于像MNIST这样的任务，它的cumbersome model几乎总是生成具有正确答案，它会具有高置信度、更多关于learned function的信息，在soft targerts中以非常小的概率比（ratios）存在。例如，一个关于2的version，可能会给出一个具有概率$$10^{-6}$$是3，以及$$10^{-9}$$是7，其它version有其它的形式。这是非常有价值的信息，它定义了一个在数据上的丰富的相似结构（例如：2看起来像3，也看起来看7），但**它在transfer stage时在cross-entropy cost function上具有非常小的影响，因为该概率会非常接近于0**。Caruana和它的同事通过使用该logits（到最终softmax的inputs）来解决该问题，而非由softmax生成的概率作为targets来学习该小模型，并且它们会最小化在logits和cumbersome model间的平方差（squared difference）。我们的更通用解决方案称为“distillation”，会提出关于final softmax的temperature，接到cumbersome model生成一个关于targets的合适soft set。当训练该小模型时，我们接着使用相同的temperature匹配这些soft targets。我们会接着展示：对该cumbersome model进行匹配logits实际是distillation的一个特例。

transfer set被用于训练小模型，可以包括无标记的数据（unlabeled data），或者我们可以使用原始的training set。我们已经发现，使用该原始的training set运行良好，特别是：如果我们添加一个小项到objective function中时，这会鼓励小模型去预测true targets，同时茶杯由cumbersome model提供的soft targets。通常，小模型不能完全匹配soft targets，并且在正常答案上的erring可以是有帮助的。

# 2.Distillation

许多networks通常会使用一个"softmax" output layer来生成类概率（class probabilities），它会对logits $$z_i$$进行转化，为每个class计算一个probability, $$q_i$$，并与其它logits进行相互比较$$z_i$$。

$$
q_i = \frac{exp(z_i / T)}{\sum_j exp(z_j / T)}
$$

...(1)

其中，T是一个temperature，通常设置为1. 将T设置得更高会生成一个在classes上的softer probability分布。

在distillation的最简形式中，通过在一个transfer set上训练模型，并使用一个对每个case在transfer set中的一个soft target分布（使用cumbersome model并在softmax中使用一个高temperature），knowledge会被转移到distilled model中。当训练该distilled model时，会使用相同高的temperature，但在它被训练后，它会使用一个temperature为1。

当correct labels被所有或部分transfer set知道时，该方法可以极大提升，也可以训练该distilled model来生成correct labels。这样做的一种方式是，使用correct labels来修正soft targets，但我们发现一种更好的方式是：简单使用一个对两个不同objective functions的weighted average。第一个objective function是使用soft targets的cross entropy，该cross entropy使用distilled model的softmax上的相同logits，但temperature=1。我们发现，获得的最好结果，会在objective function上使用一个相当低的weight。由于通过soft targets生成的梯度幅度缩放为 $$1/T^2$$，当同时使用hard和osft targets时，乘上$$T^2$$很重要。当实验使用meta-parameters时，如果用于distillation的temperature发生变化，这可以确保hard和soft targets的相对贡献仍然大致不变。

## 2.1 匹配logits是distillation的一个特征

在transfer set中每个case，对于distilled model的每个logit $$z_i$$，贡献了一个cross entropy gradient, $$dC/ dz_i$$。如果cumbersome model具有logits $$v_i$$，它会生成soft target probabilities $$p_i$$，并且transfer training会在temperature T上完成，我们给出了该gradient：

$$
\frac{\partial C}{\partial z_i} = \frac{1}{T} (q_i - p_i) = \frac{1}{T} (\frac{e^{z_i/T}}{\sum_j e^{z_j/T}} - \frac{e^{v_i/T}}{\sum_j e^{v_j/T}})
$$

...(2)

如果对于logits的幅值，temperature高，我们可以近似：

$$
\frac{\partial C}{\partial z_i}  \approx \frac{1}{T}(\frac{1+z_i/T}{N+\sum_j z_j/T} - \frac{1+v_i/T}{N + \sum_j v_j/T}
$$

...(3)

如果我们假设：对于每个transfer case，logits已经是独立零均值的，以便$$\sum_j z_j = \sum_j v_j = 0$$，等式3简化为：

$$
\frac{\partial C}{\partial z_i} \approx \frac{1}{NT^2} (z_i - v_i)
$$

...(4)

因此，在高temperature限制下，distilliation等价于最小化$$1/2(z_i - v_i)^2$$，提供的logits对于每个transfer case都是独立零均值的。在更低的temperatures上时，distilliation会花费更少的attention来matching logits，以便比平均有更多的negative。这有潜在优势，因为这些logits对于用于训练cumbersome modelcost function几乎完整无限制，因此他们可能非常有noisy。另一方面，非常负的logits可能会传达关于由cumbersome model获取knowledge的有用信息。这些效果占据着哪些是一个经验性问题。我们展示了当distilled model太小，而不能获取cumbersome model中的所有知识，intermediate temperatures会运行最好，会强烈建议忽略掉大的negative logits，这样会有用。

# 3.在MNIST上的初步实验

...

# 4.在语音识别上的实验

...

# 5.在非常大数据集上训练ensembles

...

# 6.Soft targets作为Regularizers

使用soft targets来替代hard targets的主要目的之一是，在soft targets中携带的一大堆有用的信息，不能使用单个hard target进行编码。在本节中，我们展示了，通过使用相当少的数据来满足baseline语音模型中85M参数，有非常大的影响。表5展示了只有3%的数据（大概20M样本），使用hard targets的baseline model进行训练，会导致严重的overfitting（我们做了early stopping，但在accuracy达到44.5%迅速下降），其中使用soft targets的相同模型可以恢复几乎在完整训练集上的所有信息（大概2%）.这十分深刻，注意我们不必做early-stopping：使用soft targets的系统会简单“收敛”到57%。这展示了soft targets是一个非常有效的方式，一个模型在所有数据上训练发现的regularities会与另一个模型相交流（communicating）。

## 6.1 使用soft targets来阻止specialist避免overfitting

在JFT dataset上，我们在实验上使用的specialists会将所有其它非专家类（non-specialist classes）收缩到单个垃圾类上（dustbin class）。如果我们允许specialists具有一个在所有classes上的full softmax，比起early stopping有一种更好的方式来阻止overfitting。一个specialist会在它的special classes上高度增强的数据上进行训练。这意味着，训练集的有效大小（effective size）会更小，它具有一个很强的趋势来overfit它的special classes。该问题不能通过将specialist变得更小来解决，因为我们会丢掉非常有用来自对所有non-specialist classes建模得到的transfer effects。

我们的实验会使用3%的语音数据，它强烈建议如果一个specialist会使用genrealist的weights进行初始化，我们可以让它保持几乎所有关于non-special classes的知识，通过为non-special classes使用soft targets来训练，另外使用hard targets来训练它。soft targets可以由generalist来提供。我们现在会探索该方法。

# 7.与MoE（Mixtures of Experts）的关系

使用specialists可以在数据的子集上进行训练，它与MoE（mixtures of experts）具有相似性，它使用一个gating network来计算分配每个样本到每个expert的概率。同时，由于experts会学习来处理分配给它们的样本，gating network会学习选择哪个experts来分配每个样本，基于experts对该样本的相对判别表现（relative discriminative performance）。使用该关于该experts的discriminative performance来判断该学到的分配，会比简单将input vectors进行聚类并为每个cluster分配一个expert的方式要好的多，但它使得训练很难并行化：首先，每个expert的加权训练集（weighted training set）会以依赖于所有其它experts的方式保持变更；第二，gating network需要对比在相同样本上不同experts的表现来知道如何修正它的分配概率。这些难点意味着：MoE（Mixture of experts）很少被用于那些最有收益的地方：具有海量数据集并且包括着不同的子集的任务。

对多个specialists的训练并行化更容易。我们首先训练一个gneralist model，接着使用confusion matrix来定义用于训练specialists的subsets。一旦这些subsets已经被定义，specialists可以完全独立进行训练。在test time时，我们可以使用genralist model的预测来决定，哪个specialists是相关的，并且只有这些specialists需要被运行。

# 8.讨论

我们已经展示了，distilling可以式作良好，将一个ensemble、或者从一个大的高度复杂的模型的知识进行transfer到一个更小的distilled model上。在MNIST上，当transfer set被用于训练distilled model时，即使缺少一或多个classes，distillation的效果也很好。对于一个更深的语音模型，用于android voice search，我们展示了，几乎所有的提升都通过训练一个dnn的ensemble并将它distilled到相同size的单个neural net上（它更容易部署）来达到。

对于实际的大神经网络（big neural networks），训练一个full ensemble是可行的，但我们发现，单个大网络要训练很久，可以通过学习大量specialist nets来极大提升效果，每个specialist可以学习判断在一个高度confusable cluster中的classes。到目前为上，我们并没有展示，我们可以在specialists中的知识distill到单个大的net上。


# 参考


- 1.[https://arxiv.org/pdf/1503.02531.pdf](https://arxiv.org/pdf/1503.02531.pdf)