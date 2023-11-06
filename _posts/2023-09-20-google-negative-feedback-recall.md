---
layout: post
title: google Negative User Feedback召回介绍
description: 
modified: 2023-09-20
tags: 
---

# 2.negative user feedback的训练目标

以下我们描述了将negative user feedback纳入sequential recommenders的训练目标中的方法，特别是针对在大型语料库中预测下一个推荐items set的检索任务。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/d8f410535eb6878f70de0fefd9c24f6b0c782f8f767a423307396e16895190085847c97e6ac32b08ac5d9b9a67096bb2?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1

**Loss Function.** 

顺序检索模型(Sequential retrieval models)通常用于预测postive交互的任务，使用的训练目标包括：cross-entropy loss或REINFORCE policy gradient[2,5]。在这里，我们从一个baseline模型开始，该模型会使用cross-entropy loss，并使用postive用户反馈作为训练标签。给定一个训练样例i，其中包含user $u_i$与label item $y_i$的postive交互，loss function是负对数似然：$$L_i=−log(p(y_i \mid s_i))$$，其中：

- $s_i$：表示由神经网络计算的用户状态向量，该向量汇总了用户的交互历史。

该损失函数优化了在给定用户状态$s_i$的情况下，从大型语料库中推荐标签项$y_i$的条件概率$p(y_i \mid s_i)$。概率项表示为softmax：

$$
p(y_i|s_i)=exp(s_i^T v_{y_i})/ \sum\limits_{y_i' \in A} exp(s_i^T v_{y_i'})
$$

其中:

- A：是item空间
- $v_{y_i}$：表示item embedding vector

在实践中，训练中使用sampled softmax，serving中使用最近邻搜索(NNS)来处理极大的item空间。

为了从负面用户反馈中学习，我们引入了一个“不推荐（not-to-recommend）”损失函数，即不推荐一个物品的负对数似然，即 $ L_i=−log(1−p(y_i|s_i)) $。该目标允许模型直接利用negative user feedback作为negative  labels，并与positive labels的现有损失函数一起工作。对于每个postive label添加权重$𝑟_𝑖$，对于每个postive label添加权重$𝑤_𝑖$，我们得到整体损失函数：

$$
L = − \sum\limits_{i \in D_{pos}} (r_i \cdot log(p(y_i | s_i))) − \sum\limits_{i \in D_{neg}} (w_i  \cdot log(1−p(y_i | s_i)))
$$

...(1)

其中:

- $𝐷_{pos}$和$𝐷_{neg}$分别是postive和negative label（如图1所示）。最小化此loss相当于最大化推荐具有postive feedback的物品和不推荐具有negative feedback的物品的联合概率。与仅用positive label相比，它会强化与用户兴趣的更强的一致性。

“不推荐（not-to-recommend）”损失函数解决了在training objective中建模negative user feedback的几个实际问题。例如，使用cross-entropy loss $L_i=−log(p(y_i \mid s_i))$和负值label weights可能会减少不想要物品的推荐概率，但当$p(y_i \mid s_i) \rightarrow 0$时，会导致梯度爆炸。原则上，强化学习目标支持为negative training labels分配negative reward values。实际上，我们可以用REINFORCE目标[5]替换positive labels的loss项。但是，对于negative labels，在REINFORCE推荐器中使用负回报(negative rewards)面临一个实际的挑战，在工业环境中，即使经过off-policy correction，由于极大的item空间，当$p(𝑦_𝑖 \mid s_𝑖)\rightarrow 0$时，梯度仍可能爆炸。 “not-to-recommend”损失函数规避了这些问题，因为当$𝑝(y_i \mid s_i)→0$时，梯度的保证是有限的。另一种方法是在相邻postive label的softmax负样本中包含并加权负反馈项( upweight negative feedback items)。与这种方法相比，所提出的方法将postive和negative用户反馈的loss项解耦，并且梯度允许更有针对性地从negative labels中学习。

**Negative Training Labels和Input Features**。

显式和隐式的negative用户反馈都可以作为negative训练label。label权重可以根据反馈强度、信号密度以及相对于postive标签的loss值大小进行调整。**在我们的平台上，我们将明确的dislike操作和隐式的跳过(skip)信号视为负面训练标签**。为了在模型输入中表示用户过去的negative交互，模型输入序列中的每个物品都通过二元特征（binary feature）编码dislikes，而skip则作为停留时间特征（ dwell time feature）的一部分。

# 3.真实环境

我们将“不推荐（not-to-recommend）”损失函数引入基于RNN的顺序推荐器[2,5]，该推荐器服务于数十亿用户的短内容平台，并进行实时实验以评估建模"negative user feedback"的真实世界影响。在我们的系统中，顺序推荐器是检索阶段的一部分，从庞大语料库中推荐数百个item，在进行ranking阶段[8]后，最佳的物品会被展示给用户。在实践中，negative user feedback也由下游组件（如ranking模型和启发式算法[1,18,24]）决定。即便如此，我们仍然从模型变化中看到了有意义的端到端结果。我们证明，从检索中negative user feedback可以通过：

- （1）减少不想要的推荐通过系统的机会
- （2）增强模型与用户兴趣的一致性对齐从而生成更好的推荐

下面报告的指标变化在95％置信区间下是显著的。

**建模显式的负面反馈**

在这个实验中，我们将明确的dislike反馈纳入模型的训练目标中。与不使用dislike信号的模型基线（图2a）相比，在产品主页上使用dislike信号作为输入特征和训练label的实验模型，可以将dislike率降低了2.44％。这种效果比仅使用dislike作为输入特征而不是训练label（-0.34％，不显著）、或排除dislike物品的启发式解决方案（-0.84％）要大得多。同一创作者的重复dislike率下降了9.60％，**表明模型在负面反馈后减少了类似的推荐**。实验模型将拒绝用户数（Dismissing）降低了2.05％，而观看者仅下降了0.14％。在我们的处理中，拒绝（Dismissals）并没有被显式建模，这种变化表明用户体验确实获得了改善。在我们产品的沉浸式反馈中，这种效果尤为显著。实验模型将dislike率降低了1.10％，同一创作者的重复dislike率降低了7.10％。使用基于模型的满意度指标[7]，满意的消费保持不变（+0.00％），不满意的消费减少了0.35％。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/0dbb88dad5f613eb33f20cf586fb21c418d861aca81ddf4669aac077b95a40199f9b526e1aaf3c2590565f6f3b99f048?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2 

**建模隐式负面反馈**

隐式跳过（skip）信号的密度比显式的dislike信号要高得多。我们假设更高的负面用户兴趣覆盖率可以提高整体推荐质量。在这个实验中，我们将skip item作为negative training labels纳入到沉浸式反馈上。我们观察到：整体用户享受度（enjoyment）增加了0.40％（图2b），每天至少有1小时活动的活跃用户增加了0.61％。此外，我们看到同一创作者的重复跳过率降低了1.44％，多样性增加了0.88％。

综合来看，结果表明，将显式负反馈（explicit negative feedback）纳入训练目标可以降低负面用户体验，而建模隐式负反馈（implicit negative feedback）可以提高整体用户享受度。

**衡量反应能力（RESPONSIVENESS）的反事实模拟（COUNTERFACTUAL SIMULATION）**

真实实验显示了建模负面反馈的整体好处，但没有直接衡量推荐器对这种反馈信号的响应能力（responsive）。**我们的目标是衡量响应能力（responsiveness），即当用户对某个物品做出负面反应时，推荐器可以减少多少类似的推荐**。然而，由于复杂的用户-推荐器交互，这种推荐器行为无法通过离线模型准确性、在线用户指标或相关的日志数据分析来评估。为了直接衡量推荐器的响应能力，我们需要排除混淆的用户行为，并比较反事实用户行为（counterfactual user actions）对推荐器响应的因果效应（causal effects）。

在这里，我们开发了一个框架，使用模拟用户[21]来衡量对用户反馈的响应能力（图3a）。每个用户按照随机行为轨迹来消费一系列𝑘-1个推荐物品。在第𝑘个物品上，我们在相同的交互历史下模拟多个反事实行为（例如，对其做出负面反应或不这样做）。在每个反事实分支中，我们观察推荐器输出的（𝑘+1）步并计算它们与上一个交互物品的相似度。使用基于内容和基于创作者的相似度的独立度量，**我们将相似度得分定义为：推荐与上一个交互物品相同内容簇或创作者的比例**。然后，**推荐器响应能力被计算为：【提供负反馈 vs. 不提供负反馈】两者间相似度得分的相对变化**。这告诉我们推荐器在负反馈时减少了多少类似的推荐，可以用于评估建模变化。

图3

我们使用这种方法来衡量推荐器对dislike操作的响应能力，由于其稀疏性这很难进行评估。对于反事实行为(counterfactual actions)，我们考虑一个postive交互baseline（长停留时间）与在其上添加不喜欢操作。我们运行了2000次模拟，其中𝑘=50。不使用dislike信号的baseline模型没有响应能力，需要依赖下游系统组件来响应dislike。仅添加dislike输入特征会导致类似的推荐在dislike后显著但有限地减少（-22.7％/-22.8％按内容/创作者相似度），表明dislike操作代表了内在的用户不满意。当在训练label和输入特征中都使用dislike时，模型在dislike后减少了更多类似的推荐（按内容/创作者相似度分别为-60.8％/-64.1％）（图3b）。这些结果表明，**将显式负反馈纳入训练目标可以提高模型对这种反馈的响应能力**。

# 

- 1.[https://arxiv.org/pdf/2308.12256.pdf](https://arxiv.org/pdf/2308.12256.pdf)