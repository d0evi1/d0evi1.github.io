---
layout: post
title: focal loss介绍
description: 
modified: 2022-04-30
tags: 
---

facebook在《Focal Loss for Dense Object Detection》提出了focal loss。

# 3.Focal loss

focal loss被设计用来解决one-stage object detection场景，该场景在训练期间在foreground和backgroud classes间是极不平衡的（extreme impalance）（例如：1:1000）。我们会从二分类的cross entropy（CE）开始来介绍forcal loss：

$$
CE(p, y) = \begin{cases}
-log(p),  & \text{if $y=1$} \\
3n+1, & \text{otherwise.}
\end{cases}
$$

...(1)

其中：

- $$y \in \lbrace underset{+}{-} \rbrace$$表示ground-truth class
- $$p \in [0, 1]$$是对于label y=1的class的模型估计概率

对于简洁性，我们定义了$$p_t$$：

$$
p_t = \begin{cases}
p,  & \text{if $y=1$} \\
1-p, & \text{otherwise.}
\end{cases}
$$

并重写为：$$CE(p, y) = CE(p_t) = - log(p_t)$$。

CE loss可以被看成是图1中的蓝色曲线(top)。在该图中可以发现，该loss的一个重要属性是，即便是可以被轻松分类的样本（easy classified）（$$p_t >> 0.5$$），也会带来一个具有non-trivial规模的loss。当在大量easy样本（easy examples）进行求和时，这些小的loss values会淹没掉稀有类（rare class）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/a3db84ea68f85701fc6bc2e893edc9e6e56db2f8f3a9f9fe506c74ee2dfdfc52758ebf56d65f9d36f345deffa4f977f0?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 我们提出了一种新的loss：Focal Loss，它会添加一个因子$$(1 - p_t)^{\gamma}$$到标准的cross entropy criterion中。设置$$\gamma > 0$$可以减小对于well-classified样本（$$p_t > 0.5$$）的相对loss，从而更多关注hard、misclassified样本。如实验所示，提出的focal loss可以使训练：在出现许多easy background examples下的高度精准的dense object detector.

## 3.1 Balanced Cross Entropy

解决class imbalance的一个常用方法是，为class 1引入一个weighting因子$$\alpha \in [0, 1]$$，为class -1引入$$1 - \alpha$$。惯例上，$$\alpha$$可以通过inverse class frequency来设置，或者被看成是通过cross validation设置的一个超参数。对于简洁性，我们定义了：

$$
CE(p_t) = -\alpha_t log(p_t)
$$

...(3)

该loss是对CE的一个简单扩展，我们会作为一个实验baseline进行对比。

## 3.2 Focal Loss定义

如实验所示，在dense detectors的训练期间遇到大类不均衡（large class imbalance）会淹没掉cross entropy loss。易分类负样本（Easily classified negatives）组成了loss的绝大多数，会主宰gradient。而$$\alpha$$会平衡正样本/负样本的importance，它不会区分easy/hard样本。作为替代，我们提出：将loss function变形为：对easy examples进行down-weight，从而在训练时更关注hard negatives。

更正式的，我们提出了增加一个modulating factor $$(1 - p_t)^{\gamma}$$到cross entropy loss中，可调参数为$$\gamma \geq 0$$，我们定义focal loss为：

$$
FL(p_t) = -(1-p_t)^{\gamma} log(p_t)
$$

...(4)

该focal loss在图1中根据$$\gamma \in [0, 5]$$的多个值进行可视化。我们注意到focal loss的两个特性。

- (1) 当一个样本被误分类时，$$p_t$$会很小，调节因子（modulating factor）接近1，loss不受影响。随着$$p_t \rightarrow 1$$，该因子会趋向为0，对于well-classified的样本的loss会down-weighted。
- (2) focusing参数$$\gamma$$会平滑地调节easy样本被down-weighted的rate。当$$\gamma=0$$时，FL接近于CE，随着$$\gamma$$的增加，调节因子的影响也可能增加（我们发现$$\gamma=2$$在实验中表现最好）。

直觉上，调节因子会减小来自easy examples的loss贡献，并拓宽一个样本接收到low loss的范围。例如，$$\gamma=2$$，使用$$p_t=0.9$$分类的样本会比CE低100倍loss，而使用$$p_t \approx 0.968$$则具有1000倍的更低loss。这会增加纠正误分类样本(对于$$p_t \geq 0.5$$和$$\gamma=2$$，它的loss会被缩放到至多4倍)的importance。

惯例上，我们使用一个focal loss的$$\alpha$$-balanced变种：

$$
FL(p_t) = -\alpha_t (1 - p_t)^{\gamma} log(p_t)
$$

...(5)

我们在实验中采用该格式，因为它对比non-$$\alpha$$-balanced形式在accuracy上有微小提升。最终，我们注意到，该loss layer的实现在计算loss时会组合上sigmoid操作来计算p，产生更好的数值稳定性。

而在我们的实验结果中，我们使用focal loss定义。在附录中，我们考虑focal loss的其它实例，并演示相同的效果。

## 3.3 类不平衡和模型初始化

二分类模型缺省被初始化为：对于y=1或-1具有相等的输出概率。在这样的初始化下，出现了Class Imbalance，loss会由于高频分类（frequent class）主导了total loss，造成在early training中的不稳定。为了消除它，对于rare class（foreground）在训练早期由模型估计的p值，我们引入一个“先验（prior）”概念。我们将prior通过$$\pi$$表示，并将它设置成：对于rare class的样本，以便模型的估计p很低，例如：0.01. 我们注意到，在模型初始化时，这是个变化，而在loss function上并不是。我们发现，这对于cross entropy和focal loss来说，对于heavy class imbalance的case，可以提升训练稳定性。



# 

- 1.[https://arxiv.org/pdf/1708.02002.pdf](https://arxiv.org/pdf/1708.02002.pdf)