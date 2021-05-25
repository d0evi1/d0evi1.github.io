---
layout: post
title: taobao Privileged Features Distillation介绍
description: 
modified: 2020-12-16
tags: 
---

# 1.介绍

最近几年，DNNs已经在推荐任务预测上达到了非常好的效果。然而，大多数这些工作集中在模型本身。只有有限的工作把注意力放到输入的特征方面，而它可以决定模型表现的上界（upper-bound）。在本工作中，我们主要关注于特征方面，特别是在电商推荐中的features。

**为了确保offline training与online serving的一致性，我们通常在真实应用的两个enviorments中我们使用相同的features**。然而，有一些有区分性的特征（discriminative features）会被忽略（它们只在训练时提供）。以电商环境中的CVR预测（conversion rate）为例，这里我们的目标是：**估计当用户点击了该item后购买该item概率**。在点击详情页（clicked detail page）上描述用户行为的features（例如：在整个页面上的dwell time）相当有用。**然而，这些features不能被用于推荐中的online CVR预测，因为在任意点击发生之前预测过程已经完成**。尽管这样的**post-event features**确实会在offline training记录。为了与使用privildeged information的学习相一致，**这里我们将对于预测任务具有区分性（discriminative）、但只在训练时提供的features，称为priviledged features**。

使用priviledged features的一种简单方法是：multi-task learning，例如：使用一个额外的任务来预测每个feature。然而，在multi-task learning中，每个任务不会满足一个无害保障（no-harm guarantee）（例如：priviledged features可能会伤害原始模型的学习）。更重要的，no-harm guarantee非常可能违反，因为估计priviledged features比起原始问题[20]相当具有挑战性。从实际看，当一次只使用几十个priviledged features，对于所有任务进行调参是个大挑战。

受LUPI（learning using priviledged information）【24】的启发，这里我们提出priviledged features distillation(PFD)来使用这些features。我们会训练两个模型：例如：一个student和一个teacher模型。student模型与original模型相同，它会处理offline training和online serving的features。teacher model会处理所有features，它包括：priviledged features。知识会从teacher中distill出来（例如：在本工作中的soft labels），接着被用于监督student的训练，而original hard labels（例如：{0, 1}）它会额外用来提升它的效果。在online serving期间，只有student部分会被抽出，它不依赖priviledged features作为输入，并能保证训练的一致性。对比起MTL，PFD主要有两个优点。一方面，对于预测任务，priviledged features会以一个更合适的方式来进行组合。通常，添加更多的priviledged features会产生更精准的预测。另一方面，PFD只会引入一个额外的distillation loss，不管priviledged features的数目是多少，很更容易进行平衡。

PFD不同于常用的模型萃取（model distillation：MD）[3,13]。在MD中，teacher和student会处理相同的inputs。teacher会使用比student更强的模型。例如，teachers可以使用更深的network来指导更浅的students。在PFD中，teacher和student会使用相同的模型，但会在inputs上不同。PFD与原始的LUPI【24】也不同，在PFD中的teacher network会额外处理regular features。图1给出了区别。

在本工作中，我们使用PFD到taobao推荐中。我们在两个基础预测任务上，通过使用相应的priviledged features进行实验。主要贡献有4部分：

- 在taobao推荐中定义了priviledged features，并提出了PFD来使用它们。对比起MTL来独立预测每个priviledged feature，PFD会统一所有的，并提供一个one-stop的解。
- 不同于传统的LUPI，teacher PFD会额外使用regular features，它会更好地指导student。PFD与MD互补。通过对两者进行组合，例如：PFD+MD，可以达到更进一步的提升
- 我们会通过共享公共输入组件（sharing common input components）来同步训练teacher和student。对比起传统的异步使用独立组件进行训练，这样的训练方式可以达到更好的效果，而时间开销会进一步减小。因此，该技术在online learning中是可用的，其中real-time计算需要。
- 我们会在taobao推荐的两个基础预测任务上进行实验，例如：粗排中的CTR预测，以及粗排中的CVR预测。通过对interacted features（交叉特征）进行distill是不允许的，因为在粗排中的效率问题，以及在精排CVR中的post-event features，我们可以对比baseline达到极大的提升。在on-line A/B tests中，在CTR任务上点击指标可以提升+5%。在CVR任务中，conversion指标可以提升+2.3%。

# 2.相关distillation技术

在给出我们的PFD的详细描述前，首先介绍下distillation技术。总体上，该技术的目标是，帮助non-convex的student models来更好地训练。对于model distillation，我们通常会按如下方式写出objective function：

$$

$$


...

# 3.taobao推荐中的Priviledged features

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b22fed1cbf60e8ea388c3b0acdb93a2b6d3f7dfe5dd29fcea3a6ed612aa3b61ae9163987a84df0346f820d8bf90a0751?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2

为了更好地理解priviledged features，我们首先如图2所示给出taobao推荐的一个总览。在工作推荐中通常这么做，我们采用cascaded 学习框架。在呈现给用户给，有3个stages来select/rank items：candidate generation、coarse-grained ranking、fine-grained ranking。为了在效率和accuracy间做出一个好的trade-off，越往前的cascaded stage，会采用复杂和高效的模型，对items进行scoring会具有更高的时延。在candidate generation stage，我们会选择$$10^5$$个用户可能会点击或购买的items。总之，candidate genreation会从多个sources进行混合而来，比如：协同过滤、DNN模型等。在candidate generation之后，我们会采用两个stage进行ranking，其中PFD会在这时使用。

在coarse-grained ranking stage中，我们主要会通过candidate generation stage来估计所有items的CTRs，它们接着被用来选择top-k个最高的ranked items进入到下一stage。预测模型的input主要包含了三个部分。第一部分包括：用户行为，它会记录用户点击/购买items的历史。由于用户行为是有序的，RNNs或self-attention会通常被用来建模用户的long short-term interests。第二部分由user features组成，例如：user id、age、gender等。第三部分由item features组成，例如：item id、category、brand等。通过该工作，所有features都会被转换成categorical type，我们可以为每个feature学习一个embedding。

在粗排阶段，prediction model的复杂度会被严格限制，以便让上万候选在ms内完成。这里，我们使用inner product模型来对item scores进行measure：

$$
f(X^u, X^i; W^u, W^i) \triangleq <\phi_{W^u}(X^u), \phi_{W^i}(X^i)>
$$

...(3)

其中，上标u和i分别表示user和item。$$X^u$$表示user behavior和user features的一个组合。$$\phi_W(\cdot)$$表示使用学到参数的非线性映射，$$W_{\cdot}<\cdot, \cdot>$$是内积操作。由于user侧和item侧在等式(3)中是独立的。在serving期，我们会事先离线计算关于所有items的mappings $$\phi_{W^i}(\cdot)$$。当一个请求到来时，我们只需要执行一个forward pass来获得user mapping $$\phi_{W^u}(X^u)$$，并计算关于所有candidates的inner product，它相当高效。细节如图4所示。

如图2所示，粗排不会使用任何交叉特征，例如：用户在item category上在过去24小时内的点击等。通过实验验证，添加这样的features可能大大提高预测效果。然而，这在serving时会极大地加时延，因为交叉特征依赖user和指定的item。换句话说，features会随着不同的items或users而不同。如果将它们放到等式(3)中的item或user侧。mappings $$\phi_w(\cdot)$$的inference需要执行和候选数一样多的次数，例如：$$10^5$$次。总之，non-linear mapping $$\phi_W(\cdot)$$的计算开销要比简单的inner product大许多阶。在serving期间使用交叉特征是不实际的。**这里，我们将这些交叉特征看成是：在粗排CTR预测的priviledged features**。

在精排阶段，除了估计在粗排中做的CTR外，我们也会估计所有候选的CVR，例如：如果用户点击它，那么会购买该item的概率。在电商推荐中，主要目标是最大化GMV（商品交易总量），它可以被解耦成CTR X CVR X Price。一旦为所有items估计CTR和CVR，我们可以通过expected GMVs来对它们进行排序来最大化。在CVR的定义中，很明显，用户在点击item详情页上的行为（例如：停留时长、是否观看评论、是否与卖者进行交流等），对于预测来说相当有用。然而，在任何future click发生前，CVR必须要对ranking进行估计。描述在详情页上用户行为的features在inference期间并没有提供。这里，我们可以将这些features表示成priviledged features来进行CVR预测。为了更好地理解它们，我们给出图3进行演示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b58ad836c2911bf16a9f57df68559ef1cb103de2ae1da749a2f2392bdac6be55b7cd7dc2ca863017977999572df45433?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=3.jpg&amp;size=750">

图3

# 4.Priviledged Feature Distillation

如等式(2)所示，在原始的LUPI，teacher依赖于priviledged information $$X^*$$。尽管信息量大，在本工作中的priviledged featues只能部分描述用户的偏好。使用这些features的表现要比使用常规特征（regular features）要差。另外，基于priviledged features的预测可能有时会被误导（misleading）。例如，**对于顾客来说，通常会在昂贵item上花费更多时间来最终决定，而这些items的转化率相当低。当进行CVR估计时，LUPI的teacher会依赖于priviledged features（例如：停留时间）做出预测，但不考虑regular features（例如：item price），这会导致在昂贵items上做出false positive predictions**。为了缓和它，我们会额外将常规features feed给teacher model。等式(2)的原始function可以修改如下：

$$
\underset{min}{W_s} (1-\lambda) * L_s (y, f(X; W_s)) + \lambda * L_d( f(X, X^*; W_t), f(X; W_s))
$$

...(4)

通常，添加更多信息（例如：更多features），会得到更精准的predictions。teacher $$f(X, X^*; W_t)$$这里期望会比sutdent $$f(X; W_s)$$、或者LUPI $$f(X^*; W_t)$$的teahcer更强。在上述场景上，通过考虑上priviledged features和regular features，可以使用停留时长（dwell time）来区分在不同昂贵items上的偏好程度。teacher会有更多的知识来指导student，而非误导它。通过以下实验进行验证，添加regular features到teacher中是non-trivial的，它可以极大提升LUPI的效果。从那以后，我们将该技术表示成PFD来区别LUPI。

如等式(4)所示，teacher $$f(X, X^*; W_t)$$会优先训练。然而，在我们的应用中，单独训练teacher model会花费一个较长时间。使用像等式(4)这样的distillation是相当不实际的。更可信的方式是，像[1,38,39]的方式同步地训练teacher和student。objective function接着被修改如下：

$$

$$

...（5）

尽管会节省时间，同步训练可能不稳定（un-stable）。在early stage时，teacher模型没有被well-trained，distillation loss $$L_d$$可能会使student分心（distract），并减慢训练。这里我们通过一个warm up scheme来缓和它。在early stage时，我们将等式(5)的$$\lambda$$设置为0，从那以后将它固定到一个pre-defined value，其中swapping step可以是个超参数。在我们的大规模数据集上，我们发现，这种简单的scheme可以良好地运转。不同于相互学习（mutual learning），我们只允许student来从teacher那进行学习。否则，**teacher会与student相互适应，这会降低效果**。当根据teacher参数$$W_t$$分别计算gradient时，我们会触发distillation loss $$L_d$$。算法1使用SGD更新如下。

根据该工作，所有模型都会在parameter server系统上进行训练，其中，所有参数都会存储在servers上，大多数计算会在workers上执行。训练速度主要决取于在人orkers上的计算负载以及在workers和servers间的通信量。如等式(5)所示，我们会一起训练teacher和student。参数数目和计算会加倍。使用PFD进行训练可能会比在student上单独训练更慢，这在工业界是不实际的。特别是对于在线学习，会要求实时计算，采用distillation会增加预算。这里我们会通过共享在teacher和student的所有公共输入部分来缓和该问题。由于所有features的embeddings会占据在servers上的大多数存储，通过共享通信量可以减小一半。该计算可以通过共享用户点击/购买行为的处理部分来减小，它的开销较大。正如以下实验所验证的，我们可以通过sharing来达到更好的表现。另外，对比起单独训练student，我们只会增加一些额外的时间，对于online learning来说这会使得PFD更适应些（adoptable）。

**扩展：PFD+MD**

如图1所示，PFD会从priviledged features中distill知识。作为对比，MD会从更复杂的teacher model中distill知识。两个distillation技术是互补的。一个天然扩展是，将它们进行组合来构成一个更复杂的accurate teacher来指导student。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/f9fb4b5ff2b9ac858cd7373af569666da70667037fee34aceeb50e633be1a2d29a8f4e657638cebca091349594d2be62?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1

在粗排的CTR prediction中，如等式(3)所示，我们使用inner product模型来在serving上增加效率。事实上，inner product模型会被认为是泛化的MF（gnerelized matrix factorization）。尽管我们正使用非线性映射$$\Phi_W(\cdot)$$来转移user和item inputs，该模型能力天然受限于内积操作的bi-linear结构。DNNs，它可以逼近任意函数，被认为是对于在teacher中的inner product模型的一个替代。事实上，如【22】中的定义1所示，乘积操作可以通过一个two-layers的NN（在hidden layer上只有4个neurons）来逼近任意小。因此，使用DNN的表现被认为是inner-product模型的下界（lower-bounded）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/e19f36cbaacb2ff2b91195b8272b671412ff8d0dcbf2c43110c2fba49932c422cea5bd3b2a9532e14a8ada10d36674c8?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=4.jpg&amp;size=750">

图4

在PFD+MD中，我们也采用DNN模型作为teacher network。事实上，这里的teacher model与我们在精排CTR预测使用的模型相同。本任务中的PFD+MD可以被认为是从精排中distill知识，来提升粗排。为了更好地演示，我们在图4中给出了整个框架。在serving期间，我们会只抽取student部分，它依赖于priviledged features。由于所有items的mappings $$\phi_{W^i} (X^i) $$是与users相互独立的，我们会事先对它们进行离线计算。当一个请求过来时，user mapping $$\phi_{W^u}(X^u)$$会首先计算。这之后，我们会使用所有items的mappings（它们从candidate generation阶段生成）来计算inner-product。top-k得分最高的items接着被选中并被feed给精排。基本上，我们只要执行一个forward pass来获得user mapping，并在user和所有candidates间执行高效地inner product操作，它在计算方面相当友好。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/69034a0bc169479900de0b05e88ff69b9f403f949dde46d24e65a38fcee28547ad58d8b3d9c92e6a506c7d99638fbeed?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=5.jpg&amp;size=750">

图5

# 5.实验

在taobao推荐上做了实验，目标是回答以下的研究问题：

- RQ1: PFD在粗排的CTR任务上的表现，以及在精排CVR上的表现？
- RQ2: 对于独立的PFD，我们可以通过将PFD与MD进行组合来达到额外的提升？
- RQ3: PFD对于等式(5)中的超参数$$\lambda$$敏感吗？
- RQ4: 通过共享公共输入部件（），同时训练teacher和student的效果是什么？

## 5.1 实验setting

## 5.2 粗排CTR

## 5.3 精排CVR

## 5.4 RQ3-4

# 6.结论

略

# 参考


- 1.[https://arxiv.org/pdf/1907.05171.pdf](https://arxiv.org/pdf/1907.05171.pdf)