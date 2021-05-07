---
layout: post
title: 增量训练介绍
description: 
modified: 2020-08-30
tags: 
---

华为在《A Practical Incremental Method to Train Deep CTR Models》对许多大厂在用的增量训练方式做了些总结。

# 介绍

互联网用户会训练大量在线产品和服务，因此很难区分什么对它们更有兴趣。为了减小信息过载，并满足用户的多样性需求，个性化推荐系统扮演着重要的角色。精准的个性化推荐系统有利于包括publisher和platform在内的需求侧和供给侧。

CTR预测是为了估计一个用户在特定context上、在某个推荐item上的概率。它在个性化推荐系统中扮演着重要角色，特别是在app store的商业化以及在线广告上。现在deep learning方法获得了越来越多的吸引力，因为它在预测性能和自动化特征探索上的优越性。因此，许多工业界公司都会在它们的推荐系统上部署deep ctr模型，比如：google play的Wide&Deep、Huawei AppGallery的DeepFM/PIN，Taobao的DIN和DIEN等。

然而，每件事都有两面。为了达到良好的性能，Deep CTR模型具有复杂的结构，需要在大量训练数据上进行训练许多epochs，因此它们都会具有较低的训练效率。当模型不能及时生成时，这样低效的训练（很长训练时间）会导致效果下降。我们在Huawei AppGallery上进行app推荐时观察到，当模型停止更新时，这样的效果如图1所示。**举个例子：如果模型停止更新5天，模型效果在AUC上会下降0.66%，这会导致收益和用户体验的极大损失**。因此，如何提升Deep CTR模型的训练效率并且不伤害它的效果是在推荐系统中的一个必要问题。**分布式学习(Distributed learning)和增量学习( incremental
learning )**是两种常见范式来解决该问题。分布式学习需要额外的计算资源，需要将训练数据和模型分布到多个节点上来加速训练。在另一方面，增量学习会更改训练过程：从batch mode到increment mode，它会利用最近的数据来更新当前模型。然而，工业界推荐系统的大多数deep models是以batch模式进行训练的，它会使用一个fixed-size window的训练数据来迭代训练模型。在本工作中，我们主要关注incremental方法来训练deep CTR模型，它的目标是极大提升训练效率，并且不降低模型表现。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/cf052a3b652df34d8123ce0331fdea684616ee472b6b4bf87137726196ad63f444ffce41fb1f2ad96cb5e6636e0ea177?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1  当模型停止更新不同天时，模型效果下降的表现。x轴表示training set和test set间的不同gaps

然而，大多数incremental learning方法主要关注于图片识别领域，其中：新的任务或clesses会随时间被学习。而incremental learning方法在图片识别上面临着不同的状况，比如：刚来的new features等，因此，没有必要研究该话题。在本paper中，我们提出一个实用的incremental方法：IncCTR。如图2所示，三种解耦的模块被集成到我们的模型中：Data Module、Feature Module以及Model Module。Data Module会模拟一个水库（reservoir）的功能，从历史数据和incoming数据中构建训练数据。Feature module被设计成处理来自incoming data的新features，并初始化已经存在的features和new features。Model模块会部署知识蒸馏（knowledge distillation）来对模型参数进行fine-tune，并对来自之前模型的知识与来自incoming data的知识的学习进行balance。更特别的，对于teacher model我们会观察两种不同的选择。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/3c13de63c66efd6087f8bcedd94a460a6c0bdab1213dd599e6c2d9ee58264660435da5fc0abf20a5f672f3734324e2e5?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2

主要贡献如下：

- 强调了在推荐系统中通过离线模拟进行incremental learning的必要性。我们提出了一种实用的incremental方法：IncCTR来训练deep CTR模型。
- IncCTR包括了：data模块、feature模块、以及model模块，它们分别具有构建训练数据、处理new features以及fine-tuning模型参数的功能
- 我们会在公开数据集以及Huawei APP工业界私有数据集上进行大量实验。结果表明：对比batch模式的训练，IncCTR在训练效率上具有很大提升，可以达到较好的效果。另外，在IncCTR上每个模块的ablation study可以被执行。

paper的其余部分组织如下。在第2节，我们引入先决条件来更好理解方法和应用。我们会详述我们的增量学习框架IncCTR以及其三种单独模块。在第4节中，对比实验和消融学习的结果会用来验证框架的效果。最后，第5节下个结论。

# 2.先决条件

在该节中，我们引入了关于deep ctr模型的一些概念、基础知识。

## 2.1 Deep CTR模型

。。。

## 2.2 Batch模式 vs. Increment模式

在本节中，我们会表述和比较两种不同的训练模式：batch mode与increment mode。

### 2.2.1 使用Batch Mode进行训练

在batch mode下进行训练的模型会基于一个fixed-size time window的数据进行迭代式学习。当新数据到来时，time window会向前滑动。如图3所示，“model 0”会基于day 1到day 10的数据进行训练。接着，当新数据（"day 11"）到来时，一个新模型（称为“model 1”）会基于day 2到day 11的数据进行训练。相似的，“model 2”会基于day 3到day 12.

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b174caaee9dc06a94414bd7f8a827ef0098bd2a867712df169902443943971d4c3b5f6bd258e61a0c04c33ff399e54a3?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=3.jpg&amp;size=750">

图3

### 2.2.2 使用Incremental Mode进行训练

在incremental模式下，会基于**已经存在的模型**和**新数据**进行训练。如图3所示，“Model 1”会基于已经存在的模型“Model 0”（它会基于在day 1到day 10上的数据进行训练），以及day 11的数据进行训练。接着，"Model 1"转向已经存在的模型。接着，当day 12的数据到来时，"Model 2"会基于"Model 1"和第day 12的数据进行训练。

可以看到，当使用batch mode进行训练时，两个连续的time window的训练数据会在大部分有重合。例如，day 1到day 10的数据、day 2到day 11的数据的重合部分是day 2到day 10，其中：80%的部分是共享的。在这样的环境下，使用incremental mode来替代batch mode会极大提升效率，而这样的替换又能维持效果表现。

图3

# 3.方法论

我们的incremental learning框架IncCTR如图2所示。设计了三个模块：feature、model、data，会对历史数据（historical data）和新进来数据（incoming data）进行较好的trade-off。特别的，data module会被看成是一个蓄水池（reservoir），它可以基于历史数据和新进来数据进行构建训练数据。Feature模块会处理来自incoming data的new features，并会对已经存在的features和new features进行初始化。Model模块会使用knowledge distillation来对模型参数进行fine-tune。

## 3.1 Feature Module

在推荐和信息检索场景下，feature维度通常非常高，例如：数百万 or 数十亿。这样大数目的features的出现频次符合长尾分布，其中只有少量比例的features会出现很频繁，其余很少出现。如[10]观察到的，**在模型中的一半features在整个训练数据中只出现一次**。很少出现的features很难被学好。因此，当使用batch模式训练时，features必须被归类成“frequent”或"infrequent"，通过统计每个feature的出现次数来完成。更正式的，对于一个feature x，它的出现次数S[x]大于一个预定义的阈值THR（例如：$$S[x] > THR$$）即被认为是"frequent"，并且作为单个feature进行学习。其余"infrequent" features被当成是一个特殊的dummy feature：Others。在经过这样的处理后，每个feature通过某些策略（比如：auto-increment、hash-coding）会被映射到一个唯一id上等。出于简洁性，我们会采用一个auto-increment policy F。在batch模式下，policy F从头开始构建，它会给fixed size window的训练数据中的每个features分配唯一ids，其中unique ids会1-by-1的方式自增。

然而，使用incremental模式训练会带来额外问题，因为当新数据进来时，新features会出现。如图4所示，new data的每个块都会带来一个特定比例的new features。例如，从criteo数据集上观察到，对比起在该块前存在的features集合，new data的第一块会带来12%的new features，而第14个块仍会带来4%的new features。因此，当新数据进来时，policy F需要自增更新。可能的是，一个feature x，它之前被认为是Others，在new data进来后，如果它的出现次数S[x]大于THR阈值，会被认为是一个唯一的feature。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/088c2627849fe7110a6e86ee2868a832e470ae7b40def4dcc8c350a06af2df9dee131d4b16cfabbc8c19e953e3ad58d0?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=4.jpg&amp;size=750">

图4

在分配合适的ids给所有features后，IncCTR中的feature module会对已存在featurs和new features进行初始化。当我们以batch模式训练时，feature embedding $$\epsilon$$的所有值都会被随机初始化。然而，在incremental模式下，我们会对已存在的features $$\Epsilon_{exist}$$的embedding和new features $$\Epsilon_{new}$$的embeddings独立进行初始化。

feature模块（称为：new feature分配和feature embedding初始化）的功能如算法1所示。当new data进来时，我们首先更新每个feature（第3行）的出现次数，并继承已存在的feature分配策略（第4行）。如果来自new data的一个feature是新的大于该阈值（第6行），它会被添加到该policy中，id会自增1（第7行）。feature embeddings会独立初始化，依赖于一个feature是否为新。对于一个已经存在的feature，它会继承来自已存在模型的embedding作为它的初始化（行11）。这样的继承会将历史数据的知识转移到将自增训练的模型上。对于一个new feature，它的embedding会随机初始化，因为没有先验知识提供（行12）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/964db874dec99d2f217aec5cf76c16a1dd71ec848cb4973949d6f1ad5ffafa5390c97745863bd46d72b61bbc4d5b1d8e?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=alg1.jpg&amp;size=750">

算法1

## 3.2 Model模块

在本节中，我们引入在IncCTR中的model模块，它会对模型进行合适训练，这样：模型仍会“记住（remember）”来自历史数据的知识，同时也会从new data上做出一些“升级（progress）”。

**Fine-tune**

除了已存在features的embedding外，network参数也从已存在模型继承，作为warm-start。为了使用incremental模式对模型参数进行fine-tune，我们会使用一些auxiliary tricks来达到较好表现。例如，对于$$\Epsilon_{exist}$$我们会使用一个比$$\Epsilon_{new}$$更低的learning rate。fine-tune的训练细节在算法2的第19-第25行进行表示。该模型会通过在prediction和groundtruth间进行最小化cross entropy来进行最优化。我们会对该模型训练一定数目的epochs，其中：经验上我们设置数目epoch为1（第25行）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/fabd8d00538c331ecd49de9455c74756da2b2a50ee125f186b61ef2318236099ca9b4f66d5d990b58c4da3d8d8e8df15?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=alg2.jpg&amp;size=750">

算法2

**知识蒸馏（Knowledge distillation）**

除了以上的"fine-tune"外，我们引入了knowledge distillation（KD）方法来增强来自历史数据学到的知识（名义上，为了避免灾难性忘却（catastrophic forgetting））。Hinton[6] 出于高效部署，使用KD来将一个**模型ensemble**将knowledge转移到**单个模型**上，其中KD loss会被用来保留来自较重模型（cumbersome model）的知识，通过鼓励distilled model的outputs来逼近cumbersome model。相似的，LwF[7]的作者执行KD来学习新任务，并且能保持在老任务上的知识。借用相似的思想，KD可以被用在incremental learning场景中来学习来自incoming data的新知识，并保留在historical data上的记忆。

当我们在IncCTR上使用KD时，我们提供了一些关于设计teacher model的参考。有两种选项。

- **KD-batch**：使用batch模式训练的过期模型（outdated model），可以做为teacher model的一个天然选择来对incremental model进行distill，它会保持在一个fixed-size window上的历史数据的效果。我们使用batch模式并使用这样的teacher进行训练的KD方法称为“KD-batch”。
- **KD-self**：由于batch模式训练一个模型作为teacher model，需要额外的计算资源，执行之前的incremental model作为teacher要更方便。在这种情况下，后继的incremental model会在之前incremental model的监督下进行训练。我们将这样的设计称为“KD-self”。相似的思想，在BANs中有使用，其中一个后继的student model会随机初始化，通过前一teacher model进行教导，多个student后代的ensemble是为了达到可期的表现。在图片识别领域上，所有模型会以batch模式进行训练，这与我们的框架非常不同。

当执行KD时，我们会利用soft targets $$Y_{soft}$$，它由teacher model在incoming data上生成。objective function如下进行公式化：

$$
L = L_{CE}(Y, \hat{Y}) + L_{KD} (\hat{Y}, Y_{soft}) + R \\
L_{KD}(Y, Y_{soft}) = L_{CE} ( \sigma(\frac{Z}{\tau}), \sigma(\frac{Z_{soft}}{\tau})) \\
L_{CE}(Y, \hat{Y}) = \sum\limits_{y_i \in Y} L_{CE}(y_i, \hat{y}_i)
$$

...(2) (3) (4)

新的objective function会组合标准的二元cross-entropy $$L_{CE}(\cdot)$$（其中：$$Y$$和$$\hat{Y}$$分别表示groundtruth和outputs），KD loss $$L_{KD}(\cdot)$$。KD loss $$L_{KD}(\cdot)$$是在$$\hat{Y}$$和$$Y_{soft}$$间的cross entropy（其中：$$Y_{soft}$$是teacher model的prediction），它基于logits Z和$$Z_{soft}$$。变量$$\tau$$被用来获得soft targets，R是正则项。等式(2)中的loss function的目的是，distilled model的知识应对于new data来说是精准的（第一项），而它与teacher model的知识不应该有大的不同（第二项）。

KD-batch和KD-self的训练细节在算法2中的第3行到第5行，第11行到第17行有描述。KD-batch和KD-slef间的差异是，teacher模型$$Teacher_t$$如何被训练。记住，在KD-batch中的teacher model是一个使用过期模型，而在KD-self中的teacher model是前一个incremental model。我们会在实验部分对它们的效果进行对比。给定input data的features，incremental模型$$M_t$$和teacher模型$$Teacher_t$$会做出预测，如第4和第13行。接着，incremental模型$$M_t$$通过最小化等式(2)的loss function进行最优化，如第14行。当模型训练至少一个epoch时训练过程会终止，KD loss会停止减少，第17行。

## 3.3 Data模块

从数据的角色，对于灾难性忘却问题（catastrophic forgetting）一种简单的解决方法是，不只基于new data来训练incremental模型，同时也基于一些选中的historical data。我们计划实现一个data reservoir来提供合适的训练数据给incremental training。在已存在reservoir中的一些比例的数据，和new data会互相交错构成new reservoir。在该模型中，一些问题需要进行确认，比如：在已存在的reservoir中需要保留多少比例的数据。data模块的实验不是为现在完成的，它是要完成框架的将来工作的一部分。

# 4.实验

在这部分，我们会在一个公开bechmark和私有数据集上进行实验，目标是回答以下问题：

- RQ1: IncCTR的效果 vs. batch模式的效果？
- RQ2: 在IncCTR框架中不同模块的贡献？
- RQ3: IncCTR的效率 vs. batch模式？

## 4.1 Dataset

为了评估在IncCTR框加中提出的效果和效率，我们在公开benchmark和私有数据集上做了实验。

- Criteo。该数据集用于CTR预估的benchmark算法。它包含了24天的连续流量日志，包括26个类别型features以及13个numerical features，第一行作为label，表示是否该ad被点击或未被点击
- HuaweiAPP。为了演示提出方法的效果，我们在商业数据集上做了离线实验。HuaweiAPP包含60个连续天的点击日志，包含了app features、匿名的user features和context features。

为了减小复制实验结果的目的，我们在criteo数据集上做了数据处理的细节。根据kaggle比赛，涉及datasampling、discretization以及feature filtering。出于商业原因，我们没有细出处理huaweiAPP的处理细节，但过程基本相似。

- Data sampling：考虑数据的imbalance（只有3%的样本是正），与[12]相似，我们将负样本做down sampling，将正样本比例接近50%
- 离散化：类别型和数值形features都存在在Criteo数据集中。然而，两种类型的features的分布本质是相当不同的[11]。在大多数推荐模型中，numerical features通过buckeing或logarithm被转换成categorical features。根据上述方式，我们使用logarithm作为离散方法：

$$
v \leftarrow floor(log(v)^2)
$$

...(5)

- Featrue filtering：不频繁的features通常带的信息少，可能是噪声，因此模型很难学到这样的features。因此，根据[11]，在一个特定field中的features出现次数少于20次会被设置成一个dummy feature：others。

## 4.2 评估

**Evaluation指标**：AUC和logloss（cross-entropy）. AUC和logloss的提升在0.1%才被认为是一个ctr预估模型显著提升[1, 4, 15]。

**baseline**。使用batch模式训练的模型被用于baseline，来验证IncCTR的效果和效率。为了进一步评估模型的更新延迟（delay updating）上的影响，我们考虑使用不同的delay days的baseline。更特别的，$$Batch_i(i=0,1,2,3,4,5)$$表示baseline model具有i天的延迟。

**实验细节**：为了专注于deep ctr的效果和效率，当使用batch模式和IncCTR增量方式训练时，我们选择流行的deep CTR模型DCN来进行对比。

为了模似在工业界场景的训练过程，实验会在连续几天上做出。当使用batch模式进行训练时，所有数据都会使用在fixed size window内（例如：在size-w window中的数据[s, s+w), 其中$$s \in [0, T-w]$$）。当使用增量模式训练时，只有具有新来天的（coming day）数据（例如：在window [s, s+1)中的size-1，其中$$s \in [w, T-1]$$）提供。对于增强模型，在第一个incremental step之前，会使用batch模式训练的模型进行warm-start。也就是说，我们首先训练一个使用batch模式在[0,w)上的warm-started模型，接着会在第w天的数据上训练首个增量模型。对于criteo数据集，我们设置w=7，T=23；对于HuaweiAPP我们设置w=30，T=59作为HuaweiAPP dataset。

## 4.3 RQ1: 整体表现

## 4.4 RQ2: Ablation Studies

## 4.5 RQ3: 效率

...


# 参考


- 1.[https://arxiv.org/pdf/2009.02147.pdf](https://arxiv.org/pdf/2009.02147.pdf)