---
layout: post
title: 离线/在线评估差异
description: 
modified: 2021-11-05
tags: 
---


microsoft在《Predictive Model Performance: Offline and Online Evaluations》中提出在线评估与离线评估中 AUC的特性：


# 1.介绍

对于一个常见的机器学习问题，训练和评估（or test）样本会从模型所需要的population中随机选出，预估模型会基于训练样本构建。接着，学到的模型会被应用到evaluation data上，模型的qualities会使用选中的evaluation metrics进行measure。这被称为“offline evaluation”。

另外，高度复杂的现代应用，比如：google/bing的搜索引擎，经常会在一个controlled AB testing平台上对最好的离线模型进行开展在线评估（称为online evaluation）。

现实中模型评估的一个问题是：**在离线评估中模型效果的提升有时不会收到实际效果，或者有时在在线评估时获得相反结果**。不同于静态offline evaluation，在controlled环境下的online Testing是**高度动态**的，当然，在离线建模期间都没有考虑的许多因素会对结果有重要影响。然而，这些observations会抛出一个问题：是否存在会导致这样差异在离线评估指标上的基础biases或限制。

另一个问题是：**使用不同类型的数据构建的模型所进行的效果对比，特别是那些带有稀有事件（rare events）数据**。稀有事件（rare events）会比其它事件以更低频率出现，从而导致在classes间的倾斜样本分布。在真实世界问题中，这是个相当常见的现象。**（rare events）的样本包含了在web search结果链接上的点击、展示广告的点击、在产品广告上的购买**。之前的研究已经表明，一些metrics可能会过度估计对于倾斜样本的模型效果。该observations会导致以下的问题：

- 有了该bias，**我们如何解释和对比应用不同类型数据的模型效果**？
- 例如，当我们构建对于文本广告和展示广告的预估模型时，我们可以**使用离线指标作为可对比度量（comparative measures）来预估它的真实效果吗**？
- **假设我们知道一个模型的真实效果，并且我们获得了另一个模型（the other model）相当的离线指标（offline metrics）。我们是否就可以估计该另一个模型（the other model）的真实效果呢**？如果不能，我们应使用哪种metrics进行替代？

我们提出一种新的模型评估范式：仿真指标（simulated metrics）。对于在线行为的离线仿真，我们实现了 auction simulation，并使用simulated metrics来估计该点击预估模型的在线模型效果。由于simulated metrics被设计用于模拟在线行为，我们期望更少遭受效果差异问题。另外，由于simulated metrics直接估计像user CTR等在线指标，他们可以被直接对比，即使模型基于不同数据进行构建。

# 4.评估指标

我们集中关注点击预估问题中的metrics。一个点击预估模型会估计：给定query下，广告的**位置无偏(position-unbiased) CTR**。我们将它看成是一个二分类问题。

我们将NDCG排除在外，是因为它更偏向于ranking算法，在eariler ranks位置放置更多相关结果。如第2.1节所述，在搜索广告领域，排序（ranks）不光由pClick scores（例如：预估点击）决定，也会由rank scores影响。因此，**使用NDCG的rank顺序来对pClick的效果进行measure是不合适的**。

我们也会在review中排除Precision-Recall（PR）分析，因为在PR曲线和ROC曲线间有一个连接，从而在PR曲线和AUC间会有一个连接【9】。Davis等人展示了：当且仅当在PR空间内主导的曲线，会在ROC空间中主导。

## 4.1 AUC

考虑一个二元分类器，它会产生：

- p：表示一个事件发生的概率
- 1-p：表示事件不会发生的概率

p和1-p表示：每个case是两种事件其中之一。为了预估所属class，阈值是必要的。AUC（Area under the ROC (Receiver Operating Characteristic) 曲线），提供了一个在阈值所有可能范围间的判别式衡量（discriminative measure）.

在一个混淆矩阵中，4个不同部分的概率对比：

- 真阳率- true positive rate (TPR) ：也叫做sensitivity
- 真阴率- true negative rate (TNR) ：也叫做specificity
- 假阳率- false positive rate (FPR) ：也叫做 commission
- 假阴率- false negative rate (FNR) ：也叫做 omission errors

从混淆矩阵中派生出的这4个scores和其它关于accuracy的measures，比如：precision, recall, or accuracy 都依赖于threshold。

ROC曲线是一个关于sensitivity (or TPR)的一个图形描述，是一个关于二分类的FPR的函数，随threshold变化。AUC计算如下：

- 按模型预估分的降序进行sort
- 为每个预估值计算真阳率（TPR）和假阳率（FPR）
- 绘制ROC曲线
- 使用**梯形近似（trapezoid approximation）**来计算AUC

经验上，AUC是一个关于任意scoring model的预估能力的可靠指标。对于付费搜索，AUC，特别是只在主线广告上measure的AUC，是关于模型预估能力的最可靠指标。**一个好的模型（AUC>0.8），如果AUC能提升1个点（0.01），通常具有统计显著提升（statistically significant improvement）**。

预估模型使用AUC的好处包括：

- AUC提供了一个：在所有可能阈值范围上，对整个模型效果的单值判别式得分归纳。这可以避免在阈值选择中的主观因素
- 可以应用到任意scoring function的预估模型上
- AUC得分介于[0, 1]之间，得分0.5表示随机预估，1表示完美预估
- AUC可以同时被用于预估模型的offline和online监控中

## 4.2 RIG

RIG (Relative Information Gain：相对信息增益)是一个关于log-loss的线性转换：

$$
RIG = 1 - \frac{log loss}{Entropy(\gamma)} \\
    = 1 - \frac{-c \cdot log(p) - (1-c) log(1-p)}{-\gamma \cdot log(\gamma) - (1-\gamma) log(1 - \gamma)}
$$

其中：

- c和p分别表示observed click和pClick。
- $$\gamma$$表示**评估数据的CTR**

**Log-loss表示click的期望概率（expected probability）**。最小化log-loss意味着pClick应收敛到expected click rate上，RIG score会增加。

## 4.3 MSE

MSE (Mean Squared Error)会对average squared loss进行measure：

$$
MSE(P) = \frac{\sum\limits_{i=1}^n (c_i \cdot (1 - p_i)^2 + (1 - c_i) \cdot p_i^2)}{n}
$$

其中：

- $$p_i$$和$$c_i$$分别样本i是pClick和observed click

NMSE（Normalized MSE）是**由CTR, $$\gamma$$归一化的MSE**：

$$
NMSE(P) = \frac{MSE(P)}{\gamma \cdot (1-\gamma)}
$$

## 4.4 MAE

Mean Absolute Error (MAE) 由以下公式给出：

$$
MAE(P) = \frac{1}{n} \sum\limits_{i=1}^n e_i
$$

其中：$$e_i = p_i - c_i $$是一个 absolute error.

MAE会权衡在prediction和observation间的distance，同时忽略掉到关键operating points的距离。MAE常用于measure在时序分析中的forecast error。

经验上，对于付费搜索（sponsored search），预估pClick模型的功率来说，该指标具有一个好的效果。它与AUC一起，是最可靠的指标之一。

## 4.5 Prediction Error

Prediction Error（PE）会measure由CTR归一化的平均pClick：

$$
PE(P) = \frac{avg(p)}{\gamma} - 1
$$

当平均pClick score准确估计CTR时，PE会变0。另一方面，当pClick scores相当不准时（有可能欠估计、过估计的混合，平均值与underlying CTR相似），PE可能接近0。**这使得prediction error相当不稳定，它不能被用来可靠估计分类accuracy**。

## 4.6  Simulated Metric

尽管在controlled AB testing环境下的在线实验会提供关于用户engagement方面的模型的真实效果对比指标，AB testing环境是通过一些固定参数值集合预设定的，因而，在testing环境上的模型效果指标只对应于operating points的给定集合。在多个operating points集合上开展实验，是不实际的，因为在线实验不仅耗时，而且如果新模型效果欠佳，对于用户体验和收益都很昂贵。

作为在线评估的替代，在整个可行operating points的范围（span）上，**一个模型的效果可以使用历史在线用户engagement data进行仿真。Kumar et.为federated search 开发了一种在线效果仿真方法[20]**。

Auction simulation，首先：**为给定query会离线复现ad auctions，并基于新的模型预估分、以及多个operating points集合选择一个ads集合**。

我们使用付费搜索（sponsored search）点击日志数据来实现auction simulation，并生成多个simulated metrics。Auction simulation，首先，为给定query离线复现ad auctions，并基于新模型预估分选择ads的一个集合。在仿真期间，会使用在日志中提供的(query, ad) pair的历史用户点击来预估用户点击：

- **如果(query, ad) pair在日志中被发现，但仿真的ad-position与在日志中的posiiton不同，expected CTR会通过position-biased histric CTR（或click曲线）被校准（calibirated）**。一般的，对于相同的(query, ad) pair，主线广告（mainline ads）会比sidebar ads获得更高的大CTR，在相同ad block内，对于相同的(query, ad) pair，在更高位置的广告会获得更高的CTR。
- **如果predicted(query, ad) pair不会出现在historic logs中，会使用在ad-position上的平均CTR（也被称为：reference CTR）**。

Click曲线和reference CTR来源于自在搜索广告日志中的historic user responses。

经验上，对于operating points的给定集合，auction simulation会生成高度准确的ads集合，它们会被新模型选中。 Simulated metric通常结果是在线模型效果的最强离线估计之一。

# 5.METRICS在真实世界中的问题

在本节中，我们分析了行为、限制以及缺点。注意，我们不打算建议：由于限制和缺陷，这些指标被一起排除。我们会宁可建议metrics被小心应用和说明，特别是那些会产生误导估计的metrics。

## 5.1 AUC

AUC是一个可以评估预估模型效果的相当可靠的方法，它在样本数据的特定条件下仍会有缺点。该假设是：AUC是一个需要被重新检查模型效果的充分测试指标。

首先，它忽略了预估的概率值（predicted probability values）。这使得它对于**预估概率的保序变换**不敏感。

- 一方面，这也是个优点，它使得在不同的measurement scales上生成的数值结果是可对比测试的。
- 另一方面，对于两个tests来说，生成具有相似AUC scores、并且非常不同的预估输出是相当可能的。它可能是一个较差拟合模型（poorly fitted model）（对所有预估过拟合、或者欠拟合），具有一个良好的判别能力；而一个良好拟合模型（well-fitted model），如果出现概率略微高于不出现概率时，会具有较差的判别。

表2中的示例展示了：一个poorly-fitted model，它使用大量负样本，具有非常低的pClick scores，从而有更低的CTR，反而具有较高AUC score。**在相对更高的pClick scores范围内，它会影响FPR的降低，从而提高AUC score**。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/522432bbf074f32440d9a0819586780fcc983f5f3b3dcb2c41daad3daa97b410d176a81d0c3998033b982e16b955b6a3?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t2.jpg&amp;size=750">

表2 AUC反常示例1：一个poorly fitted model，具有更高AUC，现在大量负样本集中在pClick score范围的低端(图中的第一张图展示了：better1-fitted模型)

第二，在整个ROC空间的spectrum上（包括很少操作的区域），它会总结测试效果。例如，对于付费搜索，在mainline上放置一个ad会显著影响CTR。不管ad在mainline上被展示、还是不被展示，predicted CTR如何拟合actual CTR并不是个大问题。换句话说，ROC的极左和极右通常很少用。Baker and Pinsky提出了**partial ROC曲线**作为整个ROC曲线的一个替代选择。

已经观察到，**更高的AUC并不必然意味着更好的rankings**。如表3所示，在FPR尾部上，样本分布中的变化会大量影响AUC score。然而，在模型CTR效果上的影响可能是相同的，特别是在threshold的实际operating points上。由于AUC不会判别ROC空间的多个区域，通过最优化在数据的任意一端的模型效果，一个模型可能会被训练用来最大化AUC score。这会导致在实际在线流量上，低于期望效果增益。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/188b31ec3d689ae8aa4f58049a71b568023cb5d55d00a0ae81d823885bf62a563600f9fb5eb6c869a6e5c74e73d3c218?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t3.jpg&amp;size=750">

表3 AUC反常示例2: 在FPR两端上，样本分布的变化会非常影响AUC score，尽管实际效果提升与在实际操作上很相似

第三，它会等价权衡omission和omission errors。例如，在付费搜索中，在mainline中没有放置最优ads的惩罚（penalty）（omission error）远远超过放置一个次优ads的惩罚（penalty）。当误分类代价不等时，对所有可等阈值进行汇总是有瑕疵的。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/a2a2047350986463780786775abdf5a1529d68c664d2ac58ed3429b67e1c5786557b1ca48739b6557fbb7eca8e617101?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=t4.jpg&amp;size=750">

表4 AUC反常示例3: 一个poorly fitted model与well-fitted model具有相似的AUC

最后，AUC高度依赖于数据底层分布。AUC会对两个具有不同负样本率的数据集进行计算。如表4所示，一个具有较低内在CTR的poorly-fitted模型，会与一个well-fitted模型具有相同的AUC。这意味着，**一个使用更高负样本率训练的模型，具有较高AUC score时，并不必然意味着模型具有更好的预估效果**。图1绘制了关于付费搜索和contextual ads的pClick模型的ROC曲线。如图所示，contextual ads的AUC score要比付费搜索的AUC高3%，尽管前者会更不准：付费搜索为$$\frac{avg \  pClick}{actual \ CTR} = 1.02$$，而contextual ads为0.86。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/9de05fba7dca7f1a4e71ba3edd28264ba76c8495f98f5d6470289a47ef270d4879658cb59324ebc9ce08566acd1b930f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1

## 5.2 RIG

类似于AUC，RIG的一个问题是，它对评估数据的底层分布是高度敏感的。由于评估数据的RIG score的范围会随着数据分布非常不同，我们不能仅评RIG scores来决定一个预估模型有多好。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/161851b7a05a60c1666d9627bcf1581389e3b51ceea1d58c409e9c98c4859fa68c202bd7bb920ff779b44b5c6f3c2911?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2

图2展示了RIG（实线：solid curve）和PE（虚线：doted curve）随着一个关注的CTR区间的变化情况。我们观察到，RIG scores会随着dataset中的CTR的增加而降低，即使是相同的预估模型。图2中的prediction error意味着，prediction score与true CTR有多接近。正如所期望的，click prediction error要比低的pClick score区间要高。

这种行为符合我们关于不同点击预估数据集的早期观察，它会随intrinsic CTR的level而不同。该observations建议实践时如下：

- 不应直接使用RIG scores的实际值 来对比两个预估模型效果，如果得分来自不同分布的多个数据集合。
- RIG scores可以被用于对比多个模型在同一个数据上进行test的相对效果
- 一个独立的RIG score，信息不足够去估计预估模型的效果，因为该score不仅取决于模型效果的质量，也会非常受数据分布的偏向性影响

# 6.离线和在线效果差异

实践中，关于离线评估指标，一个非常显著的问题是：离线和在线间的效果差异。**存在这样的情况：一个预估模型在离线指标上达到显著增益，在online testing环境部署时发现效果表现并不好**。

表5总结了一个点击预估模型在Bing付费搜索数据上构建的离线和在线指标，并在一个线上真实流量上进行online AB testing环境实验。Click yield（CY）是一个在线用户点击指标，它会measure每个搜索PV上广告的点击数。Mainline CY是在每次搜索PV下mainline ads的点击数。新模型vs baseline模型，会在在线环境中在user clicks上显著下降，即使在离线评估数据上AUC和RIG会有显著收益。



表5

图3对比了：在感兴趣pClick scores范围内，每个分位下（quantile），两个点击预估模型的log-loss（baseline：model-1，test：model2）。Model-2会在较低pClick score范围的分位上大量过度估计（overestimates）pClick scores。图4绘制了相同数据的prediction error。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b1fba34bedfd41112794910c5a078f6cb2d961dd50d6471c201a1b07470c173beb19d68ddb2a6a49405b72844e98b838?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/f460a90f8d8d7abbb9f888945bb659806087a15d32a8b775b014a55170a8f921159666ece337ce371444655dcb181f56?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=4.jpg&amp;size=750">

图4

对比起低pClick score范围的过度估计，pClick scores的更高范围上于click概率做出过度估计对于在线效果影响较小，因为在高pClick score范围上的广告最可能被多个模型选中。一旦展示给用户，user clicks大多数由ad-position和ads的relevances决定，而非分配的pClick scores。

另一方面，在较低pClick范围上对pClick scores的过度估计，会对在线指标做出显著负影响；对比起base model，较低质量的ads有更高机率被选中。选中的较低质量的ads，由于过度估计pClick scores，会导致对user clicks的较低rate，从而伤害在线指标。

大多数离线指标包括：RIG和AUC，不能捕获这些行为，因为：指标会贯穿pClick scores的整个范围累积该影响。

## 6.1 Simulated Metrics

我们会通过第4.6节描述的 auction simulation来计算simulated metric。 simulated click metrics的实验结果会伴随着表6中归纳的offline和online指标。我们首先会训练一个新模型，并最优化参数设定，来通过基于历史日志数据的auction simulation提供最好的期望用户点击指标。具有最好表现的click metrics如表所示。



- 1.[https://chbrown.github.io/kdd-2013-usb/kdd/p1294.pdf](https://chbrown.github.io/kdd-2013-usb/kdd/p1294.pdf)