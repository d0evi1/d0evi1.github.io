---
layout: post
title: 2014 Facebook广告点击实践
description: 
modified: 2015-08-13
tags: [ ctr ]
---

{% include JB/setup %}

# 1.介绍

Fackbook提出的gbdt+LR来预测广告点击，本文简单重温下相应的paper。

在付费搜索广告领域(sponsored search advertising)，用户query被用于检索候选广告（candidate ads），这些广告可以显式或隐式与query相匹配。**在Facebook上，广告不与query相关联，但可以通过人口统计学(demographic)和兴趣（interest）来定向(targeting)**。因而，当一个用户访问Facebook时，适合展示的广告容量（the volume of ads），比付费搜索中的要大。

为了应付每个请求上非常大量的候选广告，当用户访问Facobook时触发广告请求，我们会首先构建一连串的分类器，它们会增加计算开销。在本文中主要放在点击预测模型上，它会对最终的候选广告产生预测。

# 2.实验Setup

取2013年第4季度，某一个周的数据。为了在不同条件下维护相同的训练／测试数据，我们准备了离线训练数据，它与线上观测到的数据相似。我们将这些保存下来的离线数据划分成：训练数据和测试数据，并使用它们来模拟在线训练和预测的streaming数据。

评估的metrics: 使用预测的accuracy，而非利润和回报。我们使用**归一化熵（NE: Normalized Entropy）**作为主要的评测指标。

NE，或者更准确地被称为NCE：**归一化的Cross-Entropy**，等价于每次曝光（impression）的平均log loss，再除以如果一模型为每个曝光预测了相应后台CTR(backgroud CTR)，所对应的每个曝光的log loss平均。换句话说，预测的log loss通过backgroud CTR的熵进行归一化。backgroud CTR是训练数据集的平均期望CTR(average empirical CTR)。它可能更多描述是指关于归一化log loss（Normalized Logarithmic Loss）。值越低，通过该模型预测的效果越好。使用该归一化的原因是，backgroud CTR越接近0或1,也容易达到一个更好的log loss。除以backgroud CTR的熵，使得NE对于backgroud CTR不敏感。假设一个给定的数据集，具有N个样本，对应的labels为: \$ yi \in {-1,+1} \$，点击概率pi，其中i=1,2,...,N。平均期望CTR为p：

$$
NE=\frac{-\frac{1}{N}\sum_{i=1}^{n}(\frac{1+y_i}{2}log(p_i)+\frac{1-y_i}{2}log(1-p_i))}{-(p*log(p)+(1-p)*log(1-p))}
$$

......（1)

NE是一个必需单元，用于计算相关的信息增益(RIG: Relative Information Gain). RIG=1-NE

Calibration(校准)是一个关于平均估计CTR和期望CTR的比值。该比值相当于：期望点击数与实际观察点击数。Calibration是一个非常重要的指标，因为关于CTR的准确（accurate）和well-calibrated预测，对于在线竞价和拍卖的成功很重要。Calibration离1越小，模型越好。

注意，对于评估ranking的质量（不考虑Calibration），AUC也是一个相当好的指标。真实环境下，我们希望预测够准，而非仅仅获取最优的rank顺序，以避免欠拟合（under-delivery）或过拟合（overdelivery）。NE可以权衡预测的好坏，并隐式地影响calibration。例如，如果一个模型过度预测2倍，我们可以使用一个全局的乘子0.5来修正calibration，对应的NE也会提升，即使AUC仍相同。详见paper[12]

# 3.预测模型结构

本部分我们提出了一种混合模型结构：将GBDT与一个概率稀疏的线性分类器相串联，如图1所示。在3.1中，决策树对于输入特征的转换十分强大，可以极大增加概率线性分类器的accuracy。在3.2中，会展示训练数据的新鲜度是如何影响更准确的预测。这激发了我们使用一个在线学习方法来训练线性分类器。在3.3中，我们比较一些在线学习变种。

<img src="http://pic.yupoo.com/wangdren23/GkT2sAkF/medium.jpg">

图1: 混合模型结构。输入特征通过boosted决策树的方式进行转换，每棵树的输出作为一个类别型输入特征，输入到一个稀疏的线性分类器上。Boosted决策树提供了非常强的特征转换.

我们评估的在线学习模式，是基于SGD算法的，应用于稀疏线性分类器。在特征转换后，给定一个ad的曝光，具有一个vector形式：\$ x=(e_{i1},...,e_{in}) \$，其中ei表示第i个unit vector，\$i_1,...,i_n\$是n种类别输入特征的值。在训练阶段，我们假设，给定一个二元标签 \$y \in {-1,+1}\$，用于表示点击or未点击。

给定一个加上label的广告曝光：(x,y)，我们将激活权重的线性组合表示成：

$$
s(y,x,w)=y*w^Tx=y\sum_{j=1}{n}w_{j,i_j}
$$ 

......(2)

其中w是线性点击分值的weight vector。

对于概率回归(BOPR)，目前最state-of-art的Bayesian在线学习scheme在[7]中有描述，似然和先验如下：

$$
p(y|x,w)=\Phi(\frac{s(y,x,w)}{\beta})
$$


$$
p(w)=\prod_{i=1}^{N}N(w_k;\mu_{k},\sigma_{k}^2)
$$

## 3.1 决策树特征转换

为了提升accuracy，有两种简单的方法，来对线性分类器的输入特征进行转换。对于连续型特征，用于学习非线性变换的一个简单的trick是，将特征二值化（bin），将将二值索引(bin index)看成是一个类别型特征。线性分类器可以有效地为该特征学到一个分段(piece-wise)常量的非线性映射。学到有用的二值边界很重要，有许多方法。

第二个简单但有效地转换包含在构建tuple型输入特征中。对于类别型特征，在笛卡尔积中采用的暴力搜索方法，比如，创建一个新的类别型特征，将它看成是原始特征的所有可能值。并不是所有的组合都有用，那些不管用的可以对其进行剪枝。如果输入特征是连续的，可以做联合二值化（joint binning），使用k-d tree。

我们发现boosted决策树很强大，非常便于实现非线性和上面这种tuple型转换。**我们将每棵树看成是一个类别型特征，它把值看成是将叶子的索引。对于这种类型的特征，我们使用1-of-K的编码**。例如，考虑图1中的boosted tree模型，有2棵子树，其中第1棵子树具有3个叶子，第2棵具有2个叶子。如果一个实例在第1棵子树的叶节点2上结束，在第2棵子树的叶节点1上结束，那么整体的输入到线性分类器上的二元向量为：[0,1,0,1,0]，其中前3个实体对应于第1棵子树的叶子，后2个对应于第2棵子树。我们使用的boosted决策树为：Gradient Boosting Machine(GBM)[5]，使用的是经典的**L2-TreeBoost算法**。在每个学习迭代中，会对前面树的残差进行建模创建一棵新树。**我们可以将基于变换的boosted决策树理解成一个监督型特征编码（supervised feature encoding），它将一个实数值向量(real-valued vector)转换成一个压缩的二值向量(binary-valued vector)**。从根节点到某一叶子节点的路径，表示在特定特征上的一个规则。**在该二值向量上，再fit一个线性分类器，可以本质上学到这些规则的权重**。Boosted决策树以batch方式进行训练。

我们开展实验来展示将tree features作为线性模型的效果。在该实验中，我们比较了两个logistic regression模型，一个使用tree feature转换，另一个使用普通的(未转换)特征。我们也加了一个单独的boosted决策树模型作为对比。所表1所示：

<img src="http://pic.yupoo.com/wangdren23/GkTbV98f/medium.jpg">

相对于没有树特征转换的模型，树特征变换可以帮助减小3.4%的NE。这是非常大的相对提升。作为参考，一个典型的特征工程实验，可减小20-30%左右的相对NE。LR和树模型以独立方式运行下的比较也很有意思（LR的预测accuracy更好一点点），但它们组合起来会有大幅提升。预测acuracy的增益是很大；作为参考，特征工程的好坏可以影响NE更多。

## 3.2 数据新鲜度

点击预测系统经常被部署在动态环境上，数据分布随时间变化。我们研究了训练数据的新鲜度对于预测效果的提升情况。我们在特定的某一天训练了一个模型，并在连续的数天对该模型进行测试。我们同时运行boosted决策树模型，以及一个使用树转换的LR模型。

在该实验中，我们训练了一天的数据，在后面的6天进行评估，计算每天的NE。结果如图2所示。

<img src="http://pic.yupoo.com/wangdren23/GkU2Pm3H/medish.jpg">

图2: 预测accuracy和时间关系

随着训练集与测试集间的时延的增长，预测accuracy明显下降。一周内两种模型的NE都下降大概1%左右。

该发现表明，需要以天为基础重新训练模型。一种选择是：使用一个周期天级任务，以batch方式重新训练模型。重新训练boosted决策树的时间各有不同，具体依赖于训练样本数，树的数目，每棵树的叶子，cpu，内存等。对于单核cpu，1亿左右的样本，有可能会超过24小时来构建一个上百棵树的boosting模型。实际情况中，训练过程可以在多核机器上，内存量足够存储整个训练集，通过足够的并发，使用几个小时来完成。下一部分，我们可以使用另一种方法。boosted决策树，可以每天或者每两天进行训练，但是线性分类器可以通过在线学习方式，接近实时进行训练。

## 3.3 在线线性分类器

为了最大化数据的新鲜度，一个选择是，当广告曝光数据到达标注后，直接在线训练线性分类器。在下面的第4部分，我们会描述一些基础，用来生成实时训练数据。在这部分，我们评估了许多方式，为基于SGD的LR在线学习设置不同的learning-rate。我们接着比较BOPR模型的在线学习最好变种。

在语术(6)中，我们可以有以下选择：

1.Per-coordinate学习率。该learning rate对于特征i在第t次迭代可设置成： 

$$
\eta_{t,i} = \frac{\alpha}{\beta + \sqrt{\sum_{j=1}^{t}\Delta_{j,i}^{2}}}
$$

α, β 是两个可调的参数。

2.Per-weight均方根学习率。

$$
\eta_{t,i}=\frac{\alpha}{\sqrt{n_{t,i}}}
$$

3.Per-weight学习率：

$$
\eta_{t,i}=\frac{\alpha}{\sqrt{n_{t,i}}}
$$

4.全局学习率：

$$
\eta_{t,i}=\frac{\alpha}{\sqrt{t}}
$$

5.常数学习率：

$$
\eta_{t,i}=\alpha
$$

前三种scheme可以为每个feature独立设置learning rate。后两种为所有feature使用相同的learning rate。所有可调的参数可以通过grid search进行优化。

对于连续学习（continuous learning），我们可以将learning rate设置为一个较低的边界0.00001. 我们使用上述的learning rate scheme来在相同的数据上训练和测试LR模型。相应的试验结果如图3所示。

<img src="http://pic.yupoo.com/wangdren23/GsPkuuea/medish.jpg">

图3: 不同learning rate schema的实验. X表示不同的learning rate。左侧为calibration，y轴右侧为NE.

从上面的结果可知，使用per-coordinate learning rate的SGD可以达到最好的预测accuracy，它可以降低5%的NE，比使用per weight learning rate要低（这种最差）。该结果的结论在[8]。per-weight square root学习率和常数学习率，几乎相同。另两种则比前面的都要差很多。global学习率失败的原因主要是因为训练实例的数目在每个特征上是不平衡的(imbalance)。由于每个训练实例可能包含不同的feature，一些流行的feature比起其它feature会具有更多的训练实例。在global学习率scheme下，具有更少训练实例的features的learning rate，会降得更快，以阻止最优weight（optimum weight）的收敛。尽管per-weight的learning rate的scheme也有相同的问题，它仍会失败，因为对于所有features，它会很快地减小learning rate。训练会过早终结，而模型会收敛到一个次优的点（sub-optimal point）。这解释了为什么该scheme在所有选择中具有最差的performance。

有意思的是，需要注意BOPR更新式(3)，意味着最接近LR SGD的per-coordinate learning rate版本。BOPR的有效学习率可以指定到每个coordinate上，取决于权重的后验variance，与每个单独的coordinate相关。

。。

# 4.Online data JOINER

前面的部分确定了训练数据中新鲜度会增加预测accuracy。同时也介绍了一个简单的模型架构，其中的线性分类器这一层是在线方式训练的。

本部分介绍一个实验系统，它生成实时训练数据，通过online learning来训练线性分类器。我们将该系统看成是"online joiner"，因为它的临界操作(critical operation)是将labels(click/no-click)和训练输入(ad impressions)以在线方式join起来。相同的基础设施可以用于流式学习（stream learning），例如Google Advertising System[1]。该online joiner会输出一个实时的训练数据流到一个称为“Scribe”的基础设施上[10]。而positive labels(clicks)是定义良好的，它没有提供给用户"no click"按钮。出于该原因，如果用户看到该广告后，在一个确定的，足够长的时间周期上没有点击该广告，那么这次曝光(impression)可以认为是具有一个negative的no-click label。等待的时间窗口需要小心调节。

使用太长的时间窗口(time window)，会延迟实时训练数据，并增加内存分配开销来缓存要等待点击信号的impressions。一个过短时间的时间窗口，会造成一些点击丢失，因为相应的impression可以会被冲掉，并当成no-clicked label来对待。这种负面影响称为"点击覆盖（click coverage）"，它是一个关于所有成功的clicks与impressions相join的一个分数。作为结果，online joiner系统必须在最新性（recency）和点击覆盖(click coverage)间做出一个平衡。

<img src="http://pic.yupoo.com/wangdren23/GsPm02kk/medish.jpg">

[图4]: Online Learning Data/Model Flows

不具有完整的点击覆盖，意味着实时训练集是有偏差的(bias)：经验CTR(empirical CTR)会比ground truth要低。这是因为，一部分被标注为未点击（no-clicked）的曝光（impressions），如果等待时间足够长，会有可能会被标注成点击数据。实际上，我们会发现，在内存可控范围内，随着等待窗口的size的变大，很容易减小bias到小数范围内。另外，这种小的bias是可衡量和可纠正的。关于window size的研究可以见[6]。online joiner被设计成执行一个在ad impressions和ad clicks之间的分布式stream-to-stream join，使用一个请求ID(request ID)作为join的主键。每一时刻当一个用户在Facebook上执行一个动作时，都会生成一个请求ID，会触发新鲜的内容曝光给他们。图4展示了online joiner和online learning的数据和模型流。当用户访问Facebook时，会生成初始的数据流，会发起一个请求给ranker对候选广告进行排序。广告会被传给用户设备，并行的，每个广告、以及和它相关的用于ranking的features，会并行地添加到曝光流（impression stream）中。如果用户选择点击该广告， 那么该点击(click)会被添加到点击流中（click stream）。为了完成stream-to-stream join，系统会使用一个HashQueue，它包含了一个先入先出的队列（FIFO queue）作为一个缓存窗口；以及一个hashmap用于快速随机访问label曝光。一个HashQueue通常在kv-pair上具有三种类型的操作：enqueue, dequeue和lookup。例如，为了enqueue一个item，我们添加该item到一个队列的前面，并在hashmap中创建一个key，对应的值指向队列中的item。只有在完整的join窗口超期后(expired)，会触发一个标注的曝光(labelled impression)给训练流。如果没有点击join，它会触发一个negative的标注样本。

在该实验设置中，训练器(trainer)会从训练流中进行持续学习，并周期性发布新模型给排序器(Ranker)。这对于机器学习来说，最终会形成一个紧的闭环，特征分布的变更，或者模型表现的变更都能被捕获，学到，并在后续得到修正。

一个重要的考虑是，当使用实时训练数据生成系统进行试验时，需要构建保护机制来处理在线学习系统崩溃时引发的异常。给一个简单示例。由于某些数据基础设施的原因，点击流（click stream）过期了，那么online joiner将产生这样的数据：它具有非常小或近乎0的经验CTR（empirical CTR）。作为结果，该实时训练器会开始以非常低、或近乎0的点击概率来进行错误的预测。一个广告的期望值会自然的决策于估计的点击概率，不正确预测成非常低的CTR的一个结果是，系统会降低广告曝光数。异常检测机制在这里就很有用。例如，如果实时训练数据分布突然发生变更，你可以从online joiner上自动断开online trainer。

# 5.内存与延迟

## 5.1  boosting trees的数目

模型中树越多，训练时间越长。这部分会研究树的数目在accuracy上的影响。

我们区分了树的数目，从1到2000棵，在一个完整天的数据上训练模型，并在下一天预测效果。我们限制了每棵树的叶子节点不超过12个。与之前的试验类似，我们使用NE作为评测的metric。试验结果如图5所示。NE会随着boosted trees的数目增加而增加。然而，通过添加树获得的增益，所获得的回归会减少。几乎所有的NE提升，都来自于前500棵树。最后的1000棵树减少的NE比0.1%还要少。另外，我们看到，NE对于子模型2(submodel 2), 会在1000棵树后开始倒退。这种现象的可能原因是overfitting。因为子模型2的训练数据比子模型0和1的要小。

<img src="http://pic.yupoo.com/wangdren23/GsPm092e/medish.jpg">

图5: boosting tree的数目的实验。不同系统对应着不同的子模型。x轴表示boosting trees的数目。Y轴表示NE.

## 5.2 Boosting feature importance

特征数（feature count）是另一个模型特性，它可以影响估计的accuracy和计算性能。为了更好理解特征数的影响，我们首先为每个特征应用一个特征重要性（feature importance）。

为了衡量一个特征的重要性，我们使用**statistic Boosting Feature Importance**，它可以捕获一个feature的累积loss衰减属性（cumulative loss reduction）。在每个树节点构建中，会选择最好的feature，并将它进行split，来达到最大化平方误差衰减（squared error reduction）。由于一个feature可以被用在多棵树中，每个feature的（Boosting Feature Importance）可以通过在所有树上对一个指定feature做总的reduction的求和。

通常，只有少量的特征对可解释性(explanatory power)有主要贡献，对可解释性，而其余的特征则只有少量贡献。我们可以看到，当绘制特征数 vs. 累积特征重要性时（如图6所示），这种现象。

<img src="http://pic.yupoo.com/wangdren23/GsPm0yiF/medish.jpg">

图6: Boosting feature importance。x轴表示特征数。在y轴左侧表示log scale中抽取了特征重要性，而y轴右侧表示累积特征重要性。

从上面的结果看来，top 10的特征占握着总的特征重要性的一半，而最后300个特征只贡献了少于1%的feature importance。基于该发现，我们进一步试验，只保留top 10,20,50,100,200个特征，并评估是如何影响表现的。试验的结果如图7所示。从图中可知，我们可以看到，当增加更多的特征时，NE的回报很少。

<img src="http://pic.yupoo.com/wangdren23/GsPm0TqE/medish.jpg">

图7: 对于top features，Boosting模型的结果. 我们在y轴的左边，抽取了calibration，而在右边的y轴对应于NE.

下面，我们将研究，历史特征(historical feature)和上下文特征（contextual feature）是否有用。由于数据敏感性，我们不能展示实际使用的特征细节。一些上下文特征举例：一天里的local time，week里的天等。历史特征可以是：在一个广告上的点击累计，等。

## 5.3 历史型特征

Boosting模型中使用的特征可以归类为两类：上下文型特征（contextual features）和历史型特征（ historical features）。上下文特征的值仅仅取决于取决于一个广告所处的上下文当前信息，比如，用户使用的设备，或者用户停留的当前页面。相反的，历史特征则取决于广告(ad)或用户(user)之前的交互，例如，该广告在上周的点击通过率（click through rate），或者该用户的平均点击通过率。

<img src="http://pic.yupoo.com/wangdren23/GsPm1IxG/medish.jpg">

图8: 历史型特征百分比的结果。X轴表示特征数。Y轴表示历史型特征在top-k个重要特征中的占比。

在这一部分，我们研究了该系的表现是如何受这两种特征影响的。首先，我们检查了两种类型的相对重要性。我们通过将所有特征按重要性进行排序，接着计算在top-k个重要的特征中，历史特征的占比。结果如图8所示，从该结果看，我们可以看到，比起上下文型特征，历史型特征更具可解释性。通过重要性排序的top 10个特征全部都是历史型特征。在top 20个特征间，只有2个上下文特征，尽管历史型特征在数据集上占据着75%的特征。为了更好地理解在aggregate中每种类型的比较值，我们训练了两种Boosting模型：一种只使用上下文型特征，另一种只使用历史型特征，接着将两个模型与带所有特征的完整模型相比较。结果如表4所示。

[表4]

从该表中，我们可以再次验证，历史型特征比上下文型特征更重要。如果只有上下文型特征，我们在预测accuracy上计算4.5%的loss。相反地，没有上下文型特征，我们在预测accuracy上有1%的loss。

需要注意到，上下文型特征对于处理冷启动问题很重要。对于新用户和新广告，上下文型特征对于合理的点击率预测是必不可少的。

在下一步，我们会评估在连续几周中，只使用历史型特征的训练模型，或者只使用上下文特征的训练模型，来测试在数据新鲜度上的特征依赖性。结果如图9所示。

<img src="http://pic.yupoo.com/wangdren23/GsPm2kGJ/medish.jpg">

图9: 不同类型特征对数据新鲜度的结果。X轴表示评估时期，Y表示NE.

从该图看到，比起历史型特征，使用上下文型特征的模型更依赖于数据新鲜度。这与我们的直觉相符合，因为历史型特征描述了长期累积的用户行为，它比起上下文型特征更稳定。

# 6.大规模训练数据

Facebook的一整天的广告曝光数据是海量的。注意，我们不会展示实际数量。但一天的一定比例的数据可以有几百万的实例。一个常用的技术来控制训练开销是，减少训练数据的量。在本节中，我们评估了两种方法用于下采样：均均子抽样（uniform subsampling）和负下采样(negative down sampling)。在每个case上，我们会训练一个使用600棵树的boosted tree模型的集合，然后使用calibration和NE进行评估。

## 6.1 均匀子抽样

训练行的均匀子抽样(uniform subsampling)是一种减少数据量的好方法，因为它可以很容易实现，生成的模型不需要修改，就可以在抽样的训练数据上以及未抽样的测试数据上直接使用。在本部门，我们会评估抽样率会指数式增加。对于每个在基础数据集上进行抽样的rate，我们分别训练了一个boosted tree模型。subsampling rate为{0.001, 0.01, 0.1, 0.5, 1}。

<img src="http://pic.yupoo.com/wangdren23/GsPm2rIg/medish.jpg">

图10: 数据量的实验结果。X轴表示训练实例数。y左表示calibration。y右表示NE.

数据量及结果如图10. 它与我们的直觉相一致，越多的数据会导致更好的表现。再者，数据量展示了预测accuracy的回报会越来越少。通过使用10%的数据，NE只有在表现上，相较于整个训练数据集只有1%的减少。在该sampling rate上calibration几乎没有减少。

## 6.2 负下采样

类别不均衡（class imbalance）问题，许多研究者都有研究，它会对学到的模型具有重大的影响。在该部分，我们探索了使用负下采样（Negative down sampling）来解决类别不均衡的问题。经验上，我们使用不同负下采样率，来测试学到模型的accuracy。相应的rate为：{0.1, 0.01, 0.001, 0.0001}。结果如图11所示。

<img src="http://pic.yupoo.com/wangdren23/GsPm2Chk/medish.jpg">

图11: 负下采样的实验结果。X轴对应于不同的负下采样率。y左表示calibration，y右表示NE.

从结果上看，我们可知，负下采样率在训练模型的表现上具有极大的提升。当负下采样率设置在0.025时具有最好的表现。

## 6.3 模型Re-Calibration

负下采样可以加速训练，提升模型表现。注意，如果在一个数据集上使用负下采样来训练一个模型，它可以校正(calibrate)在下采样空间上的预测。例如，如果平均CTR在采样之前只有0.1%，我们做一个0.01的负下采样，经验CTR（empirical）大约有10%的提升。对于真实的流量实验(hive traffic experiment)，我们需要重新校准模型，会获得0.1%的回报，\$ q=\frac{p}{p+(1-p)/w} \$。其中p是在下采样空间中的预测，w是负下采样率。

略.

参考：

1.[Practical Lessons from Predicting Clicks on Ads at
Facebook](https://pdfs.semanticscholar.org/daf9/ed5dc6c6bad5367d7fd8561527da30e9b8dd.pdf)
2.[kaggle:gbdt+libffm](https://github.com/guestwalk/kaggle-2014-criteo)

