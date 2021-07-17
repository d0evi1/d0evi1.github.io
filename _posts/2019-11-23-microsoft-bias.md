---
layout: post
title: microsoft position bias介绍
description: 
modified: 2019-11-23
tags: 
---

microsoft在《Modeling and Simultaneously Removing Bias via Adversarial Neural Networks》paper中提出了一种使用adversarial network解决position bias的方法：

# 介绍

在online广告系统中，需要对给定一个query估计一个ad的ctr，或PClick。通过结合该PClick估计以及一个广告主的bid，我们可以运行一个auction来决定在一个页面上的哪个位置放置ads。这些ad的曝光(impressions)以及他们相应的features被用来训练新的PClick模型。这里，**在线广告会存在feedback loop问题**，其中：之前看到过的ads会占据着training set，在页面上越高位置的ads会由大部分正样本(clicks)组成。**该bias使得在所有ads、queries以及positions（或D）上估计一个好的PClick很难，因为features的比例过高（overrepresentation）与高CTR占据feature space有关**。

我们假设：在一个page上的position（例如：mainline、sidebar或bottom）可以概括PClick bias的一大部分。实际上，我们的目标是学习这样的一个PClick表示：它对于一个ad展示的position是不变的（invariant），也就是说，**所有potential ads对于给定页面上的一个位置，仍然会是单一的相对排序**。尽管我们可以很容易地使用一个linear function来强制在position features，但其它features的intrinsic bias相对于position来说更难被移除。

为了学习这种位置不变特性（position invariant feature）的PClick模型，我们使用ANNs（adversarial neural networks）。ANNs会使用competing loss functions来进行建模，它在tandem（[6]）下进行最优化，最近的工作[1,9]有使用它们进隐藏或加密数据。我们的ANN representation包括了4个networks：Base、Prediction、Bias、以及Bypass Networks（图1）。最终在线使用的PClick prediction是来自Bypass和Prediction networks的outputs的一个线性组合的结果，它用来预测$$\hat{y}$$。然而，在训练这些predictors期间，会使用Bias network adversary(对手)进行竞争。该Bias network只使用由Base network生成的low rank vector $$Z_A$$，尝试对position做出预测。相应的，Prediction、Bypass和Base networks会最优化一个augmented loss function，它会对Bias network进行惩罚。结果是，在传给Prediction network之前，vector $$Z_A$$与position无关。

**克服position/display biases的另一个方法是：使用multi-armed bandit方法来帮助生成更少的无偏训练数据以及covariate shift**。然而，两者都需要来自一个exploration set中的大量样本来生成更好的估计。实际上，很难获得足够量的exploration data，因为它通常会极大影响收益。我们的ANN方法无需exploration，可以应用于已存在的dataset。

为了测试模型的有效性，我们会在真实数据集和伪造实验上进行评估。我们会生成两个伪造数据集集合，并模拟online ads系统的feedback loop，并展示systematic和user position biases可以通过ANN进行处理，并生成更精准的估计。

我们也展示了：当在CTR上进行最优化时，在模型中移除bias间的一个tradeoff。我们展示了：在评估时通过使用该tradeoff，ANN架构可以在无偏数据集上恢复一个更精准的估计。

我们的贡献如下：

- 一个新的ANN表示，用于移除在线广告的position bias
- 指定一个不同的squard covariance loss，在bias组件上进行对抗最优化（adversarial optimization）
- 引入一个bypass结构来独立和线性建模position
- 详细的人工数据生成评估，演示了在在线广告系统中的feedback问题

# 2.在付费搜索中的position bias

ML应用中的feedback问题是常见的。为了演示它，我们主要关注付费广告搜索中的CTR或PClick问题。一个标准的Ad selection stack包括了：

- 选择系统（selection system）：selection system会决定ads的一个子集来传给model
- 模型阶段（model phase）：model会尝试估计在分布D上的完全概率密度函数，它是整个Ads、Queries、Positions space。特别的，我们会估计$$P(Click \mid Ad, Queries, Positions\ space)$$ 。
- 竞价阶段（auction phase）[7]：在auction阶段，广告主会对关键词竞价（bid）并对queries进行匹配。Ads和他们的positions会最终由PClicks和广告主bids所选择。我们主要关注model phase或PClick model。

出于许多原因，很难估计D。首先，一个在线Ads系统会从D中抽样一个小量的有偏部分。机器学习模型会使用许多features跨<Ad,Query>上进行估计PClick。许多丰富的features是counting features，它会会跨<Ad,QUERY>的过往进行统计信息（比如：该Ad/Query组合在过去的点击百分比）。Query Ad pairs经常出现在Ad stack中，它们具有丰富的informative feature信息：然而，**从未见过或者很少见过的Query Ad pairs并没有这些丰富的信息。因而，对于一个模型来说，保证一个Query Ad pair在online之前没有展示过很难进行ranking，这会导致feedback loop**。

第二，一个feedback loop会在training data和PClick model间形成。新的training data或ads会由之前的model的ranking顺序展示，一个新的PClick model会从之前的training data中生成。因而，**产生的Feedback Loop(FL)会偏向于过往的模型和训练数据**。

Position CTR，$$P(y \mid Position = p)$$是一个ad在一个页面上的给定位置p的点击概率。**这可以通过对在给定位置中展示的ads的CTRs做平均计算得到。具有越高ranked positions的Ads一般也会生成更高的CTRs**。之前的工作尝试建模或解释为什么存在position bias【5】。在我们的setting中，我们假设：过往ads的$$P(y \mid Position = p)$$会总结在一个在线广告系统中存在的这些issues，因为具有越高Position CTR的ads，也越可能具有与high PClicks更强相关的特性。

在理想情况下，一个PClick模型只会使用来自D中的一个大量的**随机且均匀抽样数据（RUS数据：randomly and uniformly sampled data）**。一个在线Ads stack的核心目标是：广告收入（ad revenue）。实际上，不可能获得一个大的随机抽样数据集，因为online展示许多随机的Ads和queries pair在代价太高。

# 3.背景

## 3.1 在线广告

biased FL的训练数据问题，可以通过multi-armed bandits来缓解。multi-armed bandits问题的核心是：**寻找一个合理的Exploration & Exploitation的trade off**。

在付费搜索广告的context中，会拉取一个arm，它对应的会选择一个ad进行展示。

- Exploration实际上意味着**具有较低点击概率估计的ads**，会导致在short-term revenue上的一个潜在损失（potential loss）。
- Exploitation偏向于那些**具有最高概率估计的ads**，会产生一个立即的广告收入的增益。

Bandit算法已经成功出现在展示广告文献、以及其它新闻推荐等领域。由于简洁、表现好，Thompson sampling是一个流行的方法，它对应的会根据最优的概率抽取一个arm。

这些方法在该假设下工作良好，可以探索足够的广告。在在线机器学习系统中，medium-term和short-term revenue的损失是不可接受的。**我们可以获取exploration data的一个小抽样，但通常获得足够多的exploration data开销很大，它们受训练数据极大影响**。因此，大量biased FL data仍然会被用于训练一个模型，这个问题仍存在。

另一种解决该问题的方法是：回答一个反事实的问题（answering the conterfactual question）[2]。Bottou et al.展示了如何使用反事实方法。该方法不会直接尝试最优化从D上的抽样数据的效果，而是估计PCLick models在过往表现有何不同。作者开发了importance sampling技术来估计具有置信区间的关于兴趣的conterfactual expectations。

Covariate shift是另一个相关问题，假设：$$P(Y \mid X)$$对于训练和测试分布是相同的，其中：Y是labels，X是features。然而，p(X)会随着training到testing分布发生shifts或者changes。与counterfactual文献相似，它会对loss function进行rebalance，通过在每个实例上乘以$$w(x)=\frac{p_{test}(x)}{p_{train}(x)}$$，以便影响在test set上的变化。然而，决定w(x)何时test set上不会具有足够样本变得非常困难。在我们的setting中RUS dataset不足够大来表示整个分布D。

## 3.2 Adversarial Networks

对抗网络（Adversarial networks）在最近变得流行，特别是在GANs中作为generative models的一部分。在GANs中，目标是构建一个generative model，它可以通过同时对在一个generator和discriminator network间的两个loss functions进行最优化，从一些domain中可以创建真实示例。

Adversarial networks也会用于其它目的。[1]提出使用Adversarial networks来生成某些级别的加密数据。目标是隐藏来自一个adversary的信息，。。。略


# 4.方法描述

给定一个biased Feedback Loop的training set我们描述了一种ANN结构来生成精准的PClick预测 $$\hat{y}$$。我们假设一个连续值特征b，它会对该bias进行归纳。**我们将b定义成在Ads context中的position CTR或$$P(y \mid Position=p)$$。一个input features集合X通常与b弱相关**。

## 4.1 网络结构

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/1bf0e58a34816a829c08053486a6ebfad210fda6aa1d17c1cadb95e1013409d74bda632605652fbe0d550fc147575008?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1

ANN表示包括了如图1所示的4部分：Base network、Predction network、Bias network、以及一个Bypass Network，该networks的每一部分具有参数：$$\theta_A, \theta_Y, \theta_B, \theta_{BY}$$。

- 第一个组件，Base和Prediction networks，会独立地对b进行最优化；
- 第二个组件，Bypass network，只依赖于b。

通过以这种方式解耦模型，ANN可以从bias存在时的data进行学习。

Bypass结构会直接使用b，它通过包含它的最后的hidden state $$Z_{BY}$$作为在等式1的最后sigmoid prediction函数中的一个linear term。最终的hidden states的集合用于预测$$\hat{y}$$将包含一个来自Prediction和Bypass Network的线性组合。假设：

$$
\hat{y} = sigmoid(W_Y Z_Y + W_{BY} Z_{BY} + c)
$$

...(1)

其中：

- $$Z_y$$：表示在prediction network中最后的hidden activations
- $$W_Y$$：表示weights，用来乘以$$Z_Y$$
- $$W_{BY}$$：它的定义类似
- c：标准线性offset bias项

在b上的linear bypass strategy，当直接包含b时，允许ANN独立建模b，并保留在b上相对排序（relative ranking）

给定X，Base Network会输出一个hidden activations的集合，$$Z_A$$会将输入feed给Prediction和Bias networks，如图1所示。**$$Z_A$$用于预测y效果很好，而预测b很差**。

## 4.2 Loss functions

为了完成hidden activations的期望集合，我们会最小化两个competing loss函数：

- bias loss: $$Loss_B$$
- noisy loss：$$Loss_N$$

其定义分别如下：

bias loss:

$$
Loss_B(b, \hat{b}; \theta_B) = \sum\limits_{i=0}^n (b_i - \hat{b}_i)^2
$$

...(2)

noisy loss:

$$
Loss_N(y, \hat{y}, b, \hat{b}; \theta_A, \theta_{BY}, \theta_Y) = (1-\lambda) Loss_Y(y, \hat{y}) + \lambda \cdot Cov_B(b, \hat{b})^2
$$

...(3)

bias loss函数在等式2中定义。loss function会衡量在给定$$Z_A$$时Bias network是否可以较好预测b。在图1中，只有Bias network(orange)和$$\theta_B$$会分别根据该loss function进行最优化，从而保持所有其它参数为常数。

等式3描述了noisy loss function，它会在$$\theta_A, \theta_{BY}, \theta_Y$$上进行最优化，而保持$$\theta_B$$为常数。该loss包含了$$Loss_Y(y, \hat{y})$$来表示prediction loss，可以以多种方式定义。本工作中，我们使用binary cross entropy来定义$$Loss_Y$$。

$$
Loss_Y(y, \hat{y}) = \frac{1}{n} \sum\limits_{i=0}^n y_i log(\hat{y}_i) + (1-y_i) log(\hat{y}_i)
$$

...(4)

$$Cov_B(b, \hat{b})$$是一个sample covariance的函数，它通过在一个给定minibatch中计算跨b和$$\hat{b}$$均值来计算：

$$
Cov_B(b, \hat{b})^2 = (\frac{1}{n-1} \sum\limits_{i=0}^n (b_i - \bar{b})(\hat{b}_i - \bar{\hat{b}})^2
$$

...(5)

$$Cov_B(b, \hat{b})^2$$表示来自预测噪声的距离$$\hat{b}$$。squared covariance是0，当$$\hat{b}$$与b非正相关或负相关。当存在高相关性时，只要$$\lambda$$足够大，$$Loss_N$$会被高度惩罚。

产生的$$Loss_N$$的objective function会对两种差的预测对模型进行惩罚，X用于恢复b。$$\lambda$$控制着每个项与其它有多强相关。

## 4.3 学习

实际上，covariance function会跨每个mini-batch进行独立计算，其中会从每个minibatch进行计算。两个loss function $$Loss_N$$和$$Loss_B$$会通过在相同的minibatch上使用SGD进行交替最优化(alternatively optimized)。

## 4.4 在线inference

为了在一个online setting上预测，我们会忽略Bias network，并使用其它三个networks来生成predictions： $$\hat{y}$$ . **在在线广告系统中，对于在过去在线没有看到过的数据，我们将b设置为Position 1 CTR，接着它们会feed到Bypass Network中**。

# 5.系统级bias的人工评估

我们会生成人造数据来展示自然的feedback loop或者在在线广告stack中出现的system level bias。我们首先根据一个bernoulli（概率为：$$P(Y=1)=0.1$$）生成click labels，其中Y=1表示一个点击的ad。接着，feature vectors $$x_j$$会从两个不同但重合的正态分布上生成，根据：

$$
x_j = 
\begin{cases}
N(0, \sigma), & \text{if $Y=0$} \\
N(1, \sigma), & \text{if $Y=1$}
\end{cases}
$$

其中，我们设置$$\sigma=3$$。

这会构成一个完全分布D，会采用10w样本来构建一个大的Reservoir dataset。我们接着表示通过仿真一个迭代过程来表示feedback loop，其中之前的top ranked $$x_j$$（或ads）被用于训练数据来对下一个ads进行排序。图2和3展示了这个feedback loop过程，算法2演示了该仿真。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/8f8af8ae84f7ebd6c91cb65a4b2ca60acb66e31bc358b49e01c499c6e8a641cc8633a6d4bbdff3f9d9ce103f43baccfe?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/0dc16ed7198a52f351d85651be331bbc838a62e5205a7e8d9091556d75aa84178d099e8d3d2a8a25f6e9a79ae9d006c5?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=3.jpg&amp;size=750">

图3

根据第i-1天的Reservoir数据集中的10000个实例的$$\frac{K}{2}$$候选集合使用无放回抽样随机抽取，其中K=500.模型$$M_{i-1}$$会在第i-1天上训练，并在每个候选集合上对top 2 ads进行排序，在第i天展示给用户。labels会在第i天展示，它随后会形成对可提供的$$topk_i$$训练数据供下一次迭代。


我们会重复该过程，直到迭代了T=100轮. 每次迭代中，我们会记录对于每个top 2 positions的平均position CTR $$P(y \mid Position=p)$$。**p=1表示top ranked ads，p=2表示2nd top ranked ads。我们会将position CTRs看成是连续bias term b**。为了启动该过程，我们会从Reservoir中抽取K个实例来构成$$topk_0$$。在一个在线Ads系统中，多天的训练数据通常被用来减小systermatic bias。后续评估中，我们会利用最近两天的训练数据（例如：$$M_i$$只在$$topk_i$$和$$topk_{i-1}$$上训练）。每个模型$$M_i$$是一个logistic regression classifier，它具有l2正则。我们在算法2的13行上设置参数r=0，来展示一个系统级的feedback loop bias。我们构成了testing data，从该feedback loop过程中独立，或从D中抽取10w样本来进行HeldOut RUS评估。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/7b132bbc738e8e4fb5bf3df05e40d4bb203148b1b5b180a775d7a38cda74c797e4eaac027d2a41781b537b353620964a?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=t1.jpg&amp;size=750">

表1

在过去两天的每个position的CTR如表1所示，整个CTRs的计算在所有天上计算。所有的4 CTR values可能相等，因为他们每个都与250个训练样本相关。因此，一种naive方法是预测平均CTR values。这会构建对一个adversarial Bais Network如何来预测b的一个上界。我们会在表2中记录过去最近两天数据(4 values)的average CTR，并使用该值来计算MSE。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/eb23c60dce94535e4346e25be1a7c471da080b49cac51c814ee4807ed6b9ee4936ffc7b685eb92e0311edde840615a4d?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=t2.jpg&amp;size=750">

表2

# 5.1 setup

当在FL data上训练模型时，可以生成D或RUS HeldOut data。对于该FL data，我们它使用不同的$$\lambda$$来训练一个ANNs的集合，并将b设置为它的Position CTR。除了last layers外，所有hidden activations会使用hyperbolic tangent function。Prediction network的output activation function是一个sigmoid，并且Bias Network的output activation是线性的。Bypass network包含了具有1个node的1个hidden layer，而Base、Prediction、Bias networks每个都包含了带有10个nodes的1个ideen layer。我们会执行SGD，它具有minibatch size=100以及一个learning rate=0.01. 我们会在FL data上训练100个epochs。在这个主训练过程之后，我们允许Bias network在$$Loss_B$$上训练100个epochs。理想的，在给定从Base network生成的$$Z_A$$上，这会允许Bias network来预测b。

根据对比，我们会为一个带有$$\lambda=0$$的ANN执行相同的评估。该模型可以看成是一个完全独立的vanilla neural network，它在y上进行最优化，而一个独立的Bias network可以观察和最优化$$Loss_B$$，无需对Base Network进行变更。我们会对每个模型运行10个具有不同weight initialization的实验，并上报关于y的AUC以及在b上的MSEs。

## 5.2 主要结果

为了评估来自D的一个无偏样本，我们会使用position 1 CTR, 0.464, 它从最近一天$$topk_{T-1}$$得到。

表3展示了从D中抽取的HeldOut data的AUC和LogLoss，它会在该dataset上训练一个logistic regression模型。这是构成在AUC的一个上界的理想情况。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b6dbdc53610d9248deccd736b07f8dd0ac06215e969f1f298f59847e9624184f7085989bf2b05a62e962511ef42e1e34?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=t3.jpg&amp;size=750">

表3

除了使用Bypass network的ANN架构外，我们也展示了不带Bypass network的ANN的一个变种。图4展示了在FL和RUS datasets上AUCs和MSEs。x-aixs是一个reverse log scale，$$\lambda$$从0到0.99999.随着
$$\lambda$$的增大，MSE大多数会增加FL AUC error的开销。使用Bypass network的ANN的FL MSE error会下降到0.00078(如表2所示)，它与只有平均CTR的naive方法具有相同的表现。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/e81e12d9e24b781f4da51cf1dde33e69c653f1d1ffaa3d93fe4f3bef82829f408a27bf0af0ccd26a11493e72b2c94ad0?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=4.jpg&amp;size=750">

图4

我们注意到，随着$$\lambda$$达到1，在$$Loss_N$$中的$$Loss_Y$$项变得无足轻重，因此可以是$$\lambda$$的一个集合，可以在D的AUC项上进行最优化，它在图4c上可以看到。

ANN模型从$$\lambda=0.9999$$到$$\lambda=0$$在RUS set上的AUC上会有12.6%增益，而在RUS set上只训练一个模型10%的off。

## 5.3 Bypass vs. non Bypass的结果

图4中的结果展示了使用Bypass vs. non-Bypass ANN在AUC上的微小提升，以及在RUS dataset上具有更高MSEs。我们也分析了根据给定不同的position CTRs在bypass network预测中的不同。我们会在第$$day_{T-1}$$天feed Position 1 CTR作为input到Bypass network中，和features一起做预测，$$\hat{y}_1$$，并在$$day_{T-1}$$feed Position 2 CTR来创建$$\hat{y}_2$$。

我们会计算average prediction。图4e展示了这些结果。随着MSE的增加，会在预测上有所不同。因此，我们会假设：Bypass network可以快速解释在ANN表示中的position CTR。

# 6.在user level bias上的人工评估

另一个造成position bias的factor可能是一个user level bias。用户可能会偏向于不会点在Position 1下的ads，除非是相关和用户感兴趣。我们会模拟一个额外的User level Bias信息，通过在之前的人工评估中截取position 2 ranked ad的labels来进行。算法2的12-14行的，通过使用概率r来切换observed click label从1到0. 

。。。

# 参考

- 1.[https://arxiv.org/pdf/1804.06909.pdf](https://arxiv.org/pdf/1804.06909.pdf)