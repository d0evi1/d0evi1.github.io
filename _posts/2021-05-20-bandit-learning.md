---
layout: post
title: bandit learning介绍
description: 
modified: 2021-05-20
tags: 
---


# 摘要

隐式反馈（比如：用户点击）尽管在在线信息服务中非常丰富，但在用户评估系统输出方面并没有提供实质帮助。如果没有合理地进行建模，会误导模型估计，特别是在bandit learning setting中（feedback会即时获取）。在本工作中，我们会执行使用implicit feedback的contextual bandit learning，它会建模feedback作为用户结果查看（result examination）、以及相关评估（relavance evaluation）的一部分。**由于用户的查看行为是观察不到的，我们会引入隐变量（latent variables）来建模它**。我们**会在变分贝叶斯推断（variational Bayesian inference）之上执行Thompson sampling来进行arm selection以及模型更新**。我们算法的upper regert bound analysis表明了在一个bandit setting中从implicit feedback中学习的可行性。


# 1.介绍

contextual bandit算法为现代信息服务系统提供了一种有效解决方案，可以自适应地发现items和users间的良好匹配。这类算法会使用关于user和item的side information顺序选择items提供服务给用户，并能**基于用户立即反馈（immediate user feedback）来调节选择策略（selection policy），从而最大化用户的long-term满意度**。他们被广泛部署在实际系统中：内容推荐【20，5，26】和展示广告【6，22】。

然而，user feedback的最多形式是implicit feedback（比如：clicks），而它是有偏的，并且对于系统输出的用户评估是不完整的。例如：**一个用户跳过一个推荐item，并不因为是他不喜欢该item，有可能是因为他没有查看（examine）那个展示位置（例如：position bias）**。不幸的是，在contextual bandit应用中一个常见惯例是：**简单地将未点击（no click）看成是负反馈（negative feedback）的一种形式**。这会对模型更新引入不一致性（inconsistency），由于跳过的items并非真正不相关，因而它不可避免地会导致随时间的bandit算法的次优结果。

在本工作中，我们会关注使用user click feedback来学习contextual bandits、并建模这样的implicit feedback，作为用户结果查看（result examination）和相关度评估（relevance judgment）的一个部分。 **查看假设(Examination hypothesis)**【8】在点击建模型上是一个基础假设，会假设：在一个系统上的一个用户点击的返回结果，当且仅当结果被用户查看（examination）时，它与在该时刻的用户信息是相关的。**由于一个用户的查看行为是不可观测的（unobserved）**，我们会建模它作为一个隐变量，并在一个概率模型中实现该查看假设。我们会通过在相应的contextual features上的logistic functions中定义结果查看（result examination）和相关评判（relevance judgement）的条件概率。为了执行模型更新，我们**会采用一个变分贝叶斯方法来开发一个闭式（closed form）近似即时模型参数的后验分布。 该近似也会为在bandit learning中使用Thompson sampling策略进行arm selection来铺路**。我们的有限时间分析表明，尽管在参数估计中由于引入隐式变量增加了复杂度，我们的Thompson sampling policy会基于真实后验 ，从而保证能达到一个具有高概率的sub-linear Bayesian regert。我们也会演示，基于近似后验的Thompson sampling的regret是良性有界的（well-bounded）. 另外，我们会证明：当某人在点击反馈中建模结果查看（result examination）失败时，一个线性递增的regret是可能的，因为在负反馈中模型不能区分查看驱动的跳过（examination driven skips）还是不相关驱动的跳过（relevance driven skips）。

我们会在中国的MOOC个性化教育平台上测试该算法。为了在该平台上个性化学生的学习体验，当学生正观察视频时，我们会在课程视频顶部以banner的形式推荐类似测试的问题（quiz-like quenstions）。**该算法需要决定，在一个视频中在哪里向一个特定用户展示哪个问题**。如果学生觉得展示的问题对他理解课程有用，他可能会点击该banner并阅读问题以及更多关于该问题的在线内容。因此，**我们的目标是在选中问题上最大化CTR**。该应用有多个特性，会放大bias以及点击反馈的不完整性。

- 首先，基于用户体验的考虑，为了最小化讨厌度，一个banner的展示时间会限定在几秒内。
- 第二，由于该feature对于该平台来说是新引入的，许多用户可能不会意识到：他们会在该问题上点击，并且阅读更多相关内容。

作为结果，**在一个问题上没有点击，并不能表示不相关**。我们会在4个月周期内测试该算法，其中：总共有69个问题会用到该算法上来选择超过20个主要的视频，超过10w的学习观看session。基于无偏离线评估策略，对比起标准的contextual bandits，我们的算法会达到一个8.9%的CTR提升，它不会建模用户的查看行为（examination behavior）。

# 2.相关工作

略

# 3.问题设定

我们考虑一个contextual bandit问题，它具有有限但可能较大的arms。我们将：

- arm set：表示为A
- candidate arms子集：在每个trial t=1, ..., T上，learner会观察到candidate arms的子集**$$A_t \subset A$$**，其中，每个arm a与一个context vector $$x^a$$有关，会对arm的side information进行归纳。
- 一旦arm $$a_t \in A_t$$根据一些policy $$\pi$$被选中后，对应的隐式二元反馈$$C_{a_t}$$ （例如：user click），会由learner给出作为reward。

**learner的目标是：判断它的arm selection策略来最大化它随时间的累积收益（cumulative reward）**。使得该问题唯一和挑战的是：$$C_{a_t}$$不会真正影响用户关于选中arm $$a_t$$的评估。基于查看假设【13，8】:

- 当$$C_{a_t}=1$$时，选中的$$a_t$$必须与用户在time t需要的信息相关；
- 但当$$C_{a_t}=0$$时，$$a_t$$必须相关，但用户不会查看它

不幸的是，产生的查看条件对learner来说是不可观察的。

我们会通过一个**二元隐变量$$E_{a_t}$$来建模一个用户的结果查看（result examination）**，并假设：arm a的context vector $$x_t^a$$可以被分解成：

$$(x_{C,t}^a, x_{E,t}^a)$$

- 其中：$$x_{C,t}^a$$和$$x_{E,t}^a$$分别是$$d_C$$和$$d_E$$。

相应的，用户的结果查看和相关判断决策被假设成：**通过一个$$(x_{C,t}^a, x_{E,t}^a)$$猜想、以及相应的bandit参数为$$\theta^* = (\theta_C^*, \theta_E^*)$$来进行管理**。在本文其余部分，当没有二义性引入时，我们会drop掉index a以简化概念。作为结果，我们对在arm $$a_t$$的一个observed click $$C_t$$上做出以下的生成假设：

$$
\begin{align}
P(C_t = 1 | E_t = 0, x_{C,t}) & = 0 \\
P(C_t = 1 | E_t = 1, x_{C,t}) & = \rho(x_{C,t}^T \theta_C^*) \\
P(E_t = 1 | x_{E,t}) & = \rho(x_{E,t}^T \theta_E^*)
\end{align}
$$


其中：

- $$\rho(x) = \frac{1}{1 + e^{-x}}$$

基于该假设，我们有：

$$E[C_t \mid x_t] = \rho(x_{C,t}^T \theta_C^*) \rho(x_{E,t}^T \theta_E^*)$$

**作为结果，观察到的click feedback $$C_t$$是来自该生成过程的一个样本**。我们定义$$f_{\theta}(x) := E[C \mid x, \theta] = \rho(x_C^T \theta_C) \rho(x_E^T \theta_E)$$。到达time T一个policy $$\pi$$的accumulated regret的定义如下：

$$
Regret(T, \pi, \theta^*) = \sum\limits_{t=1}^T \underset{\alpha \in A_t}{max} f_{\theta^*} (x^a) - f_{\theta^*}(x^{a_t})
$$

其中，$$x^{a_t} := (x_C^{a_t}, x_E^{a_t})$$是arm $$a_t \in A_t$$的context vector，该arm会在time t时基于历史 $$H_t := \lbrace (A_i, x_i, C_i) \rbrace_{i=1}^{t-1}$$由policy $$\pi$$ 中。Bayesian regret的定义为$$E[Regret(T, \pi, \theta^*)]$$，其中采用的期望根据在$$\theta^* $$上的先验分布采用的，它可以被写成：

$$
BayesRegret(T, \pi) = \sum\limits_{t=1}^T E[max_{a \in A_t} f_{\theta^*}(x^a) - f_{\theta^*}(x^{a_t})]
$$

在我们的在线学习环境中，objective是发现policy $$\pi$$，并最小化在T上的accumulated regret。

# 4.算法

learner需要基于从click feedback $$\lbrace x_i, C_i \rbrace_{i=1}^t$$获得的随时间的互交，估计bandit参数 $$\theta_C^*$$和$$\theta_E^*$$。理想情况下，该估计可以根据bandit model参数通过最大化data likelihood来获得。然而，在我们的bandit learning setting中examination的包含作为一个隐变量，对于参数估计和arm selection来说会造成严重挑战。由于相应最优化问题的非凸性，传统的最小二乘估计、极大似然估计可以被轻易获取，更不必说计算效率。更糟的是，两个流行的bandit learning范式，UCB principle和Thompson sampling，都需要一个关于bandit参数及不确定性的精准估计。在本节中，我们提出一个有效的新解法来解决这两个挑战，它会利用variational Bayesian inference技术来近似学习即时参数，同时桥接参数估计（parameter estimaiton）和(arm selection policy)设计。

## 4.1 Variational Bayesian进行参数估计

为了完成在第3节中定义的生成过程，我们进一步假设$$\theta_C$$和$$\theta_E$$分别遵循高斯分布 $$N(\hat{\theta}_C, \sum_C)$$以及$$N(\hat{\theta}_E, \sum_E)$$。当一个新获得的observation $$(x_C, x_E, C)$$变得可提供时，我们会对开发一个闭式近似后验感兴趣。通过在log space中使用Bayes' rule，我们有：

$$
log P(\theta_C, \theta_E | x_C, x_E, C)  \\
        = log P(C | \theta_C, \theta_E, x_C, x_E) + log P(\theta_C, \theta_E) + log const \\
        = C log \rho(x_C^T \theta_C) \rho(x_E^T \theta_E) + (1 - C) log(1 - \rho(x_C^T \theta_C) \rho(x_E^T \theta_E)) \\
        = - \frac{1}{2} (\theta_C - \hat{\theta}_C)^T \sum_C^{-1} (\theta_C - \hat{\theta}_C) - \frac{1}{2} (\theta_E - \hat{\theta}_E)^T \sum_E^{-1} (\theta_E - \hat{\theta}_E) + log const
$$

关键思想是，会为似然函数开发一个$$\theta_C$$和$$\theta_E$$的quadratic form的variational lower bound。由于 $$log \rho(x) - \frac{x}{2}$$的convexity，对应于$$x^2$$，以及logx的Jensen’s不等式，所需形式的一个lower bound是可以达到的。当C=1时，通过等式(16)，我们有：

$$
l_{C=1}(x_C, x_E, \theta) := log( \rho(x_C^T \theta_C) \rho(x_E^T \theta_E)) \geq g(x_C^T\theta, \epsilon_C) + g(x_E^T \theta, \epsilon_E)
$$

...(1)

其中，$$g(x, \epsilon) := \frac{x}{2} - \frac{\epsilon}{2} + log \rho(\epsilon) - \lambda(\epsilon)(x^2 - \epsilon^2), \lambda(\epsilon) = \frac{tanh \frac{epsilon}{2}}{4 \epsilon}, x, \epsilon \in R$$。更特别的是，$$g(x, \epsilon)$$是一个度为2的对应于x的多项式。当C=0时，通过等式（17），我们有：

$$
l_{C=0} (x_C, x_E, \theta) := log( 1-\rho(x_C^T \theta_C) \rho(x_E^T \theta_E)) \\
        \geq H(q) + qg(-x_C^T \theta, \epsilon) + qg(x_E^T \theta, \epsilon_{E,1}) + (1 - q) g(-x_E^T\theta, \epsilon_{E,2})
$$

...(2)

其中，$$H(q) := - qlog q - (1 - q) log(1-q)$$。一旦在quadratic form中的lower bound确定后，我们可以使用一个Gaussian分布来近似我们的target后验，它的均值和covariance matrix由以下等式确定：

$$
\sum_{C, post}^{-1} = \sum_C^{-1} + 2q ^{1-C} \lambda (\epsilon) x_C x_C^T \\
\hat{\theta}_{C, post} = \sum_{C,post} (\sum_{C}^{-1} \hat{\theta}_C + \frac{1}{2} (-q)^{1-C} x_C) \\
\sum_{E,post}^{-1} = \sum_{E}^{-1} + 2 \lambda(\epsilon_E) x_E x_E^T \\
\hat{\theta}_{E,post} = \sum_{E,post} (\sum_E^{-1} \hat{\theta}_E + \frac{1}{2}(2q-1)^{1-C} x_E)
$$

...(3)(4)(5)(6)

其中，下标“post”表示在高斯分布中的参数，它逼近期望的后验。连续的观测可以被顺序包含到近似后验。还剩一件事件需要决策，例如：变化参数$$(\epsilon_C, \epsilon_E, q)$$的选择。一个选这些值的常用准则是，在observations上的似然是最大化的，与【12】相似的选择，我们会选择这些变分参数的闭式更新公式：

$$
\epsilon_C = \sqrt{E_{\theta_C} [x_C^T \theta_C]^2} \\
\epsilon_E = \sqrt{E_{\theta_E} [x_E^T \theta_E]^2} \\
q = \frac{exp(g(x_C^T \theta_C, \epsilon_C) + g(x_E^T\theta_E, \epsilon_E) - g(-x_E^T \theta_E, \epsilon_E))}{ 1 + exp(g(x_C^T \theta_C, \epsilon_C) + g(x_E^T \theta_E, \epsilon_E) - g(-x_E^T \theta_E, \epsilon_E))}
$$




其中，所有期望都在近似后验下采用。经验上，我们发现，近似后验和变分参数的迭代式更新收敛相当快，以致于它通常只需要少量迭代就可以获得一个满意的局部最大值。

## 4.2 使用近似lower bound的Thompson sampling

Thompson sampling, 通常称为“概率匹配（probability matching）”，会被广泛用于bandit learning中来平衡exploration和exploitation，并且它展示了极大的经验效率。Thompson sampling需要一个模型参数的分布来采样。在标准的Thompson sampling中，我们需要从模型参数的真实后验中进行采样。但由于logistic regression不会具有一个共轭先验（conjugate prior），在我们的问题中定义的模型不会具有一个准确的后验。我们决定，根据从等式（3）到等式（6）从近似后验中抽样。之后，我们会表明：这是一个非常紧凑的后验近似。一旦$$(\bar{\theta}_C, \bar{\theta}_E)$$的采样完成，我们可以选择相应的arm $$a_t \in A_t$$，它会最大化$$\rho(x_C^T \bar{\theta}_C) \rho(x_E^T \bar{\theta}_E$$。我们将生成的bandit算法命名为examination-click bandit，或者E-C bandit，如算法1所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/a7399c65a711eb9c3e2ccf86a237fe34ca157c4423e9266534f9b0a3d0d9e6565502d3b67e1a7246f66aa6d1a9c218ae?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

算法1 E-C Bandit

# 5.Regret Analysis

略

# 6.实验

略

# 7.结论





- 1.[https://keg.cs.tsinghua.edu.cn/jietang/publications/NIPS18-Qi-et-al-Bandit-Learning-with-Implicit-Feedback.pdf](https://keg.cs.tsinghua.edu.cn/jietang/publications/NIPS18-Qi-et-al-Bandit-Learning-with-Implicit-Feedback.pdf)