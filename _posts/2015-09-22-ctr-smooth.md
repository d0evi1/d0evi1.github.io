---
layout: post
title: ctr平滑 
description: 
modified: 2015-08-25
tags: [ctr]
---

在广告系统中，一个重要的指标是CTR。ctr=点击(Click)/曝光(Impression)。

- 如果一个广告只有5次曝光，没有点击，是否表示它的ctr为0? 
- 如果一个广告曝光了4次，却有3次点击，它的ctr是否就为75%?

直觉上肯定是不对的。

一个广告的每次展示，相当于概率试验里的投硬币。客户点击/不点击广告，是一个泊努利试验(Bernoulli Trial)。多个客户点击/不点击广告，可以看成是进行一串Bernoulli试验，即二项试验(Binomial Experiment)，服从二项分布。

假设我们尝试了n次试验（有n个客户），每个客户点击该广告的概率为p（平均ctr）。该试验服从二项分布：B(n,p)。在n次试验中，观察到有k次点击的概率为：

$$
B(k,n,p)=C^{n}_{k}p^k(1-p)^{n-k}
$$

例如，如果有100个visitors，该广告的点击率为10%，点击次数的概率分布(PMF)为：

即上面公式中：n=100, 横轴为k，纵轴为p。

{% highlight python %}

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

n, p = 4, 0.1
x = np.arange(1,100)

fig, ax = plt.subplots(1, 1)
ax.plot(x, binom.pmf(x, n, p), 'bo', ms=8, label='binom pmf')
plt.show()

{% endhighlight %}

我们可得到类似这样的曲线：

<img src="http://www.marketingdistillery.com/wp-content/uploads/2014/09/CTR_binom.png">

当CTR接近10%时，有相当高的机会看到8次-12次左右的点击。如果观察的次数越少，噪声也会越多。当只有4次观测（observation）时，有65%的机会看到没有点击，30%的机会看到一次点击，5%的机会看到2次点击。

{% highlight python %}

from scipy.misc import comb

n = 4
k = 0
r = 0.1
p0 = comb(n,k)* (r)**k * (1-r)**(n-k)
### p1接近0.65

k = 1
p1 = comb(n,k)* (r)**k * (1-r)**(n-k)
### p2接近0.30

k=2
p2 = comb(n,k)* (r)**k * (1-r)**(n-k)
### p3接近0.05

{% endhighlight %}


# 2.CTR的Beta分布

假如，我们只有少量观测，是否可以去估计CTR呢？是否可以设计一个算法去模仿相应的模型数据？

为了在一个广告上模仿点击，我们首先使用一些分布上的CTR的值，接着使用它们作为在二项分布上的点击概率。这意味着我们需要两个随机变量。首先，是一个在[0,1]范围内的CTR的连续分布。第二个，是使用该CTR作为参数的二项分布。

什么样的分布是合理的？我们首先想到的是：正态分布。但它有负数，负的CTR显然没意义。另外，它是对称的，没有理由将模型限制在对称分布上。最好的分布是：Beta分布。它在[0,1]区间内是连续的，并且有各种形状。它的两个参数为：α(alpha) 和 β(beta)。

有两种方式来从数据中估计α和β的值，其中有一种方法特别有用：“均值和样本量”参数化（“Mean and Sample Size” parametrization）。

假设我们从一个很大的样本中抽取的10次曝光和2次点击来估计它的分布。假设 ν为样本量。 ν=10，μ为CTR均值。我们有2次点击和10次曝光：μ=2/10=0.2。Beta分布的参数化为：

α=μν,β=(1–μ)ν

在该例中：

α=0.2⋅10=2, β=(1–0.2)⋅10=0.8⋅10=8

beta分布的概率计算公式：

$$
f(x;\alpha,\beta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1}
$$

其中的f(x;a,β)就是实际概率，而x就是这里我们的点击率。


绘制出对应的Beta曲线：

{% highlight python %}

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

a,b=2,8
x = np.arange(0,1,0.01)

fig, ax = plt.subplots(1, 1)
ax.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')
plt.show()

{% endhighlight %}

可以得到类似的图：

<img src="http://www.marketingdistillery.com/wp-content/uploads/2014/09/CTR_beta.png">

从10次观测中的不确定性，通过该分布的扩展可以很容易解释。同理，我们可以尝试计算参数化，如果我们具有1000个观测以及200次点击。你可以注意到它将高度集中围绕在20%的CTR中。

将Beta分布与Binomial分布结合在一起，称为Beta-Binomial分布。

# 3.贝叶斯推断（Bayesian inference）

# 3.1 

在参考文献一中，提出的方法是直接使用先验CTR：

通常，我们实际展示多个广告。计算观测时，当存在不确定性时，我们会生成一个CTR的估计值。一些广告可能具有成千上万的曝光和点击，而另一些可能只有很少的点击。这时候，可以引入贝叶斯推断（Bayesian inference）。我们使用一个先验的CTR，当新观测被记录时，我们不断更新它。先验CTR有很多方式确定。如果时间足够，我们可以使用基于Mean和sample size的参数化方法。如果我们没有足够信息，我们可以使用事先分配的先验CTR(non-informative prior)，比如β(0.5,0.5):

一旦我们定义好先验CTR后，我们需要指定似然函数（likelihood function）。在我们的案例中，在结定参数集（CTR）下的观测的似然（likelihood）由二项分布给出。二项分布似然加上Beta先验，允许我们使用联合先验概率来获取一个后验分布。在我们的先验β(a,b)下，经过N次曝光观测到x次点击，得到的后验是这样一个Beta分布：β(a+x,b+N–x)。

还是使用之前的观测：4次曝光，1次点击，我们得到这样的后验：

<img src="http://www.marketingdistillery.com/wp-content/uploads/2014/09/CTR_Simple_post-297x300.png">

贝叶斯模型可以后验的总结、或者通过随机抽样(均值、中位数、标准差等）进行分析。例如，在上面的贝叶斯推断步骤后，我们希望看到：一个广告在10000次曝光下，具有0.35的CPC（Cost-per－Click）

{% highlight python %}

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

cpc = 0.35

priorA = 0.5
priorB = 0.5

clicks = 1
impressions = 4

 #rvs：生成随机数
plt.hist(beta.rvs(priorA + clicks, priorB + impressions - clicks, size=10000) * 10000 * cpc, 20, facecolor='green')

plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.grid(True)

plt.show()

{% endhighlight %}


## 3.2 数据连续性

在参考文献二中提到，在许多情况下，我们更关心CTR的趋势，而非某个绝对值快照。对于罕见的(page/ad) pair，任何时间点，CTR的估计都是很大噪声的。根据某种数据连续性，我们可以将impression/click看做是重复测量的离散集合，并在这些测量上采用了一种指数平滑（exponential smoothing），最后我们得到一个平滑的CTR。

对于一个(page/ad) pair，我们具有连续M天的测量：impressions(I1,I2,I3,...Im)和clicks(C1,C2,...,Cm)，然后我们希望估计在第M天的CTR。

\$ \hat{I} \$和 \$ \hat{C} \$各自表示平滑后的impressions和clicks。平滑后的CTR就等于\$ \frac{\hat{C}}{\hat{I}} \$。

$$
\hat{C_j}=C_j j=1
$$

$$
\hat{C_j}=\gammaC_j+(1-\gamma)\hat{C_{j-1}} j=2,...,M
$$

其中γ为平滑因子，它0<γ<1，（γ控制着平滑中的历史影响：当γ接近0，表示与历史值接近；当γ接近1，表示与历史值关系越少）。换句话说，随着时间的流逝，平滑过的\$ \hat{C_j}=C_j \$相当于变成了对过去观察值的指数式降权平均。你可以直接在CTR上应用指数平滑，而非各自对impressions/clicks做平滑。




(未完待续)

## 参考

[http://www.marketingdistillery.com/2014/09/24/bayesian-modeling-of-click-through-rate-for-small-data/](http://www.marketingdistillery.com/2014/09/24/bayesian-modeling-of-click-through-rate-for-small-data/)

[Click-Through Rate Estimation for
Rare Events in Online Advertising](http://www.cs.cmu.edu/~xuerui/papers/ctr.pdf)

[如何通俗理解beta分布？](https://www.zhihu.com/question/30269898)


