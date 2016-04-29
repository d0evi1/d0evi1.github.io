---
layout: page
title: sklearn中的gbt(gbdt/gbrt) 
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

Gradient Tree Boosting或Gradient Boosted Regression Trees（GBRT)是一个boosting的泛化表示，它使用了不同的loss函数。GBRT是精确、现成的过程，用于解决回归/分类问题。Gradient Tree Boosting模型则用于许多不同的领域：比如：网页搜索Ranking、ecology等。

GBRT的优点是：

- 天然就可处理不同类型的数据（=各种各样的features）
- 预测能力强
- 对空间外的异常点处理很健壮（通过健壮的loss函数）

GBRT的缺点是：

- 扩展性不好，因为boosting天然就是顺序执行的，很难并行化

sklearn.ensemble通过GBRT提供了分类和回归的功能。

# 2.分类

GradientBoostingClassifier支持二元分类和多元分类。以下的示例展示了如何拟合一个GBC分类器，它使用100个单层决策树作为弱学习器：

{% highlight python %}

>>> from sklearn.datasets import make_hastie_10_2
>>> from sklearn.ensemble import GradientBoostingClassifier

>>> X, y = make_hastie_10_2(random_state=0)
>>> X_train, X_test = X[:2000], X[2000:]
>>> y_train, y_test = y[:2000], y[2000:]

>>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
...     max_depth=1, random_state=0).fit(X_train, y_train)
>>> clf.score(X_test, y_test)                 
0.913...


{% endhighlight %}

弱学习器的数目（比如：回归树）通过参数n_estimators来控制；每棵树的size可以通过设置树深度max_depth或者叶子点的最大个数max_leaf_nodes来控制。learning_rate是一个范围在[0.0, 1.0]间的超参数，可以通过shrinkage来控制是否过拟合（overfitting）

注意：超过2个分类时，需要在每次迭代时引入n_classes的回归树，因此，总的索引树为（n_classes * n_estimators）。**对于分类数目很多的情况，强烈推荐你使用 RandomForestClassifier 来替代GradientBoostingClassifier**

# 4.回归

GradientBoostingRegressor支持不同的loss函数的回归，可以通过loss参数来指定；缺省的loss函数是：最小平方法（'ls'）.

{% highlight python %}
>>> import numpy as np
>>> from sklearn.metrics import mean_squared_error
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.ensemble import GradientBoostingRegressor

>>> X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
>>> X_train, X_test = X[:200], X[200:]
>>> y_train, y_test = y[:200], y[200:]
>>> est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
...     max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
>>> mean_squared_error(y_test, est.predict(X_test))    
5.00...

{% endhighlight %}

下图显示了使用最小平方法的loss函数的GradientBoostingRegressor ，在500个弱学习器上，对boston房价问题进行预测（sklearn.datasets.load_boston）。左图展示了每次迭代的训练误差和测试误差。每次迭代的训练误差存储在GBT模型的train_score_属性项上。每次迭代的测试误差通过在预测的每个阶段调用staged_predict方法返回。该图可以通过early stopping来决定树的最优个数（比如：n_estimators）。右图展示了feature的重要性，可通过feature_importances_属性来查看。

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_gradient_boosting_regression_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_gradient_boosting_regression_0011.png" alt=""></a>
</figure>

示例：

- [Gradient Boosting regression](http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#example-ensemble-plot-gradient-boosting-regression-py)
- [Gradient Boosting Out-of-Bag estimates](http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_oob.html#example-ensemble-plot-gradient-boosting-oob-py)

# 5.拟合其它的弱学习器

GradientBoostingRegressor 和 GradientBoostingClassifier都支持warm_start=True，它允许你向一个已经拟合的模型上添加更多的estimators。

{% highlight python %}
>>> _ = est.set_params(n_estimators=200, warm_start=True)  # set warm_start and new nr of trees
>>> _ = est.fit(X_train, y_train) # fit additional 100 trees to est
>>> mean_squared_error(y_test, est.predict(X_test))    
3.84...

{% endhighlight %}


# 6.控制树的size

回归树的基础学习器（base learners）的size，定义了可以被GB模型捕获的各种交互的level。通常，一棵树的深度为h，可以捕获h阶的影响因子(interactions)。控制各个回归树的size有两种方法。

如果你指定max_depth=h，那么将会长成深度为h的完整二元树。这样的树至多有2^h个叶子，以及2^h-1中间节点。

另一种方法：你可以通过指定叶子节点的数目（max_leaf_nodes）来控制树的size。这种情况下，树将使用最优搜索(best-first search)的方式生成，并以最高不纯度（impurity）的方式展开。如果树的max_leaf_nodes=k，表示具有k-1个分割节点，可以建模最高(max_leaf_nodes-1)阶的interactions。

我们发现，max_leaf_nodes=k 与 max_depth=k-1 进行比较，训练会更快，只会增大一点点的训练误差（training error）。参数max_leaf_nodes对应于gradient boosting中的变量J，与R提供的gbm包的参数interaction.depth相关，为：max_leaf_nodes == interaction.depth + 1。


# 7.数学公式

略，见下面的参考

# 8.loss函数

下面的loss函数：

回归

- 最小二乘法Least squares（'ls'）：最自然的选择，因为它的计算很简单。初始模型通过target的平均值来给出。
- 最小绝对偏差Least absolute deviation （'lad'）：一个健壮的loss函数，用于回归。初始模型通过target的中值来给出。
- Huber ('huber'): 另一个健壮的loss函数，
- Quantile ('quantile'):

分类

- Binomial deviance ('deviance')
- Multinomial deviance ('deviance')
- Exponential loss ('exponential')

# 9.正则化

## 9.1 Shrinkage

[f2001]提出了一种简单的正则化策略，它通过一个因子v将每个弱学习器的贡献进行归一化。

<figure>
	<a href="http://scikit-learn.org/stable/_images/math/47430226015a6b7a0653abed8c7d5d5e9f529404.png"><img src="http://scikit-learn.org/stable/_images/math/47430226015a6b7a0653abed8c7d5d5e9f529404.png" alt=""></a>
</figure>

参数v也被称为**学习率（learning rate）**，因为它可以对梯度下降的步长进行调整；它可以通过learning_rate参数进行设定。

参数learning_rate会强烈影响到参数n_estimators（即弱学习器个数）。learning_rate的值越小，就需要越多的弱学习器数来维持一个恒定的训练误差(training error)常量。经验上，推荐小一点的learning_rate会对测试误差(test error)更好。[HTF2009]推荐将learning_rate设置为一个小的常数（e.g. learning_rate <= 0.1），并通过early stopping机制来选择n_estimators。我们可以在[R2007]中看到更多关于learning_rate与n_estimators的关系。

## 9.2 子抽样Subsampling

[F1999]提出了随机梯度boosting，它将bagging(boostrap averaging)与GradientBoost相结合。在每次迭代时，基础分类器(base classifer)都在训练数据的一个子抽样集中进行训练。子抽样以放回抽样。subsample的典型值为：0.5。

下图展示了shrinkage的效果，并在模型的拟合优度（Goodness of Fit）上进行子抽样（subsampling）。我们可以很清楚看到：shrinkage的效果比no-shrinkage的要好。使用shrinkage的子抽样可以进一步提升模型准确率。而不带shinkage的子抽样效果差些。

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_gradient_boosting_regularization_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_gradient_boosting_regularization_0011.png" alt=""></a>
</figure>

另一个减小variance的策略是，对features进行子抽样（类比于RandomForestClassifier中的随机split）。子抽样features的数目可以通过max_features参数进行控制。

注意：使用max_features值可以极大地降低运行时长。

随机梯度boosting允许计算测试偏差（test deviance）的out-of-bag估计，通过计算没有落在bootstrap样本中的其它样本的偏差改进（i.e. out-of-bag示例）。该提升存在属性oob_improvement_中。oob_improvement_[i]表示在添加第i步到当前预测中时，OOB样本中的loss的提升。OOB估计可以被用于模型选择，例如：决定最优的迭代数。OOB估计通常很少用，我们推荐你使用交叉验证（cross-validation），除非当cross-validation十分耗时的时候。

示例：

- ［Gradient Boosting regularization］
- ［Gradient Boosting Out-of-Bag estimates］
- ［OOB Errors for Random Forests］

# 10.内省

单颗决策树可以通过内省进行可视化树结构。然而，GradientBoost模型由成百的回归树组成，不能轻易地通过对各棵决策树进行内省来进行可视化。幸运的是，已经提出了许多技术来归纳和内省GradientBoost模型。

## 10.1 feature重要程度

通常，features对于target的结果预期的贡献不是均等的；在许多情况下，大多数features都是不相关的。当内省一个模型时，第一个问题通常是：在预测我们的target时，哪些features对结果预测来说是重要的。

单棵决策树天生就可以通过选择合适的split节点进行特征选择（feature selection）。该信息可以用于计算每个feature的重要性；基本思想是：如果一个feature经常用在树的split节点上，那么它就越重要。这个重要性的概率可以延伸到决策树家族ensembles方法上，通过对每棵树的feature求简单平均即可。

GradientBoosting模型的重要性分值，可以通过feature_importances_属性来访问：

{% highlight python %}

>>> from sklearn.datasets import make_hastie_10_2
>>> from sklearn.ensemble import GradientBoostingClassifier

>>> X, y = make_hastie_10_2(random_state=0)
>>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
...     max_depth=1, random_state=0).fit(X, y)
>>> clf.feature_importances_  
array([ 0.11,  0.1 ,  0.11,  ...


{% endhighlight %}

示例：

- [Gradient Boosting regression](http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#example-ensemble-plot-gradient-boosting-regression-py)

## 10.2 局部依赖

局部依赖图（Partial dependence plots ：PDP）展示了target结果与一些目标特征（target feature）之间的依赖；边缘化（marginalizing）所有其它特征（'complement' features）。另外，我们可以内省这两者的局部依赖性。

由于人的认知的有限，目标特征的size必须设置的小些（通常：1或2），目标特征可以在最重要的特征当中进行选择。

下图展示了关于California居住情况的、4个one-way和一个two-way的局部依赖图[示例](http://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html)：

<figure>
	<a href="http://photo.yupoo.com/wangdren23/FvKfc8Yb/medish.jpg"><img src="http://photo.yupoo.com/wangdren23/FvKfc8Yb/medish.jpg" alt=""></a>
</figure>

one-way的PDP图告诉我们，target结果与target特征之间的相互关系（e.g. 线性/非线性）。左上图展示了中等收入（median income）在房价中位数（median house price）上的分布；我们可以看到它们间存在线性关系。

带有两个target特征的PDP，展示了和两个特征的相关关系。例如：上图中，两个变量的PDP展示了房价中位数（median house price）与房龄（house age）和平均家庭成员数（avg. occupants）间的关系。我们可以看到两个特征间的关系：对于AveOccup>2的，房价与房龄（HouseAge）几乎完全独立。而AveOccup<2的，房价则强烈依赖年齡。

partial_dependence模块提供了一个很方便的函数：plot_partial_dependence 来创建one-way以及two-way的局部依赖图。下例，我们展示了如何创建一个PDP：两个two-way的PDP，feature为0和1，以及一个在这两个feature之间的two-way的PDP：

{% highlight python %}

>>> from sklearn.datasets import make_hastie_10_2
>>> from sklearn.ensemble import GradientBoostingClassifier
>>> from sklearn.ensemble.partial_dependence import plot_partial_dependence

>>> X, y = make_hastie_10_2(random_state=0)
>>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
...     max_depth=1, random_state=0).fit(X, y)
>>> features = [0, 1, (0, 1)]
>>> fig, axs = plot_partial_dependence(clf, X, features) 


{% endhighlight %}


对于多分类的模块，我们需要设置类的label，通过label参数来创建PDP：

{% highlight python %}
>>> from sklearn.datasets import load_iris
>>> iris = load_iris()
>>> mc_clf = GradientBoostingClassifier(n_estimators=10,
...     max_depth=1).fit(iris.data, iris.target)
>>> features = [3, 2, (3, 2)]
>>> fig, axs = plot_partial_dependence(mc_clf, X, features, label=0) 

{% endhighlight %}

如果你需要一个局部依赖函数的原始值，而非你使用partial_dependence函数绘制的图：

{% highlight python %}
>>> from sklearn.ensemble.partial_dependence import partial_dependence

>>> pdp, axes = partial_dependence(clf, [0], X=X)
>>> pdp  
array([[ 2.46643157,  2.46643157, ...
>>> axes  
[array([-1.62497054, -1.59201391, ...

{% endhighlight %}

该函数需要两个参数：

- grid: 它控制着要评估的PDP的target特征的值
- X: 它提供了一个很方便的模式来从训练数据集上自动创建grid。

返回值axis：

-如果给定了X，那么通过这个函数返回的axes给出了每个target特征的axis.

对于在grid上的target特征的每个值，PDP函数需要边缘化树的不重要特征的预测。在决策树中，这个函数可以用来评估有效性，不需要训练集数据。对于每个grid点，会执行一棵加权树的遍历：如果一个split节点涉及到'target'特征，那么接下去的左、右分枝，每个分枝都会通过根据进入该分枝的训练样本的fraction进行加权。最终，通过访问所有叶子的平均加权得到局部依赖。对于树的ensemble来说，每棵树的结果都会被平均。

注意点：

- 带有loss='deviance'的分类，它的target结果为logit(p)
- 初始化模型后，target结果的预测越精确；PDP图不会包含在init模型中

示例：

- [Partial Dependence Plots](http://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html#example-ensemble-plot-partial-dependence-py)


参考：

1.[http://scikit-learn.org/stable/modules/ensemble.html](http://scikit-learn.org/stable/modules/ensemble.html)
