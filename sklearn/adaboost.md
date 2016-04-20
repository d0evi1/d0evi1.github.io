---
layout: page
title: sklearn中的adaboost 
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

sklearn.ensemble模块包含了很流行的boosting算法：AdaBoost.

AdaBoost的核心思想是：通过在重复修改过版本的数据上拟合一系列弱分类器（这些模型比随机猜测强，比如：小的决策树）。所有这些模型上的预测，会通过一个加权多数投票（weighted majority vote）方式来产生最终的预测。每次boosting迭代中的数据更新，会将权重w1, w2, w3, .., wN应用到每个训练样本之上。初始化时，这些权重都被置为wi=1/N，每一步都简单地在原始数据上训练一个弱分类器。对于每个后继的迭代，样本权重都各自进行更新，该学习算法会重新使用这些数据并重新进行赋权(reweight)。对于一个结定的step，通过boost模型预测错误的训练样本，在下一step上会进行加权，而预测正确的则会进行降权。随着迭代的进行，很难预测的样本将接受到不断调整。每一步的弱分类器因此会关注于解决那些在上一步错过的样本。

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_adaboost_hastie_10_2_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_adaboost_hastie_10_2_0011.png" alt=""></a>
</figure>

AdaBoost可以同时被用在分类和回归问题中。

- 对于多分类，AdaBoostClassifier实现了AdaBoost-SAMME和 AdaBoost-SAMME.R算法
- 对于回归，AdaBoostRegressor 实现了AdaBoost.R2算法

# 2.用法

下例展示了如何通过100个弱学习器（weak learners）拟合一个AdaBoost分类器：

{% highlight python %}

>>> from sklearn.cross_validation import cross_val_score
>>> from sklearn.datasets import load_iris
>>> from sklearn.ensemble import AdaBoostClassifier

>>> iris = load_iris()
>>> clf = AdaBoostClassifier(n_estimators=100)
>>> scores = cross_val_score(clf, iris.data, iris.target)
>>> scores.mean()                             
0.9...

{% endhighlight %}

弱学习器的个数，通过参数n_estimators来控制。learning_rate参数控制着在最终聚合时弱学习器的贡献。缺省情况下，弱学习器都是单层决策树（decesion stumps）。不同的弱学习器类型可以通过base_estimator参数来指定。想要获得理想的结果，需要调整最主要的参数是，n_estimators以及base estimator的复杂度（比如：在决策树下：深度：max_depth，或者 所需要的最小样本个数：min_samples_leaf）

示例：

- [Discrete versus Real AdaBoost](http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html#example-ensemble-plot-adaboost-hastie-10-2-py)
- [Multi-class AdaBoosted Decision Trees](http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#example-ensemble-plot-adaboost-multiclass-py)
- [Two-class AdaBoost](http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#example-ensemble-plot-adaboost-twoclass-py)
- [Decision Tree Regression with AdaBoost](http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_regression.html#example-ensemble-plot-adaboost-regression-py)


参考：

[http://scikit-learn.org/stable/modules/ensemble.html](http://scikit-learn.org/stable/modules/ensemble.html)
