---
layout: page
title: sklearn中的学习曲线
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

每个estimator都有各自的优点和缺点。它的泛化误差可以用bias/variance/noise来衡量。estimator的bias指的是不同训练集的平均误差。estimator的variance指的是，不同训练集的敏感程度。noise指的是数据属性。

在下图中，我们来看下一个函数f(x)=cos(3/2 * Pi * x)以及它的一些noise样本。我们使用了三个不同的estimator来拟合该函数：带多项式feature的线性回归，多项式degree分别为：1，4和15。我们看到，第一个estimator，欠拟合（high bias）；第二个estimator，拟合；第三个estimator，过拟合。

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_underfitting_overfitting_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_underfitting_overfitting_0011.png" alt=""></a>
</figure>

bias/variance是estimator的固有属性，我们可以参考它们来选择学习算法以及超参数，以便让bias/variance尽可能地低。另一个减少模型的variance的方法是：使用更多的训练数据。但是，如果true function过度复杂，而使用一个低variance的estimator来逼近它时，你应该收集更多的训练数据。

在简单的一维问题中，我们可以很容易看出estimator是bias偏大还是variance偏大。而在高维空间里，模型很难可视化。我们可以使用以下的三个demo工具来绘制：

- [Underfitting vs. Overfitting](http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#example-model-selection-plot-underfitting-overfitting-py)
- [Plotting Validation Curves](http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#example-model-selection-plot-validation-curve-py)
- [Plotting Learning Curves](http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#example-model-selection-plot-learning-curve-py)

# 2.1 交叉验证曲线

三个得分：

- training score
- validation score
- testset score

为了验证一个模型，我们需要一个scoring函数，来评估分类器的准确度。同时选择一个estimator的多个超参数，可以使用GridSearch等办法，来找到在交叉验证集中的最大得分。注意，如果我们基于一个validation score上优化超参数时，如果valadation score的bias挺大，那么泛化的效果会很差。为得到一个合适的泛化估计，我们还必须计算另一个测试集的得分。

在training score和validation score上，针对单个超参数绘制曲线，可以帮助我们发现在这些超参数上的estimator是overfitting还是underfitting。

{% highlight python %}

>>> import numpy as np
>>> from sklearn.learning_curve import validation_curve
>>> from sklearn.datasets import load_iris
>>> from sklearn.linear_model import Ridge

>>> np.random.seed(0)
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> indices = np.arange(y.shape[0])
>>> np.random.shuffle(indices)
>>> X, y = X[indices], y[indices]

>>> train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha",
...                                               np.logspace(-7, 3, 3))
>>> train_scores           
array([[ 0.94...,  0.92...,  0.92...],
       [ 0.94...,  0.92...,  0.92...],
       [ 0.47...,  0.45...,  0.42...]])
>>> valid_scores           
array([[ 0.90...,  0.92...,  0.94...],
       [ 0.90...,  0.92...,  0.94...],
       [ 0.44...,  0.39...,  0.45...]])
{% endhighlight %} 

- 如果training score和validation score都很低，那么该estimator将会是underfitting。
- 如果training score很高，但validation score很低，那么estimator将overfitting。
- 如果training score很低，而validation score很高，通常不可能。

所有的三种情况可以通过下图来发现，通过SVM的gamma值的不同来体现。

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_validation_curve_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_validation_curve_0011.png" alt=""></a>
</figure>

# 2.2 学习曲线

一条学习曲线(learning curve)表示estimator上的validation score和training score随着训练样本数的变化而变化。通过它我们要以发现，通过添加更多的训练样本，我们可以从variance error/bias error中获得多少提升。如果validation score和train score随着样本数的增加都变得很低收敛到一个值时，再增加更多的样本不会获得提升。下图展示了Navie Bayes的收敛：

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_learning_curve_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_learning_curve_0011.png" alt=""></a>
</figure>

我们必须使用这样的estimator或者参数：它可以使当前estimator学到更复杂的概念（i.e. 具有更低的bias）。对于大训练集来说，如果training score大于validation score，添加更多的训练样本将增加泛化能力。下图我们绘制了可以从更多训练样本中得到提升的 SVM曲线：

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_learning_curve_0021.png"><img src="http://scikit-learn.org/stable/_images/plot_learning_curve_0021.png" alt=""></a>
</figure>

我们可以使用函数learning_curve 来生成我们要绘制的学习曲线上的值。（使用的样本数，训练集上的平均得分，验证集上的平均得分）

{% highlight python %}
>>> from sklearn.learning_curve import learning_curve
>>> from sklearn.svm import SVC

>>> train_sizes, train_scores, valid_scores = learning_curve(
...     SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
>>> train_sizes            
array([ 50, 80, 110])
>>> train_scores           
array([[ 0.98...,  0.98 ,  0.98...,  0.98...,  0.98...],
       [ 0.98...,  1.   ,  0.98...,  0.98...,  0.98...],
       [ 0.98...,  1.   ,  0.98...,  0.98...,  0.99...]])
>>> valid_scores           
array([[ 1. ,  0.93...,  1. ,  1. ,  0.96...],
       [ 1. ,  0.96...,  1. ,  1. ,  0.96...],
       [ 1. ,  0.96...,  1. ,  1. ,  0.96...]])
       
{% endhighlight %} 



参考：

1.[http://scikit-learn.org/stable/modules/learning_curve.html](http://scikit-learn.org/stable/modules/learning_curve.html)
