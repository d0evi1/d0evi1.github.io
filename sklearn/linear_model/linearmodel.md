---
layout: page
title: sklearn中的普通最小二乘法
tagline: 介绍
---
{% include JB/setup %}

下面讲的方法用于回归，通过对输入参数进行一些线性组合来预测target值。在数学表示中，预测值为<img src="http://www.forkosh.com/mathtex.cgi?\hat{y} ">

<img src="http://www.forkosh.com/mathtex.cgi?\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p">

该模型设计了一个向量<img src="http://www.forkosh.com/mathtex.cgi?w = (w_1,
..., w_p)">作为相关系数参数：coef_，w0作为参数：intercept_。

如果想执行线性模型的分类，可以详见：Logistic regression。

# 1.最小二乘法（Ordinary Least Squares）

LinearRegression使用回归系数：<img src="http://www.forkosh.com/mathtex.cgi?w = (w_1,..., w_p)">对线性模型进行拟合，通过对数据集的观察值和预测值之间做残差平方和（RSS：residual sum of squares），进行最小化。数学表示为：

<img src="http://www.forkosh.com/mathtex.cgi?\underset{w}{min\,} {|| X w - y||_2}^2">

<figure>
    <a href="http://scikit-learn.org/stable/_images/plot_ols_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_ols_0011.png" alt=""></a>
</figure>

LinearRegression采用fit方法对X,y进行拟合，最终得到的线性模型，将w存储在coef_属性上：

{% highlight python %}

>>> from sklearn import linear_model
>>> clf = linear_model.LinearRegression()
>>> clf.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

>>> clf.coef_
array([ 0.5,  0.5])

{% endhighlight %}

然后，对于最小二乘法的回归系数估计，依赖于模型term的独立性。当terms与设计矩阵X的列相关时，具有一个近似的线性依赖，设计矩阵将与奇异矩阵相接近，结果通过观察值发现产生了一个较大的variance，最小二乘估计（least-squares）变得对随机误差高度敏感。当被采集的数据设计时如果没有经验，就有可能发生多重共线性（Multicollinearity）的情况。

示例：

- [Linear Regression Example](http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#example-linear-model-plot-ols-py)

# 2. 最小二乘法的复杂度

最小二乘法使用了X矩阵分解的奇异值。如果X的size为(n, p) ，那么该方法的cost为<img src="http://www.forkosh.com/mathtex.cgi?O(n p^2)">，假设：<img src="http://www.forkosh.com/mathtex.cgi?n \geq p">

参考：

1.[http://scikit-learn.org/stable/modules/linear_model.html](http://scikit-learn.org/stable/modules/linear_model.html)
