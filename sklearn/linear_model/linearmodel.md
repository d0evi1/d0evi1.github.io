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

在数理统计中，残差是指实际观察值与估计值（拟合值）之间的差。

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

## 最小二乘法的复杂度

最小二乘法使用了X矩阵分解的奇异值。如果X的size为(n, p) ，那么该方法的cost为<img src="http://www.forkosh.com/mathtex.cgi?O(n p^2)">，假设：<img src="http://www.forkosh.com/mathtex.cgi?n \geq p">


# 2. 岭回归(Ridge Regression)

[Ridge](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)回归通过引入对回归系数size的一个惩罚项，解决了最小二乘法存在的一些问题。ridge回归系数的优化目标为：最小化带罚项的残差平方和。

<img src="http://www.forkosh.com/mathtex.cgi?\underset{w}{min\,} {|| X w - y||_2}^2 + \alpha {||w||_2}^2}">

这里，<img src="http://www.forkosh.com/mathtex.cgi?\alpha \geq 0">表示复杂度参数，它控制着收缩（shrinkage）的程度：<img src="http://www.forkosh.com/mathtex.cgi?\alpha">越大，shrinkage程度越大，回归系数就越健壮（不容易产生多重共线性）。

<figure>
    <a href="http://scikit-learn.org/stable/_images/plot_ridge_path_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_ridge_path_0011.png" alt=""></a>
</figure>

上图的每种着色，表示coefficient vector中每个不同的feature，它们在正则化参数函数上进行展示。在相应path的终点，alpha趋向于0，相应的解趋向于普通的最小二乘法，coefficients参数具有强烈的波动。

和其它线性模型一样，Ridge的fit建模后，也会将回归系数存于coef_属性上：

{% highlight python %}

>>> from sklearn import linear_model
>>> clf = linear_model.Ridge (alpha = .5)
>>> clf.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) 
Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
      
>>> clf.coef_
array([ 0.34545455,  0.34545455])

>>> clf.intercept_ 
0.13636...

{% endhighlight %}

示例：

- [Plot Ridge coefficients as a function of the regularization](http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html#example-linear-model-plot-ridge-path-py)
- [Classification of text documents using sparse features](http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py)

## 2.1 Ridge复杂度

和普通最小二乘法的一样。

## 2.2 如何设置正则化参数？泛化交叉验证(GCV)

[RidgeCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV)实现了岭回归，内置了对alpha参数的交叉验证。该对象与GridSearchCV工作机制类似，有一点不同的是，它缺省使用泛化交叉验证（GCV），一种留一交叉验证（leave-one-out cross-validation）的有效形式。

{% highlight python %}

>>> from sklearn import linear_model
>>> clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
>>> clf.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])       
RidgeCV(alphas=[0.1, 1.0, 10.0], cv=None, fit_intercept=True, scoring=None,
    normalize=False)
    
>>> clf.alpha_                                      
0.1

{% endhighlight %}

# 3.Lasso

[Lasso](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso)是一种线性模型，用于估计稀疏的回归系数。在一些情况通常更喜欢使用更少的参数值，这样可以有效地降低变量的数目，使得解决方案独立。由于这个原因，Lasso和它的变种在压缩感知（compressed sensing）领域是基础。在特定条件下，它可以恢复非零权重的完整集合，[详见](http://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html#example-applications-plot-tomography-l1-reconstruction-py)。

数学上，它包含了一个使用<img src="http://www.forkosh.com/mathtex.cgi?\ell_1 ">作为正则项进行训练的线性模型。要最小化的目标函数为：

<img src="http://www.forkosh.com/mathtex.cgi?\underset{w}{min\,} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha ||w||_1}">

lasso估计的目标是最小化带罚项<img src="http://www.forkosh.com/mathtex.cgi?\alpha ||w||_1 ">的最小二乘，其中：<img src="http://www.forkosh.com/mathtex.cgi?\alpha">为常数，<img src="http://www.forkosh.com/mathtex.cgi? ||w||_1 ">为参数向量的l1范数。

Lasso的实现使用了坐标下降法（coordinate descent）来拟合回归系数。另外最小角回归提供了另一种实现：

{% highlight python %}

>>> from sklearn import linear_model
>>> clf = linear_model.Lasso(alpha = 0.1)
>>> clf.fit([[0, 0], [1, 1]], [0, 1])
Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
   
>>> clf.predict([[1, 1]])
array([ 0.8])

{% endhighlight %}

对于低级别的任务，函数[lasso_path](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html#sklearn.linear_model.lasso_path)很有效，它会使用所有可能值来计算回归系数。


示例：

- [Lasso and Elastic Net for Sparse Signals](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#example-linear-model-plot-lasso-and-elasticnet-py)
- [Compressive sensing: tomography reconstruction with L1 prior (Lasso)](http://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html#example-applications-plot-tomography-l1-reconstruction-py)

**注意：使用Lasso进行特征选择**

Lasso回归产生稀疏的模型，它可以用于执行特征选择，详见[L1-based feature selection](http://scikit-learn.org/stable/modules/feature_selection.html#l1-feature-selection)

**注意：随机稀疏化**

对于特征选择（feature selection）或稀疏求解（sparse recovery），有兴趣的可以使用：[Randomized sparse models.](http://scikit-learn.org/stable/modules/feature_selection.html#randomized-l1)

## 3.1 设置正则化参数

alpha参数控制着要估计的回归系数的稀疏度（degree of sparsity）。

### 使用交叉验证

sklearn提供了一些交叉验证对象来设置Lasso的alpha参数：[LassoCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV) 和 [LassoLarsCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV)。其中LassoLarsCV基于最小角回归算法。

对于高维数据集可能存在许多共线的回归，LassoCV表现通常很好。然而，LassoLarsCV的优点是，可以开发更多与alpha参数的相关值，如果样本数与观察到的数目相比非常小，通常比LassoCV更快。

图：在每个fold上的最小二乘误差：坐标下降法（训练时间：0.35s）

<figure>
    <a href="http://scikit-learn.org/stable/_images/plot_lasso_model_selection_0021.png"><img src="http://scikit-learn.org/stable/_images/plot_lasso_model_selection_0021.png" alt=""></a>
</figure>

图：在每个fold上的最小二乘误差：使用Lars（训练时间：0.17s）

<figure>
    <a href="http://scikit-learn.org/stable/_images/plot_lasso_model_selection_0021.png"><img src="http://scikit-learn.org/stable/_images/plot_lasso_model_selection_0021.png" alt=""></a>
</figure>

### 基于模型选择的信息准则

另一种方法：LassoLarsIC使用AIC(Akaike information criterion)和BIC(Bayes Information criterion)。它在寻找alpha的最优值时计算量更小，只需要一次，而当使用k-fold交叉验证时需要k+1次。。。。

# 4.Elastic Net

[ElasticNet](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet)是一个线性回归模型，同时使用L1和L2作为正则项。这种结合可以让你像Lasso那样从一个稀疏模型中学到少量非零权重，也能像Ridge模型那样仍然维持着正则化属性。我们使用l1_ratio参数来控制着L1和L2的凸结合。

当许多features彼此相关时，Elastic-net会很有用。Lasso则是从其中随机取一个，而elastic-net则选取所有的。

在Lasso和Ridge之间进行平衡的实际好处是，使得elastic-net可以继承Ridge在rotation上的稳定性。

目标函数如下：

<img src="http://www.forkosh.com/mathtex.cgi?\underset{w}{min\,} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha \rho ||w||_1 +
\frac{\alpha(1-\rho)}{2} ||w||_2 ^ 2}">

<figure>
    <a href="http://scikit-learn.org/stable/_images/plot_lasso_coordinate_descent_path_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_lasso_coordinate_descent_path_0011.png" alt=""></a>
</figure>

ElasticNetCV可以通过交叉验证来设置alpha参数（<img src="http://www.forkosh.com/mathtex.cgi?\alpha">）和l1_ratio参数（<img src="http://www.forkosh.com/mathtex.cgi?\rho">）。

示例：

- [Lasso and Elastic Net for Sparse Signals](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#example-linear-model-plot-lasso-and-elasticnet-py)
- [Lasso and Elastic Net](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html#example-linear-model-plot-lasso-coordinate-descent-path-py)


# 11.SGD



# 12. Perceptron

[Perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron)是另一个简单的算法，适用于大规模learning（large scale learning）。缺省的：

- 它不需要一个learning rate.
- 它没有正则项(不需要penalized)
- 只在错误时，才更新模型

Perceptron的最后一个特性暗示着，它比使用hinge loss的SGD要略微快一些。它会导致模型更稀疏。



参考：

1.[http://scikit-learn.org/stable/modules/linear_model.html](http://scikit-learn.org/stable/modules/linear_model.html)
