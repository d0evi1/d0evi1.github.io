---
layout: page
title: sklearn中的特征提取
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

sklearn.feature_selection模块，用于在样本集中进行特征选择和降维，改善estimators的准确率，或者提升它们在高维数据集上的效果。


# 2.移除低variance的特征

VarianceThreshold是一个简单的用于特征提取的baseline方法。它将移除所有variance不满足一些阀值的特征。缺省情况下，它会移除所有0-variance的特征（表示该feature下具有相同值）。

下面是一个示例，假设我们具有一个数据集，它具有多个boolean类型的features，我们希望移除这样的features：样本中80%都是1或者都是0（on或者off）的列。**boolean的features是Bernoulli随机变量**，这些变量的variance为：

<figure>
    <a href="http://scikit-learn.org/stable/_images/math/9dbb4af1af56391f18aa7463719c2a5173920eb4.png"><img src="http://scikit-learn.org/stable/_images/math/9dbb4af1af56391f18aa7463719c2a5173920eb4.png" alt=""></a>
</figure>

因此，我们可以使用阀值 0.8 * (1 - 0.8)：

{% highlight python %}

>>> from sklearn.feature_selection import VarianceThreshold
>>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
>>> sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
>>> sel.fit_transform(X)
array([[0, 1],
       [1, 0],
       [0, 0],
       [1, 1],
       [1, 0],
       [1, 1]])

{% endhighlight %}

正如我们期望的，VarianceThreshold将移除第一列，该列5/6>0.8以上的数据为0.

# 3. 单变量特征选择（Univariate Feature Selection）

单变量特征选择，通过单变量统计检验(univariate statistical tests)来选取最佳参数。它可以看成是estimator的预处理阶段。Scikit-learn封装了一些特征选择方法对象，实现了transform方法来完成：

- [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest)：留下topK高分的features
- [SelectPercentile](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile)：留下百分比的top高分features
- 为每个feature都使用常用的单变量统计检验（univariate statistical tests）：
 - a.[SelectFpr](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html#sklearn.feature_selection.SelectFpr)（FPR: false positive rate，假阳，即负正本判为正），
 - b.[SelectFdr](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html#sklearn.feature_selection.SelectFdr)（FDR: false discovery rate，伪发现率），
 - c.[SelectFwe](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html#sklearn.feature_selection.SelectFwe)（FWER: family wise error，多重比较谬误）
- [GenericUnivariateSelect](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html#sklearn.feature_selection.GenericUnivariateSelect)：可以使用一个可配置的策略来进行单变量特征选择（univariate feature selection）。它允许你在超参数查找中选择最好的单变量选择策略。

示例，我们使用一个卡方检验<img src="http://www.forkosh.com/mathtex.cgi?\chi^2"> 来抽取两个最佳特征：

{% highlight python %}

>>> from sklearn.datasets import load_iris
>>> from sklearn.feature_selection import SelectKBest
>>> from sklearn.feature_selection import chi2
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> X.shape
(150, 4)
>>> X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
>>> X_new.shape
(150, 2)

{% endhighlight %}

下面的对象作为scoring函数，它们将返回单变量的p-values：

- 对于回归：[f_regression](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression)
- 对于分类：[chi2](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2) 或者 [f_classif](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif)

## 稀疏数据的特征选择

如果你使用sparse数据（比如：数据表示使用sparse matrics），如果不对它作dense转换，那么只有chi2 适合处理这样的数据。

**注意：如果在分类问题上使用回归的scoring函数，你将得到无用的结果。**

总结：

- f_classif: 在label/feature之间的方差分析(Analysis of Variance:ANOVA) 的F值，用于分类.
- chi2: 非负feature的卡方检验, 用于分类.
- f_regression: 在label/feature之间的F值，用于回归.
- SelectKBest: 得到k个最高分的feature.
- SelectFpr: 基于RPR（false positive rate）检验。
- SelectFdr: 使用Benjamini-Hochberg过程。选择p值（false discovery rate）。
- SelectFwe: 
- GenericUnivariateSelect: 可配置化的单变量特征选择器.


# 4.递归特征淘汰（RFE）

给定一个外部的estimator，为feature分配权重（例如：线性模型的相关系数coefficients），递归特征淘汰（[RFE](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE)）通过递归将feature集越来越小。首先，estimator先在一个初始的feature集上进行训练，为每个feature分配权重。接着，绝对权重最小的features将从当前集中移除。在剩下的数据集下，重复该过程，直到features数达到目标期望值。

PFECV会在一个cross-validation循环上执行RFE查找最优的feature数目。

示例：

- [Recursive feature elimination](http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#example-feature-selection-plot-rfe-digits-py)
- [Recursive feature elimination with cross-validation](http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#example-feature-selection-plot-rfe-with-cross-validation-py)

# 5.使用SelectFromModel进行特征选择

SelectFromModel是一个元转换器（meta-transformer），可以用在任何在fitting后具有coef_或feature_importances_属性的estimator。如果相应的coef_ 或 feature_importances_值在提供的参数threshold之下，那么这些不重要的features将被移除。除了指定一个数值型的threshold，还内置了些string参数作为阀值的探索法（heuristics）。这些heuristics方法有："mean", "median"以及浮点乘法（比如："0.1*mean"）

示例：

- [Feature selection using SelectFromModel and LassoCV](http://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_boston.html#example-feature-selection-plot-select-from-model-boston-py)


## 5.1 L1-based特征选择

使用L1正则化作为惩罚项的线性模型，具有稀疏方式的解决方案：要估计的相关系数许多都为0。使用另一个分类器，再配合使用feature_selection.SelectFromModel选择非0系数，来达到降维的目标。特别的，稀疏方式的estimators对于线性回归中的linear_model.Lasso很有用，对于分类中的linear_model.LogisticRegression and svm.LinearSVC很有用。

{% highlight python %}
>>> from sklearn.svm import LinearSVC
>>> from sklearn.datasets import load_iris
>>> from sklearn.feature_selection import SelectFromModel
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> X.shape
(150, 4)
>>> lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
>>> model = SelectFromModel(lsvc, prefit=True)
>>> X_new = model.transform(X)
>>> X_new.shape
(150, 3)

{% endhighlight %}


对于SVM和logistic回归来说，参数C控制着稀疏性：C越小，选择到的features就越少。而对于Lasso，alpha的值越大，则选择到的features越少。

示例：

- [Classification of text documents using sparse features](http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py)


L1-recovery与压缩感知（compressive sensing）

只要满足特定条件，选择好的alpha，可以让Lasso通过少量的观察值就可以完整地恢复非零变量的完整集。特别的，训练样本数可以足够大，否则L1模型将随机执行，足够大依赖于非零系数的数目、features数目的取log、噪声的数量、非零系数的最小绝对值、设计矩阵X(design matrix)的结构。另外，设计矩阵必须可以展示特定的属性，比如不能太相关。

对于非零相关系数的恢复，没有通用规则来选择alpha参数。可以通过交叉验证（LassoCV 或 LassoLarsCV）来设置，尽管这将导致惩罚不够（under-penalized）的模型：包含少量的非相关变量，对于预测结果不利。相反地，BIC（LassoLarsIC）将设置高的alpha值。

详见：[Richard G. Baraniuk “Compressive Sensing”, IEEE Signal Processing Magazine [120] July 2007](http://dsp.rice.edu/files/cs/baraniukCSlecture07.pdf)

## 5.2 随机sparse模型

L1-based sparse模型的局限是，常在一组非常相关的features中，选择一个feature。为了缓解这个问题，可以使用随机化技术（randomization techniques），对sparse模型进行再估计多次，扰头设计矩阵；或者对数据进行子抽样，统计选择的回归需要多少次子抽样。

对于回归问题，RandomizedLasso使用Lasso来实现策略；而当分类问题时，RandomizedLogisticRegression使用logistic回归。为了得到稳定分值的所有路径，可以使用lasso_stability_path.

<figure>
    <a href="http://scikit-learn.org/stable/_images/plot_sparse_recovery_0031.png"><img src="http://scikit-learn.org/stable/_images/plot_sparse_recovery_0031.png" alt=""></a>
</figure>

注意，在检测非零features时，对于随机的sparse模型，它比标准的F检验（F statistics）更强大。也就是说，正例（the ground truth）的模型可以为sparse，它在features中只有很小一部分为非零。

示例：

- [Sparse recovery: feature selection for sparse linear models](http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_recovery.html#example-linear-model-plot-sparse-recovery-py)

## 5.3 基于树的特征选择

基于树的estimators（树：sklearn.tree、森林：sklearn.ensemble模型），可以用来计算feature的重要性，它可以用于抛弃掉那些不相关的features（sklearn.feature_selection.SelectFromModel一起使用）

{% highlight python %}

>>> from sklearn.ensemble import ExtraTreesClassifier
>>> from sklearn.datasets import load_iris
>>> from sklearn.feature_selection import SelectFromModel
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> X.shape
(150, 4)
>>> clf = ExtraTreesClassifier()
>>> clf = clf.fit(X, y)
>>> clf.feature_importances_  
array([ 0.04...,  0.05...,  0.4...,  0.4...])
>>> model = SelectFromModel(clf, prefit=True)
>>> X_new = model.transform(X)
>>> X_new.shape               
(150, 2)

{% endhighlight %}

示例：

- [Feature importances with forests of trees](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py)
- [Pixel importances with a parallel forest of trees](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances_faces.html#example-ensemble-plot-forest-importances-faces-py)

# 6.Pipeline的特征选择

特征选择通常用在数据预处理阶段。我们可以在sklearn.pipeline.Pipeline中进行处理：

{% highlight python %}

clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)

{% endhighlight %}

在该代码段内，我们使用了 sklearn.svm.LinearSVC 和 sklearn.feature_selection.SelectFromModel来评估feature的重要性，并选择最相关的features。接着，使用sklearn.ensemble.RandomForestClassifier 对这些features进行训练和转换结果。你可以执行其它的特征选择方法，并执行相同的操作来评估feature的重要性。详见：[sklearn.pipeline.Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline)


参考：

1.[http://scikit-learn.org/stable/modules/feature_selection.html](http://scikit-learn.org/stable/modules/feature_selection.html)
