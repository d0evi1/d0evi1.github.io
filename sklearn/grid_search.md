---
layout: page
title: sklearn中的GridSearch
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

参数并非从estimators中直接学到的，可以通过设置一个参数搜索空间来找到最佳的cross-validation score。通常示例包括的参数有：SVM分类器的中C、kernel和gamma，Lasso中的alpha等。

当构建一个estimator时，提供的参数可以以这种方式进行优化。更特别的是，可以使用如下的方法来给给定estimator的所有参数来找到对应的参数名和当前值：

{% highlight python %}

estimator.get_params()

{% endhighlight %}

这些参数称被提到：“超参数（hyperparameters）”，尤其在Bayesian learning中，它们与机器学习过程中的参数优化是有区别的。

一个这样的参数search包含：

- 一个estimator(regressor/classifier)
- 一个参数空间
- 一个用于searching/sampling候选参数的方法
- 一个cross-validation的scheme
- 一个score function

这样的模型允许你指定有效的搜索参数策略，[如下](http://scikit-learn.org/stable/modules/grid_search.html#alternative-cv)。在sklearn中，有两种通用方法进行sampling搜索候选参数：

- GridSearch: 暴力搜索所有参数组合
- RandomizedSearchCV: 在指定参数空间内抽样一部分候选参数

# 2.GridSearch

grid search提供了GridSearchCV，相应的参数空间param_grid设置如下：

{% highlight python %}

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

{% endhighlight %}

上例指定了两个要搜索的参数空间：一个是线性kernel，其中C值为[1,10,100,1000]；另一个则使用RBF kernel，对应的C值为[1,10,100,1000]，对应的gamma值为 [0.001, 0.0001].

GridSearchCV实例实现了通用的estimator API: 当在数据集的所有可能参数组合上进行"fitting"时，所有参数组都会被评测，并保留最优的参数组合。

示例：

- [Parameter estimation using grid search with cross-validation](http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#example-model-selection-grid-search-digits-py)
- [Sample pipeline for text feature extraction and evaluation ](http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#example-model-selection-grid-search-text-feature-extraction-py)

# 3.随机参数优化

**使用GridSearch进行参数搜索是目前最广泛使用的参数优化方法**，还有另一些方法存在。[RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.RandomizedSearchCV.html#sklearn.grid_search.RandomizedSearchCV)实现了在参数上的随机搜索，每个设置都会以可能的参数值分布进行抽样。对比穷举法，它具有两个优势：

- 1.budget的选择与参数个数和可能的值独立
- 2.增加参数不会影响性能，不会降低效果

参数设定部分和GridSearchCV类似，使用一个字典表来进行参数抽样。另外，计算开销（computation budget）, 抽取的样本数，抽样迭代次数，可以由n_iter来指定。对于每个参数，都可以指定在可能值上的分布，或者是一个离散值列表（它可以被均匀采样）。

例如：
{% highlight python %}
[{‘C’: scipy.stats.expon(scale=100), 
‘gamma’: scipy.stats.expon(scale=.1), 
‘kernel’: [‘rbf’], 
‘class_weight’:[‘auto’, None]}]

{% endhighlight %}

这个例子使用scipy.stats模块，该模块包含了许多分布方法可以用来进行抽样，包括：指数分布（expon），gamma分布(gamma)，均匀分布（uniform），或randint分布。通常每个函数都可以提供一个rvs（随机变量抽样）方法进行抽样。

注意：

scipy.stats的分布不允许以随机方式指定。作为替代，我们可以使用一个全局的numpy 随机态，它可以通过np.random.seed或np.random.set_state来设定。

对于连续的参数，比如上面的C，指定一个连续的分布十分重要，它可以完全利用随机化（randomization）。这种情况下，增加n_iter将产生一个更好的搜索。

示例：

- [Comparing randomized search and grid search for hyperparameter estimation ](http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#example-model-selection-randomized-search-py)

# 4.参数搜索tips

## 4.1 指定一个目标metric

缺省的，参数搜索会使用estimator的缺省score函数来评估参数设置。其中，分类使用[sklearn.metrics.accuracy_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)，回归使用[sklearn.metrics.r2_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)。对于其它应用，可能需要使用一个可合适的scoring函数（例如：对于unbalanced分类问题，accuracy的score是不合适的）。可选择的scoring函数可以通过GridSearchCV/RandomizedSearchCV以及其它CV工具类的scoring参数来设置。[详见](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)。

## 4.2 将estimators与参数空间组合

详见：[Pipeline: chaining estimators](http://scikit-learn.org/stable/modules/pipeline.html#pipeline)

## 4.3 模型选择：开发集与评测集

通过评估多种参数设置来进行模型选择，可以认为是使用labeled数据集来训练这些参数空间。

当评估产生的模型时，在留存样本（held-out samples）上做模型评测，不会在参数搜索过程看到：推荐你将数据划分成两部分：

- 1.开发集（development set）：对它进行GridSearchCV
- 2.评测集（evaluation set）：计算性能metrics

可以通过cross_validation.train_test_split来进行划分。

## 4.4 并列化

**n_jobs参数**进行设置。

## 4.5 容错

一些参数设置可能会导致fit1或多个folds的数据时失败。缺省的，它会引起整个搜索的失败，即使有些参数设置已经被评测过了。通过设置**error_score=0 (or =np.NaN)**, 可以让该过程更加具有容错性，对于那个存在0(或NaN)的fold数据集来说会继续进行下去。

# 5.可选择的其它暴力参数搜索（brute force parameter search）

# 5.1 模型指定的cv

一些模型可以在一些参数值范围内拟合数据，与单个参数值的拟合一样有效。这种特性可以执行一个更有效的cv来进行参数的模型选择。

一种最常用的参数策略的方式是，将正则项参数化。这种情况下，我们可以计算estimator的**正则化path（regularization path）**。

模型如下：

- linear_model.ElasticNetCV
- linear_model.LarsCV
- linear_model.LassoCV
- linear_model.LassoLarsCV
- linear_model.LogisticRegressionCV
- linear_model.MultiTaskElasticNetCV
- linear_model.MultiTaskLassoCV
- linear_model.OrthogonalMatchingPursuitCV
- linear_model.RidgeCV
- linear_model.RidgeClassifierCV

## 5.2 Information Criterion

一些模型经常提供一个information-theoretic、closed-form的公式来进行正则项参数的估计优化，通过计算单个正则项path（而非使用cv）。

比如：

- linear_model.LassoLarsIC

## 5.3 带外估计（Out of Bag Estimates）

当我们使用基于bagging的ensemble方法时，比如：使用有放回抽样来生成新的数据集，训练集中的部分仍然是见不到的。对于ensemble中的每个分类器，训练集都会遗留下另一部分数据。

这部分遗留下来的数据，可以被用于估计泛化错误（generalization error），而无需依赖于一个独立的验证集。这种估计是“免费（for free）”的，因为，不需要额外的数据就可以进行模型选择。

这些类当中实现了该方法：

- ensemble.RandomForestClassifier
- ensemble.RandomForestRegressor
- ensemble.ExtraTreesClassifier
- ensemble.ExtraTreesRegressor
- ensemble.GradientBoostingClassifier
- ensemble.GradientBoostingRegressor


参考：

1.[http://scikit-learn.org/stable/modules/grid_search.html](http://scikit-learn.org/stable/modules/grid_search.html)
