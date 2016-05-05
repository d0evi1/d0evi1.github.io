---
layout: page
title: sklearn中的logistic回归
tagline: 介绍
---
{% include JB/setup %}

# 1.logistic回归

Logistic回归是一个线性分类模型，而不是回归。Logistic回归在一些文献中也称为：logit回归，最大熵分类（MaxEnt）或者log线性分类器。在这些模型中，相应的概率描述了使用[logistic函数](http://en.wikipedia.org/wiki/Logistic_function)建模的实验结果的出现可能性。

logistic回归在sklearn中的实现可以通过[LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)类进行访问。该实现可以对多分类（one-vs-rest）logistic回归进行拟合，正则项为：L2或L1范式。

作为一个优化问题，二分类L2惩罚项的logistic回归的cost function如下：


<figure>
    <a href="http://scikit-learn.org/stable/_images/math/96fe247fe9465d26af15706141dc22e598ac7826.png"><img src="http://scikit-learn.org/stable/_images/math/96fe247fe9465d26af15706141dc22e598ac7826.png" alt=""></a>
</figure>

使用L1范式正则项的logistic回归的最优化问题为：

<figure>
    <a href="http://scikit-learn.org/stable/_images/math/3fb9bab302e67df4a9f00b8df259d326e01837fd.png"><img src="http://scikit-learn.org/stable/_images/math/3fb9bab302e67df4a9f00b8df259d326e01837fd.png" alt=""></a>
</figure>

LogisticRegression的实现方式有：“liblinear”（c++封装库，LIBLINEAR），“newton-cg”, “lbfgs” 和 “sag”。

"lbfgs” 和 “newton-cg”只支持L2罚项，对于高维数据集来说收敛更快。L1罚项会产生稀疏预测权重。

“liblinear” 基于Liblinear使用坐标下降法（CD）。对于L1罚项，sklearn.svm.l1_min_c允许C更低的界，以便得到一个非null（所有feature的权重为0）的模型。这依赖于优秀的开源库[LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/)，在sklearn中内置了。然而，在liblinear中实现的坐标下降法(CD)不能学习一个真实的multinomial（multiclass）模型；相反地，该优化问题是以“one-vs-rest”的方式进行分解，以便所有的分类都可以以二分类的方式进行训练。底层会这么去实现，LogisticRegression实例使用这种方式来解决多分类问题。

在LogisticRegression中，将multi_class参数设置成“multinomial”，并且将solver参数设置成“lbfgs” 或 “newton-cg”，将学习到一个真实的multinomial 的LR模型，这意味着它的概率估计比缺省的“one-vs-rest”设置得到更好地校正。“lbfgs”, “newton-cg” 和 “sag”这些solver不能优化L1罚项的模型，因此， “multinomial” 的设置不能学习稀疏模型。

“sag”的slover使用随机平均梯度下降法（SAGD: Stochastic Average Gradient descent）。它不能处理“multinomial”的case，它被限制在L2罚项的模型上，当在样本数很大，并且feature数也很大的大数据集上时，sag通常比其它solver更快。

你可以选择如下的solover：

| case             | solver           |
|------------------|------------------|
| 小数据集或L1罚项 | liblinear        |
| Multinomial loss | lbfgs或newton-cg |
| 大数据集         | sag              |

对于大数据集，你可以考虑使用SGDClassifier，并使用'log' loss。

示例：

- [L1 Penalty and Sparsity in Logistic Regression](http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html#example-linear-model-plot-logistic-l1-l2-sparsity-py)
- [Path with L1- Logistic Regression](http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html#example-linear-model-plot-logistic-path-py)

**与liblinear的不同点**：

使用 solver=liblinear的 LogisticRegression  或 LinearSVC 所获得的分值，与直接使用liblinear库的分值是不同的，当fit_intercept=False时，拟合的coef_ 或者要预测的数据将会为0。这是因为，对于decision_function为0的样本来说，LogisticRegression 和 LinearSVC预测负类（negative class），而liblinear预测的是正类（positive class），注意，如果fit_intercept=False的模型具有许多decision_function=0的样本，这个模型很可能是欠拟合的坏模型，我们建议你设置fit_intercept=True，并增加intercept_scaling的值。

**注意：使用稀疏logistic回归的特征选择**

使用L1罚项的LR会产生稀疏模型，因而可以用于特征选择，详见[L1-based feature selection](http://scikit-learn.org/stable/modules/feature_selection.html#l1-feature-selection)

LogisticRegressionCV 使用了内置的LR的交叉验证，用于找到最优的C参数。由于warm-starting，“newton-cg”, “sag” and “lbfgs” 的solver对于高维dense数据集执行较快。对于多分类问题，如果multi_class参数设置为“ovr”，对于每个类都获得一个最优的C；如果multi_class设置为"multinomial", 将获得一个最优的C，它使得交叉熵的loss（corss-entropy loss）最小。

# 2.随机梯度下降法(SGD)

随机梯度下降法对于拟合线性模型来说非常简单有效。当样本数很大（且feature数很大）时，这种方法特别有用。partial_fit方法允许你进行only/out-of-core学习。

[SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) 和 [SGDRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor) 分别为分类和回归提供了用来拟合线性模型的功能，可以使用不同的（convex）loss function和不同的罚项。例如：使用loss="log"，SGDClassifier可以拟合一个logistic回归模型，而使用loss="hinge"它可以拟合一个线性的SVM模型。

参考：

- [Stochastic Gradient Descent](http://scikit-learn.org/stable/modules/sgd.html#sgd)


参考：

1.[http://scikit-learn.org/stable/modules/linear_model.html](http://scikit-learn.org/stable/modules/linear_model.html#logisticregession)
