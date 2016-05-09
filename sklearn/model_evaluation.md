---
layout: page
title:  sklearn中的模型评估 
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

有三种不同的方法来评估一个模型的预测质量：

- estimator的score方法：sklearn中的estimator都具有一个score方法，它提供了一个缺省的评估法则来解决问题。
- Scoring参数：使用cross-validation的模型评估工具，依赖于内部的scoring策略。见下。
- Metric函数：metrics模块实现了一些函数，用来评估预测误差。见下。


# 2. scoring参数

模型选择和评估工具，例如： grid_search.GridSearchCV 和 cross_validation.cross_val_score，使用scoring参数来控制你的estimator的好坏。

## 2.1 预定义的值

对于大多数case而说，你可以设计一个使用scoring参数的scorer对象；下面展示了所有可能的值。所有的scorer对象都遵循：高得分，更好效果。如果从mean_absolute_error 和mean_squared_error（它计算了模型与数据间的距离）返回的得分将被忽略。

<figure>
	<a href="http://photo.yupoo.com/wangdren23/FvKhbUh3/medish.jpg"><img src="http://photo.yupoo.com/wangdren23/FvKhbUh3/medish.jpg" alt=""></a>
</figure>

## 2.2 从metric函数定义你的scoring策略

sklearn.metric提供了一些函数，用来计算真实值与预测值之间的预测误差：

- 以_score结尾的函数，返回一个最大值，越高越好
- 以_error结尾的函数，返回一个最小值，越小越好；如果使用make_scorer来创建scorer时，将greater_is_better设为False

接下去会讨论多种机器学习当中的metrics。

许多metrics并不出可以在scoring参数中配置的字符名，因为有时你可能需要额外的参数，比如：fbeta_score。这种情况下，你需要生成一个合适的scorer对象。最简单的方法是调用make_scorer来生成scoring对象。该函数将metrics转换成在模型评估中可调用的对象。

第一个典型的用例是，将一个库中已经存在的metrics函数进行包装，使用定制参数，比如对fbeta_score函数中的beta参数进行设置：

{% highlight python %}

>>> from sklearn.metrics import fbeta_score, make_scorer
>>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
>>> from sklearn.grid_search import GridSearchCV
>>> from sklearn.svm import LinearSVC
>>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=ftwo_scorer)

{% endhighlight %}

第二个典型用例是，通过make_scorer构建一个完整的定制scorer函数，该函数可以带有多个参数：

- 你可以使用python函数：下例中的my_custom_loss_func
- python函数是否返回一个score（greater_is_better=True），还是返回一个loss（greater_is_better=False）。如果为loss，python函数的输出将被scorer对象忽略，根据交叉验证的原则，得分越高模型越好。
- 对于分类的metrics：该函数如果希望处理连续值（needs_threshold=True）。缺省值为False。
- 一些额外的参数：比如f1_score中的bata或labels。

下例使用定制的scorer，使用了greater_is_better参数：

{% highlight python %}

>>> import numpy as np
>>> def my_custom_loss_func(ground_truth, predictions):
...     diff = np.abs(ground_truth - predictions).max()
...     return np.log(1 + diff)
...

>>> loss  = make_scorer(my_custom_loss_func, greater_is_better=False)
>>> score = make_scorer(my_custom_loss_func, greater_is_better=True)
>>> ground_truth = [[1, 1]]
>>> predictions  = [0, 1]
>>> from sklearn.dummy import DummyClassifier
>>> clf = DummyClassifier(strategy='most_frequent', random_state=0)
>>> clf = clf.fit(ground_truth, predictions)
>>> loss(clf,ground_truth, predictions) 
-0.69...
>>> score(clf,ground_truth, predictions) 
0.69...

{% endhighlight %}

# 2.3 实现你自己的scoring对象

你可以生成更灵活的模型scorer，通过从头构建自己的scoring对象来完成，不需要使用make_scorer工厂函数。对于一个自己实现的scorer来说，它需要遵循两个原则：

- 必须可以用(estimator, X, y)进行调用
- 必须返回一个float的值

# 3. 分类metrics

sklearn.metrics模块实现了一些loss, score以及一些工具函数来计算分类性能。一些metrics可能需要正例、置信度、或二分决策值的的概率估计。大多数实现允许每个sample提供一个对整体score来说带权重的分布，通过sample_weight参数完成。

一些二分类(binary classification)使用的case：

- matthews_corrcoef(y_true, y_pred)
- precision_recall_curve(y_true, probas_pred)
- roc_curve(y_true, y_score[, pos_label, ...])

一些多分类(multiclass)使用的case：

- confusion_matrix(y_true, y_pred[, labels])
- hinge_loss(y_true, pred_decision[, labels, ...])

一些多标签(multilabel)的case: 

- accuracy_score(y_true, y_pred[, normalize, ...])
- classification_report(y_true, y_pred[, ...])
- f1_score(y_true, y_pred[, labels, ...])
- fbeta_score(y_true, y_pred, beta[, labels, ...])
- hamming_loss(y_true, y_pred[, classes])
- jaccard_similarity_score(y_true, y_pred[, ...])
- log_loss(y_true, y_pred[, eps, normalize, ...])
- precision_recall_fscore_support(y_true, y_pred)
- precision_score(y_true, y_pred[, labels, ...])
- recall_score(y_true, y_pred[, labels, ...])
- zero_one_loss(y_true, y_pred[, normalize, ...])

还有一些可以同时用于二标签和多标签（不是多分类）问题：

- average_precision_score(y_true, y_score[, ...])
- roc_auc_score(y_true, y_score[, average, ...])

在以下的部分，我们将讨论各个函数。

# 3.1 二分类/多分类/多标签

对于二分类来说，必须定义一些matrics（f1_score，roc_auc_score）。**在这些case中，缺省只评估正例的label，缺省的正例label被标为1**（可以通过配置pos_label参数来完成）

将一个二分类matrics拓展到多分类或多标签问题时，我们可以将数据看成多个二分类问题的集合，每个类都是一个二分类。接着，我们可以通过跨多个分类计算每个二分类metrics得分的均值，这在一些情况下很有用。你可以使用average参数来指定。

- macro：计算二分类metrics的均值，为每个类给出相同权重的分值。当小类很重要时会出问题，因为该macro-averging方法是对性能的平均。另一方面，该方法假设所有分类都是一样重要的，因此macro-averaging方法会对小类的性能影响很大。
- weighted: 对于不均衡数量的类来说，计算二分类metrics的平均，通过在每个类的score上进行加权实现。
- micro： 给出了每个样本类以及它对整个metrics的贡献的pair（sample-weight），而非对整个类的metrics求和，它会每个类的metrics上的权重及因子进行求和，来计算整个份额。Micro-averaging方法在多标签（multilabel）问题中设置，包含多分类，此时，大类将被忽略。
- samples：应用在 multilabel问题上。它不会计算每个类，相反，它会在评估数据中，通过计算真实类和预测类的差异的metrics，来求平均（sample_weight-weighted）
- average：average=None将返回一个数组，它包含了每个类的得分. 

多分类（multiclass）数据提供了metric，和二分类类似，是一个label的数组，**而多标签（multilabel）数据则返回一个索引矩阵，当样本i具有label j时，元素[i,j]的值为1，否则为0**.

# 3.2 accuracy_score

accuracy_score函数计算了准确率，不管是正确预测的fraction（default），还是count(normalize=False)。

在multilabel分类中，该函数会返回子集的准确率。如果对于一个样本来说，必须严格匹配真实数据集中的label，整个集合的预测标签返回1.0；否则返回0.0.

预测值与真实值的准确率，在n个样本下的计算公式如下：

<figure>
	<a href="http://scikit-learn.org/stable/_images/math/27e20bf0b2786124f8df6383493b347e6ce8586d.png"><img src="http://scikit-learn.org/stable/_images/math/27e20bf0b2786124f8df6383493b347e6ce8586d.png" alt=""></a>
</figure>

1(x)为指示函数。

{% highlight python %}

>>> import numpy as np
>>> from sklearn.metrics import accuracy_score
>>> y_pred = [0, 2, 1, 3]
>>> y_true = [0, 1, 2, 3]
>>> accuracy_score(y_true, y_pred)
0.5
>>> accuracy_score(y_true, y_pred, normalize=False)
2

{% endhighlight %}

在多标签的case下，二分类label：

{% highlight python %}

>>> accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
0.5

{% endhighlight %}

# 3.3 Cohen’s kappa

函数cohen_kappa_score计算了Cohen’s kappa估计。这意味着需要比较通过不同的人工标注（numan annotators）的标签，而非分类器中正确的类。

kappa score是一个介于(-1, 1)之间的数. score>0.8意味着好的分类；0或更低意味着不好（实际是随机标签）

Kappa score可以用在二分类或多分类问题上，但不适用于多标签问题，以及超过两种标注的问题。

# 3.4 混淆矩阵

[confusion_matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)函数通过计算混淆矩阵，用来计算分类准确率。

缺省的，在混淆矩阵中的i,j指的是观察的数目i，预测为j，示例：

{% highlight python %}

>>> from sklearn.metrics import confusion_matrix
>>> y_true = [2, 0, 2, 2, 0, 1]
>>> y_pred = [0, 0, 2, 2, 0, 2]
>>> confusion_matrix(y_true, y_pred)
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])

{% endhighlight %}

结果为：

# 3.5 分类报告

[classification_report](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)函数构建了一个文本报告，用于展示主要的分类metrics。 下例给出了一个小示例，它使用定制的target_names和对应的label：
 
{% highlight python %}
 
>>> from sklearn.metrics import classification_report
>>> y_true = [0, 1, 2, 2, 0]
>>> y_pred = [0, 0, 2, 2, 0]
>>> target_names = ['class 0', 'class 1', 'class 2']
>>> print(classification_report(y_true, y_pred, target_names=target_names))
             precision    recall  f1-score   support

    class 0       0.67      1.00      0.80         2
    class 1       0.00      0.00      0.00         1
    class 2       1.00      1.00      1.00         2

avg / total       0.67      0.80      0.72         5

{% endhighlight %}
 
示例：
 
- [识别手写数字示例](http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#example-classification-plot-digits-classification-py)
- [使用sparse特征的文本分类](http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py)
- [使用grid search的cross-validation的参数估计](http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#example-model-selection-grid-search-digits-py)

# 3.6 Hamming loss
 
[hamming_loss](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss)计算了在两个样本集里的平均汉明距离或平均Hamming loss。
 
- <img src="http://www.forkosh.com/mathtex.cgi?\hat{y}_j ">是对应第j个label的预测值，
- <img src="http://www.forkosh.com/mathtex.cgi?y_j ">是对应的真实值
- <img src="http://www.forkosh.com/mathtex.cgi?n_\text{labels} ">是类目数

那么两个样本间的Hamming loss为<img src="http://www.forkosh.com/mathtex.cgi?L_{Hamming} ">，定义如下：

<img src="http://www.forkosh.com/mathtex.cgi?L_{Hamming}(y, \hat{y}) = \frac{1}{n_\text{labels}} \sum_{j=0}^{n_\text{labels} - 1} 1(\hat{y}_j \not= y_j)">

其中：<img src="http://www.forkosh.com/mathtex.cgi?1(x) ">为指示函数。

{% highlight python %}

>>> from sklearn.metrics import hamming_loss
>>> y_pred = [1, 2, 3, 4]
>>> y_true = [2, 2, 3, 4]
>>> hamming_loss(y_true, y_pred)
0.25

{% endhighlight %}

在多标签（multilabel）的使用二元label指示器的情况：

{% highlight python %}

>>> hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2)))
0.75

{% endhighlight %}

注意：在多分类问题上，Hamming loss与y_true 和 y_pred 间的Hamming距离相关，它与[0-1 loss](http://scikit-learn.org/stable/modules/model_evaluation.html#zero-one-loss)相类似。然而，0-1 loss会对不严格与真实数据集相匹配的预测集进行惩罚。因而，Hamming loss，作为0-1 loss的上界，也在0和1之间；预测一个合适的真实label的子集或超集将会给出一个介于0和1之间的Hamming loss.

# 3.7 Jaccard相似度系数score

[jaccard_similarity_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html#sklearn.metrics.jaccard_similarity_score)函数会计算两对label集之间的Jaccard相似度系数的平均（缺省）或求和。它也被称为Jaccard index.

第i个样本的Jaccard相似度系数（Jaccard similarity coefficient），真实标签集为<img src="http://www.forkosh.com/mathtex.cgi?y_i ">，预测标签集为：<img src="http://www.forkosh.com/mathtex.cgi?\hat{y}_j ">，其定义如下：

<img src="http://www.forkosh.com/mathtex.cgi?J(y_i, \hat{y}_i) = \frac{|y_i \cap \hat{y}_i|}{|y_i \cup \hat{y}_i|}.">

在二分类和多分类问题上，Jaccard相似度系数score与分类的正确率（accuracy）相同：

{% highlight python %}

>>> import numpy as np
>>> from sklearn.metrics import jaccard_similarity_score
>>> y_pred = [0, 2, 1, 3]
>>> y_true = [0, 1, 2, 3]
>>> jaccard_similarity_score(y_true, y_pred)
0.5
>>> jaccard_similarity_score(y_true, y_pred, normalize=False)
2

{% endhighlight %}

在多标签（multilabel）问题上，使用二元标签指示器：

{% highlight python %}

>>> jaccard_similarity_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
0.75

{% endhighlight %}

