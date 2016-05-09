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

# 3.8 准确率，召回率与F值

准确率（precision）可以衡量一个样本为负的标签被判成正，召回率（recall）用于衡量所有正例。

F-meature（包括：<img src="http://www.forkosh.com/mathtex.cgi?F_\beta">和<img src="http://www.forkosh.com/mathtex.cgi?F_1">），可以认为是precision和recall的加权调和平均（weighted harmonic mean）。一个<img src="http://www.forkosh.com/mathtex.cgi?F_\beta">值，最佳为1，最差时为0. 如果<img src="http://www.forkosh.com/mathtex.cgi?\beta">=1，那么<img src="http://www.forkosh.com/mathtex.cgi?F_\beta">和<img src="http://www.forkosh.com/mathtex.cgi?F_1">相等，precision和recall的权重相等。

[precision_recall_curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)会根据预测值和真实值来计算一条precision-recall典线。

[average_precision_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)则会预测值的平均准确率（AP: average precision）。该分值对应于precision-recall曲线下的面积。

sklearn提供了一些函数来分析precision, recall and F-measures值：

- average_precision_score：计算预测值的AP
- f1_score: 计算F1值，也被称为平衡F-score或F-meature
- fbeta_score: 计算F-beta score
- precision_recall_curve：计算不同概率阀值的precision-recall对
- precision_recall_fscore_support：为每个类计算precision, recall, F-measure 和 support
- precision_score： 计算precision
- recall_score：    计算recall

**注意：precision_recall_curve只用于二分类中。而average_precision_score可用于二分类或multilabel指示器格式**

示例：

- [使用sparse特性的文档分类](http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py)
- [使用grid search corss-validation的参数估计](http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#example-model-selection-grid-search-digits-py)
- [Precision-Recall](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#example-model-selection-plot-precision-recall-py)
- [Sparse recovery: feature selection for sparse linear models](http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_recovery.html#example-linear-model-plot-sparse-recovery-py)

## 3.8.1 二分类

在二元分类中，术语“positive”和“negative”指的是分类器的预测类别(expectation)，术语“true”和“false”则指的是预测是否正确（有时也称为：观察observation）。给出如下的定义：


|          |           实际类目（observation）           |  |
|----------|:-------------:|------:|
|预测类目（expectation）  |  TP(true positive)结果:Correct | FP(false postive)结果：Unexpected |
|  |        FN(false negative)结果: Missing   |   TN(true negtive)结果：Correct |

在这个上下文中，我们定义了precision, recall和F-measure:

<img src="http://www.forkosh.com/mathtex.cgi?\text{precision} = \frac{tp}{tp + fp}">

<img src="http://www.forkosh.com/mathtex.cgi?\text{recall} = \frac{tp}{tp + fn}">

<img src="http://www.forkosh.com/mathtex.cgi?F_\beta = (1 + \beta^2) \frac{\text{precision} \times \text{recall}}{\beta^2 \text{precision} + \text{recall}}">

这里是一个二元分类的示例：

{% highlight python %}

>>> from sklearn import metrics
>>> y_pred = [0, 1, 0, 0]
>>> y_true = [0, 1, 0, 1]
>>> metrics.precision_score(y_true, y_pred)
1.0
>>> metrics.recall_score(y_true, y_pred)
0.5
>>> metrics.f1_score(y_true, y_pred)  
0.66...
>>> metrics.fbeta_score(y_true, y_pred, beta=0.5)  
0.83...
>>> metrics.fbeta_score(y_true, y_pred, beta=1)  
0.66...
>>> metrics.fbeta_score(y_true, y_pred, beta=2) 
0.55...
>>> metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5)  
(array([ 0.66...,  1.        ]), array([ 1. ,  0.5]), array([ 0.71...,  0.83...]), array([2, 2]...))


>>> import numpy as np
>>> from sklearn.metrics import precision_recall_curve
>>> from sklearn.metrics import average_precision_score
>>> y_true = np.array([0, 0, 1, 1])
>>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
>>> precision, recall, threshold = precision_recall_curve(y_true, y_scores)
>>> precision  
array([ 0.66...,  0.5       ,  1.        ,  1.        ])
>>> recall
array([ 1. ,  0.5,  0.5,  0. ])
>>> threshold
array([ 0.35,  0.4 ,  0.8 ])
>>> average_precision_score(y_true, y_scores)  
0.79...

{% endhighlight %}

## 3.8.2 多元分类和多标签分类

在多分类（Multiclass）和多标签（multilabel）分类问题上，precision, recall, 和 F-measure的概念可以独立应用到每个label上。有一些方法可以综合各标签上的结果，通过指定average_precision_score （只能用在multilabel上）， f1_score, fbeta_score, precision_recall_fscore_support, precision_score 和 recall_score这些函数上的参数average可以做到。

注意：

- “micro”选项：表示在多分类中的对所有label进行micro-averaging产生一个平均precision，recall和F值
- “weighted”选项：表示会产生一个weighted-averaging的F值。

可以考虑下面的概念：

- y是(sample, label)pairs的预测集
- <img src="http://www.forkosh.com/mathtex.cgi?\hat{y}">是(sample, label)pairs的真实集
- L是labels的集
- S是labels的集
- <img src="http://www.forkosh.com/mathtex.cgi?\hat{y}">是y的子集，样本s,比如：<img src="http://www.forkosh.com/mathtex.cgi?y_s := \left\{(s', l) \in y | s' = s\right\}">
- <img src="http://www.forkosh.com/mathtex.cgi?y_l ">表示label l的y子集
- 同样的，<img src="http://www.forkosh.com/mathtex.cgi?y_s ">和<img src="http://www.forkosh.com/mathtex.cgi?y_l ">都是<img src="http://www.forkosh.com/mathtex.cgi?\hat{y} ">的子集
- <img src="http://www.forkosh.com/mathtex.cgi?P(A, B) := \frac{\left| A \cap B \right|}{\left|A\right|} ">
- <img src="http://www.forkosh.com/mathtex.cgi?R(A, B) := \frac{\left| A \cap B \right|}{\left|B\right|} "> 在处理<img src="http://www.forkosh.com/mathtex.cgi?B = \emptyset">时方式更不同；该实现采用<img src="http://www.forkosh.com/mathtex.cgi?R(A, B):=0">，且与P相类似。
- <img src="http://www.forkosh.com/mathtex.cgi?F_\beta(A, B) := \left(1 + \beta^2\right) \frac{P(A, B) \times R(A, B)}{\beta^2 P(A, B) + R(A, B)} ">

metrics的定义如下：

<figure>
    <a href="http://photo.yupoo.com/wangdren23/Fxc1ofvk/medish.jpg"><img src="http://photo.yupoo.com/wangdren23/Fxc1ofvk/medish.jpg" alt=""></a>
</figure>

代码：

{% highlight python %}

>>> from sklearn import metrics
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]
>>> metrics.precision_score(y_true, y_pred, average='macro')  
0.22...
>>> metrics.recall_score(y_true, y_pred, average='micro')
... 
0.33...
>>> metrics.f1_score(y_true, y_pred, average='weighted')  
0.26...
>>> metrics.fbeta_score(y_true, y_pred, average='macro', beta=0.5)  
0.23...
>>> metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5, average=None)
... 
(array([ 0.66...,  0.        ,  0.        ]), array([ 1.,  0.,  0.]), array([ 0.71...,  0.        ,  0.        ]), array([2, 2, 2]...))

{% endhighlight %}

对于多分类问题，对于一个“negative class”，有可能会排除一些标签：

{% highlight python %}

>>> metrics.recall_score(y_true, y_pred, labels=[1, 2], average='micro')
... # excluding 0, no labels were correctly recalled
0.0

{% endhighlight %}

类似的，在数据集样本中没有出现的label不能用在macro-averaging中。

{% highlight python %}

>>> metrics.precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro')
... 
0.166...

{% endhighlight %}

## 3.9 Hinge loss

[hinge_loss](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html#sklearn.metrics.hinge_loss)函数会使用hinge loss计算模型与数据之间的平均距离。它是一个单边的metric，只在预测错误（prediction erros）时考虑。(Hinge loss被用于最大间隔分类器上：比如SVM)

如果label使用+1和-1进行编码。y为真实值，w为由decision_function结出的预测决策。
hinge loss的定义如下：

<img src="http://www.forkosh.com/mathtex.cgi?L_\text{Hinge}(y, w) = \max\left\{1 - wy, 0\right\} = \left|1 - wy\right|_+">

如果超过两个label，由于[Crammer & Singer](http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf)所提到的问题 ，hinge_loss 会使用一个多元分类的变种。

如果<img src="http://www.forkosh.com/mathtex.cgi?y_w">是对于true label的预测判断（predicted decision），<img src="http://www.forkosh.com/mathtex.cgi?y_t ">则是对于其他label的预测判断的最大值，而predicted decisions由多个predicted decision输出，那么多分类的hinge loss定义如下：

<img src="http://www.forkosh.com/mathtex.cgi?L_\text{Hinge}(y_w, y_t) = \max\left\{1 + y_t - y_w, 0\right\}">

二分类问题示例：

{% highlight python %}

>>> from sklearn import svm
>>> from sklearn.metrics import hinge_loss
>>> X = [[0], [1]]
>>> y = [-1, 1]
>>> est = svm.LinearSVC(random_state=0)
>>> est.fit(X, y)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
>>> pred_decision = est.decision_function([[-2], [3], [0.5]])
>>> pred_decision  
array([-2.18...,  2.36...,  0.09...])
>>> hinge_loss([-1, 1, 1], pred_decision)  
0.3...

{% endhighlight %}

多分类问题示例：

{% highlight python %}

>>> X = np.array([[0], [1], [2], [3]])
>>> Y = np.array([0, 1, 2, 3])
>>> labels = np.array([0, 1, 2, 3])
>>> est = svm.LinearSVC()
>>> est.fit(X, Y)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
>>> pred_decision = est.decision_function([[-1], [2], [3]])
>>> y_true = [0, 2, 3]
>>> hinge_loss(y_true, pred_decision, labels)  
0.56...

{% endhighlight %}


## 3.10 Log loss

Log loss也被称为logistic回归loss，或者交叉熵loss(cross-entropy loss)，用于概率估计。它通常用在(multinomial)的LR和神经网络上，以最大期望（EM：expectation-maximization）的变种的方式，用于评估一个分类器的概率输出，而非进行离散预测。

对于二元分类，true label为：<img src="http://www.forkosh.com/mathtex.cgi?y \in \{0,1\}">，概率估计为：<img src="http://www.forkosh.com/mathtex.cgi?p = \operatorname{Pr}(y = 1)">，每个样本的log loss是对分类器给定true label的负值log似然估计(negative log-likelihood)：

<img src="http://www.forkosh.com/mathtex.cgi?L_{\log}(y, p) = -\log \operatorname{Pr}(y|p) = -(y \log (p) + (1 - y) \log (1 - p))">

当扩展到多元分类（multiclass）上时。可以将样本的true label编码成1-of-K个二元指示器矩阵Y，如果从label K集合中取出的样本i，对应的label为k，则<img src="http://www.forkosh.com/mathtex.cgi?y_{i,k} = 1">，P为概率估计矩阵，<img src="http://www.forkosh.com/mathtex.cgi?p_{i,k} = \operatorname{Pr}(t_{i,k} = 1)">。整个集合的log loss表示如下：

<img src="http://www.forkosh.com/mathtex.cgi?L_{\log}(Y, P) = -\log \operatorname{Pr}(Y|P) = - \frac{1}{N} \sum_{i=0}^{N-1} \sum_{k=0}^{K-1} y_{i,k} \log p_{i,k}">

我们再看下如何对二分类的log loss进行泛化的，注意，在二分类问题上，<img src="http://www.forkosh.com/mathtex.cgi?p_{i,0} = 1 - p_{i,1} "> 和<img src="http://www.forkosh.com/mathtex.cgi?y_{i,0} = 1 - y_{i,1}">，因而，通过在<img src="http://www.forkosh.com/mathtex.cgi?y_{i,k} \in \{0,1\} ">扩展内部和来给出二分类的log loss。

log_loss函数，通过给定一列真实值label和一个概率矩阵来计算log loss，返回值通过estimator的predict_proba返回。

{% highlight python %}

>>> from sklearn.metrics import log_loss
>>> y_true = [0, 0, 1, 1]
>>> y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
>>> log_loss(y_true, y_pred)    
0.1738...

{% endhighlight %}

y_pred中的[.9, .1]指的是，第一个样本中90%的概率是label 0。另外，log loss是非负的。

## 3.11 Matthews相关系数

[matthews_corrcoef](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef)函数计算了二元分类的[Matthew’s correlation coefficient (MCC)](http://en.wikipedia.org/wiki/Matthews_correlation_coefficient).

wikipedia是这么说的：

“The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary (two-class) classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction. The statistic is also known as the phi coefficient.”

翻译如下：

机器学习中使用的Matthews相关系数，用于度量二分类的质量。它会考虑TP/FP/TN/FP的情况，通常被认为是一个balanced的度量  ，可以用于那些有着不同size的分类中。MCC本质上是一个介于［－1，+1］之间的相关系数值。相关系数为+1，表示是一个完美的预测，0表示是一个平均随机预测（average random prediction），而-1表示是一个逆预测（inverse prediction）。这种统计方法也被称为：phi coefficient。

MCC相应的定义如下：

<img src="http://www.forkosh.com/mathtex.cgi?MCC = \frac{tp \times tn - fp \times fn}{\sqrt{(tp + fp)(tp + fn)(tn + fp)(tn + fn)}}.">

这里的示例展示了matthews_corrcoef 函数的使用：

{% highlight python %}

>>> from sklearn.metrics import matthews_corrcoef
>>> y_true = [+1, +1, +1, -1]
>>> y_pred = [+1, -1, +1, +1]
>>> matthews_corrcoef(y_true, y_pred)  
-0.33...

{% endhighlight %}

