---
layout: page
title: sklearn中的Bagging 
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

bagging和boosting都属于ensemble家族。

- 平均方法：构建各个独立的estimator，对结果求平均。其目的为了减小variance。例如：bagging、随机森林
- boosting方法：estimator按顺序构建，一层又一层，其目的是为了减小bias。例如：AdaBoost、GBDT

BAGging是Bootstrap Aggregation的简称。我们简单回顾下bagging的核心思想。

- 1.Aggregation: 也是利用分散的多组模型进行组合
- 2.Bootstrap拔靴法：有放回重复抽样 => 得到多组训练样本
- 3.分类问题=> 投票voting；回归问题：求平均

<figure>
	<a href="http://photo.yupoo.com/wangdren23/FvKhTEcV/medish.jpg"><img src="http://photo.yupoo.com/wangdren23/FvKhTEcV/medish.jpg" alt=""></a>
</figure>

在ensemble算法中，bagging方法从原始训练集进行随机抽样生成训练子集，并根据不同的训练子集构建不同的黑盒（black-box）的评估器（estimator），最后再对这些estimator的不同预测结果进行aggregate得到最终的预测值。该方法通过引入随机化，用来降低基础评估器（base estimator，比如：决策树）的variance。在许多case中，bagging是一种很简单的方法用来提升单独模型的效果，不需要放弃底层的base算法。同样是为了减小overfitting，bagging在强分类器和复杂模型之上构建时效果更佳（比如：深层决策树 Deep DT），而boosting则在弱模型之上效果更好（比如：浅层决策树 Shallow DT）。

不同的随机抽取子训练集的方法在Bagging方法会有很大差别：

- 1.Pasting：如果抽取的数据集的随机子集是sample的子集时，称为Pasting
- 2.Bagging: 如果样本抽取是放回的，则为Bagging
- 3.Random Subspace：如果抽取的数据集的随机子集是feature的子集，则为Random Subspace
- 4.Random Patches：当评估器构建在sample和feature的子集之上时，为Random Patches

在sklearn中，bagging方法使用统一的BaggingClassifier元评估器（或者BaggingRegressor），输入的参数及策略由用户指定。max_samples和max_features控制着子集的size，当bootstrap和bootstrap_features控制着sample和feature是放回抽样还是不放回抽样。当使用样本子集时，通过设置oob_score=True，泛化错误可以使用out-of-bag样本来评估。

示例：一个base estimator使用KNeighborsClassifier的bagging ensemble方法。

在上一节中，我们讲了一个cart的示例，接下去我们在那个的基础之上做些改动，将它改成一个base estimator基于cart的Bagging分类器：





