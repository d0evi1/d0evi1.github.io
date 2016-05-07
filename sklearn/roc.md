---
layout: page
title: sklearn中的roc
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

ROC (接受者操作特性曲线:Receiver Operating Characteristic)

ROC metric用来评估分类器的输出质量。

ROC曲线通常在Y轴上表现为TP率(true postive rate)，在X轴上则是FP率(false postive rate)。 这意味着ROC曲线图的左上角是理想的取值点：FP率为0，TP率为1. 这是理想情况，但这意味着**曲线下面积（AUC: area under the curve）**越大越好。

ROC的陡度（steepness）也十分重要，理想情况是：TP率最大化，FP率最小化。

# 2.多分类问题

ROC曲线一般用在二分类问题上，用来学习一个分类器的输出结果。为了将ROC曲线和ROC面积扩展到多标签分类问题（multi-label classifaction）上，需要对输出进行二值化。可以为每个label上绘制一条ROC曲线。也可以通过标签判断矩阵(label indicator matrix)的每个元素作为二分类预测（micro-averaging）来绘制ROC曲线。

另一种二分类的评估方式是micro-averaging。它给出每个分类label上对应的平均权重。

注意:

- [sklearn.metrics.roc_auc_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)
- [Receiver Operating Characteristic (ROC) with cross validation](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#example-model-selection-plot-roc-crossval-py)

参考：

1.[http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html](http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html)
