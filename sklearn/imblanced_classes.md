---
layout: page
title: 如何处理偏斜类(imbalanced classes) 
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

偏斜类是在我们处理机器学习中常遇到的情况。比如一个二分类问题，样本中class 0的样本数占90%，而class 1的样本数只占10%。当直接使用这样的训练样本进行训练时，如果采用正确率(Accuracy)来衡量，会得到一个很令人满意的分值。但是这个结果基本是不可信的，这种现象被称为“正确率谬论: [Accuracy Paradox](https://en.wikipedia.org/wiki/Accuracy_paradox)”

注意正确率(Accuracy)与准确率(Precision)的区别：

- Accuracy = (TP+TN)/(TP+FP+TN+FN)
- Precision = TP/(TP+FP)

如何处理这个问题，是机器学习中很重要的一个问题。在machinelearningmastery上，有人做了总结：

# 2.解决方案

有八种方法来处理这样的问题：

####1.**收集更多的数据（尤其是对于小类）**

####2.尝试其它的性能指标：

- 传统的方式：混淆矩阵、Precision/Recall、F1-Score/F-Score
- 新的方式：**Cohen's Kappa、ROC曲线**

####3.重新抽样

- **过采样(over-sampling)**：增加小类的copy，达到平衡
- **欠采样(under-sampling)**：删除一些大类的实例，达到平衡

关于概念，可见：[wiki](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)

一些经验：

- 对于大数据（比如：上百万），考虑使用欠采样
- 对于小数据（万级别），考虑使用过采样
- 考虑使用随机/非随机抽样方式
- 考虑测试不同的重抽样（resample）比例。（例如：在二分类中，不一样使用1:1的比例）

####4.尝试生成人造样本

- 比如：在小类中随机抽取实例的features进行构造。
- 可以考虑使用Naive Bayes对各独立特征进行抽样。
- 尝试抽样算法：**SMOTE**（Synthetic Minority Over-sampling Technique：合成少数过采样技术）。

SMOTE是一种过采样，通过对小类创建人工合成的样本（非创建copy）。该算法选择两个或多个相似的实例（使用距离计算公式），通过对比近邻实例上的不同属性项，在这个不同属性项范围内，一次只对实例上的一个属性进行扰动（perturbing）。

有许多SMOTE算法的实现，例如：

- 在Python中，[UnbalancedDataset](https://github.com/fmfn/UnbalancedDataset)模块。它提供了SMOTE的实现
- 在R中，[DMwR package](https://cran.r-project.org/web/packages/DMwR/index.html)
- 在Weka中，使用[SMOTE supervised filter](http://weka.sourceforge.net/doc.packages/SMOTE/weka/filters/supervised/instance/SMOTE.html)

####5.使用不同的算法

对于给定的问题，一定要尝试不同类型算法；不要光使用自己喜欢的算法。

决策树通常在偏斜类数据集上表现很好。可以尝试多种：C4.5, C5.0, CART, and Random Forest。

####6.使用**带惩罚项的模型**

带罚项的分类在模型训练上引入了一个额外代价，以便减小在训练时对小类的错误分类。这些惩罚项可以让模型的bias对小类更有利。

带惩罚项版本的算法有：p-SVM(penalized-SVM)和p-LDA(penalized-LDA).

也有一些带惩罚项模型的框架，例如：Weka中的CostSensitiveClassifier。

针对特定的算法来说，使用惩罚项不能重新抽样（resample），否则你会得到很差的结果。它提供了另一种方式来“平衡”分类。建立惩罚矩阵很复杂。你可以需要尝试不同的惩罚模式，并观察结果是否最佳。

####7.尝试不同的视角

有许多领域的学习是专门针对偏斜类数据集的。

你还可以考虑其它两个学习方法是：**异常检测**（anomaly detection）和**变化检测**（change detection）。

[异常检测](https://en.wikipedia.org/wiki/Anomaly_detection)：将小类看到是异常类（outliers class）。

[变化检测](https://en.wikipedia.org/wiki/Change_detection)：和异常检测类似。用于发现从使用模式或银行交易的观察值是否有变化。

####8.尝试一些脑洞

[quora上的问题](https://www.quora.com/In-classification-how-do-you-handle-an-unbalanced-training-set)

[reddit上的讨论](https://www.reddit.com/r/MachineLearning/comments/12evgi/classification_when_80_of_my_training_set_is_of/)


参考：

1.[http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/](http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
