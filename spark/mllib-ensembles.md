---
layout: page
title: spark ensembles
tagline: 介绍
---
{% include JB/setup %}

# 前言

spark.mllib中支持两种主要的ensemble算法： GradientBoostedTrees 和 RandomForest。两者都使用决策树作为基础模型。

## 一、GBT vs. RF

GBT和RF都是关于树的ensembles学习算法，但是训练过程是不同的。实际使用时有以下的权衡：

- GBT一次训练一棵树，因此训练时间比RF要长。RF可以并行训练多棵树。
- 另一方面，GBT比RF经常使用更小（更浅）的树是合理的，训练越小的树花费时间越少。
- RF很少有overfitting的倾向。在RF中训练更多的树会减少overfitting的概率，但使用GBT训练更多的树会增加overfitting的概率。（在统计语言中，使用更多的树，RF会减少variance，而GBT则减少bias）
- RF可以很容易调整，因为性能会随着树的数目单调递增提升（而GBT则会随着树的数目过大而降低）

简单的说，两种算法都很有效，对于特定数据集选择特定算法。

## 二、RF 

## 三、GBT

GBT是决策树的ensembles。GBT迭代训练决策树，为了减小loss function。类似于决策树，GBT可以处理类目型特征（categorical feature），可扩展到多分类（multiclass）设置，不需要进行特征归一化（feature scaling），并且可以捕获非线性和特征交叉。

spark.mllib支持的GBT可以用于二分类(binary classification)和回归，可以同时使用连续型特征和类目型特征。spark.mllib实现的GBT使用已经存在的[决策树实现](http://spark.apache.org/docs/latest/mllib-decision-tree.html)。详见决策树相关。

注意：GBT目前还不支持多分类。对于多分类问题，可以使用spark的决策树和RF。

### 2.1 基本算法 

Gradient boosting会迭代训练一系列的决策树。在每次迭代中，算法使用当前的ensemble来预测每个训练实例的label，接着将预测label与真实的label进行比较。数据集被重新label，以便使用弱预测对训练实例进行增强。这样，在下次迭代中，决策树将帮助纠正先前的错误。

这种特有的relabeling机制通过定义一个loss function来实现。在每次迭代后，GBT都会进一步的减小在训练数据上的loss function。

### 2.2 Losses

下表列出了spark.mllib中提供的GBT的losses。注意，每个loss都只能应用到某个分类或者某个回归，而非同时支持分类和回归。

注意：N=实例数. <img src="http://www.forkosh.com/mathtex.cgi?y_i">=实例i的label。<img src="http://www.forkosh.com/mathtex.cgi?x_i">=实例i的feature。<img src="http://www.forkosh.com/mathtex.cgi?F(x_i)">=实例i的模型预测label。

| Loss                     | Task | Formula | Description |
|:--------------------------|:----|--------|-------------:|
| Log Loss                 | 分类 |  <img src="http://www.forkosh.com/mathtex.cgi?2\sum_{i=1}^Nlog(1+e^{-2y_iF(x_i)})">       | 两倍二项式负log似然          |
| 平方误差(Squared Error)  | 回归 | <img src="http://www.forkosh.com/mathtex.cgi?\sum_{i=1}^N(y_i-F(x_i))^2">        | 也称为L2 loss。缺省loss用于回归任务     |
| 绝对误差(Absolute Error) | 回归 | <img src="http://www.forkosh.com/mathtex.cgi?\sum_{i=1}^N \mid y_i-F(x_i) \mid">        | 也称为L1 loss。对于异常类，它比平方误差更健壮     |

### 2.3 使用tips

我们包含了一点使用GBT的参数指南。这里我们忽略掉一些决策树参数，因为已经在决策树中涉及了。

- loss: 详见上面部分的loss信息（分类 vs. 回归）. 依赖于数据集，不同的loss可以给出极为不同的结果。
- numIterations: 设置在ensemble中树的数目。每次迭代产生一棵树。对于提升训练数据集的accuracy，增加它的数目会让模型更为出色。然而如果太大，那么测试时间accuracy都会有影响。
- learningRate: 该参数没必要去调整。如果算法行为看起来不稳定，增加它的值可以提升稳定性。
- algo：该算法或任务（分类/回归）可以使用设置tree[Strategy]参数来实现。

### 2.4 训练Validation

当训练的树过多时，Gradient boosting可能会overfit。为了阻止overfitting，我们在训练时需要做validate。方法runWithValidation可以使用这个选项。它会带上一对RDD作为参数，第一个RDD是训练集，第二个RDD做为验证集。

当在validation error上的提升不再超过一个固定的阀值时（通过BoostingStrategy 的validationTol参数来设定），训练将会终止。实例上，validation error会在开始减小，在最后又会增加。validation error不会单调改变，建议用户设置一个足够大的负tolerance值（negative tolerance），并且使用evaluateEachIteration（它会给出每次迭代的error或loss）来调整迭代次数以检查验证曲线（validation curve）。

### 2.5 示例

分类：

示例load一个LIBSVM数据文件，解析成一个LabeledPoint的RDD，接着使用Gradient-Boosted Trees和log loss来进行分类。并计算测试误差来衡量accuracy。

spark 源码：
examples/src/main/scala/org/apache/spark/examples/mllib/GradientBoostingClassificationExample.scala

回归：


示例load一个LIBSVM文件，解析成一个LabeledPoint的RDD，接着使用Gradient-Boosted Trees和Squared Error loss来进行分类。并在最后计算MSE来评估拟合效果好坏。

spark示例源码：

examples/src/main/scala/org/apache/spark/examples/mllib/GradientBoostingRegressionExample.scala


参考：

1.[http://spark.apache.org/docs/latest/mllib-ensembles.html](http://spark.apache.org/docs/latest/mllib-ensembles.html)
