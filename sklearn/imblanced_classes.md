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

#### 1.**收集更多的数据（尤其是对于小类）**

#### 2.尝试其它的性能指标：

- 传统的方式：混淆矩阵、Precision/Recall、F1-Score/F-Score
- 新的方式：**Cohen's Kappa、ROC曲线**

#### 3.重新抽样

- **过采样(over-sampling)**：增加小类的copy，达到平衡
- **欠采样(under-sampling)**：删除一些大类的实例，达到平衡

关于概念，可见：[wiki](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)

一些经验：

- 对于大数据（比如：上百万），考虑使用欠采样
- 对于小数据（万级别），考虑使用过采样
- 考虑使用随机/非随机抽样方式
- 考虑测试不同的重抽样（resample）比例。（例如：在二分类中，不一样使用1:1的比例）

#### 4.尝试生成人造样本

- 比如：在小类中随机抽取实例的features进行构造。
- 可以考虑使用Naive Bayes对各独立特征进行抽样。
- 尝试抽样算法：**SMOTE**（Synthetic Minority Over-sampling Technique：合成少数过采样技术）。

SMOTE是一种过采样，通过对小类创建人工合成的样本（非创建copy）。该算法选择两个或多个相似的实例（使用距离计算公式），通过对比近邻实例上的不同属性项，在这个不同属性项范围内，一次只对实例上的一个属性进行扰动（perturbing）。

有许多SMOTE算法的实现，例如：

- 在Python中，[UnbalancedDataset](https://github.com/fmfn/UnbalancedDataset)模块。它提供了SMOTE的实现
- 在R中，[DMwR package](https://cran.r-project.org/web/packages/DMwR/index.html)
- 在Weka中，使用[SMOTE supervised filter](http://weka.sourceforge.net/doc.packages/SMOTE/weka/filters/supervised/instance/SMOTE.html)

#### 5.使用不同的算法

对于给定的问题，一定要尝试不同类型算法；不要光使用自己喜欢的算法。

决策树通常在偏斜类数据集上表现很好。可以尝试多种：C4.5, C5.0, CART, and Random Forest。

#### 6.使用**带惩罚项的模型**

带罚项的分类在模型训练上引入了一个额外代价，以便减小在训练时对小类的错误分类。这些惩罚项可以让模型的bias对小类更有利。

带惩罚项版本的算法有：p-SVM(penalized-SVM)和p-LDA(penalized-LDA).

也有一些带惩罚项模型的框架，例如：Weka中的CostSensitiveClassifier。

针对特定的算法来说，使用惩罚项不能重新抽样（resample），否则你会得到很差的结果。它提供了另一种方式来“平衡”分类。建立惩罚矩阵很复杂。你可以需要尝试不同的惩罚模式，并观察结果是否最佳。

#### 7.尝试不同的视角

有许多领域的学习是专门针对偏斜类数据集的。

你还可以考虑其它两个学习方法是：**异常检测**（anomaly detection）和**变化检测**（change detection）。

[异常检测](https://en.wikipedia.org/wiki/Anomaly_detection)：将小类看到是异常类（outliers class）。

[变化检测](https://en.wikipedia.org/wiki/Change_detection)：和异常检测类似。用于发现从使用模式或银行交易的观察值是否有变化。

#### 8.尝试一些脑洞

[quora上的问题](https://www.quora.com/In-classification-how-do-you-handle-an-unbalanced-training-set)

[reddit上的讨论](https://www.reddit.com/r/MachineLearning/comments/12evgi/classification_when_80_of_my_training_set_is_of/)


# 3. UnbalancedDataset介绍

有些paper上称为上采样(up-sampling)和下采样（down-sampling）。

paper上有提到：

虽然重采样在一些数据集上取得了不错的效果 ,但是这类方法也存在一些缺陷. 上采样方法并不增加任何新的数据 ,只是重复一些样本或增加一些人工生成的稀有类样本 ,增加了训练时间. 更危险的是,上采样复制某些稀有类样本 ,或者在它周围生成新的稀有类样本 ,使得分类器过分注重这些样本 ,导致overfitting。

上采样不能从本质上解决稀有类样本的稀缺性和数据表示的不充分性 ,因此有人指出它的性能不如下采样. 但是 Japkowicz 8 ]对人工数据的一项系统研究得到了相反的结论. 下采样在去除大类样本的时候 ,容易去除重要的样本信息. 虽然有些启发式的下采样方法, 只是去除冗余样本和声样本 ,但是多数情况下这类样本只是小部分,因此这种方法能够调整的不平衡度相当有限.


## 3.1 OverSampling过采样

共有属性：

- method: 'replacement'表示放回抽样；'gaussian-perturbation'表示高斯混合法

1. OverSampler：过采样：

ratio：在原始的小类中（minority class），抽取各自样本数，：如果ratio=0.5，表示新的小类总样本块将会是1.5倍老样本块。

2.SMOTE:

SMOTE首先为每个稀有类样本随机选出几个邻近样本，并且在该样本与这些邻近样本的连线上随机取点，生成无复杂的新的稀有样本。

- k: 缺省为5. 使用多个是最近邻来构造合成样本（synthetic samples）
- m: 缺省为10. 如果小类样本很小，用来决策最邻最近邻的数目.
- out_step: 缺省为0.5。当进行推断时（extrapolating）的step size
- ratio: 缺省为1. 小类样本的块的比例
- kind: 缺省为regular。选项有：regular, borderline1, borderline2, svm
- verbose: 缺省为None。打印状态信息
- kwargs: 额外参数：sklearn SVC对象

## 3.2 UnderSampling欠采样

1.UnderSampler

- ratio: 缺省为1. 对小类元素采取多少比例的抽样

2.TomekLinks

3.ClusterCentroids

- ratio: 在小类中，根据各种样本数进行fit的聚类数，N_clusters = int(ratio * N_minority_samples) = N_maj_undersampled
- kwargs: 传给sklearn KMeans的对象参数

4.NearMiss

- version: 版本：1，2，或3.
- size_ngh: 近邻(neighbourhood)的size，计算小类样本的平均距离
- ver3_samp_ngh：NearMiss-3算法以一个resampling开始，该参数与被选中的子集的近邻数有关。
- kwargs: 传给Nearest Neighbours算法使用。

5.CondensedNearestNeighbour

- size_ngh：为最小类计算平均距离的近邻的size
- n_seeds_S：为了构建集合S抽取的样本数
- kwargs：Neareast Neighbours使用的参数

6.OneSidedSelection

- size_ngh：为最小类计算平均距离的近邻的size
- n_seeds_S：为构建集合S所要抽取的样本数
- kwargs：Neareast Neighbours的参数

7.NeighbourhoodCleaningRule

- size_ngh: 
- kwargs: 

## 3.3 ensemble sampling

1.EasyEnsemble

- ratio: 
- random_state: 随机seed.
- replacement: 是否放回抽样
- n_subsets: 生成的子集数

2.BalanceCascade

- ratio:
- random_state:
- n_max_subset:
- classifier:
- kwargs:


# 4. StratifiedKFold

k-fold交叉验证中，有一种交叉验证称为：StratifiedKFold。

<figure>
	<a href="http://photo.yupoo.com/wangdren23/FvNfSjlQ/medish.jpg"><img src="http://photo.yupoo.com/wangdren23/FvNfSjlQ/medish.jpg" alt=""></a>
</figure>


# 5. XGBoost中的处理

XGBoost中的官方文档大致这么说的：

对于一些case，比如：广告点击日志，数据集极不平衡。这会影响xgboost模型的训练，有两个方法来改进它。

- 如果你关心的预测的ranking order（AUC)：
-- 通过scale_pos_weight来平衡正负类的权重
-- 使用AUC进行评估

- 如果你关心的是预测的正确率：
-- 不能再平衡（re-balance）数据集
-- 将参数max_delta_step设置到一个有限的数（比如：1）可以获得效果提升.


参考：

1.[http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/](http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

2.[http://xgboost.readthedocs.io/en/latest/param_tuning.html](http://xgboost.readthedocs.io/en/latest/param_tuning.html)
