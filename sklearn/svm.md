---
layout: page
title: sklearn中的svm 
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

svm可以用来分隔不均衡（unbalanced）的类。

SVM的优点有：

- 1.高维空间很有效
- 2.对于维度数目大于sample数目，仍有效
- 3.在决策函数（支持向量）中使用一个训练点的子集，因此内存很高效
- 4.多样化：决策函数使用不同的kernel函数。提供了通用的kernel，但也可以指定特定的kernel。

SVM的缺点包括：

- 1.如果feature数大于sample数，会给出各差的性能
- 2.SVM不直接提供概率评估（因而，不能使用计算高昂的5-fold交差验证）

sklearn中SVM支持dense和sparse训练样本向量作为输入。

# 2.分类器

SVC, NuSVC和LinearSVC可以进行多分类。


<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_iris_0012.png"><img src="http://scikit-learn.org/stable/_images/plot_iris_0012.png" alt=""></a>
</figure>


SVC和NuSVC两者类似，在输入参数上有细微差别，并具有不同的[数学公式](http://scikit-learn.org/stable/modules/svm.html#svm-mathematical-formulation)。另一方面，LinearSVC是另一个SVM的线性实现。注意LinearSVC不接受kernel参数，因为本身已经是线性的。

SVM决策树依赖于一些训练子集，称为支持向量。这些支持向量可以由clf分类器的成员属性指定：如：support_vectors_, support_ and n_support.

## 2.1 score和probabilities

- decision_function: SVM的该方法，为每个样本给出了每个分类的score。
- probability：设为True, 表示开启类成员的概率估计（predict_proba/predict_log_proba）。在二分类问题中，概率的标准化使用Platt scaling：在SVM的score进行logistic回归， 在训练集上进行额外的cross-validation。

在Platt scaling中的cross-validation，在大训练集上开销很大。

## 2.2 实际使用时的tips

- 1.避免数据拷贝：对于SVC, SVR, NuSVC和NuSVR来说，如果传进来的数据不是C-order连续，或不是double精度的，那么在调用底层的C实现时将引起数据拷贝。你可以通过检查flags属性来确认下numpy数据是否是C连续的。 而对于LinearSVC来说，。。。
- 2.Kernel的cache size：对于SVC, SVR, NuSVC和NuSVR来说，kernel的cache size对运行时长有很大影响。如果你具有足够的RAM，推荐将cache_size设得更高，缺省为200(MB). 比如500，或1000.
- 3.设置C: C缺省为1，这是个合理的选择。如果你发现有很多噪声，可以减小该值。相应的正则项部分可以更好地估计。
- 4.支持向量机算法没有归一化，因此强烈建议进行数据归一化!!例如，在输入向量X的每个属性归一化成[0,1]或[-1,+1]，或者将它们标准化成均值为0和方差为1。注意，必须在test vector上进行相同的归一化，来获得有意义的数据。详见[数据预处理](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)
- 5.NuSVC/OneClassSVM/NuSVR的参数nu，逼近训练误差和支持向量。
- 6.在SVC中，如果分类的数据是不均衡的（比如：positive很多，negative很少），可以设置class_weight='balanced'，并尝试不同的处罚因子C.


参考：

1.[http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html](http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html)
