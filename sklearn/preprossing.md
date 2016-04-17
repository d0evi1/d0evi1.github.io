---
layout: page
title: sklearn中的数据预处理 
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

klearn.preprocessing提供了各种公共函数，来将raw feature vetor转换成另外一种更适合评估器工作的格式。

# 2.标准化(Standardization)、平均移除法（mean removal）和方差归一化（variance scaling）

数据集的标准化，在scikit中，对于众多机器学习评估器来说是必须的；如果各独立特征不进行标准化，结果标准正态分布数据差距很大：比如使用均值为0、方差为1的高斯分布.

实际上，我们经常忽略分布的形状，只是简单地通过移除每个feature的均值，将数据转换成中间值，接着通过分割非常数项的feature进行归一化。

例如，在机器学习的目标函数（objective function），比如SVM中的RBF或者线性模型中的L1和L2正则项，其中使用的元素，前提是所有的feature都是以0为中心，且方差的order都一致。如果一个feature的方差，比其它feature的order都大，那么它将在目标函数中占支配地位，从而使得estimator从其它feature上学习不到任何东西。

scale函数提供了一个快速而简单的方式：

{% highlight python %}
>>> from sklearn import preprocessing
>>> import numpy as np
>>> X = np.array([[ 1., -1.,  2.],
...               [ 2.,  0.,  0.],
...               [ 0.,  1., -1.]])
>>> X_scaled = preprocessing.scale(X)

>>> X_scaled                                          
array([[ 0.  ..., -1.22...,  1.33...],
       [ 1.22...,  0.  ..., -0.26...],
       [-1.22...,  1.22..., -1.06...]])

{% endhighlight %}

我们可以看到，归一化后的数据，均值为0，方差为1：

>>> X_scaled.mean(axis=0)
array([ 0.,  0.,  0.])

>>> X_scaled.std(axis=0)
array([ 1.,  1.,  1.])

preprocessing模块提供了另一个工具类：StandardScaler，它实现了Transformer API，来计算在一个训练集上的平均值和标准差（standard deviation），同样需要在测试集上使用相同的转换。该类也可以应用在sklearn.pipeline.Pipeline上。

>>> scaler = preprocessing.StandardScaler().fit(X)
>>> scaler
StandardScaler(copy=True, with_mean=True, with_std=True)

>>> scaler.mean_                                      
array([ 1. ...,  0. ...,  0.33...])

>>> scaler.scale_                                       
array([ 0.81...,  0.81...,  1.24...])

>>> scaler.transform(X)                               
array([[ 0.  ..., -1.22...,  1.33...],
       [ 1.22...,  0.  ..., -0.26...],
       [-1.22...,  1.22..., -1.06...]])


scaler实例，可以用在新数据上，并可以以相同的方式在训练集上转换：

>>> scaler.transform([[-1.,  1., 0.]])                
array([[-2.44...,  1.22..., -0.26...]])

通过在StandardScaler的构造函数中设置with_mean=False or with_std=False，可以禁止centering和scaling。

## 2.1 将feature归一化到一个范围内


## 2.2 归一化sparse矩阵

## 2.3 归一化异常数据

## 2.4 kernel matrics的中心化

# 3.正态分布化（Normalization）


# 4.二值化（Binarization）


# 5.将类别特征进行编码

# 6.补充缺失值

# 7.生成多态特征（polynomial features）

# 8.定制转换器

 





参考：

1.[http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)
