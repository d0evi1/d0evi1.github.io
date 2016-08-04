---
layout: page
title: ml pipelines
tagline: 介绍
---
{% include JB/setup %}

# 1.分类

## 1.1 Logistic回归

Logistic回归是一个很流行的预测二分类的方法。它是Generallized Linear model的一个特列，可以用来预测结果的发生概率。对于更多的详情，参数 spark.mllib。

spark.ml的lr当前实现只支持二分类。将来会考虑支持多分类。

当使用LogisticRegressionModel进行fitting时，不需要在数据集上解析常量非零列，对于常量非零列，Spark MLlib会输出零参数（zero cofficients）。该特性与R的glmnet相同，但与LIBSVM不同。

示例：

下例展示了使用lr、elastic net正则项的模型。elasticNetParam对应于<img src="http://www.forkosh.com/mathtex.cgi?\alpha">，而regParam则对应于<img src="http://www.forkosh.com/mathtex.cgi?\lamda">。


参考：

1.[http://spark.apache.org/docs/latest/ml-classification-regression.html](http://spark.apache.org/docs/latest/ml-classification-regression.html)
