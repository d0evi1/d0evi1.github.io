---
layout: page
title: spark协同过滤 
tagline: 介绍
---
{% include JB/setup %}

# 前言

协同过滤（CF）常用于推荐系统。这种技术的目的是，填充user-item关联矩阵中的缺失项。spark.mllib当前支持基于model的协同过滤，用户和产品通过一个小的潜因子集合进行描述，用来预测缺失项。spark.mllib使用交替最小二乘(ALS)算法来学习这些潜因子（latent factors）。spark.mllib具有以下的参数：

- numBlocks: 用于并行化计算的块的数目（设为-1表示自动配置）
- rank: 模型中潜因子的数目
- iterations: 迭代因子
- lambda: ALS中的正则化参数
- implicitPrefs: 指定是否要使用显式反馈ALS变量，还是采用隐式反馈数据
- alpha: 该参数用于ALS隐式反馈变量，管理在偏好观察中baseline的置信度

# 显式反馈vs.隐式反馈

基于CF的矩阵分解的标准方式是：将user-item矩阵中的项作为user对item的显式偏好。

现实世界中，更为常见的是隐式反馈（观看，点击，购买，like，分享等）。在spark.mllib中这种方式来自于[http://dx.doi.org/10.1109/ICDM.2008.22](http://dx.doi.org/10.1109/ICDM.2008.22)。可以直接替代评分矩阵(ranking matrix)来尝试模型，这种方法将数据看成是二元偏好与置信度的一个组合。评分（ratings）与在观察到的用户偏好上的置信度有关，而非显式评分。该模型接着会尝试找出潜因子，然后用它们来预测一个用户对相应item的偏好。

# 正则参数的归一化

从v1.1版本以后，我们就会在解每个最小二乘问题时，通过使用用户在更新潜因子时生成的评分的数目，来归一化正则项参数lambda。这种方法称为："ALS-WR"，来自论文:[http://dx.doi.org/10.1007/978-3-540-68880-8_32](http://dx.doi.org/10.1007/978-3-540-68880-8_32)。它使lambda更少依赖于数据集的规模。因此，我们可以使用从完整数据集中抽样来的样本子集中学到最好的参数，期望得到相似的性能。

# 示例

在下面的示例中，我们加载评分数据。每行包含一个用户，一个产品，和一个评分。我们使用缺省的ALS.train()方法，假设评分是显式的。我们使用MSE来评估模型。

{% highlight scala %}

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

// Load and parse the data
val data = sc.textFile("data/mllib/als/test.data")
val ratings = data.map(_.split(',') match { case Array(user, item, rate) =>
  Rating(user.toInt, item.toInt, rate.toDouble)
})

// Build the recommendation model using ALS
val rank = 10
val numIterations = 10
val model = ALS.train(ratings, rank, numIterations, 0.01)

// Evaluate the model on rating data
val usersProducts = ratings.map { case Rating(user, product, rate) =>
  (user, product)
}
val predictions =
  model.predict(usersProducts).map { case Rating(user, product, rate) =>
    ((user, product), rate)
  }
val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
  ((user, product), rate)
}.join(predictions)
val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
  val err = (r1 - r2)
  err * err
}.mean()
println("Mean Squared Error = " + MSE)

// Save and load model
model.save(sc, "target/tmp/myCollaborativeFilter")
val sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")

{% endhighlight %}

完整的代码在：examples/src/main/scala/org/apache/spark/examples/mllib/RecommendationExample.scala。

如果评分矩阵派生自另一个信息源（比如：其它信号），你可以使用trainImplicit方法来获取更好结果。

{% highlight scala %}

val alpha = 0.01
val lambda = 0.01
val model = ALS.trainImplicit(ratings, rank, numIterations, lambda, alpha)

{% endhighlight %}

# 更多教程

[training exercieses](https://databricks-training.s3.amazonaws.com/index.html)

[个性化电影推荐](https://databricks-training.s3.amazonaws.com/movie-recommendation-with-mllib.html)

参考：

1.[http://spark.apache.org/docs/latest/mllib-ensembles.html](http://spark.apache.org/docs/latest/mllib-ensembles.html)
