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

下例展示了使用lr、elastic net正则项的模型。elasticNetParam对应于<img src="http://www.forkosh.com/mathtex.cgi?\alpha">，而regParam则对应于<img src="http://www.forkosh.com/mathtex.cgi?\lambda">。

{% highlight scala %}

import org.apache.spark.ml.classification.LogisticRegression

// Load training data
val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// Fit the model
val lrModel = lr.fit(training)

// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

{% endhighlight %}

完整代码示例在：examples/src/main/scala/org/apache/spark/examples/ml/LogisticRegressionWithElasticNetExample.scala

spark.ml的lr实现也支持从训练集上抽取模型的summary。注意：在BinaryLogisticRegressionSummary上的predictions和metrics以DataFrame的方式存储，通过@transient进行注解，只在driver中支持。

LogisticRegressionTrainingSummary 提供了一个 LogisticRegressionModel的汇总信息。当前，只支持二分类，summary信息必须显式转换成BinaryLogisticRegressionTrainingSummary。这在未来会做更改。

延续之前的示例：

{% highlight scala %}

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}

// Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
// example
val trainingSummary = lrModel.summary

// Obtain the objective per iteration.
val objectiveHistory = trainingSummary.objectiveHistory
objectiveHistory.foreach(loss => println(loss))

// Obtain the metrics useful to judge performance on test data.
// We cast the summary to a BinaryLogisticRegressionSummary since the problem is a
// binary classification problem.
val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

// Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
val roc = binarySummary.roc
roc.show()
println(binarySummary.areaUnderROC)

// Set the model threshold to maximize F-Measure
val fMeasure = binarySummary.fMeasureByThreshold
val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
  .select("threshold").head().getDouble(0)
lrModel.setThreshold(bestThreshold)

{% endhighlight %}

完整示例详见：examples/src/main/scala/org/apache/spark/examples/ml/LogisticRegressionSummaryExample.scala


参考：

1.[http://spark.apache.org/docs/latest/ml-classification-regression.html](http://spark.apache.org/docs/latest/ml-classification-regression.html)
