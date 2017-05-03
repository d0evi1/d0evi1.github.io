---
layout: page
title: ml中的分类与回归
tagline: 介绍
---
{% include JB/setup %}

# 1.分类

## 1.1 Logistic回归

Logistic回归是一个很流行的预测二分类的方法。它是Generallized Linear model的一个特列，可以用来预测结果的发生概率。对于更多的详情，参数 spark.mllib。

Logistic回归提供了一个线性方法：

<img src="http://www.forkosh.com/mathtex.cgi?f(w):=\lambda R(w) + \frac{1}{n}\sum_{i=1}^{n}L(w;x_i,y_i)">

优化目标：<img src="http://www.forkosh.com/mathtex.cgi?argmin_{w}f(w)">

loss function为logistic loss：

<img src="http://www.forkosh.com/mathtex.cgi?L(w;x_i,y_i):=log(1+exp(-yw^Tx))">

对于二分类问题，算法输出一个二元logistic 回归模型。对于一个给定的数据点x，模型使用logistic function作出预测：

<img src="http://www.forkosh.com/mathtex.cgi?f(z)=\frac{1}{1+e^{-z}}">

其中<img src="http://www.forkosh.com/mathtex.cgi?z=w^Tx">

缺省的，如果<img src="http://www.forkosh.com/mathtex.cgi?f(w^Tx)">>0.5，则结果为正例，否则为负例，这不同于线性SVM，logistic回归模型的原始输出，f(z)，是一个概率解释。（比如：刚才的概率为正例）

binary logistic回归可以泛化到multinomial logistic回归上，来解决多分类问题。


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


## 决策树

## RF

RF是很流行的分类、回归方法。更多spark.ml的实现，可以参下见面的RF介绍。

示例：

下例加载libsvm格式的数据，将它split成训练集和测试集，在训练集上进行训练，在测试集上进行评估。我们使用两个feature transformer来准备数据；这可以帮助索引label的类别和类别型feature；添加metadata到DataFrame中，以便树算法进行识别。

{% highlight scala %}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load and parse the data file, converting it to a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a RandomForest model.
val rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)

{% endhighlight %}

# Tree Ensembles

DataFrame API支持两种主要的tree ensemble算法：RF和GBT。两种都使用spark.ml中的决策树作为base model。

用户可以在[MLlib Ensemble guide](http://spark.apache.org/docs/latest/mllib-ensembles.html)找到更多关于ensemble算法的信息.

该API与origin MLlib ensembles API的区别是：

- 支持DataFrame和ML Pipeline
- 分类和回归相互独立
- 使用DataFrame metadata来区别连续型feature和类别型feature
- 其中的RF具有更多的功能：feature importance的估计，也可以预测在分类中每个类的概率（a.k.a. 类的条件概率）

## RF

RF是决策树的ensembles。RF会结合许多决策树，来减小overfitting的风险。spark.ml的RF实现支持二元类和多元类的分类和回归，可以同时使用连续型和类别型feature。

### 1.输入和输出

我们列出了要预测列的类型。所有的输出列是可选的；除了需要一个输出列外，可以将相应的Param设置为空字符串。

输入列：

param名 | types | default | description
:------:|:-------:|:---------:|:-------------:
labelCol|Double| "label" | 要预测的label
featureCol|Vector| "features" | feature vector

输出列（预测）：


param名 | types | default | description  | 注意项
:-------:|:---------:|:-----------:|:--------:|:---------:
predictionCol| Double | "prediction" | 要预测的label |   
rawPredictionCol|Vector | "rawPrediction" | 对应类别的vector长度，以及在作出预测的树节点上训练实例label数 | 只用于分类
probabilityCol | Vector | "probability" | 相应类等于rawPrediction别的Vector | 只用于分类


## GBT

GBT是决策树的ensembles。为了最小化loss function，GBT会迭代训练决策树，spark.ml实现的GBT支持二分类，也可用于回归，可使用连续型和类别型feature。

### 输入和输出

我们列出了输入和输出列的类型。所有输出列是可选的；如果不需要输出列，可以将相应的
参数设置为一个空的string。

输入列

param | types | default | descrption
------|-------|---------|-------------
labelCol| Double | "label" | 要预测的label
featuresCol| Vector | "features" | Feature vector

注意，GBTClassifier目前只支持二分类的label。

输出列

param | types | Default | description 
------|-------|---------|----------------
predictionCol| Double | "prediction" | 要预测的label

在后续版本中，GBTClassifier也会输出rawPrediction和probability列。


参考：

1.[http://spark.apache.org/docs/latest/ml-classification-regression.html](http://spark.apache.org/docs/latest/ml-classification-regression.html)
