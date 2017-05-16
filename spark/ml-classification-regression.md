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

$$
f(w):=\lambda R(w) + \frac{1}{n}\sum_{i=1}^{n}L(w;x_i,y_i)
$$


优化目标 \$ argmin_{w}f(w) \$

loss function为logistic loss：

$$
L(w;x_i,y_i):=log(1+exp(-yw^Tx))
$$

对于二分类问题，算法输出一个二元logistic 回归模型。对于一个给定的数据点x，模型使用logistic function作出预测：

$$
f(z)=\frac{1}{1+e^{-z}}
$$

其中\$ z=w^Tx \$

缺省的，如果\$ f(w^Tx)>0.5 \$，则结果为正例，否则为负例，这不同于线性SVM，logistic回归模型的原始输出，f(z)，是一个概率解释。（比如：刚才的概率为正例）

binary logistic回归可以泛化到multinomial logistic回归上，来解决多分类问题。


spark.ml的lr当前实现只支持二分类。将来会考虑支持多分类。

当使用LogisticRegressionModel进行fitting时，不需要在数据集上解析常量非零列，对于常量非零列，Spark MLlib会输出零参数（zero cofficients）。该特性与R的glmnet相同，但与LIBSVM不同。

示例：

下例展示了使用lr、elastic net正则项的模型。elasticNetParam对应于\$\alpha \$，而regParam则对应于\$ \lambda \$。

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


## 1.2 决策树分类器


## 1.3 RF分类器

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

## 1.4 GBT分类器

Gradient-boosted trees (GBTs)是一个流行的分类回归方法，它使用决策树的ensembles方法。更多有关信息详见下面的GBT部分.

示例：

下面的示例加载一个LibSVM格式的数据，将它划分成training set和test sets，在training set上进行训练，在held-out test set上做评估。我们使用一个特征转换器对数据预处理，它会将label和类别型特征转换成index类别，并添加元数据到DataFrame中，以便让决策树算法可以识别到。

{% highlight scala %}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load and parse the data file, converting it to a DataFrame.
val data = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

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

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a GBT model.
val gbt = new GBTClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setMaxIter(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and GBT in a Pipeline
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Train model.  This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("precision")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println("Learned classification GBT model:\n" + gbtModel.toDebugString)

{% endhighlight %}

完整示例见：examples/src/main/scala/org/apache/spark/examples/ml/GradientBoostedTreeClassifierExample.scala

## 1.5 多层感知分类器(Multilayer perceptron classifier)

多层感知分类器(MLPC)是一个基于前馈人工神经网络（ feedforward artificial neural network）的分类器。MLPC包含了多层节点。每层都与下一层在网络上完全连接（fully connected）。输入层的节点表示输入数据。所有其它的节点，将输入映射到输出，通过结合节点权重 \$ \wv \$ 和 bias \$ \bv \$，应用一个激活函数，为输入执行线性组合。它可以以MLPC的矩阵形式，使用K+1层，如下表示：

$$

\mathrm{y}(\x) = \mathrm{f_K}(...\mathrm{f_2}(\wv_2^T\mathrm{f_1}(\wv_1^T \x+b_1)+b_2)...+b_K)

$$

中间层的节点使用sigmoid(logistic) function:

$$

\mathrm{f}(z_i) = \frac{1}{1 + e^{-z_i}} 

$$

输出层的结节使用softmax function：

$$

\mathrm{f}(z_i) = \frac{e^{z_i}}{\sum_{k=1}^N e^{z_k}} 

$$

输出层的节点数目N对应于分类的数目。

MLPC在学习模型时会执行BP算法。我们使用logistic loss function以及L-BFGS来优化。

{% highlight scala %}

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Load the data stored in LIBSVM format as a DataFrame.
val data = sqlContext.read.format("libsvm")
  .load("data/mllib/sample_multiclass_classification_data.txt")
// Split the data into train and test
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)
val layers = Array[Int](4, 5, 4, 3)
// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)
// train the model
val model = trainer.fit(train)
// compute precision on the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("precision")
println("Precision:" + evaluator.evaluate(predictionAndLabels))

{% endhighlight %}

## 1.6 one-vs-rest分类器

OneVsRest是机器学习分类方法，可以使用二分类的base classifier来构建multiclass分类。被称为"One-vs-All".

OneVsRest的实现是一个Estimator。对于base classifier，它会采用Classifier的实例，对于k类中的每个类别来创建一个二分类问题。class i的分类器被训练来预测label是不是i，以将类i与其它类区别开。

预测的完成通过评估每个二分类器来完成，输出分类器最置信的label索引。

示例：

示例展示了加载Iris dataset，解析成一个DataFrame并使用OneVsRest执行多分类。test error的计算来衡量accuracy。

{% highlight scala %}

import org.apache.spark.examples.mllib.AbstractParams
import org.apache.spark.ml.classification.{OneVsRest, LogisticRegression}
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame

val inputData = sqlContext.read.format("libsvm").load(params.input)
// compute the train/test split: if testInput is not provided use part of input.
val data = params.testInput match {
  case Some(t) => {
    // compute the number of features in the training set.
    val numFeatures = inputData.first().getAs[Vector](1).size
    val testData = sqlContext.read.option("numFeatures", numFeatures.toString)
      .format("libsvm").load(t)
    Array[DataFrame](inputData, testData)
  }
  case None => {
    val f = params.fracTest
    inputData.randomSplit(Array(1 - f, f), seed = 12345)
  }
}
val Array(train, test) = data.map(_.cache())

// instantiate the base classifier
val classifier = new LogisticRegression()
  .setMaxIter(params.maxIter)
  .setTol(params.tol)
  .setFitIntercept(params.fitIntercept)

// Set regParam, elasticNetParam if specified in params
params.regParam.foreach(classifier.setRegParam)
params.elasticNetParam.foreach(classifier.setElasticNetParam)

// instantiate the One Vs Rest Classifier.

val ovr = new OneVsRest()
ovr.setClassifier(classifier)

// train the multiclass model.
val (trainingDuration, ovrModel) = time(ovr.fit(train))

// score the model on test data.
val (predictionDuration, predictions) = time(ovrModel.transform(test))

// evaluate the model
val predictionsAndLabels = predictions.select("prediction", "label")
  .map(row => (row.getDouble(0), row.getDouble(1)))

val metrics = new MulticlassMetrics(predictionsAndLabels)

val confusionMatrix = metrics.confusionMatrix

// compute the false positive rate per label
val predictionColSchema = predictions.schema("prediction")
val numClasses = MetadataUtils.getNumClasses(predictionColSchema).get
val fprs = Range(0, numClasses).map(p => (p, metrics.falsePositiveRate(p.toDouble)))

println(s" Training Time ${trainingDuration} sec\n")

println(s" Prediction Time ${predictionDuration} sec\n")

println(s" Confusion Matrix\n ${confusionMatrix.toString}\n")

println("label\tfpr")

println(fprs.map {case (label, fpr) => label + "\t" + fpr}.mkString("\n"))

{% endhighlight %}

完整代码详见：examples/src/main/scala/org/apache/spark/examples/ml/OneVsRestExample.scala

# 2.回归



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
