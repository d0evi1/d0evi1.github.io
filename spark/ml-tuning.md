---
layout: page
title: ml 模型选择与参数调优 
tagline: 介绍
---
{% include JB/setup %}


# 一、模型选择(a.k.a 参数调优)

ML中一个很重要的任务是模型选择（model selection），或者对于给定任务，使用数据来发现最佳的模型或参数。这被称为参数调优（tuning）。Tuning可以在单个Estimators（比如：LogisticRegression）上进行，也可以在整个Pipeline上（可包含多个算法，特征化及其它步骤）进行。

MLlib支持模型选择工具： CrossValidator 和 TrainValidationSplit。这些工具需要以下的item：

- Estimator：要调优的算法或Pipeline
- ParamMap集合：要选择的参数，有时称为“parameter grid”穷举搜索
- Evaluator：要计算的metric，如何更好地对fit后的Model在测试数据上进行评估

这些模型选择工具按以下步骤工作：

- 将数据split成独立的训练集和测试集
- 对于每个(training,test) pair，迭代整个ParamMap参数空间；对于每个ParamMap，它们都会使用这些参数对Estimator进行fit，得到对应fitted后的Model，然后使用valuator评估该Model的性能。
- 选择最好性能的参数集生成模型

对于回归问题，Evaluator可以是RegressionEvaluator；对于二元分类问题，可以使用 BinaryClassificationEvaluator；对于多分类问题，可以使用MulticlassClassificationEvaluator。缺省的metric用于选择最好的ParamMap，对于每个这样的Evaluator，可以通过setMetricName方法进行override。

为了构建parameter grid，用户可以使用ParamGridBuilder 工具类。

# 二、Cross-Validation

CrossValidator会将数据集分割成几个folds，它们可以用于独立的训练集和测试集。例如：k=3 folds时，CrossValidator会生成3个(training, test) pair，每个都会使用2/3的数据作为训练集，1/3作为测试集。为了评估一个特定的ParamMap，对于在3个不同的数据pair上使用Estimator 进行fit产生3个模型，CrossValidator会计算三个evaluation metric的平均值。

在选择效果最好的ParamMap之后，CrossValidator最后会使用相应的Estimator，和最好的ParamMap，对整个数据集进行refit。

## 2.1 示例：通过cross-validation进行模型选择

下例展示了如何使用CrossValidator来选择参数。

注意，在一个参数空间内进行cross-validation是相当昂贵的。例如，在下面的示例中，param grid中的hashingTF.numFeatures具有3个值，而lr.regParam具有2个值，CrossValidator使用2-folds。这会生成(3x2)x2=12种要训练的不同模型。在实际设置中，尝试很多参数、以及使用很多folds（k=3或k=10都很常用）是很常见的。换句话说，使用CrossValidator是非常昂贵的，然后，它也是选择参数的很受认可的方法，它比启发式的手工调参更权威。

{% highlight scala %}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row

// Prepare training data from a list of (id, text, label) tuples.
val training = spark.createDataFrame(Seq(
  (0L, "a b c d e spark", 1.0),
  (1L, "b d", 0.0),
  (2L, "spark f g h", 1.0),
  (3L, "hadoop mapreduce", 0.0),
  (4L, "b spark who", 1.0),
  (5L, "g d a y", 0.0),
  (6L, "spark fly", 1.0),
  (7L, "was mapreduce", 0.0),
  (8L, "e spark program", 1.0),
  (9L, "a e c l", 0.0),
  (10L, "spark compile", 1.0),
  (11L, "hadoop software", 0.0)
)).toDF("id", "text", "label")

// Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
val tokenizer = new Tokenizer()
  .setInputCol("text")
  .setOutputCol("words")
val hashingTF = new HashingTF()
  .setInputCol(tokenizer.getOutputCol)
  .setOutputCol("features")
val lr = new LogisticRegression()
  .setMaxIter(10)
val pipeline = new Pipeline()
  .setStages(Array(tokenizer, hashingTF, lr))

// We use a ParamGridBuilder to construct a grid of parameters to search over.
// With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
// this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
val paramGrid = new ParamGridBuilder()
  .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
  .addGrid(lr.regParam, Array(0.1, 0.01))
  .build()

// We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
// This will allow us to jointly choose parameters for all Pipeline stages.
// A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
// Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
// is areaUnderROC.
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(2)  // Use 3+ in practice

// Run cross-validation, and choose the best set of parameters.
val cvModel = cv.fit(training)

// Prepare test documents, which are unlabeled (id, text) tuples.
val test = spark.createDataFrame(Seq(
  (4L, "spark i j k"),
  (5L, "l m n"),
  (6L, "mapreduce spark"),
  (7L, "apache hadoop")
)).toDF("id", "text")

// Make predictions on test documents. cvModel uses the best model found (lrModel).
cvModel.transform(test)
  .select("id", "text", "probability", "prediction")
  .collect()
  .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
    println(s"($id, $text) --> prob=$prob, prediction=$prediction")
  }

{% endhighlight %}

完整代码：examples/src/main/scala/org/apache/spark/examples/ml/ModelSelectionViaCrossValidationExample.scala

# 3. Train-Validation Split

Spark中，除了CrossValidator，还提供了TrainValidationSplit来进行参数调优。TrainValidationSplit只评估一次每次参数组合，而CrossValidator则要进行k次。它的开销更小，当训练数据集不够大时，不会产生可靠的结果。

不像CrossValidator，TrainValidationSplit会创建单个 (training, test) pair。它使用trainRatio参数将数据集split成两部分。例如：  trainRatio=0.75，TrainValidationSplit会生成一个训练集（75%）和一个测试集（25%）。

和CrossValidator类似，TrainValidationSplit最后会使用Estimator、以及最好的ParamMap，对整个数据集进行fit。

示例：通过TrainValidationSplit进行模型选择

{% highlight scala %}

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

// Prepare training and test data.
val data = spark.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")
val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)

val lr = new LinearRegression()

// We use a ParamGridBuilder to construct a grid of parameters to search over.
// TrainValidationSplit will try all combinations of values and determine best model using
// the evaluator.
val paramGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(0.1, 0.01))
  .addGrid(lr.fitIntercept)
  .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
  .build()

// In this case the estimator is simply the linear regression.
// A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
val trainValidationSplit = new TrainValidationSplit()
  .setEstimator(lr)
  .setEvaluator(new RegressionEvaluator)
  .setEstimatorParamMaps(paramGrid)
  // 80% of the data will be used for training and the remaining 20% for validation.
  .setTrainRatio(0.8)

// Run train validation split, and choose the best set of parameters.
val model = trainValidationSplit.fit(training)

// Make predictions on test data. model is the model with combination of parameters
// that performed best.
model.transform(test)
  .select("features", "label", "prediction")
  .show()

{% endhighlight %}

参考：

1.[http://spark.apache.org/docs/latest/ml-tuning.html](http://spark.apache.org/docs/latest/ml-tuning.html)
