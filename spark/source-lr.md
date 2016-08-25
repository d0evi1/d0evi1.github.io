---
layout: page
title: ml lr源码分析 
tagline: 介绍
---
{% include JB/setup %}

org.apache.spark.ml.classification.LogisticRegression

一、LR模型的参数

参数：

- threshold: 如果label 1的概率>threshold，则预测为1，否则为0.
- regParam：正则项参数
- elasticNetParam：ElasticNet混合参数。如果alpha = 0, 惩罚项是一个L2罚项。如果alpha = 1，它是一个L1 penalty。如果0 < alpha < 1，则是L1和L2的结果。缺省为0.0，为L2罚项。
- maxIter: 缺省100次。
- tol：迭代收敛的容忍度。值越小，会得到更高精度，同时迭代次数开销也大。缺省为1E-6.
- fitIntercept: 是否要拟合截距项(intercept term)，缺省为true.
- standardization：在对模型进行fit前，是否对训练特征进行标准化。模型的系数将总是返回到原始的scale上，它对用户是透明的。注意：当不使用正则化时，使用/不使用standardization，模型都应收敛到相同的解决方式上。在R的GLMNET包里，缺省行为也为true。缺省为true。
- weightCol：如果没设置或空，所有实例为有为1的权重。缺省不设置。
- treeAggregate：如果特征维度或分区数太大，该参数需要调得更大。缺省为2.

二、L1或L2?

- L2 regularization -> ridge 
- L1 regularization ->  lasso
- mix L1 and L2  -> elastic Net

对应到后面的代码里：

{% highlight scala %}

regPram = regParamL1+regParamL2
val regParamL1 = $(elasticNetParam) * $(regParam)
val regParamL2 = (1.0 - $(elasticNetParam)) * $(regParam)

{% endhighlight %}


MLlib底层的矩阵运算使用了Breeze库，Breeze库提供了Vector/Matrix的实现以及相应计算的接口（Linalg）。

两种正则化方法L1和L2。L2正则化假设模型参数服从高斯分布，L2正则化函数比L1更光滑，所以更容易计算；L1假设模型参数服从拉普拉斯分布，L1正则化具备产生稀疏解的功能，从而具备Feature Selection的能力。

ok, 训练过程：

{% highlight scala %}

  protected[spark] def train(dataset: Dataset[_], handlePersistence: Boolean):
      LogisticRegressionModel = {
      
    // 获取权重列表w，默认为1  => w
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    
    // 选取数据集上的labelCol、featuresCol、w  => instances
    val instances: RDD[Instance] =
      dataset.select(col($(labelCol)).cast(DoubleType), w, col($(featuresCol))).rdd.map {
        case Row(label: Double, weight: Double, features: Vector) =>
          Instance(label, weight, features)
      }


	 // 是否持久化
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    // 模型参数初始化
    val instr = Instrumentation.create(this, instances)
    instr.logParams(regParam, elasticNetParam, standardization, threshold,
      maxIter, tol, fitIntercept)

    //------------------------------------------------------
    // 对数据集进行treeAggregate操作.
    // MultiClassSummarizer: 统计label的数量，以及不同label相应的数据量，每个label的weightSum (numClasses, countInvalid, histogram)
    // MultivariateOnlineSummarizer: 计算vector的mean， variance, minimum, maximum, counts, and nonzero counts 
    //------------------------------------------------------
    val (summarizer, labelSummarizer) = {
      val seqOp = (c: (MultivariateOnlineSummarizer, MultiClassSummarizer),
        instance: Instance) =>
          (c._1.add(instance.features, instance.weight), c._2.add(instance.label, instance.weight))

      val combOp = (c1: (MultivariateOnlineSummarizer, MultiClassSummarizer),
        c2: (MultivariateOnlineSummarizer, MultiClassSummarizer)) =>
          (c1._1.merge(c2._1), c1._2.merge(c2._2))

      instances.treeAggregate(
        new MultivariateOnlineSummarizer, new MultiClassSummarizer
      )(seqOp, combOp, $(aggregationDepth))
    }

    // 得到各种统计状态字段
    val histogram = labelSummarizer.histogram
    val numInvalid = labelSummarizer.countInvalid
    val numClasses = histogram.length
    val numFeatures = summarizer.mean.size

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    instr.logNumClasses(numClasses)
    instr.logNumFeatures(numFeatures)

    /// main核心代码.
    val (coefficients, intercept, objectiveHistory) = {
      
      // 1.分类数合法性校验
      if (numInvalid != 0) {
        val msg = s"Classification labels should be in [0 to ${numClasses - 1}]. " +
          s"Found $numInvalid invalid labels."
        logError(msg)
        throw new SparkException(msg)
      }
      
      // 2.常量标签.
      val isConstantLabel = histogram.count(_ != 0) == 1
      
      // 3.参数合法性校验
      if (numClasses > 2) {
        val msg = s"LogisticRegression with ElasticNet in ML package only supports " +
          s"binary classification. Found $numClasses in the input dataset. Consider using " +
          s"MultinomialLogisticRegression instead."
        logError(msg)
        throw new SparkException(msg)
      } else if ($(fitIntercept) && numClasses == 2 && isConstantLabel) {
        logWarning(s"All labels are one and fitIntercept=true, so the coefficients will be " +
          s"zeros and the intercept will be positive infinity; as a result, " +
          s"training is not needed.")
        (Vectors.sparse(numFeatures, Seq()), Double.PositiveInfinity, Array.empty[Double])
      } else if ($(fitIntercept) && numClasses == 1) {
        logWarning(s"All labels are zero and fitIntercept=true, so the coefficients will be " +
          s"zeros and the intercept will be negative infinity; as a result, " +
          s"training is not needed.")
        (Vectors.sparse(numFeatures, Seq()), Double.NegativeInfinity, Array.empty[Double])
      } else {
      
        // 通过合法性校验.
        if (!$(fitIntercept) && isConstantLabel) {
          logWarning(s"All labels belong to a single class and fitIntercept=false. It's a " +
            s"dangerous ground, so the algorithm may not converge.")
        }
        
        // 计算部分. 判断feature是否有用,
        val featuresMean = summarizer.mean.toArray
        val featuresStd = summarizer.variance.toArray.map(math.sqrt)

        if (!$(fitIntercept) && (0 until numFeatures).exists { i =>
          featuresStd(i) == 0.0 && featuresMean(i) != 0.0 }) {
          logWarning("Fitting LogisticRegressionModel without intercept on dataset with " +
            "constant nonzero column, Spark MLlib outputs zero coefficients for constant " +
            "nonzero columns. This behavior is the same as R glmnet but different from LIBSVM.")
        }
        
        // 正则项系数：L1和L2范式. regPram = regParamL1+regParamL2
        val regParamL1 = $(elasticNetParam) * $(regParam)
        val regParamL2 = (1.0 - $(elasticNetParam)) * $(regParam)

		  
        val bcFeaturesStd = instances.context.broadcast(featuresStd)
        
        // 成本函数costFun
        val costFun = new LogisticCostFun(instances, numClasses, $(fitIntercept),
          $(standardization), bcFeaturesStd, regParamL2, multinomial = false, $(aggregationDepth))
          
          
        // 模型的使用=> optimizer.
        // (正则项为0： BreezeLBFGS，否则：BreezeOWLQN)
        val optimizer = if ($(elasticNetParam) == 0.0 || $(regParam) == 0.0) {
          new BreezeLBFGS[BDV[Double]]($(maxIter), 10, $(tol))
        } else {
        
   
        	// 标准化参数
          val standardizationParam = $(standardization)
          
          // L1正则
          def regParamL1Fun = (index: Int) => {
            // Remove the L1 penalization on the intercept
            if (index == numFeatures) {
              0.0
            } else {
              if (standardizationParam) {
                regParamL1
              } else {
                // If `standardization` is false, we still standardize the data
                // to improve the rate of convergence; as a result, we have to
                // perform this reverse standardization by penalizing each component
                // differently to get effectively the same objective function when
                // the training dataset is not standardized.
                if (featuresStd(index) != 0.0) regParamL1 / featuresStd(index) else 0.0
              }
            }
          }
          
          // 
          new BreezeOWLQN[Int, BDV[Double]]($(maxIter), 10, regParamL1Fun, $(tol))
        }

		 // 
        val initialCoefficientsWithIntercept =
          Vectors.zeros(if ($(fitIntercept)) numFeatures + 1 else numFeatures)

        if (optInitialModel.isDefined && optInitialModel.get.coefficients.size != numFeatures) {
          val vecSize = optInitialModel.get.coefficients.size
          logWarning(
            s"Initial coefficients will be ignored!! As its size $vecSize did not match the " +
            s"expected size $numFeatures")
        }

        if (optInitialModel.isDefined && optInitialModel.get.coefficients.size == numFeatures) {
          val initialCoefficientsWithInterceptArray = initialCoefficientsWithIntercept.toArray
          optInitialModel.get.coefficients.foreachActive { case (index, value) =>
            initialCoefficientsWithInterceptArray(index) = value
          }
          if ($(fitIntercept)) {
            initialCoefficientsWithInterceptArray(numFeatures) == optInitialModel.get.intercept
          }
        } else if ($(fitIntercept)) {
          /*
             For binary logistic regression, when we initialize the coefficients as zeros,
             it will converge faster if we initialize the intercept such that
             it follows the distribution of the labels.
             
               P(0) = 1 / (1 + \exp(b)), and
               P(1) = \exp(b) / (1 + \exp(b))
             , hence
             
               b = \log{P(1) / P(0)} = \log{count_1 / count_0}
             
           */
          initialCoefficientsWithIntercept.toArray(numFeatures) = math.log(
            histogram(1) / histogram(0))
        }


		  // 开始模型迭代计算，返回相应的模型状态 => states
        val states = optimizer.iterations(new CachedDiffFunction(costFun),
          initialCoefficientsWithIntercept.asBreeze.toDenseVector)


		  // 下面这段翻译：注意，在LR模型中，每次迭代的目标函数：
		  // objectiveHistory=(loss+ regularization) 是log似然的，
		  // 它在feature归一化后是不会变的。作为结果，optimizer的目标函数
		  // 与原始空间相同
        /*
           Note that in Logistic Regression, the objective history (loss + regularization)
           is log-likelihood which is invariant under feature standardization. As a result,
           the objective history from optimizer is the same as the one in the original space.
         */
        val arrayBuilder = mutable.ArrayBuilder.make[Double]
        var state: optimizer.State = null
        while (states.hasNext) {
          state = states.next()
          arrayBuilder += state.adjustedValue
        }

        if (state == null) {
          val msg = s"${optimizer.getClass.getName} failed."
          logError(msg)
          throw new SparkException(msg)
        }

		  // 翻译：系数在被归一化后的空间中进行训练。我们需要将它们转换成原始参数空间上
		  // 注意：归一化空间中的intercept与原始空间上相同，不需要归一化
        /*
           The coefficients are trained in the scaled space; we're converting them back to
           the original space.
           Note that the intercept in scaled space and original space is the same;
           as a result, no scaling is needed.
         */
        val rawCoefficients = state.x.toArray.clone()
        var i = 0
        while (i < numFeatures) {
          rawCoefficients(i) *= { if (featuresStd(i) != 0.0) 1.0 / featuresStd(i) else 0.0 }
          i += 1
        }
        bcFeaturesStd.destroy(blocking = false)

        if ($(fitIntercept)) {
          (Vectors.dense(rawCoefficients.dropRight(1)).compressed, rawCoefficients.last,
            arrayBuilder.result())
        } else {
          (Vectors.dense(rawCoefficients).compressed, 0.0, arrayBuilder.result())
        }
      }
    }

	 // 解除持久化
    if (handlePersistence) instances.unpersist()

	 // 生成model
    val model = copyValues(new LogisticRegressionModel(uid, coefficients, intercept))
    
    // 
    val (summaryModel, probabilityColName) = model.findSummaryModelAndProbabilityCol()
    
    // 得到对应的Summary. 
    val logRegSummary = new BinaryLogisticRegressionTrainingSummary(
      summaryModel.transform(dataset),
      probabilityColName,
      $(labelCol),
      $(featuresCol),
      objectiveHistory)
      
    // 返回m.
    val m = model.setSummary(logRegSummary)
    instr.logSuccess(m)
    m
  }


{% endhighlight %}


二、LBFGS和OWLQN


ok，我们知道，模型本身基本上核心代码就落在了这两个方法上：LBFGS和OWLQN。我们再看一下breeze库里的，这两个方法：

- breeze.optimize.LBFGS
- breeze.optimize.OWLQN

L-BFGS: Limited-memory BFGS。其中BFGS代表4个人的名字：broyden–fletcher–goldfarb–shanno
OWL-QN: (Orthant-Wise Limited-Memory Quasi-Newton)算法。

先来简单看一下breezey库。

breeze库：[https://github.com/scalanlp/breeze](https://github.com/scalanlp/breeze)

breeze库用于数值处理。它的目标是通用、简洁、强大，不需要牺牲太多性能。当前版本0.12。我们所熟悉的spark中的MLlib（经常见到的线性算法库：breeze.linalg、最优化算法库：breeze.optimize）就是在它的基础上构建的。另外它还提供了图绘制的功能（breeze.plot）

相关文档：[https://github.com/scalanlp/breeze/wiki/Quickstart](https://github.com/scalanlp/breeze/wiki/Quickstart)

[LBFGS对应的代码](https://github.com/scalanlp/breeze/blob/master/math/src/main/scala/breeze/optimize/LBFGS.scala)

[OW-LQN对应的代码](https://github.com/scalanlp/breeze/blob/master/math/src/main/scala/breeze/optimize/OWLQN.scala)


三、成本函数

{% highlight scala %}

/**
 * LogisticCostFun implements Breeze's DiffFunction[T] for a multinomial (softmax) logistic loss
 * function, as used in multi-class classification (it is also used in binary logistic regression).
 * It returns the loss and gradient with L2 regularization at a particular point (coefficients).
 * It's used in Breeze's convex optimization routines.
 */
private class LogisticCostFun(
    instances: RDD[Instance],
    numClasses: Int,
    fitIntercept: Boolean,
    standardization: Boolean,
    bcFeaturesStd: Broadcast[Array[Double]],
    regParamL2: Double,
    multinomial: Boolean,
    aggregationDepth: Int) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val coeffs = Vectors.fromBreeze(coefficients)
    val bcCoeffs = instances.context.broadcast(coeffs)
    val featuresStd = bcFeaturesStd.value
    val numFeatures = featuresStd.length

    val logisticAggregator = {
      val seqOp = (c: LogisticAggregator, instance: Instance) => c.add(instance)
      val combOp = (c1: LogisticAggregator, c2: LogisticAggregator) => c1.merge(c2)

      instances.treeAggregate(
        new LogisticAggregator(bcCoeffs, bcFeaturesStd, numClasses, fitIntercept,
          multinomial)
      )(seqOp, combOp, aggregationDepth)
    }

    val totalGradientArray = logisticAggregator.gradient.toArray
    // regVal is the sum of coefficients squares excluding intercept for L2 regularization.
    val regVal = if (regParamL2 == 0.0) {
      0.0
    } else {
      var sum = 0.0
      coeffs.foreachActive { case (index, value) =>
        // We do not apply regularization to the intercepts
        val isIntercept = fitIntercept && ((index + 1) % (numFeatures + 1) == 0)
        if (!isIntercept) {
          // The following code will compute the loss of the regularization; also
          // the gradient of the regularization, and add back to totalGradientArray.
          sum += {
            if (standardization) {
              totalGradientArray(index) += regParamL2 * value
              value * value
            } else {
              val featureIndex = if (fitIntercept) {
                index % (numFeatures + 1)
              } else {
                index % numFeatures
              }
              if (featuresStd(featureIndex) != 0.0) {
                // If `standardization` is false, we still standardize the data
                // to improve the rate of convergence; as a result, we have to
                // perform this reverse standardization by penalizing each component
                // differently to get effectively the same objective function when
                // the training dataset is not standardized.
                val temp = value / (featuresStd(featureIndex) * featuresStd(featureIndex))
                totalGradientArray(index) += regParamL2 * temp
                value * temp
              } else {
                0.0
              }
            }
          }
        }
      }
      0.5 * regParamL2 * sum
    }
    bcCoeffs.destroy(blocking = false)

    (logisticAggregator.loss + regVal, new BDV(totalGradientArray))
  }
}

{% endhighlight %}



参考：

1.[LogisticRegression源码](https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala)