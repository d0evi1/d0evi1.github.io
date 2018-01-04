---
layout: page
title: mllib的features(提取、转换、选择) 
tagline: 介绍
---
{% include JB/setup %}


* [全文目录](#1)

* [一、特征抽取](#1)

* [二、特征转换](#2)

* [三、特征选择](#3)


<h2 id="1"></h2>

# 一、特征抽取

## 1.1 TF-IDF

## 1.2 Word2Vec

Word2Vec是一个Estimator，它接收文档的word序列作为参数，训练一个Word2VecModel，该model会将每个word映射到一个唯一的固定大小的vector中。Word2VecModel将document中的所有word的平均，将每个document转换成一个vector；该vector可以被用在预测中使用的feature，也可用于相似度计算等等。可详见 [MLlib Word2Vec guide](http://spark.apache.org/docs/latest/mllib-feature-extraction.html#word2vec)

下面的代码段中，我们使用一个文档集，每个文档都表示成一个word序列串。对于每个文档，我们将它转换成一个feature vector。该feature vector可以作为参数传给相应的机器学习算法。

{% highlight scala %}

import org.apache.spark.ml.feature.Word2Vec

// Input data: Each row is a bag of words from a sentence or document.
val documentDF = spark.createDataFrame(Seq(
  "Hi I heard about Spark".split(" "),
  "I wish Java could use case classes".split(" "),
  "Logistic regression models are neat".split(" ")
).map(Tuple1.apply)).toDF("text")

// Learn a mapping from words to Vectors.
val word2Vec = new Word2Vec()
  .setInputCol("text")
  .setOutputCol("result")
  .setVectorSize(3)
  .setMinCount(0)
val model = word2Vec.fit(documentDF)
val result = model.transform(documentDF)
result.select("result").take(3).foreach(println)

{% endhighlight %}


## 1.3 CountVectorizer

CountVectorizer和CountVectorizerModel目标是帮助将一系列文本文档转换成token count的向量。如果不存在一个a-priori字典，可以使用CountVectorizer作为一个Estimator来抽取词汇表，生成一个CountVectorizerModel模型。这个模型会根据词汇表生成文档的稀疏表示，接着可以传到其它算法中（比如LDA）。

在进行fit时，CountVectorizer会选择vocabSize的top个词汇，根据语料上的词频排序。一个可选的参数是minDF，通过指定在词汇表中的单个词汇出现的最小数量的文档数，它也会影响fit过程。另一个可选的二元toggle参数，可以控制输出向量。如果设为true，所有非零的count会被设置为1.这对于一些二进制count的（非整型）离散概率模型来说很有用。

示例：

下面的DataFrame，具有两列：id和texts：

 id | texts
:----:|:----------:
 0  | Array("a", "b", "c")
 1  | Array("a", "b", "b", "c", "a")

texts中的每行是一个类型为Array[String]的文档，通过调用CountVectorizer的fit，会生成一个CountVectorizerModel模型，具有词汇表(a,b,c)。接着是转换后的输出列，"vector"：

 id | texts                           | vector
:----:|:---------------------------------:|:---------------:
 0  | Array("a", "b", "c")            | (3,[0,1,2],[1.0,1.0,1.0])
 1  | Array("a", "b", "b", "c", "a")  | (3,[0,1,2],[2.0,2.0,1.0])

每个vector表示文档在词汇表中的token count。

{% highlight scala %}

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

val df = spark.createDataFrame(Seq(
  (0, Array("a", "b", "c")),
  (1, Array("a", "b", "b", "c", "a"))
)).toDF("id", "words")

// fit a CountVectorizerModel from the corpus
val cvModel: CountVectorizerModel = new CountVectorizer()
  .setInputCol("words")
  .setOutputCol("features")
  .setVocabSize(3)
  .setMinDF(2)
  .fit(df)

// alternatively, define CountVectorizerModel with a-priori vocabulary
val cvm = new CountVectorizerModel(Array("a", "b", "c"))
  .setInputCol("words")
  .setOutputCol("features")

cvModel.transform(df).select("features").show()

{% endhighlight %}

代码详见：examples/src/main/scala/org/apache/spark/examples/ml/CountVectorizerExample.scala

<h2 id="2"></h2>

# 二、特征转换

## 2.1 Tokenizer 

Tokenization是个文本处理过程，它将句子断成独立的terms（通常为words）。[Tokenizer](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.Tokenizer)提供了该功能。下面的示例展示了如何将句子分割成一串词。

[RegexTokenizer](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.RegexTokenizer)允许更多高级特性的tokenization: 正则表达式匹配. 缺省的，参数"pattern"(regex, default:"\\s+")被用于使用空格分割输入文本。可选的，用户可以设置参数"gaps"=false，这表示正则项"pattern"为"tokens"，而非分割gaps，可以找出所有匹配出现的词作为tokenization的结果。

{% highlight scala %}

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}

val sentenceDataFrame = spark.createDataFrame(Seq(
  (0, "Hi I heard about Spark"),
  (1, "I wish Java could use case classes"),
  (2, "Logistic,regression,models,are,neat")
)).toDF("label", "sentence")

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val regexTokenizer = new RegexTokenizer()
  .setInputCol("sentence")
  .setOutputCol("words")
  .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

val tokenized = tokenizer.transform(sentenceDataFrame)
tokenized.select("words", "label").take(3).foreach(println)
val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
regexTokenized.select("words", "label").take(3).foreach(println)

{% endhighlight %}

完整代码在：examples/src/main/scala/org/apache/spark/examples/ml/TokenizerExample.scala

## 2.2 StopWordsRemover

停留词（stop words）指的是那些需要从输入排除在外的词，通常是因为这些词出现很频繁，并且没啥含义。

StopWordsRemover，允许一串字符串作为输入（比如：Tokenizer的输出），并从输入的序列中抛弃掉停留词。停留词列表由stopWords参数来指定。我们缺省提供了一个[stop words的列表](http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words)，可以通过在一个刚初始化的StopWordsRemover实例上调用getStopWrods来获取。caseSensitive参数表示是否大小写敏感（缺省为false）。

示例：

假设我们具有以下的DataFrame，具有两列（id和raw）：

 id | raw
----|----------
 0  | [I, saw, the, red, baloon]
 1  | [Mary, had, a, little, lamb]

使用StopWordsRemover，输入为raw，输出为filtered，我们可以获得以下的结果：

 id | raw                         | filtered
----|-----------------------------|--------------------
 0  | [I, saw, the, red, baloon]  |  [saw, red, baloon]
 1  | [Mary, had, a, little, lamb]|[Mary, little, lamb]


在filtered列，停留词“I”, “the”, “had”, 和 “a” 都已经被过滤掉了。

{% highlight scala %}

import org.apache.spark.ml.feature.StopWordsRemover

val remover = new StopWordsRemover()
  .setInputCol("raw")
  .setOutputCol("filtered")

val dataSet = sqlContext.createDataFrame(Seq(
  (0, Seq("I", "saw", "the", "red", "baloon")),
  (1, Seq("Mary", "had", "a", "little", "lamb"))
)).toDF("id", "raw")

remover.transform(dataSet).show()

{% endhighlight %}


## 2.3 n-gram

n-gram是一个tokens序列，n为整数。NGram可以用于将输入特征转换成n-gram。





## 2.4 Binarizer

Binarization会将数值型特征（numerical features）的阀值压到二元feature中(0/1)。

Binarizer使用公共参数：inputCol 和 outputCol，threshold来进行二值化。Feature的值如果大于threshold，就二值化为1.0；如果值小于等于threshold，就二值化为0.0. inputCol 支持：Vector和Double类型。

{% highlight scala %}

import org.apache.spark.ml.feature.Binarizer

val data = Array((0, 0.1), (1, 0.8), (2, 0.2))
val dataFrame = spark.createDataFrame(data).toDF("label", "feature")

val binarizer: Binarizer = new Binarizer()
  .setInputCol("feature")
  .setOutputCol("binarized_feature")
  .setThreshold(0.5)

val binarizedDataFrame = binarizer.transform(dataFrame)
val binarizedFeatures = binarizedDataFrame.select("binarized_feature")
binarizedFeatures.collect().foreach(println)

{% endhighlight %}

## 2.5 PCA

PCA是一个统计过程，它使用正交变换，将一个可能相关变量的观察集，转换成一个线性不相关变量的集合，这些个变量集称为主成分(principal components)。PCA类会训练一个模型，会使用PCA技术将向量投影到一个低维空间上。下例展示了，如何将一个5维的features投影到3维的主成分上。

{% highlight scala %}

import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors

val data = Array(
  Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
  Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
  Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
)
val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
val pca = new PCA()
  .setInputCol("features")
  .setOutputCol("pcaFeatures")
  .setK(3)
  .fit(df)
val pcaDF = pca.transform(df)
val result = pcaDF.select("pcaFeatures")
result.show()

{% endhighlight %}

代码详见：examples/src/main/scala/org/apache/spark/examples/ml/PCAExample.scala。

## 2.6 多项式展开（PolynomialExpansion）

[多项式展开](http://en.wikipedia.org/wiki/Polynomial_expansion)可以将你的特征展开到一个多项式空间上，它可以将原始维展开到n-degree组合上。PolynomialExpansion类提供了这个功能。下面的示例展示了将你的特征展开到3-degree的多项式空间上。

{% highlight scala %}

import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vectors

val data = Array(
  Vectors.dense(-2.0, 2.3),
  Vectors.dense(0.0, 0.0),
  Vectors.dense(0.6, -1.1)
)
val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
val polynomialExpansion = new PolynomialExpansion()
  .setInputCol("features")
  .setOutputCol("polyFeatures")
  .setDegree(3)
val polyDF = polynomialExpansion.transform(df)
polyDF.select("polyFeatures").take(3).foreach(println)


{% endhighlight %}

更多代码：examples/src/main/scala/org/apache/spark/examples/ml/PolynomialExpansionExample.scala

## 2.7 离散cosine变换(DCT)

[Discrete Cosine Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform)可以将在时间域（time domain）上长度为N的实数值序列，转换成另一个在频率域（frequency domain）内长度为N的实数值序列。DCT类提供了这个功能，将实现了[DCT-II](https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II)并将结果通过\$
1/\sqrt_2 \$进行归一化，以便将变换后的矩阵单元化。不需要在转换序列上进行shift操作（例如：转换序列的第0th位的元素，对应于DCT的第0th coefficient，而非第N/2 th）

DCT示例：

{% highlight scala %}

import org.apache.spark.ml.feature.DCT
import org.apache.spark.ml.linalg.Vectors

val data = Seq(
  Vectors.dense(0.0, 1.0, -2.0, 3.0),
  Vectors.dense(-1.0, 2.0, 4.0, -7.0),
  Vectors.dense(14.0, -2.0, -5.0, 1.0))

val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

val dct = new DCT()
  .setInputCol("features")
  .setOutputCol("featuresDCT")
  .setInverse(false)

val dctDf = dct.transform(df)
dctDf.select("featuresDCT").show(3)

{% endhighlight %}

更多代码：examples/src/main/scala/org/apache/spark/examples/ml/DCTExample.scala


## 2.8 StringIndexer

StringIndexer将一列string列的label编码成一列label索引。索引indice的值为[0, numLabels)，按label频次排序，最频繁的label会得到index 0，如果输入列是数值型的（numeric），我们可以将它转换成string，并索引string值。当对pipeline组件（比如：Estimator 或 Transformer）进行downstream时，可以利用string-indexed的label，你必须将该组件的输入列设置成string-indexed列名。在许多情况下，你可以使用setInputCol来完成。

示例：

假设，我们具有以下的DataFrame，具有两列: id和category。

 id | category
:----:|:----------:
 0  | a
 1  | b
 2  | c
 3  | a
 4  | a
 5  | c

category有三种label："a", "b"和"c"。使用StringIndexer对category作为输入列，categoryIndex作为输出列，我们可以得到下面的：

 id | category | categoryIndex
:----:|:----------:|:---------------:
 0  | a        | 0.0
 1  | b        | 2.0
 2  | c        | 1.0
 3  | a        | 0.0
 4  | a        | 0.0
 5  | c        | 1.0

“a”最频繁，所以为index=0，"c"的index为1，而"b"的index为2.

另外，要牢记，当你在一个数据集上使用StringIndexer进行fit，并使用它进行转换时，有两种策略来使用StringIndexer处理看不到的label：

- 抛出一个exception（缺省方式）
- 跳过包含unseen label的整行

示例：

回到之前的示例，数据集为：

 id | category
:----:|:----------:
 0  | a
 1  | b
 2  | c
 3  | d

如果你没有设置StringIndexer是如何处理未看到的label的，缺省下会抛出异常。然而，如果你调用：setHandleInvalid("skip")，会生成下面的数据集：

 id | category | categoryIndex
:----:|:----------:|:---------------:
 0  | a        | 0.0
 1  | b        | 2.0
 2  | c        | 1.0

注意："d"行未包含其中。

{% highlight scala %}

import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(
  Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)
indexed.show()

{% endhighlight %}



## 2.9 IndexToString

与StringIndexer相同，IndexToString会将一列label indices映射回一个包含原始label的列，作为String。常见的用例是：使用StringIndexer产生label索引，使用这些索引训练模型，并使用IndexToString，从预测的结果列中检索回原始的label。当然，你也可以自由地以你自定义的label方式工作。

示例：

使用StringIndexer构建的示例，假设我们有以下的DataFrame，具有列：id 和 categoryIndex: 

 id | categoryIndex
:----:|:---------------:
 0  | 0.0
 1  | 2.0
 2  | 1.0
 3  | 0.0
 4  | 0.0
 5  | 1.0

接着，我们使用IndexToString，将categoryIndex为作输入列，originalCategory作为输出列，我们可以检索回我们的原始label：

id | categoryIndex | originalCategory
:----:|:---------------:|:-----------------:
 0  | 0.0           | a
 1  | 2.0           | b
 2  | 1.0           | c
 3  | 0.0           | a
 4  | 0.0           | a
 5  | 1.0           | c

示例代码：

{% highlight scala %}

import org.apache.spark.ml.feature.{IndexToString, StringIndexer}

val df = spark.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c")
)).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")
  .fit(df)
val indexed = indexer.transform(df)

val converter = new IndexToString()
  .setInputCol("categoryIndex")
  .setOutputCol("originalCategory")

val converted = converter.transform(indexed)
converted.select("id", "originalCategory").show()

{% endhighlight %}

代码详见：examples/src/main/scala/org/apache/spark/examples/ml/IndexToStringExample.scala


## 2.10 OneHotEncoder

One-hot encoding会将一列label索引映射到一列二元表示的向量，至多只有一个值。这种编码允许某些期望是连续型feature的算法，比如：Logistic Regression，来使用类别型的feature。

{% highlight scala %}

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

val df = spark.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c")
)).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")
  .fit(df)
val indexed = indexer.transform(df)

val encoder = new OneHotEncoder()
  .setInputCol("categoryIndex")
  .setOutputCol("categoryVec")
val encoded = encoder.transform(indexed)
encoded.select("id", "categoryVec").show()

{% endhighlight %}

示例：examples/src/main/scala/org/apache/spark/examples/ml/OneHotEncoderExample.scala


## 2.11 VectorIndexer

VectorIndex可以用于索引Vectors类型数据集中的类别型feature。它可以自动决定哪个feature是类别型的（categorical），并将它的原始值转换成类别id索引。必须按照以下操作进行：

- 使用一个Vector类型的输入列，和一个参数maxCategories
- 基于不同的值来决定哪个feature应该是类别型的，features的最大类别数由maxCategories来指定
- 为每个类别feature，计算从0开始的类别id
- 索引类别型feature，将原始feature值转换成索引id

类别型feature的索引，允许类似的算法：决策树，Ensembles树来合理地处理类别型feature，以提升性能。

在下面的示例中：

{% highlight scala %}

import org.apache.spark.ml.feature.VectorIndexer

val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val indexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexed")
  .setMaxCategories(10)

val indexerModel = indexer.fit(data)

val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
println(s"Chose ${categoricalFeatures.size} categorical features: " +
  categoricalFeatures.mkString(", "))

// Create new column "indexed" with categorical values transformed to indices
val indexedData = indexerModel.transform(data)
indexedData.show()

{% endhighlight %}


## 2.12 Normalizer

Normalizer是一个Transformer，它可以将多行Vector的数据集进行转换，将每个Vector归一化成unit norm形式。它使用参数p，该参数指定了[p-norm](https://en.wikipedia.org/wiki/Norm_%28mathematics%29#p-norm)用于归一化。(缺省：p=2)，该归一化能将你的输入数据标准化，并改善学习算法的行为。

下面的示例，展示了如何以libsvm的格式加载数据集，接着对每行进行L2范式的归一化，和其它L无穷范式的归一化。

{% highlight scala %}

import org.apache.spark.ml.feature.Normalizer

val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Normalize each Vector using $L^1$ norm.
val normalizer = new Normalizer()
  .setInputCol("features")
  .setOutputCol("normFeatures")
  .setP(1.0)

val l1NormData = normalizer.transform(dataFrame)
l1NormData.show()

// Normalize each Vector using $L^\infty$ norm.
val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
lInfNormData.show()

{% endhighlight %}


## 2.13 StandardScaler

StandardScaler可以将多行的Vector数据进进行转换，将每个feature归一化到具有单位标准差，和零均值的范围上。它的参数有：

- withStd: 缺省为True。将数据归一化到标准差=1的范围上
- withMean: 缺省为False。在归一化前，将数据进行中心化。它会构建一个dense型的输出，因此，在稀疏数据集上对它进行操作会抛出一个异常。

StandardScaler是一个Estimator，它可以在数据集上进行fit操作，产生一个StandardScalerModel；这相当于计算总的统计。接着，该模型可以将数据集中的某个Vector列，转换成具有标准差为1和/或零均值的feature中。

注意：如果feature的标准差为0，它在Vector的对应feature上返回缺省的0.0值。

下面的示例展示了，如何在libsvm格式加载一个数据集，并接着将每个feature归一化到标准差为1.

{% highlight scala %}

import org.apache.spark.ml.feature.StandardScaler

val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false)

// Compute summary statistics by fitting the StandardScaler.
val scalerModel = scaler.fit(dataFrame)

// Normalize each feature to have unit standard deviation.
val scaledData = scalerModel.transform(dataFrame)
scaledData.show()

{% endhighlight %}

示例详见：examples/src/main/scala/org/apache/spark/examples/ml/StandardScalerExample.scala

## 2.14 MinMaxScaler

MinMaxScaler可以对多行的Vector数据集进行转换，将每个feature归一化到一个指定的范围内（比如：常用的[0,1]）。它的参数这：

- min: 缺省0.0. 转换的下界，所有features都共用.
- max: 缺省1.0. 转换的上界，所有features都共用.

MinMaxScaler会计算在数据集上的汇总统计，并产生一个MinMaxScalerModel。该模型可以将每个feature独自地转换到该指定范围中。

归一化(rescale)计算如下：

$$
Rescaled(e_i)=\frac{e_i-E_{min}}{E_{max}-E_{min}}*(max-min)+min
$$

如果：\$ E_{max}==E_{min} \$，那么:  \$Rescaled(e_i)=0.5*(max+min) \$

注意：由于零值可能被转换成非零值，对于sparse的输入，转换器的输出也必须是DenseVector。

下例展示了如何以libsvm的格式加载数据集，将将每个feature归一化到[0,1]中。

{% highlight scala %}

import org.apache.spark.ml.feature.MinMaxScaler

val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val scaler = new MinMaxScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")

// Compute summary statistics and generate MinMaxScalerModel
val scalerModel = scaler.fit(dataFrame)

// rescale each feature to range [min, max].
val scaledData = scalerModel.transform(dataFrame)
scaledData.show()

{% endhighlight %}

完整示例：examples/src/main/scala/org/apache/spark/examples/ml/MinMaxScalerExample.scala

## 2.15 MaxAbsScaler

MaxAbsScaler将每个feature转换成[-1, 1]上，通过每个feature的最大绝对值。它不会shift/center化数据，因为这样会毁了数据的稀疏性。

MaxAbsScaler会计算数据集上的汇总统计信息，并产生一个MaxAbsScalerModel。接着使用该model将每个feature单独转换到[-1,1]的范围。

下例展示了如何加载libsvm的数据集，接着将每个feature归一化到[-1,1]上。

{% highlight scala %}

import org.apache.spark.ml.feature.MaxAbsScaler

val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
val scaler = new MaxAbsScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")

// Compute summary statistics and generate MaxAbsScalerModel
val scalerModel = scaler.fit(dataFrame)

// rescale each feature to range [-1, 1]
val scaledData = scalerModel.transform(dataFrame)
scaledData.show()

{% endhighlight %}

完整的示例如下：examples/src/main/scala/org/apache/spark/examples/ml/MaxAbsScalerExample.scala

## 2.16 Bucketizer

Bucketizer会将一列连续的feature转换成一列feature buckets，对应的buckets由用户指定。它的参数如下：

- split: 该参数用于将连续型特征(continuous features)映射成buckets。如果split为n+1，则表示有n个buckets。一个bucket的定义通过在[x,y)范围上的值进行split，除了最后一个bucket，它也包含y。splits应该是严格递增的。在(-inf,inf)上的值必须显式提供以覆盖所有Double值；否则，splits外指定的值会被认为是错误的。splits的两个示例：Array(Double.NegativeInfinity, 0.0, 1.0, Double.PositiveInfinity)，Array(0.0, 1.0, 2.0)。

注意，如果你不知道目标列的上界与下界，你应该添加Double.NegativeInfinity 和 Double.PositiveInfinity作为你的边界，以防止Bucketizer 的潜在异常。

注意，splits必须严格递增：比如：s0<s1<s2<...<sn.

更多细节，详见： [Bucketizer API](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.Bucketizer)

下面的示例，展示了如何将Double列分桶成另一个index-wised列。

{% highlight scala %}

import org.apache.spark.ml.feature.Bucketizer

val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)

val data = Array(-0.5, -0.3, 0.0, 0.2)
val dataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

val bucketizer = new Bucketizer()
  .setInputCol("features")
  .setOutputCol("bucketedFeatures")
  .setSplits(splits)

// Transform original data into its bucket index.
val bucketedData = bucketizer.transform(dataFrame)
bucketedData.show()

{% endhighlight %}

完整代码：examples/src/main/scala/org/apache/spark/examples/ml/BucketizerExample.scala

## 2.17 ElementwiseProduct

ElementwiseProduct会将每个输入向量以及一个由它提供的“weight”向量相乘，使用element-wise乘法。换句话说，它将数据集的每一列都乘以一个scalar因子进行归一化。这种方式表示了input向量v，和转换向量w之间的[Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_%28matrices%29)，产生一个新的结果向量。

示例：

{% highlight scala %}

import org.apache.spark.ml.feature.ElementwiseProduct
import org.apache.spark.ml.linalg.Vectors

// Create some vector data; also works for sparse vectors
val dataFrame = spark.createDataFrame(Seq(
  ("a", Vectors.dense(1.0, 2.0, 3.0)),
  ("b", Vectors.dense(4.0, 5.0, 6.0)))).toDF("id", "vector")

val transformingVector = Vectors.dense(0.0, 1.0, 2.0)
val transformer = new ElementwiseProduct()
  .setScalingVec(transformingVector)
  .setInputCol("vector")
  .setOutputCol("transformedVector")

// Batch transform the vectors to create new column:
transformer.transform(dataFrame).show()

{% endhighlight %}

## 2.18 SQLTransformer

SQLTransformer实现了由sql语句定义的转换。当前，我们只支持这样的SQL语法："SELECT ... FROM _____THIS_____ ..." 其中的 “_____THIS_____”表示了输入数据集的底层表。select语句指定了在输出中展示相应的字段，常量，表达式，可以是任何Spark SQL支持的select语句。用户可以使用Spark SQL的内建函数和UDF来操作选中的列。例如，SQLTransformer支持如下的声明：

- SELECT a, a + b AS a_b FROM _____THIS_____
- SELECT a, SQRT(b) AS b_sqrt FROM _____THIS_____ where a > 5
- SELECT a, b, SUM(c) AS c_sum FROM _____THIS_____ GROUP BY a, b

示例：

DataFrame有三列：id, v1, v2

 id |  v1 |  v2
:----:|:-----:|:-----:
 0  | 1.0 | 3.0  
 2  | 2.0 | 5.0

SQLTransformer相应的输出可以使用这样的语句：“SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM _____THIS_____”

 id |  v1 |  v2 |  v3 |  v4
:----:|:-----:|:-----:|:-----:|:-----:
 0  | 1.0 | 3.0 | 4.0 | 3.0
 2  | 2.0 | 5.0 | 7.0 |10.0


{% highlight scala %}

import org.apache.spark.ml.feature.SQLTransformer

val df = spark.createDataFrame(
  Seq((0, 1.0, 3.0), (2, 2.0, 5.0))).toDF("id", "v1", "v2")

val sqlTrans = new SQLTransformer().setStatement(
  "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")

sqlTrans.transform(df).show()

{% endhighlight %}

代码：examples/src/main/scala/org/apache/spark/examples/ml/SQLTransformerExample.scala

## 2.19 VectorAssembler

VectorAssembler是一个transformer，它可以将给定的多列转换成单个vector列。这对于将多个raw features、以及由不同的feature transformers生成的features，组合成单个feature vector，以便于训练ML model（比如logistic regression 和决策树）。VectorAssembler接受下面的输入列类型：所有numeric类型，boolean类型，vector类型。在每行中，输入列的值都会被串联成一个指定顺序的vector。

示例：

假设我们具有一个DataFrame，它具有这样的列：id, hour, mobile, userFeatures, and clicked：

 id | hour | mobile | userFeatures     | clicked
:----:|:------:|:--------:|:------------------:|:---------:
 0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0


userFeatures是一个vector列，它包含了三个用户特征。我们希望将hour, mobile, userFeatures组合成一个叫做"features"的feature vector，并使用它来预测clicked or not。如果我们将VectorAssembler的输入列设置成hour, mobile, userFeatures，将输出列设置成features，在该转换后，会得到以下的DataFrame: 

 id | hour | mobile | userFeatures     | clicked | features
:----:|:------:|:--------:|:------------------:|:---------:|:-----------------------------:
 0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0     | [18.0, 1.0, 0.0, 10.0, 0.5]


示例代码：

{% highlight scala %}

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val dataset = spark.createDataFrame(
  Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
).toDF("id", "hour", "mobile", "userFeatures", "clicked")

val assembler = new VectorAssembler()
  .setInputCols(Array("hour", "mobile", "userFeatures"))
  .setOutputCol("features")

val output = assembler.transform(dataset)
println(output.select("features", "clicked").first())

{% endhighlight %}

## 2.20 QuantileDiscretizer

QuantileDiscretizer可以将一列连续型特征作为输入，产生一列二进制型类别型特征（binned categorical features）。二进制的数目通过参数numBuckets进行设定。bin选择的范围，可以使用一个近似算法（详见文档[approxQuantile](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.DataFrameStatFunctions)）。这种逼近的precision可以通过relativeError参数进行控制。如果该参数设置为0，会得到精准的分位数（注意：计算精准分位数是一项开销很昂贵的操作）。bin的上下界可以是(-Infinity, +Infinity)，这样可以覆盖整个实数集。

示例：

假设我们具有一个DataFrame，它的列为：id, hour：

 id | hour
:----:|:------:
 0  | 18.0
----|------
 1  | 19.0
----|------
 2  | 8.0
----|------
 3  | 5.0
----|------
 4  | 2.2

hour是一个连续型feature，Double类型。我们希望将连续型feature转换成一个类别型feature。通过设置 numBuckets =3，我们可以得到以下的DataFrame:

id | hour | result
:----:|:------:|:------:
 0  | 18.0 | 2.0
----|------|------
 1  | 19.0 | 2.0
----|------|------
 2  | 8.0  | 1.0
----|------|------
 3  | 5.0  | 1.0
----|------|------
 4  | 2.2  | 0.0


{% highlight scala %}

import org.apache.spark.ml.feature.QuantileDiscretizer

val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
var df = spark.createDataFrame(data).toDF("id", "hour")

val discretizer = new QuantileDiscretizer()
  .setInputCol("hour")
  .setOutputCol("result")
  .setNumBuckets(3)

val result = discretizer.fit(df).transform(df)
result.show()

{% endhighlight %}

<h2 id="3"></h2>

# 三、特征选择

## 3.1 VectorSlicer

VectorSlicer是个转换器：它的输入是一个feature vector，输出为一个新的feature vector（是原始feature的子集）。它对于从一个vector列中抽取feature来说很管用。

VectorSlicer接受指定indice的向量列，接着输出一个新的向量列，它们的值通过indice来选定。有两种类型的indice：

- 1.Integer类型的indice，表示vector的索引，通过setIndices()设置
- 2.String类型的indice，表示vector的feature名，通过setNames()设置。它需要向量列具备一个AttributeGroup，因为它的实现需要与Attribute的名字相匹配。

可以通过同过integer和string进行指定。也就是说，你可以同时使用integer索引和string名。必须至少选择一个feature。不允许重复的feature，因此在选择indices和names时不能重复。注意，如果features的名字被选中了，如果输入的属性是空的会抛出异常。

输出的vector会对feature进行排序，选中的indices按首位，接下去才是选中的names。

** 示例 ** 

假设我们具有一个DataFrame，它有一列userFeatures：

userFeatures
------------------
 [0.0, 10.0, 0.5]


userFeatures


## 3.2 RFormula

RFormula通过指定一个[R model formula](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/formula.html)选择列。当前，我们支持R操作符的一个限定集，包括：‘~’, ‘.’, ‘:’, ‘+’, 和 ‘-‘。基本操作符是：

- ~: 分隔target和terms
- +: concat terms，"+ 0"就意味着removing intercept
- -: 移除一个term，"- 1"意味着removing intercept
- :: 交互
- .: 所有列除了target

假设a和b都是double的列，我们可以使用下面的简单示例来展示RFormula的效果：

- y ~ a + b => 表示模型 y ~ w0 + w1 * a + w2 * b，w0是intercept，而w1和w2是coefficients
- y ~ a + b + a:b -1 => 表示模型：y ~ w1 * a + w2 * b + w3 * a * b，其中，w1, w2, w3是coefficients

RFormula会生成一个vector的feature列，和一个double或string类型的label列。和在R中使用线性回归的formula一样，string类型的input列将会进行one-hot encoded，而numeric列则会被转成double。如果label列是string类型，它将会首先被StringIndexer转换成double。如果label列在DataFrame中不存在，输出的label列将会从formula中指定的响应变量上被创建。

示例：

假设，DataFrame有相应的列：id, country, hour, 和 clicked

id | country | hour | clicked
:---:|:---------:|:------:|:---------:
 7 | "US"    | 18   | 1.0
 8 | "CA"    | 12   | 0.0
 9 | "NZ"    | 15   | 0.0

如果我们使用RFormula，相应的formula string为：clicked ~ country + hour，这意味着，我们希望基于country和hour去预测clicked，在转换后，我们应该可以得到下面的DataFrame：

id | country | hour | clicked | features         | label
:---:|:---------:|:------:|:---------:|:------------------:|:-------:
 7 | "US"    | 18   | 1.0     | [0.0, 0.0, 18.0] | 1.0
 8 | "CA"    | 12   | 0.0     | [0.0, 1.0, 12.0] | 0.0
 9 | "NZ"    | 15   | 0.0     | [1.0, 0.0, 15.0] | 0.0

相应的代码如下：

{% highlight scala %}

import org.apache.spark.ml.feature.RFormula

val dataset = spark.createDataFrame(Seq(
  (7, "US", 18, 1.0),
  (8, "CA", 12, 0.0),
  (9, "NZ", 15, 0.0)
)).toDF("id", "country", "hour", "clicked")
val formula = new RFormula()
  .setFormula("clicked ~ country + hour")
  .setFeaturesCol("features")
  .setLabelCol("label")
val output = formula.fit(dataset).transform(dataset)
output.select("features", "label").show()

{% endhighlight %}

代码：examples/src/main/scala/org/apache/spark/examples/ml/RFormulaExample.scala

## 3.3 ChiSqSelector

ChiSqSelector表示使用卡方分布的选择。它也可以在类别型feature的数据集上进行操作。ChiSqSelector会基于[卡方检验](https://en.wikipedia.org/wiki/Chi-squared_test)来排序feature，接着会选择（筛选）出最独立的top features。

示例：

我们具有一个DataFrame，相应的列为：id, features, 和 clicked，它用于我们要预测的target：

id | features              | clicked
:---:|:-----------------------:|:---------:
 7 | [0.0, 0.0, 18.0, 1.0] | 1.0
 8 | [0.0, 1.0, 12.0, 0.0] | 0.0
 9 | [1.0, 0.0, 15.0, 0.1] | 0.0


我们使用ChiSqSelector，设置：numTopFeatures = 1, 接着根据我们的对应clicked的label作为最后一列，它会在features上选择最有用的一列feature：

id | features              | clicked | selectedFeatures
:---:|:-----------------------:|:---------:|:------------------:
 7 | [0.0, 0.0, 18.0, 1.0] | 1.0     | [1.0]
 8 | [0.0, 1.0, 12.0, 0.0] | 0.0     | [0.0]
 9 | [1.0, 0.0, 15.0, 0.1] | 0.0     | [0.1]

示例代码：

{% highlight scala %}

import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors

val data = Seq(
  (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
  (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
  (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
)

val df = spark.createDataset(data).toDF("id", "features", "clicked")

val selector = new ChiSqSelector()
  .setNumTopFeatures(1)
  .setFeaturesCol("features")
  .setLabelCol("clicked")
  .setOutputCol("selectedFeatures")

val result = selector.fit(df).transform(df)
result.show()

{% endhighlight %}

详见代码：examples/src/main/scala/org/apache/spark/examples/ml/ChiSqSelectorExample.scala

参考：

1.[http://spark.apache.org/docs/latest/ml-features.html](http://spark.apache.org/docs/latest/ml-features.html)
