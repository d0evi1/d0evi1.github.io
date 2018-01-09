---
layout: page
title:  tensorflow中的FeatureColumn
tagline: 
---
{% include JB/setup %}

# 介绍

FeatureColumn表示数据中的单个特征。FeatureColumn可以表示像“高度（height）”这样的**连续型值量**，或者它可以表示像“眼珠颜色（eye_color）”这样的**类别型变量**：可能的值为{'blue’, ‘brown’, ‘green’}。

在“height”这样的连续型特征和“eye_color”这种类别型特征的情况下，数据中的单个值在输入模型之前可能会被转换成一个**数字序列**。FeatureColumn这个抽象类可以将特征作为单个语义单元来操作。您可以指定转换并选择要包括的特征，而无需进行处理就可以feed给模型的tensor中的指定索引。（用名称索引，而非数字进行索引）

# 1.sparse列

FeatureColumn会自动处理将类别型的值自动转换成向量（vectors）：

{% highlight python %}
eye_color = tf.feature_column.categorical_column_with_vocabulary_list(
    "eye_color", vocabulary_list=["blue", "brown", "green"])

{% endhighlight %}

对于**不知道所有可能值的类别型特征**，也可以生成FeatureColumns。这种情况你需要使用：categorical_column_with_hash_bucket()，**它会使用hash函数来分配索引给特征值**：

{% highlight python %}

education = tf.feature_column.categorical_column_with_hash_bucket(
    "education", hash_bucket_size=1000)
    
{% endhighlight %}

# 2.特征交叉列

**因为线性模型会为不同的features分配独立的weights，它们不能学到特征值的特定组合的相对重要性**。如果你有一个特征为：“favorite_sport(喜欢的运动)”，另一个特征为：“home_city（家乡城市）”, 当你尝试去预测一个人是否喜欢穿红色衣服，你的线性模型并不会学到这样的特性（两个特征都是类别型）: 来自St.Louis的爱好棒球运动的粉丝是很喜欢穿红色衣服（圣路易斯红雀队）。

你可以通过创建一个新的'favorite_sport_x_home_city'特征。特征值为两个源特征的组合，例如： 'baseball_x_stlouis'。这种特征组合称为“特征交叉（feature cross）”。

crossed_column()函数很容易构建出交叉特征：

{% highlight python %}

sport_x_city = tf.feature_column.crossed_column(
    ["sport", "city"], hash_bucket_size=int(1e4))

{% endhighlight %}

# 3.连续型列

你可以指定一个连续型特征：

{% highlight python %}

age = tf.feature_column.numeric_column("age")

{% endhighlight %}

尽管对于一个实数，一个连续型特征一般是能够直接输入到模型中的。tf.learn也提供了对连续型列进行转换(即：Bucketization)。

# 4.Bucketization列

分桶化（Bucketization）会将一个连续型column转换为一个类别型column。这种转换可以从特征交叉中的连续型特征学到：**对于特定的范围(range)，具有特定的重要性**。

Bucketization将可能值的范围分割成几个子区间（subranges），称为桶（buckets）：

{% highlight python %}

age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

{% endhighlight %}

落到bucket中的所对应的值，会变为类别型label。

## 4.1 输入函数input_fn

FeatureColumn为你的模型的输入数据提供了一个关于如何表示和转换数据的规范。但它们本身不提供数据。需要你通过一个输入函数来提供数据。

输入函数（input function）必须返回一个关于tensors的字典（dictionary）。每个key对应于FeatureColumn的名字。每个key的value是一个tensor：它包含着所有数据实例的特征的值。input function详见[Building Input Functions with tf.estimator ](https://www.tensorflow.org/get_started/input_fn)，线性模型中的input_fn详见代码[linear models tutorial code](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/learn/wide_n_deep_tutorial.py).

输入函数被传给：train() 和 evaluate()，它们会调用training和testing。

# 4.2 线性estimators

Tensorflow estimator类提供了一个统一的training 和 evaluation组件来用于回归和分类模型。它们会处理training和evaluation loops的细节，让用户更关注模型输入和架构本身。

为了构建一个线性estimator，你可以使用tf.estimator.LinearClassifier 和 tf.estimator.LinearRegressor 来处理分类和回归。

对于所有的tensorflow estimators，为了运行，你可以：

- 1.实例化estimator类。对于两个linear estimator类，你将一个FeatureColumn's列表传给构造函数
- 2.调用estimator的train()方法来训练
- 3.调用estimator的evaluate()来看效果

例如：

{% highlight python %}

e = tf.estimator.LinearClassifier(
    feature_columns=[
        native_country, education, occupation, workclass, marital_status,
        race, age_buckets, education_x_occupation,
        age_buckets_x_race_x_occupation],
    model_dir=YOUR_MODEL_DIRECTORY)
e.train(input_fn=input_fn_train, steps=200)
# Evaluate for one step (one pass through the test data).
results = e.evaluate(input_fn=input_fn_test)

# Print the stats for the evaluation.
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

{% endhighlight %}


# Wide & deep learning

tf.estimator API提供了一个estimator类，来让你做关于一个linear model和deep NN model的jointly training。这种方法会结合线性模型的能力来"记住（memorize）"关键特征，并具有神经网络的generalization能力。使用tf.estimator.DNNLinearCombinedClassifier来创建wide&deep模型：

{% highlight python %}

e = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=YOUR_MODEL_DIR,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])

{% endhighlight %}

# 总结

feature columns提供了一种将数据映射到模型的机制。

- **bucketized_column**: 会为离散型dense input创建一个_BucketizedColumn。
- **crossed_column**：创建一个_CrossedColumn，用于执行特征交叉。
- **embedding_column**：创建一个_EmbeddingColumn，将稀疏数据feeding进一个DNN
- **hashed_embedding_column**: 会使用hashing为一个sparse feature创建_HashedEmbeddingColumn。值v的第i个embedding component，可以通过一个embedding weight（它的index为pair<v,i>的一个fingerprint）进行检索。
- **one_hot_column**: 创建在DNN中使用的一个_OneHotColumn。
- **real_valued_column**：为dense numeric数据创建一个_RealValuedColumn。
- **shared_embedding_columns**：创建一个_EmbeddingColumn列表，它们共享相同的embedding。


# 参考：

- 1.[tensorflow feature_columns](https://www.tensorflow.org/versions/r0.12/api_docs/python/contrib.layers/feature_columns)
- 2.[linear模型中提到的feature column](https://www.tensorflow.org/tutorials/linear)
