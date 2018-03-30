---
layout: page
title:  分布式Estimator
tagline: 
---
{% include JB/setup %}

# 

在paper 1一文中提到（老版本）：

Estimators可以在单机上进行训练、评估、以及模型导出。对于生产环境，以及对大规模训练数据的建模，需要使用分布式执行Estimators，它可以充分利用tensorflow对分布式训练的支持。分布式执行的核心是Experiment类，它将带有两个input_fn（训练和评估）的Estimator进行group。架构图2所示。

<img src="http://pic.yupoo.com/wangdren23/He7cTZTz/medish.jpg">

图2: Experiment接口总览

在每个tensorflow集群上，有许多个在数服务器和许多个worker tasks。大多数workers被训练进程所处理，它会调用Estimator的train()方法。workers的其中之一是指定leader，负责管理checkpoints和其它维护工作。目前，在Tensorflow Estimators中replica training的主要模式是between-graph replication和asynchronous training。然而，它可以被轻易扩展成支持其它的replicated training settings。有了这种架构，梯度下降训练可以以并行方式执行。

通过在固定数目的parameter servers上运行不同数目的workers，我们已经评估过tensorflow estimators的可扩展性（scaling）。我们在一个大的内部推荐数据集上（有1000亿样本）训练了一个DNN模型，花了48个小时，并给出了每秒处理training steps的平均数目。图3展示了：每秒处理的global steps与workers数目达到线性可扩展性。

<img src="http://pic.yupoo.com/wangdren23/Hee8k5Fa/medish.jpg">

有一个特殊的worker会为Experiment处理评估过程，来评估效果以及进行模型导出。它会在一个连续的loop中运行，并调用Estimator中带有input_fn的evaluate()方法。为了避免竞态（race conditions）以及不一致的模型参数状态，评估过程总是会加载最新的checkpoint，并基于该checkpoint的模型参数来计算评估metrics。作为一个简单的扩展，Experiment也支持带有input_fn的evaluation，在实际深度学习中，对于检测overfitting很有用。

更进一步，我们也提供了工具类RunConfig和runner，可以让在集群上使用和配置Experiment进行分布式训练的方式更容易些。RunConfig持有着所有Experiment/Estimator所需的执行相关配置，包括：ClusterSpec，模型输出目录，checkpoints配置等。特别的，RunConfig指定了当前task的任务类型，它允许所有tasks共享相同的binary，但以一个不同的模式运行，比如：parameter server，training 或者continual evaluation。runner是一个简单的工具类，用来构建Runconfig，比如：解析环境变量，使用RunConfig执行Experiment/Estimator。有了该设计，Experiment/Estimator可以很容易地通过多种执行框架（包括：end-to-end机器学习pipelines，以及超参数tuning）进行共享。

# 说明

在源码中[experiment](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)提到，Experiment类已经切换到了tf.estimator.train_and_evaluate。


# 3.tf.estimator.train_and_evaluate

该方法用于estimator进行训练和评测。

该工具函数可以通过使用给定的estimator进行训练、测估（可选）、并导出模型。所有训练相关的设置由train_spec所持有，包括训练的input_fn以及训练用的max_steps，等。所有评估和导出相关的设置由eval_spec持有，包括评估用的input_fn，steps等。

该工具函数提供了对于本地（非分布式）和分布式配置的一致的行为。当前只支持的分布式训练配置是between-graph replication。

- Overfitting：为了避免overfitting，推荐设置训练的input_fn来shuffle训练数据。同时也推荐：在执行评估前，以一个更长（多个epochs）的方式来训练模型，input pipeline会为每个训练重新开始启动。这对于本地训练和评估特别重要。
- 停止条件（stop condition）：为了可靠地同时支持分布式和非分布式配置，对于模型训练，只支持的停止条件是traing_spec.max_steps。如果traing_spec.max_steps是None，模型会永久训练。如果停止条件不同，使用时需小心。例如，假设模型希望只训练一个epoch的数据，配置的input_fn在运行完一个epoch后会抛出OutOfRangeError，这会停止Estimator.train。对于一个三训练worker(three-training-worker)的分布式配置，每个training worker可以独立运行整个epoch。因而，模型会使用三个epoch训练数据来替代一个epoch。

本地（非分布式）训练的示例：

{% highlight python %}

# Set up feature columns.
categorial_feature_a = categorial_column_with_hash_bucket(...)
categorial_feature_a_emb = embedding_column(
    categorical_column=categorial_feature_a, ...)
...  # other feature columns

estimator = DNNClassifier(
    feature_columns=[categorial_feature_a_emb, ...],
    hidden_units=[1024, 512, 256])

# Or set up the model directory
#   estimator = DNNClassifier(
#       config=tf.estimator.RunConfig(
#           model_dir='/my_model', save_summary_steps=100),
#       feature_columns=[categorial_feature_a_emb, ...],
#       hidden_units=[1024, 512, 256])

# Input pipeline for train and evaluate.
def train_input_fn: # returns x, y
  # please shuffle the data.
  pass
def eval_input_fn_eval: # returns x, y
  pass

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

{% endhighlight %}

上述代码可以更改，即可用于分布式训练。（请确保所有worker的RunConfig.model_dir都设置成相同的目录，比如：一个所有worker都能读和写的共享文件系统）。只需要一个额外的工作，为每个对应的worker将合理地设置环境变量TF_CONFIG。

详见：https://www.tensorflow.org/deploy/distributed

设置环境变量依赖于平台。例如，在Linux上，它可以按以下方式：

	$ TF_CONFIG='<replace_with_real_content>' python train_model.py

对于在TF_CONFIG中的内容，确保训练集群的ClusterSpec像起来像：

	cluster = {"chief": ["host0:2222"],
	           "worker": ["host1:2222", "host2:2222", "host3:2222"],
	           "ps": ["host4:2222", "host5:2222"]}

对于主worker，TF_CONFIG的示例如下：

{% highlight python %}

# This should be a JSON string, which is set as environment variable. Usually
# the cluster manager handles that.
TF_CONFIG='{
    "cluster": {
        "chief": ["host0:2222"],
        "worker": ["host1:2222", "host2:2222", "host3:2222"],
        "ps": ["host4:2222", "host5:2222"]
    },
    "task": {"type": "chief", "index": 0}
}'

{% endhighlight %}

注意，主worker也可以做模型训练的工作，与其它**从worker（non-chief workers）**相似。除了模型训练外，它还管理一些额外工作，比如，checkpoint的保存和恢复，写summaries，等。

对于从worker的TF_CONFIG示例，如下：

{% highlight python %}

# This should be a JSON string, which is set as environment variable. Usually
# the cluster manager handles that.
TF_CONFIG='{
    "cluster": {
        "chief": ["host0:2222"],
        "worker": ["host1:2222", "host2:2222", "host3:2222"],
        "ps": ["host4:2222", "host5:2222"]
    },
    "task": {"type": "worker", "index": 0}
}'

{% endhighlight %}

其中task.index被设置成0, 1, 2. 在本例中，各自对应于各个从workers。

{% highlight python %}

# This should be a JSON string, which is set as environment variable. Usually
# the cluster manager handles that.
TF_CONFIG='{
    "cluster": {
        "chief": ["host0:2222"],
        "worker": ["host1:2222", "host2:2222", "host3:2222"],
        "ps": ["host4:2222", "host5:2222"]
    },
    "task": {"type": "ps", "index": 0}
}'

{% endhighlight %}

其中task.index应设置成0和1, 在本例中，各自对应parameter servers。

对于evaluator task的示例。Evaluator是一个特殊的任务，它不是训练集群的一部分。只有一个。用于模型评估。

{% highlight python %}

# This should be a JSON string, which is set as environment variable. Usually
# the cluster manager handles that.
TF_CONFIG='{
    "cluster": {
        "chief": ["host0:2222"],
        "worker": ["host1:2222", "host2:2222", "host3:2222"],
        "ps": ["host4:2222", "host5:2222"]
    },
    "task": {"type": "evaluator", "index": 0}
}'

{% endhighlight %}

参数：

- estimator: 一个用于训练和评估的Estimator实例。
- train_spec: 一个TrainSpec实例，用来指定训练的设置（specification）
- eval_spec：一个EvalSpec实例，用来指定evaluation和export的设置

抛出：

- ValueError： 如果环境变量TF_CONFIG没有被正确设置.


# 参考

1.[TensorFlow Estimators: Managing Simplicity vs. Flexibility in High-Level Machine Learning Frameworks](https://arxiv.org/pdf/1708.02637.pdf)

2.[tf.estimator.train_and_evaluate](https://cloud.google.com/blog/big-data/2018/02/easy-distributed-training-with-tensorflow-using-tfestimatortrain-and-evaluate-on-cloud-ml-engine)
