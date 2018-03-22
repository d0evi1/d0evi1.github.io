---
layout: page
title:  定制estimators
tagline: 
---
{% include JB/setup %}

# 介绍

该文档介绍了定制的Estimators，演示了如何来创建一个定制的Estimator，它会模仿pre-made Estimator(DNNClassifier)的行为特性来解决iris问题。

为了下载和访问示例，可以调用以下命令：

	git clone https://github.com/tensorflow/models/
	cd models/samples/core/get_started

在本文档中，我们会使用[custom_estimator.py](https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py)。你可以使用以下的命令来运行它：

	python custom_estimator.py

如果你比较急，可以比较custom_estimator.py与[premade_estimator.py](https://github.com/tensorflow/models/blob/master/samples/core/get_started/premade_estimator.py)。

# 1.pre-made vs. custom

如下图所示，pre-made Estimators是基类tf.estimator.Estimator的子类，而定制的estimators是tf.estimator.Estimator的实例：

<img src="https://www.tensorflow.org/images/custom_estimators/estimator_types.png">

pre-made Estimators是已经做好的。但有时候，你需要对一个Estimator的行为做更多控制。这时候就需要定制Estimators了。你可以创建一个定制版的Estimator来做任何事。如果你希望hidden layers以某些不常见的方式进行连接，可以编写一个定制的Estimator。如果你想为你的模型计算一个唯一的metric，可以编写一个定制的Estimator。基本上，如果你想为特定的问题进行优化，你可编写一个定制的Estimator。

模型函数（model_fn）用来实现ML算法。pre-made Estimators和custom Estimators的唯一区别在于：

- 对于pre-made Estimator，有人已经为你编写了model_fn
- 对于custom Estimator，你必须自己编写model_fn

你的模型函数可以实现很多算法，定义许多种hidden layers和metrics。类似于input_fn，所有的model_fn必须接收一个标准group的输入参数，并返回一个标准group的输出值。正如input_fn可以利用Dataset API，model_fn可以利用Layers API和Metrics API。

让我们来看如何使用一个定制版Estimator来解决Iris problem。牢记，这是Iris model的组织架构：

<img src="https://www.tensorflow.org/images/custom_estimators/full_network.png">

我们的iris实现包含了4个特征，两个hidden layers，以及一个logits output layer

# 2.编写一个input_fn

定制版Estimator的实现使用了与pre-make Estimator相同的input_fn：

{% highlight python %}

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

{% endhighlight %}

该input_fn会构建一个input pipeline，它会yields相应的(features, labels) pair的batch，其中features是一个关于特征的字典。

# 2.创建feature_column

你必须定义你的模型的feature columns来指定模型应该使用每个特征。不管是pre-make Estimators还是custom Estimators，两者定义feature_column相同。

下面的代码为每个input feature创建了一个简单的numeric_column，表示input feature的值会被直接作为模型的输入使用：

{% highlight python %}

# Feature columns describe how to use the input.
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

{% endhighlight %}

# 3.编写一个model_fn

模型函数使用如下的签名：

{% highlight python %}

def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration

{% endhighlight %}

前两个参数是features和labels的batch，它们由input_fn返回；也就是说，features和labels是你的模型所要用来处理的数据。model参数表示调用者（caller）是否会进行training，predicting，或evaluation。

caller会将参数params传递给一个Estimator的构造器。任何传给该构造器的params接着会被传给model_fn。在[custom_estimator.py](https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py)中，下面的代码会创建estimator，并设置params来配置模型。该配置step与在之前[DNNClassifier一节](https://www.tensorflow.org/get_started/premade_estimators)中如何配置相似。

{% highlight python %}

classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
        'feature_columns': my_feature_columns,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [10, 10],
        # The model must choose between 3 classes.
        'n_classes': 3,
    })

{% endhighlight %}

为了实现一个常见的模型函数，你需要做以下事：

- 1.定义模型
- 2.为三种不同的mode指定额外的计算：
	- Predict
	- Evaluate
	- Train

## 3.1 定义模型

基本的DNN模型必须定义以下三个部分：

- 一个input layer
- 一或多个hidden layer
- 一个output layer

### 定义input layer

model_fn的第一行会调用tf.feature_column.input_layer来将feature字典和feature_columns转换成模型所要的输入，如下所示：

{% highlight python %}

# Use `input_layer` to apply the feature columns.
net = tf.feature_column.input_layer(features, params['feature_columns'])

{% endhighlight %} 

该行代码会应用你的feature column所定义的转换（transformations），来创建模型的input layer。

<img src="https://www.tensorflow.org/images/custom_estimators/input_layer.png">

### hidden layers

如果你创建一个DNN，你必须定义一或多个hidden layers。 Layers API提供了丰富的函数来定义所有类型的hidden layers，包含conv，pooling，dropout layers。对于Iris，我们可以简单地调用tf.layers.dense来创建hidden layers，它的维度由params['hidden_layers']来定义。在一个dense layer中，每个节点会与前一层的每个节点相连接。这里是相关的代码：

{% highlight python %}

# Build the hidden layers, sized according to the 'hidden_units' param.
for units in params['hidden_units']:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)


{% endhighlight %}

该代码会：

- units参数定义了在一个给定layer中output neurons的数目
- activation参数定义了activation function——该处为ReLu

net变量表示网络的当前top layer。在第一次迭代中，net表示input layer。在每次loop迭代中，tf.layers.dense会创建一个新的layer，它会用变量net来接受前一层layer的输出作为它的输入。

在创建了两个hidden layers后，我们的网络看起来如下所示。为了简化，该图并没有展示每一层units的数目。

<img src="https://www.tensorflow.org/images/custom_estimators/add_hidden_layer.png">

注意，tf.layers.dense提供了许多额外的能力，包括可以设置多个正则参数。为了简化的目的，我们接收其它参数的缺省值。

### output layer

我们通过再次调用tf.layers.dense来定义output layer，这次无需activation function：

{% highlight python %}

# Compute logits (1 per class).
logits = tf.layers.dense(net, params['n_classes'], activation=None)

{% endhighlight %}

这里，net表示最终的hidden layer。因而，layers的完整集如下连接：

<img src="https://www.tensorflow.org/images/custom_estimators/add_logits.png">

最终的hidden layer会feeds给output layer

当定义一个output layer时，units参数指定了outputs的数目。因而，通过设置units给params['n_classese']，模型会为每个class产生一个output值。output vector的每个元素会包含一个分数，或“logit”，它们的计算与Iris的类有关：Setosa, Versicolor, or Virginica,等。

后续，这些logits会通过tf.nn.softmax函数转换成对应的概率。

## 3.2 实现training，evaluation，prediction

创建一个model_fn的最终step是，编写分支代码来实现prediction，evaluation和training。

当调用Estimator的train，evaluate，或者predict方法时，model_fn会被调用。回顾下模型函数的函数签名：

{% highlight python %}

def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys, see below
   params):  # Additional configuration

{% endhighlight %}

注意第三个参数，如下表如下，当调用train，evaluate，或者predict时，Estimator框架会调用对应mode参数下的模型函数：

- train(): ModeKeys.TRAIN
- evaluate(): ModeKeys.EVAL
- predict(): 	ModeKeys.PREDICT

例如，假设你实例化一个custom Estimator来生成一个名为classifier的对象，接着你做出如下的调用：

{% highlight python %}

classifier = tf.estimator.Estimator(...)
classifier.train(input_fn=lambda: my_input_fn(FILE_TRAIN, True, 500))

{% endhighlight %}

Estimator框架会调用你的模型函数，其中mode设置为ModeKeys.TRAIN。

你的模型函数必须提供代码来处理所有这三种mode值。对于每个mode值，你的代码必须返回一个tf.estimator.EstimatorSpec的实例，它会包含caller所需的信息。让我们检查下每个mode。

### Predict

当Estimator的predict方法被调用时，model_fn会接收mode=ModeKeys.PREDICT。在这种情况下，model_fn必须返回一个包含prediction的tf.estimator.EstimatorSpec。

模型必须在做出预测之前被训练。被训练后的模型存储在model_dir目录下，该目录可以在你实例化Estimator时确定。

生成预测的代码如下：

{% highlight python %}

# Compute predictions.
predicted_classes = tf.argmax(logits, 1)
if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

{% endhighlight %}

预测字典包含了当运行在prediction mode下模型返回的所有东西。

<img src="https://www.tensorflow.org/images/custom_estimators/add_predictions.png">

predictions包含了以下三个key/value pairs：

- class_ids 持有类id(0, 1, 或2)，表示该样本最可能的模型预测.
- probabilities会包含三个概率（在本例中：0.02, 0.95, 0.03）
- logits 包含了原始的logit值（在本例中，-1.3, 2.6, -0.9）

通过tf.estimator.EstimatorSpec的predictions参数，我们回返回该字典给caller。Estimator的predict方法将yield这些字典。

#### 计算loss

对于training和evaluation，我们需要计算模型的loss。这是我们要优化的objective函数。

我们可以调用tf.losses.sparse_softmax_cross_entropy来计算loss。该函数返回的值会是最小的，逼近0, 正例的概率（index label）接近1.0。返回的loss值会随着正例概率的降低变得越来越大。

该函数会返回在整个batch上的平均。

{% highlight python %}

# Compute loss.
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

{% endhighlight %}

### Evaluate

当Estimator的evaluate方法被调用时，model_fn会接收mode=ModeKeys.EVAL。在这种情况下，模型函数必须返回一个包含模型loss以及可选的一或多个metrics的tf.estimator.EstimatorSpec。

尽管返回metrics是可选中，大多数定制的Estimators会至少返回一个metric。Tensorflow提供了一个Metrics模块tf.metrics来计算公共metrics。为了简洁，我们只会返回accuracy。tf.metrics.accuracy函数会比较predictions与true values（也就是input_fn提供的labels）。tf.metrics.accuracy函数需要labels和predictions具有相同的shape。以下是tf.metrics.accuracy的调用代码：

{% highlight python %}

# Compute evaluation metrics.
accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')
{% endhighlight %}

为evaluation返回的EstimatorSpec通常包含以下的信息：

- loss：模型loss
- eval_metric_ops：是一个可选的metrics字典

因而，我们会创建一个包含sole metric的字典。如果我们已经计算了其它metrics，我们将它们作为额外的key/value pairs添加到相同的字典中。接着，我们会给tf.estimator.EstimatorSpec的eval_metric_ops参数传递该字典。代码如下：

{% highlight python %}
metrics = {'accuracy': accuracy}
tf.summary.scalar('accuracy', accuracy[1])

if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=metrics)

{% endhighlight %}

在TRAIN和EVAL模式下，tf.summary.scalar会将accuracy提供给TensorBaord。

### Train

当Estimator的train方法被调用时，model_fn会使用mode=ModeKeys.TRAIN进行调用。在这种情况下，model_fn必须返回一个包含loss和一个训练操作的EstimatorSpec。

构建训练操作将需要一个optimizer。我们会使用tf.train.AdagradOptimizer, 因为我们会模仿DNNClassifier，它缺省也使用Adagrad。tf.train包提供了许多种optimizer，你可以自由尝试。

以下代码构建了optimizer:

{% highlight python %}
optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)

{% endhighlight %}

接下来，我们在loss上使用optimizer的minimize方法来构建训练操作。

minimize方法会接收一个global_step参数。TensorFlow使用该参数来统计已经处理的training steps的数目（以便知道何时结束一个training run）。再者，global_step对于TensorBoard graphs正确工作是必须的。可以简单调用tf.train.get_global_step，并将结果传给minimize的global_step参数。

以下代码用于训练模型：

{% highlight python %}

train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

{% endhighlight %}

对于training返回的EstimatorSpec，必须具有以下的字段集：

- loss：它包含了loss function的值
- train_op: 它会执行一个training step

这里的代码会调用EstimatorSpec: 

{% highlight python %}
return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

{% endhighlight %}

至此，模型函数完成。

# 定制Estimator

通过Estimator基类来实例化定制的Estimator：

{% highlight python %}

# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
        'feature_columns': my_feature_columns,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [10, 10],
        # The model must choose between 3 classes.
        'n_classes': 3,
    })

{% endhighlight %}

这里的params字典的作用与DNNClassifier中的key-word参数相同；params字典让你无需修改model_fn的代码即可以配置你的Estimator。

代码的其余部分与pre-made estimator相似。例如：训练一个模型时：

{% highlight python %}

# Train the Model.
classifier.train(
    input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
    steps=args.train_steps)
{% endhighlight %}

# TensorBoard

你可以在tensorboard中查看custom Estimator的训练结果。为了看到上报，启动TensorBoard：

	# Replace PATH with the actual path passed as model_dir
	tensorboard --logdir=PATH

接着，通过浏览http://localhost:6006开启tensorboard

所有pre-make Estimators会自动记录一些信息到tensorboard中。对于custom Estimators，tensorboard只提供了一个缺省日志（一个关于loss的graph），加上你显式告诉tensorboard要记录的信息。对于custom Estimator，tensorboard会生成如下：

<img src="https://www.tensorflow.org/images/custom_estimators/accuracy.png">

<img src="https://www.tensorflow.org/images/custom_estimators/loss.png">

<img src="https://www.tensorflow.org/images/custom_estimators/steps_per_second.png">

tensorboard会显示三个图

这三个图说明如下：

- global_step/sec：一个性能指示器，展示了随着模型训练，每秒处理多少batches（gradient updates）
- loss：上报的loss
- accuracy: 通过以下两行所记录的accuracy
	- eval_metric_ops={'my_accuracy': accuracy})，evaluation期
	- tf.summary.scalar('accuracy', accuracy[1])，training期

对于传输一个global_step到你的optimizer的minimize方法中很重要，这些tensorboard图是主要原因之一。如果没有它，该模型不能为这些图记录x坐标。

在my_accuracy和loss图中，要注意以下事项：

- 橙色线表示training
- 蓝色点表示evaluation

在训练期，summaries（橙色线）会随着batches的处理被周期性记录，这就是为什么它会变成一个x轴。

相反的，对于evaluate的每次调要，evaluation过程只会在图中有一个点。该点包含了整个evaluation调用的平均。在该图中没有width，因为它会被在特定training step（单个checkpoint）的某个模型态下整个进行评估。

如下图所建议的，你可以使用左侧的控制面板来选择性地disable/enable 上报。

<img src="https://www.tensorflow.org/images/custom_estimators/select_run.jpg">

# 总结

尽管pre-make Estimator是一种创建新模型的有效方式，你常需要定制Estimators所提供的额外的灵活性。幸运的是，pre-make和custom Estimators遵循相同的编程模式。唯一的区别是，对于custom Estimator你需要一个model_fn.

对于更多信息，可以参见：

- [官方MNIST tensorflow实现](https://github.com/tensorflow/models/tree/master/official/mnist)，它会实现一个custom Estimator
- [官方模型repository](https://github.com/tensorflow/models/tree/master/official)，包含了许多使用custom Estimator的示例
- [TensorBoard视频](https://www.youtube.com/watch?v=eBbEDRsCmv4&feature=youtu.be)
- [底层API](https://www.tensorflow.org/programmers_guide/low_level_intro)
 
，# 参考

[tensorflow Custom Estimators](https://www.tensorflow.org/get_started/custom_estimators)
