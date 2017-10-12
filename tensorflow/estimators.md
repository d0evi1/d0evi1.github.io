---
layout: page
title:  tensorflow中的estimators
tagline: 
---
{% include JB/setup %}

# 使用tf.estimator中创建Estimators

tf.estimator框架可以很方便地通过高级Estimator API来构建和训练机器学习模型。Estimator提供了一些类，你可以快速配置实例化来创建regressors和classifiers：

- tf.estimator.LinearClassifier: 线性分类模型
- tf.estimator.LinearRegressor: 线性回归模型
- tf.estimator.DNNClassifier: 创建一个神经网络分类模型
- tf.estimator.DNNRegressor: 创建一个神经网络回归模型
- tf.estimator.DNNLinearCombinedClassifier: 创建一个deep&wide分类模型
- tf.estimator.DNNLinearCombinedRegressor: 创建一个deep&wide回归模型

如果tf.estimator预定义的模型不能满足你的需求时，该怎么办？可能你需要更多关于模型的细粒度控制，比如：定制自己的loss function进行优化，或者为每个layer指定不同的activation function。或者你要实现一个排序系统或推荐系统，上述的classifier和regressor不适合这样的预测。

本文描述了如何使用tf.estimator来**创建你自己的Estimator**，基于生理特征来预测鲍鱼（abalones）的年龄:

- 实例化一个Estimator
- 构建一个定制的model function
- 使用tf.feature_column和tf.layers来配置一个神经网络
- 从tf.losses中选择一个合适的loss function
- 为你的模型定义一个training op
- 生成和返回predictions

# 1.准备

首先，你需要了解tf.estimator API基础，比如：feature columns, input functions, 以及 train()/evaluate()/predict()操作。如果你从未了解过，你可以参考：

- [tf.estimator Quickstart](https://www.tensorflow.org/get_started/estimator)
- [TensorFlow Linear Model Tutorial](https://www.tensorflow.org/tutorials/wide)
- [Building Input Functions with tf.estimator](https://www.tensorflow.org/get_started/input_fn)

# 2.鲍鱼年龄预测

**通常估计鲍鱼的年龄通过贝壳的环数来确定**。然而，由于该任务需要: 切割、染色、通过显微器观看贝壳。因而去发现预测年龄的其它方法还是很有必要的。

鲍鱼的生理特征：

- Length: 鲍鱼的长度(最长方向上；单位：mm)
- Diameter：鲍鱼的直径（长度的垂直测量，单位：mm）
- Height：鲍鱼的高度（贝壳内的肉；单位：mm）
- Whole Weight：整个鲍鱼的重量（单位：克 grams）
- Shucked Weight：鲍鱼肉的重量（单位：克）
- Viscera Weight：出血后的重量
- Shell Weight：贝壳重量

<img src="https://www.tensorflow.org/images/abalone_shell.jpg">

# 3.setup

数据集：

- [abalone_train.csv](http://download.tensorflow.org/data/abalone_train.csv)：3,320个样本
- [abalone_test.csv](http://download.tensorflow.org/data/abalone_test.csv) : 850个样本
- [abalone_predict](http://download.tensorflow.org/data/abalone_predict.csv)：7个预测

## 3.1 加载CSV数据集

{% highlight python %}

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

# Import urllib
from six.moves import urllib

import numpy as np
import tensorflow as tf

FLAGS = None

# 开启loggging.
tf.logging.set_verbosity(tf.logging.INFO)


# 定义下载数据集.
def maybe_download(train_data, test_data, predict_data):
  """Maybe downloads training data and returns train and test file names."""
  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/abalone_train.csv",
        train_file.name)
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/abalone_test.csv", test_file.name)
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s" % test_file_name)

  if predict_data:
    predict_file_name = predict_data
  else:
    predict_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/abalone_predict.csv",
        predict_file.name)
    predict_file_name = predict_file.name
    predict_file.close()
    print("Prediction data is downloaded to %s" % predict_file_name)

  return train_file_name, test_file_name, predict_file_name
  
  
# 创建main()函数，加载train/test/predict数据集.

def main(unused_argv):
  # Load datasets
  abalone_train, abalone_test, abalone_predict = maybe_download(
    FLAGS.train_data, FLAGS.test_data, FLAGS.predict_data)

  # Training examples
  training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_train, target_dtype=np.int, features_dtype=np.float64)

  # Test examples
  test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_test, target_dtype=np.int, features_dtype=np.float64)

  # Set of 7 examples for which to predict abalone ages
  prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_predict, target_dtype=np.int, features_dtype=np.float64)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--train_data", type=str, default="", help="Path to the training data.")
  parser.add_argument(
      "--test_data", type=str, default="", help="Path to the test data.")
  parser.add_argument(
      "--predict_data",
      type=str,
      default="",
      help="Path to the prediction data.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

{% endhighlight %}

## 3.2 实例化一个Estimator

当使用 tf.estimator提供的类（比如：DNNClassifier）来定义一个模型时，在构造函数中应用所有的配置参数：

{% highlight python %}
my_nn = tf.estimator.DNNClassifier(feature_columns=[age, height, weight],
                 hidden_units=[10, 10, 10],
                 activation_fn=tf.nn.relu,
                 dropout=0.2,
                 n_classes=3,
                 optimizer="Adam")
{% endhighlight %}

你不需要编写额外的代码来指示tensorflow来如何训练模型，计算loss以及返回预测；该logic已经被整合到DNNClassifier中了。

当你要从头创建自己的estimator时，构造函数会接受两个高级参数进行模型配置，model_fn和params。

{% highlight python %}

nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

{% endhighlight %}

- model_fn: 一个函数对象，它包含了所有前面提到的逻辑：支持training, evaluation, prediction。你只负责实现功能。下面会讲述所何构建model_fn。
- params：一个可选的超参数字典（例如：learning_rate，dropout），它们会被传给model_fn。

**注意：正如tf.estimator预定义的regressors和classifiers，Estimator的initializer也接受通用的配置参数：model_dir 和 config。**

对于鲍鱼年龄预测器，该模型会接受一个超参数：learning rate。可以在你的代码初始位置将LEARNING_RATE定义为一个常数。

{% highlight python %}

tf.logging.set_verbosity(tf.logging.INFO)

# Learning rate for the model
LEARNING_RATE = 0.001

{% endhighlight %}

**注意：这里LEARNING_RATE设为0.001, 你可以随需调整该值来在模型训练期间达到最好的结果。**

接着，添加下面的代码到main()中，创建model_params字典，它包含了learning rate，用它来实例化Estimator:

{% highlight python %}

# Set model params
model_params = {"learning_rate": LEARNING_RATE}

# Instantiate Estimator
nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

{% endhighlight %}

## 3.3 构建model_fn

Estimator API模型函数的基本框架为：

{% highlight python %}

def model_fn(features, labels, mode, params):
   # Logic to do the following:
   # 1. Configure the model via TensorFlow operations
   # 2. Define the loss function for training/evaluation
   # 3. Define the training operation/optimizer
   # 4. Generate predictions
   # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
   return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)

{% endhighlight %}


model_fn接受3个参数：

- **features**：一个字典（dict），它包含的特征会通过input_fn传给模型
- **labels**：一个Tensor，它包含的labels会通过input_fn传给模型。对于predict()的调用该labels会为空，该模型会infer出这些值。
- **mode**: tf.estimator.ModeKeys的其中一种字符串值，可以表示model_fn会被哪种上下文所调用：
	- tf.estimator.ModeKeys.TRAIN： model_fn将在training模式下通过一个train()被调用
	- tf.estimator.ModeKeys.EVAL： model_fn将在evaluation模式下通过一个evaluate()被调用
	- tf.estimator.ModeKeys.PREDICT：model_fn将在predict模式下通过一个predict()被调用

**model_fn也接受一个params参数**，它包含了一个用于训练的超参数字典（所上所述）

该函数体会执行下面的任务：

- 配置模型：对于这里的示例任务，就是一个NN.
- 定义loss function：来计算模型的预测与target值有多接近
- 定义training操作：指定了optimizer算法来通过loss function最小化loss。

**model_fn必须返回一个tf.estimator.EstimatorSpec对象**，它包含了以下的值：

- **mode(必须)**。该模型以何种模式运行。通常，这里你将返回model_fn的mode参数
- **predictions（在PREDICT模式下必须）**。它是一个字典，包含了所选Tensor（它包含了模型预测）的key names，例如：predictions = {"results": tensor_of_predictions}。在PREDICT模式下，你在EstimatorSpec中返回的该字典，会接着被predict()返回。因此，你可以你喜欢的格式来构建它
- **loss（在EVAL 和 TRAIN模式下必须）**。一个Tensor包含了一个标量的loss值：模型的loss function的输出（将在下面详细说明）。在TRAIN模式下用于error handling和logging，在EVAL模式下自动当成是一个metric。
- **train_op（在TRAIN模式下必须）**。一个Op，它可以运行训练的一个step。
- **eval_metric_ops（可选参数）**。一个name/value pairs的字典，它指定了在EVAL模式下模型运行的metrics。其中，name是一个你要选择的metrics的标签，值就是你的metric计算的结果。tf.metrics 模块提供了一些常用metrics的预定义函数。下面的eval_metric_ops包含了一个“accuracy” metric，使用tf.metrics.accuracy进行计算：eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels, predictions) }。如果你不指定eval_metric_ops，loss只会在evaluation期间被计算。

## 3.4 使用tf.feature_column和tf.layers来配置一个NN

构建一个神经网络必须**创建和连接input layer/hidden layers/output layer**。

input layer是一系列节点（模型中每个特征一个节点），它们接受被传给model_fn的features参数所指定的feature数据。如果features包含了一个n维的Tensor，包含所有的特征数据，那么你可以将它看成是input layer。如果features包含了一个通过一个input function传给模型的feature columns字典，你可以使用tf.feature_column.input_layer函数来将它转化成一个input-layer Tensor：

{% highlight python %}

input_layer = tf.feature_column.input_layer(
    features=features, feature_columns=[age, height, weight])

{% endhighlight %}

如上所述，input_layer()会采用两个必须的参数：

- **features**：从string keys到Tensors的映射，包含了相对应的feature数据。它与model_fn中的features参数相同.
- **feature_columns**：在模型中的所有FeatureColumns的列表—— age, height, weight。

**NN的input layer，接着必须通过一个activation function（它会对前一层执行一个非线性变换）被连接到一或多个hidden layers**。最后一层的hidden layer接着连接到output layer上。tf.layers提供了tf.layers.dense函数来构建fully connected layers。该activation通过activation参数进行控制。activation参数可选的一些选项为：

- **tf.nn.relu**：下面的代码创建了一层units节点，它会使用一个ReLU激活函数，与前一层的input_layer完全连接：

{% highlight python %}

hidden_layer = tf.layers.dense( inputs=input_layer, units=10, activation=tf.nn.relu)

{% endhighlight %}

- **tf.nn.relu6**: 下面的代码创建一层units节点，它们使用一个ReLU 6的激活函数与前一层的hidden_layer完全连接

{% highlight python %}

second_hidden_layer = tf.layers.dense( inputs=hidden_layer, units=20, activation=tf.nn.relu)

{% endhighlight %}

- **None**：以下的代码创建了一个units节点的层，它与前一层的second_hidden_layer完全连接，不使用activation函数，只是一个线性变换：

{% highlight python %}

output_layer = tf.layers.dense( inputs=second_hidden_layer, units=3, activation=None)

{% endhighlight %}

**其它的activation函数(比如sigmoid)**也是可能的，例如：

{% highlight python %}

output_layer = tf.layers.dense(inputs=second_hidden_layer,
                               units=10,
                               activation_fn=tf.sigmoid)

{% endhighlight %}

上述代码创建了神经网络层：output_layer, 它与second_hidden_layer通过一个sigmoid激活函数（tf.sigmoid）完全连接，**对于预定义的activation，可参见[API docs](https://www.tensorflow.org/api_guides/python/nn#activation_functions)**

完整的预测器代码如下：

{% highlight python %}

def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features["x"]) with relu activation
  first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.layers.dense(
      first_hidden_layer, 10, activation=tf.nn.relu)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.layers.dense(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"ages": predictions}
  ...

{% endhighlight %}

这里，因为你使用numpy_input_fn来创建abalone Datasets，features是一个关于{"x": data_tensor}的字典，因此，features["x"]是input layer。该网络包含了两个hidden layers，每个layer具有10个节点以及一个ReLU激活函数。output layer没有activation函数，可以通过tf.reshape 到一个1维的tensor，以便捕获模型的predictions，它被存到predictions_dict中。

## 3.5 为模型定义loss

通过model_fn返回的EstimatorSpec必须包含loss：一个Tensor代表loss值，它可以量化模型的predict value在训练和评测期间与label value间的优化。tf.losses模块提供了很方便的函数来计算loss：

- **absolute_difference(labels, predictions)**: 使用 absolute-difference formula（L1 loss）来计算loss
- **log_loss(labels, predictions)**：使用 logistic loss forumula来计算loss（通常在LR中）
- **mean_squared_error(labels, predictions)**：使用MSE（即L2 loss）来计算loss

下面的示例，为model_fn添加了一个loss的定义，它使用mean_squared_error():

{% highlight python %}

def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features["x"]) with relu activation
  first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.layers.dense(
      first_hidden_layer, 10, activation=tf.nn.relu)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.layers.dense(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"ages": predictions}

  # Calculate loss using mean squared error
  loss = tf.losses.mean_squared_error(labels, predictions)
  ...

{% endhighlight %}

**evaluation的metrics可以被添加到一个eval_metric_ops字典中**。下面的代码定义了一个rmse metrics，它会为模型预测计算root mean squared error。

## 3.6 为模型定义training op

**training op定义了当对训练数据进行fit模型的所用的优化算法**。通常当训练时，目标是最小化loss。一种简单的方法是，创建training op来实例化一个tf.train.Optimizer子类，并调用minimize方法。

下面的代码为model_fn定义了一个training op，使用上述计算的loss value，在params中传给的learning rate，gradient descent optimizer，对于global_step，函数 tf.train.get_global_step会小心的生成一个整型变量：

{% highlight python %}

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=params["learning_rate"])
train_op = optimizer.minimize(
    loss=loss, global_step=tf.train.get_global_step())

{% endhighlight %}

对于optimizers的列表，详见API guide。

## 3.7 完整的model_fn

这里给出了最终完整的model_fn。下面的代码配置了NN；**定义了loss和training op；并返回一个包含mode，predictions_dict, loss, 和 train_op的EstimatorSpec对象**。

{% highlight python %}

def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features["x"]) with relu activation
  first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.layers.dense(
      first_hidden_layer, 10, activation=tf.nn.relu)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.layers.dense(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])

  # Provide an estimator spec for `ModeKeys.PREDICT`.
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"ages": predictions})

  # Calculate loss using mean squared error
  loss = tf.losses.mean_squared_error(labels, predictions)

  # Calculate root mean squared error as additional eval metric
  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(labels, tf.float64), predictions)
  }

  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params["learning_rate"])
  train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())

  # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)

{% endhighlight %}

## 3.8 运行模型

你已经实例化了一个Estimator，并在model_fn中定义了自己的行为；接着你可以去train/evaluate/predict。

将下面的代码添加到main()的尾部来对训练数据进行fit，并评估accuracy：

{% highlight python %}

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)

# Train
nn.train(input_fn=train_input_fn, steps=5000)

# Score accuracy
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(test_set.data)},
    y=np.array(test_set.target),
    num_epochs=1,
    shuffle=False)

ev = nn.evaluate(input_fn=test_input_fn)
print("Loss: %s" % ev["loss"])
print("Root Mean Squared Error: %s" % ev["rmse"])

{% endhighlight %}

**注意：上述代码使用input function（对于training (train_input_fn) ， 对于evaluation (test_input_fn)），来将 feature(x)和label(y) Tensors feed给模型，更多input fun的定义详见input_fn.**

运行代码可以看到：

{% highlight python %}

...
INFO:tensorflow:loss = 4.86658, step = 4701
INFO:tensorflow:loss = 4.86191, step = 4801
INFO:tensorflow:loss = 4.85788, step = 4901
...
INFO:tensorflow:Saving evaluation summary for 5000 step: loss = 5.581
Loss: 5.581

{% endhighlight %}

当运行时，model_fn所返回的mean squared error loss分值会输出。

为了预测age，可以添加下面的代码到main()中：

{% highlight python %}

# Print out predictions
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": prediction_set.data},
    num_epochs=1,
    shuffle=False)
predictions = nn.predict(input_fn=predict_input_fn)
for i, p in enumerate(predictions):
  print("Prediction %s: %s" % (i + 1, p["ages"]))

{% endhighlight %}

这里，predict() 函数会返回predictions的结果作为一个iterable。for loop会枚举和打印所有的结果。重新运行该代码，你可以看到如下输出：

{% highlight python %}

...
Prediction 1: 4.92229
Prediction 2: 10.3225
Prediction 3: 7.384
Prediction 4: 10.6264
Prediction 5: 11.0862
Prediction 6: 9.39239
Prediction 7: 11.1289

{% endhighlight %}

## 其它资源

- [Layers](https://www.tensorflow.org/api_guides/python/contrib.layers)
- [Losses](https://www.tensorflow.org/api_guides/python/contrib.losses)
- [Optimization](https://www.tensorflow.org/api_guides/python/contrib.layers#optimization)

# 参考

[tensorflow input_fn](https://www.tensorflow.org/get_started/input_fn)
