---
layout: page
title:  tensorflow中的input_fn
tagline: 
---
{% include JB/setup %}

# 使用tf.estimator构建input function

本文介绍如何在tf.estimator中创建输入函数（input function）。你将了解到：构建一个input_fn可以进行预处理，并将数据feed给你的模型。接着，你可以实现一个input_fn，它可以将training/evaluation/prediction data这些数据feed给一个神经网络的regressor来预测房价中位数。

## 1.使用input_fn定制input pipelines

input_fn用于将feature和target data传递给Estimator的train/evaluate/predict方法。用户可以在input_fn中做特征工程或进行数据预处理。下面是一个示例：

{% highlight python %}

import numpy as np

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float32)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)

classifier.train(input_fn=train_input_fn, steps=2000)

{% endhighlight %}

## 2. input_fn的详解

下面代码展示了input_fn的基本骨架：

{% highlight python %}

def my_input_fn():

    # Preprocess your data here...

    # ...then return：
    #  1) a mapping of feature columns to Tensors with the corresponding feature data,
    #  2) a Tensor containing labels
    return feature_cols, labels


{% endhighlight %}

输入函数的body部分包含了特定的逻辑，可以进行数据预处理，比如：将badcase example去掉，或者进行feature scaling。

输入函数必须返回以下两种值：包含最终feature和label的数据，最后会feed给你的模型：

- feature_cols: 一个包含key/value pairs的dict，它将feature column names映射到对应的Tensors（或SparseTensor s）上，它们包含对应的feature data.
- labels: 一个包含label(target)值的Tensor：你的模型要预测的目标值。

## 3.将feature data转换成Tensors

如果你的feature/label data是一个python array或pandas dataframes或numpy arrays，你可以使用下面的方法来构建input_fn：

{% highlight python %}

import numpy as np
 
# numpy input_fn.
my_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x_data)},
    y=np.array(y_data),
    ...)
    
import pandas as pd
# pandas input_fn.
my_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=pd.DataFrame({"x": x_data}),
    y=pd.Series(y_data),
    ...)


{% endhighlight %}

对于sparse, categorical data（大部分数据为0），你可以使用SparseTensor来替代，它通过3个参数进行实例化：

- dense_shape： tensor的shape。传入一个list，在每个维度上对元素数目进行索引。dense_shape=[3,6] 表示一个3x6的2D tensor，dense_shape=[2,3,4]表示一个2x3x4的3D tensor，dense_shape=[9] 表示一个包含9个元素的1D tensor。
- indices: 在你的tensor中非零元素的索引。传入一个关于terms的list。每个term它自身是一个包含非零元素的索引列。（元素从0开始索引，比如：[0,0]表示在2D tensor中的第一行、第一列）例如，indices=[[1,3], [2,4]] 表示[1,3] 和 [2,4]具有非零值。
- values: 一个1D tensor的值。在values中的Term i 对应于在indices中的第i个term，并指定它的值。例如：indices=[[1,3], [2,4]]，values=[18, 3.6]表示：[1,3]的值为18, [2,4]的值为3.6.

以下的代码定义了一个2D SparseTensor，它具有3行、5列。index [0,1]的元素具有值6, [2,4] 的值为0.5.

{% highlight python %}

sparse_tensor = tf.SparseTensor(indices=[[0,1], [2,4]],
                                values=[6, 0.5],
                                dense_shape=[3, 5])

{% endhighlight %}

对应的dense tensor为：

{% highlight python %}

[[0, 6, 0, 0, 0]
 [0, 0, 0, 0, 0]
 [0, 0, 0, 0, 0.5]]

{% endhighlight %}

## 4.将input_fn的数据传给模型

为了将数据feed给你的模型进行训练，你可以简单地将创建好的input_fn传给train操作即可。

{% highlight python %}

classifier.train(input_fn=my_input_fn, steps=2000)

{% endhighlight %}

注意：input_fn参数必须是一个函数对象（function object），比如：input_fn=my_input_fn。而不是函数调用的返回值 (input_fn=my_input_fn())。这意味着如果你尝试将带参数的函数传递给train中的input_fn，会导致TypeError: 

{% highlight python %}

classifier.train(input_fn=my_input_fn(training_set), steps=2000)

{% endhighlight %}

然而，如果你想参数化你的input_fn，有其它的办法可以做到。你可以使用一个wrapper函数，它不带任何参数，作为你的input_fn，使用它来调用你想要参数的input function。例如：

{% highlight python %}

def my_input_fn(data_set):
  ...

def my_input_fn_training_set():
  return my_input_fn(training_set)

classifier.train(input_fn=my_input_fn_training_set, steps=2000)

{% endhighlight %}

另一种方法是，你可以使用Python的functools.partial 函数来构建一个新的函数对象，所有参数值都固定：

{% highlight python %}

classifier.train(
    input_fn=functools.partial(my_input_fn, data_set=training_set), steps=2000)

{% endhighlight %}

第三种方法是，将你的input_fn封装在一个lambda函数中，做为参数传给input_fn：

{% highlight python %}

classifier.train(input_fn=lambda: my_input_fn(training_set), steps=2000)

{% endhighlight %}

如上设计input pipeline的一个大的优点是：可以接受一个dataset参数——传递相同的input_fn，只需要调整个dataset函数就可以共享给evaluate和predict操作:

{% highlight python %}

classifier.evaluate(input_fn=lambda: my_input_fn(test_set), steps=2000)

{% endhighlight %}

这种方法增强了代码的可维护性：不必为每个操作类型定义多个input_fn（比如：input_fn_train，input_fn_test，input_fn_predict）

最终，你可以使用在tf.estimator.inputs中的方法来从numpy 或 pandas datasets中创建input_fn。额外的好处是，你可以使用更多的参数，比如：num_epochs和shuffle来控制在数据上的input_fn如何迭代。

{% highlight python %}

import pandas as pd

def get_input_fn_from_pandas(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pdDataFrame(...),
      y=pd.Series(...),
      num_epochs=num_epochs,
      shuffle=shuffle)


import numpy as np

def get_input_fn_from_numpy(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.numpy_input_fn(
      x={...},
      y=np.array(...),
      num_epochs=num_epochs,
      shuffle=shuffle)

{% endhighlight %}


# 2. Boston房价预测的神经网络模型

[Boston CSV data sets](https://www.tensorflow.org/get_started/input_fn#setup) 使用以下特征描述来构建NN：

- CRIM：每个capita的犯罪率
- ZN：居住用地允许25000+平方英尺lots的部分
- INDUS: 非零售业占地部分
- NOX: 氮氧化物浓度(每1000w)
- RM: 每个住所的平均房间数
- AGE: 在1940年后业主自主的比例
- DIS: 离Boston地区就业中心的距离
- TAX: 每$10,000的不动产税率
- PTRATIO: 学生-教师比率

我们要预测的label是MEDV，业主自主住宅的中位数（单位：千元金）

## 2.1 Setup

下载：

- [boston_train.csv](http://download.tensorflow.org/data/boston_train.csv)
- [boston_test.csv](http://download.tensorflow.org/data/boston_test.csv)
- [boston_predict.csv](http://download.tensorflow.org/data/boston_predict.csv)

以下部分提供了：如何创建input function，feed datasets给NN regressor，进行train和evaluate模型，以及最后做出预测。

完整代码见：[boston.py](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/input_fn/boston.py)

{% highlight python %}

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

# INFO: 输出更多log信息
tf.logging.set_verbosity(tf.logging.INFO)


# 区分features/labels.
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"


# 构建input_fn.可以很方便地换数据集参数：train/test/predict. 
# num_epochs: 控制着迭代的轮数；对于训练数据，设置为None，直到所需训练steps数达到后input_fn才返回；对于evaluate 和 predict, 设置为1, input_fn会在数据上迭代一次, 接着抛出OutOfRangeError。该error会发信号给Estimator以便停止evaluate或predict。
# shuffle: 是否对数据进行shuffle。对于evaluate 和 predict，设置为False，input_fn会顺序化访问数据. 对于train，置为True.
def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)


def main(unused_argv):
  # Load datasets
  training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
  test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                         skiprows=1, names=COLUMNS)

  # Set of 6 examples for which to predict median house values
  prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

  # Feature cols. 所有特征都是连续型值变量。
  feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  # 每个隐层需要的节点数。（两个隐层，每个隐层10个节点）
  regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                        hidden_units=[10, 10],
                                        model_dir="/tmp/boston_model")

  # Train
  regressor.train(input_fn=get_input_fn(training_set), steps=5000)

  # Evaluate loss over one epoch of test_set.
  ev = regressor.evaluate(
      input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  # Print out predictions over a slice of prediction_set.
  y = regressor.predict(
      input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
  # .predict() returns an iterator of dicts; convert to a list and print
  # predictions
  predictions = list(p["predictions"] for p in itertools.islice(y, 6))
  print("Predictions: {}".format(str(predictions)))

if __name__ == "__main__":
  tf.app.run()

{% endhighlight %}


每100个steps可以看到：

{% highlight python %}

INFO:tensorflow:Step 1: loss = 483.179
INFO:tensorflow:Step 101: loss = 81.2072
INFO:tensorflow:Step 201: loss = 72.4354
...
INFO:tensorflow:Step 1801: loss = 33.4454
INFO:tensorflow:Step 1901: loss = 32.3397
INFO:tensorflow:Step 2001: loss = 32.0053
INFO:tensorflow:Step 4801: loss = 27.2791
INFO:tensorflow:Step 4901: loss = 27.2251
INFO:tensorflow:Saving checkpoints for 5000 into /tmp/boston_model/model.ckpt.
INFO:tensorflow:Loss for final step: 27.1674.

{% endhighlight %}


# 参考

[tensorflow input_fn](https://www.tensorflow.org/get_started/input_fn)
