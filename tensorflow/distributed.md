---
layout: page
title:  分布式tensorflow
tagline: 
---
{% include JB/setup %}

# 介绍

该文档展示了如何创建tensorflow servers的一个集群，以及如何将计算图（computation graph）跨集群进行分布。我们假设你已经对tensorflow的基本概念已经很熟悉。

# 一、hello distributed TensorFlow!

执行下面代码，可以看到一个简单的tensorflow cluster示例：

{% highlight python %}

# Start a TensorFlow server as a single-process "cluster".
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> server = tf.train.Server.create_local_server()
>>> sess = tf.Session(server.target)  # Create a session on the server.
>>> sess.run(c)
'Hello, distributed TensorFlow!'

{% endhighlight %}

[tf.train.Server.create_local_server](https://www.tensorflow.org/api_docs/python/tf/train/Server#create_local_server)会创建一个单进程的cluster，它是一个in-process server。

# 二、创建一个cluster

TensorFlow "**cluster**"指的是一个任务集合（a set of "tasks"），它们会以分布式方式执行一个tensorflow graph。每个任务（**task**）与一个tensorflow的"**server**"相关，该server包含了一个“**master**”：用于创建sessions，以及一个"**worker**"：在graph中执行op。一个cluster也可以被分割成一或多个"**jobs**"：其中每个job包含了一或多个tasks。

即：

- cluster -> [1-n] task -> [1]server -> ([1]master => sessions, [1]worker => op)
- cluster -> [1-n] job -> [1-n] task

为了创建一个cluster，你需要在cluster上为每个task启动tensorflow server。每个task通常会运行在一个不同的机器上，但你也可以在相同的机器上运行多个tasks（例如：为了控制不同的GPU device）。在每个task上，做以下事情：

- 1.创建一个[tf.train.ClusterSpec](https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec)，它在cluster中描述了所有的tasks。这对于每个task来说是相同的
- 2.创建一个tf.train.Server，将tf.train.ClusterSpec传给构造器，使用一个job name和task index来标识local task。

## 2.1 创建一个tf.train.ClusterSpec来描述cluster

cluster指定字典会将job names映射到网络地址列表上。将该字典传递给tf.train.ClusterSpec构造器。例如：

tf.train.ClusterSpec

如果使用：

	tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

那么对应的tasks为：

	/job:local/task:0
	/job:local/task:1

如果使用：

	tf.train.ClusterSpec({
    "worker": [
        "worker0.example.com:2222",
        "worker1.example.com:2222",
        "worker2.example.com:2222"
    ],
    "ps": [
        "ps0.example.com:2222",
        "ps1.example.com:2222"
    ]})

那么对应的tasks为：

	/job:worker/task:0
	/job:worker/task:1
	/job:worker/task:2
	/job:ps/task:0
	/job:ps/task:1

## 2.2 在每个task上创建一个tf.train.Server实例

一个tf.train.Server对象包含了一个local devices集合，一个关于与tf.train.ClusterSpec中的其它tasks相连接的connections集合，以及一个可以使用上述集合进行一个分布式计算的tf.Session。每个server是一个指定名字的job的成员，在该job内有一个task index。一个server可以与在cluster中的任何其它server进行通信。

例如，为了使用运行在localhost:2222和localhost:2223上的两个servers来启动一个cluster，可以在本地机器上以两种不同方式运行以下的代码段：

{% highlight python %}

# In task 0:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)

{% endhighlight %}



{% highlight python %}

# In task 1:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)

{% endhighlight %}

注意：人工指定这些cluster specifications可能很乏味，特别是对于大集群（large clusters）。我们正开发工具来以可编程的方式启动tasks，例如：使用一个cluster manager（[Kubernetes](http://kubernetes.io/)）。如果存在特别的cluster manager你希望支持，可以提交Github issue。

# 三、在模型中指定分布式设备

为了在一个特别的进程上放置ops，你可以使用相同的tf.device 函数来指定ops是否运行在CPU或GPU上。例如：

{% highlight python %}

with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...

with tf.Session("grpc://worker7.example.com:2222") as sess:
  for _ in range(10000):
    sess.run(train_op)

{% endhighlight %}

在上述示例中，在ps job的两个task上创建变量，模型的对计算敏感部分在worker job上被创建。tensorflow将会插入合适的在jobs间的data transfers。(从ps到worker进行forward pass，从worker到ps应用gradients)

# 四、Replicated training

一个常见的训练配置，被称为“数据并行化（data parallelism）”，涉及到在一个worker job上的多个tasks在不同的mini-batch数据上训练相同的模型，更新位于在ps job上的一或多个tasks上的共享参数。所有tasks通常运行在不同的机器上。有许多种方式来指定在tensorflow中的结构，我们可以构建库来简化指定一个replicated model的工作量。可能的方法包括：

- **in-graph replication**。在这种方法中，客户端会构建单个tf.Graph，它包含了一个参数集合（在位于/job:ps绑定的tf.Variable节点），以及模型的计算敏感部分的多个拷贝，每个都绑定到在/job:worker上的一个不同的任务上。
- **between-graph replication**。这种方法中，对于每个/job:worker task存在一个独立的client，通常和worker task在相同的进程中。每个client会构建一个相似的包含这些参数（在使用[tf.train.replica_device_setter](https://www.tensorflow.org/api_docs/python/tf/train/replica_device_setter)将这些参数映射到相同的tasks上之前，先绑定到/job:ps上）的graph，以及模型的计算敏感部分的单个copy，绑定到在/job:worker中的local task上。
- **异步训练（Asynchronous training）**。在该方法中，该graph的每个replica都具有一个独立的训练loop，它的执行不需要协同。上述多种形式的replication可以兼容。
- **同步训练（Synchronous training）**。在该方法中，所有replicas会读取当前参数相同的值，并列计算梯度，接着将它们一起应用。它与in-graph replication（例如：在[cifar-10多cpu训练中](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py)使用梯度平均）、以及between-graph相兼容。（例如：使用[tf.train.SyncReplicasOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer)）


## 将它们放置在一起：示例训练程序

下面的代码展示了一个分布式训练器程序，它实现了between-graph replication和asynchronous training。它包含了parameter server和worker tasks。

{% highlight python %}

import argparse
import sys

import tensorflow as tf

FLAGS = None

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      loss = ...
      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

{% endhighlight %}

如果想使用两个parameter servers，以及两个workers来启动trainer，可以使用下面的命令行：

{% highlight python %}
# On ps0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=0
# On ps1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=1
# On worker0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=0
# On worker1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=1

{% endhighlight %}

# 五、术语

## Client

一个client通常是一个程序，它构建了一个tensorflow graph以及构建了一个tensorflow::Session来与一个cluster相交互。Clients通常使用Python或C++编写。单个client进程可以直接与多个tensorflow servers交互（详见“replicated training”），而单个server可以服务多个clients。

## Cluster

一个tensorflow cluster由一或多个"jobs"组成，每个job可以划分成一或多个tasks。一个cluster通常被用于一个特别高级的目标，比如：使用多台机器并行训练一个神经网络。一个cluster可以通过一个tf.train.ClusterSpec对象来定义。

## Job

一个job由一列tasks组成，它们通常服务于一个常用的目的。例如，一个名为“ps”的job（表示“parameter server”）通常寄宿着存储和更新变量的节点；而一个名为“worker”的job通常寄宿着执行对计算敏感tasks的无状态节点。在一个job中的tasks会运行在不同的机器上。job集合的角色相当灵活，一个worker可以维持一些状态。

## Master service

这是一个RPC server，它提供了远程访问分布式设备，并扮演着一个session target的角色。master service实现了tensorflow::Session接口，负责协调跨一或多个"worker service"间的工作。所有tensorflow server都实现了master service。

## Task

一个task对应于一个指定的TensorFlow server，通常对应单个进程。一个task属于一个特殊的"job"，可以在tasks的job列表中通过索引进行标识。

## Tensorflow server

一个运行着一个tf.train.Server实例的进程，它是一个cluster的一个成员，可以导出一个“master service”和"worker service"。

## Worker service

一个RPC service，可以使用它的local devices执行一个tensorflow graph的一部分。一个worker service实现了[ worker_service.proto](https://www.github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/core/protobuf/worker_service.proto)。所有tensorflow server实现了worker service。


# 参考

[tensorflow 分布式](https://www.tensorflow.org/deploy/distributed)
