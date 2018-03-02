---
layout: page
title:  tensorflow GPU
tagline: 
---
{% include JB/setup %}

# 一、支持的设备

在一个常见系统中，会存在多个计算设备（devices）。在tensorflow中，支持的设备类型是：CPU和GPU。它们被表示成strings。例如：

- "/cpu:0"： 你机器的CPU
- “/device:GPU:0”：机器的GPU，如果你只有一块GPU
- “/device:GPU:1”：你机器的第二块GPU

如果某个tensorflow操作同时具有CPU和GPU实现，当该操作被分配到一个设备时，GPU设备会有优先权。例如：matmul操作同时具有CPU和GPU kernels。在一个具有cpu:0和gpu:0的设备上，gpu:0会被选中来运行matmul。

# 二、记录设备放置信息(placement)

为了找到你的操作（op）和tensors被分配的设备，需要使用log_device_placement配置选项置为True来创建session：

{% highlight python %}

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

{% endhighlight %}

你可以看到以下的输出：

{% highlight python %}

Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/device:GPU:0
a: /job:localhost/replica:0/task:0/device:GPU:0
MatMul: /job:localhost/replica:0/task:0/device:GPU:0
[[ 22.  28.]
 [ 49.  64.]]

{% endhighlight %}

# 三、人工设置device placement

如果你希望某个特别的操作运行在一个你选中的设备上，而非系统自动选择，你可以使用tf.device来创建一个设备上下文，所有在该上下文的操作都具有相同的设备分配（device assigment）。

{% highlight python %}

# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

{% endhighlight %}

你将看到，a和b被分配给cpu:0。因为没有为MatMul操作显式指定一个设备，tensorflow runtime会基于该操作和提供的设备（本例为gpu:0）选择一个设备，如有需要会在设备间自动拷贝tensors。

{% highlight python %}

Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/cpu:0
a: /job:localhost/replica:0/task:0/cpu:0
MatMul: /job:localhost/replica:0/task:0/device:GPU:0
[[ 22.  28.]
 [ 49.  64.]]

{% endhighlight %}

# 四、允许GPU内存增长

缺省的，tensorflow会把几乎所有GPU的所有GPU内存（CUDA_VISIBLE_DEVICES）对进程可见。这样做是为了更有效地在设备上使用相对宝贵的GPU内存，以减小[内存碎片](https://en.wikipedia.org/wiki/Fragmentation_(computing))。

在一些情况下，进程只分配所提供内存的一个子集是适宜的，或者只按需增长内存使用。tensorflow提供了两个Config选项来控制它。

第一个是allow_growth选项，它会基于runtime内存分配来尝试分配更多GPU内存：它开始时会分配少量内存，当随着Session运行越来越需要更多GPU内存时，我们会根据Tensorflow进程需求来扩展GPU内存范围。注意，我们不需要释放内存，因为可能会导致更糟糕的内存碎片。为了开启该选项，可以在ConfigProto中设置该选项：

{% highlight python %}

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)

{% endhighlight %}

第二个方法是per_process_gpu_memory_fraction选项，它决定着每个可见GPU应分配内存的整体内存量的比例。例如，你可以告诉tensorflow，只为每个GPU分配40%的总内存量：

{% highlight python %}

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)

{% endhighlight %}

如果你想为tensorflow进程真实界定可提供的GPU内存量，这方法会有用。

# 五、在多GPU系统中使用单个GPU

如果你的系统具有超过一个GPU，缺省情况下会选中带有最小ID的GPU。如果你希望在一个不同的GPU上运行，你需要显示进行指定：

{% highlight python %}

# Creates a graph.
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

{% endhighlight %}

如果你指定的设备不存在，你将得到InvalidArgumentError：

{% highlight python %}

InvalidArgumentError: Invalid argument: Cannot assign a device to node 'b':
Could not satisfy explicit device specification '/device:GPU:2'
   [[Node: b = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [3,2]
   values: 1 2 3...>, _device="/device:GPU:2"]()]]

{% endhighlight %}

当指定的设备不存在时，如果你希望tensorflow自动选择一个存在的、可支持的设备来运行运算操作，你可以设置allow_soft_placement=True：

{% highlight python %}

# Creates a graph.
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with allow_soft_placement and log_device_placement set
# to True.
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
# Runs the op.
print(sess.run(c))

{% endhighlight %}

# 六、使用多GPU

如果你想在多GPU上运行tensorflow，你可以以多塔结构（multi-tower）的方式来构建你的模型，每个tower被分配到一个不同的GPU上。例如：

{% highlight python %}

# Creates a graph.
c = []
for d in ['/device:GPU:2', '/device:GPU:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))

{% endhighlight %}

你将得到以下的输出：

{% highlight python %}

Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K20m, pci bus
id: 0000:02:00.0
/job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: Tesla K20m, pci bus
id: 0000:03:00.0
/job:localhost/replica:0/task:0/device:GPU:2 -> device: 2, name: Tesla K20m, pci bus
id: 0000:83:00.0
/job:localhost/replica:0/task:0/device:GPU:3 -> device: 3, name: Tesla K20m, pci bus
id: 0000:84:00.0
Const_3: /job:localhost/replica:0/task:0/device:GPU:3
Const_2: /job:localhost/replica:0/task:0/device:GPU:3
MatMul_1: /job:localhost/replica:0/task:0/device:GPU:3
Const_1: /job:localhost/replica:0/task:0/device:GPU:2
Const: /job:localhost/replica:0/task:0/device:GPU:2
MatMul: /job:localhost/replica:0/task:0/device:GPU:2
AddN: /job:localhost/replica:0/task:0/cpu:0
[[  44.   56.]
 [  98.  128.]]

{% endhighlight %}

[cifar10 tutorial](https://www.tensorflow.org/tutorials/deep_cnn)是一个很好的示例，演示了如何使用多GPU进行训练。

# 参考

[tensorflow using_gpu](https://www.tensorflow.org/programmers_guide/using_gpu)
