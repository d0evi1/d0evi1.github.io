---
layout: page
title:  模型文件工具开发指南
tagline: 
---
{% include JB/setup %}

# 介绍

大多数用户不需要关心tensorflow在磁盘上存储数据的内部细节，但是工具开发者除外。例如，你可能希望分析模型，或者将tensorflow格式与其它格式相互转换。该指南会解释一些关于如何与持有模型数据的主文件工作的细节，以便更容易开发这些工具。

# 1. Protocol Buffers

所有tensorflow文件格式都是基于protobuf，因此你需要先对它熟悉下。总体上，你需要在文本文件中定义数据结构，通过protobuf工具生成c、python或者其它语言的类（classes）来加载、保存、访问数据。

# 2. GraphDef

tensorflow的计算基础是Graph对象。它会持有一个节点网络，每个节点代表了一个操作（op），它会与其它op作为输入或输出进行相互连接。在你已经创建一个Graph对象后，你可以通过调用as_graph_def()来保存它，它会返回一个GraphDef对象。

GraphDef类是一个由Protobuf库根据在[tensorflow/core/framework/graph.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)的定义创建的对象。protobuf工具会解析该文本文件，生成代码来加载、存储和操控graph定义。如果你看到一个用来表示一个模型的tensorflow独立文件，它会包含关于这些由protobuf代码保存的GraphDef对象一个序列化版本。

生成的代码用于从磁盘中保存和加载GraphDef。该代码实际上会以如下方式加载模型：

{% highlight python %}

graph_def = graph_pb2.GraphDef()

{% endhighlight %}

这行会创建一个空的GraphDef对象，该类会被由graph.proto中的文本定义所创建。

{% highlight python %}
with open(FLAGS.graph, "rb") as f:

{% endhighlight %}

这里我们得到一个关于路径的文件句柄，传给该脚本：

{% highlight python %}
if FLAGS.input_binary:
	graph_def.ParseFromString(f.read())
else:
	text_format.Merge(f.read(), graph_def)

{% endhighlight %}

# 3.文本格式 vs.二进制格式?

实际上，一个protobuf可以保存成两种不同的格式。TextFormat是一种人类可读的格式，方便调式和编辑，但当存储数值型数据（比如：weights）是会变得很大。你可以看到在[graph_run_run2.pbtxt](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/demo/data/graph_run_run2.pbtxt)中的一个小示例。

二进制格式文件比文本格式更小，但它不可读。我们会告诉用户支持一个flag来标识输入文件是binary还是text。你可以发现在[inception_v3 archive](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz)，（inception_v3_2016_08_28_frozen.pb）中是一个很大的二进制文件。

该API本身可能有些混淆，二进制调用实际上是ParseFromString()，其中你使用text_format模块的一个工具函数来加载文本文件。

# 4.节点（Nodes）

一旦你加载一个文件到graph_def变量中，你可以访问在它内部的数据。对于大多数实际目的，最重要部分是保存在节点成员中的节点列表。这里的代码可以这样循环：

{% highlight python %}

for node in graph_def.node

{% endhighlight %}

每个node是一个NodeDef对象，在[tensorflow/core/framework/node_def.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto)中有定义。这些是tensorflow graph的基础构建块，每一个定义了一个与输入连接相关的op。以下是一个NodeDef的成员。

## 4.1 name

每一个node都应有一个唯一的id，不能被graph中的其它node所使用。如果你在使用python API构建一个graph时不指定id，系统会生成：一部分对应着op名，比如“MatMul”，它会与一个递增的数字相连，比如"5"。在当需要定义节点间的连接，以及当运行graph时设置inputs和outputs时，会用到该name。

## 4.2 op

op定义了要运行的操作，例如："Add", "MatMul", or "Conv2D"。当一个graph运行时，op名可以通过一个注册表（registry）查询来发现对应的实现。该注册表由REGISTER_OP()宏调用得到，位于[tensorflow/core/ops/nn_ops.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/nn_ops.cc)。

## 4.3 input

一个字符串列表，每个元素表示另一个节点的name，可选的，可以跟一个冒号和一个输出端口号。例如，一个node具有两个输入，必须具有像这样的一列：["some_node_name", "another_node_name"]，它等价于["some_node_name:0", "another_node_name:0"]，将该节点的第一个input定义为来自名为"some_node_name"节点的第一个输出，第二个输入来自于"another_node_name"的第一个输出。

## 4.4 device

在大多数情况下，你可以忽略它，因为它定义了在分布式环境下运行一个node，或者你希望强制该op在CPU或GPU上。

## 4.5 attr

这是一个key/value存储，它持有了一个node的所有属性。它们是节点的永久属性，在运行时（runtime）这些属性（比如：卷积filters的size，constant ops的值等）不会发生变化。因为它们可以是许多类型的属性值（从string类型，到int型，到tensor array值），会有一个独立的protobuf文件来定义相应数据结构持有它们，在[ tensorflow/core/framework/attr_value.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)中。

每个属性具有一个唯一的name字符串，当该op被定义时，想要的属性会列出。如果一个属性在一节点中不存在，但它某op定义列表中具有一个缺省，缺省情况下当该graph被创建时会使用该属性。

你可以在python中通过调用node.name, node.op来访问所有成员，在GraphDef中存储的节点列表是该模型架构的一个完整定义。

# 5.Freezing

关于Freezing一个令人困惑的部分是，在训练期间权重（weights）通常不会保存在文件格式中。它们会存在在独立的checkpoint文件中，在graph的Variable ops在初始化时会加载最新的值。当你想部署到生产环境中时，这种独立文件的方式通常并不方便，因此，[freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)脚本会采用一个graph定义、一个checkpoints集合，来将它们freeze到单个文件中。

整个过程是，加载GraphDef，从最新的checkpoint文件中抽取所有变量的值，接着使用一个Const（它具有在属性中存储的权重的数值型数据）来替换每个Variable op，接着去掉没有关联的节点（不会在forward inference中被使用），保存产生的GraphDef到一个输出文件中。

# 6.Weight格式

如果你正处理用于表示神经网络的tensorflow模型，一个最常见的问题是，抽取和解析weight值。一种常见的存储方法是，例如由freeze_graph脚本创建的graphs中，作为Const ops将包含的weights当成是Tensors。它们在[tensorflow/core/framework/tensor.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto)中有定义，并包含了关于数据的size、type、value信息。在python中，你可以从一个NodeDef中获取一个TensorProto对象，通过调用类似这样的操作：some_node_def.attr['value'].tensor来表示一个Const op。

这给出了一个用于表示weight数据的对象。该数据本身会被存储在一列带有后缀_val（表示对象类型）的列表中，例如：float_val表示32-bit的float数据类型。

当在不同的框架间进行转换时，卷积权重值的顺序通常很难去处理。在Tensorflow中，对于Conv2D操作的filter的weights被存储在第二个input中，顺序为[filter_height, filter_width, input_depth, output_depth]，其中filter_count会通过一个到邻近值的移动均值进行递增。

# 参考

[A Tool Developer's Guide to TensorFlow Model Files](https://www.tensorflow.org/extend/tool_developers/)
