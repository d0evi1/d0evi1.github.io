---
layout: page
title:  tensorflow serving一个模型
tagline: 
---
{% include JB/setup %}

# 介绍

该tutorial会教你如何使用tensorflow serving组件来export一个训练好的tensorflow模型，并使用标准的tensorflow_model_server来提供服务。如果你已经对tensorflow serving很熟悉，你可以知道更多关于server内部工作机制，详见[advance_tutorial](https://www.tensorflow.org/serving/serving_advanced)。

该tutorial会使用简单的softmax regression模型来处理手写图片分类（MNIST 数据集）。

该教程的代码包含了两个部分：

- 一个Python文件，[mnist_saved_model.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_saved_model.py)，训练和导出该模型。
- 一个ModelServer二进制，它可以使用apt-get来安装、或者从C++文件[main.cc](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/model_servers/main.cc)中编译得到。Tensorflow Serving ModelServer会发现新的导出模型，接着运行一个gRPC服务来为它们提供服务。

在开始之前，请完成[先决条件](https://www.tensorflow.org/serving/setup#prerequisites)。

**注意：下面所有的bazel build命令使用标准的-c opt标志。为了更进一步优化build，可参见[instructions](https://www.tensorflow.org/serving/setup#optimized)**。

# 1.训练和导出tensorflow模型

mnist_saved_model.py中，Tensorflow graph会在sess中加载，输出的tensor(image)为x，输出的tensor（softmax score）为y。

接着我们使用tensorflow的[SavedModelBuilder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/builder.py)模块来导出模型。SavedModelBuilder会保存训练模型的一个“快照（snapshot）”进行可靠存储，以便在inference时能被加载。

SavedModel的格式，详见[SavedModel Readme.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md)。

下面的代码展示了保存一个模型到磁盘的整体过程：

{% highlight python %}

export_path_base = sys.argv[-1]
export_path = os.path.join(
      compat.as_bytes(export_path_base),
      compat.as_bytes(str(FLAGS.model_version)))
print 'Exporting trained model to', export_path
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           'predict_images':
               prediction_signature,
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               classification_signature,
      },
      legacy_init_op=legacy_init_op)
builder.save()

{% endhighlight %}

SavedModelBuilder.__init__会采用以下的参数：

- export_path是导出目录的路径

SavedModelBuilder会在目录不存在时创建目录。在本例中，我们将命令行参数和FLAGS.model_version拼接一起来获得导出的目录。

FLAGS.model_version指定了该模型的版本。当要导出相同模型的更新版本时，你可以指定一个更大的整数。每个version都将被导到给定路径下的一个不同目录下。

你可以使用SavedModelBuilder.add_meta_graph_and_variables() 添加meta graph和variables到该builder中，使用以下的参数：

- sess：tensorflow session，它会持有你导出的训练模型.
- tags：tags集合，用于保存元图(meta graph)。这种情况下，由于我们打算在serving中使用graph，我们会使用来自预定义的SavedModel标签常数的serve tag。更多细节，详见tag_constants.py。
- signature_def_map：指定了一个map：关于签名(signature)的key由用户提供，它指向一个添加到meta graph中的tensorflow::SignatureDef。signature指定了模型导出的类型、以及当运行inference时所绑定的input/output tensors。

特殊的signature key（serving_default）指定了缺省的serving signature。缺省的serving signature def key，会随着与签名相关的其它常数，被定义成SavedModel签名常数的一部分。更多细节详见signature_constants.py。

另外，为了帮助构建signature defs，SavedModel API提供了[签名工具类](https://www.tensorflow.org/api_docs/python/tf/saved_model/signature_def_utils)。特定的，在mnist_saved_model.py上面的代码片段中，我们使用了signature_def_utils.build_signature_def()来构建predict_signature和classification_signature。

正如示例中predict_signature是如何被定义的，该工具类也会采用以下的参数：

- inputs={'images': tensor_info_x} ：指定了input tensor的信息
- outputs={'scores': tensor_info_y}：指定了scores tensor的信息
- method_name：inference所使用的方法。对于预测请求，它会设置成tensorflow/serving/predict。更多方法名，详见：[signature_constants.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py)

注意，tensor_info_x和tensor_info_y具有tensorflow::TensorInfo 所定义的[protobuf](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto)的结构。为了更轻易构建tensor信息，Tensorflow SavedModel API也提供了[utils.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/utils.py)

另外，也要注意，images和scores是tensor的别名。它们可以是任何唯一的strings。

例如，如果x指的是带有名为"long_tensor_name_foo"的tensor，y指的是名为“generated_tensor_name_bar”的tensor，builder将存储tensor的逻辑名称到实际名称（'images'->'long_tensor_name_foo'），和('scores' -> 'generated_tensor_name_bar')。这允许该用户在运行inference时使用它们的逻辑名称来索引这些tensors。

**注意：除了以上的描述外，文档相关签名定义结构，以及如何设置它们可以在此发现。**

运行时，先清除已经存在的导出目录：

	$>rm -rf /tmp/mnist_model

如果你想安装tensorflow和tensorflow-serving-api PIP安装包，你可以使用简单的python命令来运行所有python代码（export和client）。为了安装PIP包，参考[以下命令](https://www.tensorflow.org/serving/setup#pip)。它也可能使用Bazel来构建必要的依赖和无需任何安装包来运行所有代码。指引的其余所有部分必须具有同时Bazel和PIP选项的指令。

	$>bazel build -c opt //tensorflow_serving/example:mnist_saved_model
	$>bazel-bin/tensorflow_serving/example/mnist_saved_model /tmp/mnist_model
	Training model...
	
	...
	
	Done training!
	Exporting trained model to /tmp/mnist_model
	Done exporting!

或者如果你已经安装了tensorflow-serving-api，你可以运行：

	python tensorflow_serving/example/mnist_saved_model.py /tmp/mnist_model

现在，看一下导出目录：

	$>ls /tmp/mnist_model
	1

如上所述，一个子目录将会被创建，以导出该模型的每个版本（version）。FLAGS.model_version具有缺省值为1, 因此会创建相应的子目录1。

	$>ls /tmp/mnist_model/1
	saved_model.pb variables

每个version的子目录包含了以下文件：

- saved_model.pb：是序列化的tensorflow::SavedModel。它包含了模型的一或多个graph定义，以及模型元数据（比如：签名）。
- variables：那些持有关于序列化图变量的文件。

你的Tensorflow模型可以被导出和ready。

# 2.使用标准ModelServer来加载导出模型

如果你想使用一个本地编译好的ModelServer，可以运行以下命令：

	$>bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
	$>bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/

如果你更喜欢跳过编译、安装，可以使用apt-get，使用以下[命令](https://www.tensorflow.org/serving/setup#aptget)。以以下命令运行server：

	tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/

# 3.测试Server

我们可以使用提供的[mnist_client](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_client.py)工具类来测试server。client会下载MNIST测试数据，将它们作为请求发送到该server上，并计算inference的error rate。

使用Bazel运行：

	$>bazel build -c opt //tensorflow_serving/example:mnist_client
	$>bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000
	...
	Inference error rate: 10.5%

可选的，如果你安装PIP包，运行：

	python tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=localhost:9000

我们期望，对于初始的1000个测试图片，对于训练好的Softmax模型有91%左右的accuracy，我们获得了10.5%的inference error rate。这证实了server会成功加载和运行训练好的模型。



# 参考

[https://www.tensorflow.org/serving/serving_basic](https://www.tensorflow.org/serving/serving_basic)
