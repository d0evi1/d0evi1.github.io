---
layout: page
title:  tensorflow estimator使用SavedModel
tagline: 
---
{% include JB/setup %}

在训练完一个Estimator模型后，你希望创建一个service：使用该模型来对请求返回相应的预测结果。你可以在本机运行这样的service，或者将它部署到云平台上。

为了让一个已训练好的Estimator进行serving，你必须将它导出成标准的SavedModel格式。该文会解释：

- 如何指定输出节点(output nodes)和相应的APIs（Classify, Regress, or Predict）。
- 如何将你的模型导出成SavedModel格式
- 如何在一个local server上提供模型服务，并请求预测

# 1.准备serving inputs

在训练时，input_fn()会获取数据，在模型使用前对数据进行预处理。相似的，在serving时，需要一个serving_input_receiver_fn()，它会接收inference请求，并为该模型做相应的预处理。该函数具有以下的功能：

- 增加**placeholders**到graph中，serving系统在获得inference请求时会进行feed数据
- 增加了**额外ops**：可以将原有输入格式的数据转换成模型所需特征tensors

该函数会返回一个[tf.estimator.export.ServingInputReceiver](https://www.tensorflow.org/api_docs/python/tf/estimator/export/ServingInputReceiver)对象，**它会将placeholders打包，并生成相应的特征Tensors**。

**一个典型的模式(pattern)是，inference请求以序列化的tf.Example(S)到达，因此，serving_input_receiver_fn()会创建单个string placeholder来接收他们。接着，serving_input_receiver_fn()会通过添加一个tf.parse_example op到graph中来负责解析tf.Example(S)。**

当编写这样的serving_input_receiver_fn()时，你必须传递一个parsing sepcification给tf.parse_example来告诉parser：哪些特征名，以及如何将它们映射成Tensor(s)。一个parsing specification会采用一个字典格式，将特征名映射到tf.FixedLenFeature, tf.VarLenFeature, 以及tf.SparseFeature。注意，该parsing specification不应包含任何label column或weight columns，因为这在serving time时是不需提供的——而训练时的input_fn()则相反。

组成在一起，即：

{% highlight python %}

feature_spec = {'foo': tf.FixedLenFeature(...),
                'bar': tf.VarLenFeature(...)}

def serving_input_receiver_fn():
  """An input receiver that expects a serialized tf.Example."""
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[default_batch_size],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

{% endhighlight %}

tf.estimator.export.build_parsing_serving_input_receiver_fn工具类提供了通用的input receiver。


**注意：当训练一个模型提供服务并在一个local server上使用Predict API，不需要parsing阶段，因为模型会接受原始特征数据。**

即使你不需要parsing或者其它的输入预处理——也就是说，serving系统会直接feed相应的特征Tensors——你必须仍要提供一个serving_input_receiver_fn()，它会为特征Tensors创建placeholders，并将它们传进去。tf.estimator.export.build_raw_serving_input_receiver_fn可以提供该功能。

如果这些工具类不能满足你的需要，你可以自由地编写自己的serving_input_receiver_fn()。一种情况是，如果训练input_fn()合并提炼了一些预处理逻辑，在serving时必须重新被使用。为了减少training-serving交叉的风险，我们推荐你在一个函数中重新封装这样的处理逻辑，然后它可以被input_fn()和serving_input_receiver_fn()调用。

注意，serving_input_receiver_fn() 也会决定签名（signature）的input部分。也就是说，当编写一个serving_input_receiver_fn()时，你必须告知parser：期望什么签名（signatures）、如何将它们映射到你的模型所期望的输入。相反地，signature的output部分由模型来决定。

# 2.指定一个定制模型的输出

当编写一个定制的model_fn时，你必须填入tf.estimator.EstimatorSpec返回值中的export_outputs元素。它是一个关于{name: output}的字典，描述了要导出和在serving时使用的输出签名（output signatures）。

在常见的只做单个预测的情况下，该dict只包含一个元素，name是不重要的。在一个multi-headed模型中，每个head由在该dict中的一个entry所表示。这种情况下，name是关于你的选择的一个string，它可以在serving时被用于请求一个指定的head。

每个output值必须是一个ExportOutput对象，比如tf.estimator.export.ClassificationOutput, tf.estimator.export.RegressionOutput, 或者 tf.estimator.export.PredictOutput。

这样的输出类型（output types）可以直接映射到[Tensorflow Serving APIs](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto)上，因此决定着哪些请求类型会被使用。

**注意：在multi-headed的情况下，model_fn会返回export_outputs字典：将会为每个element生成一个SignatureDef，使用相同的key命名。这些SignatureDefs与它们的输出（outputs）不同，通过相应的ExportOutput entry提供。输入总是由serving_input_receiver_fn提供。一个inference请求必须通过name指定head。一个head必须使用signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY进行命名，它表示当一个inference请求没有指定一个head（？？？）时，哪个SignatureDef会被服务到。**

# 3.执行export操作

为了导出你训练好的Estimator，需要调用tf.estimator.Estimator.export_savedmodel，需要填入export_dir_base和serving_input_receiver_fn参数。

	estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn,
                            strip_default_attrs=True)


该方法会构建一个新的graph，通过首先调用serving_input_receiver_fn()来获取特征Tensor(s)，接着基于这些特征来调用该Estimator的model_fn()生成模型graph。它会启动一个新的Session，接着恢复最近的模型checkpoint到该session中（如有需要，可传递一个不同的checkpoint）。最后，它会在给定的export_dir_base下创建一个时间戳导出目录（比如：export_dir_base/<timestamp>），并将一个SavedModel写到该目录下。

# 4.本地为导出模型提供服务

为了本地部署，你可以使用Tensorflow Serving来为你的模型提供服务，它会加载一个SavedModel，并将它导出成一个gRPC服务。

	bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_base_path=$export_dir_base

现在，你有一个server，它会通过gRPC监听来自9000端口的inference请求。

# 5.从一个local server上请求预测

该server会根据PredictionService gRPC API的服务定义为gRPC请求提供服务。

从API service定义，gRPC框架可以以多种语言生成客户端库，以提供远端访问。如果在一个使用Bazel build tool的项目中，这些库会自动构建，并通过依赖提供。

	deps = [
	    "//tensorflow_serving/apis:classification_proto_py_pb2",
	    "//tensorflow_serving/apis:regression_proto_py_pb2",
	    "//tensorflow_serving/apis:predict_proto_py_pb2",
	    "//tensorflow_serving/apis:prediction_service_proto_py_pb2"
	  ]

Python客户端代码可以导出这些库：

{% highlight python %}

from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

{% endhighlight %}

**注意：prediction_service_pb2定义了该service，是必需的。然而，一个典型的client只需要classification_pb2, regression_pb2, 和 predict_pb2三者的其中之一作为请求类型。**

发送一个gRPC请求，接着通过组装一个包含了请求数据的protobuf、并将它传给service stub来完成。注意，protobuf请求如保被创建，可以参考[generated protocol buffer API](https://developers.google.com/protocol-buffers/docs/reference/python-generated)。

{% highlight python %}

from grpc.beta import implementations

channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = classification_pb2.ClassificationRequest()
example = request.input.example_list.examples.add()
example.features.feature['x'].float_list.value.extend(image[0].astype(float))

result = stub.Classify(request, 10.0)  # 10 secs timeout

{% endhighlight %}

在本例中返回的结果是一个ClassificationResponse protobuf。

这只是一个骨架，详见[Tensorflow Serving](https://www.tensorflow.org/deploy/index)。

**注意：ClassificationRequest和RegressionRequest包含了一个tensorflow.serving.Input的protobuf，它会依次包含一列 tensorflow.Example protobuf。相反地，PredictRequest包含了一个关于feature names到由TensorProto编码的values的映射。相应的：当使用Classify 和 Regress APIs时，TensorFlow Serving 会将序列化的tf.Examples feed给graph，因此，你的serving_input_receiver_fn()应包含一个tf.parse_example() Op。当使用通用的Predict API时，Tensorflow Serving会将原始的特征数据feed给graph，因而可以使用serving_input_receiver_fn()直接通过。**

# 6.使用CLI来inspect和execute SavedModel

你可以使用SavedModel Command Line Interface (CLI) 来inspect和execute一个SavedModel。例如，你可以使用CLI来inspect模型的SignatureDef s。CLI允许你快速求证：input tensor的dtype和shape是否与模型相匹配。再者，如果你想测试你的模型，你可以使用CLI来做一些心智检查（sanity check）：通过传递不同格式的样本输入(例如：python表示)，接着获取output。

## 6.1 安装SavedModel CLI

你可以以下面两种方式安装Tensorflow:

- 通过安装一个预先构建的tensorflow binary
- 通过源代码来构建tensorflow

如果你通过一个预先构建的tensorflow binary来安装tensorflow，接着，可以在你的系统目录bin\saved_model_cli下找到已经安装的SavedModel CLI。

如果你想从源代码构建tensorflow，你必须运行以下命令来构建saved_model_cli：

$ bazel build tensorflow/python/tools:saved_model_cli

## 6.2 命令总览

SavedModel CLI支持以下两种命令来操作SavedModel中的一个MetaGraphDef：

- show: 可以展示在一个SavedModel中一个MetaGraphDef上的计算（computation）
- run: 在MetaGraphDef上运行一个computation

## 6.3 show命令

一个SavedModel包含了一或多个MetaGraphDefs，通过它们的tag-sets进行标识。为了为该模型提供服务，你可能想知道在每个模型中分别是哪个类型的SignatureDefs，以及它们的inputs和outputs。show命令让你以层次化顺序检查SavedModel的内容。以下是它的语法：

	usage: saved_model_cli show [-h] --dir DIR [--all] 
				[--tag_set TAG_SET] [--signature_def SIGNATURE_DEF_KEY]
				
例如，以下命令展示了在SavedModel中所有可提供的MetaGraphDef tag-sets:

	$ saved_model_cli show --dir /tmp/saved_model_dir
	The given SavedModel contains the following tag-sets:
	serve
	serve, gpu

以下命令展示了在一个MetaGraphDef中所有可提供的SignatureDef keys：

	$ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve
	The given SavedModel `MetaGraphDef` contains `SignatureDefs` with the
	following keys:
	SignatureDef key: "classify_x2_to_y3"
	SignatureDef key: "classify_x_to_y"
	SignatureDef key: "regress_x2_to_y3"
	SignatureDef key: "regress_x_to_y"
	SignatureDef key: "regress_x_to_y2"
	SignatureDef key: "serving_default"

如果一个MetaGraphDef在一个tag-set取多个tags，你必须指定所有tags，每个tag通过一个：号分割。例如：

	$ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve,gpu
	
为了展示某个指定SignatureDef所有的inputs和outputs TensorInfo，可以将SignatureDef key传给signature_def选项。当你想知道tensor的key/value、dtype和shape时非常有用。例如：

	$ saved_model_cli show --dir \
	/tmp/saved_model_dir --tag_set serve --signature_def serving_default
	
	The given SavedModel SignatureDef contains the following input(s):
	  inputs['x'] tensor_info:
	      dtype: DT_FLOAT
	      shape: (-1, 1)
	      name: x:0
	The given SavedModel SignatureDef contains the following output(s):
	  outputs['y'] tensor_info:
	      dtype: DT_FLOAT
	      shape: (-1, 1)
	      name: y:0
	Method name is: tensorflow/serving/predict

## 6.4 run命令

调用run命令来运行一个graph computation，传递inputs接着会显示outputs。以下是语法：

	usage: saved_model_cli run [-h] --dir DIR --tag_set TAG_SET --signature_def
	                           SIGNATURE_DEF_KEY [--inputs INPUTS]
	                           [--input_exprs INPUT_EXPRS]
	                           [--input_examples INPUT_EXAMPLES] [--outdir OUTDIR]
	                           [--overwrite] [--tf_debug]


run命令提供了以下三种方法来传递inputs给模型：

- --inputs: 允许你传入在文件中的numpy ndarray
- --input_exprs: 允许你传入Python表达式
- --input_examples: 允许传入tf.train.Example

### --inputs

传入文件中的input data，指定--inputs选项，它会采用以下的通用格式：

	--inputs <INPUTS>

其中INPUTS是以下格式：

- input_key=filename
- input_key=filename[<variable_name>]

你可以传入多个inputs。如果你传入多个inputs，使用一个分号来分割每个INPUTS。

saved_model_cli会使用numpy.load来加载filename。filename可以是如下格式：

- .npy
- .npz
- pickle格式

一个.npy文件总是包含一个numpy ndarray。因此，当从一个.npy文件加载时，它的内容将被直接被配给指定的input tensor。如果你在.npy文件中指定了一个variable_name，会忽略variable_name并抛出一个warning。

当从一个.npz (zip)文件加载时，你可以可选地指定一个variable_name来标识在zip文件中的该variable，来加载input tensor key。如果你不指定一个variable_name，SavedModel CLI可以确认在zip文件中只包含一个文件，并为指定的input tensor key来加载它。

当从一个pickle文件加载时，如果没有指定variable_name，在pickle文件中不管是什么，都将被传给指定的input tensor key。否则，SavedModel CLI将假设在pickle文件中存储的一个字典，它的value对应于将被使用的variable_name。

### --input_exprs

为了通过python表达式传入inputs，可以指定--input_eprs选项。当你没有数据文件时这会很有用。。。。示例：

	`<input_key>=[[1],[2],[3]]`

除了Python表达式外，你也可以传入numpy函数。例如：

	`<input_key>=np.ones((32,32,3))`
	
### --input_examples

为了传入tf.train.Example作为输入，可以指定该选项。对于每个input key，它会采用一个dict列表，其中每个dict是一个tf.train.Example实例。dict key是features，values是每个feature的value lists。例如：

	`<input_key>=[{"age":[22,24],"education":["BS","MS"]}]`

### 保存输出

缺省的，SavedModel CLI会将output写到stdout上。如果一个目录通过--outdir选项传入，outputs将被保存成npy文件。

使用--overwrite 覆盖写入已经存在的output files。

### tfdbg

略.

# 7.一个SavedModel目录的结构

当你以SavedModel格式保存一个模型时，TensorFLow会创建一个SavedModel目录，它包含了相应的子目录和文件：

	assets/
	assets.extra/
	variables/
	    variables.data-?????-of-?????
	    variables.index
	saved_model.pb|saved_model.pbtxt

单个SavedModel可以表示多个graphs。在这种情况下，在SavedModel中的所有graph会共享单个checkpoints(variables)和assets的集合。例如，下图展示了在SavedModel上包含了三个MetaGraphDefs，三者共享checkpoints和assets的相同集合：

<img src="https://www.tensorflow.org/images/SavedModel.svg">

每个graph与一个tags的指定集合相关，它会在load或restore操作时允许验证。

# 参考

[using_savedmodel_with_estimators](https://www.tensorflow.org/guide/saved_model#using_savedmodel_with_estimators)
