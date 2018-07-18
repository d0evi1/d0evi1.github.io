---
layout: page
title:  tensorflow 构建标准ModelServer
tagline: 
---
{% include JB/setup %}


该tutorial会展示你如何使用tensorflow serving组件来构建标准的tensorflow ModelServer，能动态发现模型的新版本并提供服务。

该totorial会使用与serving_basic相同的示例。

- python文件：[mnist_saved_model.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_saved_model.py)，用于训练和导出多个版本的模型。
- c++文件：[main.cc](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/model_servers/main.cc)标准的tensorflow ModelServer，用于发现新导出模型并运行一个gRPC服务来提供服务。

该tutorial有以下几部分：

- 1.训练和导出一个模型
- 2.使用ServerCore来管理模型版本
- 3.使用SessionBundleSourceAdapterConfig来配置batching
- 4.使用ServerCore来提供服务
- 5.运行和测试服务

# 1.训练和导出模型

先删除已经存在的导出目录：

	$>rm -rf /tmp/mnist_model

训练100次迭代并导出第1个版本的模型：

	$>bazel build -c opt //tensorflow_serving/example:mnist_saved_model
	$>bazel-bin/tensorflow_serving/example/mnist_saved_model --training_iteration=100 --model_version=1 /tmp/mnist_model

训练2000次并导出第二个版本的模型：

	$>bazel-bin/tensorflow_serving/example/mnist_saved_model --training_iteration=2000 --model_version=2 /tmp/mnist_model

正如在mnist_saved_model.py所见，训练和导出与serving_basic教程相同。为了演示的方便，将第一版本称为v1,第二个版本称为v2——我们期望后者能达到更好的分类accuracy。你可以看到在mnist_model目录下运行的每次训练的数据。

	$>ls /tmp/mnist_model
	1  2

# 2.ServerCore

现在，假设v1和v2是在runtime时动态生成的，比如：新算法试验的进行，或者模型使用新的数据集进行训练。在生产环境下，你可能希望构建一个server来支持渐近式上线（gradual rollout），在其中，v2可以被发现，加载，实验，监控，或者回滚。另外，你可以希望在上线v2时撤下v1。TensorFlow Serving支持两种选项——其中，一种对于在切换期继续提供服务，另一种可以最小化资源使用（比如：RAM）。

Tensorflow Serving的Manager就是做这个事的。它会处理tensorflow模型的完整生命周期：加载(loading)、服务(serving)以及版本更迭时的上传（uploading）。在该教程中，你将在Tensorflow Serving的ServerCore上构建你的server。它内部会封装一个AspiredVersionsManager。

{% highlight c++ %}

int main(int argc, char** argv) {
  ...

  ServerCore::Options options;
  options.model_server_config = model_server_config;
  options.servable_state_monitor_creator = &CreateServableStateMonitor;
  options.custom_model_config_loader = &LoadCustomModelConfig;

  ::google::protobuf::Any source_adapter_config;
  SavedModelBundleSourceAdapterConfig
      saved_model_bundle_source_adapter_config;
  source_adapter_config.PackFrom(saved_model_bundle_source_adapter_config);
  (*(*options.platform_config_map.mutable_platform_configs())
      [kTensorFlowModelPlatform].mutable_source_adapter_config()) =
      source_adapter_config;

  std::unique_ptr<ServerCore> core;
  TF_CHECK_OK(ServerCore::Create(options, &core));
  RunServer(port, std::move(core));

  return 0;
}

{% endhighlight %}

ServerCore::Create() 会使用一个ServerCore::Options参数。这里是一些常用的选项：

- ModelServerConfig：指定要被加载的模型。
- PlatformConfigMap：将平台名（比如：tensorflow）映射到PlatformConfig，用于创建SourceAdapter。SourceAdapter会适配StoragePath(模型版本被发现的路径)给模型Loader（从存储路径上加载模型版本，并提供状态转换接口给Manager）。如果PlatformConfig包含了SavedModelBundleSourceAdapterConfig, 将会创建一个SavedModelBundleSourceAdapter。

SavedModelBundle是tensorflow的一个关键组件。它表示了一个tensorflow模型，从一个给定路径上加载，并提供了与Tensorflow相同的Session::Run接口来运行inference。SavedModelBundleSourceAdapter会适配存储路径给Loader<SavedModelBundle>，以便模型的生命周期可以通过Manager被管理。请注意，SavedModelBundle是SessionBundle(不再使用)的后继。推荐用户使用SavedModelBundle作为SessionBundle的支持。

ServerCore内部会做以下事情：

- 实例化一个FileSystemStoragePathSource，它会监控在model_config_list声明的模型导出路径。
- 使用PlatformConfigMap实例化一个SourceAdapter，模型平台在model_config_list中配置，并与FileSystemStoragePathSource相连。这种方式下，一旦一个新的模型版本在导出路径下被发现，SavedModelBundleSourceAdapter会适配给一个Loader<SavedModelBundle>。
- 实例化一个Manager的特定实现AspiredVersionsManager，它管理着所有由SavedModelBundleSourceAdapter创建的Loader实例。ServerCore导出了Manager接口，通过代理该调用给AspiredVersionsManager。

当一个新版本可用时，该AspiredVersionsManager会加载新版本，伴随着该动作也会卸载老版本。如果你想启动定制化，推荐理解组件的内部创建机制，以及如何配置它们。

值得一提是，tensorflow的设计非常灵活和可扩展。你可以构建许多插件来定制系统行为，采用核心组件：ServerCore和AspiredVersionsManager。例如，你可以构建一个数据源插件，它会监控云存储，或者你可以构建一个版本策略插件，它可以以不同方式做版本切换——事实上，你也可以构建模型插件来服务非tensorflow模型。这些主题超出了该教程的范围。

# 3.Batching

另一个我们希望在生产环境中具有的典型server特性是Batching。当inference请求以大batches方式运行时，现代硬件加速（GPU等）用于机器学习inference通常能达到最佳的计算效率。

当创建SavedModelBundleSourceAdapter时，Batching可以通过提供合适的SessionBundleConfig开启。在这种情况下，我们设置了BatchingParameters，使用了许多缺省值。Batching可以通过设置定制的timeout、batch_size进行fine-tuned。详细的，可以参考BatchingParameters。

{% highlight python %}

SessionBundleConfig session_bundle_config;
// Batching config
if (enable_batching) {
  BatchingParameters* batching_parameters =
      session_bundle_config.mutable_batching_parameters();
  batching_parameters->mutable_thread_pool_name()->set_value(
      "model_server_batch_threads");
}
*saved_model_bundle_source_adapter_config.mutable_legacy_config() =
    session_bundle_config;

{% endhighlight %}

达到full batch时，inference请求会在内部被合并成单个大的request (tensor)，然后调用tensorflow::Session::Run()（它其中在GPU的实际效果增益来自于它）。

# 与Manager一起提供Serve

如上所述，tensorflow serving Manager被设计成一个通用组件，可以处理loading、serving、unloading以及版本切换。它的API构建围绕以下几点：

- Servable：Servable是任何透明对象，可以被用于服务客户端请求。一个servable的size和granularity是很灵活的，。。。
- Servable Version：Servables是版本控制的，Manager可以管理servable的一或多个version。
- Servable Stream：一个servable stream是一个servable的版本序列，会随版本号递增
- Model：是一个由一或多个servable表示的机器学习模型。servables示例有：
	- tensorflow session或者其wrapper，比如SavedModelBundle
	- 其它类型的机器学习模型
	- Vocabulary lookup tables
	- Embedding lookup tables

一个复合模型（composite model）可以由多个独立的servables进行表示，或者单个composite servable。一个servable可对应于一个Model的一部分，例如，一个跨多个Manager实例进行共享的超大lookup table。

本教程中：

- Tensorflow模型通过一种servable进行表示：SavedModelBundle。SavedModelBundle内部由一个tensorflow::Session、以及与之成对的一些元数据（比如：哪个graph被加载进session，以及如何在inference时运行它）组成。
- 存在一个文件系统目录，它包含了tensorflow导出的一个stream，在它们自己的子目录（名字为版本号）上的每个stream。外部目录可以被认为是模型的servable stream的序列化表示。每次导出对应于要加载的一个servables。
- AspiredVersionsManager会监控导出的stream，动态管理所有SavedModelBundle servables的生命周期。

TensorflowPredictImpl::Predict接着会：

- 从manager中请求SavedModelBundle（通过ServerCore）
- 使用generic signatures来将在PredictRequest中的逻辑tensor名映射到实际tensor名上，并将值绑定到tensors上。
- 运行inference

# 测试和运行Server

将第一版本的export拷贝到监控目录，并启动server。

	$>mkdir /tmp/monitored
	$>cp -r /tmp/mnist_model/1 /tmp/monitored
	$>bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
	$>bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --enable_batching --port=9000 --model_name=mnist --model_base_path=/tmp/monitored

该server会每秒触发日志：“Aspiring version for servable ..”, 这意味着它发现了export，然后它正跟踪它后续的生存期。

使用参数--concurrency=10运行测试。它会发送并发请求给该server，接着触发你的batching逻辑：

	$>bazel build -c opt //tensorflow_serving/example:mnist_client
	$>bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000 --concurrency=10
	...
	Inference error rate: 13.1%

接着copy第二个export版本到监控目录下，重新运行测试：

	$>cp -r /tmp/mnist_model/2 /tmp/monitored
	$>bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000 --concurrency=10
	...
	Inference error rate: 9.5%

这证实了你的server可以自动发现新版本并使用它进行serving！


# 参考

[https://www.tensorflow.org/serving/serving_advanced](https://www.tensorflow.org/serving/serving_advanced)
