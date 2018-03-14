---
layout: page
title:  tensorflow 架构
tagline: 
---
{% include JB/setup %}

# 介绍

我们设计tensorflow的目的是**进行大规模分布式training和inference，同时能够足够灵活地支持使用新的机器学习模型和系统级优化来支持实验**。

该文档描述的系统架构，可以支持可扩展性和灵活性。本文档假设你已经基本熟悉tensorflow的编程概念，比如：computation graph，operations，sessions。

该文档面向这样的人群：由于当前API功能不支持而想对tensorflow进行扩展的开发者，想优化tensorflow的硬件工程师，致力可扩展和分布式机器学习系统的实现者，以及任何想看tensorflow底层实现的人。读了该文档后，你会理解tensorflow的架构，以便你读取和修改核心tensorflow的代码。

# 1.总览

tensorflow runtime是一个跨平台的库。图1展示了它的总体架构。C API以不同语言将用户级代码与core runtime相分离开。

<img src="https://www.tensorflow.org/images/layers.png">

图一

该文档主要关注以下几层：

- **Client**
	- 将计算定义成一个dataflow graph
	- 使用一个session来初始化graph execution
- **Distributed Master**
	- 从graph中剪枝出一个指定的subgraph，通过Session.run()中参数来定义
	- 将subgraph划分成多个部分，运行在不同的进程和设备上
	- 将图块(graph piece)分布到worker services上
	- 通过worker services初始化图块（graph piece）的执行
- **Worker Services(每个任务一个)**
	- 使用与所使用硬件(CPU, GPU等)相对应的kernel实现调度graph op的执行 
	- 从其它worker services上发送和接收op结果
- **Kernel Implementations**
	- 为单个graph ops执行计算

图2展示了这些组件的交互。"/job:worker/task:0"和"/job:ps/task:0"两者都是带有worker services的任务（tasks）。“PS”表示“parameter server”：这是一个负责存储和更新模型参数的task。其它tasks随着参数优化的进行将更新发送到这些参数上。这种特殊的任务切分方式并不是必需的，但对于分布式训练很常见。

<img src="https://www.tensorflow.org/images/diag1.svg">
图2

注意，Distributed Master和Worker Service只存在于分布式tensorflow中。单进程版本的tensorflow包含了一个特殊的Session实现，它会扮演distributed master的角色，但只会与在本地进程上的devices间相互通信。

下节详细描述了核心的Tensorflow layers，并通过一个示例graph来说明。

# 2.Client

用户编写client Tensorflow程序来构建计算图（computation graph）。该程序即可以直接对单独的ops进行组合，也可以使用像Estimators API这样的通用库来组合神经网络layers和更高级的抽象。tensorflow支持多种client语言，我们优先使用Python和C++，因为google的内部用户对这些语言更熟悉。随着特性变得更稳定，我们通常将它们移植到C++，以便用户可以从其它client语言来访问一个优化版的实现。大多数训练库仍是用python编写的。但C++能支持高效的inference。

client会创建一个session，它会将graph定义作为一个[tf.GraphDef](https://www.tensorflow.org/api_docs/python/tf/GraphDef)的proto buffer发送到distributed master。当client评估在图中一个或多个节点时，evaluation会触发对distributed master的一个调用来初始化计算（computation）。

在图3中，client已经构建了一个graph，将权重(w)应用到一个特征向量(x)上，添加一个bias项(b)，并将结果保存到一个变量(s)上。

<img src="https://www.tensorflow.org/images/graph_client.svg">
图3

代码：[tf.Session](https://www.tensorflow.org/api_docs/python/tf/Session)

# 3.Distributed master

distributed master的作用：

- 将graph进行裁剪以获取**subgraph**，来支持client评估节点
- 为每个特定的设备，将graph进行分区以获取**graph分片（pieces）**
- 将这些分片（pieces）进行缓存，以便他们在后续操作中被复用

由于master可以看到一个step的整体计算，它会应用标准优化，比如：公共子表达式消除（common subexpression elimination）和常量合并（constant folding）。接着会跨任务集合对优化的subgraphs进行协同执行。

<img src="https://www.tensorflow.org/images/graph_master_cln.svg">

图4

图5展示了我们的示例graph的一个可能划分。distributed master将模型参数进行分组，以便将它们一起放置到parameter server上。

<img src="https://www.tensorflow.org/images/graph_split1.svg">
图5

其中图的边（graph edges）按分区进行切割，distributed master会将send和receive节点插入到在distributed tasks间的传递信息中（pass information）。

<img src="https://www.tensorflow.org/images/graph_split2.svg">
图6

distributed master接着将graph分片（pieces）传到distributed tasks上。

<img src="https://www.tensorflow.org/images/graph_workers_cln.svg">

图7

代码：

- [MasterService API definition](https://www.github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/core/protobuf/master_service.proto)
- [Master interface](https://www.github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/core/distributed_runtime/master_interface.h)

# 4.Worker Service

worker servie在每个task中会处理以下事情：

- 处理来自master的请求
- 为组成一个local subgraph的ops调度kernel的执行
- 直接调解在tasks间的通信

为了以较低负载运行large graphs我们优化了worker service。我们当前的实现可以每秒执行上万个subgraphs，它可以允许大量replicas做出快速的、细粒度的training steps。worker service会分派kernels给local devices，并且当可能时以并行方式运行kernels，例如：使用多个CPU cores或者GPU streams。

我们为每个源设备类型(source device)和目标设备类型(target device)的对（pair）指定Send和Recv操作：

- 在本地CPU和GPU设备间的Transfers，使用cudaMemcpyAsync() API来重叠计算（overlap computation）和数据转移（data transfer）
- 在两个本地GPU间的Transfers，使用peer-to-peer DMA来避免通过宿主CPU昂贵的copy操作

对于tasks间的转移，tensorflow使用两种协议，包括：

- 通过TCP的gRPC
- 通过以太网（Converged Ethernet）的RDMA

我们已经准备支持NVIDIA's NCCL库来进行多GPU通信（详见：tf.contrib.nccl）

<img src="https://www.tensorflow.org/images/graph_send_recv.svg">

图8

相关代码：

- WorkerService API definition
- Worker interface
- Remote rendezvous (for Send and Recv implementations)

# 5.Kernel实现

runtime包含了超过200种的标准ops，包含：数学型op（mathematical），数组操作op（array manipulation），控制流op（control flow），状态管理op（state management）等。每种这样的操作都具有为多种设备优化的kernel实现。许多op kernels使用Eigen::Tensor来实现，它会为多核CPUs和GPUs使用c++模板来生成有效的并行代码；然而，我们已经实现了量化库[quantization](https://www.tensorflow.org/performance/quantization)，它会允许在以下环境中（比如：移动设备、或者高吞吐数据中心应用）更快的inference，并且使用[gemmlowp](https://github.com/google/gemmlowp)低精度矩阵库来加速量化计算（quantized computation）。

如果将一个子计算表示成ops的一个组合很难或者很低效时，用户可以注册额外的kernels编写c++实现来提供一个高效的实现。例如，对于一些性能严格的操作（比如：ReLU和Sigmoid激活函数，以及相应的梯度），我们推荐注册你自己的fused kernels。[XLA compiler](https://www.tensorflow.org/performance/xla/index)具有一个自动kernel fusion的实验版的实现。

相关代码

- [OpKernel interface](https://www.github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/core/framework/op_kernel.h)

# 参考

[tensorflow Architecture](https://www.tensorflow.org/extend/architecture/)
