---
layout: page
title:  tensorflow serving架构
tagline: 
---
{% include JB/setup %}

# 介绍

TensorFlow Serving是一个灵活的、高性能的机器学习模型serving系统，专为生产环境设计。Tensorflow Serving使得部署新算法和新实验更加简单，同时又能保持相同的服务器架构和APIs。Tensorflow Serving提供了开箱即用的tensorflow模型集成，并可以很容易地扩展到服务其它类型的模型。

# 1.核心概念

为了理解tensorflow serving的架构，你需要理解以下概念：

## 1.1 Servables

Servables是在Tensorflow Serving中的核心抽象。**Servables是客户端（clients）用于执行计算（例如：一次lookup或inference）的底层对象**。

一个Servable的大小（size）和粒度（granularity）是很灵活的。单个Servable必须包含任何东西：可以是某个模型的lookup table的某个分片(shard)，也可以是inference models的一个tuple。Servable可以是任何类型和接口，这样可以保证灵活性和将来的改进：

- streaming结果
- experimental APIs
- 异步模式的操作

Servables不会管理它们的生命周期（lifecycle）。

典型的servables包含以下：

- 一个TensorFlow SavedModelBundle (tensorflow::Session)
- 一个用于embedding或vocabulary查询的lookup table

## 1.1.1 Servable Versions

在单个服务实例的生命周期中，TensorFlow Serving可以处理一或多个版本（versions）的servable。这使得在这段时间内可以加载新算法配置、权重、其它数据。Versions使得超过一个以上的version被并行加载，支持渐近回滚（gradual rollout）和试验（experimentation）。在服务时，对于特定的模型，clients可以请求最新版本或一个特定的版本id。

## 1.1.2 Servable Streams

一个Servable stream是关于一个servable的versions序列，通过递增的版本号进行排序。

## 1.2 Models

TensorFlow Serving将一个模型（Model）表示成一或多个servables。**一个机器学习模型（Model）可以包含一或多个算法（包括学到的权重）、lookup tables 或者 embedding tables**。

你可以将一个**复合模型（composite model）**表示成以下之一：

- 多个独立的servables
- 单个composite servable

一个servable也可以对应一个模型的一部分。例如，一个大的lookup table可以跨多个tensorflow Serving实例共享。

## 1.2 Loaders

**Loaders管理着一个servable的生命周期**。Loader API允许公共设施独立于特定的学习算法，数据或者产品用例。特别的，Loaders会使用标准化API来**加载或卸载一个servable**。

## 1.3 Sources

**Sources是插件模块，它可以查找和提供servables**。每个Source提供了零或多个servable streams。对于每个servable stream，一个Source会为每个version提供一个Loader实例，使得可以被加载。（一个Source实际上会使用零或多个SourceAdapters链接在一起，链（chain）上的最后一个item会触发（emits）该Loaders。

对于Sources的TensorFlow Serving的接口，可以从特定存储系统中查找servables。Tensorflow Serving包括了公共索引Source的实现。例如，Source可以访问类似RPC的机制，可以poll一个文件系统。

Sources可以维持跨多个servables或versions进行共享的状态（state）。这对于使用delta(diff)来在versions间进行更新的servables很有用。

## 1.4 Aspired Versions

**Aspired versions表示Servable Versions的集合，可以被加载和ready**。Sources会与该servable versions集合进行通讯，一次一个servable stream。当一个Source给出一个aspired versions的新列表到Manager时，它会为该servable stream取代之前的列表。该Manager会unload任何之前已加载的versions，使它们不再出现在该列表中。

## 1.5 Managers

**Managers处理Servable的完整生命周期**，包括：

- loading Servalbes
- serving Servables
- unloading Servables

Manager会监听Source，并跟踪所有的versions。Manager会尝试满足Sources的请求，但如果需要的resources不提供的话，也可以拒绝加载一个aspired version。Managers也可以延期一个“unload”操作。例如，一个Manager可以等待unload直到一个更新的version完成loading，基于一个策略(Policy)来保证在所有时间内至少一个version被加载。

Tensorflow Serving Manager提供了一个单一的接口——GetServableHandle()——给clients来访问已加载的servable实例。

## 1.6 Core

TensorFlow Serving Core（通过标准Tensorflow Serving API）管理着以下的servables：

- lifecycle
- metrics

TensorFlow Serving Core将servables和loaders看成是透明的对象。


# 2.一个Servable的生命周期

<img src="https://www.tensorflow.org/serving/images/serving_architecture.svg">

通俗的说：

- 1.Sources会为Servable Versions创建Loaders。
- 2.Loaders会作为Aspired Versions被发送给Manager，manager会加载和提供服务给客户端请求。

更详细的：

- 1.一个Source插件会为一个特定的version创建一个Loader。该Loaders包含了在加载Servable时所需的任何元数据。
- 2.Source会使用一个callback来通知该Aspired Version的Manager。
- 3.该Manager应用该配置过的Version Policy来决定下一个要采用的动作，它会被unload一个之前已经加载过的version，或者加载一个新的version。
- 4.如果该Manager决定它是否安全，它会给Loader所需的资源，并告诉该Loader来加载新的version。
- 5.Clients会告知Manager，即可显式指定version，或者只请求最新的version。该Manager会为该Servable返回一个handle。

例如，一个Source表示一个TensorFlow graph，频繁更新模型权重。该weights存储在磁盘中的一个文件中。

- 1.Source决定着一个关于模型权重的新version。它会创建一个Loader，并包含着一个指针指向在磁盘中的模型数据。
- 2.Source会通知该Aspired Version的Dynamic Manager
- 3.该Dynamic Manager会应用该Version Policy，并决定加载新的version
- 4.Dynamic Manager会告诉Loaders，它有足够的内存。Loader会使用新的weights来实例化Tensorflow graph。
- 5.一个client会向最新version的模型请求一个handle，接着Dynamic Manager返回一个handle给Servables的新version。

# 3.Extensibility

Tensorflow serving提供了一些扩展点，使你可以添加新功能。

## 3.1 Version Policy

Version Policy可以指定version序列，在单个servable stream中加载或卸载。

tensorflow serving包含了两个策略（policy）来适应大多数已知用例。分别是：Availability Preserving Policy(避免没有version加载的情况；在卸载一个老version前通常会加载一个新的version)， Resource Preserving Policy（避免两个version同时被加载，这需要两倍的资源；会在加载一个新version前卸载老version）。对于tensorflow serving的简单用法，一个模型的serving能力很重要，资源消耗要低，Availability Preserving Policy会确保在新version加载前卸载老version。对于TensorFlow Serving的复杂用法，例如跨多个server实例管理version，Resource Preserving Policy需要最少的资源（对于加载新version无需额外buffer）

## 3.2 Source

新的资源（Source）可以支持新的文件系统、云供应商、算法后端。Tensorflow Serving提供了一些公共构建块来很方便地创建新的资源。例如，Tensorflow Serving包含了围绕一个简单资源的polling行为的封装工具类。对于指定的算法和数据宿主servables，Sources与Loaders更紧密相关。

## 3.3 Loaders

Loaders是用于添加算法和数据后端的扩展点。Tensorflow就是这样的一个算法后端。例如，你实现了一个新的Loader来进行加载、访问、卸载一个新的servable类型的机器学习模型实例。我们会为lookup tables和额外的算法创建Loaders。

## 3.4 Batcher

会将多个请求打包（Batching）成单个请求，可以极大减小执行inference的开销，特别是像GPU这样的硬件加速存在的时候。Tensorflow Serving包含了一个请求打包组件（request batching widget），使得客户端（clients）可以很容易地将特定类型的inferences跨请求进行打包成一个batch，使得算法系统可以更高效地处理。详见[Batching Guide](/tensorflow/serving/batching)。



# 参考

[tensorflow serving架构](https://www.tensorflow.org/serving/architecture_overview)
