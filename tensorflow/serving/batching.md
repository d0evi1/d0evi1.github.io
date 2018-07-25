---
layout: page
title:  tensorflow serving batching
tagline: 
---
{% include JB/setup %}

# 介绍

当一个tensorflow模型进行serving时，将单个模型inference请求进行batching对于请求来说相当重要。特别的，batching对于解锁由硬件加速器(例如：GPU)的高吞吐量来说很重要。tensorflow serving存在一个库（library）来对请求（requests）进行batching，以及对这些batches进行调度。该library自身不与GPUs相绑定，可以被用于以下情况：多个小任务处理分组(group)一起协同运行（该文档假设GPUs是为了简化表述）。它提供了一个特别的tensorflow Session API，同时也提供了用于以其它粒度进行batch的底层APIs。

该library当前分隔在两个位置： 

- 1) tensorflow/contrib/batching (core API and implementation)
- 2) tensorflow_serving/batching (higher-level and experimental code)

该library提供了多个可选的类（classes）以供选择。这些选择的原因是，有许多合理的方式来执行batching。**不存在“最佳”的方法，因为不同的用例具有不同的需求**：

- API偏好选择：Tensor API vs. general API; 异步 vs 同步 (synchronous vs. asynchronous)
- 除了GPU外，模型是否有很大的CPU组件
- server是否需要将请求（requests）交错到（interleave）多个模型（或者版本version）中
- 这是用于在线服务（online serving）还是批处理（bulk processing，比哪：map-reduce jobs）？

更进一步，其中一些部署需要高级能力来挤榨出最大性能，其它一些可能只想简单合理的执行。

# 2.Simple Batching

如果你刚接触batching library或者只有基本需求，你可以只关注BatchingSession或者BasicBatchScheduler。

## 2.1 BatchingSession

BatchingSession会添加batching到一个标准的tensorflow::Session中，接着由你使用单个tensor（非batching）的方式调用Session::Run() ，你可以获得你看不见的batching收益。如果你的应用使用tensorflow，可以搭配Session:Run()的synchronous API，该抽象（abstraction）效果不错——请求线程调用Session::Run() 时会阻塞，等待其它调用将它们分组（group）成同一个batch。**为了在synchronous API上达到好的吞吐量（throughput），我们推荐你将客户端线程数设置成batch_size的两倍**。

**BatchingSession可以与其它batch调度器（包括BasicBatchScheduler）一起使用**。它提供了一种方式来将限定每个Session:Run()的调用能阻塞多久。使用BatchingSession的最简单方式是使用CreateRetryingBasicBatchingSession()来创建: 它可以给你返回一个使用一个BasicBatchScheduler底层的tensorflow::Session()对象，也可以处理来自从调度队列溢出的重试请求。你可以提供一些关键参数来管理传给底层BasicBatchScheduler的批请求（batched requests）的调度和执行；下面会有详细介绍。BasicBatchScheduler具有一个限定size的队列；你可以设置参数来管理当队列满时Session::Run()是否会失败；或者以一定的时延进行重试多少次，等等。

最终的配置参数是allowed_batch_sizes。该参数是可选的。如果未设置，那么batch size会是任意在[1, 最大允许size(比如:1024)]间的任何数。取决于你的环境，batch_size很大可能会带来问题。allowed_batch_sizes参数可以让你将batch size限定在一个固定集，比如：128, 256, 512, 1024. BatchingSession会保持这种限制：通过使用空数据（dummy data）补足不合理size的batches来达到下一个合法的size。

## 2.2 BasicBatchScheduler

BasicBatchScheduler是比BatchingSession更底层的一个抽象。它不与tensors/Tensorflow相绑定，这使得它很灵活。对于那些处理均匀请求的servers来说很适合（详见basic_batch_scheduler.h）。

BasicBatchScheduler提供了一个异步API，它会与BatchScheduler共享。该API通过一个BatchTask类进行模板化，封装了关于batch工作的一个unit。使用一个非阻塞Schedule()方法来enqueue一个任务进行处理。一旦一个batch的任务准备好要被处理，会在处理该batch的单个线程上调用一个callback。如何使用该API的一个好示例可以详见batching_session.cc的BatchingSession实现。

# 3.Batching Scheduling参数及Tuning

这些参数管理着batch scheduling：

- max_batch_size：batch允许的最大size。该参数管理着吞吐量/时延间的tradeoff，也可以避免。
- batch_timeout_micros：在执行一个batch之前，要等待的最大时间量（即使它没有达到max_batch_size），用于控制尾部时延（tail latency），详见basic_batch_scheduler.h。
- num_batch_threads：并行度，例如，并发处理batches的最大数目
- max_enqueued_batches：可以进入scheduler队列的最大batches的数目。它被用于限定队列时延，通过不让那些长时间的请求入内，而非构建一个大的缓充（backlog）。

## 3.1 性能调优

batch scheduling参数的最佳值取决于你的模型、系统和环境、以及你的吞吐量和时延要求。可以通过实验来选择合理值。这里有相应的指南。

### 总体指南

首先，试验时你应将max_enqueued_batches设置成无限大（infinity）。接着，设置你的生产环境，可以按如下去做：

- 如果你正执行online serving，取决于将请求路由到server实例中策略，可以设置max_enqueued_batches等于num_batch_threads，以便能在当特定的server正忙时能最小化排队时延（queueing delay）。
- 如果是批处理任务，可将max_enqueued_batches设置成一个大值，但不能太高，以避免out-of-memory crashes。

第二，出于系统架构的原因，你需要将限制batch size可能集合（比如：100, 200 或者 400,而非在1到400间的任意值）：如果你正使用BatchingSession，你可以设置allowed_batch_sizes参数。否则，你可以通过使用空数据来补齐batches来安排你的callback。

### CPU-only:One 方法

如果你的系统是CPU-only（无GPU），可以考虑使用以下值：

- num_batch_threads：等于CPU core的数目；
- max_batch_size：设置成无穷大(infinity)
- batch_timeout_micros：0
- batch_timeout_micros：试验值在1-10ms （1000-10000 us）范围，记住，0也可以是最优值

#### GPU:One 方法

如果你的模型使用一个GPU设备来完成inference工作的一部分，考虑以下的方法：

- 1.将num_batch_threads设置成CPU core的数目
- 2.将batch_timeout_micros临时设置成无穷大，然后调节max_batch_size来在吞吐量和平均时延的平衡上达到你想要的值。
- 3.对于online serving，调节batch_timeout_micros来控制尾部时延（tail latency）。batches通常会填满max_batch_size，但对于进来的请求偶尔也会填不满，为了避免引了时延毛刺，对于在队列中未填满的batch来说需要去做处理。batch_timeout_micros通常只有几毫秒（ms），具体取决于上下文和目标。0也是一个可考虑的值，对于某些workloads可能效果很好。（对于批处理任务，选择一个较大值，可能是几秒，来确保好的吞吐量，又不至于对最后未满的batch等太久）

# 4.多个模型，多个版本或者多个子任务的Servers

一些server实例会同时服务多种请求类型（比如：多种模型、一个模型的多种版本）。另一情况下，单个请求可以分解成子请求，涉及到多个不同的servables（比如：一个推荐系统可能具有一个triggering model，它会决定是否准备一个推荐，根据选择实际推荐的一个模型）。第三种情况是，分桶序列模型请求会对相同的长度的请求进行batch，最小化padding。

总的来说，对于每种请求或子任务，如果它们在底层计算资源上共享的话，各自使用一个独立的batch scheduler效果并不会好——每个scheduler会运行在它们自己的线程上，该线程会与其它线程竞争访问资源。较好的方法是，使用单个scheduler时与单个线程池一起搭配使用，可以认出多个不同类型的任务，避免同种任务的batches与其它任务的batches相交错。

SharedBatchScheduler会处理这种情况。它表示一个队列的抽象，接受请求来调度某一种特定的任务。每个batch都包含了一种类型的任务（来自某一队列）。该scheduler会通过不同类型的batches相交错来确保公平性。

该队列实现了BatchScheduler API，因此他们可以在任何使用简单scheduler（非共享）的地方被使用，包含与BatchingSession的情况。队列可以一直进行添加和移除，这对于新模型版本的迁移（客户端会指定一个特定版本）的情况很有用：clients可以学到最新的版本，server必须处理两种版本的请求，SharedBatchScheduler会处理两种类型请求batches的交叉。

# 5.混合CPU/GPU/IO Workloads

除了主要的GPU工作量外，一些模型会执行重要的CPU工作。而核心矩阵操作则在GPU上运行，次要操作则发生在CPU上，例如：embedding lookup，vocabulary lookup，quantization/dequantization. 具体取决于GPU如何被管理，将CPU和GPU steps的整个序列进行batching成一个unit可能会减少GPU的利用率。

非GPU（Non-GPU）的预处理和后处理可以在**请求线程(request threads)**中被执行，而batch scheduler只用于GPU部分的工作。

另外，非GPU工作可以在batch线程（batch threads）上完成，在batch scheduler调用的callback中完成。在一个batch完全形成之前，为了允许该callback来执行non- batched工作，你可以使用StreamingBatchScheduler。这样的设计是为了让servers非常精准的控制时延，并能很好地控制pipeline的每个stage。

StreamingBatchScheduler将拒绝一个任务，如果该scheduler当前没有能力（capacity）处理它。如果你想自动重试被拒绝的任务，你可以在batch scheduler之上叠一个BatchSchedulerRetrier。另外存在一个很便利的函数用来创建一个streaming scheduler与retrier的搭配是：“CreateRetryingStreamingBatchScheduler()”

当将模型inference logic分割成多个不同的阶段（phases）来优化时延或利用率时，记住对于一个给定的请求，每个阶段都应使用相同版本的该模型。确保该特征的一个好方式是，协调在每个阶段(phases)上跨线程使用哪个ServableHandle对象。

最后，对于inference中I/O敏感的阶段，比如：查询磁盘或远程servers，可能会受益于batching来掩盖他们的时延。你可以使用两个batch scheduler实例：一个用于batch这些lookups，另一个用于batch那些GPU工作量。

# 参考
[batching](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/batching/README.md)
