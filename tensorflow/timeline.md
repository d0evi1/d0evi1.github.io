---
layout: page
title:  tf timeline介绍
tagline: 
---
{% include JB/setup %}

# 介绍

虽然tensorflow在很多地方提到过使用timeline进行profiling，但是对于这一块的tutorial是少之又少，不过还好在2016年的[tensorflow isue](https://github.com/tensorflow/tensorflow/issues/1824#issuecomment-244251867)上有相关介绍。

在runtime中集成了一个基础的CUPTI GPU tracer。（注：CUPTI: the CUDA Performance Tool Interface）你可以运行一个step，并开启tracing，它会记录执行的ops以及加载的GPU kernels。这里是一个示例：

{% highlight python %}

run_metadata = tf.RunMetadata()
_, l, lr, predictions = sess.run(
            [optimizer, loss, learning_rate, train_prediction],
            feed_dict=feed_dict,
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_metadata)

{% endhighlight %}

在step完成后，run_metadata会包含一个StepStats protobuf，它包含了许多时序信息(timing information)，以tensorflow device分组的形式存在。CUPTI GPU tracing会以一些名为“/gpu:0/stream:56”/"/gpu:0/memcpy"这样的额外设备的形式而出现。


注意：为了获取GPU tracing，你必须保证libcupti.so在你的LD_LIBRARY_PATH中。通常可以在/usr/local/cuda/extras/lib64找到。

使用这种信息的最简单方式是，以如下形式加载该stats到一个"Timeline"中：

{% highlight python %}

from tensorflow.python.client import timeline
trace = timeline.Timeline(step_stats=run_metadata.step_stats)

{% endhighlight %}

Timeline类可以被用于以Chrome Tracing的格式生成一个JSON trace文件:

{% highlight python %}

trace_file = open('timeline.ctf.json', 'w')
trace_file.write(trace.generate_chrome_trace_format())

{% endhighlight %}

为了观看trace，可以在chrome浏览器中导航到'chrome://tracing'，然后使用“load”按钮来加载该文件。

关于分布式训练的trace，目前还未实现【2016年】，但issue中有讨论。

# 2. show_memory=True

GPU tracing只能使用FULL_TRACE才行。然而该标记会影响在RunMetadata.StepStats protobuf中的东西。chrome trace file的大小通过StepStats中的信息来决定。例如，如果想添加在所有内部tensors上的shape/size信息，可以添加show_memory=True，这会使JSON输出文件更大。

# 3.另一点问题
	
来自rizar的提问：


thank you for your explanations. The reason why I thought that this information could be there, is because when I generate a timeline in the Chrome trace format and load it in the browsers, for some of the tensors I can see a span in it (see a screenshot below). I tried to understand what this span means and made a wrong hypothesis that it may correspond to the period when the tensor was kept in memory. But from what you are saying it does not seem to be the case. That said, I am still curious: what do the spans in the timeline stand for, why do some tensors have a span associated with them, while other just a creation moment?

show_memory模式下，有些tensors可以看到一个span，是否对应该tensor在memory中的period？那么这些在timeline中的spans代表了啥？为什么一些tensors有一个span，而另一些只有一个创建点？

prb12的解答：

由tf.Timeline生成的tensor(memory) views通常并不能很好地表示它的进度，这些选项默认是关闭的。

NodeExecStats proto实际上没有提供足够的信息来准确跟踪tensor的依赖。为了这样做，你实际需要知道完整的执行数据流图(dataflow graph)——这与input GraphDef并不相同，因为placer和optimizer可以做出更实际的变更。

当tf.Timeline被写入时，通过解析NodeExecStats的timeline_label这一field，可以获得一个关于graph的合理近似，来构建一些dataflow依赖。这在时序上会越来越不可靠(robust)。

然而，通过RunMetadata proto的partition_graphs字段来可编程地检索optimized graph是可行的，这可以提供关于tensor lifetime的一种准确思想。关于tf.Timeline的tensor lifetime和memory views，可以使用这些GraphDefs（如果提供的话）来重写。（然而，目前还没时间去做这个事【2017年】）

# 4.show_dataflow=True

	generate_chrome_trace_format(self, show_dataflow=True, show_memory=False)

上面说的dataflow，可以使用该选项，默认是提供的。

# 5.

这里具体总结一下几个重要的部分。

以下是作者prb12的部分：

1.在未来的一段时间内，不可能有很多时间去写一篇关于timeline和tracing的tutorial

2.chrome:tracing上对应的pid不是真实pid，只是为了

3.tracing机制的设计是为了捕获单个step.

4.在标题下的所有行都是在相同的tensorflow gpu device上被分派的ops。由于多个ops会在host上被并行触发，他们的执行会在时序上重合。一个简单的分箱算法是：将它们分配到多行上，以便在UI上不重叠在一起（注意：这与host threads不是1:1的对应关系）

# 6.影响性能的Sparse操作

[issue 8321](https://github.com/tensorflow/tensorflow/issues/8321)

[issue 5408](https://github.com/tensorflow/tensorflow/issues/5408)

TF的SparseToDense操作只有CPU kernel。

# 参考

[tensorflow input_fn](https://www.tensorflow.org/get_started/input_fn)
