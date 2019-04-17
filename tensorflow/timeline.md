---
layout: page
title:  tensorflow中的estimators
tagline: 
---
{% include JB/setup %}

# 介绍

虽然tensorflow在很多地方提到过使用timeline进行profiling，但是对于这一块的tutorial是少之又少，不过还好在2016年的[tensorflow isue](https://github.com/tensorflow/tensorflow/issues/1824#issuecomment-244251867)上有相关介绍。这里具体总结一下几个重要的部分。

以下是作者prb12的部分：

1.在未来的一段时间内，不可能有很多时间去写一篇关于timeline和tracing的tutorial

2.chrome:tracing上对应的pid不是真实pid，只是为了

3.tracing机制的设计是为了捕获单个step.

4.在标题下的所有行都是在相同的tensorflow gpu device上被分派的ops。由于多个ops会在host上被并行触发，他们的执行会在时序上重合。一个简单的分箱算法是：将它们分配到多行上，以便在UI上不重叠在一起（注意：这与host threads不是1:1的对应关系）








# 参考

[tensorflow input_fn](https://www.tensorflow.org/get_started/input_fn)
