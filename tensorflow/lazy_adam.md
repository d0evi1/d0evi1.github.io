---
layout: page
title:  tensorflow LazyAdam
tagline: 
---
{% include JB/setup %}

# LazyAdam

它是Adam optimizer的变种，可以更有效地处理稀疏更新（sparse updates）。

原始的Adam算法为每个训练变量（tranable variable）维护着两个移动平均累加器；该累加器会在每个step上被更新。该class为稀疏变量（sparse variables）的梯度更新提供了lazier handling机制。它只会为出现在当前batch中的稀疏变量（sparce variable indices）更新移动平均累积，而非为所有indices更新累积。对比原始的Adam optimizer，它可以为一些应用在模型训练吞吐上提供大的提升。然而，它与原始Adam算法有一些不同的语义，可能会导致不同的期望结果（empirical results）。

注意，当前不支持amsgrad，该参数只能为False。

https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/optimizers/lazy_adam.py#L49-L92




# 参考

[tensorflow xla](https://www.tensorflow.org/performance/xla/)
