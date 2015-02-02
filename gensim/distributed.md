---
layout: page
title: 分布式计算 
---
{% include JB/setup %}

#为什么需要分布式计算？

构建一个包含几百万文档的语料的语义表示，并让一直运行？你可以使和一些空闲的机器进行处理？通过将任务分割成多个更小的子任务，并行地传给一些计算节点，分布式计算可以加速计算。

在gensim的上下文中，计算节点通过ip/port进行区分，通信方式为TCP/IP。整个机器的集合称为一个集群。这种分布式十分粗粒度，因此网络允许高延迟。

注意

使用分布式计算的主要原因是，可以运行地更快。在gensim中，大多数时间消耗都是在NumPy中的线性代数低级路由中，而非gensim代码。为NumPy安装一个快速的[BLAS库](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)可以提升性能15倍！因此，在你需要购买额外的计算机前，考虑安装一个快速的、线程化，并在你的特定机型上进行过优化的BLAS库（而非一个通用的二进制库）。可选的方案包括：一些提供商的BLAS库（Intel的MKL, AMD的ACML, OSX的vecLib，Sun的Sunperf...）或者一些开源选择（GotoBLAS, ALTAS）.

如果想查看正在使用的BLAS和LAPACK，在shell中输入：

python -c 'import scipy; scipy.show_config()'

# 先决条件

gensim使用Pyro(Python Remote Objects)在节点间进行通过，version >=4.27。这个库


[英文文档](http://radimrehurek.com/gensim/distributed.html)
