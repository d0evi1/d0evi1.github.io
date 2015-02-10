---
layout: page
title: 分布式计算 
---
{% include JB/setup %}

# 1.为什么需要分布式计算？

构建一个包含几百万文档的语料的语义表示，并让一直运行？你可以使和一些空闲的机器进行处理？通过将任务分割成多个更小的子任务，并行地传给一些计算节点，分布式计算可以加速计算。

在gensim的上下文中，计算节点通过ip/port进行区分，通信方式为TCP/IP。整个机器的集合称为一个集群。这种分布式十分粗粒度，因此网络允许高延迟。

注意

使用分布式计算的主要原因是，可以运行地更快。在gensim中，大多数时间消耗都是在NumPy中的线性代数低级路由中，而非gensim代码。为NumPy安装一个快速的[BLAS库](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)可以提升性能15倍！因此，在你需要购买额外的计算机前，考虑安装一个快速的、线程化，并在你的特定机型上进行过优化的BLAS库（而非一个通用的二进制库）。可选的方案包括：一些提供商的BLAS库（Intel的MKL, AMD的ACML, OSX的vecLib，Sun的Sunperf...）或者一些开源选择（GotoBLAS, ALTAS）.

如果想查看正在使用的BLAS和LAPACK，在shell中输入：

python -c 'import scipy; scipy.show_config()'

# 2.先决条件

gensim使用Pyro(Python Remote Objects)在节点间进行通过，version >=4.27。这个库通过底层socket以及RPC来通讯。Pyro是一个纯python库，因此，安装很简单，只需要将所有*.py文件拷到python的import path中即可：

    sudo easy_install Pyro4

你不需要安装Pyro来运行gensim，但是如果你不安装，那么，你就不能使用分布式特性。（所有的操作都将按序进行）

# 3.核心概念

gensim提供的API很简单。你在代码中不必做任何更改，即可以运行在集群上!

你只需要在每个集群节点上运行一个worker脚本，来启动计算。运行脚本将告诉gensim，它将节点做为slave节点来代理任务运行。在初始化时，gensim中的算法将进行查找，并将所有的worker节点做为slave。

## 3.1 节点（Node）

一个逻辑工作单元。可以认为是一台物理机器，但是你可以在一台机器上运行多个worker，这样就成了多个逻辑节点。

## 3.2 集群（Cluster）

一些节点通过TCP/IP通信。目前，使用网络广播用于发现和连接所有通信节点，因此节点必须使用相同的广播域。

## 3.3 Worker

每个节点上创建的一个处理进程。如果你想从你的集群中移除一个节点，可以简单地杀死它的worker进程。

## 3.4 Dispatcher

dispatcher主要负责所有的计算、排队、分派任务给worker。计算从不和worker节点直接通信，只能通过dispatcher。与worker不同的是，在集群中一次只有一个激活的dispatcher。

# 提供的分布式算法

- [分布式LSA](http://d0evi1.github.io/gensim/dis_lsi)
- [分布式LDA](http://d0evi1.github.io/gensim/dis_lda)

[英文文档](http://radimrehurek.com/gensim/distributed.html)
