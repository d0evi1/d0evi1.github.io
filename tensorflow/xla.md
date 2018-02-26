---
layout: page
title:  tensorflow XLA
tagline: 
---
{% include JB/setup %}

# XLA总览

**注意：XLA仍在实验阶段，并且在alpha版本。大多数用例并不会在性能上做提升（加速或减小内存使用）。我们这么早放出XLA是为了开源社区能为开发做出贡献，可以创建一个路线图来集成硬件加速。**

XLA（Accelerated Linear Algebra:加速线性代数）是一个为线性代数设计的特定领域编译器，可以优化TensorFlow的计算。结果可以在速度、内存使用上进行加速，以及在服务端和移动平台上可移植。最初，大多数用户不会从XLA上获益很多，但仍欢迎尝试通过JIT编译（just-in-time）或AOT编译（ahead-of-time）来使用XLA。特别鼓励面向新硬件加速的开发者们尝试XLA。

XLA框架仍处试验阶段，并且仍在活跃开发中。特别的，存在的操作语义不大可能会变更，人们希望添加更多的操作来覆盖重要的用例。XLA团队欢迎社区的各种关于缺失功能的feedback和contributions。

# 为什么要构建XLA

使用Tensorflow XLA有以下一些目的：

- 提升执行速度。
- 提升内存利用率。
- 减小mobile footprint
- 得升可移植性。

# XLA是如何工作的？

XLA的输入语言被称为“HLO IR”，或者就称为"HLO（High Level Optimizer）"。HLO的语义在[Operation Semantics](https://www.tensorflow.org/performance/xla/operation_semantics)有描述。将HLO看成是compiler IR很方便。

XLA会利用在HLO中定义的graphs（“computations”），将它们编译成不同架构的机器指令。XLA是模块化的，很容易插入到其它[面向一些著名HW架构](https://www.tensorflow.org/performance/xla/developing_new_backend)的后端。x64 和 ARM64的CPU后端，以及NVIDIA GPU后端，都在TensorFlow源树中。

下图展示了XLA的编译过程：

<img src="https://www.tensorflow.org/images/how-does-xla-work.png">

XLA会伴随着一些优化（optimizations）和分析传递（analysis passes），它们是target-independent的，比如：CSE；target-independent operation fusion，以及buffer analysis来分配计算所需的运行时内存。

在target-independent阶段（steps）后，XLA发送HLO计算给一个backend。该backend会执行进一步的HLO-level的优化，此时会伴随着特定目标的信息，必须牢记。例如，XLA GPU backend可能会执行operation fusion，这对于GPU 编程模型会特别有用，决定着如何将该计算划分成streams。在该阶段（stages），backends可以模式匹配（pattern-match）由优化库调用的特定操作（operations）或组合（combinations）。

下一部是特定目标的代码生成（target-specific code generation）。包含了XLA 的CPU 和 GPU backends, 会使用LLVM来进行low-level IR，优化，以及代码生成。这些backends如有必要会发射LLVM IR，来以一个有效方式表示XLA HLO计算，接着调用LLVM来从LLVM IR触发原生代码（native code）。

GPU backend目前支持通过LLVM NVPTX backend 的NVIDIA GPUs，CPU backend 支持多CPU ISAs.

# 支持平台

XLA当前支持x86-64的JIT编译，以及NVIDIA GPUs，以及x86-64 和ARM的AOT编译。


# 参考

[tensorflow xla](https://www.tensorflow.org/performance/xla/)
