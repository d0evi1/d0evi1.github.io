---
layout: page
title: models.lsimodel - 隐含语义分析 
---
{% include JB/setup %}

python 隐含语义分析LSA（或者LSI）模块。

通过奇异值分解（SVD）的方式进行实现。SVD可以在任何时间添加新值进行更新（实时，增量，有效内丰训练）

该模块实际上包含了许多算法，用来分解大语料，允许构建LSI模块：

- 语料比RAM大：只需要常量内存，对于语料的size来说 (尽管依赖特征集的size)
- 语料是流化的：文档只能按顺序访问，非随机访问
- 语料不能被暂时存储：每个文档只能被查看一次，必须被立即处理（one-pass算法）
- 对于非常大的语料，可以通过使用集群进行分布式运算

English Wikipedia的性能（2G的语料position，在最终的TF-IDF矩阵中，3.2M文档，100k特征，0.5G的非零实体），请求 top 400 LSI因素：

<table>
    <tr>
        <th>算法</th>
        <th>单机串行</th>
        <th>分布式</th>
    </tr>
    <tr>
        <td>one-pass merge算法</td>
        <td>5h 14m</td>
        <td>1h 41m</td>
    </tr>

    <tr>
        <td>multi-pass stochastic算法（2阶迭代）</td>
        <td>5h 39m</td>
        <td>N/A</td>
    </tr>
</table>

serial = Core 2 Duo MacBook Pro 2.53Ghz, 4GB RAM, libVec

distributed = cluster of four logical nodes on three physical machines, each with dual core Xeon 2.0GHz, 4GB RAM, ATLAS


