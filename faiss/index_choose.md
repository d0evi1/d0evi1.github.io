---
layout: page
title:  faiss如何选择一个index？
tagline: 
---
{% include JB/setup %}

# 1.介绍

选择一个index并非显而易见，因此下面有些本质问题，可以帮助你选择index。主要应用于L2 distances。我们会说明：

- 对于每个问题的index_factory string
- 如果存在参数，我们会将它们表示成相应的ParameterSpace参数

# 2.执行少量searches的情况

如果你打算只执行少量的searches（比如：1000-10000），index的building time不会被分摊到search time上。那么，直接计算是最高效的选择。

这可以通过一个“Flat” index完成。如果整个dataset不能fit进RAM，你可以一个接一个地构建small indexes，并将search结果进行组合。

当query vectors的数目有限时，最好的indexing方法是：无需索引，直接使用brute-force搜索。当dataset vector不能fit进RAM or GPU RAM时，这会有问题。在这种case中，对database vectors xb以切片方式进行searh xq vectors是高效的。为了维护top-k最近邻，可以使用一个ResultHeap结构：

[https://github.com/facebookresearch/faiss/blob/master/benchs/bench_all_ivf/datasets.py#L75](https://github.com/facebookresearch/faiss/blob/master/benchs/bench_all_ivf/datasets.py#L75)

# 3.是否需要exact结果？

那么，选择“Flat”！

可以保证exact结果的index只有IndexFlatL2或IndexFlatIP。它为其它indexes提供了baseline。它不会压缩vectors，但不会在它们之上增加开销。它不支持添加ids（add_with_ids），只能顺序添加，因此如果你想使用add_with_ids，需要使用“IDMap,Flat”. flat index不需要训练，没有参数。

支持GPU: yes

# 4.内存是否是个关注点？

记住，所有Faiss indexes是存储在RAM中的。以下会考虑，是否需要exact结果，RAM是限制因素，在内存限制下我们可以对precision-speed tradeoff进行优化。

## a.如果内存不是问题，使用"HNSWx"

如果你具有大量RAM，或者dataset很小，HNSW是最优选择，它是个非常快和精准的index。$$4 <= x <= 64$$是每个vector的links数目，越高表示越精准但也会使用更多内存。speed-accuracy tradeoff通过efSearch参数设置。内存使用量是每个vector占用$$(d*4 + x*2*4)$$ bytes。

HNSW不只支持顺序adds（非add_with_ids），因此，不需要前缀IDMap。HNSW不需要training，也不支持从index中移除vector。

支持GPU：no

## b.如果可能是个问题，那么"...,Flat"

"..."意味着dataset在之前必须执行一个聚类(clustering)。在clustering之后，"Flat"会将vectors组织成"buckets"，因此它不会压缩它们，storage size与原始dataset的size相同。在speed和accuracy间的tradeoff通过nprobe参数进行设置。

是否支持GPU：yes

## c.如果相当重要，那么"PCARx,...,SQ8"

如果存储整个vector太昂贵，这会执行两个操作：

- 执行一个维度x的PCA来降维
- 每个vector component上执行一个scalar quantization到1字节(byte)

接着，总存储是：每个vector为x bytes。

SQ4和SQ6也可以支持（每个vector component为4 or 6 bits）

是否支持GPU: yes（除了SQ6外）

## d.如果非常重要，那么"OPQx_y,...,PQx"

PQx会使用一个product quantizer来对vectors进行压缩，输出x字节的codes。x通常<=64, 对于更大的codes，SQ通常更精准、更快。OPQ是一个对vectors的线性变换，使得他们更容易压缩。y是一个维度：

- y 是x的倍数 （必需）
- y<=d，d是input vectors的维度（更好）
-  y<=4\*x (更好)

# 5.数据集有多大？

该问题被用于填充上面提到（...）的聚类选择(clustering options)。dataset会被聚类成buckets，在search时，通常只有一部分buckets会被访问（nprobe个buckets）。该聚类(clustering)会在dataset vectors的一个有代表性的抽样上执行，通常是该databset的一个抽样。我们将为该抽样说明最优的size。

## a.如果<1M vectors："...,IVFx,..."

当x介于4\*sqrt(N) - 16\*sqrt(N)间时，其中N是dataset的size。只需要使用k-means对vectors进行聚类。你需要30\*x 和 256 \*x的vectors进行训练（越大越好）。

支持GPU: yes

## b.如果1M-10M："...,IVF65536_HNSW32,..."

IVF组合上HNSW，使用HNSW来做cluster assignment。你将需要介于30\*65536 和 256 * 65536间的vectors来进行training。

支持GPU: no (如果想在GPU上，使用上面的IVF)

## c.如果 10M-100M："...,IVF262144_HNSW32,..."

与上相似，将65536替换成262144(2^18)。注意，训练会变慢。可以只在GPU上训练，其它运行在CPU上，见train_ivf_with_gpu.ipynb.

## d.如果100M-1B: "...,IVF1048576_HNSW32,..."

与上相似。训练会更慢。


# 参考

[faiss choosing](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
