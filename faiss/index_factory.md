---
layout: page
title:  faiss index_factory
tagline: 
---
{% include JB/setup %}

# 介绍

index_factory函数会将一个string进行翻译，来生成一个composite faiss index。该string是一个逗号分割的列表。它的目标是，帮助index结构的构建，特别是它们被嵌套时。index_factory参数通常包含：**一个preprocessing组件、inverted file以及一个encoding组件**。这里总结了index_factory的components和参数。

示例：

	index=index_factory(128, "PCA80,FLat")

会为一个128D的vectors生成一个index，它会将它们使用PCA降维到80D上，接着做穷举搜索（exhaustive search）。

	index = index_factory(128, "OPQ16_64,IMI2x8,PQ8+16")

输入128D的vectors，使用OPQ转换成64D的16个block，使用一个inverted multi-index 2x8 bits（=65536 inverted lists），接着使用一个size=8, 16bytes的PQ进行重定义。

# 1.前缀(Prefixes)



| String  | Class name  |  注释   |
|---|---|---|
|  IDMap | IndexIDMap  |  <div style="width: 150pt">用于在不支持它的indexes上开启add\_with\_ids，主要用于flat indexes</div>|
{:.table-striped}

# 2.vector转换

这些strings会map成VectorTransform对象，在它们被索引前被应用到vectors上：

| string  | class name  | output维度  | 注释   |
|---|---|---|---|
| PCA64,PCAR64,PCAW64,PCAWR64  | PCAMatrix  |  64 |  使用一个PCA变换来降维. W= whitening, R= random rotation. PCAR对于使用一个Flat或scalar quantizer的预处理特别有用 |
| OPQ16, OPQ16_64  |  OPQMatrix | d,64  |  将vectors进行旋转，以便可以通过一个PQ进行有效编码成16个subvectors. 64是output维度，因为它通常有用，也可以降维。如果output维度没指定，那么与input维度相同 |
| RR64  | RandomRotation  | 64  | 在输入数据上做Random rotation。维度可以随input增加or减少  |
| L2norm  | NormalizationTransform  | d  | 对input vectors进行L2-normailizes  |
| ITQ256, ITQ  |  ITQMatrix | 256, d  | 对input vectors做ITQ转换。当vectors被编码成LSH时很有用  |
| Pad128  | RemapDimensionsTransform  | 128   | 对input vectors使用0s进行padding到128维  |

# 3.非穷举搜索组件

inverted files全部继随自IndexIVF。非穷举搜索组件（non-exhaustive component）指定了coarse quantizer(constructor的第一个参数)应该是什么。

| String  | Quantizer class  | cnetroids数目  | 注释  |
|---|---|---|---|
| IVF4096  | IndexFlatL2或IndexFlatIP  | 4096  | 使用一个flat quantizer，构建IndexIVF变种的一种  |
| IMI2x9  | MultiIndexQuantizer  | 2^(2 * 9) = 262144  | 使用更多centroids构建一个IVF，也更可能平衡些  |
| IVF65536_HNSW32  | IndexHNSWFlat  | 65536  | quantizer使用一个flat index训练，但索引使用一个HNSW。这会让quantization更快一些  |

HNSW索引继承自IndexHNSW。IndexHNSW会依赖于一个flat storage，它会存储实际vectors。在HNSW32中，32编码了links的数目，最低的level会使用最多的内存，具有32 x 2 links，或者每个vector为32 x 2 x 4=256 bytes。

| String  | Storage class  | 注释  |
|---|---|---|
| HNSW32  | IndexFlatL2  | 最有用的HNSW变种，因为当links被存储时，对于压缩vectors来说意义不大  |
| HNSW32_SQ8  | IndexScalarQuantizer  |  SQ8 scalar quantizer |
| HNSW32_PQ12  | IndexPQ  |  PQ12x8 index  |
| HNSW32_16384+PQ12  | Index2Layer  | 第1层是一个flat index，PQ会对quantizer的residual进行编码  |
| HNSW32_2x10+PQ12  | Index2Layer  | 第1层是一个IMI index，PQ会对quantizer的residual编码  |

# 4.Encodings

| String  | Class name(Flat/IVF)  | code size(bytes)  | 注释  |
|---|---|---|---|
| Flat  | IndexFlat,IndexIVFFlat  | 4\*d  | vectors不做任何处理，直接存储  |
| PQ16, PQ16x12  | IndexPQ, IndexIVFPQ	  | 16, ceil(16 \* 12 / 8)  |  使用PQ codes，为每12 bits使用16 codes。当bits的数目被忽略时，它被设置成8 （IndexIVFPQ只支持8-bit的encodings。使用后缀"np"不会训练Polysemous排列，它们可能会慢些） |
| SQ4, SQ8, SQ6, SQfp16  | IndexScalar Quantizer, IndexIVF ScalarQuantizer  | 4\*d/8, d, 6\*d/8, d\*2  |  Scalar quantizer encoding |
| Residual128, Residual3x10  | Index2Layer  | 4\*d/8, d, 6\*d/8, d\*2  |  Residual encoding. 将vectors量化成128centroids或者2x10的MI centroids。应根据PQ或SQ来实际编码residual。只作为一个codec使用 |
| ZnLattice3x10_6  | IndexLattice  | ceil(log2(128) / 8), ceil(3\*10 / 8)  | Lattice codec  |
| LSH, LSHrt, LSHr, LSHt  |  IndexLSH | ceil(d / 8)  |  Binarizes |




# 参考

[faiss index_factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
