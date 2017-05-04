---
layout: page
title: MLlib的数据类型 
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

MLLib支持存储在单机上的local vectors和metrices，也支持分布式的matrics（背后通过一或多个RDD实现）。local vectors和local matrices都是简单数据类型，作为公共接口使用。底层的线性算法操作则由Breeze和jblas来实现。MLlib中，监督学习的一个训练样本，被称为“labeled point”。

# 2.Local vector

存储在单机上的local vector，由一个整数类型的从0开始的索引(indice)，double类型的值(value)组成。MLlib支持两种类型的local vectors: dense和sparse。dense vector 背后通过一个double array来表示它的条目值，而sparse vector则由两个并列数组实现：索引(indices)和值(values)。例如，一个vector(1.0, 0.0, 3.0)，可以表示成dense格式：[1.0, 0.0, 3.0]，也可以表示成sparse格式：(3, [0, 2], [1.0, 3.0])，其中，3就是vector的size。

示例：

{% highlight scala %}

import org.apache.spark.mllib.linalg.{Vector, Vectors}

// Create a dense vector (1.0, 0.0, 3.0).
val dv: Vector = Vectors.dense(1.0, 0.0, 3.0)
// Create a sparse vector (1.0, 0.0, 3.0) by specifying its indices and values corresponding to nonzero entries.
val sv1: Vector = Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0))
// Create a sparse vector (1.0, 0.0, 3.0) by specifying its nonzero entries.
val sv2: Vector = Vectors.sparse(3, Seq((0, 1.0), (2, 3.0)))

{% endhighlight%}

注意：scala 缺省会import scala.collection.immutable.Vector，你必须显式使用MLlib的Vector: import org.apache.spark.mllib.linalg.Vector。

# 3.Labeled point

labeled point是一个local vector，可以是dense或sparse，它与一个label/response相关联。在MLlib中，labeled points被用于见监督学习算法。我们使用一个double来存储一个label，因此，它可以同时用于分类和回归。对于二分类，一个label可以是0（negative）或1(positive). 对于多分类，label可以从0开始：0, 1, 2, ...

[LabeledPoint](https://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.mllib.regression.LabeledPoint)

{% highlight scala %}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

// Create a labeled point with a positive label and a dense feature vector.
val pos = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))

// Create a labeled point with a negative label and a sparse feature vector.
val neg = LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))


{% endhighlight %}

## Sparse data

实际上，稀疏的训练数据很常见。MLlib支持读取以LIBSVM格式存储的训练样本，缺省使用LIBSVM/LIBLINEAR格式。它是一个文本格式，用于表示一个labeld sparse feature vector:

	label index1:value1 index2:value2 ...

其中索引从1开始，以升序排列。加载时，特征的索引会被转换成从0开始.

[MLUtils.loadLibSVMFile](https://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.mllib.util.MLUtils$)

{% highlight scala %}

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

{% endhighlight %}

# 4.Local matrix

local matrix由整数型的行索引、列索引(indices)，以及浮点型的值(values)组成，存储在单机上。MLlib支持dense matrices，它的条目值存储在单个double array上，以列为主（column-major）的顺序。而sparse matrices，它是非零条目值以压缩稀疏列(CSC: Compressed Sparse Column)的格式存储，以列为主（column-major）的顺序。例如，下面的dense matrix：

<img src="http://www.forkosh.com/mathtex.cgi?\[ \begin{pmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \\ 5.0 & 6.0 \end{pmatrix} \] ">

被存储成一维的array [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]，matrix的size为(3,2).

local matrices的基类是[Matrix](https://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.mllib.linalg.Matrix)，它提供了两种实现：DenseMatrix和SparseMatrix. 我们推荐使用Matrices的工厂方法来创建local matrices。记住，MLlib的local matrices被存成以列为主的顺序。

{% highlight scala %}

import org.apache.spark.mllib.linalg.{Matrix, Matrices}

// Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
val dm: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))

// https://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.mllib.linalg.Matrices$
// Create a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0))
// colPtrs: Array[Int], rowIndices: Array[Int]
val sm: Matrix = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8))

{% endhighlight %}


# 5.Distributed matrix

distributed matrix由long型的行索引和列索引，以及double型的值组成，以一或多个RDD的方式分布式存储。**选择合适的格式来存储分布式大矩阵相当重要。**将一个分布式矩阵转换成一个不同的格式，可能需要一个全局的shuffle，计算代价高昂！至今支持三种类型的分布式矩阵。

基本类型被称为RowMatrix。

一个RowMatrix是以行为主的分布式矩阵，它没有行索引，只是一个特征向量的集合。背后由一个包含这些行的RDD实现，其中，每个行(row)都是一个local vector。我们假设，对于一个RowMatrix，列的数目并不大，因而，单个local vector可以合理地与driver进行通信，也可以使用单个节点被存储/操作。

一个IndexedRowMatrix与一个RowMatrix相似，但有行索引，它可以被用来标识出行(rows)以及正在执行的join操作（executing joins）。

一个CoordinateMatrix是一个分布式矩阵，以[coordinate list(COO)](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_.28COO.29)的格式进行存储，后端由一个包含这些条目的RDD实现。

注意：

**一个分布式矩阵的底层RDD实现必须是确定的(deterministic)，因为我们会对matrix size进行cache。总之，使用非确定的RDD会导致errors。**

## RowMatrix

RowMatrix是面向row的分布式矩阵，没有行索引，背后由一个包含这些行的RDD实现，基中，每个行是一个local vector。因为每个row都被表示成一个local vector，列的数目被限制在一个整数范围内，实际使用时会更小。

RowMatrix可以通过一个RDD[Vector]实例被创建。接着，我们可以计算它的列归纳统计（column summary statistics），以及分解(decompositions)。[QR deceompsition](https://en.wikipedia.org/wiki/QR_decomposition)的格式：A=QR，其中Q是一个正交矩阵（orthogonal matrix），R是一个上三角矩阵(upper triangular matrix)。对于SVD和PCA，请参考[降维]()这一节。

{% highlight scala %}

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix

val rows: RDD[Vector] = ... // an RDD of local vectors
// Create a RowMatrix from an RDD[Vector].
val mat: RowMatrix = new RowMatrix(rows)

// Get its size.
val m = mat.numRows()
val n = mat.numCols()

// QR decomposition 
val qrResult = mat.tallSkinnyQR(true)

{% endhighlight %}

## IndexedRowMatrix

IndexedRowMatrix与RowMatrix相似，但有行索引。它背后由一个带索引的行的RDD实现，因此，每个行可以被表示成long型索引和local vector。

一个IndexedRowMatrix由RDD[IndexedRow]实例实现，其中，IndexedRow是一个在(Long,Vector)上的封装wrapper。IndexedRowMatrix可以被转换成一个RowMatrix，通过drop掉它的行索引来完成。


{% highlight scala %}

import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}

val rows: RDD[IndexedRow] = ... // an RDD of indexed rows
// Create an IndexedRowMatrix from an RDD[IndexedRow].
val mat: IndexedRowMatrix = new IndexedRowMatrix(rows)

// Get its size.
val m = mat.numRows()
val n = mat.numCols()

// Drop its row indices.
val rowMat: RowMatrix = mat.toRowMatrix()

{% endhighlight %}


## CoordinateMatrix

CoordinateMatrix是一个分布式矩阵，背后由一个包含这些条目(entries)的RDD实现。每个条目(entry)是一个三元组（tuple）：(i: Long, j: Long, value: Double)， 其中: i是行索引，j是列索引，value是entry value。当矩阵的维度很大，并且很稀疏时，推荐使用CoordinateMatrix。

CoordinateMatrix可以通过一个RDD[MatrixEntry]实例来创建，其中MatrixEntry是一个(Long,Long,Double)的Wrapper。通过调用toIndexedRowMatrix，一个CoordinateMatrix可以被转化成一个带有稀疏行的IndexedRowMatrix。CoordinateMatrix的其它计算目前不支持。

{% highlight scala %}

import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}

val entries: RDD[MatrixEntry] = ... // an RDD of matrix entries
// Create a CoordinateMatrix from an RDD[MatrixEntry].
val mat: CoordinateMatrix = new CoordinateMatrix(entries)

// Get its size.
val m = mat.numRows()
val n = mat.numCols()

// Convert it to an IndexRowMatrix whose rows are sparse vectors.
val indexedRowMatrix = mat.toIndexedRowMatrix()

{% endhighlight %}

## BlockMatrix

BlockMatrix是一个分布式矩阵，背后由一个MatrixBlocks的RDD实现，其中MatrixBlock是一个tuple: ((Int,Int),Matrix)，其中(Int,Int)是block的索引，Matrix是由rowsPerBlock x colsPerBlock的sub-matrix。BlockMatrix支持方法：add和multiply。BlockMatrix也有一个helper function：validate，它可以被用于确认BlockMatrix的设置是否正确。

BlockMatrix 可以由一个IndexedRowMatrix或CoordinateMatrix，通过调用toBlockMatrix很容易地创建。toBlockMatrix缺省会创建1024x1024的blocks。可以通过toBlockMatrix(rowsPerBlock, colsPerBlock)进行修改。

{% highlight scala %}

import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}

val entries: RDD[MatrixEntry] = ... // an RDD of (i, j, v) matrix entries
// Create a CoordinateMatrix from an RDD[MatrixEntry].
val coordMat: CoordinateMatrix = new CoordinateMatrix(entries)
// Transform the CoordinateMatrix to a BlockMatrix
val matA: BlockMatrix = coordMat.toBlockMatrix().cache()

// Validate whether the BlockMatrix is set up properly. Throws an Exception when it is not valid.
// Nothing happens if it is valid.
matA.validate()

// Calculate A^T A.
val ata = matA.transpose.multiply(matA)

{% endhighlight %}


# 参考

[mllib-data-types](https://spark.apache.org/docs/1.6.1/mllib-data-types.html)

1.[http://spark.apache.org/docs/latest/mllib-ensembles.html](http://spark.apache.org/docs/latest/mllib-ensembles.html)

2.[SCIPY CSC格式](http://www.scipy-lectures.org/advanced/scipy_sparse/csc_matrix.html)

3.[CSC格式](http://www.cs.colostate.edu/~mcrob/toolbox/c++/sparseMatrix/sparse_matrix_compression.html)
