---
layout: page
title: sklearn中的矩阵因式分解问题 
tagline: 介绍
---
{% include JB/setup %}

1. 主成分分析：PCA

PCA用于将多变量数据集分解成一系列的正交化（orthogonal）的主成分（它们之间有最大的variance）。在sklearn中，PCA的实现是一个转换器对象（transformer），它可以在fit函数中学习n个主成分，可以用于新数据上，并将它们投影到这些主成分上。


可选参数whiten=True，当将每个主成分归一化到单位方差上（unit variance），可以将数据投影到奇异矩阵（singular space）上。如果下流模型（down-stream model）希望在同性的信号上做出很强的预测，通常在这之前使用PCA会很管用：比如：RBF kernel的SVM，以及K-Means聚类算法。

下例给出了iris数据集的示例，它包含4个feature，投影到2维上，来解释：

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_pca_vs_lda_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_pca_vs_lda_0011.png" alt=""></a>
</figure>

PCA对象提供了一个PCA的概率解释，基于它的variance给出数据的一种可能性解释。它实现了一个score方法，可以用在cross-validation上。

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_pca_vs_fa_model_selection_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_pca_vs_fa_model_selection_0011.png" alt=""></a>
</figure>

示例：

- [Comparison of LDA and PCA 2D projection of Iris dataset](http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#example-decomposition-plot-pca-vs-lda-py)
- [Model selection with Probabilistic PCA and Factor Analysis (FA)](http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#example-decomposition-plot-pca-vs-fa-model-selection-py)

# 2. 增量PCA (Incremental PCA)

[PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)对象很有用，但对于大数据集很受限制。最大的限制是：PCA只支持批处理（batch processing），这意味着所有的数据都会被处理，在内存中进行fit。[IncrementalPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA) 对象使用不同的处理方式，它可以进行局部计算（partial computations），和PCA的结果几乎一致，处理数据的方式是：minibatch。IncrementalPCA可以实现核外并行（out-of-core）的PCA：

- partial_fit方法：使用该方法，可以地从本地硬盘、或网络数据库上顺序地获取数据块。
- fit方法：调用该方法使用的是基于内存映射文件（numpy.memmap）的方式。

IncrementalPCA只存储主成分估计（estimates of component），以及噪声的方差（noise variances），以便增量地更新explained_variance_ratio_属性值。它所使用的内存决取于每次batch时的样本数，而非在这个数据集中要处理的所有样本数。

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_incremental_pca_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_incremental_pca_0011.png" alt=""></a>
</figure>

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_incremental_pca_0021.png"><img src="http://scikit-learn.org/stable/_images/plot_incremental_pca_0021.png" alt=""></a>
</figure>

示例：

- [Incremental PCA](http://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html#example-decomposition-plot-incremental-pca-py)

# 3. Approximate PCA

通常，将数据投影到一个低维空间中，并且保留大多数的variance，通过drop掉奇异向量上具有低奇异值的成分。

例如，如果你对64x64像素的灰度图进行面部识别（face recognition），该数据的维度就是4096，如果使用一个kernel=RBF的SVM对这样的数据进行训练会很慢。我们知道数据的内在维度（intrinsic dimensionality ）是远低于4096的，因为所有的人脸图片基本相类似。这些样本主要集中在一些主要维度上（比如：200维）。PCA算法可以对数据进行线性转换，即能降低维度，也能同时保留下大多数可解释的variance。

[RandomizedPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.RandomizedPCA.html#sklearn.decomposition.RandomizedPCA)在这种情况下很有用：由于我们要丢弃掉大多数奇异向量，将计算限制到一个奇异向量的近似估计上更有效。

例如，下图显示了Olivetti数据集中的16个样本画像（中心为0.0）。在右侧，有16个奇异向量被reshape成画像。由于我们只需要top 16个奇异向量，size为：样本数=400， feature数=64x64=4096，计算时间需要小于1s：

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_faces_decomposition_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_faces_decomposition_0011.png" alt=""></a>
</figure>

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_faces_decomposition_0021.png"><img src="http://scikit-learn.org/stable/_images/plot_faces_decomposition_0021.png" alt=""></a>
</figure>

RandomizedPCA 可以替代PCA进行降维，我们需要指定维度值n_components作为输入参数。

假设：

- <img src="http://www.forkosh.com/mathtex.cgi?n_{max} = max(n_{samples}, n_{features})">
- <img src="http://www.forkosh.com/mathtex.cgi?n_{min} = min(n_{samples}, n_{features})">

RandomizedPCA的时间复杂度为：

<img src="http://www.forkosh.com/mathtex.cgi?O(n_{max}^2 \cdot n_{components})">

而PCA的时间复杂度要：

<img src="http://www.forkosh.com/mathtex.cgi? O(n_{max}^2 \cdot n_{min})">

RandomizedPCA 的内存占用为：

<img src="http://www.forkosh.com/mathtex.cgi? 2 \cdot n_{max} \cdot n_{components}">

而PCA的为：

<img src="http://www.forkosh.com/mathtex.cgi?  n_{max}
\cdot n_{min}">

**注意：RandomizedPCA中的inverse_transform实现，并不是真正的inverse transform转换，即使whiten=False时（缺省）**。

示例：

- [Faces recognition example using eigenfaces and SVMs](http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html#example-applications-face-recognition-py)
- [Faces dataset decompositions](http://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#example-decomposition-plot-faces-decomposition-py)

# 4.Kernel PCA

[KernelPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA)是PCA的一个扩展，它通过使用[kernel](http://scikit-learn.org/stable/modules/metrics.html#metrics)，进行非线性维度的降维。它有许多应用：去噪（denoising），压缩（compression），结构化预测（structured prediction，核依赖估计：kernel dependency estimation）。KernelPCA同时支持transform和inverse_transform。

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_kernel_pca_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_kernel_pca_0011.png" alt=""></a>
</figure>

示例：

- [Kernel PCA](http://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html#example-decomposition-plot-kernel-pca-py)

# 5. Sparse PCA (SparsePCA和MiniBatchSparsePCA)

[SparsePCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA) 是PCA的一个变种，它的目的是从稀疏的特征中抽取出最好的用于构建数据的主成分。

[MiniBatchSparsePCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchSparsePCA.html#sklearn.decomposition.MiniBatchSparsePCA)是SparsePCA的一个变种，它更快，但相对不准一些。对于给定的迭代次数，它的提速通过在小块特征集上进行迭代来完成。

主成分分析（PCA）的缺点是：通过该方法抽取的主成分具有特有的密度表示（dense expressions），例如：它们具有非零的相关系数（coefficients），可以表示成原始变量的线性组合。这很难解释。在许多情况下，真实的主成分可以想象成稀疏矩阵，例如：面部识别，主成分仅仅是脸部特征。

稀疏的主成分分析更吝啬，更可解释，更强调原始特征在样本差异下的贡献。

下面的示例展示了使用SparsePCA从Olivetti面部数据库中抽取的16个主成分。你可以看到，正则项是如何产生许多个0的. 更进一步，数据的自然结构造成了非零相关系数正交（vertically adjacent）。该模型不能增强这种数学现像：每个主成分是一个向量 <img src="http://www.forkosh.com/mathtex.cgi? h \in \mathbf{R}^{4096}">，不必关心正交，除非在可视化成64x64像素的图象时。这个现像（下面局部展示的主成分受数据本身结构的影响），这使得一些局部模式（local patterns）可以最小化重构误差（minimize reconstruction error）。另外，目前存在着许多sparsity-inducing范式，它们用来解释正交和许多不同类型的结构；详见[Jen09](http://scikit-learn.org/stable/modules/decomposition.html#jen09).

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_faces_decomposition_0021.png"><img src="http://scikit-learn.org/stable/_images/plot_faces_decomposition_0021.png" alt=""></a>
</figure>
<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_faces_decomposition_0051.png"><img src="http://scikit-learn.org/stable/_images/plot_faces_decomposition_0051.png" alt=""></a>
</figure>

注意，Sparse PCA有许多不同的公式。这里实现的方式是基于[Mri09](http://scikit-learn.org/stable/modules/decomposition.html#mrl09)。要解决的优化问题是：PCA问题（字典学习）在各主成分上带有一个<img src="http://www.forkosh.com/mathtex.cgi?\ell_1 ">罚项：

<figure>
	<a href="http://scikit-learn.org/stable/_images/math/fc84de1d23be724722d5ac285c78f4c77b780b36.png"><img src="http://scikit-learn.org/stable/_images/math/fc84de1d23be724722d5ac285c78f4c77b780b36.png" alt=""></a>
</figure>

当样本很少时，sparsity-inducing <img src="http://www.forkosh.com/mathtex.cgi?\ell_1 ">范式可以阻止学习到的主成分的噪声。**罚项的度（The degree of penalization：这里指的是稀疏度：sparsity），可以通过超参数alpha进行调整。值越小，正则项因子的力度就越温和，值越大则将会把许多相关系数shrink到0。**

注意：当尝试在线学习算法时，MiniBatchSparsePCA类没有实现partial_fit，因为该算法针对的是features方向，而不是samples方向。

示例：

- [Faces dataset decompositions](http://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#example-decomposition-plot-faces-decomposition-py)

参考：

[Mrl09](http://www.di.ens.fr/sierra/pdfs/icml09.pdf)
[Jen09](http://scikit-learn.org/stable/modules/www.di.ens.fr/~fbach/sspca_AISTATS2010.pdf)



参考：

1.[http://scikit-learn.org/stable/modules/decomposition.html#pca](http://scikit-learn.org/stable/modules/decomposition.html#pca)
