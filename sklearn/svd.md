---
layout: page
title: sklearn中的降维：SVD 和LSA
tagline: 介绍
---
{% include JB/setup %}

# svd和LSA

[TruncatedSVD](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD) 实现了奇异值分解（SVD）的一个变种，它只需要计算k个最大的奇异值，参数k由用户指定。

当TruncatedSVD用于term-doc矩阵上时（通过CountVectorizer 或 TfidfVectorizer返回），该变换就是LSA（潜语义分析），因为它将这样的矩阵转换到一个隐含(semantic)的低维空间上。特别的，LDA与同义（synonymy）和多义（polysemy）经常对比（两者都意味着每个词都有多个意思），这造成了term-doc矩阵过于稀疏，以至于使用余弦相似度进行计算时通常不相似。

**注意：LSA经常以LSI(latent semantic indexing)的方式被大家熟知，尽管它严格意义上指的是在信息检索领域用于保存索引。**

数学上，TruncatedSVD将训练样本X，产生一个低维的近似矩阵Xk:

<img src="http://www.forkosh.com/mathtex.cgi?X \approx X_k = U_k \Sigma_k V_k^\top">

在这项操作后，<img src="http://www.forkosh.com/mathtex.cgi? U_k \Sigma_k^\top">是转换后带有k个features的训练集（在API中称为: n_components）。

为了在测试集X上进行这样的转换，我们也需要乘上Vk:

<img src="http://www.forkosh.com/mathtex.cgi?X' = X V_k">

**注意：大多数在自然语言处理（NLP）以及信息检索（IR）文献中的LSA方法，交换了矩阵X的axes，它的shape为：n_features × n_samples。而我们以不同的方式来表示以便更好地适配sklearn API，但奇异值本身是一致的。**

TruncatedSVD和PCA很相似，但不同的是，它在样本矩阵X上直接运行，而非它们的协方差矩阵（covariance matrices)。当X的列（每个feature）已经从feature值中提取后，在结果矩阵上进行TruncatedSVD与PCA相同。在实际上术语中， 这意味着TruncatedSVD转换器接受scipy.sparse 参数矩阵，不需要dense矩阵；即使对于中等size的docment集，使用dense矩阵会填满整个内存。

TruncatedSVD转换器可以在任何（稀疏）特征矩阵上运行，推荐在LDA文档处理时对原始词频TF进行tf-idf矩阵转换。特别的，次线性归一化（sublinear scaling）和IDF可以通过参数(sublinear_tf=True, use_idf=True) 进行设置，使得feature的值更接近高斯分布（Gaussian distribution），从而补偿对文本数据进行LSA的误差。

示例：

- [Clustering text documents using k-means](http://scikit-learn.org/stable/auto_examples/text/document_clustering.html#example-text-document-clustering-py)


参考：

1.[http://scikit-learn.org/stable/modules/decomposition.html#lsa](http://scikit-learn.org/stable/modules/decomposition.html#lsa)
