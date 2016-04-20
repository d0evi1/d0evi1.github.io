---
layout: page
title: sklearn中的随机森林 
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍


sklearn.ensemble模块包含了两种基于随机决策树的平均算法：RandomForest算法和Extra-Trees算法。这两种算法都采用了很流行的树设计思想：perturb-and-combine思想。这种方法会在分类器的构建时，通过引入随机化，创建一组各不一样（diverse）的分类器。这种ensemble方法的预测会给出各个分类器预测的平均。

和其它分类器相比，forest分类器可以使用这两个数组更好的进行拟合（fit）：

- X数组：一个sparse或dense数组，它的训练样本的size为[n_samples, n_features]；
- Y数组：一个size为[n_samples]的Y数组，它表示训练样本的target值:（注意：是大写的Y） 

{% highlight python %}

>>> from sklearn.ensemble import RandomForestClassifier
>>> X = [[0, 0], [1, 1]]
>>> Y = [0, 1]
>>> clf = RandomForestClassifier(n_estimators=10)
>>> clf = clf.fit(X, Y)

{% endhighlight %}


# 2.主题

## 2.1 RandomForests

在随机森林（RF）中，该ensemble方法中的每棵树都基于一个通过可放回抽样（bootstrap）得到的训练集构建。另外，在构建树的过程中，当split一个节点时，split的选择不再是对所有features的最佳选择。相反地，在features的子集中随机进行split反倒是最好的split方式。这种随机的后果是，整个forest的bias通常会略微增大，但是由于结果会求平均，因此，它的variance会降低，会修正偏大的bias，从而得到一个更好的模型。

sklearn的随机森林（RF）实现通过对各分类结果预测求平均得到，而非让每个分类器进行投票（vote）。

## 2.2 Ext-Trees

在Ext-Trees中(详见ExtraTreesClassifier和 ExtraTreesRegressor)，该方法中，随机性在划分时会更进一步进行计算。在随机森林中，会使用侯选feature的一个随机子集，而非查找最好的阀值，对于每个候选feature来说，阀值是抽取的，选择这种随机生成阀值的方式作为划分原则。通常情况下，在减小模型的variance的同时，适当增加bias是允许的。

{% highlight python %}

>>> from sklearn.cross_validation import cross_val_score
>>> from sklearn.datasets import make_blobs
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.ensemble import ExtraTreesClassifier
>>> from sklearn.tree import DecisionTreeClassifier

>>> X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
...     random_state=0)

>>> clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,
...     random_state=0)
>>> scores = cross_val_score(clf, X, y)
>>> scores.mean()                             
0.97...

>>> clf = RandomForestClassifier(n_estimators=10, max_depth=None,
...     min_samples_split=1, random_state=0)
>>> scores = cross_val_score(clf, X, y)
>>> scores.mean()                             
0.999...

>>> clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
...     min_samples_split=1, random_state=0)
>>> scores = cross_val_score(clf, X, y)
>>> scores.mean() > 0.999
True

{% endhighlight %}

在iris数据集上的分类：

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_forest_iris_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_forest_iris_0011.png" alt=""></a>
</figure>

# 2.3 参数

当使用这些方法时，最主要的参数是调整n_estimators 和 max_features。n_estimators指的是森林中树的个数。树数目越大越好，但会增加计算开销。另外，注意如果超出限定数量后，计算结果将停止。max_features指的是，当划分一个节点时，feature的随机子集的size。该值越小，variance会变小，但bias会变大。

根据经验，

- 对于回归问题：好的缺省值是max_features=n_features；
- 对于分类问题：好的缺省值是max_features=sqrt(n_features)。n_features指的是数据中的feature总数。

当设置max_depth=None，以及min_samples_split=1时，通常会得到好的结果（完全展开的树）。但需要注意，这些值通常不是最优的，并且会浪费RAM内存。最好的参数应通过cross-validation给出。另外需要注意：

- 在随机森林中，缺省时会使用bootstrap进行样本抽样(bootstrap=True) ；
- 而extra-trees中，缺省策略为不使用bootstrap抽样 (bootstrap=False)；

当使用bootstrap样本时，泛化误差可能在估计时落在out-of-bag样本中。此时，可以通过设置oob_score=True来开启。

# 2.4 并行化

该模块可以并行构建多棵树，以及并行进行预测，通过n_jobs参数来指定。如果n_jobs=k，则计算被划分为k个job，并运行在k核上。如果n_jobs=-1，那么机器上所有的核都会被使用。注意，由于进程间通信的开销，加速效果并不会是线性的（job数k不会提升k倍）。通过构建大量的树，比起单棵树所需的时间，性能也能得到很大提升。（比如：在大数据集上）

示例：

- [Plot the decision surfaces of ensembles of trees on the iris dataset](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html#example-ensemble-plot-forest-iris-py)
- [Pixel importances with a parallel forest of trees](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances_faces.html#example-ensemble-plot-forest-importances-faces-py)
- [Face completion with a multi-output estimators](http://scikit-learn.org/stable/auto_examples/plot_multioutput_face_completion.html#example-plot-multioutput-face-completion-py)

# 2.5 特征重要性评估

在树的一个决策点上的一个feature的相对rank（i.e. 深度：depth），可以用来判断该feature相对于预测目标的重要程度。在树顶端使用的feature，对于输入样本中的大部分的最终预测决策来说，贡献挺大。样本中期望的块（expected fraction of the samples）可以被用于feature重要性程度的估计。

通过对随机树之间的期望活动率求均值，我们可以减小评估器的varaince，并使用它进行特征选择（feature selection）。

下例显示了各独立像素在人脸识别中的相对重要性，通过着色来表示。（使用了ExtraTreesClassifier模型）：

<figure>
	<a href="http://scikit-learn.org/stable/_images/plot_forest_importances_faces_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_forest_importances_faces_0011.png" alt=""></a>
</figure>

实际上，这些评估器在拟合的模型上保存着一个属性：feature_importances_。这是一个{n_features,)的数组，它的值为正，求和为1.0. 该值越高，那么该feature对于预测函数的贡献也越大。

示例：

- [Pixel importances with a parallel forest of trees](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances_faces.html#example-ensemble-plot-forest-importances-faces-py)
- [Feature importances with forests of trees](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py)


# 2.6 总的随机森林Embedding

RandomTreesEmbedding实现了一个无监督的数据转换器。使用一个完全随机树的森林，RandomTreesEmbedding会进行编码数据，通过数据点结束的叶子节点建立索引。该索引以one-of-K的方式，进行高维、稀疏二元编码。该编码的计算十分有效，可以作为基础用在其它学习算法上。编码的size与稀疏性受树的数量、以及每棵树的最大深度的影响。对于在ensemble方法中的每棵树而言，该编码包含了一个接一个的entry。这种编码的size至多有（n_estimators * 2 ** max_depth），在随机森林中最大数目的叶子节点。

邻近数据点，很可能在一棵树的相同叶子上，该转换执行了一个显式的、无参的密度估计。

示例：

- [Hashing feature transformation using Totally Random Trees](http://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_embedding.html#example-ensemble-plot-random-forest-embedding-py)
- [Manifold learning on handwritten digits: Locally Linear Embedding, Isomap... compares non-linear dimensionality reduction techniques on handwritten digits.](http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#example-manifold-plot-lle-digits-py)
- [Feature transformations with ensembles of trees compares supervised and unsupervised tree based feature transformations.](http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#example-ensemble-plot-feature-transformation-py)

参考：

[http://scikit-learn.org/stable/modules/ensemble.html](http://scikit-learn.org/stable/modules/ensemble.html)
