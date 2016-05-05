---
layout: page
title: sklearn中的梯度下降法（SGD）
tagline: 介绍
---
{% include JB/setup %}

# 介绍

梯度下降法（SGD）是一个简单有效的方法，用于判断使用凸loss函数（convex loss function）的分类器（SVM或logistic回归）。即使SGD在机器学习社区已经存在了很久，到它被广泛接受也是最近几年。

SGD被成功地应用在大规模稀疏机器学习问题上（large-scale and sparse machine learning），经常用在文本分类及自然语言处理上。假如数据是稀疏的，该模块的分类器可以轻松地解决这样的问题：超过10^5的训练样本、超过10^5的features。

SGD的优点是：

- 高效
- 容易实现（有许多机会进行代码调优）

SGD的缺点是：

- SGD需要许多超参数：比如正则项参数、迭代数。
- SGD对于特征归一化（feature scaling）是敏感的。

# 2.分类

**注意：在对模型进行拟合前，必须重排(permute/shuffle)你的训练数据集； 或者在每次迭代后使用shuffle=True 进行shuffle。**

SGDClassifier类实现了一个为分类设计的普通随机梯度下降学习，它支持不同的loss函数和罚项。

<figure>
    <a href="http://scikit-learn.org/stable/_images/plot_sgd_separating_hyperplane_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_sgd_separating_hyperplane_0011.png" alt=""></a>
</figure>

和其它分类器一样，SGD必须使用两个array数姐：一个数组X（其size为[n_samples, n_features]）：保存着训练样本；一个数组Y：保存着训练样本的target值（class label）：

{% highlight python %}

>>> from sklearn.linear_model import SGDClassifier
>>> X = [[0., 0.], [1., 1.]]
>>> y = [0, 1]
>>> clf = SGDClassifier(loss="hinge", penalty="l2")
>>> clf.fit(X, y)
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)

{% endhighlight %}

在fit后，模型就可以用来进行预测：

{% highlight python %}

>>> clf.predict([[2., 2.]])
array([1])

{% endhighlight %}

SGD可以为训练数据拟合线性模型。属性coef_保存着模型参数：

{% highlight python %}

>>> clf.coef_                                         
array([[ 9.9...,  9.9...]])

{% endhighlight %}

属性intercept_保存着intercept(aka offset or bias):

{% highlight python %}

>>> clf.intercept_                                    
array([-9.9...])

{% endhighlight %}

通过设置参数fit_intercept来控制模型是否使用intercept（一个偏差超平面 biased hyperplane）

为了得到与该超平面的有符号距离，可以使用SGDClassifier.decision_function：

{% highlight python %}

>>> clf.decision_function([[2., 2.]])                 
array([ 29.6...])

{% endhighlight %}

loss function可以通过loss参数进行设置。SGDClassifier支持下面的loss函数：

- loss="hinge": (soft-margin)线性SVM.
- loss="modified_huber": 带平滑的hinge loss.
- loss="log": logistic回归
- 以及下面所提到的所有回归的loss

前两个loss函数都是lazy的，如果相应的一个示例违反了这个margin约束，它们只更新模型参数，这使得训练非常有效，并可能导致稀疏的模型，即使使用了L2罚项。

使用loss="log" 或 loss="modified_huber"可以启用predict_proba方法，它给出了一个概率估计向量，每个样本x都有对应一个P(y|x)：

{% highlight python %}

>>> clf = SGDClassifier(loss="log").fit(X, y)
>>> clf.predict_proba([[1., 1.]])                      
array([[ 0.00...,  0.99...]])

{% endhighlight %}

通过penalty参数，可以设置对应的惩罚项。SGD支持下面的罚项：

- penalty="l2": 对coef_的L2范数罚项
- penalty="l1": 对coef_的L1范数罚项
- penalty="elasticnet": L2和L1的convex组合; (1 - l1_ratio) * L2 + l1_ratio * L1

缺省设置为：penalty="l2"。L1罚项会导致稀疏解决方案，产生许多零系数。Elastic Net可以解决L1罚项在高度相关属性上的一些不足。参数l1_ratio控制着L1和L2罚项的convex组合。

SGDClassifier支持多分类，它以"one-vs-all(OVA)"的方式通过结合多个二分类来完成。对于K个类中的每个类来说，一个二分类器可以通过它和其它K-1个类来进行学习得到。在测试时，我们会为每个分类器要计算置信度（例如：到超平面的有符号距离），并选择最高置信度的类。下图展示了对iris数据集进行OVA方式分类。虚线表示三个OVA分类器，背影色展示了三个分类器产生的决策边界。

<figure>
    <a href="http://scikit-learn.org/stable/_images/plot_sgd_iris_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_sgd_iris_0011.png" alt=""></a>
</figure>

在多分类问题上，coef_是一个二维数组：shape=[n_classes, n_features]，intercept_是一个一维数组：shape=[n_classes]. coef_的第i行保存着OVA分类器对第i类的权重向量；这些类通过升序方式索引（详见属性classes_）。注意，原则上，因为他们允许创建一个概率模型，loss="log" 和 loss="modified_huber"更合适one-vs-all分类。

SGDClassifier 支持带权重的分类和带权重的实例，通过fit参数：class_weight 和 sample_weight。详见：SGDClassifier.fit。

示例：

- [SGD: Maximum margin separating hyperplane](http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_separating_hyperplane.html#example-linear-model-plot-sgd-separating-hyperplane-py)
- [Plot multi-class SGD on the iris dataset](http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_iris.html#example-linear-model-plot-sgd-iris-py)
- [SGD: Weighted samples](http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_weighted_samples.html#example-linear-model-plot-sgd-weighted-samples-py)
- [Comparing various online solvers](http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html#example-linear-model-plot-sgd-comparison-py)
- [SVM: Separating hyperplane for unbalanced classes](http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html#example-svm-plot-separating-hyperplane-unbalanced-py)

SGDClassifier 支持平均随机梯度下降法（ASGD）。可以通过设置`average=True`来开启平均功能。ASGD通过对一个样本每次进行SGD迭代的系数做平均。当使用ASGD时，learning_rate可以变得更大，从而在训练时对一些数据集进行加速。

对于使用logistic loss的分类，另一个使用平均策略的SGD变种是：随机平均梯度（SAG）算法。由LogisticRegression提供。

# 3.回归

SGDRegressor类实现了一个普通的随机梯度下降学习，它支持不同的loss函数和罚项来拟合线性回归模型。SGDRegressor对于大数据量训练集（》10000）的回归问题很合适。对于其它问题：我们推荐你使用Ridge, Lasso, or ElasticNet。

loss函数可以通过loss参数进行设置。SGDRegressor支持以下的loss函数：

- loss="squared_loss": 普通最小二乘法
- loss="huber": 对于稳健回归（robust regression）的Huber loss
- loss="epsilon_insensitive": 线性SVM

Huber 和 epsilon-insensitive loss函数可以用于稳健回归（robust regression）。敏感区的宽度可以通过参数epsilon来指定。该参数依赖于target变量的规模。

SGDRegressor和SGDClassifier一样支持ASGD。通过设置`average=True`来开启。

对于使用最小二乘loss和l2罚项的回归来说，可以使用SGD的另一个变种SAG算法，由Ridge的solver提供。

# 4.稀疏数据的SGD

**注意：稀疏实现与dense实现相比会产生不同结果，由于在intercept的learning rate上有shrink。**

稀疏矩阵由scipy.sparse支持。出于最大化的效率，使用CSR矩阵格式可以由scipy.sparse.csr_matrix定义。

示例：

- [Classification of text documents using sparse features](http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py)

# 5.复杂度

SGD的主要优点在于它的高效，它与训练样本数成线性关系。如果X是一个size(n,p)的训练矩阵，具有一个cost为<img src="http://www.forkosh.com/mathtex.cgi?O(k n \bar p)">，其中k为迭代数（epochs），而<img src="http://www.forkosh.com/mathtex.cgi?\bar p">是每个样本的非零属性平均数。

最近的理论结果表明，优化正确率问题的runtime并不会随着训练样本大小的增加而增加。

# 6.实用tips

- SGD对于特征归一化（feature scaling）很敏感，因此强烈推荐你对数据进行归一化。例如，将输入向量X的每个属性归一化到[0,1]或者[-1,1]，或者将它标准化到均值为0、方差为1。注意，必须对测试集向量也使用相同的归一化操作以获取有意义的结果。这可以通过StandardScaler做到：

{% highlight python %}

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)  # Don't cheat - fit only on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # apply same transformation to test data

{% endhighlight %}

如果你的属性项本身就做了归一化（例如：词频、或者指示器feature），那就不需要再做归一化了。

- 可以通过GridSearchCV找到一个合适的正则项<img src="http://www.forkosh.com/mathtex.cgi?\alpha">，通常范围在：10.0**-np.arange(1,7)。
- 经验上，我们发现SGD在接近10^6的训练样本时进行收敛。因此，第一步合理的猜想是，将迭代数设置成：n_iter = np.ceil(10**6 / n)，其中n是训练集的size。
- 如果你在使用PCA提取的feature上使用SGD，我们发现，对于feature值进行归一化是明智的，通过一些常数c，以便训练数据的平均L2范式等于1。
- 我们发现ASGD在当feature很大和eta0很大时运转很好。

参考：

- [“Efficient BackProp” Y. LeCun](http://scikit-learn.org/stable/modules/yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

# 7.数学公式

给定训练集： <img src="http://www.forkosh.com/mathtex.cgi?(x_1, y_1), \ldots, (x_n, y_n)">，

且满足<img src="http://www.forkosh.com/mathtex.cgi? x_i \in \mathbf{R}^n">和<img src="http://www.forkosh.com/mathtex.cgi?y_i \in \{-1,1\}">，

我们的目标是学到一个线性scoring函数：

<img src="http://www.forkosh.com/mathtex.cgi? f(x) = w^T x + b">

它的模型参数为：

<img src="http://www.forkosh.com/mathtex.cgi? w \in \mathbf{R}^m">

intercept为：

<img src="http://www.forkosh.com/mathtex.cgi?b \in \mathbf{R}">。

为了进行预测，我们还需要看一个f(x)的符号。一个通用的选择是，通过最小化正则化的训练误差来得到最优模型参数：


<figure>
    <a href="http://scikit-learn.org/stable/_images/math/4782697c94f74995ee99624f00633bba53f5245f.png"><img src="http://scikit-learn.org/stable/_images/math/4782697c94f74995ee99624f00633bba53f5245f.png" alt=""></a>
</figure>

其中L是loss函数，计算模型是否拟合，R是正则项（aka penalty），用于惩罚模型复杂度；<img src="http://www.forkosh.com/mathtex.cgi?\alpha > 0 ">是一个非负的超参数。

对于不同分类器，L有不同选择，比如：

- Hinge: (soft-margin)SVM
- Log: Logistic回归
- Least-Squares: Ridge回归
- Epsilon-Insensitive: (soft-margin)SVM

上面所有的loss函数可以被认为是误分类的上边界（0-1 loss），如图如下：

<figure>
    <a href="http://scikit-learn.org/stable/_images/plot_sgd_loss_functions_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_sgd_loss_functions_0011.png" alt=""></a>
</figure>

正则项R有如下的选择：

- L2范数：<img src="http://www.forkosh.com/mathtex.cgi?R(w) := \frac{1}{2} \sum_{i=1}^{n} w_i^2">
- L1范数：<img src="http://www.forkosh.com/mathtex.cgi?R(w) := \sum_{i=1}^{n} |w_i|">，会导致稀疏。
- Elastic Net: <img src="http://www.forkosh.com/mathtex.cgi?R(w) := \frac{\rho}{2} \sum_{i=1}^{n} w_i^2 + (1-\rho) \sum_{i=1}^{n} |w_i|">，它是L2和L1的convex组合，其中<img src="http://www.forkosh.com/mathtex.cgi?\rho ">通过1 - l1_ratio给出。

下图展示了当<img src="http://www.forkosh.com/mathtex.cgi?R(w) = 1">在参数空间内不同的正则项的轮廓。

<figure>
    <a href="http://scikit-learn.org/stable/_images/plot_sgd_penalties_0011.png"><img src="http://scikit-learn.org/stable/_images/plot_sgd_penalties_0011.png" alt=""></a>
</figure>

## 7.1 SGD

随机梯度下降(Stochastic gradient descent)是一个优化方法，用于非限制性优化问题。与它对应的是批梯度下降（batch gradient descent），SGD逼近E(w,b)的真实梯度，它在一次迭代时只考虑一个训练样本。

SGDClassifier实现了一阶的SGD学习。该算法在训练集上进行迭代，对于每个样本，根据下面的更新规则更新模型参数：

<figure>
    <a href="http://scikit-learn.org/stable/_images/math/98cdc3aed40cb93594dbaaf045ea3e1abbd8edcb.png"><img src="http://scikit-learn.org/stable/_images/math/98cdc3aed40cb93594dbaaf045ea3e1abbd8edcb.png" alt=""></a>
</figure>

其中，<img src="http://www.forkosh.com/mathtex.cgi?\eta">是学习率，它控制着参数空间的步长。截距(intercept)b的更新相类似，但没有正则项。

学习率<img src="http://www.forkosh.com/mathtex.cgi?\eta">即可以是常数，也可以逐渐衰减。对于分类，缺省的学习率(learning_rate='optimal')，通过下式给定：

<img src="http://www.forkosh.com/mathtex.cgi?\eta^{(t)} = \frac {1}{\alpha  (t_0 + t)}">

其中，t是时间步长（总共有：n_samples * n_iter的时间步长），<img src="http://www.forkosh.com/mathtex.cgi?t_0 ">则基于Léon Bottou提出的启发式学习，以便期望的初始更新与期望的权值size是可以相比较的。可以在BaseSGD的_init_t中找到精确定义。

对于回归，缺省的学习率schedule是inverse scaling（learning_rate='invscaling'），通过：

<img src="http://www.forkosh.com/mathtex.cgi?\eta^{(t)} = \frac{eta_0}{t^{power\_t}}">

其中，<img src="http://www.forkosh.com/mathtex.cgi?eta_0">和<img src="http://www.forkosh.com/mathtex.cgi?power\_t ">这两个超参数由用户通过参数eta0 和 power_t来控制。

对于一个常数值的学习率来说，可以使用learning_rate='constant' ，并使用eta0来指定学习率。

该模型的参数可以通过属性coef_和intercept_来访问：

- coef_: 权重w
- intercept_: 参数b

# 8.实现细节

SGD的实现受Léon Bottou提出的Stochastic Gradient SVM的影响。与SvmSGD相类型，weight向量被表示成一个标量（scalar）和一个向量（vector）的乘积，它允许L2正则化进行更新。对于稀疏特征向量的情况，截距（intercept）的更新使用一个更小的学习率（乘以0.01）来说明事实上更新更频繁。训练样本按顺序被选择，在每一个被观察到的示例后学习率会变慢。

我们采用的学习率schedule是：Shalev-Shwartz et al. 2007提出的。对于多分类，使用”one-versus-all“方法。我们使用由Tsuruoka et al. 2009提出的截断梯度（truncated gradient）算法进行L1正则化（和Elastic Net）。代码使用cython编写。

参考：

- “[Stochastic Gradient Descent](http://leon.bottou.org/projects/sgd)” L. Bottou - Website, 2010
- “[The Tradeoffs of Large Scale Machine Learning](http://leon.bottou.org/slides/largescale/lstut.pdf)” L. Bottou - Website, 2011.
- “[Pegasos: Primal estimated sub-gradient solver for svm](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.74.8513)” S. Shalev-Shwartz, Y. Singer, N. Srebro - In Proceedings of ICML ‘07.
- “[Stochastic gradient descent training for l1-regularized log-linear models with cumulative penalty](http://www.aclweb.org/anthology/P/P09/P09-1054.pdf)” Y. Tsuruoka, J. Tsujii, S. Ananiadou - In Proceedings of the AFNLP/ACL ‘09.

参考：

1.[http://scikit-learn.org/stable/modules/sgd.html](http://scikit-learn.org/stable/modules/sgd.html)
