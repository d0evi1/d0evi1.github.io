---
layout: page
title: sklearn中的cart分类树 
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

cart是比较决策树中一个比较经典的算法，具体的算法原理不在此解释，它用于分类，后者用于回归。

以分类树为例。

一、回顾下简单决策树的要点：

- 1.树的生长：分枝策略(or 特征选择)=>树的生成   (局部最优)
- 2.树的剪枝：剪枝策略 => 解决overfit（全局最优）

二、再回顾下C&RT：

- 1.CART是二元树，即二切分。
- 2.树的生长：分支原则：purifying. 纯不纯?
- 3.树的剪枝：

树的生成（from台大ML课程）：
<figure>
	<a href="http://photo.yupoo.com/wangdren23/FvKjz6s4/medish.jpg"><img src="http://photo.yupoo.com/wangdren23/FvKjz6s4/medish.jpg" alt=""></a>
</figure>

树的剪枝：

<figure>
	<a href="http://photo.yupoo.com/wangdren23/FvKiNvsQ/medish.jpg"><img src="http://photo.yupoo.com/wangdren23/FvKiNvsQ/medish.jpg" alt=""></a>
</figure>


我们都知道[gini系数](http://baike.baidu.com/link?url=IToPYLzxGl_nGQ30Axpz4HJMHbRKvo47HE_Qvx0S-8chg_W4ix8kzU6KOFzyk1WLec4WGEl4SOHolsOy_r_ml_)在经济学里用来衡量收入分配是否不均。gini系数越大，贫富差距越大。cart分类树采用gini系数来衡量一个节点纯不纯，其分支目标即是最小化gini系数。cart回归数采用的还是平方误差，此处不讨论。

剪枝采用正则项进行判断。

详细可参考李航的《统计学习方法》.

# 2. cart实现

scikit-learn上也有一个它的实现 DecisionTreeClassifier/DecisionTreeRegressor。本文以它的应用，做为一个示例。

sklearn提供了一个CART的优化实现，但仍有个较大的总题：就是剪枝部分没有实现。

tree的实现在我的mac目录下，为：
/Library/Python/2.7/site-packages/scikit_learn-0.15.2-py2.7-macosx-10.9-intel.egg/sklearn/tree/tree.py


其中，部分实现是用cython写的，可以github上看到其源码。

在_tree.pyx中，是一棵二元树（binary decision tree）。由并列的数组一起表示。

- node_count : 节点总数目。包括：internal nodes + leaves

- capacity：<=node_count
- max_depth: 最大深度
- children_left：children_left[i]表示node i的左节点.
- children_right:children_right[i]表示node i的右节点.
- feature: feature[i]表示internal node i进行分割的feature.
- threshold:threshold[i]表示internal node i的threshold.
- value:包含了每个节点的预测值
- impurity：impurity[i]表示node i的不纯度impurity.
- n_node_samples：n_node_samples[i]表示node i下所有的训练样本数.

- weighted_n_node_samples: weighted_n_node_samples[i]表示node i的样本权重。

上面是树的定义。

那么，在训练时(fit)，需要注意什么呢？

| 参数                     | 说明                                                                                                                                                                                                                                         |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| criterion                | 用来权衡划分的质量。缺省‘gini’： 即 Gini impurity。 或者‘entropy’： 信息增益. 其实现可参考： _creterion.pyx, 包含了分类和回归.缺省为：gini.                                                                                                  |
| splitter                 | 划分方式有三种：best, presort-best, random. 它的实现在： _splitter.pyx，它会根据criterion对每个节点进行划分. best是最优划分，random: 随机划分. 缺省为：best                                                                                  |
| max_features             | 当进行best划分时，会考虑max_features. 缺省: None                                                                                                                                                                                             |
| max_depth                | 树的最大深度。缺省为：None                                                                                                                                                                                                                   |
| min_samples_split        | 对于一个中间节点（internal node），必须有min个samples才对它进行分割。缺省为：2                                                                                                                                                               |
| min_samples_leaf         | 对一个叶子节点（left node），必须有min个samples认为它是叶子节点。缺省为：1                                                                                                                                                                   |
| min_weight_fraction_leaf | 在一个叶子节点上，输入样本必须有min个最小权重块。缺省为：0                                                                                                                                                                                   |
| max_leaf_nodes           | 以最好优先（best-first）的方式使用该值生成树。如果为None:不限制叶子节点的数目。如果不为None，则忽略max_depth。缺省为：None                                                                                                                   |
| class_weight             | 分类和权重以这种形式关联在一起：{class_label: weight}。如果示给定，那么所有的分类都认为具有权重1. 对于多分类总题，可以给出一个list of dicts，顺序与y列一致。“balanced”模式：自动调整权重。n_samples / (n_classes * np.bincount(y))详见文档。 |
| random_state             | 随机种子                                                                                                                                                                                                                                     |
| presort                  | bool, 是否对数据进行预先排序，以便在fitting时加快最优划分。对于大数据集，使用False，对于小数据集，使用True.                                                                                                                                  |


## 2.1 模型及参数

cart, 可参考：[sklearn 决策树](http://scikit-learn.org/stable/modules/tree.html)

## 2.2 模型评价


模型评价部分，可参考：[sklearn模型评测](http://scikit-learn.org/stable/modules/model_evaluation.html)

## 2.3 参数优化及选择

sklearn中，使用[Grid Search](http://scikit-learn.org/stable/modules/grid_search.html)对假设函数的参数进行最优化。列出你要测试的参数，然后Grid Search使用穷举搜索（exhaustive search）的方式，遍历你的模型参数组合，来训练和评估你的模型。因而，它的计算代价比较高昂。为此，sklearn内置提供了并行化实现。而RandomizedSearchCV则以特定分布的方式进行抽样和搜索。

一个Grid Search包含了：

- 1.一个estimator
- 2.一个参数空间
- 3.用于抽样或搜索修选参数的方法: GridSearchCV和RandomizedSearchCV
- 4.cross-validation的scheme
- 5.score函数


以一个示例作为解释，下面的示例会遍历288种参数组合，然后选出其中一个最优的参数，这里的判断标准是cross-validation中的选的score方式。

{% highlight python %}

#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import tree

from sklearn.datasets import load_iris

from matplotlib import pyplot
import scipy as sp
import numpy as np
from matplotlib import pylab

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

print(__doc__)

# Loading the Digits dataset
iris = load_iris()

X = iris.data 
y=iris.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.75, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }


clf = tree.DecisionTreeClassifier()

print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(clf, tuned_parameters, cv=10)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()

print("use the best estimator to predict...")

y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

{% endhighlight %}

随机参数优化：RandomizedSearchCV

它通过参数进行随机搜索，每一参数设定会通过一个参数值分布进行抽样。对比穷举法，具有两个优势：

- 1.参数个数和可能的值可以独立可以选择budget
- 2.增加参数不会影响性能，不会降低效果

参数设定部分和GridSearchCV类似，使用一个字典表来进行参数抽样。另外，计算开销（computation budget）, 抽取的样本数，抽样迭代次数，可以由n_iter来指定。对于每个参数，都可以指定在可能值上的分布，或者是一个离散值列表（均匀采样）。

例如：

[{'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
  'kernel': ['rbf'], 'class_weight':['auto', None]}]

这个例子使用scipy.stats模块，该模块包含了许多分布方法可以用来抽样，包括：指数分布（expon），gamma分布(gamma)，均匀分布（uniform），或randint分布。通常每个函数都可以提供一个rvs（随机变量抽样）方法进行抽样。

使用RandomizedGrid的示例如下：

{% highlight python %}
#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import tree

from sklearn.datasets import load_iris

from matplotlib import pyplot
import scipy as sp
import numpy as np
from matplotlib import pylab

from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
from time import time

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report


# Loading the Digits dataset
iris = load_iris()

X = iris.data 
y=iris.target

print X.shape
print y.shape

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.75, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = {"criterion": ["gini", "entropy"],
              "min_samples_split": sp_randint(1, 20),
              "max_depth": sp_randint(1, 20),
              "min_samples_leaf": sp_randint(1, 20),
              "max_leaf_nodes": sp_randint(2,20),
              }

clf = tree.DecisionTreeClassifier()

print("# Tuning hyper-parameters")
print()

n_iter_search = 288 
clf = RandomizedSearchCV(clf, \
        param_distributions=tuned_parameters, \
        n_iter=n_iter_search, \
        cv=10)

start = time()
clf.fit(X_train, y_train)


print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time() - start), n_iter_search))

print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()

print("use the best...")
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

{% endhighlight %}


缺省时，在search参数时，通过score来比较优劣。分类使用
sklearn.metrics.accuracy_score，回归使用sklearn.metrics.r2_score。但是有时候这并不有效，比如：分类中的倾斜类，使用f1值更好。你可以根据你的优化目标设计或选择scoring，[详见]{http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter}。

# 2.4 实际使用注意事项

- 1.如果feature数过多，那么决策树很容易overfit。样本数与特征数之比合适与否相当重要，一棵只有少量样本的树，在高维空间中很可能overfit.
- 2.可以考虑执行降维，比如：[PCA](http://scikit-learn.org/stable/modules/decomposition.html#pca)/[ICA](http://scikit-learn.org/stable/modules/decomposition.html#ica)/或[特征选取](http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection)，让你的决策树可以更好的发现特征。
- 3.可以通过export进行可视化。
- 4.使用max_depth阻止过拟合
- 5.使用min_samples_split或min_samples_leaf来控制叶结点的样本数. 节点样本数越小意味着树越容易overfit。这两者间的区别是，min_samples_leaf可以保证一个叶子节点的最小数目，而min_samples_split则可以创建更独裁的小节点。
- 6.在训练之前，平衡下你的数据集，以便阻止树偏向于大类。类的平衡可以通过对每个类抽样相等的样本数，或者通过为每个类对样本权重（sample_weight）的和进行归一化到相类似的值。注意，基于weight的事前剪枝的criteria（比如min_weight_fraction_leaf），在大类上比基于样本权重的criteria（比如：min_samples_leaf）具有更小的bias。
- 7.如果样本是带权重的，则使用基于权重的事前剪枝策略(比如：min_weight_fraction_leaf)进行结构优化很容易。
- 8.所有决策树内部都使用np.float32。如果训练数据不是这个格式，会生成该数据集的一个copy.
- 9.如果输入矩阵X非常稀疏，推荐在调用fit前转成csc_matrix，在predict前转成csr_matrix. 可以加快速度。

# 2.5 偏斜类

举个栗子，对一个二元分类来说，你得到的一个结果，可能是：


             precision    recall  f1-score   support

          0       0.96      1.00      0.98     14591
          1       0.26      0.01      0.02       613

avg / total       0.93      0.96      0.94     15204

class 0和class 1的样本的数目差异很大，对于class 0来说，这个预测的效果看起来还不错；如果求平均来看，它的precision竟然有0.93，f1有0.94，看起来也是很不错的，但是对于class 1来说，准确率为可怜的0.26，f1值更低，这显然不是一个好的预测。

对于这种问题，我们应该在sklearn中怎么去解决它？

对于二分类问题，sklearn中scoring函数缺省使用类别1的score作为评判标准。上面的问题很简单。如果你想让0类作为主类，设置pos_label参数就好。

对于多分类问题，详见scoring那一节。

# 2.5 决策树的导出

可以用dot文件导出，并用graphviz打开查看. 当训练fit完后，即可导出.

{% highlight python %}

clf = clf.fit(iris.data, iris.target)
from sklearn.externals.six import StringIO
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

{% endhighlight %}


