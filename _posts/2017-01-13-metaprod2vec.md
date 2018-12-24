---
layout: post
title: meta-prod2vec介绍
description: 
modified: 2017-01-13
tags: 
---

我们来看下criteo公司的Flavian Vasile提出的《Meta-Prod2Vec Product Embeddings Using
Side-Information for Recommendation》：

# 1.介绍

Prod2Vec算法只使用由商品序列(product sequences)确立的局部商品共现信息来创建商品的distributed representations，但没有利用商品的元信息(metadata)。Prod2Vec的作者提出了一个算法扩展：以序列结构的方式利用上下文内容信息，但该方法只针对文本元信息，产生的结构是层次化的，因而会缺失一些side informations项。我们结合了在推荐上的工作，使用side information并提出了Meta-Prod2Vec，它是一个通用方法，可以以一种很简单高效的方法，来增加类别型(categorical) side information到Prod2vec模型中。将额外项信息作为side information的用法（例如：只在训练时提供），受推荐系统在实时打分时要保存的特征值数量限制的启发。这种情况下，只在训练时使用的metadata，当提升在线效果时，可以让内存占用量（memory footprint）保持常数（假设一个已经存在的推荐系统会使用item embeddings）。

在30Music的listening和playlists数据集的子集上，我们展示了我们的方法可以极大提升推荐系统的效果，实现开销和集成开销很小。

在第2节，我们会回顾相关工作。第3节，展示Meta-Prod2vec方法。第4节，展示实验实置和30Music数据集上的结果。第5节总结。

# 3.Meta-Prod2Vec

## 3.1 Prod2Vec

在Prod2Vec的paper[9]中，Grbovic等人提供了在从电子邮件的商品收据序列上使用Word2Vec算法。更正式的，结定一个商品序列的集合S: $$s = (p_1, p_2, ..., p_M), s \in S$$，目标函数是发现一个D维的真实表示$$u_p \in R^D $$，以便相似的商品可以在产生的向量空间内更接近。

原算法（Word2Vec）原本是一个高度可扩展的预测模型，用于从文本中学习词向量，属于自然语言神经网络模型。在该领域大多数工作都是基于Distributional Hypothesis[27]，它宣称在相同的上下文中出现的词更接近，即使意思不同。

该hypothesis可以被应用于更大的上下文中，比如：在线电商，音乐和媒体消费，这些服务的基础是CF算法。在CF设置中，服务的使用者（用户）被当成上下文使用，在该上下文中哪些商品共现过，从而在CF中产生经典的item 共现（co-occurence）方法。基于co-count的推荐方法和Word2Vec间的相似关系由Omer[16]确定；该作者展示了embedding方法的目标函数与矩阵分解（它包含了局部共现items的the Shifted Positive PMI）很接近，其中PMI表示Point-Wise Mutual Information：

$$
PMI_{ij} = log(\frac{X_{ij} \cdot |D|}{X_i X_j})
$$

$$
SPMI_{ij} = PMI(i, j) - log k
$$

其中$$X_i$$和$$X_j$$是item频次，$$X_{ij}$$是i和j共现的次数，D是数据集的size，k是由负样本(negatives)与正样本（positives）的比率。

### Prod2Vec Objective目标函数

在[23]中作者展示了Word2Vec的目标函数（与Prod2Vec相似），可以被重写成：给定目标商品（Word2Vec-SkipGram模型，通常在大数据集上表现很好），最小化加权交叉熵（期望和建模后的上下文商品的条件分布）。接着，条件分布的预测被建模成一个关于在目标商品和上下文商品向量间内积的softmax。

$$
L_{P2V} = L_{J|I}(\theta) = \sum_{ij} (-X_{ij}^{POS} log q_{j|i}(\theta) -(X_i - X_{ij}^{POS} log(1-q_{j|i}(\theta)))) \\ = \sum_{ij} X_i(-p_{j|i} log q_{j|i}(\theta) - p_{\neg j|i} log q_{\neg j|i}(\theta)) \\ = \sum_i X_i H(p_{\cdot | i}, q_{\cdot|i}(\theta))
$$

这里，$$H(p_{\cdot \mid i}, q_{\cdot \mid i}(\theta))$$是期望概率$$p_{\cdot \mid i}$$的交叉熵，表示基于输入商品$$i \in I$$和预测条件概率$$q_{\cdot \mid i}$$, 在输出空间J上看过的任何商品：

$$
q_{j|i}(\theta) = \frac{e^{w_i^T w_j}} { e^{w_i^T w_j} + \sum_{ j' \in (V_{J-j})} e^{W_i^T W_j'}}
$$

其中，$$X_i$$表示商品i的输入频次，$$X_{ij}^{POS}$$是商品对(product pair)(i,j)在训练数据中被观察到的频次数目。

<img src="http://pic.yupoo.com/wangdren23_v/e8131d66/medium.png">

图一: Prod2Vec架构

对于Prod2Vec产生的结构，如图1所示，使用一个只有单个hidden layer和一个softmax output layer的NN，位于中心窗口的所有商品的输入空间被训练成用于预测周围商品的值。

然而，由Prod2Vec生成的商品embeddings，只考虑了用户购买序列的信息，也就是局部共现信息（local co-occurrence information）。尽管这比在协同过滤（CF）中的全局共现频次更加丰富，它不会考虑其它类型的item信息（比如：item的metadata）。

例如，假设输入是关于已经被归好类商品的序列，标准的Prod2Vec embeddings不会建模以下的交互：

- 给定关于商品p（属于类别c）的当前访问，下一个访问的商品$$p'$$更可能属于相同的类别c
- 给定当前类别c，下一个更可能的访问类别是c，或者一个相关的类别$$c'$$（例如：在购买一件游泳衣后，很可能会有一个防晒油的购买行为，它们属于一个完全不同的商品类目，但相近）
- 给定当前商品p，下一个类目更可能是c或者一个相关类目$$c'$$
- 给定当前类别c，被访问的当前商品更可能是p或$$p'$$

前面提取，作者对Prod2Vec算法作为扩展，会同时考虑商品序列和商品文本信息。如果将该扩展方法应用于非文本元数据上，加上商品预列信息，该算法会建模商品元数据和商品id间的依赖，但不会将元数据序列和商品id序列连接在一起。

## 3.2 Meta-Prod2Vec

在第一节，已经有相关工作使用side information进行推荐，尤其是结合CF和CB的混合方法。在embeddings的方法中，最相关的工作是Doc2Vec模型，其中words和paragraph会进行联合训练（jointly），但只有paragraph embedding会被用于最终的任务中。

我们提出了相似的架构，在NN的输入空间和输出空间中同时包含side information，在嵌入的items和metadata间的交互相互独立进行参数化，如图2所示。

<img src="http://pic.yupoo.com/wangdren23_v/3476895d/medium.png">

图二: Prod2Vec架构

### Meta-Prod2Vec目标函数

Meta-Prod2Vec的loss扩展了Prod2Vec的loss，它会考虑4种涉及item元信息的额外交互项：

$$
L_{MP2V} = L_{J|I} + \lambda \times (L_{M|I} + L_{J|M} + L_{M|M} + L_{I|M}) 
$$

其中：M是元数据空间（例如：在30Music数据集中的artist ids），$$\lambda$$是正则参数。我们列出了新的交互项：

- $$L_{I \mid M}$$：给定输入商品的元信息M，所观察到的输入商品ids的条件概率 与 其预测的条件概率间的加权交叉熵。该side-information与其它三种类型不同，因为它会将item建模成关于它的metadata的函数。这是因为，在大多数情况下，该item的metadata与id更通用，可以部分解释指定id的observation。
- $$L_{J \mid M}$$: 给定输入商品的元信息M，所观察到的它的上下文商品id的条件概率 与 其预测的条件概率间的加权交叉熵. 一个架构是，正常的Word2vec loss被看成是，只有该交叉项与Doc2Vec模型提出的很接近，其中，我们可以使用一个更通用类型的item metadata来替代document id information。
- $$L_{M \mid I}$$: 给定输入商品I，它的上下文商品元信息值M的条件概率、与预测的条件概率 间的加权交叉熵。
- $$L_{M \mid M}$$: 给定输入商品的元信息M，它的上下文商品元信息值M的条件概率，与预测的条件概率 间的加权交叉熵。该模型会建模观察到的metadata序列，以及在它内表示关于metadata的Word2Vec-like的embedding。

总之，$$L_{J \mid I}$$和$$L_{M \mid M}$$会分别对items序列和metadata序列的似然建模进行编码loss项。$$L_{I \mid M}$$表示在给定元信息的情况下item id的条件似然，$$L_{J \mid M}$$和$$L_{M \mid I}$$表示在item ids和metadata间的cross-item交叉项。如图3如示，我们展示了由Prod2Vec因子分解出的item matrix，以及另一个由Meta-Prod2Vec分解出的item matrix。

<img src="http://pic.yupoo.com/wangdren23_v/0c301967/medium.png">

图三：将MetaProd2Vec看成是items和metadata扩展矩阵的矩阵分解

Meta-Prod2Vec的更通用等式，为4种类型的side information（$$\lambda_{mi}, \lambda_{jm}, \lambda_{mm}, \lambda_{im}$$）引入了一个独立的$$\lambda$$。

在第4节，我们将分析每种类型的side-information的相对重要性。当使用多种源的metadata时，每种源在全局loss中将具有它自己的项以及它自己的正则参数。

根据softmax正则因子，我们有机会将items的输出空间和metadata的输出空间选择是否进行独立开来。与在Word2Vec中使用的简化假设相似，这允许每个共现的商品对（product pairs）被可以被独立地进行预测（predicted）和拟合（fitted）（因此，给定一个输入商品，在输出商品集上添加一个隐式的相互排斥限制），我们在相同的空间中嵌入该商品和它的metadata，因此允许它们共享归一化限制(normalization constraint)。

Word2Vec算法的一个吸引点是它的可扩展性，它使用Negative Sampling loss在所有可能词的空间上近似原始softmax loss，可以只在正共现上使用抽样的少量负样本来拟合模型，最大化一个修改版本似然函数$$L_{SG-NS}(\theta)$$：

$$
L_{J|I}(\theta) = \sum_{ij} (- X_{ij}^{POS} log q_{j|i}(\theta) - (X_{ij}^{NEG} log(1 - q_{j|i}(\theta))) \\ \approx L_{SG-NS}(\theta)
$$

其中：

$$
L_{SG-NS}(\theta) = \sum_{ij} - X_{ij}^{POS} (log \sigma(w_i^T w_j)) - k E_{c_N ~ P_D} log \sigma(-w_i^T w_N))
$$

其中，$$P_D$$概率分布用于抽样负上下文样本，k是一个超参数，它指定了每个正样本对应的负样本的数目。side information loss项$$L_{I \mid M}, L_{J \mid M}, L_{M \mid I}, L_{M \mid M}$$根据相同的公式进行计算，其中$$i,j$$分别索引input/output空间。

在Meta-Prod2Vec中，将商品(products)和元信息(metadata)共同嵌入（co-embed）到$$L_{SG-NS}(\theta)$$loss中的影响是，对于任意正样本对（positive pair）潜在的负样本集合，会包含items和metadata值的联合。

### 最终对推荐系统的影响

由于共享embedding空间，用于Prod2Vec的训练算法保持不变。唯一一不同是，在新版本的训练对（training pairs）的生成阶段，原始item对会得到涉及metadata的额外pairs的协助。在线推荐系统中，假设我们增加一个涉及item embeddings的解决方案，在线系统不会增加任何改变（因为我们只在训练时使用metadata），对在线内存占用没有任何影响。

# 4.实验

如下。首先描述了评估任务设置，metrics和baselines。接着，我们报告了在30Music开放数据集上的实验。

## 4.1 Setup

我们评估了在事件预测任务（next event prediction task）上的推荐方法。我们考虑用户与items交叉的时间顺序序列。我们将每个序列分割成训练集、验证集、测试集。我们将拟合Prod2Vec和Meta-Prod2Vec模型的embedding，在每个用户序列的前n-2个元素上，在第n-1个元素上测试超参数效果，最终通过训练前n-1个items，并预测第n个item。

我们使用在训练序列中的最后一个item作为query item，我们使用下述的其中一种方法来推荐最相似的商品。

在第1节所示，由于技术限制，需要让内存占用保持常量，我们只在训练时对item的metadata感兴趣。因此，我们不会与将metadata直接用于预测时的方法做比较，比如：先前的基于内容的推荐（CB）embedding方法，用户和item被表示成item content embeddings的线性组合，其中商品通过关联的图片内容embeddings进行表示。

我们使用以下的评估metrics，对所有用户做平均：

- K上的点击率(HR@K)：如果测试商品出现在推荐商品的top K列表中，它就等于1/K。
- Normalized Discounted Cumulative Gain(NDCG@K): 在推荐商品列表中，测试商品有更高的ranks。

使用上述metrics，我们比较了以下方法：

- BestOf：top产品按它们的流行度进行检索。
- CoCounts: 标准CF方法，它使用共现向量与其它items的cosine相似度。
- Standalone Prod2Vec: 通过Word2Vec在商品序列上获取的向量进行cosine相似度计算得到推荐结果。
- Standalone Meta-Prod2Vec: 它增强了Prod2Vec，使用item side information，并使用生成的商品embedding来计算cosine相似度。和Prod2Vec一样，目标是进一步解决冷启动问题。
- Mix(Prod2Vec,CoCounts): 一个ensemble方法，它返回使用两者线性组合来返回top items。 $$Mix(Prod2Vec, CoCounts)= \alpha * Prod2Vec + (1-\alpha) * CoCounts $$
- Mix(Meta-Prod2Vec,CoCounts): 一个ensemble方法，$$Mix(MetaProd2Vec, CoCounts)= \alpha * MetaProd2Vec + (1-\alpha) * CoCounts $$， 它返回使用两者线性组合来返回top items。

## 4.2 数据集

30Music dataset：它从Last.fm API中抽取的一个关于listening和playlists的集合。在该数据集上，我们评估了关于推荐下一首歌预测的推荐方法。对于meta-prod2vec算法，我们利用track metadata，命名为artist信息。我们在一个100k用户sessions数据集的样本上运行实验，生成的vocabulary size为433k首歌和67k artists。

## 4.3 结果

见paper本身。


# 参考

- 1.[Meta-Prod2Vec - Product Embeddings Using
Side-Information for Recommendation](https://arxiv.org/pdf/1607.07326.pdf)
