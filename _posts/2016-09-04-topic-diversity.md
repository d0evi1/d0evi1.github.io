---
layout: post
title: topic diversity介绍
description: 
modified: 2016-09-04
tags: 
---

Joseph A. Konstan教授(coursera.明大.推荐系统课程)在2005年《Improving Recommendation Lists Through Topic Diversification》中提出了主题多样性。虽然这篇paper比较老，但比较经典，值得一看：


# 1.介绍

推荐系统的目标是：基于用户过往偏好、历史购买、人口统计学信息（demographic information），提供用户关于他们可能喜欢的产品的推荐。大多数成功的系统会利用CF（collaborative filtering），许多商业系统（Amazon.com）利用这些技术来提供个性化推荐列表给他们的顾客。

尽管state-of-the art CF系统的accuracy很出色，实际中已经发现：一些可能的结果会影响用户满意度。在Amazon.com上，许多推荐看起出在内容上很“相似（similar）”。例如，顾客已经购买了许多关于赫尔曼·黑塞（Hermann Hesse）的诗集，可能会获得这样的推荐列表：其中所有top-5的条目都只包含了该作者的书。当纯粹考虑accuracy时，所有这些推荐看起来都是出色的，因为该用户很明显喜欢赫尔曼·黑塞写的书。另一方面，假设，该活跃用户在赫尔曼·黑塞之外还有其它兴趣，比如：历史小说以及世界游行方面的书，那么该item推荐集合就看起来较差了，缺乏多样性。

传统上，推荐系统项目会关注于使用precision/recall 或者MAE等metrics来优化accuracy。现在的一些研究开始关注pure accuracy之外的点，真实用户体验是必不可少的。

## 1.1 贡献

我们通过关于真实用户满意度，而非pure accuracy，来解决之前提到的不足。主要贡献如下：

- 主题多样性(topic diversification)：我们提出了一种方法，根据该活跃用户的完整范围的兴趣，来平衡top-N推荐列表。我们的新方法会同时考虑：suggestions给出的accuracy，以及在特定主题上的用户兴趣范围。主题多样性的分析包含：user-based CF和item-based CF.
- Intra-list相似度指标（similarity metric）：
- accuracy vs. satisfaction：

# 2.CF

略

# 3.评估指标

为了判断推荐系统的质量和效果，评估指标是必要的。许多评价只关注于accuracy并忽略了其它因素，例如：推荐的novelty（新奇）和serendipity（意外发现），以及推荐列表items的diversity。

以下给出了一些流行的指标。

## 3.1 Accuracy Metrics

Accuracy metrics主要有两个：

第1, 为了判断单个预测的accuracy（例如：对于商品$$b_k$$的预测$$w_i(b_k)$$与$$a_i$$的accuracy ratings $$r_i(b_k)$$有多偏离）。这些指标特别适合于，预测会随商品被展示的任务（比如：annotation in context）

### 3.1.1 MAE

### 3.1.2 RECALL/Precision

## 3.2 Accuracy之外

尽管accuracy指标很重要，还有其它不能被捕获的用户满意度特性。non-accuracy metrics与主流的研究兴趣相距较远。

### 3.2.1 覆盖度（Coverage）

在所有non-accuracy评估指标上，coverage是最常见的。coverage指的是：对于要预测的问题域（problem domain）中元素（elements）部分的百分比。

### 3.2.2 Noverlty和Serendipity

一些推荐器会生成高度精准的结果，但在实际中没啥用（例如：在食品杂货店中给顾客推荐香焦）。尽管高度精准，注意，几乎每人都喜欢和购买香焦。因此，他们的推荐看起来太过明显，对顾客没啥太多帮助。

Novelty和serendipity指标可以衡量推荐的"非显而易见性（non-obviousness）"，避免“随机选取（cherry picking）[12]”。对于serendipity的一些简单measure，可以采用推荐items的平均流行度。分值越低表示越高的senrendipity。

## 3.3 Intra-List Similarity

我们提出一个新指标，它的目的是捕获一个list的diversity。这里，diversity指的是所有类型的特性，例如：genre、author、以及其它的有辩识度的特性。基于一个任意函数（arbitrary function）：$$c_o: B \times B \rightarrow [-1, +1] $$，来衡量在商品$$b_k, b_e$$间的相似度$$c_o(b_k, b_e)$$，我们将$$a_i$$的推荐列表$$P_{w_i}$$的intra-list similarity定义如下：

$$
ILS(P_{w_i}) = \frac{\sum\limits_{b_k \in \Im P_{w_i}} \sum\limits_{b_e \in \Im P_{w_i}, b_k \neq b_e} c_o(b_k, b_e)}{2}
$$

...(5)

分值越高表示越低的diversity。我们会在后面涉及到的关于ILS的一个令人感兴趣的数学特性：排列不敏感（permutation-insensitivity），例如：$$S_N$$是关于在$$N=\|P_{w_i}\|$$的所有排列的对称分组（symetric group）：

$$
\forall \delta_i, \delta_j \in S_N: ILS(P_{w_i} \circ \delta_i) = ILS(P_{w_i} \circ \delta_j)
$$

...(6)

这里，通过在一个top-N list $$P_{w_i}$$上对推荐的位置进行简单重设置，不会影响$$P_{w_i}$$的intra-list similarity。

# 4.topic diversification

acurracy指标的一个主要问题是，它不能捕获更宽泛的用户满意度，会隐藏在已存在系统中的一些明显缺陷。例如，推荐出一个非常相似items的列表（比如：对于一个很少使用的用户，根据author、genre、topic推出），尽管该列表的平均accuracy可能很高。

该问题在之前被其它研究者觉察过，由Ali[1]创造了新词“投资组合效应（portfolio effect）”。我们相信：item-based CF系统特别容易受该效应的影响。从item-based TV recommender TiVo[1]、以及Amazon.com recommender的个性化体验都是item-based。例如，这篇paper的作者只获得了关于Heinlein的书籍推荐，另一个则抱怨推荐的书籍全是Tolkien的写作。

在用户满意户上的负面分歧的原因暗示着，“protfolio effects”是很好理解的，已经在经济学界被广泛研究，术语为“边际收益递减规律（law of diminishing marginal returns）【30】”。该规律描述了饱和效应（saturation effects）：当商品p被不断地获得（acquired）或消费（consumed）时，它的增量效用（incremental utility）会稳定的递减。例如，假设你被提供了你喜欢的酒。假设：$$p_1$$表示你愿意为该商品支付的价格。假设你被提供了第二杯特别的酒，你趋向于花费的的金额$$p_2$$会越来越少，（例如：p_1 > p_2）。以此类推：$$p_3, p_4$$。

我们提出了一种方法“topic diversification”来处理该问题，并便推荐列表更多样化，更有用。我们的方法是现有算法的一个扩展，可以被应用到top推荐列表上。

## 4.1 Taxonomy-based Similarity指标

函数：$$c^*: 2^B \times 2^B \rightarrow [-1,+1]$$，表示在两个商品集合之间的相似度，这构成了topic diversification的一个必要部分。对于taxonomy-driven filtering[33]，我们使用我们的指标对$$c^*$$实例化，尽管其它content-based similarity measures可能看起来也适合。在商品集合间的相似度计算指标基于他们的分类（classification）得到。每个商品属于一或多个分类，它们以分类的taxonomies进行层次化排列，以机器可读的方式描述了这些商品。

classification taxonomies存在不同领域。Amazon.com为books/DVSs/CDs/电子产品等制作了非常大的taxonomies。图1表示一个sample taxonomy。另外，在Amazon.com上的所有商品的内容描述与它们的domain taxonomies相关。特征化的主题可以包含：author、genre、audience。

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/4b23454112e2389e3f8cafacde407483b06df0302c22862265192dbe0029849a17f42c9bd8e4f4e2157945c318f2e5dc?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1

## 4.2 topic diversification算法

算法1展示了完整的topic diversification算法。

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/04881ae27b5d1feaccfef6ff93810d5988f6b20517204d34ada2306e2af39fe6a1254e9d0b2e40457f77a2329f4d1138?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=a1.jpg&amp;size=750">

算法1

函数$$P_{w_i *}$$表示新的推荐列表，它由使用topic diversification产生。对于每个list entry $$z \in [2,N]$$，我们从候选商品集$$B_i$$中收集了这些商品b，它们不会出现在在$$P_{w_i *}$$中的位置$$o < z$$上，并使用集合$$\lbrace P_{w_i *}(k) \mid k \in [1,z] \rbrace$$来计算相似度，它包含了所有新推荐的前导rank z。

根据$$c^*(b)$$以逆序方式对所有商品b进行排序（sorting），我们可以获得不相似rank（dissimilarity rank） $$P_{c^*}{rev}$$。该rank接着会使用原始推荐rank $$P_{w_i}$$根据多样化因子$$\Theta_F$$进行合并，生成最终rank $$P_{w_i *}$$。因子$$\Theta_F$$定义了dissimilarity rank $$P_{c^*}^{rev}$$应用在总体输出上的影响（impact）。大的$$\Theta_F \in [0.5, 1]$$喜欢多样性（diversification）胜过$$a_i$$的原始相关顺序，而较低的$$\Theta_F \in [0, 0.5]$$生成的推荐列表与原始rank $$P_{w_i}$$更接近。对于实验分析，我们使用diversification因子：$$\Theta_F \in [0, 0.9]$$。

注意，有序的input lists $$P_{w_i}$$必须是大于最终的top-N list。对于我们的后续实验，我们会使用top-50 input lists来产生最终的top-10推荐。

## 4.3 推荐依赖

为了实验topic diversification，我们假设：推荐商品$$P_{w_i}(o)$$和$$P_{w_i}(p), o, p \in N$$，和它们的内容描述一起，会产生一个相互影响，它会被现有方法所忽略：通常，对于推荐列表的items来说，只有相关度加权排序（relevance weight ordering） $$o < p \Rightarrow w_i(P_{w_i}(o)) \geq w_i(P_{w_i}(p))$$必须保持，假设没有其它依赖。

在topic diversification的case中，推荐相互依赖意味着：一个item b与之前推荐间的当前dissimilarity rank，会扮演一个重要角色，可能影响新的ranking。

## 4.4 渗透压（Osmotic Pressure）类比

dissimilarity效应与分子生物学上的渗透压和选择性渗透（selective permeability）相类似。将商品$$b_o$$（它来自兴趣$$d_o$$的特定领域）稳定插入到推荐列表中，等价于：从来自一个特定物质的分子通过细胞膜传到细胞质中。随着浓度$$d_o$$（它属于膜的选择性通透性）的增加，来自其它物质d的分子b的压力会上升。对于一个给定主题$$d_p$$，当压力(pressure)足够高时，它最好的商品$$b_p$$可能“散开（diffuse）“到推荐列表中，尽管他们的原始排序（original rank）$$P_{w_i}^{-1}(b)$$可能不如来自流行域（prevailing domain）$$d_o$$。相应的，对于$$d_p$$的压力会下降，为另一个压力上升的domain铺路。

这里的topic diversification类似于细胞膜的选择性渗透性，它允许细胞来将细胞质的内部组成维持在所需要级别上。

# 参考

- 1.[Improving recommendation lists through topic diversification](https://dl.acm.org/citation.cfm?id=1060754)
