---
layout: post
title: starspace介绍
description: 
modified: 2018-05-02
tags: 
---

Facebook在2017年提出了《StarSpace:
Embed All The Things!》，我们可以看下：

# 介绍

StarSpace是一种神经嵌入模型（neural embedding model ），可以广泛解决多种问题：

- 文本分类，或者其它标注任务，比如：语义分类
- 实体排序，比如：给定一个query，对web文档进行排序
- 基于CF的推荐，例如：文档、音乐、或视频推荐
- 基于内容的推荐，其中，内容通过离散特征（例如：词）定义
- 图谱嵌入（mebedding graphs），例如，Freebase这样的多关系图谱
- 学习词、句、文档的embeddings

# 相关工作

隐文本表示（或称为embeddings），是在一个大语料上使用非监督学习得到的关于词、文档的向量表示。相应的方法有：Bengio 2003 embedding, Mikolov word2vec 2013, Bojanowski fastText 2017。

在监督式embeddings领域，方法有：SSI，WSABIE， (Tang, Qin, and Liu 2015), (Zhang and LeCun 2015), (Conneau et al. 2016), TagSpace(Weston, Chopra, and Adams 2014) and fastText(Joulin et al. 2016) 。

在推荐领域，embedding模型很成功，有：SVD
(Goldberg et al. 2001)， SVD++
(Koren and Bell 2015)，(Rendle 2010; Lawrence and Urtasun 2009; Shi et al. 2012)。这些方法中的一些方法主要关注于：CF的setup，其中user IDs和movie IDs具有单独的embeddings，比如：在Netflix挑战赛的setup（例如： (Koren and Bell 2015),新的users和items不能天然合并进去）。我们展示了StarSpace是如何天然地迎合该settings和基于content的settings，其中users和items可以被表示成features，具有天然的out-of-sample扩展，而非只是一个固定集合。

在知识图谱中的链接预测上。有s(Bordes et al. 2013) and (GarciaDuran, Bordes, and Usunier 2015)。StarSpace也可以应用于此，匹配TransE方法。

# 模型

StarSpace模型由学习实体（learning entities）组成，每个通过一个离散特征（bag-of-features）的集合描述。一个像（文档、句子）这样的实体可以通过bag-of-words或n-grams进行描述，一个像user这样的实体可以通过bag-of-documents、movies、items的方式进行描述。重要的是，StarSpace模型可以自由比较不同类型的实体（entities）。例如，一个用户实体可以与一个item实体、或者一个文档实体(推荐)、一个文带标签实体的文档实体等进行比较。这可以通过学习来完成，并将它们嵌入到相同的空间中，这样的比较才有意义————通过根据各自的metric进行最优化。

字典D，特征为F，那么有一个D X F的矩阵，其中$$F_i$$表示第i维特征（行），它有d维的embedding，我们可以使用$$\sum_{i \in a} F_i$$来嵌入一个实体a。

这就是说，像其它embedding模型，我们的模型通过为在该集合（称为字典，它包含了像words这样的特征）中的每个离散特征分配一个d维向量。实体由特征组成（比如：文档），被表示成一个bag-of-features，它们的embeddings可以隐式被学到。注意，一个实体可以由像单个特征（唯一）组成，比如：单个词(word)、名字(name)、user ID、Item ID。

为了训练我们的模型，我们必须学到比较实体。特别的，我们希望最小化以下的loss function：

$$
\sum_{(a,b)\inE^+, b^- \in E^-} L^{batch} (sim(a,b), sim(a,b_1^-), ..., sim(a,b_k^-))
$$

注意：

- 正实体对 positive entity pairs (a,b)的生成器来自于集合$$E^+$$。这是任务非独立的，将会在下面描述。
- 负实体$$b_i^-$$的生成器来自于集合$$E^-$$。我们使用一个k-negative sampling策略（Mikolov 2013）来为每个batch update选择k个这样的负样本对。我们会从在该实体集内随机选择，它们可能出现在相似函数的第二个参数上（例如：对于文本标注任务, a是文档，b是标签，因此我们可以从labels集合上抽样$$b^-$$）。k的因素分析将在第4部分讨论。
- 相似函数$$sim(\cdot, \cdot)$$。在我们的txxik，我们的实现有：cosine相似度和内积，可以通过一个参数进行选定。总之，对于较少数目的label features（比如：分类），它们的工作机制很像；对于更大数目（比如：句子或文档相似度）时，cosine更好。
- loss function为 $$L_{batch}$$，它比较了positive pair (a,b)，negative pairs(a, b_i^-)，其中i=1,..., k. 我们也实现了两个概率：margin ranking loss（例如：$$max(0, \mu-sim(a,b))$$，其中$$\mu$$是margin参数），negative loss loss of softmax。所有的实验都使用前者，因为表现更好。

我们使用SGD进行最优化，例如，每个SGD step是从在outer sum中$$E^+$$上的一个抽样，在多CPU上使用Adagrad，hogwild。我们也应用一个max norm的embeddings来将学到的向量限制在球半径为r的空间$$R^d$$上。

在测试时，可以使用学到的函数$$sim(\cdot,\cdot)$$来测量实体间的相似度。例如，对于分类，在测试时为给定输入a预测一个label，使用$$max_{\hat{b}} sim(a,\hat{b})$$来表示可能的label$$\hat{b}$$。或者，对于ranking，通过similarity对实体进行排序。另外，embedding向量可以直接被下游任务使用，例如：word embbeding models。然而，如果$$sim(\cdot,\cdot)$$直接满足你的应用需要，我们推荐使用StarSpace，它很擅长这一块。

接着，我们会描述该模型如何被应用到其它任务上：每个case会描述$$E^+$$和$$E^-$$是如何生成的。

$$Multiclass分类（例如：文本分类）$$：positive pair generator直接来自于一个标注数据训练集，(a,b) pairs，其中，a是文档（bags-of-words），b是labels（单个features）。负实体($$b^-$$)从可能的labels集合中被抽样。

$$Multilabel Classification$$: 在该case中，每个文档a可以具有多个正标签，其中之一在每个SGD step中从b中抽样，来实现multilabel分类。

$$基于CF的推荐$$：训练数据包含了：一个用户集合，其中每个用户通过bag-of-items进行描述（字典里的唯一特征）。positive pair生成器会选择一个user，选择a作为该user ID的唯一单个特征，以及单个item b。负实体$$b^-$$从该可能的items中进行抽样。

**基于CF的推荐，使用out-of-sample用户扩展**：经典CF的一个问题是，不能泛化到新用户上，每个user ID可以学到一个独立的mebedding。与之前方法使用相同的训练数据，可以使用StarSpace学到一个新模型。该方法中，positive pair genterator会选择一个user，选择a作为除它之外的所有item，b作为left out item。也就是说，该模型会学到估计：如果一个用户喜欢一个item，可以将该用户建模成，。

**基于内容的推荐**：该任务包含了一个用户集合，其中，每个用户通过一个bag-of-items进行描述，其中每个item通过一个bag-of-features（来自字典）进行描述。例如，对于文档推荐，每个用户通过bag-of-documents进行描述，其中，每个文档通过bag-of-words进行描述。现在，a可以被选成除它外的所有items，b表示left out item。该系统可以扩展到新items和新users上，只要两都被特征化。

**多关系知识图谱(例如：逻接预测)**：给定一个graph三元组(h,r,t)，包含了一个head conecpt：h，一个relation：r，一个tail concept t，例如：(Beyonce,´born-in, Houston），一个可以从该graph中学习embeddings。对h, r, t的实例可以被定义成字典里中唯一features。我们可以随机均匀选择：

- (i) a由bag-of-features：h和r 组成，其中b只包含t；
- (ii) a由h组成，b包含了r和t.

负实体$$b^-$$从可能的concepts中抽样得到。学到的embeddings可以通过学到的sim(a,b)被用于回答link prediction问题，比如：(Beyonce, born-in, ?) ´ or (?, born-in, Houston).

**信息检索IR（例如：文档搜索）和文档嵌入**：给定监督式训练数据，包含了（搜索关键词，相关文档）pairs，可以直接训练一个信息检索模型：a包含了搜索关键词，b是一个相关文档，$$b^-$$是另一个不相关的文档。如果只提供了非监督式训练数据，它包含了未标注文档的集合，从文档中选择a的一个方法是，作为随机关键词，b作为保留词。注意，两种方法可以隐式地学习文档嵌入，可以被用于该目的。

**学习word embedding**：我们可以使用StarSpace来学习非监督式word embeddings，它使用训练raw text组成的数据。我们会选择a作为一个窗口的词（例如：4个words，两边各两个），b作为中间词。

**学习sentence embeddings**：当你可以直接学习句子的embeddings，使用上述方法来学习word embeddings、并使用它们来嵌入看起来不是最优的。给定一个未标注文档的训练集，它们由句子组成，我们选择a和b作为来自相同文档的一个句子对(pair of sentences)；$$b^-$$是来自其它文档的句子。直觉上，句子间的语义相似度在同一个文档内共享（如果文档很长的话，也可以只选择在特定距离内的句子）。再者，embeddings将会自动为关于句子长度的words sets进行最优化，因此，训练时间会与测试时间相匹配，而使用短窗口进行训练来使用word embeddings进行特殊学习——window-based embbedings可以变差，当一个句子中的words总和变得更大时。

**多任务学习**：任何这些任务可以组合，如果它们共享在基础字典F中的一些特征时可以同时训练。例如，可以使用非监督式word/sentence embedding，并结合监督式分类，来给出半监督式学习。

## 实验

略。


# 参考

- 1.[https://arxiv.org/pdf/1709.03856.pdf](https://arxiv.org/pdf/1709.03856.pdf)
