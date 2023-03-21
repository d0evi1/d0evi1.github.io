---
layout: post
title: google SSL双塔模型介绍
description: 
modified: 2022-05-10
tags: 
---


google在2021《Self-supervised Learning for Large-scale Item Recommendations》中在双塔模型中使用了SSL：


# 介绍

在本paper中关注的推荐任务是：给定一个query，从一个很大的item catalog中找出与该query最相关的items。large-scale item推荐的问题已经被多个应用广泛使用。一个推荐任务可以根据query类型划分为：

- i) 个性化推荐： 此时query是一个用户
- ii) item to item推荐：此时query是一个item
- iii) 搜索（search）：此时query是一个自由文本片段

为了建模一个query和一个item间的交叉，广告使用的方法是：embedding-based neural networks。推荐任务通常被公式化为一个极端分类问题：其中每个item被表示成在output space中的一个dense vector。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/e635c8c69de7b04db764f8679d51e4761195b2696b801784c14acf1ea289781bd37aa5cbdda663d7be68da932e0da2e1?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1 

该paper主要关注于two-tower DNNs（见图1），它在许多真实推荐系统中很常见。在本架构中，一个neural network会将item features集合编码成一个embedding，使得它可以用于检索cold-start items。另外，two-tower DNN结构使得可以实时serving大量的items，将一个top-k NN搜索问题转成一个NIPS（Maximum-Inner-Product Search），可以在sublinear的复杂度中求解。

Embedding-based deep models通常具有大量参数，因为他们会使用高维embeddings进行构建，可以表示高维稀疏features，比如：topics或item IDs。在许多已经存在的文献中，训练这些模型的loss functions会被公式化为一个监督学习问题。监督学习需求收集labels（比如：clicks）。现代推荐系统会从用户侧收集数十亿的footprints，并提供大量的训练数据来构建deep models。然而，当它建模一个大量items问题时，数据仍会非常稀疏，因为：

- 高度倾斜的数据分布：queries和items通常会以一个二八分布（power-law）方式高度倾斜。因此，只有一小部分流行的items会获得大量的交互。这使得长尾items的训练数据非常稀疏。
- 缺乏显式用户反馈：用户经常提供许多隐式正向反馈，比如：点击或点赞。然而，像显式反馈（item评分、用户喜好度反馈、相关得分）等非常少。

自监督学习（SSL：Self-supervised learning）会提供一个不同的视角来通过unlabeled data来提升deep表征学习。基本思想是：使用不同的数据增强（data augmentations）来增强训练数据，监督任务会预测或重新设置原始样本作为辅助任务（auxiliary tasks）。自监督学习（SSL）已经被广泛用在视觉、NLP领域。CV中的一个示例是：图片随机旋转，训练一个模型来预估每个增强的输入图片是如何被旋转的。在NLU中，masked language任务会在BERT模型中被引入，来帮助提升语言模型的预训练。相似的，其它预训练任务（比如：预估周围句子、将wikipedia文章中的句子进行链接）已经被用于提升dual-encoder type models.对比起常规则的监督学习，SSL提供互补目标（complementary objectives），来消除人工收集labels的前提。另外，SSL可以利用input features的内部关系来自主发现好的语义表示。

SSL在CV和NLP中的广泛使用，但在推荐系统领域还没有被研究过。最近的研究[17,23,41]，会研究一些正则技术，它用来设计强制学到的表示（例如：一个multi-layer perception的output layer），不同样本会相互远离，在整个latent embedding space中分散开来。尽管SSL中共享相似的精神，这些技术不会显式构建SSL任务。对比起CV和NLU应用中的建模，推荐模型会使用极度稀疏的input，其中高维categorical featurers是one-hot（）编码的，比如：item IDs或item categories。这些features通常被表示为在深度模型中的可学习embedding vectors。由于在CV和NLU中大多数模型会处理dense input，创建SSL任务的已存在方法不会直接应用于在推荐系统中的稀疏模型。最近，一些研究研究SSL来改进在推荐系统中的sequential user modeling。。。。

# 2.相关工作

。。。

# 3.方法

我们使用自监督学习框架来进行DNN建模，它会使用大词汇表的categorical features。特别的，一个常见的SSL框架会在第3.1中引入。在第3.2节中，我们提出一个数据增强方式，用来构建SSL任务，并详述 spread-out regularization的connections。最后，在第3.3节中，我们通过一个multitask learning框架，描述了如何使用SSL来改进factorized models（图1中的 two-tower DNNs）。

## 3.1 框架

受SimCLR框架的启发，对于visual representation learning，我们采用相似的对比学习算法来学习categorical features的表示。基本思想是两部分：首先，对于相似的训练样本，我们使用不同的数据增强来进行学习表示；接着使用contrastive loss函数来鼓励从相同训练样本中学习的representations是相似的。contrastive loss会被用于训练two-tower DNNs（见：[23, 39]），尽管这里的目标是：使得正样本（postive item）会与它相应的queries一致。

我们会考虑关于N个item样本的一个batch：$$x_1, \cdots, x_N$$，其中：$$x_i \in X$$表示了样本i的一个特征集合，在推荐系统场景中，一个样本由一个query、一个item或一个query-item pair构成。假设：存在一个关于转换函数的pair：$$h, g: X \rightarrow X$$，它会将$$x_i$$分别增强为$$y_i, y_i'$$：

$$
y_i \leftarrow h(x_i), y_i' \leftarrow g(x_i)
$$

...(1)

给定关于样本i的相同输入，我们希望学习在增强后（augmentation）不同的表示$$y_i, y_i'$$，确认模型仍能识别$$y_i$$和$$y_i'$$代表相同的input i。换句话说，contrastive loss会学习最小化在$$y_i, y_i'$$间的不同。同时，对于不同的样本i和j，contrastive loss会最大化在数据不同增强后在学到的$$y_i, y_i'$$间的representations的不同。假设：$$z_i, z_i'$$表示：在由两个nueral networks $$H, G: \rightarrow R^d$$编码后$$y_i, y_i'$$的embeddings，也就是说：

$$
z_i \rightarrow H(y_i), z_i' \leftarrow G(y_i')
$$

...(2)

我们将$$z_i, z_i'$$看成是postive pairs，$$(z_i, z_j')$$看成是negative pairs，其中：$$i \neq j$$。
假设$$s(z_i, z_j') = <z_i,z_j'> / \| z_i \| \cdot \|z_j'\|$$。为了鼓励上述的属性，我们将一个关于N个样本$$\lbrace  x_i \rbrace$$的batch的SSL loss定义为：

$$
L_{self} ( \lbrace x_i \rbrace; H, G) := -\frac{1}{N} \sum\limits_{i \in [N]} log \frac{exp(s(z_i, z_i')) / \tau}{\sum\limits_{i \in [N]} exp(s(z_i, z_j'))/ \tau}
$$

...(3)

其中：$$\tau$$是一个对于softmax temperature可调的超参数。上述的loss function学习了一个健壮的embedding space，使得：在数据增强后相似items会相互更近些，随机样本则会被push更远。整个框架如图2所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/d3acfa0a979fb7e545601b12960598267648e850559562452436a8ea62e189fc9f9c71adc566efa468ed3f16772a6fac?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图2

**Encoder结构**

对于categorical features的输入样本，H, G通常使用一个input layer构建，在这之前使用一个multi-layer perceptron（MLP）。input layer通常是一个关于normalized dense features和多个sparse feature embeddings的concatenation，其中，sparse feature embeddings是学到的representations，会存储在embedding tables中（作为对比，CV和语言模型的input layers，会直接作用于raw inputs中）。为了使用SSL对监督学习更容易，对于network H和G，我们会共享的sparse features的embedding table。这取决于数据增强(h, g)的技术，H和G的MLPs可以被完全或部分共享。

**Connection with Spread-out Regularization**

在特例中，其中：(h,g)是identical map, H,G是相同的neural network，在等式(3)中的loss function接着被减为：

$$
- N^{-1} \sum_i log \frac{exp(1/\tau)}{ exp(1 / \tau) + \sum_{j \neq i} exp(s(z_i, z_j) / \tau)}
$$

它会鼓励不同样本的representations具有很小的cosine相似度。该loss与[41]中引入的e spread-out regularization相近，除了：原始提出会使用square loss，例如：$$N^{-1} \sum_i \sum_{j \neq i} <z_i, z_j>^2$$，会替代softmax。spread-out regularization已经被证明是会改进大规模检索模型的泛化性。在第4节中，我们展示了，引入特定的数据增强，使用SSL-based regularization可以进一步提升模型效果。

## 3.2 two-stage data augmentation

我们引入了数据增强，例如，在图2中的h和g。给定一个item features的集合，关键思想是：通过将部分信息进行masking，创建两个augmented examples。一个好的transformation和data augmentation应在该数据上做出最小量的假设，以便它可以被应用于大量任务和模型上。masking的思想是，受到在BERT的Masked Language Modeling的启发。不同于sequential tokens，features的集合不会有顺序，使得masking方式是一个开放问题， 我们会通过探索特征相关性（feature corelation）来找到masking模式。我们提出了相关特征masking（Correlated Feature Masking (CFM)），通过知道feature correlations，对于categorical features进行裁剪。

在详究masking细节前，我们首先提出一种two-stage augmentation算法。注意，无需augmentation，input layer会通过将所有categorical featuresr embeddings进行concatenating来创建。该two-stage augmentation包含：

- Masking：通过在item features集合上使用一个masking模式。我们会在input layer上使用一个缺省的 embedding来表示被masked的features。
- Dropout：对于多个值的categorical features，我们会使用一定概率来丢弃掉每个值。它会进一步减少input信息，并增加SSL任务的hardness

masking阶段可以被解释成一个关于dropout 100%的特例，我们的策略是互补masking（complementary masking）模式，我们会将feature set分割成两个排它的feature sets到两个增强样本上。特别的，我们可以随机将feature set进行split到两个不相交的subsets上。我们将这样的方法为Random Feature Masking（RFM），它会使用作为我们的baselines。我们接着介绍Correlated Feature Masking(CFM) ，其中，当创建masking patterns时，我们会进一步探索feature相关性。

**Categorical Feature的互信息**

如果masked features会被随机选中，(h, g)必须从$$2^kk$$个不同的masking patterns上抽样，这对于SSL任务会天然地导致不同的效果。例如，SSL contrastive learning任务必须利用在两个增强样本间高度相关features的shortcut，使得SSL任务太easy。为了解决该问题，我们提出将features根据feature相关性进行分割，通过互信息进行measure。两个类别型features的互信息如下：

$$
MI(V_i, V_j) = \sum\limits_{v_i \in V_i, v_j \in V_j} P(v_i, v_j) log \frac{P(v_i, v_j)}{P(v_i)p(v_j)}
$$

...(4)

其中，$$V_i, V_j$$表示它们的词汇集合（vocab sets）。所有features的pairs的互信息可以被预计算好。

**相关特征掩码（Correlated Feature Masking）**

有了预计算互信息，我们提出Correlated Feature Masking (CFM)，对于更有意义的SSL任务，它会利用feature-dependency patterns。对于masked features的集合，$$F_m$$，我们会寻找将高度相关的features一起进行mask。我们会首先从所有可能的features $$F=\lbrace f_1, \cdots, f_k \rbrace$$中均匀抽样一个seed feature $$f_{feed}$$；接着根据与$$f_{seed}$$的互信息，选择top-n个最相关的features $$F_c = \lbrace f_{c,1}, \cdots, f_{c,n} \rbrace$$。我们会选择$$n = \lceil k / 2 \rceil$$，以便关于features的masked and retained set，会具有完全相同的size。我们会变更每个batch的seed feature，以便SSL任务可以学习多种masking patterns。

## 3.3 Multi-task训练

，为了确保SSL学到的representations可以帮助提升主要监督任务（比如：回归或分类）的学习，我们会利用一个 multi-task training策略，其中：主要(main)监督任务和辅助(auxiliary) SSL任务会进行联合优化（jointly optimized）。准确的，

- $$\lbrace (q_i, x_i)\rbrace$$是一个关于query-item pairs的batch，它从训练数据分布$$D_{train}$$抽样得到；
- $$\lbrace x_i \rbrace$$是一个从item分布$$D_{item}$$抽样得到的items的batch；

那么，joint loss为：

$$
L = L_{main} (\lbrace q_i, x_i) \rbrace) + \alpha \cdot L_{self} (\lbrace x_i \rbrace)
$$

...(5)

其中：

- $$L_{main}$$：是main task的loss function，它可以捕获在query和item间的交叉
- $$\alpha$$：是regularization strength

**不同样本分布（Heterogeneous Sample Distribution）**

来自$$D_{train}$$的边缘item分布（The marginal item distribution）通常会遵循二八定律（power-law）。因此，对于$$L_{self}$$使用training item分布会造成学到的feature关系会偏向于head items。作为替代，对于$$L_{self}$$我们会从corpus中均匀抽样items。换句话说，$$D_{item}$$是均匀item分布。实际上，我们发现：**对于main task和ssl tasks使用不同分布（Heterogeneous Distribution）对于SSL能达到更优效果来说非常重要**。


<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/09fc357b71ed182bb3daa52290ae81b9a0ef8fa04a6621951c4a4b72a30c342369d9ccf7cd72d7c45b72c3120a1b4e4f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.jpg&amp;size=750">

图3

**Main task的loss**

对于依赖objectives的main loss来说有多个选择。在本paper中，对于优化top-k accuracy，我们考虑batch softmax loss。详细的，如果$$q_i, x_i$$是关于query和item样本$$(q_i, x_i)$$的embeddings（它会通过两个neural networks编码得到），接着对于一个关于pairs $$\lbrace (q_i, x_i) \rbrace_{i=1}^N$$的batch和temperature $$\tau$$，batch softmax cross entropy loss为：

$$
L_{main} = - \frac{1}{N} \sum\limits_{i \in [N]} log \frac{exp(s(q_i, x_i)/\tau)}{\sum_{j \in [N]} exp(s(q_i, x_j) / \tau)}
$$

...(6)

其它Baselines。如第2节所示，对于main task，我们使用two-tower DNNs作为baseline模型。对比起经典的MF和分类模型，two-tower模型对于编码item features具有独特性。前两种方法可以被用于大规模item检索，但他们只基于IDs学到item embeddings，不符合使用SSL来利用item feature relations。



# 4.离线实验

评估


[https://dl.acm.org/doi/pdf/10.1145/3459637.3481952]{https://dl.acm.org/doi/pdf/10.1145/3459637.3481952}