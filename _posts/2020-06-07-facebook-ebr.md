---
layout: post
title: baidu Query-Ad Matching算法介绍
description: 
modified: 2020-06-05
tags: 
---

facebook在《Embedding-based Retrieval in Facebook Search》介绍了它们的推荐策略。

# 摘要

社交网络中的搜索（比如：Facebook）会比其它经典web搜索构成更大挑战：除了query text外，很重要的是，考虑上搜索者（searcher）的context来提供相关结果。它们的社交图谱（social graph）是这些context的一个主要部分，这是Facebook search的独特之处。而embedding-based retrieval(EBR)被应用到web搜索引擎中已经好多年了，Facebook search仍主要基于Boolean matching模型。在本paper中，我们会讨论使用统一embedding framebook来构建semantic embeddings来进行个性化搜索，该系统会在一个经典搜索系统中基于一个inverted index来提供embedding-based retrieval服务。我们讨论了一些tricks和experiences在整个系统的end-to-end optimization上，包括ANN参数tuning和full-stack optimization。最终，我们将整个过程表述成两个selected advanced topics。我们会为Facebook Search的verticals上评估EBR，并在online A/B experimenets上获取大的metrics提升。我们相信，该paper会提供给你在search engines上开发embeddinb-based retrieval systems一些观点和经验。

# 1.介绍

search engine是一个很重要的工具，帮助用户访问大量在线信息。在过去十年中，已经有许多技术来提升搜索质量，特别是Bing和Google。由于很难从query text上准确计算搜索意图以及文档的意义表示，大多数搜索技术基于许多term matching方法，一些case上keyword matching可以很好的解决问题。然而，对于semantic matching仍然是个大挑战，需要解决：期望结果可能与query text不匹配，但能满足用户搜索意图的情况。

在最近几年，deep learning在语音识别，CV、NLP上得到了巨大进步。在它们之中，embedding已经被证明是成功的技术贡献。本质上，embedding可以将ids的sparse vector表示成一个dense feature vector，它也被称为：semantic embedding，可以提供语义的学习。一旦学到该embeddings，它可以被用于query和documents的表示，应用于搜索引擎的多个stages上。由于该技术在其它领域的巨大成功，它在IR领域以及工业界也是个活跃的研究主题，被看成是next generation search technology。

总之，搜索引擎由：

- 一个recall layer：目标是检索一个相关文档集合（低时延和低开销），通常被称为“retrieval”；
- 一个precision layer：目标是使用复杂算法和模型对最想要的文档进行top rank，通常被称为“ranking”

组成。而embeddings可以被应用到这两个layers上，它通常在retrieval layer上使用embeddings机会更多些，因为它在系统底层（bottom），通常会是瓶颈。在retrieval中的embeddings的应用被称为“embedding-based retrieval”或"EBR"。出于简洁性，EBR是一种使用embeddings来表示query和documents的技术，接着retrieval问题被转换成一个在embedding space上的NN search问题。

EBR在搜索问题中是个挑战，因为数据的规模很大。不同于ranking layers：该layers通常会在每个session考虑数百个documents，retrieval layer需要在search engine的index上处理billions或trillions的文档。在embeddings的training和serving上都具在大规模的挑战。第二，不同于CV任务中embedding-based retrieval，search engine通常在retrieval layer上需要同时包括：embedding-based retrieval和term matching-based retrieval来进行打分。

Facebook search，作为一个社交搜索引擎，与传统搜索引擎相比具有独特挑战。在facebook search中，搜索意图（search intent）不只依赖于query text，同时对于发起该query的用户以及searcher所处的context具有很深的依赖。因此，facebook的embedding-based retrieval不是一个text embedding问题，而是一个IR领域的活跃的研究主题。另外，它是个相当复杂的问题，需要一起考虑：text、user、context。

为了部署ebr，我们开发了一个方法来解决modeling、serving、full-stack optimization的挑战。在modeling上，我们提出了unified embedding，它是一个two-sided model，一个side是：query text、searcher、context组成的search request，另一个side是：document。为了有效地训练该模型，我们开发了方法来从search log中挖掘训练数据，并从searcher、query、context、documents中抽取featrues。为了快速进行模型迭代，我们采用了离线评估集来进行一个recall metric评估。

对于搜索引擎，构建retrieval models具有独特挑战，比如：如何构建representative training task建模和有效、高效学习。我们调查了两个不同的方法：

- hard mining：来有效解决representing和learning retrieval 任务的挑战
- ensemble embedding：将模型划分成多个stages，其中每个stage具有不同的recall/precision tradeoff

在模型被开发后，我们需要在tetrieval stack上进行开发以支持高效的模型serving。使用已存在的retrieval和embedding KNN来构建这样的系统很简单，然而我们发现这是次优（suboptimal）方案，原因如下：

- 1) 从我们的初始实验看存在巨大性能开销
- 2) 由于dual index，存在维护开销
- 3) 两个candidate sets 可能会有大量重合，整体上低效

因此，我们开发了一个混合retrieval framework来将embedding KNN与Boolean matching进行整合，来给retrieval的文档进行打分。出于该目的，我们采用Faiss来进行embedding vector quantization，并结合inverted index-based retrieval来提供混合检索系统。除了解决上述挑战外，该系统也有两个主要优点：

- 1) 它可以允许embedding和term matching进行joint optimization来解决search retrieval问题
- 2) 它允许基于term matching限制的embedding KNN, 它不仅可以帮助解决系统性能开销，也可以提升embedding KNN results的precision

search是一个multi-stage ranking系统，其中retrieval是第一个stage，紧接着还有ranking、filtering等多个stages。为了整体优化系统来返回new good results，假设在结尾有new bad results，我们会执行later-stage optimization。特别的，我们合并embedding到ranking layers中，并构建一个training data feedback loop来actively learn来从embedding-based retrieval中标识这些good和bad results。图1是EBR系统的一个图示。我们在facebook search的verticals上评估了EBR，它在A/B实验上具有大的提升。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b9ff5151f853c3f5ffb37ecdff94950374dd3acf9f6d40eb732bbc507079170e7d760d6c73da40f16a0e902936f72556?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1

# 2.模型

我们将搜索检索任务公式化成一个recall optimization问题。特别的，给定一个search query，它的target result set $$T=\lbrace t_1, t_2, \cdots, t_N \rbrace$$，模型返回top K个结果：$$\lbrace d_1, d_2, \cdots, d_K \rbrace$$，我们希望最大化top k个结果的recall：

$$
recall@K = \frac{\sum_{i=1}^K d_i \in T}{N}
$$

...(1)

target results是基于特定准则与给定query相关的documents。例如，它可以是user clicks的结果，或者是基于human rating的相关文档。

我们基于query和documents之间的距离计算，将一个ranking problem公式化为recall optimization。query和documents使用一个neural network model被编码成dense vectors，我们基于它使用cosine similarity作为距离metric。我们提出使用triplet loss来近似recall objective来学习neural network encoder，它也被称为embedding model。

而semantic embedding通常被公式化成在IR上的text embedding problem，它在facebook search上是低效的，它是一个个性化搜索引擎，不仅会考虑text query，也会考虑searcher的信息、以及search task中的context来满足用户个性化信息的需要。以people search作为一个示例，它具有上千个名为“john Smith”的user profiles，实际的target person是使用"John Simth"作为query搜索的这个用户，很可能是它们的朋友或者相互认识。为了建模该问题，我们提出了unified embedding，它不只会考虑text，也会在生成的embeddings上考虑user和context信息。

## 2.1 评估metrics

由于我们的最终目标是，通过online A/B test，以端到端的方式来达到质量提升，开发离线metrics很重要，在在线实验前可以快速评估模型质量，从复杂的online实验setup中隔离问题。我们提出在整个index上运行KNN search，接着使用等式(1)中的recall@K作为模型评估指标。特别的，我们会抽样10000个search sessions来收集query和target result set pairs作为evaluation set，接着报告在10000 sessions上的平均recall@K。

## 2.2 Loss function

对于一个给定的triplet $$(q^{(i)}, d_{+}^{(i)}, d_{-}^{(i)})$$，其中：

- $$q^{(i)}$$是一个query
- $$d_{(+)}^{(i)}$$和$$d_{(-)}^{(i)}$$分别是相关的positive和negative documents

triplet loss的定义如下：

$$
L = \sum\limits_{i=1}^N max(0, D(q^{(i)}, d_{+}^{(i)}) - D(q^{(i)}, d_{-}^{(i)}) + m)
$$

...(2)

其中，$$D(u, v)$$是一个在vector u和v间的distance metric，m是在positive和negative pairs间的margin，N是triplets的总数。该loss function的意图是：通过一个distance margin从negative pair中分离出positive pair。我们发现：调整margin value很重要——最优的margin value会随不同的训练任务变化很大，不同的margin values会产生5-10%的KNN recall variance。

我们相似，使用random samples来为triplet loss生成negative pairs可以逼近recall optimization任务。原因如下：

当candidate pool size为n时，如果我们在训练数据中对每个positive抽样n个negatives，该模型将会在recall@top1的位置上进行最优化。假设实际的serving candidate pool size为N，我们可以近似最优化recall@topK $$ K \approx  N/n$$。在第2.4节，我们将验证该hypothesis并提供不同positive和negative label定义的对比。


## 2.3 Unified Embedding模型

为了学习最优化triplet loss的embeddings，我们的模型由三个主要部分组成：

- 一个query encoder $$E_Q = f(Q)$$：它会生成一个query embedding
- 一个document encoder $$E_D = g(D)$$：它会生成一个document embedding
- 一个相似度函数 $$S(E_Q, E_D)$$：它会生成一个在query Q和document D间的分数

一个encoder是一个neural network，它会将一个input转换成一个低维的dense vector，被称为embedding。在我们的模型中，这两个encoders $$f(\cdot)$$和$$g(\cdot)$$缺省是两个独立的networks，但可以选择共享部分参数。对于相似度函数，我们选择cosine相似度，因为它是embedding learning中常用的一种相似度：

$$
S(Q, D) = cos(E_Q, E_D) = \frac{<E_Q, E_D>}{|| E_Q || \cdot || E_D ||}
$$

...(3)

该distance被用于在等式(2)中的loss function，因此cosine distance被定义成：$$1 - cos(E_Q, E_D)$$

encoders的inputs可以从conventional text embedding模型中区分unified embedding。unified embedding可以编码textual、social以及其它有意义的contextual features来各自表示query和document。例如，对于query side，我们可以包含searcher location以及其它social connections；而对于document side，以社群搜索（groups search）为例，我们可以包括aggregated location以及关于一个Facebook group的social clusters。

大多数features是具有较高基数（cardinality）的categorical features，它们可以是one-hot或multi-hot vectors。对于每个categorical feature，一个embedding lookup layer会被插入来学习，输出它的dense vector表示，接着feed给encoders。对于multi-hot vectors，最终的feature-level embedding会使用一个关于多个embeddings的加权组合。图2表示我们的unified embedding模型架构。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b76d72344662d37778263f9d882ca19a616991ebce7841a5ffad90fe064959c783fd01c10c1554c24d3c181e92af9efb?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.png&amp;size=750">

图2

## 2.4 训练数据的挖掘

对于一个检索任务，定义positive和negative labels是non-trivial问题。这里我们会基于模型的recall metric来对比一些选择（options）。对于负样本（negative），我们会在以下的两种negatives options中进行实验；而使用click作为正样本（positive）：

- 随机抽样（random samples）：对于每个query，我们会随机从document pool中随机抽样documents作为negatives
- 非点击曝光（non-click impressions）：对于每个query，我们会在相同的session中随机抽样这样的曝光未点击数据来生成negatives

对比起使用random negative，使用non-click impressions作为负样本训练的模型会具有更差的model recall：对于people embedding model来说在recall上相差55%的regression。我们认为：这是因为对于hard cases（它们在一或多个因子上会与query相match）存在negatives bias，在索引index中的大部分documents都是easy cases，它们不需要与query完全匹配。将所有negatives作为这样的hard negatives会将训练数据的表示变更成为真实的retrieval任务，这会引入non-trivial bias到学到的embeddings中。

我们也会使用不同方式的mining positives进行实验，并发现以下的有意思的现象：

- 点击（clicks）：它使用点击结果作为正样本，因为clicks表示用户对于结果的反馈，它很可能与用户的搜索意图相匹配
- 曝光（impressions）：我们将retrieval看成是一个ranker的近似，但它执行的很快。因此，我们希望设计retrieval模型来学习返回相同集合的结果，它们可以被ranker排得更高。从这个意义上说，对于retrieval model learning来说，展示或曝光给用户的所有结果是相同的。

我们的实验结果表明，两种定义效果相当；模型使用click vs. impressions进行训练，给定相同的data volume，会导致相似的recalls。另外，我们会对click-based训练数据与impression-based数据达成一致，然而我们没有观察到在click-based模型上有额外的收益。这表明增加impression data不会提供额外的值，模型也不会从增加的训练数据量上获得收益。

我们以上的研究表明，使用click作为positive以及random为作negative可以提供一个合理的模型表现。在它之上，我们进一步探索hard mining策略来提升模型区分相似结果间不同之处的能力。我们将在6.1节讨论。

# 3.特征工程

unified embedding model的一个优点是，它可以吸收除text之外的不同features来提升模型表现。我们观察到在不同的verticals上的一致性，unified embedding对比text embedding会更高效。例如，对于event search，当从text切换到unfied embeddings上会有+18%的recall提升，对于group search则会有+16%的recall提升。unified embeddings的效果高度依赖于对informative features的识别和制作。表1表明通过增加每个新的feature category到gropup embedding model（text features作为baseline）中的增量提升。在该节中，我们会讨论一些重要的features，它们会对主要的模型提升有贡献。

**Text features**

对于text embedding，character n-gram[7]是一个常用的方式来表示text。对比word n-grams它的优点有两方面。首先，由于有限的vocab size，embedding lookup table具有一个更小的size，在训练期间可以更高效的学习。第二，对于query侧（比如：拼写差异或错误）或者document侧（facebook存在大量内容）来说，subword representation对于out-of-vocab问题来说是健壮的。我们对比了使用character n-grams vs. word n-grams的模型发现前者的效果更好。然而，在characger trigrams的top，包括：word n-gram represantations会额外提供较小但一致的模型提升（+1.5%的recall gain）。注意，由于word n-grams的cardinality通常会非常高（例如：对于query trigrams有352M），需要进行hashing来减小embedding lookup table的size。即使有hash冲突的缺点，添加word n-ngrams仍会提供额外收益。

对于一个facebook entity，抽取text features的主要字段（field）是people entities的name、或者non-people entities的title。对于Boolean term matching技术，我们发现使用纯粹text features训练的embeddings特别擅长于解决以下两种场景：

- Fuzzy text match。例如，该模型允许学习在query "kacis creations"与"Kasie's creations" page间的匹配，而term-based match则不能
- Optionalization。例如，对于query "mini cooper nw"，该模型可以学习检索期望的群组（expected group）：Mini cooper owner/drivers club。通过丢弃“nw”来得到一个optional term match。

**Location features**

Location match在许多搜索场景具有优点，比如：对于本地business/groups/events的搜索。为了让embedding模型在生成output embeddings时考虑上locations，我们添加location features到query和document side features上。对于query side，我们会抽取searcher的city、region、country、language。对于document side，我们会添加公开提供的信息，比如：由admin打标记上的explicit group locaiton。有了这些text features，该模型可以成功学到query与results间相匹配的implicit location。表2展示了在group search中，由text embedding model vs. text+location embedding model分别返回的top相似文档上的一个side-by-side的对比。我们可以看到，带location features可以学到将localtion信号混合到embeddings中，ranking文档具有相同的location，因为来自Louisville, Kentucky的searcher会有更高的位置。


# 4.Serving

## 4.1 ANN

略

## 4.2 系统实现

为了将embedding-based retrieval集成到我们的serving stack中，我们实现了在Unicorm（facebook的搜索引擎）中NN search的一级支持。Unicorn会将每个document表示成一个bag-of-terms，它们可以是任意strings，可以表示关于document的二元属性，通常会使用它们的语义进行命名。例如，一个用户john居住在Seattle，会具有以下term：text:john以及location:seattle。terms可以有与它们绑定的payloads。

一个query可以是在该terms上的任意一个Boolean expression。例如，以下query可以返回所有具有在名字中有john和smithe、并且居住在Seattle或Menlo Park的people:

	(and (or (term location:seattle)
			(term location:menlo_park))
		(and  (term text:john)
			(term text:smithe)))
			
为了支持NN，我们将文档表示进行扩展来包含embeddings，每个具有一个给定的string key，并添加一个 (nn <key> : radius <radius>) query operator，它会匹配那些<key> embedding在query embedding的特定radius范围内的所有documents。

在indexing时，每个document embedding会被quantized，并转化成一个term（对于它的coarse cluster）以及一个payload（对于quantized residual）。在query time时，(nn)在内部会被重写成一个(or)的terms：它与离query embedding (probes)最近的coarse clusters相关，来匹配上些term payload在一定radius限制内的documents. probes的数目可以通过一个额外的属性：nprobe来指定，无需重写一个独立的系统，我们可以从已经存在的系统中继承所有的features，比如：realtime updates、高效的query planning及execution，以及支持multi-hop queries。

最后支持top-K NN queries，我们只会选择与query最近的K个documents，接着对query的剩余部分进行评估。然而，从实验上看，我们发现radius mode可以给出在系统效果和结果质量上更好的trade-off。一个可能的原因是，radius mode允许一个constrained NN search，但topK mode提供一个更宽松的operation，它需要扫描所有index来获得topK的结果。因此，我们在生产环境中使用radius-based matching。

### 4.2.1 Hybrid Retrieval

由于具有(nn) operator作为我们Boolean query language的一部分，我们可以支持hybrid retrieval表达式，它可以是embeddings和terms的任意组合。这对于model-based fuzzy matching来说很有用，可以提升像拼写变种（spelling variations）、optionalization等的cases。而从其它retrieval expression的部分进行复用和获益。例如，一个误拼的query: john smithe，它会寻找一个人名为john smith in Seattle or Menlo Park的人；搜索表达式和上面的类似。

该表达式在检索在question中的user时会失败，因为term text:smithe会在匹配document时会失败。我们可以通过(nn) operator添加fuzzy matching到该表达式中：

	（and (or (term location:seattle)
			 (term location:menlo_park))
		   (or (and (term text:john)
		   		   (term text:smithe))
		     (nn model-141709009 : radius 0.24 :nprobe 16)))
		     
其中model-141795009是embedding的key。在该case中，当query(john smithe) embedding和document(john smith) embedding间的cosine distance小于0.24时，target user会被检索到 。

### 4.2.2 Model Serving

我们可以以如下方式对embedding model进行serving。在训练完two-sided embedding model后，我们将模型分解成一个query embedding model以及一个document embedding model，接着分别对两个models进行serving。对于query embedding，我们会部署一个online embedding inference来进行实时inference。对于documents，我们会使用Spark来以batch offline的方式进行model inference，接着将生成的embeddings与其它metadata发布到forward index上。我们做这样额外的embedding quantization包括：coarse quantization、PQ，并将它发布到inverted index中。

## 4.3 Query和Index Selection

为了提升EBR的效率和质量，我们会执行query和index selection。我们会应用query selection技术来克服像over-triggering、huge capacity cost以及junkines increase。我们不会trigger EBR来进行特定quereis，因为EBR在没有提供额外value时会很弱，比如：一些searchers发起的easy queries会寻找一个之前搜索和点击过的的特定query。在index侧，我们会做index selection来让搜索更快。例如，我们只会选择月活用户、最近events、popular pages以及groups。


# 5. later-stage optimization

facebook search ranking是一个复杂的multi-stage ranking系统，其中每个stage会渐进式地改进（refine）来自前一stage的结果。在该stack的最底部是retrieval layer，其中会使用embedding based retrieval。来自retrieval layer的结果接着通过一个ranking layers的stack进行排序（sort）和过滤（filter）。在每个stage的model对通过前一layer返回的结果分布进行最优化。然而，由于当前ranking stages是为已经存在的搜索场景所设计的，这会导致由EBR返回的新结果被已经存在的rankers进行次优排序（sub-optimally ranked）。为了解决该问题，我们提出两种方法：

- Embedding作为ranking feature。对embedding相似度进一步进行propagating不仅可以帮助ranker识别来自EBR的新结果，也可以提供一个针对所有结果的通用语义相似度measure。我们探索了许多options来抽取基于embeddings的features，包括：在query和result embedding间的cosine相似度、Hadamard product、raw embeddings。从实验研究上看，cosine similarity feature比其它选项效果更好
- 训练数据反馈闭环（feedback loop）。由于EBR可以提升retrieval recall，对比term matching，它可以具有一个更低的precision。为了解决precision问题，我们基于human rating pipeline构建了一个封闭的feedback loop。特别的，我们会在EBR后对结果进行日志，接着发送这些结果到human raters来标记是否相关。我们使用这些人工标注的数据来重新训练相关性模型，从而它可以被用于过滤来自EBR中不相关的结果，只保留相关的结果。这被证明是一个有用的技术，来达到在EBR中recall提升的一个高precision。


# 6.高级主题

EBR需要大量研究来继续提升效果。我们研究了两个重要的embedding modeling领域：hard mining和embedding ensemble，来持续增强SOTA的EBR。

## 6.1 Hard mining

在文本/语义/社交匹配上，对于一个retrieval任务的数据空间，具有多样性的数据分布，对于一个embedding模型来说，在这样的空间上进行高效学习需要设计一个特别的training dataset。为了解决该问题，hard mining是一个主要方向，对于embedding learning来说也是一个活跃的领域。然而，大多数CV的研究用于分类任务，而搜索检索没有“classes”的概念，因此唯一的问题是已经存在的技术不一定能work。在该方向上，我们划分为两部分：hard negative mining和hard positive mining。

### 6.1.1 Hard negative mining（HNM)

当分析我们的embedding模型时，我们发现：给定一个query，来自embeddings的topK个结果通常具有相同的名字，尽管会有social features，该模型不会总是对高于其它的target结果进行排序。这驱使我们相信：模型不能合理利用social features，这很可能是因为：negative training data很容易，因为他们是随机样本，通常具有不同的名字。为了使得模型在区分相似结果间的不同表现的更好，我们可以使用在embedding space中与positive样本更接近的样本作为hard negatives。

**online hard negative mining**

由于模型训练是基于mini-batch更新的，hard negatives会在每个动态的batch中被高效选中。每个batch由n个positive pairs组成（$$\lbrace q^{(i)}, d_{+}^{(i)} \rbrace_{i=1}^{n}$$）。接着对于每个query $$q^{(i)}$$，我们会构成一个小的document pool，它使用所有其它positive documents $$\lbrace d_{+}^{(1)}, \cdots,  d_{+}^{(n)} \mid j \neq i \rbrace$$，并选择具有最高相似度得分的documents作为hardest negatives来创建training triplets。允许online hard negative mining是我们模型提升的一个主要contributor。它可以极地提升跨所有verticals的embedding模型质量：对于people search的recall具有+8.38%的recall；对于groups search具有+7%的recall，对于events search具有+5.33%的recall。我们也观察到：最优的setting是每个positive最多有两个hard negatives。使用超过两个hard negatives会启动regress model quality。

online HNM的一个限制是，具有来自random samples的任意hard negative的probability可能很低，因此不能产生hard enough negatives。接着，我们会基于整个result pool来生成harder negatives，也就是：offline Hard Negative Mining。

**Offline hard negative mining**

offline hard negative mining具有以下的procedure：

- 1) 为每个query生成top K的结果
- 2) 基于hard selection strategy选择hard negatives
- 3) 使用最新生成的triplets来重新训练embedding模型
- 4) 该过程可以是迭代式的

我们执行大量实验来对比offline hard negative mining和online hard negative mining。一个发现是，使用hard negatives进行简单训练的模型，比不过random negatives训练的模型。更进一步分析表明：“hard”模型会在non-text features上放置更多的weights，但在text match上会比"easy"模型表现更差。因此，我们会调整sampling strategy，并最终生成能胜过online HNM模型的一个模型。

第一个insight是关于hard selection strategy。我们发现，使用hardest examples并不是最好的strategy。我们会对比来自不同的rank positions中的抽样，并发现在rank 101-500间的抽样能达到最好的model recall。第二个insight是retrieval task optimization。我们的hypothesis是：训练数据中的easy negatives的出现仍然是必要的，因为检索模型是在一个input space上操作的，它混合了不同levels的hardness数据组成。因此，我们探索一些方式来与hard negatives一起集成random negatives，包括：从一个easy model中的transfer learning。从经验上看，以下两种技术具有最好的效果：

- **Mix easy/hard training**：在训练中混合random和hard negatives是有益的。在easy:hard=100:1时，可以增加easy/hard negatives的ratio可以提升model recall和饱和度。
- 从“hard”模型到"easy"模型的transfer learning：从easy到hard模型的transfer learning不会生成一个更好的模型，从hard到easy的transfer learning可以达到一个更好的model recall的提升

最后，在training data中为每个data point计算穷举KNN是非常time-consuming的，由于计算资源限制，总的模型训练时间会变得不切实际。对于offline hard negative mining算法来说，具有一个高效地top K generation很重要。。。


## 6.2 embedding ensemble

# 7.结论



# 参考

- 1.[https://arxiv.org/pdf/2006.11632.pdf](https://arxiv.org/pdf/2006.11632.pdf)