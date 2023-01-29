---
layout: post
title: facebook Memory Networks介绍
description: 
modified: 2021-09-19
tags: memory-network
---


facebook在《MEMORY NETWORKS》中提出了memory networks：

# 摘要

我们描述了一种称为"memory networks"的新学习模型。memory networks会让inference components与一个long-term meory component进行组合；他们会学习如何使用这些联合机制。其中：long-term memory可以读取和写入，我们的目标是使用它进行预估。我们会在QA场景对该模型进行研究，其中，long-term memory会有效扮演着一个动态的knowledge base，输出是一个textual response。我们会在一个大规模QA任务上对它们进行评估，并从一个仿真世界生成一个很少并且复杂的toy task。在后者，我们展示了这些模型的强大之处，它们会将多个支持的句子串联来回答需要理解动词强度的问题.

# 1.介绍

大多数机器学习模型缺少一种简单方式来读取和写入long-term memory component的部分，并将它与inference进行无缝组合。因而，他们不能利用现代计算机的最大优势。例如，**考虑这样一个任务：告诉一个关于事实（facts）或故事（story）的集合，接着回答与该主题相关的问题**。通常这会通过一个语言建模器（language modeler，比如：RNN）来达到。这些模型会被训练成：当读取一个words流后，预测并输出下一个word(s)或者集合。**然而，它们的memory（通过hidden states和weights进行encode）通常很小，没有被分得足够开来准确记住来自过往知识的事实（知识会被压缩到dense vectors中）**。RNNs很难执行记忆，例如：简单的copying任务：读入一个句子输出相同的句子（Zaremba 2014）。其它任务也类似，例如：在视觉和音频领域，需要long-term memory来watch一个电影并回答关于它的问题。

在本工作中，我们介绍了一种新的模型，称为memory networks，它会尝试解决该问题。中心思想是：通过将成功的学习策略（其它机器学习文献中）进行组合，使用可读可写的memory component进行inference。该模型接着会被训练来学习如何有效操作memory component。我们引入了第二节中的通用框架，并提出一个在文本领域关于QA的指定实现。

# 2.Memory Networks

一个memory network包含了：

- **1个memory： m**（一个通过使用$$m_i$$索引的objects的数组）
- **4个components：I、G、O、R**


<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/df77a6b0a0322e52f31f301b3d788899df7247fbae7386965839ecfafe955ca8fdfc6f9e6017cb58e390d87b53c01b0b?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

其中：
- I（input feature map）: 将incoming input转换成内部的feature representation
- G（generalization）： 给定new input更新旧的memories。我们称该genrealization为：在该stage，对于该network存在一个机会来压缩和泛化该memories，以便后续进一步使用
- O（output feature map）：给定new input和当前memory state，生成一个新的output（在feature representation space中）
- R（response）：将output转化成期望的response format。例如，一个文本reponse或一个action response。

给定一个input x（例如：一个input character，word或sentence，一个image或一个音频信号），该模型的flow如下：

- 1.将x转换成一个interal feature representation：$$I(x)$$
- 2.给定new input，更新memories $$m_i$$：$$m_i = G(m_i, I(x), m), \forall i$$
- 3.给定new input和该memories，计算output features o：$$o = O(I(x), m)$$
- 4.最终，将output features o解码给出最终的response: $$r = R(o)$$

该过程会同时被应用到train和test时，但在这样的过程间存在差别，也就是说，**memories会在test time时已经被存储下来，但I、G、O、R模型参数不会被更新**。Memory networks会覆盖许多种可能实现。I、G、O、R的components可以潜在使用任何机器学习文献的思想：例如：使用你最喜欢的模型（SVMs，或者decision trees等）

**I component**

component I可以使用标准的预处理：例如：关于文本输入的解析、共指、实体解析。**它可以将input编码成一个internal feature representation**，例如：将text转换成一个sparse或dense feature vector。

**G component**

G的最简单形式是，将I(x)存储到在memory中的一个"slot"上：

$$
m_{H(x)} = I(x)
$$

...(1)

其中：**$$H(.)$$是一个slot选择函数**。也就是说，G会更新关于index $$H(x)$$对应的m，但memory的其它部分不会动。G的许多复杂变种可以回溯并基于当前input x的新证据来更新更早存储的memories。**如果input在character和word级别，你可以将inputs（例如：将它们分割成 chunks）进行group，并将每个chunk存储到memory slot中**。

如果memory很大（例如：考虑Freebase或Wikipedia），你需要将memories进行合理组织。这可以通过使用前面描述的slot choosing function H来达到：例如，它可以被设计、或被训练、或者通过entity或topoic来存储记忆。因此，出于规模上的效率考虑，G（和O）不需要在所有memories上进行操作：他们可以只在一个关于候选的恢复子集上进行操作（只在合适主题上操作相应的memories）。在我们的实验中，我们会探索一个简单变种。

**如果memory满了（full），会有一个“遗忘过程（forgetting）”，它通过H来实现，它会选择哪个memory需要被替换**，例如：H可以对每个memory的功率（utility）进行打分，并对用处最少的一个进行覆盖写。我们不会在该实验中进行探索。

**O和R components**

O组件通常负责读取memory和执行inferenece，例如：计算相关的memories并执行一个好的response。R component则通过给定O来生成最终的response。例如，在一个QA过程中，O能找到相关的memories，接着R能生成关于该回答的实际wording，**例如：R可以是一个RNN，它会基于output O生成**。我们的假设是：如果在这样的memories上没有conditioning，那么这样的RNN会执行很差。

# 3.文本的MEMNN实现

**一个memory network的特殊实例是：其中的components是neural networks。我们称之为memory neural networks（MemNNs）**。在本节中，我们描述了关于一个MemNN的相对简单实现，它具有文本输入和输出。

## 3.1 基础模型

在我们的基础结构中，I模块会采用一个input text。我们首先假设这是一个句子：也可以是一个关于事实的statement，或者待回答的一个问题（后续我们会考虑word-based input sequences）。该text会以原始形式被存储到下一个提供的memory slot中，例如：S(x)会返回下一个空的memory slot N：$$m_N = x, N = N+1$$。G模块只会被用来存储该新的memory，因此，旧的memories不会被更新。更复杂的模型会在后续描述。

inference的核心依赖于O和R模块。**O模块会通过由给定的x寻找k个supporting memories来生成output features**。我们使用k直到2，但该过程可以被泛化到更大的k上。对于k=1，最高的scoring supporing memory会被检索：

$$
o_1 = O_1(x, m) = \underset{i=1,\cdots,N}{argmax} \  s_O(x, m_i)
$$

...(2)

其中：**$$s_O$$是一个打分函数，它会对在sentenses x和$$m_i$$的pair间的match程度进行打分**。如果k=2，我们会接着找到一个second supporting memory，它基于前一迭代给出：

$$
o_2 = O_2(x, m) =  \underset{i=1,\cdots,N}{argmax} \ s_O([x, m_{o_1}], m_i)
$$

...(3)

其中，候选的supporing memory $$m_i$$现在分别由原始的input和第一个supporting memory进行打分，其中：方括号内表示一个list。最终的output o是$$[x, m_{o_1}, m_{o_2}]$$，它是module R的输入。

最终，R需要生成一个文本response r。最简单的response是返回$$m_{o_k}$$，例如：输出我们检索的之前公开的句子。为了执行真实的句子生成，我们可以采用一个RNN来替代。在我们的实验中，我们也会考虑一个简单的评估折中方法，其中，我们会通过将它们排序，将文本响应限制到单个word（输出的所有words会在模型中被见过）：

$$
r = \underset{w \in W}{argmax} \ s_R([x, m_{o_1}, m_{o_2}], w)
$$

...(4)

其中：**W是字典中所有words的集合，并且$$S_R$$是一个关于match程度的得分函数**。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/9a2730ff74ab642da0e1badb8ecb45916276b2cc4b4b2617e2aef249fd2341a5dd6e7ac948b5fc196bae37a86623f913?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.jpg&amp;size=750">

图1

一个示例任务如图1所示。为了回答该问题：x = "where is the milk now?"，O module会首先对所有memories进行打分，例如：所有之前见过的句子，与x相反来检索最可能相关的事实：该case中的$$m_{o_1}$$="Joe left the milk"。接着，它会由给定的$$[x, m_{o_1}]$$再次搜索该memory来找出第二可能的fact，也就是 $$m_{o_2}$$="joe travelled to the office"（Joe在倒掉牛奶之前到过的最后的地方）。最终，R module使用等式(4)对给定的$$[x, m_{o_1}, m_{o_2}]$$的words进行打分，并输出 r = "office"。

在我们的实验中，scoring functions $$S_O$$和$$S_R$$具有相同的形式，一个embedding model：

$$
s(x, y) = \phi_x(x)^T U^T \phi_y(y)
$$

...(5)

其中：U是一个$$n \times D$$的矩阵，其中，D是features的数目，n是embedding的维度。$$\phi_x$$和$$\phi_y$$的角色是，将原始文本映射到D维feature space中。最简单的feature space是选择一个BOW的representation，我们为$$s_O$$选择 $$D = 3 \mid W \mid$$，例如，在字典中的每个word会具有三个不同的representations：一个是为$$\phi_y(.)$$，另一个是为$$\phi_x(.)$$，依赖于关于input arguments的words是否来自于实际的input x或者来自于supporting memories，以便他们可以不同方式建模。相似的，我们会为$$S_R$$使用$$D=3\mid W \mid$$。$$S_O$$和$$S_R$$会使用不同的weight矩阵$$U_O$$和$$U_R$$。

**training** 我们以一个fully supervised setting方式训练，其中给定想要的inputs和responses，supporting sentences会在训练数据中被标记（在test data中，我们只输了inputs）。也就是说，在训练期间，我们知道在等式 （2）和(3)中的max functions的最佳选择。训练会接着使用一个margin ranking loss和SGD来执行。特别的，对于一个给定的question x，它具有true response r，supporting sentences $$m_{o_1}$$和$$m_{o_2}$$（当k=2），我们会通过模型参数$$U_O$$和$$U_R$$进行最小化：

$$

$$

...(6)(7)(8)

其中，$$\bar{f}, \bar{f}', \bar{r}$$是对比正确labels所有其它选择，$$\gamma$$是margin。在SGD的每个step，我们会抽样$$\bar{f}, \bar{f}', \bar{r} $$，而非计算每个训练样本的总和。

在我们的MemNN中，采用一个RNN作为R component，我们会使用在一个语言建模任务中的标准log likelihood来替换最后一项，其中RNN会被feed序列$$[x, o_1, o_2, r]$$。在test time时，我们会根据给定的$$[x, o_1, o_2]$$输出prediction r。对比起最简单的模型，使用k=1并输出localted memory $$m_{o_1}$$作为response r，只会使用第一项进行训练。

下面章节我们会考虑扩展基础模型。

## 3.2 WORD序列作为input

如果input是word级别，而非sentence级别，那就在流中到达的words（例如，使用RNNs），并且还未被分割成statements和questions，我们需要将该方法进行修改。接着我们添加一个“segmentation” function，它会输入关于words的至今未被分割的最近序列。当该segmenter触发时（表示当前序列是一个segment）我们会将该序列写到memory中，接着如之前方式进行处理。segmenter会与其它组件相似建模，以一个embedding model的形式：

$$
seg(c) = W_{seg}^T U_S \phi_{seg} (c)
$$

...(9)

其中，$$W_{seg}$$是一个vector（实际上在embedding space中一个线性分类器的参数），并且c是input words序列，表示成使用一个独立字典的BOW。如果 $$seg(c) > \gamma $$，其中$$\gamma$$是margin，接着，该序列会被识别成一个segment。这种方式下，我们的MemNN在它的写操作中具有一个学习组件。我们将该segmenter考虑成概念的第一个证明：当然，你也可以设计得更复杂。附录B中给出了训练机制的详情。

## 3.3 通过Hashing方式的高效memory

如果存储的memories集合非常大，将等式（2）和等式（3）中所有内容进行打分会非常昂贵。作为替代，我们会探索hashing tricks来加速lookup：将input I(x)进行hash到一或多个buckets中，接着只有在相同buckets中的memories $$m_i$$会进行打分score。我们会研究hashing的两种方式：

- i) 采用hashing words
- ii) 将word embeddings进行聚类

对于(i)我们会考虑许多buckets，它们是字典里的words，接着对于一个给定的句子，我们将所有对应words hash到相应的buckets中。(i)的问题是：一个memory $$m_i$$只会考虑，它与input I(x)是否只共享至少一个word。方法(ii)尝试通过聚类方式解决该问题。在训练完embedding matrix $$U_O$$后，我们会运行K-means来将word vectors $$U(O)_i$$进行聚类，并给出K个buckets。我们接着将一个给定的句子中每个单独的word都落到各自的buckets中。由于word vectors趋向于与它们的同义词更近，他们会聚在一起，我们接着对这些相应的memories进行打分。input和memory间完全精准匹配则会通过定义进行打分。K的选择用于控制speed-accuracy间的trade-off。

## 3.4 建模write time

我们扩展我们的模型，来说明当一个memory slot是何时被写入。当回答关于确定事实的问题时（比如：“what is the capital of France?”），这并不重要。但当回答关于一个story的问题时（如图1），这就很重要。一种明显的方式是，添加额外的features到representations $$\phi_x$$和$$\phi_y$$中，它们会编码一个给定memory $$m_j$$的index j，假设：j会遵循write time（例如：没有memory slot rewriting）。然而，这需要处理绝对时间，而非相对时间。对于后续过程我们具有更成功经验：通过替代scoring input，与s的候选pairs，我们会在三元组 $$s_{O_t}(x, y, y')$$上学习一个funciton：

$$
s_{O_t} (x, y, y') = \phi_x(x)^T U_{O_t}^T U_{O_t} (\phi_y(y) - \phi_y(y') + \phi_t(x, y, y'))
$$

...(10)

$$\phi_t(x, y, y')$$使用三个新features，它会采用值0或1: 不管x是否大于y，x是否大于y', y大于y'。（也就是说，我们会将所有$$\phi$$ embeddings的维度扩展到3，并在当没有使用时将这三个维度设置成0）。现在，如果$$s_{O_t}(x, y, y') > 0$$，该模型会偏爱y胜过y'，如果$$s_{O_t}(x, y, y')<0$$则它偏好y'。等式(2)和(3)的argmax会通过在memories i=$$1, \cdots, N$$上的一个loop进行替换，在每个step时将胜出的memory(y或y')保持住，并且总是将当前的胜者与下一memory $$m_i$$进行对比。在当time features被移除之前，该过程等价于argmax。更多细节由附录C给出。

## 3.5 建模之前未见过的words

对于那些读了很多text的人来说，新的words仍会持续引入。例如：word "Boromir"第一次出现在《指环王》中。那么机器学习模型如何处理这些呢？理想的，只要见过一次就能work。一个可能的方法是使用一个language model：给定邻近词语，预测可能的word，并且假设新的word与它们相似。我们提出的方法会采用该思想，但将它组装到我们的网络$$s_O$$和$$s_R$$中，而非一个独立的step。

具体的，对于每个见过的word，我们会存储共现过的BOW，一个bag用于左侧context，另一个用于右侧context。任意未知的word可以被这样的features进行表示。因而，我们会将我们的feature representation D从$$3 \mid W \mid $$增加到$$t \mid W \mid$$来建模这样的contexts（对于每个bag有$$\mid W \mid$$ features）。我们的模型会在训练期间使用一种“dropout”技术来学习处理新words: d%的可能性我们会假装没有见过一个词，因而对于那个word来说不会有一个n维embedding，我们会使用context来替代表示它。

## 3.6 精准匹配和unseen words

embedding models不能有效使用exact word matches，因为维度n过低。一种解决方法是，对一个pair x,y使用下式进行打分：

$$
\Phi_x(x)^T U^T \Phi_y(y) + \lambda \Phi_x(x)^T \Phi_y(y)
$$

也就是说，“bags of words”会将得分与学到的embedding score进行茶杯。另外，我们的建议是，保留在n维embedding space中，但使用matching features扩展feature representation D，例如：每个word一个。一个matching feature表示着：一个词是否在x和y中同时出现。也就是说，我们会使用$$\Phi_x(x)^T U^T \Phi_y(y, x)$$进行打分，其中：$$\Phi_y$$实际基于x的条件构建：如果在y中的一些words与在x中的words相匹配，我们会将这些matching features设置为1。unseen words可以被相似建模，基于它们的context words使用matching features。接着给出一个feature space $$D = 8 \mid W \mid$$。

# 

略



- 1.[https://arxiv.org/pdf/1410.3916.pdf](https://arxiv.org/pdf/1410.3916.pdf)