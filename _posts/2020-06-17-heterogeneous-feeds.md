---
layout: post
title: Heterogeneous Feeds predict优化介绍
description: 
modified: 2020-06-15
tags: 
---

ali在多年前提的一个《An End-to-end Model of Predicting Diverse Ranking On Heterogeneous Feeds》，我们来看下它的实现。

# 介绍

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/5f3dccb61a3ff96dfff597173299b5ef2ec271b3127e3551d88a7ea533c5304290b12acececf119a8c1a4b85bbc0c553?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1 Item Search Engine和Content Search Engine间的关系


# 2.基本概念

## 2.1 商业背景

alibaba自己拥有CSE(Content Search Engine)和ISE（Item Search engine），它会相互交互来为用户创建在线购物环境。在ISE中的所有items与CSE中的一群feed相关。用户可以自由地travel和search在两个搜索引擎。

阿里巴巴ISE和CSE的混合使用，对用户在线购物有收益。总之，用户会在阿里巴巴上日志化，他们与搜索引擎的交互总是有意图的。然而，当面对大量items时，如果他们只搜索在ISE中的items，他们可能会迷茫。例如，阿里的官网，像服装这样的热门品类可能包含上万的items。每个items会使用数十个keywords进行打标记（比如：风格、slim cut、韩版等）。如果没有任意指导，用户们从一个item候选的大集合中区分合适的items是面临挑战的。

因此，**为了帮助用户避免疑惑，并提供它们可靠的购物建议，CSE会成为用户的购物指导**。给定来自用户的queries，CSE会组织一个合适的feed ranking list作为一个返回结果，替代item ranking list。**feeds可以被表示成post(article)、list(item list)以及video的形式**。他们由电商领域（clothing、travelling、化妆品）专家的"淘宝达人（Daren）"生产。在它们的feeds流量，“达人”会介绍特定items的优点和缺点，并对特定subjects基于它们的领域知识提出个人建议。

- 一个post feed是一篇文章，它会介绍特定items的属性；
- 一个list feed是一个推荐items的集合；
- 一个video feed是一个用来演示建议items的short video。

通过达人的建议，用户可以做出更好的购物选择。

每天在生产环境中的数据会经验性地展示在两个搜索引擎间的user travel rate是否频繁。在用户跳到CSE中前，他们通常已经在ISE中搜索过记录。这表明用户实际上希望购买来自“达人”的建议。提供更好的CSE搜索结果可以帮助用户更方便的定向到合适的items，从而做出之后的购买，这是电商的主要目标。然而，仍然有挑战。首先，feed types是异构的，不同类型的feeds的匹配度（fitness）与一个给定的query是不可比的。例如，一个list feed是否比一个post feed更好依赖于用户体验。第二，大量用户进入阿里CSE会携带着来自ISE的用户行为。如何处理跨域信息并构建user profiles来形成一个在CSE上的个性化feed ranking需要进一步探索。

## 2.2 数据准备

在我们的方法中，我们的目标是：**给定一个user u发起的一个query q，返回一个异构feed ranking list $$R_1(feed) \mid u, q$$**。

在Top K个ranked feed中的每个item都会被安置（locate），并在CSE中从上到下分别在一个"slot"中展示。为了学习independent Multi-armed Bandit（iMAB）模型以及personalized Markov DNN(pMDNN)模型，需要获取**slot相关的统计数据（全局信息）**以及**用户点击流数据（个性化信息）**这两者。

### 2.2.1 slot相关的统计数据

用户偏好的一个假设是：**在每个slot中的feed type相互独立**。

对于每个slot，三个候选feed types的概率（post、list、video）会**遵循它们自己的Beta分布**。因此，给定在一个slot s上的候选类型T，为了估计一个feed type $$\theta$$的先验分布$$p(\theta \mid \alpha, \beta, T, s)$$，必须知道每个slot的所有slot相关统计信息，以便估计$$\alpha$$和$$\beta$$。slot相关的统计数据包含了两部分：在线实时数据和离线历史数据。在线实时数据指的是流数据：每天由用户生成的对于一个特定slot type的点击数(click)、曝光数(display)。离线历史数据指的是：过去n天page view（pv）、以及item page view（ipv）的总数。在线天数据是streaming data，只能在实时中观察到。而离线历史数据可以被追踪、并能从仓库中获取。我们会在表1中计算和展示top 5 slots的统计数据。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/306effdbd7e89591e270a7b1460e62b2d579da9b671c6d2a64a2ce82623dbf44522f0e1fdada9806b207e2cd831c1faf?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=t1.jpg&amp;size=750">

表1 在CSE的top 5 slots上的feed历史数据

从表1中看到，在每个相关slots的pv和ipv的总数会有不同，video feeds在CSE中的top 5 slots很难出现。这表示：从全局上看，用户在不同feed types会偏向不同的feed types。

### 2.2.2 用户点击流数据

来自ISE和CSE的用户行为序列数据对于训练一个个性化feed ranking结果来说是有用的。**为了构建user profile，我们会设置一个window size w，它只考虑用户在ISE上的最新w个行为**。该行为可以表示为两种类型的triplet：<user, issue, query>和<user, click, item>。

- 用户在items上的点击次数表明users和items间的关系，
- 而用户(users)发起（issue）相同query的次数表明：users和queries间的关系强度。

基于此，可以在相同的latent space上从每个user/query中学习得到一个给定维度的embedding。另外，**在每个slot中的feed type通过one-hot encoder进行编码。最后，所有users、queries、feed types可以被表示成vectors**。示例如表2所示。前两列指的是每个user $$f_u$$学到的表示，以及一个issued query $$f_q$$。第三列指的是在每个slot中feed type $$f_t$$的one-hot表示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/0689a49c79b288b56e8ab2f953d3da07dbef6426e4fcf74909e6f0cdb999bfc42ee1b08d8a8d37876054fbf161a41cc1?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=t2.jpg&amp;size=750">

表2 对于每个slot的用户个性化数据的示例

# 3.方法

对于用户体验，我们希望观察到更好的异构feeds ranking。整个过程包含了Heterogeneous Type Sorting step以及Homogeneous Feed Ranking step。

- 对于第一个step，对于slot independent场景，会设计一个independent Multi-Armed Bandit(iMAB)模型；
- 对于slot denpendent场景，会设计一个改进版的personlized Markov DNN(pMDNN)模型。

第3.1节和3.2节会分别引入两个模型。对于第二个step，会使用一个DSSM模型来在每个slot上分配合适类型的feeds。第3.3节会详细介绍，pMDNN可以与DSSM一起训练来构成一个end-to-end模型。

## 3.1 independent Multi-Armed Bandit

在iMAB模型中，**异构feed ranking的评估指标是：在ipv和pv间的ratio $$\theta$$**。更高的$$\theta$$意味着：当一个用户浏览在CSE中的一个feed时，用户更可能会点击该feed。因此，$$\theta$$可以根据用户的实际需要来用于评估heterogeneous feed ranking的匹配度（fitness）。因此，**对于每个independent slot，我们会为每个feed type估计一个先验ratio $$\theta$$分布，并倾向于选择能够生成最高$$\theta$$值的feed type**。

理论上，由于Beta分布可以天然地表示成：由两个参数$$\alpha$$和$$\beta$$控制的任意类型的分布，它会假设每个type的ratio $$\theta$$具有一个先验分布遵循：$$\theta_i \sim B(\alpha_i^0, \beta_i^0)$$，

其中：

- $$i \in \mu = \lbrace post, list, video \rbrace$$。
- $$\alpha_i^0$$是type i的历史ipv数，
- $$\beta_i^0$$是type i历史pv数和ipv数间的差。

由于$$B(\alpha_i^0, \beta_i^0)$$的期望是：$$\frac{\alpha_i^0}{\alpha_i^0 + \beta_i^0}$$，它是ipv和pv间的历史ratio。因此，后验ratio分布可以通过在线实时流数据进行每天更新，表示成：

$$
\theta_i \mid D_i \sim B(\alpha_i^0 + \lambda D^{ipv}, \beta_i^0 + \lambda (D^{pv} - D^{ipv})) 
$$

其中：

- $$D_i$$指的是每天到来的feed type i
- $$\lambda$$是时间影响因子，因为新数据会对更新ratio分布有影响。

最终，我们会使用一个two step sampling策略来选择每个slot的type。首先，对于每个feed type i，会被随机生成一个value $$\theta_i$$，因为在pv和ipv间的ratio的估计遵循以下的概率分布：

$$
p(\theta_i | D_i) = \frac{(\theta_i)^{\alpha_i,-1}(1-\theta_i)^{\beta_i,-1}}{B(\alpha_i, \beta_i,)}
$$

...(1)

其中，给定$$\alpha_i$$和$$\beta_i,$$，

- $$B(\alpha_i, \beta_i,)$$是一个常数
- $$\alpha_i,=\alpha_0+\lambda D_i^{ipv}$$
- $$\beta_i,=\beta_0 + \lambda(D_i^{pv} - D_i^{ipv})$$

第二，对于每个feed type，会应用一个softmax函数到所有feed types上来生成一个归一化的selection概率：

$$
p(i) = \frac{exp(\theta_i)}{\sum_{i \in \mu} exp(\theta_j)}
$$

...(2)

其中：

- i指的是三种feed types的其中之一
- $$\theta_i$$是公式1展示了根据后验概率分布$$D(\theta)$$生成的随机值

这种方式中，在所有slots中的feed types会独立选中。整个过程的伪代码如算法1所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/2ceac90ce872330ed31b006821c7f8c231ceeb9b328fca298d9e64315f65cd462a8c586f61341730b570e6f6a7ab53f3?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=a1.jpg&amp;size=750">

算法1

## 3.2 peronalized Markov DNN

Dependent heterogneous feed type selection会由三个因素决定：user、query、以及在相同page页上之前的slot feed types。 第一，不同users会在相同queries下对于items具有不同的偏好。例如，当一个用户搜索"dress"时，她可能会愿意看到关于dress描述的文章post。而对于其它user，他们可能更喜欢lists，因为他们想看到更多item选项而非单个item介绍。第二，在当前slot上的feed types的用户偏好可能会受之前feed types的潜在影响，它可以被看成是一个Markov process。例如，没有用户愿意看到在所有slots上看到相同类型，他们或多或少希望看到不同types的feeds。第三，不同queries在所有slots上应产生不同的feed type allocation。为了一起集成用户偏好、query、以及推荐的previous feed types，对于第i个slot，我们提出了一个pMDNN模型来生成推荐的feed type $$t_i \mid (user, query, t_1, \cdots, t_{i-1})$$。整个模型可以解耦成两个子任务（sub tasks）：包括一个user&query表示学习任务、以及一个personlized slot type的预测任务，如图2所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/52fbd3f901e0935336d3662d7377d21877360bbb618b2deb2b8adc5e5c9dd3558b3554b4edafc2db9a044a3da98423e6?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2

### 3.2.1 User和Query的表示

基于统计数据，在CSE中有80%的用户来自于ISE。因此，通过使用它们的用户行为序列数据，我们可以构建一个graph来描述users、queries、items间的关系。之后，node2vec会在最后使用一个skip gram模型来学习users和queries的embeddings。详细的pipeline如图2所示，目标函数如下：

$$
O(\upperset{\rightarrow}{f_v}) = log \sigma(\bar{f_t} \cdot \bar{f_v})) + k E_{u \in P_{noise}} [-log \sigma(\bar{f_u} \cdot \bar{f_v}))]
$$

...(3)

$$\bar{f_v}$$是当前node v的embedding。t是node v的一个positive neighbour node；u是v的一个negative sampled node。这意味着，给定一个node v，我们需要学习一个node embedding表示，它可以最大化概率来生成它的positive neighbor node u，并最小化概率来生成它的negative node node sets$$P_{noise}$$。

图2的中间部分表明，如何训练node embedding表示。input layer是node的one-hot encoding。weight matrix W是所有nodes的embedding，它可以帮助将input one-hot encoding node投影到一个$$\mid D \mid$$维的latent space上。接着，最大化概率来生成node u的neighour nodes。

在最后，所有users和queries可以使用一个给定长度维度的embedding representations。并且匀们使用user & query embeddings作为输入来进行slot feed type预测。

### 3.2.2 Type prediction

对于给定users、queries、以及previous slot feed types信息，我们希望预测每个slot的feed types。因此，目标函数为：

$$
\underset{\phi}{argmax} \prod_{i=1}^K p(\phi(X_i) = c | u_i, q_i, f_i)
$$

...(4)

其中，$$X_i$$是对于第i个slot的input feature vectors，它与user $$u_i$$、queries $$q_i$$以及previous slot feed types $$f_i$$相关。$$\Phi$$是input feature vectors到output feed type的转换函数。c是当前slot的true feed type。我们的目标是最大化成功预测slot feed types的joint probability。

为了简化我们的pMDNN模型，并加速运行速度，只有slot feed type的一阶Markov过程会被应用到该模型上。它意味着预测第i个slot feed type，只有第(i-1)个slot feed type会对它有latent影响。而这对于一个user u的第一个slot feed type来说会带来一个问题。因为它没有previous slot feed type信息。对于一个user u，为了给第一个slot生成一个伪信息，user u喜欢的item i会在ISE中根据观看次数和停留时长被检测到。接着，我们会在ISE中将item i映射到在CSE中与它相关的feed f中，并使用f的type作为一个替代。

我们使用给定的user、query的embedding以及previous slot types来构建pMDNN模型来推荐feed type。input layer是user embedding(U)、query embedding(Q)和之前slot types(T)的concatenation。User和query embedding会通过在constructed graph上进行node2vec学到。整个input layer的构建可以看成是：

$$
X = U \oplus Q \oplus T
$$

...(5)

最后，会在input layer上附加三个fully connected hidden layers。每个layer会使用线性分类器和cross entropy作为loss function。在每个hidden layer中的Activation function被设置为ReLu，output layer会使用Softmax作为activation function。通过gradient descent和BP，我们会训练模型直至收敛。outpu layer是一个vector，它包含了：，在通过一个softmax activation function之后，在每个指定slot上关于三种feed types的一个概率分布。

$$
L_1 = ReLu(w_0 \cdot X) \\
L_2 = ReLu(w_1 \cdot L1) \\
L_3 = ReLu(w_2 \cdot L2) \\
L = Softmax(w_3 \cdot L_3)
$$

...(6)

L表示当前slot feed type的true label。pMDNN模型会在离线阶段被训练，我们可以管理trained model来预测实时用户请求。$$L_1, L_2, L_3$$分别表示三个hidden layers。图2的第一部分展示了这个工作流。

## 3.3 异构feed ranking

下一step是对homogeneous feeds进行排序，并填充相关的slots。例如，如果$$slot_i, slot_j, slot_k$$被选中具有"post" feed type，我们需要rank所有post feeds并选择在当前query下具有最高relevance score的top 3 feeds。由于所有types的feeds与文本信息相关（比如：title），会使用一个已经存在的DSSM来对所有post feeds进行rank来在三个slots上进行填充。

在DSSM中，我们不会为每个word会使用一个one-hot表示进行编码，而是使用word hashing方法来利用n-gram模型来分解每个word。这会减小word表示的维度。

最后，一个DNN模型可以使用query和feeds作为input layer，给定跨训练信的queries，并通过最大化点击文档的likelihood进行模型参数训练。相等的，该模型需要最小化以下的loss function：

$$
L(\wedge) = -log \prod_{Q, D^+} p(D^ | Q)
$$

...(7)

其中，$$\wedge$$表示NN的参数集。$$D^+$$表示具有true label的feed，Q是user-issued query。该模型会使用gradient-based数值优化算法进行训练。

最后，给定一个query，所有candidate feeds可以通过由该模型计算的generative probability进行排序。它可以使用pMDNN进行训练来公式化成一个end-to-end模型，如图2所示。而它仍需要从iMAB模型进行训练得到。

# 参考


- 1.[]()