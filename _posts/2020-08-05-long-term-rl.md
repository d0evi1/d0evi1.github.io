---
layout: post
title: long-term engagement RL1介绍
description: 
modified: 2021-01-02
tags: 
---

jd在《Reinforcement Learning to Optimize Long-term User
Engagement in Recommender Systems》对long-term engagement做了建模。

# 摘要

Feed流机制在推荐系统中广泛被采用，特别是移动Apps。Feed streaming setting提供用户无限feeds的交互形式。这种形式下，**一个好的推荐系统会更关注用户的粘性，通常以long-term user engagement的方式进行measure**。这与经典的instant metrics相差较大。直接对long-term engagement直行优化是一个non-trivial问题，因为learning target通常不能由传统的supervised learning方法提供。尽管RL天然适合于对long term rewards最大化的最优化问题，应用RL来最优化long-term user engagement仍然面临着以下挑战：

- 用户行为是多变的，它通常包含两者：**instant feedback（比如：clicks）**以及**delayed feedback（比如：停留时长（dwell time）,再次访问（revisit））**；
- 另外，执行有效的off-policy learning仍然不成熟，特别是当结合上bootstrapping和function approximation时。

为了解决该问题，在该工作中，我们引入了一个RL framework——FeedRec来最优化long-term user engagement。FeedRec引入了两个部分：

- 1) 一个Q-Network，它以hierarchical LSTM的方式设计，会建模复杂的用户行为
- 2) 一个S-Network，它会模拟enviroment，协助Q-Network并避免在policy learning上的收敛不稳定。

实验表明，FeedRec在优化long-term user engagement上要胜于SOTA的方法。

# 介绍

推荐系统通过建议最匹配用户需求和喜好的商品，在信息搜索任务中帮助用户进行发现。最近，用户可以浏览由无限刷的feeds流生成的items，比如：Yahoo News的新闻流，Facebook的social流，Amazon的商品流。特别的，**当与商品流交互时，用户会点击items并浏览items的详情。同时，用户可能也会跳过不够吸引人的items，并继续下刷，并有可能由于过多冗余的、不感兴趣的items的出现而离开系统**。在这样的环境下，优化点击（clicks）不再是黄金法则。最大化用户的交互满意度，有两部分：

- instant engagement（比如：click）
- long-term engagement（比如：粘性 stickiness）：通常表示用户会继续更长时间停留在streams上，并在后续重复打开streams

然而，大多数传统的推荐系统只关注于优化instant metrics（比如：CTR：点击率、CVR：转化率 conversion rate）。随着更深的交互，一个商品feed流推荐系统应不仅带来更高的CTR，同时也能保持用户与系统的活跃度。**Delayed metrics通常更复杂，包括：在Apps上的dwell time，page-view的深度，在两个visits间的internal time等**。不幸的是，由于建模delayed metrics的难度，直接优化delayed metrics非常具有挑战性。而一些前置工作[28]开始研究一些long-term/delayed metrics的最优化，希望找到一种系统解决方案来最优化overall engagement metrics。

直觉上，RL天生是最大化long-term rewards的，可以是一个unified framework来最优化instant和long-term user engagement。使用RL来最优化long-term user engagement本身并不是一个non-trivial问题。正如提到的，long-term user engagement是非常复杂的（例如：在多变行为上的measure，比如：dwell time, revisit），需要大量的enviroment interactions来建模这样的long term行为，并有效构建一个推荐agent。作为结果，通过在线系统从头构建一个recommender agent的代价很高，因为许多与不成熟的推荐agent的交互会伤害用户体验，甚至惹恼用户。另一种可选的方法是利用logged data构建一个离线的recommender agent，其中，off-policy learning方法会缓和trial-and-error search的开销。不幸的是，在实际推荐系统中，**包括Monte Carlo(MC)和temporal difference(TD)在内的当前方法，对于offline policy learning具有缺陷：MC-based方法会有high variance的问题，尤其是在实际应用中当面对大量action space（例如：数十亿candidate items）；TD-based方法可以通过使用bootstrapping技术在估计时提升效率，然而，会遇到另一个大问题：Deadly Triad（致命的三）**：例如：当将function approximation、bootstrapping、offline training给合在一起时，会引起不稳定（instability）和分歧（divergence）问题。不幸的是，推荐系统中的SOTA方法，使用neural结构设计，在offline policy learning中会不可避免地遇到Deadly Triad问题。

为了克服复杂行为和offline policy learning的问题，我们这里提出了一个RL-based framework，称为FeedRec，来提升推荐系统中的long-term user engagement。特别的，我们将feed streaming推荐看成是一个Markov decision process(MDP)，并设计了一个Q-Network来直接最优化user engagement的metrics。为了避免在offline Q-learning中收敛不稳定的问题，我们会进一步引入S-Network，它会模拟environments，来协助policy learning。在Q-Network中，为了捕获多变的用户long-term行为的信息，会通过LSTM来建模用户行为链（user behavior chain），它包括所有的rough behaviors，比如：click、skip、browser、ordering、dwell、revisit等。当建模这样的细粒度用户行为时，会有两个问题：特定用户actions的数目是十分不平衡的（例如：click要比skips少很多）；long-term user behavior的表示更复杂。我们会进一步使用temporal cell集成hierarchical LSTM到Q-Network来对fine-grained用户行为进行characterize。

另一方面，为了充分利用历史logged data，并避免在offline Q-Learning中的Deadly Triad问题，我们引入了一个environment模型，称为S-network，来模拟environment并生成仿真的用户体验，协助offline policy learning。我们会在模拟数据集和真实电商数据集上进行实验。

主要贡献如下：

- 1) 提出了一个RL模型FeedRec，它会直接对user engagement进行最优化（同时对instant和long-term user engagement）
- 2) 为了建模多变的用户行为，它通常同时包括：instant engagement（比如：click和order）以及long-term engagement（例如：dwell time、revisit等），提出了hiearchical LSTM的结构的Q-Network
- 3) 为了确保在off-policy learning的收敛，设计了一个有效、安全的训练框架
- 4) 实验结果表明，提出的算法要胜过SOTA baseline

# 2.相关工作

略

# 3.问题公式化

## 3.1 Feed Streaming推荐

在feed流推荐中，推荐系统会在离散的time steps上与一个user $$u \in U$$进行交互：

在每个time step t上，agent会feeds一个item $$i_t$$，并从该user上接收一个feedback $$f_t$$，其中$$i_t \in I$$会来自推荐的item set，$$f_t \in F$$是用户在$$i_t$$上的feedback/behavior，包括：点击、购买、跳过、离开等。交互过程形成一个序列：$$X_t = \lbrace u, (i_1, f_1, d_1), \cdots, (i_t, f_t, d_t) \rbrace$$，其中：$$d_t$$是在推荐上的dwell time，它表示用户在推荐上的体验偏好。

给定$$X_t$$，agent需要为下一个item step生成$$i_{i+1}$$，它的目标是：**最大化long term user engagement**，例如：总点击数（total clicks） 或 浏览深度（browsing depth）。在本工作中，我们关注于在feed streaming场景如何提升所有items的期望质量（expected quality）。

## 3.2 Feed Streams的MDP公式

一个MDP可以通过$$M=<S, A, P, R, \gamma>$$进行定义，其中：

- S是state space
- A是action space
- $$P: S \times A \times S \rightarrow R$$是转移函数（transition function）
- $$R: S \times A \rightarrow R$$是mean reward function ，其中：r(s, a)是immediate goodness
- $$\gamma \in [0,1]$$是discount factor。

一个(stationary) policy $$\pi: S \times A \rightarrow [0, 1]$$会在actions上分配每个state $$s \in S$$一个distribution，其中：$$a \in A$$具有概率$$\pi (a \mid s)$$。

在feed流推荐中，$$<S, A, P>$$设置如下：

- State S：是一个states的集合。我们会在time step t上的 state设计成浏览序列$$s_t = X_{t-1}$$。在开始时，$$s_1 = \lbrace u \rbrace$$只会包含用户信息。在time step t上，$$s_t = s_{t-1} \oplus \lbrace (i_{t-1}, f_{t-1}, d_{t-1})\rbrace$$会使用old state $$s_{t-1}$$进行更新，
- Action A：是一个关于actions的有限集合。可提供的actions依赖于state s，表示为A(s)。$$A(s_1)$$会使用所有recalled items进行初始化。$$A(s_t)$$会通过从$$A(s_{t-1})$$中移除推荐items进行更新，action $$a_t$$是正在推荐的item $$i_t$$
- Transition P：是transition function，其中$$p(s_{t+1} \mid s_t, i_t)$$指的是，在$$s_t$$上采取action $$i_t$$之后看到state $$s_{t+1}$$的概率。在我们的case中，来自用户的feedback $$f_t$$的不确定性会根据$$t_t$$和$$s_t$$进行。

## 3.3 User Engagement和Reward function

之前提到，像传统推荐中，即时指标（instant metrics，比如：点击、购买等）不是用户engagement/satisfactory的唯一衡量，long term engagement更重要，它通常会以delayed metrics进行衡量，例如：浏览深度（browsing depth）、用户重复访问（user revisits）、在系统上的停留时长（dwell time）。RL learning会通过对reward functions进行设计，提供一种方式来直接对instant和delayed metrics进行直接最优化。

reward function $$R: S \times A \rightarrow R$$可以以不同的形式进行设计。我们这里会假设：在每一step t上的user engagement reward $$r_t(m_t)$$会以不同metrics进行加权求和的形式（weighted sum），来线性（linearly）地对它进行实例化：

$$
r_t = \omega^T m_t
$$

...(1)

其中：

- $$m_t$$由不同metrics的column vector组成
- $$\omega$$是weight vector

接着，我们根据instant metrics和delayed metrics给出一些reward function的实例。

### Instant metrics

在instant user engagement中，我们会具有clicks、purchase（商业上）等。instant metrics的公共特性是：**这些metrics由current action即时触发**。此处我们以click为例，第t次feedback的click数可以定义成：

$$
m_t^c = \#clicks(f_t)
$$

### Delayed metrics

delayed metrics包括：browsing depth、dwell time、user revisit等。这些metrics通常会被用于衡量long-term user engagement。**delayed metrics会由之前的行为触发，其中一些会具有long-term dependency**。这里提供会提供delayed metrics的两个示例reward functions：

 **1.深度指标（Depth metrics）**
 
 由于无限下刷机制，浏览的深度是在feed流场景下的一个特殊指标器（special indicator），它会与其它类型的推荐相区别。在观看了第t个feed之后，如果用户仍然在系统中并且继续下刷，系统会对该feed进行reward。直觉上，depth $$m_t^d$$的metric可以被定义成：
 
 $$
 m_t^d = \#scans(f_t)
 $$
 
 其中，$$\#scans(f_t)$$是第t个feedback的scans的数目。
 
 **2.返回时间指标（Return time metric）**

当用户对推荐items很满意时，通常他会更经常性地使用该系统。因此，在两个visits间的间隔时间（interval time）可以影响系统的用户满意度。return time $$m_t^r$$可以被设计成时间的倒数（reciprocal of time）:

$$
m_t^r = \frac{\beta}{v^r}
$$

其中：

- $$v^r$$表示在两次visits间的time
- $$\beta$$是超参数

从以上示例（click metric、depth metric以及return time metrics），我们可以清楚看到：$$m_t = [m_t^c, m_t^d, m_t^r]^T$$。注意，在MDP setting中，累积收益（cumulative rewards）是可以被最大化的，也就是说，我们实际上对总浏览深度（total browsing depth）、未来访问频次进行最优化，它们通常是long term user engagement。

# 4. 推荐系统的Policy learning

为了估计future reward（例如：未来用户粘性），对于推荐项$$I_T$$的expected long-term user engagement会使用Q-value进行表示：

$$
Q^{\pi} (s_t, i_t) = E_{i_k \sim \pi} [r_t + \sum\limits_{k=1}^{T_t} \gamma^k r_{t+k}]
$$

...(2)

其中：

- $$\gamma$$是discount factor，用来对current rewards和future rewards的importance进行balance
- $$Q^{*}(s_t, i_t)$$具有由optimal policy达到的最大expected reward，应遵循optimal Bellman方程[24]：

$$
Q^{*}(s_t, i_t) = E_{s_{t+1}} [r_t + \gamma max_{i'} Q^{*}(s_{t+1}, i') | s_t, i_t
$$

...(3)

给定$$Q^{*}$$，推荐项$$i_t$$会使用最大的$$Q^{*}(s_t, i_t)$$选中。在真实推荐系统中，由于大量的users和items，为每个state-action pairs估计action-value function $$Q^{*}(s_t, i_t)$$是可行的。因而，使用函数逼近（例如，neural networks）来估计action-value function很灵活和实际，例如：$$Q^{*}(s_t, i_t) \approx Q(s_t, i_t; \theta_q)$$。实际上，neural networks对于跟踪用户在推荐中的兴趣表现很好。在本paper中，我们提到，使用参数$$\theta_q$$的nural network function approximator作为一个Q-network。Q-network可以通过最小化mean-squared loss function进行训练，定义如下：

$$
l(\theta_q) = E_{(s_t, i_t, r_t, s_{t+1}) \sim M} [(y_t - Q(s_t, i_t; \theta_q))^2] \\
y_t = r_t + \gamma max_{i_{t+1 \in I}} Q(s_{t+1, i+1; \theta_q}
$$

...(4)

其中，$$M = \lbrace (s_t, i+t, r_t, s_{t+1}) \rbrace$$是一个大的replay buffer，它会存储过去的feeds，我们从中以mini-batch training的方式进行抽样。通过对loss function进行微分，我们可以达到以下的gradient：

$$
\nabla_{\theta_q} l(\theta_q) = E_{(s_t, i_t, r_t, s_{t+1}) \sim M} [(r+\gamma max_{i_{t+1}} Q(s_{t+1}, i_{t+1}; \theta_q) - Q(s_t, i_t; \theta_q)) \nabla_{\theta_q} Q(s_t, i_t; \theta_q)] 
$$

...(5)

实例上，通过SGD来最优化loss function通常计算效果，而非计算以上gradient的full expectations。

## 4.1 Q-Network

Q-network的设计对于性能很重要。在long-term user engagement最优化中，用户的交互行为反复无常（例如：除了click外，还有dwell time、revisit、skip等），这使得建模变得non-trivial。为了有效对这些engagement进行最优化，我们会首先从这样的行为收到信息传给Q-Network

### 4.1.1 Raw Behavior Embedding Layer

该layer的目的是，采用所有raw behavior信息，它们与long term engagement有关，来distill用户的state以便进一步最优化。给定observation $$s_t= \lbrace u, (i_1, f_1, d_1) \cdots, (i_{t-1}, f_{t-1}, d_{t-1}) \rbrace$$，我们让$$f_t$$是在$$i_t$$上所有用户行为的可能类型，包括：点击、购买、跳过、离开等，其中$$d_t$$是该行为的dwell time。$$\lbrace i_t \rbrace$$的整个集合首先被转成embedding vectors $$\lbrace i_t \rbrace$$。为了表示将信息feedback给item embedding，我们将$$\lbrace i_t \rbrace$$投影到一个feedback-dependent空间中，通过使用一个projection matrix来对embedding进行乘积，如下：

$$
i_t^' = F_{f_t} i_t
$$

其中，$$F_{f_t} \in R^{H \times H}$$是对于一个特定feedback $$f_t$$的投影矩阵，为了进一步建模时间信息，会使用一个time-LSTM来跟踪user state随时间的变化：

$$
h_{r, t} = Time-LSTM(i_t^', d_t)
$$

...(6)

其中，Time-LSTM会建模dwell time，通过引入一个由$$d_t$$控制的time gate：

$$
g_t = \sigma(i_t^', W_{ig} + \sigma(d_t W_{gg}) + b_g) \\
c_t = p_t \odot c_{t-1} + e_t \odot g_t \odot \sigma(i_t^' W_{ic} + h_{t-1} W_{hc} + b_c) \\
o_t = \sigma(i_t^' W_{io} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)
$$

其中，$$c_t$$是memory cell。
。。。

### 4.1.2 Hierarchical Behavior Layer

为了捕获我变的用户行为信息，所有rough behaviors会顺序feed到raw Behavior Embedding layer中。在实际中，特定user actions的数目是极其不均的（例如：点击数要少于skips数）。因此，直接利用raw Behavior Embedding Layer的output会造成Q-Network从sparse user behaviors中丢失信息，例如：购买信息会被skips信息会淹没。另外，每种类型的用户行为具有它自己的特性：在一个item上的点击，通常能表示用户当前的偏好，在一个item上的购买会暗示着用户兴趣的转移（shifting），而skip的因果关系更复杂，可能是：随意浏览（casual browsing）、中立（neutral）、或者觉得不喜欢（annoyed）等。

为了更好表示user state，如图1所示，我们提供一种hierarchical behavior layer来加到raw beahiors embedding layers中，主要的用户行为，比如：click、skip、purchase会使用不同的LSTM pipelines进行独立跟踪：

$$
h_{k,t} = LSTM-k(h_{r,t})
$$

f_t是第k个行为，其中，不同的用户行为（例如：第k个行为）会通过相应的LSTM layer来进行捕获，以避免被大量行为支配（intensive behavior dominance）、并捕获指定的特性。最后，state-action embedding会通过将不同用户的behavior layer和user profile进行concate起来：

$$
s_t = concat[h_{r,t}, h_{1,t}, h_{\dot, t}, h_{k,t}, u]
$$

其中，u是对于一个指定用户的embedding vector。

### 4.1.3 Q-value layer

Q-value的逼近（approximation）通过具有dense state embedding的input的MLP来完成，item embedding如下：

$$
Q(s_t, i_t; \theta_q) = MLP(s_t, i_t)
$$

$$\theta_q$$的值会通过SGD进行更新，梯度计算如等式(5)所示。

## 4.2 off-policy learning任务

有了Q-learning based framework，在学习一个稳定的推荐policy之前，我们在模型中通过trial&error search来训练参数。然而，由于部署不满意policies的开销以及风险，对于在线训练policy几乎是不可能的。一个可选的方式是，在部署前使用logged data D训练一个合适的policy，它通过一个logging policy $$\pi_b$$收集到。不幸的是，等式(4)的Q-Learning framework会到到Deadly Trial问题，当对函数近似（function approximation）、bootstrapping以及offline training进行组合时，会出现这种不稳定和差异问题。

为了避免在offline Q-Learning中的不稳定和差异问题，我们进一步引入一个user simulator（指的是S-Network），它会对environment进行仿真，并帮助policy learning。特别的，在每轮推荐中，会使用真实user feedback进行对齐（aligning），S-Network需要生成用户的响应$$f_t$$、dwell time $$d_t$$、revisited time $$v^r$$，以及一个二元变量$$L_T$$，它表示用户是否会离开平台。如图2所示，simulated user feedback的生成使用S-Network $$S(\theta_s)$$，它是一个multi-head neural network。State-action embedding被设计在与Q-Network中相同的结构中，但具有独立的参数。layer $$(s_t, i_t)$$会跨所有任务进行共享，其它layer （图2中的$$(s_t, i_t)$$）是task-specific的。因为dwell time和用户的feedback是inner-session的行为，$$\hat{f}_t$$和$$\hat{d}_t$$的计算如下：

$$
\hat{f}_t = softmax(W_f x_f + b_f) \\
\hat{d}_t = W_d x_f + b_d \\
x_f = tanh(W_{xf} [s_t, i_t] + b_{xf})
$$

其中：

$$X_*$$和$$b_*$$是weight项和bias项。$$[s_t, i_t]$$是state action feature的核心。revisiting time的生成、以及离开平台（inter-session 行为）的完成如下：

$$
\hat{l}_t = sigmoid(x_f^T w_l + b_l) \\
\hat{v}_r = W_v x_l + b_d \\
x_l = tanh(W_{xl} [s_t, i_t] + b_{xl})
$$

## 4.3 Simulator Learning

略

# 5.Simulation研究

略

# 6.实验





# 介绍




# 4.实验

略

# 参考


- 1.[https://arxiv.org/pdf/2001.03025.pdf](https://arxiv.org/pdf/2001.03025.pdf)