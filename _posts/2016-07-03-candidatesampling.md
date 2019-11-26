---
layout: post
title: Candidate Sampling介绍
description: 
modified: 2016-07-03
tags: 
---

# 1.介绍

我们有一个multi-class或者multi-label问题，其中每个训练样本$$(x_i, T_i)$$包含了一个上下文$$x_i$$，以及一个关于target classes $$T_i$$的小集合（该集合在一个关于可能classes的大型空间L的范围之外）。例如，我们的问题可能是：给定前面的词汇，预测在句子中的下一个词。

## 1.1 F(x,y)

我们希望学习到一个兼容函数（compatibility function） $$F(x,y)$$，它会说明关于某个class y以及一个context x间的兼容性。**例如——给定该上下文，该class的概率**。

“穷举（Exhaustive）”训练法，比如：softmax和logistic regression需要我们为每个训练样本，对于每个类$$y \in L$$去计算F(x,y)。当$$\mid L \mid$$很大时，计算开销会很大。

## 1.2 $$T_i$$、$$C_i$$、$$S_i$$

- target classes（正样本）: $$T_i$$
- candidate classes（候选样本）: $$C_i$$
- randomly chosen sample of classes（负样本）: $$S_i$$

“候选采样（Candidate Sampling）”训练法，涉及到构建这样一个训练任务：对于每个训练样本$$(x_i, T_i)$$，我们只需要为**候选类（candidate classes）$$C_i \subset L$$**评估F(x,y)。通常，候选集合$$C_i$$是target classes和随机选中抽样的classes(非正例) $$S_i \subset L$$的合集(union)。

$$
C_i = T_i \cup S_i
$$

随机样本$$S_i$$可以依赖（或者不依赖）$$x_i$$和/或$$T_i$$。

训练算法会采用神经网络的形式训练，其中：用于表示F(x,y)的layer会通过BP算法从一个loss function中进行训练。

<img src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/7fd6372a58aaf13d8a89f379634b9a132d357cd795084da410793d51219da5cba916c10b9ff6eae38eeb612a114cf7ab?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=1.jpg&amp;size=750">

图1

- $$Q(y \mid x)$$: 被定义为：给定context x，根据抽样算法在sampled classes的集合中得到class y的概率（或：expected count）。
- $$K(x)$$：是一个任意函数（arbitrary function），不依赖于候选类（candidate class）。由于softmax涉及到一个归一化（normalization），加上这种函数不会影响到计算概率。
- logistic training loss为：

$$
 \sum\limits_i (\sum\limits_{y \in POS_i} log(1+exp(-G(x,y)) + \sum\limits_{y \in NEG_i} log(1+exp(G(x_i,y)) )
$$
 
- softmax training loss为：

$$
\sum\limits_i (-G(x_i,t_i) + log (\sum\limits_{y \in POS_i \cap NEG_i} exp(G(x_i,y))))
$$

- NCE 和Negatvie Sampling可以泛化到$$T_i$$是一个multiset的情况。在这种情况中，$$P(y \mid x)$$表示在$$T_i$$中y的期望数（expected count）。相似的，NCE，Negative Sampling和Sampled Logistic可以泛化到$$S_i$$是一个multiset的情况。在这种情况下，$$Q(y \mid x)$$表示在$$S_i$$中y的期望数（expected count）。

# Sampled Softmax

参考：[http://arxiv.org/abs/1412.2007](http://arxiv.org/abs/1412.2007)

假设我们有一个单标签问题（single-label）。每个训练样本$$(x_i, \lbrace t_i \rbrace)$$包含了一个context以及一个target class。我们将**$$P(y \mid x)$$作为：给定context x下，一个target class y的概率**。

我们可以训练一个函数F(x,y)来生成softmax logits——也就是说，该class在给定context上的相对log概率（relative log probabilities）：

$$
F(x,y) \leftarrow log(P(y|x)) + K(x)
$$

其中，K(x)是一个不依赖于y的任意函数(arbitrary function)。

在full softmax训练中，对于每个训练样本$$(x_i,\lbrace t_i \rbrace)$$，我们会为在$$y \in L$$中的所有类计算logits $$F(x_i,y)$$。**如果类L总数很大，计算很会昂贵**。

## 抽样函数：$$Q(y \mid x)$$

在"Sampled Softmax"中，对于每个训练样本$$(x_i, \lbrace t_i \rbrace)$$，**我们会根据一个选择抽样函数：$$Q(y \mid x)$$来选择一个关于“sampled” classese的小集合$$S_i \subset L$$**。每个被包含在$$S_i$$中的类，它与概率$$Q(y \mid x_i)$$完全独立。

$$
P(S_i = S|x_i) = \prod_{y \in S} Q(y|x_i) \prod_{y \in (L-S)} (1-Q(y|x_i))
$$

我们创建一个候选集合$$C_i$$，它包含了关于target class和sampled classes的union：

$$
C_i = S_i \cup \lbrace t_i \rbrace
$$

我们的训练任务是为了指出：**在给定集合$$C_i$$上，在$$C_i$$中哪个类是target class**。

对于每个类$$y \in C_i$$，给定我们的先验$$x_i$$和$$C_i$$，我们希望计算target class y的后验概率。

使用Bayes' rule：[bayes](https://math.stackexchange.com/questions/549887/bayes-theorem-with-multiple-random-variables)

$$
P(Z|X,Y) = P(Y,Z|X) P(X) / P(X,Y) = P(Y,Z|X) P(Y|X)
$$...(b)

得到：

$$
P(t_i=y|x_i,C_i) = P(t_i=y,C_i|x_i) / P(C_i|x_i) \\
=P(t_i=y|x_i) P(C_i|t_i=y,x_i) / P(C_i|x_i) \\
=P(y|x_i)P(C_i|t_i=y,x_i) / P(C_i|x_i)
$$

接着，为了计算$$P(C_i \mid t_i=y,x_i)$$，我们注意到为了让它发生，$$S_i$$可以包含y或也可以不包含y，但必须包含$$C_i$$所有其它元素，并且必须不包含在$$C_i$$任意classes。因此：

$$
\begin{align}
P(t_i=y|x_i, C_i) & = P(y|x_i) \prod_{y' \in C_i - \lbrace y \rbrace} Q({y'}|x_i) \prod_{y' \in (L-C_i)} (1-Q({y'}|x_i)) / P(C_i | x_i) \nonumber\\
& = \frac{P(y|x_i)}{Q(y|x_i)} \prod_{ {y'} \in C_i} Q({y'}|x_i) \prod_{ {y'} \in (L-C_i)} (1-Q({y'}|x_i))/P(C_i|x_i) \nonumber\\
& = \frac{P(y|x_i)}{Q(y|x_i)} / K(x_i,C_i)
\end{align} \nonumber\\
$$

其中：**$$K(x_i,C_i)$$是一个与y无关的函数**。因而：

$$
log(P(t_i=y | x_i, C_i)) = log(P(y|x_i)) - log(Q(y|x_i)) + {K'} (x_i,C_i)
$$

这些是relative logits，应feed给一个softmax classifier，来预测在$$C_i$$中的哪个candidates是正样本（true）。

因此，我们尝试训练函数F(x,y)来逼近$$log(P(y \mid x))$$，它会采用在我们的网络中的layer来表示F(x,y)，减去$$log(Q(y \mid x))$$，然后将结果传给一个softmax classifier来预测哪个candidate是true样本。

$$
training \ softmax \ input = F(x,y) - log(Q(y|x))
$$

从该classifer对梯度进行BP，可以训练任何我们想到的F。


# 

以tensorflow中的tf.random.log_uniform_candidate_sampler为例。

它会使用一个log-uniform(Zipfian)base分布。

该操作会随样从抽样分类(sampled_candidates)中抽取一个tensor，范围为[0, range_max)。

sampled_candidates的元素会使用base分布被无放回投样（如果：unique=True），否则会使用有放回抽样。

对于该操作，基础分布是log-uniform或Zipf分布的一个近似：

$$
P(class) = \frac{(log(class+2) - log(class+1))} { log(range\_max + 1)}
$$

当target classes近似遵循这样的一个分布时，该sampler很有用——例如，如果该classes以一个字母序表示的词语，并以频率降序排列。如果你的classes没有通过词频降序排列，就不需要使用该op。

另外，该操作会返回tensors: true_expected_count， 

## sampled_softmax_loss

{% highlight python %}

def _compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None,
                            seed=None):
    # 核心代码实现：
    if isinstance(weights, variables.PartitionedVariable):
        weights = list(weights)
    if not isinstance(weights, list):
        weights = [weights]

    # labels_flat:  batch_size.
    with ops.name_scope(name, "compute_sampled_logits",
                        weights + [biases, inputs, labels]):
        if labels.dtype != dtypes.int64:
            labels = math_ops.cast(labels, dtypes.int64)

        labels_flat = array_ops.reshape(labels, [-1])

    # 抽取num_sampled个样本.
    if sampled_values is None:
        sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
            true_classes=labels,
            num_true=num_true,
            num_sampled=num_sampled,
            unique=True,
            range_max=num_classes,
            seed=seed)

    # 这三个值不会进行反向传播
    sampled, true_expected_count, sampled_expected_count = (array_ops.stop_gradient(s) for s in sampled_values)

    # 转成int64
    sampled = math_ops.cast(sampled, dtypes.int64)

    # label + sampled (labels基础上拼接上抽样出的labels)
    all_ids = array_ops.concat([labels_flat, sampled], 0)

    # 合并在一起，使用all_ids一起进行查询
    all_w = embedding_ops.embedding_lookup(
        weights, all_ids, partition_strategy=partition_strategy)

    # 分割出label的weight.
    true_w = array_ops.slice(all_w, [0, 0],
                             array_ops.stack(
                                 [array_ops.shape(labels_flat)[0], -1]))

    # 分割出sampled weight.
    sampled_w = array_ops.slice(
        all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])

    # user_vec * item_vec
    sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)

    # bias一起查询.
    all_b = embedding_ops.embedding_lookup(
        biases, all_ids, partition_strategy=partition_strategy)

    # true_b is a [batch_size * num_true] tensor
    # sampled_b is a [num_sampled] float tensor
    true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
    sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

    # element-wise product.
    dim = array_ops.shape(true_w)[1:2]
    new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
    row_wise_dots = math_ops.multiply(
        array_ops.expand_dims(inputs, 1),
        array_ops.reshape(true_w, new_true_w_shape))

    # true label对应的logits, bias.
    dots_as_matrix = array_ops.reshape(row_wise_dots,
                                       array_ops.concat([[-1], dim], 0))
    true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
    true_b = array_ops.reshape(true_b, [-1, num_true])


    true_logits += true_b
    sampled_logits += sampled_b

    # 减去先验概率.
    if subtract_log_q:
      # Subtract log of Q(l), prior probability that l appears in sampled.
      true_logits -= math_ops.log(true_expected_count)
      sampled_logits -= math_ops.log(sampled_expected_count)

    # 输出logits，拼接在一起.
    out_logits = array_ops.concat([true_logits, sampled_logits], 1)

    # 输出的labels.
    out_labels = array_ops.concat([
        array_ops.ones_like(true_logits) / num_true,
        array_ops.zeros_like(sampled_logits)
    ], 1)

    return out_logits, out_labels

{% endhighlight %}

得到logits和labels后，就可以计算softmax_cross_entropy_with_logits_v2了。

## log_uniform_candidate_sampler

	tf.random.log_uniform_candidate_sampler(
    true_classes,
    num_true,
    num_sampled,
    unique,
    range_max,
    seed=None,
    name=None
	)


使用一个log-uniform(Zipfian)的基础分布来采样classes集合。

该操作会对一个sampled classes（sampled_candidates） tensor从范围[0, range_max)进行随机抽样。

sampled_candidates的elements会从基础分布进行无放回抽样（如果unique=True）或者有放回抽样（unique=False）。

对于该操作的base distribution是一个近似的log-uniform or Zipfian分布：

$$
P(class) = (log(class + 2) - log(class + 1)) / log(range\_max + 1)
$$

**当target classes近似遵循这样的一个分布时，该sampler很有用——例如，如果该classes表示字典中的词以词频降序排列时。如果你的classes不以词频降序排列，无需使用该op**。

另外，该操作会返回true_expected_count和sampled_expected_count的tensors，它们分别对应于表示每个target classes(true_classes)以及sampled classes（sampled_candidates）在sampled classes的一个平均tensor中期望出现的次数。这些值对应于在上面的$$Q(y \mid x)$$。如果unique=True，那么它是一个post-rejection概率，我们会近似计算它。

# paper2

另外在paper 2《On Using Very Large Target Vocabulary for Neural Machine Translation》的第3节部分，也做了相应的讲解。

## paper2 第3节

在该paper中，提出了一种model-specific的方法来训练具有一个非常大的目标词汇表（target vocabulary）的一个神经网络机器翻译（NMT）模型。使用这种方法，训练的计算复杂度会是target vocabulary的size的常数倍。另外，该方法允许我们高效地使用一个具有有限内存的快速计算设备(比如：一个GPU)，来训练一个具有更大词汇表的NMT模型。

训练一个NMT的计算低效性，是由等式(6)的归一化常数引起的。为了避免计算该归一化常数的成长复杂度(growing complexity)，我们这里提出：在每次更新时使用target vocab的一个小子集$$V'$$。提出的方法基于早前Bengio 2008的工作。

等式(6), next target word的概率：

$$
p(y_t | y_{<t},x) = \frac{1}{Z} exp \lbrace w_t^T \phi(y_{t-1}, z_t, c_t) + b_t \rbrace
$$

...(6)

其中Z是归一化常数（normalization constant）：

$$
Z = \sum\limits_{k:y_k \in V} exp \lbrace w_k^T \phi (y_{t-1}, z_t, c_t) + b_k \rbrace
$$

...(7)

假设我们考虑在等式(6)中output的log-probability。梯度的计算由一个正部分(positive part)和负部分(negative part)组成：

$$
\nabla log p(y_t \mid y_{<t}, x) = \nabla \epsilon(y_t) - \sum\limits_{k:y_k \in V} p(y_k | y_{<t}, x) \nabla \epsilon(y_k)
$$

...(8)

其中，我们将energy $$\epsilon$$定义为：

$$
\epsilon(y_i) = w_j^T \phi(y_{i-1}, z_j, c_j) + b_j
$$

梯度的第二项（negative）本质上是energy的期望梯度：

$$
E_p[\nabla \epsilon(y)]
$$

...(9)

其中P表示$$p(y \mid y_{<t}, x)$$。

提出的方法主要思想是，通过使用少量samples进行importance sampling来逼近该期望，或者该梯度的负项。给定一个预定义的分布Q和从Q中抽取的一个$$V'$$样本集合，我们使用下式来逼近等式(9)中的期望：

$$
E_P [\nabla \epsilon(y)] \approx \frac{w_k}{\sum_{k':y_{k'} \in V'} w_{k'}} \nabla \epsilon(y_k)
$$

...(10)

其中：

$$
w_k = exp \lbrace \epsilon(y_k) - log Q(y_k) \rbrace
$$

...(11)

该方法允许我们在训练期间，使用target vocabulary的一个小子集来计算归一化常数，每次参数更新时复杂度更低。直觉上，在每次参数更新时，我们只会更新与correct word $$w_t$$以及在$$V'$$中的sampled words相关的vectors。一旦训练完成，我们可以使用full target vocabulary来计算每个target word的output probability。

尽管提出的方法可以天然地解决计算复杂度，使用该方法本质上不能在每个句子对(sentance pair：它们包含了多个target words)更新时保证参数数目。

实际上，我们会对training corpus进行分区(partition)，并在训练之前为每个分区（partition）定义一个target vocabulary子集$$V'$$。在训练开始之前，我们会顺序检查训练语料中的每个target sentence，并累积唯一的target word，直到唯一的target words达到了预定义的阀值$$\tau$$。然后在训练期间将累积的vocabulary用于corpus的partition。我们会重复该过程，直到达到训练集的结尾。假设对于第i个partition的target words的子集用$$V_i'$$表示。

这可以理解成，对于训练语料的每个partition具有一个单独的分布$$Q_i$$。该分布$$Q_i$$会为在子集$$V_i'$$中包含的所有target words分配相等的概率质量(probability mass)，其它words则具有为0的概率质量，例如：

$$
Q_i(y_i) = 
\begin{cases}
\frac{1}{|V_i'|} & \text{if $y_t \in  V_i'$} \\
0	& otherwise.
\end{cases} 
$$

提议分布(proposal distribution)的选择会抵消掉correction term——来自等式(10)-(11)中importance weight的$$log Q(y_k)$$，使得该方法等价于，使用下式来逼近等式(6)中的准确output probability：

$$
p(y_t | y_{<t}, x) = \frac{exp \lbrace w_t^T \phi(y_{t-1}, z_t, c_t) + b_t) \rbrace}{ \sum_{k:y_k \in V'} exp \lbrace w_k^T \phi(y_{t-1}, z_t, c_t) + b_k \rbrace}
$$

需要注意的是，Q的这种选择会使得estimator有偏。

对比常用的importance sampling，提出的该方法可以用来加速，它会利用上现代计算机的优势来完成matrix-matrix vs. matrix-vector乘法。

# 参考

- [https://www.tensorflow.org/extras/candidate_sampling.pdf](https://www.tensorflow.org/extras/candidate_sampling.pdf)
- [On Using Very Large Target Vocabulary for Neural Machine Translation](https://arxiv.org/pdf/1412.2007.pdf)
