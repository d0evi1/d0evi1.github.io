---
layout: post
title: transformer介绍
description: 
modified: 2018-05-08
tags: 
---

google在2017年paper《Attention Is All You Need》提出了transformer，我们可以看下：

# 1.介绍

在序列建模(sequence modeling)和转换问题(transduction problems，比如：语言建模和机器翻译)上，RNN, LSTM和GRNN已经成为state-of-art的方法。大多数努力源自于recurrent语言模型和encoder-decoder架构的持续推动。

recurrent模型通常会沿着输入和输出序列的符号位置(symbol position)进行因子计算。在计算时对位置（position）进行排列，他们可以生成一个hidden states $$h_t$$的序列，它是关于前一hidden state $$h_{t-1}$$和位置t做为输入的一个函数。**这种天然的序列特性排除了在训练样本中的并发性(这对于更长序列长度很重要)，因为内存约束会限制在样本上进行batching**。最近工作表明，因子分解tricks[18]和条件计算[26]可以在计算效率上进行有效提升，同时也会提升模型效果。然而，序列化计算的基本限制仍然存在。

Attention机制已经在许多任务中成为序列建模（sequence modeling）和转化模型（transduction model）的不可欠缺的一个部件，它可以在无需考虑input或output序列上的距离[2,16]的情况下来建模依赖(dependencies)。除了极少的cases外，几乎所有这样的attention机制都会与一个recurrent network一起使用。

**在该工作中，我们提出了Transformer，这种模型结构可以避免recurrence，它完全依赖attention机制来抽取在input和output间的全局依赖性**。Transformer允许更大的并行度(parallelization)，在eight P100 GPUs上训练12小时后，在翻译质量上可以达到一种新的state-of-art效果。

# 2.背景

减小序列化计算开销的目标，也构成了Extended Neural GPU
[20], ByteNet [15] and ConvS2S [8]的基础，它们都使用CNN作为基本构建块，对所有input和output positions并行计算hidden representations。在这些模型中，两个专门的input或output positions的相关信号所需要的ops数目，会随着positions间的距离而增长：这对于ConvS2S是线性的，对于ByteNet是成log关系。**这使得很难学习在较远位置（distant positions）间的依赖[11]**。在Transformer中，操作（operations）的数目可以**减小到常数级别**，虽然在有效识别率上会有损失的代价（因为会对attention-weighted positions进行平均），我们会使用第3.2节中的Multi-Head Attention来消除这现象。

self-attention，有时被称为"intra-attention"，是一种与单个序列上不同位置有关的attention机制，它的目的是**计算该序列的一种表示（representation）**。self-attention已经被成功用于许多任务，包括：阅读理解(reading comprehension)、抽象式摘要(abstractive summarization)、文字蕴含（textual entailment）、独立句子表示任务[4,22,23,19]。

end-to-end memory networks基于一个recurrent attention机制（而非基于sequence-aligned recurrence），已经展示出在单语言问答上和语言建模任务上很好的效果[28]。

据我们所知，**Transformer是首个完全依赖于self-attention的转换模型（transduction model），它无需使用sequence-aligned RNNs或convolution，就可以计算input和output的表示(representations)**。在以下部分，我们会描述Transformer、motivation self-attention、以及在模型上的优点[14,15]，[8]。

# 3.模型结构

大多数比赛采用的神经序列转换模型(neural sequence transduction models)都有一个encoder-decoder结构[5,2,29]。这里，encoder会将一个关于符号表示$$(x_1, \cdots, x_n)$$的输入序列映射到一个连续表示$$z=(z_1, \cdots, z_n)$$的序列上。在给定z后，decoder接着生成一个关于符号(symbols)的output序列$$(y_1, \cdots, y_m)$$，一次一个元素。在每个step上，模型是自回归的（auto-regressive），当生成下一输出时，它会消费前面生成的符号作为额外输入。

Transformer会遵循这样的总体架构：它使用stacked self-attention、point-wise FC-layers的encoder-decoder，如图1的左右所示：

<img src="http://pic.yupoo.com/wangdren23_v/b4bb3caf/81ba10cc.png" alt="1.png">

图1 Transformer模型结构

## 3.1 Encoder Stacks和Decoder Stacks

**Encoder**：encoder由一个N=6的相同层（identical layers）的stack组成。**每一layer具有两个sub-layers。第1个是一个multi-head self-attention机制，第2个是一个简单的position-wise FC 前馈网络**。我们在两个sub-layers的每一个上采用一个residual connection[10]，后跟着layer nomalization[1]。也就是说：**每一sub-layer的output是 $$LayerNorm(x + Sublayer(x))$$**，其中Sublayer(x)是通过sub-layer自身实现的函数。为了促进这些residual connections，模型中的所有sub-layers以及embedding layers会生成维度 $$d_{model}=512$$的outputs。

**Decoder**：该decoder也由一个N=6的相同层（identical layers）的stacks组成。除了包含在每个encoder layer中的两个sub-layers之外，**decoder会插入第三个sub-layer，从而在encoder stack的output上执行multi-head attention**。与encoder相似，我们在每个sub-layers周围采用residual connections，后跟layer normalization。**同时我们在decoder stack中修改了self-attention sub-layer，来阻止position与后序位置有联系**。这种masking机制，结合上output embeddings由一个位置偏移(offset by one position)的事实，可以确保对于位置i的预测只依赖于在位置小于i上的已知outputs。

## 3.2 Attention

attention函数可以被描述成：将一个query和一个key-value pairs集合映射到一个output上，其中：query, keys, values和output都是向量(vectors)。output由对values进行加权计算得到，其中为每个value分配的weight通过query和对应的key的一个兼容函数计算得到。


<img src="http://pic.yupoo.com/wangdren23_v/ba75c826/4d85c908.png" alt="2.png">

图2 (左) Scaled Dot-Product Attention (右) Multi-Head Attention，包含了并行运行的多个attention layers

### 3.2.1  归一化点乘Attention（Scaled Dot-Product Attention）

我们将这种特别的attention称为"Scaled Dot-Product Attention"（图2）。输入包含：querys、维度为$$d_k$$的keys、以及维度为$$d_v$$的values。我们会计算query和所有keys的点乘（dot products），每个除以$$\sqrt{d_k}$$，并使用一个softmax函数来获取在values上的weights。

实际上，我们会同时在一个queries集合上计算attention函数，并将它们打包成一个矩阵Q。keys和values也一起被加包成矩阵K和V。我们会计算矩阵的outputs：

$$
Attention(Q, K, V) = softmax(\frac{Q K^T}{ \sqrt{d_k}}) V
$$

...(1)

两种最常用的attention函数是：additive attention[2]，dot-product（multiplicative） attention。**dot-product attention等同于我们的算法，除了缩放因子$$\frac{1}{\sqrt{d_k}}$$**。additive attention会使用一个单hidden layer的前馈网络来计算兼容函数。两者在理论复杂度上很相似，**dot-product attention更快，空间效率更高，因为它使用高度优化的矩阵乘法代码来实现**。

**如果$$d_k$$值比较小，两种机制效果相似; 如果$$d_k$$值很大，additive attention效果要好于dot-product attention**。对于$$d_k$$的大值我们表示怀疑，dot-product在幅度上增长更大，在具有极小梯度值的区域上使用softmax函数。为了消除该影响，我们将dot-product缩放至$$\frac{1}{\sqrt{d_k}}$$。


### 3.2.2 Multi-Head Attention

我们不会使用$$d_{model}$$维的keys、values和queries来执行单个attention函数，**我们发现：使用学到的不同线性投影将queries、keys和values各自投影到$$d_k$$、$$d_k$$、$$d_v$$维上是有好处的**。在关于queries、keys和values的每一个投影版本上，我们会并行执行attention函数，生成$$d_v$$维的output values。这些值被拼接在一起(concatenated)，再进行投影，产生最后值，如图2所示。

Multi-head attention允许模型联合处理在不同位置处来自不同表示子空间的信息。使用单个attention head，求平均会禁止这样做。


$$
MultiHead(Q, K, V) = Concat(head_1, \cdots, head_h) W^O \\
head_i = Attention(Q W_i^Q, KW_i^K, V W_i^V)
$$

其中，投影是参数矩阵：

$$
W_i^Q \in R^{d_{model} \ \ \times d_k}, \\
W_i^K \in R^{d_{model} \ \ \times d_k}, \\
W_i^V \in R^{d_{model} \ \ \times d_v}, \\
W^O \in R^{h d_v \ \times d_{model}}
$$


在本工作中，我们使用h=8的并行attention layers或heads。对于每一者，我们会使用$$d_k = d_v = d_{model}/h = 64$$ 。由于每个head的维度缩减，总的计算开销与具有完整维度的single-head attention相似。

### 3.2.3 在模型中Attention的应用

Transformer以三种不同的方式使用multi-head attention：

- **"encoder-decoder attention" layers中**：queries来自前一decoder layer，memory keys和values来自encoder的output。这允许在decoder中的每个position会注意(attend)在输入序列中的所有位置。这种方式模仿了在seq2seq模型中典型的encoder-decoder attention机制[31,2,8]。
- **encoder中**：encoder包含了self-attention layers。在一个self-attention layer中，所有的keys, values和queries来自相同的地方：在encoder中的前一layer的output。在encoder中每个position可以注意（attend）在encoder的前一layer中的所有位置。
- **decoder中**：相似的，在decoder中self-attention layers允许在decoder中的每一position注意到在decoder中的所有positions，直到包含该position。我们需要阻止在decoder中的左侧信息流，来保留自回归(auto-regressive)属性。我们通过对softmax（它对应到无效连接）的输入的所有值进行掩码（masking out，设置为$$-\infty$$）来实现该scaled dot-product attention内部。见图2.

## 3.3 Position-wise前馈网络

除了attention sub-layers之外，**在我们的encoder和decoder中的每一层，包含了一个FC前馈网络，它可以独自和等同地应用到每个position上**。在两者间使用一个ReLU来包含两个线性转换。

$$
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2 
$$

...(2)

其中，线性转换在不同的positions上是相同的，在层与层间它们使用不同参数。另一种方式是，使用kernel size为1的两个convolutions。输入和输出的维度是$$d_{model}=512$$，inner-layer具有维度$$d_{ff}=2048$$。

## 3.4 Embedding和softmax

与其它序列转换模型相似，我们使用学到的embeddings来将input tokens和output tokens转换成$$d_{model}$$维的向量。我们也使用常见的学到的线性转换和softmax函数来将decoder output转换成要预测的下一token的概率。在我们的模型中，我们在两个embedding layers和pre-softmax线性转换间共享相同的权重矩阵，这与[24]相同。在embedding layers中，我们会使用$$\sqrt{d_{model}}$$乘以这些权重。

## 3.5 Positional Encoding

由于我们的模型不包含recurrence和convolution，为了利用序列的顺序，我们必须注意一些与相关性(relative)或tokens在序列中的绝对位置有关的信息。我们添加"positional encoding"到在encoder和decoder stacks的底部的input embeddings中。该positional encodings与该embeddings具有相同的维度$$d_{model}$$，因而两者可以求和。positinal encodings有许多选择，可以采用学到（learned）或者固定（fixed）。

在本工作中，我们使用不同频率的sin和cosine函数：

$$
PE_{(pos, 2i)} = sin(pos / 10000 ^{2i/d_{model}}) \\
PE_{(pos, 2i+1)} = cos(pos / 10000 ^{2i/d_{model}})
$$

其中，**pos是position，i是维度**。也就是说：**positional encoding的每个维度对应于一个正弦曲线（sinusoid）**。波长(wavelengths)形成了一个从$$2 \pi$$到$$10000 \cdot 2\pi$$的几何过程。我们选择该函数是因为：我们假设它允许该模型可以很容易学到通过相对位置来进行关注（attend），因为对于任意固定offset k，$$PE_{pos+k}$$可以被表示成一个关于$$PE_{pos}$$的线性函数。

我们也使用学到的positional embeddings进行实验，发现两者版本几乎生成相同的结果（见表3 第E行）。我们选择正弦曲线版本，是因为它可以允许模型对序列长度长于训练期遇到的长度进行推导。

# 4.为什么用self-attention

在本节中，我们比较了self-attention layers与recurrent layers、convolutional layers的多个方面（它们常用于将一个变长序列的符号表示$$(x_1, \cdots, x_n)$$映射到另一个等长的序列$$(z_1, \cdots, z_n)$$上，其中：$$x_i, z_i \in R^d$$），比如：在一个常用的序列转换encoder或decoder中的一个hidden layer。**启发我们使用self-attention主要有三方面考虑**：

- 1.**每一layer的总体计算复杂度**
- 2.**可以并行计算的计算量**，通过所需序列操作(ops)的最小数目进行衡量
- 3.**在长范围依赖（long-range dependencies）间的路径长度**。学习长范围依赖在许多序列转换任务中是一个关键挑战。影响该能力（学习这样的依赖）一个的关键因素是，forward和backward信号的路径长度必须在网络中可穿越（traverse）。在input和output序列中任意位置组合间的路径越短，学习长范围依赖就越容易[11]。这里，我们也比较了由不同layer types构成的网络上，在任意两个input和output positions间最大路径长度。

<img src="http://pic.yupoo.com/wangdren23_v/0ebdf80d/994ef067.png" alt="t1.png">

表1

如表1所示，**一个self-attention layer会使用常数数目的序列执行操作（sequentially executed operations）来连接所有positions**；而一个recurrent layer需要O(n)个序列操作（sequential operations）。**根据计算复杂度，当序列长度n比representation维度d要小时(通常大多数情况下，使用state-of-art模型的句子表示，比如：word-piece和byte-pair表示)，self-attention layers要比recurrent layers快**。为了提升非常长序列任务的计算性能，self-attention可以限制到只考虑在input序列中围绕各自output position为中心的一个size=r的邻居。这可以将最大路径长度增大到$$O(n/r)$$。我们在未来会计划研究该方法。

kernel宽度$$k < n$$的单个convolutional layer，不会连接上input和output positions的所有pairs。在连续kernels的情况下，这样做需要一个$$O(n/k)个$$ convolutional layers的stack；在扩大卷积(dilated convoluitons)的情况下需要$$O(log_k(n))$$，这会增加在网络中任意两个positions间的最长路径的长度。卷积层(convolutional layers)通常要比recurrent layers开销更大，会乘以一个因子k。然而，可分离卷积(Separable convolutions)，将复杂度减小到$$O(k \cdot n \cdot d + n \cdot d^2)$$。有了$$k=n$$，然而，一个可分离卷积的复杂度等于一个self-attention layer和一个point-wise前馈layer，在我们的模型中采用该方法。

另一个好处是，**self-attention可以生成更多可解释模型**。我们从我们的模型中内省(inspect)出attention分布，并在附录部分讨论示例。单独的attention heads不仅可以很明确地学习执行不同的任务，出现在展示行为中的多个（）还可以与句子的形态结构和语义结构相关。

# 5.训练

## 5.1 训练数据和Batching

我们在标准的WMT 2014 English-German dataset上进行训练，它包含了将近450w句子对（sentence pairs）。句子使用byte-pair encoding进行编码，它具有一个37000 tokens的共享的source-target词汇表。对于英译法，我们使用更大的WMT 2014-English-French数据集，它包含了36M句子，32000个word-piece词汇表。句子对(sentence pairs)通过近似的序列长度进行打包。每个training batch包含了一个句子对集合，它会近似包含25000个source tokens和25000个target tokens。

## 5.2 硬件与schedule

在8块nvidia P100 GPUs上进行模型训练。对于我们的base models，它使用paper上描述的超参数，每个training step会花费0.4s。我们会为base models训练10w个steps 或12小时。对于我们的大模型（表3底部描述），step time是1s。大模型会训练30000 steps（3.5天）。

## 5.3 Optimizer

我们使用Adam optimizer，$$\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$$。我们会根据训练过程调整learning rate，根据以下公式：

$$
lrate = d_{model}^{-0.5} \cdot min(step\_num ^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
$$

...(3)

这对应于为前warmup_steps阶段线性增加learning rate，然后之后与step_num的平方根成比例减小。我们使用的warmup_steps=4000.

# 6.结果


## 6.1 机器翻译

在WMT 2014 English-to-German翻译任务上，big transformer model(见表2: Transformer(big))的效果要比之前最好的模型（包括ensembles）要好2.0 BLEU，达到一个新的state-of-art BLEU分:28.4. 该模型的配置列在了表3的底部。训练会在8张P100 GPUs上训练3.5天。我们的base model胜过之前发布的所有模型和ensembles，训练开销只是其他模型的一小部分。

<img src="http://pic.yupoo.com/wangdren23_v/5a19edc7/ef7d969e.png" alt="t2.png">

表2: 

在WMT 2014 English-to-French翻译任务上，我们的big model的BLEU得分为41.0, 比之前发布的single models都要好，训练开销只有之前state-of-art model的1/4. 对于English-to-French所训练的Transformer(big)模型，使用dropout rate为：$$P_{drop}=0.1$$，而非0.3。

对于base models，我们使用一个single model，它通过最后的5个checkpoint进行平均获得，每个checkpoint会有10分钟的时间间隔。对于big models，我们则对最后的20个checkpoints进行平均得到。我们使用的beam search的beam size为4, length penalty为 α = 0.6. 这些超参数会在实验之后选择。我们在推断(inference)期间设置最大的output length为: (input length+50)，当可能时会提前终止。

表2归纳了我们的结果，并比较了与其它模型结构间的翻译质量和训练开销。我们估计了用于训练一个模型的浮点操作的数目，乘以训练时间，所使用的GPUs数目，以及每个GPU的持续的(sustained)单精度浮点能力(single-precision floating-point capacity)。


## 6.2 模型变种

为了评估Transformer中不同组件的重要性，我们以不同的方式区分我们的base model，并在数据集newstest2013上测量了在English-to-German翻译上的效果。我们使用前一节描述的beam serach，但没有进行checkpoint averaging。我们的结果在表3中。

<img src="http://pic.yupoo.com/wangdren23_v/a2c0f8a3/617f5c68.png" alt="t3.png">

表3

在表3 rows(A)，我们会使用不同的attention heads数目、attention key和value维度，来保持常数级的计算量，如3.2.2节所描述的。而single-head attention是0.9 BLEU，它比最佳setting要差，如果有太多heads质量也会下降。

在表3 rows(B)，我们观察到减小attention key size $$d_k$$会伤害模型质量。这建议我们，决定兼容并不容易，一个dot-product更复杂的兼容函数可能会更有意义。进一步观察（C）和（D），模型越大越好，dropout在避免over-fitting上更有用。在row(E)上，我们使用已经学到的positional embedding[8]来替换了我们的sinusoidal positional encoding，结果与base model几乎相同。

# 7.结论

Transformer是首个完全基于attention的序列转换模型（sequence transduction model），它使用multi-headed self-attention来替换在encoder-decoder架构中常用的recurrent layers。

对于翻译任务，Transformer训练要比基于recurrent或convolutional layers的结构要快很多。在WMT 2014 English-to-German和WMT 2014 English-to-French翻译任务上，我们达到了一个新的state-of-the-art效果。

我们对attention-based模型的将来很激动，计划应用到其它任务上。我们计划将Transformer扩展到涉及输入输出形态的非文本问题，研究local, restricted attention mechanisms以有效处理大的inputs和outputs（比如：图片、音频、视频）。生成更少序列是另一个研究目标。

代码在：[https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)



# 参考

- 1.[https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
