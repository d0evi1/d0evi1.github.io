---
layout: post
title: Skip-thought Vectors介绍
description: 
modified: 2016-08-09
tags: [cnn]
---

# 1.介绍

对于词（words）的分布式组成语义的算法开发是一个长期存在的开放性难题。最近几年的算法有：将word vectors映射到sentence vectors（包括recursive networks, recurrent networks, convolutional networks，以及recursive-convolutional方法）。所有的这些方法会生成句子表示，传给一个监督式任务，依赖一个class label来对组成权重（composition weights）做BP算法。因而，这些方法能学到高质量句子表示，但只能对自己的特定任务进行调整。paragraph vector是另一种方法，它通过引入一个分布式句子索引作为模型的一部分，以非监督式学习进行句子表示。

本文中，我们考虑了另一种loss function，可以用于任何组成操作（composition operator）上。考虑以下的问题：是否存在一个任务，它对应的loss允许我们学习高度泛化的句子表示？受使用word vector学习的启发，我们提出了一个目标函数，它从句子级别上抽象了skip-gram模型。也就是说，它不再使用单个word来预测它的上下文，我们会encode一个句子。因而，任何组成操作（composition operator）都适用于一个句子编码器(sentence encoder)，只是目标函数被修改了而已。图1展示了该模型，我们会调用我们的skip-thoughts模型和向量。

<img src="http://pic.yupoo.com/wangdren23/HqyASSUo/medish.jpg">

图1: skip-thoughts模型。给定一个tuple（$$s_{i-1}, s_i, s_{i+1}$$），$$s_i$$表示book中的第i个句子，$$s_i$$被编码并尝试重构前一个句子$$s_{i+1}$$和下一个句子$$s_{i+1}$$。在本例中，输入的句子三元组：I got back home. I could see the cat on the steps. This was strange. 未绑定箭头被连接到encoder output上。颜色表示哪个component共享参数。(与skip-gram有点像)

<img src="http://pic.yupoo.com/wangdren23/HqATZTnf/medish.jpg">

表1: BookCorpus dataset的统计信息

我们的模型依赖于一个关于连续文本的训练语料。我们选择使用一个小说集合BookCorpus dataset来训练我们的模型。这些书由未出版的作者编写。该dataset具有6种不同的种类：Romance, Fantasy, Science fiction , Teen等。表1高亮了该语料的统计。伴随着故事，书包含着对话，感情（emotion）和广泛的字符交叉。另外，训练集的量很大，不会偏向于任何特定领域或应用。表2展示了该语料中句子的最近邻。该结果展示了skip-thought vectors精确地捕获了编码后的句子的语义和结构。

<img src="http://pic.yupoo.com/wangdren23/HqAUDnUT/medish.jpg">

表2: 在每个样本中，第一个句子是一个query，而第二个句子是它的最近邻。通过从语料中随机抽取5w个句子中，通过计算cosine相似度进行最近邻分数排序。

我们以新的setting评估了我们的向量：在学到skip-thoughts后，冻结模型，使用encoder作为一个泛化的特征抽取器（generic feature extractor）以用于特定任务。在我们的实验中，我们考虑了8个任务：句义相关性，段落检测，图像句子排序以及其它5个标准的分类benchmarks。在这些实验中，我们抽取了skip-thought向量，并训练了线性模型来评估它的表示（representations），没有任何额外的参数fine-tuning。结果说明，skip-thoughts提出的泛化表示对所有任务都很robust。

一个难点是，这样的实验会构建一个足够大的词汇表来编码句子。例如，一个从wikipedia文章中的句子可能包含了与我们的词汇表高度不一样的名词。为了解决这个问题，我们学到了一个mapping：从一个模型传递给另一个模型。通过使用cbow模型预训练好的word2vec表示，我们学到了这样的一个线性映射：将在word2vec空间中的一个词映射到encoder词汇表空间中的一个词上。学到的该mapping会使用所有单词，它们共享相同的词汇表。在训练后，出现在word2vec中的任何word，可以在encoder word embedding空间中获得一个vector。

# 2.方法

## 2.1 引入skip-ghought vectors

我们使用encoder-decoder模型框架来对待skip-thoughts。也就是说，一个encoder会将words映射到一个句子向量（sentence vector）上，一个decoder会用于生成周围的句子。在该setting中，一个encoder被用于将一个英文句子映射到一个向量。decoder接着根据该向量来生成一个关于源英文句子（source sentence）的翻译（translation）。已经探索了许多encoder-decoder pair选择，包括：ConvNet-RNN，RNN-RNN，LSTM-LSTM。源句子表示（source sentence representation）也可以通过使用一个attention机制来动态变化，用于说明任何时候只有相关的才用于翻译（translation）。**在我们的模型中，我们使用一个带GRU activations的RNN encoder，以及一个conditional GRU的RNN decoder**。该模型组合近似等同于神经机器翻译中的RNN encoder-decoder【11】。GRU展示了在序列建模任务中效果比LSTM要好，并且更简单。GRU units只有两个gates，不需要使用一个cell。而我们的模型则使用RNN，只要能在它之上进行BP算法，任何encoder和decoder可以被使用。

假设我们给定了一个句子的tuple：$$(s_{i-1}, s_i, s_{i+1})$$。假设$$w_i^t$$表示了句子中的第t个word，$$x_i^t$$表示它的word embedding。我们将模型描述成三部分：encoder，decoder，目标函数。

**Encoder**：假设$$w_i^1, ..., w_i^N$$是句子$$s_i$$中的words，其中N表示在句子中的words数目。在每个step中，encoder会产生一个hidden state：$$h_i^t$$，它可以解释成序列$$w_i^1,...,w_i^t$$的表示（representation）。hidden state：$$h_i^N$$可以表示整个句子。

为了encode一个句子，我们对下面的等式进行迭代（这里去掉了下标i）：

$$
r^t = \sigma(W_r x^t + U_r h^{t-1}) 	
$$ ... (1)

$$
z^t = \sigma(W_z x^t + U_z h^{t-1})
$$ ... (2)

$$
\bar{h}^t = tanh(W x^t + U (r^t \odot h^{t-1}) 
$$ ... (3)

$$
h^t = (1-z^t) \odot h^{t-1} + z^t \odot \bar{h}^t
$$ ...(4)

其中 $$\bar{h}^t$$是在时间t提出的状态更新，$$z^t$$是update gate，$$r^t$$是reset gate（$$\odot$$）表示一个component-wise product。两个update gates会采用0到1间的值。

**Decoder**: decoder是一个神经语言模型，它的条件是encoder output $$h_i$$。该计算与encoder相似，除了我们引入了矩阵$$C_z，C_r$$，以及C，它们被用于偏置由句子向量计算得到的update gate，reset gate和hidden state。一个decoder会被用于下一个句子$$s_{i+1}$$，而第二个decoder被用于前一句子$$s_{i-1}$$。Separate参数被用于每个decoder，除了词汇矩阵V，它的权重矩阵会连接decoder的hidden state，以便计算在词上的一个分布。我们在下一个句子$$s_{i+1}$$上描述decoder，通过一个在前一句子$$s_{i-1}$$上的类似计算得到。假设$$h_{i+1}^t$$表示decoder在时间t的hidden state。对下面的等式进行迭代（丢掉下标i+1）：

$$
r^t = \sigma(W_r^d x^{t-1} + U_r^d h^{t-1} + C_r h_i)
$$ ...(5)

$$
z^t = \sigma(W_z^d x^{t-1} + U_z^d h^{t-1} + C_z h_i)
$$ ...(6)

$$
\bar{h}^t = tanh(W^d x^{t-1} + U^d (r^t \odot h^{t-1} + C h_i
$$ ...(7)

$$
h_{i+1}^t = (1-z^t) \odot h^{t-1} + z^t \odot \bar{h}^t
$$ ...(8)

给定$$h_{i+1}^t$$，单词$$w_{i+1}^t$$的概率给出了前(t-1) words，encoder vector为：

$$
P(w_{i+1}^t | w_{i+1}^{<t}, h_i) \propto exp(v_{w_{i+1}^t} h_{i+1}^t )
$$ ...(9)

其中，$$v_{w_{i+1}^{t}}$$表示V的行，对应于word $$w_{i+1}^t$$。对于前面句子$$s_{i-1}$$可以做一个类似的计算。

**目标函数**。给定一个tuple $$(s_{i-1}, s_i, s_{i+1})$$，目标优化为：

$$
\sum_{t} log P(w_{i+1}^t | w_{i+1}^{<t}, h_i) + \sum_{t} log P(w_{i-1}^t | w_{i-1}^{<t}, h_i)
$$ ...(10)

总的目标函数是对所有这样的training tuples进行求和。

## 2.2 词汇表膨胀

现在，我们会描述如何将我们的encoder的词汇表扩展到那些在训练期间未见过的词上。假设我们有个被训练的模型引入了词表示（word representations），假设$$V_{rnn}$$表示了RNN的词向量空间。我们假设词汇表$$V_{w2v}$$比$$V_{rnn}$$更大。我们的目标是构建一个mapping f: $$ V_{w2v} \rightarrow V_{rnn}$$，它由一个矩阵W进行参数化，比如：$$v'=Wv$$，其中$$v \in V_{w2v}$$， $$v' \in V_{rnn}$$。受[15]的启发，它会学到在词空间转移之间的线性映射，我们为矩阵W求解一个非正则的L2线性回归loss。这样，对于编码中的句子，任何来自$$V_{w2v}$$的词可以被映射到$$V_{rnn}$$。

# 3 实验

在我们的实验中，我们在BookCorpus dataset上评估了我们的encoder作为一个通用的feature extractor的性能。实验setup如下：

- 使用学到的encoder作为一个feature extractor，抽取所有句子的skip-thought vectors
- 如果该任务涉及到计算句子对（pairs of sentences）之间的分数，会计算pairs间的component-wise features。
- 在抽取的features上训练一个线性分类器，在skip-thoughts模型中没有任何额外的fine-tuning或者backpropagation。

我们限定在线性分类器主要出于两个原因。第一个是为了直接评估计算向量的representation quality。如果使用非线性模型可能会获得额外的性能增益，但这超出了我们的目标。再者，它可以允许更好地分析学到的representations的优点和缺点。第二个原因是，重现（reproducibility）很简单。

## 3.1 训练细节

为了引入skip-thought vectors，我们在我们的book corpus上训练了两个独立的模型。一个是带有2400维的unidirectional encoder，它常被称为uni-skip。另一个是一个bidirectional model，forward和backward每个各1200维。该模型包含了两个encoder，它们具有不同的参数：一个encoder会给出正确顺序的句子，另一个会给出逆序的句子。输出接着被拼接形成一个2400维的向量。我们将该模型称为bi-skip。对于训练，我们会初始化所有的recurrent矩阵：进行正交初始化。non-recurrent weights则从一个[-0.1, 0.1]的均匀分布上进行初始化。使用mini-batches的size=128, 如果参数向量的norm超过10, 梯度会被裁减（clip）。我们使用Adam算法进行optimization。模型会被训练两周。另外作为一个额外的试验，我们使用一个组合模型导出了试验结果，包含了uni-skip和bi-skip，会生成4800维的向量。我们称该模型为combine-skip。

在被模型被训练后，我们接着使用词汇扩展，将word embedding映射到RNN encoder空间上。会使用公开提供的CBOW word vectors[2]。被训练的skip-thought会有一个词汇size=20000个词。在从CBOW模型中移除多个word样本后，会产生一个词汇size=930911个词。这样，即使被训练的skip-thoughts模型具有20000个词，在词汇表扩展后，我们可以对930991个可能的词进行成功编码。

由于我们的目标是将skip-thoughts作为一个通用的feature extractor进行评估，我们将文本预处理保持最低水平。当编码新句子时，除了基本的tokenization，不会有额外的预处理。这样做可以测试我们的vectors的健壮性。作为一个额外的baseline，我们也会考虑来自uni-skip模型学到的word vectors的平均（mean）。我们将该baseline称为bow。这会决定着在BookCorpus上训练的标准baseline的效果。

## 3.2 语义相关性
## 3.3 段落检测
## 3.4 Image-sentence ranking
## 3.5 Classification benchmarks

略

## 3.6 skip-thoughts可视化

t-sne.

# 参考

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)