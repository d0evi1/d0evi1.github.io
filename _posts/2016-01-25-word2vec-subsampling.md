---
layout: post
title: word2vec中的subsampling
description: 
modified: 2016-01-10
tags: [word2vec]
---

# 介绍

Mikolov在paper[Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)中，提到高频词的subsampling问题，以下我对相关节选进行的简单中文翻译：

在非常大的语料中，最高频的词汇，可能出现上百万次（比如：in, the, a这些词）。这样的词汇通常比其它罕见词提供了更少的信息量。例如，当skip-gram模型，通过观察"France"和"Paris"的共现率(co-occurrence)，Skip-gram模型可以从中获益；而"France"和"the"的共现率则对模型贡献很少；而几乎每个词都常在句子中与"the"共同出现过。该思路也常用在相反的方向上，高频词的向量表示，在上百万样本训练完后不会出现大变化。

为了度量这种罕见词与高频词间存在不平衡现象，我们使用一个简单的subsampling方法：训练集中的每个词wi，以下面公式计算得到的概率进行抛弃：

<img src="http://www.forkosh.com/mathtex.cgi?P(w_i)=1-\sqrt{\frac{t}{f(w_i)}}">

f(wi)是wi的词频，t为选中的一个阀值，通常为10^-5周围(0.00001)。我们之所以选择该subsampling公式，是因为：它可以很大胆的对那些词频大于t的词进行subsampling，并保留词频的排序(ranking of the frequencies)。尽管subsampling公式的选择是拍脑袋出来的（启发式的heuristically），我们发现它在实践中很有效。它加速了学习，并极大改善了罕见词的学习向量的准确率（accuracy）。

# 具体实现

有道之前的<deep learning实战之word2vec>中，提到的该subsampling描述也不准确。在当中的描述是：

<img src="http://www.forkosh.com/mathtex.cgi?P(w_i)=1-(\sqrt{\frac{sample}{freq(w_i)}}+\frac{sample}{freq(w_i)})">


而实际中，采用的是：

<img src="http://www.forkosh.com/mathtex.cgi?P(w_i)=\frac{random}{65535}-(\sqrt{\frac{sample}{freq(w_i)}}+\frac{sample}{freq(w_i)})">

部分代码：

{% highlight c %}

// 进行subsampling，随机丢弃常见词，保持相同的频率排序ranking.
// The subsampling randomly discards frequent words while keeping the ranking same
if (sample > 0) {

  // 计算相应的抛弃概率ran.
  real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;

  // 生成一个随机数next_random.
  next_random = next_random * (unsigned long long)25214903917 + 11;

  // 如果random/65535 - ran > 0, 则抛弃该词，继续
  if (ran < (next_random & 0xFFFF) / (real)65536) 
      continue;
}

{% endhighlight %}

为此，我简单地写了一个python程序，做了个测试。程序托管在github上，[点击下载](https://github.com/d0evi1/word2vec_insight/blob/master/subsampling.py)

下面提供了三种方法，最终生成的图可以看下。对于前两种方法，基本能做到与原先词频正相关，最后使用时，需要设置一个阀值，砍掉高频词。而最后一种方法，效果也不错（虽然偶有会存留高频词，或者低频词也同样被砍掉）。而word2vec采用的正是第3种方法(大于0的采样点会被抛弃掉)。

<img src="http://pic.yupoo.com/wangdren23/G7O9UJ6m/medish.jpg">
