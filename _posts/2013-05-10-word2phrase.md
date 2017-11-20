---
layout: post
title: word2phrase解读 
description: 
modified: 2013-05-10
tags: [word2vec word2phrase 互信息]
---

word2vec的源码中，自带着另一个工具word2phrase，之前一直没有关注，可以用来发现短语. 代码写的很精练，不到300行，那么我们可以看一下，这不到300行的代码是如何实现短语发现的。

# 一、参数

- train:   训练文件（已经分好词的文件）
- debug: 是否开启调试模式 (debug=2，在训练时会有更多的输出信息)
- output: 输出文件
- min-count: 最小支持度，小于该值的词将不会被采用，缺省为5
- threshold:  超过该得分，才形成短语，该值越高，形成的短语也会越少。(default:100)

# 二、词汇表

{% highlight c %}

struct vocab_word {
    long long cn;    //
    char *word;    //
};

{% endhighlight %}


每个词汇由vacab_word构成，由这些词汇形成一个大的词汇表：vacab_hash。（最大条目：500M个）

# 三、模型训练 TrainModel（）

内部原理很简单：是一个分词后2-gram的模型，统计出现频次. 本质是利用互信息(mi)计算内聚程度。（内部做了一些内存优化＝>牺牲了一点点准确性，如果想更准，不如自己写map-reduce或spark）

公式为：

score = (pab - min_count) / (real)pa / (real)pb * (real)train_words;

可以简化为：

$$
\frac{P_{ab}}{P_{a}P_{b}}
$$


互信息只是简单的多个log2而已.

如果你有做过新词发现等经验，就知道这并不是一个比较优的方法，目前流行的做法是：互信息＋min(左右邻字信息熵). 这里的字换成词即是短语发现.
