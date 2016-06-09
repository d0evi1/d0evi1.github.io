---
layout: post
title: 乱语（无意义/随机字符串）识别 
description: 
modified: 2013-07-03
tags: [无意义识别 乱语识别 随机串识别 markov链 马尔可夫链]
---

我们知道，在这个ugc的年代里，网络上各种各样支持文本输出的地方（比如：无意义的微博及评论、利用随机生成器注册id等等），经常出现一些无意义的乱语（英文叫：gibberish），比如“asdfqwer”。如何去识别它，是一个有意思的主题。

直觉跳出最简单的一种方式是，收集一本较全的英文字典，然后过滤一次，看看目标是否落在字典中。这是一种方式。准确率会有大幅提升，但是肯定还是会有一些case是难以解决的，比如：两个有意义的词刚好连在一起，却不在字典中。

另外我们再进一步思考一下，是否有什么算法进行训练，进而识别直接识别是否是乱语呢？可以考虑使用markov链模型。

对于一句话或短语来说：“hello buddy”，每个字母都与上下两个字母有关，这种关系可以通过2-gram来表示。比如：he, el, ll, lo, o[space], [space]b, ...。

我们可以建立这样一个状态转移矩阵：

|  字母       | a | b | c | ... | [space] |
|:---------|:--|:--|:---|:-----|:---------|
| a       | Paa  |  Pab |  Pac |     |         |
| b       | Pba  | ...  |   |     |         |
| c       | Pca  |   |   |     |         |
| ...     |   |   |   |     |         |
| [space] |   |   |   |     |         |


在一个语料库， 我们会统计这些2-gram的频数，并将它们进行归一化。这样每个字母后的下一个字母出现都有一个概率分布（26个字母加空格）。

对于字母a，它的下一个输入为b的组成的2-gram的状态转换概率为：

<img src="http://www.forkosh.com/mathtex.cgi?P_{ab}=log(\frac{count(ab)}{\sum{count(ai)}})"> 

为什么不是直接概率，而要使用log呢？

- 由于字典很大，但一些词的出现概率会过小，计算机下溢(undeflow problem)将它们当成0来处理。
- 我们最后要求的是整个“hello buddy”的概率，即：p = prob(he) * prob(el) * prob(ll) * ... prob(dy)的概率，这在英文2gram的语言模型，而使用log可以利用log的性质：log(ab)=log(a)+log(b)
- 最后，再通过e转换回正常概率即可. <img src="http://www.forkosh.com/mathtex.cgi?e^{log(p)}=p">


如何判断是否是乱语？

我们可以收集一些乱语（bad），还有一些比较好的短语或语汇(good)。然后分别统计出对应bad以及good的平均转移概率。

平均转移概率=概率/转移次数

阀值的选取？

一般来说，good的平均转移概率，要比bad的大。可以选择平均：

thresh = (min(good_probs)+max(bad_probs))/2

- 大于thresh，认为是good。
- 小于等于thresh，认为是bad。

ok，利用这个模型，就可以识别字典之外的乱语了。如果你的训练语料很标准的话，那么相应的效果就会好很多。


参考：

- [why log](https://squarecog.wordpress.com/2009/01/10/dealing-with-underflow-in-joint-probability-calculations/)
- [相应的算法](https://github.com/rrenaud/Gibberish-Detector)
