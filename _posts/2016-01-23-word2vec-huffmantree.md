---
layout: post
title: word2vec中的Huffman树
description: 
modified: 2016-01-23
tags: [word2vec+Huffman]
---

# 介绍

如果你在大学期间学过信息论或数据压缩相关课程，一定会了解Huffman Tree。建议首先在wikipedia上重新温习下Huffman编码与Huffman Tree的基本概念: [Huffman编码wiki](https://zh.wikipedia.org/wiki/%E9%9C%8D%E5%A4%AB%E6%9B%BC%E7%BC%96%E7%A0%81) 

简单的说（对于文本中的字母或词），Huffman编码会将出现频率较高（也可认为是：权重较大）的字（或词）编码成短码，而将罕见的字(或词)编码成长码。对比长度一致的编码，能大幅提升压缩比例。

而Huffman树指的就是这种带权路径长度最短的二叉树。权指的就是权重W（比如上面的词频）；路径指的是：从根节点到叶子节点的路径长度L；带权路径指的是：树中所有的叶结点的权值乘上其到根结点的路径长度。WPL=∑W*L，它是最小的。

# word2vec的Huffman-Tree实现

为便于word2vec的Huffman-Tree实现，我已经将它单独剥离出来，具体代码托管到github上: **[huffman_tree代码下载](https://github.com/d0evi1/word2vec_insight/blob/master/huffman_tree.cpp)**。示例采用的是wikipedia上的字母：

即：F:2, O:3, R:4, G:4, E:5, T:7 

这里有两个注意事项：

- 1.对于单个节点分枝的编码，wikipedia上的1/0分配是定死的：即左为0，右为1（但其实分左右是相对的，左可以调到右，右也可以调到左）。而word2vec采用的方法是，两侧中哪一侧的值较大则为1，值较小的为0。当然你可以认为（1为右，0为左）。
- 2.word2vec会对词汇表中的词汇预先从大到小排好序，然后再去创建Huffman树。在CreateBinaryTree()调用后，会生成最优的带权路径最优的Huffman-Tree。

最终生成的图如下：

<img src="http://pic.yupoo.com/wangdren23/G7Fugo2a/medish.jpg">

此图中，我故意将右边的T和RG父结节调了下左右，这样可以跳出上面的误区（即左为0，右为1。其实是按大小来判断0/1）

相应的数据结构为：

{% highlight c %}

/**
 * word与Huffman树编码
 */
struct vocab_word {
  long long cn;     // 词在训练集中的词频率
  int *point;       // 编码的节点路径
  char *word,       // 词
       *code,       // Huffman编码，0、1串
       codelen;     // Huffman编码长度
};

{% endhighlight %}

最后，上面链接代码得到的结果：

	word=T	cn=7	codelen=2	code=10	point=4-3-
	word=E	cn=5	codelen=2	code=01	point=4-2-
	word=G	cn=4	codelen=3	code=111	point=4-3-1-
	word=R	cn=4	codelen=3	code=110	point=4-3-1-
	word=O	cn=3	codelen=3	code=001	point=4-2-0-
	word=F	cn=2	codelen=3	code=000	point=4-2-0-

整个计算过程设计的比较精巧。使用了三个数组：count[]/binary[]/parent_node[]，这三个数组构成相应的Huffman二叉树。有vocab_size个叶子节点。最坏情况下，每个节点下都有一个叶子节点，二叉树的总节点数为vocab_size * 2 - 1就够用了。代码使用的是 vocab_size * 2 + 1。

当然，如果你有兴趣关注下整棵树的构建过程的话，也可以留意下这部分输出：

{% highlight c %}
count[]:	7 5 4 4 3 2 1000000000000000 1000000000000000 1000000000000000 1000000000000000 1000000000000000 1000000000000000
binary[]:	0 0 0 0 0 0 0 0 0 0 0 0
parent[]:	0 0 0 0 0 0 0 0 0 0 0 0
	
--step 1:
count[]:	7 5 4 4 3 2 5 1000000000000000 1000000000000000 1000000000000000 1000000000000000 1000000000000000
binary[]:	0 0 0 0 1 0 0 0 0 0 0 0
parent[]:	0 0 0 0 6 6 0 0 0 0 0 0
	
--step 2:
count[]:	7 5 4 4 3 2 5 8 1000000000000000 1000000000000000 1000000000000000 1000000000000000
binary[]:	0 0 1 0 1 0 0 0 0 0 0 0
parent[]:	0 0 7 7 6 6 0 0 0 0 0 0
	
--step 3:
count[]:	7 5 4 4 3 2 5 8 10 1000000000000000 1000000000000000 1000000000000000
binary[]:	0 1 1 0 1 0 0 0 0 0 0 0
parent[]:	0 8 7 7 6 6 8 0 0 0 0 0
	
--step 4:
count[]:	7 5 4 4 3 2 5 8 10 15 1000000000000000 1000000000000000
binary[]:	0 1 1 0 1 0 0 1 0 0 0 0
parent[]:	9 8 7 7 6 6 8 9 0 0 0 0
	
--step 5:
count[]:	7 5 4 4 3 2 5 8 10 15 25 1000000000000000
binary[]:	0 1 1 0 1 0 0 1 0 1 0 0
parent[]:	9 8 7 7 6 6 8 9 10 10 0 0

{% endhighlight %}


## 参考

1.[Huffman编码](https://zh.wikipedia.org/wiki/%E9%9C%8D%E5%A4%AB%E6%9B%BC%E7%BC%96%E7%A0%81)



