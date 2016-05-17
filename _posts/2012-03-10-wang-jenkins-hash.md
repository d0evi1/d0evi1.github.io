---
layout: post
title: Wang/Jenkins Hash算法
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2012-12-24
tags: [hash算法]
---

Wang/Jenkins Hash算法在网上提到的也甚多，但是很少有人或有文章能系统地能将该算法的来龙去脉说明白。于是，我就充当了该苦工，幸好还是找到了一些东西，尝试着将该算法简单道来。

最早，Bob Jenkins提出了多个基于字符串通用Hash算法（搜Jenkins Hash就知道了），而Thomas Wang在Jenkins的基础上，针对固定整数输入做了相应的Hash算法。因而，其名字也就成了Wang/Jenkins Hash，其64位版本的 Hash算法如下：


{% highlight c %}

uint64_t hash(uint64_t key) {
    key = (~key) + (key << 21); // key = (key << 21) - key - 1;
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8); // key * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4); // key * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}
{% endhighlight %}


其关键特性是：

- 1.雪崩性（更改输入参数的任何一位，就将引起输出有一半以上的位发生变化）
- 2.可逆性（input ==> hash ==> inverse_hash ==> input）

其逆Hash函数为：


{% highlight c %}

uint64_t inverse_hash(uint64_t key) {
    uint64_t tmp;

    // Invert key = key + (key << 31)
    tmp = key-(key<<31);
    key = key-(tmp<<31);

    // Invert key = key ^ (key >> 28)
    tmp = key^key>>28;
    key = key^tmp>>28;

    // Invert key *= 21
    key *= 14933078535860113213u;

    // Invert key = key ^ (key >> 14)
    tmp = key^key>>14;
    tmp = key^tmp>>14;
    tmp = key^tmp>>14;
    key = key^tmp>>14;

    // Invert key *= 265
    key *= 15244667743933553977u;

    // Invert key = key ^ (key >> 24)
    tmp = key^key>>24;
    key = key^tmp>>24;

    // Invert key = (~key) + (key << 21)
    tmp = ~key;
    tmp = ~(key-(tmp<<21));
    tmp = ~(key-(tmp<<21));
    key = ~(key-(tmp<<21));

    return key;
}

{% endhighlight %}


由上述的算法实现可知，原始的hash算法过程是非常快的，而其逆Hash算法则比较慢一些。

参考：

- 1.[jenkins 32位Hash算法](http://burtleburtle.net/bob/hash/integer.html)
- 2.[Geoffrey Irving's blog](http://naml.us/blog/tag/thomas-wang)
