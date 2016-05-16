---
layout: page
title: matplot中的一些常用作图
tagline: 介绍
---
{% include JB/setup %}

# 1.介绍

在介绍scipy之前，我们先来回顾下，统计学原理中出现的一些概念，附带的链接，直接指定了对应的scipy中的链接：

## 2.1 各种平均数

- 1.[中心值/均值(average加权平均)](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.average.html)
- 2.[众数(mode)](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html#scipy.stats.mode)
- 3.[中位数(median)](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.median.html)
- 4.[均值(mean:几何平均)](http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html#numpy.mean)

## 2.2 变差

- 1.[极差(range)](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.ptp.html#numpy.ptp)
- 2.[四分位数极差](http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html)
- 3.[标准差(standard deviation)](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.std.html)
- 4.[方差(variance)](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.var.html#numpy.var)

## 2.3 标准误差

- 1.[标准误差(standard error)](http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.sem.html)

{% highlight python %}

numpy.std([0,4,8,12,16])/math.sqrt(5-1)

{% endhighlight %}

## 2.4 标准得分

- 1.[标准得分z-score(standard scores)](http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.mstats.zscore.html)


- [scipy](http://docs.scipy.org/doc/scipy/reference/stats.html)
- [numpy](http://docs.scipy.org/doc/numpy-1.10.1/reference/routines.statistics.html)
