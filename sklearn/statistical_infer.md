---
layout: post
title: 统计推断的一些基本概念 
description: 
modified: 2014-07-21
tags: [样本估计 假设检验 零假设]
---

# 1.介绍

建议读一下《统计学-基本概念和方法》。本文基本概念来自于它。

统计推断（statistical inference）: 从样本数据得出与总体参数值有关的结论。
包括 估计(estimation) 和 假设检验(hypothesis testing)

机器学习中的对应概念也来自于此。

为什么要用样本替代总体？

- 统计总体代价高
- 由于样本的不准确，可引入抽样误差

## 1.1 样本与总体

两者的参数：




## 2 估计

估计主要有这么两类：

- 点估计(point estimation): 是一个用来估计参数值的**数**.
- 区间估计(interval estimation): 是一个用于参数估计值的**区间**.

注意点：

- 点估计：无偏估计 vs. 有偏估计
- 区间估计：抽样误差、置信区间、置信水平

### 2.1 置信区间

置信区间的表示：[统计量-抽样误差, 统计量+抽样误差]

数学表示：

样本个数为n，总体个数为N

1.

- 标准正态分布为：<img src="http://www.forkosh.com/mathtex.cgi?Z - N(0,1)">，z值为：z=(x-μ)/σ 

- [标准正态分布表查表](http://static.oschina.net/uploads/img/201509/14164301_5fGD.png)

2.

- 样本比例的数学期望：<img src="http://www.forkosh.com/mathtex.cgi?E(p)=\pi">
- 样本比例的方差（重复抽样）： <img src="http://www.forkosh.com/mathtex.cgi?\sigma^{2}=\frac{\pi(1-\pi)}{n}">
- 样本比例的方差（不重复抽样）：<img src="http://www.forkosh.com/mathtex.cgi?\sigma^{2}=\frac{\pi(1-\pi)}{n}(\frac{N-n}{N-1})">


#### 一、总体百分数的置信区间:

<img src="http://www.forkosh.com/mathtex.cgi?[P-1.96\sqrt\frac{P(100-P)}{n}, P+1.96\：\frac{P(100-P)}{n}]">

（p=0.975, z=1.96）

它的一个可用于快速计算的95%置信区间的近似计算（令P=50）：

<img src="http://www.forkosh.com/mathtex.cgi?[P-\frac{100}{\sqrt{n}}, P+\frac{100}{\sqrt{n}}]">

为什么大多数样本要求有1200个响应者的原因。（95%置信区间，100/sqrt(n)=3，n=1111）

#### 二、总体均值的置信区间：

- [t分布](http://baike.baidu.com/link?url=6-_ll1N1LVImC7NPGslgac2snge1iwyLrFhzW59E-O6RHOHiLwpE4XslYRT_DL13l9KX2u4KbTQU0yfYwcc8Ba)
- [t分布表](http://www.360doc.com/content/12/0307/17/7598058_192529468.shtml)

由n个独立的、服从正态分布的观测组成一个样本，样本均值为：<img src="http://www.forkosh.com/mathtex.cgi?\overline{x}">，标准差为：s. 

总体均值的置信区间为：

<img src="http://www.forkosh.com/mathtex.cgi?[\overline{x}-t^{*}\frac{s}{\sqrt{n}},\overline{x}+t^{*}\frac{s}{\sqrt{n}}], \ \ d.f.=n-1">

<img src="http://www.forkosh.com/mathtex.cgi?t^{*}">是t变量的一个值，可以从自由度n-1的t分布表中查到。只需要找到<img src="http://www.forkosh.com/mathtex.cgi?t^{*}">使得95%的t变量都落在<img src="http://www.forkosh.com/mathtex.cgi?(-t^{*},+t^{*})">之间。

### 3.两个百分比之差的置信区间

一个样本有n1个观测，另一个有n2个观测；对应的样本百分比分别为P1和P2。则两个总体百分比之差的95%置信区间是：

<img src="http://www.forkosh.com/mathtex.cgi?[(P_{1}-P_{2})-1.96\sqrt{\frac{P_{1}(100-P_{1})}{n_{1}}+\frac{P_{2}(100-P_{2})}{n_{2}}}, (P_{1}-P_{2})+1.96\sqrt{\frac{P_{1}(100-P_{1})}{n_{1}}+\frac{P_{2}(100-P_{2})}{n_{2}}}]">

#### 4.两个均值之差的置信区间

- 抽样1：n1个观测值，样本均值<img src="http://www.forkosh.com/mathtex.cgi?\overline{x_{1}}">，样本标准差<img src="http://www.forkosh.com/mathtex.cgi?s_{1}">
- 抽样2：n2个观测值，样本均值<img src="http://www.forkosh.com/mathtex.cgi?\overline{x_{2}}">，样本标准差<img src="http://www.forkosh.com/mathtex.cgi?s_{2}">

两个总体均值之差(<img src="http://www.forkosh.com/mathtex.cgi?\mu_{1}-\mu_{2}">)的置信区间为：

<img src="http://www.forkosh.com/mathtex.cgi?[(\overline{x_1}-\overline{x_2})-t^{*}s\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}},(\overline{x_1}-\overline{x_2})+t^{*}s\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}">

自由度为：n1+n2-2

查对应自由度的t分布表，查到<img src="http://www.forkosh.com/mathtex.cgi?t^{*}">的值，使t变量落入<img src="http://www.forkosh.com/mathtex.cgi?-t^{*}">到<img src="http://www.forkosh.com/mathtex.cgi?t^{*}">的概率为0.95。

# 3.假设检验

## 3.1 基本过程

估计和假设检验，都关注的是**总体参数**，两者区别：

- 估计：找参数值等于多少
- 假设检验：看参数值是否等于某个感兴趣的值

两个假设：

- 零假设（null hypothesis）：<img src="http://www.forkosh.com/mathtex.cgi?H_0">:参数=值
- 备择假设（alternative hypothesis）：<img src="http://www.forkosh.com/mathtex.cgi?H_a">:参数<img src="http://www.forkosh.com/mathtex.cgi?\neq">值

如果样本数据证明零假设不成立，则拒绝(reject)零假设，倾向于备择假设。

注意：之所以叫“零”，原因是假设的内容总是没有差异或没有改变，或变量间没有关系等。

由于样本信息量受限，有可能提供错误答案：

- 第一类错误（α错误、type I error）：零假设正确，但判断错误
- 第二类错误（β错误、type II error）：零假设错误，但判断正确

如何衡量零假设？答：p值（p-value）。

- p值定义：当零假设正确时，得到的结果数据是否等于实际观察或者比实际观察更极端的概率。
- p值越小，拒绝零假设的理由就越充分
- 大小：20分之一，即0.05. p值<=0.05都认为是小的(Ronald Fisher)
- p值的意义：在某总体的许多样本中，某一类数据出现的经常程度。

统计显著

- 如果零假设被拒绝(p值很小)，则样本结果是统计显著（Statistical significant）的。

如何计算p值？标准得分(standard score)

零假设（均值之差、或者单个均值） => t分布的t变量

(详见<统计学>第七章公式部分)

单均值检验：

- 零假设：<img src="http://www.forkosh.com/mathtex.cgi?H_0: \mu=\mu_0">
- 样本：n个观测，均值为<img src="http://www.forkosh.com/mathtex.cgi?\overline{x}">，标准差为s。
- 进行t变换，计算统计量：<img src="http://www.forkosh.com/mathtex.cgi?t=\frac{\overline{x}-\mu_0}{s/\sqrt{n}} \ d.f.=n-1">

根据对应自由度，查对应t分布表得：p值.

两均值差检验：

- 零假设：<img src="http://www.forkosh.com/mathtex.cgi?H_0: \mu_1-\mu_2=0">
- 样本数据A：有n1个观测，均值为<img src="http://www.forkosh.com/mathtex.cgi?\overline{y}_{1}">，标准差是s1；样本数据B：有n2个观测，均值为<img src="http://www.forkosh.com/mathtex.cgi?\overline{y}_{2}">，标准差是s2。
- 进行t变换，计算统计量：<img src="http://www.forkosh.com/mathtex.cgi?t=\frac{\overline{y}_{1}-\overline{y}_{2}}{s\sqrt{1/n_{1}+1/n_{2}}} / / d.f.=n_1+n_2-2">

基中s为联合方差：<img src="http://www.forkosh.com/mathtex.cgi?s^2=\frac{(n_1-1)s_{1}^2+(n_2-1)s_{2}^2}{n_1+n_2-2}">

代入，查表，得到p值，判断是否拒绝零假设。


# 3.2 总体比例

总体比例检验：

- 零假设：<img src="http://www.forkosh.com/mathtex.cgi?H_0:\pi=\pi_{0}">
- 样本比例：p，样本观测量：n
- z变换：<img src="http://www.forkosh.com/mathtex.cgi?z=\frac{p-\pi_{0}}{\sqrt{\frac{\pi_{0}(1-\pi_{0})}{n}}}">

得到z值，查找标准正态分布表，得到p值，判断零假设是否正确。

比例差的检验：

- 零假设：<img src="http://www.forkosh.com/mathtex.cgi?H_0:\pi_{1}-\pi_{2}=0">
- 两个样本比例之差：p1-p2，样本观测个数分别为：n1,n2
- z变换：<img src="http://www.forkosh.com/mathtex.cgi?z=\frac{(p_1-p_2)-(\pi_1-\pi_2)}{\sqrt{\frac{\pi_{1}(1-\pi_{1})}{n_1}}+\sqrt{\frac{\pi_{2}(1-\pi_{2})}{n_2}}}">

得到z值，查找标准正态分布表，得到p值，判断零假设是否正确。

# 3.3 其它

置信区间与假设检验：

- 置信区间：给了我们一个参数值的可能范围
- 假设检验：一个可能值



# 参考：

- 1.<统计学-基本概念和方法>
- 2.http://zy.swust.net.cn/10/2/tjxyl/kj/6-4-1.htm
















