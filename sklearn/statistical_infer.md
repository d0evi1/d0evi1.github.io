---
layout: page
title: 统计推断的一些基本概念 
tagline: 介绍
tags: [样本估计 假设检验 零假设]
---
{% include JB/setup %}

观测值=真值+非统计错误+随机性

观测值可以是：百分比、均值之差、数值。

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

| 类型   |      均值     |  标准差 | 百分比 |
|:----------|:------------|:------|:--------|
| 总体参数(未知) | <img src="http://www.forkosh.com/mathtex.cgi?\mu"> | <img src="http://www.forkosh.com/mathtex.cgi?\sigma"> |   <img src="http://www.forkosh.com/mathtex.cgi?\Pi">     |
| 样本统计量(已知) |  <img src="http://www.forkosh.com/mathtex.cgi?\overline{x}">   |  s |   P     |


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

- 样本个数为n，总体个数为N
- 标准正态分布为：<img src="http://www.forkosh.com/mathtex.cgi?Z - N(0,1)">，z值为：z=(x-μ)/σ 
- [标准正态分布表查表](http://static.oschina.net/uploads/img/201509/14164301_5fGD.png)

2.

- 样本比例的数学期望：<img src="http://www.forkosh.com/mathtex.cgi?E(p)=\pi">
- 样本比例的方差（重复抽样）： <img src="http://www.forkosh.com/mathtex.cgi?\sigma^{2}=\frac{\pi(1-\pi)}{n}">
- 样本比例的方差（不重复抽样）：<img src="http://www.forkosh.com/mathtex.cgi?\sigma^{2}=\frac{\pi(1-\pi)}{n}(\frac{N-n}{N-1})">


### 一、总体百分数的置信区间:

<img src="http://www.forkosh.com/mathtex.cgi?[P-1.96\sqrt\frac{P(100-P)}{n}, P+1.96\：\frac{P(100-P)}{n}]">

（p=0.975, z=1.96）

它的一个可用于快速计算的95%置信区间的近似计算（令P=50）：

<img src="http://www.forkosh.com/mathtex.cgi?[P-\frac{100}{\sqrt{n}}, P+\frac{100}{\sqrt{n}}]">

为什么大多数样本要求有1200个响应者的原因。（95%置信区间，100/sqrt(n)=3，n=1111）

### 二、总体均值的置信区间：

- [t分布](http://baike.baidu.com/link?url=6-_ll1N1LVImC7NPGslgac2snge1iwyLrFhzW59E-O6RHOHiLwpE4XslYRT_DL13l9KX2u4KbTQU0yfYwcc8Ba)
- [t分布表](http://www.360doc.com/content/12/0307/17/7598058_192529468.shtml)

由n个独立的、服从正态分布的观测组成一个样本，样本均值为：<img src="http://www.forkosh.com/mathtex.cgi?\overline{x}">，标准差为：s. 

总体均值的置信区间为：

<img src="http://www.forkosh.com/mathtex.cgi?[\overline{x}-t^{*}\frac{s}{\sqrt{n}},\overline{x}+t^{*}\frac{s}{\sqrt{n}}], \ \ d.f.=n-1">

<img src="http://www.forkosh.com/mathtex.cgi?t^{*}">是t变量的一个值，可以从自由度n-1的t分布表中查到。只需要找到<img src="http://www.forkosh.com/mathtex.cgi?t^{*}">使得95%的t变量都落在<img src="http://www.forkosh.com/mathtex.cgi?(-t^{*},+t^{*})">之间。

### 3.两个百分比之差的置信区间

一个样本有n1个观测，另一个有n2个观测；对应的样本百分比分别为P1和P2。则两个总体百分比之差的95%置信区间是：

<img src="http://www.forkosh.com/mathtex.cgi?[(P_{1}-P_{2})-1.96\sqrt{\frac{P_{1}(100-P_{1})}{n_{1}}+\frac{P_{2}(100-P_{2})}{n_{2}}}, (P_{1}-P_{2})+1.96\sqrt{\frac{P_{1}(100-P_{1})}{n_{1}}+\frac{P_{2}(100-P_{2})}{n_{2}}}]">

### 4.两个均值之差的置信区间

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

如何计算p值？=> 使用标准得分(standard score)


零假设（均值之差、或者单个均值） => t分布的t变量

(详见<统计学>第七章公式部分)

单均值检验：

- 零假设：<img src="http://www.forkosh.com/mathtex.cgi?H_0: \mu=\mu_0">
- 样本：n个观测，均值为<img src="http://www.forkosh.com/mathtex.cgi?\overline{x}">，标准差为s。
- 进行t变换，计算统计量：<img src="http://www.forkosh.com/mathtex.cgi?t=\frac{\overline{x}-\mu_0}{s/\sqrt{n}} \ , d.f.=n-1">

根据对应自由度，查对应t分布表得：p值.

两均值差检验：

- 零假设：<img src="http://www.forkosh.com/mathtex.cgi?H_0: \mu_1-\mu_2=0">
- 样本数据A：有n1个观测，均值为<img src="http://www.forkosh.com/mathtex.cgi?\overline{y}_{1}">，标准差是s1；样本数据B：有n2个观测，均值为<img src="http://www.forkosh.com/mathtex.cgi?\overline{y}_{2}">，标准差是s2。
- 进行t变换，计算统计量：<img src="http://www.forkosh.com/mathtex.cgi?t=\frac{\overline{y}_{1}-\overline{y}_{2}}{s\sqrt{1/n_{1}+1/n_{2}}} , d.f.=n_1+n_2-2">

基中s为联合方差：<img src="http://www.forkosh.com/mathtex.cgi?s^2=\frac{(n_1-1)s_{1}^2+(n_2-1)s_{2}^2}{n_1+n_2-2}">

代入，查表，得到p值，判断是否拒绝零假设。


## 3.2 总体比例

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

## 3.3 其它

置信区间与假设检验：

- 置信区间：给了我们一个参数值的可能范围
- 假设检验：一个可能值

# 4.卡方检验

## 4.1 行-列表资料检验

2x2列联表: 对应现实中的两变量，每个变量有两种取值。

| a   |  b  | a+b |
|-----|:---:|----:|
| c   |  d  | c+d |
| a+c | b+d |   n |

字母a,b,c,d代表每个格子中观测的个数。整个表的总和记作n。

当每个变量只有两个取值时，用<img src="http://www.forkosh.com/mathtex.cgi?\phi">表示两个分类变量的相关性，取值[0,1]。<img src="http://www.forkosh.com/mathtex.cgi?\phi">越大，关系越强。

<img src="http://www.forkosh.com/mathtex.cgi?\phi=\frac{ad-bc}{\sqrt{(a+b)(c+d)(a+c)(b+d)}}">

如果分类变量>=2个，每个分类变量取值>=2个，使用V来度量变量间的相关程度。取值为[0,1]间。

<img src="http://www.forkosh.com/mathtex.cgi?V=\sqrt{\frac{\chi^2}{n(L-1)}}">

其中: n为观测的个数，L是min(行数,列数)。（2行，4列，L=2），V是<img src="http://www.forkosh.com/mathtex.cgi?\phi">的推广。

<img src="http://www.forkosh.com/mathtex.cgi?\chi^2">变量：度量了观测到的频率与期望频率间有多大的差异。

2x2表：<img src="http://www.forkosh.com/mathtex.cgi?\chi^2=n\phi^2">

对于更大的表：<img src="http://www.forkosh.com/mathtex.cgi?\chi^2=\sum_{i=1}^n{\frac{(O_i-E_i)^2}{E_i}}">

其中期望E_i的计算为：期望频率=(对应行总和x对应列总和)/表的总和。

可以利用卡方来计算p-value，这样得到的是一个近似真实的p-value。

自由度（d.f.）用来度量表的大小。

d.f.=(行数-1)x(列数-1)

零假设：两个变量不相关.


# 5.回归分析与相关分析

- 回归分析（regression analysis）：是一个或多个自变量的变化是如何影响因变量的一种方法
- 相关分析（correlation anaysis）：是两个数值变量间关系的强弱

简单回归分析：两个变量的回归分析 => 散点图

- 相关系数r：[-1, 1]  或称为：(线性相关系数，Pearson相关系数,每次积相关系数)
- 另一个量：<img src="http://www.forkosh.com/mathtex.cgi?r^2">

用直线来度量：

- 回归直线（regression line）
- 回归方程（regression equation）：y=a+bx
- 回归系数（regression coefficient）：b

计算回归直线？

- 最小二乘法（least squares）：垂直距离和最小

用回归分析进行预测

- 预测值：把自变量的值代入回归直线的方程，就得到了因变量的预测值
- <img src="http://www.forkosh.com/mathtex.cgi?\hat{y}=36.1+15.3(4)">

残差的影响：

- 残差变量（residual variable）：除自变量外其他所有变量对因变量的影响的变量

不同的平方和：

- 总平方和（total sum of squares）：<img src="http://www.forkosh.com/mathtex.cgi?TSS=\sum{(y_i-\overline{y})^2}">。即(观测-平均)^2 的求和，度量自变量和残差变量在因变量上的总效应
- 残差平方和（RSS:residual sum of squares）：<img src="http://www.forkosh.com/mathtex.cgi?RSS=\sum{(y_i-\hat{y})^2}">。即观测点到回归直线的垂直距离的平方和 (也叫误差平方和)
- 回归平方和（regression sum of squares）：<img src="http://www.forkosh.com/mathtex.cgi?RegrSS=\sum{(\hat{y}_i-\overline{y})^2}">。

- TSS=RegrSS+RSS
- <img src="http://www.forkosh.com/mathtex.cgi?r^2=\frac{RegrSS}{TSS}">
- 残差变量占总效应的比例：<img src="http://www.forkosh.com/mathtex.cgi?1-r^2">

(注：有些地方把RSS标成ESS，把RegrSS标成RSS)

观测点离回归直线越近，RSS越小，相关系数r越大

从样本推广到总体

- 总体的回归系数<img src="http://www.forkosh.com/mathtex.cgi?\beta">
- 可不可信：置信区间 or 假设检验

<img src="http://www.forkosh.com/mathtex.cgi?\beta">的置信区间：

- [<img src="http://www.forkosh.com/mathtex.cgi?b-t^{*}s_b">, <img src="http://www.forkosh.com/mathtex.cgi?b+t^{*}s_b">]
- <img src="http://www.forkosh.com/mathtex.cgi?s_b=\sqrt{\frac{RSS/(n-2)}{\sum{(x_i-\overline{x})^2}}">

其中b是观测的回归系数，t*是，sb是b的标准误差。

均方

- 均方：对应平方和除以相应的自由度
- 残差均方（RMS）： RSS/D.F.
- F比：RMS/RegrMS

使用统计量F，求得p值来检验零假设。（零假设：变量间没有关系）

# 6.方差分析(ANOVA)

## 6.1 方差分析

方差分析（analysis of variance）：在研究分类型自变量对数量型因变量的影响时，用来对比因变量在不同组中的平均值的统计方法。

变量间的对应关系：

a.散点图，方差分析与回归分析的不同：

- 回归分析：水平轴，自变量是数量变量
- 方差分析：水平轴，自变量是分类变量

b.盒子图：

- 对比中位数

关系有多强：

正规的方差分析中，用的不是中位数，而是每一组观测的均值。方差分析更适合的名字应该是均值分析。

- 分类型自变量平方和：<img src="http://www.forkosh.com/mathtex.cgi?CSS=\sum{n_i(\overline{y}_i-\overline{y})^2}"> 。即：(组均值-总均值)^2 之和
- 残差平方和：<img src="http://www.forkosh.com/mathtex.cgi?RSS=\sum\sum{(y_{ij}-\overline{y}_i)^2}">(观测-组均值)^2 之和
- 总平方和：<img src="http://www.forkosh.com/mathtex.cgi?TSS=\sum\sum{(y_{ij}-\overline{y})^2}">。(观测-总均值)^2 之和
- R方：<img src="http://www.forkosh.com/mathtex.cgi?R^2">=分类型自变量平方和/总平方和

对应的自由度：

- 分类型自变量的自由度：k-1
- 残差的自由度：n-k
- 总自由度：n-1

均方：

- 分类型变量的均方：CMS=分类型自变量平方和/(k-1)
- 残差均方：RMS=残差平方和/(n-k)

F比：

- F=分类型变量的均方/残差均方

查找F分布表对应的p值，判断是否拒绝零假设。


## 6.2 配对分析

略。

# 参考：

- 1.<统计学-基本概念和方法>
- 2.http://zy.swust.net.cn/10/2/tjxyl/kj/6-4-1.htm













