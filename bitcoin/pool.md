---
layout: page
title: btc的挖矿
tagline: 介绍
---
{% include JB/setup %}

## 1.1 挖矿

合法区块的bitcoins数为B。

假设：在块头构建中，协议中的hash函数和伪随机数生成器的量足够，一个给定的计算是否产生一个合法区块可以认为是一个随机事件，任何所有的计算hash彼此相互独立。
该量（quantity）称为计算难度（difficulty），标为D，由btc网络进行周期性调整，它决定着发现一个合法块的难度。选择的目标值，每个计算hash都会有<img src="http://www.forkosh.com/mathtex.cgi?\frac{1}{2^{32}D}">的概率产生一个合法区块。

一个矿工(miner)的hash率（hashrate）为h，挖矿的周期为t，总共计算hash次数为ht，因此，可以平均找到<img src="http://www.forkosh.com/mathtex.cgi?\frac{ht}{2^{32}D}">个块。期望的回报为：<img src="http://www.forkosh.com/mathtex.cgi?\frac{htB}{2^{32}D}">

示例：Bob买了台专用矿机，它可以执行每秒数10亿次hash计算。h = 1 Ghash/s = 10^9 hash/s.如果一天连续24小时（86400s）挖矿，难度D=1690906，区块回报为B=50BTC，那么他将平均得到<img src="http://www.forkosh.com/mathtex.cgi?(ht)/(2^{32}D)=(10^{9}hash/s*86400s)/(2^{32}*1690906)\approx0.0119 blocks">，换算成btc为：0.0119B=0.595BTC。

## 1.2 solo挖矿模式的variance

solo模式，具有常数的hashrate为：h，这是个泊松过程（Poisson process），相应的rate为：<img src="http://www.forkosh.com/mathtex.cgi?\frac{h}{2^{32}D}">。

发现的区块链的数目服从泊松分布：

<img src="http://www.forkosh.com/mathtex.cgi?\lambda=\frac{ht}{2^{32}D}">，

该值也是发现的区块链数目的variance。

对应回报的方差（variance）为：

<img src="http://www.forkosh.com/mathtex.cgi?\lambda*B^{2}=\frac{htB^{2}}{2^{32}*D}">

相应的标准差（standard deviation）：

<img src="http://www.forkosh.com/mathtex.cgi? \frac{\sqrt{\lambda*B^{2}}}{\lambda*B}=\frac{1}{\sqrt{\lambda}}=\sqrt{\frac{2^{32}D}{ht}}">

示例：前例中，Bob的回报（payout）具有

- 方差为：<img src="http://www.forkosh.com/mathtex.cgi?0.0119B^{2} = 29.75BTC^{2}">, 
- 标准差为：<img src="http://www.forkosh.com/mathtex.cgi?\sqrt{29.75BTC^{2}}\approx5.454BTC">，它为期望的917%倍。

事实上，Bob这一整天下来接收任何回报的概率为：

<img src="http://www.forkosh.com/mathtex.cgi?1-exp(-\lambda)\approx1.18%">

solo模式的variance是较大的。对于一个独立硬件的参与者来说，平均等3个月才能接收到一些回报。需要注意的是，该过程是完全随机且无记忆的--如果用户3个月过去后还没有发现一个块，他可能还需平均再等3个月。另外，随机BTC收益获取的难度变得越来越难，相应的variance也会增加。

这种情况是有问题的，有如下原因：

- 由于货币供应是concave的，货币供应的variance会直接减少个人收入。很难开展财务计划。
- 由于缺少常规的回报，使得技术上更难验证btc系统是否正确工作
- 由于收入源具有这么高的方差，感觉被征了许多情感税

## 1.3 pool模式

pool模式允许矿工进行合作来发现区块，根据他们的贡献进行分配回报。如果所有矿工的hashrate为H，那么矿池将在周期t内平均获得：<img src="http://www.forkosh.com/mathtex.cgi?\frac{Ht}{2^{32}D}">个区块，平均回报为：<img src="http://www.forkosh.com/mathtex.cgi?\frac{HtB}{2^{32}D}">。单个矿工的hashrate为：h=qH（其中：q是该矿工对该矿池中总算力的贡献），它将获得总收益的q倍，<img src="http://www.forkosh.com/mathtex.cgi?q\frac{HtB}{2^{32}D}=\frac{htB}{2^{32}D}">，该值与solo模式的期望回报一致。但是，它的variance将变得更低--

- 总回报的variance为：<img src="http://www.forkosh.com/mathtex.cgi?q\frac{HtB}{2^{32}D}=\frac{HtB^{2}}{2^{32}D}">，
- 单人回报的variance为：<img src="http://www.forkosh.com/mathtex.cgi?q^{2}\frac{HtB^{2}}{2^{32}D}=q\frac{htB^{2}}{2^{32}D}">，该值为solo模式下variance的q倍. (0<q<<1)

因而，矿池越大，矿工越少，潜在收益更大。

矿池通常由一个**矿池操作人（pool operator）**来维护，它会在相应的服务上花费一定的费用。这通常是区块回报的一个固定百分比：f。因此，对于每个发现的区块，operator都将收到一笔fB的费用，余下的(1-f)B将分配给矿工。因此，单个矿工的实际期望回季收入为：

<img src="http://www.forkosh.com/mathtex.cgi?\frac{(1-f)htB}{2^{32}D}">

为了确定矿池是否工作良好，矿工们发现并提交块头的shares，hashes（如果D=1，）。每个hash具有一个概率<img src="http://www.forkosh.com/mathtex.cgi?\frac{1}{2^{32}}">成为一个share。假如使用的hash函数正确，如果没有做发现区块必需的工作，或者搜索区块时没有采用这种方法来发现shares，那也不可能发现shares。一个矿工发现的shares的数目，与hashes计算次数成比例。

每个share都有一个概率<img src="http://www.forkosh.com/mathtex.cgi?p=\frac{1}{D}">成为合法区块。因而，如果一个矿工花同样在solo模式下的挖矿算力，他的期望贡献是<img src="http://www.forkosh.com/mathtex.cgi?pB">。因而，在一个公平的矿池中，矿工在每次提交一个share时将平均接收到pB的回报，对于operator则收到<img src="http://www.forkosh.com/mathtex.cgi?(1-f)pB">的费用。

这里的主要问题就转变成了：如何来划分回报，以使每个矿工可以根据它的贡献来得到公平的share，这是个很重要的问题。业界提出了许多方法，有好有坏。下面章节，将详细讨论每种方法。

需要注意一下，矿池潜在会提升一个矿工的variance（通过因子q，矿工对矿池的hashrate贡献度）。然而，如果一个矿工的hashrate过小，根据他发现shares的数目的方差，将不会出现这种可能性。这里存在一个最大可能改进因子，它依赖着回报系统的实现。挖矿中间可能会增加variance，它也依赖于回报系统。然而，这些情况都不会影响平均回报的量，在一个公平的矿池中，它对于每个提供的share都有<img src="http://www.forkosh.com/mathtex.cgi?(1-f)pB">。


# 2.简单回报系统

## 2.2 PPS(Pay-per-share)

在PPS系统中，operator不是一个被动的中间人。它实际会吸收所有单个矿工都会遇到的variance。当一个参与者（participant）提交一个share时，它立即获得回报<img src="http://www.forkosh.com/mathtex.cgi?(1-f)pB">，对应于它的贡献，减去oprator的费用（fee）--不管事实上发现了多少区块。operator会保留着所有发现区块的回报。

每份share的支付都有一个决策值。这种方式对于矿工来说具有以下的优点：

- 每份share在回报时都具有0方差。
- 无需等待时间（发现区块）即可获得收益
- 很容易描述精确的收益
- 很容易验证回报，没有损失，无需担心部分operator或者部分组织的不诚信。
- 不会因为矿池跳槽（pool-hopping）而受损失

然而，这对于矿池operator来说是风险最高的回报系统--为了提供variance=0支付给参与者，它必须承担所有的variance。他可以在短期获得很好的回报--他获取整个区块奖励，支出比平均shares数还要少--但是在长期来看会大幅损失。它的variance与solo的variance相同。为了补偿它的风险，operator将比其它方法支付一个更大的费用，这就是PPS方法的缺点。

如果operator不正确平衡矿池的费用以及他的财产准备金，矿池有很大可能会破产。在附录C中，为了让破产概率在<img src="http://www.forkosh.com/mathtex.cgi?\delta">以下，operator应该至少有如下的准备金：

<img src="http://www.forkosh.com/mathtex.cgi?R=\frac{Bln\frac{1}{\delta}}{2f}">

需要的准备金比大多数人预期的还要高。管理这样的矿池最适合那些知道如何能负现管理自己风险的人，期望稳定的矿工应回避那些没有适当经营的PPS矿池，它们随时会关停。



# Appenddix C：

## PPS矿池的安全网

在附录中，我们会讨论一个PPS矿池的operator要准备多少准备金，才能让矿池合理运转。准备金越大，可以确保矿池破产的概率越低。

假设没有受到黑客攻击，非法区块链等，operator的财务平衡可以通过Markov链来建模：

<img src="http://www.forkosh.com/mathtex.cgi? X_{t+1}-X_{t}=\{ \begin{aligned} &-(1-f)pB+B & w.p. & & p \\ &-(1-f)pB & w.p. & & 1-p \end{aligned}">

每次提交share可以当成一个step。每次都具有期望<img src="http://www.forkosh.com/mathtex.cgi?fpB">，方差近似为：<img src="http://www.forkosh.com/mathtex.cgi?pB^{2}">，根据中心极限定理，该随机过程的长期行为等价于：

<img src="http://www.forkosh.com/mathtex.cgi? X_{t+1}-X_{t}=\{ \begin{aligned} &+\sqrt{p}B & w.p. & & (1+f\sqrt{p})/2 \\ &-\sqrt{p}B & w.p. & & (1-f\sqrt{p})/2 \end{aligned}">


它们具有相同的期望和方差。这等价于：

<img src="http://www.forkosh.com/mathtex.cgi? X_{t+1}-X_{t}=\{ \begin{aligned} &+1 & w.p. & & (1+f\sqrt{p})/2 \\ &-1 & w.p. & & (1-f\sqrt{p})/2\end{aligned}">

初始条件通过因子进行<img src="http://www.forkosh.com/mathtex.cgi?\sqrt{p}/2">缩放。

<img src="http://www.forkosh.com/mathtex.cgi?a_n">表示，从状态n开始要达到0的概率（表示矿池破产）。我们在第一步得到的条件，表示：<img src="http://www.forkosh.com/mathtex.cgi?q=(1+f\sqrt{p})/2">,

<img src="http://www.forkosh.com/mathtex.cgi?a_n=qa_{n+1}+(1-q)a_{n-1}">

这个递归方程的多项式特征方程是：<img src="http://www.forkosh.com/mathtex.cgi?q\lambda^{2}-\lambda+(1-q)">，

因而，它具有通解：<img src="http://www.forkosh.com/mathtex.cgi?a_n=A+B((1-q)/q)^{n}">。

代入初始值（边界条件）：<img src="http://www.forkosh.com/mathtex.cgi?a_0=1,a_{\infty}=0">，

我们具有A=0, B=1，因而：

<img src="http://www.forkosh.com/mathtex.cgi?a_n=(\frac{1-q}{q})^{n}=(\frac{1-f\sqrt{p}}{1+f\sqrt{p}})^{n} \approx exp(-2fn\sqrt{p})">

如果operator以一个R的准备金启动，矿池的破产概率为：

<img src="http://www.forkosh.com/mathtex.cgi?\delta=a_{R/(\sqrt{p}B)} \approx exp(\frac{-2fR\sqrt{p}}{\sqrt{p}B}) = exp(\frac{-2fR}{B})">

相反地，为了维持一个破产概率最大为<img src="http://www.forkosh.com/mathtex.cgi?\delta">，矿池应至少保有准备金：

<img src="http://www.forkosh.com/mathtex.cgi?R=\frac{Bln(\frac{1}{\delta})}{2f}">

例如，如果B=50BTC, <img src="http://www.forkosh.com/mathtex.cgi?\delta=1/1000">和f=0.05(矿池费用为5%)，准备金应该为：

<img src="http://www.forkosh.com/mathtex.cgi?R=\frac{50BTC*ln1000}{2*0.05} \approx 3454BTC">

如果oprator尝试以f=0.01，准备金为500BTC启动，那么：

<img src="http://www.forkosh.com/mathtex.cgi?\delta=exp(\frac{-2*0.01*500BTC}{50 BTC}) \approx 0.819">

因而，它实际上具有81.9%的机率破产。














