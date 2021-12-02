---
layout: post
title: PID广告控制介绍
description: 
modified: 2020-09-10
tags: 
---

《Feedback Control of Real-Time Display Advertising》在PID中引入了feedback control。

# 3.RTB feedback control系统



图2展示了RTB feedback control系统的图示。传统的bidding strategy可以表示为在DSP bidding agent中的bid calculator module。controller会扮演着根据bid calculator调整bid价格的角色。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/ecfce45fcc3f2e22ec815c4fe656c7e6d087370ea7874a0b6f46c724bb19e2b4a4459d50a2f5a5bf2da85b98fc6fafca?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=2.jpg&amp;size=750">

图2 集成在RTB系统中的feedback controller

特别的，monitor会接收到来自ad exchange的auction win通知和来自ad tracking系统的用户点击反馈，它整体上会看成是dynamic system。接着，当前的KPI值（比如：AWR和eCPC）会被计算。如果该任务会使用reference value来控制eCPC，在reference eCPC和measured eCPC间的error因子会被计算，并接着会发送到control function中。输出控制信号会被发送到actuator中，它会使用control signal来调整来自bid calculator的原始bid price。调整后的bid price会将合理的ad（qualified ad）打包成bid response，并发送回ad exchange进行auction。

# 3.1 Actuator

对于在t时刻的bid request，auctuator会考虑当前控制信息$$\phi(t)$$来将bid价格从$$b(t)$$调整到一个新的值$$b_a(t)$$。在我们的模型中，控制信号，它会在下一节中以数学形式定义，这在bid price上的一个增益。总之，当控制信号$$\phi(t)$$为0时，不会进行bid调整。这是不同的actuator模型，在我们的工作中，我会选择使用：

$$
b_a(t) = b(t) exp(\lbrace \phi(t) \rbrace)
$$ 

...(2)

其中，当$$\phi(t) = 0$$时，该模型会满足$$b_a(t)$$。其它比如线性模型$$b_a(t) = b(t) (1+\phi(t))$$的模型也会在我们的研究中，但当一个大的负向控制信号被发到actuator时执行效果很差，其中linear actuator通常会响应一个负或零的bid，这在我们场景中是无意义的。相反的，指数模型是合适的，因为它天然会避免一个负的竞价，从而解决上述缺点。在之后的研究中，我们会基于指数形式的actuator model上报分析。

## 3.2 PID controller 

我们考虑的第一个controller是经典的PID controller。如名字所示，一个PID controller会从基于error因子的比例因子、积分因子和微分因子的一个线性组合上生成控制信号：

$$
e(t_k) = x_r - x(t_k), \\
\phi(t_{k+1}) \rightarrow \lambda_P e(t_k) + \lambda_I \sum\limits_{j=1}^k e(t_j) \Delta t_j + \lambda_D \frac{\Delta e(t_k)}{\Delta t_k}
$$

...(3)(4)

其中：

- error factor $$e(t_k)$$是$$x_r$$减去当前控制变量值$$x(t_k)$$的reference value
- 更新时间间隔给定如下$$\Delta t_j = t_j - t_{j-1} $$,error factors的变化是$$\Delta e(t_k) = e(t_k) - e(t_{k-1})$$，其中: $$\lambda_P, \lambda_I, \lambda_D$$是每个control factor的weight参数。

注意，这里的control factors都是在离散时间$$(t_1, t_2, \cdots) $$上的，因为bidding事件是离散的，它实际上会周期性更新control factors。**所有的control factors $$(\phi(t), e(t_k), \lambda_P, \lambda_I, \lambda_D)$$仍会在两个updates间保持相同**，在等式(2)中的控制信号$$\phi(t)$$等于$$\phi(t_k)$$。我们看到P factor会趋向于将当前变量值push到reference value；I factor会减小从当前时间开始的累计error；D factor会控制该变量的波动。

## 3.3 Waterlevel-based Controller

Waterlevel-based（WL） Controller是另一种feedback model，它会通过water level来切换设备控制：

$$
\phi(t_{k+1}) \rightarrow \phi(t_k) + \gamma(x_r - x(t_k))
$$

...(5)

其中，$$\gamma$$是对于在指数scale下的$$\phi(t_k)$$次更新的step size参数。

对比起PID， WL controller只会使用变量与reference value间的差异。另外，它会提供一个顺序控制信号。也就是说，下个control信号会基于前者进行调整。


## 3.4 为点击最大化设置References

假设：feedback controller是用来分发广告主的KPI目标的一个有效工具。在该节中，我们会演示feedback control机制可以被当成一个**模型无关（model-free）的点击最大化框架（click maximisation framework）**，它可以嵌入到任意bidding策略，并执行：在不同channels上，通过设置smart reference values来进行竞价预算分配。

当一个广告主在指定目标受众时（通常会组合上广告曝光上下文分类）来进行它指定的campaign，来自独立channels（比如：不同的广告交易平台(ad exchanges)、不同的用户regions、不同的用户PC/model设备等）的满足目标规则（target rules）的曝光（impressions）。通常，DSP会集合许多ad exchanges，并分发来自所有这些广告交易平台(ad exchanges)的所需ad曝光（只要曝光能满足target rule），尽管市场价格会大有不同。图3展示了：**对于相同的campaign，不同的广告交易平台(ad exchanges)会有不同的eCPC**。如【34】中所说，在其它channels上（比如：user regions和devices上）也会有所不同。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/20fe6d05772ce1e0acc3bfc82846acde16b666b530416a4ad4157c4c4f0e95b2b4348dbe1451e050d611220b0cea0057?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=3.jpg&amp;size=750">

图3 跨不同ad exchanges的不同eCPCs。Dataset: iPinYou

**开销差异提供给广告主一个机会，可以基于eCPC来最优化它的campaign效果**。为了证实这点，假设一个DSP被集成到两个广告交易平台A和B。对于在该DSP中的一个campaign，**如果来自A的eCPC要比B的高，这意味着来自平台B的库存要比平台A的费用更高效**，接着会重新分配一些预算给A和B，这将潜在减小该campaign的整体eCPC。实际上，**预算重分配（budget reallocation）可以通过对平台A减小竞价、并增加平台B的竞价来完成**。这里，我们正式提出一个用于计算每个交易平台的**均衡eCPC模型：它可以被用作最优化reference eCPC来进行feedback control，并在给定预算约束下生成一个最大数目的点击**。

数学上，假设对于一个给定的ad campaign，存在n个交易平台（可以是其它channels），比如：1, 2, ..., n，它们对于一个target rule具有ad volume。在我们的公式里，我们关注最优化点击，而转化率公式可以相似被获取到。假设：

- $$\epsilon_i$$：是在交易平台i上的eCPC
- $$c_i(\epsilon_i)$$：是campaign在交易平台i上调整竞价后使得eCPC为$$\epsilon_i$$，所对应的**在该campaign的lifetime中获得的点击数**

对于广告主，他们希望在给定campaign预算B下，最大化campaign-level的点击数（会使得成本$$\epsilon_i$$越小）：

$$
max_{\epsilon_1,\epsilon_2,\cdots,\epsilon_n} \sum_i c_i(\epsilon_i) \\
s.t. \sum_i c_i(\epsilon_i) \epsilon_i = B
$$

...(6)(7)

它的拉格朗日项为：

$$
L(\epsilon_1,\epsilon_2,\cdots,\epsilon_n, \alpha) = \sum_i  c_i(\epsilon_i)  - \alpha ( c_i(\epsilon_i) \epsilon_i - B)
$$

...(8)

其中，$$\alpha$$是Lagrangian乘子。接着我们采用它在$$\epsilon_i$$上的梯度，并假设它为0

$$
\frac{\partial L(\epsilon_1,\epsilon_2,\cdots,\epsilon_n, \alpha)}{\partial \epsilon_i} = c_i'(\epsilon_i) - \alpha(c_i' (\epsilon_i) \epsilon_i + c_i(\epsilon_i)) = 0 \\
\frac{1}{\alpha} = \frac{c_i' (\epsilon_i) \epsilon_i + c_i(\epsilon_i)}{c_i'(\epsilon_i)} = \epsilon_i + \frac{c_i(\epsilon_i)}{c_i'(\epsilon_i)}
$$

...(9) (10)

其中，对于每个交易平台i都适用于该等式。这样，我们可以使用$$\alpha$$来桥接任意两个平台i和j的等式：

$$
\frac{1}{\alpha} = \epsilon_i + \frac{c_i(\epsilon_i)}{c_i'(\epsilon_i)} = \epsilon_j + \frac{c_j(\epsilon_j)}{c_j'(\epsilon_j)}
$$

...(11)

因此，最优解条件给定如下：

$$
\frac{1}{\alpha} = \epsilon_1 + \frac{c_i(\epsilon_1)}{c_1'(\epsilon_1)} = \epsilon_2 + \frac{c_2(\epsilon_2)}{c_2'(\epsilon_2)} = ... = \epsilon_n + \frac{c_n(\epsilon_n)}{c_n'(\epsilon_n)} \\
\sum_i c_i(\epsilon_i) \epsilon_i = B
$$

...(12)(13)

有了足够的数据样本，我们可以发现，$$c_i(\epsilon_i)$$通常是一个concave和smooth函数。一些示例如图4所示。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/730cef0bd8b1fd91dd48f9c9a7860097afa302603bfbc8bb946de038ed5afa11a8b6709f723128f26855bb37715d57fd?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=4.jpg&amp;size=750">

图4 与eCPC相反的clicks数目。clicks和eCPC会跨整个iPinYou 每campaign的training dataset计算，通过对等式(1)中的$$b_0$$进行调参进行计算

基于该观察，可以将$$c_i(\epsilon_i)$$定义成一个通用多项式：

$$
c_i(\epsilon_i) = c_i^* a_i  (\frac{\epsilon_i}{\epsilon_i^*})^{b_i}
$$

...(14)

其中，

- $$\epsilon_i^*$$是在交易平台i在训练数据周期期间，该campaignad库存的历史平均eCPC
- $$c_i^*$$：相应的点击数（click number）

这两个因子可以直接从训练数据中获得。参数$$a_i$$和$$b_i$$可以进行调参来拟合训练数据。

等式(14)转成(12)：

$$
\frac{1}{\alpha} = \epsilon_i + \frac{c_i(\epsilon_i)}{c_i^'(\epsilon_i)} = \epsilon_i +  ... = (1+\frac{1}{b_i} \epsilon_i
$$

...(15)

我们将等式(12)重写：

$$
\frac{1}{\alpha} = (1+\frac{1}{b_1}) \epsilon_1 =  (1+\frac{1}{b_2}) \epsilon_2 = ... =  (1+\frac{1}{b_n}) \epsilon_n \\
\epsilon_i = \frac{b_i}{\alpha (b_i+1)}
$$

...(16) (17)

有意思的是，等式(17)中的equilibrium不在相同交易平台的eCPCs的state中。作为替代，当在平台间重新分配任意预算量时，不会做出更多的总点击；例如，在一个双平台的情况下，当来自一个平台的点击增加等于另一个的下降时，会达到平衡。更特别的，对于广告交易平台i，我们从等式(17)观察到，如果它的点击函数$$c_i(\epsilon_i)$$相当平，例如：在特定区域，点击数会随着eCPC的增加而缓慢增加，接着学到的$$b_i$$会很小。这意味着因子$$\frac{b_i}{b_i + 1}$$也会很小；接着等式(17)中，我们可以看到在广告交易平台i中最优的eCPC应相当小。

将等式(14)和等式(17)代入等式(7)中：

$$
\sum_i \frac{c_i^* a_i}{\epsilon_i^{* b_i}} (\frac{b_i}{b_i + 1})^{b_i + 1} (\frac{1}{\alpha})^{b_i + 1} = B
$$

...(18)

出于简洁，我们将每个ad交易平台i的参数 $$\frac{c_i^* a_i}{\epsilon_i^{* b_i}} (\frac{b_i}{b_i + 1})^{b_i + 1} $$ 表示为$$\delta_i$$。这给出了一个更简洁的形式：

$$
\sum_i \delta_i (\frac{1}{\alpha})^{b_i + 1}  = B
$$

...(19)

等式(19)对于$$\alpha$$没有封闭解（closed form slove）。然而，由于$$b_i$$非负，$$\sum_i \delta_i (\frac{1}{\alpha})^{b_i + 1} $$随着$$\frac{1}{\alpha}$$单调增，你可以轻易获得$$\alpha$$的解，通过使用一个数值求解：比如：SGD或Newton法。最终，基于求得的$$\alpha$$，我们可以发现对于每个ad交易平台i的最优的eCPC $$\epsilon_i$$。实际上，这些eCPCs是我们希望campaign对于相应的交易平台达到reference value。。。

作为特例，如果我们将campaign的整体容量看成一个channel，该方法可以被直接用于一个通用的bid optimisation tool。它会使用campaign的历史数据来决定最优化的eCPC，接着通过控制eCPC来执行click optimisation来将最优的eCPC设置为reference。注意，该multi-channel click最大化框架可以灵活地合并到任意竞价策略中。

# 4.实证研究

## 4.1 评估setup

**Dataset** 我们在iPinYou DSP上收集的公开数据集上测试了我们的系统。它包含了在2013年的10天内来自9个campains的日志数据，包含了64.75M的bid记录，以及19.5M的曝光数据，14.79K的点击以及16K条人民币的消费数据。根据数据发布者，每个campaign最近3天的数据会被split成test data，剩余作为training data。dataset的磁盘大小为35GB。关于dataset的更多统计和分析提供在[34]。该dataset是以record-per-row的格式存在，其中，每行包含了三个部分：

- i) 该次拍卖（auction）的features，比如：time、location、IP地址、URL/发布者的domain、ad slot size、用户兴趣分段等。每条记录的features会被indexed成0.7M维度的稀疏二元vector，它会feed到一个关于竞价策略的logistic regression CTR estimator中
- ii) 拍卖获胜价格，它是该bid赢得该次auction的和hreshold
- iii) 用户在ad impression上的feedback，比如：点击 或 未点击

**评估协议（evaluation protocol）**

我们遵循之有在bid optimisation上的研究的evaluation protocol。特别的，对于每条数据记录，我们会将feature信息传递到我们的bidding agent中。我们的bidding agent会基于CTR预估以及等式（1）中的参数生成一个新的bid。我们接着会对生成的bid 与 记录的实际auction winning price进行对比。如果该bid比auction winning price要更高，我们可以知道bidding agent会在该次auction上胜出，并且获得该次ad impression。如果该次ad impression record会带来一次click，那么该placement会生成一个正向结果（一次click），开销等于winning price。如果没有发生click行为，该placement会产生一次负向结果，并浪费钱。control参数会每2小时更新（作为一轮）。

值得一提的是，历史的用户反馈会被广泛用来评估信息检索系统和推荐系统。他们所有都会使用历史点击作为一个proxy来关联训练prediction model，也会形成ground truth。相似的，我们的evaluation protocol会保留user contexts、displayed ads（creatives 等）、bid requests
、以及auction environment不变。我们希望回答：在相同的context下，如果广告主给出一个不同或更好的竞价策略或采用一个feedback loop，他们是否能获得在有限预算下的更好点击。对于用户来说只要没有任何变化，点击仍会相同。该方法对于评估bid optimisation来说很好，并且在展示广告工业界被广泛采用。

**Evaluation Measures**

我们在feedback control系统中采用一些常用的measures。我们将errorband定义为在reference value上$$\pm 10%$$的区间。如果在该区域内控制变量settles，我们认为该变量被成功控制。convergence的speed（）也同样重要。特别的，我们会评估rise time来确认该控制变量进入到error band中有多快。我们也会使用settling time来评估受控变量成功限制在error band内有多快。然而，快收敛（fast convergence）会带来控制不准（inaccurate control）问题。在settling（称为稳态：）之后，我们使用RMSE-SS来评估在controlled variable value与reference value间的RMSE。最后，我们会通过计算在该settling后该变量值的标准差，来measure该control stability，称为“SD-SS”。

对于bid optimisation的效果，我们会使用campaign的总达成点击数（total achieved click number）和eCPC来作为主要评估measures。我们也会监控与效果相关的曝光（比如：曝光数、AWR和CPM）。

**实证研究的组织**

包含了在控制两个KIPs（eCPC和AWR）的以下5个部分。

- i) 4.2节，我们会回答提出的feedback control系统是否实际上能控制这些KPIs
- iii) 4.3节，会集中在PID controller，并研究它在设置target variable上的特性
- iii) 4.4节， 我们关注PID controller并研究它在设置target变量上的属性
- iv) 4.5节，我们利用PID controllers作为一个bid optimization tool，并研究了它在跨多个广告交易平台间优化campaign点击和eCPC上的效果。
- v) 最后，讨论PID controller调参

## 4.2 control容量

对于每个campaign，我们会检查在两个KPIs上的两个controllers。我们首先调整在训练数据上的控制参数来最小化settling time。接着我们采用在test data上的controllers并观察效果。在每个campaign上的详细控制效果如表1所示。图5展示了controlled KPI vs. timesteps（例如：轮）曲线。曲水平线意味着reference。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/4ea198e84181652260c3274c5616844eeca81c48879f02336ef82afbb028381316b48240f9bdc464f542c7b13de50bb1?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=5.jpg&amp;size=750">

图5 eCPC和AWR上的控制效果

我们从结果看到：

- (i) 所有PID controllers可以设置在error band内的KIPs（小于40轮的settling time），它意味着PID control可以在给定reference value上设置两个KPIs。
- (ii) 在eCPC上的WL controller，在test data上不会work，即使我们可以找出在训练数据上的好参数。这是因为当面对RTB的巨大动态性时，WL controller会尝试通过短时效果反馈（transient performance feedbacks）影响平均系统行为。
-  (iii)对于在AWR上的WL，大多数campaigns是可控的，但仍有两个campaigns会在设置 reference value时失败。
-  (iv) 对比在AWR上的PID，WL总是产生更高的RMSE-SS和SD-SS，但比overshot百分比要更低。这种control settings会具有一个相当短的rise time，通常会面对一个更高的overshot。
-  (v) 另外，我们观察到，具有更高CTR estimator AUC效果的campaigns，通常会获得更短的settling time。

根据上述结果，PID controller在test RTB cases上完胜WL controller。我们相信，这是因为在PID controller中的integral factor可以帮助减小累积误差（例如：RMSE-SS），derivative factor可以帮助减小变量波动（variable fluctuation，例如：SD-SS）。设置AWR比eCPC要更容易。这主要是因为AWR只依赖于市场价格分布，而eCPC会额外涉及user feedback，例如：CTR，而prediction会具有很大不确定性。

## 4.3 控制难度（control difficulty）

在本节中，我们会通过添加更高或更低的reference values来进一步扩展control capability实验进行对比。我们的目标是，研究不同levels的reference values在control difficulty上的影响。我们遵循相同的模式来训练和测试controllers。然而，替代展示准确的performance value效果，我们这里只关注不同refrence settings上的效果对比。

在达成settling time、RMSE-SS和SD-SS的分布，以及三个refrence levels的setting，如图6(a)(b)所示，使用PID来控制eCPC和AWR。我们观察到，平均setting time、RMSE-SS、SD-SS，会随着refrence values变高而减小。这表明：具有更高reference的eCPC和AWR的控制任务，会更容易达成，因为可以竞价更高来获胜更多、并且花费更多。随着reference越高，越接近初始performance value，控制信号不会带来更严重的bias或易变性（volatility），这会导致更低的RMSE-SS和SD-SS。对于page limit，使用WL的control效果不会在这里呈述。结果与PID相近。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/0f5d7ec0ea575c1ff3d003cabe8c1ed01521a044bf91c9c7358565ec1dcee6f5f998153cd493197915bf4703679a4f59?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=6.jpg&amp;size=750">

图6 使用PID的控制难度对比

图7给出了具有三个reference levels两个controllers的特定控制曲线，它在一个样本campaign 3386。我们发现reference value会远离控制变量的初始值，会在eCPC和AWR上为settling带来更大的难度。这建议广告主在设置一个模糊控制目标会引入unsettling或更大易变性的风险。广告主应尝试找出在target value和practical control performance间的一个最好trade-off。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/7e9508b02ebd33c62760d652742000a692b8edbb720c28f27ffb74407025f8669015fe618a07828c271807a26cadd970?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=7.jpg&amp;size=750">

图7 campaign 3386在eCPC和AWR上使用不同reference values的控制效果

## 4.4 PID setting：静态 vs. 动态references

比例因子、微分因子、积分因子的组合，可以使PID feedback自动高效调整在control lifetime期间的settling过程。可选的，你可以经验性地调整reference value，以便达到期望的reference value。对于eCPC control的示例，如果campaign的达成eCPC（achived eCPC）要比intial reference value要更高，刚好在耗尽首个一半预算，广告主会希望降低reference以便加快下调，最终在耗尽完预算之前达到它初始的eCPC目标。PID feedback controller会通过它的积分项来隐式地处理这样的问题。在本节中，我们研究了RTB feedback control机制，对于广告主是否必要仍必要，来根据campaign的实时效果来有目的地调整reference value。

**Dynamic Freference Adjustment Model**

为了模拟广告主的策略，在预算限制下来动态变更eCPC和AWR的reference value，我们提出一种dynamic reference adjustment model来计算在$$t_k$$后的新reference $$x_r(t_{k+1})$$：

$$
x_r(t_{k+1}) = \frac{(B - s(t_k)) x_r x(t_k)}{B x(t_k) - s(t_k) x_r }
$$

...(20)

其中：

- $$x_r$$是初始feference value
- $$x(t_k)$$是在timestep $$t_k$$时的达成KPI（eCPC或AWR）
- B是campaign budget
- $$s(t_k)$$是开销

我们可以看到，等式(20)中：

- 当$$x_r(t_k) = x_r$$, $$x_r(t_{k+1})$$会设置成与$$x_r$$相同；
- 当$$x_r(t_k) > x_r$$时，$$x_r(t_{k+1})$$会设置得比$$x_r$$更低，反之

出于可读性，我们会在附录详细介绍微分项。使用等式(20)我们可以计算新的reference eCPC/AWR $$x_r(t_{k+1})$$，并使用它来替换等式(3)中的$$x_r$$来计算error factor，以便做出动态reference control。

**结果与讨论**

图8展示了具有基于等式(20)计算的动态reference PID control的效果。campaign效果会在当预算耗尽时停止。从该图看到，对于eCPC和AWR control，动态reference会采用一个激进模式，将eCPC或AWR沿着原始的reference value（虚线）进行推进。这实际上会模拟一些广告主的策略：当performance低于reference时，更高的dynamic reference会将总效果更快地推向intial reference。另外，对于AWR control，我们可以看到，当预算即将耗尽时，dynamic reference会起伏不定。这是因为当所剩预算不够时，reference value会通过等式(20)设置得过高或过低，以便将效果push到初始目标。很显然这是一个低效的解决方案。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/44da0a625cd1446841e5705272e99b6993df7b0b0dc0195cd6238db7f8382a8be90d796418cfa4a3c8db4fbe8d398b06?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=8.jpg&amp;size=750">

图8 使用PID的动态reference control

另外，我们直接对比了PID在dynamic-reference controllers（dyn）和标准静态reference（st）上的数量控制效果。除了settling time外，我们也对比了settling cost，它表示在settling前的花费预算。在所有campaigns上整体效果，eCPC control如图9(a)所示；AWR control如图9(b)所示。结果表明：

- (i) 对于eCPC control，dynamic reference controllers不会比static-reference controller效果更好 
- (ii) 对于AWR control，dynamic-reference controllers可以减小settling time和cost，但accuracy（RMSE-SS）和stability（SD-SS）会比static-reference controllers更糟。这是因为dynamic reference本身会带来易变性（如图8）。这些结果表明，PID controller提供了一个够好的方式来朝着预先指定的reference来设置变量，无需动态调整reference来加速使用我们的方法。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/251d37508fa3e292405759ed38d3ba719e2754a4a84606731e64fdaada73d93153a1b434032d4cc032e1bd0c806d8323?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=9.jpg&amp;size=750">

图9 使用PID的Dynamic vs. static reference 

## 4.5 click maximisation的reference setting

我们已经研究了feedback control如何用于点击最大化的目标。如第3.4节所示，bid requests通常来自不同的广告交易平台，其中，市场支配力和CPM价格是不同的。给定一个预算限制，控制eCPC在每个交易平台上的点击数最大化，可以通过各自在每个平台上设置最优的eCPC reference来达到。

在该实验中，我们为每个广告平台构建了一个PID feedback controller，其中，reference eCPCs通过等式(17)和等式(19)进行计算。我们会基于每个campaign的训练数据来训练PID参数，接着在test data上测试bidding的效果。如表3所示，在所有广告交易平台上的eCPC，对于所有测试campaigns都会设置在reference value上（settling time少于40）。我们将多个平台eCPC feedback control方法称为multiple。除了multiple外，我们也测试了一个baseline方法，它会在所有ad交易平台上分配一个单一最优均匀eCPC reference，表示为“uniform”。我们也会使用没有feedback control的线性bidding策略作为baseline，表示为“none”。

在多个evaluation measures上的对比如图10所示。我们观察到：

- (i) 在achieved clicks数目和eCPC上，feedback-control-enabled bidding策略（uniform和multiple）要极大胜过non-controlled bidding策略（none）。这表明，合理的controlling eCPCs会导致在最大化clicks上的一个最优解 
- (ii) 通过重分配预算，在不同广告交易平台上设置不同reference eCPCs，multiple会进一步胜过uniform
- (iii) 在impression相关的指标上，feedback-control-enabled bidding策略会比non-controlled bidding stategy获得更多曝光，通过减少它们的bids（CPM）以及AWR，但达成更多的bid volumes。这建议我们：通过分配更多预算到具有更低值的impressions上，可以潜在生成更多点击。作为一个副产品，它会证实【33】的理论。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/570b8e6e48ff1cfb8b099f0e0384aabfab50b958b161d5bfe55637fa245dc772d6392d09e7b9ace6886a298c3f3442a0?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=10.jpg&amp;size=750">

图10 Bid最优化效果

作为示例，图11会绘制在campaign 1458上的三个方法的settling效果。三条曲线是在三个交易平台上的reference eCPCs。我们可以看到：在三个广告交易平台上的eCPCs。我们可以看到，在三个交易平台上的eCPs成功设置在reference eCPCs。同时，campaign-level eCPC (multiple)会比uniform和none设置在一个更低值。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/14836f47b8553605dce2bbdf9694a11ae633ac26632e5fb6111be31712780e74e758fe99245df32b0b6c5f2529c8b0db?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=11.jpg&amp;size=750">

图11 多交易平台的feedback control的settlement

## 4.6 PID参数调整

在该节中，我们共享了一些关于PID controller调参以及在线更新的经验。

**参数搜索**

经验上，$$\lambda_D$$不会极大地改变control效果。$$\lambda_D$$只需要一个很小值，比如：$$1 \times 10^{-5}$$，会减小overshoot并能轻微缩短settling time。这样，参数搜索的焦点是$$\lambda_P$$和$$\lambda_I$$。我们不会使用开销很大的grid search，我们会执行一个adaptive coordinate search。对于每个update，我们会fix住一个参数，更改另一个来寻找能产生更短settling time的最优值，对于每次shooting，这种line searching step length会指数收缩。通常在3或4次迭代后，会达到局部最优解（local optima），我们会发现这样的解与grid search是高度可比的。

**设置$$\phi(t)$$的bounds**

我们也发现：设置控制信号$$\phi(t)$$的上下界对于KIPs可控来说很重要。由于在RTB中的动态性，用户CTR在一个周期内下降是正常的，这会使得eCPC更高。相应的feedback可能产生在bids上一个大的负向增益（nagative gain），这会导致极低的bid price，并且在剩余rounds上没有win、click以及没有额外开销。在这种情况下，一个合理的低下限(-2)的目标可以消除上述极端影响，可以阻止严重的负向控制信号。另外，一个上限(5)为了避免在远超reference value的变量增长。

**在线参数更新**

由于DSP会在feedback control下运行，收集的数据会被立即使用来训练一个新的PID controller并更新老的。我们研究了PID参数使用最近数据进行在线更新的可能性。特别的，在使用训练数据初始化PID参数后，我们会在每10轮上对controller进行重训练（例如：在round 10, 20, 30），在test stage使用所有之前的数据并使用与training stage相同的参数搜索方法。在re-training中的参数搜索会在每个controller上花费10分钟，它比round周期（2小时）要更短。图12展示了分别使用在线和郭线PID参数的control效果。可以看到，在每10轮后，在线调参的PID在管理控制围绕reference value的eCPC上会比离线更有效，产生更短的settling time以及更低的overshoot。另外，当切换参数时，没有明显的干扰或不稳定发生。有了在线参数更新，我们可以开始基于several-hour的training data来训练controllers并采用新数据来自适应更新参数来提升control效果。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/b64be8f332eb6b3f86b6696ae835486471918c88c1810a563b82b79eef8a8734e30cba0918550fd77192da715e6fc1c3?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=12.jpg&amp;size=750">

图12 在线/离线参数更新的控制

# 5.在线部署与测试

略





# 参考

- 1.[https://wnzhang.net/papers/fc-wsdm.pdf](https://wnzhang.net/papers/fc-wsdm.pdf)