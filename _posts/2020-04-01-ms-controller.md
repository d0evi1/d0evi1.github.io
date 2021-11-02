---
layout: post
title: MS 流量控制介绍
description: 
modified: 2020-03-25
tags: 
---

# 1.摘要

我们描述了一个实时竞价算法，来进行基于效果的展示广告分配。在效果展示广告中的核心问题是：匹配campaigns到广告曝光，它可以公式化成一个受限最优化问题（constrained optimization problem）：在有限的预算和库存下，可以最大化收益目标。当前实践是，在一个曝光粒度的可追踪级别上，离线求解最优化问题（例如：placement level），并且在预计算的静态分发模式下，基于online的方式服务广告。尽管离线方法会以全局视角来达到最优目标，但它不能扩展到以单个曝光级别上进行广告分发决策。因此，我们提出一个实时竞价算法，它可以细粒度地进行曝光评估（例如，使用实时转化数据来定向用户），并且根据实时约束（例如：预算消耗级别）来调整value-based的竞价。因此，我们展示了在一个LP（线性规划：linear programming ）的primal-dual公式，这种简单实时竞价算法确实是个在线解法，通过将dual problem的该最优解作为input，可以解决原始的主问题。换句话说，在给定与一个离线最优化的相同级别的经验下，在线算法会保障离线达到的最优化。经验上，我们会开发和实验两个实时竞价算法，来自适应市场的非稳态：一个方法会根据实时约束满意级别，使用控制理论方法来调整竞价；另一个方法则会基于历史竞价静态建模来调整竞价。最后，我们展示了实验结果。

# 1.介绍

略

# 2.背景

略

# 3.效果展示最优化：一个LP公式

为了为在线竞价生成基本算法形式，并且确立它的最优化（optimality），我们通过对基本效果展示广告最优化看成是一个LP问题。在基础设定中，曝光会被评估，并且单独分配，在demand-side侧约束下（例如：预算限制），会以曝光分发目标的形式给出。该公式会捕获所有的理论本质，并且实际的细微差异会在第6节被讨论。假设：我们首先定义以下概念：

- 1. i会索引n个曝光，j会索引m个campaigns
- 2.$$p_{ij}$$表示曝光i分配到campaign j上的CTR，$$q_j$$表示campaign j的CPC；$$v_{ij}=p_{ij}q_{j}$$是这种assignment的eCPI
- 3.$$g_j$$是campaign j的曝光分发目标
- 4.$$x_{ij}$$是决策变量，它表示曝光i是否分配到campaign j（$$x_{ij}=1$$）或不是（$$x_{ij}=0$$）

我们将如下LP公式化成primal：

$$
max \sum_{ij} v_{ij} x_{ij} \\
s.t. \forall j, \sum_i x_{ij}  \leq g_j, \\
\forall i, \sum_j x_{ij} \leq 1, \\
x_{ij} \leq 0
$$

...(6)

dual problem接着：

$$
min_{\alpha, \beta} \sum_j g_j \alpha_j + \sum_i \beta_i \\
s.t. \forall i, j, \alpha_j + \beta_i \gt v_{ij} \\
\alpha_j, \beta_i \gt 0
$$

...(7)

重点注意的是：由于一个曝光，


# 5.1 基于控制论的竞价调整（bid adjustment）

基于经典控制理论的一个简单控制设计是，使用PI controller（proportialnal-intergral controller），这是proportional-integral-derivative (PID) controller的一种常用形式。据我们所知，在缺少低层过程的先验知识时，PID controller是最好的controller【3】。正式的，假设t表示time，$$r_j(t)$$和$$r'_j(t)$$分别是winning bids在time t上的期望概率（desired probabilities）和真实概率（observed probabilities）; $$e_j(t) = r_j(t) - r'_j(t)$$是在time t时的measure error。PI controller会采用如下形式：

$$
\alpha_j(t+1) \leftarrow \alpha_j(t) - k_1 e_j(t) - k_2 \int_0^t e_j(\tau) d\tau
$$

...(9)

这里:

- $$k_1$$是P项（比例增益：proportional gain）
- $$k_2$$是I项（积分增益：integeral gain）

两者都是待调参数。实际中，出于在线计算效率和曝光到达的离散性，time t不需要实时跟踪。假设：$$t \in [1, \cdots, T]$$会索引足够小的时间间隔（time intervals），其中T是在online bidding的整个duration内的intervals数目；在每个interval之后只会更新$$\alpha_j$$。

另一个更简单的控制方法：受水位标（Waterlevel）【4】的启发，在资源分配问题（resource allocation problems，比如：在保留位置上分发展示广告）上，有一个在线/快速近拟算法。waterlevel-based方法的更新公式：

$$
\alpha_j(t+1) \leftarrow a_j(t) e^(\gamma (\frac{x_j(t)}{g_j} - \frac{1}{T})), /forall j
$$

...(10)
其中：

- $$x_j(t)$$表示获胜的campaign j在time interval t期间的曝光数；
- 指数因子$$\gamma$$：是一个可调参数，它控制着算法根据erroe meaured $$\frac{x_j(t)}{g_j} - \frac{1}{T}$$来控制有多快。如果初始的$$\alpha_j$$（例如：由offline dual求解得到）对于未来的运行来说确实是最优的，我们希望将$$\gamma$$变为0

注意，在error项$$\frac{x_j(t)}{g_j} - \frac{1}{T}$$中，我们假设知道在time intervals上具有一个均匀的曝光流（impression stream）。该假设并不重要，因为它可以通过添加一个时间依赖先验（time-dependent prior）来被很容易地移除。另外，Water-based方法的更新具有一个很nice的链条特性：

$$
\alhpa_j(t+1) = \alpha_j(t) exp(\gamma(x_j(t) / g_j - 1/T)) \\
= \alpha_j(t-1) exp(\gamma ( \sum\limits_{\tau=t-1}^t x_j(\tau) / g_j - 2/T)) \\
= ... \\
= a_j(1) exp(\gamma(\sum\limits_{\tau=1}^t x_j(\tau) / g_j - t/T))

$$

...(11)

## 5.2 Model-based的竞价调整（Bid Adjustment）

我们的model-based方法从现代控制理论【9】中抽理出来，其中系统（在我们的case中是竞价市场）状态的一个数学模型是，用于生成一个控制信号（在我们的case为：竞价调整$$\alpha_j$$）。更正式的，我们会假设：在胜选竞价（winning bids）上有一个参数分布P：

$$
w \sim P(\theta)
$$

...(12)

其中，$$\theta$$是模型参数。我们使用泛化形式，因为一个合适参数选择应通过数据来进行经验调节，并且可能是domain-dependent的。一些合理的选择是：在winning bids【13】的square-root（均方根）上有一个log-normal分布【7】以及一个Gaussian分布，但两者都不可以天然处理negative bids。在我们竞价调整的加法形式如等式（3），一个negative bid  $$b_{ij} = v_{ij} - \alpha_j < 0$$意味着：竞价者（bidder）不能满足由acquiring impression i的最小间隔，因而展示了整个value book的一个hidden part。我们：

- 将概率分布函数PDF表示成$$f(w;\theta)$$
- 将累积分布函数CDF表示成$$F(w;\theta)$$
- inverse CDF(逆函数)表示为$$F^{-1}(p; \theta)$$。
- 分布参数$$\theta$$的MLE由历史胜选竞价{w}的充分统计得到，它可以在线更新可读（例如：第一，第二时刻）。有了bidding landscape的静态模型后，我们可以通过使用bidding $$b_{ij} = v_{ij} - \alpha_j$$生成获胜概率：

$$
p(w \leq b_{ij}) = \int_{-\inf}^{b_{ij}} f(w;\theta) dw = F(b_{ij}; \theta)
$$

假设：对于所有impression i来说，winning bids会遵循一个单一分布是不现实的（通常是一个mixture model）。因此对于来自一个位置（placement）的一组同质曝光（homogeneous impressions）来说，会满足分布$$P(\theta)$$。实际上，我们会使用impression granularity level来安排$$P(\theta)$$，它同时具有supply-side和demand-side的拘束。现在假设我们只关注异构曝光（homogeneous impressions）。

我们希望将学到的胜选概率（winning probability）与未来的竞价行为相联系，来达到分发目标。假设：$$r_j$$是赢得剩余曝光的期望概率，表示campaign j满足目标$$g_j$$。在未来曝光i上，会尝试使用$$b_{ij} = F^{-1}(r_j;\theta)$$来竞价。然而，这种纯基于目标的方法，在使用feedback来显式控制future bids时会失败，因为会丢掉closed-loop controller（PID controller）上具有的稳定性、健壮性等优点来建模不确定性。换句话说，纯目标驱动的方法不会学到过往竞价（past bidding）做出的errors。我们提出一个model-based controller来解决该限制，并且利用由bidding landscape学到的知识。竞价调整的公式如下：

$$
\alpha_j(t+1) \leftarrow \alpha_j(t) - \gamma(F^{-1}(r_j(t)) - F^{-1}(r_j^' (t))), \forall j
$$

...(14)

其中，$$r_j(t)$$和$$r_j^'(t)$$分别是在time t上的期望胜选概率和真实胜选概率。乘法因子$$\gamma$$是一个可调参数，它控制着在对应于errors的一次更新上做出的rate。对于经典方法，model-based方法不会直接操作measured errors；作为替代，它会通过一个compact model $$(P(\theta))$$来将一个error signal（获胜概率error）转换成一个control signal（updated $$\alpha_j$$）。

当缺少一个好的参数时这种方式不好，一个非参数模型在也可以使用。我们需要维护一个empirical CDF作为一个two-way lookup table $$(F(w;D)和F^{-1}(p;D))$$来进行online inference。

# 6.效果展示广告优化：一个实际公式

我们已经开发了在算法1中的基本算法形式，并确定了在给定稳定曝光到达假设下的最优解。在基本的LP公式下，constraints会被编码成impression分发目标，曝光会被单独进行分配和评估。我们将LP问题直接使用商业约束进行公式化，主要是：demand-side预算限制以及supply-side的库存量；接着讨论在真实系统中要考虑的几个方面。假设我们首先更新以下概念：

- 1.i表示索引n个impression groups（比如：placements），在一个group中的impressions会被看成是不可区分的，因而对于给定一个campaign来说会生成一个相同的CTR估计。
- 2.$$g_j$$是对于campaign j的预算最高限额（budget cap）
- 3.$$h_i$$表示对于group i来说的曝光量限制或预测
- 4.$$x_{ij}$$表示：来自group i分配给campaign j的曝光数
- 5.$$w_i$$：表示来自group i的每次曝光的（traffic acquisition）开销（cost），例如：在Vickrey acution中的第二价格

注意：我们会在同一个解法中做出CTR预测和supply constraint。避免刮脂效应(cream-skimming)问题很重要。如果CTR预估比起即时的supply constraint来说更细粒度，一个optimization方法是，在每个impression group总是分配impressions更高的CTR机会，这很明显是不现实的。我们会引入cost term $$w_i$$来泛化生成optimization给其它players，例如：参与到一个second-prece auction的一个ad network或者demand-side平台。primal LP变为：

$$
max_x \sum_{i,j} (v_{ij} - w_{i}) x_{ij} \\
s.t. \forall j, \sum_j v_{ij} x_{ij} \leq g_j, \\
\forall i, \sum_j x_{ij} \leq h_i, \\
x_{ij} \gt 0
$$

接着该dual problem是：

$$
min_{\alpha, \beta} \sum_j g_j \alpha_j + \sum_i h_i \beta_i \\

$$


# 6.实验评估



# 参考

- 1.[https://www.nikhildevanur.com/pubs/rtb-perf.pdf](https://www.nikhildevanur.com/pubs/rtb-perf.pdf)