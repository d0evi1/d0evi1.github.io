---
layout: post
title: whole page优化介绍
description: 
modified: 2020-06-15
tags: 
---

yahoo在2016年《Beyond Ranking: Optimizing Whole-Page Presentation》的一篇paper里提出了针对whole-page优化的方法，我们来看下。

# 摘要

现代搜索引擎会从不同verticals中聚合结果：网页、新闻、图片、视频、购物（shopping）、知识页卡（knowledge cards）、本地地图（local maps）等。不同于“ten blue links”，**这些搜索结果天然是异构的，并且不再以list的形式在页面上呈现**。这样的变化直接挑战着在ad hoc搜索中传统的“ranked list”形式。因此，为这种异构结果group发现合适的呈现（presentation）对于现代搜索引擎来说很重要。

我们提出了一个新框架来学习**最优的页面呈现（page presentation）**，从而在搜索结构页（SERP：search result page）上渲染异构结果。**页面呈现被广泛定义成在SERP上呈现一个items集合的策略，它要比一个ranked list要更丰富些。它可以指定：item位置、image sizes、文本字体、其它样式以及其它在商业和设计范围内的变体**。学到的presentation是content-aware的，例如：为特定的queries和returned results定制化的。模拟实验表明，框架可以为相关结果自动学习到更吸引眼球的呈现。在真实数据上的实现表明，该框架的简单实例可以胜过在综合搜索结果呈现上的先进算法。这意味着框架可以从数据中学到它自己的结果呈现策略，无需“probability ranking principle”。

# 1.介绍

十年前，搜索引擎返回"十个蓝色链接（ten blue links）"。结果呈现（Result presentation）很简单：通过估计的相关度对webpages进行排序。当用户往下扫描列表时会省力些，可以在top ranks点击到他希望的信息。这种“**probability ranking principle**”在1970年开始已经存在很久，并且在eye-tracking研究[20,19]以及搜索日志分析[24,15]中被证实。

今天的搜索引擎会返回比“ten blue links”更丰富的结果。除了webpages，结果可能还包括：**news、images、video、shopping、结构化知识、本地商业地图等**。每个特定corpus可以通过一个垂类搜索引擎（vertical search engine）索引；他们联合（federated）在一起为用户信息提供服务。不同于“ten blue links”，**垂类搜索结果具有不同的视角外观、布局和大小（sizes）**。他们会在页面上跨多列，不会严格限制在mainline list上（图1）。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/a230b179b24a16c10888c7d42a1e86be2dfeb5b41135e6eb4a35f5b9e5fb0ad816a160314b25a4552d5a0c1581bd49c6?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=f1.jpg&amp;size=750">

图1 现代搜索引擎结果页

综合搜索结果会在搜索结果页（SERPs）上转移用户交互样式（user interaction patterns）。**人的眼球会自发地被图片结果（graphical results）吸引，从而产生一个显著的attention bias：vertical bias**[12, 26, 31]。更有趣的是，**在某个垂类结果（vertical result）周围的blue links被点击的概率也会增加**[12]。在垂类结果中，在整个SERP上的用户满意度不能从成对结果的偏好判断中可靠地推断出。

这些观察表明：**用户不会按顺序扫描由综合搜索（federated search）返回的结果**。尽管常规的"ranked list"公式仍会被用于综合搜索结果呈现上[5,4]，它本质上是一个一阶近似（first-order approximation）的问题。

本文中，我们提出了一个新的框架，它可以为在SERP上的异构搜索结果学习最优的呈现（presentation）。**Page presentation被定义成：在SERP上呈现一个异构items集合的策略（strategy），它比一个ranked list更具表现力。它可以指定item positions、图片尺寸、文本字体、以及任意在商业限束下的元素变化风格**。一个presentation的好坏可以通过用户满意度指标来判断：更好的呈现（presentation）会让用户更满意。该框架会学到一个scoring function，它会将搜索结果和其它在SERP上的presentation映射成用户满意度。接着，给定由一个new query的搜索结果，该框架能计算出一个能最大化用户满意度的presentation。

该框架相当通用。首先，我们可以灵活定义page presentation的范围。它可以将item positions（水平和垂直位置都行）、以及元素风格（图片大小、文本字体等）编码。ranked list只是它的一个特例。第二，不同应用场景可以采用不同的用户满意度指标。它不受click-based指标的限制，但也会将其它交互行为考虑其中，比如：停留时长（dwelling time）和首次点击时间（time-to-first-click）。最后，该框架也可以在其它交互搜索场景上实现，比如：**移动端或tablelet devices的搜索结果呈现**、在线社交网络的多媒体展示feeds、电商中的items布局（arranging） 。

我们在synthetic和real data上都做了实验，演示了该框架的强大。仿真实验展示了框架可以适配不同类型的attention bias，并可以学会呈现相关结果来捕获用户眼球。这意味着我们的方法可以直接针对由综合搜索带来的挑战，其中，用户不会按顺序扫描结果，并且结果不会以ranked list的形式展示。在real data实验中，framework的简单实现效果要好于在综合搜索结果排序上的先进算法。这是因为：ranked list在结果呈现上使用probability ranking principle的ranking算法，而我们的框架不认可它的存在。然而，它会纯粹从数据中学到它自己的结果呈现准则（result presentation principle），并可以达到SOTA的效果。

主要贡献：

- 1.我们对新问题（whole-page presentation optimization）进行公式化，它扩展了在ad hoc search上的异构文档排序
- 2.我们提出了一个通用框架，它会为综合搜索结果（federated search results）计算最优化的呈现（presentation）
- 3.在synthetic和real data上的实验表明：提出的框架可以解决新问题

# 2.问题公式化

Page Presentation Optimization的问题声明如下：“**给定在一个页面上要展示的搜索结果，发现可以最大化用户满意度（user satisfaction）的最优呈现**”。我们假设以下setting：搜索引擎返回针对某一query的一个结果集，并在SERP上根据一些呈现策略（presentation strategy）对items进行渲染。由于SERP会被展示给用户，用户可以与它交互来获得特定的用户满意度。现在假设我们在setting中定义了些重要概念：

**定义1（Page Content）**

Page Content是在页面上被展示的搜索结果集合。每个搜索结果是一个item。在接收到用户的query后，搜索引擎后台会返回一个关于k个items的集合。每个item被表示成为一个vector $$x_i$$。注意，不同的users和不同的queries会生成不同的items集合，因此$$x_i$$也可以从实际users和queries中编码信息。**Page Content被呈现成k个item vectors的concatenation：$$x^{\top}= (x_1^{\top}, \cdots, x_i^{\top}, \cdots, x_k^{\top})$$**。x的domain通过由后台返回的所有可能page content进行定义，被表示为X。

**定义2（Page Presentation）**

Page Presentation定义了要展示的page content x，比如：position、vertical type、size、color等。**它可以被编码成一个vector p**。p的domain通过在符合商业和设计约束下所有可能的page presentations进行定义，表示成P。

**定义3（Search Result Page, SERP）**

当Page Content x根据呈现策略p被排布在页面上时，会生成一个SERP。换句话说，content x和presentation p唯一地决定了一个SERP。**它可以被呈现成一个tuple $$(x, p) \in X \times P$$**。

**定义4（User Response）**

User Response包含了它在SERP上的动作（actions），比如：点击数、点击位置、点击停留时长、第一次点击的时间。**该信息被编码成一个vector y**。y的domain由所有可能的user responses进行定义，表示为Y。

**定义5（User Satisfaction）**

用户体验会随着他与SERP交互来确定满意度。我们假设，**用户的满意度可以被校准成一个real value $$s \in R$$：s越大意味着满意度越高**。

User Response是满意度的一个很强指标。直觉上，如果用户打开了SERP，并在top result上进行了合适的点击，接着在这些结果上花了较长的停留时间，表明他很可能会享受这些结果。

有了以上定义，我们可以将问题公式化：

**Page Presentation Optimization**：对于一个给定page content $$x \in X$$，我们的目标是发现这样的presentation $$p \in P$$：当$$SERP (x, p)$$被呈现给用户时，他的满意度得分可以被最大化。

如果我们假设，存在一个scoring function $$F: X \times P \rightarrow R $$，它可以将SERP(x, p) 映射到用户满意度得分s上，接着，Page Presentation Optimization问题可以正式写成：

$$
\underset{p \in P}{max} \ F(x, p)
$$

它遵循在Presentation p上的constraints。

Page Presentation Optimization问题是全新且充满挑战的。它之所以为新是因为page presentation可以被灵活定义，这使我们可以学习全新的方式来展示信息。检索和推荐系统通常会使用一个ranked list来展示异构内容（homogeneous content）。由于异构结果是网页形式编排，以极大化用户使用率的方式通过合理方式进行呈现很重要。该问题很具挑战性是因为：对于发现scoring function来将整个SERP（内容和呈现）映射成user satisfaction是非常不明确的。我们在下节提出我们的解决框架。

# 3.Presentation Optimization framework

我们提出了一种suprevised learning方法来解决page presentation optimization。该部分会建立一整个框架，包括：数据收集方法、scoring function $$F(\cdot, \cdot)$$的设计、learning和optimization阶段。在下一节，我们描述了实际的框架实例。

## 3.1 通过Exploration的数据收集

supervised learning需要labelled训练数据。在数据收集中的警告（caveat）是，**正常的搜索流量（normal search traffic）不能被当成训练数据来学习scoring function $$F(x,p)$$**。这是因为：**在正常的搜索流量中，search engine具有一个deterministic policy来呈现page content x，它由在系统中已经存在的model/rules所控制**。换句话说，Page Presentation p是由给定的Page Content x唯一确定的。然而，我们期望：随着我们通过不同的Page Presentations进行搜索，模型F可以告诉我们用户满意度。x和p间的混杂（confouding）会对学到的模型产生bias，这是一个非常严重的问题。

**为了消除混杂（confouding），我们会分配“Presentation Exploration Bucket”来做随机实验**。对于在该bucket中的请求，我们会使用随机的Page Presentation对Page Content进行组织。这里“随机（random）”意味着：会在商业和设计约束下均匀地抽取Presentation strategies，这样用户体验也不会损伤太大。更进一步，Presentation Exploration traffic由一个非常小量控制，因此不会影响整体的搜索服务质量。在这种方式下的数据收集保证了scoring function的无偏估计。

对用户随机曝光结果并不是我们想要的，也可以雇人工标注者来标记page，或者从使用不同的固定呈现策略（fixed presentation strategy）的多个buckets来收集数据，因为每个互联网公司都会测试他们的UI变化。由于我们已经在生产系统上开发了一种通过exploration framework的数据收集， 我们选择采用该方法来进行数据收集。

## 3.2 Learning Stage

Page Presentation Optimization的核心是，估计scoring function $$s = F(x, p)$$。我们可以考虑以下两个方法：

- (I) Direct方法：收集page-wise的用户满意度ratings，并直接对SERP和用户满意度间的依赖关系建模。该依赖路径（dependency path）是“$$(x,p) \rightarrow s$$”。
- (II) Factorized方法：首先，在SERP上预测user response y，接着寻找一个函数来从这些responses上measure用户满意度。该依赖路径（dependency path）是“$$(x,p) \rightarrow y \rightarrow s$$”。

方法(I)是简单的。然而，它非常难（当数据规模很大时，获得对于entire SERP的显式用户评分（explicit user rating）s很困难）。为了构建这样的数据集，我们需要大量的observations和人工标注来克服训练数据的稀疏性。

方法(II)分两步：

- 第一步：预测在一个给定页上的user responses；
- 第二步：基于它的page-wise response对用户满意度进行measure。

引入user response变量y可以在概念上进行分开：

- 一方面，在page上的user response是一个与页面交互的直接结果（direct consequence）
- 另一方面，用户满意度通常只通过从user responses中进行估计（比如：总点击数、或停留时长）

在方法（II）中，$$F(\cdot,\cdot)$$的复杂依赖被解耦成两个相对独立的因子。在实际上，**方法（II）对于当前的web技术来说更现实**，因为在SERP上的user response可以通过javascript很容易收集，而显式地询问用户来评估whole page是非常罕见的。因此，我们采用factorized方法。

在factorized方法中，第一步是学习一个user response模型：

$$
y = f(x, p), \forall x \in X, p \in P
$$

这是一个supervised learning任务；f(x,p)的实际形式可以被灵活选择。**我们可以简单地为在y中的每个component $$y_i$$构建一个模型（我注：类似于多目标?），或者我们可以直接使用结构化的输出预测（structured output prediction）联合预测y的所有component**。在任意case中，用户在页面上的responses同时依赖于content（相关的、多样化的、吸引人的）和presentation（是否接近top、在图片块周围、或以big size展示）。

第二步是一个utility function，它定义了一个用户满意度指标：

$$
s = g(y), \forall y \in Y
$$

基于page-wise user responses寻找合适的用户满意度指标不在本paper关注范围内，在[21,30,38]中有研究主题。确定，实践者通常会将指标定义成细粒度的user responses的聚合，比如：CTR、长停留点击（long-dwell-time clicks），首次点击时间（time-to-first-click）。

最终，对于整个SERP我们的scoring function为：

$$
s = F(x,p)= (g \circ f) (x, p) = g(f(x, p))
$$

## 3.3 Optimization stage

对于给定的content x，通过求解以下的optimization问题，我们计算了最优的presentation $$p^*$$：

$$
\underset{p \in P}{max} g(f(x,p))
$$

它遵循presentation p的约束（constraints）。

计算该optimization问题的计算开销，依赖于objective function $$F=g \circ f$$的实际形式，以及在presentation p上的constraints。在下一节中，我们展示了对于f和g的特定实例，$$p^*$$可以被相当有效地进行计算。

# 4.Presentation Optimization Framework

本节描述了该framework的实现，包括特征表示、用户满意度metric、两个user response模型以及它们的learning和optimization stages。我们会围绕l2r来展示该framework。

## 4.1 Features

在一个SERP上的content和presentation两者会同时在一个feature vector上进行表示，它会作为user response模型的input。

## Content Features

content features包含了query和相应搜索结果的信息，这与l2r中所使用的相似。我们采用与[23]中所使用相近的content features来进行实验对比：

- **全局结果集特征（Global result set features）**：由所有返回结果派生的features。他们指示了每个垂类（vertical）内容的是否有提供（availability）。
- **Query特征（Query features）**：词汇特征，比如：query unigrams、bigrams、共现统计等。我们也会使用query分类器的outputs、基于query features的历史session等
- **语料级特征（Corpus Level Features）**：来自每个vertical及web文档的query-independent features，比如：历史ctr、用户偏好等
- **搜索结果特征（search result features）**：从每个搜索结果中抽取得到。它是一个统计归纳特征列表（比如：每个单独结果的相关度得分、ranking features等）。对于一些verticals，我们也会抽取一些domain-specific meta features，比如：电影是否是在屏幕上，在movie vertical中是否提供电影海报，在news vertical中最新几小时的新闻文章的点击数。

## Presentation Features

Presentation features会在SERP上被展示的搜索结果进行编码，它是在框架中的新features。具体示例包括：

- **Binary indicators**：是否在某一位置上展示某个item。该scheme可以编码在线框（wireframe）中的position，比如：一个list或多列panels。假设在frame中存在k个positions，会展示k个items。假设i是items的索引，j是positions的索引，$$1 \leq i, j \leq k$$。item i的presentation，$$p_i$$，是一个1-of-k的binary encoding vector。如果document i被放置在position j，那么$$p_i$$的第j个component是1，其余为0. 在本case中，我们将$$p_i$$的值表示为$$p_{ij}-1$$。page presentation $$p^{\top} = (p_1^{\top}, \cdots, p_k^{\top})$$包含了$$k \times k$$的二元指示变量（binary indicator variables）、本质上编码了k个对象(objects)的排列（permutation）。
- **Categorical features**：page items的离散（discrete）属性，比如：一个item的多媒体类型（text还是image），一个textual item的字体（typeface）
- **Numerical features**：pape items的连续（continuous）属性，比如：一个graphical item的亮度、以及对比度
- **其它特征**：page content和presentation间的特定交叉可能会影响user response，比如：“在graphical item之上紧接一个textual item”

在实际实验中，我们会使用两种类型的presentation features。我们会使用binary indicators来编码items的位置。对于本地的搜索结果（local search results），我们会将presentation size编码成一个categorical feature（"single" vs. "multiple"条目）。

## 4.2 用户满意度metric

我们会假设：用户满意度指标 g(y)是对y中components的加权和的某种形式：

$$
g(y) = c^{\top} y
$$

在该实验中，我们使用关于k items的click-skip metric[23]：

$$
g(y) = \sum\limits_{i=1}^k y_i
$$

其中：

- 如果item i被点击，则有$$y_i=1$$；
- **如果item i被跳过并且它下面的某些item被点击**，则有$$y_i=-1$$。

一个skip通常表示浪费性检查（wasted inspection），因此我们会将它设置成一个单位的negative utility。该metric会强烈地偏向于在top positions上的邻近点击（adjacent click）。

## 4.3 User Response Models

我们会使用两个模型来预测page-wise user response：

- 第一个模型会采用在content和presentation间的特征二阶交叉（features quadratic interaction）。它会允许一个有效的最优化阶段（optimization stage）
- 第二个模型使用gradient boosted decision tree来捕获在content和presentation间的更复杂、非线性交叉。我们期等它来生成更好的效果.

**二阶特征模型（Quadratic Feature model）**

首先，假设我们考虑一个关于user response模型的简单实现，它可以在optimization stage上进行高效求解。由于它使用x和p间的二阶交叉特征（quadratic features），我们称之为“Quadratic Feature Model”。

假设：对于k个items存在k个positions。

- Page content x：是关于k个item vectors的concatenation；
- Page Presentation p：使用二元指示$$p \in \lbrace 0,1 \rbrace^{k \times k}$$进行编码，如第4.1节定义。
- vec：该模型也包含了x和p间的完全交叉作为features。假设 vec(A)表示包含了在matrix A中所有elements的row vector，一列挨一列，从左到右。

Quadratic Feature Model的增广特征向量（augmented feature vector）$$\phi$$为：

$$
\phi^{\top} = (x^{\top}, p^{\top}, vec(xp^{\top}))
$$

假设：

- $$y \in R^k$$是User Response vector；
- 每个component $$y_i$$是在item i上的一个User Response

线性模型$$f_i$$被用于预测在y中的每个$$y_i$$：

$$
y_i = f_i(\phi) = w_i^{\top} \phi = u_i^{\top} x + v_i^{\top} p + x^{\top} Q_i p
$$

...(1)

$$u_i, v_i, Q_i$$分别是content-only特征、presentation-only特征、content-presentation二阶交叉特征各自对应的参数。参数$$w_i = \lbrace u_i, v_i, Q_i \rbrace$$可以使用正则线性回归来估计。为了避免overfitting，我们会将$$u_i$$和$$v_i$$的$$L_2$$ norm进行正则化，并进一步在$$Q_i$$上利用low-rank regularization来处理二阶特征的稀疏性。

总之， 我们具有了k个这样的模型，每个模型会预测y中的一个$$y_i$$。为了在概念上将k个模型分组，假设将系数(cofficients)写成：

$$
U=(u_1,\cdots, u_k)^{\top}, \\
V=(v_1, \cdots, v_k)^{\top}, \\
Q=diag(Q_1, \cdots, Q_k)
$$

其中，将x和p“拷贝（copy）” k次来获得以下：

- matrix：$$X=diag(x^{\top}, \cdots, x^{\top})$$ 
- vector：$$t^{\top}=(p^{\top}, \cdots, p^{\top})$$

为了声明维度，如果$$x \in R^n, p \in R^m$$，那么：

$$
U \in R^{k \times n}, \\
V \in R^{k \times m}, \\
X \in R^{k \times nk}, \\
Q \in R^{nk \times mk}
$$

其中：$$t \in R^{mk}$$，user response model可以被写成：

$$
y = f(x, p) = Ux + Vp + XQ_t
$$

将用户满意度metric表示为：$$g(y) = c^{\top} y$$。那么scoring function $$F=g \circ f$$为：

$$
F(x,p) = g(f(x, p)) \\
= c^{\top} Ux + c^{\top} Vp + c^T XQ_t \\
= c^{\top} Ux + a^{\top} p
$$

...(2)

其中，$$a=V^{\top} c + \sum\limits_{i=1}^k c_i Q_i^{\top} x$$是一个已知vector。

最后，optimization stage会找到将(2)最大化的p，并满足在p上的constraints。由于给定了page content x，在(2)中的第一项是一个常数，可以丢弃。第二项$$a^T p$$是一个关于p的线性项。由于$$p \in \lbrace  0, 1\rbrace^{k \times k}$$会编码成一个k排列（k-permutation），在$$a \in R^{k \times k}$$中的每个component表示成用户满意度的增益，如果item i被放置在position j上，$$1 \leq i,j \leq k$$。因此，optimzation问题会化简成：最大二部图匹配（maximum bipartite matching），这是线性分配问题的一个特例。它可以通过Hungarian算法高效求解，时间复杂度为：$$O(\mid p \mid^3)=O(k^6)$$。在一个具有2GHz CPU的单核计算机上，对于50个items，该问题可以在10ms内求解。

**GBDT**

为了捕获在content x和presentation p间更复杂的非线性交叉，我们将前面章节的quadratic feature model $$f_i$$替换成gbdt模型：$$h_i^{GBDT}$$。GBDT是学习非线性函数的一个非常有效的方法。

我们的feature vector为：

$$
\phi^{\top} = (x^{\top}, p^{\top})
$$

在y中的每个user response $$y_i$$通过一个GBDT模型来预测：

$$
y_i = h_i^{GBDT}(x, p)
$$

用户满意度指标是：$$g(y) = c^{\top} = \sum\limits_{i=1}^k c_i y_i$$

在optimization阶段，由于每个$$h_i$$是一个无参数模型，我们不能根据p来获得$$F(x,p) = \sum\limits_{i=1}^k c_i h_i^{GBDT}(x, p)$$的解析形式。也就是说，在p上的optimization是棘手的。**在实际settings中，由于商业和设计约束，p的搜索空间通常会被减技到十位数的可能值**。我们可以实现并行枚举(paralled enumeration)来快速发现最优的presentation来最大化用户满意度。

## 4.4 特例：L2R

当我们将page presentation限定在一个ranked list时，假设：当更相关的结果被放在top ranks上时，用户会更满意，那么presentation optimization会化简成传统的ranking问题。

，，，

# 5.仿真研究

通过仿真（simulation），我们展示了presentation optimization framework的潜能。我们使用synthetic dataset，以便我们可以知道“ground truth”机制来最大化用户满意度，因此，我们可以轻易地确认该算法是否可以真正学到最优的page presentation来最大化用户满意度。在该研究中，我们已经有两个目标：

- (1) 我们展示了该framework允许page presentation的通用定义
- (2) 我们使用position bias和item-specific bias两者来展示framework可以自动适配用户交叉习惯

## 5.1 总览

我们首先给定一个关于simulation workflow的总览。**该仿真的“Presentation Exploration Bucket”会生成一个包含由random presentation的items set组合成的页面**。每次生成一个新页面时，每个item被分配一些从一个底层分布中抽样的reward（例如：相关信息）。仿真的“user”会具有一个特定类型的attention bias：

- (1) position bias：比起其它地方，用户会花更多注意力在页面的特定区域（图2a所示）
- (2) vertical bias（或item-specific bias）：某一特定类型item及它的周围会更吸引人的注意力

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/a9a917ddf7587ddf8fb7a3f14428b74da0cbbd1083eaf07af8429f5afa1c9029b5f16f79688dd7576eb6a6026c88657f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=f2.jpg&amp;size=750">

图2 不同类型的user attention bias

(**当"Presentation Exploration Bucket"生成一个page时，“user”带有attention bias去检查它时，就发生一次“interaction”。当用户检查一个item时，他会接受到相应的reward。对该page的用户满意度是rewards的总和**。Page Content、Presentation、以及被检查的items和positions（user responses），会变成框架希望学习的数据。最终，我们会测试该框架是否成功学到用户的attention bias。给定items的一个新集合，我们会希望看到，该框架会将具有更高rewards的items放置到更容易获得注意力的positions上来达到最大化用户满意度。因此，为了对模型在user attention bias上的当前置信（current belief）进行可视化，我们可以在该page上绘制item rewards的分布。

## 5.2 数据生成过程

在“search engine”侧，一个page（不管是1D list还是2D grid）包含了k个positions。Page Content $$x=(x_1, \cdots, x_k)^T$$ 以及 $$x_i \sim N(\mu_i, \sigma)$$表示了k个items的内在奖励（intrinsic reward）。对于1-D list我们设置k=10，对于2-D grid设置k=7 x 7。 $$\mu_i$$是从[0, 1]中抽取的随机数字，$$\sigma=0.1$$。page presentation p从k-permutations中随机均匀抽取。**whole page被表示成：(x, p)**。

在"user"侧，attention bias按如下方式仿真：

**Position bias**：检查position j是否是一个具有参数$$p_j$$的Bernoulli随机变量。一个真实示例是：top-down position bias，当user与一个ranked list交互时常观察到。

**Item-specific bias**：检查item i是否是一个具有参数$$p_i$$的Bernoulli随机变量。一个真实示例是：vertical bias，当用户与一个包含了垂直搜索结果（vertical search results：images、videos、maps等）交互时常观察到。

接着，"user"会与页面(x, p)进行交互：k个binary values会从k个Bernoulli分布中抽取得到，并且被记录成一个user response vector $$y \in \lbrace 0, 1 \rbrace^k$$。如果item i被examine到，$$y_i=1$$，该user收到一个reward $$x_i$$。用户满意度等于examined items的reward总和。我们会生成10w个pages来训练4.3节中的Quadratic Feature Model。

## 5.3 讨论

为了对学到的最优的presentation进行可视化，我们会选择一个随机的x，并计算相应的最优的presentation $$p^*$$，接着根据$$p^*$$来安置$$x_i$$。一个page可以被可视化成关于$$x_i$$的rewards的热力图（heat map）。理想情况下，具有更高reward的items（“better content”）应该放置在具有更高概率user attention的position上。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/2078c92e27115bf2df3c7e60dfb2b5d5d38b6373e2ba5789d0ee0405012f008b805986121bc9e633221cc892d7e8020c?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=f3.jpg&amp;size=750">

图3 1-D list中的top position bias和presentation

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/94c493368548e7361e44d064a70bccf9d3c944ac1e13b1b3e80c22789a4343e15c76736612f520a9530c94aab88de0e1?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=f4.jpg&amp;size=750">

图4 2-D canvas上的top-left position bias

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/3474ca896a820ad8b7af01b2a4a575e6826d294f6d4b1882ca9519b8458dafed33cdb98ac65af5c669e2f2566234e51e?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=f5.jpg&amp;size=750">

图5 2-D canvas上的two-end position bias和presentation

图3、4、5在多个position biases上对presentation结果进行了可视化。我们可以看到，**该算法的确能学到将“更好内容（better content）”放置到具有更多user attention的position上**。由于Page Presentation的定义是通用的，对于1-D list和2-D grid它都可以处理。另外，它可以捕获在2-D canvas上的position bias的复杂分布：在图4上的top-left position bias，以及在图5上的top-bottom position bias。

图6展示了在item-specific bias下的结果可视化。这是个很有意思的case，其中在page上的一个item是非常夺人眼球的，并且它也会吸引用户的attention到它周围的items上（例如：一个image会吸引用户的眼球，同时也会吸引注意力在在大标题(caption)和描述文本上）。另外假设：对于那些远离eye-catchy item的items，用户的attention会进一步下降。那么，最优的presentation strategy是放置item在page的中心，以便whole page会分派最大的reward。在图6中，我们可以看到：当该item（深红色区域）位于页面中心时，用户满意度值s是最高的。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/68aeb5c0315ccd287d6b9f28e36296f84cf74f9dd07d6d79e78dbda08faa3fbfe40967b28ecc6771ce336720a36bd2a9?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=f6.jpg&amp;size=750">

图6 Item-Specific bias。s: page-wise user satisfaction。当一个specific item(例如：图片)吸引了用户的注意力时，它周围的结果也会受到关注，那么: **当垂类（vertical）被放置在页面中心时，page-wise reward最高**

# 6.真实数据实验

通过在一个商业搜索引擎上获取的real-world dataset，我们展示了page presentation最优化框架的效果。

## 6.1 数据收集

我们使用一个非常小部分的搜索流量作为presentation exploration buckets。该数据会通过2013年收集。垂类搜索结果（vertical search results）的presentation会包括：news、shopping和local listings进行探索（exploration）。**在exploration buckets中，web结果的顺序会保持不变，verticals会被随机地slot落到具有均匀概率的positions上**。随机生成的SERPs不会受系统ranking算法的影响。如3.1节所示，**当训练模型时，还需要消除page content confouding**。该exploration SERP接着被呈现给正常方式交互的用户。用户在SERP上做出response，与page-wise content信息（比如：query、以及来自后端的document features）会被日志化。

## 6.2 方法

我们使用两个pointwise ranking模型作为baseline方法。他们使用4.1节描述的content features进行训练。第一个baseline方法在生产环境中使用（logit-rank）[23]。它会为每个垂类结果（vertical result, 包括web result）估计一个logistic regression模型：

$$
y_i = \sigma(w_i^T x_i)
$$

其中，$$y_i$$是一个binary target变量，它表示结果是否被点击（$$y_i = +1$$）或被跳过（$$y_i=-1$$）（4.2节有描述），其中$$\sigma(\cdot)$$是归一化到[-1, 1]间的sigmoid link function。

第二个baseline方法使用GBDT来学习一个pointwise ranking function(GBDT-RANK)，它使用一个GBDT regressor：

$$
y_i = h_i^{GBDT}(x_i)
$$

我们会评估4.3节中presentation optimization框架的两种实现：Quadratic Feature Model (QUAD-PRES)和GBDT Model（GBDT-PRES）。他们使用pase-wise信息(x,p)来预测user response vector，例如：clicks和skips的vector。

在实现中，我们使用Vowpal Wabbit来学习logistic regression模型，XGBoost来学习GBDT模型。模型的超参会在一个holdout数据集上进行调参。

## 6.3 评估

**我们使用一半的exploration SERAP作为训练集（1月到6月），其余做为测试集。它包含了上亿的pageview，从真实流量中获取**。对比标准的supervised learning setup，它很难做一个无偏的离线效果评估，因为该任务存在天然交互性。这是因为offline data $$x^{(n)}, p^{(p)}, y^{(n)}$$使用一个指定的logging policy收集得到，因此我们只能观察到对于一个指定page presentation $$p^{(n)}$$的user response $$y^{(n)}$$。

在离线评估中，当给定Page Content $$x^{(n)}$$时，该算法会输出一个presentation $$p^{*(n)} \neq p^{(n)}$$，但在离线时我们观察不到user response，因此不能评估它的好坏。**为了解决该问题，我们使用一个offline policy evaluation方法[28]来评估online推荐系统**。它的实现很简单，可以提供一个无偏的效果估计（unbiased performance estimate），依赖于通过random exploration的数据收集。

给定通过random exploration收集到的一个关于events的stream：

 $$
 (x^{(n)}, p^{(n)}, Pr(p^{(n)}), y^{(n)})
 $$
 
其中：

- $$x^{(n)}$$：Page Content
- $$p(n)$$：对应的Page presentation
- $$Pr(p^{(n)})$$：指的是从均匀随机曝光中生成$$SERP(X^{(n)}, p^{(n)})$$的概率
- $$y^{(n)}$$：得到的User Response

对于N个offline events的平均用户满意度可以计算为：

$$
\bar{s} = \frac{1}{N} \sum\limits_{n=1}^N \frac{g(y^{n}) 1_{\lbrace  p^{*(n)} == p^{(n)}\rbrace}}{Pr(p^{(n)})}
$$

其中：

- $$1_{\lbrace \cdot \rbrace}$$是indicator function
- $$g(y^{(n)})$$是对SERP n的用户满意度

这意味着该算法会在这些exploration SERPs上会评估：**哪个presentation会与算法选的相匹配（match）**；否则在离线评估中该SERP会被抛弃。

随着match沿页面往下走，match rate会下降（表1）。**如果我们需要针对预测$$p^{*(n)}$$和实际$$p^{(n)}$$间的exact match，那么大部分test set会被抛弃，效果评估会趋向于具有大的variance，从而不可信**。我们的评估只关注在第一、第二、第三个webpage result之上的垂类结果。注意，第一个webpage result不总是在top rank上；top rank经常被垂类结果（vertical results）占据。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/33e137e0af2cf51c9c1e79281729958462f5bc368e2d9ae48c98396db5eb809b99f4cf166b45e1caa8e583bdb943dc86?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=t1.jpg&amp;size=750">

表1 random exploration presentation p与predicted optimal presentation $$p^*$$间的match rate。“Until Web1”意味着p和$$p^*$$会在第1个webpage结果之上编码相同的presentation

## 6.4 结果

表2展示了平均page-wise用户满意度。可以看到whole-page优化方法要胜过ranking方法，由于ranking方法会使用probability ranking principle通过相关度来对结果进行排序，它会假设存在一个top-down position bias。**QUAD-PRES和GBDT-PRES不会做出这样的假设，它们只会纯粹从数据中学到它们自己的result presentation principle**。GBDT模型的效果好于logistic regression模型的原因是：logistic regression假设线性边界，而GBDT则是对非线性边界建模。

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/8f5fe4ee820edaad8edefd09c5d9ecf03b453cb0dcb9963c230777579339850e03aab941db14ee1f7aebc53264f48f9c?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;uin=402636034&amp;fname=t2.jpg&amp;size=750">

表2

**注意：在我们的关于用户满意度metric（$$g(y)$$）的定义中，一个skip会产生negative utility $$y_i=-1$$**。QUAD-PRES和GBDT-PRES通常会比baseline方法更好，这是因为会考虑retrieved items、page presentation、以及在entire SERP上的交互，不只是单个结果。presentation-blind模型会建模logit-rank和gbdt-rank，它们总是希望将最可能获得点击的结果放置在top位置上。然而，**对于特定queries，人们可能倾向于跳过图片结果（graphical results）（例如：当用户真实搜索意图是information时，会跳过shopping ads）**。在这样的case中，一个click会趋向于发生在top rank之下。作为对比，presentation optimization方法会同时考虑在页面上的结果和它们的position。这会生成更易觉察的结果安置（arrangement）。当我们越往下的SERP时，我们可以看到GBDT-PRES会吸引更多的点击，并具有越少的跳过。

表3、4、5展示了在Web1、Web2、Web3上的CTR。"S. Local"表示本地商业结果的单个(single)条目（比如：餐馆）；“M. Local”表示本地商业结果的多个（mutiple）条目。对于相同的vertical/genre结果，会呈现不同的size。在CTR项上，ranking方法会具有非常强的效果，因为他们会直接对高CTR进行最优化。然而，whole-page最优化方法仍具竞争力，通过考虑page-wise信息，有时会具有更好的CTR。

有意思的是，对于News vertical，对SERP上的其它结果、以及presentation并不会有过多帮助。作为对比，明知的（knowing）page-wise结果可以帮助提升top-ranked local listings上的CTR一大截。一个可能的解释是：新闻与常规网页类似，包含了丰富的文本信息以及他们的内容相关度可以通过标准ranking函数进行建模。另一方面，local listings...


# 7.相关工作

## 7.1 检索中的文档排序


## 7.2 综合搜索（Federated Search）

综合搜索（Federated Search或aggregated search）指的是通过一个特定垂类搜索引擎集合，在SERP上聚合结果。通常，来自不同垂类的内容是异构的，并且视觉上更丰富。综合搜索具有两个子任务：垂类选择（vertical selection）和结果呈现（result presentation）。给定一个query，vertical selection的任务会精准地决定哪个候选垂类提供可能的相关结果。在从候选垂类中获得结果后，result presentation则将垂类结果进行合并（merge）来生成相同页上的网页结果。

该paper主要关注result presentation。之前的方法将它看成是一个ranking问题[5,34,23]。特别的，[5,23]采用pointwise ranking函数来排序结果（results）和块（blocks），而[5,34]也构建了pairwise的偏好判断来训练一个ranking函数。[14]考虑了图片和电商结果（shopping results）的2-D grid presentation。比起rankedlist和2-D grid，我们的框架则允许presentation的灵活定义，允许任意的frames、image sizes、文本字体。

综合搜索结果极大地改变了SERP的景观（landscape），它也反过来要求在评估方法上做出变更。[9]提出了一个whole-page relevance的概念。他们主张Cranfield-style evaluation对于在现代SERP上量化用户的整体体验是不充分的。它建议通过为多个SERP elements分配等级来评估整页相关度（whole-page relevance）。我们的框架通过定义一个合适的用户满意度指标来体现该思想。

我们的工作与商业搜索或在线广告竞价的whole-page最优化相关，主要关注于优化搜索服务提供者的回报。我们的框架更通用，并且可以通过更改optimization objective来应用到这些问题上。

## 7.3 搜索行为建模（Search Behavior Modeling）

为了分发好的搜索体验，在SERP上理解用户行为很重要。眼球跟踪实验和点击日志分析观察到：用户在浏览blue-link-only的SERPs上会遵从序列顺序。更低ranked results会使用更低概率被examined到。在另一方面，这些结果也证实了probability ranking principle，鼓励搜索引擎在top上放置更多相关结果。另一方面，当使用click-through data作为相关性来评估训练ranking functions时，必须处理position bias。

由于异构结果出现在SERP上，用户在浏览结果时不再遵循序列顺序。


# RL公式

Y. Wang在<Optimizing Whole-Page Presentation for Web Search>中，补充了RL的方式。

# 8. RUNTIME-EFFICIENT PAGE PRESENTATION POLICY

在本节中，我们提供了在page presentation optimization问题上的一个关于RL的新视角。第三节的方法实验是求解一个RL问题的众多可能方式之一。基于新公式，我们提供了一个policy learning方法，它可以求解相同问题，运行也很高效。最终，我们通过仿真实验展示新方法的高效性。

## 8.1 RL Formulation

我们引入普用的RL setup，接着将page presentation optimization转化成为一个RL问题。

在一个通用的RL setup中，一个agent会扮演着一个随机环境（stochasitc environment）的角色，它会在一个timesteps序列上顺序选择actions，最大化累积收益。它可以被看成是以下的MDP（Markov decision process）：

- state space S
- action space A
- initial state分布$$p(s_0)$$，state转移动态分布$$p(s_{t+1}|s_t, a_t)$$，满足Markov性质：$$p(s_{t+1}| s_0, a_0, \cdots, s_t, a_t) = p(s_{t+1}| s_t, a_t)$$
- 一个reward function $$r: S \times A \rightarrow R$$
- 在给定state上选择actions的一个policy: $$S \rightarrow P(A)$$，其中$$P(A)$$是在A上measure的概率集合，$$\theta \in R^m$$是一个关于m个参数的vector。$$\pi_{\theta}(a \mid s)$$是在state s上采取action a的概率。一个deterministic policy是一个特例，其中：一个action a在任意state上满足$$\pi_{\theta}(a \mid s) = 1$$
- 该agent会使用它的policy来与MDP交互来给出关于states、actions、rewards的一个trajectory（$$S \times A \times R$$）：$$h_{0:T}=s_0,a_0,r_0, \cdots, s_T, a_T, r_T$$。cumulative discounted reward(return)是：$$R_{\gamma} = \sum\limits_{t=0}^{\infty} \gamma^t r(s_t, a_t)$$，其中，discount factor $$\gamma \in [0, 1]$$决定了future rewards的present value
- action-value function $$Q^{\pi} (s, a) = E[R_{\gamma} \mid s_0=s, a_0=a;\pi]$$是在state s上采用action a的expected return，接着following policy $$\pi . Q^*(s,a) = max_{\pi} E[R_{\gamma} \mid s_0=s, a_0=a; \pi]$$是最优的action-value function
- agent的目标是获得一个policy $$\pi$$，它可以最大化expected return，表示为$$J(\pi)=E[R_{\gamma} \ \pi]$$

在page presentation optimization中，agent是这样的算法：对于每个到来的search query，它决定了在对应SERP上page content的presentation。相关概念对应如下：

- (1) 一个query的page content x是state，state space为X
- (2) page presentation p是一个action，action space为P
- (3) intial state distribution $$p(x)$$通过query分布来决定。由于我们不会建模在搜索引擎与用户之间的顺序交互，因此没有state transition dynamics
- (4) reward function是在一个给定SERP上的用户满意度v，我们会通过scoring function $$v=F(x,p)$$进行估计。在state-action space $$X \times P$$中的每个点是一个SERP
- (5) 一个policy会为给定的page content x选择一个presentation strategy p。这也是公式(1)的page presentation optimization问题。
-(6)由于没有state transition，收益 $$R_{\gamma}=v$$，discount factor $$\gamma=0$$，effective time horizon $$T=0$$
- (7) 由于该policy不会在initial timestep之后起效果，action-value function等于reward function，$$Q^{\pi}=F(x,p), \forall \pi$$，因而：$$F(x,p) = Q^*(s,a)$$
- (8)expected return $$J(\pi) = E[v \mid \pi] = E_{x \sim p(x)} [E_{p \sim \pi(p \mid x)}[F(x,p)]]$$是agent希望最大化的平均用户满意度

因此，page presentation optimization问题可以被看成是一个RL问题。第3节的方法实际上是一个Q-learning方法。它首先在exploration数据上通过supervised learning来学习最优的action-value function（在我们的case中，它与reward function/scoring function相一致）F(x,p)。接着，它通过选择能最大化optimal action-value function $$\pi(\ \mid x)=1$$的action来生成一个deterministic policy，如果：p能求解$$max_{p \in P}F(x,p)$$以及$$\pi(p \mid x)=0$$

该方法的一个主要缺点是，它必须在运行时为每个query求解一个组合最优问题（combinatorial optimization problem），这对于$$F(\cdot, \cdot)$$的复杂函数形式来说很难。幸运的是，在RL中将该问题进行重塑对于runtime-efficient solutions带来了新曙光。以下我们描述了policy learning的一种解决方案。

## 8.2 Page Presentation的Policy learning

我们会找寻一个新的page presentation算法：

- (1) 它在runtime时很高效
- (2) 表现力足够，例如：能捕获在一个页面上的综合交互
- (3) 通过exploration buckets上收集到的数据进行离线训练

这些需求对于Web-scale online应用来说很重要，因为：

- (1) runtime高效性直接影响着用户体验
- (2) 不同的items在一个SERP上进行展示时可能会有依赖
- (3) 搜索算法的离线更新可以减少未知exploration行为的风险

一个policy-based agent会满足上述所有要求。通过experience，agent会学到一个policy $$\pi_{\theta}(a \mid s)$$而非一个value function。在runtime时，对于给定的state s，它会从$$\pi_{\theta}(a \mid s)$$中抽取一个action a，在action space上不会执行optimization。在RL场景中，action space是高维或连续的，通常采用policy-based agent。

在我们的问题setting中，agent会从exploration data中学到一个policy $$\pi_{\theta}(p \mid x)$$。对于一个search presentation policy，它更希望是deterministic的，因为搜索引擎用户更希望搜索服务是可预期和可靠的。我们可以将一个deterministic policy写成：$$p.= \pi_{\theta}(x)$$。在runtime时，给定page content x，policy会输出一个page presentation p，它会渲染内容到页面上。为了捕获在page contents间的复杂交叉，$$\pi_{\theta}(\cdot)$$可以采用非线性函数形式。

现在，我们描述presentation policy的设计和训练：

### 8.2.1 Policy Funciton设计

policy $$\pi_{\theta}$$采用page content vector作为input，并输出一个presentation vector。output vector会编码一个关于k个items的排列（permutation）：$$x^T=(x_1^{top},\cdots, x_k^{\top})$$，同时带有其它categorical和numerical性质（例如：image sizes、font types）。它不会去询问$$\pi_{\theta}$$来直接输出一个关于k-permutation作为$$k \times k$$的binary indicators的一个vector，我们会考虑隐式输出一个k-permutation的函数。至少有两种方法可以考虑：

- (1) 通用排序方法（Generalized ranking approach）：$$\pi_{\theta}$$会为每个item输入一个sorting score，它定义了一个顺序：可以将k个items安排在k个位置上。
- (2) 通用的seq2seq方法（Generalized sequence-to-sequence approach）：$$\pi_{\theta}$$会为每个item-position pair输出一个matching score，这需要在k个items和k个positions间的一个二部图匹配（bipartite matching）

注意，在上面的两种方法中，k个positions可以采用任意任意layout，不限于1-D list。如果$$\pi_{\theta}$$会综合考虑所有items和它们的依赖，那么两种方法都是可行的。在本文中，我们会考虑方法（1）。我们在未来会探索方法(2)。

在方法(1)中，每个item i会具有一个sorting score $$f_{\theta}(\bar{x_i})$$。该sorting scores会通过相同的function $$f_{\theta}$$来生成。feature vector $$\bar{x_i}$$对于每个item i是不同的，并且会包含整个page content feature x。这可以通过将item i的维度放置到x的前面、并将item i的原始维度设置为0来达到。也就是说：对于$$i=1, \cdots, k$$， 有$$\bar{x_i}^{\top}= (x_i^{\top}, x_1^{\top}, \cdots, x_{i-1}^{\top}, 0^{\top}, x_{i+1}^{\top}, \cdots, x_k^{\top}$$。这允许函数$$f_{\theta}$$考虑整个page content，但会为每个item输出一个不同的score。为了捕获在items间的复杂交互，每个$$F_{\theta}$$是一个LambdaMart模型（例如：GBDT模型）。

### 7.2.2 Policy Training

有多种方法以离线方式来训练一个policy，包括：model-based方法、off-policy actor-critic方法。这里我们使用一个model-based方法，它需要首先学到reward function和state transition dynamics。在我们的setting中，reward function是scoring function $$F(x, p)$$，不存在state transition。因此我们采用两个steps来训练policy $$\pi_{\theta}$$：

- (1) 从exploration data中学习scoring function F(x, p)
- (2) 通过最大化expected return $$J(\pi_{\theta}) = E_{x \sim p(x)} [F(x, \pi_{\theta}(x))]$$来训练policy $$\pi_{\theta}$$

注意，step(1)和step(2)涉及到optimization step $$max_{p \in P} F(x, p)$$。它允许我们选择关于F和$$\pi$$的复杂函数形式来捕获每个页面间的复杂依赖。

在RL文献中，policy通常根据policy gradient方法进行训练。也就是说，$$\pi_{\theta}$$会通过沿$$\Delta_{\theta}J$$方法上移动$$\theta$$来逐渐改进。对于deterministic policies，计算$$\Delta_{\theta}J$$是non-trivial的，正如我们的case。我们在LambdaMart中使用$$\lambda-gradients$$的思想来最优化policy。

由于我们的policy $$\pi_{\theta}$$会通过soring来生成一个permutation，我们可以将$$F(x, \pi_{\theta}(x))$$看成是一种“list-wise objective”。确实：x包含了k个items，$$p=\pi_{\theta}(x)$$定义了一个关于它们的permutation，尽管该layout并不是一个"list"。在$$F(x, \pi_{\theta}(x))$$与常规listwise IR measures间（比如：NDCG和MAP）的差异是：NDCG和MAP基于人工分配的per-item relevance，而$$F(x, \pi_{\theta}(x))$$会自动从数据中学到。对$$J(\pi_{\theta})$$最优化会转化成对listwise objective $$F(x,\pi_{\theta}(x))$$进行最优化。listwise l2r方法LambdaMart很适合该场景，它可以对大多数IR measures进行优化。

LambdaMart使用了一系列的回归树（regression trees），每个会估计$$\lambda$$。对于一个query q，$$\lambda_i \in R$$是一个分配给文档$$d_i$$的数字，它表示当前sorting score $$u_i$$会被改变多少来提升由q检索到的ranked list的NDCG。因此，$$\lambda$$是NDCG对应于soring scores的gradients。他们可以在预测的ranking中，通过交换两个documents $$d_i$$和$$d_j$$进行计算，接着观察在NDCG上的变化：

$$
...
$$

其中，。。。

## 8.3 仿真实验

在与第5节中相同的仿真setup下，我们实现了上述的policy-based算法。我们的目标是展示：policy-based方法可以同时学到1D和2D layouts，它的表现与原始的page presentation算法（Q-learning算法）是可比的。

表6展示了在1D和2D场景上的page presentation算法的用户满意度。由于仿真用户满意度是stochastic的，我们会在每个setup上运行1000次求平均方差和标准差。

- Ideal算法如图3(b)和4(b)所示。它会将具有最高reward的item放置到具有最高检查可能性的位置上。它会给出效果的上界。
- Q-Learning算法是QUAD-PRES。它的作用如图3(c)和4(c)所类似
- 我们包含了Policy算法的两个变种。他们在scoring function F的实现上有差异。一个使用factorized方法，其中每个item的reward会使用一个GBDT模型来估计，并且pagewise用户满意度是估计的item rewards的求和。另一个使用direct方法，其中，单个GBDT模型会使用整个SERP信息(x,p)来直接预测pasewise的用户满意度。
- RANDOM算法则将items进行随机shuffle到各个位置上。它给出了算法的下界。

在两种场景下，Q-LEARNING的效果几乎与IDEAL一样好。使用factorized F的policy与在1D case中的Q-LEARNING差不多，比在2D case中的Q-LEARNING稍微差一些。这是因为我们使用了一个policy function $$\pi_{\theta}(x)$$来逼近在Q-LEARNING中的"$$argmax_{p \in P} F(x, p)$$" global optimization过程。它会在presentation质量和runtime效率间做一个tradeoff。

scoring function在policy-based方法上扮演着重要角色。在direct方法中，F会丢失pagewise用户满意度的内部结果，它对于泛化到未见过的presentations是必要的，可以在policy training期间提供accurate feedback。factorized方法会保留在$$F=g \circ f$$的结果，因此它会比direct方法训练一个更好的policy。

# 参考


- 1.[Beyond Ranking: Optimizing Whole-Page Presentation](http://www-personal.umich.edu/~qmei/pub/wsdm2016-ranking.pdf)
- 2.[Optimizing Whole-Page Presentation for Web Search](https://dl.acm.org/doi/pdf/10.1145/3204461)