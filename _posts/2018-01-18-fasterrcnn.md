---
layout: post
title: Faster R-CNN介绍
description: 
modified: 2018-01-18
tags: [PNN]
---

Faster R-CNN由Ross Girshick等人提出。

# 总览

在目标检测领域的最新进展来源于**候选区域法（region proposal methods）**以**基于区域的卷积神经网络（region-based convolutional neural networks）**的成功。尽管region-based CNN开销很大，但如果通过跨候选块（proposals）共享卷积，可以极大地减小开销。当忽略掉在候选区域（region proposals）上花费的时间时，Fast R-CNN通过使用极深网络已经达到了接近实时的准确率。现在，在主流的检测系统中，在测试时间上都存在着proposals的计算瓶劲。

候选区域法（Region proposal methods）通常依赖于开销低的特征以及比较经济的inference模式(schemes)。**选择性搜索法（Selective Search）**是其中一种最流行的方法之一，它会基于已经开发的底层特征(low-level features)，对超像素 (superpixels)进行贪婪式合并。当与其它有效的检测网络[paper 2]进行对比时，Selective Search方法会更慢些，在CPU的实现上每张图片需要2秒耗时。EdgeBoxes[6]方法提供了在proposal上的质量和速率上的最佳权衡，每张图片0.2秒。尽管如此，候选区域（region
proposal）阶段步骤仍然会像该检测网络一样消耗相当多的运行时（running time）。

有人注意到，fast RCNN(fast region-based CNN)可以利用GPU，而在研究中使用的候选区域法(region proposal methods)则通常在CPU上实现，使得这样的运行时比较变得不公平。很明显，一种用于加速proposal计算的方法就是：在GPU上重新实现。这是一个有效的工程解决方案，但重新实现会忽略下游的检测网络（down-stream detection network），从而失去共享计算的机会。

本paper中展示了一种新方法：**使用DNN来计算候选（proposals）**，来产生一个优雅并有效的解决方案，在给定检测网络的计算下，其中proposal计算的几乎没有开销。我们引入了新的**Region Proposal Networks（RPNs）**来在最新的目标检测网络[1][2]中共享卷积层。通过在测试时（test-time）共享卷积，计算proposals的边缘开销很小（例如：每张图片10ms）。

我们观察到，由region-based dectectors（例如：Fast R-CNN）所使用的卷积特征图，也可以被用于生成候选区域（region proposals）。在这些卷积特征（convolutional features）之上，我们通过添加一些额外的卷积层（conv layers）构建了一个RPN，这些layers可以对一个常规网格(regular grid)上的每个位置，同时对区域边界（region bounds）进行回归(regression)、以及生成目标得分（objectness scores）。RPN是这样一种**完全卷积网络（FCN: fully convolutional network）**[7]，它可以以end-to-end方式训练，特别适用于生成检测候选（detection proposals）。

<img src="http://pic.yupoo.com/wangdren23/HI4Ky9cg/medish.jpg">

图1: 多种scales和sizes下的不同模式（schemes）。(a) 构建图片和feature maps的金字塔，在所有scales上运行分类器 (b) 使用多个scales/sizes的filters，在feature map上运行  (c) 使用在回归函数中参照框(reference box)金字塔

RPN被设计成使用一个较广范围的比例（scales）和高宽比（aspect ratios）来高效地预测region proposals。对比于上面使用图片金字塔（图1，a）的方法或者过滤器金字塔（图1，b），我们引入了新的**锚点边框“anchor” boxes**，在多个不同尺度和高宽比的情况下充当参照（references）。我们的scheme可以被看成是一个对参照（references）进行回归的金字塔（图1,c），它可以避免枚举多个不同尺度和高宽比的图片或filters。当使用单尺度图片进行训练和测试时，该模型执行很好，并且能提升运行速度。

为了将RPN和Fast R-CNN目标检测网络进行统一，我们提出了一个training scheme，它可以轮流为region proposal任务和目标检测任务进行fine-tuning，并保持proposals固定。该scheme可以快速收敛，并生成一个使用卷积特征(可在任务间共享)的统一网络。

我们在PASCAL VOC benchmarks上进行综合评估，其中**使用Fast R-CNN的RPNs准确率比使用Fast R-CNN的Selective Search(baseline)要好**。同时，我们的方法没有Selective Search在测试时的计算开销——可以在10ms内有效运行proposals。使用昂贵的极深网络，我们的检测方法在GPU上仍然有5fps（包含所有steps）的帧率，这是一个在速率和准确率上实际可行的目标检测系统。我们也在MS COCO数据集上做了测试，并研究了在PASCAL VOC数据集上使用COCO数据进行提升。代码在：[matlab code](https://github.com/shaoqingren/faster_rcnn) 和 [python code](https://github.com/
rbgirshick/py-faster-rcnn)。

该paper的预览版在此前有发布。在此之后，RPN和Faster R-CNN的框架已经被其它方法实现并实现，比如：3D目标检测[13], part-based detection[14], instance segmentation[15]，image captioning[16]。我们的快速有效目标检测系统已经在比如Pinterests等商业系统中使用。

在ILSVRC和COCO 2015比赛中，Faster R-CNN和RPN是在ImageNet detection, ImageNet
localization, COCO detection, and COCO segmentation等众多领域第1名方法的基础。RPNs可以从数据中学到propose regions，这可以从更深和更昂贵特征中受益（比如101-layer residual nets）。Faster R-CNN和RPN也可以被许多其它参赛者使用。这些结果表明我们的方法不仅是一个有效的解决方案，也是一种有效方法来提升目标检索的准确率。

# 2.相关工作

**候选目标（Object Proposals）法**。在object proposal methods中有大量文献。可以在[19],[20],[21]中找到。广泛被使用的object proposal methods中包含了以下方法：

- 基于grouping super-pixels的方法（Selective Search, CPMC, MCG）
- 基于滑动窗口的方法（objectness in windows[24], EdgeBoxes [6])。

Object proposal methods被看成是dectectors之外独立的一个模块。

**深度网络法**：R-CNN法可以训练一个CNN的end-to-end网络来将proposal regions分类成目标类别（object categories）或是背景(background)。R-CNN主要扮演分类器的角色，它不会预测对象的边界（除了通过bounding box regression进行重定义）。它的准确率依赖于region proposal模块的性能。许多papers[25],[9],[26],[27]提出了使用深度网络来预测目标的bounding boxes。在OverFeat方法中[9]，会训练一个FC layer来为单个目标的定位任务预测box的坐标。FC-layer接着会转化成一个conv-layer来检测多个特定类别的目标。MultiBox方法[26],[27]会从一个最后一层为FC layer（可以同时预测多个未知类的boxes）的网络中生成region proposals，生成OverFeat方式下的单个box。这些未知类别的boxes可以被用于R-CNN的proposals。对比于我们的fully conv scheme，MultiBox proposal网络可以被用于单个图片的裁减或多个大图片的裁减。

。。。

# 3.Faster R-CNN

我们的目标检测系统，称为Faster R-CNN，由两个模块组成。第一个模块是深度完全卷积网络，它用于生成候选区域；第二个模块是Fast R-CNN检测器，它会使用这些候选区域。整个系统是一个统一的网络，使用了最近神经网络中的流行术语：attention机制，RPN模块会告诉Fast R-CNN模块去看哪里。在3.1节中，我们介绍了该网络的设计和属性。在3.2节中，我们开发算法来训练两个模块，并共享特征。

## 3.1 RPN

一个RPN会将一张图片（任意size）作为输入，输出一个矩形候选目标集合，每一个都有一个目标得分(objetness score)。我们使用一个完全卷积网络(fully conv network)将该过程建模，会在该部分描述。由于我们的最终目标是使用一个Fast R-CNN网络来共享计算，我们假设两个网络共享一个公共的卷积层(conv layers)集合。在我们的实验中，我们研究了ZF model[5]：它有5个共享的conv layers；以及VGG16 [3]
：它有13个共享的conv layers。

为了生成候选区域（region proposals），我们在由最后一个共享conv layer所输出的conv feature map上滑动一个小网络。该小网络会将一个在input conv feature map上的n x n的空间窗口作为输入。每个滑动窗口被映射到一个更低维的feature（ZF:256-d, VGG: 512-d）上。该feature会被fed进两个相邻的FC-Layer上——一个box-regression layer（reg），另一个是box-classification layer (cls)。在本paper中，我们使用n=3, 注意在输入图片上的有效接受域（effective receptive field）非常大(ZF: 171 pixels, VGG: 228 pixels)。该mini-network如图3(左）所示。注意，由于mini-network以滑动窗口的方式操作，FC-Layers会跨所有空间位置被共享。该结果很自然地使用一个n x n的conv layer进行实现，接着两个同级的1x1 conv layers（reg和cls）

### 3.1.1 Anchors

在每个滑动窗口位置上，我们同时预测多个候选区域（region proposals），其中每个位置的最大可能候选数量被表示成k。因而reg layer具有4k的输出，它可以编码k个boxes的坐标；cls layer输出2k个得分，它用来估计每个proposal是object还是非object的概率。k个候选(proposals)被相对参数化到k个参考框（reference boxes），我们称之为锚点（anchors）。一个anchor位于当前的滑动窗口的中心，与一个scale和aspect ratio（图3, 左）相关联。缺省的，我们使用3个scales和3个aspect ratios，在每个滑动位置上产生k=9个anchors。对于一个size=W x H（通常~2400）卷积特征图（conv feature map），总共就会有WHk个anchors。

#### 平移不变的Anchors

我们的方法的一个重要属性是：平移不变性（translation invariant）， 对于该anchors、以及用于计算相对于该anchors的proposals的该functions都适用。如果在一个图片中移动一个object，该proposal也会平移，相同的函数应能预测在该位置的proposal。这种平移不变特征由方法5所保证。作为比较，MultiBox方法[27]使用k-means来生成800个anchors，它并没有平移不变性。因而，MultiBox不会保证：如果一个object发生平移仍会生成相同的proposal。

平移不变性也会减小模型的size。MultiBox具有一个(4+1) x 800维的FC output layer，其中我们的方法具有一个(4+2) x 9维的conv output layer，anchors数为k=9个。结果是，我们的output layer具有$$ 2.8 \times 10^4 $$个参数(VGG-16: $$512 \times (4+2) \times 9$$)，比MultiBox的output layer的参数（$$ 6.1 \time 10^6 $$）要少两阶。如果考虑上特征投影层（feature projection layers），我们的proposal layers仍比MultiBox的参数少一阶。我们希望我们的方法在小数据集上（比如：PASCAL VOC）更不容易overfit。

<img src="">

图3: 

#### Multi-Scal anchors as Regression References

关于anchors的设计，提供了一种新的scheme来发表多个scales（以及aspect ratios）。如图1所示，具有两个流行的方法来进行multi-scale预测。第一种方法基于image/feature 金字塔，比如：DPM和基于CNN的方法。这些图片以多种scales进行resize，在每个scale上计算feature maps（HOG或deep conv features）（如图1(a)所示）。该方法通常很有用，但耗时严重。第二种方法是在feature maps上使用多个scales（或aspect ratios）的滑动窗口。例如，在DPM中，不同aspect ratios的模型使用不同的filter sizes（比如：5x7和7x5）进行单独训练。如果该方法用于解决multi scales，它可以被认为是一个“过滤器金字塔（pyramid of filters）”（如图1(b)所示）。第二种方法通常与第一种方法联合被采纳。

作为比较，我们的基于anchor的方法构建了一个关于anchors的金字塔，它效率更高。我们的方法会进行分类和回归bounding boxes，使用multi scales和aspect ratios的anchor boxes。它只取决于单一尺度的图片和feature maps，以及使用单一size的filters（在feature map上滑动窗口）。我们通过实验展示了该scheme用于解决multiple scales和sizes的效果（表8）。

由于该multi-scale设计基于anchors，我们可以简单地使用在单一尺度的图片上计算得到的conv features，这也可以由Fast R-CNN detector来完成。multi-scale anchors的设计是共享特征的核心关键（无需额外开销来解决scales问题）。

### 3.1.2 Loss函数

为了训练RPN，我们为每个anchor分配一个二元分类label（是object、不是object）。我们分配一个正向label给两种类型的anchors：

- (i) 具有与一个ground truth box的IoU（Intersection-over-Union）重合率最高的anchor/anchors
- (ii) 具有一个与任意ground-truth box的IoU重合度高于0.7的anchor

注意，单个ground-truth box可以分配一个正向label给多个anchors。通常第二个条件足够决定正样本；但我们仍采用了第一个条件，原因是有些罕见的case在第二个条件下会找不到正样本。

假如它的相对所有ground-truth boxes的IoU ratio低于0.3, 我们分配一个负向label给一个非正anchor. 即非正，也非负的anchors对训练目标贡献不大。

有了上述定义，我们根据在Fast R-CNN中的多任务loss来最小化目标函数。一张图片的loss function如下所示：

$$
L(\{p_i\}, \{t_i\}) = \frac{1}{N_{cls}} \sum_{i} L_{cls} (p_i, p^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* L_{reg}(t_i, t_i^*)
$$
...(1)

这里，

- i表示在mini-batch中的一个anchor的索引
- $$p_i$$表示anchor i是object的预测概率。
- 如果该anchor为正，ground-truth label $$p_i^*$$是1;否则为0. 
- $$t_i$$是一个向量，表示要预测的bounding box的4个参数化坐标
- $$t_i^*$$是与一个正锚点（positive anchor）相关的ground-truth box。
- $$L_{cls}$$是分类loss，它是关于两个类别的log loss。
- $$L_{reg}(t_i, t_i^*) = R (t_i - t_i^*)$$是回归loss，其中R是robust loss function（L1平滑）。
- $$p_i^* L_{reg}$$意味着regression loss当为正锚点时（$$p_i^*=1$$）激活，否则禁止（$$p_i^*=0$$）

两个项(term)通过$$N_{cls}$$和$$N_{reg}$$被归一化，通过一个参数$$\lambda$$进行加权。在我们当前实现中（释出的代码），等式(1)中的cls项通过mini-batch size进行归一化（例如：$$N_{cls}=256$$），reg项通过anchor位置的数目进行归一化（例如：$$N_{reg} ~ 2400$$）。缺省的，我们设置$$\lambda=10$$，接着cls和reg两项会被加权。通过实验我们发现，结果对于$$\lambda$$的值在一个宽范围内是敏感的（见表9）。我们也注意到，归一化（normalization）不是必需的，可以简化。

表9

对于bounding box回归，我们采用了以下的4坐标的参数化：

$$
t_x = (x-x_a) / w_a, t_y = (y-y_s) / h_a
$$

$$
t_w = log(w/w_a), t_h = log(h/h_a)
$$

$$
t_x^*=(x^* - x_a)/w_a, t_y^*=(y^* - y_a) / h_a
$$

$$
t_w^* = log(w^*/w_a), t_h^* = log(h^*/h_a)
$$

...(2)

其中，x, y, w和h表示box的中心坐标、宽、高。变量x, $$x_a$$，以及$$x^*$$分别是预测box，anchor box，ground-truth box（y,w,h也类似）。这可以被认为是从一个anchor box到一个接近的ground-truth box的bounding-box regression。

然而，我们的方法与之前的基于RoI（Region of Interest）的方法不同，通过一种不同的方式达成bounding-box regression。bounding-box regression在由特定size的RoI上的features来执行，该regression weights被所有region sizes共享。在我们的公式中，用于回归的该features在feature maps上的空间size上(3x3)相同。为了应付不同的size，会学到k个bounding-box regressors集合。每个regressor负责一个scale和一个aspect ratio，k个regressors不会共享权重。由于anchors的这种设计，仍然能预测不同size的boxes，即使features是一个固定的size/scale。

### 3.1.3 训练RPNs

RPN可以通过BP和SGD以end-to-end的方式进行训练。我们根据"以图片为中心（image-centric）"的抽样策略来训练网络。从单个图片中提取的每个mini-batch，包含了许多正负样本锚点。它可以为所有anchors的loss functions进行优化，但会偏向主导地位的负样本。作为替代，我们在一个图片上随机抽取256个锚点，来计算一个mini-batch的loss函数，其中抽样到的正负锚点的比例为1:1。如果在一个图片中正样本数少于128个，我们将将该mini-batch以负样本进行补齐。

我们通过从一个零均值、标准差为0.01的高斯分布中抽取权重，来随机初始化所有new layers。所有其它layers（比如：共享的conv layers）通过ImageNet分类得到的预训练模型进行初始化。接着调整ZF net的所有layers，conv3_1以及来保存内存。我们在PASCAL VOC数据集上，对于mini-batches=60k使用使用learning rate=0.001，对于mini-batch=20k使用learning rate=0.0001. 我们使用一个momentum=0.9, weight decay=0.0005, 代码用Caffe实现。

## 3.2 为RPN和Rast R-CNN共享特征

我们已经描述了如何去训练一个网络来进行region proposal的生成，无需考虑基于region的目标检测CNN会使用这些proposals。对于检测网络，我们采用Fast R-CNN。接着，我们描述的算法会学到一个统一的网络，它由RPN和Fast R-CNN组成，它们会共享conv layers（如图2）。

图2

如果RPN和Fast R-CNN独立训练，会以不同的方式修改它们的conv layers。因此需要开发一个技术来允许在两个网络间共享conv layers，而非学习两个独立的网络。我们讨论了三种方式来训练特征共享的网络：

- (i) 交替训练（Alternating training）。在这种方案中，我们首先训练RPN，接着使用这些proposals来训练Fast R-CNN。 该网络会通过Fast R-CNN进行调参，接着被用于初始化RPN，然后反复迭代该过程。这种方案被用于该paper中的所有实验。
- (ii) 近似联合训练（Approximate joint training）。在这种方案中，RPN和Fast R-CNN网络在训练期间被合并到一个网络中，如图2所示。在每个SGD迭代过程中，forward pass会生成region proposals（当训练一个Fast R-CNN detector时，他们被看成是固定的、预计算好的proposals）。backward propagation会和往常一样进行，其中对于共享的layers来说，来自RPN loss的Fast R-CNN loss的后向传播信号是组合在一起的。该方案很容易实现。但该方案会忽略到关于proposal boxes坐标的导数（derivative w.r.t. the proposal
boxes’ coordinates）, 也就是网络响应，因而是近似的。在我们的实验中，我们期望发现该求解会产生闭式结果，并减少大约25-50%的训练时间（对比alternating training）。该求解在python代码中包含。
- (iii) 非近似联合训练（Non-approximate joint training）。根据上述讨论，由RPN预测的bounding boxes也是输入函数。在Fast R-CNN中的RoI pooling layer会接受conv features，以及预测的bounding boxes作为输入，因而一个理论合理的BP解也与box坐标的梯度有关。这些梯度在上面的approximate joint training会被忽略。在非近似方法中，我们需要一个RoI pooling layer，它是box坐标的微分。这是一个非平凡问题，解可以通过一个"RoI warping" layer给出[15]。（超出本paper讨论范围）

#### 4-step Alternating Training

在该paper中，采用了一个实用的4-step training算法来通过alternating优化来学习共享特征。在第一个step中，会如3.1.3节描述来训练RPN。该网络使用一个ImageNet-pre-trained模型来初始化，为region proposal任务来进行end-to-end的fine-tuning。在第二个step中，我们训练了一个独立的Fast R-CNN dectection网络，它会使用由第一步的RPN生成的proposals。该检测网络也使用ImageNet-pre-trained模型初始化。在此时，这两个网络不共享conv layers。在第三个step中，我们使用detector网络来初始化RPN training，但我们会固定共享的conv layers的能数，只对对于RPN唯一的layers进行fine-tune。最后，保持共享的conv layer固定，对Fast R-CNN的唯一layers进行fine-tune。这样，两个网络会共享conv layers，并形成一个统一网络。相类似的alternating training会运行很多次迭代，直到观察到不再有提升。

## 3.3 实现细节

我们在单一scale的图片上，训练和测试两个region proposal以及目标检测网络。我们re-scale这些图片，以至它们更短的边: s=600 pixels。Multi-scale特征抽取（使用一个图片金字塔image pyramid）可以提升accuracy，但不会有好的speed-accuracy的平衡。在re-scale的图片上，对于ZF和VGG nets来说，在最后一层conv layer上的总stride为16 pixels，在一个典型的PASCAL image上在resizing（~500x375）之前接近10 pixels。尽管这样大的stride会提供好的结果，但accuracy会使用一个更小的stride进行进一步提升。

对于anchors，我们使用3个scales，box areas分别为：$$128^2, 256^2, 512^2$$个pixels，3个aspect ratios分别为：1:1, 1:2, 2:1. 对于一个特定数据集，这些超参数并不是精心选择的，我们提供了消融实验。我们的解不需要一个图片金字塔或是过滤器金字塔来预测多个scales的regions，节约运行时间。图3(右）展示了在一个关于scales和sapect ratios范围内我们方法的能力。表1展示了对于每个anchor使用ZF net学到的平均proposal size。我们注意到，我们的算法允许预测比底层的receptive field更大。这样的预测是不可能的——如果一个object只有中间部分可见，仍能infer出一个object的其它部分。

该anchor boxes会交叉图片的边界，需要小心处理。在训练期间，我们忽略了所有交叉边界anchors（cross-boundary anchors），因而他们不会对loss有贡献。对于一个典型的1000x600的图片，共有20000 (~60x40x9)个anchors。由于忽略的cross-boundary anchors的存在，训练期每个图片有大约6000个anchors。如果boundary-crossing outliers在训练期被忽略，他们会引入大的、难的来纠正在目标函数中错误项，训练不会收敛。在测试期，我们仍应用完全卷积的RPN到整个图片上。这也会生成cross-boundary的proposal boxes，我们会将image boundary进行裁减。

表2

一些RPN proposals高度相互重叠。为了减小冗余，我们在proposal regions上基于它们的cls分值采用了NMS(non-maxinum suppression)。我们为NMS将IoU阀值固定为0.7，可以为每张图片留下2000个proposal regions。NMS不会对最终的检测accuracy有害，实际上会减小proposals的数目。在NMS后，我们使用top-N排序后的proposal regions进行detection。然后，我们使用2000个RPN proposals训练Fast R-CNN，但在测试时评估不同数目的proposals。

# 4.实验

## 4.1 PASCAL VOC

在PASCAL VOC 2007 detection benchmark上进行评估。该数据集包含了5k个trainval images，以及5k个test images，object类别超过20个。我们也提供了PASCAL VOC 2012 benchmark。对于ImageNet pre-trained network，我们使用ZF net的"fast"版本：它具有5个conv layers以及3个FC layers，以及公开的VGG-16 model：它具有13个conv layers以及3个FC layers。我们使用mAP（ mean Average Precision）进行评估detection，因为实际的目标验测的metric（而非关注目标的proposal proxy metrics）。

表2展示了使用不同region proposal methords的训练和测试结果。对于Selective Search(SS)[4]方法，我们通过"fast"模式生成了大约2000个proposals。对于EdgeBoxes（EB）[6]方法，我们通过缺省的EB setting将IoU设置为0.7来生成proposals。在Fast R-CNN框架下，SS的mAP具有58.7%，而EB的mAP具有58.6%。RPN和Fast R-CNN达到的完整结果为，mAP具有59.9%，仅使用300个proposals。使用RPN会比SS或EB生成一个更快的检测系统，因为共享卷积计算；更少的proposals也会减小region-wise FC layers的开销（表5）。

RPN上的Ablation实验。为了研究RPN作为proposal method的行为，我们做了一些ablation研究。首先，我们展示了在RPN和Fast R-CNN检测网络间共享卷积层（conv layers）的效果。为了达到这个，我们在第二个step后停止训练过程。使用独立的网络将结果减小到58.7%（RPN+ZF,unshared, 表2）。我们观察到这是因为在第三个step中，当detector-tuned features被用于fine-tune该RPN时，proposal质量会被提升。

接着，我们放开RPN对Fast R-CNN训练的影响。出于该目的，我们训练了一个Fast R-CNN模型，使用2000个SS proposals和ZF net。我们固定该detector，通过更改在测试时的proposal regions，来评估该detection的mAP。在这些ablation实验中，RPN不会与detector共享features。

在测试时，将SS替换成300 RPN proposals会产生mAP=56.8%。在mAP中的该loss是由于在training/testing proposals间的不一致性造成的。该结果会当成baseline。

# 评测

略，详见paper。

# 参考

- 1.[https://arxiv.org/pdf/1506.01497.pdf](https://arxiv.org/pdf/1506.01497.pdf)
