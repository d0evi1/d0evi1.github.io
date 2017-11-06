---
layout: post
title: Active Deep Learning验证码识别
description: 
modified: 2015-11-12
tags: [深度学习]
---

我们先来看下慕尼黑大学的paper：《CAPTCHA Recognition with Active Deep Learning》。

# 2.介绍

常用的方法是，以两个相互独立的步骤来检测图片中的文本：定位在文本中的词（words）或单个字符（characters）的区域，进行分割（segmenting）并接着对它们进行识别。另外，可以使用一个字典来排除不可能的词（words）。例如，Mori and Malik提出的方法来使用一个411个词的字典来识别CAPTCHAs。等等，**然而，在现代CAPTCHAs中，单个字符不能轻易地使用矩形窗口进行分割，另外字符可以相互有交叠**。这些CAPTCHAs与手写文本相类似，LeCun提出使用CNN来解决手写数字识别。这些CNNs被设计成用于构建layer by layer体系来执行分类任务。在2014年，Google fellow提出使用deep CNN来结合定位（localzation)，分割（segmentation）以及多字符文本的识别。jaderberg等提出了一种在自然场景图片中的文本识别。然而，对于训练过程，它们人工模拟创建了一个非常大的文本图片的集合。**相反的，我们则使用一个更小的训练集**。通过探索Active Learning，我们在运行期对该神经网络进行微调（fine-tune），我们的网络接收已正确分好类但高度不确定的测试样本的前馈输入(feed）。

# 3.用于CAPTCHA识别的Deep CNN

<img src="http://pic.yupoo.com/wangdren23/GRpFgn5r/medish.jpg">

图2. 用于captcha识别的卷积神经网络。该CNN由三个conv layer，三个pooling layer，两个fully-connected layer组成。最后的layer会输出所有数字的概率分布，我们可以由此计算预测数据（prediction）以及它的不确定度

我们提出了一种deep CNN来解决CAPTCHA问题。**我们的方法在识别整个sequence时没有预分割（pre-segmentation）**。我们使用如图2所示的网络结构。**我们主要关注6数字的CAPTCHAs**。每个数字（digit）在output layer中由62个神经元表示。我们定义了一个双向映射函数（bijection）：\$\sigma(x) \$，它将一个字符 \$ x \in \{ '0' ... '9' , 'A' ... 'Z', 'a' ... 'z' \} \$映射到一个整数\$ l \in \{ 0 , ... , 61 \}\$上。

我们分配第一个62输出神经元（output neurons）到第一个序列数字上，第二个62神经元到第二个数字上，以此类推。这样，对于一个数字 \$x_i\$神经元索引n被计算成 \$ n=i * 62 + \theta(x_i) \$，其中\$ i \in {0,...,5} \$是数字索引，例如，output layer具有\$ 6 * 62 = 372 \$个神经元。为了预测一个数字，我们考虑相应的62个神经元，并将它们归一化成总和（sum）为1. 图4展示了一个神经网络输出的示例。这里，对于首个数字的预测字符索引(predicted character index)是\$c_0=52\$，预测标签为\$ x=\theta^{-1}(c_0)='q'\$。

<img src="http://pic.yupoo.com/wangdren23/GRpVTAbj/medish.jpg">

图4. 对于CAPTCHA "qnnivm"的神经网络样本输出. 左图：每个数字有62个输出。黑箱展示了第一个数字的输出。右图：第一个数字的概率分布。总和归一化为1.

# 4.使用 Active Learning来减少所需训练数据

为了获得一个好的分类accuracy，CNNs通常需要一个非常大的训练集。然而，收集数百万的人工标注的CAPTCHAs是不可行的。因而，我们提出了使用Active Learning（见图3）。主要思想是，只有在必要时才添加新的训练数据，例如，如果样本的信息量足够用于重新训练（re-learning）。这个决定是基于预测文本的不确定性，它会使用best-versus-secnond-best的策略来计算。

<img src="http://pic.yupoo.com/wangdren23/GRpIrgS1/medish.jpg">

图3. Active Learning流程图. 我们首先在一个小的数据集上训练。接着，分类器被应用到一些新数据上，产生一个label prediction和一个相关的不确定度。有了该不确定度，分类器可以决定是否请求一个ground truth。在我们的case中，该query的完成通过使用prediction来解决给定的CAPTCHA，以及使用正确分类的样本。接着训练数据会增加，learning会被再次执行。在我们的方法中，我们使用一个deep CNN，它可以使用新加的训练样本被更有效的重训练。

## 4.1 获取不确定性

如上所述，通过将相应的网络输出的求和归一化为1，我们可以估计每个数字的预测分布。这样我们可以使用"best-vs-second-best"来计算整体的不确定度\$ \eta \$：

$$
\eta = \frac{1}{d} * \sum_{i=1}^{d} \frac{argmax{P(x_i) \ argmaxP(x_i)}}}{argmaxP(x_i)}
$$

...(2)

其中，\$P(x_i)\$是数字\$d_i\$所对应的所有网络输出集。这样，我们通过对每个数字的最佳预测（best prediction）来分割第二佳预测（second-best）。

## 4.2 查询groud truth信息

我们的CAPTCHA识别问题是场景独特的：无需人工交互就可以执行学习。我们通过只使用这些数据样本进行重新训练（re-training）即可完成这一点：分类器已经为它们提供了一个正常label。然而，简单使用所有这些正确分类的样本进行re-training将非常低效。事实上，训练会越来越频繁，因为分类器越来越好，因而会将这些样本正确分类。为了避免这一点，我们使用不确定值来表示上述情况：在每个learning round通过预测不确定度来区分正确分类的测试样本，以及使用最不确定的样本来进行retraining。我们在试验中发现，这种方法会产生了一个更小规模的训练样本，最不确定的样本对于学习来说信息量越大。

# 5.实验评估

我们在自动生成CAPTCHAs上试验了我们的方法。所有实验都使用caffe框架在NVIDIA GeForce GTC 750 Ti GPU上执行。

## 5.1 数据集生成

由于没有人工标注好的CAPTCHA数据集，我们使用脚本来生成CAPTCHAs。在自动生成期间，我们确保它在数据集上没有重复。

我们使用Cool PHP CAPTHCA框架来生成CAPTCHAs。它们由固定长度6个歪曲文本组成，类似于reCAPTCHA。它们具有size：180 x 50. 我们修改了该框架让它生成黑白色的图片。另外，我们已经禁止了阴影（shadow）和贯穿文本的线。我们也没有使用字典词，而是使用随机字符。因此，我们已经移除了该规则：每个第二字符必须是元音字符（vowel）。我们的字体是：“AntykwaBold”。图5展示了生成的一些样本。

<img src="http://pic.yupoo.com/wangdren23/GRpX4kD7/medish.jpg">

图5: 实验中所使用的CAPTCHAs样本

## 5.2 网络的设计

我们使用如图2所示的网络。

- 卷积层(conv layers)具有的size为：48, 64和128. 它们具有一个kernel size: 5x5，padding size：2。
- pooling layers的window size为2x2。
- 第一个和第三个pooling layers，还有第一个conv layer的stride为2. 
- 该网络有一个size=3072的fully connected layer，还有一个二级的fully connected layer(分类器)具有output size=372.

我们也在每个卷积层和第一个fully conntected layer上添加了ReLU和dropout。每次迭代的batch size为：64.

## 5.3 量化评估

我们使用SGD来训练网络。然而，对比其它方法，我们以独立的方式为所有数字训练该网络。learning rate通过\$ \alpha = \alpha_{0} * (1+\gamma * t)^{-\beta} \$的方式变更，其中，基础的learning rate为\$ \alpha_{0} = 10 ^{-2}\$，\$ \beta=0.75, \gamma=10^{-4}\$，其中，t为当前迭代轮数。我们设置momentum \$ \mu=0.9 \$，正则参数\$\lambda=5 * 10^{-4} \$。

最昂贵的部分是获取训练样本，我们的方法的目标是，降小初始训练集所需的size。因而，我们首先使用一个非常小的初始训练集（含10000张图片）来进行 \$5 * 10^4\$迭代。我们只达到9.6%的accuracy（迭代越大甚至会降低accuracy）。因而，我们希望使用Active Learning。

首先，我们再次使用含10000张图片的初始训练集进行 \$5 * 10^4\$迭代。然后，我们分类 \$5 * 10^4\$ 张测试图片。接着，我们从正确分类的数据中选取新的训练样本。我们可以全取，或者基于不确定度（uncertainty）只取\$5 * 10^3\$个样本：即有最高的不确定度，最低的不确定度，或者随机。不确定度的计算如4.1节所述。一旦新的选中的样本被添加到训练集中，我们重新训练该网络\$5 * 10^4\$迭代。接着，我们遵循相同的过程。我们在总共20次Active learning rounds rounds(epoch)中应用该算法。在每次\$5 * 10^3\$迭代后，在一个固定的验证集上计算accuracy。我们在正确但不确定的预测上获取了最好的表现（见图6）。所有的结果是两种运行的平均。

<img src="http://pic.yupoo.com/wangdren23/GSmwJs7b/medish.jpg">

图6: Active Deep Learning的学习曲线. 上图：训练集在每次迭代后随样选中样本的增加而增加。当使用所有正确的样本时（黑色曲线），在\$ 50 \dot 10^4 \$我们停止向训练集添加新的图片，因为训练集的size已经超过了 \$ 3 \dot 10^6 \$。 下图：只在新样本上重新训练该网络。竖直的黑线表示每轮Active Learning epoch的结束。

然而，在训练集上增加样本数需要存储。再者，增加迭代次数可以从累积的集合上受益，但它会占据更长的训练时间。对于所有这些原因，我们建议：在每次迭代进行重训练网络时，只使用选中的样本。因而，我们再次使用使用含10000张图片的初始训练集进行 \$5 \dot 10^4\$迭代训练。接着，对\$10^5\$次测试图片进行分类，使用\$10^4\$正确分类的图片进行替换，并再训练\$2.5 \dot 10^5\$。接着，我们遵循该过程，根据下面的规则来减小迭代次数：在6轮前使用\$2.5 \dot 10^4\$，在6-11轮使用\$2 \dot 10^4\$，在11-16轮使用\$1.5 \dot 10^4\$，在16-21轮使用\$ 1 \dot 10^4\$，在21-40轮使用\$5 \dot 10^3\$。我们再次使用正确但不确定的预测来获取最好的表现（见图6）。这是合理的，因为该网络会正确分类图片，仍对预测仍非常不确定。因而，它可以事实上学到：它对于分类确定是正确的。一旦有争议：误分类样本的学习应产生更好的结果。事实上应该如此，然而实际上并不可能。


# 参考

- 0.[CAPTCHA Recognition with Active Deep Learning](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwiYrqy28ZfXAhXFTrwKHc6fDT4QFggsMAA&url=%68%74%74%70%73%3a%2f%2f%76%69%73%69%6f%6e%2e%69%6e%2e%74%75%6d%2e%64%65%2f%5f%6d%65%64%69%61%2f%73%70%65%7a%69%61%6c%2f%62%69%62%2f%73%74%61%72%6b%2d%67%63%70%72%31%35%2e%70%64%66&usg=AOvVaw2F4P4WkfNXDLPgG4t9cOf1)
