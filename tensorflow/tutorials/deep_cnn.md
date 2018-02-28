---
layout: page
title:  tensorflow deep_cnn教程
tagline: 
---
{% include JB/setup %}

# 卷积神经网络

**注意：该教程面向tensorflow高级用户**

# 一、总览

CIFAR-10分类在机器学习中是一个常见的benchmark问题。该问题是为了将RGB 32x32 像素的图片分类成10个类别：

	airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

关于更多详情，可参见[CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html)和[Tech Report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)。

## 1.1 目标

该教程的目标是构建一个相对小的CNN来识别图片。在该过程中，该教程会：

- 1.为网络结构，训练和评估提出一个权威的结构
- 2.提供一个模板来构建更大和更复杂的模型

选择cifar-10的原因是，它足够复杂，可以体验许多tensorflow的特性来扩展到大模型上。同时，该模型足够小，训练很快，对于尝试新想法和体验新技术很理想。

## 1.2 教程要点

cifar-10教程展示了许多重要的结构来在tensorflow中设计更大和更复杂模型：

- 核心数学组件包括：conv，relu，max_pooling，[local response normalization](https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization) ([AlexNet paper 第3.3节](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf))
- 在训练期间网络激活函数的可视化，包括：输入图片、losses以及activations和gradients的分布
- 提供了Routines来计算学习参数的[移动平均([moving average](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage))]，以及在评估期间使用这些平均来增强预测效果
- 提供了[learing rate schedule](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay)的实现，可以随时间系统性衰减
- 为输入数据做prefetching queues操作，将模型与磁盘延迟以及昂贵的预处理相隔离

我们也提供了该模型的[多GPU版本](https://www.tensorflow.org/tutorials/deep_cnn#training_a_model_using_multiple_gpu_cards)：

- 配置一个模型来跨多GPU显卡来并行训练
- 在多GPU间共享和更新变量

我们希望该教程，对于视觉任务提供一个入口来基于tensorflow构建更大的CNN。

## 1.3 模型架构

cifar-10教程中的模型是一个多层架构（multi-layer），它包含了交替的卷积层（convolutions ）和非线性层（nonlinearities）。这些层最后接在完全连接层（fully connected layers）上来生成一个softmax classifier。该模型遵循Alex Krizhevsky描述的架构，只前几层上略有不同。

该模型达到的最高效果是86%的accuracy，只需要在单GPU上训练几个小时。详见代码和[下面](https://www.tensorflow.org/tutorials/deep_cnn#evaluating_a_model)。它包含了1,068,298个可学习参数，需要19.5M个multiply-add 操作来在单个图片上进行inference计算。

# 二、代码组织

详见：[models/tutorials/image/cifar10/](https://www.tensorflow.org/code/tensorflow_models/tutorials/image/cifar10/)

- cifar10_input.py：读取原生的cifar-10二进制文件格式
- cifar10.py：构建cifar-10模型
- cifar10_train.py：在单cpu或gpu上训练一个cifar-10模型
- cifar10_multi_gpu_train.py：在多GPU上训练一个cifar-10模型
- cifar10_eval.py：评估cifar-10模型的预测效果

# 三、CIFAR-10模型

cifar-10网络大部分包含在cifar10.py中。完整的训练图（training graph）包含了765个op。我们发现，通过使用以下模块构建graph，我们可以让代码更复用：

- [Model inputs](https://www.tensorflow.org/tutorials/deep_cnn#model_inputs)：inputs()和distorted_inputs()会添加一些op：它们各自会读取和预处理cifar图片来进行评估、训练
- [Model prediction](https://www.tensorflow.org/tutorials/deep_cnn#model_prediction)：inference()会添加执行inference的op，例如：分类。
- [Model training](https://www.tensorflow.org/tutorials/deep_cnn#model_training)：loss()和train()会添加op：它们会计算loss，gradients，变量更新和可视化的summaries

## 3.1 模型输入

模型的输入部分通过函数inputs() 和 distorted_inputs() 进行构建，它们会从CIFAR-10 二进制文件中读取图片。这些文件包含了固定字节长度的records，因为我们可以使用tf.FixedLengthRecordReader。关于Reader类的工作机制可详见[Reading Data](https://www.tensorflow.org/api_guides/python/reading_data#reading-from-files)

这些图片会进行下列的处理：

- 被裁减成24x24像素，evaluation使用centrally方式，而training使用[randomly方式](https://www.tensorflow.org/api_docs/python/tf/random_crop)
- 他们会[近似变白：approximately whitened](https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization)，以便让模型对于动态范围不敏感

对于training过程，我们额外使用一系列随机扭曲来人工伪造增加数据集的size：

- 对图片从左到右进行[随机翻转](https://www.tensorflow.org/api_docs/python/tf/image/random_flip_left_right)
- 对[图片亮度](https://www.tensorflow.org/api_docs/python/tf/image/random_brightness)进行随机扭曲
- 对[图片对比度](https://www.tensorflow.org/api_docs/python/tf/image/random_contrast)进行随机扭曲

可以看到[Images](https://www.tensorflow.org/api_guides/python/image)页列出了许多可选的扭曲方式。我们也绑定了一个[tf.summary.image](https://www.tensorflow.org/api_docs/python/tf/summary/image) 到图片上，以便我们可以在tensorboard上进行可视化。这是个好的惯例来验证输入是否被正确构建。

<img src="https://www.tensorflow.org/images/cifar_image_summary.png">

从磁盘中读取文件，并对它们进行扭曲，可以使用一个不寻常的处理时间。为了防止这些操作减缓训练，我们在16个不同的线程中运行它们，它们会持续填满一个tensorflow[队列](https://www.tensorflow.org/api_docs/python/tf/train/shuffle_batch)。


## 3.2 模型预测

模型的预测部分通过inference()函数进行构建，它会添加op来计算预测的logits。模型的这部分如下进行组织：

- conv1: convolution and rectified linear activation.
- pool1: max pooling.
- norm1: local response normalization
- conv2: convolution and rectified linear activation.
- norm2: local response normalization.
- pool2: max pooling.
- local3: fully connected layer with rectified linear activation.
- local4: fully connected layer with rectified linear activation.
- softmax_linear: linear transformation to produce logits.

下面是由tensorboard生成关于inference操作的graph：

<img src="https://www.tensorflow.org/images/cifar_graph.png">

**练习：inference的输出是一个未归一化的logits。尝试编辑该网络结构，使用tf.nn.softmax，来返回归一化的predictions。**

inputs()和inference()函数提供了所有必要的组件来执行一个模型的evaluation。我们现在将焦点转移至构建op来训练一个模型。

**练习：在inference()的模型架构与在[cuda-convnet](https://code.google.com/p/cuda-convnet/)中指定的cifar-10模型有些许不同。特别的，Alex原始模型的顶层是locally connected而非fully connected。尝试编辑架构在顶层来准确生成个locally connected架构。**

## 3.3 模型训练

训练一个执行N-way分类的网络的常用方法是，通过softmax regression来使用[multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)。Softmax回归使用一个softmax非线性层到网络输出上，计算归一化预测与label index间的cross-entropy。对于正则项，我们使用常用的[weight decay](https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)到所有可学习变量上。该模型的目标函数是对所有cross entropy loss以及所有这些weight decay项进行求和，然后通过loss()函数返回。

我们在tensorboard中使用tf.summary.scalar进行可视化：

<img src="https://www.tensorflow.org/images/cifar_loss.png">

我们尝试使用标准的[梯度下降算法](https://en.wikipedia.org/wiki/Gradient_descent)来训练模型，其中learning rate随时间做指数衰减[exponentially decays](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay)

<img src="https://www.tensorflow.org/images/cifar_lr_decay.png">

train()函数会添加以下通过计算梯度和更新学习变量来最小化目标函数所需的op（详见：tf.train.GradientDescentOptimizer）。它会返回一个op：该op会为图片的一个batch执行所有训练和更新模型所需的计算。

# 四、启动和训练模型

我们构建完模型后，接着使用cifar10_train.py脚本来启动它，运行训练操作。

	python cifar10_train.py

**注意：在cifar-10教程中，当你第一次运行任何target时，cifar-10 dataset会被自动下载。dataset大约为~160MB。**

你可以看到输出为：

{% highlight python %}

Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
2015-11-04 11:45:45.927302: step 0, loss = 4.68 (2.0 examples/sec; 64.221 sec/batch)
2015-11-04 11:45:49.133065: step 10, loss = 4.66 (533.8 examples/sec; 0.240 sec/batch)
2015-11-04 11:45:51.397710: step 20, loss = 4.64 (597.4 examples/sec; 0.214 sec/batch)
2015-11-04 11:45:54.446850: step 30, loss = 4.62 (391.0 examples/sec; 0.327 sec/batch)
2015-11-04 11:45:57.152676: step 40, loss = 4.61 (430.2 examples/sec; 0.298 sec/batch)
2015-11-04 11:46:00.437717: step 50, loss = 4.59 (406.4 examples/sec; 0.315 sec/batch)
...

{% endhighlight %}

脚本会在每10个step后报告了total loss，以及被处理的最后一个batch的速率。一些注释：

- 第1个batch的数据可能非常慢（几分钟），随着预处理线程填满fullfing queue，有20000张处理好的cifar图片。
- 上报的loss是最近batch的平均loss。记住，该loss是对cross entropy和所有weight decay项的求和。
- 注意一个batch的处理速率。上述的数字显示了我们在tesla K40c上获取。如果我们运行在一个CPU上，可能会性能更慢

**练习：当实验时，有时很讨厌，第一个training step可能会花费更长时间。尝试减小初始填满队列的图片数目。可以在cifar10_input.py中搜索min_fraction_of_examples_in_queue**

cifar10_train.py会周期性地在[checkpoint files](https://www.tensorflow.org/programmers_guide/variables#saving-and-restoring)中保存所有模型参数，但它不会评估该模型。checkpoint文件会通过cifar10_eval.py来使用并衡量预测的效果。

如果你遵循之前的steps，接着你已经可以开始训练一个cifar-10模型了。

从cifar10_train.py中返回的终端文本提供了关于模型如何被训练的最小视角。我们希望更深入地了解训练期间的模型：

- loss是否已经真的减小，或者还是noise状态？
- 模型是否被提供了合适的图片？
- gradients, activations 和 weights是否合理？
- learning rate当前在哪？

tensorboard提供了上述功能，可以通过cifar10_train.py中的tf.summary.FileWriter周期性导出并展示。

例如，我们可以监视在训练期间，在local3特征上的activations分布以及稀疏程度：

<img src="https://www.tensorflow.org/images/cifar_sparsity.png">

<img src="https://www.tensorflow.org/images/cifar_activations.png">

单个loss function，以及total loss，也可以随时间进行跟踪。然而，loss展示了相当大量的noise，归因于在训练中使用过小的batch_size。实际上，我们发现，除了对它们的原始值外，它对于可视化移动平均相当有用。可以参见脚本是如何使用tf.train.ExponentialMovingAverage。

# 五、评估一个模型

现在，我们可以如何使用训练后的模型来评估一个hold-out数据集。该模型通过脚本cifar10_eval.py进行评估。它会使用inference()函数来构建模型，并使用在cifar-10的evaluation set中的所有10000张图片。它会计算precision@1: top预测与图片真实label间的差异。

为了监控在训练期模型是如何提升的，evaluation脚本会在由cifar10_train.py创建的最新checkpoint文件上周期性运行。

	python cifar10_eval.py

注意，不要在相同的GPU上同时运行评估和评练脚本，否则你将内存溢出（out of memory）。如果可以，考虑到在一个独立的GPU上运行evaluation，或者当在相同的GPU上运行evaluation时暂停训练。

你可以看到以下输出：

{% highlight python %}

2015-11-06 08:30:44.391206: precision @ 1 = 0.860
...

{% endhighlight %}

该脚本只会周期性返回precision@1 —— 在这种情况下，它会返回86%的accuracy。cifar10_eval.py也会导出summaries，它们可以被tensorboard可视化。这些summaries提供了模型在评估期的额外视角。

训练脚本会计算所有可学习变量的移动平均[moving average](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)。evaluation脚本会使用移动平均版本来替换所有学到的模型参数。这种替换会在评估期增强模型效果。

**练习：使用平均参数，在precision@1上可以增强大约3%的预测效果。编辑cifar10_eval.py，不再为模型使用平均参数，并验证预测效果下降（drops）**

# 五、使用多CPU显卡训练模型

现代的工作站为进行科学计算，可能包含了多个GPUs。tensorflow可以利用这些环境来跨多卡并行运行训练操作。

以并行、分布式方式训练一个模型，需要对训练进程进行协同(coordinating)。我们把**model replica**称作：在某个数据子集上训练的模型的一个copy。

使用模型参数的异步更新，会导致次优（sub-optimal）的训练效果，因为单个model replica可以在一个模型参数的旧拷贝（stale copy）上被训练。相反地，使用完全同步的更新，将导致与最慢的model replica一样慢。

在一个拥有多GPU卡的工作站上，每个GPU都有相似的速率，包含足够的内存来运行一整个CIFAR-10模型。然而，我们以如下的方式选择设计我们的训练系统：

- 在每个GPU上放置一个独立的model replica
- 等待所有GPU完成处理一个batch的数据后，再同步更新模型参数

这里是该模型的图：

<img src="https://www.tensorflow.org/images/Parallelism.png">

注意，每个GPU会为每个唯一的batch数据计算inference和gradients。该setup可以有效允许跨GPUs分割一个更大batch的数据。

该setup需要所有GPUs共享模型参数。一个己知的事实是，将数据转移入和转移出GPU是相当慢的。出于该原因，我们决定在CPU上保存和更新所有的模型参数（见绿框）。当一个新batch的数据被所有GPU处理后，模型参数的一个新集合会被转移到GPU。

GPUs在操作上是同步的。所有gradients会从GPUs被累积，并进行平均（见绿框）。模型参数会使用跨所有model replicas的梯度平均进行更新。

## 在设备上放置变量和op

在device上放置ops和variables需要一些特别的抽象。

我们需要的第一个抽象是，一个用于为单个model replica计算inference和gradients的function。在代码中，我们将该抽象称为一个“tower”。我们必须为每个tower设置两个属性：

- 在一个tower中为所有操作提供一个唯一的名字。tf.name_scope可以提供唯一名字。例如，所有在第一个tower中的所有op会使用tower_0 ，例如：tower_0/conv1/Conv2D
- 在一个tower内使用一个首选的硬件设备来运行op。可以通过tf.device指定。例如，所有在第一个tower中的ops会位于device('/device:GPU:0')的范围内，表示他们应运行在第一个GPU上。

所有变量都被限制在该CPU上，为了在一个多GPU版本中共享它们，可以通过tf.get_variable来访问。

## 在多GPU卡上启动和训练模型

如果你的机器上安装了多个GPU卡，你可以使用cifar10_multi_gpu_train.py来更快地训练模型。该版本的训练脚本会跨多GPU卡对模型并行化。

	python cifar10_multi_gpu_train.py --num_gpus=2

注意：GPU卡的数目缺省为1 。另外，如果只提供了1个GPU，所有计算都会放置在它上面，即使你想要更多GPU。

**练习：cifar10_train.py的缺省设置是运行的batch_size=128. 可以在2 GPU上运行cifar10_multi_gpu_train.py，每个batch_size=64, 然后比较训练速度**

# 6.下一步

**练习：下载[街景门牌号Street View House Numbers (SVHN) ](http://ufldl.stanford.edu/housenumbers/)**数据集。修改该教程的脚本进行预测。


# 参考

[tensorflow deep_cnn](https://www.tensorflow.org/tutorials/deep_cnn)
