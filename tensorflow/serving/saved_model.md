---
layout: page
title:  tensorflow serving一个模型
tagline: 
---
{% include JB/setup %}

该文档描述了SavedModel，这是Tensorflow模型的通用序列化格式。

SavedModel提供了一个语言中立的格式来保存机器学习模型，它可恢复且对外界不透明。它允许高层系统和工具来生成、消费、和转换tensorflow模型。

# 1.特性

以下是SavedModel的特性总结 ：

- 共享单个variables和assets集合的多个graph，可以被添加到单个SavedModel中。每个graph与一个特定的tags集合相关，允许在一次加载和恢复操作期进行验证。
- 支持SignatureDefs
	- 被用于inference任务的Graphs，通常具有一个inputs/outputs集合。称为一个Signature。
	- SavedModel使用SignatureDefs来广泛支持那些需要被保存的signatures。
	- 对于Tensorflow Serving中公共使用的SignatureDefs，详见[这里](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/signature_defs.md)
- 支持Assets
	- 对于那些初始化时依赖于外部文件的ops，比如：vocabularies, SavedModel通过assets来支持它
	- Assets被拷贝到SavedModel位置上，当加载一个特定meta graph def时可以被读取。
- 在生成SavedModel之前，支持清除（clear）设备。

以下是那些在SavedModel中不支持的特性的一个总结。使用SavedModel的高层框架和工具提供了这些。

- 隐式版本
- 垃圾回收
- 原子写操作到SavedModel location上

# 2.背景

SavedModel是已经存在的tensorflow原语(比如Saver和MetaGraphDef）之上进行管理和构建的。特别的，SavedModel封装了一个Saver。Saver原先被用于生成变量的checkpoints。SavedModel将已经存在的标准[tensorflow inference模型格式](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/session_bundle/README.md)替换成导出tensorflow graphs来进行serving。

# 3.组件

一个SavedModel目录具有以下的结构：

	assets/
	assets.extra/
	variables/
	    variables.data-?????-of-?????
	    variables.index
	saved_model.pb

- SavedModel protocol buffer
	- saved_model.pb 或者 saved_model.pbtxt
	- 包含了graph定义
- Assets
	- 子目录称为assets
	- 包含了辅助文件：比如vocabularies
- extra assets
	- 是一个子目录，其中高层库和用户可以添加他们自己的assets，与模型一起共存，但不会被graph加载.	
	- 该子目录不能通过SavedModel库进行管理
- Variables
	- 称为variables的子目录
	- 包含了Saver的输出
		- variables.data-?????-of-?????
		- variables.index

# 4.APIs

APIs用于构建和加载一个SavedModel。

## 4.1 Builder

在Python中实现了SavedModel Builder.

SavedModelBuilder提供了保存多个meta graph defs、相关variables和assets的功能。

为了构建一个SavedModel，第一个meta graph必须与variables一起保存。接下来的meta graph会简单地与其它graph defs一起保存。如果assets需要保存和写入或者拷贝至磁盘上，当meta graph def被添加时它们必须被提供。如果多个meta graph defs与相同名字的一个asset相关，只有第一个version会被保留。

## 4.2 Tags

每个添加到SavedModel的meta graph必须使用用户指定的tags进行注解。tags提供了一种方式来标识指定的要加载和恢复的meta graph，与共享的variables和assets集合一起。这些tags通常使用它的功能（比如：serving或traing）、以及可能的硬件（比如：GPU）来注解一个MetaGraph。

## 4.3 用法

builder的常见用法如下：

{% highlight python %}

export_dir = ...
...
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       [tf.saved_model.tag_constants.TRAINING],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets)
...
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph(["bar-tag", "baz-tag"])
...
builder.save()

 
{% endhighlight %} 

## 4.4 缺省值属性

当添加一个meta graph到SavedModel bundle中时，SavedModelBuilder类允许用户来控制缺省值属性是否必须从NodeDefs中脱离。SavedModelBuilder.add_meta_graph_and_variables和SavedModelBuilder.add_meta_graph方法允许一个Boolean标识strip_default_attrs来控制该行为。

如果strip_default_attrs为False，导出的MetaGraphDef在所有NodeDef实例中必须具有缺省值属性。这会破坏向前兼容性：

- 一个已经存在的Op(Foo)被更新成包含一个新的属性（T）和一个缺省（Bool），版本号101.
- 一个模型生产者（比如一个Trainer）二进制会选取该变化（version 101）来OpDef，重新导出一个使用Op(Foo)的已经存在的模型。
- 一个模型消费者（比如Tensorflow Serving）运行一个老的二进制（version 100）不会具有属性T，但会尝试导入该模型。该模型消费者不会识别在NodeDef中的属性T，因此会在加载该模型时失败。

通过设置strip_default_attrs=True，模型生产者可以剥离在NodeDefs中的任何缺省值属性。这可以帮助确保新加入的属性值不会造成老的模型消费者使用新的训练二进行进行加载模型时失败。

注意：如果你关心向前兼容性，可以将strip_default_attrs=True。

## 4.5 Loader

SavedModel loader通过C++和Python来实现。

### Python

Python版本的SavedModel loader提供了load和restore功能。load操作需要session来恢复graph的def和variables，tags用于标识meta graph def。在一次load上，variables和assets的子集作为特定meta graph def的一部分进行支持，可以在session中被恢复。

{% highlight python %}

export_dir = ...
...
with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)
  ...

{% endhighlight %}

### C++

SavedModel loader 的C++版本提供了一个API来从一个path上加载一个SavedModel，允许SessionOptions和RunOptions。与Python版本相类似，C++版本需要指定与加载的graph相关的tags。SavedModel的加载版本被称为：SavedModelBundle，它包含了meta graph def和session。

{% highlight C++ %}

const string export_dir = ...
SavedModelBundle bundle;
...
LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagTrain},
               &bundle);

{% endhighlight %}

## 4.6 Constants

SavedModel提供了很多灵活性来构建和加载多种用例下的Tensorflow graphs。对于大多数情况，SavedModel's APIs提供了在Python和C++中的一个常数集合，很容易复用和跨工具持久共享。

### Tag constants

tags集合可以被用于唯一标识一个在SavedModel保存的MetaGraphDef。常用tags的一个子集：

- [python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/tag_constants.py)
- [c++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/tag_constants.h)

### Signature constants

SignatureDefs被用于定义一个计算（computation）的signature. 通常用于input keys，output keys以及method names：

- [python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py)
- [c++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/signature_constants.h)

# 参考
[SaveModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md)
