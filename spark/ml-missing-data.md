---
layout: page
title: ml missing values 
tagline: 介绍
---
{% include JB/setup %}

spark mllib本身没有提供missing values的处理机制。

常用的两种处理missing values的方法有：

- 1.drop掉对应missing values的数据点（不推荐使用）
- 2.对于数值型的特征，使用均值（median）进行填充；而对于类别型特征，使用最常出现的类别进行填充。

第一种方法不会使用所有的数据集，第2种方法则对有许多gap的数据集使用过宽的标准。

其它方法：使用机器学习方法进行预测等。

spark mllib推荐使用DataFrame处理数据。对应的：

1.drop方法

{% highlight scala %}

// method 1
var train_removed = train.na.drop()

// numerical data
var avgAge = train.select(mean("Age")).first()(0).asInstanceOf[Double]
var train_fill_num = train.na.fill(avgAge, Seq("Age"))

// categorical data, most frequence: A
var train_fill_cat = train_data.na.fill("A", Seq("Rank"))

{% endhighlight %}