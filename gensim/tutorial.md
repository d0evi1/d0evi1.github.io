---
layout: page
title: 教程 
---
{% include JB/setup %}

教程的组织由一系列示例组成，包含一系列gensim的特性.

示例分成以下几部分：

- [语料和矢量空间]()
    - String转成向量
    - 语料流
    - 语料格式
    - 兼容NumPy和SciPy
- 主题和变换
    - 变换接口
    - 提供的转换
- 相似查询
    - 相似接口
    - next?
- 英文Wikipedia示例
    - 准备语料
    - LSA
    - LDA
- 分布式计算
    - 为什么要分布式计算?
    - 先决条件
    - 核心概念
    - 提供的分布式算法 

## 先决条件

所有的示例都可以通过Python解析器shell直接copy运行.  IPython的cpaste命令可以拷贝代码段，包含前导的>>>字符.

gensim使用Python的标准logging模块来记录不同优先级的日志；如果想激活日志，运行：

    >>> import logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
