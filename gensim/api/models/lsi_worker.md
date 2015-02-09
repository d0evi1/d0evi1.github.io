---
layout: page
title: models.lsi_worker -  分布式LSI的worker
---
{% include JB/setup %}

用法：%(program)s

    Worker("slave")进程，用于计算分布式LSI上使用。需要在你的集群上运行该脚本。如果利用机器的多核特性（注意：内存footprint会依次递增），你也可以在单机上运行多次。

示例：python -m gensim.models.lsi_worker

[英文原版](http://radimrehurek.com/gensim/models/lsi_worker.html)
