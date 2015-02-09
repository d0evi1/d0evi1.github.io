---
layout: page
title: models.lsi_worker -  分布式LSI的worker
---
{% include JB/setup %}

USAGE: %(program)s SIZE_OF_JOBS_QUEUE

    Dispatcher进程，用来分派分布式LSI计算。在你的集群上的每个节点都运行该脚本一次。

示例：python -m gensim.models.lsi_dispatcher

--------------------------------------------------------------

class gensim.models.lsi_dispatcher.Dispatcher(maxsize=0)

    dispatcher对象，用于和独立的worker进行进行通信。

    在任何时间，只能有一个dispatcher运行。

    注意：dispatcher的构造函数不能进行完全初始化，可以再使用initialize()函数。

--------------------------------------------------------------

exit()
    
    终止所有注册的worker，然后终止dispatcher。

--------------------------------------------------------------

getstate()

    在所有的worker间合并projection，并返回最后的projection.
    
--------------------------------------------------------------

getworkers()

    返回所有注册worker的pyro URI.

--------------------------------------------------------------

initialize(**model_params)

    model_params：该参数用于初始化各独立的 workers (worker.intialize())

--------------------------------------------------------------

jobdone(*args, **kwargs)

    一个worker已经完成了它的job。记录事件，接着异步传输给worker。

    这种方式下，控制流基本上在dispatcher.jobdone() 和 worker.requestjob()之间摆动。

--------------------------------------------------------------

jobsdone()

    封装了 self._jobsdone，通过代理进行远程访问。

--------------------------------------------------------------
reset()
    
    为新的分解初始化所有的workers。


[英文原版](http://radimrehurek.com/gensim/models/lsi_dispatcher.html)




