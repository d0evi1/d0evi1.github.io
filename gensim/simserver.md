---
layout: page
title: 文档相似服务器 
---
{% include JB/setup %}


gensim 0.7.x系列版本正在改进性能和API. 0.8.x将会有新特性：文档相似服务器.

该项目的源码从gensim中独立出来，叫simserver，你可以在github上进行clone。

# 1.什么是文档相似服务器?

概念上，该服务的作用如下：

- 1.从plain text的语料库中进行语义模型训练（无注解和mark-up标记等）
- 2.使用语义模型建立索引
- 3.对相似文档进行索引查询（查询可以是一个索引中存在的id，或者一个文本）

代码：

    >>> from simserver import SessionServer
    >>> server = SessionServer('/tmp/my_server') # resume server (or create a new one)

    >>> server.train(training_corpus, method='lsi') # create a semantic model
    >>> server.index(some_documents) # convert plain text to semantic representation and index it
    >>> server.find_similar(query) # convert query to semantic representation and compare against index
    >>> ...
    >>> server.index(more_documents) # add to index: incremental indexing works
    >>> server.find_similar(query)
    >>> ...
    >>> server.delete(ids_to_delete) # incremental deleting also works
    >>> server.find_similar(query)
    >>> ...

注意：

这里的"语义"涉及到天然语义－－LSA, LDA等。无需做语义网，人工资料标签、或者详细语义推测。

# 2.它擅长什么？

文本文档的数字库。更进一步说，它可以以一种更抽象的方式帮你标注，组织和索引文档，通过比较关键词查询来完成。

# 3.它的独特之处？

- 1.内存独立. gensim有唯一性算法来进行统计分析，对比于RAM，允许你很快为大训练集语料（比RAM还大）创建专门的语义模型。
- 2.内存独立(还是). 存储成文件的共享索引可以根据需要还原到磁盘/mmap。因此，你可以检索大语料。对比RAM，索引文档的数目独立。
- 3.高效。gensim使用python的NumPy和SciPy库来有效进建立索引和查询。
- 4.健壮。索引的修改是事务型的，你可以commit/rollback整个索引session。也就是说，在一个session期间，服务对于查询来说仍有效（当session启动时会使用它的状态）。断电时服务会停留在一个恒定的状态（隐式的rollback）。
- 5.纯python。技术上，NumPy和SciPy通过C和Fortran进行封装，但是gensim本身是纯Python实现的。无需编译，只需root权限即可安装。
- 6.并发支持。底层的服务对象是线程安全的，因此可以做为daemon-server：clients通过RPC连接上它，然后在远端进行训练/索引/查询。
- 7.跨网络，跨平台和跨语言。当python服务器使用pyro的TCP运行时，客户端可以通过java/.Net的Pyrolite进行连接。

文档的其余部分会更进一步解释特性。

# 4.先决条件

假设你已经安装了gensim。你需要sqlitedict包，它封装了线程安全的sqlite3模块。

    $ sudo easy_install -U sqlitedict

为了测试远程服务器的功能，安装Pyro4（Python Remote Objects, 版本>=4.8）:

    $ sudo easy_install Pyro4

注意：

不要忘记初始化日志消息：
    
    >>> import logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 5.文档？

如果是文本文档，那么期待的格式为：

    >>> document = {'id': 'some_unique_string',
    >>>             'tokens': ['content', 'of', 'the', 'document', '...'],
    >>>             'other_fields_are_allowed_but_ignored': None}

选择该格式是因为，它符合JSON，因此先容易以任何语言进行序列化和传输。所有的字符串都必须是utf8编码。

# 6.语料？

一个文档序列。语料中的任何文档：... 可以进行迭代/Generator来完成。这里使用普通的list（会消耗更多内存）。

    >>> from gensim import utils
    >>> texts = ["Human machine interface for lab abc computer applications",
    >>>          "A survey of user opinion of computer system response time",
    >>>          "The EPS user interface management system",
    >>>          "System and human system engineering testing of EPS",
    >>>          "Relation of user perceived response time to error measurement",
    >>>          "The generation of random binary unordered trees",
    >>>          "The intersection graph of paths in trees",
    >>>          "Graph minors IV Widths of trees and well quasi ordering",
    >>>          "Graph minors A survey"]
    >>> corpus = [{'id': 'doc_%i' % num, 'tokens': utils.simple_preprocess(text)}
    >>>           for num, text in enumerate(texts)]
    
由于corpora可以更大，因此推荐的方式：让客户端将它们分成几个chunk，然后将它们上传给服务器：

    >>> utils.upload_chunked(server, corpus, chunksize=1000) # send 1k docs at a time

# 7.上传什么，上传到哪？

如果你在代码中直接使用相似服务对象（simserver.SessionServer实例）－－非远程访问－－运行良好。如果使用远程服务，从一个不同的进程/机器是没有必要的。

文档相似可以做为一个长期运行的服务存在，每台机器都有一个daemon进程。在本例中，它可以调用一个服务对象server。

我们可以先本地运行。打开shell：

    >>> from gensim import utils
    >>> from simserver import SessionServer
    >>> service = SessionServer('/tmp/my_server/') # or wherever

初始化一个新的服务，位置放在/tmp/my_server（你需要有写权限）

注意：

服务通过它的本地位置来进行完全定义。如果你使用一个已经存在的位置，服务对象将会从相应的index处进行resume。也就是说，clone一个服务，只需copy它的目录即可。copy操作将完整地复制一个server出来。

# 8.模型训练

我们可以进行索引试试：
    
    >>> service.index(corpus)
    AttributeError: must initialize model for /tmp/my_server/b before indexing documents

嗯，好像不行。服务必须以语义表示的方式进行文档检索，这与你给的文本不一样。我们必须首先让service知道如何将普通文本与语义模型进行转换：

    >>> service.train(corpus, method='lsi')

很简单。method='lsi' 参数意味着，我们使用LSI进行模型训练，缺省通过tf-idf 400维来表示我们的小语料。更多信息后续会讲到。

注意，为了让语义模型比较ok，你必须在语料训练时注意：

- 你希望在检索时，出现合理的文档相似。在法语语料上进行训练，而当检索英语文档时不会有任何用。
- 够大（至少成千上万的文档），以便静态分布可以。不要在生产环境中使用类似我们的示例语料（只9个。。）

# 9.检索文档

    >>> service.index(corpus) # index the same documents that we trained on...

检索可以通过任何文档进行，此处仍使用9文档的语料。。

可以进行删除文档：

    >>> service.delete(['doc_5', 'doc_8']) # supply a list of document ids to be removed from the index

当你传文档时，必须在建好索引的文档中要有相似的id，被检索的文档通过新的输入完全重写（=最新的数目，每个service的文档id都是唯一）
    
    >>> service.index(corpus[:3]) # overall index size unchanged (just 3 docs overwritten)

索引/删除/重写，可以和查询穿插进行。你不必首先建立好所有文档的索引，然后启动查询，可以增量检索。

# 10.查询

有两种类型的查询：

1.通过id:

    >>> print(service.find_similar('doc_0'))
    [('doc_0', 1.0, None), ('doc_2', 0.30426699, None), ('doc_1', 0.25648531, None), ('doc_3', 0.25480536, None)]

结果为3-tuples，doc_n是在检索时提供的一个文档id，0.30426699则与doc_n查询的相似度，最后一个None指的是，你可以在检索时，为每个文档附加一个"payload"。这个payload对象（可以是任何东西）会在查询后返回。如果你不想在检索期间指定任务doc['payload']，那么可以通过指定None来完成。

2.或者通过文档（使用 document['tokens']；这里id将被忽略）:

    >>> doc = {'tokens': utils.simple_preprocess('Graph and minors and humans and trees.')}
    >>> print(service.find_similar(doc, min_score=0.4, max_results=50))
    [('doc_7', 0.93350589, None), ('doc_3', 0.42718196, None)]

# 11.远程访问

至今为止，我们做的操作都在本地python shell上进行。我很喜欢Pyro，一个RPC纯python包，我们可以通过Pyro进行远程访问。Pyro会负责所有的socket监听/请求，路由/数据编解码/线程分派，它会节省很多麻烦。

为了创建一个相似的server，我们只需要创建一个simserver.SessionServer对象，并且使用Pyro daemon来为远程访问进行注册。有一个示例脚本，包含了simserver，可以运行它：

    $ python -m simserver.run_simserver /tmp/testserver

你可以使用ctrl+c来终止进程。

现在再次打开你的python shell，在另一个终端，或者另一个机器上：

    >>> import Pyro4
    >>> service = Pyro4.Proxy(Pyro4.locateNS().lookup('gensim.testserver'))

现在，service是一个代理对象：当你运行run_server.py脚本时，物理执行每个调用，它可以是完全不同的计算机（在另一个网络广播域中），你无需知道：

    >>> print(service.status())
    >>> service.train(corpus)
    >>> service.index(other_corpus)
    >>> service.find_similar(query)
    >>> ...

需要提一下的是，lrmen，pyro的作者，最近又发布了pyrolite。这个包允许你通过Java/.NET来创建pyro代码。

# 12.并发

这里越来越有趣了。因为我们开始远程访问service时，多个client同时创建proxy会发生什么？如果它们同时修改index会发生什么？

答案是：SessionServer是线程安全的，因此，当你的所有client通过Pyro派生请求线程时，不会发生混乱。

这意味着：

- 1.可以同时发生service.find_similar查询（总而言之，多个同时调用是"read-only"的）
- 2.当两个client同时修改时（index/train/delete/drop_index/...），会有内部锁进行串行保证。
- 3.当一个client修改index时，所有其它的client的查询仍会看到原始的index。只有当这些修改commit之后，它们才可见。

# 13.你认为的可见是什么？

service内部使用事务。这意味着每个修改都通过service的clone来完成。如果出于某种原因，session的修改失败（代码异常，断电，关闭服务器，当session到来时client处理不过来），它都将rollback。这也意味着，其它的client可以在index更新期间继续查询原始的index。

该机制对用户而言是被隐藏的，缺省时自动提交auto-committing（上面的示例就是），但是自动提交也可以显式关闭：

    >>> service.set_autosession(False)
    >>> service.train(corpus)
    RuntimeError: must open a session before modifying SessionServer
    >>> service.open_session()
    >>> service.train(corpus)
    >>> service.index(corpus)
    >>> service.delete(doc_ids)
    >>> ...

任何修改都对其它client可见。也就是说，其它client调用index/train/etc将被阻塞，直到session被commit/rollback－－它不能同时是两个开放的session。

为了结束一个session:

    >>> service.rollback() # discard all changes since open_session()

或者：

    >>> service.commit() # make changes public; now other clients can see changes/acquire the modification lock

# 其它东西

TODO: 文档定制解析 (utils.simple_preprocess). 不同的模式（非lsi）。索引优化service.optimize()

TODO: 增加一些更大的数字；等。

[英文原版](http://radimrehurek.com/gensim/simserver.html)

    

