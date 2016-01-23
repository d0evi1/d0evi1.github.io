---
layout: page
title         : movielens数据集(movielens Dataset)
author        : 译者：d0evi1 2016.1.10
---
{% include JB/setup %}

# 简介

###ml-data.tar.gz      

    压缩文件. 用于重新u数据文件: gunzip ml-data.tar.gz; tar xvf ml-data.tar; mku.sh


###u.data:
    
    完整的u数据集，10w条ratings打分数据：943个user在1682个item上。每个user至少为20个电影打分。users和items从1开始，数字连续。数据是随机排序的。通过tab进行分割，如：

    user_id | item_id | rating | timestamp。

    timestamps是至1/1/1970 UTC以来的unix秒.


###u.info:
    
    在u.data数据集中user、item、rating的总数.


###u.item:

    items(电影)的信息；通过tab进行分割：

    movie_id | movie_title | release_date | video_release_date |

    IMDb URL | unknown | Action | Adventure | Animation |

    Children's | Comedy | Crime | Documentary | Drama | 

    Fantasy | Film-Noir | Horror | Musical | Mystery | 

    Romance | Sci-Fi | Thriller | War | Western |

    最后19个字段都是类型(genres)，值为1,则表示电影为该类型，为0则表示不是该类型；一部电影可以有多个类型。电影id和u.data数据集中的一致.


###u.genre     
    
    类型列表


###u.user      

    人口统计学(Demographic)信息。通过tab列表进行分割。

            user id | age | gender | occupation | zip code

            用户id与u.data数据集中一致.


###u.occupation
    职业信息.


###u1.base/u1.test ...  
    
    该数据集, u1.base和u1.test到u5.base/u5.test通过80%/20%进行分割成：训练集和测试集。每个u1, ..., u5都与测试集相互独立；5组可以进行交叉难（你可以重复试验每个训练集和测试集，对结果求平均）。该数据集通过mku.sh对u.data进行切分生成。


###ua.base/ua.test      

    该数据集，通过将u.data进行切分获得：在测试集中，每个用户都有10个打分(rating)。ua.test和ub.test是不相交的。该数据集也通过mku.sh生成。


###allbut.pl            

    该脚本用来生成测试集和训练集，但在训练集中，只有n个user打分。 


###mku.sh               

    该脚本用来生成所有的u文件数据集.


[官方原址](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt)
