---
layout: post
title: 一种简单的个性化导航实现
description: 
modified: 2015-08-25
tags: [个性化 机器学习]
---

{% include JB/setup %}

移动端时代的挑战：手机屏更小，输入更不便，信息过载问题更严重。

用户获取信息的方式：浏览 vs. 查询

点击距离(click-distance):

click-distance(i) = selects(i) + scrolls(i)   i为item的意思。

## 1 个性化用户兴趣

两种点击：

- static hit-table:大众的点击数据，one-size-fits-all
- user hit-table:个人的点击数据

其中static hit-table如下：

<figure>
	<a href="http://pic.yupoo.com/wangdren23/FCBSudVA/medish.jpg"><img src="http://pic.yupoo.com/wangdren23/FCBSudVA/medish.jpg" alt=""></a>
</figure>

某一个用户的hit-table如下：

<figure>
	<a href="http://pic.yupoo.com/wangdren23/FCBSJ0Bt/medish.jpg"><img src="http://pic.yupoo.com/wangdren23/FCBSJ0Bt/medish.jpg" alt=""></a>
</figure>

然后根据此计算这个用户对每个item的喜好概率. 概率计算：

- \$ P(B|A)=(20+10)/(40+100)=0.214 \$
- \$ P(C|A)=(20+90)/(40+100)=0.786 \$
- \$ P(D|A)=P(B|A)P(D|B)=(30/140)(10+5)/(20+10) = 0.107 \$
- \$ P(E|A)=P(B|A)P(E|B)=(30/140)(10+5)/(20+10) = 0.107 \$
- \$ P(F|A)=P(C|A)P(F|C)=(110/140)(10+80)/(20+90)=0.642 \$
-\$ P(G|A)=P(C|A)P(G|C)=(110/140)(10+10)/(20+90)=0.142 \$

该用户的喜好排序为：C>F>B>G>D>E

## 2 个性化调整

ok，计算好了之后。需要对每个用户做menu的调整。调整方式采用的是：垂直提升（vertical promotions）。举个例子，原先如果是三层：根菜单－父菜单-菜单选项。菜单选项提升到父菜单级别，父菜单提升到根菜单级别。别外同级之间的相对位置也会进行调整。

## 3 指标评测

- 平均点击距离（是否降低）
- 平均每个session的平均导航时间（是否降低）
- 平均内容浏览时间（是否提升）


参考：

1.personalization techniques and recommender systems, Gulden Uchyigit etc.

