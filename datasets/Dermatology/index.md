---
layout: page
title         : 皮肤学数据集(Dermatology Dataset)
author        : 译者：d0evi1 2016.1.10
---
{% include JB/setup %}

# 数据来源于：

Original Owners: 

1. 

Nilsel Ilter, M.D., Ph.D., 

Gazi University, 

School of Medicine 

06510 Ankara, Turkey 

Phone: +90 (312) 214 1080 

2.

H. Altay Guvenir, PhD., 

Bilkent University, 

Department of Computer Engineering and Information Science, 

06533 Ankara, Turkey 

Phone: +90 (312) 266 4133 

Email: guvenir '@' cs.bilkent.edu.tr 

Donor: 

H. Altay Guvenir, 

Bilkent University, 

Department of Computer Engineering and Information Science, 

06533 Ankara, Turkey 

Phone: +90 (312) 266 4133 

Email: guvenir '@' cs.bilkent.edu.tr

# 数据集介绍

该数据集包含了34个属性，其中33个通过线性值(linear valued)表示，还有一个标称（nominal）。

在皮肤学中，皮肤病：[erythemato-squamous病](https://en.wikipedia.org/wiki/Erythema)，
它的鉴别诊断是个头疼的问题。它们在erythema和scaling特征上有着相同的临床特征，只有很
少的区别。这一族群的疾病有：银屑病（psoriasis）、脂溢性皮炎(seboreic dermatitis)、lichen planus、
pityriasis rosea, cronic dermatitis, and pityriasis rubra pilaris。通常对于这类疾病，活体组织
检查(biopsy)是必不可少的，但不幸的是，这类疾病有着相近的病理学特征。对于鉴别诊断来说，另一个难点是，
一种疾病上出现另一种疾病在开始病发阶段出现的特征，在接下来的阶段才会有不同的特征。病人首先会在临床上
评估12项特征。之后，皮肤样本会进行其它22项组织病理学特征的评估。然后再通过在显微镜下，这些特征的评估值，
来分析决定。

该领域的数据集中，家族历史（family history）特征如果任何这样的特征被观察到了，那么，该特征值为1,
否则为0. 年龄（age）特征只表示病人的年纪。其它特征（如：clinical 和 histopathological）则给定了一个
范围值（0-3）。这里，0表示没有出现该特征，3表示最大量。1和2则表示一个相对值。

病人的names和id number最近从该数据集中被移除。

# 属性介绍

## 临床属性：(范围值取：0, 1, 2, 3，其它取0,1)

1: erythema 

2: scaling

3: definite borders

4: itching

5: koebner phenomenon

6: polygonal papules

7: follicular papules

8: oral mucosal involvement 

9: knee and elbow involvement 

10: scalp involvement 

11: family history, (0 or 1) 

34: Age (linear) 


##组织病理学特征：（取：0, 1, 2, 3）

12: melanin incontinence 

13: eosinophils in the infiltrate 

14: PNL infiltrate 

15: fibrosis of the papillary dermis 

16: exocytosis 

17: acanthosis 

18: hyperkeratosis 

19: parakeratosis 

20: clubbing of the rete ridges 

21: elongation of the rete ridges 

22: thinning of the suprapapillary epidermis 

23: spongiform pustule 

24: munro microabcess 

25: focal hypergranulosis 

26: disappearance of the granular layer 

27: vacuolisation and damage of basal layer 

28: spongiosis 

29: saw-tooth appearance of retes 

30: follicular horn plug 

31: perifollicular parakeratosis 

32: inflammatory monoluclear inflitrate 

33: band-like infiltrate


Enjoy!

[官方原址](http://archive.ics.uci.edu/ml/datasets/Dermatology)
