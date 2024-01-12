---
layout: post
title: Cross Attention介绍
description: 
modified: 2023-11-13
tags: 
---

Vaclav Kosar在《Cross-Attention in Transformer Architecture》这篇文章里提出了一种cross attention方法。其实在很多地方有在用。

# 介绍

交叉注意力（Cross attention）是：

- 一种在Transformer架构中的attention机制，可以将两种不同embedding序列进行混合
- 这两个序列必须具有相同的维度
- 这两个序列可以是不同的模态（例如文本、图像、声音）
- 其中一个序列作为Query输入，定义了输出长度。另一个序列则产生Key和Value输入，用于attention计算

交叉注意力机制使得模型能够关注来自两个序列的相关信息，这在图像字幕或多模式机器翻译等任务中非常有用。

# Cross-attention应用

- [image-text classification with Perceiver](https://vaclavkosar.com/ml/Multimodal-Image-Text-Classification)
- [机器翻译：cross attention帮助解码器去预测被翻译文本的下一个token](https://vaclavkosar.com/ml/cross-attention-in-transformer-architecture)

# Cross-attention vs Self-attention

除了输入之外，cross attention的计算方式与self-attention相同。cross attention以不对称的方式组合了**两个相同维度的独立embedding序列**，而self-attention的输入是单个embedding序列。其中一个序列作为query输入，而另一个序列作为key和value输入。在SelfDoc中的一种cross attention可选方式是：使用来自一个序列的query和value，而key则来自另一个序列。

前馈层（feed forward layer）与cross-attention相关，不同之处是：前馈层会使用softmax，并且其中一个输入序列是静态的。《[Augmenting Self-attention with Persistent Memory paper]{https://vaclavkosar.com/ml/Feed-Forward-Self-Attendion-Key-Value-Memory}》一文表明，前馈层的计算方式与self-attention相同。


<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/29edc8db5a5c21e525b1c59db26db38343f9cdbd1aacee1346eb1ace1d9c255a9a2e60ddbc17a17d8bc7f710decd9750?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=1.jpg&amp;size=750">

图1

# Cross-attention算法

- 假设我们有两个embeddings（token）序列S1和S2
- 从序列S1中计算键（Key）和值（Value）
- 从序列S2中计算查询（Queries）
- 使用Key和Query来计算注意力矩阵（Attention Matrix）
- 将queries应用于注意力矩阵
- 输出序列具有与序列S2相同的维度和长度

在一个等式中：

$$
softmax((W_Q S_2)(W_K S_1)^T)W_V S_1
$$

# Cross-attention可选方式

Feature-wise Linear Modulation Layer是一个更简单的可选方式，它不要求：输入必须是个序列，并且是线性计算复杂度的。这可以使用稳定扩散（Stable Diffusion）生成图像。在这种情况下，交叉注意力用于使用文本提示为图像生成器中的UNet层中的变压器进行条件编码。构造函数显示了我们如何使用不同的维度，并且如果您使用调试器逐步执行代码，还可以看到两种模态之间的不同序列长度。

# Cross-attention实现

在Diffusers library中的[cross attention实现](https://github.com/huggingface/diffusers/blob/4125756e88e82370c197fecf28e9f0b4d7eee6c3/src/diffusers/models/cross_attention.py)可以使用Stable Diffusion生成图像。在这个case中，cross-attention被用于【使用文本prompt为图像生成器中的UNet层中的condition transformers】。构造函数显示了我们如何使用不同的维度，并且如果您使用调试器逐步执行代码，还可以看到两种模态之间的不同序列长度。

{% highlight python %}
class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """
{% endhighlight %}

特别是在这部分中，您可以看到查询（query）、键（key）和值（value）是如何相互作用的。这是编码器-解码器架构，因此query是从encoder的hidden states中创建得到的。

{% highlight python %}
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
{% endhighlight %}

# 流行结构中的cross-attention

## Transformer Decoder中的cross-attention

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/f2330b55f6c40c681672c24a547021b6fccf2abe7e793539a33b36ee0584842687290bdd58cf64afdfeadd050a0b80ed?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=2.png&amp;size=750">

## Stable Diffusion中的cross-attenion

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/2b60e8e67c4445e3cdc654fe6166f16182a993b90d8181d8c3e48e1d391cd04e99564f0e865d8ed03c4c390134f02592?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=3.png&amp;size=750">

## Perceiver IO中的Cross-Attention

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/77e0aba25c3503754510325f62ecd4252247e33798154ff518e63d6c34f0d698f622c30222e3846cfebfab60183edd61?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=4.png&amp;size=750">

## SelfDoc中的Cross-Attention

<img alt="图片名称" src="https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/db917067846c9ce9d7a63bbafb17e9e67c38e6fec56c6fbd2e6ce3ed201110da1175b3ff2a47ab3599f294e472fe7f6f?pictype=scale&amp;from=30113&amp;version=3.3.3.3&amp;fname=5.png&amp;size=750">


- [https://vaclavkosar.com/ml/cross-attention-in-transformer-architecture](https://vaclavkosar.com/ml/cross-attention-in-transformer-architecture)