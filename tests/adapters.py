from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    from notebook.linear import Linear
    linear=Linear(out_features=d_out,in_features=d_in)
    linear.load_state_dict({"weight":weights})
    out_features=linear.forward(in_features)
    return out_features


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    from notebook.embedding import Embedding
    
    emb=Embedding(num_embedding=vocab_size,embedding_dim=d_model)
    emb.load_state_dict({'weight':weights})
    token_emb=emb.forward(token_ids)
    return token_emb


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    from notebook.swiglu import SwiGLU
    
    swiglu=SwiGLU(d_model=d_model,d_ff=d_ff)
    swiglu.load_state_dict({'w1_weight':w1_weight,'w2_weight':w2_weight,'w3_weight':w3_weight})
    out_features=swiglu.forward(in_features)
    return out_features
    


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    from notebook.utiltool import UtilTool
    
    return UtilTool.scaled_dot_product_attention(Q,K,V,mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    给定一个朴素（未经优化的）非批量版本多头注意力机制中的键（key）、查询（query）和值（value）投影权重，
    返回一个优化后的批量实现版本的输出结果。这个实现应该在一个矩阵乘法操作中同时处理所有注意力头的
    键、查询和值投影变换。
    这个函数不应该使用RoPE（旋转位置编码）。
    请参考Vaswani等人2017年论文的第3.2.2节。

    参数：
        d_model (int): 前馈网络输入和输出的维度大小。
        num_heads (int): 多头注意力机制中使用的注意力头数量。
        max_seq_len (int): 最大序列长度，如果你的实现需要预先缓存数据的话。
        q_proj_weight (Float[Tensor, "d_k d_in"]): 查询（Q）投影的权重矩阵
        k_proj_weight (Float[Tensor, "d_k d_in"]): 键（K）投影的权重矩阵
        v_proj_weight (Float[Tensor, "d_k d_in"]): 值（V）投影的权重矩阵
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出投影的权重矩阵
        in_features (Float[Tensor, "... sequence_length d_in"]): 要运行实现的输入特征张量。

    返回值：
        Float[Tensor, " ... sequence_length d_out"]: 包含优化后的批量多头注意力实现结果的输出张量，
        该实现使用了给定的QKV投影权重和输入特征。
    """
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # print('d_model',d_model)
    # print('num_heads',num_heads)
    # print('q_proj_weight.shape',q_proj_weight.shape)
    # print('k_proj_weight.shape',k_proj_weight.shape)
    # print('v_proj_weight.shape',v_proj_weight.shape)
    # print('o_proj_weight.shape',o_proj_weight.shape)
    # print('in_features.shape',in_features.shape)
    
    from notebook.multi_head_attention import MultiHeadAttention
    seq_len=in_features.shape[-2]
    mha=MultiHeadAttention(d_model,num_heads,seq_len)
    mha.load_state_dict({
        'q_proj_weight':q_proj_weight,
        'k_proj_weight':k_proj_weight,
        'v_proj_weight':v_proj_weight,
        'o_proj_weight':o_proj_weight
    })
    result = mha.forward(in_features)
    return result
    


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    from notebook.multi_head_attention import MultiHeadAttention
    mha=MultiHeadAttention(d_model,num_heads,max_seq_len,theta)
    mha.load_state_dict({
        'q_proj_weight':q_proj_weight,
        'k_proj_weight':k_proj_weight,
        'v_proj_weight':v_proj_weight,
        'o_proj_weight':o_proj_weight
    })
    result = mha.forward(in_features,True)
    return result
    


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    from notebook.rotary_positional_embedding import RotaryPositionalEmbedding
    rpe=RotaryPositionalEmbedding(theta,d_k,max_seq_len)
    return rpe.forward(in_query_or_key,token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定预规范 Transformer 块的权重和输入特征，
    返回在该输入特征上运行 Transformer 块的输出。

    此函数应使用 RoPE。
    根据你的实现，你可能只需要将相关参数传递给 TransformerBlock 构造函数，
    或者你可能需要初始化自己的 RoPE 类并将其传递进去。

    参数:
        d_model (int): Transformer 块输入的维度。
        num_heads (int): 多头注意力中使用的头数。`d_model` 必须能被
            `num_heads` 整除。
        d_ff (int): 前馈网络内层的维度。
        max_seq_len (int): 如果你的实现需要预缓存，这是最大序列长度。
        theta (float): RoPE 参数。
        weights (dict[str, Tensor]):
            参考实现的状态字典。
            该字典的键包括:
            - `attn.q_proj.weight`
                所有 `num_heads` 个注意力头的查询投影矩阵。
                形状为 (d_model, d_model)。
                行按 (num_heads, d_k) 形状的矩阵排列，
                因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `attn.k_proj.weight`
                所有 `num_heads` 个注意力头的键投影矩阵。
                形状为 (d_model, d_model)。
                行按 (num_heads, d_k) 形状的矩阵排列，
                因此 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `attn.v_proj.weight`
                所有 `num_heads` 个注意力头的值投影矩阵。
                形状为 (d_model, d_model)。
                行按 (num_heads, d_v) 形状的矩阵排列，
                因此 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `attn.output_proj.weight`
                多头自注意力输出投影的权重矩阵。
                形状为 (d_model, d_model)。
            - `ln1.weight`
                Transformer 块中第一个 RMSNorm
                的仿射变换权重。
                形状为 (d_model,)。
            - `ffn.w1.weight`
                FFN 中第一个线性变换的权重矩阵。
                形状为 (d_model, d_ff)。
            - `ffn.w2.weight`
                FFN 中第二个线性变换的权重矩阵。
                形状为 (d_ff, d_model)。
            - `ffn.w3.weight`
                FFN 中第三个线性变换的权重矩阵。
                形状为 (d_model, d_ff)。
            - `ln2.weight`
                Transformer 块中第二个 RMSNorm
                的仿射变换权重。
                形状为 (d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            要运行实现的输入张量。

    返回:
        Float[Tensor, "batch sequence_length d_model"] 在输入特征上运行
        Transformer 块并使用 RoPE 的输出张量。
    """
    from notebook.transformer_block import TransformerBlock
    
    tb = TransformerBlock(d_model,num_heads,d_ff,max_seq_len,theta)
    tb.attn.load_state_dict({
        'q_proj_weight':weights['attn.q_proj.weight'],
        'k_proj_weight':weights['attn.k_proj.weight'],
        'v_proj_weight':weights['attn.v_proj.weight'],
        'o_proj_weight':weights['attn.output_proj.weight'],
    })
    tb.ffn.load_state_dict({
        'w1_weight':weights['ffn.w1.weight'],
        'w2_weight':weights['ffn.w2.weight'],
        'w3_weight':weights['ffn.w3.weight'],
    })
    tb.ln1.load_state_dict({
        'weight':weights['ln1.weight']
    })
    tb.ln2.load_state_dict({
        'weight':weights['ln2.weight']
    })
    o=tb.forward(in_features)
    return o
    


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """给定 Transformer 语言模型的权重和输入索引，返回在该输入索引上执行前向传播的输出。

    此函数应使用 RoPE。

    Args:
        vocab_size (int): 输出词汇表中要预测的唯一项的数量。
        context_length (int): 一次最多处理的 token 数量。
        d_model (int): 模型嵌入和子层输出的维度。
        num_layers (int): 要使用的 Transformer 层数。
        num_heads (int): 多头注意力中使用的头数。`d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈网络内层的维度（第 3.3 节）。
        rope_theta (float): RoPE $\Theta$ 参数。
        weights (dict[str, Tensor]):
            参考实现的参数字典。{num_layers} 指的是 `0` 到 `num_layers - 1` 之间的整数（层索引）。
            该字典的键如下：
            - `token_embeddings.weight`
                Token 嵌入矩阵。形状为 (vocab_size, d_model)。
            - `layers.{num_layers}.attn.q_proj.weight`
                所有 `num_heads` 个注意力头的查询投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                这些行按 (num_heads, d_k) 形状的矩阵排序，
                因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.k_proj.weight`
                所有 `num_heads` 个注意力头的键投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                这些行按 (num_heads, d_k) 形状的矩阵排序，
                因此 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.v_proj.weight`
                所有 `num_heads` 个注意力头的值投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                这些行按 (num_heads, d_v) 形状的矩阵排序，
                因此 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.output_proj.weight`
                多头自注意力输出投影的权重。
                形状为 ((d_model / num_heads) * num_heads, d_model)。
            - `layers.{num_layers}.ln1.weight`
                Transformer 模块中应用的第一个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `layers.{num_layers}.ffn.w1.weight`
                FFN 中第一个线性变换的权重。
                形状为 (d_model, d_ff)。
            - `layers.{num_layers}.ffn.w2.weight`
                FFN 中第二个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `layers.{num_layers}.ffn.w3.weight`
                FFN 中第三个线性变换的权重。
                形状为 (d_model, d_ff)。
            - `layers.{num_layers}.ln2.weight`
                Transformer 模块中应用的第二个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `ln_final.weight`
                对最终 Transformer 模块输出应用的 RMSNorm 的仿射变换权重。
                形状为 (d_model, )。
            - `lm_head.weight`
                语言模型输出嵌入的权重。
                形状为 (vocab_size, d_model)。
        in_indices (Int[Tensor, "batch_size sequence_length"]): 用于运行语言模型的输入索引张量。形状为 (batch_size, sequence_length)，其中
            `sequence_length` 至多为 `context_length`。

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: 包含每个 token 的预测归一化
            下一个词分布的张量。
    """
    from notebook.transformer_lm import TransformerLM
    transformerLM=TransformerLM(vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta)
    weight_map={
        'embedding.weight':weights['token_embeddings.weight'],
        'norm.weight':weights['ln_final.weight'],
        'linear.weight':weights['lm_head.weight']
    }
    for layers_idx in range(num_layers):
        weight_map[f'transformer_blocks.{layers_idx}.attn.q_proj_weight'] = weights[f'layers.{layers_idx}.attn.q_proj.weight']
        weight_map[f'transformer_blocks.{layers_idx}.attn.k_proj_weight'] = weights[f'layers.{layers_idx}.attn.k_proj.weight']
        weight_map[f'transformer_blocks.{layers_idx}.attn.v_proj_weight'] = weights[f'layers.{layers_idx}.attn.v_proj.weight']
        weight_map[f'transformer_blocks.{layers_idx}.attn.o_proj_weight'] = weights[f'layers.{layers_idx}.attn.output_proj.weight']
        weight_map[f'transformer_blocks.{layers_idx}.ln1.weight'] = weights[f'layers.{layers_idx}.ln1.weight']
        weight_map[f'transformer_blocks.{layers_idx}.ffn.w1_weight'] = weights[f'layers.{layers_idx}.ffn.w1.weight']
        weight_map[f'transformer_blocks.{layers_idx}.ffn.w2_weight'] = weights[f'layers.{layers_idx}.ffn.w2.weight']
        weight_map[f'transformer_blocks.{layers_idx}.ffn.w3_weight'] = weights[f'layers.{layers_idx}.ffn.w3.weight']
        weight_map[f'transformer_blocks.{layers_idx}.ln2.weight'] = weights[f'layers.{layers_idx}.ln2.weight']

    transformerLM.load_state_dict(weight_map)
    result = transformerLM.forward(in_indices)
    return result
    


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """

    from notebook.rmsnorm import RMSNorm
    
    norm=RMSNorm(d_model=d_model,eps=eps)
    norm.load_state_dict({'weight':weights})
    out_features=norm.forward(in_features)
    return out_features



def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    from notebook.utiltool import UtilTool
    return UtilTool.softmax(in_features,dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    from notebook.Tokenizer import Tokenizer
    return Tokenizer(vocab,merges,special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    with open(input_path) as f:
        raw_str=f.read()
        from notebook.Tokenizer import Tokenizer
        vocab,word_bytes_freq,pair_freq=Tokenizer.__init_train_variables__(raw_str,special_tokens)
        vocab,merge_list=Tokenizer.train_bpe(vocab,word_bytes_freq,pair_freq,vocab_size)
        return vocab,merge_list