"""
This module defines all the models in the experiment. 
Each model is described in detail in the paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch_geometric.utils import to_dense_batch
from transformers import AutoModel
import numpy as np
from mind.main.batch import MINDBatch, ContentsEncoded, Word2VecMINDBatch
from typing import Set


def init_weights(m: nn.Module):
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)

    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias)

    if isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def is_precomputed(x):
    return type(x) is torch.Tensor


def prune_linear_layer(layer, index, dim=0):
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
        layer.weight.device
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = torch.ones(n_heads, head_size)
    heads = (
        set(heads) - already_pruned_heads
    )  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


class DistillMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = nn.Dropout(p=0.2)

        self.q_lin = nn.Linear(in_features=dim, out_features=dim)
        self.k_lin = nn.Linear(in_features=dim, out_features=dim)
        self.v_lin = nn.Linear(in_features=dim, out_features=dim)
        self.out_lin = nn.Linear(in_features=dim, out_features=dim)

        self.pruned_heads: Set[int] = set()
        self.attention_head_size = self.dim // self.n_heads

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.attention_head_size, self.pruned_heads
        )
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = self.attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        query,
        key,
        value,
        mask,
        head_mask=None,
        output_attentions=False,
    ):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """group heads"""
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (
            (mask == 0).view(mask_reshp).expand_as(scores)
        )  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(
            scores, dim=-1
        )  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.attention = DistillMultiHeadSelfAttention(dim, n_heads)
        # self.sa_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

    def forward(
        self, x, attn_mask=None, head_mask=None, output_attentions: bool = False
    ):
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            (
                sa_output,
                sa_weights,
            ) = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            sa_output = sa_output[0]

        if output_attentions:
            return (sa_weights,) + sa_output

        return sa_output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, attention_dropout=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = nn.Dropout(p=attention_dropout)

        # Have an even number of multi heads that divide the dimensions
        if self.dim % self.n_heads != 0:
            # Raise value errors for even multi-head attention nodes
            raise ValueError(
                f"self.n_heads: {self.n_heads} must divide self.dim: {self.dim} evenly"
            )

        self.q_lin = nn.Linear(in_features=dim, out_features=dim)
        self.k_lin = nn.Linear(in_features=dim, out_features=dim)
        self.v_lin = nn.Linear(in_features=dim, out_features=dim)
        self.out_lin = nn.Linear(in_features=dim, out_features=dim)

        self.attention_head_size = self.dim // self.n_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask=None,
    ):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (
            (mask == 0).view(mask_reshp).expand_as(scores)
        )  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(
            scores, dim=-1
        )  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        return context


class AdditiveAttention(nn.Module):
    """
    additive attention
    """

    def __init__(self, dim1=300, dim2=300):
        """
        dim1: dim of hi
        dim2: dim of qw
        """
        super().__init__()
        self.V = nn.Linear(dim1, dim2)  # includes Vw and vw
        self.qw = nn.Parameter(torch.rand(dim2, 1))
        nn.init.xavier_uniform_(self.qw)
        self.V.apply(init_weights)

    def forward(self, context, mask=None):
        """Additive Attention
        Args:
            context (tensor): [B, seq_len, dim]
            mask (tensor): [B, seq_len]
        Returns:
            outputs, weights: [B, dim], [B, seq_len]
        """
        x = torch.tanh(self.V(context))  # [B, seq_len, dim2]
        a = torch.matmul(x, self.qw).squeeze(2)  # [B, seq_len]
        if mask is not None:
            a = a.masked_fill(mask == False, torch.finfo(x.dtype).min)
        alpha = torch.softmax(a, dim=1)  # B, seq_len

        return torch.bmm(alpha.unsqueeze(1), context).squeeze(1), alpha


class CoAttention(nn.Module):
    """
    implement co-attention with mask, dynamic user encoder
    """

    def __init__(self, dim1=300, dim2=300):
        """
        dim1: dim of hi
        dim2: dim of qw
        """
        super().__init__()
        self.V = nn.Linear(dim1, dim2)  # includes Vw and vw
        self.qw = nn.Parameter(torch.rand(dim2, 1))
        nn.init.xavier_uniform_(self.qw)
        self.V.apply(init_weights)

    def forward(self, x_hist, x_cand, mask_hist=None):
        # x_hist: [B, L1, dim]
        # x_cand: [B, L2, dim]
        # mask_hist: [B, L1]
        attn = torch.matmul(x_cand, x_hist.transpose(1, 2))  # [B, L2, L1]
        L2 = attn.shape[1]
        mask_hist = mask_hist.unsqueeze(1).repeat(1, L2, 1)  # [B, L2, L1]
        if mask_hist is not None:
            attn = attn.masked_fill(mask_hist == False, torch.finfo(attn.dtype).min)
        attn_hist = torch.softmax(attn, dim=-1)  # [B, L2, L1]
        x = torch.matmul(attn_hist, x_hist)  # [B, L2, dim]

        return x


class WordEmbeddingNewsEncoder(nn.Module):
    """
    news encoder for word2vec version NRMS
    """

    def __init__(
        self,
        wordembedding_path: str,
    ):
        super().__init__()
        wordembedding_array = np.load(wordembedding_path).astype(np.float32)
        V, D = wordembedding_array.shape
        self.dim = D
        self.embedding = nn.Embedding(V, D)  # No set padding_idx
        self.embedding.weight.data.copy_(torch.from_numpy(wordembedding_array))

        self.pos_embedding = nn.Embedding(512, D)
        self.self_attn = MultiHeadSelfAttention(self.dim, 6)
        self.additive_attn = AdditiveAttention(D, D)

    def forward(self, inputs, mask=None):
        title = inputs["title"]
        if title.ndim == 1:
            title = title.unsqueeze(0)  # (1, L)
        title_len = title.size(1)
        x_t = self.embedding(title)

        title_pos = (
            torch.arange(title_len, device=title.device)
            .unsqueeze(0)
            .repeat(title.size(0), 1)
        )

        x_t = x_t + self.pos_embedding(title_pos)
        # print("x_t.shape: ", x_t.shape)  # torch.Size([248, 37, 300])
        if mask is not None:
            x_t = self.self_attn(x_t, x_t, x_t, x_t, mask=mask)
        else:
            mask = torch.ones(
                (title.shape[0], title.shape[1]), device=x_t.device
            ).bool()
            x_t = self.self_attn(x_t, x_t, x_t, mask=mask)

        x, alpha = self.additive_attn(x_t)

        return x


class BERTEmbeddingNewsEncoder(nn.Module):
    """
    news encoder for BERT version NRMS (PLM-NR)
    """

    def __init__(
        self,
        pretrained_model_name: str,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dim = self.bert.config.hidden_size
        self.self_attn = MultiHeadSelfAttention(self.dim, 8)
        self.additive_attn = AdditiveAttention(self.dim, self.dim)

    def forward(self, inputs):
        x_t = self.bert(**inputs["title"])[0]

        title = inputs["title"]
        mask = title["attention_mask"]
        if mask is not None:
            x_t = self.self_attn(x_t, x_t, x_t, mask=mask)
        else:
            mask = torch.ones(
                (title.shape[0], title.shape[1]), device=x_t.device
            ).bool()
            x_t = self.self_attn(x_t, x_t, x_t, mask=mask)
        x, alpha = self.additive_attn(x_t)

        return x


class BERTCLSEmbeddingNewsEncoder(nn.Module):
    """
    news encoder with CLS (PLM fusion module)
    """

    def __init__(
        self,
        pretrained_model_name: str,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dim = self.bert.config.hidden_size

        # self.pos_embedding = nn.Embedding(512, self.dim)
        self.self_attn = MultiHeadSelfAttention(self.dim, 8)
        self.additive_attn = AdditiveAttention(self.dim, self.dim)

    def forward(self, inputs):
        # print(inputs["title"]["input_ids"].shape)  # torch.Size([202, 41])
        # print(inputs["title"]["input_ids"][0])  # torch.Size([202, 41]), CLS is the first token in sequence
        x_t = self.bert(**inputs["title"], output_hidden_states=True)
        x_t = torch.cat(
            [item[:, 0, :].unsqueeze(1) for item in x_t["hidden_states"][1:]], dim=1
        )  # (202, 6, 768)
        # print("x_t.shape: ", x_t.shape)  # torch.Size([202, 6, 768])
        x_t = self.self_attn(
            x_t,
            x_t,
            x_t,
            mask=torch.ones((x_t.shape[0], x_t.shape[1]), device=x_t.device).bool(),
        )

        x, alpha = self.additive_attn(x_t)

        return x


class PLMNR(nn.Module):
    """
    BERT version NRMS (PLM-NR)
    """

    def __init__(
        self,
        pretrained_model_name: str,
        sa_pretrained_model_name: str,
    ):
        super(PLMNR, self).__init__()
        self.encoder = BERTEmbeddingNewsEncoder(pretrained_model_name)
        dim = self.encoder.dim

        # Use pre trained self attention.
        bert = AutoModel.from_pretrained(sa_pretrained_model_name)
        self.self_attn = TransformerBlock(dim, 12)
        self.additive_attn = AdditiveAttention(dim, dim)

    def forward(self, inputs: MINDBatch):
        if is_precomputed(inputs["x_hist"]):
            x_hist = inputs["x_hist"]
        else:
            x_hist = self.encoder(inputs["x_hist"])

        x_hist, mask_hist = to_dense_batch(x_hist, inputs["batch_hist"])
        x_hist = self.self_attn(x_hist, attn_mask=mask_hist)  # DistilBERT

        if is_precomputed(inputs["x_cand"]):
            x_cand = inputs["x_cand"]
        else:
            x_cand = self.encoder.forward(inputs["x_cand"])
        x_cand, mask_cand = to_dense_batch(x_cand, inputs["batch_cand"])
        x_hist, _ = self.additive_attn(x_hist)
        logits = torch.bmm(x_hist.unsqueeze(1), x_cand.permute(0, 2, 1)).squeeze(1)
        logits = logits[mask_cand]

        targets = inputs["targets"]
        if targets is None:
            return logits

        if self.training:
            criterion = nn.CrossEntropyLoss()
            # criterion = LabelSmoothingCrossEntropy()
            loss = criterion(logits.reshape(targets.size(0), -1), targets)
        else:
            # In case of val, targets are multi label. It's not comparable with train.
            with torch.no_grad():
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, targets.float())

        return loss, logits


class FusionNR(nn.Module):
    """
    BERT version NRMS with PLM fusion module
    use CLS sequence as the representation of news
    """

    def __init__(
        self,
        pretrained_model_name: str,
        sa_pretrained_model_name: str,
    ):
        super(FusionNR, self).__init__()
        self.encoder = BERTCLSEmbeddingNewsEncoder(pretrained_model_name)
        dim = self.encoder.dim

        # Use pre trained self attention.
        bert = AutoModel.from_pretrained(sa_pretrained_model_name)
        self.self_attn = bert.transformer.layer[-1]  # DistilBERT

        self.additive_attn = AdditiveAttention(dim, dim)

    def forward(self, inputs: MINDBatch):
        if is_precomputed(inputs["x_hist"]):
            x_hist = inputs["x_hist"]
        else:
            x_hist = self.encoder(inputs["x_hist"])

        x_hist, mask_hist = to_dense_batch(x_hist, inputs["batch_hist"])
        x_hist = self.self_attn(x_hist, attn_mask=mask_hist)[0]  # DistilBERT

        if is_precomputed(inputs["x_cand"]):
            x_cand = inputs["x_cand"]
        else:
            x_cand = self.encoder.forward(inputs["x_cand"])
        x_cand, mask_cand = to_dense_batch(x_cand, inputs["batch_cand"])
        x_hist, _ = self.additive_attn(x_hist, mask_hist)
        logits = torch.bmm(x_hist.unsqueeze(1), x_cand.permute(0, 2, 1)).squeeze(1)
        logits = logits[mask_cand]

        targets = inputs["targets"]
        if targets is None:
            return logits

        if self.training:
            criterion = nn.CrossEntropyLoss()
            # criterion = LabelSmoothingCrossEntropy()
            loss = criterion(logits.reshape(targets.size(0), -1), targets)
        else:
            # In case of val, targets are multi label. It's not comparable with train.
            with torch.no_grad():
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, targets.float())

        return loss, logits


class MLFBERT(nn.Module):
    """
    BERT version NRMS with PLM fusion module
    use CLS token as news embedding
    use co-attention to get user embedding
    """

    def __init__(
        self,
        pretrained_model_name: str,
        sa_pretrained_model_name: str,
    ):
        super(MLFBERT, self).__init__()
        self.encoder = BERTCLSEmbeddingNewsEncoder(pretrained_model_name)
        dim = self.encoder.dim

        # Use pre trained self attention.
        bert = AutoModel.from_pretrained(sa_pretrained_model_name)
        self.self_attn = bert.transformer.layer[-1]  # DistilBERT
        self.additive_attn = CoAttention(dim, dim)

    def forward(self, inputs: MINDBatch):
        if is_precomputed(inputs["x_hist"]):
            x_hist = inputs["x_hist"]
        else:
            x_hist = self.encoder(inputs["x_hist"])

        x_hist, mask_hist = to_dense_batch(x_hist, inputs["batch_hist"])
        x_hist = self.self_attn(x_hist, attn_mask=mask_hist)[0]  # DistilBERT

        if is_precomputed(inputs["x_cand"]):
            x_cand = inputs["x_cand"]
        else:
            x_cand = self.encoder.forward(inputs["x_cand"])
        x_cand, mask_cand = to_dense_batch(x_cand, inputs["batch_cand"])

        x_hist = self.additive_attn(x_hist, x_cand, mask_hist)

        # print("x_hist.shape: ", x_hist.shape)  # torch.Size([8, 41, 768])
        # print("x_cand.shape: ", x_cand.shape)  # torch.Size([8, 41, 768])
        logits = (
            torch.matmul(x_hist.unsqueeze(2), x_cand.unsqueeze(3))
            .squeeze(-1)
            .squeeze(-1)
        )
        logits = logits[mask_cand]

        targets = inputs["targets"]
        if targets is None:
            return logits

        if self.training:
            criterion = nn.CrossEntropyLoss()
            # criterion = LabelSmoothingCrossEntropy()
            loss = criterion(logits.reshape(targets.size(0), -1), targets)
        else:
            # In case of val, targets are multi label. It's not comparable with train.
            with torch.no_grad():
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, targets.float())

        return loss, logits


class Word2VecNRMS(nn.Module):
    """
    Word2Vec version NRMS
    """

    def __init__(
        self,
        wordembedding_path: str,
    ):
        super(Word2VecNRMS, self).__init__()
        self.encoder = WordEmbeddingNewsEncoder(wordembedding_path)

        dim = self.encoder.dim
        self.self_attn = MultiHeadSelfAttention(dim, 6)
        self.additive_attn = AdditiveAttention(dim, dim)

    def forward(self, inputs: Word2VecMINDBatch):
        if is_precomputed(
            inputs["x_hist"]
        ):  # The type of x_hist can be rd2vecContentsEncoded or torch.Tensor,
            # depending on the training set or verification set
            x_hist = inputs["x_hist"]
        else:
            x_hist = self.encoder(inputs["x_hist"])
        # If it is the validation set
        # print("inputs['x_hist'].shape: ", inputs['x_hist'].shape)  # torch.Size([L, 300])
        # print("type(x_hist): ", type(x_hist))  # <class 'torch.Tensor'>
        # print("x_hist.shape: ", x_hist.shape)  # torch.Size([L, 300])

        # If it is the training set
        # print("before dense batch x_hist:", torch.sum(torch.isnan(x_hist)))  #
        x_hist, mask_hist = to_dense_batch(
            x_hist, inputs["batch_hist"]
        )  # Do padding, dynamic batching on the history sequence
        # print("to dense batch, x_hist:", torch.sum(torch.isnan(x_hist)))  # > 0
        # print("after x_hist.shape: ", x_hist.shape)  # torch.Size([8, L1, 300])
        # print("mask_hist.shape: ", mask_hist.shape)  # torch.Size([8, L1]
        # print(mask_hist)  # BoolTensor
        x_hist = self.self_attn(
            x_hist, x_hist, x_hist, mask=mask_hist
        )  # Weights used to initialize self_attn

        # print("x_hist: ", torch.sum(torch.isnan(x_hist)))  # > 0
        x_hist, _ = self.additive_attn(x_hist, mask_hist)

        if is_precomputed(inputs["x_cand"]):
            x_cand = inputs["x_cand"]
        else:
            x_cand = self.encoder(inputs["x_cand"])
        x_cand, mask_cand = to_dense_batch(x_cand, inputs["batch_cand"])  #
        # print("x_cand.shape: ", x_cand.shape)  # torch.Size([8, 128, 300])
        # print("x_hist: ", torch.sum(torch.isnan(x_hist)))  # with nan
        # print("x_cand: ", torch.sum(torch.isnan(x_cand)))  # without nan
        logits = torch.bmm(x_hist.unsqueeze(1), x_cand.permute(0, 2, 1)).squeeze(1)

        logits = logits[mask_cand]
        # print("x_hist: ", torch.sum(torch.isnan(x_hist)))  # with nan

        targets = inputs["targets"]
        if targets is None:
            return logits

        if self.training:
            criterion = nn.CrossEntropyLoss()  # 1:K multi-category task, 1 0 0 0 0
            # criterion = LabelSmoothingCrossEntropy()
            loss = criterion(logits.reshape(targets.size(0), -1), targets)
            # print("training loss: ", loss)  # nan
        else:
            # In case of val, targets are multi label. It's not comparable with train.
            with torch.no_grad():
                criterion = nn.BCEWithLogitsLoss()  # Each individual forecast
                loss = criterion(logits, targets.float())

        return loss, logits
