import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Literal
from architecture.entmax import entmax15



def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -10000)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def attention_entmax15(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    p_attn = entmax15(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h: int, encoding_size: int,
                 attention_func: Literal["classic", "propsparse", "entmax15", "propsparse_entmax15", "fourier1", "no"]="classic",
                 dropout=0.1, feature_dimension=1, **kwargs):

        super(MultiHeadedAttention, self).__init__()

        self.h = h
        self.feature_dimension = feature_dimension
        self.encoding_size = encoding_size


        if attention_func == "entmax15":
            self.attention = attention_entmax15
        elif attention_func == "classic":
            self.attention = attention
        elif attention_func == "no":
            self.attention = None
            return  # no need to initialize the rest

        self.lin_query = nn.Linear(encoding_size, encoding_size)
        self.lin_key = nn.Linear(encoding_size, encoding_size)
        self.lin_values = nn.Linear(encoding_size, encoding_size)

        self.lin_fin = nn.Linear(encoding_size, encoding_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        nfeatures = query.size(2)
        npatches = query.size(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.lin_query(query).view(nbatches, nfeatures, self.h, -1, self.encoding_size // self.h)
        key = self.lin_key(key).view(nbatches, nfeatures, self.h, -1, self.encoding_size // self.h)
        value = self.lin_values(value).view(nbatches, nfeatures, self.h, -1, self.encoding_size // self.h)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.view(nbatches, npatches, nfeatures, self.encoding_size)

        return self.lin_fin(x), attn.sum(dim=2) if attn is not None else None