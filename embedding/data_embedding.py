from torch import nn
import torch
import math
import numpy as np


class PositionalEncoding2(nn.Module):

    def __init__(self, d_embedd: int, batch_size: int, data_window_size: int):
        super().__init__()

        position = torch.arange(data_window_size).unsqueeze(1)
        pe = torch.zeros(batch_size, data_window_size, d_embedd)
        for i in range(d_embedd):
            pe[:, :, i] = (0.5 + 0.5 * torch.sin(position * (i/d_embedd))).squeeze(-1)
        self.register_buffer('pe', pe)

    def forward(self):
        return self.pe


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = 0.5 + 0.5 * torch.sin(position * div_term)
        pe[:, 1::2] = 0.5 + 0.5 * torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, start, end):
        return self.pe[:, start:end].detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, freq='t'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        if freq == 't':
            self.minute_embed = nn.Embedding(minute_size, d_model)
        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.month_embed = nn.Embedding(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        t = hour_x + weekday_x + day_x + month_x + minute_x
        t = (t - t.min()) / (t.max() - t.min())
        return t


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='t'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        t = self.embed(x.to(torch.float32))
        t = (t - t.min()) / (t.max() - t.min())
        return t + t.min().abs()


class TimeEmbedding(nn.Module):
    def __init__(self, encoding_size, time_embedding, seq_len, pred_len, freq, **kwargs):
        super(TimeEmbedding, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pos_encoding = time_embedding.split("+")
        self.time_feature_embedding = None
        self.temporal_embedding = None
        self.positional_embedding = None


        if "timeF" in self.pos_encoding:
            self.time_feature_embedding = TimeFeatureEmbedding(d_model=encoding_size, freq=freq)
        if "tempEmb" in self.pos_encoding:
            self.temporal_embedding = TemporalEmbedding(d_model=encoding_size, freq=freq)
        if "posEmb" in self.pos_encoding:
            self.positional_embedding = PositionalEmbedding(d_model=encoding_size, max_len=seq_len + pred_len)


    def forward(self, time_embedding_input, time_embedding_target):

        xs, ys = [], []
        if self.positional_embedding is not None:
            xs.append(self.positional_embedding(start=0, end=self.seq_len).repeat(time_embedding_input.shape[0], 1, 1))
            ys.append(self.positional_embedding(start=self.seq_len, end=self.seq_len + self.pred_len).repeat(time_embedding_input.shape[0], 1, 1))

        if self.temporal_embedding is not None:
            xs.append(self.temporal_embedding(time_embedding_input))
            ys.append(self.temporal_embedding(time_embedding_target))

        if self.time_feature_embedding is not None:
            xs.append(self.time_feature_embedding(time_embedding_input))
            ys.append(self.time_feature_embedding(time_embedding_target))

        if len(xs) >0:
            return self.normalize(torch.sum(torch.stack(xs), dim=0)), self.normalize(torch.sum(torch.stack(ys), dim=0))
        return None, None

    def normalize(self, x, ref=None):
        """
        Normalize time embeddings to match the input data scale.
        If `ref` is provided, normalize using its mean and std.
        """
        if ref is not None:
            mean = ref.mean(dim=(1, 2), keepdim=True)
            std = ref.std(dim=(1, 2), keepdim=True) + 1e-5
            return (x - mean) / std
        else:
            mean = x.mean()
            std = x.std() + 1e-5
            return (x - mean) / std


