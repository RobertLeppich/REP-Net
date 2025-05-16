import lightning as pl
from torch import nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from embedding.data_embedding import TimeEmbedding


def conv_output_dimension(window_size, dilation, padding, kernel_size, stride):
    return int(((window_size + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)


def calc_patch_sizes(seq_len, patch_size, stride):
    return math.floor((seq_len - patch_size) / stride) + 1


def transpose_output_dimension(window_size, dilation, padding, kernel_size, stride):
    return (window_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1


class EncodingLayerSingleLinear(nn.Module):

    def __init__(self, patch_size, encoding_size, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.encoding_size = encoding_size

        self.linear1 = nn.Linear(self.patch_size, self.encoding_size)

    def forward(self, x):
        return self.linear1(x).squeeze(-1)

class EncodingLayerLinear(nn.Module):

    def __init__(self, patch_size, encoding_size, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.encoding_size = encoding_size

        self.linear1 = nn.Linear(self.patch_size, self.encoding_size * 2, bias=False)
        self.linear2 = nn.Linear(self.encoding_size * 2, self.encoding_size)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x))).squeeze(-1)


class GLUEncodingLayerLinear(nn.Module):

    def __init__(self, patch_size, encoding_size, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.encoding_size = encoding_size

        self.linear1 = nn.Linear(self.patch_size, self.encoding_size * 2, bias=False)
        self.linear2 = nn.Linear(self.encoding_size * 2, self.encoding_size * 2)

    def forward(self, x):
        return F.glu(self.linear2(F.gelu(self.linear1(x)))).squeeze(-1)


class MatmulEncodingLayer(nn.Module):

    def __init__(self, patch_size, encoding_size, **kwargs):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(patch_size, encoding_size), requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.encoding)


class MatmulLinearEncodingLayer(nn.Module):

    def __init__(self, patch_size, encoding_size, **kwargs):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(patch_size, encoding_size), requires_grad=True)
        self.linear = nn.Linear(encoding_size, encoding_size)

    def forward(self, x):
        return self.linear(torch.matmul(x, self.encoding))


class MatmulLinearGLUEncodingLayer(nn.Module):

    def __init__(self, patch_size, encoding_size, **kwargs):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(patch_size, encoding_size), requires_grad=True)
        self.linear = nn.Linear(encoding_size, encoding_size * 2)

    def forward(self, x):
        return F.glu(self.linear(torch.matmul(x, self.encoding)))


class EncodingLayerCNN1(nn.Module):

    def __init__(self, patch_size, encoding_size, feature_dimension, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.encoding_size = encoding_size

        self.conv1 = nn.Conv1d(feature_dimension, self.encoding_size * feature_dimension, 3, groups=feature_dimension, padding=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.linear = nn.Linear(patch_size, 1)

    def forward(self, x):
        pre_shape = x.shape
        x = self.conv1(x.flatten(0,1))
        return self.linear(x.unflatten(0, (pre_shape[:2]))).squeeze(-1).unflatten(-1, (pre_shape[2], -1))

class EncodingLayerCNN2(nn.Module):

    def __init__(self, patch_size, encoding_size, feature_dimension, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.encoding_size = encoding_size

        self.conv1 = nn.Conv1d(feature_dimension, self.encoding_size * feature_dimension, 3, groups=feature_dimension, padding=1)
        self.conv2 = nn.Conv1d(self.encoding_size * feature_dimension, self.encoding_size * feature_dimension, 3, groups=feature_dimension, padding=1)
        self.linear = nn.Linear(patch_size, 1)

    def forward(self, x):
        pre_shape = x.shape
        x = self.conv1(x.flatten(0,1))
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.linear(x.unflatten(0, (pre_shape[:2]))).squeeze(-1).unflatten(-1, (pre_shape[2], -1))
        return x

class EncodingLayerCNN3(nn.Module):

    def __init__(self, patch_size, encoding_size, feature_dimension, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.encoding_size = encoding_size

        self.conv1 = nn.Conv1d(feature_dimension, self.encoding_size * feature_dimension, 3, groups=feature_dimension, padding=1)
        self.conv2 = nn.Conv1d(self.encoding_size * feature_dimension, self.encoding_size * feature_dimension, 3, groups=feature_dimension, padding=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.linear = nn.Linear(patch_size, 1)

    def forward(self, x):
        pre_shape = x.shape
        x = self.conv1(x.flatten(0,1))
        x = F.pad(x, (1,1))
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.pad(x, (1,1))
        x = self.max_pool2(x)
        x = self.linear(x.unflatten(0, (pre_shape[:2]))).squeeze(-1).unflatten(-1, (pre_shape[2], -1))
        return x


class EncodingLayerFilter(nn.Module):

    def __init__(self, patch_size, encoding_size, n=100):
        super().__init__()

        self.permutation = self.calc_filters(patch_size, n=n)
        self.embedding = nn.Embedding(n, encoding_size)

    def forward(self, x):
        x = self.scale_last_dim(x)
        r = self.permutation.to(x.device) - x
        t = r.sum(dim=-1)
        result = torch.argmin(t.abs(), keepdim=True, dim=0)
        return self.embedding(result.permute(1, 2, 3, 0)).squeeze(-2)

    @staticmethod
    def calc_filters(p, step=0.01, n=10000):
        random_tensor = torch.rand(n, p)
        quantized_tensor = torch.floor(random_tensor / step) * step
        return quantized_tensor.view(n, 1, 1, 1, p)

    @staticmethod
    def scale_last_dim(x, eps=1e-8):
        # Compute min and max along the last dimension, keeping the dimension for broadcasting
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        # Normalize: subtract the minimum and divide by the range.
        # The epsilon is added to avoid division by zero in case x_max equals x_min.
        return (x - x_min) / (x_max - x_min + eps)


representation_modules = {
    "single_linear": EncodingLayerSingleLinear,
    "linear": EncodingLayerLinear,
    "glu": GLUEncodingLayerLinear,
    "simple_linear": MatmulEncodingLayer,
    "double_linear": MatmulLinearEncodingLayer,
    "double_linear_glu": MatmulLinearGLUEncodingLayer,
    "cnn_1": EncodingLayerCNN1,
    "cnn_2": EncodingLayerCNN2,
    "cnn_3": EncodingLayerCNN3
}

class RepresentationLayer(nn.Module):

    def __init__(self, conv_dims, encoding_size, seq_len, pred_len, time_embedding, freq, time_embedding_size, representation_module, **kwargs):
        super().__init__()
        self.conv_sizes = []
        self.encoding_size = encoding_size

        representation_module = representation_modules[representation_module]
        self.representation_layer = []
        self.encoding_layer = nn.ModuleList()
        self.encoding_layer_time = nn.ModuleList()

        self.time_embedding = TimeEmbedding(encoding_size=time_embedding_size, time_embedding=time_embedding, seq_len=seq_len, pred_len=pred_len, freq=freq) \
            if len(time_embedding) > 0 else None

        for cover_size, dilation, stride in conv_dims:
            #conv_size_layer = conv_output_dimension(seq_len, dilation, 0, patch_size, stride)
            amount_patches = calc_patch_sizes(seq_len, cover_size, stride)
            assert amount_patches > 0, "Invalid patch sizes"

            self.conv_sizes.append(amount_patches)

            self.representation_layer.append((cover_size, dilation, stride))



            if self.time_embedding is not None:
                self.encoding_layer.append(
                    representation_module(patch_size=math.ceil(cover_size / dilation), encoding_size=encoding_size // 2, **kwargs))
                self.encoding_layer_time.append(
                    EncodingLayerSingleLinear(patch_size=cover_size * time_embedding_size, encoding_size=encoding_size // 2))
            else:
                self.encoding_layer.append(
                    representation_module(patch_size=math.ceil(cover_size / dilation), encoding_size=encoding_size, **kwargs))

    def get_representation_size(self):
        return sum(self.conv_sizes)

    def get_representations(self):
        return self.representation_layer

    def get_patch_amount(self):
        return self.conv_sizes

    def forward(self, data, time_embedding_x, time_embedding_y):

        result = []
        if self.time_embedding is not None:
            time_embedding_x, time_embedding_y = self.time_embedding(time_embedding_x, time_embedding_y)

        for i, (cover_size, dilation, stride) in enumerate(self.representation_layer):

            repr = data.unfold(dimension=1, size=cover_size, step=stride)[:, :, :, ::dilation]
            repr = self.encoding_layer[i](repr)

            if len(self.encoding_layer_time) > 0:
                time_repr = time_embedding_x.unfold(dimension=1, size=cover_size, step=stride)
                time_repr = self.encoding_layer_time[i](time_repr.flatten(start_dim=-2, end_dim=-1))
                result.append(torch.concat([repr, time_repr.unsqueeze(-2).repeat(1, 1, repr.size(-2), 1)], dim=-1))
            else:
                result.append(repr)
        return torch.concat(result, axis=1)


    def _calc_representations(self, data, layer, cover_size, dilation, stride):
        x = data.unfold(dimension=1, size=cover_size, step=stride)[:, :, :, ::dilation]
        # batch, patches, feature, data

        x = self.encoding_layer[layer](x)
        # batch, patches, feature, encoding_size
        return x


    def normalize(self, x, ref=None):
        """
        Normalize time embeddings to match the input data scale.
        If `ref` is provided, normalize using its mean and std.
        """
        if ref is not None:
            mean = ref.mean(dim=(1, 3), keepdim=True)
            std = ref.std(dim=(1, 3), keepdim=True) + 1e-5
            return (x - mean) / std
        else:
            mean = x.mean()
            std = x.std() + 1e-5
            return (x - mean) / std