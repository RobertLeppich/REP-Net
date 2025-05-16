import lightning as pl
from torch import nn
import torch
import math

def calc_patch_sizes(seq_len, patch_size, stride):
    return math.floor((seq_len - patch_size) / stride) + 1


class MergeLayer(nn.Module):

    def __init__(self, target_size, encoding_size, patch_amounts, pred_len, representations, lstm_layer, **kwargs):
        super().__init__()
        self.conv_size = 0
        self.encoding_size = encoding_size
        self.patch_amounts = patch_amounts
        self.representations = representations
        self.pred_len = pred_len

        self.projection_layer = nn.ModuleList()
        self.lstm_layer = nn.ModuleList()
        self.representations = representations
        self.pred_patches = []

        for i, (cover_size, dilation, stride) in enumerate(representations):
            self.projection_layer.append(nn.Linear(encoding_size * patch_amounts[i], target_size, bias=False))

            if lstm_layer is not None and lstm_layer > 0:
                self.lstm_layer.append(nn.LSTM(input_size=encoding_size, hidden_size=encoding_size, batch_first=True, num_layers=lstm_layer))

            self.pred_patches.append(calc_patch_sizes(pred_len, cover_size, stride))

    def forward(self, data, output):

        splits = torch.split(data, self.patch_amounts, dim=1)
        result = []
        for i in range(len(self.patch_amounts)):
            # batch, feature, #patches * encoding_size
            input_patches = splits[i].transpose(1,2).flatten(start_dim=0, end_dim=1)

            if len(self.lstm_layer) > 0:
                input_patches, _ = self.lstm_layer[i](input_patches)

            input_patches = torch.flatten(input_patches.unflatten(0, (output.size(0), output.size(-1))), start_dim=-2, end_dim=-1)
            input_patches = self.projection_layer[i](input_patches)

            result.append(input_patches.unsqueeze(-1))

        # compare_ts(result[0].transpose(2, 1).squeeze(-1), result[1].transpose(2, 1).squeeze(-1), result[2].transpose(2, 1).squeeze(-1))
        return torch.sum(torch.concat(result, dim=-1), -1).transpose(-1, -2)

