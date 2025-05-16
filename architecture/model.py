import torch
from mpmath import residual
from sympy.abc import lamda
from torch.cuda import device
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch.nn as nn
import lightning as pl
from architecture.loss_functions import ForecastingLoss
from architecture.multiHeadAttention import MultiHeadedAttention
from architecture.metrics import calc_metrics
from architecture.RevIN import RevIN
from architecture.representation import RepresentationLayer
from architecture.merge import MergeLayer
import numpy as np


class EncoderLayer(nn.Module):

    def __init__(self, representation_size, encoding_size: int, h: int, dropout: float, attention_func, feature_dimension: int,
                 tsrm_fc=False, glu_layer=True, **kwargs):
        super().__init__()
        if attention_func is None:
            self._self_attention = None
        else:
            self._self_attention = MultiHeadedAttention(encoding_size=encoding_size, h=h, dropout=dropout,
                                                        attention_func=attention_func,
                                                        feature_dimension=feature_dimension,
                                                        **kwargs)
        self.tsrm_fc = tsrm_fc
        self.glu_layer = glu_layer

        self._dropout = nn.Dropout(p=dropout)

        self.positional_mixer = nn.Linear(representation_size, representation_size)

        if not tsrm_fc:
            self.feature_mixer = nn.Linear(encoding_size, encoding_size)
            self.feature_mixer2 = nn.Linear(encoding_size, encoding_size)
        else:
            self.feature_mixer = nn.Linear(encoding_size * feature_dimension, encoding_size * feature_dimension)
            self.feature_mixer2 = nn.Linear(encoding_size * feature_dimension, encoding_size * feature_dimension)

        self.glu_linear = nn.Linear(encoding_size, encoding_size * 2)

        self.pos_norm = nn.LayerNorm(representation_size)
        self.feature_norm = nn.LayerNorm(encoding_size)
        self.attn_norm = nn.LayerNorm(encoding_size)
        self.glu_norm = nn.LayerNorm(encoding_size)

    def forward(self, x):

        # Feed forward positional
        residual = x
        x = self.pos_norm(x.transpose(1,-1)).transpose(1,-1)
        x = self.positional_mixer(x.transpose(-1, 1)).transpose(-1, 1)
        x = nn.functional.relu(x)
        x = self._dropout(x)
        x += residual

        # Self attention
        if self._self_attention is not None:
            residual = x
            x = self.attn_norm(x)
            x, attn = self._self_attention(query=x, key=x, value=x)
            x = nn.functional.gelu(x)
            x = self._dropout(x)
            x += residual

        # GLU
        residual = x
        x = self.glu_norm(x)
        if self.glu_layer:
            x = self.glu_linear(x)
            x = nn.functional.glu(x)
            x = self._dropout(x)

        if not self.tsrm_fc:
            x = self.feature_mixer(x)
        else:
            pre_shape = x.shape
            x = self.feature_mixer(x.flatten(-2, -1)).reshape(pre_shape)

        x = self._dropout(x)
        x += residual

        return x, None


class Transformations(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.global_minimums = None
        self.requires_transform = False
        self.revin = RevIN(num_features=config["feature_dimension"], affine=True, subtract_last=False) if config.get(
            "revin", True) else None

        self.batch_size = config["batch_size"]
        self.feature_dimension = config["feature_dimension"]

        if self.config["phase"] != "pretrain" and self.config["task"] == "forecasting":
            self.finetuning_fc = True
            self.pred_len = self.config["pred_len"]
        else:
            self.finetuning_fc = False
            self.pred_len = 0

    def transform(self, x, mask):

        # revin
        if self.revin is not None:
            x = self.revin(x, "norm")

        if mask is not None:
            x = torch.masked_fill(x, mask, -1)

        #x = x.transpose(-2, -1).reshape(self.batch_size * self.feature_dimension, -1).unsqueeze(-1)
        return x

    def reverse(self, x):

        #x = x.squeeze(-1).reshape(self.batch_size, self.feature_dimension, -1).transpose(-2, -1)

        if self.revin is not None:
            x = self.revin(x, "denorm")

        return x


class REPNet(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()

        self.value_transformer = Transformations(config)

        self.config = config

        self.learning_rate = config["learning_rate"]

        self.representation = RepresentationLayer(**config)
        self.merge_layer = MergeLayer(target_size=config.get("pred_len", None) or config["seq_len"],
                                      representations=self.representation.get_representations(),
                                      patch_amounts=self.representation.get_patch_amount(), **config)

        self.encoding_layer = nn.ModuleList([EncoderLayer(representation_size=self.representation.get_representation_size(), **config) for _ in range(config["N"])])
        self.float()

    def forward(self, encoding: torch.Tensor, time_embedding_x, time_embedding_y, mask=None):

        encoding = self.value_transformer.transform(encoding, mask)

        encoding = self.representation.forward(encoding, time_embedding_x, time_embedding_y)
        # batch, patches, features, encoding_size
        attention_weights = []

        for i in range(len(self.encoding_layer)):
            encoding, attn = self.encoding_layer[i](encoding)


        encoding = self.merge_layer(encoding, torch.zeros((*time_embedding_y.shape[:-1], self.config["feature_dimension"]), device=encoding.device))
        encoding = self.value_transformer.reverse(encoding).contiguous()

        return encoding, None
        # encoding = self.encoding_ff(encoding, x_mark)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=self.config.get("weight_decay", 0.))
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss", "frequency": 1, "strict": True}}


    def _step(self, input_batch, idx, phase="train"):
        if len(input_batch) == 2:
            input_data, input_target = input_batch
        elif len(input_batch) == 3:
            input_data, input_target, meta = input_batch
        else:
            input_data, input_target, embedding_x, embedding_y = input_batch

        loss = self._run(input_data, input_target, embedding_x, embedding_y, determine_metrics=phase == "test",
                         phase=phase)
        return loss

    def training_step(self, input_batch, idx):
        return self._step(input_batch, idx, phase="train")

    def validation_step(self, input_batch, idx):
        return self._step(input_batch, idx, phase="val")

    def test_step(self, input_batch, idx):
        return self._step(input_batch, idx, phase="test")

    def _run(self, input_data, input_target, embedding_x, embedding_y, determine_metrics=True, calc_real=False,
             phase="train") -> "Loss":
        ...


    def on_validation_epoch_end(self) -> None:
        self.lr_schedulers().step(self.trainer.callback_metrics["val_loss"])


class REPNetForecasting(REPNet):
    def __init__(self, config: dict):

        super().__init__(config)
        self.loss = ForecastingLoss(config)

    def _run(self, input_data, input_target, embedding_x, embedding_y, determine_metrics=True, calc_real=False,
             phase="train"):

        input_data = input_data.float()
        output, attn_map = self.forward(input_data, embedding_x, embedding_y)

        horizon_output = output[:, -self.config["pred_len"]:, :]
        horizon_target = input_target[:, -self.config["pred_len"]:, :]

        loss = self.loss(prediction=horizon_output.float(),
                         target=horizon_target.float())
        if determine_metrics:
            metrics = calc_metrics(output=horizon_output, target=horizon_target, prefix=f"{phase}_")
            metrics.update({"loss": loss})
            metrics.update({"memory_during_test": torch.cuda.memory_reserved() / (1024 ** 2)})

            self.log_dict(metrics, batch_size=input_data.shape[0])
        if phase == "val":
            self.log("val_loss", loss, prog_bar=True)
        else:
            self.log_dict({phase + "_loss": loss, "loss": loss}, batch_size=input_data.shape[0])

        return loss
