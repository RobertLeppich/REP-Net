from torchmetrics.functional.regression import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from torchmetrics.functional.classification import f1_score, precision, recall
import torch
import numpy as np

def rescale(data, scaler, meta):
    result = []
    for i in range(len(data)):
        batch = data[i]
        scale = scaler[meta[i]] if meta is not None else scaler
        result.append(scale.inverse_transform(batch.view(-1, batch.shape[1])).reshape(batch.shape))
    return torch.tensor(np.stack(result))

def calc_metrics(output=None, target=None, prefix=""):

    output, target = output.reshape(-1), target.reshape(-1)
    ts_loss = {
        # missing
        prefix + "mae_missing": float(mean_absolute_error(output, target)),
        prefix + "mse_missing": float(mean_squared_error(output, target)),
        prefix + "rmse_missing": float(torch.sqrt(mean_squared_error(output, target))),
    }

    return ts_loss
