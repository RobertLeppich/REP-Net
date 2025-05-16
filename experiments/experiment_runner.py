import torch
from lightning.pytorch.callbacks import DeviceStatsMonitor
from architecture.model import REPNetForecasting
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import lightning.pytorch as pl
import numpy as np
import os
import time
from datetime import timedelta
import json
import yaml
import hashlib
import random
from data_provider.data_factory import data_provider

torch.set_float32_matmul_precision("medium")

class ExperimentRun:

    def __init__(self, name: str, config: dict, exp_base_config: dict=None):

        # load default experimental setup
        if exp_base_config is None:
            exp_base_config = yaml.safe_load(open("experiments/configs/experiments_setup.yaml", "r"))

        # update task specific config
        config.update(exp_base_config)
        self.exp_run_config = config
        self.name = name

        self.run_id = self.generate_run_id()

    def generate_run_id(self, phase=None):
        h = hashlib.md5()
        h.update((phase or self.name).encode())
        h.update(json.dumps(self.exp_run_config, sort_keys=True).encode())

        return h.hexdigest()

    def create_log_path(self, checkpoint_path="model_checkpoints", name=None, run_id=None):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        return os.path.join(checkpoint_path, name or self.name,
                            run_id or self.run_id)

    def set_seeds(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def run(self, dry_run=False, speed_run=False):
        exp_run_config = self.exp_run_config

        exp_run_config["dry_run"] = dry_run
        exp_run_config["name"] = self.name
        exp_run_config["num_workers"] = exp_run_config.get("num_workers", 2) if not dry_run else 0
        self.set_seeds(exp_run_config["shuffle_seed"])

        try:

            print(
                f"Start {self.name}: ({self.run_id}) \nconfig:{str(exp_run_config)}")
            log_path = self.create_log_path()
            start_time = time.time()
            metrics = self.run_config(exp_run_config, path=log_path, speed_run=speed_run)
            elapsed_time = time.time() - start_time

            print(
                f"Finished {self.name}: ({self.run_id})\n{str(metrics)}\nElapsed Time: {elapsed_time}”")

        except Exception as se:
            if dry_run:
                raise se
            print(f"ERROR: {se}")

    def run_config(self, config, path: str, speed_run=False):

        train_data, train_loader = data_provider(config, flag="train")
        validation_data, validation_loader = data_provider(config, flag="val")
        test_data, test_loader = data_provider(config, flag="test")

        config["train_steps"] = len(train_loader)

        model = REPNetForecasting(config=config)

        print(f"Trainable Parameter: {ModelSummary(model).trainable_parameters}")

        early_stopping = EarlyStopping(monitor="val_loss" if not speed_run else "loss",
                                       min_delta=config["earlystopping_min_delta"],
                                       patience=config["earlystopping_patience"], verbose=True, mode="min")
        checkpoint_callback = ModelCheckpoint(dirpath=path if not config["dry_run"] else "dry_run_" + path,
                                              save_top_k=1, monitor="val_loss", every_n_epochs=1)
        timer_callback = Timer(duration=timedelta(days=2))

        trainer = pl.Trainer(devices=1, accelerator="gpu",
                             precision="16-mixed",
                             max_epochs=20 if not speed_run else 1,
                             max_steps=-1 if not speed_run else 500,
                             gradient_clip_val=0.5,
                             callbacks=([early_stopping, timer_callback, checkpoint_callback])
                             if not config["dry_run"] else [early_stopping, checkpoint_callback],
                             default_root_dir=path, profiler=None if config["dry_run"] else None)

        model.training = True
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

        model.training = False
        if not speed_run:
            trainer.test(ckpt_path="best", dataloaders=test_loader)
        else:
            trainer.test(model, dataloaders=test_loader)

        mets = {k: v.item() for k, v in trainer.logged_metrics.items()}
        mets.update({"version": trainer.logger.version if trainer.logger is not None else None,
                     "train_duration": timer_callback.time_elapsed("train"),
                     "validation_duration": timer_callback.time_elapsed("validate"),
                     "test_duration": timer_callback.time_elapsed("test"),
                     "model_size": torch.cuda.memory_reserved() / (1024 ** 2),
                     "trainable_parameter": ModelSummary(model).trainable_parameters
                     })
        if "loss" not in mets.keys():
            raise IOError("No loss in metrics")

        # mlflow.log_metrics(mets)
        [log.finalize("success") for log in trainer.loggers]

        torch.cuda.empty_cache()
        return mets
