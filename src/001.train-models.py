"""
This script serves as the entry point for training and validation of various neural recommendation models.

1. Import Statements: The necessary libraries, including PyTorch, PyTorch Lightning, and custom-defined modules and classes, are imported.

2. AverageMeterSet: A data class encapsulating average meters to track training and validation metrics like loss, ROC, MRR, and NDCG.

3. PLModule: A class defining the training structure for a PyTorch Lightning model, supporting various base models like 
NRMS, PLMNR, FusionNR, and MLFBERT. Methods are implemented for training, validation, optimizer configuration, etc.

4. Training Logic: The script includes the training loop with logging, metrics calculation, validation, checkpointing, 
and tensorboard integration. Special functions handle training steps, validation steps, and end-of-epoch events.

5. Main Execution Block: If executed directly, the script configures logging, loads parameters, 
and initiates the training process by calling the train function with appropriate parameters and data module.

This file encapsulates the main components and overall functionality for the training and validation procedures of the specified models.
"""

from dataclasses import dataclass
from functools import cached_property
from logging import getLogger, FileHandler
from pathlib import Path
from time import time
from typing import Dict, Sequence, Any, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import auroc
from pytorch_lightning.utilities import move_data_to_device
from torch.optim import Adam
from tqdm import tqdm

from libs.pytorch_lightning.logging import configure_logging
from libs.torch.avg_meter import AverageMeter
from libs.torch.metrics import ndcg_score, mrr_score
from mind.main.data_module import MINDDataModule
from mind.main.dataset import MINDDatasetVal
from mind.params import ModuleParams, Params
from models.expmodels import (
    NRMS,
    PLMNR,
    FusionNR,
    MLFBERT,
)


@dataclass(frozen=True)
class AverageMeterSet:
    train_loss: AverageMeter = AverageMeter()
    val_loss: AverageMeter = AverageMeter()
    val_roc: AverageMeter = AverageMeter()
    val_mrr: AverageMeter = AverageMeter()
    val_ndcg10: AverageMeter = AverageMeter()
    val_ndcg5: AverageMeter = AverageMeter()


class PLModule(pl.LightningModule):
    def __init__(self, hparams: Dict):
        super().__init__()
        self.hparams = hparams
        # print("type(hparams):", type(hparams))  # dict
        if self.hparams["base_model"] == "PLMNR":
            self.model = PLMNR(
                pretrained_model_name=self.hp.pretrained_model_name,
                sa_pretrained_model_name=self.hp.sa_pretrained_model_name,
            )
        elif self.hparams["base_model"] == "FusionNR":
            self.model = FusionNR(
                pretrained_model_name=self.hp.pretrained_model_name,
                sa_pretrained_model_name=self.hp.sa_pretrained_model_name,
            )
        elif self.hparams["base_model"] == "MLFBERT":
            self.model = MLFBERT(
                pretrained_model_name=self.hp.pretrained_model_name,
                sa_pretrained_model_name=self.hp.sa_pretrained_model_name,
            )
        elif self.hparams["base_model"] == "NRMS":
            self.model = NRMS(
                wordembedding_path=self.hp.wordembedding_path,
            )
        else:
            raise ValueError(f"Unknown base_model: {self.hparams['base_model']}")

        self.am = AverageMeterSet()
        self.total_processed = 0

    def training_step(self, batch, batch_idx):
        loss, y_score = self.model.forward(batch)
        with torch.no_grad():
            n_processed = batch["batch_cand"].max() + 1
            self.am.train_loss.update(loss.detach(), n_processed)
            self.total_processed += n_processed
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, y_score = self.model.forward(batch)
        y_true = batch["targets"]
        n_processed = batch["batch_cand"].max() + 1

        for n in range(n_processed):
            mask = batch["batch_cand"] == n
            s, t = y_score[mask], y_true[mask]
            s = torch.softmax(s, dim=0)
            # print("s.shape:", s.shape)  # torch.Size([L])
            self.am.val_roc.update(auroc(s, t))
            self.am.val_mrr.update(mrr_score(s, t))
            self.am.val_ndcg10.update(ndcg_score(s, t))
            self.am.val_ndcg5.update(ndcg_score(s, t, k=5))

        self.am.val_loss.update(loss, n_processed)

    def training_epoch_end(self, outputs: Sequence[Any]):
        self.log("train_loss", self.am.train_loss.compute())

    @torch.no_grad()
    def validation_epoch_end(self, outputs: Sequence[Any]):
        self.log("val_loss", self.am.val_loss.compute())
        self.log("val_roc", self.am.val_roc.compute())
        self.log("val_mrr", self.am.val_mrr.compute())
        self.log("val_ndcg10", self.am.val_ndcg10.compute())
        self.log("val_ndcg5", self.am.val_ndcg5.compute())

    @torch.no_grad()
    def on_validation_epoch_start(self):
        # Pre compute feature of uniq candidates in val to save time.
        val_dataset = cast(MINDDatasetVal, self.val_dataloader().dataset)

        if self.total_processed == 0:  # just placeholder
            val_dataset.init_dummy_feature_map(self.model.encoder.dim)
            return

        encoder = self.model.encoder.eval()
        inputs = val_dataset.uniq_news_inputs
        feats = {
            k: encoder.forward(move_data_to_device(v, self.device)).squeeze().cpu()
            for k, v in tqdm(inputs.items(), desc="Encoding val candidates")
        }
        val_dataset.news_feature_map = feats  # assign to dataset

    def configure_optimizers(self):
        opt = Adam(
            self.model.parameters(),
            lr=self.hp.lr,
            weight_decay=self.hp.weight_decay,
        )
        return [opt]

    @cached_property
    def hp(self) -> ModuleParams:
        return ModuleParams.from_dict(dict(self.hparams))


def train(params: Params):
    seed_everything(params.d.seed)

    tb_logger = TensorBoardLogger(
        params.t.save_dir,
        name=f"expmodels",
        version=str(int(time())),
    )

    log_dir = Path(tb_logger.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = getLogger("lightning")
    logger.addHandler(FileHandler(log_dir / "train.log"))
    logger.info(params.pretty())

    callbacks = [
        # LearningRateMonitor(),
    ]
    if params.t.checkpoint_callback:
        callbacks.append(
            ModelCheckpoint(
                monitor="val_roc",
                save_last=True,
                verbose=True,
                mode="max",
            )
        )

    trainer = pl.Trainer(
        max_epochs=params.t.epochs,
        gpus=params.t.gpus,
        tpu_cores=params.t.num_tpu_cores,
        logger=tb_logger,
        precision=params.t.precision,
        resume_from_checkpoint=params.t.resume_from_checkpoint,
        weights_save_path=params.t.weights_save_path,
        checkpoint_callback=params.t.checkpoint_callback,
        callbacks=callbacks,
        deterministic=True,
        benchmark=True,
        accumulate_grad_batches=params.t.accumulate_grad_batches,
        val_check_interval=params.t.val_check_interval,
        gradient_clip_val=0.5,  #
    )
    net = PLModule(params.m.to_dict())  # set model
    dm = MINDDataModule(params.d, params.m.base_model)  # set datamodule

    trainer.fit(net, datamodule=dm)


if __name__ == "__main__":
    configure_logging()
    params = Params.load()
    print("params: ", params)
    train(params)
