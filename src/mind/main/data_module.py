"""
This module defines the MINDDataModule class for loading the MIND dataset within a PyTorch Lightning framework:

- `__init__`: Initializes the class with dataset parameters, tokenizer, and base model.
- `setup`: Prepares the training and validation datasets.
- `train_dataloader`: Returns a DataLoader for the training dataset, with specific configurations like batch size, shuffling, and number of workers.
- `val_dataloader`: Returns a DataLoader for the validation dataset with similar configurations but without shuffling.
"""

from multiprocessing import cpu_count
from typing import Optional, Union, Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mind.main.dataset import (
    MINDDatasetTrain,
    MINDDatasetVal,
    get_train_dataset,
    get_val_dataset,
    MINDCollateTrain,
    MINDCollateVal,
)
from mind.params import DataParams


class MINDDataModule(pl.LightningDataModule):
    def __init__(self, params: DataParams, base_model: str):
        super().__init__()
        self.params = params
        self.train_dataset: Optional[MINDDatasetTrain] = None
        self.val_dataset: Optional[MINDDatasetVal] = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            params.pretrained_model_name,
            use_fast=True,
        )
        self.base_model = base_model

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = get_train_dataset(
            base_dir=self.params.mind_path, base_model=self.base_model
        )  # MINDDatasetTrain
        self.val_dataset = get_val_dataset(  # MINDDatasetVal
            base_dir=self.params.mind_path,
            # This tokenizer must be different instance from others.
            tokenizer=AutoTokenizer.from_pretrained(
                self.params.pretrained_model_name,
                use_fast=False,
            ),
            base_model=self.base_model,
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            collate_fn=MINDCollateTrain(self.tokenizer, self.base_model),
            shuffle=True,
            # num_workers=cpu_count(),
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(
        self, *args, **kwargs
    ) -> Union[DataLoader, Sequence[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            collate_fn=MINDCollateVal(self.base_model),
            shuffle=False,
            # num_workers=cpu_count(),
            num_workers=2,
            pin_memory=True,
        )
