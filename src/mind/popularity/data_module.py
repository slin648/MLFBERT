"""
The PopularityDataModule class handles the preparation and loading of datasets for popularity-based recommendation.
It includes methods to set up training and validation datasets, along with data loaders for training and validation.
"""

from multiprocessing import cpu_count
from typing import Optional, Union, Sequence

import pytorch_lightning as pl
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mind.dataframe import load_popularity_df, load_news_df
from mind.params import DataParams
from mind.popularity.dataset import PopularityDataset, PopularityCollate


# noinspection PyAbstractClass
class PopularityDataModule(pl.LightningDataModule):
    def __init__(self, params: DataParams):
        super().__init__()
        self.params = params
        self.train_dataset: Optional[PopularityDataset] = None
        self.val_dataset: Optional[PopularityDataset] = None
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def setup(self, stage: Optional[str] = None):
        df_p = load_popularity_df(self.params.mind_path)
        df_p = df_p[df_p["popularity"] > 0]
        df_n = load_news_df(self.params.mind_path)
        df = df_p.merge(df_n, left_index=True, right_index=True, how="left")

        if self.params.train_all:
            df_train = df
            df_val = df.iloc[:10]
        else:
            kf = KFold(
                n_splits=self.params.n_splits,
                random_state=self.params.seed,
                shuffle=True,
            )
            train_idx, val_idx = list(kf.split(df))[self.params.fold]
            df_train = df.iloc[train_idx]
            df_val = df.iloc[val_idx]

        self.train_dataset = PopularityDataset(df_train)
        self.val_dataset = PopularityDataset(df_val)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            collate_fn=PopularityCollate(self.tokenizer),
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
            collate_fn=PopularityCollate(self.tokenizer),
            shuffle=False,
            # num_workers=cpu_count(),
            num_workers=2,
            pin_memory=True,
        )
