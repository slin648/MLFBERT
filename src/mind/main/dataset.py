"""
dataset.py: Dataset and collate functions for MIND (Microsoft News Dataset).

This module defines datasets and collate functions to handle data processing 
and batching for the Microsoft News Dataset (MIND).

Classes:
    - MINDDatasetTrain: Dataset for training data.
    - MINDDatasetVal: Dataset for validation (and test) data.
    - MINDCollateTrain: Collate function for the training set.
    - MINDCollateVal: Collate function for the validation/test set.

Functions:
    - _make_batch_assignees: Utility function to create batch assignees.
    - _load_df: Load dataframes (behaviors and news) based on the given split (train/valid/test).
    - get_train_dataset: Load the training dataset.
    - get_val_dataset: Load the validation dataset.
    - get_test_dataset: Load the test dataset.

Note:
    1. The datasets handle sampling, tokenization, and feature extraction.
    2. The collate functions deal with batching and additional processing.
    3. Data are handled differently based on the model type (e.g., Transformer vs. Word2Vec).
"""

from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Sequence, Mapping, Union, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from mind.main.batch import (
    MINDBatch,
    ContentsEncoded,
    Word2VecMINDBatch,
    Word2VecContentsEncoded,
)
from mind.dataframe import load_behaviours_df, load_news_df


@dataclass
class MINDDatasetTrain(Dataset):
    df_behaviours: pd.DataFrame
    df_news: pd.DataFrame
    base_model: str
    n_neg: int = 4  # K=4
    hist_size: int = 50  # history size

    def __getitem__(self, idx):
        bhv = self.df_behaviours.iloc[idx]

        histories = bhv["histories"]
        candidates = bhv["candidates"]
        labels = bhv["labels"]

        histories = np.random.permutation(histories)[
            : self.hist_size
        ]  # Randomly select 50 historical news, there may not be 50, there is no padding here
        candidates, labels = self._sample_candidates(candidates, labels)

        histories = self.df_news.loc[
            histories
        ]  # Select the features of historical news (n_id), which is actually a dataframe
        candidates = self.df_news.loc[
            candidates
        ]  # Select features for candidate news (n_id)
        # print("labels:", labels)  # labels: [False False False  True False]
        labels = labels.argmax()
        # print("after, labels:", labels)  # labels: 3

        return histories, candidates, labels

    def __len__(self):
        return len(self.df_behaviours)
        # return 100  # TODO for debug

    def _sample_candidates(self, candidates, labels):
        """
        negative sampling
        """
        pos_id = np.random.permutation(np.where(labels)[0])[0]

        neg_ids = np.array([]).astype(int)
        while len(neg_ids) < self.n_neg:
            neg_ids = np.concatenate(
                (
                    neg_ids,
                    np.random.permutation(np.where(~labels)[0]),
                )
            )
        neg_ids = neg_ids[: self.n_neg]  #  Sample K negative samples

        indices = np.concatenate(([pos_id], neg_ids))
        indices = np.random.permutation(indices)
        candidates = np.array(candidates)[indices]
        labels = labels[indices]
        return candidates, labels


@dataclass
class MINDDatasetVal(Dataset):
    df_behaviours: pd.DataFrame
    df_news: pd.DataFrame
    tokenizer: InitVar[PreTrainedTokenizer]
    base_model: str
    hist_size: int = 50
    news_feature_map: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self, tokenizer):
        self._uniq_news_inputs = self._make_uniq_news_inputs(
            tokenizer
        )  # Call this method after __init__() to count all nids

    def __getitem__(self, idx):
        assert (
            self.news_feature_map
        ), "news_feature_map is empty. Set it before to get an item."

        bhv = self.df_behaviours.iloc[idx]

        # TODO consider more
        histories = bhv["histories"][: self.hist_size]
        candidates = bhv["candidates"]
        labels = bhv["labels"]

        # Use precomputed features.
        histories = torch.stack(
            [self.news_feature_map[idx] for idx in histories], dim=0
        )
        candidates = torch.stack(
            [self.news_feature_map[idx] for idx in candidates], dim=0
        )
        # print("Valid labels: ", labels)  # Negative sampling not done [ True False False False False False False False]
        return histories, candidates, labels

    def __len__(self):
        return len(self.df_behaviours)
        # return 100  # TODO for debug

    def init_dummy_feature_map(self, dim: int):
        self.news_feature_map = {
            k: torch.randn(dim) for k, v in self.uniq_news_inputs.items()
        }

    @property
    def uniq_news_inputs(self):
        return self._uniq_news_inputs

    def _make_uniq_news_inputs(self, tokenizer):
        """
        Segment the n_id of the union between all click histories and all candidates
        """
        print("Valid, self.df_behaviours.shape: ", self.df_behaviours.shape)  #
        # print(self.df_behaviours.columns)  # Index(['histories', 'candidates', 'labels'], dtype='object')
        histories = np.concatenate(self.df_behaviours["histories"].values)
        candidates = np.concatenate(self.df_behaviours["candidates"].values)
        # print("histories.shape: ", histories.shape)  # (2362514,)
        # print("candidates.shape: ", candidates.shape)  # (2658091,)

        indices = set(histories) | set(
            candidates
        )  # Find the n_id union between all click history and all candidates
        inputs = self.df_news.loc[list(indices)].to_dict("records")
        if "Word2Vec" not in self.base_model:
            inputs = [
                {
                    "title": tokenizer(
                        x["title"],
                        x["category"],
                        # x['abstract'],
                        return_tensors="pt",
                        return_token_type_ids=False,
                        truncation=True,
                    ),
                }
                for x in inputs
            ]
        else:
            inputs = [
                {
                    "title": torch.from_numpy(
                        np.array(
                            [int(item) for item in x["title_id"].split(";")]
                            + [int(item) for item in x["category_id"].split(";")]
                        ).astype(np.int64)
                    ),  # Must be Tensor, news_feature_map defined
                }
                for x in inputs
            ]

        return dict(zip(indices, inputs))


def _make_batch_assignees(items: Sequence[Sequence[Any]]) -> torch.Tensor:
    sizes = torch.tensor([len(x) for x in items])
    batch = torch.repeat_interleave(torch.arange(len(items)), sizes)
    return batch


@dataclass
class MINDCollateTrain:
    tokenizer: PreTrainedTokenizer
    base_model: str

    def __call__(self, batch):
        histories, candidates, targets = zip(*batch)

        batch_hist = _make_batch_assignees(histories)
        batch_cand = _make_batch_assignees(candidates)

        x_hist = self._tokenize_df(pd.concat(histories), self.base_model)  # dict
        x_cand = self._tokenize_df(pd.concat(candidates), self.base_model)

        if "Word2Vec" not in self.base_model:
            return MINDBatch(
                batch_hist=batch_hist,
                batch_cand=batch_cand,
                x_hist=x_hist,
                x_cand=x_cand,
                targets=torch.tensor(targets),
            )
        else:
            return Word2VecMINDBatch(
                batch_hist=batch_hist,
                batch_cand=batch_cand,
                x_hist=x_hist,
                x_cand=x_cand,
                targets=torch.tensor(targets),
            )

    def _tokenize(self, x: List[str]):
        return self.tokenizer(
            x,
            return_tensors="pt",
            return_token_type_ids=False,
            padding=True,
            truncation=True,
        )

    def _tokenize_df(self, df: pd.DataFrame, base_model: str):
        def parse(lst):
            """
            Each element in the list ['18;19;20;21;22;23;24;22;25;23;26;27;28', '2']
            """
            new_lst = []
            for item in lst:
                new_str = ";".join(item)
                new_lst.append(
                    [int(wid) for wid in new_str.split(";")]
                )  # each length is different
            # dynamic batching
            max_len = max([len(item) for item in new_lst])
            padded_lst = []
            for item in new_lst:
                padded_lst.append(item + [1] * (max_len - len(item)))
            return torch.from_numpy(np.array(padded_lst)).long()

        if "Word2Vec" not in base_model:
            return {
                "title": self._tokenize(df[["title", "category"]].values.tolist()),
            }
        else:
            return {"title": parse(df[["title_id", "category_id"]].values.tolist())}


@dataclass
class MINDCollateVal:
    base_model: str
    is_test: bool = False  # Is it a real test set, no ground truth

    def __call__(self, batch):
        # It gets precomputed inputs.
        histories, candidates, targets = zip(*batch)

        batch_hist = _make_batch_assignees(histories)
        batch_cand = _make_batch_assignees(candidates)

        x_hist = torch.cat(
            histories
        )  # This is different from Train, the random vector used
        # print("x_hist.shape: ", x_hist.shape)  # (L, 300), not (B, L, 300), because there is no padding for the length
        x_cand = torch.cat(candidates)
        # print("x_cand.shape: ", x_cand.shape)  # (L, 300)
        if self.is_test:
            targets = None
        else:
            targets = np.concatenate(targets)
            targets = torch.from_numpy(targets)
        if "Word2Vec" not in self.base_model:
            return MINDBatch(
                batch_hist=batch_hist,
                batch_cand=batch_cand,
                x_hist=x_hist,
                x_cand=x_cand,
                targets=targets,
            )
        else:
            return Word2VecMINDBatch(
                batch_hist=batch_hist,
                batch_cand=batch_cand,
                x_hist=x_hist,
                x_cand=x_cand,
                targets=targets,
            )


def _load_df(base_dir: Union[str, Path], base_model: str, split: str):
    df_b = load_behaviours_df(base_dir)
    df_b = df_b[df_b["split"] == split].reset_index(drop=True)
    df_b = df_b[["histories", "candidates", "labels"]]

    df_n = load_news_df(base_dir, base_model)  # notice Word2Vec version NRMS
    df_n["category"] = df_n["category"] + " > " + df_n["subcategory"]

    if "Word2Vec" not in base_model and "Abstract" not in base_model:
        df_n = df_n[["title", "category"]]  # n_id is index
    elif "Word2Vec" not in base_model and "Abstract" in base_model:
        df_n = df_n[
            [
                "title",
                "category",
                "abstract",
            ]
        ]
    else:
        df_n = df_n[["title", "category", "category_id", "title_id", "abstract_id"]]

    return df_b, df_n


def get_train_dataset(base_dir: Union[str, Path], base_model: str):
    df_b, df_n = _load_df(base_dir, base_model, "train")  # behavior and news

    return MINDDatasetTrain(
        df_behaviours=df_b,
        df_news=df_n,
        base_model=base_model,
    )


def get_val_dataset(
    base_dir: Union[str, Path], tokenizer: PreTrainedTokenizer, base_model
) -> MINDDatasetVal:
    df_b, df_n = _load_df(base_dir, base_model, "valid")
    print("validation set: len(df_b): ", len(df_b), "len(df_n): ", len(df_n))

    return MINDDatasetVal(
        df_behaviours=df_b,
        df_news=df_n,
        tokenizer=tokenizer,
        base_model=base_model,
    )


def get_test_dataset(
    base_dir: Union[str, Path], tokenizer: PreTrainedTokenizer
) -> MINDDatasetVal:
    df_b, df_n = _load_df(base_dir, "MLFBERT", "test")

    return MINDDatasetVal(
        df_behaviours=df_b,
        df_news=df_n,
        tokenizer=tokenizer,
        base_model="MLFBERT",
    )
