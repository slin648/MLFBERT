"""
This module provides functionalities to load and preprocess datasets related to user behaviors and news information.
- load_behaviours_df: Loads and pre-processes user behavior data, including training, validation, and testing sets.
- load_news_df: Loads and concatenates news data based on the given model type.
- load_popularity_df: Aggregates information to generate a popularity dataframe.
- load_popularity_df_test: Constructs a test dataframe for popularity evaluation.
Several helper functions are also defined to assist with loading data in different formats.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from libs.pandas.cache import pd_cache


def load_behaviours_df(base_dir: Union[Path, str]) -> pd.DataFrame:
    """Data preprocessing, convert data to pandas.DataFrame format"""
    base_dir = Path(base_dir)
    # 'b_id', 'u_id', 'time', 'histories', 'impressions'
    df_train = _load_behaviours_df(base_dir / "train/behaviors.tsv")
    df_val = _load_behaviours_df(base_dir / "valid/behaviors.tsv")
    if (base_dir / "test").exists():
        df_test = _load_behaviours_df(base_dir / "test/behaviors.tsv")
    else:
        df_test = pd.DataFrame()

    df_train["split"] = "train"
    df_val["split"] = "valid"
    df_test["split"] = "test"

    df: pd.DataFrame = pd.concat((df_train, df_val, df_test), ignore_index=True)

    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
    df = df[df["histories"].notna()].reset_index(drop=True)  #
    df = df[df["histories"] != ""].reset_index(drop=True)  #
    df = df[df["histories"] != " "].reset_index(drop=True)  #
    # df['histories'] = df['histories'].fillna('').str.split()  # Users with no history, filled with an empty string
    df["histories"] = df["histories"].str.split()  # Note that split() will str -> list
    print("df.shape:", df.shape)

    # Candidates is the part of the impression that removes -1 and -0
    df["candidates"] = (
        df["impressions"].str.strip().str.replace(r"-[01]", "", regex=True).str.split()
    )
    df["labels"] = None  # Test has no labels.
    df.loc[df["split"] != "test", "labels"] = (
        df[df["split"] != "test"]["impressions"]
        .str.strip()
        .str.replace(r"N\d*-", "", regex=True)
        .str.split()
        .apply(lambda x: np.array(x).astype(bool))
    )
    df = df.drop(columns=["impressions"])
    print("df.shape:", df.shape)
    # if drop_no_hist:  # Whether to remove users without historical information
    df = df[df["histories"].apply(len) > 0].reset_index(drop=True)
    print("after remove cold-start users, df.shape:", df.shape)
    return df


def _load_behaviours_df(tsv_path: Union[Path, str]):
    df = pd.read_table(
        tsv_path,
        header=None,
        names=[
            "b_id",
            "u_id",
            "time",
            "histories",
            "impressions",
        ],
    )
    return df


def load_news_df(base_dir: Union[Path, str], base_model) -> pd.DataFrame:
    base_dir = Path(base_dir)

    df_train = (
        _load_news_df(base_dir / "train/news.tsv")
        if "Word2Vec" not in base_model
        else _load_news_df_word2vec_version(base_dir / "train/news_numerized.parquet")
    )
    df_val = (
        _load_news_df(base_dir / "valid/news.tsv")
        if "Word2Vec" not in base_model
        else _load_news_df_word2vec_version(base_dir / "valid/news_numerized.parquet")
    )
    if (base_dir / "test").exists():
        df_test = (
            _load_news_df(base_dir / "test/news.tsv")
            if "Word2Vec" not in base_model
            else _load_news_df_word2vec_version(
                base_dir / "test/news_numerized.parquet"
            )
        )
    else:
        df_test = pd.DataFrame()

    df: pd.DataFrame = pd.concat((df_train, df_val, df_test), ignore_index=True)
    df = df.drop_duplicates(subset=["n_id"])  # 'n_id' is the news id, a required field
    df = df.set_index("n_id", drop=True)

    df["abstract"] = df["abstract"].fillna("")

    return df


def _load_news_df(tsv_path):
    df = pd.read_table(
        tsv_path,
        header=None,
        names=[
            "n_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
        ],
        usecols=range(6),
    )
    return df


def _load_news_df_word2vec_version(tsv_path):
    """
    If it is a word2vec version model, use this function to read news_numerized.parquet data
    """
    df = pd.read_parquet(tsv_path)
    df.columns = [
        "n_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "title_entities",
        "abstract_entities",
        "category_id",
        "title_id",
        "abstract_id",
    ]

    df = df[
        [
            "n_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "category_id",
            "title_id",
            "abstract_id",
        ]
    ]
    df["category_id"] = df["category_id"].fillna("1")
    df["title_id"] = df["title_id"].fillna("1")

    return df


def load_popularity_df(base_dir) -> pd.DataFrame:
    df_b = load_behaviours_df(base_dir, drop_no_hist=False)
    df_b = df_b[df_b["split"] != "test"]

    df_imp_flat = pd.concat(
        (
            df_b["candidates"].explode(),
            df_b["labels"].explode(),
        ),
        axis=1,
    ).reset_index(drop=True)
    df_imp_flat["labels"] = df_imp_flat["labels"].astype(np.uint8)

    df_p: pd.DataFrame = df_imp_flat.groupby("candidates").agg({"labels": [np.mean]})
    df_p.columns = ["popularity"]

    return df_p


def load_popularity_df_test(base_dir) -> pd.DataFrame:
    df_b = load_behaviours_df(base_dir, drop_no_hist=False)
    df_b = df_b[df_b["histories"].apply(len) == 0]
    df_b = df_b[df_b["split"] == "test"]

    df = pd.DataFrame(index=df_b["candidates"].explode().unique())
    df["popularity"] = None

    return df
