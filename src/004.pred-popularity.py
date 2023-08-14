"""
This script computes and predicts the popularity of news articles based on user behaviors.
It includes the following functionalities:

1. `_load_behaviours_df`: Reads the behaviors data from a TSV file.
2. `compute_news_popularity`: Calculates the global click-through rate for each news item.
3. `compute_news_click_ratio`: Computes the average probability of a click per recommendation for each news item.
4. `load_behaviours_df`: Processes and converts data into pandas DataFrame format, handling training, validation, and testing splits.
5. `main`: Orchestrates the loading of the behaviors data, computation of click ratios, and saving of predictions.

The script reads data from the specified behaviors path and produces a Parquet file containing the popularity predictions.
"""

import os
import copy
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from pytorch_lightning.metrics.functional import auroc
from libs.torch.metrics import ndcg_score, mrr_score
import warnings

warnings.filterwarnings("ignore")


def _load_behaviours_df(tsv_path):
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


def compute_news_popularity(behaviors_path):
    """
    Global Click-Through Rate
    """
    df = _load_behaviours_df(behaviors_path)
    # print(df.shape)
    # print(df[["histories", "impressions"]].head(3))
    news2proba = {}
    click_num = 0
    for idx, row in df.iterrows():
        # print(row["impressions"].split())  # ['N55689-1', 'N35729-0']
        # break
        click_num += len(row["impressions"].split())
        for impression in row["impressions"].split():
            news_id = impression.split("-")[0]
            if news_id not in news2proba:
                news2proba[news_id] = 0
            news2proba[news_id] += 1
    # print("click_num:", click_num)
    for k, v in news2proba.items():
        news2proba[k] = v / click_num

    return news2proba


def compute_news_click_ratio(behaviors_path):
    """
    Average Probability of Click per Recommendation
    """
    df = _load_behaviours_df(behaviors_path)
    # print(df.shape)
    # print(df[["histories", "impressions"]].head(3))
    news2proba = {}
    news2impression_num = {}
    # click_num = 0
    for idx, row in df.iterrows():
        # print(row["impressions"].split())  # ['N55689-1', 'N35729-0']
        # break
        # click_num += len(row["impressions"].split())
        for impression in row["impressions"].split():
            news_id = impression.split("-")[0]
            if news_id not in news2impression_num:
                news2impression_num[news_id] = 0
            news2impression_num[news_id] += 1

            is_click = int(impression.split("-")[1])
            if is_click == 1:
                if news_id not in news2proba:
                    news2proba[news_id] = 0
                news2proba[news_id] += 1

    # print("click_num:", click_num)
    for k, v in news2proba.items():
        news2proba[k] = v / news2impression_num[k]

    return news2proba


def load_behaviours_df(base_dir):
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

    df_copy = copy.deepcopy(df)

    df = df[df["histories"].notna()].reset_index(drop=True)  #
    df = df[df["histories"] != ""].reset_index(drop=True)  #
    df = df[df["histories"] != " "].reset_index(drop=True)  #
    # df['histories'] = df['histories'].fillna('').str.split()
    # Users with no history, filled with an empty string
    df["histories"] = df["histories"].str.split()  # Note that split() will str -> list
    # print("df.shape:", df.shape)

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
    # print("df.shape:", df.shape)
    # if drop_no_hist:  # Whether to remove users without historical information
    df = df[df["histories"].apply(len) > 0].reset_index(drop=True)
    # print("after remove cold-start users, df.shape:", df.shape)

    # Process df_copy
    df_copy["candidates"] = (
        df_copy["impressions"]
        .str.strip()
        .str.replace(r"-[01]", "", regex=True)
        .str.split()
    )
    df_copy["labels"] = None  # Test has no labels.
    df_copy.loc[df_copy["split"] != "test", "labels"] = (
        df_copy[df_copy["split"] != "test"]["impressions"]
        .str.strip()
        .str.replace(r"N\d*-", "", regex=True)
        .str.split()
        .apply(lambda x: np.array(x).astype(bool))
    )
    df_copy = df_copy.drop(columns=["impressions"])

    return df, df_copy


def main():
    base_dir = "../MIND_DATASET"
    news2proba = compute_news_click_ratio(os.path.join(base_dir, "train/behaviors.tsv"))

    df_clean, df_whole = load_behaviours_df(base_dir)
    split = "test"
    df_clean = df_clean[df_clean["split"] == split].reset_index(drop=True)
    df_whole = df_whole[df_whole["split"] == split].reset_index(drop=True)

    b_id_whole = set(list(df_whole["b_id"].values))
    b_id_clean = set(list(df_clean["b_id"].values))

    b_id_no_history = b_id_whole - b_id_clean
    # users without history
    df = df_whole[df_whole["b_id"].isin(b_id_no_history)].reset_index(drop=True)
    # print("valid size:", len(df))

    sub_rows = []
    predictions = []
    b_ids = []
    # labels = []
    for idx, row in df.iterrows():
        b_ids.append(row["b_id"])
        cur_predictions = []
        for nid in row["candidates"]:
            if nid in news2proba:
                cur_predictions.append(news2proba[nid])
            else:
                # predictions.append(0.5)  # append 0 or 0.5
                cur_predictions.append(0.0)  # append 0 or 0.5
        # for t in row["labels"]:
        #     labels.append(t)

        predictions.append(
            "[{}]".format(
                ",".join(
                    list(  # .astype(str)
                        (np.argsort(np.array(cur_predictions) * -1) + 1).astype(str)
                    )
                )
            )
        )
        sub_rows.append(f"{b_ids[-1]} {predictions[-1]}")

    df_popular = pd.DataFrame(
        index=b_ids,
        data=sub_rows,
        columns=["preds"],
    )

    out_dir = Path("../predictions")
    out_dir.mkdir(exist_ok=True)

    df_popular.to_parquet(out_dir / "sub_popularity.pqt")


if __name__ == "__main__":
    main()
