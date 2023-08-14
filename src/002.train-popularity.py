"""
This script focuses on computing and analyzing the popularity of news articles based on user click-through behaviors.

1. Import Statements: Necessary libraries and modules are imported, including os, pandas, numpy, PyTorch, and specific metrics.

2. Data Loading Functions:
   - _load_behaviours_df: A helper function to load behaviors data from a given TSV file.
   - load_behaviours_df: A function to load and preprocess behaviors data for training, validation, and testing.

3. Popularity Computation Functions:
   - compute_news_popularity: A function that computes the global click-through rate for each news article.
   - compute_news_click_ratio: A function that calculates the average probability of a click per recommendation for each news article.

4. Main Execution Block:
   - The main part of the script computes click-through rates for the news articles, preprocesses the dataset, and splits it into validation sets.
   - It then computes predictions and evaluates them using AUC (Area Under the Receiver Operating Characteristic Curve), MRR (Mean Reciprocal Rank), and NDCG (Normalized Discounted Cumulative Gain) at different levels.
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
    split = "valid"
    df_clean = df_clean[df_clean["split"] == split].reset_index(drop=True)
    df_whole = df_whole[df_whole["split"] == split].reset_index(drop=True)

    b_id_whole = set(list(df_whole["b_id"].values))
    b_id_clean = set(list(df_clean["b_id"].values))

    b_id_no_history = b_id_whole - b_id_clean
    # users without history
    df = df_whole[df_whole["b_id"].isin(b_id_no_history)].reset_index(drop=True)
    print("valid size:", len(df))
    # print("df.columns:", df.columns)
    predictions = []
    labels = []
    for idx, row in df.iterrows():
        for nid in row["candidates"]:
            if nid in news2proba:
                predictions.append(news2proba[nid])
            else:
                # predictions.append(0.5)  # append 0 or 0.5
                predictions.append(0.0)  # append 0 or 0.5
        for t in row["labels"]:
            labels.append(t)

    # print(len(predictions), len(labels))
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)

    auc = auroc(predictions, labels)
    mrr = mrr_score(predictions, labels)
    ndcg10 = ndcg_score(predictions, labels)
    ndcg5 = ndcg_score(predictions, labels, k=5)
    print("auc:", auc)
    print("mrr:", mrr)
    print("ndcg10:", ndcg10)
    print("ndcg5:", ndcg5)


if __name__ == "__main__":
    main()
