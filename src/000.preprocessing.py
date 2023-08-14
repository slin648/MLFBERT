"""
This script performs preprocessing steps needed when utilizing word2vec for word embedding instead of BERT.
It parses news data for training, validation, and test sets, converts words to their corresponding IDs based on frequency,
and generates pre-trained word embeddings using word2vec. This includes handling words not found in the pre-trained embedding file.
The resulting numerized and embedded data are then saved for further processing.
Functions:
    - parse_news(train_dir, val_dir, test_dir): Parses news data and converts words to IDs.
    - generate_word_embedding(source, word2int): Generates word embeddings using word2vec.
"""

import os
import pandas as pd
import json
import math
from tqdm import tqdm
from os import path
from pathlib import Path
import random
from nltk.tokenize import word_tokenize
import numpy as np
import csv
from tqdm import tqdm


SMALL_VERSION = 1
WORD_FREQ_THRETHOLD = 2


def parse_news(train_dir, val_dir, test_dir=None):
    """
    Parse news for training set and validation set
    """
    news = pd.read_table(
        os.path.join(train_dir, "news.tsv"),
        header=None,
        usecols=[0, 1, 2, 3, 4, 6, 7],
        quoting=csv.QUOTE_NONE,
        names=[
            "id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "title_entities",
            "abstract_entities",
        ],
    )
    news.title_entities.fillna("[]", inplace=True)
    news.abstract_entities.fillna("[]", inplace=True)
    news.fillna(" ", inplace=True)

    word2freq = {}  # word -> frequency
    categorys = list(news["category"].values)
    titles = list(news["title"].values)

    sentences = categorys + titles  #  + abstracts
    for sentence in tqdm(sentences):
        if sentence.strip() != "":
            for i, w in enumerate(word_tokenize(sentence)):  # do not lower case
                word2freq[w] = word2freq.get(w, 0) + 1
    word2int = {"[UNK]": 0, "[PAD]": 1}
    for word, freq in word2freq.items():
        if freq >= WORD_FREQ_THRETHOLD:
            word2int[word] = len(word2int)
    print("len(word2freq): ", len(word2freq))  # 48124
    print("len(word2int): ", len(word2int))  # 25957

    # convert word to id
    def convert_word_to_id(sentence):
        if sentence.strip() == "":
            return "0"  # ["UNK"]
        else:
            return ";".join(
                [
                    str(word2int[w]) if w in word2int else "0"
                    for w in word_tokenize(sentence)
                ]
            )

    news["category_id"] = news["category"].apply(convert_word_to_id)
    news["title_id"] = news["title"].apply(convert_word_to_id)
    news["abstract_id"] = news["abstract"].apply(convert_word_to_id)

    valid_news = pd.read_table(
        os.path.join(val_dir, "news.tsv"),
        header=None,
        usecols=[0, 1, 2, 3, 4, 6, 7],
        quoting=csv.QUOTE_NONE,
        names=[
            "id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "title_entities",
            "abstract_entities",
        ],
    )  # TODO try to avoid csv.QUOTE_NONE
    valid_news.title_entities.fillna("[]", inplace=True)
    valid_news.abstract_entities.fillna("[]", inplace=True)
    valid_news.fillna(" ", inplace=True)

    valid_news["category_id"] = valid_news["category"].apply(convert_word_to_id)
    valid_news["title_id"] = valid_news["title"].apply(convert_word_to_id)
    valid_news["abstract_id"] = valid_news["abstract"].apply(convert_word_to_id)

    if test_dir is not None:
        test_news = pd.read_table(
            os.path.join(test_dir, "news.tsv"),
            header=None,
            usecols=[0, 1, 2, 3, 4, 6, 7],
            quoting=csv.QUOTE_NONE,
            names=[
                "id",
                "category",
                "subcategory",
                "title",
                "abstract",
                "title_entities",
                "abstract_entities",
            ],
        )
        test_news.title_entities.fillna("[]", inplace=True)
        test_news.abstract_entities.fillna("[]", inplace=True)
        test_news.fillna(" ", inplace=True)

        test_news["category_id"] = test_news["category"].apply(convert_word_to_id)
        test_news["title_id"] = test_news["title"].apply(convert_word_to_id)
        test_news["abstract_id"] = test_news["abstract"].apply(convert_word_to_id)

        test_news.to_parquet(os.path.join(test_dir, "news_numerized.parquet"))
    news.to_parquet(os.path.join(train_dir, "news_numerized.parquet"))
    valid_news.to_parquet(os.path.join(val_dir, "news_numerized.parquet"))

    with open(os.path.join(train_dir, "word2int.txt"), "w", encoding="utf-8") as fo:
        for word, idx in word2int.items():
            fo.write(f"{word}\t{idx}\n")

    return word2int


def generate_word_embedding(source, word2int):
    """
    Generate from pretrained word embedding file
    If a word not in embedding file, initial its embedding by N(0, 1)
    Args:
        source: path of pretrained word embedding file, e.g. glove.840B.300d.txt
        word2int_path: vocabulary file when words in it will be searched in pretrained embedding file
    """
    words = []
    ids = []
    for word, idx in word2int.items():
        words.append(word)
        ids.append(idx)
    wordint_df = pd.DataFrame({"word": words, "id": ids})

    source_embedding = pd.read_table(
        source,
        index_col=0,
        sep=" ",
        header=None,
        quoting=csv.QUOTE_NONE,
        names=range(300),
    )

    # word, vector
    source_embedding.index.rename("word", inplace=True)
    source_embedding.reset_index(inplace=True)
    # print(source_embedding.head(3))
    """
      word         0        1        2         3         4  ...      294       295       296      297      298      299
    0    , -0.082752  0.67204 -0.14987 -0.064983  0.056491  ... -0.35598  0.053380 -0.050821 -0.19180 -0.37846 -0.06589
    1    .  0.012001  0.20751 -0.12578 -0.593250  0.125250  ... -0.28135  0.063500  0.140190  0.13871 -0.36049 -0.03500
    2  the  0.272040 -0.06203 -0.18840  0.023225 -0.018158  ... -0.10280 -0.018168  0.114070  0.13015 -0.18317  0.13230
    """
    # word, int, vector
    merged = wordint_df.merge(source_embedding, how="inner", on="word").reset_index(
        drop=True
    )

    missed_words = set(words) - set(list(merged.word.values))

    missed_embedding = pd.DataFrame(
        data=np.random.normal(size=(len(missed_words), 300))
    )
    missed_embedding["word"] = list(missed_words)
    missed_embedding["id"] = missed_embedding["word"].map(word2int)

    final_embedding = (
        pd.concat([merged, missed_embedding[list(merged.columns)]])
        .sort_index()
        .reset_index(drop=True)
    )
    final_embedding = final_embedding.sort_values(by="id").reset_index(
        drop=True
    )  # sort by word id
    print(final_embedding["id"].head(30))
    # print(final_embedding[range(300)])
    """
            0         1        2         3         4         5         6    ...       293       294       295       296       297       298       299
    0 -0.245000  0.505550 -0.11252  0.219670 -0.211820 -0.013613  0.110440  ... -0.236990 -0.790720  0.027771 -0.130300 -0.483410 -0.261070  0.658770
    1  0.207153 -0.750994  0.48160 -0.275664  0.453296 -0.458481 -1.005986  ... -1.446503 -0.721708 -0.281144 -0.933893  1.351543  0.666261  0.027421
    """
    np.save("./wordembedding.npy", final_embedding[range(300)].values)

    print(
        f"Rate of word missed in pretrained embedding: {(len(missed_words)-1)/len(word2int):.4f}"  # 0.0441
    )


if __name__ == "__main__":
    train_dir = "../MIND_DATASET/train"
    val_dir = "../MIND_DATASET/valid"
    test_dir = None if SMALL_VERSION else "../MIND_DATASET/test"

    word2int = parse_news(train_dir, val_dir, test_dir)
    # word2int = {"Apple": 1, "kdkdkddk": 2}
    print("Generate word embedding")
    generate_word_embedding("../MIND_DATASET/glove.840B.300d.txt", word2int)
