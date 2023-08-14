"""
This module defines TypedDict classes to represent the structure of batches used in the MIND dataset:

- `ContentsEncoded`: A dictionary subclass defining the structure of the encoded content, containing title, abstract, category, and subcategory. Only the title is actively used.
- `MINDBatch`: Describes a batch structure specific to the MIND dataset, including batch history, batch candidates, and other content details like title and abstract.
- `Word2VecContentsEncoded`: Similar to `ContentsEncoded`, but represents the Word2Vec encoded content where all fields are tensors.
- `Word2VecMINDBatch`: Similar to `MINDBatch`, but tailored to Word2Vec encoding.
"""

import torch
from typing import TypedDict, Optional, Union
from transformers import BatchEncoding


class ContentsEncoded(TypedDict):
    """dict is converted to ContentsEncoded, itself is a Dict subclass"""

    # Only title is used.
    title: BatchEncoding
    abstract: BatchEncoding
    category: torch.Tensor
    subcategory: torch.Tensor


class MINDBatch(TypedDict):
    batch_hist: torch.Tensor
    batch_cand: torch.Tensor
    x_hist: Union[ContentsEncoded, torch.Tensor]
    x_cand: Union[ContentsEncoded, torch.Tensor]
    targets: Optional[torch.Tensor]


class Word2VecContentsEncoded(TypedDict):
    # Only title is used.
    title: torch.Tensor  # word id of dynamic padding
    abstract: torch.Tensor
    category: torch.Tensor
    subcategory: torch.Tensor


class Word2VecMINDBatch(TypedDict):
    batch_hist: torch.Tensor
    batch_cand: torch.Tensor
    x_hist: Union[Word2VecContentsEncoded, torch.Tensor]
    x_cand: Union[Word2VecContentsEncoded, torch.Tensor]
    targets: Optional[torch.Tensor]
