import torch
import numpy as np


def dcg_score(y_score: torch.Tensor, y_true: torch.Tensor, k=10) -> torch.Tensor:
    y_true = y_true.float()
    y_score = y_score.float()

    order = torch.argsort(y_score).flip([0])
    y_true = torch.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = torch.log2(torch.arange(len(y_true), device=y_true.device).float() + 2)

    return torch.sum(gains / discounts)


def ndcg_score(y_score: torch.Tensor, y_true: torch.Tensor, k=10) -> torch.Tensor:
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_score, y_true, k)
    return actual / best


def mrr_score(y_score: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    y_true_np = y_true.cpu().numpy()
    y_score_np = y_score.cpu().numpy()
    order = np.argsort(y_score_np)[::-1]
    y_true = np.take(y_true_np, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    x = np.sum(rr_score) / np.sum(y_true)
    return torch.tensor(x).to(y_score.device)


# def mrr_score(y_true, y_score):
#     order = np.argsort(y_score)[::-1]
#     y_true = np.take(y_true, order)
#     rr_score = y_true / (np.arange(len(y_true)) + 1)
#     return np.sum(rr_score) / np.sum(y_true)
