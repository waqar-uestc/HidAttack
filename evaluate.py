import torch
import numpy as np

def evaluate_mrr(rating, ground_truth, top_k):
    _, rating_k = torch.topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()
    mrr = 0
    rank_index = None
    for i, v in enumerate(rating_k):
        if v in ground_truth:
            rank_index = ground_truth.index(v)+1
            mrr += (1/rank_index)
    mrr_score =mrr/(len(rating_k))
    return mrr_score

def evaluate_recall(rating, ground_truth, top_k):
    _, rating_k = torch.topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()
    hit = 0
    for i, v in enumerate(rating_k):
        if v in ground_truth:
            hit += 1

    recall = hit / len(ground_truth)
    return recall


def evaluate_precision(rating, ground_truth, top_k):
    _, rating_k = torch.topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()

    hit = 0
    for i, v in enumerate(rating_k):
        if v in ground_truth:
            hit += 1

    precision = hit / top_k
    return precision


def evaluate_ndcg(rating, ground_truth, top_k):
    _, rating_k = torch.topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()
    dcg, idcg = 0., 0.

    for i, v in enumerate(rating_k):
        if i < len(ground_truth):
            idcg += (1 / np.log2(2 + i))
        if v in ground_truth:
            dcg += (1 / np.log2(2 + i))

    ndcg = dcg / idcg
    return ndcg
