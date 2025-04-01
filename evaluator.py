# evaluator.py
# 功能：计算推荐结果的评价指标，如 Recall@K, NDCG@K, Hit@K

import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score


def recall_at_k(pred_items, true_items, k):
    hits = len(set(pred_items[:k]) & set(true_items))
    return hits / len(true_items) if true_items else 0


def ndcg_at_k(pred_items, true_items, k):
    dcg = 0.0
    for i, item in enumerate(pred_items[:k]):
        if item in true_items:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
    return dcg / idcg if idcg > 0 else 0


def hit_at_k(pred_items, true_items, k):
    return int(len(set(pred_items[:k]) & set(true_items)) > 0)

def auc_for_user(scores, true_items, all_items):
    labels = np.zeros(len(all_items), dtype=int)
    for i in true_items:
        labels[i] = 1
    try:
        return roc_auc_score(labels, scores)
    except:
        return np.nan  # 若全为正或负则 AUC 不定义


def evaluate(model, user_emb, item_emb, test_user_dict, K=20, device='cpu'):
    """
    test_user_dict: {user_id: [item_ids]}
    """
    model.eval()
    recalls, ndcgs, hits, auc_scores = [], [], [], []

    with torch.no_grad():
        for user_id in tqdm(test_user_dict, desc=f"Evaluating Top-{K}", total=len(test_user_dict)):
            true_items = test_user_dict[user_id]
            scores = model.full_sort_predict(user_emb, item_emb, torch.tensor([user_id]).to(device))
            scores = scores.squeeze().cpu().numpy()
            top_k_items = np.argsort(-scores)[:K]

            recalls.append(recall_at_k(top_k_items, true_items, K))
            ndcgs.append(ndcg_at_k(top_k_items, true_items, K))
            hits.append(hit_at_k(top_k_items, true_items, K))
            auc_scores.append(auc_for_user(scores, true_items, list(range(len(scores)))))

    return {
        f"Recall@{K}": np.mean(recalls),
        f"NDCG@{K}": np.mean(ndcgs),
        f"Hit@{K}": np.mean(hits),
        f"AUC": np.nanmean(auc_scores)
    }