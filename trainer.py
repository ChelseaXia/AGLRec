# trainer.py
# 功能：训练 LightGCN 模型，使用 BPR loss 和优化器，支持评估与早停

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


def bpr_loss(user_emb, pos_emb, neg_emb, reg_lambda=1e-4):
    """
    Bayesian Personalized Ranking Loss
    """
    pos_scores = torch.sum(user_emb * pos_emb, dim=1)
    neg_scores = torch.sum(user_emb * neg_emb, dim=1)
    diff = pos_scores - neg_scores

    loss = -torch.mean(F.logsigmoid(diff))
    reg = reg_lambda * (
        user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)
    ) / 2
    return loss + reg


def train(model, train_loader, adj_mat, epochs, lr=0.001, weight_decay=1e-4,
          eval_func=None, early_stop_round=10, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_metric = 0
    best_epoch = 0
    stop_counter = 0

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            users, pos_items, neg_items = [x.to(device) for x in batch]
            user_emb, item_emb = model(adj_mat)

            u_emb = user_emb[users]
            i_emb = item_emb[pos_items]
            j_emb = item_emb[neg_items]

            # 数值打印
            # with torch.no_grad():
            #    pos_scores = torch.sum(u_emb * i_emb, dim=1)
            #    neg_scores = torch.sum(u_emb * j_emb, dim=1)
            #    print("pos_scores:", pos_scores.min().item(), pos_scores.max().item())
            #    print("neg_scores:", neg_scores.min().item(), neg_scores.max().item())
            #    print("score_diff:", (pos_scores - neg_scores).min().item(), (pos_scores - neg_scores).max().item())
            #    print("Batch embedding std:", u_emb.std().item(), i_emb.std().item(), j_emb.std().item())

            loss = bpr_loss(u_emb, i_emb, j_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch}] Loss: {total_loss:.4f}")

        if eval_func is not None:
            model.eval()
            user_emb, item_emb = model(adj_mat)
            metric = eval_func(model, user_emb, item_emb)
            print(f"[Eval] Metric: {metric:.4f}")

            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                stop_counter = 0
            else:
                stop_counter += 1

            if stop_counter >= early_stop_round:
                print(f"[Early Stop] at Epoch {epoch}, Best Epoch: {best_epoch}, Best Metric: {best_metric:.4f}")
                break

    return model
