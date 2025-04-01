# model.py
# 功能：定义 LightGCN 模型结构，支持融合 LLM 的用户/物品嵌入

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3,
                 user_init_emb=None, item_init_emb=None,
                 user_llm_emb=None, item_llm_emb=None,
                 fusion_mode='concat'):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.fusion_mode = fusion_mode  # 'concat' or 'sum'

        # 初始化 ID embedding，数值范围控制在合理区间
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        # LLM 融合（可选）
        self.user_llm_emb = user_llm_emb
        self.item_llm_emb = item_llm_emb
        if self.user_llm_emb is not None:
            user_llm_tensor = torch.tensor(user_llm_emb, dtype=torch.float32)
            self.user_llm_embedding = nn.Parameter(user_llm_tensor, requires_grad=False)
        if self.item_llm_emb is not None:
            item_llm_tensor = torch.tensor(item_llm_emb, dtype=torch.float32)
            self.item_llm_embedding = nn.Parameter(item_llm_tensor, requires_grad=False)

        # Projection 层：如果使用 concat 模式
        if fusion_mode == 'concat':
            if self.user_llm_emb is not None:
                user_llm_dim = self.user_llm_emb.shape[1]
                self.user_projection = nn.Linear(embedding_dim + user_llm_dim, embedding_dim)
            if self.item_llm_emb is not None:
                item_llm_dim = self.item_llm_emb.shape[1]
                self.item_projection = nn.Linear(embedding_dim + item_llm_dim, embedding_dim)

    def forward(self, adj_mat):
        device = next(self.parameters()).device
        adj_mat = adj_mat.to(device)

        user_ids = torch.arange(self.num_users, device=device)
        item_ids = torch.arange(self.num_items, device=device)

        all_user_emb = self.user_embedding(user_ids)
        all_item_emb = self.item_embedding(item_ids)

        # 融合逻辑：分别判断 user 和 item
        if self.user_llm_emb is not None:
            if self.fusion_mode == 'sum':
                all_user_emb = all_user_emb + self.user_llm_embedding.to(device)
            elif self.fusion_mode == 'concat':
                user_cat = torch.cat([all_user_emb, self.user_llm_embedding.to(device)], dim=1)
                all_user_emb = self.user_projection(user_cat)

        if self.item_llm_emb is not None:
            if self.fusion_mode == 'sum':
                all_item_emb = all_item_emb + self.item_llm_embedding.to(device)
            elif self.fusion_mode == 'concat':
                item_cat = torch.cat([all_item_emb, self.item_llm_embedding.to(device)], dim=1)
                all_item_emb = self.item_projection(item_cat)

        # 归一化初始 embedding
        all_user_emb = F.normalize(all_user_emb, p=2, dim=1)
        all_item_emb = F.normalize(all_item_emb, p=2, dim=1)

        emb = torch.cat([all_user_emb, all_item_emb], dim=0)
        embs = [emb]

        for layer in range(self.num_layers):
            emb = torch.sparse.mm(adj_mat, emb)
            embs.append(emb)

        out = torch.stack(embs, dim=1).mean(dim=1)
        out = F.normalize(out, p=2, dim=1)

        final_user_emb, final_item_emb = torch.split(out, [self.num_users, self.num_items])
        return final_user_emb, final_item_emb

    def get_embedding(self):
        return self.user_embedding.weight, self.item_embedding.weight

    def predict(self, user_emb, item_emb, user_idx, item_idx):
        u = user_emb[user_idx]
        i = item_emb[item_idx]
        return (u * i).sum(dim=1)

    def full_sort_predict(self, user_emb, item_emb, user_idx):
        u = user_emb[user_idx]
        scores = torch.matmul(u, item_emb.T)
        return scores
