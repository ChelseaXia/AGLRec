# main.py
# 功能：主入口，控制整个训练和推荐流程，支持语义增强、保存模型与评估推荐效果

import os
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from config import CONFIG as cfg
from llm_embedding import LLMEncoder, build_item_embeddings, build_user_embeddings
from data_loader import encode_id_mappings, build_interaction_matrix, build_train_dataset, InteractionDataset, build_adj_matrix
from model import LightGCN
from trainer import train
from evaluator import evaluate
from recommend import generate_top_k_recommendations
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import json


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


def main():
    os.makedirs("saved", exist_ok=True)

    mode = cfg.get('mode', 'train')  # 'train' or 'inference'
    print(f"[模式] 当前为 {mode.upper()} 模式")

    print("[Step 1] 读取数据...")
    anime_df = pd.read_csv(cfg['anime_path'], sep='\t')
    interactions_df = pd.read_csv(cfg['interaction_path'], sep='\t')
    interactions_df = interactions_df.sort_values(by='updated_at')  # 按时间排序

    print("[Step 2] 构建 ID 映射...")
    user2id, item2id, id2user, id2item = encode_id_mappings(interactions_df)
    num_users, num_items = len(user2id), len(item2id)
    with open(cfg['mapping_path'], 'wb') as f:
        pickle.dump((user2id, item2id, id2user, id2item), f)

    print("[Step 3] 构建交互矩阵...")
    interaction_matrix = build_interaction_matrix(interactions_df, num_users, num_items, user2id, item2id)
    adj_csr = normalize_adj(build_adj_matrix(interaction_matrix)).tocoo()
    indices = torch.LongTensor([adj_csr.row, adj_csr.col])
    values = torch.FloatTensor(adj_csr.data)
    adj_mat = torch.sparse_coo_tensor(indices, values, adj_csr.shape).coalesce()

    if mode == 'train':
        print("[Step 4] 构建训练数据集（按用户 leave-one-out）...")
        interactions_df = interactions_df.sort_values(by=['user_id', 'updated_at'])
        train_rows = []
        test_rows = []
        
        for uid, user_df in interactions_df.groupby('user_id'):
            if len(user_df) < 2:
                train_rows.append(user_df)
            else:
                test_rows.append(user_df.iloc[-1])
                train_rows.append(user_df.iloc[:-1])
        
        train_df = pd.concat(train_rows).reset_index(drop=True)
        test_df = pd.DataFrame(test_rows).reset_index(drop=True)
        
        # 打印冷启动信息（应为 0）
        train_user_ids = set(train_df['user_id'])
        test_user_ids = set(test_df['user_id'])
        cold_users = test_user_ids - train_user_ids
        print(f"[Check] 冷启动用户数: {len(cold_users)} / {len(test_user_ids)}")
        
        train_pairs = build_train_dataset(train_df, user2id, item2id)
        train_loader = DataLoader(
            InteractionDataset(train_pairs, num_items),
            batch_size=cfg['batch_size'], shuffle=True
        )


    print("[Step 5] 生成或加载 LLM 嵌入...")
    user_llm = None
    item_llm = None
    
    if cfg["use_user_llm"] or cfg["use_item_llm"]:
        encoder = LLMEncoder(model_name=cfg['embedding_model'])
        print(f"[INFO] 当前使用设备：{encoder.device}")
    
    if cfg["use_item_llm"]:
        if os.path.exists(cfg['item_embedding_path']):
            item_llm = np.load(cfg['item_embedding_path'])
            print("已加载 item 嵌入")
        else:
            item_embed_dict = build_item_embeddings(anime_df, encoder)
            item_llm = np.zeros((num_items, item_embed_dict[next(iter(item_embed_dict))].shape[0]))
            for sid, vec in item_embed_dict.items():
                if sid in item2id:
                    item_llm[item2id[sid]] = vec
            np.save(cfg['item_embedding_path'], item_llm)
            print("已保存 item 嵌入")
    
    if cfg["use_user_llm"]:
        if os.path.exists(cfg['user_embedding_path']):
            user_llm = np.load(cfg['user_embedding_path'])
            print("已加载 user 嵌入")
        else:
            user_embed_dict = build_user_embeddings(interactions_df, encoder)
            llm_dim = next(iter(user_embed_dict.values())).shape[0]
            user_llm = np.zeros((num_users, llm_dim))
            for uid, idx in user2id.items():
                if uid in user_embed_dict:
                    user_llm[idx] = user_embed_dict[uid]
            np.save(cfg['user_embedding_path'], user_llm)
            print("已保存 user 嵌入")
    
        if os.path.exists(cfg['item_embedding_path']):
            item_llm = np.load(cfg['item_embedding_path'])
            print("已加载 item 嵌入")
        else:
            item_embed_dict = build_item_embeddings(anime_df, encoder)
            item_llm = np.zeros((num_items, item_embed_dict[next(iter(item_embed_dict))].shape[0]))
            for sid, vec in item_embed_dict.items():
                if sid in item2id:
                    item_llm[item2id[sid]] = vec
            np.save(cfg['item_embedding_path'], item_llm)
            print("已保存 item 嵌入")
    
        if os.path.exists(cfg['user_embedding_path']):
            user_llm = np.load(cfg['user_embedding_path'])
            print("已加载 user 嵌入")
        else:
            user_embed_dict = build_user_embeddings(interactions_df, encoder)
            llm_dim = next(iter(user_embed_dict.values())).shape[0]
            user_llm = np.zeros((num_users, llm_dim))
            for uid, idx in user2id.items():
                if uid in user_embed_dict:
                    user_llm[idx] = user_embed_dict[uid]
            np.save(cfg['user_embedding_path'], user_llm)
            print("已保存 user 嵌入")

    print("[Step 6] 初始化 LightGCN 模型...")
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=cfg['embedding_dim'],
        num_layers=cfg['num_layers'],
        user_llm_emb=user_llm,
        item_llm_emb=item_llm,
        fusion_mode=cfg['fusion_mode']
    )

    if mode == 'train':
        print("[Step 7] 开始训练...")

        def eval_func(model, user_emb, item_emb):
            test_user_dict = defaultdict(list)
            for _, row in test_df.iterrows():
                uid, sid = row['user_id'], row['subject_id']
                if uid in user2id and sid in item2id:
                    test_user_dict[user2id[uid]].append(item2id[sid])
            metrics = evaluate(model, user_emb, item_emb, test_user_dict, K=cfg['top_k'], device=cfg['device'])
            for k, v in metrics.items():
                print(f"[Eval] {k}: {v:.4f}")

            test_users = set(test_df['user_id'])
            train_users = set(train_df['user_id'])
            cold_users = test_users - train_users
            print(f"[DEBUG] 冷启动用户数: {len(cold_users)} / {len(test_users)}")

            test_items = set(test_df['subject_id'])
            train_items = set(train_df['subject_id'])
            cold_items = test_items - train_items
            print(f"[DEBUG] 冷启动物品数: {len(cold_items)} / {len(test_items)}")

            return sum(metrics.values()) / len(metrics)  # 综合指标平均作为 early stop 标准

        model = train(model, train_loader, adj_mat, epochs=cfg['epochs'], lr=cfg['lr'],
                      weight_decay=cfg['weight_decay'], early_stop_round=cfg['early_stop_round'],
                      eval_func=eval_func, device=cfg['device'])
        torch.save(model, cfg['model_path'])
    else:
        print("[Step 7] 加载已训练模型...")
        model = torch.load(cfg['model_path'], map_location=cfg['device'])

    user_emb, item_emb = model.forward(adj_mat)
    torch.save(user_emb, cfg['user_emb_path'])
    torch.save(item_emb, cfg['item_emb_path'])

    print("[Step 8] 生成推荐结果...")
    train_user_dict = defaultdict(list)
    for _, row in interactions_df.iterrows():
        uid, sid = row['user_id'], row['subject_id']
        if uid in user2id and sid in item2id:
            train_user_dict[user2id[uid]].append(item2id[sid])
    with open(cfg['train_user_dict_path'], 'wb') as f:
        pickle.dump(train_user_dict, f)

    recs = generate_top_k_recommendations(model, user_emb, item_emb,
                                          user2id, id2item, train_user_dict, cfg['top_k'], cfg['device'])
    recs = {int(k): [int(i) for i in v] for k, v in recs.items()}

    with open('saved/recommendations.json', 'w', encoding='utf-8') as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)

    print("[Step 9] 全部完成，结果已保存")


if __name__ == '__main__':
    main()
