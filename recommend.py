# recommend.py
# 功能：使用训练好的模型给每个用户生成 Top-K 推荐列表

import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm


def generate_top_k_recommendations(model, user_emb, item_emb,
                                    user2id: Dict[int, int],
                                    id2item: Dict[int, int],
                                    train_user_dict: Dict[int, List[int]],
                                    K: int = 20,
                                    device='cpu') -> Dict[int, List[int]]:
    """
    返回每个用户的 Top-K 推荐结果，过滤掉训练集已交互项
    """
    model.eval()
    all_recs = {}
    num_users = user_emb.shape[0]

    with torch.no_grad():
        for uid in tqdm(range(num_users), desc="Generating Recommendations"):
            train_items = set(train_user_dict.get(uid, []))
            scores = model.full_sort_predict(user_emb, item_emb, torch.tensor([uid]).to(device))
            scores = scores.squeeze().cpu().numpy()

            scores[list(train_items)] = -np.inf  # 不推荐训练集中看过的
            top_k = np.argsort(-scores)[:K]
            top_k_raw_ids = [id2item[iid] for iid in top_k]
            user_raw_id = [k for k, v in user2id.items() if v == uid][0]
            all_recs[user_raw_id] = top_k_raw_ids

    return all_recs


# 使用示例：
if __name__ == '__main__':
    # 假设已有训练完成的模型和 embedding
    from model import LightGCN
    from data_loader import encode_id_mappings
    import pickle

    # 加载模型、embedding、id 映射等
    model = torch.load('best_model.pth')
    user_emb = torch.load('user_emb.pt')
    item_emb = torch.load('item_emb.pt')
    with open('id_mappings.pkl', 'rb') as f:
        user2id, item2id, id2user, id2item = pickle.load(f)
    with open('train_user_dict.pkl', 'rb') as f:
        train_user_dict = pickle.load(f)

    top_k_recs = generate_top_k_recommendations(model, user_emb, item_emb,
                                                user2id, id2item, train_user_dict, K=20)
    for uid in list(top_k_recs.keys())[:5]:
        print(f"用户 {uid} 推荐：{top_k_recs[uid]}")