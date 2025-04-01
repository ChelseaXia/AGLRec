# data_loader.py
# 功能：数据加载与预处理，构建 user-item 图、训练数据、负采样

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
from collections import defaultdict
from tqdm import tqdm


class InteractionDataset(Dataset):
    """
    三元组 (user, pos_item, neg_item) 用于 BPR 训练
    """
    def __init__(self, user_item_pairs, num_items):
        self.user_item_pairs = user_item_pairs
        self.num_items = num_items
        self.user_pos_dict = defaultdict(set)
        for u, i in user_item_pairs:
            self.user_pos_dict[u].add(i)

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, pos = self.user_item_pairs[idx]
        while True:
            neg = np.random.randint(0, self.num_items)
            if neg not in self.user_pos_dict[user]:
                break
        return user, pos, neg


def build_interaction_matrix(interactions_df, num_users, num_items, user2id, item2id):
    """
    构建稀疏的 user-item 矩阵 A，后续用于 Graph 构造
    """
    rows = interactions_df['user_id'].map(user2id)
    cols = interactions_df['subject_id'].map(item2id)
    data = np.ones(len(interactions_df))
    mat = sp.coo_matrix((data, (rows, cols)), shape=(num_users, num_items))
    return mat


def build_train_dataset(interactions_df, user2id, item2id):
    """
    返回训练用的正样本列表 [(u, i)]，用于构建 BPR 三元组
    """
    interactions_df = interactions_df[interactions_df['rate'] > 0]
    user_item_pairs = [
        (user2id[u], item2id[i])
        for u, i in tqdm(zip(interactions_df['user_id'], interactions_df['subject_id']),
                         total=len(interactions_df), desc="Building train pairs")
        if u in user2id and i in item2id
    ]
    return user_item_pairs


def encode_id_mappings(interactions_df):
    """
    构建 user_id 和 item_id 的连续编码
    """
    unique_users = interactions_df['user_id'].unique()
    unique_items = interactions_df['subject_id'].unique()

    user2id = {uid: i for i, uid in enumerate(unique_users)}
    item2id = {iid: i for i, iid in enumerate(unique_items)}
    id2user = {i: uid for uid, i in user2id.items()}
    id2item = {i: iid for iid, i in item2id.items()}
    return user2id, item2id, id2user, id2item


def build_adj_matrix(interaction_matrix):
    """
    构建对称邻接矩阵 A_hat = A + A.T（用于 GCN）
    """
    user_item_mat = interaction_matrix
    item_user_mat = user_item_mat.T
    upper_mat = sp.vstack([sp.hstack([sp.coo_matrix((user_item_mat.shape[0], user_item_mat.shape[0])), user_item_mat]),
                           sp.hstack([item_user_mat, sp.coo_matrix((user_item_mat.shape[1], user_item_mat.shape[1]))])])
    return upper_mat.tocsr()


# 使用示例（主函数）
if __name__ == '__main__':
    interactions_df = pd.read_csv('interactions.tsv', sep='\t')
    user2id, item2id, id2user, id2item = encode_id_mappings(interactions_df)
    num_users, num_items = len(user2id), len(item2id)

    interaction_mat = build_interaction_matrix(interactions_df, num_users, num_items, user2id, item2id)
    adj_mat = build_adj_matrix(interaction_mat)

    user_item_pairs = build_train_dataset(interactions_df, user2id, item2id)
    dataset = InteractionDataset(user_item_pairs, num_items)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    for u, i, j in dataloader:
        print(u.shape, i.shape, j.shape)
        break
