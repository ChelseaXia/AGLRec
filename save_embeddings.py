# save_embeddings.py
# 功能：加载 anime_df 和 interactions_df，使用 LLMEncoder 生成并保存嵌入

import pandas as pd
import numpy as np
import os
from llm_embedding import LLMEncoder, build_item_embeddings, build_user_embeddings


def save_embeddings(anime_path, interaction_path,
                    item_out_path="item_llm_embeddings.npy",
                    user_out_path="user_llm_embeddings.npy",
                    model_name="BAAI/bge-m3"):
    # 加载数据
    print("[1] 读取数据...")
    anime_df = pd.read_csv(anime_path, sep='\t')
    interactions_df = pd.read_csv(interaction_path, sep='\t')

    # 初始化模型
    print("[2] 初始化编码模型...")
    encoder = LLMEncoder(model_name=model_name)

    # 构建并保存 item embedding
    print("[3] 构建 item embedding...")
    item_emb_dict = build_item_embeddings(anime_df, encoder)
    item_ids_sorted = sorted(item_emb_dict.keys())
    item_embeddings = np.array([item_emb_dict[iid] for iid in item_ids_sorted])
    np.save(item_out_path, item_embeddings)
    print(f"✅ item embedding 保存至：{item_out_path}，维度 = {item_embeddings.shape}")

    # 构建并保存 user embedding
    print("[4] 构建 user embedding...")
    user_emb_dict = build_user_embeddings(interactions_df, encoder)
    user_ids_sorted = sorted(user_emb_dict.keys())
    user_embeddings = np.array([user_emb_dict[uid] for uid in user_ids_sorted])
    np.save(user_out_path, user_embeddings)
    print(f"✅ user embedding 保存至：{user_out_path}，维度 = {user_embeddings.shape}")


# 使用示例：
if __name__ == '__main__':
    save_embeddings(
        anime_path="anime_df.tsv",
        interaction_path="interactions_df.tsv",
        item_out_path="item_llm_embeddings.npy",
        user_out_path="user_llm_embeddings.npy",
        model_name="BAAI/bge-m3"
    )