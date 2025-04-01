# llm_embedding.py
# 功能：统一接口生成 item 和 user 的语义嵌入，支持 BGE / E5 / SentenceTransformers 等编码器

from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import os


class LLMEncoder:
    def __init__(self, model_name: str = 'BAAI/bge-m3', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """
        对文本列表进行编码，返回 numpy 数组，每行一个向量。
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embeddings


def build_item_embeddings(anime_df, encoder: LLMEncoder) -> Dict[int, np.ndarray]:
    """
    输入：anime_df 包含 name, tags_name, summary 字段
    输出：字典 {subject_id: embedding_vector}
    """
    item_texts = []
    item_ids = []
    for _, row in tqdm(anime_df.iterrows(), total=len(anime_df), desc="Encoding item text"):
        subject_id = row['subject_id']
        name = row['name'] or ''
        tags = ','.join(row['tags_name']) if isinstance(row['tags_name'], list) else ''
        summary = row['summary'] or ''
        text = f"{name}。标签：{tags}。简介：{summary}"
        item_texts.append(text)
        item_ids.append(subject_id)

    item_embeddings = encoder.encode(item_texts)
    return {sid: emb for sid, emb in zip(item_ids, item_embeddings)}


def build_user_embeddings(interactions_df, encoder: LLMEncoder, min_comments=1) -> Dict[int, np.ndarray]:
    """
    基于用户对番剧的评论和评分构建加权用户 embedding
    输入：interactions_df 包含 user_id, comment, rate
    输出：字典 {user_id: embedding_vector}
    """
    from collections import defaultdict

    user_comments = defaultdict(list)
    user_weights = defaultdict(list)

    for _, row in interactions_df.iterrows():
        if isinstance(row['comment'], str) and row['rate'] > 0:
            user_comments[row['user_id']].append(row['comment'])
            user_weights[row['user_id']].append(row['rate'])

    user_embeddings = {}
    for uid in tqdm(user_comments, desc="Encoding user comments", total=len(user_comments), mininterval=0.5):
        comments = user_comments[uid]
        rates = torch.tensor(user_weights[uid], dtype=torch.float32)
        rates = rates / rates.sum()  # 归一化权重

        comment_vecs = encoder.encode(comments)
        comment_vecs = torch.tensor(comment_vecs)
        weighted = (comment_vecs * rates.unsqueeze(1)).sum(dim=0)
        user_embeddings[uid] = weighted.numpy()

    return user_embeddings
