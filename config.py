# config.py
# 功能：统一管理路径、超参数、模型设置等配置项
import torch

CONFIG = {
    # 数据路径
    "anime_path": "data/anime.tsv",
    "interaction_path": "data/interactions.tsv",
    "user_path": "data/user.tsv",

    # 嵌入保存路径
    "item_embeddisng_path": "saved/item_llm_embeddings.npy",
    "user_embedding_path": "saved/user_llm_embeddings.npy",

    # LLM 模型选择
    "embedding_model": "BAAI/bge-m3",  # 或 sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    
    # 控制是否使用 LLM 语义嵌入
    "use_user_llm": False,
    "use_item_llm": False,
    
    # GCN 模型超参数
    "embedding_dim": 64,
    "num_layers": 1,
    "fusion_mode": "concat",  # "sum" 或 "concat"

    # 训练参数
    "epochs": 20,
    "lr": 0.001,
    "batch_size": 512,
    "weight_decay": 1e-4,
    "reg_lambda": 1e-2,
    "early_stop_round": 10,

    # 推荐设置
    "top_k": 10,

    # 保存路径
    "model_path": "saved/best_model.pth",
    "user_emb_path": "saved/user_emb.pt",
    "item_emb_path": "saved/item_emb.pt",
    "mapping_path": "saved/id_mappings.pkl",
    "train_user_dict_path": "saved/train_user_dict.pkl",

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 训练模式
    "mode": "train"  # "train"或"inference"
}


if __name__ == '__main__':
    for k, v in CONFIG.items():
        print(f"{k:30s}: {v}")