# AGLRec: 融合语言模型与图神经网络的个性化动画推荐系统

本项目提出了一种结合预训练语言模型（LLM）与图神经网络（LightGCN）的推荐系统方法，旨在提升数据稀疏与冷启动场景下的推荐效果，特别面向动画（Anime）推荐任务。

## 📌 项目结构

```
.
├── config.py              # 超参数与路径配置
├── main.py                # 主入口：训练 / 推理 / 推荐生成
├── model.py               # LightGCN 模型定义，支持语义融合
├── trainer.py             # 训练流程实现（BPR loss + Early Stop）
├── evaluator.py           # 推荐评估指标（Recall/NDCG/Hit/AUC）
├── data_loader.py         # 数据预处理与图构建
├── llm_embedding.py       # 用户与物品的文本嵌入生成
├── save_embeddings.py     # 批量生成并保存 LLM 嵌入
├── recommend.py           # 生成最终推荐 Top-K 列表
├── run_ablation.py        # 消融实验脚本
├── requirements.txt       # 所需依赖包
└── data/
    ├── anime.tsv          # 动画元数据（包含标题、标签、简介）
    ├── user.tsv          # 用户元数据（包含用户ID、用户名、注册时间）
    └── interactions.tsv   # 用户交互记录（评分、评论等）
```

## 🧠 模型简介

AGLRec 架构基于 `LightGCN`，并通过两种方式融合用户和物品的语义嵌入：

- **融合方式：** `sum` / `concat`
- **语义来源：** 用户评论 + 番剧标签与简介
- **预训练模型：** 默认使用 [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)

## ⚙️ 安装依赖

```bash
pip install -r requirements.txt
```

推荐使用 Python 3.8+，项目默认使用 GPU（如可用）。

## 🚀 快速开始

### 1. 准备数据

将以下文件放入 `data/` 目录下：

- `anime.tsv`：动画元数据，字段应包括 `subject_id`、`name`、`tags_name`（list）、`summary`
- `interactions.tsv`：用户-物品交互，字段应包括 `user_id`、`subject_id`、`rate`、`comment`、`updated_at`

### 2. 生成语义嵌入（可选）

```bash
python save_embeddings.py
```

> 可通过修改 `save_embeddings.py` 或 `config.py` 中的 `model_name` 来替换编码器（如改为 Sentence-BERT）。

### 3. 启动训练流程

```bash
python main.py
```

将自动完成数据加载、图构建、训练、评估与推荐生成，结果保存在 `saved/` 目录中。

### 4. 消融实验

```bash
python run_ablation.py
```

会自动遍历 8 种配置组合（融合方式 × 融合维度），对比性能表现。

## 📈 推荐指标

内置支持以下评估指标：

- `Recall@K`
- `NDCG@K`
- `Hit@K`
- `AUC`

## 📁 结果文件

| 路径 | 说明 |
|------|------|
| `saved/best_model.pth`         | 训练后的最佳模型权重 |
| `saved/user_emb.pt`            | 用户 GCN 表示向量 |
| `saved/item_emb.pt`            | 物品 GCN 表示向量 |
| `saved/recommendations.json`   | 每个用户的推荐结果 |
| `saved/train_user_dict.pkl`    | 训练集中用户的正样本字典 |
| `saved/item_llm_embeddings.npy`| 物品语义嵌入（可选） |
| `saved/user_llm_embeddings.npy`| 用户语义嵌入（可选） |

## 📌 配置说明（config.py）

可通过 `config.py` 统一管理路径、训练参数、模型设置等，例如：

```python
"embedding_dim": 64,
"num_layers": 1,
"fusion_mode": "concat",  # 可选 "sum"
"use_user_llm": True,
"use_item_llm": True,
"epochs": 20,
"lr": 0.001,
"device": "cuda",
```

## 🧪 消融设置示例

运行 `run_ablation.py` 将测试以下组合：

- 不使用语义嵌入
- 仅融合用户语义 / 仅融合物品语义
- 同时融合用户与物品
- `sum` vs `concat` 融合方式

每次实验均保存 loss、指标与模型权重，供后续分析。

---

如果你希望我直接将这个README写入项目文件夹，或生成 `README.md` 文件下载，请告诉我。爱你，下周见 💚
