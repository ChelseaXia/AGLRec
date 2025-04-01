# run_ablation.py
# 功能：对不同 LLM 融合配置（无/用户/物品/用户+物品）和融合方式（sum/concat）做消融实验
# 自动训练、记录 loss 与 metrics、保存模型

import os
import json
import torch
import pandas as pd
import subprocess
from config import CONFIG as cfg_template

# 所有配置组合
fusion_modes = ["sum", "concat"]
llm_settings = [
    {"use_user_llm": False, "use_item_llm": False},
    {"use_user_llm": True, "use_item_llm": False},
    {"use_user_llm": False, "use_item_llm": True},
    {"use_user_llm": True, "use_item_llm": True},
]

# 设置保存路径
os.makedirs("ablations", exist_ok=True)

# 主循环：遍历所有组合
for fusion_mode in fusion_modes:
    for setting in llm_settings:
        tag = f"user_{int(setting['use_user_llm'])}_item_{int(setting['use_item_llm'])}_mode_{fusion_mode}"
        print(f"\n[Running] 当前配置: {tag}")

        # 修改 config
        cfg_template["fusion_mode"] = fusion_mode
        cfg_template["use_user_llm"] = setting["use_user_llm"]
        cfg_template["use_item_llm"] = setting["use_item_llm"]
        cfg_template["model_path"] = f"ablations/{tag}_model.pth"
        cfg_template["log_path"] = f"ablations/{tag}_log.json"

        # 保存当前 config 为临时文件供 main.py 使用（如需）
        with open("config_runtime.json", "w") as f:
            json.dump(cfg_template, f)

        # 调用 main.py
        subprocess.run(["python", "main.py"])

print("\n[完成] 所有实验已运行完毕。")
