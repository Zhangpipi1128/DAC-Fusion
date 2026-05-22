# -*- coding: utf-8 -*-
"""
Kaggle Notebook — 第一个 Code Cell（SCRUBD + 可选依赖）。

用法：在 Kaggle 中新建 Code Cell，将本文件「全文」复制进去运行，再运行完整 kaggle.py。

说明：
  - 使用 subprocess 调用 pip/git，避免依赖 IPython 的 ! 魔法。
  - 克隆 SCRUBD 到 /kaggle/working/SCRUBD，并 chdir 到数据目录（labels.csv / solidity_codes）。
  - 可选：pip 安装 sentence-transformers（仅当 RUN_ST_BASELINE=1 时需要）。
"""
from __future__ import annotations

import os
import subprocess
import sys

WORK = "/kaggle/working"
SCRUBD_DIR = os.path.join(WORK, "SCRUBD")
SCRUBD_DATA = os.path.join(SCRUBD_DIR, "SCRUBD-CD", "data")

# 封装执行系统命令
def run(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, **kwargs)


def main() -> None:
    os.makedirs(WORK, exist_ok=True)

    # Sentence-BGE 基线（VULN_IR_FULL_PAPER_RUN 会开启 RUN_ST_BASELINE）
    if os.environ.get("RUN_ST_BASELINE", "0").strip().lower() in ("1", "true", "yes", "on") or os.environ.get(
        "VULN_IR_FULL_PAPER_RUN", "0"
    ).strip().lower() in ("1", "true", "yes", "on"):
        run([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])

    if not os.path.isdir(os.path.join(SCRUBD_DIR, ".git")) and os.path.isdir(SCRUBD_DIR):
        print("目录已存在（非git克隆），跳过 clone:", SCRUBD_DIR)
    elif not os.path.isdir(SCRUBD_DIR):
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/sujeetc/SCRUBD.git",
                SCRUBD_DIR,
            ],
            cwd=WORK,
        )
    else:
        print("已存在 SCRUBD 仓库:", SCRUBD_DIR)

    if not os.path.isdir(SCRUBD_DATA):
        raise SystemExit(f"未找到 SCRUBD 数据目录: {SCRUBD_DATA}，请检查仓库结构是否变更。")
    os.chdir(SCRUBD_DATA)

    os.environ.setdefault("SCRUBD_LABELS", os.path.join(SCRUBD_DATA, "labels.csv"))
    os.environ.setdefault("SCRUBD_CODES", os.path.join(SCRUBD_DATA, "solidity_codes"))
    os.environ.setdefault("RUN_ST_BASELINE", "0")


# 1) 快速调试模式：关闭（必须0，否则跑不出论文结果）
os.environ.setdefault("QUICK_RUN", "0")

# 2) 训练阶段：只跑 Stage1，完全关闭 Stage2 难负例挖掘
os.environ.setdefault("RUN_STAGE1", "1")
os.environ.setdefault("RUN_STAGE2_HARD", "0")

# 3) 训练模式：window（论文核心：滑动窗口一致性训练）
os.environ.setdefault("TRAIN_POS_MODE", "window")

# 4) 融合模型：开启 DAC-Fusion（论文最终模型）
os.environ.setdefault("ENABLE_DAC_FUSION", "1")

# 5) 多种子实验：开启（5个种子，论文标准）
os.environ.setdefault("RUN_MULTI_SEED", "1")

# 6) 只跑基线：关闭（我们要跑自己的模型）
os.environ.setdefault("RUN_BASELINES_ONLY", "0")

# 7) 窗口聚合策略：max（论文默认最优）
os.environ.setdefault("WINDOW_AGG", "max")

# 8) 消融实验：开启
os.environ.setdefault("EVAL_ABLATION_WINDOW", "1")
os.environ.setdefault("EVAL_ABLATION_TRUNC", "1")
os.environ.setdefault("EVAL_DAC_ABLATION_COMPONENTS", "1")
os.environ.setdefault("VULN_IR_FULL_PAPER_RUN", "1")
os.environ.setdefault("RUN_TRUNC_TRAIN_DAC_ABLATION", "1")

# 9) 效率测试：开启
os.environ.setdefault("MEASURE_EFFICIENCY", "1")

# 10) 案例可视化：关闭（开源不需要）
os.environ.setdefault("DUMP_CASE", "0")

# 11) 数据集路径（自动）
os.environ.setdefault("SCRUBD_LABELS", os.path.join(SCRUBD_DATA, "labels.csv"))
os.environ.setdefault("SCRUBD_CODES", os.path.join(SCRUBD_DATA, "solidity_codes"))

# 12) 输出目录
os.environ.setdefault("VULN_IR_OUT", os.path.join(WORK, "output"))

print("=== 已加载 论文最终版固定配置 ===")



if __name__ == "__main__":
    main()
