# -*- coding: utf-8 -*-
# 
# ================================================================================
# Vuln-IR：基于 CodeBERT 的智能合约漏洞语义检索（完整可运行脚本）
# ================================================================================
# 功能概览（按 main() 执行顺序）：
#   1) 读取标签 CSV 与 Solidity 源码，构造 vuln_type（支持 SCRUBD、显式 vuln_type 列、二进制列映射；见 load_dataframe）
#   2) 为每条样本生成多条自然语言 query，构建 Vuln-IR 检索基准
#   3) 按合约 sample_id 划分 train / val / test，避免数据泄漏（与论文 §3.1、§4.1 合约级划分一致）
#   4) 加载 CodeBERT，Stage1：窗口一致性的对比学习（InfoNCE + in-batch negatives）
#   5) Stage2（可选）：合约级 hard negative（每条 query 3 个难负样本）二阶段训练
#   6) 评测：整段截断检索（baseline）+ 滑窗 chunk 聚合检索（window）
#   7) 可选：外部 JSONL（VULN_IR_DATA_SOURCE=external_jsonl；VULN_IR_EXTERNAL_JSONL_DIR）
#   8) 可选：Sentence-Transformers 稠密基线（RUN_ST_BASELINE=1；与 Graph/UniX 并列，需 pip install sentence-transformers）
#   9) 可选：开源小 LLM 重排序基线（RUN_LLM_RERANK_BASELINE=1；BM25@K 召回后单次生成选序号，见 LLM_RERANK_* 环境变量）
#   10) 可选：RUN_BASELINES_ONLY=1 只跑基线（跳过训练/主模型）
#   11) DAC 消融列（见 EVAL_DAC_ABLATION_COMPONENTS、RUN_TRUNC_TRAIN_DAC_ABLATION）：多种子汇总与 ablation_table_mean.csv 追加行

# 运行方式：
#   - Kaggle：直接运行本文件（已适配路径）。Notebook 环境默认 VULN_IR_FULL_PAPER_RUN≈开：
#       五种子 + DAC 子消融 + trunc_first 对照 + 案例门控热力图 + BM25/Graph/UniX/ST 基线。
#       快速试跑请设 QUICK_RUN=1，或显式 VULN_IR_FULL_PAPER_RUN=0 关闭全套。
#   - 本地一次跑出同样全套：VULN_IR_FULL_PAPER_RUN=1 python kaggle.py
#   - 本地：设置环境变量 SCRUBD_LABELS / SCRUBD_CODES，或改下方 CONFIG

# 依赖：torch, transformers, pandas, tqdm, numpy
# 可选：sentence-transformers（仅当 RUN_ST_BASELINE=1 时需要）
# ================================================================================
# 

# ==============================================================================
# Kaggle 环境：克隆数据集（取消下面的注释即可在 Kaggle 使用）
# ==============================================================================
# import subprocess
# subprocess.run(["git", "clone", "https://github.com/sujeetc/SCRUBD"])
# import os
# os.chdir("/kaggle/working/SCRUBD/SCRUBD-CD/data")

import os
import sys
import gc
import random
import json
import re
import math
import traceback
import time
from collections import OrderedDict
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel

# ==============================================================================
# 路径与超参数（按需修改；注释标明用途）
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "microsoft/codebert-base"
graph_model_name = "microsoft/graphcodebert-base"  # 额外强 baseline
unixcoder_model_name = "microsoft/unixcoder-base"  # 主流同类方法 baseline

# ========== 适配Kaggle路径 ==========
CSV_PATH = os.environ.get(
    "SCRUBD_LABELS",
    "/kaggle/working/SCRUBD/SCRUBD-CD/data/labels.csv",  
)
CODE_BASE_DIR = os.environ.get(
    "SCRUBD_CODES",
    "/kaggle/working/SCRUBD/SCRUBD-CD/data/solidity_codes", 
)


# 导出划分与 query 的目录（可选，用于论文复现）
OUTPUT_DIR = os.environ.get("VULN_IR_OUT", "/kaggle/working/vuln_ir_dataset") 

# 窗口：滑窗切分 token id，用于长合约；需与训练 Stage1 一致
window_max_length = 256
window_stride = 128

# 训练
batch_size = 8
lr = 2e-5
# Stage2（hard negative）单独学习率：通常要比 Stage1 小，减轻在验证集上的过拟合/伤泛化。
# 若验证仍明显低于 Stage1，可再降到 1e-6 或减小 epochs_stage2。
lr_stage2 = 5e-6
temperature = 0.05
epochs_stage1 = 8

# Stage2 hard negative（合约级，每条 query 取 HARD_K 个难负合约）
RUN_STAGE2_HARD = False  # Stage1-only 消融：跳过二阶段
HARD_K = 3
CAND_TOPN = 50  # 从相似度 Top-N 候选里排除正例后取 hard neg
epochs_stage2 = 2
CONTRACT_REP_WINDOWS = 3  # 每个合约用几个窗口来构建代表向量（越大越稳，越慢）
STAGE1_POS_WINDOWS = 3  # Stage1 每个样本采样几个正例窗口（多正例更稳）
# Stage1 正例采样模式：
#   - "window": 现有做法，随机采样窗口正例（窗口一致性训练）
#   - "trunc_first": 仅使用首窗口作为正例（近似“截断训练”）
TRAIN_POS_MODE = os.environ.get("TRAIN_POS_MODE", "window").strip().lower()
if TRAIN_POS_MODE not in ("window", "trunc_first"):
    raise ValueError(f"Invalid TRAIN_POS_MODE={TRAIN_POS_MODE}, expected 'window' or 'trunc_first'")

# ---------- 验证集早停 / 最佳 checkpoint（选项 A）----------
# 每个 epoch 后在 val_records 上算 window MRR；若连续 patience 轮无提升则提前结束该阶段。
EARLY_STOP_PATIENCE = 2  # 设为 0 表示不早停，只仍会在阶段结束后加载「验证 MRR 最高」的权重
VAL_EVAL_TOP_CHUNK = None  # None 表示与 window_eval_top_chunk 相同；验证可改小以加速
STAGE2_MIN_IMPROVEMENT = 0.002  # Stage2 验证 MRR 至少超过 Stage1 这么多才保留 Stage2 权重

# 评测：整段编码最大长度（截断 baseline）
eval_trunc_max_length = 128

# 窗口评测：先取相似度最高的 top_chunk 个 chunk 再按合约聚合（加速 + 避免漏窗）
window_eval_top_chunk = 2000
window_agg = "max"  # "max" / "mean" / "lse" / "topk_mean"
# topk_mean：每个合约只取 top-k 个 chunk 分数求均值（常用于提升 Hit@1 稳定性）
WINDOW_TOPK_MEAN_K = 5
# lse：logsumexp pooling 的温度（越小越接近 max；建议 0.05 起步）
WINDOW_LSE_TAU = 0.05
EVAL_BOTH_AGG = False  # 关闭双聚合消融，先按当前最优 mean 聚合稳定跑
ENABLE_FUSION = True  # trunc + window 融合评测开关
FUSION_ALPHA_GRID = [i / 20 for i in range(21)]  # alpha 从 0.00 到 1.00，步长 0.05
ENABLE_NORM_FUSION = True  # 先做按-query分数归一化，再搜索 alpha
ENABLE_RRF_FUSION = True  # rank-level 融合（对分数尺度不敏感）
RRF_K = 60
# DAC-Fusion: Distribution-Alignment Contrastive Fusion (Oracle-guided gate)
ENABLE_DAC_FUSION = os.environ.get("ENABLE_DAC_FUSION", os.environ.get("ENABLE_GATE_FUSION", "1")).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
DAC_EPOCHS = int(os.environ.get("DAC_EPOCHS", os.environ.get("GATE_EPOCHS", "20")))
DAC_LR = float(os.environ.get("DAC_LR", os.environ.get("GATE_LR", "1e-3")))
DAC_NEG_PER_QUERY = int(os.environ.get("DAC_NEG_PER_QUERY", os.environ.get("GATE_NEG_PER_QUERY", "32")))
DAC_HIDDEN_DIM = int(os.environ.get("DAC_HIDDEN_DIM", os.environ.get("GATE_HIDDEN_DIM", "256")))
DAC_DROPOUT = float(os.environ.get("DAC_DROPOUT", "0.2"))
DAC_TEMP = float(os.environ.get("DAC_TEMP", "0.05"))
# 在已训练好的「窗口一致性」模型上，额外评测 DAC 子模块消融（不增加主训练时间）
#   - dac_gate_raw_*：自适应门 + 原始 trunc/window 分数（不做按 query 的 temperature-softmax）
#   - dac_temp_fixed_*：按 query 对两路分数做 temperature-softmax 后，仅用验证集网格搜索的固定 α 融合（无门控）
EVAL_DAC_ABLATION_COMPONENTS = os.environ.get("EVAL_DAC_ABLATION_COMPONENTS", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# 同一 seed 内追加一次 TRAIN_POS_MODE=trunc_first 的 Stage1，用于论文表「基础 InfoNCE（无窗口一致性）+ DAC-Fusion」
# 约倍增 Stage1 时间；默认关闭，完整消融表设 RUN_TRUNC_TRAIN_DAC_ABLATION=1
RUN_TRUNC_TRAIN_DAC_ABLATION = os.environ.get("RUN_TRUNC_TRAIN_DAC_ABLATION", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
PRINT_ALPHA_TRACE = os.environ.get("PRINT_ALPHA_TRACE", "1").strip().lower() in ("1", "true", "yes", "on")
ENABLE_ADV_ANALYSIS = os.environ.get("ENABLE_ADV_ANALYSIS", "1").strip().lower() in ("1", "true", "yes", "on")
SIG_BOOTSTRAP_ROUNDS = int(os.environ.get("SIG_BOOTSTRAP_ROUNDS", "2000"))
CASE_BAD_RANK = int(os.environ.get("CASE_BAD_RANK", "50"))
CASE_GOOD_RANK = int(os.environ.get("CASE_GOOD_RANK", "3"))
CASE_TOPN = int(os.environ.get("CASE_TOPN", "20"))
DUMP_CASE_GATE_HEATMAP = os.environ.get("DUMP_CASE_GATE_HEATMAP", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
RUN_GRAPHCODEBERT_BASELINE = os.environ.get("RUN_GRAPHCODEBERT_BASELINE", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
graph_eval_trunc_max_length = 128
RUN_UNIXCODER_BASELINE = os.environ.get("RUN_UNIXCODER_BASELINE", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
unix_eval_trunc_max_length = 128


# 仅跑基线评测（跳过 CodeBERT 训练、截断/滑窗/融合等主流程）。
RUN_BASELINES_ONLY = os.environ.get("RUN_BASELINES_ONLY", "0").strip().lower() in ("1", "true", "yes", "on")
# BM25 基线：默认开启；只想跑部分基线时可设 RUN_BM25_BASELINE=0
RUN_BM25_BASELINE = os.environ.get("RUN_BM25_BASELINE", "1").strip().lower() in ("1", "true", "yes", "on")
# Sentence-Transformers 稠密基线（论文 Table 2：Sentence-BGE；需 pip install sentence-transformers）
RUN_ST_BASELINE = os.environ.get("RUN_ST_BASELINE", "0").strip().lower() in ("1", "true", "yes", "on")
ST_MODEL_NAME = os.environ.get("ST_MODEL_NAME", "BAAI/bge-small-en-v1.5")
ST_MAX_CODE_CHARS = int(os.environ.get("ST_MAX_CODE_CHARS", "8000"))
# 可选：BM25@K + 小 LLM 重排序（默认关闭；见 kaggle_notebook_llm_rerank_env.py）
RUN_LLM_RERANK_BASELINE = os.environ.get("RUN_LLM_RERANK_BASELINE", "0").strip().lower() in ("1", "true", "yes", "on")
LLM_RERANK_MODEL_NAME = os.environ.get("LLM_RERANK_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
LLM_RERANK_TOPK = int(os.environ.get("LLM_RERANK_TOPK", "12"))
LLM_RERANK_CODE_CHARS = int(os.environ.get("LLM_RERANK_CODE_CHARS", "500"))
LLM_RERANK_MAX_NEW_TOKENS = int(os.environ.get("LLM_RERANK_MAX_NEW_TOKENS", "12"))
LLM_RERANK_PROMPT_MAX_LENGTH = int(os.environ.get("LLM_RERANK_PROMPT_MAX_LENGTH", "3072"))


def refresh_data_source_globals_from_environ():
    """在 main 里若改写了与数据相关的 os.environ，同步模块全局变量。"""
    global DATA_SOURCE, EXTERNAL_JSONL_DIR
    DATA_SOURCE = os.environ.get("VULN_IR_DATA_SOURCE", "scrubd").strip().lower()
    EXTERNAL_JSONL_DIR = os.environ.get("VULN_IR_EXTERNAL_JSONL_DIR", "").strip()


def verify_paths():
    """验证数据路径是否存在（scrubd CSV 或 external_jsonl 三文件）。"""
    if DATA_SOURCE == "external_jsonl":
        if not EXTERNAL_JSONL_DIR or not os.path.isdir(EXTERNAL_JSONL_DIR):
            print("错误: VULN_IR_DATA_SOURCE=external_jsonl 时需设置 VULN_IR_EXTERNAL_JSONL_DIR 为有效目录")
            return False
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            p = os.path.join(EXTERNAL_JSONL_DIR, name)
            if not os.path.isfile(p):
                print(f"错误: 缺少文件 {p}")
                return False
        print(f"外部 JSONL 目录: {EXTERNAL_JSONL_DIR}")
        print(f"输出目录: {OUTPUT_DIR}")
        return True
    if not os.path.exists(CSV_PATH):
        print(f"警告: CSV 文件不存在: {CSV_PATH}")
        print("请确保已经正确克隆 SCRUBD 数据集")
        return False
    if not os.path.exists(CODE_BASE_DIR):
        print(f"警告: 代码目录不存在: {CODE_BASE_DIR}")
        print("请确保已经正确克隆 SCRUBD 数据集")
        return False
    print(f"CSV 路径: {CSV_PATH}")
    print(f"代码目录: {CODE_BASE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    return True


# 随机种子（保证划分与采样可复现）
# 可通过环境变量覆盖：SEED=2026 python kaggle.py
SEED = int(os.environ.get("SEED", "42"))

# ---------- 多种子实验 ----------
# 设为 True 后，会按 SEED_LIST 自动重复训练/评测，并输出均值与标准差。
RUN_MULTI_SEED = True
SEED_LIST = [42, 123, 456, 789, 2026]
SAVE_MULTI_SEED_RESULTS = True
# Kaggle 易中途断连：每跑完一个 seed 追加写入 JSONL，避免前面白跑
SAVE_PARTIAL_AFTER_EACH_SEED = True
PARTIAL_RESULTS_JSONL = "per_seed_results.jsonl"  # 位于 OUTPUT_DIR 下

# ---------- 极速模式（先尽快出结果） ----------
QUICK_RUN = os.environ.get("QUICK_RUN", "0").strip().lower() in ("1", "true", "yes", "on")
if QUICK_RUN:
    # 训练侧：尽量缩短时间
    epochs_stage1 = min(epochs_stage1, 1)
    RUN_STAGE2_HARD = False
    epochs_stage2 = 0
    CONTRACT_REP_WINDOWS = min(CONTRACT_REP_WINDOWS, 1)
    STAGE1_POS_WINDOWS = min(STAGE1_POS_WINDOWS, 1)
    EARLY_STOP_PATIENCE = 1

    # 评测侧：缩小候选窗口规模、减少 alpha 搜索
    window_eval_top_chunk = min(window_eval_top_chunk, 400)
    VAL_EVAL_TOP_CHUNK = 400
    FUSION_ALPHA_GRID = [0.0, 0.5, 1.0]
    ENABLE_NORM_FUSION = False
    ENABLE_RRF_FUSION = False

    # DAC：保留但减小开销（不想跑可直接 ENABLE_DAC_FUSION=0）
    DAC_EPOCHS = min(DAC_EPOCHS, 6)
    DAC_NEG_PER_QUERY = min(DAC_NEG_PER_QUERY, 8)
    EVAL_DAC_ABLATION_COMPONENTS = False
    RUN_TRUNC_TRAIN_DAC_ABLATION = False

    # 关闭耗时基线
    RUN_GRAPHCODEBERT_BASELINE = False
    RUN_UNIXCODER_BASELINE = False
    RUN_ST_BASELINE = False
    RUN_LLM_RERANK_BASELINE = False
    RUN_BM25_BASELINE = False

    # 只跑单种子，最快产出
    RUN_MULTI_SEED = False
    SAVE_MULTI_SEED_RESULTS = False
    SAVE_PARTIAL_AFTER_EACH_SEED = False

# ---------- 一键完整实验（论文/审稿补强：五种子 + DAC 全消融 + trunc 对照 + 案例热力图 + ST 基线等）----------
# 优先级：显式 VULN_IR_FULL_PAPER_RUN=1/0 >（未显式设置且 Kaggle Notebook 且非 QUICK_RUN 时默认开启）> 关闭
_on_kaggle_notebook = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").strip())
_explicit_full = "VULN_IR_FULL_PAPER_RUN" in os.environ
_full_raw = os.environ.get("VULN_IR_FULL_PAPER_RUN", "").strip().lower()
if _full_raw in ("1", "true", "yes", "on"):
    VULN_IR_FULL_PAPER_RUN = True
elif _full_raw in ("0", "false", "no", "off"):
    VULN_IR_FULL_PAPER_RUN = False
elif not _explicit_full and _on_kaggle_notebook and not QUICK_RUN:
    VULN_IR_FULL_PAPER_RUN = True
else:
    VULN_IR_FULL_PAPER_RUN = False

if VULN_IR_FULL_PAPER_RUN and not QUICK_RUN:
    RUN_MULTI_SEED = True
    SAVE_MULTI_SEED_RESULTS = True
    SAVE_PARTIAL_AFTER_EACH_SEED = True
    SEED_LIST = [42, 123, 456, 789, 2026]
    ENABLE_FUSION = True
    ENABLE_DAC_FUSION = True
    ENABLE_NORM_FUSION = True
    ENABLE_RRF_FUSION = True
    EVAL_DAC_ABLATION_COMPONENTS = True
    RUN_TRUNC_TRAIN_DAC_ABLATION = True
    ENABLE_ADV_ANALYSIS = True
    DUMP_CASE_GATE_HEATMAP = True
    RUN_BM25_BASELINE = True
    RUN_GRAPHCODEBERT_BASELINE = True
    RUN_UNIXCODER_BASELINE = True
    RUN_ST_BASELINE = True

# 全局模型句柄（main 里赋值后，encode_text 等才能用）
tokenizer = None
encoder = None

# 全局数据（main 里赋值）：df、各划分记录
df = None
train_records = None
val_records = None
test_records = None
train_loader = None


# ==============================================================================
# 工具：随机种子
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EfficiencyTracker:
    """统一记录各方法的编码时间、检索时间、显存占用"""
    def __init__(self):
        self.records = OrderedDict()
    
    def record(self, method_name, encode_time, search_time, 
               memory_mb=None, num_queries=None, num_candidates=None):
        self.records[method_name] = {
            'encode_time_s': encode_time,
            'search_time_s': search_time,
            'total_time_s': encode_time + search_time,
            'memory_mb': memory_mb if memory_mb else 
                        (torch.cuda.max_memory_allocated() / 1024**2 
                         if torch.cuda.is_available() else 0),
            'num_queries': num_queries,
            'num_candidates': num_candidates,
        }
    
    def summary(self):
        """打印效率对比表"""
        print("\n" + "=" * 80)
        print("【效率分析】各方法推理时间与显存对比")
        print("=" * 80)
        print(f"{'Method':<25} {'Encode(s)':<12} {'Search(s)':<12} {'Total(s)':<12} {'Memory(MB)':<12}")
        print("-" * 73)
        for name, rec in self.records.items():
            print(f"{name:<25} {rec['encode_time_s']:<12.2f} {rec['search_time_s']:<12.2f} "
                  f"{rec['total_time_s']:<12.2f} {rec['memory_mb']:<12.1f}")
        print("=" * 80)
        return self.records
    
    def to_csv(self, filepath):
        """导出效率数据到 CSV"""
        rows = []
        for name, rec in self.records.items():
            rows.append({'method': name, **rec})
        pd.DataFrame(rows).to_csv(filepath, index=False, encoding='utf-8')
        print(f"效率分析已导出到: {filepath}")

# 全局效率追踪器
efficiency_tracker = EfficiencyTracker()

def cleanup_gpu():
    """每个 seed 结束后释放显存，降低 Kaggle OOM / 被系统 kill 的概率。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def append_partial_result(row: dict):
    """将单个 seed 的指标追加写入 JSONL，中断后仍可保留已完成种子。"""
    if not SAVE_PARTIAL_AFTER_EACH_SEED:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, PARTIAL_RESULTS_JSONL)

    def _to_jsonable(v):
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    row_safe = {k: _to_jsonable(v) for k, v in row.items()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row_safe, ensure_ascii=False) + "\n")
    print(f"已追加保存本 seed 结果到: {path}")


def load_partial_results():
    """
    读取已保存的 per-seed JSONL（若存在），用于断点续跑。
    返回:
      rows: list[dict]
      completed: set[int]
    """
    path = os.path.join(OUTPUT_DIR, PARTIAL_RESULTS_JSONL)
    if not os.path.isfile(path):
        return [], set()
    rows = []
    completed = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
                try:
                    completed.add(int(obj.get("seed")))
                except Exception:
                    pass
    return rows, completed


# ==============================================================================
# 1) 窗口：把长合约切成 token 窗口（不 decode，避免二次分词漂移）
# ==============================================================================
def build_id_windows(code_text, tokenizer_obj, max_length=256, stride=128):
    """
    输入完整合约字符串，输出若干段 input_ids 窗口（list[list[int]]）。
    """
    code_text = (code_text or "").strip()
    if not code_text:
        return []
    # 先拿“无 special token”的原始 token 序列，再对每个窗口手动拼 special token。
    # 这样能避免不同 transformers 版本下 tokenizer API 差异导致的 AttributeError。
    # 先 tokenize 再转 id，可避免 tokenizer 在超长字符串上抛出长度告警。
    tokens = tokenizer_obj.tokenize(code_text)
    ids = tokenizer_obj.convert_tokens_to_ids(tokens)

    windows = []
    cls_id = tokenizer_obj.cls_token_id
    sep_id = tokenizer_obj.sep_token_id
    reserved = 0 
    if cls_id is not None:
        reserved += 1
    if sep_id is not None:
        reserved += 1

    # 预留 special token 位置，保证窗口总长度不超过 max_length
    payload_len = max(1, max_length - reserved)
    step = max(1, min(stride, payload_len))
    start = 0 # 给每个窗口添加special token
    while start < len(ids):
        end = min(start + payload_len, len(ids))
        win_payload = ids[start:end]
        win_ids = []
        if cls_id is not None:
            win_ids.append(cls_id)
        win_ids.extend(win_payload)
        if sep_id is not None:
            win_ids.append(sep_id)
        windows.append(win_ids)
        if end == len(ids):
            break
        start += step
    return windows


# ==============================================================================
# 2) 批量编码窗口 id（推理/建索引用，no_grad）
# ==============================================================================
def encode_id_windows_in_batches(list_of_id_windows, max_length, batch_size=32):
    """将 chunk 窗口批量编码为 L2 归一化向量，用于 chunk 索引。"""
    global tokenizer, encoder
    embs = []
    encoder.eval()
    with torch.no_grad(): # 推理模式,只生成向量
        for i in tqdm(range(0, len(list_of_id_windows), batch_size), desc="Encode chunks(ids)"):
            batch_ids = list_of_id_windows[i : i + batch_size]
            inputs = tokenizer.pad(
                {"input_ids": batch_ids},
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)
            out = encoder(**inputs).last_hidden_state[:, 0, :] # 补齐窗口
            out = F.normalize(out, dim=-1) # 向量标准化
            embs.append(out.cpu())
    return torch.cat(embs, dim=0)


# ==============================================================================
# 3) 文本 query 编码（训练时需梯度；评测时 no_grad=True）
# ==============================================================================
def encode_text(texts, max_length=128, no_grad=True):
    """
    对自然语言 query（或短文本）做 [CLS] 向量，默认 L2 归一化。
    训练时务必 no_grad=False，否则 loss 无法反传。
    """
    global tokenizer, encoder
    # 让输入 tensor 跟随 encoder 所在设备，避免中途把 encoder 移到 CPU/GPU 后发生 device mismatch
    enc_device = next(encoder.parameters()).device
    inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).to(enc_device)
    if no_grad:
        with torch.no_grad():
            out = encoder(**inputs).last_hidden_state[:, 0, :]
    else:
        out = encoder(**inputs).last_hidden_state[:, 0, :]
    return F.normalize(out, dim=-1)


# ==============================================================================
# 4) 读取 SCRUBD：标签 CSV + 源码目录
# ==============================================================================
def get_code_file_map(code_dir: str):
    """
    建立 合约名/词干 -> 文件路径 的确定性映射，避免 str in filename 误匹配。
    便于快速通过合约名找到代码文件
    """
    stem_to_path = {}
    if not os.path.isdir(code_dir):
        return stem_to_path
    for fname in os.listdir(code_dir):
        if not fname.endswith(".sol"):
            continue
        stem = os.path.splitext(fname)[0]
        stem_to_path[stem] = os.path.join(code_dir, fname)
    return stem_to_path


def read_solidity(stem_to_path: dict, name) -> str:
    """根据 labels 中的 Smart Contract 字段读取源码。"""
    key = str(name).strip()
    if not key:
        return ""
    if key in stem_to_path:
        path = stem_to_path[key]
    elif key.endswith(".sol") and os.path.isfile(os.path.join(CODE_BASE_DIR, key)):
        path = os.path.join(CODE_BASE_DIR, key)
    else:
        path = stem_to_path.get(key.replace(".sol", ""), "")
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def _canonical_vuln_label(v) -> str:
    """统一漏洞标签字符串；空、nan、normal 视为无漏洞。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "normal"):
        return ""
    return s


def _parse_binary_vuln_map():
    """
    环境变量 VULN_IR_BINARY_VULN_MAP，格式：列名:漏洞类型,...
    例：RE:reentrancy,UX:unchecked_exception,IO:integer_overflow
    """
    raw = os.environ.get("VULN_IR_BINARY_VULN_MAP", "").strip()
    if not raw:
        return None
    out = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        k, v = k.strip(), v.strip()
        if k and v:
            out[k] = v
    return out or None


def _row_vuln_from_binary(row: pd.Series, col_to_type: dict) -> str:
    """按 col_to_type 顺序取第一个值为 1 的列，返回对应漏洞类型；否则 normal。"""
    for col, vtype in col_to_type.items():
        if col not in row.index:
            continue
        val = row[col]
        try:
            iv = int(val)
        except (TypeError, ValueError):
            iv = 1 if val in (True, "1", 1.0, "true", "True", "yes", "Y") else 0
        if iv == 1:
            return vtype
    return "normal"


def load_dataframe(csv_path: str, code_dir: str):
    """
    读取 labels.csv，生成 vuln_type，并挂上 code 列。

    漏洞类型来源（优先级）：
      1) CSV 已有列 vuln_type
      2) 环境变量 VULN_IR_VULN_COLUMN 指定的列名
      3) 环境变量 VULN_IR_BINARY_VULN_MAP（如 RE:reentrancy,UX:unchecked_exception）
      4) 若存在 RE、UX 列：兼容原 SCRUBD 二分类

    丢弃无有效漏洞标签或 code 为空的行。
    """
    df_local = pd.read_csv(csv_path)
    env_vuln = os.environ.get("VULN_IR_VULN_COLUMN", "").strip()

    if "vuln_type" in df_local.columns:
        df_local["vuln_type"] = df_local["vuln_type"].map(_canonical_vuln_label)
    elif env_vuln and env_vuln in df_local.columns:
        df_local["vuln_type"] = df_local[env_vuln].map(_canonical_vuln_label)
    else:
        bmap = _parse_binary_vuln_map()
        if bmap is not None:
            df_local["vuln_type"] = df_local.apply(
                lambda row: _canonical_vuln_label(_row_vuln_from_binary(row, bmap)), axis=1
            )
        elif "RE" in df_local.columns and "UX" in df_local.columns:
            default_map = {"RE": "reentrancy", "UX": "unchecked_exception"}
            df_local["vuln_type"] = df_local.apply(
                lambda row: _canonical_vuln_label(_row_vuln_from_binary(row, default_map)), axis=1
            )
        else:
            raise SystemExit(
                "无法推断漏洞类型：请在 CSV 中提供 vuln_type 列，或设置环境变量 "
                "VULN_IR_VULN_COLUMN，或 VULN_IR_BINARY_VULN_MAP（列名:类型,...），"
                "或使用含 RE、UX 列的 SCRUBD labels.csv。"
            )

    df_local = df_local[df_local["vuln_type"].str.len() > 0].reset_index(drop=True)
    stem_map = get_code_file_map(code_dir)
    df_local["code"] = df_local["Smart Contract"].apply(lambda n: read_solidity(stem_map, n))
    df_local = df_local[df_local["code"].str.strip().str.len() > 0].reset_index(drop=True)
    return df_local


# ==============================================================================
# 5) 构建 query 集：多模板 × 多样 query（Vuln-IR 基准）
# ==============================================================================
# 定义漏洞描述模版
REENTRANCY_TEMPLATES = [
    "reentrancy vulnerability in smart contract",
    "external call before state update",
    "reentrancy attack due to fallback function",
    "missing checks-effects-interactions pattern",
    "reentrant call caused by call.value or low-level call",
]

UNCHECKED_TEMPLATES = [
    "unchecked exception in low-level call",
    "low-level call return value not checked",
    "unchecked send or call return",
    "missing require for external call result",
    "silent failure due to unchecked call",
]

KNOWN_QUERY_TEMPLATES = {
    "reentrancy": REENTRANCY_TEMPLATES,
    "unchecked_exception": UNCHECKED_TEMPLATES,
}


def query_templates_for_vuln(lab: str) -> list:
    """按漏洞类型取 query 模板；未知类型用语义泛化模板。"""
    if lab in KNOWN_QUERY_TEMPLATES:
        return list(KNOWN_QUERY_TEMPLATES[lab])
    readable = str(lab).replace("_", " ").strip() or "vulnerability"
    return [
        f"{readable} vulnerability in Solidity smart contract",
        f"security flaw related to {readable}",
        f"code defect: {readable}",
        f"potential {readable} exploit in smart contract",
        f"scan for {readable} issues",
    ]


def build_query_records(df_local: pd.DataFrame, max_queries_per_sample: int = 3):
    """
    每条合约样本随机抽取若干条自然语言 query，形成 (query, code, sample_id, vuln_type) 记录。
    reset_index 后行号即 sample_id。
    """
    records = []
    for idx, row in df_local.iterrows():
        lab = row["vuln_type"]
        pool = query_templates_for_vuln(lab)
        k = min(max_queries_per_sample, len(pool))
        for q in random.sample(pool, k):
            records.append(
                {
                    "sample_id": int(idx),
                    "query": q,
                    "code": row["code"],
                    "vuln_type": lab,
                }
            )
    return records


def split_by_contract(query_records, train_ratio=0.7, val_ratio=0.15):
    """
    按 sample_id（合约级）划分，避免同一合约的 query 同时出现在 train 与 test。
    """
    all_ids = sorted({r["sample_id"] for r in query_records})
    random.shuffle(all_ids)
    n = len(all_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = set(all_ids[:n_train])
    val_ids = set(all_ids[n_train : n_train + n_val])
    test_ids = set(all_ids[n_train + n_val :])

    def pick(id_set):
        return [r for r in query_records if r["sample_id"] in id_set]

    return pick(train_ids), pick(val_ids), pick(test_ids), train_ids, val_ids, test_ids


def export_jsonl(records, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str):
    """读取 JSONL，每行一条与 export_jsonl 相同结构的记录。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def dataframe_from_query_splits(train_records, val_records, test_records):
    """
    由 train/val/test 记录构建全局 df（index=sample_id），供评测与滑窗索引使用。
    要求同一 sample_id 的 code、vuln_type 在各划分中一致。
    """
    sid_to = {}
    for r in train_records + val_records + test_records:
        sid = int(r["sample_id"])
        code = r["code"]
        vt = r.get("vuln_type", "")
        if sid in sid_to:
            if sid_to[sid]["code"] != code:
                raise ValueError(f"sample_id={sid} 的 code 在不同划分中不一致")
        else:
            sid_to[sid] = {"code": code, "vuln_type": vt}
    idx = sorted(sid_to.keys())
    return pd.DataFrame([{**sid_to[i], "sample_id": i} for i in idx]).set_index("sample_id")


# ==============================================================================
# 6) PyTorch Dataset / DataLoader
# ==============================================================================
class VulnIRDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "query": r["query"],
            "code": r["code"],
            "vuln_type": r["vuln_type"],
            "sample_id": r["sample_id"],
        }


def collate_fn(batch):
    return {
        "queries": [b["query"] for b in batch],
        "codes": [b["code"] for b in batch],
        "vuln_types": [b["vuln_type"] for b in batch],
        "sample_ids": [b["sample_id"] for b in batch],
    }


# ==============================================================================
# 7) Stage1：窗口一致性训练（随机窗口作正例，与 window 评测对齐）
# ==============================================================================
def encode_id_windows_train(list_of_id_windows):
    """训练时编码窗口，需要保留梯度。"""
    global tokenizer, encoder
    inputs = tokenizer.pad(
        {"input_ids": list_of_id_windows},
        padding="max_length",
        max_length=window_max_length,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)
    return encoder(**inputs).last_hidden_state[:, 0, :]


def run_stage1_training(val_records=None, unique_val_ids=None, train_pos_mode=None):
    """
    Stage1：对比学习。
    train_pos_mode / TRAIN_POS_MODE="window" 时为窗口一致性训练；
    "trunc_first" 时仅用首窗口作正例（近似截断训练，与截断推理证据路径更一致）。
    若提供 val_records / unique_val_ids：每个 epoch 后在验证集上算 window MRR，
    保存验证 MRR 最高的 encoder；支持 EARLY_STOP_PATIENCE 早停。
    """
    global tokenizer, encoder, train_loader, df

    mode = (train_pos_mode if train_pos_mode is not None else TRAIN_POS_MODE).strip().lower()
    if mode not in ("window", "trunc_first"):
        raise ValueError(f"Invalid train_pos_mode={mode}, expected 'window' or 'trunc_first'")

    use_val = val_records is not None and len(val_records) > 0 and unique_val_ids is not None and len(unique_val_ids) > 0
    best_val_mrr = -1.0
    best_state_cpu = None
    patience_left = EARLY_STOP_PATIENCE if use_val and EARLY_STOP_PATIENCE > 0 else None

    def multi_positive_contrastive_loss(q_emb, c_emb_multi, temp=0.05):
        """
        q_emb: [B, D]
        c_emb_multi: [B, P, D]，每个 query 对应 P 个正例窗口。
        采用多正例对比损失：logsumexp(正例) - logsumexp(全部候选)。
        让查询向量和代码窗口向量对齐。
        """
        bsz, pnum, dim = c_emb_multi.shape
        q_emb = F.normalize(q_emb, dim=-1)
        c_emb_multi = F.normalize(c_emb_multi, dim=-1)

        c_flat = c_emb_multi.reshape(bsz * pnum, dim)  # [B*P, D]
        logits = (q_emb @ c_flat.t()) / temp            # [B, B*P]

        # 为每一行构造正例 mask：第 i 行的正例列为 [i*P, i*P+P)
        pos_mask = torch.zeros_like(logits, dtype=torch.bool)
        for i in range(bsz):
            s = i * pnum
            e = s + pnum
            pos_mask[i, s:e] = True

        # 多正例InfoNCE
        logits_pos = logits.masked_fill(~pos_mask, float("-inf"))
        pos_term = torch.logsumexp(logits_pos, dim=1)
        all_term = torch.logsumexp(logits, dim=1)
        loss = -(pos_term - all_term).mean()
        return loss

    sample_windows_cache = {}

    def get_windows_for_sample(sid: int):
        sid = int(sid)
        if sid not in sample_windows_cache:
            code = df.loc[sid, "code"]
            sample_windows_cache[sid] = build_id_windows(
                code, tokenizer, max_length=window_max_length, stride=window_stride
            )
        return sample_windows_cache[sid]

    encoder.train()
    optimizer = AdamW(encoder.parameters(), lr=lr)
    print(f"Stage1 train_pos_mode={mode}")

    for epoch in range(epochs_stage1):
        encoder.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}"):
            queries = batch["queries"]
            sample_ids = batch["sample_ids"]

            q_emb = encode_text(queries, max_length=64, no_grad=False)

            pos_windows = []
            for sid in sample_ids:
                wins = get_windows_for_sample(sid)
                if not wins:
                    fb = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.eos_token_id
                    one = [[fb] for _ in range(STAGE1_POS_WINDOWS)]
                    pos_windows.append(one)
                else:
                    if mode == "trunc_first":
                        picked = [wins[0] for _ in range(STAGE1_POS_WINDOWS)]
                    else:
                        k = min(STAGE1_POS_WINDOWS, len(wins))
                        picked = random.sample(wins, k)
                        if k < STAGE1_POS_WINDOWS:
                            picked += [random.choice(wins) for _ in range(STAGE1_POS_WINDOWS - k)]
                    pos_windows.append(picked)

            flat = [w for one in pos_windows for w in one]  # [B*P]
            c_flat = encode_id_windows_train(flat)           # [B*P, D]
            c_emb_multi = c_flat.view(len(queries), STAGE1_POS_WINDOWS, -1)
            loss = multi_positive_contrastive_loss(q_emb, c_emb_multi, temp=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Stage1 Epoch {epoch+1} mean loss: {total_loss / len(train_loader):.4f}")

        # ---------- 验证集：MRR + 最佳权重 + 早停 ----------
        if use_val:
            val_mrr = validate_window_mrr(val_records, unique_val_ids, show_progress=False)
            print(f"Stage1 Epoch {epoch+1} 验证集 window MRR: {val_mrr:.6f}")
            if val_mrr > best_val_mrr + 1e-8:
                best_val_mrr = val_mrr
                best_state_cpu = _encoder_state_dict_cpu()
                print("  -> 更新最佳 checkpoint（验证 MRR 提升）")
                if patience_left is not None:
                    patience_left = EARLY_STOP_PATIENCE
            elif patience_left is not None:
                patience_left -= 1
                print(f"  -> 验证未提升，早停计数 {patience_left}/{EARLY_STOP_PATIENCE}")
                if patience_left == 0:
                    print("Stage1 早停：连续多轮验证 MRR 未提升，结束本阶段训练。")
                    break

    if use_val and best_state_cpu is not None:
        _load_encoder_state_dict_cpu(best_state_cpu)
        print(f"Stage1 结束：已加载验证集 MRR 最高的权重（best val MRR={best_val_mrr:.6f}）")
    elif use_val:
        print("Stage1 结束：验证集未产生有效 checkpoint，保留最后一轮权重。")

    return best_val_mrr if use_val else -1.0, best_state_cpu


# ==============================================================================
# 8) 为 Stage2 准备：训练集每个合约一个代表向量（用于挖 hard neg）
# ==============================================================================
def build_train_contract_index():
    """
    对每个训练合约随机选一个代表窗口并编码，得到 [N_train, D] 矩阵，
    与 train_contract_ids 顺序一致，供 query 与全体训练合约相似度检索。
    """
    global tokenizer, encoder, df, train_records

    train_contract_ids = sorted({r["sample_id"] for r in train_records}) # 去重合约ID

    # 为每个合约采样多个窗口并聚合，减少“单窗口偶然性”导致的 hard-neg 噪声。
    flat_windows = []
    contract_offsets = []  # (start_idx, end_idx) in flat_windows
    for sid in tqdm(train_contract_ids, desc="Sample rep windows"):
        wins = build_id_windows(df.loc[sid, "code"], tokenizer, window_max_length, window_stride)
        if not wins:
            fb = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.eos_token_id
            chosen = [[fb]]
        else:
            k = min(CONTRACT_REP_WINDOWS, len(wins))
            chosen = random.sample(wins, k)
        s = len(flat_windows)
        flat_windows.extend(chosen)
        e = len(flat_windows)
        contract_offsets.append((s, e))

    # 一次性批量编码，再按合约聚合（max pooling）。
    all_emb = encode_id_windows_in_batches(
        flat_windows, max_length=window_max_length, batch_size=32
    )  # [sum_k, D]

    reps = []
    for s, e in contract_offsets:
        reps.append(torch.max(all_emb[s:e], dim=0).values)
    train_contract_emb = torch.stack(reps, dim=0)  # [N_contract, D]

    print(f"[Stage2 prep] num train contracts: {len(train_contract_ids)}, emb shape: {train_contract_emb.shape}")
    return train_contract_ids, train_contract_emb


# ==============================================================================
# 9) Stage2：合约级 hard negative + 小分类式 InfoNCE（正例在候选第 0 位）
# ==============================================================================
def mine_hard_negatives(train_records_local, train_contract_ids, train_contract_emb):
    """
    对每条训练 query：用当前模型算 query 与所有训练合约向量的相似度，
    采用“混合难负例”策略（默认 2 个异类 + 1 个同类）生成 HARD_K 个负样本。
    """
    global tokenizer, encoder

    train_records_hard = []
    encoder.eval()
    with torch.no_grad():
        for rec in tqdm(train_records_local, desc="Mine hard negatives"):
            q = rec["query"]
            pos_sid = int(rec["sample_id"])
            q_emb = encode_text([q], max_length=64, no_grad=True).cpu()
            sims = (q_emb @ train_contract_emb.t())[0]
            top_idx = torch.topk(sims, k=min(CAND_TOPN, sims.numel())).indices.tolist()

            pos_lab = df.loc[pos_sid, "vuln_type"]
            inter_negs = []  # 异类
            intra_negs = []  # 同类（但不同合约）
            for j in top_idx:
                sid = int(train_contract_ids[j])
                if sid == pos_sid:
                    continue
                lab = df.loc[sid, "vuln_type"]
                if lab != pos_lab:
                    inter_negs.append(sid)
                else:
                    intra_negs.append(sid)
                if len(inter_negs) + len(intra_negs) >= CAND_TOPN:
                    break

            # 目标配比：2 个异类 + 1 个同类（若 HARD_K 改了则自动适配）
            target_inter = min(2, HARD_K)
            target_intra = max(0, HARD_K - target_inter)

            neg_sids = []
            neg_sids.extend(inter_negs[:target_inter])
            neg_sids.extend(intra_negs[:target_intra])

            # 不足时从剩余候选补齐
            leftovers = [x for x in (inter_negs[target_inter:] + intra_negs[target_intra:]) if x not in neg_sids]
            for sid in leftovers:
                if len(neg_sids) >= HARD_K:
                    break
                neg_sids.append(sid)

            # 仍不足时随机补齐（优先异类）
            inter_pool = [x for x in train_contract_ids if x != pos_sid and df.loc[x, "vuln_type"] != pos_lab]
            intra_pool = [x for x in train_contract_ids if x != pos_sid and df.loc[x, "vuln_type"] == pos_lab]
            while len(neg_sids) < HARD_K and (inter_pool or intra_pool):
                pool = inter_pool if inter_pool else intra_pool
                cand = random.choice(pool)
                if cand not in neg_sids:
                    neg_sids.append(cand)

            new_r = dict(rec)
            new_r["hard_neg_ids"] = neg_sids[:HARD_K]
            train_records_hard.append(new_r)

    return train_records_hard


class HardNegDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "query": r["query"],
            "pos_sid": int(r["sample_id"]),
            "neg_sids": [int(x) for x in r["hard_neg_ids"]],
        }


def hard_collate(batch):
    return {
        "queries": [b["query"] for b in batch],
        "pos_sids": [b["pos_sid"] for b in batch],
        "neg_sids": [b["neg_sids"] for b in batch],
    }


def hard_nce_loss(q_emb, cand_emb, temp=0.05):
    """
    q_emb: [B, D]
    cand_emb: [B, 1+HARD_K, D]，第 0 维为正例窗口，其余为难负合约窗口
    """
    q_emb = F.normalize(q_emb, dim=-1)
    cand_emb = F.normalize(cand_emb, dim=-1)
    logits = torch.einsum("bd,bkd->bk", q_emb, cand_emb) / temp
    labels = torch.zeros(q_emb.size(0), dtype=torch.long, device=q_emb.device)
    return F.cross_entropy(logits, labels)


def run_stage2_training(train_records_hard, val_records=None, unique_val_ids=None, stage1_best_mrr=-1.0, stage1_state_cpu=None):
    """
    Stage2：hard negative 训练。
    同样支持验证 MRR 最佳权重与早停；若整阶段验证未超过 Stage1 的 best MRR，则回退到 Stage1 权重。
    """
    global tokenizer, encoder, df

    use_val = val_records is not None and len(val_records) > 0 and unique_val_ids is not None and len(unique_val_ids) > 0
    best_val_mrr = -1.0
    best_state_cpu = None
    patience_left = EARLY_STOP_PATIENCE if use_val and EARLY_STOP_PATIENCE > 0 else None

    def pick_win(sid):
        wins = build_id_windows(df.loc[sid, "code"], tokenizer, window_max_length, window_stride)
        if wins:
            return random.choice(wins)
        fb = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.eos_token_id
        return [fb]

    loader = DataLoader(
        HardNegDataset(train_records_hard),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=hard_collate,
    )

    encoder.train()
    optimizer = AdamW(encoder.parameters(), lr=lr_stage2)
    print(f"Stage2 优化器学习率 lr_stage2={lr_stage2}（Stage1 使用 lr={lr}）")

    for epoch in range(epochs_stage2):
        encoder.train()
        total = 0.0
        for batch in tqdm(loader, desc=f"Stage2 hard Epoch {epoch+1}"):
            queries = batch["queries"]
            pos_sids = batch["pos_sids"]
            neg_sids_list = batch["neg_sids"]

            q_emb = encode_text(queries, max_length=64, no_grad=False)

            flat = []
            for ps, ns in zip(pos_sids, neg_sids_list):
                one = [pick_win(ps)]
                for nid in ns:
                    one.append(pick_win(nid))
                flat.extend(one)

            c_flat = encode_id_windows_train(flat)
            c_emb = c_flat.view(len(queries), 1 + HARD_K, -1)

            loss = hard_nce_loss(q_emb, c_emb, temp=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        print(f"Stage2 Epoch {epoch+1} mean loss: {total / len(loader):.4f}")

        if use_val:
            val_mrr = validate_window_mrr(val_records, unique_val_ids, show_progress=False)
            print(f"Stage2 Epoch {epoch+1} 验证集 window MRR: {val_mrr:.6f}")
            if val_mrr > best_val_mrr + 1e-8:
                best_val_mrr = val_mrr
                best_state_cpu = _encoder_state_dict_cpu()
                print("  -> 更新 Stage2 最佳 checkpoint")
                if patience_left is not None:
                    patience_left = EARLY_STOP_PATIENCE
            elif patience_left is not None:
                patience_left -= 1
                print(f"  -> 验证未提升，早停计数 {patience_left}/{EARLY_STOP_PATIENCE}")
                if patience_left == 0:
                    print("Stage2 早停。")
                    break

    if use_val and best_state_cpu is not None:
        _load_encoder_state_dict_cpu(best_state_cpu)
        print(f"Stage2 结束：已加载本阶段验证 MRR 最高权重（best val MRR={best_val_mrr:.6f}）")
        # 若 Stage2 没超过 Stage1，回退（避免二阶段把检索变差）
        if (
            stage1_state_cpu is not None
            and stage1_best_mrr > -0.5
            and best_val_mrr < stage1_best_mrr + STAGE2_MIN_IMPROVEMENT
        ):
            _load_encoder_state_dict_cpu(stage1_state_cpu)
            print(
                f"Stage2 最佳验证 MRR ({best_val_mrr:.6f}) 未达到 Stage1+阈值 "
                f"({stage1_best_mrr:.6f}+{STAGE2_MIN_IMPROVEMENT:.4f})，已回退到 Stage1 权重。"
            )
            best_val_mrr = stage1_best_mrr
    elif use_val:
        print("Stage2 结束：验证集未产生有效 checkpoint，保留最后一轮权重。")

    return best_val_mrr if use_val else -1.0, best_state_cpu


# ==============================================================================
# 10) 评测：整段截断（instance）与滑窗聚合（window）
# ==============================================================================
def evaluate_instance(test_recs, code_embeddings_tensor, code_id_list, k_list=(1, 5, 10)):
    """query 与「每个测试合约一个向量」比相似度，Hit@k / MRR。"""
    hits = {k: 0 for k in k_list}
    mrr_sum = 0.0
    num = 0
    code_emb = code_embeddings_tensor
    enc_device = next(encoder.parameters()).device
    if code_emb.device != enc_device:
        code_emb = code_emb.to(enc_device)

    encode_time_total = 0.0
    search_time_total = 0.0
    
    encoder.eval()
    with torch.no_grad():
        for rec in tqdm(test_recs, desc="Evaluate (trunc)"):
            q = rec["query"]
            target_id = rec["sample_id"]
            
            t0 = time.time()
            q_emb = encode_text([q], max_length=64, no_grad=True)
            encode_time_total += time.time() - t0
            
            t0 = time.time()
            sims = (q_emb @ code_emb.t())[0]
            rank_idx = torch.argsort(sims, descending=True)
            ranked_ids = [code_id_list[i] for i in rank_idx.tolist()]
            search_time_total += time.time() - t0
            
            num += 1
            for kk in k_list:
                if target_id in ranked_ids[:kk]:
                    hits[kk] += 1
            for r, cid in enumerate(ranked_ids):
                if cid == target_id:
                    mrr_sum += 1.0 / (r + 1)
                    break
    
    # 记录效率
    global efficiency_tracker
    efficiency_tracker.record(
        'Trunc',
        encode_time=encode_time_total,
        search_time=search_time_total,
        num_queries=num,
        num_candidates=len(code_id_list)
    )
    print(f"[Efficiency] Trunc: encode={encode_time_total:.2f}s, search={search_time_total:.2f}s, "
          f"total={encode_time_total+search_time_total:.2f}s ({num} queries × {len(code_id_list)} candidates)")
    
    return {k: hits[k] / num for k in k_list}, mrr_sum / num


def evaluate_window(
    test_recs,
    chunk_emb_tensor,
    chunk_to_contract_list,
    k_list=(1, 5, 10),
    top_chunk=None,
    agg="max",
    show_progress=True,
):
    """
    先算 query 与每个 chunk 的相似度，再对同一合约下的 chunk 分数做 max/mean 聚合。
    top_chunk：只取相似度最高的前 top_chunk 个 chunk 参与聚合，加速且长合约必用较大值。
    """
    hits = {k: 0 for k in k_list}
    mrr_sum = 0.0
    num = 0
    chunk_emb = chunk_emb_tensor
    if chunk_emb.device != device:
        chunk_emb = chunk_emb.cpu()

    encode_time_total = 0.0
    search_time_total = 0.0

    encoder.eval()
    iterator = tqdm(test_recs, desc="Evaluate (window)") if show_progress else test_recs
    with torch.no_grad():
        for rec in iterator:
            q = rec["query"]
            target_id = rec["sample_id"]

            t0 = time.time()
            q_emb = encode_text([q], max_length=64, no_grad=True).cpu()
            encode_time_total += time.time() - t0

            t0 = time.time()
            sims = (q_emb @ chunk_emb.t())[0]

            if top_chunk is not None and top_chunk < sims.numel():
                top_idx = torch.topk(sims, k=top_chunk).indices.tolist()
            else:
                top_idx = torch.argsort(sims, descending=True).tolist()

            # 把每个窗口相似度按合约分组
            contract_scores = defaultdict(list)
            for i in top_idx:
                cid = chunk_to_contract_list[i]
                contract_scores[cid].append(float(sims[i]))

            if agg == "max":
                scored = [(cid, max(ss)) for cid, ss in contract_scores.items()]
            elif agg == "mean":
                scored = [(cid, sum(ss) / len(ss)) for cid, ss in contract_scores.items()]
            elif agg == "topk_mean":
                k = max(1, int(WINDOW_TOPK_MEAN_K))
                scored = []
                for cid, ss in contract_scores.items():
                    ss_sorted = sorted(ss, reverse=True)
                    topk = ss_sorted[: min(k, len(ss_sorted))]
                    scored.append((cid, sum(topk) / len(topk)))
            elif agg == "lse":
                # logsumexp pooling：tau 越小越接近 max，但更平滑，常能提升 top1 排序稳定性
                tau = float(WINDOW_LSE_TAU)
                scored = []
                for cid, ss in contract_scores.items():
                    x = np.array(ss, dtype=np.float32) / max(tau, 1e-6)
                    m = float(x.max())
                    s = (m + math.log(float(np.exp(x - m).sum()))) * tau
                    scored.append((cid, float(s)))
            else:
                raise ValueError("agg must be 'max'/'mean'/'topk_mean'/'lse'")

            scored.sort(key=lambda x: x[1], reverse=True)
            ranked_contracts = [cid for cid, _ in scored]
            search_time_total += time.time() - t0
            
            num += 1
            for kk in k_list:
                if target_id in ranked_contracts[:kk]:
                    hits[kk] += 1
            for r, cid in enumerate(ranked_contracts):
                if cid == target_id:
                    mrr_sum += 1.0 / (r + 1)
                    break
        
    # 记录效率
    global efficiency_tracker
    efficiency_tracker.record(
        f'Window ({agg})',
        encode_time=encode_time_total,
        search_time=search_time_total,
        num_queries=num,
        num_candidates=len(set(chunk_to_contract_list))
    )
    print(f"[Efficiency] Window({agg}): encode={encode_time_total:.2f}s, search={search_time_total:.2f}s, "
          f"total={encode_time_total+search_time_total:.2f}s ({num} queries × {len(chunk_emb)} chunks)")

    return {k: hits[k] / num for k in k_list}, mrr_sum / num


def precompute_window_contract_scores(records, chunk_emb_tensor, chunk_to_contract_list, top_chunk, agg):
    """
    预计算每条 query 对每个合约的 window 聚合分数，返回：
    - qids: 与 records 对齐的 sample_id 列表
    - score_matrix: [num_queries, num_contracts]
    - contract_ids: 列索引对应的 contract id 顺序
    """
    contract_ids = sorted(set(chunk_to_contract_list))
    cid_to_col = {cid: i for i, cid in enumerate(contract_ids)}
    score_matrix = np.full((len(records), len(contract_ids)), -1e9, dtype=np.float32)

    chunk_emb = chunk_emb_tensor
    if chunk_emb.device != device:
        chunk_emb = chunk_emb.cpu()

    encoder.eval()
    with torch.no_grad():
        for qi, rec in enumerate(tqdm(records, desc=f"Precompute window scores ({agg})")):
            q = rec["query"]
            q_emb = encode_text([q], max_length=64, no_grad=True).cpu()
            sims = (q_emb @ chunk_emb.t())[0]  # [Nc]

            if top_chunk is not None and top_chunk < sims.numel():
                top_idx = torch.topk(sims, k=top_chunk).indices.tolist()
            else:
                top_idx = torch.argsort(sims, descending=True).tolist()

            bucket = defaultdict(list)
            for i in top_idx:
                cid = chunk_to_contract_list[i]
                bucket[cid].append(float(sims[i]))

            for cid, arr in bucket.items():
                if agg == "max":
                    s = max(arr)
                elif agg == "mean":
                    s = sum(arr) / len(arr)
                elif agg == "topk_mean":
                    k = max(1, int(WINDOW_TOPK_MEAN_K))
                    arr_sorted = sorted(arr, reverse=True)
                    topk = arr_sorted[: min(k, len(arr_sorted))]
                    s = sum(topk) / len(topk)
                elif agg == "lse":
                    tau = float(WINDOW_LSE_TAU)
                    x = np.array(arr, dtype=np.float32) / max(tau, 1e-6)
                    m = float(x.max())
                    s = (m + math.log(float(np.exp(x - m).sum()))) * tau
                else:
                    raise ValueError("agg must be 'max'/'mean'/'topk_mean'/'lse'")
                score_matrix[qi, cid_to_col[cid]] = float(s)

    qids = [int(r["sample_id"]) for r in records]
    return qids, score_matrix, contract_ids


def precompute_window_scores_with_local_embeddings(records, chunk_emb_tensor, chunk_to_contract_list, top_chunk, agg):
    """
    预计算 window 分数 + query-conditioned 局部向量（每个 query/contract 取最高分 chunk 向量）：
    - qids: [Q]
    - score_matrix: [Q, C]
    - contract_ids: [C]
    - local_embs: [Q, C, D]，D 为 embedding 维度
    - query_embs: [Q, D]
    """
    contract_ids = sorted(set(chunk_to_contract_list))
    cid_to_col = {cid: i for i, cid in enumerate(contract_ids)}
    num_q, num_c = len(records), len(contract_ids)
    emb_dim = int(chunk_emb_tensor.shape[1])

    score_matrix = np.full((num_q, num_c), -1e9, dtype=np.float32)
    local_embs = np.zeros((num_q, num_c, emb_dim), dtype=np.float32)
    query_embs = np.zeros((num_q, emb_dim), dtype=np.float32)

    chunk_emb = chunk_emb_tensor
    if chunk_emb.device != device:
        chunk_emb = chunk_emb.cpu()

    encoder.eval()
    with torch.no_grad():
        for qi, rec in enumerate(tqdm(records, desc=f"Precompute window+local ({agg})")):
            q = rec["query"]
            q_emb = encode_text([q], max_length=64, no_grad=True).cpu()
            query_embs[qi] = q_emb[0].numpy().astype(np.float32)
            sims = (q_emb @ chunk_emb.t())[0]  # [Nc]

            if top_chunk is not None and top_chunk < sims.numel():
                top_idx = torch.topk(sims, k=top_chunk).indices.tolist()
            else:
                top_idx = torch.argsort(sims, descending=True).tolist()

            bucket_scores = defaultdict(list)
            bucket_best = {}
            for i in top_idx:
                cid = chunk_to_contract_list[i]
                s = float(sims[i])
                bucket_scores[cid].append(s)
                # 局部向量使用该 query 下该 contract 的最高分窗口向量
                if (cid not in bucket_best) or (s > bucket_best[cid][0]):
                    bucket_best[cid] = (s, int(i))

            for cid, arr in bucket_scores.items():
                if agg == "max":
                    s = max(arr)
                elif agg == "mean":
                    s = sum(arr) / len(arr)
                elif agg == "topk_mean":
                    k = max(1, int(WINDOW_TOPK_MEAN_K))
                    arr_sorted = sorted(arr, reverse=True)
                    topk = arr_sorted[: min(k, len(arr_sorted))]
                    s = sum(topk) / len(topk)
                elif agg == "lse":
                    tau = float(WINDOW_LSE_TAU)
                    x = np.array(arr, dtype=np.float32) / max(tau, 1e-6)
                    m = float(x.max())
                    s = (m + math.log(float(np.exp(x - m).sum()))) * tau
                else:
                    raise ValueError("agg must be 'max'/'mean'/'topk_mean'/'lse'")

                cj = cid_to_col[cid]
                score_matrix[qi, cj] = float(s)
                best_chunk_idx = bucket_best[cid][1]
                local_embs[qi, cj, :] = chunk_emb[best_chunk_idx].numpy().astype(np.float32)

    qids = [int(r["sample_id"]) for r in records]
    return qids, score_matrix, contract_ids, local_embs, query_embs


def precompute_trunc_contract_scores(records, code_embeddings_tensor, code_id_list):
    """
    预计算每条 query 对每个合约的 trunc 相似度分数，返回：
    - qids: 与 records 对齐的 sample_id 列表
    - score_matrix: [num_queries, num_contracts]
    - contract_ids: 列索引对应的 contract id 顺序
    """
    contract_ids = list(code_id_list)
    code_emb = code_embeddings_tensor
    enc_device = next(encoder.parameters()).device
    if code_emb.device != enc_device:
        code_emb = code_emb.to(enc_device)

    score_rows = []
    qids = []
    encoder.eval()
    with torch.no_grad():
        for rec in tqdm(records, desc="Precompute trunc scores"):
            q = rec["query"]
            qids.append(int(rec["sample_id"]))
            q_emb = encode_text([q], max_length=64, no_grad=True)
            sims = (q_emb @ code_emb.t())[0].detach().cpu().numpy().astype(np.float32)
            score_rows.append(sims)
    return qids, np.vstack(score_rows), contract_ids


def export_full_query_ranking(score_matrix, contract_ids, qids, out_csv, score_col="score", rank_col="rank"):
    """
    导出完整候选排名（每条 query 对所有 contract 的排序）到 CSV。
    query_idx 使用 score_matrix 的行号（与内部分析保持一致）。
    """
    rows = []
    for qi in range(score_matrix.shape[0]):
        row = score_matrix[qi]
        order = np.argsort(row)[::-1]
        for rk, cj in enumerate(order, start=1):
            rows.append(
                {
                    "query_idx": int(qi),
                    "query_sample_id": int(qids[qi]),
                    "candidate_contract_id": int(contract_ids[cj]),
                    rank_col: int(rk),
                    score_col: float(row[cj]),
                }
            )
    df_rank = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df_rank.to_csv(out_csv, index=False, encoding="utf-8")
    return df_rank


def evaluate_from_score_matrix(qids, score_matrix, contract_ids, k_list=(1, 5, 10)):
    """
    从预计算分数矩阵直接计算 Hit@k / MRR。
    """
    hits = {k: 0 for k in k_list}
    mrr_sum = 0.0
    num = len(qids)
    for i, target_id in enumerate(qids):
        row = score_matrix[i]
        rank = np.argsort(row)[::-1]
        ranked_ids = [contract_ids[j] for j in rank.tolist()]
        for kk in k_list:
            if target_id in ranked_ids[:kk]:
                hits[kk] += 1
        for r, cid in enumerate(ranked_ids):
            if cid == target_id:
                mrr_sum += 1.0 / (r + 1)
                break
    return {k: hits[k] / num for k in k_list}, mrr_sum / num


def evaluate_from_score_matrix_with_details(qids, score_matrix, contract_ids, k_list=(1, 5, 10)):
    """
    返回整体指标 + 每条 query 的 rank/rr（用于显著性检验与案例分析）。
    """
    hits = {k: 0 for k in k_list}
    mrr_sum = 0.0
    num = len(qids)
    cid_to_col = {cid: j for j, cid in enumerate(contract_ids)}
    rr = np.zeros(num, dtype=np.float32)
    rank_pos = np.zeros(num, dtype=np.int32)

    for i, target_id in enumerate(qids):
        row = score_matrix[i]
        rank = np.argsort(row)[::-1]
        ranked_ids = [contract_ids[j] for j in rank.tolist()]
        for kk in k_list:
            if target_id in ranked_ids[:kk]:
                hits[kk] += 1
        # 1-based rank
        tgt_col = cid_to_col.get(target_id, None)
        if tgt_col is None:
            rank_pos[i] = len(contract_ids) + 1
            rr[i] = 0.0
            continue
        pos = int(np.where(rank == tgt_col)[0][0]) + 1
        rank_pos[i] = pos
        rr[i] = 1.0 / float(pos)
        mrr_sum += rr[i]
    return {k: hits[k] / num for k in k_list}, mrr_sum / num, rr, rank_pos


def paired_bootstrap_test(a_rr: np.ndarray, b_rr: np.ndarray, rounds: int = 2000):
    """
    对 query 级 RR 做 paired bootstrap：
    返回 delta_mean, p_value, ci_low, ci_high。
    """
    a = np.asarray(a_rr, dtype=np.float32)
    b = np.asarray(b_rr, dtype=np.float32)
    n = min(len(a), len(b))
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    a = a[:n]
    b = b[:n]
    diff = a - b
    obs = float(diff.mean())
    if rounds <= 0:
        return obs, np.nan, np.nan, np.nan

    rng = np.random.default_rng(42)
    boot = np.empty(rounds, dtype=np.float32)
    for i in range(rounds):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(diff[idx].mean())
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5]).tolist()
    # 两侧 p 值：以 0 为零假设中心
    p_left = float((boot <= 0.0).mean())
    p_right = float((boot >= 0.0).mean())
    p_val = 2.0 * min(p_left, p_right)
    p_val = min(max(p_val, 0.0), 1.0)
    return obs, p_val, float(ci_low), float(ci_high)


def paired_rank_biserial_from_diff(diff: np.ndarray) -> float:
    """
    配对场景下的 rank-biserial（简化为符号占比版本）：
      r = (n_pos - n_neg) / (n_pos + n_neg)
    取值范围 [-1, 1]，越接近 1 表示前者更优。
    """
    d = np.asarray(diff, dtype=np.float32)
    n_pos = float((d > 0).sum())
    n_neg = float((d < 0).sum())
    den = n_pos + n_neg
    if den <= 0:
        return 0.0
    return float((n_pos - n_neg) / den)


def wilcoxon_signed_rank_pvalue(diff: np.ndarray) -> float:
    """
    计算 Wilcoxon signed-rank 双侧 p 值。
    若环境缺少 scipy 或样本无效，返回 np.nan。
    """
    d = np.asarray(diff, dtype=np.float32)
    d = d[np.isfinite(d)]
    # 全 0 时检验无意义
    if d.size == 0 or np.allclose(d, 0.0):
        return np.nan
    try:
        from scipy.stats import wilcoxon  # type: ignore
        return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided", correction=False).pvalue)
    except Exception:
        return np.nan


def _bucketize_by_token_length(sample_ids):
    """
    对给定 sample_id 按 code token 长度分桶（T1/T2 分位点）：
      short / medium / long
    返回 sid->bucket, sid->token_len, thresholds
    """
    lens = []
    sid_len = {}
    for sid in sample_ids:
        code = str(df.loc[sid, "code"])
        tok_len = len(tokenizer.encode(code, add_special_tokens=False))
        sid_len[int(sid)] = int(tok_len)
        lens.append(tok_len)
    if not lens:
        return {}, {}, (0, 0)
    q1, q2 = np.quantile(np.array(lens, dtype=np.float32), [0.33, 0.67]).tolist()
    q1 = int(round(q1))
    q2 = int(round(q2))
    sid_bucket = {}
    for sid, l in sid_len.items():
        if l <= q1:
            sid_bucket[sid] = "short"
        elif l <= q2:
            sid_bucket[sid] = "medium"
        else:
            sid_bucket[sid] = "long"
    return sid_bucket, sid_len, (q1, q2)
def search_best_fusion_alpha(val_qids, trunc_scores, window_scores, contract_ids, alpha_grid):
    """
    在验证集上网格搜索 alpha：
      fused = alpha * trunc + (1-alpha) * window
    以 MRR 为目标选最优 alpha。
    """
    best_alpha, best_mrr, best_hit = None, -1.0, None
    for alpha in alpha_grid:
        fused = alpha * trunc_scores + (1.0 - alpha) * window_scores
        hit, mrr = evaluate_from_score_matrix(val_qids, fused, contract_ids)
        if PRINT_ALPHA_TRACE:
            print(f"[AlphaTrace] alpha={alpha:.2f} val_mrr={mrr:.6f} val_hit1={hit.get(1, 0.0):.4f}")
        if mrr > best_mrr:
            best_alpha, best_mrr, best_hit = alpha, mrr, hit
    return best_alpha, best_hit, best_mrr


def search_best_fusion_alpha_temp_softmax(val_qids, trunc_scores, window_scores, contract_ids, alpha_grid, temp):
    """
    在验证集上：对 trunc / window 分数矩阵分别做按 query 的 temperature-softmax（与 DAC 方程 (5) 前一致），
    再线性融合 s = α·norm_trunc + (1-α)·norm_window，以 MRR 选最优 α（无门控，用于「仅温度归一化 + 固定权重」消融）。
    """
    norm_t = _temperature_softmax_rowwise(trunc_scores, temp=temp)
    norm_w = _temperature_softmax_rowwise(window_scores, temp=temp)
    best_alpha, best_mrr, best_hit = None, -1.0, None
    for alpha in alpha_grid:
        fused = alpha * norm_t + (1.0 - alpha) * norm_w
        hit, mrr = evaluate_from_score_matrix(val_qids, fused, contract_ids)
        if PRINT_ALPHA_TRACE:
            print(f"[AlphaTrace-TempNorm] alpha={alpha:.2f} val_mrr={mrr:.6f} val_hit1={hit.get(1, 0.0):.4f}")
        if mrr > best_mrr:
            best_alpha, best_mrr, best_hit = alpha, mrr, hit
    return best_alpha, best_hit, best_mrr


def fuse_temp_norm_fixed_alpha(trunc_scores: np.ndarray, window_scores: np.ndarray, alpha: float, temp: float):
    """温度归一化（按 query softmax）后固定 α 融合；用于 dac_temp_fixed_* 推理。"""
    norm_t = _temperature_softmax_rowwise(trunc_scores, temp=temp)
    norm_w = _temperature_softmax_rowwise(window_scores, temp=temp)
    a = float(alpha)
    return (a * norm_t + (1.0 - a) * norm_w).astype(np.float32)


# def rowwise_zscore(x: np.ndarray) -> np.ndarray:
#     """按 query（行）做 z-score，缓解 trunc/window 分数尺度不一致。"""
#     mu = x.mean(axis=1, keepdims=True)
#     sd = x.std(axis=1, keepdims=True)
#     return (x - mu) / np.maximum(sd, 1e-6)
def rowwise_minmax(x: np.ndarray) -> np.ndarray:
    """按 query（行）做 Min-Max 归一化，将分数严格映射到 [0, 1] 区间，消除量纲差异。"""
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    # 加上 1e-6 防止除以 0
    return (x - x_min) / (x_max - x_min + 1e-6)


class DACGate(nn.Module):
    """
    DAC-Gate:
      输入 [v_g, v_l, |v_g-v_l|]，输出 gate 权重 g in [0,1]。
    """
    def __init__(self, embed_dim=768, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, v_g, v_l):
        v_diff = torch.abs(v_g - v_l)
        x = torch.cat([v_g, v_l, v_diff], dim=-1)
        return self.net(x)


def _temperature_softmax_rowwise(x: np.ndarray, temp: float = 0.05) -> np.ndarray:
    t = max(float(temp), 1e-6)
    z = x / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def _build_dac_training_samples(
    val_qids,
    contract_ids,
    val_q_embs,
    val_trunc_scores,
    val_window_scores,
    val_local_embs,
    contract_emb_matrix,
    neg_per_query=32,
):
    """
    验证集构建 Oracle-guided 训练样本：
      - 正样本: 真实匹配 contract
      - 负样本: hard negatives
      - Golden gate: 1 if sim_g > sim_l else 0
    """
    cid_to_idx = {cid: i for i, cid in enumerate(contract_ids)}
    t_norm = rowwise_minmax(val_trunc_scores)
    w_norm = rowwise_minmax(val_window_scores)

    vg_list, vl_list, gt_list = [], [], []
    for qi, qid in enumerate(val_qids):
        if qid not in cid_to_idx:
            continue
        pos = cid_to_idx[qid]
        rank = np.argsort(0.5 * t_norm[qi] + 0.5 * w_norm[qi])[::-1]
        negs = [j for j in rank.tolist() if j != pos][: max(1, int(neg_per_query))]
        idxs = [pos] + negs

        q_vec = val_q_embs[qi]
        for cj in idxs:
            v_g = contract_emb_matrix[cj]
            v_l = val_local_embs[qi, cj]
            sim_g = float(np.dot(q_vec, v_g))
            sim_l = float(np.dot(q_vec, v_l))
            golden = 1.0 if sim_g > sim_l else 0.0
            vg_list.append(v_g.astype(np.float32))
            vl_list.append(v_l.astype(np.float32))
            gt_list.append([golden])

    if not vg_list:
        raise RuntimeError("DAC gate training data is empty.")
    return (
        np.stack(vg_list).astype(np.float32),
        np.stack(vl_list).astype(np.float32),
        np.array(gt_list, dtype=np.float32),
    )


def train_dac_gate(
    val_qids,
    contract_ids,
    val_q_embs,
    val_trunc_scores,
    val_window_scores,
    val_local_embs,
    contract_emb_matrix,
):
    v_g, v_l, golden = _build_dac_training_samples(
        val_qids,
        contract_ids,
        val_q_embs,
        val_trunc_scores,
        val_window_scores,
        val_local_embs,
        contract_emb_matrix,
        neg_per_query=DAC_NEG_PER_QUERY,
    )
    gate = DACGate(
        embed_dim=v_g.shape[1],
        hidden_dim=DAC_HIDDEN_DIM,
        dropout=DAC_DROPOUT,
    ).to(device)
    print(
        f"[DAC-Gate] training samples={v_g.shape[0]}, embed_dim={v_g.shape[1]}, "
        f"epochs={DAC_EPOCHS}, lr={DAC_LR}, neg_per_query={DAC_NEG_PER_QUERY}"
    )
    optimizer = torch.optim.Adam(gate.parameters(), lr=DAC_LR)
    criterion = nn.MSELoss()

    vg_t = torch.from_numpy(v_g).to(device)
    vl_t = torch.from_numpy(v_l).to(device)
    target_t = torch.from_numpy(golden).to(device)

    gate.train()
    for ep in range(DAC_EPOCHS):
        pred = gate(vg_t, vl_t)
        loss = criterion(pred, target_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (ep + 1) % max(1, DAC_EPOCHS // 5) == 0:
            print(f"[DAC-Gate] epoch={ep+1}/{DAC_EPOCHS}, loss={float(loss.item()):.6f}")
    gate.eval()
    return gate


def apply_dac_fusion(
    gate,
    trunc_scores: np.ndarray,
    window_scores: np.ndarray,
    local_embs: np.ndarray,
    contract_emb_matrix: np.ndarray,
    temp: float = 0.05,
    return_g: bool = False,
    fusion_mode: str = "full",
):
    """
    DAC 推理变体：
      - full：门控 g + 对 trunc/window 做按 query 的 temperature-softmax 后融合（论文默认）
      - gate_raw：门控 g + 直接在原始 cosine 分数上融合 g*trunc + (1-g)*window（无温度归一化消融）
    fusion_mode="temp_fixed" 时请改用 fuse_temp_norm_fixed_alpha（无需 gate 前向）。
    """
    fusion_mode = (fusion_mode or "full").strip().lower()
    if fusion_mode not in ("full", "gate_raw"):
        raise ValueError(f"apply_dac_fusion fusion_mode must be 'full' or 'gate_raw', got {fusion_mode}")
    if gate is None:
        raise ValueError("apply_dac_fusion requires gate for fusion_mode full/gate_raw")

    qn, cn = trunc_scores.shape
    emb_dim = contract_emb_matrix.shape[1]
    v_g = np.broadcast_to(contract_emb_matrix[None, :, :], (qn, cn, emb_dim)).reshape(-1, emb_dim).astype(np.float32)
    v_l = local_embs.reshape(-1, emb_dim).astype(np.float32)

    vg_t = torch.from_numpy(v_g).to(device)
    vl_t = torch.from_numpy(v_l).to(device)
    with torch.no_grad():
        g = gate(vg_t, vl_t).detach().cpu().numpy().reshape(qn, cn).astype(np.float32)

    if fusion_mode == "full":
        norm_t = _temperature_softmax_rowwise(trunc_scores, temp=temp)
        norm_w = _temperature_softmax_rowwise(window_scores, temp=temp)
        fused = g * norm_t + (1.0 - g) * norm_w
    else:
        fused = g * trunc_scores + (1.0 - g) * window_scores
    gate_mean_g = float(g.mean())
    if return_g:
        return fused.astype(np.float32), g.astype(np.float32), gate_mean_g
    return fused.astype(np.float32), gate_mean_g


def rrf_fuse_rankings(score_a: np.ndarray, score_b: np.ndarray, rrf_k: int = 60) -> np.ndarray:
    """
    Reciprocal Rank Fusion (RRF):
      fused(d) = 1/(k + rank_a(d)) + 1/(k + rank_b(d))
    """
    ra = np.argsort(score_a)[::-1]
    rb = np.argsort(score_b)[::-1]
    rank_pos_a = np.empty_like(ra)
    rank_pos_b = np.empty_like(rb)
    rank_pos_a[ra] = np.arange(1, len(ra) + 1)
    rank_pos_b[rb] = np.arange(1, len(rb) + 1)
    k = float(rrf_k)
    return 1.0 / (k + rank_pos_a.astype(np.float32)) + 1.0 / (k + rank_pos_b.astype(np.float32))


def evaluate_rrf_from_score_matrices(qids, trunc_scores, window_scores, contract_ids, rrf_k=60, k_list=(1, 5, 10)):
    fused = np.zeros_like(trunc_scores, dtype=np.float32)
    for i in range(fused.shape[0]):
        fused[i] = rrf_fuse_rankings(trunc_scores[i], window_scores[i], rrf_k=rrf_k)
    return evaluate_from_score_matrix(qids, fused, contract_ids, k_list=k_list)


def simple_tokenize(text: str):
    """
    轻量 tokenizer（BM25 baseline 用）：仅保留代码/文本里的标识符形态 token。
    """
    if not isinstance(text, str):
        return []
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


class BM25OkapiLite:
    """
    纯 Python 版 BM25（无额外依赖），用于快速稀疏检索基线。
    """
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N > 0 else 0.0
        self.df = defaultdict(int)
        self.tfs = []
        for doc in corpus_tokens:
            tf = defaultdict(int)
            for t in doc:
                tf[t] += 1
            self.tfs.append(tf)
            for t in tf.keys():
                self.df[t] += 1
        self.idf = {}
        for t, dfi in self.df.items():
            self.idf[t] = math.log(1 + (self.N - dfi + 0.5) / (dfi + 0.5))

    def get_scores(self, query_tokens):
        scores = np.zeros(self.N, dtype=np.float32)
        if self.N == 0:
            return scores
        for i, tf in enumerate(self.tfs):
            dl = self.doc_len[i]
            norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-12))
            s = 0.0
            for t in query_tokens:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (f + norm)
            scores[i] = s
        return scores


def evaluate_bm25_contract(test_recs, unique_test_ids, k_list=(1, 5, 10)):
    """
    BM25 合约级检索 baseline：query 文本 vs 测试集合约代码文本。
    """
    t0 = time.time()
    id_to_code = {sid: df.loc[sid, "code"] for sid in unique_test_ids}
    doc_ids = sorted(unique_test_ids)
    corpus = [simple_tokenize(id_to_code[sid]) for sid in doc_ids]
    bm25 = BM25OkapiLite(corpus)
    build_time = time.time() - t0

    hits = {k: 0 for k in k_list}
    mrr_sum = 0.0
    num = 0

    search_time_total = 0.0
    for rec in tqdm(test_recs, desc="Evaluate (bm25)"):
        target_id = rec["sample_id"]
        q_tokens = simple_tokenize(rec["query"])
        t0 = time.time()
        scores = bm25.get_scores(q_tokens)
        rank = np.argsort(scores)[::-1]
        ranked_ids = [doc_ids[i] for i in rank.tolist()]
        search_time_total += time.time() - t0
        num += 1
        for kk in k_list:
            if target_id in ranked_ids[:kk]:
                hits[kk] += 1
        for r, cid in enumerate(ranked_ids):
            if cid == target_id:
                mrr_sum += 1.0 / (r + 1)
                break
    # 记录效率（BM25 无编码阶段，build_time 计为 encode_time）
    global efficiency_tracker
    efficiency_tracker.record(
        'BM25',
        encode_time=build_time,
        search_time=search_time_total,
        num_queries=num,
        num_candidates=len(doc_ids)
    )
    print(f"[Efficiency] BM25: build={build_time:.2f}s, search={search_time_total:.2f}s, "
          f"total={build_time+search_time_total:.2f}s ({num} queries)")

    return {k: hits[k] / num for k in k_list}, mrr_sum / num


def evaluate_graphcodebert_contract(test_recs, unique_test_ids, k_list=(1, 5, 10)):
    """
    GraphCodeBERT 零微调 baseline（合约级）：
      - 用 graphcodebert-base 对 query / code 分别编码（CLS）
      - 余弦相似度检索，计算 Hit@k / MRR
    """
    print(f"\n========== GraphCodeBERT baseline: {graph_model_name} ==========")
    t0 = time.time()
    graph_tokenizer = AutoTokenizer.from_pretrained(graph_model_name)
    graph_encoder = AutoModel.from_pretrained(graph_model_name).to(device)
    graph_encoder.eval()
    load_time = time.time() - t0

    if graph_tokenizer.pad_token is None:
        graph_tokenizer.pad_token = graph_tokenizer.eos_token

    doc_ids = sorted(unique_test_ids)
    id_to_code = {sid: df.loc[sid, "code"] for sid in doc_ids}

    # 编码测试合约
    encode_time_total = 0.0
    code_vecs = []
    with torch.no_grad():
        for sid in tqdm(doc_ids, desc="GraphCodeBERT encode contracts"):
            code = id_to_code[sid]
            inputs = graph_tokenizer(
                [code],
                truncation=True,
                padding="max_length",
                max_length=graph_eval_trunc_max_length,
                return_tensors="pt",
            ).to(device)
            t0 = time.time()
            emb = graph_encoder(**inputs).last_hidden_state[:, 0, :]  # [1, D]
            emb = F.normalize(emb, dim=-1)
            encode_time_total += time.time() - t0
            code_vecs.append(emb.cpu())
    code_mat = torch.cat(code_vecs, dim=0)  # [N, D]

    # 检索计时
    hits = {k: 0 for k in k_list}
    mrr_sum = 0.0
    num = 0
    search_time_total = 0.0
    with torch.no_grad():
        for rec in tqdm(test_recs, desc="Evaluate (graphcodebert)"):
            q = rec["query"]
            target_id = rec["sample_id"]
            q_inputs = graph_tokenizer(
                [q],
                truncation=True,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            ).to(device)

            t0 = time.time()
            q_emb = graph_encoder(**q_inputs).last_hidden_state[:, 0, :]
            q_emb = F.normalize(q_emb, dim=-1).cpu()

            sims = (q_emb @ code_mat.t())[0]  # [N]
            rank = torch.argsort(sims, descending=True).tolist()
            search_time_total += time.time() - t0
            
            ranked_ids = [doc_ids[i] for i in rank]
            num += 1

            for kk in k_list:
                if target_id in ranked_ids[:kk]:
                    hits[kk] += 1
            for r, cid in enumerate(ranked_ids):
                if cid == target_id:
                    mrr_sum += 1.0 / (r + 1)
                    break

    # 记录效率
    global efficiency_tracker
    efficiency_tracker.record(
        'GraphCodeBERT',
        encode_time=encode_time_total,
        search_time=search_time_total,
        num_queries=num,
        num_candidates=len(doc_ids)
    )
    print(f"[Efficiency] GraphCodeBERT: load={load_time:.1f}s, encode={encode_time_total:.2f}s, "
          f"search={search_time_total:.2f}s, total={encode_time_total+search_time_total:.2f}s")


    # 释放显存，避免影响后续流程
    del graph_encoder
    cleanup_gpu()

    return {k: hits[k] / num for k in k_list}, mrr_sum / num


def evaluate_unixcoder_contract(test_recs, unique_test_ids, k_list=(1, 5, 10)):
    """
    UniXCoder 零微调 baseline（合约级）：
      - 用 unixcoder-base 对 query / code 分别编码（CLS）
      - 余弦相似度检索，计算 Hit@k / MRR
    """
    print(f"\n========== UniXCoder baseline: {unixcoder_model_name} ==========")
    t0 = time.time()
    ux_tokenizer = AutoTokenizer.from_pretrained(unixcoder_model_name)
    ux_encoder = AutoModel.from_pretrained(unixcoder_model_name).to(device)
    ux_encoder.eval()
    load_time = time.time() - t0

    if ux_tokenizer.pad_token is None:
        ux_tokenizer.pad_token = ux_tokenizer.eos_token

    doc_ids = sorted(unique_test_ids)
    id_to_code = {sid: df.loc[sid, "code"] for sid in doc_ids}

    encode_time_total = 0.0
    code_vecs = []
    with torch.no_grad():
        for sid in tqdm(doc_ids, desc="UniXCoder encode contracts"):
            code = id_to_code[sid]
            inputs = ux_tokenizer(
                [code],
                truncation=True,
                padding="max_length",
                max_length=unix_eval_trunc_max_length,
                return_tensors="pt",
            ).to(device)
            t0 = time.time()
            emb = ux_encoder(**inputs).last_hidden_state[:, 0, :]
            emb = F.normalize(emb, dim=-1)
            encode_time_total += time.time() - t0
            code_vecs.append(emb.cpu())
    code_mat = torch.cat(code_vecs, dim=0)

    hits = {k: 0 for k in k_list}
    mrr_sum = 0.0
    num = 0
    search_time_total = 0.0
    with torch.no_grad():
        for rec in tqdm(test_recs, desc="Evaluate (unixcoder)"):
            q = rec["query"]
            target_id = rec["sample_id"]
            q_inputs = ux_tokenizer(
                [q],
                truncation=True,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            ).to(device)

            t0 = time.time()
            q_emb = ux_encoder(**q_inputs).last_hidden_state[:, 0, :]
            q_emb = F.normalize(q_emb, dim=-1).cpu()

            sims = (q_emb @ code_mat.t())[0]
            rank = torch.argsort(sims, descending=True).tolist()
            search_time_total += time.time() - t0

            ranked_ids = [doc_ids[i] for i in rank]
            num += 1
            for kk in k_list:
                if target_id in ranked_ids[:kk]:
                    hits[kk] += 1
            for r, cid in enumerate(ranked_ids):
                if cid == target_id:
                    mrr_sum += 1.0 / (r + 1)
                    break
    global efficiency_tracker
    efficiency_tracker.record(
        'UniXCoder',
        encode_time=encode_time_total,
        search_time=search_time_total,
        num_queries=num,
        num_candidates=len(doc_ids)
    )
    print(f"[Efficiency] UniXCoder: load={load_time:.1f}s, encode={encode_time_total:.2f}s, "
          f"search={search_time_total:.2f}s, total={encode_time_total+search_time_total:.2f}s")


    del ux_encoder
    cleanup_gpu()
    return {k: hits[k] / num for k in k_list}, mrr_sum / num


def evaluate_sentence_transformer_baseline(test_recs, unique_test_ids, k_list=(1, 5, 10)):
    """
    Sentence-Transformers 句向量模型合约级检索基线（与 GraphCodeBERT / UniXCoder 并列的可选稠密对照）。
    需 pip install sentence-transformers；模型名见 ST_MODEL_NAME。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("【ST baseline】未安装 sentence-transformers，跳过。pip install sentence-transformers")
        return {k: np.nan for k in k_list}, np.nan

    print(f"\n========== Sentence-Transformer baseline: {ST_MODEL_NAME} ==========")
    st_model = SentenceTransformer(ST_MODEL_NAME, device=str(device))
    doc_ids = sorted(unique_test_ids)
    global df
    id_to_code = {}
    for sid in doc_ids:
        c = df.loc[sid, "code"]
        if ST_MAX_CODE_CHARS and len(c) > ST_MAX_CODE_CHARS:
            c = c[:ST_MAX_CODE_CHARS]
        id_to_code[sid] = c

    texts = [id_to_code[sid] for sid in doc_ids]
    code_emb = st_model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    code_mat = torch.from_numpy(np.asarray(code_emb, dtype=np.float32))

    hits = {k: 0 for k in k_list}
    mrr_sum = 0.0
    num = 0
    with torch.no_grad():
        for rec in tqdm(test_recs, desc="Evaluate (sentence-transformers)"):
            q = rec["query"]
            target_id = rec["sample_id"]
            q_emb = st_model.encode(
                [q],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            qv = torch.from_numpy(np.asarray(q_emb, dtype=np.float32))
            sims = (qv @ code_mat.t())[0]
            rank = torch.argsort(sims, descending=True).tolist()
            ranked_ids = [doc_ids[i] for i in rank]
            num += 1
            for kk in k_list:
                if target_id in ranked_ids[:kk]:
                    hits[kk] += 1
            for r, cid in enumerate(ranked_ids):
                if cid == target_id:
                    mrr_sum += 1.0 / (r + 1)
                    break

    del st_model
    cleanup_gpu()
    return {k: hits[k] / num for k in k_list}, mrr_sum / num


def _parse_llm_rerank_choice(text: str, topk: int) -> int:
    """从模型输出中解析 0～topk 的整数；0 表示「都不相关」；失败返回 -1。"""
    for m in re.finditer(r"-?\d+", text):
        v = int(m.group(0))
        if 0 <= v <= topk:
            return v
    return -1


def evaluate_llm_rerank_bm25_baseline(test_recs, unique_test_ids, k_list=(1, 5, 10)):
    """
    开源小 LLM 重排序基线（适合 Kaggle，无 API 费用）：
      1) BM25 对全测试库打分，取 Top-K 合约；
      2) 将 K 段截断源码编号写入单条指令，让模型**只输出一个整数**（1..K 最相关，0 为均不相关）；
      3) 将选中候选置于首位，其余 Top-K 按 BM25 原序接上，再拼接 K 之后的 BM25 序，得到最终排序并算 MRR/Hit@k。

    说明：若金标合约未进入 BM25@K，本基线无法召回，属「BM25 召回 + LLM 精排」管道的固有限制。
    """
    from transformers import AutoModelForCausalLM

    global df
    print(
        f"\n========== LLM rerank baseline: BM25@{min(LLM_RERANK_TOPK, len(unique_test_ids))} + "
        f"{LLM_RERANK_MODEL_NAME} =========="
    )
    id_to_code = {sid: df.loc[sid, "code"] for sid in unique_test_ids}
    doc_ids = sorted(unique_test_ids)
    corpus = [simple_tokenize(id_to_code[sid]) for sid in doc_ids]
    bm25 = BM25OkapiLite(corpus)

    hf_tok = AutoTokenizer.from_pretrained(LLM_RERANK_MODEL_NAME, trust_remote_code=True)
    if hf_tok.pad_token_id is None:
        hf_tok.pad_token_id = hf_tok.eos_token_id
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_RERANK_MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    llm = llm.to(device)
    llm.eval()

    topk = min(LLM_RERANK_TOPK, len(doc_ids))
    hits = {k: 0 for k in k_list}
    mrr_sum = 0.0
    num = 0

    for rec in tqdm(test_recs, desc="Evaluate (llm_rerank@bm25)"):
        target_id = rec["sample_id"]
        q_tokens = simple_tokenize(rec["query"])
        scores = bm25.get_scores(q_tokens)
        rank_full = np.argsort(scores)[::-1].tolist()
        ranked_full = [doc_ids[i] for i in rank_full]
        topk_ids = ranked_full[:topk]
        rest_ids = ranked_full[topk:]

        lines = [
            "You are a smart contract security assistant.",
            "Given the QUERY and numbered Solidity CODE excerpts, reply with ONLY ONE integer:",
            f"from 1 to {topk} for the index of the excerpt most relevant to the query, or 0 if none fit.",
            "",
            "QUERY:",
            rec["query"].strip(),
            "",
        ]
        for j, sid in enumerate(topk_ids, start=1):
            code = id_to_code[sid]
            if len(code) > LLM_RERANK_CODE_CHARS:
                code = code[: LLM_RERANK_CODE_CHARS] + "\n// ... [truncated]"
            lines.append(f"{j}. ```solidity\n{code}\n```")
        lines.append("")
        lines.append("Answer (one integer only):")
        user_content = "\n".join(lines)

        if getattr(hf_tok, "chat_template", None):
            messages = [{"role": "user", "content": user_content}]
            prompt = hf_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = hf_tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=LLM_RERANK_PROMPT_MAX_LENGTH,
            ).to(device)
        else:
            inputs = hf_tok(
                user_content,
                return_tensors="pt",
                truncation=True,
                max_length=LLM_RERANK_PROMPT_MAX_LENGTH,
            ).to(device)

        with torch.inference_mode():
            out = llm.generate(
                **inputs,
                max_new_tokens=LLM_RERANK_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=getattr(hf_tok, "eos_token_id", None) or hf_tok.pad_token_id,
            )
        gen = hf_tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pick = _parse_llm_rerank_choice(gen, topk)
        if 1 <= pick <= topk:
            chosen = topk_ids[pick - 1]
            new_top = [chosen] + [x for i, x in enumerate(topk_ids) if i != pick - 1]
        else:
            new_top = list(topk_ids)

        final_rank = new_top + rest_ids
        num += 1
        for kk in k_list:
            if target_id in final_rank[:kk]:
                hits[kk] += 1
        for r, cid in enumerate(final_rank):
            if cid == target_id:
                mrr_sum += 1.0 / (r + 1)
                break

    del llm
    cleanup_gpu()
    return {k: hits[k] / num for k in k_list}, mrr_sum / num


def _metric_slug_vuln(vuln: str) -> str:
    """用于结果列名的稳定后缀（避免空格与特殊字符）。"""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(vuln).strip()).strip("_").lower()
    return s or "unknown"


def evaluate_by_vuln_group(test_recs, eval_fn, labels_map):
    """
    按漏洞类型分组评测；类型集合由测试集 labels_map 中出现的 vuln_type 决定。
    eval_fn: 接收一组 records，返回 (hit_at_k, mrr)。
    labels_map: sample_id -> vuln_type。
    """
    labels_in_test = sorted(
        {_canonical_vuln_label(labels_map.get(int(r["sample_id"]), "")) for r in test_recs} - {""}
    )
    groups = {lab: [] for lab in labels_in_test}
    for r in test_recs:
        lab = _canonical_vuln_label(labels_map.get(int(r["sample_id"]), ""))
        if lab in groups:
            groups[lab].append(r)

    out = {}
    for lab, recs in groups.items():
        slug = _metric_slug_vuln(lab)
        if len(recs) == 0:
            out[f"{slug}_mrr"] = np.nan
            out[f"{slug}_hit1"] = np.nan
            out[f"{slug}_hit5"] = np.nan
            out[f"{slug}_hit10"] = np.nan
            continue
        h, m = eval_fn(recs)
        out[f"{slug}_mrr"] = m
        out[f"{slug}_hit1"] = h.get(1, np.nan)
        out[f"{slug}_hit5"] = h.get(5, np.nan)
        out[f"{slug}_hit10"] = h.get(10, np.nan)
    return out


def build_test_trunc_embeddings(unique_test_ids):
    """测试集每个合约一条截断向量（baseline）。"""
    global df
    code_embeddings_list = []
    code_id_list = []
    encoder.eval()
    with torch.no_grad():
        for sid in tqdm(sorted(unique_test_ids), desc="Encode test contracts (trunc)"):
            code = df.loc[sid, "code"]
            emb = encode_text([code], max_length=eval_trunc_max_length, no_grad=True).cpu()
            code_embeddings_list.append(emb)
            code_id_list.append(sid)
    mat = torch.cat(code_embeddings_list, dim=0)
    mat = F.normalize(mat, dim=-1)
    return mat, code_id_list


def build_chunk_index_for_contract_ids(unique_ids, desc="Build chunk windows"):
    """
    给定一组合约 sample_id，构建滑窗 chunk 索引（验证集 / 测试集共用逻辑）。
    """
    global df
    chunk_id_windows = []
    chunk_to_contract = []
    for cid in tqdm(sorted(unique_ids), desc=desc):
        code = df.loc[cid, "code"]
        wins = build_id_windows(code, tokenizer, window_max_length, window_stride)
        for w in wins:
            chunk_id_windows.append(w)
            chunk_to_contract.append(cid)
    chunk_emb = encode_id_windows_in_batches(
        chunk_id_windows, max_length=window_max_length, batch_size=32
    )
    return chunk_emb, chunk_to_contract


def build_test_chunk_index(unique_test_ids):
    """测试集：滑窗 + 批量编码 chunk（内部转调通用函数）。"""
    return build_chunk_index_for_contract_ids(unique_test_ids, desc="Build test windows")


# ==============================================================================
# 验证：保存/加载最佳 encoder（CPU 上存 state_dict，省显存）
# ==============================================================================
def _encoder_state_dict_cpu():
    """把当前 encoder 权重拷到 CPU，用于 best checkpoint。"""
    return {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}


def _load_encoder_state_dict_cpu(state_cpu):
    """从 CPU 上的 state_dict 恢复到当前 device 上的 encoder。"""
    encoder.load_state_dict({k: v.to(device) for k, v in state_cpu.items()})


def validate_window_mrr(val_recs, unique_val_ids, top_chunk=None, show_progress=False):
    """
    在验证集上计算 window 检索 MRR（与测试口径一致：chunk 聚合）。
    每个 epoch 调用会重算 chunk 嵌入（因 encoder 已更新），略慢但正确。
    """
    if top_chunk is None:
        top_chunk = window_eval_top_chunk
    if VAL_EVAL_TOP_CHUNK is not None:
        top_chunk = VAL_EVAL_TOP_CHUNK

    chunk_emb, chunk_map = build_chunk_index_for_contract_ids(
        unique_val_ids, desc="Val: build windows"
    )
    _, mrr = evaluate_window(
        val_recs,
        chunk_emb,
        chunk_map,
        top_chunk=top_chunk,
        agg=window_agg,
        show_progress=show_progress,
    )
    return float(mrr)


# ==============================================================================
# 运行入口：单种子 / 多种子
# ==============================================================================
def run_baselines_only_seed(seed_value: int):
    """
    只加载数据并评测已打开的基线（不训练、不跑 CodeBERT 截断/滑窗/融合）。
    典型用法：RUN_BASELINES_ONLY=1，按需打开 BM25 / Graph / UniX / RUN_ST_BASELINE / RUN_LLM_RERANK_BASELINE 等。
    """
    global tokenizer, encoder, df, train_records, val_records, test_records, train_loader
    tokenizer, encoder = None, None
    train_loader = None

    set_seed(seed_value)
    print(f"\n================ Seed = {seed_value}（仅基线评测）================\n")
    print("Device:", device)
    print("DATA_SOURCE:", DATA_SOURCE)
    print("RUN_BASELINES_ONLY=1：跳过训练与主模型")

    any_baseline = (
        RUN_BM25_BASELINE
        or RUN_GRAPHCODEBERT_BASELINE
        or RUN_UNIXCODER_BASELINE
        or RUN_ST_BASELINE
        or RUN_LLM_RERANK_BASELINE
    )
    if not any_baseline:
        raise SystemExit(
            "RUN_BASELINES_ONLY=1 时至少需要开启一个基线（例如 RUN_BM25_BASELINE=1、"
            "RUN_GRAPHCODEBERT_BASELINE、RUN_UNIXCODER_BASELINE、RUN_ST_BASELINE 或 RUN_LLM_RERANK_BASELINE）。"
        )

    if DATA_SOURCE == "external_jsonl":
        if not EXTERNAL_JSONL_DIR:
            raise SystemExit("DATA_SOURCE=external_jsonl 但未设置 VULN_IR_EXTERNAL_JSONL_DIR")
        train_records = load_jsonl(os.path.join(EXTERNAL_JSONL_DIR, "train.jsonl"))
        val_records = load_jsonl(os.path.join(EXTERNAL_JSONL_DIR, "val.jsonl"))
        test_records = load_jsonl(os.path.join(EXTERNAL_JSONL_DIR, "test.jsonl"))
        df = dataframe_from_query_splits(train_records, val_records, test_records)
        print("【外部 JSONL】有效合约数:", len(df))
    else:
        df = load_dataframe(CSV_PATH, CODE_BASE_DIR)
        print("有效样本数:", len(df))
        query_records = build_query_records(df, max_queries_per_sample=3)
        train_records, val_records, test_records, _, _, _ = split_by_contract(query_records)

    test_ids = sorted({r["sample_id"] for r in test_records})
    print("test query 条数:", len(test_records), "test 合约数:", len(test_ids))

    hit_bm25, mrr_bm25 = {1: np.nan, 5: np.nan, 10: np.nan}, np.nan
    if RUN_BM25_BASELINE:
        hit_bm25, mrr_bm25 = evaluate_bm25_contract(test_records, test_ids)
        print("【BM25】Hit@k:", hit_bm25, "MRR:", mrr_bm25)

    hit_graph, mrr_graph = {1: np.nan, 5: np.nan, 10: np.nan}, np.nan
    if RUN_GRAPHCODEBERT_BASELINE:
        hit_graph, mrr_graph = evaluate_graphcodebert_contract(test_records, test_ids)
        print("【GraphCodeBERT】Hit@k:", hit_graph, "MRR:", mrr_graph)

    hit_unix, mrr_unix = {1: np.nan, 5: np.nan, 10: np.nan}, np.nan
    if RUN_UNIXCODER_BASELINE:
        hit_unix, mrr_unix = evaluate_unixcoder_contract(test_records, test_ids)
        print("【UniXCoder】Hit@k:", hit_unix, "MRR:", mrr_unix)

    hit_st, mrr_st = {1: np.nan, 5: np.nan, 10: np.nan}, np.nan
    if RUN_ST_BASELINE:
        hit_st, mrr_st = evaluate_sentence_transformer_baseline(test_records, test_ids)
        print("【Sentence-Transformer】Hit@k:", hit_st, "MRR:", mrr_st)

    hit_llm_rerank, mrr_llm_rerank = {1: np.nan, 5: np.nan, 10: np.nan}, np.nan
    if RUN_LLM_RERANK_BASELINE:
        hit_llm_rerank, mrr_llm_rerank = evaluate_llm_rerank_bm25_baseline(test_records, test_ids)
        print("【LLM rerank @BM25】Hit@k:", hit_llm_rerank, "MRR:", mrr_llm_rerank)

    result = {
        "seed": seed_value,
        "trunc_hit1": np.nan,
        "trunc_hit5": np.nan,
        "trunc_hit10": np.nan,
        "trunc_mrr": np.nan,
        "window_hit1": np.nan,
        "window_hit5": np.nan,
        "window_hit10": np.nan,
        "window_mrr": np.nan,
        "window_primary_agg": window_agg,
        "window_alt_agg": "",
        "window_alt_hit1": np.nan,
        "window_alt_hit5": np.nan,
        "window_alt_hit10": np.nan,
        "window_alt_mrr": np.nan,
        "bm25_hit1": hit_bm25.get(1, np.nan),
        "bm25_hit5": hit_bm25.get(5, np.nan),
        "bm25_hit10": hit_bm25.get(10, np.nan),
        "bm25_mrr": mrr_bm25,
        "graph_hit1": hit_graph.get(1, np.nan),
        "graph_hit5": hit_graph.get(5, np.nan),
        "graph_hit10": hit_graph.get(10, np.nan),
        "graph_mrr": mrr_graph,
        "unix_hit1": hit_unix.get(1, np.nan),
        "unix_hit5": hit_unix.get(5, np.nan),
        "unix_hit10": hit_unix.get(10, np.nan),
        "unix_mrr": mrr_unix,
        "st_hit1": hit_st.get(1, np.nan),
        "st_hit5": hit_st.get(5, np.nan),
        "st_hit10": hit_st.get(10, np.nan),
        "st_mrr": mrr_st,
        "llm_rerank_hit1": hit_llm_rerank.get(1, np.nan),
        "llm_rerank_hit5": hit_llm_rerank.get(5, np.nan),
        "llm_rerank_hit10": hit_llm_rerank.get(10, np.nan),
        "llm_rerank_mrr": mrr_llm_rerank,
        "fusion_alpha": np.nan,
        "fusion_hit1": np.nan,
        "fusion_hit5": np.nan,
        "fusion_hit10": np.nan,
        "fusion_mrr": np.nan,
        "fusion_norm_alpha": np.nan,
        "fusion_norm_hit1": np.nan,
        "fusion_norm_hit5": np.nan,
        "fusion_norm_hit10": np.nan,
        "fusion_norm_mrr": np.nan,
        "rrf_k": np.nan,
        "rrf_hit1": np.nan,
        "rrf_hit5": np.nan,
        "rrf_hit10": np.nan,
        "rrf_mrr": np.nan,
        "gate_mean_g": np.nan,
        "gate_hit1": np.nan,
        "gate_hit5": np.nan,
        "gate_hit10": np.nan,
        "gate_mrr": np.nan,
        "dac_gate_raw_mean_g": np.nan,
        "dac_gate_raw_hit1": np.nan,
        "dac_gate_raw_hit5": np.nan,
        "dac_gate_raw_hit10": np.nan,
        "dac_gate_raw_mrr": np.nan,
        "dac_temp_fixed_alpha": np.nan,
        "dac_temp_fixed_hit1": np.nan,
        "dac_temp_fixed_hit5": np.nan,
        "dac_temp_fixed_hit10": np.nan,
        "dac_temp_fixed_mrr": np.nan,
        "dac_trunc_train_mean_g": np.nan,
        "dac_trunc_train_hit1": np.nan,
        "dac_trunc_train_hit5": np.nan,
        "dac_trunc_train_hit10": np.nan,
        "dac_trunc_train_mrr": np.nan,
        "p_fusion_vs_trunc": np.nan,
        "p_fusion_vs_window": np.nan,
        "p_dac_vs_fusion": np.nan,
        "stage1_best_val_mrr": np.nan,
    }
    return result


def run_one_seed(seed_value: int):
    """
    运行一次完整流程并返回核心指标，便于后续做多种子统计。
    """
    global tokenizer, encoder, df, train_records, val_records, test_records, train_loader

    if RUN_BASELINES_ONLY:
        return run_baselines_only_seed(seed_value)

    set_seed(seed_value)
    print(f"\n================ Seed = {seed_value} ================\n")
    print("Device:", device)
    print("DATA_SOURCE:", DATA_SOURCE)
    print("TRAIN_POS_MODE:", TRAIN_POS_MODE)
    print("CSV:", CSV_PATH)
    print("Code dir:", CODE_BASE_DIR)

    # ---------- 数据 ----------
    if DATA_SOURCE == "external_jsonl":
        if not EXTERNAL_JSONL_DIR:
            raise SystemExit("DATA_SOURCE=external_jsonl 但未设置 VULN_IR_EXTERNAL_JSONL_DIR")
        train_records = load_jsonl(os.path.join(EXTERNAL_JSONL_DIR, "train.jsonl"))
        val_records = load_jsonl(os.path.join(EXTERNAL_JSONL_DIR, "val.jsonl"))
        test_records = load_jsonl(os.path.join(EXTERNAL_JSONL_DIR, "test.jsonl"))
        df = dataframe_from_query_splits(train_records, val_records, test_records)
        print("【外部 JSONL】有效合约数:", len(df))
        print(df["vuln_type"].value_counts())
        print("train/val/test query 条数:", len(train_records), len(val_records), len(test_records))
    else:
        df = load_dataframe(CSV_PATH, CODE_BASE_DIR)
        print("有效样本数:", len(df))
        print(df["vuln_type"].value_counts())

        query_records = build_query_records(df, max_queries_per_sample=3)
        print("查询集总条数:", len(query_records))

        train_records, val_records, test_records, _, _, _ = split_by_contract(query_records)
        print("train/val/test query 条数:", len(train_records), len(val_records), len(test_records))

    # 每个种子单独导出，防止覆盖
    seed_out_dir = os.path.join(OUTPUT_DIR, f"seed_{seed_value}")
    os.makedirs(seed_out_dir, exist_ok=True)
    export_jsonl(train_records, os.path.join(seed_out_dir, "train.jsonl"))
    export_jsonl(val_records, os.path.join(seed_out_dir, "val.jsonl"))
    export_jsonl(test_records, os.path.join(seed_out_dir, "test.jsonl"))
    print("已导出划分到:", seed_out_dir)

    # ---------- 模型 ----------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(
        VulnIRDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    unique_val_ids = sorted({r["sample_id"] for r in val_records})

    # ---------- Stage1 ----------
    print("\n========== Stage1: 窗口一致性对比学习 ==========")
    print(
        f"（验证：每 epoch 后计算 val window MRR；最佳权重会保留；EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE}）"
    )
    stage1_best_mrr, stage1_state_cpu = run_stage1_training(val_records, unique_val_ids)

    # ---------- Stage2（可选）----------
    if RUN_STAGE2_HARD:
        print("\n========== Stage2: 合约级 Hard Negative ==========")
        train_contract_ids, train_contract_emb = build_train_contract_index()
        train_records_hard = mine_hard_negatives(train_records, train_contract_ids, train_contract_emb)
        run_stage2_training(
            train_records_hard,
            val_records=val_records,
            unique_val_ids=unique_val_ids,
            stage1_best_mrr=stage1_best_mrr,
            stage1_state_cpu=stage1_state_cpu,
        )

    # ---------- 测试 ----------
    print("\n========== 构建测试集索引并评测 ==========")
    test_ids = sorted({r["sample_id"] for r in test_records})

    code_embeddings, code_id_list = build_test_trunc_embeddings(test_ids)
    hit_at_k, mrr = evaluate_instance(test_records, code_embeddings, code_id_list)
    print("【整段截断检索】Hit@k:", hit_at_k)
    print("【整段截断检索】MRR:", mrr)

    chunk_embeddings, chunk_to_contract = build_test_chunk_index(test_ids)
    hit_w, mrr_w = evaluate_window(
        test_records,
        chunk_embeddings,
        chunk_to_contract,
        top_chunk=window_eval_top_chunk,
        agg=window_agg,
    )
    print(f"【滑窗聚合检索】Window({window_agg}) Hit@k:", hit_w)
    print(f"【滑窗聚合检索】Window({window_agg}) MRR:", mrr_w)

    # 可选：同一套权重下，额外评估另一种聚合策略（不需要重训）
    alt_agg = None
    hit_w_alt, mrr_w_alt = None, None
    if EVAL_BOTH_AGG:
        alt_agg = "mean" if window_agg == "max" else "max"
        hit_w_alt, mrr_w_alt = evaluate_window(
            test_records,
            chunk_embeddings,
            chunk_to_contract,
            top_chunk=window_eval_top_chunk,
            agg=alt_agg,
        )
        print(f"【滑窗聚合检索】Window({alt_agg}) Hit@k:", hit_w_alt)
        print(f"【滑窗聚合检索】Window({alt_agg}) MRR:", mrr_w_alt)

    hit_bm25, mrr_bm25 = {1: np.nan, 5: np.nan, 10: np.nan}, np.nan
    if RUN_BM25_BASELINE:
        hit_bm25, mrr_bm25 = evaluate_bm25_contract(test_records, test_ids)
        print("【BM25 合约级基线】Hit@k:", hit_bm25)
        print("【BM25 合约级基线】MRR:", mrr_bm25)

    # ---------- GraphCodeBERT baseline ----------
    hit_graph, mrr_graph = {1: np.nan, 5: np.nan, 10: np.nan}, np.nan
    if RUN_GRAPHCODEBERT_BASELINE:
        # 释放 CodeBERT encoder 的 GPU 显存，避免和 baseline 模型叠加导致 Kaggle OOM
        encoder_on_cuda = next(encoder.parameters()).is_cuda
        if encoder_on_cuda:
            encoder.to("cpu")
            cleanup_gpu()
        hit_graph, mrr_graph = evaluate_graphcodebert_contract(test_records, test_ids)
        print("【GraphCodeBERT 合约级基线】Hit@k:", hit_graph)
        print("【GraphCodeBERT 合约级基线】MRR:", mrr_graph)
        if encoder_on_cuda:
            encoder.to(device)
            cleanup_gpu()

    # ---------- UniXCoder baseline ----------
    hit_unix, mrr_unix = {1: np.nan, 5: np.nan, 10: np.nan}, np.nan
    if RUN_UNIXCODER_BASELINE:
        # 释放 CodeBERT encoder 的 GPU 显存，避免和 baseline 模型叠加导致 Kaggle OOM
        encoder_on_cuda = next(encoder.parameters()).is_cuda
        if encoder_on_cuda:
            encoder.to("cpu")
            cleanup_gpu()
        hit_unix, mrr_unix = evaluate_unixcoder_contract(test_records, test_ids)
        print("【UniXCoder 合约级基线】Hit@k:", hit_unix)
        print("【UniXCoder 合约级基线】MRR:", mrr_unix)
        if encoder_on_cuda:
            encoder.to(device)
            cleanup_gpu()

    # ---------- Sentence-Transformer 句向量基线（可选）----------
    hit_st, mrr_st = {1: np.nan, 5: np.nan, 10: np.nan}, np.nan
    if RUN_ST_BASELINE:
        encoder_on_cuda = next(encoder.parameters()).is_cuda
        if encoder_on_cuda:
            encoder.to("cpu")
            cleanup_gpu()
        hit_st, mrr_st = evaluate_sentence_transformer_baseline(test_records, test_ids)
        print("【Sentence-Transformer 合约级基线】Hit@k:", hit_st)
        print("【Sentence-Transformer 合约级基线】MRR:", mrr_st)
        if encoder_on_cuda:
            encoder.to(device)
            cleanup_gpu()

    # ---------- 开源小 LLM + BM25@K 重排序基线（可选）----------
    hit_llm_rerank, mrr_llm_rerank = {1: np.nan, 5: np.nan, 10: np.nan}, np.nan
    if RUN_LLM_RERANK_BASELINE:
        encoder_on_cuda = next(encoder.parameters()).is_cuda
        if encoder_on_cuda:
            encoder.to("cpu")
            cleanup_gpu()
        hit_llm_rerank, mrr_llm_rerank = evaluate_llm_rerank_bm25_baseline(test_records, test_ids)
        print("【LLM rerank@BM25】Hit@k:", hit_llm_rerank, "MRR:", mrr_llm_rerank)
        if encoder_on_cuda:
            encoder.to(device)
            cleanup_gpu()

    # ---------- trunc + window 融合评测 ----------
    fusion_alpha = np.nan
    fusion_hit, fusion_mrr = None, np.nan
    fusion_norm_alpha = np.nan
    fusion_norm_hit, fusion_norm_mrr = None, np.nan
    rrf_hit, rrf_mrr = None, np.nan
    rrf_k_used = np.nan
    gate_hit, gate_mrr = None, np.nan
    gate_mean_g = np.nan
    dac_gate_raw_hit, dac_gate_raw_mrr = None, np.nan
    dac_gate_raw_mean_g = np.nan
    dac_temp_fixed_alpha = np.nan
    dac_temp_fixed_hit, dac_temp_fixed_mrr = None, np.nan
    dac_trunc_train_hit1 = dac_trunc_train_hit5 = dac_trunc_train_hit10 = np.nan
    dac_trunc_train_mrr = np.nan
    dac_trunc_train_mean_g = np.nan
    trunc_rr = window_rr = fusion_rr = gate_rr = None
    trunc_rank = window_rank = fusion_rank = gate_rank = None
    p_fusion_vs_trunc = np.nan
    p_fusion_vs_window = np.nan
    p_dac_vs_fusion = np.nan
    if ENABLE_FUSION:
        print("========== 融合评测：验证集搜索最优 alpha ==========")
        # 用验证集涉及的合约建索引
        val_ids = sorted({r["sample_id"] for r in val_records})
        val_code_emb, val_code_id_list = build_test_trunc_embeddings(val_ids)
        val_chunk_emb, val_chunk_map = build_chunk_index_for_contract_ids(val_ids, desc="Build val windows (fusion)")

        # 预计算 val 的 trunc/window 分数矩阵
        vq1, v_trunc, cids_t = precompute_trunc_contract_scores(val_records, val_code_emb, val_code_id_list)
        vq2, v_window, cids_w, v_local_embs, v_q_embs = precompute_window_scores_with_local_embeddings(
            val_records, val_chunk_emb, val_chunk_map, top_chunk=window_eval_top_chunk, agg=window_agg
        )
        if cids_t != cids_w:
            raise RuntimeError("Fusion error: val contract id order mismatch between trunc and window.")
        if vq1 != vq2:
            raise RuntimeError("Fusion error: val query order mismatch between trunc and window.")

        fusion_alpha, val_fusion_hit, val_fusion_mrr = search_best_fusion_alpha(
            vq1, v_trunc, v_window, cids_t, FUSION_ALPHA_GRID
        )
        print(f"最优 alpha={fusion_alpha:.2f}，验证集融合 Hit@k={val_fusion_hit}，验证集融合 MRR={val_fusion_mrr:.6f}")

        # 在测试集上应用最优 alpha
        tq1, t_trunc, tcids_t = precompute_trunc_contract_scores(test_records, code_embeddings, code_id_list)
        tq2, t_window, tcids_w, t_local_embs, _ = precompute_window_scores_with_local_embeddings(
            test_records, chunk_embeddings, chunk_to_contract, top_chunk=window_eval_top_chunk, agg=window_agg
        )
        if tcids_t != tcids_w:
            raise RuntimeError("Fusion error: test contract id order mismatch between trunc and window.")
        if tq1 != tq2:
            raise RuntimeError("Fusion error: test query order mismatch between trunc and window.")

        # 导出测试集完整 trunc 排名（便于后续给 Top-k 候选回填 rank_trunc）
        trunc_rank_csv = os.path.join(seed_out_dir, "test_trunc_full_ranking.csv")
        trunc_rank_df = export_full_query_ranking(
            score_matrix=t_trunc,
            contract_ids=tcids_t,
            qids=tq1,
            out_csv=trunc_rank_csv,
            score_col="trunc_score",
            rank_col="rank_trunc",
        )
        print(f"已导出完整 trunc 排名到: {trunc_rank_csv}")

        _, _, trunc_rr, trunc_rank = evaluate_from_score_matrix_with_details(tq1, t_trunc, tcids_t)
        _, _, window_rr, window_rank = evaluate_from_score_matrix_with_details(tq1, t_window, tcids_t)

        t_fused = fusion_alpha * t_trunc + (1.0 - fusion_alpha) * t_window
        fusion_hit, fusion_mrr, fusion_rr, fusion_rank = evaluate_from_score_matrix_with_details(tq1, t_fused, tcids_t)
        print(f"【融合检索】Fusion(alpha={fusion_alpha:.2f}) Hit@k:", fusion_hit)
        print(f"【融合检索】Fusion(alpha={fusion_alpha:.2f}) MRR:", fusion_mrr)

        # if ENABLE_NORM_FUSION:
        #     # 分数先按 query 归一化，再搜索 alpha
        #     v_trunc_z = rowwise_zscore(v_trunc)
        #     v_window_z = rowwise_zscore(v_window)
        #     fusion_norm_alpha, val_norm_hit, val_norm_mrr = search_best_fusion_alpha(
        #         vq1, v_trunc_z, v_window_z, cids_t, FUSION_ALPHA_GRID
        #     )
        #     print(f"最优 alpha(norm)={fusion_norm_alpha:.2f}，验证集融合(norm) Hit@k={val_norm_hit}，MRR={val_norm_mrr:.6f}")

        #     t_trunc_z = rowwise_zscore(t_trunc)
        #     t_window_z = rowwise_zscore(t_window)
        #     t_fused_norm = fusion_norm_alpha * t_trunc_z + (1.0 - fusion_norm_alpha) * t_window_z
        #     fusion_norm_hit, fusion_norm_mrr = evaluate_from_score_matrix(tq1, t_fused_norm, tcids_t)
        #     print(f"【融合检索】Fusion-Norm(alpha={fusion_norm_alpha:.2f}) Hit@k:", fusion_norm_hit)
        #     print(f"【融合检索】Fusion-Norm(alpha={fusion_norm_alpha:.2f}) MRR:", fusion_norm_mrr)
        if ENABLE_NORM_FUSION:
            # 分数先按 query 归一化 (使用 Min-Max 代替 Z-Score)，再搜索 alpha
            v_trunc_norm = rowwise_minmax(v_trunc)
            v_window_norm = rowwise_minmax(v_window)
            fusion_norm_alpha, val_norm_hit, val_norm_mrr = search_best_fusion_alpha(
                vq1, v_trunc_norm, v_window_norm, cids_t, FUSION_ALPHA_GRID
            )
            print(f"最优 alpha(norm)={fusion_norm_alpha:.2f}，验证集融合(norm) Hit@k={val_norm_hit}，MRR={val_norm_mrr:.6f}")

            t_trunc_norm = rowwise_minmax(t_trunc)
            t_window_norm = rowwise_minmax(t_window)
            t_fused_norm = fusion_norm_alpha * t_trunc_norm + (1.0 - fusion_norm_alpha) * t_window_norm
            fusion_norm_hit, fusion_norm_mrr = evaluate_from_score_matrix(tq1, t_fused_norm, tcids_t)
            print(f"【融合检索】Fusion-Norm(alpha={fusion_norm_alpha:.2f}) Hit@k:", fusion_norm_hit)
            print(f"【融合检索】Fusion-Norm(alpha={fusion_norm_alpha:.2f}) MRR:", fusion_norm_mrr)

        if ENABLE_RRF_FUSION:
            rrf_k_used = float(RRF_K)
            rrf_hit, rrf_mrr = evaluate_rrf_from_score_matrices(tq1, t_trunc, t_window, tcids_t, rrf_k=RRF_K)
            print(f"【融合检索】RRF(k={RRF_K}) Hit@k:", rrf_hit)
            print(f"【融合检索】RRF(k={RRF_K}) MRR:", rrf_mrr)

        t_gate_g = None  # for optional case-study gate heatmap dumping
        if ENABLE_DAC_FUSION:
            print("========== 融合评测：DAC-Fusion 动态门控 ==========")
            contract_emb_np = val_code_emb.detach().cpu().numpy().astype(np.float32)
            gate_model = train_dac_gate(
                val_qids=vq1,
                contract_ids=cids_t,
                val_q_embs=v_q_embs,
                val_trunc_scores=v_trunc,
                val_window_scores=v_window,
                val_local_embs=v_local_embs,
                contract_emb_matrix=contract_emb_np,
            )
            t_contract_emb_np = code_embeddings.detach().cpu().numpy().astype(np.float32)
            if DUMP_CASE_GATE_HEATMAP:
                t_gate_fused, t_gate_g, gate_mean_g = apply_dac_fusion(
                    gate=gate_model,
                    trunc_scores=t_trunc,
                    window_scores=t_window,
                    local_embs=t_local_embs,
                    contract_emb_matrix=t_contract_emb_np,
                    temp=DAC_TEMP,
                    return_g=True,
                )
            else:
                t_gate_fused, gate_mean_g = apply_dac_fusion(
                    gate=gate_model,
                    trunc_scores=t_trunc,
                    window_scores=t_window,
                    local_embs=t_local_embs,
                    contract_emb_matrix=t_contract_emb_np,
                    temp=DAC_TEMP,
                    return_g=False,
                )
                t_gate_g = None
            gate_hit, gate_mrr, gate_rr, gate_rank = evaluate_from_score_matrix_with_details(tq1, t_gate_fused, tcids_t)
            print(f"【融合检索】DAC-Fusion(mean g={gate_mean_g:.4f}) Hit@k:", gate_hit)
            print(f"【融合检索】DAC-Fusion(mean g={gate_mean_g:.4f}) MRR:", gate_mrr)

            if EVAL_DAC_ABLATION_COMPONENTS:
                print("========== DAC 子模块消融：门控+原始分数 / 温度归一化+固定 α ==========")
                t_gate_raw_fused, dac_gate_raw_mean_g = apply_dac_fusion(
                    gate=gate_model,
                    trunc_scores=t_trunc,
                    window_scores=t_window,
                    local_embs=t_local_embs,
                    contract_emb_matrix=t_contract_emb_np,
                    temp=DAC_TEMP,
                    return_g=False,
                    fusion_mode="gate_raw",
                )
                dac_gate_raw_hit, dac_gate_raw_mrr, _, _ = evaluate_from_score_matrix_with_details(
                    tq1, t_gate_raw_fused, tcids_t
                )
                print(
                    f"【DAC 消融】Gate+RawScores(mean g={dac_gate_raw_mean_g:.4f}) Hit@k:",
                    dac_gate_raw_hit,
                    "MRR:",
                    dac_gate_raw_mrr,
                )

                dac_temp_fixed_alpha, _, _ = search_best_fusion_alpha_temp_softmax(
                    vq1, v_trunc, v_window, cids_t, FUSION_ALPHA_GRID, DAC_TEMP
                )
                t_temp_fixed = fuse_temp_norm_fixed_alpha(t_trunc, t_window, dac_temp_fixed_alpha, DAC_TEMP)
                dac_temp_fixed_hit, dac_temp_fixed_mrr, _, _ = evaluate_from_score_matrix_with_details(
                    tq1, t_temp_fixed, tcids_t
                )
                print(
                    f"【DAC 消融】TempNorm+FixedAlpha(α={dac_temp_fixed_alpha:.2f}) Hit@k:",
                    dac_temp_fixed_hit,
                    "MRR:",
                    dac_temp_fixed_mrr,
                )

        if ENABLE_ADV_ANALYSIS and trunc_rr is not None and window_rr is not None and fusion_rr is not None:
            print("========== 进阶分析：长度分桶 / 显著性 / 案例样本 ==========")
            # 1) 显著性检验（query-level paired bootstrap）
            sig_rows = []
            for name, rr in [
                ("fusion_vs_trunc", fusion_rr - trunc_rr),
                ("fusion_vs_window", fusion_rr - window_rr),
                ("dac_vs_trunc", (gate_rr - trunc_rr) if gate_rr is not None else None),
                ("dac_vs_window", (gate_rr - window_rr) if gate_rr is not None else None),
                ("dac_vs_fusion", (gate_rr - fusion_rr) if gate_rr is not None else None),
            ]:
                if rr is None:
                    continue
                # paired_bootstrap_test 需要两个序列，这里将 diff+0 的等价形式用于复用
                delta, p_val, ci_l, ci_h = paired_bootstrap_test(rr, np.zeros_like(rr), rounds=SIG_BOOTSTRAP_ROUNDS)
                p_wil = wilcoxon_signed_rank_pvalue(rr)
                eff_rbc = paired_rank_biserial_from_diff(rr)
                sig_rows.append(
                    {
                        "comparison": name,
                        "delta_mrr": float(delta),
                        "p_value": float(p_val),  # backward compatibility: bootstrap p
                        "p_value_bootstrap": float(p_val),
                        "p_value_wilcoxon": float(p_wil) if not np.isnan(p_wil) else np.nan,
                        "effect_size_rbc": float(eff_rbc),
                        "ci95_low": float(ci_l),
                        "ci95_high": float(ci_h),
                        "rounds": int(SIG_BOOTSTRAP_ROUNDS),
                    }
                )
            if sig_rows:
                sig_df = pd.DataFrame(sig_rows)
                sig_csv = os.path.join(seed_out_dir, "significance_tests.csv")
                sig_df.to_csv(sig_csv, index=False, encoding="utf-8")
                print(f"已导出显著性检验结果到: {sig_csv}")
                for row in sig_rows:
                    if row["comparison"] == "fusion_vs_trunc":
                        p_fusion_vs_trunc = float(row["p_value"])
                    elif row["comparison"] == "fusion_vs_window":
                        p_fusion_vs_window = float(row["p_value"])
                    elif row["comparison"] == "dac_vs_fusion":
                        p_dac_vs_fusion = float(row["p_value"])

            # 2) 长度分桶分析
            sid_bucket, sid_len, (q1, q2) = _bucketize_by_token_length(test_ids)
            bucket_rows = []
            methods = {
                "trunc": (trunc_rr, trunc_rank),
                "window": (window_rr, window_rank),
                "fusion": (fusion_rr, fusion_rank),
            }
            if gate_rr is not None and gate_rank is not None:
                methods["dac_fusion"] = (gate_rr, gate_rank)

            for bname in ("short", "medium", "long"):
                idx = [i for i, qid in enumerate(tq1) if sid_bucket.get(int(qid), "unknown") == bname]
                if not idx:
                    continue
                avg_len = float(np.mean([sid_len[int(tq1[i])] for i in idx]))
                for mname, (m_rr, m_rank) in methods.items():
                    rr_sub = np.asarray([m_rr[i] for i in idx], dtype=np.float32)
                    rk_sub = np.asarray([m_rank[i] for i in idx], dtype=np.int32)
                    bucket_rows.append(
                        {
                            "bucket": bname,
                            "method": mname,
                            "num_queries": int(len(idx)),
                            "avg_token_len": avg_len,
                            "mrr": float(rr_sub.mean()),
                            "hit10": float((rk_sub <= 10).mean()),
                            "q1_token_len": int(q1),
                            "q2_token_len": int(q2),
                        }
                    )
            if bucket_rows:
                bucket_df = pd.DataFrame(bucket_rows)
                bucket_csv = os.path.join(seed_out_dir, "length_bucket_metrics.csv")
                bucket_df.to_csv(bucket_csv, index=False, encoding="utf-8")
                print(f"已导出长度分桶结果到: {bucket_csv}")

            # 3) 案例样本候选（trunc 差、window 好）
            case_rows = []
            for i, qid in enumerate(tq1):
                tr = int(trunc_rank[i])
                wr = int(window_rank[i])
                if tr >= CASE_BAD_RANK and wr <= CASE_GOOD_RANK:
                    sid = int(qid)
                    code = str(df.loc[sid, "code"])
                    case_rows.append(
                        {
                            "query_idx": int(i),
                            "sample_id": sid,
                            "query": str(test_records[i]["query"]),
                            "bucket": sid_bucket.get(sid, "unknown"),
                            "token_len": int(sid_len.get(sid, 0)),
                            "trunc_rank": tr,
                            "window_rank": wr,
                            "fusion_rank": int(fusion_rank[i]) if fusion_rank is not None else np.nan,
                            "dac_rank": int(gate_rank[i]) if gate_rank is not None else np.nan,
                            "trunc_rr": float(trunc_rr[i]),
                            "window_rr": float(window_rr[i]),
                            "fusion_rr": float(fusion_rr[i]),
                            "dac_rr": float(gate_rr[i]) if gate_rr is not None else np.nan,
                            "code_preview": code[:600].replace("\n", " "),
                        }
                    )
            if case_rows:
                case_df = pd.DataFrame(case_rows)
                case_df["rank_gap_trunc_minus_window"] = case_df["trunc_rank"] - case_df["window_rank"]
                case_df = case_df.sort_values(
                    ["rank_gap_trunc_minus_window", "window_rank"], ascending=[False, True]
                ).head(max(1, CASE_TOPN))
                case_csv = os.path.join(seed_out_dir, "case_study_candidates.csv")
                case_df.to_csv(case_csv, index=False, encoding="utf-8")
                print(f"已导出案例候选到: {case_csv}")

                # Optional: dump gate weights for a heatmap case study.
                # 支持两种模式：
                #  - 默认：使用 case_df 里“trunc 差 / window 好”的最佳样本（best = case_df.iloc[0]）
                #  - 强制：通过环境变量 DUMP_CASE_QUERY_IDX 指定 query_idx（避免 case_df 为空导致不导出）
                dump_qi_env = os.environ.get("DUMP_CASE_QUERY_IDX", "").strip()
                if DUMP_CASE_GATE_HEATMAP and t_gate_g is not None and (dump_qi_env or not case_df.empty):
                    if dump_qi_env:
                        best_qi = int(dump_qi_env)
                    else:
                        best = case_df.iloc[0]
                        best_qi = int(best["query_idx"])
                    top_n = int(os.environ.get("DUMP_CASE_GATE_HEATMAP_TOPN", "20"))

                    cand_idx_sorted = np.argsort(t_gate_fused[best_qi])[::-1][:top_n]
                    heat_rows = []
                    for rk, cj in enumerate(cand_idx_sorted, start=1):
                        cid = int(tcids_t[cj])
                        code_cand = str(df.loc[cid, "code"])
                        cand_tok_len = int(len(tokenizer.encode(code_cand, add_special_tokens=False)))
                        if cand_tok_len <= q1:
                            cand_bucket = "short"
                        elif cand_tok_len <= q2:
                            cand_bucket = "medium"
                        else:
                            cand_bucket = "long"
                        heat_rows.append(
                            {
                                "seed": seed_value,
                                "query_idx": best_qi,
                                "query_sample_id": int(tq1[best_qi]),
                                "candidate_contract_id": cid,
                                "candidate_dac_rank": rk,
                                "candidate_token_len": cand_tok_len,
                                "candidate_bucket": cand_bucket,
                                "gate_weight_g": float(t_gate_g[best_qi, cj]),
                                "dac_score": float(t_gate_fused[best_qi, cj]),
                                "trunc_score": float(t_trunc[best_qi, cj]),
                                "window_score": float(t_window[best_qi, cj]),
                                "code_preview": code_cand[:400].replace("\n", " "),
                            }
                        )

                    heat_df = pd.DataFrame(heat_rows)
                    # 自动回填 rank_trunc（基于完整 trunc 排名）
                    if 'trunc_rank_df' in locals() and trunc_rank_df is not None and not trunc_rank_df.empty:
                        heat_df = heat_df.merge(
                            trunc_rank_df[["query_idx", "candidate_contract_id", "rank_trunc"]],
                            on=["query_idx", "candidate_contract_id"],
                            how="left",
                        )
                    heat_csv = os.path.join(seed_out_dir, "case_gate_heatmap_data.csv")
                    heat_df.to_csv(heat_csv, index=False, encoding="utf-8")
                    print(f"已导出门控权重热力图数据到: {heat_csv}")

    # ---------- 按漏洞类型分组评测 ----------
    labels_map = {int(i): df.loc[i, "vuln_type"] for i in test_ids}

    grp_trunc = evaluate_by_vuln_group(
        test_records,
        lambda recs: evaluate_instance(recs, code_embeddings, code_id_list),
        labels_map,
    )
    grp_window = evaluate_by_vuln_group(
        test_records,
        lambda recs: evaluate_window(
            recs,
            chunk_embeddings,
            chunk_to_contract,
            top_chunk=window_eval_top_chunk,
            agg=window_agg,
            show_progress=False,
        ),
        labels_map,
    )
    if RUN_BM25_BASELINE:
        grp_bm25 = evaluate_by_vuln_group(
            test_records,
            lambda recs: evaluate_bm25_contract(recs, test_ids),
            labels_map,
        )
    else:
        grp_bm25 = {}
    print("【分组-trunc】", grp_trunc)
    print("【分组-window】", grp_window)
    print("【分组-bm25】", grp_bm25 if RUN_BM25_BASELINE else "(已跳过 BM25)")

    # ---------- 可选：trunc_first Stage1 再训一遍，仅评测「基础 InfoNCE + 完整 DAC」（不覆盖主流程 window 权重）----------
    if (
        RUN_TRUNC_TRAIN_DAC_ABLATION
        and TRAIN_POS_MODE == "window"
        and ENABLE_FUSION
        and ENABLE_DAC_FUSION
    ):
        print("\n========== 消融：trunc_first Stage1 + DAC-Fusion（独立第二套 Stage1）==========")
        window_encoder_state = _encoder_state_dict_cpu()
        run_stage1_training(val_records, unique_val_ids, train_pos_mode="trunc_first")
        code_emb_ab, code_id_ab = build_test_trunc_embeddings(test_ids)
        chunk_emb_ab, chunk_map_ab = build_test_chunk_index(test_ids)
        val_ids_ab = sorted({r["sample_id"] for r in val_records})
        val_code_ab, val_id_ab = build_test_trunc_embeddings(val_ids_ab)
        val_chunk_ab, val_map_ab = build_chunk_index_for_contract_ids(
            val_ids_ab, desc="Val windows (trunc-train DAC ablation)"
        )
        vq_a, v_tr_a, cid_a = precompute_trunc_contract_scores(val_records, val_code_ab, val_id_ab)
        vq_b, v_win_a, cid_b, v_loc_a, v_qe_a = precompute_window_scores_with_local_embeddings(
            val_records, val_chunk_ab, val_map_ab, top_chunk=window_eval_top_chunk, agg=window_agg
        )
        if cid_a != cid_b or vq_a != vq_b:
            raise RuntimeError("trunc-train DAC ablation: val contract/query order mismatch.")
        gate_ab = train_dac_gate(
            val_qids=vq_a,
            contract_ids=cid_a,
            val_q_embs=v_qe_a,
            val_trunc_scores=v_tr_a,
            val_window_scores=v_win_a,
            val_local_embs=v_loc_a,
            contract_emb_matrix=val_code_ab.detach().cpu().numpy().astype(np.float32),
        )
        tq_a, t_tr_a, tcid_a = precompute_trunc_contract_scores(test_records, code_emb_ab, code_id_ab)
        tq_b, t_win_a, tcid_b, t_loc_a, _ = precompute_window_scores_with_local_embeddings(
            test_records, chunk_emb_ab, chunk_map_ab, top_chunk=window_eval_top_chunk, agg=window_agg
        )
        if tcid_a != tcid_b or tq_a != tq_b:
            raise RuntimeError("trunc-train DAC ablation: test order mismatch.")
        t_fused_ab, dac_trunc_train_mean_g = apply_dac_fusion(
            gate=gate_ab,
            trunc_scores=t_tr_a,
            window_scores=t_win_a,
            local_embs=t_loc_a,
            contract_emb_matrix=code_emb_ab.detach().cpu().numpy().astype(np.float32),
            temp=DAC_TEMP,
            return_g=False,
            fusion_mode="full",
        )
        dac_trunc_h, dac_trunc_train_mrr, _, _ = evaluate_from_score_matrix_with_details(tq_a, t_fused_ab, tcid_a)
        dac_trunc_train_hit1 = dac_trunc_h.get(1, 0.0)
        dac_trunc_train_hit5 = dac_trunc_h.get(5, 0.0)
        dac_trunc_train_hit10 = dac_trunc_h.get(10, 0.0)
        print(
            f"【trunc_first + DAC-Fusion】mean g={dac_trunc_train_mean_g:.4f} Hit@k:",
            dac_trunc_h,
            "MRR:",
            dac_trunc_train_mrr,
        )
        _load_encoder_state_dict_cpu(window_encoder_state)
        del code_emb_ab, chunk_emb_ab, val_code_ab, val_chunk_ab, gate_ab, t_fused_ab
        cleanup_gpu()

    result = {
        "seed": seed_value,
        "train_pos_mode": TRAIN_POS_MODE,
        "trunc_hit1": hit_at_k.get(1, 0.0),
        "trunc_hit5": hit_at_k.get(5, 0.0),
        "trunc_hit10": hit_at_k.get(10, 0.0),
        "trunc_mrr": mrr,
        "window_hit1": hit_w.get(1, 0.0),
        "window_hit5": hit_w.get(5, 0.0),
        "window_hit10": hit_w.get(10, 0.0),
        "window_mrr": mrr_w,
        "window_primary_agg": window_agg,
        "window_alt_agg": alt_agg if alt_agg is not None else "",
        "window_alt_hit1": hit_w_alt.get(1, 0.0) if hit_w_alt is not None else np.nan,
        "window_alt_hit5": hit_w_alt.get(5, 0.0) if hit_w_alt is not None else np.nan,
        "window_alt_hit10": hit_w_alt.get(10, 0.0) if hit_w_alt is not None else np.nan,
        "window_alt_mrr": mrr_w_alt if mrr_w_alt is not None else np.nan,
        "bm25_hit1": hit_bm25.get(1, 0.0),
        "bm25_hit5": hit_bm25.get(5, 0.0),
        "bm25_hit10": hit_bm25.get(10, 0.0),
        "bm25_mrr": mrr_bm25,
        "graph_hit1": hit_graph.get(1, np.nan),
        "graph_hit5": hit_graph.get(5, np.nan),
        "graph_hit10": hit_graph.get(10, np.nan),
        "graph_mrr": mrr_graph,
        "unix_hit1": hit_unix.get(1, np.nan),
        "unix_hit5": hit_unix.get(5, np.nan),
        "unix_hit10": hit_unix.get(10, np.nan),
        "unix_mrr": mrr_unix,
        "st_hit1": hit_st.get(1, np.nan),
        "st_hit5": hit_st.get(5, np.nan),
        "st_hit10": hit_st.get(10, np.nan),
        "st_mrr": mrr_st,
        "llm_rerank_hit1": hit_llm_rerank.get(1, np.nan),
        "llm_rerank_hit5": hit_llm_rerank.get(5, np.nan),
        "llm_rerank_hit10": hit_llm_rerank.get(10, np.nan),
        "llm_rerank_mrr": mrr_llm_rerank,
        "fusion_alpha": fusion_alpha,
        "fusion_hit1": fusion_hit.get(1, 0.0) if fusion_hit is not None else np.nan,
        "fusion_hit5": fusion_hit.get(5, 0.0) if fusion_hit is not None else np.nan,
        "fusion_hit10": fusion_hit.get(10, 0.0) if fusion_hit is not None else np.nan,
        "fusion_mrr": fusion_mrr,
        "fusion_norm_alpha": fusion_norm_alpha,
        "fusion_norm_hit1": fusion_norm_hit.get(1, 0.0) if fusion_norm_hit is not None else np.nan,
        "fusion_norm_hit5": fusion_norm_hit.get(5, 0.0) if fusion_norm_hit is not None else np.nan,
        "fusion_norm_hit10": fusion_norm_hit.get(10, 0.0) if fusion_norm_hit is not None else np.nan,
        "fusion_norm_mrr": fusion_norm_mrr,
        "rrf_k": rrf_k_used,
        "rrf_hit1": rrf_hit.get(1, 0.0) if rrf_hit is not None else np.nan,
        "rrf_hit5": rrf_hit.get(5, 0.0) if rrf_hit is not None else np.nan,
        "rrf_hit10": rrf_hit.get(10, 0.0) if rrf_hit is not None else np.nan,
        "rrf_mrr": rrf_mrr,
        "gate_mean_g": gate_mean_g,
        "gate_hit1": gate_hit.get(1, 0.0) if gate_hit is not None else np.nan,
        "gate_hit5": gate_hit.get(5, 0.0) if gate_hit is not None else np.nan,
        "gate_hit10": gate_hit.get(10, 0.0) if gate_hit is not None else np.nan,
        "gate_mrr": gate_mrr,
        "dac_gate_raw_mean_g": dac_gate_raw_mean_g,
        "dac_gate_raw_hit1": dac_gate_raw_hit.get(1, 0.0) if dac_gate_raw_hit is not None else np.nan,
        "dac_gate_raw_hit5": dac_gate_raw_hit.get(5, 0.0) if dac_gate_raw_hit is not None else np.nan,
        "dac_gate_raw_hit10": dac_gate_raw_hit.get(10, 0.0) if dac_gate_raw_hit is not None else np.nan,
        "dac_gate_raw_mrr": dac_gate_raw_mrr,
        "dac_temp_fixed_alpha": dac_temp_fixed_alpha,
        "dac_temp_fixed_hit1": dac_temp_fixed_hit.get(1, 0.0) if dac_temp_fixed_hit is not None else np.nan,
        "dac_temp_fixed_hit5": dac_temp_fixed_hit.get(5, 0.0) if dac_temp_fixed_hit is not None else np.nan,
        "dac_temp_fixed_hit10": dac_temp_fixed_hit.get(10, 0.0) if dac_temp_fixed_hit is not None else np.nan,
        "dac_temp_fixed_mrr": dac_temp_fixed_mrr,
        "dac_trunc_train_mean_g": dac_trunc_train_mean_g,
        "dac_trunc_train_hit1": dac_trunc_train_hit1,
        "dac_trunc_train_hit5": dac_trunc_train_hit5,
        "dac_trunc_train_hit10": dac_trunc_train_hit10,
        "dac_trunc_train_mrr": dac_trunc_train_mrr,
        "p_fusion_vs_trunc": p_fusion_vs_trunc,
        "p_fusion_vs_window": p_fusion_vs_window,
        "p_dac_vs_fusion": p_dac_vs_fusion,
        "stage1_best_val_mrr": stage1_best_mrr,
    }
    # 分组指标写入结果字典（前缀区分模型）
    for k, v in grp_trunc.items():
        result[f"trunc_{k}"] = v
    for k, v in grp_window.items():
        result[f"window_{k}"] = v
    for k, v in grp_bm25.items():
        result[f"bm25_{k}"] = v


    # ========== 效率分析汇总 ==========
    global efficiency_tracker
    if len(efficiency_tracker.records) > 0:
        efficiency_tracker.summary()
        efficiency_csv = os.path.join(seed_out_dir, "efficiency_analysis.csv")
        efficiency_tracker.to_csv(efficiency_csv)
        
        # 将核心效率指标也写入 result
        for name, rec in efficiency_tracker.records.items():
            slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            result[f"eff_{slug}_encode_s"] = rec['encode_time_s']
            result[f"eff_{slug}_search_s"] = rec['search_time_s']
            result[f"eff_{slug}_total_s"] = rec['total_time_s']
            result[f"eff_{slug}_memory_mb"] = rec['memory_mb']


    return result


def summarize_multi_seed(results):
    """
    打印多种子实验的均值/标准差，并可导出 csv。
    """
    if not results:
        return
    df_res = pd.DataFrame(results)
    print("\n========== Multi-seed Summary ==========")
    print(df_res)
    metrics = [
        "unix_hit1", "unix_hit5", "unix_hit10", "unix_mrr",
        "st_hit1", "st_hit5", "st_hit10", "st_mrr",
        "llm_rerank_hit1", "llm_rerank_hit5", "llm_rerank_hit10", "llm_rerank_mrr",
        "graph_hit1", "graph_hit5", "graph_hit10", "graph_mrr",
        "fusion_alpha", "fusion_hit1", "fusion_hit5", "fusion_hit10", "fusion_mrr",
        "fusion_norm_alpha", "fusion_norm_hit1", "fusion_norm_hit5", "fusion_norm_hit10", "fusion_norm_mrr",
        "rrf_k", "rrf_hit1", "rrf_hit5", "rrf_hit10", "rrf_mrr",
        "gate_mean_g", "gate_hit1", "gate_hit5", "gate_hit10", "gate_mrr",
        "dac_gate_raw_mean_g",
        "dac_gate_raw_hit1", "dac_gate_raw_hit5", "dac_gate_raw_hit10", "dac_gate_raw_mrr",
        "dac_temp_fixed_alpha",
        "dac_temp_fixed_hit1", "dac_temp_fixed_hit5", "dac_temp_fixed_hit10", "dac_temp_fixed_mrr",
        "dac_trunc_train_mean_g",
        "dac_trunc_train_hit1", "dac_trunc_train_hit5", "dac_trunc_train_hit10", "dac_trunc_train_mrr",
        "p_fusion_vs_trunc", "p_fusion_vs_window", "p_dac_vs_fusion",
        "bm25_hit1", "bm25_hit5", "bm25_hit10", "bm25_mrr",
        "trunc_hit1", "trunc_hit5", "trunc_hit10", "trunc_mrr",
        "window_hit1", "window_hit5", "window_hit10", "window_mrr",
        "window_alt_hit1", "window_alt_hit5", "window_alt_hit10", "window_alt_mrr",
        "stage1_best_val_mrr",
        # ===== 新增：效率分析指标 =====
        "eff_trunc_total_s", "eff_trunc_memory_mb",
        "eff_window_max_total_s", "eff_window_max_memory_mb",
        "eff_bm25_total_s", "eff_bm25_memory_mb",
        "eff_graphcodebert_total_s", "eff_graphcodebert_memory_mb",
        "eff_unixcoder_total_s", "eff_unixcoder_memory_mb",
    ]
    print("\n均值 ± 标准差：")
    for m in metrics:
        if m not in df_res.columns:
            continue
        mean_v = df_res[m].mean()
        std_v = df_res[m].std(ddof=0)
        print(f"{m}: {mean_v:.6f} ± {std_v:.6f}")

    if SAVE_MULTI_SEED_RESULTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_csv = os.path.join(OUTPUT_DIR, "multi_seed_results.csv")
        df_res.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n已导出多种子结果到: {out_csv}")

        # 额外导出一个模型对比表（均值±标准差）
        compare_rows = []
        compare_spec = {
            "unixcoder": ["unix_hit1", "unix_hit5", "unix_hit10", "unix_mrr"],
            "sentence_transformers": ["st_hit1", "st_hit5", "st_hit10", "st_mrr"],
            "llm_rerank_bm25": ["llm_rerank_hit1", "llm_rerank_hit5", "llm_rerank_hit10", "llm_rerank_mrr"],
            "graphcodebert": ["graph_hit1", "graph_hit5", "graph_hit10", "graph_mrr"],
            "fusion": ["fusion_hit1", "fusion_hit5", "fusion_hit10", "fusion_mrr"],
            "fusion_norm": ["fusion_norm_hit1", "fusion_norm_hit5", "fusion_norm_hit10", "fusion_norm_mrr"],
            "rrf": ["rrf_hit1", "rrf_hit5", "rrf_hit10", "rrf_mrr"],
            "dac_fusion": ["gate_hit1", "gate_hit5", "gate_hit10", "gate_mrr"],
            "dac_gate_raw": ["dac_gate_raw_hit1", "dac_gate_raw_hit5", "dac_gate_raw_hit10", "dac_gate_raw_mrr"],
            "dac_temp_fixed": ["dac_temp_fixed_hit1", "dac_temp_fixed_hit5", "dac_temp_fixed_hit10", "dac_temp_fixed_mrr"],
            "dac_trunc_train_plus_dac": [
                "dac_trunc_train_hit1",
                "dac_trunc_train_hit5",
                "dac_trunc_train_hit10",
                "dac_trunc_train_mrr",
            ],
            "bm25": ["bm25_hit1", "bm25_hit5", "bm25_hit10", "bm25_mrr"],
            "trunc": ["trunc_hit1", "trunc_hit5", "trunc_hit10", "trunc_mrr"],
            "window": ["window_hit1", "window_hit5", "window_hit10", "window_mrr"],
            # ===== 新增：效率对比组 =====
            "efficiency_total_time": [
                "eff_trunc_total_s",
                "eff_window_max_total_s", 
                "eff_bm25_total_s",
                "eff_graphcodebert_total_s",
                "eff_unixcoder_total_s",
            ],
            "efficiency_memory": [
                "eff_trunc_memory_mb",
                "eff_window_max_memory_mb",
                "eff_bm25_memory_mb",
                "eff_graphcodebert_memory_mb",
                "eff_unixcoder_memory_mb",
            ],
        }
        for name, cols in compare_spec.items():
            if not all(c in df_res.columns for c in cols):
                continue
            row = {"model": name}
            for c in cols:
                row[f"{c}_mean"] = float(df_res[c].mean())
                row[f"{c}_std"] = float(df_res[c].std(ddof=0))
            compare_rows.append(row)
        df_compare = pd.DataFrame(compare_rows)
        cmp_csv = os.path.join(OUTPUT_DIR, "model_compare_mean_std.csv")
        df_compare.to_csv(cmp_csv, index=False, encoding="utf-8")
        print(f"已导出模型对比表到: {cmp_csv}")

        # 分组指标导出（按漏洞类型；列名随数据集中类型变化）
        existing_grp_cols = sorted(
            c
            for c in df_res.columns
            if (
                c.startswith(("trunc_", "window_", "bm25_"))
                and c.endswith("_mrr")
                and len(c.split("_")) >= 3
            )
        )
        if existing_grp_cols:
            grp_summary = []
            for c in existing_grp_cols:
                grp_summary.append(
                    {"metric": c, "mean": float(df_res[c].mean()), "std": float(df_res[c].std(ddof=0))}
                )
            grp_df = pd.DataFrame(grp_summary)
            grp_csv = os.path.join(OUTPUT_DIR, "grouped_mrr_mean_std.csv")
            grp_df.to_csv(grp_csv, index=False, encoding="utf-8")
            print(f"已导出分组 MRR 汇总到: {grp_csv}")

        # 额外导出“消融表”文本：聚焦核心模块贡献（可直接贴论文）
        train_mode_label = "trunc_train" if TRAIN_POS_MODE == "trunc_first" else "window_aware_train"
        window_variant_label = (
            f"Trunc-Train + Window-Infer ({window_agg})"
            if TRAIN_POS_MODE == "trunc_first"
            else f"Window-Aware-Train + Window-Infer ({window_agg})"
        )
        ablation_rows = [
            {"variant": f"{train_mode_label} + trunc_infer", "hit1": float(df_res["trunc_hit1"].mean()), "hit5": float(df_res["trunc_hit5"].mean()), "hit10": float(df_res["trunc_hit10"].mean()), "mrr": float(df_res["trunc_mrr"].mean())},
            {"variant": window_variant_label, "hit1": float(df_res["window_hit1"].mean()), "hit5": float(df_res["window_hit5"].mean()), "hit10": float(df_res["window_hit10"].mean()), "mrr": float(df_res["window_mrr"].mean())},
            {"variant": f"{train_mode_label} + fusion", "hit1": float(df_res["fusion_hit1"].mean()), "hit5": float(df_res["fusion_hit5"].mean()), "hit10": float(df_res["fusion_hit10"].mean()), "mrr": float(df_res["fusion_mrr"].mean())},
        ]
        if "gate_mrr" in df_res.columns and not df_res["gate_mrr"].isna().all():
            ablation_rows.append(
                {
                    "variant": f"{train_mode_label} + DAC-Fusion",
                    "hit1": float(df_res["gate_hit1"].mean()),
                    "hit5": float(df_res["gate_hit5"].mean()),
                    "hit10": float(df_res["gate_hit10"].mean()),
                    "mrr": float(df_res["gate_mrr"].mean()),
                }
            )
        if "dac_gate_raw_mrr" in df_res.columns and not df_res["dac_gate_raw_mrr"].isna().all():
            ablation_rows.append(
                {
                    "variant": f"{train_mode_label} + DAC (adaptive gate, raw scores; no temp-softmax norm)",
                    "hit1": float(df_res["dac_gate_raw_hit1"].mean()),
                    "hit5": float(df_res["dac_gate_raw_hit5"].mean()),
                    "hit10": float(df_res["dac_gate_raw_hit10"].mean()),
                    "mrr": float(df_res["dac_gate_raw_mrr"].mean()),
                }
            )
        if "dac_temp_fixed_mrr" in df_res.columns and not df_res["dac_temp_fixed_mrr"].isna().all():
            ablation_rows.append(
                {
                    "variant": f"{train_mode_label} + temp-norm two-branch fusion (fixed alpha, no gate)",
                    "hit1": float(df_res["dac_temp_fixed_hit1"].mean()),
                    "hit5": float(df_res["dac_temp_fixed_hit5"].mean()),
                    "hit10": float(df_res["dac_temp_fixed_hit10"].mean()),
                    "mrr": float(df_res["dac_temp_fixed_mrr"].mean()),
                }
            )
        if "dac_trunc_train_mrr" in df_res.columns and not df_res["dac_trunc_train_mrr"].isna().all():
            ablation_rows.append(
                {
                    "variant": "trunc_first InfoNCE (no window-consistency) + DAC-Fusion (aux Stage1 in same script)",
                    "hit1": float(df_res["dac_trunc_train_hit1"].mean()),
                    "hit5": float(df_res["dac_trunc_train_hit5"].mean()),
                    "hit10": float(df_res["dac_trunc_train_hit10"].mean()),
                    "mrr": float(df_res["dac_trunc_train_mrr"].mean()),
                }
            )
        if RUN_STAGE2_HARD:
            ablation_rows.append({"variant": "window + stage2(hard-neg)", "hit1": np.nan, "hit5": np.nan, "hit10": np.nan, "mrr": np.nan})
        abl_df = pd.DataFrame(ablation_rows)
        abl_csv = os.path.join(OUTPUT_DIR, "ablation_table_mean.csv")
        abl_df.to_csv(abl_csv, index=False, encoding="utf-8")
        print(f"已导出消融表到: {abl_csv}")

        # ===== 新增：打印效率汇总 =====
        efficiency_cols = [
            ("eff_trunc_total_s", "Trunc"),
            ("eff_window_max_total_s", "Window(Max)"),
            ("eff_bm25_total_s", "BM25"),
            ("eff_graphcodebert_total_s", "GraphCodeBERT"),
            ("eff_unixcoder_total_s", "UniXCoder"),
        ]
        print("\n【效率分析】推理总时间 (均值 ± 标准差)：")
        for col, name in efficiency_cols:
            if col in df_res.columns:
                mean_v = df_res[col].mean()
                std_v = df_res[col].std(ddof=0)
                print(f"  {name:<20}: {mean_v:8.2f} ± {std_v:6.2f} s")
        
        memory_cols = [
            ("eff_trunc_memory_mb", "Trunc"),
            ("eff_window_max_memory_mb", "Window(Max)"),
            ("eff_bm25_memory_mb", "BM25"),
            ("eff_graphcodebert_memory_mb", "GraphCodeBERT"),
            ("eff_unixcoder_memory_mb", "UniXCoder"),
        ]
        print("\n【效率分析】GPU 显存占用 (均值 ± 标准差)：")
        for col, name in memory_cols:
            if col in df_res.columns:
                mean_v = df_res[col].mean()
                std_v = df_res[col].std(ddof=0)
                print(f"  {name:<20}: {mean_v:8.1f} ± {std_v:6.1f} MB")


def main():
    # Kaggle/Notebook 中避免 stdout 缓冲导致“看不到实时输出”
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    refresh_data_source_globals_from_environ()
    if QUICK_RUN:
        print(">>> QUICK_RUN=1: 已启用极速模式（单种子、少epoch、关闭重基线）")
    else:
        print(">>> QUICK_RUN=0: 标准训练模式（用于观察融合提升）")
    print(
        f">>> Config: VULN_IR_FULL_PAPER_RUN={VULN_IR_FULL_PAPER_RUN}, QUICK_RUN={QUICK_RUN}, "
        f"epochs_stage1={epochs_stage1}, ENABLE_DAC_FUSION={ENABLE_DAC_FUSION}, "
        f"DAC_EPOCHS={DAC_EPOCHS}, window_eval_top_chunk={window_eval_top_chunk}, "
        f"EVAL_DAC_ABLATION_COMPONENTS={EVAL_DAC_ABLATION_COMPONENTS}, "
        f"RUN_TRUNC_TRAIN_DAC_ABLATION={RUN_TRUNC_TRAIN_DAC_ABLATION}, "
        f"RUN_ST_BASELINE={RUN_ST_BASELINE}, DUMP_CASE_GATE_HEATMAP={DUMP_CASE_GATE_HEATMAP}"
    )

    # 路径不存在则直接退出，避免长时间训练后才发现读不到数据
    if not verify_paths():
        print("路径校验失败，请检查 CSV / solidity_codes 是否已放到 Kaggle 对应目录。")
        raise SystemExit(1)

    if RUN_MULTI_SEED:
        print(f"启用多种子实验：{SEED_LIST}")
        existing_rows, completed = load_partial_results()
        # 只保留当前 SEED_LIST 内的历史结果，并按 seed 去重（后出现覆盖先出现）
        existing_map = {}
        for r in existing_rows:
            try:
                s = int(r.get("seed"))
            except Exception:
                continue
            if s in SEED_LIST:
                existing_map[s] = r
        if existing_map:
            done_sorted = sorted(existing_map.keys())
            print(f"检测到已完成 seed（将跳过）: {done_sorted}")

        all_results = [existing_map[s] for s in sorted(existing_map.keys())]
        pending = [sd for sd in SEED_LIST if sd not in existing_map]
        if not pending:
            print("所有 seed 已在 partial 结果中，直接汇总。")

        for sd in pending:
            try:
                r = run_one_seed(sd)
                all_results.append(r)
                append_partial_result(r)
            except Exception as e:
                err_path = os.path.join(OUTPUT_DIR, f"error_seed_{sd}.txt")
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(err_path, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
                print(f"\n!!! seed={sd} 异常，详情已写入: {err_path}")
                raise
            finally:
                cleanup_gpu()
        summarize_multi_seed(all_results)
        print("\n全部完成（多种子）。")
    else:
        try:
            r = run_one_seed(SEED)
            append_partial_result(r)
        finally:
            cleanup_gpu()
        print("\n全部完成（单种子）。")


if __name__ == "__main__":
    main()