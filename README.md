# DAC-Fusion
Official implementation of DAC-Fusion for natural-language to vulnerable smart-contract retrieval: window-consistent contrastive training, truncation–window dual-branch retrieval, and instance-adaptive fusion on SCRUBD (5 seeds).


# Overview
Retrieving vulnerable smart contracts from natural-language descriptions is critical for IoT security analysis. 
Existing dense retrievers suffer from **train–inference misalignment**:
- Training: often uses fixed truncated code prefixes
- Inference: relies on sliding-window aggregation for long contracts

This inconsistency makes models fail to capture local vulnerability patterns in long smart contracts.

We propose:
1. **Window-consistency contrastive learning** — align training with sliding-window inference  
2. **DAC-Fusion** — instance-adaptive fusion of global truncation scores and local window scores

Our method consistently improves long-contract retrieval performance.


# Key Contributions
- Strict **contract-level train/val/test split** to avoid data leakage
- Window-aligned contrastive learning to bridge train–inference gap
- DAC-Fusion: adaptive gating + temperature-normalized score fusion
- Five-seed evaluation with significance test & length-bucket analysis
- Superior long-contract retrieval performance (Hit@10)


# Dataset
Experiments use the **SCRUBD (https://github.com/sujeetc/SCRUBD)** benchmark (reentrancy + unchecked exception; 439 contracts; contract-level split).

Random seeds: 42, 123, 456, 789, 2026. Please cite the original SCRUBD paper when using the data.


# Main results (5-seed mean ± std)
From reproduce/per_seed_results.jsonl. (matches the paper, Table 2)


# Installation
```bash
git clone https://github.com/Zhangpipi1128/DAC-Fusion.git
cd DAC-Fusion
pip install -r requirements.txt
```


# Reproduction
**Kaggle (Recommended)**
-Run kaggle_notebook_cell1.py to clone SCRUBD and set environment.
-Run kaggle.py with:
      QUICK_RUN=0
      VULN_IR_FULL_PAPER_RUN=1
      Results will be saved to /kaggle/working/output.
-Outputs: VULN_IR_OUT/per_seed_results.jsonl and per-seed CSVs (default on Kaggle: /kaggle/working/output).


# Citation
If you use this code before formal publication, please cite the manuscript (update the entry after camera-ready / acceptance):

```bibtex
@misc{zhang2026dacfusion,
  title={Bridging Train--Inference Misalignment for Long-Code Vulnerability Retrieval},
  author={Zhang, Huiling and Wu, Weiqiang and Wu, Guangfu and Wu, Rui},
  year={2026},
  note={Submitted to IoTCIT 2026}
}
```


# License
MIT License


# Acknowledgements
This repository is built on CodeBERT, GraphCodeBERT, UniXCoder, SCRUBD, Sentence-BGE.
Thanks to the open-source community.

