# Open Cross Domain Transfer Learning with Graph Isomorphism Networks (GIN)

A research codebase for open cross-domain transfer learning on graph tasks using Graph Isomorphism Networks. 

This project combines two main projects:
1. **JSSP (Job Shop Scheduling)** implements the L2S (Learning to Schedule) algorithm with DRL-guided improvement heuristic, where GINs are used to encode complete solutions for the scheduling problem.
  * based on the work of [Zhang et al.] in their paper "Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling". Their original repository can be found [here](https://github.com/zcaicaros/L2S).
2. **DDI (Drug-Drug Interaction)** implements a GNN-based model for predicting drug-drug interactions across multiple datasets, where GINs are used to encode drug molecular graphs.
  * based on the work of [Abbas et al.] in their paper "Graph Neural Network-Based Drug-Drug Interaction Prediction". Their original repository can be found [here](https://github.com/khushnood/DrugDruginteractionPredictionBasedOnGNN).

## 🔍 Project Overview

The project focuses on transferring knowledge from a source domain (Job Shop Scheduling Problem - JSSP) to a target domain (Drug-Drug Interaction Prediction - DDI) using GINs as feature extractors. The codebase includes implementations for both domains, training scripts, and analysis tools for evaluating transfer learning performance.

While much of the code from both projects has been refactored and optimized for better GPU utilization, the core algorithms and ideas were derived from the original research papers, and we acknowledge the contributions of the original authors. If you make use of the code/experiment or algorithms in your work, please cite the respective papers as indicated in the citations section below.

## 📁 Repository Structure

- `DDI/`
  - `dataset/` : `biosnapddi`, `drugbankddi`, `ogbl_ddi`
  - `src/` : classes and functions for data loading, model definition, training, and evaluation
  - `scripts/` : `train.py`, `run_experiments.py`
- `JSSP/`
  - `src/` : DRL environment, policy, message passing
  - `scripts/` : `train.py`, `run_experiments.py`
  - `results/`: pretrained model weights, logs, and evaluation results
- `transfer_learning/`
  - `scripts/`: `analyze_results.py`, `plot_results.py`
  - `results/`: aggregated metrics, plots, and statistical analysis outputs

## ⚙️ Environment Setup

### Conda Environment For DDI Experiments
```bash
conda create -n ddi_env python=3.8
conda activate ddi_env
pip install -r DDI/requirements.txt
```

### Conda Environment For JSSP Experiments
```bash
conda create -n jssp_env python=3.8
conda activate jssp_env
pip install -r JSSP/requirements.txt
```

## 🚀 Running Experiments
### JSSP Pretraining
```bash
cd JSSP
python src/train.py --config configs/pretrain_config.yaml
# TODO: Update repo with config files
```

## 📊 Analyzing Results
```bash
cd transfer_learning
python scripts/analyze_results.py # TODO add parse args to scripts 
```

## 📚 Citations
The code in the L2S folder is based on the work of [Zhang et al.] in their paper "Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling". Though their code has been refactored and adapted for these experiments, the core algorithm and ideas are derived from their research. If you make use of the code/experiment or L2S algorithm in your work, please cite their paper (Bibtex below).

```
@InProceedings{
zhang2024deep,
title={Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling},
author={Zhang, Cong and Cao, Zhiguang and Song, Wen and Wu, Yaoxin and Zhang, Jie},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024}
}
```

The code in the DDI folder is based on the work of [Abbas et al.] in their paper "Graph Neural Network-Based Drug-Drug Interaction Prediction". Though their code has been refactored and adapted for these experiments, the core training algorithm was derived from their research. If you make use of the code/experiment in your work, please cite their paper (Bibtex below).

```@InProceedings{
author = {Author, A. and Author, B. and Author, C.},
title = {Graph Neural Network-Based Drug-Drug Interaction Prediction},
booktitle = {TODO: Update with actual conference/journal name},
year = {2025}
}```