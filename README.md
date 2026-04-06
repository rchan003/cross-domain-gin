# Open Cross Domain Transfer Learning with Graph Isomorphism Networks (GIN)

A research codebase for open cross-domain transfer learning on graph tasks using Graph Isomorphism Networks. 

This project combines two main projects:
1. **JSSP (Job Shop Scheduling)** implements the L2S (Learning to Schedule) algorithm with DRL-guided improvement heuristic, where GINs are used to encode complete solutions for the scheduling problem.
2. **DDI (Drug-Drug Interaction)** implements a GNN-based model for predicting drug-drug interactions across multiple datasets, where GINs are used to encode drug molecular graphs.

## 🔍 Project Overview

The project focuses on investigating the ability of GINs to transfer knowledge from a source domain (Job Shop Scheduling Problem - JSSP) to a target domain (Drug-Drug Interaction Prediction - DDI). The codebase includes implementations for both domains, training scripts, and analysis tools for evaluating transfer learning performance.

The core algorithms and ideas were derived from the work of [Zhang et al.] and [Abbas et al.], and we acknowledge the contributions of the original authors. If you make use of the code/experiment or algorithms in your work, please cite the respective papers as indicated in the citations section below.

## 📁 Repository Structure
```
TODO: Update with actual repo structure
```


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
### JSSP
The code in the L2S folder is based on the work of [Zhang et al.] in their paper "Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling". Their original repository can be found [here](https://github.com/zcaicaros/L2S). If you make use of the code/experiment or L2S algorithm in your work, please cite their paper below. 

```text
Cong Zhang, Zhiguang Cao, Wen Song, Yaoxin Wu, & Jie Zhang. (2024). Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling.
```

### DDI 
The code in the DDI folder is based on the work of [Abbas et al.] in their paper "Graph Neural Network-Based Drug-Drug Interaction Prediction". Their original repository can be found [here](https://github.com/khushnood/DrugDruginteractionPredictionBasedOnGNN). If you make use of the code/experiment in your work, please cite their paper below.

```text
Abbas, K., Hao, C., Yong, X. et al. Graph neural network-based drug-drug interaction prediction. Sci Rep 15, 30340 (2025). https://doi.org/10.1038/s41598-025-12936-1
```