# Open Cross Domain Transfer Learning with Graph Isomorphism Networks (GIN)

A research codebase for open cross-domain transfer learning on graph tasks using Graph Isomorphism Networks.

This project combines two main projects:

1. **JSSP (Job Shop Scheduling)** implements the L2S (Learning to Schedule) algorithm with DRL-guided improvement heuristic, where GINs are used to encode complete solutions for the scheduling problem.
2. **DDI (Drug-Drug Interaction)** implements a GNN-based model for predicting drug-drug interactions across multiple datasets, where GINs are used to encode drug molecular graphs.

## 🔍 Project Overview

The project focuses on investigating the ability of GINs to transfer knowledge from a source domain (Job Shop Scheduling Problem - JSSP) to a target domain (Drug-Drug Interaction Prediction - DDI). The codebase includes implementations for both domains, training scripts, and analysis tools for evaluating transfer learning performance.

The core algorithms and ideas were derived from the work of [Zhang et al.] and [Abbas et al.], and we acknowledge the contributions of the original authors. If you make use of the code/experiment or algorithms in your work, please cite the respective papers as indicated in the citations section below.

## 📁 Repository Structure

```bash
cross-domain-gin/
├── DDI/                        # Drug-Drug Interaction Prediction code and experiments
│   ├── src/                    # Source code for DDI experiments
│   ├── scripts/                # Scripts for running DDI experiments
│   ├── logs/                   # Logs from DDI experiments
│   └── results/                # Results from DDI experiments
├── JSSP/                       # Job Shop Scheduling Problem code and experiments
│   ├── src/                    # Source code for JSSP experiments
│   ├── scripts/                # Scripts for running JSSP experiments
│   ├── logs/                   # Logs from JSSP experiments
│   └── results/                # Results from JSSP experiments
├── transfer_learning/          # Code for analyzing transfer learning results
│   ├── scripts/                # Scripts for plotting, analysis, and model visualization
│   ├── experiment_results/     # Aggregated transfer-learning plots/statistics outputs
│   └── model_diagrams/         # Saved architecture diagrams (e.g., JSSP actor variants)
├── config.py                   # Central experiment config for JSSP/DDI run_experiments
```

## 📦 Requirements

- Python 3.8
- Git LFS
- CUDA 12 compatible GPU (optional, but recommended)
- Graphviz (optional, for visualizing models)

## ⚙️ Environment Setup

### Conda Environment For JSSP Project

```bash
conda create -n jssp_env python=3.8
conda activate jssp_env
python -m pip install -r JSSP/requirements.txt
```

### Conda Environment For DDI Project

```bash
conda create -n ddi_env python=3.8
conda activate ddi_env
python -m pip install -r DDI/requirements.txt
```

## 🚀 Running Experiments

### 1. Create Pretrained GIN Models on JSSP

First, run the JSSP experiments to create pretrained GIN models. Update the JSSP experiment settings in `config.py`. Then run the command below to start the experiments. 

Logs and results will be saved under `JSSP/logs` and `JSSP/results` in subdirectories named after each experiment. The best pretrained model (`best_gin_incumbent.pth`) from each experiment will be used for transfer learning in the DDI experiments.

```bash
cd JSSP
python scripts/run_experiments.py
```

### 2. DDI Training with Transfer Learning

Run the transfer learning experiments using the pretrained JSSP in models by first setting the DDI experiment configuration in `config.py`. Then run the command below to start the experiments. Logs and results will be saved under `DDI/logs` and `DDI/results` in subdirectories named after `DDI_EXPERIMENT_SET_NAME`.

For each pretrained JSSP model (`best_gin_incumbent.pth`) in the `JSSP/results` folder, the code performs multiple training runs under two conditions: random initialization (control) and initialization from pretrained weights. Each paired run uses the same random seed schedule to ensure a fair comparison.

```bash
cd DDI
python scripts/run_experiments.py
```

### 3. Analyzing Results

After DDI experiments finish, use the scripts under `transfer_learning/scripts/` to aggregate CSV logs, compare random versus pretrained initialization, and produce figures and statistical summaries. Run **`plot_results.py` first**: it scans `DDI/results/<experiment-set>/` (where `<experiment-set>` matches `DDI_EXPERIMENT_SET_NAME` in `config.py`), builds a cached `raw_data.csv`, and writes plots under `transfer_learning/experiment_results/<experiment-set>/`. Then run **`analyze_results.py`** on the same experiment set: it reads `transfer_learning/experiment_results/<experiment-set>/data/raw_data.csv` and writes reports under `transfer_learning/experiment_results/<experiment-set>/stats/`.

Both scripts take a **required** `--run-name` that must be the same string as your DDI experiment set folder name (for example, the value of `DDI_EXPERIMENT_SET_NAME`). Optional flags control metrics, confidence intervals, plot styling, epoch limits, and (for analysis) p-value and effect-size thresholds. Use `--help` on each script for the full argument list.

```bash
cd transfer_learning
python scripts/plot_results.py --run-name <experiment-set>
python scripts/analyze_results.py --run-name <experiment-set>
```

## 📚 Citations

### JSSP

The code in the L2S folder is based on the work of [Zhang et al.] in their paper "Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling". Their original repository can be found here in the [L2S GitHub](https://github.com/zcaicaros/L2S). If you make use of the code/experiment or L2S algorithm in your work, please cite their paper below.

```text
Cong Zhang, Zhiguang Cao, Wen Song, Yaoxin Wu, & Jie Zhang. (2024). Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling.
```

### DDI

The code in the DDI folder is based on the work of [Abbas et al.] in their paper "Graph Neural Network-Based Drug-Drug Interaction Prediction". Their original repository can be found here in the [DDI Github](https://github.com/khushnood/DrugDruginteractionPredictionBasedOnGNN). If you make use of the code/experiment in your work, please cite their paper below.

```text
Abbas, K., Hao, C., Yong, X. et al. Graph neural network-based drug-drug interaction prediction. Sci Rep 15, 30340 (2025). https://doi.org/10.1038/s41598-025-12936-1
```

## 📝 TODO

- [ ] Add more comments to code
- [ ] Add docstrings to functions
- [ ] Move DDI logs into their experiment results folder instead of having them all in one logs folder
