# GNN-based DDI Prediction

This repository contains code for the paper **"GNN-based DDI Prediction"**, which explores the use of Graph Neural Networks (GNNs) for predicting drug-drug interactions (DDIs) across various datasets.

---

## Environment Setup

To ensure reproducibility, all dependencies are listed in the `environment.yml` file. You can create the conda environment using:

```bash
conda env create -f environment.yml
conda activate gnn-ddi  # replace with the name defined in your .yml file

python src/main.py --model "gcn_skipconnection" --dataset "biosnapddi"


For more information see the args help.