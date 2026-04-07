"""
Project-level experiment configuration.

This file is intended to centralize experiment settings that are shared by
runner scripts. Sections are grouped by project to keep source (JSSP) and
target (DDI) experiment settings separated.

See more details about the parameters in the JSSP/train.py and DDI/train.py scripts.
"""

# ============================================================================
# JSSP: Configurations for `JSSP/scripts/run_experiments.py`
# ============================================================================

JSSP_BATCH_SIZE = 64  # Number of JSSP instances solved in parallel per training batch.
JSSP_NUM_EPOCHS = 1  # Runner-level epoch count used to derive total episodes.
JSSP_NUM_EPISODES = JSSP_BATCH_SIZE * JSSP_NUM_EPOCHS  # Total episodes = batch_size * num_epochs.

# JSSP_SEARCH_SPACE: Combination of parameter values to try in the JSSP experiments.
# Each combination of values is an experiment, thus the number of experiments is the 
# product of the lengths of the lists in the SEARCH_SPACE.
JSSP_SEARCH_SPACE = {
    "embedding_type": ["gin"],  # Encoder variant(s); choices: "gin", "dghan", "gin+dghan".
    "hidden_dim": [128],  # Hidden feature width for encoder/policy layers.
    "lr": [1e-4],  # Learning rate values to sweep.
    "embedding_layer": [3],  # Number of encoder layers (GIN/DGHAN depth).
    "j": [10],  # Number of jobs (problem scale dimension).
    "m": [10],  # Number of machines (problem scale dimension).
}

# JSSP_FIXED_PARAMS: Parameter values that are fixed for all JSSP experiments.
JSSP_FIXED_PARAMS = {
    "l": 1,  # Minimum processing time bound for synthetic instance generation.
    "h": 99,  # Maximum processing time bound (exclusive) for synthetic generation.
    "init_type": "fdd-divide-mwkr",  # Initial solution rule; choices: "plist", "spt", "fdd-divide-mwkr".
    "reward_type": "yaoxin",  # Reward style; choices: "yaoxin", "consecutive".
    "gamma": 1.0,  # Discount factor for return calculation.
    "policy_layer": 4,  # Number of policy network layers after graph embeddings.
    "heads": 1,  # Attention heads for DGHAN; relevant when embedding_type uses DGHAN.
    "drop_out": 0.0,  # Dropout for DGHAN layers; relevant when embedding_type uses DGHAN.
    "steps_learn": 10,  # Environment steps between policy updates.
    "transit": 500,  # Maximum transitions per episode rollout.
    "batch_size": JSSP_BATCH_SIZE,  # Batch size forwarded to JSSP training script.
    "episodes": JSSP_NUM_EPISODES,  # Total episodes forwarded to JSSP training script.
    "step_validation": 1,  # Validate every N training batches.
}


# ============================================================================
# DDI: Configurations for `DDI/scripts/run_experiments.py`
# ============================================================================

DDI_EXPERIMENT_SET_NAME = "ogbl-ddi-random-sampling"  # Descriptive name for this set of experiments.
DDI_SAVE_RESULTS = True  # Whether DDI train runs should save logs/summaries.

# DDI_SEARCH_SPACE: Combination of parameter values to try in the DDI experiments.
# Each combination of values is an experiment, thus the number of experiments is the 
# product of the lengths of the lists in the SEARCH_SPACE.
DDI_SEARCH_SPACE = {
    "init_mode": ["pretrained", "random"],  # Initialization modes to compare; choices: "pretrained", "random".
}

# DDI_FIXED_PARAMS: Parameter values that are fixed for all DDI experiments.
DDI_FIXED_PARAMS = {
    "dataset": "ogbl-ddi",  # Dataset choice; choices: "ogbl-ddi", "drugbankddi", "biosnapddi".
    "loss_function": "binary_cross_entropy",  # Loss; choices: "pairwise_ranking", "binary_cross_entropy".
    "runs": 20,  # Number of repeated runs per configuration.
    "device": 0,  # CUDA device index; use -1 in DDI train.py for CPU.
    "batch_size": 18000,  # Positive-edge minibatch size used during training.
    "epochs": 25,  # Number of epochs per run.
    "eval_steps": 1,  # Evaluate every N epochs.
    "lr": 1e-6,  # Learning rate for Adam optimizer.
    "dropout": 0.1,  # Dropout in link predictor MLP.
    "num_neg_per_pos": 1,  # Number of negatives sampled per positive edge.
    "save_results": DDI_SAVE_RESULTS,  # Forwarded save flag to DDI train.py.
}
