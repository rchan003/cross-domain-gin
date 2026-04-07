import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# cd /mnt/4tb/rachel_thesis/DDI
# python3 src/rachel/stats.py

# --- CONFIGURATION ---
BASE_DIR = ROOT_DIR / "transfer_learning" / "experiment_results"
METRIC_CHOICES = ["loss", "accuracy", "auc", "ap", "f1", "recall", "test"]

# ----------------------------------------------------------------
# 1. STATISTICAL UTILITIES
# ----------------------------------------------------------------


def calculate_cohens_d(group1, group2):
    """Calculates Cohen's d (Effect Size) between two independent groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0


def get_effect_size_label(d_val, small_threshold, medium_threshold, large_threshold):
    """Qualitative label for Cohen's d magnitude."""
    d = abs(d_val)
    if d < small_threshold:
        return "Negligible"
    elif d < medium_threshold:
        return "Small"
    elif d < large_threshold:
        return "Medium"
    else:
        return "Large"


# ----------------------------------------------------------------
# 2. ANALYSIS METHODS
# ----------------------------------------------------------------


def run_pointwise_analysis(df, metric, output_subfolder, p_value_threshold):
    """Calculates p-values for every epoch across all groups."""
    is_high_better = False if metric == "loss" else True
    results = []

    # Define groups to iterate: Global (None), plus Arch and Scale
    groups = [("Global", "All_Data", None, None)]
    for split in ["architecture", "scale"]:
        for val in df[split].unique():
            groups.append((split, val, split, val))

    for scope, label, col, val in groups:
        sub_df = df[df[col] == val] if col else df
        epochs = sorted(sub_df["epoch"].unique())

        epoch_sig_data = []
        for e in epochs:
            edata = sub_df[sub_df["epoch"] == e]
            rand = edata[edata["initialization"] == "random"][metric]
            pre = edata[edata["initialization"] == "pretrained"][metric]

            if len(rand) > 1 and len(pre) > 1:
                _, p = stats.ttest_ind(pre, rand, equal_var=False)
                # Significant if p is below threshold and direction is correct
                is_sig = (p < p_value_threshold) and (
                    (pre.mean() > rand.mean())
                    if is_high_better
                    else (pre.mean() < rand.mean())
                )
                epoch_sig_data.append(
                    {"epoch": e, "p_value": p, "is_significant": is_sig}
                )

        # Log summary for this metric/group
        sig_df = pd.DataFrame(epoch_sig_data)
        if not sig_df.empty:
            sig_only = sig_df[sig_df["is_significant"]]
            results.append(
                {
                    "Analysis_Scope": scope,
                    "Group_Value": label,
                    "Metric": metric,
                    "Onset_Epoch": (
                        sig_only["epoch"].min() if not sig_only.empty else None
                    ),
                    "Total_Sig_Epochs": len(sig_only),
                    "Significant_at_End": sig_df.iloc[-1]["is_significant"],
                }
            )

            # Save raw p-values for plotting ribbons later
            raw_path = output_subfolder / f"pvals_{metric}_{label}.csv"
            sig_df.to_csv(raw_path, index=False)

    return results


def run_impact_analysis(
    df,
    metric,
    scope_label,
    p_value_threshold,
    small_threshold,
    medium_threshold,
    large_threshold,
    group_col=None,
    group_val=None,
):
    """Calculates Best-Performance P-value and Cohen's d."""
    sub_df = df[df[group_col] == group_val] if group_col else df
    is_high_better = False if metric == "loss" else True

    # Extract peak performance per run
    if is_high_better:
        best_per_run = (
            sub_df.groupby(["initialization", "architecture", "scale", "run"])[metric]
            .max()
            .reset_index()
        )
    else:
        best_per_run = (
            sub_df.groupby(["initialization", "architecture", "scale", "run"])[metric]
            .min()
            .reset_index()
        )

    rand_scores = best_per_run[best_per_run["initialization"] == "random"][metric]
    pre_scores = best_per_run[best_per_run["initialization"] == "pretrained"][metric]

    if len(rand_scores) < 2 or len(pre_scores) < 2:
        return None

    _, p_val = stats.ttest_ind(pre_scores, rand_scores, equal_var=False)
    d_val = calculate_cohens_d(pre_scores, rand_scores)

    return {
        "Analysis_Scope": scope_label,
        "Group_Value": group_val if group_val else "All Data",
        "Metric": metric,
        "Pretrain_Best_Mean": pre_scores.mean(),
        "Random_Best_Mean": rand_scores.mean(),
        "P_Value": p_val,
        "Cohens_D": d_val,
        "Effect_Size": get_effect_size_label(
            d_val, small_threshold, medium_threshold, large_threshold
        ),
        "Conclusion": "Significant" if p_val < p_value_threshold else "Not Significant",
    }


# ----------------------------------------------------------------
# 3. MAIN EXECUTION
# ----------------------------------------------------------------


def main(args):
    data_path = BASE_DIR / args.run_name / "data" / "raw_data.csv"
    output_dir = BASE_DIR / args.run_name / "stats"

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_pvals_dir = output_dir / "raw_epoch_pvalues"
    raw_pvals_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    # Standardize metric name for hits@k
    if "test" in df.columns:
        df = df.rename(columns={"test": "hits_at_k"})

    if args.max_epoch is not None:
        print(f"[INFO] Filtering data to epochs <= {args.max_epoch}")
        df = df[df["epoch"] <= args.max_epoch]

    metrics_to_run = [m if m != "test" else "hits_at_k" for m in args.metrics]

    pointwise_results = []
    impact_results = []

    for m in metrics_to_run:
        if m not in df.columns:
            continue
        print(f"Processing Metric: {m.upper()}")

        # 1. Pointwise (All Epochs) Analysis
        pointwise_results.extend(
            run_pointwise_analysis(df, m, raw_pvals_dir, args.p_value_threshold)
        )

        # 2. Impact (Best Performance) Analysis
        # Global
        impact_results.append(
            run_impact_analysis(
                df,
                m,
                "Global",
                args.p_value_threshold,
                args.effect_size_small,
                args.effect_size_medium,
                args.effect_size_large,
            )
        )
        # Groups
        for split in ["architecture", "scale"]:
            for val in df[split].unique():
                res = run_impact_analysis(
                    df,
                    m,
                    split,
                    args.p_value_threshold,
                    args.effect_size_small,
                    args.effect_size_medium,
                    args.effect_size_large,
                    split,
                    val,
                )
                if res:
                    impact_results.append(res)

    # --- SAVE SUMMARIES ---
    scope_order = ["Global", "architecture", "scale"]

    # Save Impact Report
    impact_df = pd.DataFrame(impact_results)
    impact_df["Analysis_Scope"] = pd.Categorical(
        impact_df["Analysis_Scope"], categories=scope_order, ordered=True
    )
    impact_df = impact_df.sort_values(["Analysis_Scope", "Metric", "Group_Value"])
    impact_df.to_csv(output_dir / "master_impact_report.csv", index=False)

    # Save Significance Report
    sig_report_df = pd.DataFrame(pointwise_results)
    sig_report_df["Analysis_Scope"] = pd.Categorical(
        sig_report_df["Analysis_Scope"], categories=scope_order, ordered=True
    )
    sig_report_df = sig_report_df.sort_values(
        ["Analysis_Scope", "Metric", "Group_Value"]
    )
    sig_report_df.to_csv(output_dir / "complete_significance_report.csv", index=False)

    print(f"\n[DONE] Analysis complete.")
    print(f"Master Impact Report: {output_dir / 'master_impact_report.csv'}")
    print(
        f"Epoch Significance Report: {output_dir / 'complete_significance_report.csv'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Result analysis for transfer learning experiments"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Experiment-results directory name under transfer_learning/experiment_results. \
        Plot results must be run first to create the data directory for this experiment. \
        Must be the same as the --run-name used in plot_results.py.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=False,
        choices=METRIC_CHOICES,
        default=METRIC_CHOICES,
        help="Metrics to analyze. Provide one or more values.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        required=False,
        default=None,
        help="Maximum epoch to include in analysis",
    )
    parser.add_argument(
        "--p-value-threshold",
        type=float,
        required=False,
        default=0.05,
        help="P-value cutoff used to determine statistical significance.",
    )
    parser.add_argument(
        "--effect-size-small",
        type=float,
        required=False,
        default=0.2,
        help="Cohen's d boundary between negligible and small effects.",
    )
    parser.add_argument(
        "--effect-size-medium",
        type=float,
        required=False,
        default=0.5,
        help="Cohen's d boundary between small and medium effects.",
    )
    parser.add_argument(
        "--effect-size-large",
        type=float,
        required=False,
        default=0.8,
        help="Cohen's d boundary between medium and large effects.",
    )
    args = parser.parse_args()
    main(args)
