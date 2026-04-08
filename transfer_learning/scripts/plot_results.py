import argparse
import re
from enum import Enum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METRIC_CHOICES = ["loss", "accuracy", "auc", "ap", "f1", "recall", "test"]
CI_CHOICES = [90, 95, 99]
DASH_STYLE_CHOICES = ["solid", "dashed"]

# DDI CSV logs one row per Hits@K; loss/accuracy/etc. repeat — dedupe for non-test metrics.
_DEDUP_KEYS = ["run", "epoch", "initialization", "architecture", "scale"]


def iter_metrics_with_k(metrics, df):
    """Yield (metric, test_hits_k) — one plot per K for test when hits@k is present."""
    for m in metrics:
        if m not in df.columns:
            continue
        if m == "test" and "hits@k" in df.columns:
            for k in sorted(df["hits@k"].dropna().unique().astype(int)):
                yield m, int(k)
        else:
            yield m, None


def prepare_df_for_plot_metric(df, metric, test_hits_k=None):
    """
    Match analyze_results: one row per run/epoch/treatment for scalar metrics so CIs are not
    artificially tight. For test (Hits@K), filter to one K per plot (caller loops all K).
    """
    df = df.copy()
    if not all(k in df.columns for k in _DEDUP_KEYS):
        return df

    if metric == "test" and "hits@k" in df.columns:
        if test_hits_k is None:
            raise ValueError(
                "metric 'test' with hits@k column requires test_hits_k (internal error)."
            )
        return df[df["hits@k"] == test_hits_k].copy()

    if "hits@k" in df.columns:
        df = df.drop_duplicates(subset=_DEDUP_KEYS, keep="first").copy()
    return df


class PlotType(Enum):
    GLOBAL = auto()
    SCALE_COMBINED = auto()
    SCALE_INDIVIDUAL = auto()
    ARCH_COMBINED = auto()
    ARCH_INDIVIDUAL = auto()


def parse_details(folder_name):
    """Extracts Architecture and Scale from directory names."""
    arch = re.search(r"(L\d+).*(H\d+)", folder_name)
    scale = re.search(r"(\d+x\d+)", folder_name)

    return (
        f"{arch.group(1)}_{arch.group(2)}" if arch else "Unknown",
        scale.group(1) if scale else "Unknown_Scale",
    )


def find_finished_subdir(experiment_dir: Path, prefix: str):
    for subdir in experiment_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith(prefix):
            if (subdir / "finished.txt").exists():
                return subdir
    return None


def load_data(root_dir, data_dir, force=False):
    cache_path = data_dir / "raw_data.csv"
    if cache_path.exists() and not force:
        print(f"[CACHE] Loading: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"[SCAN] Processing results from: {root_dir}")
    all_results = []
    for exp_dir in filter(Path.is_dir, root_dir.iterdir()):
        arch, scale = parse_details(exp_dir.name)
        for variant in ["control", "pretrained"]:
            target_dir = find_finished_subdir(exp_dir, variant)
            if not target_dir:
                continue
            for csv_file in target_dir.glob("*.csv"):
                df = pd.read_csv(csv_file)
                df["initialization"] = (
                    "random" if variant == "control" else "pretrained"
                )
                df["architecture"] = arch
                df["scale"] = scale
                all_results.append(df)

    if not all_results:
        raise ValueError(f"No valid experiment data found in {root_dir}")

    full_df = pd.concat(all_results, ignore_index=True)
    if "hits@k" in full_df.columns:
        full_df["hits@k"] = (
            pd.to_numeric(full_df["hits@k"], errors="coerce").fillna(0).astype(int)
        )
        full_df = full_df[full_df["hits@k"] > 0]

    full_df = full_df.sort_values(by=["architecture", "scale"], ascending=True)
    full_df.to_csv(cache_path, index=False)
    return full_df


def format_plot_data(df, metric, split_col, plot_type):
    plot_df = df.copy()
    # Multi-K in one frame: hue by K. Single-K plot: hue by split_col (e.g. arch / scale).
    is_hits = (
        metric == "test"
        and "hits@k" in plot_df.columns
        and plot_df["hits@k"].nunique() > 1
    )

    # Check if this is any of the "Combined" view types
    is_combined = plot_type in [
        PlotType.SCALE_COMBINED,
        PlotType.ARCH_COMBINED,
    ]

    if is_hits:
        if is_combined:
            label = f"{split_col} & metric"
            plot_df[label] = (
                plot_df[split_col].astype(str)
                + " hits@"
                + plot_df["hits@k"].astype(str)
            )
        else:
            label = "K Value"
            plot_df[label] = plot_df["hits@k"].astype(str)
        s_col = label
    else:
        s_col = split_col

    split_name = s_col.title().replace("_", " ")
    plot_df.rename(columns={s_col: split_name}, inplace=True)
    style_name = "Initialization"
    plot_df.rename(columns={"initialization": style_name}, inplace=True)

    return plot_df, split_name, style_name


def get_title(metric_label, plot_type, group, confidence_interval, test_hits_k=None):
    if plot_type == PlotType.GLOBAL:
        base = f"{metric_label} For All Experiments ({confidence_interval}% CI)"
    elif plot_type == PlotType.SCALE_COMBINED:
        base = f"{metric_label} Across All Scales ({confidence_interval}% CI)"
    elif plot_type == PlotType.SCALE_INDIVIDUAL:
        base = f"{metric_label} for Scale: {group} ({confidence_interval}% CI)"
    elif plot_type == PlotType.ARCH_COMBINED:
        base = f"{metric_label} Across Architectures ({confidence_interval}% CI)"
    elif plot_type == PlotType.ARCH_INDIVIDUAL:
        base = f"{metric_label} for Architecture: {group} ({confidence_interval}% CI)"
    else:
        base = f"{metric_label} Plot"
    if test_hits_k is not None:
        return f"{base} — Hits@K={test_hits_k}"
    return base


def plot_line(
    df,
    metric,
    output_folder,
    split_col,
    plot_type,
    confidence_interval,
    dash_styles,
    force=False,
    group=None,
    test_hits_k=None,
):
    if metric == "test" and test_hits_k is not None:
        fname = f"test_k{test_hits_k}.png"
    elif metric == "test":
        fname = "test.png"
    else:
        fname = f"{metric}.png"
    save_path = output_folder / fname
    if save_path.exists() and not force:
        return

    df = prepare_df_for_plot_metric(df, metric, test_hits_k)
    if df.empty:
        print(f"[WARN] Skipping plot (no rows): metric={metric}, path={save_path}")
        return

    plot_df, split_name, style_name = format_plot_data(df, metric, split_col, plot_type)
    metric_label = "hits@k" if metric == "test" else metric
    metric_label = (
        metric_label.upper() if len(metric_label) < 4 else metric_label.capitalize()
    )
    title = get_title(
        metric_label, plot_type, group, confidence_interval, test_hits_k=test_hits_k
    )

    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    sns.lineplot(
        data=plot_df,
        x="epoch",
        y=metric,
        hue=split_name,
        style=style_name,
        dashes=dash_styles,
        errorbar=("ci", confidence_interval),
        palette="husl",
        linewidth=1.8,
        alpha=0.8,
    )

    plt.xlabel("Epoch")
    plt.ylabel(metric_label)
    plt.title(title, fontsize=14, pad=15)
    plt.legend(
        title="Legend", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0
    )
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_section(
    df,
    base_path,
    split_col,
    metrics,
    confidence_interval,
    dash_styles,
    force=False,
):
    # Mapping split columns to PlotTypes
    type_map = {
        "scale": (PlotType.SCALE_COMBINED, PlotType.SCALE_INDIVIDUAL),
        "architecture": (PlotType.ARCH_COMBINED, PlotType.ARCH_INDIVIDUAL),
    }
    comb_type, ind_type = type_map[split_col]

    print(f"\n  [SUB-SECTION] Combined")
    for m, test_k in iter_metrics_with_k(metrics, df):
        plot_line(
            df,
            m,
            base_path / "combined",
            split_col,
            comb_type,
            confidence_interval,
            dash_styles,
            force,
            test_hits_k=test_k,
        )

    print(f"\n  [SUB-SECTION] Individual")
    for group in df[split_col].unique():
        group_df = df[df[split_col] == group]
        for m, test_k in iter_metrics_with_k(metrics, group_df):
            plot_line(
                group_df,
                m,
                base_path / "individual" / str(group),
                "initialization",
                ind_type,
                confidence_interval,
                dash_styles,
                force,
                group,
                test_hits_k=test_k,
            )


def main(args):
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir
        else BASE_DIR / "DDI" / "results" / args.run_name
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else BASE_DIR / "transfer_learning" / "experiment_results" / args.run_name
    )

    dash_styles = {
        "random": "" if args.random_dash_style == "solid" else (2, 2),
        "pretrained": "" if args.pretrained_dash_style == "solid" else (2, 2),
    }

    for sub in [
        "data",
        "plots/global",
        "plots/split_by_architecture",
        "plots/split_by_scale",
    ]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    df = load_data(results_dir, output_dir / "data", force=args.force)
    if args.max_epoch is not None:
        print(f"[INFO] Limiting plots to epochs <= {args.max_epoch}")
        df = df[df["epoch"] <= args.max_epoch]

    print("\n[SECTION 1/3] Global Aggregate")
    for m, test_k in iter_metrics_with_k(args.metrics, df):
        plot_line(
            df,
            m,
            output_dir / "plots" / "global",
            "initialization",
            PlotType.GLOBAL,
            args.confidence_interval,
            dash_styles,
            args.force,
            test_hits_k=test_k,
        )

    print("\n[SECTION 2/3] Architecture Analysis")
    generate_section(
        df,
        output_dir / "plots" / "split_by_architecture",
        "architecture",
        args.metrics,
        args.confidence_interval,
        dash_styles,
        args.force,
    )

    print("\n[SECTION 3/3] Scale Analysis")
    generate_section(
        df,
        output_dir / "plots" / "split_by_scale",
        "scale",
        args.metrics,
        args.confidence_interval,
        dash_styles,
        args.force,
    )

    print(f"\nPipeline Complete. Outputs: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thesis Plotting Suite")
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Experiment set name used to build default input/output directories.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=False,
        default=None,
        help="Optional explicit DDI results directory. Overrides --run-name default path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default=None,
        help="Optional explicit output directory. Overrides --run-name default path.",
    )
    parser.add_argument(
        "--confidence-interval",
        type=int,
        required=False,
        choices=CI_CHOICES,
        default=95,
        help="Confidence interval for seaborn error bands.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=False,
        choices=METRIC_CHOICES,
        default=METRIC_CHOICES,
        help="Metrics to plot. Provide one or more values.",
    )
    parser.add_argument(
        "--random-dash-style",
        type=str,
        required=False,
        choices=DASH_STYLE_CHOICES,
        default="solid",
        help='Line style for "random" initialization curves.',
    )
    parser.add_argument(
        "--pretrained-dash-style",
        type=str,
        required=False,
        choices=DASH_STYLE_CHOICES,
        default="dashed",
        help='Line style for "pretrained" initialization curves.',
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Overwrite existing plots/cache",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        required=False,
        default=None,
        help="Maximum epoch to include in plots",
    )
    args = parser.parse_args()
    main(args)
