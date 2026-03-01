#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FF8 = [
    "uniface_ff",
    "blendface_ff",
    "mobileswap_ff",
    "e4s_ff",
    "facedancer_ff",
    "fsgan_ff",
    "inswap_ff",
    "simswap_ff",
]


def _split_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _load_auc_acc_delta(pissa_only_csv: Path, pissa_plus_ln_csv: Path) -> pd.DataFrame:
    cols = ["dataset", "video_auc", "video_acc"]
    df_only = pd.read_csv(pissa_only_csv)[cols].copy()
    df_plus = pd.read_csv(pissa_plus_ln_csv)[cols].copy()

    df_only = df_only.rename(columns={"video_auc": "video_auc_pissa_only", "video_acc": "video_acc_pissa_only"})
    df_plus = df_plus.rename(columns={"video_auc": "video_auc_pissa_plus_ln", "video_acc": "video_acc_pissa_plus_ln"})

    merged = df_only.merge(df_plus, on="dataset", how="inner")
    merged["delta_auc"] = merged["video_auc_pissa_plus_ln"] - merged["video_auc_pissa_only"]
    merged["delta_acc"] = merged["video_acc_pissa_plus_ln"] - merged["video_acc_pissa_only"]
    return merged


def _toy_map(kind: str, size: int = 41) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(x, y)
    base = np.exp(-2.4 * (xx**2 + yy**2))

    if kind == "left":
        bias = 1.0 - 0.7 * xx
    elif kind == "right":
        bias = 1.0 + 0.7 * xx
    elif kind == "top":
        bias = 1.0 - 0.7 * yy
    elif kind == "bottom":
        bias = 1.0 + 0.7 * yy
    else:
        bias = np.ones_like(base)

    z = np.clip(base * bias, 0.0, None)
    z = z / np.maximum(z.sum(), 1e-8)
    return z


def make_axis_explainer(out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    kinds = [
        ("left", "x < 0 (left-biased)"),
        ("right", "x > 0 (right-biased)"),
        ("top", "y < 0 (top-biased)"),
        ("bottom", "y > 0 (bottom-biased)"),
    ]
    for ax, (kind, title) in zip(axes.flatten(), kinds):
        z = _toy_map(kind)
        ax.imshow(z, cmap="magma")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate(
            "",
            xy=(0.95, 0.5),
            xytext=(0.05, 0.5),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.4, color="white"),
        )
        ax.annotate(
            "",
            xy=(0.5, 0.95),
            xytext=(0.5, 0.05),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.4, color="white"),
        )
        ax.text(0.03, 0.05, "x", color="white", transform=ax.transAxes, fontsize=10)
        ax.text(0.52, 0.95, "y", color="white", transform=ax.transAxes, fontsize=10)

    fig.suptitle(
        "How to Read Spatial Skew (Rollout Attention)\n"
        "Sign = direction, |value| = bias strength",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_rollout_arrow_plot(df: pd.DataFrame, ff8: List[str], out_path: Path):
    d = df[df["dataset"].isin(ff8)].copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    x_only = d["rollout_spatial_skew_x_mean_pissa_only"].to_numpy(dtype=float)
    y_only = d["rollout_spatial_skew_y_mean_pissa_only"].to_numpy(dtype=float)
    x_plus = d["rollout_spatial_skew_x_mean_pissa_plus_ln"].to_numpy(dtype=float)
    y_plus = d["rollout_spatial_skew_y_mean_pissa_plus_ln"].to_numpy(dtype=float)

    ax.scatter(x_only, y_only, s=62, color="#1f77b4", label="pissa_only", alpha=0.9)
    ax.scatter(x_plus, y_plus, s=70, marker="s", color="#ff7f0e", label="pissa_plus_ln", alpha=0.95)

    for _, row in d.iterrows():
        x1 = float(row["rollout_spatial_skew_x_mean_pissa_only"])
        y1 = float(row["rollout_spatial_skew_y_mean_pissa_only"])
        x2 = float(row["rollout_spatial_skew_x_mean_pissa_plus_ln"])
        y2 = float(row["rollout_spatial_skew_y_mean_pissa_plus_ln"])
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.1, alpha=0.8),
        )
        ax.text(x2 + 0.004, y2 + 0.002, str(row["dataset"]), fontsize=8)

    ax.axhline(0, color="black", lw=0.9, alpha=0.6)
    ax.axvline(0, color="black", lw=0.9, alpha=0.6)
    ax.grid(alpha=0.25)
    ax.set_xlabel("rollout skew x  (left - | + right)")
    ax.set_ylabel("rollout skew y  (top - | + bottom)")
    ax.set_title("Rollout Spatial Skew: mode shift by dataset")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_abs_delta_heatmap(df: pd.DataFrame, ff8: List[str], out_path: Path):
    d = df[df["dataset"].isin(ff8)].copy()
    if d.empty:
        return
    d = d.set_index("dataset")
    mat = pd.DataFrame(
        {
            "abs_score_skew_all_delta": d["score_skew_all_pissa_plus_ln"].abs() - d["score_skew_all_pissa_only"].abs(),
            "abs_rollout_x_delta": d["rollout_spatial_skew_x_mean_pissa_plus_ln"].abs()
            - d["rollout_spatial_skew_x_mean_pissa_only"].abs(),
            "abs_rollout_y_delta": d["rollout_spatial_skew_y_mean_pissa_plus_ln"].abs()
            - d["rollout_spatial_skew_y_mean_pissa_only"].abs(),
        }
    )
    mat = mat.loc[[x for x in ff8 if x in mat.index]]
    arr = mat.to_numpy(dtype=float)
    vmax = max(0.02, float(np.nanmax(np.abs(arr))))

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title("Abs-skew delta (pissa_plus_ln - pissa_only)\nnegative = less skew (better)")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:+.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("delta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_dataset_before_after_grid(df: pd.DataFrame, datasets: List[str], out_path: Path, metrics_df: pd.DataFrame | None = None):
    d = df[df["dataset"].isin(datasets)].copy()
    if d.empty:
        return

    # Keep dataset order stable.
    d["dataset"] = pd.Categorical(d["dataset"], categories=datasets, ordered=True)
    d = d.sort_values("dataset")
    metric_map = {}
    if metrics_df is not None and not metrics_df.empty:
        metric_map = {str(r["dataset"]): r for _, r in metrics_df.iterrows()}

    # Shared axes range so movement size is comparable across panels.
    vals = []
    for col in [
        "rollout_spatial_skew_x_mean_pissa_only",
        "rollout_spatial_skew_y_mean_pissa_only",
        "rollout_spatial_skew_x_mean_pissa_plus_ln",
        "rollout_spatial_skew_y_mean_pissa_plus_ln",
    ]:
        vals.extend(d[col].astype(float).tolist())
    lim = max(0.8, float(np.nanmax(np.abs(vals))) + 0.08)

    n = len(d)
    ncols = 4 if n >= 8 else min(3, n)
    nrows = int(np.ceil(n / max(1, ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.1 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, (_, row) in enumerate(d.iterrows()):
        ax = axes_flat[i]
        x1 = float(row["rollout_spatial_skew_x_mean_pissa_only"])
        y1 = float(row["rollout_spatial_skew_y_mean_pissa_only"])
        x2 = float(row["rollout_spatial_skew_x_mean_pissa_plus_ln"])
        y2 = float(row["rollout_spatial_skew_y_mean_pissa_plus_ln"])

        ax.axhline(0, color="black", lw=0.7, alpha=0.5)
        ax.axvline(0, color="black", lw=0.7, alpha=0.5)
        ax.grid(alpha=0.22)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(str(row["dataset"]), fontsize=10)

        ax.scatter([x1], [y1], s=42, color="#1f77b4", label="before", zorder=3)
        ax.scatter([x2], [y2], s=46, marker="s", color="#ff7f0e", label="after", zorder=3)
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
        )
        ax.text(x1 + 0.01, y1 + 0.01, "B", fontsize=8, color="#1f77b4")
        ax.text(x2 + 0.01, y2 + 0.01, "A", fontsize=8, color="#ff7f0e")
        ds = str(row["dataset"])
        if ds in metric_map:
            m = metric_map[ds]
            auc_b = float(m["video_auc_pissa_only"])
            auc_a = float(m["video_auc_pissa_plus_ln"])
            acc_b = float(m["video_acc_pissa_only"])
            acc_a = float(m["video_acc_pissa_plus_ln"])
            d_auc = float(m["delta_auc"])
            d_acc = float(m["delta_acc"])
            metric_txt = (
                f"AUC {auc_b:.4f}->{auc_a:.4f} ({d_auc:+.4f})\n"
                f"ACC {acc_b:.4f}->{acc_a:.4f} ({d_acc:+.4f})"
            )
            ax.text(
                0.03,
                0.04,
                metric_txt,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.0,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#666666", alpha=0.78, lw=0.5),
            )
        ax.set_xlabel("rollout-map skew x", fontsize=8)
        ax.set_ylabel("rollout-map skew y", fontsize=8)

    # Turn off unused panels.
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    # Global legend and title.
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="Before (pissa_only)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff7f0e", markersize=8, label="After (pissa_plus_ln)"),
        plt.Line2D([0], [0], color="#555555", lw=1.2, label="Shift direction"),
    ]
    fig.suptitle("Dataset-wise rollout skew before/after LN", fontsize=12.5, fontweight="bold", y=0.985)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=True,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.08,
        "Skew target: attention rollout map (class-token -> patch), metric: rollout_spatial_skew_x/y_mean",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.965])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Create intuitive skew interpretation figures.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("weights/mode_ablation_robustness/final_pissa_vs_pissa_ln/skew_comparison.csv"),
        help="Path to skew_comparison.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights/mode_ablation_robustness/final_pissa_vs_pissa_ln"),
        help="Output directory",
    )
    parser.add_argument(
        "--ff8-list",
        type=str,
        default=",".join(DEFAULT_FF8),
        help="Comma-separated FF8 dataset list order.",
    )
    parser.add_argument(
        "--pissa-only-summary",
        type=Path,
        default=Path("weights/dinov3-h+_PiSSA_Base/best_models/test_mode_summary.csv"),
        help="Path to pissa_only test_mode_summary.csv",
    )
    parser.add_argument(
        "--pissa-plus-ln-summary",
        type=Path,
        default=Path("weights/dinov3-h+_GenD_LN/best_models/test_mode_summary.csv"),
        help="Path to pissa_plus_ln test_mode_summary.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ff8 = _split_csv_list(args.ff8_list)
    _ensure_dir(args.output_dir)

    df = pd.read_csv(args.input_csv)
    metrics_df = None
    if args.pissa_only_summary.exists() and args.pissa_plus_ln_summary.exists():
        metrics_df = _load_auc_acc_delta(args.pissa_only_summary, args.pissa_plus_ln_summary)
    else:
        print(
            "warning: AUC/ACC summaries not found; panel-level AUC/ACC annotations will be skipped. "
            f"missing? only={args.pissa_only_summary.exists()} plus={args.pissa_plus_ln_summary.exists()}"
        )

    p1 = args.output_dir / "skew_axis_explainer.png"
    p2 = args.output_dir / "rollout_skew_xy_arrow_compare.png"
    p3 = args.output_dir / "abs_skew_delta_heatmap.png"
    p4 = args.output_dir / "rollout_skew_before_after_by_dataset.png"

    make_axis_explainer(p1)
    make_rollout_arrow_plot(df, ff8, p2)
    make_abs_delta_heatmap(df, ff8, p3)
    make_dataset_before_after_grid(df, ff8, p4, metrics_df=metrics_df)

    print(f"saved: {p1}")
    print(f"saved: {p2}")
    print(f"saved: {p3}")
    print(f"saved: {p4}")


if __name__ == "__main__":
    main()
