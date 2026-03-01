#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODE_OUTPUTS = {
    "pissa_only": "./weights/dinov3-h+_PiSSA_Base",
    "ln_only": "./weights/dinov3-h+_LN_only",
    "pissa_plus_ln": "./weights/dinov3-h+_GenD_LN",
}
DEFAULT_MODES = ["pissa_only", "ln_only", "pissa_plus_ln"]
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
DEFAULT_OOD4 = ["Celeb-DF-v2", "DFDC", "DFDCP", "UADFV"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare mode robustness metrics and generate final figures/report."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: likelihood_submission_ln).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("weights/mode_ablation_robustness"),
        help="Root directory for run_summary and final outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for final comparison artifacts (default: <root>/final).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="pissa_only,ln_only,pissa_plus_ln",
        help="Comma-separated mode list.",
    )
    parser.add_argument(
        "--ff8-list",
        type=str,
        default=",".join(DEFAULT_FF8),
        help="Comma-separated FF8 dataset names.",
    )
    parser.add_argument(
        "--ood4-list",
        type=str,
        default=",".join(DEFAULT_OOD4),
        help="Comma-separated OOD4 dataset names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print plan without generating outputs.",
    )
    return parser.parse_args()


def _split_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _safe_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _safe_std(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.std(ddof=0))


def _cvar25(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    arr = np.sort(arr)
    k = max(1, int(math.ceil(arr.size * 0.25)))
    return float(arr[:k].mean())


def _load_output_map(repo_root: Path, modes: List[str], run_summary_csv: Path) -> Dict[str, Path]:
    mode_to_output: Dict[str, Path] = {}
    if run_summary_csv.exists():
        df = pd.read_csv(run_summary_csv)
        for _, row in df.iterrows():
            mode = str(row.get("mode", "")).strip()
            output_dir = str(row.get("output_dir", "")).strip()
            if mode and output_dir:
                mode_to_output[mode] = Path(output_dir).resolve()

    for mode in modes:
        if mode in mode_to_output:
            continue
        if mode not in DEFAULT_MODE_OUTPUTS:
            continue
        mode_to_output[mode] = (repo_root / DEFAULT_MODE_OUTPUTS[mode]).resolve()
    return mode_to_output


def _load_mode_auc_map(summary_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(summary_csv)
    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        ds = str(row.get("dataset", "")).strip()
        if not ds or ds.upper() == "MEAN":
            continue
        val = row.get("video_auc", np.nan)
        try:
            fval = float(val)
        except Exception:
            continue
        if np.isfinite(fval):
            out[ds] = fval
    return out


def _ordered_union(dataset_maps: Dict[str, Dict[str, float]], ff8: List[str], ood4: List[str]) -> List[str]:
    known = ff8 + [x for x in ood4 if x not in ff8]
    seen = set(known)
    extra = []
    for _, auc_map in dataset_maps.items():
        for ds in auc_map:
            if ds not in seen:
                seen.add(ds)
                extra.append(ds)
    return known + sorted(extra)


def _plot_auc_by_dataset_grouped(
    out_path: Path,
    dataset_order: List[str],
    modes: List[str],
    dataset_maps: Dict[str, Dict[str, float]],
):
    fig, ax = plt.subplots(figsize=(max(10, len(dataset_order) * 0.7), 5))
    x = np.arange(len(dataset_order), dtype=np.float32)
    width = 0.8 / max(1, len(modes))

    for i, mode in enumerate(modes):
        vals = [dataset_maps.get(mode, {}).get(ds, np.nan) for ds in dataset_order]
        pos = x - 0.4 + (i + 0.5) * width
        ax.bar(pos, vals, width=width, label=mode, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_order, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("video AUC")
    ax.set_title("AUC by Dataset (Mode Comparison)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_auc_heatmap(
    out_path: Path,
    dataset_order: List[str],
    modes: List[str],
    dataset_maps: Dict[str, Dict[str, float]],
):
    mat = np.full((len(dataset_order), len(modes)), np.nan, dtype=np.float32)
    for i, ds in enumerate(dataset_order):
        for j, mode in enumerate(modes):
            v = dataset_maps.get(mode, {}).get(ds, np.nan)
            mat[i, j] = np.nan if not np.isfinite(v) else float(v)

    fig, ax = plt.subplots(figsize=(max(6, len(modes) * 1.6), max(5, len(dataset_order) * 0.38)))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_yticks(np.arange(len(dataset_order)))
    ax.set_yticklabels(dataset_order)
    ax.set_title("Dataset x Mode AUC Heatmap")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("video AUC")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_robustness_frontier(out_path: Path, metrics_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in metrics_df.iterrows():
        x = float(row["mean_auc_all"])
        y = float(row["worst_auc_all"])
        mode = str(row["mode"])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        ax.scatter([x], [y], s=80)
        ax.text(x + 0.001, y + 0.001, mode, fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("mean_auc_all")
    ax.set_ylabel("worst_auc_all")
    ax.set_title("Robustness Frontier")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ff8_vs_ood4_boxplot(
    out_path: Path,
    modes: List[str],
    dataset_maps: Dict[str, Dict[str, float]],
    ff8: List[str],
    ood4: List[str],
):
    fig, ax = plt.subplots(figsize=(max(7, len(modes) * 2.6), 5))
    positions = []
    labels = []
    series = []
    pos = 1
    for mode in modes:
        ff_vals = [dataset_maps.get(mode, {}).get(ds, np.nan) for ds in ff8]
        ff_vals = [x for x in ff_vals if np.isfinite(x)]
        ood_vals = [dataset_maps.get(mode, {}).get(ds, np.nan) for ds in ood4]
        ood_vals = [x for x in ood_vals if np.isfinite(x)]

        if ff_vals:
            series.append(ff_vals)
            positions.append(pos)
            labels.append(f"{mode}\nFF8")
        pos += 1
        if ood_vals:
            series.append(ood_vals)
            positions.append(pos)
            labels.append(f"{mode}\nOOD4")
        pos += 1

    if series:
        ax.boxplot(series, positions=positions, widths=0.7, patch_artist=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("video AUC")
    ax.set_title("FF8 vs OOD4 AUC Distribution by Mode")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _fmt(v: object) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:.6f}"
    return str(v)


def _winner_report(metrics_df: pd.DataFrame) -> str:
    if metrics_df.empty:
        return "비교 가능한 모드 지표가 없습니다."

    eps = 1e-12
    df = metrics_df.copy()
    df = df[np.isfinite(df["robustness_ratio"].astype(float))]
    if df.empty:
        return "robustness_ratio 계산 가능한 모드가 없습니다."

    best_ratio = float(df["robustness_ratio"].max())
    best_cvar = float(df["cvar25_auc"].max())

    pissa_row = metrics_df[metrics_df["mode"] == "pissa_plus_ln"]
    if not pissa_row.empty:
        pr = float(pissa_row.iloc[0]["robustness_ratio"])
        pc = float(pissa_row.iloc[0]["cvar25_auc"])
        if np.isfinite(pr) and np.isfinite(pc) and pr >= best_ratio - eps and pc >= best_cvar - eps:
            return "PiSSA+LN이 robustness_ratio와 cvar25_auc 모두 1위여서 분포 차이 강건성 우위로 판정됩니다."

    winner = df.sort_values(["robustness_ratio", "cvar25_auc"], ascending=False).iloc[0]
    return (
        f"{winner['mode']}가 robustness_ratio={winner['robustness_ratio']:.6f}, "
        f"cvar25_auc={winner['cvar25_auc']:.6f}로 강건성 우위입니다."
    )


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    root_dir = (repo_root / args.root).resolve() if not args.root.is_absolute() else args.root.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = root_dir / "final"
    elif not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = _split_csv_list(args.modes) or DEFAULT_MODES[:]
    ff8 = _split_csv_list(args.ff8_list)
    ood4 = _split_csv_list(args.ood4_list)
    run_summary_csv = root_dir / "run_summary.csv"
    mode_to_output = _load_output_map(repo_root, modes, run_summary_csv)

    print("[*] Compare robustness configuration")
    print(f"    - root: {root_dir}")
    print(f"    - output_dir: {output_dir}")
    print(f"    - modes: {modes}")
    print(f"    - ff8: {ff8}")
    print(f"    - ood4: {ood4}")

    dataset_maps: Dict[str, Dict[str, float]] = {}
    metrics_rows: List[Dict[str, object]] = []
    for mode in modes:
        out_dir = mode_to_output.get(mode)
        if out_dir is None:
            print(f"[Warn] output_dir not found for mode={mode}, skip")
            continue

        summary_csv = out_dir / "best_models" / "test_mode_summary.csv"
        analysis_metrics_csv = out_dir / "best_models" / "analysis_paper" / "tables" / "dataset_metrics.csv"
        if not summary_csv.exists():
            print(f"[Warn] missing summary CSV: {summary_csv}")
            continue
        if not analysis_metrics_csv.exists():
            print(f"[Warn] missing analysis dataset_metrics CSV: {analysis_metrics_csv}")

        auc_map = _load_mode_auc_map(summary_csv)
        dataset_maps[mode] = auc_map

        all_vals = list(auc_map.values())
        ff_vals = [auc_map.get(ds, np.nan) for ds in ff8]
        ff_vals = [x for x in ff_vals if np.isfinite(x)]
        ood_vals = [auc_map.get(ds, np.nan) for ds in ood4]
        ood_vals = [x for x in ood_vals if np.isfinite(x)]

        mean_all = _safe_mean(all_vals)
        mean_ff8 = _safe_mean(ff_vals)
        mean_ood4 = _safe_mean(ood_vals)
        worst_all = float(np.min(all_vals)) if all_vals else float("nan")
        cvar25 = _cvar25(all_vals)
        std_all = _safe_std(all_vals)
        ood_drop = mean_ff8 - mean_ood4 if np.isfinite(mean_ff8) and np.isfinite(mean_ood4) else float("nan")
        ratio = mean_ood4 / mean_ff8 if np.isfinite(mean_ff8) and mean_ff8 > 0 and np.isfinite(mean_ood4) else float("nan")

        metrics_rows.append(
            {
                "mode": mode,
                "output_dir": str(out_dir),
                "summary_csv": str(summary_csv),
                "analysis_dataset_metrics_csv": str(analysis_metrics_csv),
                "mean_auc_all": mean_all,
                "mean_auc_ff8": mean_ff8,
                "mean_auc_ood4": mean_ood4,
                "worst_auc_all": worst_all,
                "cvar25_auc": cvar25,
                "std_auc": std_all,
                "ood_drop": ood_drop,
                "robustness_ratio": ratio,
                "num_datasets": len(all_vals),
            }
        )

    if args.dry_run:
        print("[*] Dry-run completed. No figure/report files were generated.")
        return

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = output_dir / "mode_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    dataset_order = _ordered_union(dataset_maps, ff8, ood4)
    details_rows = []
    for ds in dataset_order:
        group = "ff8" if ds in ff8 else ("ood4" if ds in ood4 else "other")
        for mode in modes:
            details_rows.append(
                {
                    "dataset": ds,
                    "group": group,
                    "mode": mode,
                    "video_auc": dataset_maps.get(mode, {}).get(ds, np.nan),
                }
            )
    details_df = pd.DataFrame(details_rows)
    details_csv = output_dir / "dataset_mode_auc.csv"
    details_df.to_csv(details_csv, index=False)

    fig_grouped = output_dir / "auc_by_dataset_grouped.png"
    fig_heatmap = output_dir / "auc_heatmap.png"
    fig_frontier = output_dir / "robustness_frontier.png"
    fig_box = output_dir / "ff8_vs_ood4_boxplot.png"
    _plot_auc_by_dataset_grouped(fig_grouped, dataset_order, modes, dataset_maps)
    _plot_auc_heatmap(fig_heatmap, dataset_order, modes, dataset_maps)
    _plot_robustness_frontier(fig_frontier, metrics_df)
    _plot_ff8_vs_ood4_boxplot(fig_box, modes, dataset_maps, ff8, ood4)

    report_md = output_dir / "report.md"
    lines = []
    lines.append("# Mode Robustness Comparison")
    lines.append("")
    lines.append(f"- Generated from root: `{root_dir}`")
    lines.append(f"- Modes: `{', '.join(modes)}`")
    lines.append(f"- FF8: `{', '.join(ff8)}`")
    lines.append(f"- OOD4: `{', '.join(ood4)}`")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append("")
    headers = [
        "mode",
        "mean_auc_all",
        "mean_auc_ff8",
        "mean_auc_ood4",
        "worst_auc_all",
        "cvar25_auc",
        "std_auc",
        "ood_drop",
        "robustness_ratio",
        "num_datasets",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in metrics_df.iterrows():
        vals = [_fmt(row.get(h)) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(_winner_report(metrics_df))
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Metrics CSV: `{metrics_csv}`")
    lines.append(f"- Dataset-Mode AUC CSV: `{details_csv}`")
    lines.append(f"- Figure: `{fig_grouped}`")
    lines.append(f"- Figure: `{fig_heatmap}`")
    lines.append(f"- Figure: `{fig_frontier}`")
    lines.append(f"- Figure: `{fig_box}`")

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[*] Saved: {metrics_csv}")
    print(f"[*] Saved: {details_csv}")
    print(f"[*] Saved: {fig_grouped}")
    print(f"[*] Saved: {fig_heatmap}")
    print(f"[*] Saved: {fig_frontier}")
    print(f"[*] Saved: {fig_box}")
    print(f"[*] Saved: {report_md}")


if __name__ == "__main__":
    main()

