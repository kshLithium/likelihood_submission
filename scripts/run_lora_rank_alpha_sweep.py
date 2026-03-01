#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


SWEEP_ROOT = Path("weights/lora_rank_alpha_ln01_sweep")
EXPERIMENTS = [
    {"r": 1, "alpha": 2, "name": "r1_a2_ln01"},
    {"r": 2, "alpha": 4, "name": "r2_a4_ln01"},
    {"r": 4, "alpha": 4, "name": "r4_a4_ln01"},
    {"r": 4, "alpha": 8, "name": "r4_a8_ln01"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 4 LoRA (rank, alpha) experiments with LN LR scale fixed to 0.1."
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: likelihood_submission_ln)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands/env and exit without executing training/evaluation.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated experiment names to run (e.g. r1_a2_ln01,r2_a4_ln01).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when any experiment fails.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="3",
        help="CUDA_VISIBLE_DEVICES value for all runs (default: 3).",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="offline",
        choices=["offline", "disabled", "online"],
        help="W&B mode for training runs (default: offline).",
    )
    return parser.parse_args()


def read_base_lr(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return float(cfg.get("learning_rate", 1e-3))


def parse_mean_video_auc(summary_csv_path):
    if not summary_csv_path.exists():
        return None
    with open(summary_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("dataset", "")).strip().upper() != "MEAN":
                continue
            raw = row.get("video_auc")
            if raw in (None, "", "None"):
                return None
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def run_command(cmd, env, cwd):
    start = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, env=env, check=True)
    elapsed_min = (time.perf_counter() - start) / 60.0
    return elapsed_min


def build_markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                if col.endswith("_wall_time_min"):
                    values.append(f"{value:.2f}")
                elif col == "mean_video_auc":
                    values.append(f"{value:.4f}")
                else:
                    values.append(f"{value}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(repo_root, rows):
    summary_dir = repo_root / SWEEP_ROOT
    summary_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "r",
        "alpha",
        "base_lr",
        "ln_lr_scale",
        "effective_ln_lr",
        "mean_video_auc",
        "output_dir",
        "model_pt",
        "adapter_config",
        "test_summary_csv",
        "train_wall_time_min",
        "eval_wall_time_min",
    ]

    csv_path = summary_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = summary_dir / "summary.md"
    md_content = build_markdown_table(rows, columns)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content + "\n")

    print(f"[*] Summary CSV: {csv_path}")
    print(f"[*] Summary MD : {md_path}")
    print("\n" + md_content)


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config_path = repo_root / "config" / "config.yaml"
    train_script = repo_root / "train.py"

    base_lr = read_base_lr(config_path)
    ln_lr_scale = 0.1
    effective_ln_lr = base_lr * ln_lr_scale

    selected = set(x.strip() for x in args.only.split(",") if x.strip())
    experiments = [exp for exp in EXPERIMENTS if not selected or exp["name"] in selected]
    if not experiments:
        raise SystemExit("[Error] No experiments selected.")

    print("[*] Sweep configuration")
    print(f"    - repo_root: {repo_root}")
    print(f"    - base_lr: {base_lr}")
    print(f"    - ln_lr_scale: {ln_lr_scale}")
    print(f"    - effective_ln_lr: {effective_ln_lr}")
    print(f"    - cuda_visible_devices: {args.cuda_visible_devices}")
    print(f"    - wandb_mode: {args.wandb_mode}")
    print(f"    - experiments: {[exp['name'] for exp in experiments]}")

    rows = []
    for exp in experiments:
        r = int(exp["r"])
        alpha = int(exp["alpha"])
        name = exp["name"]
        output_dir = (repo_root / SWEEP_ROOT / name).resolve()
        best_models_dir = output_dir / "best_models"

        env_common = os.environ.copy()
        env_common["OUTPUT_DIR"] = str(output_dir)
        env_common["LORA_RANK"] = str(r)
        env_common["LORA_ALPHA"] = str(alpha)
        env_common["LAYERNORM_LR_SCALE"] = str(ln_lr_scale)
        env_common["WANDB_RUN_NAME"] = name
        env_common["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        env_common["WANDB_MODE"] = str(args.wandb_mode)

        train_cmd = [sys.executable, str(train_script)]
        eval_cmd = [sys.executable, str(train_script)]

        print(f"\n{'=' * 80}")
        print(f"[*] Experiment: {name} (r={r}, alpha={alpha})")
        print(f"    output_dir: {output_dir}")
        if args.dry_run:
            print("    [DRY-RUN] train:")
            print(
                "      CUDA_VISIBLE_DEVICES="
                + str(args.cuda_visible_devices)
                + " TEST_MODE=false "
                + " ".join(train_cmd)
            )
            print("    [DRY-RUN] eval:")
            print(
                "      CUDA_VISIBLE_DEVICES="
                + str(args.cuda_visible_devices)
                + " TEST_MODE=true  "
                + " ".join(eval_cmd)
            )
            rows.append(
                {
                    "r": r,
                    "alpha": alpha,
                    "base_lr": base_lr,
                    "ln_lr_scale": ln_lr_scale,
                    "effective_ln_lr": effective_ln_lr,
                    "mean_video_auc": None,
                    "output_dir": str(output_dir),
                    "model_pt": str(best_models_dir / "model.pt"),
                    "adapter_config": str(best_models_dir / "adapter_config.json"),
                    "test_summary_csv": str(best_models_dir / "test_mode_summary.csv"),
                    "train_wall_time_min": None,
                    "eval_wall_time_min": None,
                }
            )
            continue

        train_minutes = None
        eval_minutes = None
        error = None

        try:
            env_train = env_common.copy()
            env_train["TEST_MODE"] = "false"
            train_minutes = run_command(train_cmd, env_train, repo_root)

            env_eval = env_common.copy()
            env_eval["TEST_MODE"] = "true"
            eval_minutes = run_command(eval_cmd, env_eval, repo_root)
        except subprocess.CalledProcessError as exc:
            error = f"Command failed with exit code {exc.returncode}"
            print(f"[Error] {name}: {error}")

        summary_csv = best_models_dir / "test_mode_summary.csv"
        mean_video_auc = parse_mean_video_auc(summary_csv)
        if mean_video_auc is None:
            print(f"[Warn] Could not read MEAN video AUC from: {summary_csv}")

        row = {
            "r": r,
            "alpha": alpha,
            "base_lr": base_lr,
            "ln_lr_scale": ln_lr_scale,
            "effective_ln_lr": effective_ln_lr,
            "mean_video_auc": mean_video_auc,
            "output_dir": str(output_dir),
            "model_pt": str(best_models_dir / "model.pt"),
            "adapter_config": str(best_models_dir / "adapter_config.json"),
            "test_summary_csv": str(summary_csv),
            "train_wall_time_min": train_minutes,
            "eval_wall_time_min": eval_minutes,
        }
        if error is not None:
            row["mean_video_auc"] = None
        rows.append(row)

        if error is not None and args.stop_on_error:
            break

    if args.dry_run:
        print("\n[*] Dry-run completed.")
        return

    write_summary(repo_root, rows)


if __name__ == "__main__":
    main()
