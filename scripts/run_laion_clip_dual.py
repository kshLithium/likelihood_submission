#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path


MODEL_SPECS = {
    "h14": {
        "alias": "h14",
        "model_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "output_dir": "./weights/clip-h14_laion2b_s32b_b79k_pissa_LN-01",
    },
    "g14": {
        "alias": "g14",
        "model_id": "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        "output_dir": "./weights/clip-g14_laion2b_s12b_b42k_pissa_LN-01",
    },
}

MODEL_ORDER = ["h14", "g14"]
DEFAULT_BATCH_SEQUENCE = [192, 160, 128, 96, 64, 48, 32, 24]
OOM_KEYWORDS = [
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "outofmemoryerror",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LAION CLIP H14/g14 sequential experiments with OOM-aware batch fallback."
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: likelihood_submission_ln)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="h14,g14",
        help="Comma-separated model aliases to run: h14,g14 or all (default: h14,g14).",
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
        help="W&B mode for train/test runs (default: offline).",
    )
    parser.add_argument(
        "--batch-sequence",
        type=str,
        default="192,160,128,96,64,48,32,24",
        help="Batch-size fallback sequence (default: 192,160,128,96,64,48,32,24).",
    )
    parser.add_argument(
        "--max_batch_trials",
        type=int,
        default=0,
        help="Limit number of batch trials from sequence (0 means all).",
    )
    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=20,
        help="analysis_clip sample_per_class (default: 20).",
    )
    parser.add_argument(
        "--analysis_seed",
        type=int,
        default=42,
        help="analysis_clip seed (default: 42).",
    )
    parser.add_argument(
        "--analysis_num_workers",
        type=int,
        default=8,
        help="analysis_clip num_workers (default: 8).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop all experiments immediately when one model fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands/env and exit without executing.",
    )
    return parser.parse_args()


def parse_models(raw_models):
    tokens = [x.strip().lower() for x in str(raw_models).split(",") if x.strip()]
    if not tokens or "all" in tokens:
        tokens = MODEL_ORDER[:]

    invalid = [x for x in tokens if x not in MODEL_SPECS]
    if invalid:
        raise ValueError(f"Unknown model aliases: {invalid}. Valid: {list(MODEL_SPECS)}")

    # Preserve canonical order
    seen = set()
    ordered = []
    for alias in MODEL_ORDER:
        if alias in tokens and alias not in seen:
            ordered.append(alias)
            seen.add(alias)
    return ordered


def parse_batch_sequence(raw_seq, max_trials):
    if raw_seq is None or str(raw_seq).strip() == "":
        sequence = DEFAULT_BATCH_SEQUENCE[:]
    else:
        vals = []
        for tok in str(raw_seq).split(","):
            tok = tok.strip()
            if not tok:
                continue
            vals.append(int(tok))
        if not vals:
            raise ValueError("batch-sequence is empty after parsing.")
        sequence = vals

    # Keep order, remove duplicates
    unique = []
    seen = set()
    for b in sequence:
        if b <= 0:
            raise ValueError(f"Invalid batch size in sequence: {b}")
        if b not in seen:
            unique.append(b)
            seen.add(b)

    if max_trials > 0:
        unique = unique[:max_trials]
    if not unique:
        raise ValueError("No batch trials available after max_batch_trials filtering.")
    return unique


def resolve_output_dir(repo_root, output_dir_str):
    p = Path(output_dir_str)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def run_command_stream(cmd, env, cwd, oom_keywords):
    keywords = [k.lower() for k in oom_keywords]
    tail = []
    oom_detected = False

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lower = line.lower()
        if any(k in lower for k in keywords):
            oom_detected = True
        tail.append(line.rstrip("\n"))
        if len(tail) > 120:
            tail = tail[-120:]

    returncode = proc.wait()
    elapsed_min = (time.perf_counter() - start) / 60.0
    return returncode, elapsed_min, oom_detected, "\n".join(tail)


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


def build_markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if value is None:
                vals.append("")
            elif isinstance(value, float):
                if col.endswith("_minutes"):
                    vals.append(f"{value:.2f}")
                elif col == "mean_video_auc":
                    vals.append(f"{value:.6f}")
                else:
                    vals.append(str(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(summary_dir, rows):
    summary_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_id",
        "alias",
        "status",
        "batch_size_used",
        "train_minutes",
        "eval_minutes",
        "analysis_minutes",
        "mean_video_auc",
        "test_summary_csv",
        "analysis_report_md",
        "output_dir",
        "error_message",
    ]

    csv_path = summary_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = summary_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown_table(rows, columns) + "\n")

    print(f"[*] Summary CSV: {csv_path}")
    print(f"[*] Summary MD : {md_path}")


def update_config_to_last_model(config_path, model_id, output_dir):
    text = config_path.read_text(encoding="utf-8")
    replacements = [
        (r"(?m)^(\s*clip:\s*).*$", rf'\1true'),
        (r'(?m)^(\s*clip_model:\s*).*$', rf'\1"{model_id}"'),
        (r'(?m)^(\s*output_dir:\s*).*$', rf'\1"{output_dir}"'),
    ]

    changed = text
    for pattern, repl in replacements:
        changed, n = re.subn(pattern, repl, changed, count=1)
        if n != 1:
            raise RuntimeError(f"Failed to update config line for pattern: {pattern}")

    config_path.write_text(changed, encoding="utf-8")
    print(f"[*] Updated config to last model: {config_path}")


def model_run_name(alias):
    return f"laion_clip_{alias}_pissa_ln01"


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    train_script = repo_root / "train.py"
    analysis_script = repo_root / "scripts" / "analyze_clip_best_models.py"
    config_path = repo_root / "config" / "config.yaml"

    aliases = parse_models(args.models)
    batch_trials = parse_batch_sequence(args.batch_sequence, args.max_batch_trials)
    summary_dir = (repo_root / "weights" / "laion_clip_dual_summary").resolve()

    print("[*] LAION dual run configuration")
    print(f"    - repo_root: {repo_root}")
    print(f"    - models: {aliases}")
    print(f"    - gpu: {args.cuda_visible_devices}")
    print(f"    - batch_trials: {batch_trials}")
    print(f"    - wandb_mode: {args.wandb_mode}")
    print(f"    - dry_run: {args.dry_run}")

    rows = []
    for alias in aliases:
        spec = MODEL_SPECS[alias]
        output_dir = resolve_output_dir(repo_root, spec["output_dir"])
        best_models_dir = output_dir / "best_models"
        analysis_out_dir = best_models_dir / "analysis_paper"
        summary_csv = best_models_dir / "test_mode_summary.csv"
        report_md = analysis_out_dir / "report.md"

        row = {
            "model_id": spec["model_id"],
            "alias": alias,
            "status": "failed_other",
            "batch_size_used": None,
            "train_minutes": None,
            "eval_minutes": None,
            "analysis_minutes": None,
            "mean_video_auc": None,
            "test_summary_csv": str(summary_csv.resolve()),
            "analysis_report_md": str(report_md.resolve()),
            "output_dir": str(output_dir),
            "error_message": "",
        }

        base_env = os.environ.copy()
        base_env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        base_env["WANDB_MODE"] = str(args.wandb_mode)
        base_env["USE_CLIP"] = "true"
        base_env["CLIP_MODEL"] = spec["model_id"]
        base_env["OUTPUT_DIR"] = str(output_dir)
        base_env["WANDB_RUN_NAME"] = model_run_name(alias)

        print("\n" + "=" * 88)
        print(f"[*] Model: {alias}")
        print(f"    model_id: {spec['model_id']}")
        print(f"    output_dir: {output_dir}")

        if args.dry_run:
            print("    [DRY-RUN] train/test/analysis commands")
            for b in batch_trials:
                print(
                    f"      BATCH_SIZE={b} TEST_MODE=false {sys.executable} {train_script}"
                )
            print(f"      BATCH_SIZE=<selected> TEST_MODE=true  {sys.executable} {train_script}")
            print(
                "      "
                + f"{sys.executable} {analysis_script} "
                + f"--weights_root {best_models_dir} --output_dir {analysis_out_dir} "
                + f"--sample_per_class {args.sample_per_class} --seed {args.analysis_seed} "
                + f"--device cuda --num_workers {args.analysis_num_workers}"
            )
            row["status"] = "ok"
            rows.append(row)
            continue

        train_ok = False
        failed_non_oom = False
        oom_seen = False

        for batch in batch_trials:
            print(f"\n    [train] Trying batch_size={batch}")
            env_train = base_env.copy()
            env_train["TEST_MODE"] = "false"
            env_train["BATCH_SIZE"] = str(batch)
            cmd_train = [sys.executable, str(train_script)]
            rc, minutes, oom_detected, tail = run_command_stream(
                cmd_train, env_train, repo_root, OOM_KEYWORDS
            )
            if rc == 0:
                train_ok = True
                row["batch_size_used"] = batch
                row["train_minutes"] = minutes
                row["error_message"] = ""
                print(f"    [train] success with batch_size={batch} ({minutes:.2f} min)")
                break

            if oom_detected:
                oom_seen = True
                row["error_message"] = f"OOM at batch_size={batch}"
                print(f"    [train] OOM detected at batch_size={batch}, trying next fallback.")
                continue

            failed_non_oom = True
            row["error_message"] = (
                f"Train failed (non-OOM) at batch_size={batch}, returncode={rc}\n{tail}"
            )
            print(f"    [train] failed (non-OOM), returncode={rc}")
            break

        if not train_ok:
            if failed_non_oom:
                row["status"] = "failed_other"
            elif oom_seen:
                row["status"] = "failed_oom"
            else:
                row["status"] = "failed_other"
                row["error_message"] = "Train failed before OOM/non-OOM classification."
            rows.append(row)
            if args.stop_on_error:
                break
            continue

        # Evaluation (TEST_MODE=true)
        print("\n    [eval] Running TEST_MODE=true")
        env_eval = base_env.copy()
        env_eval["TEST_MODE"] = "true"
        env_eval["BATCH_SIZE"] = str(row["batch_size_used"])
        cmd_eval = [sys.executable, str(train_script)]
        rc, minutes, _oom_eval, tail = run_command_stream(cmd_eval, env_eval, repo_root, OOM_KEYWORDS)
        row["eval_minutes"] = minutes
        if rc != 0:
            row["status"] = "failed_other"
            row["error_message"] = f"Eval failed, returncode={rc}\n{tail}"
            rows.append(row)
            if args.stop_on_error:
                break
            continue

        row["mean_video_auc"] = parse_mean_video_auc(summary_csv)
        if row["mean_video_auc"] is None:
            row["error_message"] = "Eval finished but could not parse MEAN video_auc from summary CSV."

        # Analysis report generation
        print("\n    [analysis] Generating analysis_paper")
        env_analysis = base_env.copy()
        cmd_analysis = [
            sys.executable,
            str(analysis_script),
            "--weights_root",
            str(best_models_dir),
            "--output_dir",
            str(analysis_out_dir),
            "--sample_per_class",
            str(args.sample_per_class),
            "--seed",
            str(args.analysis_seed),
            "--device",
            "cuda",
            "--num_workers",
            str(args.analysis_num_workers),
        ]
        rc, minutes, _oom_analysis, tail = run_command_stream(
            cmd_analysis, env_analysis, repo_root, OOM_KEYWORDS
        )
        row["analysis_minutes"] = minutes
        if rc != 0:
            row["status"] = "failed_other"
            row["error_message"] = f"Analysis failed, returncode={rc}\n{tail}"
            rows.append(row)
            if args.stop_on_error:
                break
            continue

        row["status"] = "ok"
        rows.append(row)

    if args.dry_run:
        print("\n[*] Dry-run completed. No commands executed and no files were written.")
        return

    # User preference: leave config at the last (g14) model setting.
    final = MODEL_SPECS["g14"]
    update_config_to_last_model(config_path, final["model_id"], final["output_dir"])
    write_summary(summary_dir, rows)


if __name__ == "__main__":
    main()
