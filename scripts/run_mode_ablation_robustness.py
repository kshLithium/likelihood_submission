#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MODE_SPECS: Dict[str, Dict[str, str]] = {
    "pissa_only": {
        "use_peft": "true",
        "use_layernorm_tuning": "false",
        "output_dir": "./weights/dinov3-h+_PiSSA_Base",
    },
    "ln_only": {
        "use_peft": "false",
        "use_layernorm_tuning": "true",
        "output_dir": "./weights/dinov3-h+_LN_only",
    },
    "pissa_plus_ln": {
        "use_peft": "true",
        "use_layernorm_tuning": "true",
        "output_dir": "./weights/dinov3-h+_GenD_LN",
    },
}

MODE_ORDER = ["pissa_only", "ln_only", "pissa_plus_ln"]
DEFAULT_BATCH_SEQUENCE = [192, 160, 128, 96, 64]
OOM_KEYWORDS = [
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "outofmemoryerror",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 3-way mode ablation (PiSSA/LN/PiSSA+LN) with robustness analysis."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: likelihood_submission_ln).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="pissa_only,ln_only,pissa_plus_ln",
        help="Comma-separated modes to run.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force training even when best_models/model.pt already exists.",
    )
    parser.add_argument(
        "--batch-sequence",
        type=str,
        default="192,160,128,96,64",
        help="Batch-size fallback sequence for training.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=20,
        help="analysis_clip sample_per_class.",
    )
    parser.add_argument(
        "--datasets-preset",
        type=str,
        default="ff8_ood4",
        choices=["ff8", "ff8_ood4"],
        help="TEST_DATASET_PRESET for TEST_MODE evaluation.",
    )
    parser.add_argument(
        "--analysis-seed",
        type=int,
        default=42,
        help="analysis_clip seed.",
    )
    parser.add_argument(
        "--analysis-num-workers",
        type=int,
        default=8,
        help="analysis_clip num_workers.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="offline",
        choices=["offline", "disabled", "online"],
        help="W&B mode for train/eval runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands and exit without executing.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop remaining modes when any mode fails.",
    )
    return parser.parse_args()


def parse_modes(raw: str) -> List[str]:
    tokens = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not tokens:
        tokens = MODE_ORDER[:]
    invalid = [x for x in tokens if x not in MODE_SPECS]
    if invalid:
        raise ValueError(f"Unknown modes: {invalid}. Valid: {list(MODE_SPECS)}")
    out: List[str] = []
    for m in MODE_ORDER:
        if m in tokens and m not in out:
            out.append(m)
    return out


def parse_batch_sequence(raw: str) -> List[int]:
    if raw is None or str(raw).strip() == "":
        return DEFAULT_BATCH_SEQUENCE[:]
    vals = []
    seen = set()
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        b = int(tok)
        if b <= 0:
            raise ValueError(f"Invalid batch size: {b}")
        if b not in seen:
            seen.add(b)
            vals.append(b)
    if not vals:
        raise ValueError("batch-sequence is empty")
    return vals


def run_command_stream(
    cmd: List[str],
    env: Dict[str, str],
    cwd: Path,
    oom_keywords: Optional[List[str]] = None,
) -> Tuple[int, float, bool, str]:
    keywords = [k.lower() for k in (oom_keywords or [])]
    tail: List[str] = []
    oom_detected = False
    start = time.perf_counter()

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
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

    rc = proc.wait()
    elapsed_min = (time.perf_counter() - start) / 60.0
    return rc, elapsed_min, oom_detected, "\n".join(tail)


def parse_mean_video_auc(summary_csv: Path) -> Optional[float]:
    if not summary_csv.exists():
        return None
    with open(summary_csv, "r", encoding="utf-8", newline="") as f:
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


def build_markdown_table(rows: List[Dict[str, object]], columns: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        vals = []
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                if c.endswith("_minutes"):
                    vals.append(f"{v:.2f}")
                elif c == "mean_video_auc":
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            elif v is None:
                vals.append("")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(summary_root: Path, rows: List[Dict[str, object]]):
    summary_root.mkdir(parents=True, exist_ok=True)
    columns = [
        "mode",
        "status",
        "trained",
        "batch_size_used",
        "train_minutes",
        "eval_minutes",
        "analysis_minutes",
        "mean_video_auc",
        "output_dir",
        "model_pt",
        "test_summary_csv",
        "analysis_report_md",
        "error_message",
    ]

    csv_path = summary_root / "run_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = summary_root / "run_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown_table(rows, columns) + "\n")

    print(f"[*] Summary CSV: {csv_path}")
    print(f"[*] Summary MD : {md_path}")


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    train_script = repo_root / "train.py"
    analysis_script = repo_root / "scripts" / "analyze_clip_best_models.py"
    modes = parse_modes(args.modes)
    batch_sequence = parse_batch_sequence(args.batch_sequence)
    summary_root = (repo_root / "weights" / "mode_ablation_robustness").resolve()

    print("[*] Mode ablation configuration")
    print(f"    - repo_root: {repo_root}")
    print(f"    - modes: {modes}")
    print(f"    - batch_sequence: {batch_sequence}")
    print(f"    - datasets_preset: {args.datasets_preset}")
    print(f"    - sample_per_class: {args.sample_per_class}")
    print("    - CUDA_VISIBLE_DEVICES: 2,3")
    print(f"    - force_retrain: {args.force_retrain}")
    print(f"    - dry_run: {args.dry_run}")

    rows: List[Dict[str, object]] = []
    for mode in modes:
        spec = MODE_SPECS[mode]
        output_dir = (repo_root / spec["output_dir"]).resolve()
        best_models_dir = output_dir / "best_models"
        model_pt = best_models_dir / "model.pt"
        test_summary_csv = best_models_dir / "test_mode_summary.csv"
        analysis_out_dir = best_models_dir / "analysis_paper"
        analysis_report_md = analysis_out_dir / "report.md"

        row: Dict[str, object] = {
            "mode": mode,
            "status": "failed_other",
            "trained": False,
            "batch_size_used": None,
            "train_minutes": None,
            "eval_minutes": None,
            "analysis_minutes": None,
            "mean_video_auc": None,
            "output_dir": str(output_dir),
            "model_pt": str(model_pt),
            "test_summary_csv": str(test_summary_csv),
            "analysis_report_md": str(analysis_report_md),
            "error_message": "",
        }

        env_common = os.environ.copy()
        env_common["CUDA_VISIBLE_DEVICES"] = "2,3"
        env_common["WANDB_MODE"] = str(args.wandb_mode)
        env_common["USE_CLIP"] = "false"
        env_common["OUTPUT_DIR"] = str(output_dir)
        env_common["USE_PEFT"] = str(spec["use_peft"])
        env_common["USE_LAYERNORM_TUNING"] = str(spec["use_layernorm_tuning"])
        env_common["TEST_DATASET_PRESET"] = str(args.datasets_preset)
        env_common["WANDB_RUN_NAME"] = f"mode_{mode}"

        print("\n" + "=" * 92)
        print(f"[*] Mode: {mode}")
        print(f"    - output_dir: {output_dir}")
        print(f"    - use_peft: {spec['use_peft']}")
        print(f"    - use_layernorm_tuning: {spec['use_layernorm_tuning']}")

        need_train = args.force_retrain or (not model_pt.exists())
        if args.dry_run:
            print(f"    [DRY-RUN] train_required={need_train}")
            if need_train:
                for b in batch_sequence:
                    print(f"      BATCH_SIZE={b} TEST_MODE=false {sys.executable} {train_script}")
            print(f"      TEST_MODE=true  {sys.executable} {train_script}")
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

        if need_train:
            print("    [train] model.pt 없음 또는 force-retrain=true -> 학습 수행")
            trained_ok = False
            fail_non_oom = False
            for batch in batch_sequence:
                print(f"    [train] trying batch_size={batch}")
                env_train = env_common.copy()
                env_train["TEST_MODE"] = "false"
                env_train["BATCH_SIZE"] = str(batch)
                rc, minutes, oom, tail = run_command_stream(
                    [sys.executable, str(train_script)],
                    env_train,
                    repo_root,
                    OOM_KEYWORDS,
                )
                if rc == 0:
                    row["trained"] = True
                    row["batch_size_used"] = batch
                    row["train_minutes"] = minutes
                    trained_ok = True
                    break
                if oom:
                    row["error_message"] = f"OOM at batch_size={batch}"
                    continue
                row["error_message"] = f"Train failed (returncode={rc})\n{tail}"
                fail_non_oom = True
                break

            if not trained_ok:
                row["status"] = "failed_other" if fail_non_oom else "failed_oom"
                rows.append(row)
                if args.stop_on_error:
                    break
                continue
        else:
            print("    [train] 기존 model.pt 존재 -> 학습 스킵")

        print("    [eval] TEST_MODE=true 실행")
        env_eval = env_common.copy()
        env_eval["TEST_MODE"] = "true"
        if row["batch_size_used"] is not None:
            env_eval["BATCH_SIZE"] = str(row["batch_size_used"])
        rc, minutes, _oom, tail = run_command_stream(
            [sys.executable, str(train_script)],
            env_eval,
            repo_root,
            OOM_KEYWORDS,
        )
        row["eval_minutes"] = minutes
        if rc != 0:
            row["status"] = "failed_other"
            row["error_message"] = f"Eval failed (returncode={rc})\n{tail}"
            rows.append(row)
            if args.stop_on_error:
                break
            continue

        row["mean_video_auc"] = parse_mean_video_auc(test_summary_csv)

        print("    [analysis] analysis_clip 실행")
        env_analysis = env_common.copy()
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
        rc, minutes, _oom, tail = run_command_stream(cmd_analysis, env_analysis, repo_root, OOM_KEYWORDS)
        row["analysis_minutes"] = minutes
        if rc != 0:
            row["status"] = "failed_other"
            row["error_message"] = f"Analysis failed (returncode={rc})\n{tail}"
            rows.append(row)
            if args.stop_on_error:
                break
            continue

        row["status"] = "ok"
        rows.append(row)

    if args.dry_run:
        print("[*] Dry-run completed.")
        return

    write_summary(summary_root, rows)


if __name__ == "__main__":
    main()

