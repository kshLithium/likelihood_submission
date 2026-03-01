import os
import sys
import json
import random
import builtins
from contextlib import nullcontext
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from datasets import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DefaultDataCollator
)

# -----------------------------------------------------------------------------
# [1] Import Project Modules
# -----------------------------------------------------------------------------
# 프로젝트 루트를 모듈 검색 경로에 추가
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import (
    # Config Variables
    OUTPUT_DIR,
    MODEL_ID, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    MULTITASK, SEED, NUM_WORKERS, USE_PEFT, DDP_TIMEOUT_SECONDS,
    LORA_RANK, LORA_ALPHA, WANDB_RUN_NAME,
    LAYERNORM_PATTERNS, LAYERNORM_LR_SCALE,
    USE_EFFORT, EFFORT_RANK,
    TEST_MODE, TEST_NUM_FRAMES, NUM_FRAMES_PER_VIDEO,
    USE_CLIP, CLIP_MODEL,
    # Utils
    is_main_process
)
from src.dataset import (
    load_all_data,
    load_all_data_video_level,
    DATASETS,
    TEST_DATASETS,
    RESOLVED_TEST_DATASET_PRESET,
    init_augs,
    apply_transforms_robust,
    validate_paths,
    validate_images_openable,
    balance_real_fake,
    balance_three_class
)
from src.models import (
    DINOv3ForClassification,
    build_peft_model,
    build_effort_model
)


# -----------------------------------------------------------------------------
# [2] Helper Functions
# -----------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """
    Validation 단계에서 정확도(Accuracy)와 AUC를 계산합니다.
    (training.py 원본 로직 유지)
    """
    logits, labels = eval_pred

    # Tuple로 들어오는 경우 처리 (HuggingFace Trainer 특성)
    if isinstance(logits, tuple):
        logits = logits[0]

    if MULTITASK:
        # Multi-class (3 classes)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = probs.argmax(axis=-1).reshape(-1)
        acc = accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, probs, multi_class="ovr")
        except Exception:
            auc = 0.5
    else:
        # Binary Classification (Real vs Fake)
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs > 0.5).astype(int).reshape(-1)
        acc = accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.5

    return {"accuracy": acc, "auc": auc}


def _is_layernorm_param(param_name):
    lname = param_name.lower()
    return any(pattern in lname for pattern in LAYERNORM_PATTERNS)


def _build_optimizer(model, training_args):
    if LAYERNORM_LR_SCALE <= 0:
        raise ValueError(f"layernorm_lr_scale must be > 0, got {LAYERNORM_LR_SCALE}")

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters found for optimizer setup.")

    ln_params = [param for name, param in trainable if _is_layernorm_param(name)]
    other_params = [param for name, param in trainable if not _is_layernorm_param(name)]

    ln_lr = LEARNING_RATE * LAYERNORM_LR_SCALE
    optimizer_grouped_parameters = []
    if other_params:
        optimizer_grouped_parameters.append({"params": other_params, "lr": LEARNING_RATE})
    if ln_params:
        optimizer_grouped_parameters.append({"params": ln_params, "lr": ln_lr})

    ln_count = sum(p.numel() for p in ln_params)
    other_count = sum(p.numel() for p in other_params)
    print("[*] Optimizer LR groups:")
    print(f"    - base_lr: {LEARNING_RATE} (params: {other_count})")
    print(
        f"    - layernorm_lr: {ln_lr} "
        f"(scale: {LAYERNORM_LR_SCALE}, params: {ln_count})"
    )

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )
    return optimizer


# -----------------------------------------------------------------------------
# [3] Shared evaluation helper (used by callback & test mode)
# -----------------------------------------------------------------------------
def _apply_val_transform(batch):
    return apply_transforms_robust(batch, is_train=False)


def _evaluate_single_dataset(model, ds_conf, device, num_frames, batch_size, num_workers):
    """
    Evaluate a single test dataset. Returns dict with video_auc, video_acc, frame_auc, etc.
    Video-level aggregation: mean of frame probabilities.
    """
    ds_name = ds_conf["name"]
    video_data = load_all_data_video_level([ds_conf], num_frames=num_frames)
    if not video_data:
        return None

    video_data = validate_paths(video_data, f"test_{ds_name}", max_workers=16, sample_limit=50)
    if not video_data:
        return None

    df = pd.DataFrame(video_data)
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.set_transform(_apply_val_transform)

    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=DefaultDataCollator().__call__,
        drop_last=False,
    )

    all_probs, all_labels = [], []
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        for batch in dataloader:
            pv = batch["pixel_values"]
            if isinstance(pv, list):
                pv = torch.stack(pv)
            pv = pv.to(device)

            logits = model(pv).logits
            if MULTITASK:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                probs = torch.sigmoid(logits).cpu().float().numpy().reshape(-1)

            all_probs.append(probs)
            all_labels.append(batch["labels"].numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    video_ids = df["video_id"].tolist()

    result = {"num_frames": len(all_probs)}

    # Frame count mismatch fallback
    if len(all_probs) != len(video_ids):
        try:
            frame_auc = roc_auc_score(all_labels, all_probs) if not MULTITASK else roc_auc_score(all_labels, all_probs, multi_class="ovr")
        except Exception:
            frame_auc = 0.5
        result.update({"frame_auc": frame_auc, "video_auc": None})
        return result

    # Aggregate by video_id (MEAN)
    video_probs_map = defaultdict(list)
    video_labels_map = {}
    for vid_id, prob, label in zip(video_ids, all_probs, all_labels):
        video_probs_map[vid_id].append(prob)
        video_labels_map[vid_id] = label

    video_mean_probs, video_gt_labels = [], []
    for vid_id in video_probs_map:
        video_mean_probs.append(np.array(video_probs_map[vid_id]).mean(axis=0))
        video_gt_labels.append(video_labels_map[vid_id])

    video_mean_probs = np.array(video_mean_probs)
    video_gt_labels = np.array(video_gt_labels)

    try:
        if MULTITASK:
            video_auc = roc_auc_score(video_gt_labels, video_mean_probs, multi_class="ovr")
            video_preds = video_mean_probs.argmax(axis=-1)
        else:
            video_auc = roc_auc_score(video_gt_labels, video_mean_probs)
            video_preds = (video_mean_probs > 0.5).astype(int)
    except Exception:
        video_auc = 0.5
        video_preds = (video_mean_probs > 0.5).astype(int) if not MULTITASK else video_mean_probs.argmax(axis=-1)

    video_acc = accuracy_score(video_gt_labels, video_preds)

    try:
        frame_auc = roc_auc_score(all_labels, all_probs) if not MULTITASK else roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except Exception:
        frame_auc = 0.5

    result.update({
        "video_auc": video_auc,
        "video_acc": video_acc,
        "frame_auc": frame_auc,
        "num_videos": len(video_probs_map),
    })
    return result


def _extract_and_save_state_dict(trainer_model, save_path):
    """Extract state_dict from trainer model (handling DDP/PEFT) and save to .pt file."""
    m = trainer_model
    if hasattr(m, "module"):
        m = m.module

    def _to_cpu_contiguous_state_dict(sd):
        packed = {}
        for key, value in sd.items():
            if torch.is_tensor(value):
                packed[key] = value.detach().cpu().contiguous().clone()
            else:
                packed[key] = value
        return packed

    if USE_PEFT:
        # 어댑터(LoRA + modules_to_save; 예: classifier/LN) 파라미터만 저장
        from peft import get_peft_model_state_dict
        state_dict = get_peft_model_state_dict(m)
        state_dict = _to_cpu_contiguous_state_dict(state_dict)
    else:
        state_dict = _to_cpu_contiguous_state_dict(m.state_dict())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state_dict, save_path)
    if USE_PEFT:
        adapter_cfg_path = os.path.join(os.path.dirname(save_path), "adapter_config.json")
        adapter_cfg = {
            "lora_rank": int(LORA_RANK),
            "lora_alpha": int(LORA_ALPHA),
            "layernorm_lr_scale": float(LAYERNORM_LR_SCALE),
            "learning_rate": float(LEARNING_RATE),
            "clip": bool(USE_CLIP),
            "clip_model": str(CLIP_MODEL),
        }
        with open(adapter_cfg_path, "w", encoding="utf-8") as f:
            json.dump(adapter_cfg, f, ensure_ascii=False, indent=2)
    return save_path


def _dist_barrier_if_needed():
    """Synchronize all ranks in distributed runs."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        else:
            torch.distributed.barrier()


# -----------------------------------------------------------------------------
# [4] Best-per-dataset callback
# -----------------------------------------------------------------------------
class BestModelPerDatasetCallback(TrainerCallback):
    """
    After each epoch, evaluate each TEST_DATASET independently.
    When a dataset achieves a new best video-level AUC (mean aggregation),
    save the model as model/best_{dataset_name}.pt.
    Also saves model/model.pt as the overall best (mean of all dataset AUCs).
    """

    def __init__(self, test_datasets, num_frames, batch_size, num_workers):
        self.test_datasets = [ds for ds in test_datasets if ds.get("enabled", True)]
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.best_auc_per_dataset = {}  # ds_name -> best video_auc
        self.best_mean_auc = -1.0       # best mean across all datasets
        self.model_dir = os.path.join(OUTPUT_DIR, "best_models")
        os.makedirs(self.model_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            _dist_barrier_if_needed()
            return

        # In DDP, rank0-only callback with wrapped model can hang on collectives.
        # Evaluate with the raw module on main rank only, then barrier-sync all ranks.
        eval_model = model.module if hasattr(model, "module") else model

        if not is_main_process():
            _dist_barrier_if_needed()
            return

        epoch = int(state.epoch)
        device = next(eval_model.parameters()).device
        eval_model.eval()

        print(f"\n{'='*60}")
        print(f"[Callback] Epoch {epoch} — Per-dataset evaluation (mean video AUC, {self.num_frames} frames)")
        print(f"{'='*60}")

        epoch_aucs = {}

        for ds_conf in self.test_datasets:
            ds_name = ds_conf["name"]
            print(f"\n  [*] Evaluating: {ds_name}")

            result = _evaluate_single_dataset(
                eval_model, ds_conf, device,
                num_frames=self.num_frames,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

            if result is None:
                print(f"      [Warning] No data for {ds_name}")
                continue

            video_auc = result.get("video_auc")
            if video_auc is None:
                print(f"      Frame-level AUC: {result.get('frame_auc', 'N/A'):.4f} (video mismatch)")
                continue

            epoch_aucs[ds_name] = video_auc
            prev_best = self.best_auc_per_dataset.get(ds_name, -1.0)

            print(f"      Video AUC: {video_auc:.4f}  (prev best: {prev_best:.4f})")
            if result.get("video_acc") is not None:
                print(f"      Video Acc: {result['video_acc']:.4f}, Frame AUC: {result['frame_auc']:.4f}, #Videos: {result['num_videos']}")

            if video_auc > prev_best:
                self.best_auc_per_dataset[ds_name] = video_auc
                safe_name = ds_name.replace("/", "_").replace(" ", "_")
                ds_model_dir = os.path.join(self.model_dir, safe_name)
                os.makedirs(ds_model_dir, exist_ok=True)
                save_path = os.path.join(ds_model_dir, "model.pt")
                print(f"      *** NEW BEST for {ds_name}! Saving to {save_path} ***")
                _extract_and_save_state_dict(model, save_path)
                # Ensure model is back on device
                eval_model.to(device)

        # Overall best (mean of all dataset AUCs)
        if epoch_aucs:
            mean_auc = np.mean(list(epoch_aucs.values()))
            print(f"\n  [*] Epoch {epoch} Mean Video AUC: {mean_auc:.4f} (prev best: {self.best_mean_auc:.4f})")
            if mean_auc > self.best_mean_auc:
                self.best_mean_auc = mean_auc
                save_path = os.path.join(self.model_dir, "model.pt")
                print(f"      *** NEW OVERALL BEST! Saving to {save_path} ***")
                _extract_and_save_state_dict(model, save_path)
                eval_model.to(device)

        # Print summary table
        print(f"\n  {'Dataset':<25} {'Epoch AUC':>10} {'Best AUC':>10}")
        print(f"  {'-'*47}")
        for ds_name in [ds["name"] for ds in self.test_datasets]:
            ea = f"{epoch_aucs[ds_name]:.4f}" if ds_name in epoch_aucs else "N/A"
            ba = f"{self.best_auc_per_dataset[ds_name]:.4f}" if ds_name in self.best_auc_per_dataset else "N/A"
            print(f"  {ds_name:<25} {ea:>10} {ba:>10}")
        if epoch_aucs:
            print(f"  {'-'*47}")
            print(f"  {'MEAN':<25} {np.mean(list(epoch_aucs.values())):>10.4f} {self.best_mean_auc:>10.4f}")
        print(f"{'='*60}\n")

        eval_model.train()
        torch.cuda.empty_cache()
        _dist_barrier_if_needed()


# -----------------------------------------------------------------------------
# [5] Test Mode: Per-dataset best model → Video-level AUC (mean, 32 frames)
# -----------------------------------------------------------------------------
def run_test_mode():
    """
    For each TEST_DATASET:
      1) Try to load model/best_{dataset_name}.pt (per-dataset best)
      2) Fallback to model/model.pt (overall best)
    Evaluate with TEST_NUM_FRAMES frames, mean video-level AUC.
    """
    model_dir = os.path.join(OUTPUT_DIR, "best_models")
    fallback_path = os.path.join(model_dir, "model.pt")

    print("=" * 60)
    print("[TEST MODE] Evaluating best models per dataset")
    print(f"    - Model dir: {model_dir}")
    print(f"    - Frames per video: {TEST_NUM_FRAMES}")
    print(f"    - Video aggregation: MEAN")
    print(f"    - Test dataset preset: {RESOLVED_TEST_DATASET_PRESET}")
    print("=" * 60)

    init_augs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_summary = {}

    for ds_conf in TEST_DATASETS:
        if not ds_conf.get("enabled", True):
            continue

        ds_name = ds_conf["name"]
        safe_name = ds_name.replace("/", "_").replace(" ", "_")
        best_path = os.path.join(model_dir, safe_name, "model.pt")

        # Choose model: per-dataset best > overall best
        if os.path.exists(best_path):
            model_path = best_path
            print(f"\n[*] {ds_name}: Using per-dataset best model: {model_path}")
        elif os.path.exists(fallback_path):
            model_path = fallback_path
            print(f"\n[*] {ds_name}: Using overall best model (fallback): {model_path}")
        else:
            print(f"\n[*] {ds_name}: No model found, skipping.")
            continue

        # Load model
        model = DINOv3ForClassification(MODEL_ID, class_weights=None)
        if USE_PEFT:
            model = build_peft_model(model)
            adapter_state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model, adapter_state_dict)
        elif USE_EFFORT:
            model = build_effort_model(model, r=EFFORT_RANK)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        result = _evaluate_single_dataset(
            model, ds_conf, device,
            num_frames=TEST_NUM_FRAMES,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        del model
        torch.cuda.empty_cache()

        if result is None:
            print(f"    [Warning] No test data for {ds_name}")
            continue

        video_auc = result.get("video_auc")
        if video_auc is None:
            print(f"    Frame-level AUC: {result.get('frame_auc', 'N/A'):.4f}")
            results_summary[ds_name] = {"frame_auc": result.get("frame_auc", 0.5), "video_auc": "N/A"}
            continue

        print(f"    Videos: {result.get('num_videos', 'N/A')}")
        print(f"    Video-level AUC:  {video_auc:.4f}")
        print(f"    Video-level Acc:  {result.get('video_acc', 'N/A')}")
        print(f"    Frame-level AUC:  {result.get('frame_auc', 'N/A'):.4f}")

        results_summary[ds_name] = result

    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("[TEST MODE] RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<25} {'Video AUC':>10} {'Video Acc':>10} {'Frame AUC':>10} {'#Videos':>8}")
    print("-" * 65)
    for ds_name, m in results_summary.items():
        va = f"{m['video_auc']:.4f}" if isinstance(m.get('video_auc'), float) else str(m.get('video_auc', 'N/A'))
        vc = f"{m['video_acc']:.4f}" if isinstance(m.get('video_acc'), float) else str(m.get('video_acc', 'N/A'))
        fa = f"{m['frame_auc']:.4f}" if isinstance(m.get('frame_auc'), float) else str(m.get('frame_auc', 'N/A'))
        nv = str(m.get('num_videos', 'N/A'))
        print(f"{ds_name:<25} {va:>10} {vc:>10} {fa:>10} {nv:>8}")

    valid_aucs = [m["video_auc"] for m in results_summary.values() if isinstance(m.get("video_auc"), float)]
    mean_video_auc = None
    if valid_aucs:
        mean_video_auc = float(np.mean(valid_aucs))
        print("-" * 65)
        print(f"{'MEAN':.<25} {mean_video_auc:>10.4f}")
    print("=" * 60)

    # Persist test summary for sweep/result aggregation.
    summary_rows = []
    for ds_name, m in results_summary.items():
        summary_rows.append(
            {
                "dataset": ds_name,
                "video_auc": float(m["video_auc"]) if isinstance(m.get("video_auc"), float) else None,
                "video_acc": float(m["video_acc"]) if isinstance(m.get("video_acc"), float) else None,
                "frame_auc": float(m["frame_auc"]) if isinstance(m.get("frame_auc"), float) else None,
                "num_videos": int(m["num_videos"]) if isinstance(m.get("num_videos"), (int, np.integer)) else None,
            }
        )
    if mean_video_auc is not None:
        summary_rows.append(
            {
                "dataset": "MEAN",
                "video_auc": mean_video_auc,
                "video_acc": None,
                "frame_auc": None,
                "num_videos": None,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["dataset", "video_auc", "video_acc", "frame_auc", "num_videos"],
    )
    summary_csv_path = os.path.join(model_dir, "test_mode_summary.csv")
    summary_json_path = os.path.join(model_dir, "test_mode_summary.json")
    os.makedirs(model_dir, exist_ok=True)
    summary_df.to_csv(summary_csv_path, index=False)

    summary_payload = {
        "output_dir": OUTPUT_DIR,
        "test_num_frames": int(TEST_NUM_FRAMES),
        "test_dataset_preset": str(RESOLVED_TEST_DATASET_PRESET),
        "test_datasets": [str(ds["name"]) for ds in TEST_DATASETS if ds.get("enabled", True)],
        "lora_rank": int(LORA_RANK),
        "lora_alpha": int(LORA_ALPHA),
        "layernorm_lr_scale": float(LAYERNORM_LR_SCALE),
        "learning_rate": float(LEARNING_RATE),
        "mean_video_auc": mean_video_auc,
        "results": summary_rows,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(f"[*] Test summary CSV saved to: {summary_csv_path}")
    print(f"[*] Test summary JSON saved to: {summary_json_path}")


def main():
    if not is_main_process():
        builtins.print = lambda *args, **kwargs: None

    # TEST MODE: model/model.pt로 테스트 데이터셋 평가
    if TEST_MODE:
        return run_test_mode()

    # 1. 시드 고정 (재현성) - config에 값이 있을 때만
    if SEED is not None:
        from transformers import set_seed
        set_seed(SEED)

    print(f"[*] Starting Training...")
    print(f"    - Output Dir: {OUTPUT_DIR}")
    print(f"    - Model ID: {MODEL_ID}")
    print(f"    - Encoder: {'CLIP' if USE_CLIP else 'DINO'}")
    print(f"    - Test dataset preset: {RESOLVED_TEST_DATASET_PRESET}")
    if USE_CLIP:
        print(f"    - CLIP Model: {CLIP_MODEL}")
    print(f"    - Multitask: {MULTITASK}")
    print(f"    - LoRA: r={LORA_RANK}, alpha={LORA_ALPHA}")
    print(
        f"    - LayerNorm LR: {LEARNING_RATE * LAYERNORM_LR_SCALE} "
        f"(base={LEARNING_RATE}, scale={LAYERNORM_LR_SCALE})"
    )

    # 2. 데이터 로드
    print("\n" + "=" * 60)
    print("[*] Loading Datasets...")
    print("=" * 60)
    train_list, _ = load_all_data(DATASETS, train_only=False)
    _, test_list = load_all_data(TEST_DATASETS, test_only=True)

    if not train_list:
        sys.exit("[Error] No training data.")

    if not test_list:
        print("[Warning] No test data. Splitting train...")
        # random.shuffle(train_list)
        split_seed = 42
        split_rng = random.Random(split_seed)
        split_rng.shuffle(train_list)
        print(f"[*] Split seed: {split_seed}")
        val_ratio = 0.01
        split = int(len(train_list) * (1 - val_ratio))
        test_list = train_list[split:]
        train_list = train_list[:split]

    train_list = validate_paths(train_list, "training", max_workers=16, sample_limit=200)
    print(f"[*] Valid Train Size: {len(train_list)}")

    test_list = validate_paths(test_list, "test", max_workers=16, sample_limit=200)
    print(f"[*] Valid Test Size: {len(test_list)}")

    test_list = validate_images_openable(test_list, "test", max_workers=16, sample_limit=50)
    print(f"[*] Readable Test Size: {len(test_list)}")

    if not train_list:
        sys.exit("[Error] All training data dropped!")

    # 데이터 밸런싱
    if MULTITASK:
        train_list = balance_three_class(train_list)
    else:
        train_list = balance_real_fake(train_list)

    class_weights_val = None
    print("[*] Class weights disabled.")

    train_df = pd.DataFrame(train_list)
    train_dataset = Dataset.from_pandas(train_df)
    val_df = pd.DataFrame(test_list)
    eval_dataset = Dataset.from_pandas(val_df)

    init_augs()
    train_dataset.set_transform(lambda b: apply_transforms_robust(b, True))
    eval_dataset.set_transform(lambda b: apply_transforms_robust(b, False))

    # 4. 모델 준비
    model = DINOv3ForClassification(MODEL_ID, class_weights=class_weights_val)
    if USE_PEFT:
        model = build_peft_model(model)
    elif USE_EFFORT:
        model = build_effort_model(model, r=EFFORT_RANK)
    
    # 5. Training Arguments 설정 (training.py의 하드코딩 값 유지)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE // 4,  # training batch size의 1/4
        
        # Optimizer & Scheduler
        lr_scheduler_type="cosine",
        warmup_steps=200,       # training.py 원본 값
        
        # Memory Management
        eval_accumulation_steps=1, # 검증 시 예측값을 즉시 CPU로 옮겨 VRAM 누수 방지
        
        # Hardware Acceleration (training.py 원본 값)
        fp16=False,
        bf16=True,
        tf32=True,
        
        # Logging & Saving (단 1개의 pt 파일만 남기도록 설정)
        run_name=WANDB_RUN_NAME,
        report_to="wandb",
        save_strategy="no",
        eval_strategy="epoch",
        load_best_model_at_end=False,
        
        # DataLoader
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],
        
        # DDP
        ddp_find_unused_parameters=False,
        ddp_timeout=DDP_TIMEOUT_SECONDS,
    )

    # 6. Callback: 매 epoch마다 각 테스트셋 평가 + best model 저장
    #    학습 중에는 빠르게 8프레임으로 best epoch 탐색, 나중에 test mode에서 32프레임 평가
    best_model_callback = BestModelPerDatasetCallback(
        test_datasets=TEST_DATASETS,
        num_frames=NUM_FRAMES_PER_VIDEO,  # 학습 중 빠른 평가: 8 frames
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    optimizer = _build_optimizer(model, training_args)

    # 7. Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[best_model_callback],
        optimizers=(optimizer, None),
    )

    # 8. 학습 시작
    print("[*] Initiating Trainer.train()...")
    trainer.train()
    
    # 9. 최종 모델 저장 (callback이 이미 best를 저장했지만, 마지막 epoch도 저장)
    if is_main_process():
        final_model_dir = os.path.join(OUTPUT_DIR, "best_models")
        print(f"[*] Saving final (last epoch) model to {final_model_dir}...")
        
        last_model_path = os.path.join(final_model_dir, "model_last.pt")
        _extract_and_save_state_dict(trainer.model, last_model_path)
        print(f"[*] Last epoch model saved to: {last_model_path}")

        # 만약 callback에서 model.pt가 저장되지 않았으면 (테스트셋 없는 경우 등) last를 복사
        best_model_path = os.path.join(final_model_dir, "model.pt")
        if not os.path.exists(best_model_path):
            import shutil
            shutil.copy2(last_model_path, best_model_path)
            print(f"[*] No best model found, copied last epoch as model.pt")

        print(f"[*] Done! Best models per dataset saved in: {final_model_dir}")
        # List saved models
        saved = [f for f in os.listdir(final_model_dir) if f.endswith(".pt")]
        for f in sorted(saved):
            size_mb = os.path.getsize(os.path.join(final_model_dir, f)) / (1024*1024)
            print(f"    - {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
