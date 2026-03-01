import argparse
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from peft import set_peft_model_state_dict
from torchvision import transforms

from .dataset_core import TEST_DATASETS, parse_dataset_json_video_level
from .models import DINOv3ForClassification, build_peft_model
from .utils import IMAGE_SIZE, MODEL_ID
import src.models as models_mod


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]
EPS = 1e-8


def _log(msg: str):
    print(f"[analysis] {msg}")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stable_name_seed(base_seed: int, name: str) -> int:
    token = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return int(token, 16) + int(base_seed)


def _safe_float(x) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _normalize_01(arr: np.ndarray, eps: float = EPS) -> np.ndarray:
    arr = arr.astype(np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - vmin) / (vmax - vmin)


def _prob_from_logits(logits: torch.Tensor) -> float:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.shape[-1] == 1:
        return float(torch.sigmoid(logits[0, 0]).item())
    if logits.shape[-1] >= 2:
        prob = torch.softmax(logits[0], dim=-1)[1]
        return float(prob.item())
    return float("nan")


def _moment_skew(values: Sequence[float], eps: float = EPS) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std < eps:
        return 0.0
    z = (arr - mean) / std
    return float(np.mean(z ** 3))


def _attention_map_stats(attn_map: np.ndarray, eps: float = EPS) -> Dict[str, float]:
    p = np.asarray(attn_map, dtype=np.float64)
    p = np.maximum(p, 0.0)
    s = float(p.sum())
    if s < eps:
        h, w = p.shape
        return {
            "skew_x": 0.0,
            "skew_y": 0.0,
            "com_x": 0.5,
            "com_y": 0.5,
        }
    p = p / s

    h, w = p.shape
    x = np.arange(w, dtype=np.float64)
    y = np.arange(h, dtype=np.float64)

    mx = p.sum(axis=0)
    my = p.sum(axis=1)

    mu_x = float((mx * x).sum())
    mu_y = float((my * y).sum())

    std_x = float(np.sqrt(np.maximum(((x - mu_x) ** 2 * mx).sum(), 0.0)))
    std_y = float(np.sqrt(np.maximum(((y - mu_y) ** 2 * my).sum(), 0.0)))

    if std_x < eps:
        skew_x = 0.0
    else:
        skew_x = float((((x - mu_x) / std_x) ** 3 * mx).sum())

    if std_y < eps:
        skew_y = 0.0
    else:
        skew_y = float((((y - mu_y) / std_y) ** 3 * my).sum())

    return {
        "skew_x": skew_x,
        "skew_y": skew_y,
        "com_x": float(mu_x / max(1.0, w - 1.0)),
        "com_y": float(mu_y / max(1.0, h - 1.0)),
    }


def _vec_to_square(vec: torch.Tensor) -> torch.Tensor:
    n = int(vec.numel())
    side = int(round(math.sqrt(n)))
    if side * side != n:
        side = int(math.floor(math.sqrt(n)))
        side = max(1, side)
        vec = vec[: side * side]
    return vec.reshape(side, side)


def _last_attention_map(attn_last: torch.Tensor) -> np.ndarray:
    # attn_last: [heads, tokens, tokens]
    cls_patch = attn_last.mean(dim=0)[0, 1:]
    square = _vec_to_square(cls_patch)
    return square.detach().cpu().float().numpy()


def _attention_rollout_map(attentions: Sequence[torch.Tensor], eps: float = EPS) -> np.ndarray:
    # attentions each: [1, heads, tokens, tokens]
    tokens = attentions[0].shape[-1]
    eye = torch.eye(tokens, device=attentions[0].device, dtype=attentions[0].dtype)
    rollout = torch.eye(tokens, device=attentions[0].device, dtype=attentions[0].dtype)

    for layer_attn in attentions:
        a = layer_attn[0].mean(dim=0)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(eps)
        rollout = a @ rollout

    cls_patch = rollout[0, 1:]
    square = _vec_to_square(cls_patch)
    return square.detach().cpu().float().numpy()


def _layer_profile(attentions: Sequence[torch.Tensor], eps: float = EPS) -> Tuple[np.ndarray, np.ndarray]:
    entropies: List[float] = []
    max_conc: List[float] = []

    for layer_attn in attentions:
        cls_patch = layer_attn[0].mean(dim=0)[0, 1:]
        p = torch.clamp(cls_patch, min=0)
        p = p / p.sum().clamp_min(eps)
        entropy = -(p * torch.log(p + eps)).sum() / math.log(max(2, p.numel()))
        entropies.append(float(entropy.item()))
        max_conc.append(float(p.max().item()))

    return np.asarray(entropies, dtype=np.float32), np.asarray(max_conc, dtype=np.float32)


def _save_overlay_png(image_np: np.ndarray, heatmap: np.ndarray, out_path: str, title: str):
    hm = _normalize_01(heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image_np)
    axes[0].set_title("image")
    axes[1].imshow(hm, cmap="magma")
    axes[1].set_title("heatmap")
    axes[2].imshow(image_np)
    axes[2].imshow(hm, cmap="magma", alpha=0.45)
    axes[2].set_title(title)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_meanmap_figure(
    dataset: str,
    out_path: str,
    last_real: Optional[np.ndarray],
    last_fake: Optional[np.ndarray],
    roll_real: Optional[np.ndarray],
    roll_fake: Optional[np.ndarray],
):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    maps = [last_real, last_fake, roll_real, roll_fake]
    titles = [
        "LastAttn (Real)",
        "LastAttn (Fake)",
        "Rollout (Real)",
        "Rollout (Fake)",
    ]
    for ax, m, t in zip(axes.reshape(-1), maps, titles):
        if m is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.imshow(_normalize_01(m), cmap="magma")
        ax.set_title(t)
        ax.axis("off")
    fig.suptitle(f"{dataset}: Mean Attention Maps", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_score_hist(dataset: str, out_path: str, probs_real: List[float], probs_fake: List[float]):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    if probs_real:
        ax.hist(probs_real, bins=bins, alpha=0.6, label="real", color="#4C78A8", density=False)
    if probs_fake:
        ax.hist(probs_fake, bins=bins, alpha=0.6, label="fake", color="#F58518", density=False)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fake probability")
    ax.set_ylabel("count")
    ax.set_title(f"{dataset}: Score Histogram")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_layer_profile_plot(
    dataset: str,
    out_path: str,
    ent_real: Optional[np.ndarray],
    ent_fake: Optional[np.ndarray],
    max_real: Optional[np.ndarray],
    max_fake: Optional[np.ndarray],
):
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    if ent_real is not None:
        axes[0].plot(ent_real, label="real", color="#4C78A8")
    if ent_fake is not None:
        axes[0].plot(ent_fake, label="fake", color="#F58518")
    axes[0].set_ylabel("entropy")
    axes[0].set_title("Layer-wise CLS Attention Entropy")
    axes[0].legend()

    if max_real is not None:
        axes[1].plot(max_real, label="real", color="#4C78A8")
    if max_fake is not None:
        axes[1].plot(max_fake, label="fake", color="#F58518")
    axes[1].set_ylabel("max concentration")
    axes[1].set_xlabel("layer index")
    axes[1].set_title("Layer-wise CLS Max Concentration")
    axes[1].legend()

    fig.suptitle(f"{dataset}: Layer Profile")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_lora_heatmap(dataset: str, out_path: str, lora_df: pd.DataFrame):
    if lora_df.empty:
        return

    projs = ["q_proj", "k_proj", "v_proj", "o_proj"]
    layers = sorted(lora_df["layer"].unique().tolist())
    mat = np.zeros((len(layers), len(projs)), dtype=np.float32)

    layer_to_idx = {layer: i for i, layer in enumerate(layers)}
    proj_to_idx = {proj: j for j, proj in enumerate(projs)}

    for _, row in lora_df.iterrows():
        li = layer_to_idx[int(row["layer"])]
        proj = str(row["proj"])
        if proj not in proj_to_idx:
            continue
        pj = proj_to_idx[proj]
        mat[li, pj] = float(row["delta_norm"])

    fig, ax = plt.subplots(figsize=(5.5, 7))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(projs)))
    ax.set_xticklabels(projs)
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([str(x) for x in layers])
    ax.set_xlabel("projection")
    ax.set_ylabel("layer")
    ax.set_title(f"{dataset}: LoRA Delta Norm Heatmap")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("||B@A||_F * (alpha/r)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _set_attention_impl_eager(backbone) -> bool:
    changed = False
    try:
        cfg = getattr(backbone, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            if getattr(cfg, "_attn_implementation") != "eager":
                cfg._attn_implementation = "eager"
                changed = True
    except Exception:
        pass

    try:
        vcfg = getattr(getattr(backbone, "vision_model", None), "config", None)
        if vcfg is not None and hasattr(vcfg, "_attn_implementation"):
            if getattr(vcfg, "_attn_implementation") != "eager":
                vcfg._attn_implementation = "eager"
                changed = True
    except Exception:
        pass

    return changed


def _sanitize_attentions(attentions) -> List[torch.Tensor]:
    if attentions is None:
        return []
    valid: List[torch.Tensor] = []
    for a in attentions:
        if a is None:
            continue
        if not torch.is_tensor(a):
            continue
        if a.ndim != 4:
            continue
        valid.append(a)
    return valid


def _extract_lora_rank(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    for k, v in state_dict.items():
        if k.endswith("lora_A.weight") and getattr(v, "ndim", None) == 2:
            return int(v.shape[0])
    return None


def _read_adapter_config(model_dir: str) -> Dict[str, object]:
    cfg_path = os.path.join(model_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _strip_uniform_prefix(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if all(str(k).startswith(prefix) for k in keys):
        return {str(k)[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _infer_is_clip_from_state_keys(state_dict: Dict[str, torch.Tensor]) -> bool:
    keys = list(state_dict.keys())
    if any("vision_model." in str(k) for k in keys):
        return True
    if any(".backbone.layer." in str(k) for k in keys):
        return False
    return False


def _infer_runtime_config(
    state_dict: Dict[str, torch.Tensor],
    adapter_cfg: Dict[str, object],
) -> Dict[str, object]:
    keys = list(state_dict.keys())
    has_lora = any(".lora_" in str(k) for k in keys)
    has_ln_saved = any(
        ".lora_" not in str(k)
        and ("layer_norm" in str(k) or ".norm" in str(k))
        and (str(k).endswith(".weight") or str(k).endswith(".bias"))
        for k in keys
    )

    clip_value = adapter_cfg.get("clip", None)
    if clip_value is None:
        resolved_use_clip = _infer_is_clip_from_state_keys(state_dict)
    else:
        resolved_use_clip = bool(clip_value)

    peft_value = adapter_cfg.get("use_peft", None)
    resolved_use_peft = bool(peft_value) if peft_value is not None else bool(has_lora)

    ln_value = adapter_cfg.get("use_layernorm_tuning", None)
    if ln_value is None:
        resolved_use_layernorm_tuning = bool(has_ln_saved) if resolved_use_peft else bool(models_mod.USE_LAYERNORM_TUNING)
    else:
        resolved_use_layernorm_tuning = bool(ln_value)

    resolved_rank = int(adapter_cfg.get("lora_rank") or (_extract_lora_rank(state_dict) or 0))
    resolved_alpha = int(adapter_cfg.get("lora_alpha") or (2 if resolved_rank > 0 else 0))
    resolved_clip_model = str(adapter_cfg.get("clip_model", models_mod.CLIP_MODEL))

    return {
        "use_clip": bool(resolved_use_clip),
        "clip_model": resolved_clip_model,
        "use_peft": bool(resolved_use_peft),
        "use_layernorm_tuning": bool(resolved_use_layernorm_tuning),
        "lora_rank": int(resolved_rank),
        "lora_alpha": int(resolved_alpha),
        "has_lora": bool(has_lora),
        "has_ln_saved": bool(has_ln_saved),
    }


def _compute_weight_stats(
    state_dict: Dict[str, torch.Tensor],
    dataset: str,
    alpha: int,
    rank: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    def _parse_layer_proj_from_lora_key(key: str) -> Tuple[Optional[int], Optional[str]]:
        parts = key.split(".")

        # CLIP-style: ... encoder.layers.{idx}.self_attn.{proj}.lora_A.weight
        if "layers" in parts and "self_attn" in parts:
            try:
                return int(parts[parts.index("layers") + 1]), parts[parts.index("self_attn") + 1]
            except Exception:
                pass

        # DINO-style: ... backbone.layer.{idx}.attention.{proj}.lora_A.weight
        if "layer" in parts and "attention" in parts:
            try:
                return int(parts[parts.index("layer") + 1]), parts[parts.index("attention") + 1]
            except Exception:
                pass

        # Fallback for unexpected key layouts.
        layer_idx = None
        proj = None
        for i, tok in enumerate(parts[:-1]):
            if tok in ("layer", "layers"):
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except Exception:
                    continue
        for tok in parts:
            if tok in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = tok
                break

        return layer_idx, proj

    rows: List[Dict[str, object]] = []

    alpha = int(alpha)
    rank = max(1, int(rank))
    scale = float(alpha / rank)

    ln_sq_sum = 0.0
    ln_param_count = 0
    classifier_weight_norm = float("nan")
    classifier_bias_norm = float("nan")

    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            continue

        if k.endswith("classifier.weight"):
            classifier_weight_norm = float(v.float().norm().item())
        elif k.endswith("classifier.bias"):
            classifier_bias_norm = float(v.float().norm().item())

        if ".lora_" not in k and ("layer_norm" in k or ".norm" in k) and (k.endswith(".weight") or k.endswith(".bias")):
            val = v.float().reshape(-1)
            ln_sq_sum += float((val * val).sum().item())
            ln_param_count += int(val.numel())

        if not k.endswith("lora_A.weight"):
            continue

        b_key = k.replace("lora_A.weight", "lora_B.weight")
        if b_key not in state_dict:
            continue

        layer_idx, proj = _parse_layer_proj_from_lora_key(k)
        if layer_idx is None or proj is None:
            continue

        a = state_dict[k].float()
        b = state_dict[b_key].float()
        delta = b @ a

        rows.append(
            {
                "dataset": dataset,
                "layer": layer_idx,
                "proj": proj,
                "a_norm": float(a.norm().item()),
                "b_norm": float(b.norm().item()),
                "delta_norm": float(delta.norm().item() * scale),
                "rank": rank,
                "alpha": alpha,
            }
        )

    rows = sorted(rows, key=lambda x: (int(x["layer"]), str(x["proj"])))
    lora_df = pd.DataFrame(rows)

    weight_summary = {
        "classifier_weight_norm": classifier_weight_norm,
        "classifier_bias_norm": classifier_bias_norm,
        "layernorm_param_l2": float(math.sqrt(max(0.0, ln_sq_sum))),
        "layernorm_param_count": int(ln_param_count),
    }
    return lora_df, weight_summary


def _parse_test_auc_map(weights_root: str) -> Dict[str, float]:
    auc_map: Dict[str, float] = {}
    csv_path = os.path.join(weights_root, "test_mode_summary.csv")
    if not os.path.exists(csv_path):
        return auc_map

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds = str(row.get("dataset", "")).strip()
                if not ds or ds.upper() == "MEAN":
                    continue
                raw = row.get("video_auc")
                if raw in (None, "", "None"):
                    continue
                auc_map[ds] = float(raw)
    except Exception:
        return {}

    return auc_map


@dataclass
class AnalysisConfig:
    weights_root: str
    output_dir: str
    sample_per_class: int
    seed: int
    device: str
    num_workers: int
    datasets: Optional[List[str]] = None


class ClipBestModelAnalyzer:
    def __init__(self, cfg: AnalysisConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")

        self.tables_dir = os.path.join(cfg.output_dir, "tables")
        self.figures_dir = os.path.join(cfg.output_dir, "figures")
        self.sample_maps_dir = os.path.join(cfg.output_dir, "sample_maps")

        _ensure_dir(cfg.output_dir)
        _ensure_dir(self.tables_dir)
        _ensure_dir(self.figures_dir)
        _ensure_dir(self.sample_maps_dir)

        self.transform_clip = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )
        self.transform_dino = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=DINO_MEAN, std=DINO_STD),
            ]
        )

        self.video_auc_map = _parse_test_auc_map(cfg.weights_root)
        self.warnings: List[str] = []
        self.model_shape_reference: Dict[str, Tuple[int, int, int]] = {}

    def _select_test_datasets(self) -> List[Dict[str, object]]:
        selected = [d for d in TEST_DATASETS if d.get("enabled", True)]
        if self.cfg.datasets:
            allow = {x.strip() for x in self.cfg.datasets if x.strip()}
            selected = [d for d in selected if d.get("name") in allow]
        return selected

    def _sample_dataset_items(self, ds_conf: Dict[str, object]) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
        ds_name = str(ds_conf["name"])
        items = parse_dataset_json_video_level(
            str(ds_conf["json"]),
            "test",
            str(ds_conf.get("root", "")),
            ds_name,
            num_frames=1,
        )
        items = [x for x in items if os.path.exists(str(x.get("image_path", "")))]

        by_label = {0: [], 1: []}
        for item in items:
            label = int(item.get("label", -1))
            if label in by_label:
                by_label[label].append(item)

        sample_seed = _stable_name_seed(self.cfg.seed, ds_name)
        rng = random.Random(sample_seed)

        sampled: List[Dict[str, object]] = []
        counts = {"available_real": len(by_label[0]), "available_fake": len(by_label[1])}
        for lbl in [0, 1]:
            pool = by_label[lbl]
            want = int(self.cfg.sample_per_class)
            take = min(len(pool), want)
            if take < want:
                self.warnings.append(
                    f"{ds_name}: label={lbl} samples 부족 ({take}/{want}), 가능한 수만 사용"
                )
            chosen = rng.sample(pool, k=take) if take > 0 else []
            sampled.extend(chosen)

        rng.shuffle(sampled)
        counts.update({
            "sampled_real": sum(1 for x in sampled if int(x["label"]) == 0),
            "sampled_fake": sum(1 for x in sampled if int(x["label"]) == 1),
        })
        return sampled, counts

    def _load_model_for_dataset(self, ds_name: str):
        model_dir = os.path.join(self.cfg.weights_root, ds_name)
        model_path = os.path.join(model_dir, "model.pt")
        selected_model_dir = model_dir
        if not os.path.exists(model_path):
            fallback_model_path = os.path.join(self.cfg.weights_root, "model.pt")
            if os.path.exists(fallback_model_path):
                model_path = fallback_model_path
                selected_model_dir = self.cfg.weights_root
                self.warnings.append(f"{ds_name}: per-dataset model.pt 누락 -> overall model.pt fallback 사용")
            else:
                self.warnings.append(f"{ds_name}: model.pt 누락 -> 스킵")
                return None, None, None, None, None

        adapter_cfg = _read_adapter_config(selected_model_dir)
        if not adapter_cfg and selected_model_dir != self.cfg.weights_root:
            adapter_cfg = _read_adapter_config(self.cfg.weights_root)

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dict = _strip_uniform_prefix(state_dict, "module.")
        resolved_cfg = _infer_runtime_config(state_dict, adapter_cfg)

        # Patch src.models runtime globals so we can reuse the training-side builder.
        prev_values = {
            "USE_CLIP": models_mod.USE_CLIP,
            "CLIP_MODEL": models_mod.CLIP_MODEL,
            "LORA_RANK": models_mod.LORA_RANK,
            "LORA_ALPHA": models_mod.LORA_ALPHA,
            "USE_LAYERNORM_TUNING": models_mod.USE_LAYERNORM_TUNING,
            "USE_EFFORT": models_mod.USE_EFFORT,
        }
        models_mod.USE_CLIP = bool(resolved_cfg["use_clip"])
        models_mod.CLIP_MODEL = str(resolved_cfg["clip_model"])
        models_mod.LORA_RANK = int(max(1, resolved_cfg["lora_rank"])) if resolved_cfg["use_peft"] else models_mod.LORA_RANK
        models_mod.LORA_ALPHA = int(max(1, resolved_cfg["lora_alpha"])) if resolved_cfg["use_peft"] else models_mod.LORA_ALPHA
        models_mod.USE_LAYERNORM_TUNING = bool(resolved_cfg["use_layernorm_tuning"])
        models_mod.USE_EFFORT = False

        try:
            model = DINOv3ForClassification(MODEL_ID, class_weights=None)
            if resolved_cfg["use_peft"]:
                model = build_peft_model(model)
                set_peft_model_state_dict(model, state_dict)
            else:
                model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            model.eval()
            _set_attention_impl_eager(model.backbone)
        finally:
            models_mod.USE_CLIP = prev_values["USE_CLIP"]
            models_mod.CLIP_MODEL = prev_values["CLIP_MODEL"]
            models_mod.LORA_RANK = prev_values["LORA_RANK"]
            models_mod.LORA_ALPHA = prev_values["LORA_ALPHA"]
            models_mod.USE_LAYERNORM_TUNING = prev_values["USE_LAYERNORM_TUNING"]
            models_mod.USE_EFFORT = prev_values["USE_EFFORT"]

        model_hash = _sha256_file(model_path)
        return model, model_path, model_hash, state_dict, {
            "lora_rank": int(resolved_cfg["lora_rank"]),
            "lora_alpha": int(resolved_cfg["lora_alpha"]),
            "clip": bool(resolved_cfg["use_clip"]),
            "clip_model": str(resolved_cfg["clip_model"]),
            "use_peft": bool(resolved_cfg["use_peft"]),
            "use_layernorm_tuning": bool(resolved_cfg["use_layernorm_tuning"]),
        }

    def _analyze_one_sample(self, model, image_path: str, label: int, model_shape_key: str, is_clip: bool) -> Dict[str, object]:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_np = np.asarray(img_resized)

        transform = self.transform_clip if is_clip else self.transform_dino
        x = transform(img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            cls_out = model(x)
            fake_prob = _prob_from_logits(cls_out.logits)
            backbone_out = model.backbone(
                pixel_values=x,
                output_attentions=True,
                output_hidden_states=True,
            )

        attentions = _sanitize_attentions(getattr(backbone_out, "attentions", None))
        if not attentions:
            # Some backends can return None attentions with SDPA/flash paths.
            # Retry once with eager attention implementation.
            changed = _set_attention_impl_eager(model.backbone)
            if changed:
                with torch.inference_mode():
                    backbone_out = model.backbone(
                        pixel_values=x,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                attentions = _sanitize_attentions(getattr(backbone_out, "attentions", None))

        if not attentions:
            raise RuntimeError("attention tensors are empty or None after eager retry")

        num_layers = len(attentions)
        num_heads = int(attentions[-1].shape[1])
        num_tokens = int(attentions[-1].shape[-1])
        current_shape = (num_layers, num_heads, num_tokens)
        if model_shape_key not in self.model_shape_reference:
            self.model_shape_reference[model_shape_key] = current_shape
        expected_shape = self.model_shape_reference[model_shape_key]
        shape_ok = current_shape == expected_shape

        last_map = _last_attention_map(attentions[-1][0])
        rollout_map = _attention_rollout_map(attentions)
        layer_entropy, layer_max = _layer_profile(attentions)

        last_stats = _attention_map_stats(_normalize_01(last_map))
        roll_stats = _attention_map_stats(_normalize_01(rollout_map))

        return {
            "image_np": img_np,
            "fake_prob": float(fake_prob),
            "last_map": last_map,
            "rollout_map": rollout_map,
            "layer_entropy": layer_entropy,
            "layer_max": layer_max,
            "last_stats": last_stats,
            "roll_stats": roll_stats,
            "shape_ok": bool(shape_ok),
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_tokens": num_tokens,
            "label": int(label),
        }

    def run(self):
        _log(f"device={self.device}")
        _log(f"weights_root={self.cfg.weights_root}")
        _log(f"output_dir={self.cfg.output_dir}")

        sample_rows: List[Dict[str, object]] = []
        dataset_rows: List[Dict[str, object]] = []
        lora_rows: List[Dict[str, object]] = []
        hash_rows: List[Dict[str, object]] = []

        datasets = self._select_test_datasets()
        if not datasets:
            raise RuntimeError("분석 대상 TEST_DATASETS가 없습니다.")

        for ds_conf in datasets:
            ds_name = str(ds_conf["name"])
            _log(f"processing dataset={ds_name}")

            model, model_path, model_hash, state_dict, resolved_cfg = self._load_model_for_dataset(ds_name)
            if model is None:
                continue

            hash_rows.append(
                {
                    "dataset": ds_name,
                    "model_path": model_path,
                    "sha256": model_hash,
                }
            )

            sampled_items, sample_counts = self._sample_dataset_items(ds_conf)
            if not sampled_items:
                self.warnings.append(f"{ds_name}: 샘플 0개 -> 스킵")
                del model
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            ds_sample_dir = os.path.join(self.sample_maps_dir, ds_name)
            _ensure_dir(ds_sample_dir)

            lora_df, weight_summary = _compute_weight_stats(
                state_dict,
                dataset=ds_name,
                alpha=int(resolved_cfg["lora_alpha"]),
                rank=int(resolved_cfg["lora_rank"]),
            )
            if not lora_df.empty:
                lora_rows.extend(lora_df.to_dict(orient="records"))
                _save_lora_heatmap(
                    ds_name,
                    os.path.join(self.figures_dir, f"{ds_name}_lora_layer_heatmap.png"),
                    lora_df,
                )

            probs_real: List[float] = []
            probs_fake: List[float] = []
            last_maps_real: List[np.ndarray] = []
            last_maps_fake: List[np.ndarray] = []
            roll_maps_real: List[np.ndarray] = []
            roll_maps_fake: List[np.ndarray] = []
            ent_real: List[np.ndarray] = []
            ent_fake: List[np.ndarray] = []
            max_real: List[np.ndarray] = []
            max_fake: List[np.ndarray] = []
            shape_bad_count = 0

            for idx, item in enumerate(sampled_items):
                image_path = str(item["image_path"])
                label = int(item["label"])
                label_name = "real" if label == 0 else "fake"
                sample_key = hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:10]
                sample_id = f"{label_name}_{idx:02d}_{sample_key}"

                try:
                    out = self._analyze_one_sample(
                        model,
                        image_path,
                        label,
                        model_shape_key=model_hash,
                        is_clip=bool(resolved_cfg["clip"]),
                    )
                except Exception as exc:
                    self.warnings.append(f"{ds_name}: sample 분석 실패 ({image_path}) - {exc}")
                    continue

                if not out["shape_ok"]:
                    shape_bad_count += 1

                fake_prob = float(out["fake_prob"])
                if label == 0:
                    probs_real.append(fake_prob)
                    last_maps_real.append(out["last_map"])
                    roll_maps_real.append(out["rollout_map"])
                    ent_real.append(out["layer_entropy"])
                    max_real.append(out["layer_max"])
                else:
                    probs_fake.append(fake_prob)
                    last_maps_fake.append(out["last_map"])
                    roll_maps_fake.append(out["rollout_map"])
                    ent_fake.append(out["layer_entropy"])
                    max_fake.append(out["layer_max"])

                last_png = os.path.join(ds_sample_dir, f"{sample_id}_lastattn.png")
                roll_png = os.path.join(ds_sample_dir, f"{sample_id}_rollout.png")
                _save_overlay_png(out["image_np"], out["last_map"], last_png, "last attention")
                _save_overlay_png(out["image_np"], out["rollout_map"], roll_png, "attention rollout")

                sample_rows.append(
                    {
                        "dataset": ds_name,
                        "sample_id": sample_id,
                        "label": label,
                        "image_path": image_path,
                        "model_path": model_path,
                        "model_sha256": model_hash,
                        "fake_prob": fake_prob,
                        "pred_label": int(fake_prob >= 0.5),
                        "num_layers": int(out["num_layers"]),
                        "num_heads": int(out["num_heads"]),
                        "num_tokens": int(out["num_tokens"]),
                        "shape_ok": bool(out["shape_ok"]),
                        "last_skew_x": float(out["last_stats"]["skew_x"]),
                        "last_skew_y": float(out["last_stats"]["skew_y"]),
                        "last_com_x": float(out["last_stats"]["com_x"]),
                        "last_com_y": float(out["last_stats"]["com_y"]),
                        "rollout_skew_x": float(out["roll_stats"]["skew_x"]),
                        "rollout_skew_y": float(out["roll_stats"]["skew_y"]),
                        "rollout_com_x": float(out["roll_stats"]["com_x"]),
                        "rollout_com_y": float(out["roll_stats"]["com_y"]),
                        "last_layer_entropy": float(out["layer_entropy"][-1]),
                        "last_layer_max_concentration": float(out["layer_max"][-1]),
                    }
                )

            if shape_bad_count > 0:
                self.warnings.append(f"{ds_name}: shape mismatch sample 수 {shape_bad_count}")

            last_real_mean = np.mean(last_maps_real, axis=0) if last_maps_real else None
            last_fake_mean = np.mean(last_maps_fake, axis=0) if last_maps_fake else None
            roll_real_mean = np.mean(roll_maps_real, axis=0) if roll_maps_real else None
            roll_fake_mean = np.mean(roll_maps_fake, axis=0) if roll_maps_fake else None

            _save_meanmap_figure(
                ds_name,
                os.path.join(self.figures_dir, f"{ds_name}_meanmap_real_vs_fake.png"),
                last_real_mean,
                last_fake_mean,
                roll_real_mean,
                roll_fake_mean,
            )

            _save_score_hist(
                ds_name,
                os.path.join(self.figures_dir, f"{ds_name}_score_hist.png"),
                probs_real,
                probs_fake,
            )

            ent_real_mean = np.mean(ent_real, axis=0) if ent_real else None
            ent_fake_mean = np.mean(ent_fake, axis=0) if ent_fake else None
            max_real_mean = np.mean(max_real, axis=0) if max_real else None
            max_fake_mean = np.mean(max_fake, axis=0) if max_fake else None

            _save_layer_profile_plot(
                ds_name,
                os.path.join(self.figures_dir, f"{ds_name}_layer_profile.png"),
                ent_real_mean,
                ent_fake_mean,
                max_real_mean,
                max_fake_mean,
            )

            probs_all = probs_real + probs_fake
            roll_skew_x_all = [r["rollout_skew_x"] for r in sample_rows if r["dataset"] == ds_name]
            roll_skew_y_all = [r["rollout_skew_y"] for r in sample_rows if r["dataset"] == ds_name]
            last_skew_x_all = [r["last_skew_x"] for r in sample_rows if r["dataset"] == ds_name]
            last_skew_y_all = [r["last_skew_y"] for r in sample_rows if r["dataset"] == ds_name]

            dataset_rows.append(
                {
                    "dataset": ds_name,
                    "model_path": model_path,
                    "model_sha256": model_hash,
                    "video_auc": _safe_float(self.video_auc_map.get(ds_name)),
                    "sampled_real": int(sample_counts["sampled_real"]),
                    "sampled_fake": int(sample_counts["sampled_fake"]),
                    "available_real": int(sample_counts["available_real"]),
                    "available_fake": int(sample_counts["available_fake"]),
                    "score_mean_real": float(np.mean(probs_real)) if probs_real else float("nan"),
                    "score_mean_fake": float(np.mean(probs_fake)) if probs_fake else float("nan"),
                    "score_skew_all": _moment_skew(probs_all),
                    "score_skew_real": _moment_skew(probs_real),
                    "score_skew_fake": _moment_skew(probs_fake),
                    "rollout_spatial_skew_x_mean": float(np.mean(roll_skew_x_all)) if roll_skew_x_all else float("nan"),
                    "rollout_spatial_skew_y_mean": float(np.mean(roll_skew_y_all)) if roll_skew_y_all else float("nan"),
                    "last_spatial_skew_x_mean": float(np.mean(last_skew_x_all)) if last_skew_x_all else float("nan"),
                    "last_spatial_skew_y_mean": float(np.mean(last_skew_y_all)) if last_skew_y_all else float("nan"),
                    "layer_entropy_mean_real": float(np.mean(ent_real_mean)) if ent_real_mean is not None else float("nan"),
                    "layer_entropy_mean_fake": float(np.mean(ent_fake_mean)) if ent_fake_mean is not None else float("nan"),
                    "layer_maxconc_mean_real": float(np.mean(max_real_mean)) if max_real_mean is not None else float("nan"),
                    "layer_maxconc_mean_fake": float(np.mean(max_fake_mean)) if max_fake_mean is not None else float("nan"),
                    "classifier_weight_norm": float(weight_summary["classifier_weight_norm"]),
                    "classifier_bias_norm": float(weight_summary["classifier_bias_norm"]),
                    "layernorm_param_l2": float(weight_summary["layernorm_param_l2"]),
                    "layernorm_param_count": int(weight_summary["layernorm_param_count"]),
                    "lora_rank": int(resolved_cfg["lora_rank"]),
                    "lora_alpha": int(resolved_cfg["lora_alpha"]),
                    "clip": bool(resolved_cfg["clip"]),
                    "clip_model": str(resolved_cfg["clip_model"]),
                    "shape_mismatch_count": int(shape_bad_count),
                }
            )

            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        sample_df = pd.DataFrame(sample_rows)
        dataset_df = pd.DataFrame(dataset_rows)
        lora_df = pd.DataFrame(lora_rows)
        hash_df = pd.DataFrame(hash_rows)

        if not hash_df.empty:
            dup_groups = {}
            for sha, group in hash_df.groupby("sha256"):
                datasets_sorted = sorted(group["dataset"].tolist())
                dup_groups[sha] = datasets_sorted
            hash_df["is_duplicate"] = hash_df["sha256"].map(lambda x: len(dup_groups.get(x, [])) > 1)
            hash_df["duplicate_group"] = hash_df["sha256"].map(lambda x: ",".join(dup_groups.get(x, [])))

            if not dataset_df.empty:
                dataset_df = dataset_df.merge(
                    hash_df[["dataset", "is_duplicate", "duplicate_group"]],
                    on="dataset",
                    how="left",
                )

        sample_csv = os.path.join(self.tables_dir, "sample_metrics.csv")
        dataset_csv = os.path.join(self.tables_dir, "dataset_metrics.csv")
        lora_csv = os.path.join(self.tables_dir, "lora_layer_stats.csv")
        hash_csv = os.path.join(self.tables_dir, "model_hashes.csv")

        sample_df.to_csv(sample_csv, index=False)
        dataset_df.to_csv(dataset_csv, index=False)
        lora_df.to_csv(lora_csv, index=False)
        hash_df.to_csv(hash_csv, index=False)

        self._write_report(dataset_df, hash_df)

        _log(f"saved: {sample_csv}")
        _log(f"saved: {dataset_csv}")
        _log(f"saved: {lora_csv}")
        _log(f"saved: {hash_csv}")

    def _write_report(self, dataset_df: pd.DataFrame, hash_df: pd.DataFrame):
        report_path = os.path.join(self.cfg.output_dir, "report.md")

        def _md_table(headers: List[str], rows: List[List[object]]) -> str:
            lines = []
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                vals = []
                for v in row:
                    if isinstance(v, float):
                        if np.isnan(v):
                            vals.append("")
                        else:
                            vals.append(f"{v:.6f}")
                    else:
                        vals.append(str(v))
                lines.append("| " + " | ".join(vals) + " |")
            return "\n".join(lines)

        lines: List[str] = []
        run_name = os.path.basename(os.path.dirname(os.path.abspath(self.cfg.weights_root))) or "best_models"
        lines.append(f"# {run_name} Best Models Analysis Report")
        lines.append("")
        lines.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- Weights root: `{self.cfg.weights_root}`")
        lines.append(f"- Output dir: `{self.cfg.output_dir}`")
        lines.append(f"- Sample policy: real={self.cfg.sample_per_class}, fake={self.cfg.sample_per_class} per dataset")
        lines.append(f"- Device used: `{self.device}`")
        lines.append("")

        if not dataset_df.empty:
            lines.append("## Dataset Summary")
            lines.append("")
            headers = [
                "dataset",
                "video_auc",
                "sampled_real",
                "sampled_fake",
                "score_skew_all",
                "rollout_spatial_skew_x_mean",
                "rollout_spatial_skew_y_mean",
                "is_duplicate",
            ]
            rows = dataset_df[headers].values.tolist()
            lines.append(_md_table(headers, rows))
            lines.append("")

        if not hash_df.empty:
            lines.append("## Model Hash Duplicates")
            lines.append("")
            hash_rows = []
            for sha, group in hash_df.groupby("sha256"):
                datasets = sorted(group["dataset"].tolist())
                hash_rows.append([sha, len(datasets), ", ".join(datasets)])
            hash_rows.sort(key=lambda x: (-x[1], x[0]))
            lines.append(_md_table(["sha256", "count", "datasets"], hash_rows))
            lines.append("")

        if not dataset_df.empty:
            lines.append("## Per-dataset Figures")
            lines.append("")
            for ds_name in dataset_df["dataset"].tolist():
                mean_rel = f"figures/{ds_name}_meanmap_real_vs_fake.png"
                hist_rel = f"figures/{ds_name}_score_hist.png"
                heatmap_rel = f"figures/{ds_name}_lora_layer_heatmap.png"
                profile_rel = f"figures/{ds_name}_layer_profile.png"

                lines.append(f"### {ds_name}")
                lines.append("")
                lines.append(f"- Mean maps: `{mean_rel}`")
                lines.append(f"- Score histogram: `{hist_rel}`")
                if os.path.exists(os.path.join(self.cfg.output_dir, heatmap_rel)):
                    lines.append(f"- LoRA layer heatmap: `{heatmap_rel}`")
                else:
                    lines.append("- LoRA layer heatmap: (not generated)")
                lines.append(f"- Layer profile: `{profile_rel}`")
                lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            lines.append("")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        _log(f"saved: {report_path}")


def run_analysis(
    weights_root: str,
    output_dir: str,
    sample_per_class: int,
    seed: int,
    device: str,
    num_workers: int,
    datasets: Optional[List[str]] = None,
):
    cfg = AnalysisConfig(
        weights_root=weights_root,
        output_dir=output_dir,
        sample_per_class=int(sample_per_class),
        seed=int(seed),
        device=str(device),
        num_workers=int(num_workers),
        datasets=datasets,
    )
    analyzer = ClipBestModelAnalyzer(cfg)
    analyzer.run()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze best_models with attention maps and weight statistics",
    )
    parser.add_argument(
        "--weights_root",
        type=str,
        default="/workspace/likelihood_submission_ln/weights/CLIP-L14_pissa_LN-01/best_models",
        help="Root directory containing dataset-specific best model folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/likelihood_submission_ln/weights/CLIP-L14_pissa_LN-01/best_models/analysis_paper",
        help="Output directory for figures/tables/report",
    )
    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=20,
        help="Number of samples per class(real/fake) for each dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Preferred device (cuda falls back to cpu when unavailable)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Reserved worker count for compatibility",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated dataset names to restrict analysis",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()] if args.datasets else None

    run_analysis(
        weights_root=args.weights_root,
        output_dir=args.output_dir,
        sample_per_class=args.sample_per_class,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        datasets=datasets,
    )


if __name__ == "__main__":
    main()
