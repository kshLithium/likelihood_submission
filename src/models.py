import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, CLIPVisionModel
from transformers.modeling_outputs import ImageClassifierOutput

from .utils import (
    MULTITASK,
    LABEL_SMOOTHING,
    BACKBONE_DIR,
    USE_EFFORT,
    EFFORT_RANK,
    LORA_RANK,
    LORA_ALPHA,
    USE_LAYERNORM_TUNING,
    LAYERNORM_PATTERNS,
    USE_GEND_L2_HEAD,
    USE_CLIP,
    CLIP_MODEL,
)
from .utils import label_smoothing_cross_entropy
from .effort import apply_svd_residual_to_model


def _resolve_local_model_path(model_id):
    """Resolve a model path if it exists locally, otherwise return None."""
    if not model_id:
        return None
    if os.path.isdir(model_id):
        return model_id
    candidate = os.path.join(BACKBONE_DIR, model_id)
    if os.path.isdir(candidate):
        return candidate
    return None


def _resolve_local_backbone_path(model_id):
    """
    Resolve local DINO backbone path only.
    Priority:
      1) model_id itself as a local directory
      2) <BACKBONE_DIR>/<model_id>
    """
    resolved = _resolve_local_model_path(model_id)
    if resolved is not None:
        return resolved
    candidate = os.path.join(BACKBONE_DIR, model_id)

    raise FileNotFoundError(
        "Local backbone not found.\n"
        f"  - model_id path: {os.path.abspath(model_id)}\n"
        f"  - backbone path: {os.path.abspath(candidate)}\n"
        "Only local loading is supported. Put the backbone files under "
        "<backbone_dir>/<backbone_model> or set backbone_model to a local directory path."
    )


def _enable_layernorm_finetuning(backbone, patterns):
    """Enable gradients only for backbone parameters matching LayerNorm keywords."""
    patterns = [str(p).lower() for p in patterns if str(p).strip()]
    if not patterns:
        patterns = ["norm1", "norm2", "norm"]

    tuned_names = []
    for name, param in backbone.named_parameters():
        lname = name.lower()
        if any(pattern in lname for pattern in patterns):
            param.requires_grad = True
            tuned_names.append(name)

    return tuned_names


def _collect_layernorm_modules_to_save(backbone, patterns):
    """
    Collect backbone module names to persist via PEFT `modules_to_save`.
    We derive module names from parameter names that match LN patterns so this
    stays aligned with `_enable_layernorm_finetuning`.
    """
    patterns = [str(p).lower() for p in patterns if str(p).strip()]
    if not patterns:
        patterns = ["norm1", "norm2", "norm"]

    module_names = set()
    for name, _ in backbone.named_parameters():
        lname = name.lower()
        if not any(pattern in lname for pattern in patterns):
            continue
        if "." not in name:
            continue
        module_name = name.rsplit(".", 1)[0]
        module_names.add(f"backbone.{module_name}")

    return sorted(module_names)


class DINOv3ForClassification(nn.Module):
    def __init__(self, model_id, class_weights=None):
        super().__init__()
        self.is_clip = bool(USE_CLIP)

        if self.is_clip:
            local_clip_path = _resolve_local_model_path(CLIP_MODEL)
            clip_source = local_clip_path if local_clip_path is not None else CLIP_MODEL
            local_only = local_clip_path is not None
            source_kind = "local" if local_only else "huggingface_hub"
            print(f"[*] Encoder type: CLIP")
            print(f"[*] Loading Backbone (CLIP/{source_kind}): {clip_source}")
            try:
                self.backbone = CLIPVisionModel.from_pretrained(
                    clip_source,
                    local_files_only=local_only,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load CLIP vision encoder.\n"
                    f"  - requested clip_model: {CLIP_MODEL!r}\n"
                    f"  - tried source: {clip_source!r}\n"
                    f"  - local_files_only: {local_only}\n"
                    "Check the clip_model path/name and make sure weights are available "
                    "locally or cached from Hugging Face."
                ) from exc
        else:
            local_backbone_path = _resolve_local_backbone_path(model_id)
            print("[*] Encoder type: DINO")
            print(f"[*] Loading Backbone (local-only): {local_backbone_path}")
            self.backbone = AutoModel.from_pretrained(
                local_backbone_path,
                trust_remote_code=True,
                local_files_only=True,
            )

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.use_gend_l2_head = bool(USE_GEND_L2_HEAD)
        self.use_layernorm_tuning = bool(USE_LAYERNORM_TUNING and not USE_EFFORT)
        if USE_LAYERNORM_TUNING and USE_EFFORT:
            print("[*] LayerNorm tuning requested but disabled because EFFORT is enabled.")
        if self.use_layernorm_tuning:
            tuned = _enable_layernorm_finetuning(self.backbone, LAYERNORM_PATTERNS)
            trainable_backbone = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total_backbone = sum(p.numel() for p in self.backbone.parameters())
            ratio = 100.0 * trainable_backbone / max(1, total_backbone)
            print(
                f"[*] GenD LN tuning: enabled for {len(tuned)} tensors, "
                f"trainable backbone params={trainable_backbone}/{total_backbone} ({ratio:.4f}%)"
            )
        else:
            print("[*] GenD LN tuning: disabled.")

        self.config = self.backbone.config
        self.hidden_size = getattr(self.config, "hidden_size", getattr(self.config, "embed_dim", 1024))

        self.num_labels = 3 if MULTITASK else 1
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
        print(f"[*] GenD L2-normalized head: {'enabled' if self.use_gend_l2_head else 'disabled'}")

        # label smoothing value (from CLI). Apply to CE and (optionally) BCE when >0.
        self.label_smoothing = float(LABEL_SMOOTHING) if LABEL_SMOOTHING is not None else 0.0

        if MULTITASK:
            if class_weights is not None and hasattr(class_weights, '__len__') and len(class_weights) == 3:
                weight_tensor = torch.tensor(class_weights).float()
                self.loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
                print(f"[*] Loss initialized (CE, weights: {class_weights})")
            else:
                self.loss_fct = nn.CrossEntropyLoss()
                print("[*] Loss initialized (CE, No weights)")
        else:
            if class_weights is not None:
                if hasattr(class_weights, '__getitem__') and len(class_weights) > 0:
                    weight_val = class_weights[0] if len(class_weights) == 1 else class_weights[1]
                elif isinstance(class_weights, (float, int)):
                    weight_val = class_weights
                else:
                    weight_val = 1.0

                pos_weight_tensor = torch.tensor(weight_val).float()
                self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
                print(f"[*] Loss initialized with pos_weight: {weight_val}")
            else:
                self.loss_fct = nn.BCEWithLogitsLoss()
                print("[*] Loss initialized (BCE, No weights)")

    def forward(self, pixel_values, labels=None):
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            cls_token = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_token = outputs.pooler_output
        else:
            raise RuntimeError(
                "Backbone output does not contain last_hidden_state or pooler_output."
            )
        if cls_token.dtype != self.classifier.weight.dtype:
            cls_token = cls_token.to(self.classifier.weight.dtype)
        if self.use_gend_l2_head:
            cls_token = F.normalize(cls_token, p=2, dim=1)
        logits = self.classifier(cls_token)

        loss = None
        if labels is not None:
            if MULTITASK:
                # For multi-class, use label smoothed CE if smoothing > 0
                if getattr(self, 'label_smoothing', 0.0) > 0.0:
                    loss = label_smoothing_cross_entropy(logits, labels.long(), smoothing=self.label_smoothing)
                else:
                    loss = self.loss_fct(logits, labels.long())
            else:
                # Binary case: BCEWithLogitsLoss accepts targets in [0,1]. If smoothing enabled, move labels toward opposite class.
                targets = labels.float().unsqueeze(1)
                if getattr(self, 'label_smoothing', 0.0) > 0.0:
                    eps = float(self.label_smoothing)
                    # Positive -> 1 - eps, Negative -> eps
                    targets = targets * (1.0 - eps) + (1.0 - targets) * eps
                loss = self.loss_fct(logits, targets)

        return ImageClassifierOutput(loss=loss, logits=logits)


def build_peft_model(model):
    linear_names = [n for n, m in model.backbone.named_modules() if isinstance(m, nn.Linear)]
    target_modules = [p for p in ["q_proj", "k_proj", "v_proj", "o_proj"] if any(p in n for n in linear_names)]
    if not target_modules:
        target_modules = [p for p in ["up_proj", "down_proj"] if any(p in n for n in linear_names)]
    if not target_modules:
        target_modules = list({n.split(".")[-1] for n in linear_names})
    print(f"[*] Target Modules: {target_modules}")

    ln_modules_to_save = []
    if getattr(model, "use_layernorm_tuning", False):
        ln_modules_to_save = _collect_layernorm_modules_to_save(model.backbone, LAYERNORM_PATTERNS)
        preview = ", ".join(ln_modules_to_save[:5]) if ln_modules_to_save else "(none)"
        print(
            f"[*] LN modules_to_save: {len(ln_modules_to_save)} modules "
            f"(preview: {preview})"
        )
    else:
        print("[*] LN modules_to_save: disabled")

    modules_to_save = ["classifier"] + ln_modules_to_save

    print(f"[*] LoRA config: r={LORA_RANK}, alpha={LORA_ALPHA}")
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=modules_to_save,
        init_lora_weights="pissa",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def build_effort_model(model, r=None):
    """Apply Effort (SVD Residual) fine-tuning to the backbone's self-attention layers."""
    if r is None:
        r = EFFORT_RANK

    # Detect target modules from backbone
    linear_names = [n for n, m in model.backbone.named_modules() if isinstance(m, nn.Linear)]
    target_modules = [p for p in ["q_proj", "k_proj", "v_proj", "o_proj"] if any(p in n for n in linear_names)]
    if not target_modules:
        target_modules = [p for p in ["up_proj", "down_proj"] if any(p in n for n in linear_names)]
    if not target_modules:
        target_modules = list({n.split(".")[-1] for n in linear_names})
    print(f"[Effort] Target Modules: {target_modules}")

    model = apply_svd_residual_to_model(model, r=r, target_modules=target_modules)
    return model
