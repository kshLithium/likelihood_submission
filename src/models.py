import os
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel
from transformers.modeling_outputs import ImageClassifierOutput

from .utils import MULTITASK, LABEL_SMOOTHING, BACKBONE_DIR
from .utils import label_smoothing_cross_entropy


def _resolve_local_backbone_path(model_id):
    """
    Resolve local backbone path only.
    Priority:
      1) model_id itself as a local directory
      2) <BACKBONE_DIR>/<model_id>
    """
    if os.path.isdir(model_id):
        return model_id

    candidate = os.path.join(BACKBONE_DIR, model_id)
    if os.path.isdir(candidate):
        return candidate

    raise FileNotFoundError(
        "Local backbone not found.\n"
        f"  - model_id path: {os.path.abspath(model_id)}\n"
        f"  - backbone path: {os.path.abspath(candidate)}\n"
        "Only local loading is supported. Put the backbone files under "
        "<backbone_dir>/<backbone_model> or set backbone_model to a local directory path."
    )


class DINOv3ForClassification(nn.Module):
    def __init__(self, model_id, class_weights=None):
        super().__init__()
        local_backbone_path = _resolve_local_backbone_path(model_id)
        print(f"[*] Loading Backbone (local-only): {local_backbone_path}")
        self.backbone = AutoModel.from_pretrained(
            local_backbone_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.config = self.backbone.config
        self.hidden_size = getattr(self.config, "hidden_size", getattr(self.config, "embed_dim", 1024))

        self.num_labels = 3 if MULTITASK else 1
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

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
        outputs = self.backbone(pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
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

    peft_config = LoraConfig(
        r=1,
        lora_alpha=2,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
        init_lora_weights="pissa",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
