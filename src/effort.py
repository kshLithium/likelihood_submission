"""
Effort: Orthogonal Subspace Decomposition for Efficient Fine-Tuning.

Instead of LoRA/PiSSA, this decomposes pre-trained weights via SVD:
  W = U_r @ diag(S_r) @ Vh_r   +   U_res @ diag(S_res) @ Vh_res
         (frozen top-r)                  (trainable residual)

Two regularisation losses keep the residual well-behaved:
  1) Orthogonality loss  – U/V columns stay orthonormal
  2) Keep-SV loss        – Frobenius norm doesn't drift from the original
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
# SVDResidualLinear
# ------------------------------------------------------------------ #
class SVDResidualLinear(nn.Module):
    """Drop-in replacement for nn.Linear with SVD-based residual training."""

    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        # Main (frozen) weight
        self.weight_main = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Placeholders – filled by `replace_with_svd_residual`
        self.S_r: nn.Parameter | None = None
        self.U_r: nn.Parameter | None = None
        self.V_r: nn.Parameter | None = None

        self.S_residual: nn.Parameter | None = None
        self.U_residual: nn.Parameter | None = None
        self.V_residual: nn.Parameter | None = None

        self.weight_original_fnorm: float | None = None

    # ---- forward -------------------------------------------------- #
    def forward(self, x):
        if self.S_residual is not None:
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)

    # ---- regularisation losses ------------------------------------ #
    def compute_orthogonal_loss(self):
        """|| concat(U_r, U_res) @ concat(…)^T - I || + same for V."""
        if self.S_residual is None:
            return torch.tensor(0.0, device=self.weight_main.device)

        U_cat = torch.cat([self.U_r, self.U_residual], dim=1)       # (out, full_rank)
        V_cat = torch.cat([self.V_r, self.V_residual], dim=0)       # (full_rank, in)

        UUT = U_cat @ U_cat.t()
        VVT = V_cat @ V_cat.t()

        I_U = torch.eye(UUT.size(0), device=UUT.device, dtype=UUT.dtype)
        I_V = torch.eye(VVT.size(0), device=VVT.device, dtype=VVT.dtype)

        loss = 0.5 * torch.norm(UUT - I_U, p="fro") + 0.5 * torch.norm(VVT - I_V, p="fro")
        return loss

    def compute_keepsv_loss(self):
        """| ||W_current||_F^2 - ||W_original||_F^2 |"""
        if self.S_residual is None or self.weight_original_fnorm is None:
            return torch.tensor(0.0, device=self.weight_main.device)

        W_cur = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        cur_fnorm = torch.norm(W_cur, p="fro")
        loss = torch.abs(cur_fnorm ** 2 - self.weight_original_fnorm ** 2)
        return loss


# ------------------------------------------------------------------ #
# Factory: nn.Linear  →  SVDResidualLinear
# ------------------------------------------------------------------ #
def replace_with_svd_residual(module: nn.Linear, r: int) -> SVDResidualLinear:
    """Create an SVDResidualLinear initialised from a pre-trained Linear."""
    in_f, out_f = module.in_features, module.out_features
    has_bias = module.bias is not None

    new = SVDResidualLinear(in_f, out_f, r, bias=has_bias, init_weight=module.weight.data.clone())
    if has_bias and module.bias is not None:
        new.bias.data.copy_(module.bias.data)

    new.weight_original_fnorm = torch.norm(module.weight.data, p="fro").item()

    # SVD  (torch.linalg.svd → U, S, Vh)
    U, S, Vh = torch.linalg.svd(module.weight.data.float(), full_matrices=False)
    r = min(r, len(S))

    # 1. Frozen principal components
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    new.weight_main.data.copy_((U_r @ torch.diag(S_r) @ Vh_r).to(module.weight.dtype))

    # 2. Trainable residual
    U_res = U[:, r:]
    S_res = S[r:]
    Vh_res = Vh[r:, :]

    if len(S_res) > 0:
        new.S_residual = nn.Parameter(S_res.clone().contiguous().to(module.weight.dtype))
        new.U_residual = nn.Parameter(U_res.clone().contiguous().to(module.weight.dtype))
        new.V_residual = nn.Parameter(Vh_res.clone().contiguous().to(module.weight.dtype))

        new.S_r = nn.Parameter(S_r.clone().contiguous().to(module.weight.dtype), requires_grad=False)
        new.U_r = nn.Parameter(U_r.clone().contiguous().to(module.weight.dtype), requires_grad=False)
        new.V_r = nn.Parameter(Vh_r.clone().contiguous().to(module.weight.dtype), requires_grad=False)

    return new


# ------------------------------------------------------------------ #
# Recursive applicator
# ------------------------------------------------------------------ #
_DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]


def apply_svd_residual_to_model(
    model: nn.Module,
    r: int = 16,
    target_modules: list[str] | None = None,
):
    """
    Walk the model tree, replacing target Linear layers with SVDResidualLinear.
    Then freeze everything except the residual SVD parameters and biases.
    """
    if target_modules is None:
        target_modules = _DEFAULT_TARGETS

    replaced = 0

    def _recurse(parent: nn.Module):
        nonlocal replaced
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear) and any(t in name for t in target_modules):
                new_mod = replace_with_svd_residual(child, r)
                setattr(parent, name, new_mod)
                replaced += 1
            else:
                _recurse(child)

    _recurse(model)

    # Freeze / unfreeze
    for pname, param in model.named_parameters():
        if any(k in pname for k in ("S_residual", "U_residual", "V_residual", "bias", "classifier")):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Report
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Effort] Replaced {replaced} Linear layers (rank r={r})")
    print(f"[Effort] Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model


# ------------------------------------------------------------------ #
# Loss aggregation helpers  (used by CustomTrainer)
# ------------------------------------------------------------------ #
def collect_effort_losses(model: nn.Module):
    """Return summed orthogonality and keep-SV losses over all SVDResidualLinear modules."""
    ortho_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    keepsv_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0

    for m in model.modules():
        if isinstance(m, SVDResidualLinear):
            ortho_loss = ortho_loss + m.compute_orthogonal_loss()
            keepsv_loss = keepsv_loss + m.compute_keepsv_loss()
            count += 1

    return ortho_loss, keepsv_loss, count
