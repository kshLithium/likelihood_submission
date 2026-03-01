import os
import yaml
import torch
import torch.nn.functional as F
from PIL import ImageFile
import timm.data

# =========================================================
# [Config Loading Section] - config.yaml 로드
# =========================================================
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_CURRENT_DIR, "..", "config", "config.yaml")

if os.path.exists(_CONFIG_PATH):
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f) or {}
else:
    # 파일이 없을 경우(단위 테스트 등)를 대비한 기본값
    _cfg = {}

# [핵심 설정] 잘리거나 손상된 이미지도 에러 없이 로드 시도
ImageFile.LOAD_TRUNCATED_IMAGES = True

# [Hotfix] timm 호환성 패치
def _apply_timm_hotfix():
    if not hasattr(timm.data, "ImageNetInfo"):
        class MockImageNetInfo:
            def __init__(self, *args, **kwargs):
                pass

            def index_to_description(self, index):
                return ""

        timm.data.ImageNetInfo = MockImageNetInfo

    if not hasattr(timm.data, "infer_imagenet_subset"):
        timm.data.infer_imagenet_subset = lambda *args, **kwargs: None


_apply_timm_hotfix()

def _parse_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    sval = str(value).strip().lower()
    if sval in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if sval in {"0", "false", "f", "no", "n", "off", "", "none", "null"}:
        return False
    return default


def _parse_optional_str(value):
    if value is None:
        return None
    sval = str(value).strip()
    if sval.lower() in {"", "none", "null"}:
        return None
    return sval


def _get_with_env(cfg_key, env_key, default, cast):
    if env_key:
        raw = os.environ.get(env_key)
        if raw is not None:
            try:
                return cast(raw)
            except Exception as exc:
                raise ValueError(
                    f"Invalid env value for {env_key}={raw!r} "
                    f"(expected {cast.__name__})"
                ) from exc
    return cast(_cfg.get(cfg_key, default))


# 다른 파일에서 import할 변수들
DATA_ROOT_BASE = _cfg.get("data_root_base", "/data/train_data")
JSON_BASE = _cfg.get("json_base", "/data/train_data/json")

MODEL_ID = _cfg.get("backbone_model", "facebook/dinov3-vith16plus-pretrain-lvd1689m")
_DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"
USE_CLIP = _get_with_env("clip", "USE_CLIP", False, lambda v: _parse_bool(v, False))
CLIP_MODEL = _get_with_env(
    "clip_model",
    "CLIP_MODEL",
    _DEFAULT_CLIP_MODEL,
    lambda v: _parse_optional_str(v) or _DEFAULT_CLIP_MODEL,
)
BACKBONE_DIR = _cfg.get("backbone_dir", "./backbone")
OUTPUT_DIR = _get_with_env("output_dir", "OUTPUT_DIR", "./outputs", str)

IMAGE_SIZE = int(_cfg.get("image_size", 224))
NUM_FRAMES_PER_VIDEO = int(_cfg.get("num_frames_per_video", 12))
TARGET_COMPRESSION = _cfg.get("target_compression", "c23")

NUM_EPOCHS = int(_cfg.get("epochs", 1))
BATCH_SIZE = _get_with_env("batch_size", "BATCH_SIZE", 32, int)
LEARNING_RATE = float(_cfg.get("learning_rate", 1e-4))
DDP_TIMEOUT_SECONDS = int(_cfg.get("ddp_timeout_seconds", 10800))

MULTITASK = _parse_bool(_cfg.get("multitask", False), False)
LABEL_SMOOTHING = float(_cfg.get("label_smoothing", 0.0))

SEED = _cfg.get("seed", None)
NUM_WORKERS = int(_cfg.get("num_workers", 8))
USE_PEFT = _get_with_env("use_peft", "USE_PEFT", False, lambda v: _parse_bool(v, False))
LORA_RANK = _get_with_env("lora_rank", "LORA_RANK", 1, int)
LORA_ALPHA = _get_with_env("lora_alpha", "LORA_ALPHA", 2, int)

# GenD-style LN fine-tuning settings
_DEFAULT_LAYERNORM_PATTERNS = ["norm1", "norm2", "norm"]
_layernorm_patterns = _cfg.get("layernorm_patterns", _DEFAULT_LAYERNORM_PATTERNS)
if isinstance(_layernorm_patterns, str):
    _layernorm_patterns = [p.strip() for p in _layernorm_patterns.split(",") if p.strip()]
if not isinstance(_layernorm_patterns, (list, tuple)) or len(_layernorm_patterns) == 0:
    _layernorm_patterns = _DEFAULT_LAYERNORM_PATTERNS

USE_LAYERNORM_TUNING = _get_with_env(
    "use_layernorm_tuning",
    "USE_LAYERNORM_TUNING",
    True,
    lambda v: _parse_bool(v, True),
)
LAYERNORM_PATTERNS = [str(p).lower() for p in _layernorm_patterns]
USE_GEND_L2_HEAD = _parse_bool(_cfg.get("use_gend_l2_head", True), True)
LAYERNORM_LR_SCALE = _get_with_env("layernorm_lr_scale", "LAYERNORM_LR_SCALE", 1.0, float)

# Effort (Orthogonal Subspace Decomposition) settings
USE_EFFORT = _parse_bool(_cfg.get("use_effort", False), False)
EFFORT_RANK = int(_cfg.get("effort_rank", 16))
EFFORT_ORTHO_WEIGHT = float(_cfg.get("effort_ortho_weight", 0.1))
EFFORT_KEEPSV_WEIGHT = float(_cfg.get("effort_keepsv_weight", 0.01))

WANDB_PROJECT = _cfg.get("wandb_project", "Hecto_deepfake")
WANDB_RUN_NAME = _get_with_env("wandb_run_name", "WANDB_RUN_NAME", None, _parse_optional_str)

# Test mode
TEST_MODE = _get_with_env("test_mode", "TEST_MODE", False, lambda v: _parse_bool(v, False))
TEST_NUM_FRAMES = int(_cfg.get("test_num_frames", 32))
TEST_DATASET_PRESET = _get_with_env(
    "test_dataset_preset",
    "TEST_DATASET_PRESET",
    "ff8",
    lambda v: str(v).strip().lower() or "ff8",
)


# =========================================================
# [Utility Functions Section] - 기존 유틸 함수들
# =========================================================

def is_main_process():
    """
    현재 프로세스가 메인 프로세스(Rank 0)인지 확인합니다.
    """
    return int(os.environ.get("RANK", "0")) == 0

def label_smoothing_cross_entropy(pred, target, smoothing=0.1):
    """
    Cross entropy loss with label smoothing for multi-class classification.
    """
    if smoothing <= 0.0:
        return F.cross_entropy(pred, target)

    num_classes = pred.size(1)
    with torch.no_grad():
        true_dist = torch.full_like(pred, fill_value=smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)

    log_probs = F.log_softmax(pred, dim=1)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))
