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
        _cfg = yaml.safe_load(f)
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

# 다른 파일에서 import할 변수들
DATA_ROOT_BASE = _cfg.get("data_root_base", "/data/train_data")
JSON_BASE = _cfg.get("json_base", "/data/train_data/json")

MODEL_ID = _cfg.get("backbone_model", "facebook/dinov3-vith16plus-pretrain-lvd1689m")
OUTPUT_DIR = _cfg.get("output_dir", "./outputs")

IMAGE_SIZE = int(_cfg.get("image_size", 224))
NUM_FRAMES_PER_VIDEO = int(_cfg.get("num_frames_per_video", 12))
TARGET_COMPRESSION = _cfg.get("target_compression", "c23")

NUM_EPOCHS = int(_cfg.get("epochs", 1))
BATCH_SIZE = int(_cfg.get("batch_size", 32))
LEARNING_RATE = float(_cfg.get("learning_rate", 1e-4))

MULTITASK = _cfg.get("multitask", False)
LABEL_SMOOTHING = float(_cfg.get("label_smoothing", 0.0))

SEED = _cfg.get("seed", None)
NUM_WORKERS = int(_cfg.get("num_workers", 8))
USE_PEFT = bool(_cfg.get("use_peft", False))


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