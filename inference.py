"""
Unified deepfake competition inference script.
Includes:
1) data preprocessing (face detection/cropping)
2) model loading
3) model inference
4) final submission CSV generation
"""

import argparse
import heapq
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import imageio.v3 as iio
import numpy as np
import onnxruntime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, CLIPVisionModel
from transformers.modeling_outputs import ImageClassifierOutput

# timm compatibility patch (required by some transformers backbones)
import timm.data

if not hasattr(timm.data, "ImageNetInfo"):
    class MockImageNetInfo:
        def __init__(self, *args, **kwargs):
            pass

        def index_to_description(self, index):
            return ""

    timm.data.ImageNetInfo = MockImageNetInfo

if not hasattr(timm.data, "infer_imagenet_subset"):
    timm.data.infer_imagenet_subset = lambda *args, **kwargs: None


# ==============================================================================
# Constants
# ==============================================================================
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".jfif", ".webp")
CROP_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".jfif")
ORIGINAL_EXTS = (".png", ".jpg", ".jpeg", ".jfif", ".webp", ".mp4", ".mov", ".avi", ".mkv", ".webm")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
DEFAULT_INPUT_ROOT = os.path.join(SCRIPT_DIR, "test_data")
DEFAULT_CROPPED_DIR = os.path.join(SCRIPT_DIR, "cropped")
DEFAULT_OUTPUT_CSV = os.path.join(SCRIPT_DIR, "result", "submission.csv")
DETECTOR_MODEL_FILE = os.path.join(SCRIPT_DIR, "retinaface", "det_10g.onnx")

IMAGE_SIZE = 224
VIDEO_AGG = "topk_confidence"
DEFAULT_LAYERNORM_PATTERNS = ["norm1", "norm2", "norm"]
DEFAULT_USE_CLIP = False
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"
DEFAULT_BACKBONE_DIR = "./backbone"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Embedded backbone config (formerly model/model_config/config.json)
DINOV3_BACKBONE_CONFIG = {
    "apply_layernorm": True,
    "architectures": ["DINOv3ViTModel"],
    "attention_dropout": 0.0,
    "drop_path_rate": 0.0,
    "dtype": "float16",
    "hidden_act": "silu",
    "hidden_size": 1280,
    "image_size": 224,
    "initializer_range": 0.02,
    "intermediate_size": 5120,
    "key_bias": False,
    "layer_norm_eps": 1e-05,
    "layerscale_value": 1.0,
    "mlp_bias": True,
    "model_type": "dinov3_vit",
    "num_attention_heads": 20,
    "num_channels": 3,
    "num_hidden_layers": 32,
    "num_register_tokens": 4,
    "out_features": ["stage32"],
    "out_indices": [32],
    "patch_size": 16,
    "pos_embed_jitter": None,
    "pos_embed_rescale": 2.0,
    "pos_embed_shift": None,
    "proj_bias": True,
    "query_bias": True,
    "reshape_hidden_states": True,
    "rope_theta": 100.0,
    "stage_names": [
        "stem",
        "stage1",
        "stage2",
        "stage3",
        "stage4",
        "stage5",
        "stage6",
        "stage7",
        "stage8",
        "stage9",
        "stage10",
        "stage11",
        "stage12",
        "stage13",
        "stage14",
        "stage15",
        "stage16",
        "stage17",
        "stage18",
        "stage19",
        "stage20",
        "stage21",
        "stage22",
        "stage23",
        "stage24",
        "stage25",
        "stage26",
        "stage27",
        "stage28",
        "stage29",
        "stage30",
        "stage31",
        "stage32",
    ],
    "transformers_version": "5.0.0",
    "use_gated_mlp": True,
    "value_bias": True,
}


# ==============================================================================
# Runtime config helpers
# ==============================================================================
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


def _resolve_backbone_dir(path_value):
    raw = _parse_optional_str(path_value) or DEFAULT_BACKBONE_DIR
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(os.path.join(SCRIPT_DIR, raw))


def _resolve_local_model_path(model_id, backbone_dir):
    if not model_id:
        return None
    if os.path.isdir(model_id):
        return model_id
    candidate = os.path.join(backbone_dir, model_id)
    if os.path.isdir(candidate):
        return candidate
    return None


def _load_runtime_config():
    config_path = os.path.join(SCRIPT_DIR, "config", "config.yaml")
    cfg = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"[Warn] Failed to read config.yaml ({config_path}): {exc}")
            cfg = {}

    use_clip = _parse_bool(cfg.get("clip", DEFAULT_USE_CLIP), DEFAULT_USE_CLIP)
    clip_model = _parse_optional_str(cfg.get("clip_model", DEFAULT_CLIP_MODEL)) or DEFAULT_CLIP_MODEL
    backbone_dir = _resolve_backbone_dir(cfg.get("backbone_dir", DEFAULT_BACKBONE_DIR))
    return {
        "path": config_path,
        "has_config": os.path.exists(config_path),
        "use_clip": bool(use_clip),
        "clip_model": clip_model,
        "backbone_dir": backbone_dir,
    }


RUNTIME_CONFIG = _load_runtime_config()


# ==============================================================================
# Common helpers
# ==============================================================================
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def topk_confidence_mean(probs, topk):
    if not probs:
        return 0.5
    arr = np.asarray(probs, dtype=np.float32)
    if arr.size == 0:
        return 0.5
    k = int(topk) if topk is not None else 1
    k = max(1, min(k, arr.size))
    conf = np.abs(arr - 0.5)
    idx = np.argpartition(-conf, k - 1)[:k]
    return float(arr[idx].mean())


def load_original_extension_map(original_data_root):
    if not original_data_root or not os.path.isdir(original_data_root):
        print(f"[Warn] original_data_root not found: {original_data_root}")
        return {}

    ext_map = {}
    for root, _, files in os.walk(original_data_root):
        for filename in files:
            stem, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext not in ORIGINAL_EXTS:
                continue
            if stem in ext_map and ext_map[stem] != ext:
                continue
            ext_map[stem] = ext

    print(f"[*] Loaded original extension map: {len(ext_map)} items from {original_data_root}")
    return ext_map


def with_submission_ext(item_id, item_type, original_ext_map):
    if os.path.splitext(item_id)[1]:
        return item_id
    if item_id in original_ext_map:
        return f"{item_id}{original_ext_map[item_id]}"
    return f"{item_id}.mp4" if item_type == "video" else f"{item_id}.png"


def collect_expected_filenames(original_data_root):
    expected = set()
    if not original_data_root or not os.path.isdir(original_data_root):
        return expected

    for root, _, files in os.walk(original_data_root):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in ORIGINAL_EXTS:
                expected.add(filename)
    return expected


def clean_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def format_elapsed(seconds):
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


# ==============================================================================
# Detector (merged from preprocessing/detector.py)
# ==============================================================================
def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class RetinaFace:
    def __init__(self, model_file=None, session=None, providers=None):
        self.model_file = model_file
        self.session = session
        if providers is None:
            providers = ["CPUExecutionProvider"]
        if self.session is None:
            assert self.model_file is not None
            assert os.path.exists(self.model_file)
            sess_options = onnxruntime.SessionOptions()
            num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
            self.session = onnxruntime.InferenceSession(self.model_file, sess_options)
            self.session.set_providers(providers)
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        self.input_name = input_cfg.name
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self.use_kps = True
        else:
            raise ValueError(f"Unsupported RetinaFace output count: {len(outputs)}")

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])
        nms_thresh = kwargs.get("nms_thresh", None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get("det_thresh", None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get("input_size", None)
        if input_size is not None:
            if self.input_size is not None:
                print("warning: det_size is already set in detection model, ignore")
            else:
                self.input_size = input_size

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc] * stride
            if self.use_kps:
                kps_preds = net_outs[idx + self.fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds).reshape((-1, 5, 2))
                kpss_list.append(kpss[pos_inds])
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None, max_num=0, metric="default"):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep


def max_spread_permutation_pq(n, start=0):
    if not (0 <= start < n):
        raise ValueError("`start` must be in the range [0, n-1]")

    chosen = [start]
    dist = {i: abs(i - start) for i in range(n) if i != start}
    heap = [(-d, i) for i, d in dist.items()]
    heapq.heapify(heap)

    while heap:
        while True:
            neg_d, candidate = heapq.heappop(heap)
            current = -neg_d
            if dist.get(candidate, -1) == current:
                break

        chosen.append(candidate)
        del dist[candidate]

        for other in list(dist.keys()):
            new_d = abs(other - candidate)
            if new_d < dist[other]:
                dist[other] = new_d
                heapq.heappush(heap, (-new_d, other))

    return chosen


def get_video_frame_ids(source_path):
    try:
        props = iio.improps(source_path, plugin="pyav")
        total_frames = props.shape[0]
    except Exception as e:
        print(f"Warning: Video {source_path} cannot be opened! {e}")
        return [], 0

    if total_frames <= 0:
        print(f"Warning: Video {source_path} has no frames to extract.")
        return [], total_frames

    frame_ids = max_spread_permutation_pq(total_frames, start=total_frames // 2)

    seen = set()
    unique_ids = []
    for fid in frame_ids:
        fid = int(fid)
        if 0 <= fid < total_frames and fid not in seen:
            seen.add(fid)
            unique_ids.append(fid)

    return unique_ids, total_frames


def align_face(img, landmarks, target_size=None, scale=1.3):
    dst = np.array(
        [
            [0.34, 0.46],
            [0.66, 0.46],
            [0.5, 0.64],
            [0.37, 0.82],
            [0.63, 0.82],
        ],
        dtype=np.float32,
    )

    if target_size is None:
        desired_dists = np.linalg.norm(landmarks[:, None, :] - landmarks[None, :, :], axis=-1)
        dst_dists = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=-1)
        upper = np.triu_indices(len(dst), k=1)
        dst_dists = dst_dists[upper]
        desired_dists = desired_dists[upper]

        approx_size = np.round(np.mean(desired_dists / dst_dists) * scale).astype(int)
        target_size = (approx_size, approx_size)

    dst[:, 0] = dst[:, 0] * target_size[0]
    dst[:, 1] = dst[:, 1] * target_size[1]

    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.0
    y_margin = target_size[1] * margin_rate / 2.0

    dst[:, 0] += x_margin
    dst[:, 1] += y_margin

    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

    src = landmarks.astype(np.float32)
    M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
    img = cv2.warpAffine(img, M, target_size, flags=cv2.INTER_LINEAR)

    return img


def crop_face_bbox(img, bbox, target_size=None, scale=1.0):
    x1, y1, x2, y2 = bbox[:4]
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w *= scale
    h *= scale

    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))

    img_h, img_w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    if target_size is not None:
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    return crop


def bbox_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def greedy_iou_match(prev_tracks, curr_boxes, iou_thresh):
    if not prev_tracks or curr_boxes.size == 0:
        return {}

    prev_boxes = np.stack([t["bbox"] for t in prev_tracks], axis=0)
    iou_matrix = np.zeros((prev_boxes.shape[0], curr_boxes.shape[0]), dtype=np.float32)
    for i in range(prev_boxes.shape[0]):
        for j in range(curr_boxes.shape[0]):
            iou_matrix[i, j] = bbox_iou(prev_boxes[i], curr_boxes[j])

    matches = {}
    used_prev = set()
    used_curr = set()
    while True:
        max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        max_iou = float(iou_matrix[max_idx])
        if max_iou < iou_thresh:
            break

        i, j = int(max_idx[0]), int(max_idx[1])
        if i in used_prev or j in used_curr:
            iou_matrix[i, j] = -1.0
            continue

        matches[j] = (int(prev_tracks[i]["id"]), max_iou)
        used_prev.add(i)
        used_curr.add(j)
        iou_matrix[i, :] = -1.0
        iou_matrix[:, j] = -1.0

    return matches


def process_video(
    source_path,
    output_root,
    model,
    scale,
    target_size,
    num_frames,
    track_iou,
    multi_score_thres,
    multi_confirm_frames,
):
    frame_root = output_root
    frame_save_path = os.path.join(frame_root, "frames")
    ids_root = frame_root

    def should_enable_multiface():
        if multi_confirm_frames <= 0:
            return False
        for frame_idx in range(multi_confirm_frames):
            try:
                frame_rgb = iio.imread(source_path, index=frame_idx, plugin="pyav")
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Warning: Failed to read frame {frame_idx} of {source_path}. {e}")
                return False
            try:
                dets, _ = model.detect(frame)
            except Exception as e:
                print(f"Error during detection: {e}")
                return False
            if dets is None or len(dets) == 0:
                return False
            high = (dets[:, 4] >= multi_score_thres).sum()
            if high < 2:
                return False
        return True

    track_ids = should_enable_multiface()

    frame_ids, _ = get_video_frame_ids(source_path)
    if not frame_ids:
        print(f"Warning: No frames extracted from {source_path}.")
        return ids_root if track_ids else frame_save_path

    target_frames = num_frames if num_frames and num_frames > 0 else None
    max_attempts = target_frames * 3 if target_frames else None

    attempted = 0
    if track_ids:
        candidates = []
        for frame_id in frame_ids:
            if max_attempts is not None and attempted >= max_attempts:
                break
            attempted += 1

            try:
                frame_rgb = iio.imread(source_path, index=frame_id, plugin="pyav")
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Warning: Failed to read frame {frame_id} of {source_path}. Skipping. {e}")
                continue

            try:
                preds = model.detect(frame)
            except Exception as e:
                print(f"Error during detection: {e}")
                continue

            xyxy, landmarks = preds
            if len(xyxy) == 0:
                continue

            candidates.append(
                {
                    "frame_id": int(frame_id),
                    "dets": xyxy.copy(),
                    "kpss": None if landmarks is None else landmarks.copy(),
                }
            )

        if not candidates:
            print(f"No faces were saved from {source_path}.")
            return ids_root

        candidates.sort(key=lambda c: c["frame_id"])
        selected = candidates
        if target_frames is not None:
            if len(candidates) >= target_frames:
                pick_idx = np.linspace(0, len(candidates) - 1, target_frames, endpoint=True, dtype=int)
                selected = [candidates[i] for i in pick_idx]
            else:
                print(
                    f"Warning: Only {len(candidates)} frames available for {source_path} "
                    f"(requested {target_frames}, oversampled {max_attempts})."
                )

        os.makedirs(ids_root, exist_ok=True)
        prev_tracks = []
        next_track_id = 0
        for cand in selected:
            frame_id = cand["frame_id"]
            dets = cand["dets"]
            kpss = cand["kpss"]

            try:
                frame_rgb = iio.imread(source_path, index=frame_id, plugin="pyav")
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Warning: Failed to read frame {frame_id} of {source_path}. Skipping. {e}")
                continue

            curr_boxes = dets[:, :4]
            match_map = greedy_iou_match(prev_tracks, curr_boxes, track_iou)
            assignments = []

            for det_idx in range(curr_boxes.shape[0]):
                if det_idx in match_map:
                    track_id, _ = match_map[det_idx]
                else:
                    track_id = next_track_id
                    next_track_id += 1

                bbox = dets[det_idx]
                landmarks = None
                if kpss is not None and len(kpss) > det_idx:
                    landmarks = kpss[det_idx]

                id_dir = os.path.join(ids_root, f"ID_{track_id:03d}")
                os.makedirs(id_dir, exist_ok=True)
                out_path = os.path.join(id_dir, f"frame_{frame_id:04d}.png")

                if landmarks is not None:
                    aligned_face = align_face(frame, landmarks, target_size=target_size, scale=scale)
                else:
                    aligned_face = crop_face_bbox(frame, bbox, target_size=target_size, scale=scale)
                    if aligned_face is None:
                        continue

                ok = cv2.imwrite(out_path, aligned_face)
                if not ok:
                    print(f"Warning: Failed to write frame {frame_id} for {source_path}.")

                assignments.append({"id": track_id, "bbox": bbox[:4].copy()})

            prev_tracks = assignments

        return ids_root

    aligned_candidates = []
    for frame_id in frame_ids:
        if max_attempts is not None and attempted >= max_attempts:
            break
        attempted += 1

        try:
            frame_rgb = iio.imread(source_path, index=frame_id, plugin="pyav")
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Warning: Failed to read frame {frame_id} of {source_path}. Skipping. {e}")
            continue

        try:
            preds = model.detect(frame)
        except Exception as e:
            print(f"Error during detection: {e}")
            continue

        xyxy, landmarks = preds
        if len(xyxy) == 0:
            continue

        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = np.argmax(areas)
        selected_landmarks = landmarks[idx]

        aligned_face = align_face(frame, selected_landmarks, target_size=target_size, scale=scale)
        aligned_candidates.append((frame_id, aligned_face))

    if not aligned_candidates:
        print(f"No faces were saved from {source_path}.")
        return frame_save_path

    aligned_candidates.sort(key=lambda x: x[0])
    selected = aligned_candidates
    if target_frames is not None:
        if len(aligned_candidates) >= target_frames:
            pick_idx = np.linspace(0, len(aligned_candidates) - 1, target_frames, endpoint=True, dtype=int)
            selected = [aligned_candidates[i] for i in pick_idx]
        else:
            print(
                f"Warning: Only {len(aligned_candidates)} frames available for {source_path} "
                f"(requested {target_frames}, oversampled {max_attempts})."
            )

    os.makedirs(frame_save_path, exist_ok=True)
    for frame_id, aligned_face in selected:
        frame_filename = os.path.join(frame_save_path, f"frame_{frame_id:04d}.png")
        ok = cv2.imwrite(frame_filename, aligned_face)
        if not ok:
            print(f"Warning: Failed to write frame {frame_id} for {source_path}.")

    return frame_save_path


def process_image(source_path, output_root, model, scale, target_size):
    root, _ = os.path.splitext(output_root)
    output_root = root
    base = os.path.splitext(os.path.basename(source_path))[0]
    target_path = os.path.join(output_root, f"{base}.png")

    try:
        img_rgb = iio.imread(source_path)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Failed to read image {source_path}: {e}")
        return None

    try:
        preds = model.detect(img)
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

    xyxy, landmarks = preds
    if len(xyxy) == 0:
        return None

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = np.argmax(areas)
    selected_landmarks = landmarks[idx]

    aligned_face = align_face(img, selected_landmarks, target_size=target_size, scale=scale)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    ok = cv2.imwrite(target_path, aligned_face)
    if not ok:
        print(f"Warning: Failed to write image {target_path}.")
        return None
    return target_path


def find_media_files(start_dir, extensions):
    files = []
    for root, _, filenames in os.walk(start_dir):
        for name in filenames:
            if name.lower().endswith(extensions):
                files.append(os.path.join(root, name))
    return sorted(files)


def get_output_root(source_path, input_root, output_root, is_video):
    rel = os.path.relpath(source_path, input_root) if os.path.isdir(input_root) else os.path.basename(source_path)
    rel_no_ext = os.path.splitext(rel)[0]
    category = "videos" if is_video else "images"

    rel_norm = os.path.normpath(rel_no_ext)
    parts = rel_norm.split(os.sep)
    if parts and parts[0] in ("images", "videos") and parts[0] == category:
        rel_no_ext = os.path.join(*parts[1:]) if len(parts) > 1 else parts[0]

    return os.path.join(output_root, category, rel_no_ext)


def process_dataset(
    input_path,
    output_folder,
    model,
    num_workers,
    scale,
    target_size,
    num_frames,
    track_iou,
    multi_score_thres,
    multi_confirm_frames,
):
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = find_media_files(input_path, VIDEO_EXTS + IMAGE_EXTS)

    if not files:
        print(f"No files found in {input_path}")
        return

    def process_one(source_path):
        is_video = source_path.lower().endswith(VIDEO_EXTS)
        output_root = get_output_root(source_path, input_path, output_folder, is_video)

        if is_video:
            try:
                return process_video(
                    source_path,
                    output_root,
                    model,
                    scale=scale,
                    target_size=target_size,
                    num_frames=num_frames,
                    track_iou=track_iou,
                    multi_score_thres=multi_score_thres,
                    multi_confirm_frames=multi_confirm_frames,
                )
            except Exception as e:
                print(f"Error processing video {source_path}: {e}")
                return None

        try:
            return process_image(
                source_path,
                output_root,
                model,
                scale=scale,
                target_size=target_size,
            )
        except Exception as e:
            print(f"Error processing image {source_path}: {e}")
            return None

    files = sorted(files)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_one, file) for file in files]
        for future in tqdm(futures, desc=f"Preprocess {input_path}", leave=True):
            future.result()



def parse_target_size(target_size_str):
    try:
        width, height = map(int, target_size_str.split(","))
        return (width, height)
    except ValueError:
        if "none" in target_size_str.lower():
            return None
        raise ValueError("Invalid target_size format. Use 'width,height' or 'none'.")


def prepare_model_cuda_only(det_thres=0.5, nms_thresh=0.4):
    model_file = DETECTOR_MODEL_FILE
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Detector model file not found: {model_file}. "
            "Place det_10g.onnx in likelihood_submission/retinaface/."
        )

    try:
        onnxruntime.preload_dlls(cuda=True, cudnn=True, msvc=False)
    except Exception as e:
        print(f"[CUDA CHECK] Warning: preload_dlls failed: {e}")

    available = onnxruntime.get_available_providers()
    print(f"[CUDA CHECK] ONNXRuntime available providers: {available}")
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable in this environment. "
            "Install/repair CUDA-enabled ONNXRuntime and CUDA libraries."
        )

    requested = ["CUDAExecutionProvider"]
    print(f"[CUDA CHECK] Requesting providers: {requested} (CPU fallback disabled)")

    try:
        model = RetinaFace(model_file, providers=requested)
        model.prepare(
            ctx_id=0,
            nms_thresh=nms_thresh,
            input_size=(640, 640),
            det_thresh=det_thres,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize RetinaFace with CUDAExecutionProvider. "
            "CPU fallback is disabled by design."
        ) from e

    active = model.session.get_providers()
    print(f"[CUDA CHECK] Active session providers: {active}")
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            "CUDAExecutionProvider was not activated. "
            "Refusing to continue because CPU fallback is disabled."
        )

    print("[CUDA CHECK] CUDAExecutionProvider is active. Cropping will run on GPU.")
    return model


# ==============================================================================
# Dataset for cropped images (merged from inference_last.py)
# ==============================================================================
class CompetitionDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.items, stats = self._build_items(data_root)
        print(f"[*] Found {stats['total_images']} images in {data_root}")
        if stats["mode"] == "structured":
            print(f"    - images: {stats['image_items']} items")
            print(f"    - videos: {stats['video_items']} items ({stats['video_frames']} frames)")
        else:
            print(f"    - flat: {stats['total_images']} files")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path = item["path"]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"[Error] loading {path}: {e}")
            image = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
        return image, item["group_key"], item["item_id"], item["item_type"]

    def _is_image_file(self, filename):
        return filename.lower().endswith(CROP_IMAGE_EXTS)

    def _collect_images_recursive(self, root_dir):
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if self._is_image_file(file):
                    image_paths.append(os.path.join(root, file))
        image_paths.sort()
        return image_paths

    def _collect_images_flat(self, root_dir):
        image_paths = []
        with os.scandir(root_dir) as it:
            for entry in it:
                if entry.is_file() and self._is_image_file(entry.name):
                    image_paths.append(entry.path)
        image_paths.sort()
        return image_paths

    def _build_items(self, data_root):
        items = []
        stats = {
            "mode": "flat",
            "total_images": 0,
            "image_items": 0,
            "video_items": 0,
            "video_frames": 0,
        }

        images_root = os.path.join(data_root, "images")
        videos_root = os.path.join(data_root, "videos")
        has_structured = os.path.isdir(images_root) or os.path.isdir(videos_root)

        if has_structured:
            stats["mode"] = "structured"

            if os.path.isdir(images_root):
                with os.scandir(images_root) as it:
                    image_dirs = sorted([entry for entry in it if entry.is_dir()], key=lambda e: e.name)
                for entry in image_dirs:
                    image_files = self._collect_images_flat(entry.path)
                    if not image_files:
                        continue
                    if len(image_files) > 1:
                        print(f"[Warn] Multiple images found in {entry.path}, using first.")
                    img_path = image_files[0]
                    item_id = entry.name
                    items.append(
                        {
                            "path": img_path,
                            "group_key": f"image::{item_id}",
                            "item_id": item_id,
                            "item_type": "image",
                        }
                    )
                    stats["image_items"] += 1

            if os.path.isdir(videos_root):
                with os.scandir(videos_root) as it:
                    video_dirs = sorted([entry for entry in it if entry.is_dir()], key=lambda e: e.name)
                for entry in video_dirs:
                    frames_dir = os.path.join(entry.path, "frames")
                    item_id = entry.name

                    if os.path.isdir(frames_dir):
                        frame_paths = self._collect_images_recursive(frames_dir)
                        if not frame_paths:
                            continue
                        for frame_path in frame_paths:
                            items.append(
                                {
                                    "path": frame_path,
                                    "group_key": f"video::{item_id}",
                                    "item_id": item_id,
                                    "item_type": "video",
                                }
                            )
                        stats["video_items"] += 1
                        stats["video_frames"] += len(frame_paths)
                        continue

                    with os.scandir(entry.path) as it:
                        sub_dirs = sorted([sub for sub in it if sub.is_dir()], key=lambda e: e.name)

                    sub_groups = []
                    for sub in sub_dirs:
                        frame_paths = self._collect_images_recursive(sub.path)
                        if frame_paths:
                            sub_groups.append((sub.name, frame_paths))

                    if sub_groups:
                        for sub_id, frame_paths in sub_groups:
                            for frame_path in frame_paths:
                                items.append(
                                    {
                                        "path": frame_path,
                                        "group_key": f"video::{item_id}::{sub_id}",
                                        "item_id": item_id,
                                        "item_type": "video",
                                    }
                                )
                            stats["video_frames"] += len(frame_paths)
                        stats["video_items"] += 1
                        continue

                    frame_paths = self._collect_images_recursive(entry.path)
                    if not frame_paths:
                        continue
                    for frame_path in frame_paths:
                        items.append(
                            {
                                "path": frame_path,
                                "group_key": f"video::{item_id}",
                                "item_id": item_id,
                                "item_type": "video",
                            }
                        )
                    stats["video_items"] += 1
                    stats["video_frames"] += len(frame_paths)

        else:
            image_paths = self._collect_images_recursive(data_root)
            for path in image_paths:
                items.append(
                    {
                        "path": path,
                        "group_key": f"image::{path}",
                        "item_id": os.path.splitext(os.path.basename(path))[0],
                        "item_type": "image",
                    }
                )

        stats["total_images"] = len(items)
        return items, stats


# ==============================================================================
# Model (merged from inference_last.py)
# ==============================================================================
class DINOv3ForClassification(nn.Module):
    def __init__(self, num_labels=1, normalize_inputs=True, use_clip=False, clip_model=DEFAULT_CLIP_MODEL):
        super().__init__()
        self.is_clip = bool(use_clip)
        self.clip_model = _parse_optional_str(clip_model) or DEFAULT_CLIP_MODEL

        if self.is_clip:
            local_clip_path = _resolve_local_model_path(self.clip_model, RUNTIME_CONFIG["backbone_dir"])
            clip_source = local_clip_path if local_clip_path is not None else self.clip_model
            local_only = local_clip_path is not None
            source_kind = "local" if local_only else "huggingface_hub"
            print(f"[*] Encoder type: CLIP")
            print(f"[*] Loading CLIPVisionModel ({source_kind}): {clip_source}")
            try:
                self.backbone = CLIPVisionModel.from_pretrained(
                    clip_source,
                    local_files_only=local_only,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load CLIP vision model.\n"
                    f"  - requested clip_model: {self.clip_model!r}\n"
                    f"  - tried source: {clip_source!r}\n"
                    f"  - local_files_only: {local_only}\n"
                    "Check clip_model path/name and ensure weights are available "
                    "locally or cached from Hugging Face."
                ) from exc
            config = self.backbone.config
        else:
            print("[*] Encoder type: DINO")
            print("[*] Loading Backbone config (embedded)")
            config_kwargs = dict(DINOV3_BACKBONE_CONFIG)
            model_type = config_kwargs.pop("model_type")
            config = AutoConfig.for_model(model_type, **config_kwargs)
            self.backbone = AutoModel.from_config(config, trust_remote_code=True)
        self.normalize_inputs = bool(normalize_inputs)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.config = config
        self.hidden_size = getattr(self.config, "hidden_size", getattr(self.config, "embed_dim", 1280))

        if num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {num_labels}")
        self.num_labels = int(num_labels)

        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
        print(f"[*] GenD L2-normalized head: {'enabled' if self.normalize_inputs else 'disabled'}")

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
        if self.normalize_inputs:
            cls_token = F.normalize(cls_token, p=2, dim=1)
        if cls_token.dtype != self.classifier.weight.dtype:
            cls_token = cls_token.to(self.classifier.weight.dtype)
        logits = self.classifier(cls_token)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.long())

        return ImageClassifierOutput(loss=loss, logits=logits)


def _resolve_model_pt_path(weight_path):
    if os.path.isdir(weight_path):
        return os.path.join(weight_path, "model.pt")
    return weight_path


def _load_state_dict_from_pt(model_pt_path):
    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"model.pt not found: {model_pt_path}")

    state_obj = torch.load(model_pt_path, map_location="cpu")
    if isinstance(state_obj, dict) and "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
        state_obj = state_obj["state_dict"]
    if not isinstance(state_obj, dict):
        raise TypeError(f"Unsupported checkpoint format in {model_pt_path}: {type(state_obj)}")
    return state_obj


def _infer_num_labels_from_state_dict(state_dict, default_num_labels=1):
    weight = state_dict.get("classifier.weight")
    if weight is None and "model.classifier.weight" in state_dict:
        weight = state_dict["model.classifier.weight"]
    if weight is None and "module.classifier.weight" in state_dict:
        weight = state_dict["module.classifier.weight"]
    if weight is not None and getattr(weight, "ndim", None) == 2:
        return int(weight.shape[0])
    return int(default_num_labels)


def _is_adapter_state_dict(state_dict):
    """Check if the state dict contains only adapter (LoRA) weights."""
    return any("lora_" in k for k in state_dict.keys())


def _normalize_layernorm_patterns(patterns):
    if patterns is None:
        patterns = DEFAULT_LAYERNORM_PATTERNS
    patterns = [str(p).lower() for p in patterns if str(p).strip()]
    if not patterns:
        patterns = DEFAULT_LAYERNORM_PATTERNS
    return patterns


def _collect_layernorm_modules_to_save(backbone, patterns=None):
    patterns = _normalize_layernorm_patterns(patterns)
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


def _collect_layernorm_modules_from_checkpoint(state_dict, patterns=None):
    patterns = _normalize_layernorm_patterns(patterns)
    prefix = "base_model.model.backbone."

    module_names = set()
    for key in state_dict.keys():
        if not key.startswith(prefix):
            continue
        if ".lora_" in key:
            continue
        if not (key.endswith(".weight") or key.endswith(".bias")):
            continue

        module_name = key[len("base_model.model."):].rsplit(".", 1)[0]
        if any(pattern in module_name.lower() for pattern in patterns):
            module_names.add(module_name)

    return sorted(module_names)


def _build_peft_model_for_inference(
    model,
    state_dict=None,
    layernorm_patterns=None,
    lora_rank=1,
    lora_alpha=2,
):
    """Build a PEFT model matching the training config for inference."""
    from peft import LoraConfig, get_peft_model
    linear_names = [n for n, m in model.backbone.named_modules() if isinstance(m, nn.Linear)]
    target_modules = [p for p in ["q_proj", "k_proj", "v_proj", "o_proj"] if any(p in n for n in linear_names)]
    if not target_modules:
        target_modules = [p for p in ["up_proj", "down_proj"] if any(p in n for n in linear_names)]
    if not target_modules:
        target_modules = list({n.split(".")[-1] for n in linear_names})

    if state_dict is not None:
        ln_modules_to_save = _collect_layernorm_modules_from_checkpoint(state_dict, layernorm_patterns)
        source = "checkpoint"
    else:
        ln_modules_to_save = _collect_layernorm_modules_to_save(model.backbone, layernorm_patterns)
        source = "backbone"

    preview = ", ".join(ln_modules_to_save[:5]) if ln_modules_to_save else "(none)"
    print(
        f"[*] LN modules_to_save ({source}): {len(ln_modules_to_save)} modules "
        f"(preview: {preview})"
    )
    modules_to_save = ["classifier"] + ln_modules_to_save

    print(f"[*] Inference LoRA config: r={lora_rank}, alpha={lora_alpha}")
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,  # inference 시 dropout 비활성화
        bias="none",
        modules_to_save=modules_to_save,
        init_lora_weights="pissa",
    )
    model = get_peft_model(model, peft_config)
    return model


def _infer_lora_rank_from_state_dict(state_dict):
    for key, value in state_dict.items():
        if not key.endswith("lora_A.weight"):
            continue
        if hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[0])
    return None


def _resolve_adapter_config_path(weight_path, model_pt_path):
    candidates = []
    if os.path.isdir(weight_path):
        candidates.append(os.path.join(weight_path, "adapter_config.json"))
    candidates.append(os.path.join(os.path.dirname(model_pt_path), "adapter_config.json"))

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        norm = os.path.abspath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.exists(norm):
            return norm
    return None


def _load_adapter_config(weight_path, model_pt_path):
    cfg_path = _resolve_adapter_config_path(weight_path, model_pt_path)
    if cfg_path is None:
        return None, None

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            print(f"[Warn] adapter_config is not a dict: {cfg_path}")
            return None, cfg_path
        return payload, cfg_path
    except Exception as exc:
        print(f"[Warn] Failed to read adapter_config.json ({cfg_path}): {exc}")
        return None, cfg_path


def _resolve_lora_hparams(weight_path, model_pt_path, state_dict, cli_rank=None, cli_alpha=None):
    adapter_cfg, adapter_cfg_path = _load_adapter_config(weight_path, model_pt_path)

    rank_source = "default"
    alpha_source = "default"

    if cli_rank is not None:
        lora_rank = int(cli_rank)
        rank_source = "cli"
    elif adapter_cfg is not None and adapter_cfg.get("lora_rank") is not None:
        lora_rank = int(adapter_cfg["lora_rank"])
        rank_source = f"adapter_config:{adapter_cfg_path}"
    else:
        inferred_rank = _infer_lora_rank_from_state_dict(state_dict)
        if inferred_rank is not None:
            lora_rank = int(inferred_rank)
            rank_source = "state_dict"
        else:
            lora_rank = 1

    if cli_alpha is not None:
        lora_alpha = int(cli_alpha)
        alpha_source = "cli"
    elif adapter_cfg is not None and adapter_cfg.get("lora_alpha") is not None:
        lora_alpha = int(adapter_cfg["lora_alpha"])
        alpha_source = f"adapter_config:{adapter_cfg_path}"
    else:
        lora_alpha = 2

    if lora_rank <= 0:
        raise ValueError(f"Invalid LoRA rank: {lora_rank}")
    if lora_alpha <= 0:
        raise ValueError(f"Invalid LoRA alpha: {lora_alpha}")

    print(f"[*] Resolved LoRA rank={lora_rank} ({rank_source})")
    print(f"[*] Resolved LoRA alpha={lora_alpha} ({alpha_source})")
    return lora_rank, lora_alpha


def _resolve_clip_settings(weight_path, model_pt_path, cli_clip=None, cli_clip_model=None):
    adapter_cfg, adapter_cfg_path = _load_adapter_config(weight_path, model_pt_path)
    cfg_source = f"config:{RUNTIME_CONFIG['path']}" if RUNTIME_CONFIG["has_config"] else "default"

    clip_source = cfg_source
    clip_model_source = cfg_source

    if cli_clip is not None:
        use_clip = bool(cli_clip)
        clip_source = "cli"
    elif adapter_cfg is not None and adapter_cfg.get("clip") is not None:
        use_clip = _parse_bool(adapter_cfg.get("clip"), RUNTIME_CONFIG["use_clip"])
        clip_source = f"adapter_config:{adapter_cfg_path}"
    else:
        use_clip = bool(RUNTIME_CONFIG["use_clip"])

    cli_clip_model_norm = _parse_optional_str(cli_clip_model)
    adapter_clip_model = None
    if adapter_cfg is not None:
        adapter_clip_model = _parse_optional_str(adapter_cfg.get("clip_model"))

    if cli_clip_model_norm is not None:
        clip_model = cli_clip_model_norm
        clip_model_source = "cli"
    elif adapter_clip_model is not None:
        clip_model = adapter_clip_model
        clip_model_source = f"adapter_config:{adapter_cfg_path}"
    else:
        clip_model = RUNTIME_CONFIG["clip_model"]

    print(f"[*] Resolved clip={use_clip} ({clip_source})")
    print(f"[*] Resolved clip_model={clip_model!r} ({clip_model_source})")
    return bool(use_clip), clip_model


def load_model(
    weight_path,
    device,
    num_labels=None,
    lora_rank=None,
    lora_alpha=None,
    clip=None,
    clip_model=None,
):
    print(f"[*] Loading model from {weight_path}...")

    model_pt_path = _resolve_model_pt_path(weight_path)
    state_dict = _load_state_dict_from_pt(model_pt_path)
    inferred_num_labels = _infer_num_labels_from_state_dict(state_dict)
    if num_labels is None:
        num_labels = inferred_num_labels
    print(f"[*] Using num_labels={num_labels} (model.pt suggests {inferred_num_labels})")

    resolved_clip, resolved_clip_model = _resolve_clip_settings(
        weight_path=weight_path,
        model_pt_path=model_pt_path,
        cli_clip=clip,
        cli_clip_model=clip_model,
    )
    model = DINOv3ForClassification(
        num_labels=num_labels,
        use_clip=resolved_clip,
        clip_model=resolved_clip_model,
    )

    if _is_adapter_state_dict(state_dict):
        print("[*] Detected adapter-only checkpoint, rebuilding PEFT model...")
        resolved_rank, resolved_alpha = _resolve_lora_hparams(
            weight_path=weight_path,
            model_pt_path=model_pt_path,
            state_dict=state_dict,
            cli_rank=lora_rank,
            cli_alpha=lora_alpha,
        )
        model = _build_peft_model_for_inference(
            model,
            state_dict=state_dict,
            lora_rank=resolved_rank,
            lora_alpha=resolved_alpha,
        )
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, state_dict)
    else:
        model.load_state_dict(state_dict, strict=True)

    model = model.float()
    model.to(device)
    model.eval()
    return model, resolved_clip, resolved_clip_model


# ==============================================================================
# Unified pipeline
# ==============================================================================
def run_preprocessing(args):
    if args.skip_preprocess:
        print(f"[*] Skipping preprocessing. Reusing crops: {args.cropped_dir}")
        return args.cropped_dir

    if args.clean_cropped:
        print(f"[*] Cleaning cropped directory: {args.cropped_dir}")
        clean_dir(args.cropped_dir)
    else:
        os.makedirs(args.cropped_dir, exist_ok=True)

    print("[*] Initializing face detector...")
    detector = prepare_model_cuda_only(det_thres=args.det_thres, nms_thresh=args.nms_thresh)

    print("[*] Running preprocessing (crop/alignment)...")
    process_dataset(
        input_path=args.input_root,
        output_folder=args.cropped_dir,
        model=detector,
        num_workers=args.crop_workers,
        scale=args.scale,
        target_size=args.target_size,
        num_frames=args.num_frames,
        track_iou=args.track_iou,
        multi_score_thres=args.multi_score_thres,
        multi_confirm_frames=args.multi_confirm_frames,
    )
    return args.cropped_dir


def run_model_inference(args, cropped_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    model, resolved_clip, resolved_clip_model = load_model(
        args.weight,
        device,
        num_labels=args.num_labels,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        clip=args.clip,
        clip_model=args.clip_model,
    )
    args.clip = resolved_clip
    args.clip_model = resolved_clip_model
    original_ext_map = load_original_extension_map(args.input_root)

    if resolved_clip:
        norm_mean, norm_std = CLIP_MEAN, CLIP_STD
        norm_name = "CLIP"
    else:
        norm_mean, norm_std = IMAGENET_MEAN, IMAGENET_STD
        norm_name = "ImageNet/DINO"
    print(f"[*] Inference normalization: {norm_name} mean={norm_mean}, std={norm_std}")

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )

    dataset = CompetitionDataset(data_root=cropped_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    group_stats = {}
    print(f"[*] Starting model inference on {cropped_dir} ...")

    with torch.inference_mode():
        for images, group_keys, item_ids, item_types in tqdm(dataloader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            logits = outputs.logits
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)

            current_num_labels = logits.shape[1]
            if current_num_labels == 1:
                fake_probs = torch.sigmoid(logits[:, 0]).cpu().numpy()
            elif current_num_labels == 2:
                probs = F.softmax(logits, dim=1)
                fake_probs = probs[:, 1].cpu().numpy()
            else:
                raise ValueError(f"Expected binary logits (1 or 2), got {current_num_labels}.")

            for group_key, item_id, item_type, prob in zip(group_keys, item_ids, item_types, fake_probs):
                if group_key not in group_stats:
                    group_stats[group_key] = {
                        "item_id": item_id,
                        "item_type": item_type,
                        "probs": [],
                    }
                group_stats[group_key]["probs"].append(float(prob))

    results = []
    video_candidates = {}
    for info in group_stats.values():
        probs = info["probs"]
        if info["item_type"] == "video":
            agg_prob = topk_confidence_mean(probs, topk=args.topk)
            video_id = info["item_id"]
            video_candidates.setdefault(video_id, []).append(float(agg_prob))
        else:
            agg_prob = float(np.mean(probs)) if probs else 0.0
            mapped_filename = with_submission_ext(info["item_id"], info["item_type"], original_ext_map)
            results.append({"filename": mapped_filename, "prob": float(agg_prob)})

    for video_id, probs in video_candidates.items():
        video_prob = max(probs) if probs else 0.0
        mapped_filename = with_submission_ext(video_id, "video", original_ext_map)
        results.append({"filename": mapped_filename, "prob": float(video_prob)})

    if args.ensure_all_inputs:
        expected_files = collect_expected_filenames(args.input_root)
        current_files = {r["filename"] for r in results}
        missing = sorted(expected_files - current_files)
        if missing:
            print(f"[Warn] {len(missing)} items had no face crops. Filling prob=0.5.")
            for filename in missing:
                results.append({"filename": filename, "prob": 0.5})

    df = pd.DataFrame(results, columns=["filename", "prob"])
    if not df.empty:
        df = df.sort_values("filename", ascending=True).reset_index(drop=True)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[*] Results saved to {args.output_csv}")


# ==============================================================================
# CLI
# ==============================================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="Unified inference: preprocessing + model inference + submission export"
    )

    parser.add_argument("--weight", type=str, default=DEFAULT_MODEL_DIR, help="Path to model dir or model.pt")
    parser.add_argument("--input_root", type=str, default=DEFAULT_INPUT_ROOT, help="Root folder of raw test data")
    parser.add_argument("--cropped_dir", type=str, default=DEFAULT_CROPPED_DIR, help="Preprocessed output directory")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Final submission csv path")

    parser.add_argument("--skip_preprocess", action="store_true", help="Skip detector preprocessing and reuse cropped_dir")
    parser.add_argument("--clean_cropped", action="store_true", help="Delete cropped_dir before preprocessing")
    parser.add_argument("--crop_workers", type=int, default=16, help="Preprocessing thread count")
    parser.add_argument("--loader_workers", type=int, default=16, help="Dataloader workers")

    parser.add_argument("--det_thres", type=float, default=0.5, help="RetinaFace detection threshold")
    parser.add_argument("--nms_thresh", type=float, default=0.4, help="RetinaFace NMS threshold")
    parser.add_argument("--scale", type=float, default=1.3, help="Face alignment scale")
    parser.add_argument("--target_size", type=str, default="none", help="Crop target size as width,height or none")
    parser.add_argument("--num_frames", type=int, default=32, help="Max frames extracted per video")
    parser.add_argument("--track_iou", type=float, default=0.1, help="IoU threshold for ID tracking")
    parser.add_argument("--multi_score_thres", type=float, default=0.71, help="Multi-face activation score threshold")
    parser.add_argument(
        "--multi_confirm_frames",
        type=int,
        default=1,
        help="Initial frames required to have >=2 high-score faces",
    )

    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_labels", type=int, default=None, help="Override classifier output dimension")
    parser.add_argument("--lora_rank", type=int, default=None, help="Override LoRA rank for adapter checkpoints")
    parser.add_argument("--lora_alpha", type=int, default=None, help="Override LoRA alpha for adapter checkpoints")
    parser.add_argument(
        "--clip",
        dest="clip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use CLIP vision encoder (priority: CLI > adapter_config > config.yaml > default).",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default=None,
        help="CLIP model id/path override (priority: CLI > adapter_config > config.yaml > default).",
    )
    parser.add_argument("--video_agg", type=str, default=VIDEO_AGG, choices=[VIDEO_AGG], help="Video aggregation")
    parser.add_argument("--topk", type=int, default=12, help="Top-k for topk_confidence")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--no_ensure_all_inputs",
        action="store_false",
        dest="ensure_all_inputs",
        help="Do not fill missing inputs with prob=0.5",
    )
    parser.set_defaults(ensure_all_inputs=True)

    args = parser.parse_args()
    args.target_size = parse_target_size(args.target_size)
    args.clip_model = _parse_optional_str(args.clip_model)
    return args


def main():
    pipeline_start = time.perf_counter()
    args = get_args()
    cfg_label = RUNTIME_CONFIG["path"] if RUNTIME_CONFIG["has_config"] else "(not found)"
    print(f"[*] Runtime config source: {cfg_label}")
    print(
        f"[*] Runtime defaults: clip={RUNTIME_CONFIG['use_clip']}, "
        f"clip_model={RUNTIME_CONFIG['clip_model']!r}"
    )
    seed_everything(args.seed)

    preprocess_start = time.perf_counter()
    cropped_dir = run_preprocessing(args)
    preprocess_elapsed = time.perf_counter() - preprocess_start
    print(f"[*] 소요시간(시작 -> 전처리 완료): {format_elapsed(preprocess_elapsed)}")

    inference_start = time.perf_counter()
    run_model_inference(args, cropped_dir)
    inference_elapsed = time.perf_counter() - inference_start
    print(
        f"[*] 소요시간(전처리 완료 -> 추론+결과파일 작성 완료): "
        f"{format_elapsed(inference_elapsed)}"
    )

    total_elapsed = time.perf_counter() - pipeline_start
    print(
        f"[*] 최종 총 소요시간(전처리+추론): {format_elapsed(preprocess_elapsed + inference_elapsed)} "
        f"(측정 전체: {format_elapsed(total_elapsed)})"
    )


if __name__ == "__main__":
    main()
