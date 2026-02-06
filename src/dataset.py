import os
import json
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

# Config는 기존 구조 유지 가정
from .utils import (
    DATA_ROOT_BASE, NUM_FRAMES_PER_VIDEO, TARGET_COMPRESSION, 
    MULTITASK, IMAGE_SIZE, JSON_BASE
)
from .utils import is_main_process

# ---------------------------------------------------------
# Constants & Catalogs
# ---------------------------------------------------------

_GENAI_DATASETS = {
    "ddim", "dit", "e4e", "e4s", "pixart", "rddm", "sd2.1", "sit",
    "stylegan2", "stylegan3", "styleganxl", "vqgan", "uniface",
    "midjourney", "collabdiff", "heygen", "dalle2", "dcface",
    "palette", "styleclip", "stargan", "starganv2",
}

LABEL_MAP = {
    "ffhq": 0, "imdb_wiki": 0, "celeba_real": 0, "UTKFace_real": 0, "UTKFace-Cropped": 0,
    "vggface2-224_real": 0, "vggface2-224": 0, "CASIA-WebFace_crop_real": 0,
    "FF-real": 0, "original": 0, "real": 0, "YouTube-real": 0, "Celeb-real": 0, "CelebDFv2_real": 0,
    "FF-SH": 1, "FF-F2F": 1, "FF-DF": 1, "FF-FS": 1, "FF-NT": 1, "FF-FH": 1,
    "Deepfakes": 1, "Face2Face": 1, "FaceSwap": 1, "NeuralTextures": 1,
    "fake": 1, "Celeb-synthesis": 1, "CelebDFv2_fake": 1,
    "FF-actors-real": 0, "FF-DFD": 1, "FaceShifter": 1, "DeepFakeDetection": 1,
    "blendface_real": 0, "blendface_fake": 1, "facedancer_real": 0, "facedancer_fake": 1,
    "fomm_real": 0, "fomm_fake": 1, "inswap_real": 0, "inswap_fake": 1,
    "simswap_real": 0, "simswap_fake": 1, "hyperreenact_real": 0, "hyperreenact_fake": 1,
    "danet_real": 0, "danet_fake": 1, "faceswap_real": 0, "faceswap_fake": 1,
    "facevid2vid_real": 0, "facevid2vid_fake": 1, "fsgan_real": 0, "fsgan_fake": 1,
    "lia_real": 0, "lia_fake": 1, "mcnet_real": 0, "mcnet_fake": 1,
    "mobileswap_real": 0, "mobileswap_fake": 1, "MRAA_real": 0, "MRAA_fake": 1,
    "one_shot_free_real": 0, "one_shot_free_fake": 1, "pirender_real": 0, "pirender_fake": 1,
    "sadtalker_real": 0, "sadtalker_fake": 1, "tpsm_real": 0, "tpsm_fake": 1,
    "wav2lip_real": 0, "wav2lip_fake": 1, "ddim_real": 0, "ddim_fake": 1,
    "DiT_real": 0, "DiT_fake": 1, "e4e_real": 0, "e4e_fake": 1,
    "e4s_real": 0, "e4s_fake": 1, "pixart_real": 0, "pixart_fake": 1,
    "rddm_real": 0, "rddm_fake": 1, "sd2.1_real": 0, "sd2.1_fake": 1,
    "SiT_real": 0, "SiT_fake": 1, "StyleGAN2_real": 0, "StyleGAN2_fake": 1,
    "StyleGAN3_real": 0, "StyleGAN3_fake": 1, "StyleGANXL_real": 0, "StyleGANXL_fake": 1,
    "VQGAN_real": 0, "VQGAN_fake": 1, "stargan_real": 0, "stargan_fake": 1,
    "starganv2_real": 0, "starganv2_fake": 1, "uniface_real": 0, "uniface_fake": 1,
    "CollabDiff_real": 0, "CollabDiff_fake": 1, "deepfacelab_real": 0, "deepfacelab_fake": 1,
    "DFDC_real": 0, "DFDC_fake": 1, "DFDCP_real": 0, "DFDCP_fake": 1,
    "heygen_real": 0, "heygen_fake": 1, "MidJourney_real": 0, "MidJourney_fake": 1,
    "styleclip_real": 0, "styleclip_fake": 1, "UADFV_real": 0, "UADFV_fake": 1,
    "whichisreal_real": 0, "whichisreal_fake": 1, "whichfaceisreal_real": 0, "whichfaceisreal_fake": 1
}

DATASETS = [
    { "name": "FaceForensics++", "json": f"{JSON_BASE}/FaceForensics++.json", "enabled": True },
    { "name": "Celeb-DF-v2", "json": f"{JSON_BASE}/Celeb-DF-v2.json", "enabled": True },
    { "name": "celeba_data", "json": f"{JSON_BASE}/celeba_data.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/celeba_data" },
    { "name": "UTKFace-Cropped", "json": f"{JSON_BASE}/UTKFace-Cropped.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/UTKFace-Cropped" },
    { "name": "ffhq", "json": f"{JSON_BASE}/ffhq.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/ffhq" },
    { "name": "imdb_wiki", "json": f"{JSON_BASE}/imdb_wiki.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/imdb_wiki" },
    { "name": "whichisreal", "json": f"{JSON_BASE}/whichisreal.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/whichfaceisreal" },
    { "name": "DFDC", "json": f"{JSON_BASE}/DFDC.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/DFDC" },
    { "name": "DFD", "json": f"{JSON_BASE}/DFD.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/DFD" },
    { "name": "CollabDiff", "json": f"{JSON_BASE}/CollabDiff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/CollabDiff" },
    { "name": "DFDCP", "json": f"{JSON_BASE}/DFDCP.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/DFDCP" },
    { "name": "UADFV", "json": f"{JSON_BASE}/UADFV.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/UADFV" },
    { "name": "styleclip", "json": f"{JSON_BASE}/styleclip.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/styleclip" },
    { "name": "deepfacelab", "json": f"{JSON_BASE}/deepfacelab.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/deepfacelab" },
    { "name": "heygen", "json": f"{JSON_BASE}/heygen.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/heygen" },
    # DF40 - FF Source
    { "name": "blendface_ff", "json": f"{JSON_BASE}/blendface_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/blendface/ff" },
    { "name": "facedancer_ff", "json": f"{JSON_BASE}/facedancer_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/facedancer/ff" },
    { "name": "fomm_ff", "json": f"{JSON_BASE}/fomm_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/fomm/ff" },
    { "name": "inswap_ff", "json": f"{JSON_BASE}/inswap_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/inswap/ff" },
    { "name": "simswap_ff", "json": f"{JSON_BASE}/simswap_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/simswap/ff" },
    { "name": "hyperreenact_ff", "json": f"{JSON_BASE}/hyperreenact_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/hyperreenact/ff" },
    { "name": "danet_ff", "json": f"{JSON_BASE}/danet_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/danet/ff" },
    { "name": "faceswap_ff", "json": f"{JSON_BASE}/faceswap_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/faceswap/ff" },
    { "name": "facevid2vid_ff", "json": f"{JSON_BASE}/facevid2vid_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/facevid2vid/ff" },
    { "name": "fsgan_ff", "json": f"{JSON_BASE}/fsgan_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/fsgan/ff" },
    { "name": "lia_ff", "json": f"{JSON_BASE}/lia_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/lia/ff" },
    { "name": "mcnet_ff", "json": f"{JSON_BASE}/mcnet_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/mcnet/ff" },
    { "name": "mobileswap_ff", "json": f"{JSON_BASE}/mobileswap_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/mobileswap/ff" },
    { "name": "MRAA_ff", "json": f"{JSON_BASE}/MRAA_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/MRAA/ff" },
    { "name": "one_shot_free_ff", "json": f"{JSON_BASE}/one_shot_free_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/one_shot_free/ff" },
    { "name": "pirender_ff", "json": f"{JSON_BASE}/pirender_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/pirender/ff" },
    { "name": "sadtalker_ff", "json": f"{JSON_BASE}/sadtalker_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/sadtalker/ff" },
    { "name": "tpsm_ff", "json": f"{JSON_BASE}/tpsm_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/tpsm/ff" },
    { "name": "wav2lip_ff", "json": f"{JSON_BASE}/wav2lip_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/wav2lip/ff" },
    { "name": "ddim_ff", "json": f"{JSON_BASE}/ddim_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/ddim/ff" },
    { "name": "DiT_ff", "json": f"{JSON_BASE}/DiT_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/DiT/ff" },
    { "name": "e4e_ff", "json": f"{JSON_BASE}/e4e_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/e4e/ff" },
    { "name": "e4s_ff", "json": f"{JSON_BASE}/e4s_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/e4s/ff" },
    { "name": "pixart_ff", "json": f"{JSON_BASE}/pixart_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/pixart/ff" },
    { "name": "rddm_ff", "json": f"{JSON_BASE}/rddm_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/RDDM/ff" },
    { "name": "sd2.1_ff", "json": f"{JSON_BASE}/sd2.1_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/sd2.1/ff" },
    { "name": "SiT_ff", "json": f"{JSON_BASE}/SiT_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/SiT/ff" },
    { "name": "StyleGAN2_ff", "json": f"{JSON_BASE}/StyleGAN2_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/StyleGAN2/ff" },
    { "name": "StyleGAN3_ff", "json": f"{JSON_BASE}/StyleGAN3_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/StyleGAN3/ff" },
    { "name": "StyleGANXL_ff", "json": f"{JSON_BASE}/StyleGANXL_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/StyleGANXL/ff" },
    { "name": "VQGAN_ff", "json": f"{JSON_BASE}/VQGAN_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/VQGAN/ff" },
    { "name": "uniface_ff", "json": f"{JSON_BASE}/uniface_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/uniface/ff" },
    # DF40 - CDF Source
    { "name": "blendface_cdf", "json": f"{JSON_BASE}/blendface_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/blendface/cdf" },
    { "name": "facedancer_cdf", "json": f"{JSON_BASE}/facedancer_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/facedancer/cdf" },
    { "name": "fomm_cdf", "json": f"{JSON_BASE}/fomm_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/fomm/cdf" },
    { "name": "inswap_cdf", "json": f"{JSON_BASE}/inswap_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/inswap/cdf" },
    { "name": "simswap_cdf", "json": f"{JSON_BASE}/simswap_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/simswap/cdf" },
    { "name": "hyperreenact_cdf", "json": f"{JSON_BASE}/hyperreenact_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/hyperreenact/cdf" },
    { "name": "danet_cdf", "json": f"{JSON_BASE}/danet_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/danet/cdf" },
    { "name": "faceswap_cdf", "json": f"{JSON_BASE}/faceswap_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/faceswap/cdf" },
    { "name": "facevid2vid_cdf", "json": f"{JSON_BASE}/facevid2vid_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/facevid2vid/cdf" },
    { "name": "fsgan_cdf", "json": f"{JSON_BASE}/fsgan_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/fsgan/cdf" },
    { "name": "lia_cdf", "json": f"{JSON_BASE}/lia_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/lia/cdf" },
    { "name": "mcnet_cdf", "json": f"{JSON_BASE}/mcnet_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/mcnet/cdf" },
    { "name": "mobileswap_cdf", "json": f"{JSON_BASE}/mobileswap_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/mobileswap/cdf" },
    { "name": "MRAA_cdf", "json": f"{JSON_BASE}/MRAA_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/MRAA/cdf" },
    { "name": "one_shot_free_cdf", "json": f"{JSON_BASE}/one_shot_free_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/one_shot_free/cdf" },
    { "name": "pirender_cdf", "json": f"{JSON_BASE}/pirender_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/pirender/cdf" },
    { "name": "sadtalker_cdf", "json": f"{JSON_BASE}/sadtalker_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/sadtalker/cdf" },
    { "name": "tpsm_cdf", "json": f"{JSON_BASE}/tpsm_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/tpsm/cdf" },
    { "name": "wav2lip_cdf", "json": f"{JSON_BASE}/wav2lip_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/wav2lip/cdf" },
    { "name": "ddim_cdf", "json": f"{JSON_BASE}/ddim_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/ddim/cdf" },
    { "name": "DiT_cdf", "json": f"{JSON_BASE}/DiT_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/DiT/cdf" },
    { "name": "e4e_cdf", "json": f"{JSON_BASE}/e4e_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/e4e/cdf" },
    { "name": "e4s_cdf", "json": f"{JSON_BASE}/e4s_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/e4s/cdf" },
    { "name": "rddm_cdf", "json": f"{JSON_BASE}/rddm_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/RDDM/cdf" },
    { "name": "sd2.1_cdf", "json": f"{JSON_BASE}/sd2.1_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/sd2.1/cdf" },
    { "name": "SiT_cdf", "json": f"{JSON_BASE}/SiT_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/SiT/cdf" },
    { "name": "StyleGAN2_cdf", "json": f"{JSON_BASE}/StyleGAN2_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/StyleGAN2/cdf" },
    { "name": "StyleGAN3_cdf", "json": f"{JSON_BASE}/StyleGAN3_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/StyleGAN3/cdf" },
    { "name": "StyleGANXL_cdf", "json": f"{JSON_BASE}/StyleGANXL_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/StyleGANXL/cdf" },
    { "name": "VQGAN_cdf", "json": f"{JSON_BASE}/VQGAN_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/VQGAN/cdf" },
    { "name": "uniface_cdf", "json": f"{JSON_BASE}/uniface_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/uniface/cdf" },
    { "name": "pixart_cdf", "json": f"{JSON_BASE}/pixart_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/pixart/cdf" },
    { "name": "stargan", "json": f"{JSON_BASE}/stargan.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/stargan" },
    { "name": "starganv2", "json": f"{JSON_BASE}/starganv2.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/starganv2" },
    { "name": "midjourney_ff", "json": f"{JSON_BASE}/midjourney_ff.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/midjourney/ff" },
    { "name": "midjourney_cdf", "json": f"{JSON_BASE}/midjourney_cdf.json", "enabled": True, "root": f"{DATA_ROOT_BASE}/midjourney/cdf" },
]

TEST_DATASETS = []

# ---------------------------------------------------------
# Path Helpers
# ---------------------------------------------------------

def _dataset_base_name(dataset_name):
    name = dataset_name
    if name.endswith("_ff"):
        name = name[:-3]
    if name.endswith("_cdf"):
        name = name[:-4]
    return name.lower()

def _is_ai_generated_dataset(dataset_name):
    base = _dataset_base_name(dataset_name)
    if base in _GENAI_DATASETS:
        return True
    return any(key in base for key in _GENAI_DATASETS)

def _derive_dataset_base(dataset_name):
    if dataset_name.endswith("_ff"):
        return dataset_name[:-3], "ff"
    if dataset_name.endswith("_cdf"):
        return dataset_name[:-4], "cdf"
    return dataset_name, None

def _strip_dataset_prefix(path, base_name, split_name):
    if split_name and path.startswith(f"{base_name}/{split_name}/"):
        return path[len(f"{base_name}/{split_name}/"):]
    if path.startswith(f"{base_name}/"):
        return path[len(base_name) + 1:]
    if split_name and path.startswith(f"{split_name}/"):
        return path[len(split_name) + 1:]
    return path

def _normalize_path(path):
    path = path.replace("\\", "/")
    path = path.lstrip("./")
    path = path.replace("DF40_train/", "").replace("DF40/", "")
    path = path.replace("/cdf/cdf/", "/cdf/").replace("/ff/ff/", "/ff/")
    return path.lstrip("/")

def smart_path_fixer(image_path, root_dir, dataset_name):
    if not image_path:
        return None

    path = image_path.replace("\\", "/")

    if os.path.isabs(path) and os.path.exists(path):
        return path

    path = _normalize_path(path)

    candidate = os.path.join(DATA_ROOT_BASE, path)
    if os.path.exists(candidate):
        return candidate

    base_name, split_name = _derive_dataset_base(dataset_name)
    stripped = _strip_dataset_prefix(path, base_name, split_name)
    candidate = os.path.join(root_dir, stripped)
    if os.path.exists(candidate):
        return candidate

    if "/Celeb-real/" in path or "/YouTube-real/" in path:
        cdf_candidates = []
        if "/Celeb-real/" in path:
            cdf_candidates.append(path.replace("/Celeb-real/", "/Fake_from_Celeb-real/"))
        if "/YouTube-real/" in path:
            cdf_candidates.append(path.replace("/YouTube-real/", "/Fake_from_Youtube-real/"))

        for cand in cdf_candidates:
            full_cand = os.path.join(DATA_ROOT_BASE, cand)
            if os.path.exists(full_cand):
                return full_cand

            stripped_cand = _strip_dataset_prefix(cand, base_name, split_name)
            p_root = os.path.join(root_dir, stripped_cand)
            if os.path.exists(p_root):
                return p_root

    if "e4e" in dataset_name.lower() and "inversions" in path:
        basename = os.path.basename(path)
        name, _ = os.path.splitext(basename)
        if name.isdigit():
            file_id = int(name)
            bucket = (file_id - 1) // 32
            candidates = [bucket, bucket + 1, bucket - 1]
            for b in candidates:
                if b < 0:
                    continue
                p_guess = os.path.join(root_dir, f"{b:03d}", basename)
                if os.path.exists(p_guess):
                    return p_guess

    if "/frames/" in stripped:
        suffix = stripped.split("/frames/", 1)[1]
        p_frames = os.path.join(root_dir, "frames", suffix)
        if os.path.exists(p_frames):
            return p_frames

    return os.path.join(root_dir, stripped)

# ---------------------------------------------------------
# Parsing & Loading
# ---------------------------------------------------------

def parse_dataset_json(json_path, split_name, data_root, dataset_name, target_comp=TARGET_COMPRESSION):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        print(f"[Error] JSON Load Failed: {json_path}")
        return []

    flat_data = []
    root_node = None
    if isinstance(data, dict):
        if dataset_name in data:
            root_node = data[dataset_name]
        elif "FaceForensics++" in data and "FaceForensics" in dataset_name:
            root_node = data["FaceForensics++"]
        else:
            for k, v in data.items():
                if dataset_name.lower() in k.lower():
                    root_node = v
                    break

    if not root_node or not isinstance(root_node, dict):
        valid_fallback = False
        for k in data.keys():
            for l_key in LABEL_MAP:
                if l_key.lower() in k.lower():
                    valid_fallback = True
                    break
            if valid_fallback:
                break

        if valid_fallback:
            root_node = data
        else:
            print(f"[Warning] Could not find root node for {dataset_name} in JSON")
            return []

    for class_label, class_data in root_node.items():
        mapped_label = None
        best_len = -1
        for key, val in LABEL_MAP.items():
            if key.lower() in class_label.lower():
                if len(key) > best_len:
                    mapped_label = val
                    best_len = len(key)

        if mapped_label is None:
            continue

        if MULTITASK and mapped_label == 1:
            mapped_label = 2 if _is_ai_generated_dataset(dataset_name) else 1

        if split_name not in class_data:
            continue

        sub_info = class_data[split_name]

        priorities = [target_comp, 'c40', 'c23']
        video_dict = None

        first_val = next(iter(sub_info.values())) if sub_info else None
        if isinstance(first_val, dict) and ('frames' in first_val or 'label' in first_val):
            video_dict = sub_info
        else:
            for p in priorities:
                if p in sub_info:
                    video_dict = sub_info[p]
                    break

        if not video_dict:
            continue

        for video_id, content in video_dict.items():
            raw_frames = content.get('frames', [])
            if not raw_frames:
                continue

            frames = sorted(raw_frames)
            total_frames = len(frames)

            if mapped_label == 0:
                selected_frames = frames
            elif total_frames > NUM_FRAMES_PER_VIDEO:
                indices = np.linspace(0, total_frames - 1, NUM_FRAMES_PER_VIDEO, dtype=int)
                selected_frames = [frames[i] for i in indices]
            else:
                selected_frames = frames

            for raw_path in selected_frames:
                full_p = smart_path_fixer(raw_path, data_root, dataset_name)
                flat_data.append({"image_path": full_p, "label": mapped_label})

    return flat_data

def load_all_data(datasets_config, train_only=False, test_only=False):
    train_all, test_all = [], []
    train_seen, test_seen = set(), set()

    for ds in datasets_config:
        if not ds.get("enabled", False):
            continue
        print(f"[*] Processing: {ds['name']}")

        if "root" in ds:
            base_root = ds["root"]
        else:
            base_root = f"{DATA_ROOT_BASE}/{ds['name']}"

        if not test_only:
            t = parse_dataset_json(ds['json'], 'train', base_root, ds['name'], TARGET_COMPRESSION)

            if not t and "_cdf" in ds['name']:
                print(f"    [Info] {ds['name']} train split is empty (Common for CDF). Fallback to using 'test' split data for training.")
                t = parse_dataset_json(ds['json'], 'test', base_root, ds['name'], TARGET_COMPRESSION)

            new_t = []
            for item in t:
                path = item['image_path']
                if path and path not in train_seen:
                    train_seen.add(path)
                    new_t.append(item)

            train_all.extend(new_t)
            print(f"    -> Train found: {len(new_t)} (Raw: {len(t)})")

        if not train_only:
            v = parse_dataset_json(ds['json'], 'test', base_root, ds['name'], TARGET_COMPRESSION)

            new_v = []
            for item in v:
                path = item['image_path']
                if path and path not in test_seen:
                    test_seen.add(path)
                    new_v.append(item)

            test_all.extend(new_v)
            print(f"    -> Test found:  {len(new_v)} (Raw: {len(v)})")

    return train_all, test_all

# ---------------------------------------------------------
# Augmentation & Validation
# ---------------------------------------------------------

train_aug = None
val_aug = None

def init_augs(image_size=IMAGE_SIZE):
    global train_aug, val_aug
    transforms_list = [
        A.HorizontalFlip(p=0.5),
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]

    train_aug = A.Compose(transforms_list)
    print(f"[*] Train Augmentations: {transforms_list}")

    val_aug = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_aug, val_aug

def apply_transforms_robust(batch, is_train=True):
    if train_aug is None or val_aug is None:
        init_augs()

    pixel_values, labels = [], []
    transform = train_aug if is_train else val_aug
    error_count = 0
    logged = 0
    max_logs = 5

    for path, label in zip(batch['image_path'], batch['label']):
        if path is None or not os.path.exists(path):
            error_count += 1
            if logged < max_logs:
                print(f"[Warning] Dropped missing file: {path}")
                logged += 1
            continue

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                image_np = np.array(img)
            processed = transform(image=image_np)['image']
            pixel_values.append(processed)
            labels.append(label)
        except Exception as e:
            error_count += 1
            if logged < max_logs:
                print(f"[Warning] Dropped corrupted file: {path} ({e})")
                logged += 1
            continue

    if error_count > 0:
        print(f"[Warning] Dropped {error_count} samples in this batch.")

    return {"pixel_values": pixel_values, "labels": labels}

def validate_paths(samples, split_label, max_workers=16, sample_limit=200):
    def check_path(item):
        if os.path.exists(item['image_path']):
            return item
        return None

    desc = "Checking Train Paths" if "train" in split_label else "Checking Test Paths"
    print(f"[*] Validating paths for {len(samples)} {split_label} samples...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        valid_list = list(
            tqdm(
                executor.map(check_path, samples),
                total=len(samples),
                desc=desc,
                disable=not is_main_process(),
            )
        )

    valid_list = [x for x in valid_list if x is not None]
    invalid_count = len(samples) - len(valid_list)
    print(f"[*] Path Validation Results ({'Train' if 'train' in split_label else 'Test'}):")
    print(f"    - Original: {len(samples)}")
    print(f"    - Valid:    {len(valid_list)}")
    print(f"    - Invalid:  {invalid_count} (Dropped)")
    if invalid_count > 0:
        missing_samples = [x['image_path'] for x in samples if not os.path.exists(x['image_path'])][:sample_limit]
        label = "train" if "train" in split_label else "test"
        print(f"[!] Sample missing {label} paths: {missing_samples}")

    return valid_list

def validate_images_openable(samples, split_label, max_workers=16, sample_limit=50):
    def check_openable(item):
        path = item["image_path"]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                _ = img.size
            return item, None
        except Exception as e:
            return None, f"{path} ({e})"

    print(f"[*] Validating image readability for {len(samples)} {split_label} samples...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(check_openable, samples),
                total=len(samples),
                desc=f"Checking {split_label} readability",
                disable=not is_main_process(),
            )
        )

    valid_list = [item for item, err in results if item is not None]
    invalid_errors = [err for item, err in results if err is not None]
    invalid_count = len(samples) - len(valid_list)

    print(f"[*] Readability Validation Results ({split_label}):")
    print(f"    - Original: {len(samples)}")
    print(f"    - Valid:    {len(valid_list)}")
    print(f"    - Invalid:  {invalid_count} (Dropped)")
    if invalid_errors:
        preview = invalid_errors[:sample_limit]
        print(f"[!] Sample unreadable {split_label} paths: {preview}")

    return valid_list

def balance_real_fake(train_list):
    labels_raw = [x['label'] for x in train_list]
    real_count = sum(1 for x in labels_raw if x == 0)
    fake_count = len(labels_raw) - real_count

    print(f"\n[Raw Stats] Real(0): {real_count}, Fake(1+): {fake_count}")
    return train_list

def balance_three_class(train_list):
    labels_raw = [x['label'] for x in train_list]
    c0 = sum(1 for x in labels_raw if x == 0)
    c1 = sum(1 for x in labels_raw if x == 1)
    c2 = sum(1 for x in labels_raw if x == 2)

    print(f"\n[Raw Stats] Real(0): {c0}, Fake0(1): {c1}, Fake1(2): {c2}")
    return train_list