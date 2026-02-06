import os
import sys
import random
import builtins
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score

from transformers import (
    TrainingArguments,
    Trainer,
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
    DATA_ROOT_BASE, JSON_BASE, OUTPUT_DIR,
    MODEL_ID, IMAGE_SIZE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    MULTITASK, LABEL_SMOOTHING, SEED, NUM_WORKERS, USE_PEFT,
    # Utils
    is_main_process
)
from src.dataset import (
    load_all_data,
    DATASETS,
    TEST_DATASETS,
    init_augs,
    apply_transforms_robust,
    validate_paths,
    validate_images_openable,
    balance_real_fake,
    balance_three_class
)
from src.models import (
    DINOv3ForClassification,
    build_peft_model
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


def main():
    if not is_main_process():
        builtins.print = lambda *args, **kwargs: None

    # 1. 시드 고정 (재현성) - config에 값이 있을 때만
    if SEED is not None:
        from transformers import set_seed
        set_seed(SEED)

    print(f"[*] Starting Training...")
    print(f"    - Output Dir: {OUTPUT_DIR}")
    print(f"    - Model ID: {MODEL_ID}")
    print(f"    - Multitask: {MULTITASK}")

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
    model = build_peft_model(model)
    
    # 5. Training Arguments 설정 (training.py의 하드코딩 값 유지)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        
        # Optimizer & Scheduler
        lr_scheduler_type="cosine",
        warmup_steps=200,       # training.py 원본 값
        
        # Hardware Acceleration (training.py 원본 값)
        fp16=False,
        bf16=True,
        tf32=True,
        
        # Logging & Saving (WandB 제거 -> none)
        report_to="none",
        save_strategy="steps",
        save_steps=200,         # training.py 원본 값
        save_total_limit=10,    # training.py 원본 값
        eval_strategy="epoch",
        load_best_model_at_end=False,
        
        # DataLoader
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],
        
        # DDP
        ddp_find_unused_parameters=False,
    )

    # 6. Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    # 7. 학습 시작
    print("[*] Initiating Trainer.train()...")
    trainer.train()
    
    # 8. 최종 모델 저장
    trainer.save_model(OUTPUT_DIR)
    if is_main_process():
        print("[*] Done!")


if __name__ == "__main__":
    main()