## Hecto Deepfake Challenge 제출 코드 (likelihood_submission)

이 디렉터리는 대회 제출용 최소 필수 구조를 기준으로 정리된 코드입니다.

### 1) 프로젝트 구조

```text
likelihood_submission/
├── model/
│   ├── model.pt
│   └── model_config/
├── src/
│   ├── models.py
│   ├── dataset.py
│   └── utils.py
├── config/
│   └── config.yaml
├── env/
│   ├── Dockerfile
│   └── requirements.txt
├── train_data/
├── test_data/
├── train.py
├── eval.py
├── inference.py
└── README.md
```

### 2) 추론 환경 기준 (대회 안내 기준)

- GPU: L40S (48GB VRAM)
- CPU: 16 vCPUs (96GB RAM)
- DISK: 80GB
- CUDA: 11.8 ~ 12.6
- PyTorch: 2.5.0 권장

본 저장소의 Docker 이미지는 `pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime`를 사용합니다.

### 3) 환경 구성

#### 3-1. Docker 사용 (권장)

`likelihood_submission` 디렉터리에서:

```bash
docker build -f env/Dockerfile -t hecto-deepfake:latest .
```

실행:

```bash
docker run --gpus all --rm -it \
  -v $(pwd):/workspace/your_submission \
  -w /workspace/your_submission \
  hecto-deepfake:latest \
  python inference.py --clean_cropped
```

#### 3-2. 로컬 직접 설치

```bash
pip install -r env/requirements.txt
```

`torch==2.5.0` + CUDA 호환 환경을 별도로 맞춰주세요.

### 4) 추론 실행

기본값 실행(전처리 + 추론 + 제출파일 생성):

```bash
python inference.py
```

권장 실행(기존 크롭 결과 제거 후 전체 재생성):

```bash
python inference.py --clean_cropped
```

기본 경로:

- 입력 평가 데이터: `./test_data`
- 전처리 결과: `./cropped`
- 모델 가중치: `./model`
- RetinaFace onnx: `./retinaface/det_10g.onnx`
- 출력 CSV: `./result/submission.csv`

### 5) 추론 출력 형식

생성 파일:

- `result/submission.csv`

컬럼 형식:

- `filename`
- `prob`

### 6) inference.py 단계별 처리

`inference.py`는 아래 전 과정을 단일 엔트리포인트에서 수행합니다.

1. 데이터 전처리(이미지/영상 face detection, crop/alignment)
2. 모델 로드(`model/model.pt`)
3. 프레임/이미지 단위 추론
4. 영상 후처리 집계 및 최종 CSV 저장

또한 아래 시간 로그를 출력합니다.

- 시작 -> 전처리 완료
- 전처리 완료 -> 추론+결과파일 작성 완료
- 최종 총 소요시간(전처리+추론)

### 7) 학습 실행

```bash
python train.py
```

학습 설정은 `config/config.yaml`을 사용합니다.

### 8) 설정 파일

`config/config.yaml`의 데이터 경로는 제출 이식성을 위해 상대경로를 사용합니다.

- `data_root_base: ./train_data`
- `json_base: ./config/dataset_json`

### 9) 주의사항

- 추론은 오프라인 환경을 가정합니다.
- `test_data`는 추론 전용으로 사용하세요.
- `retinaface/det_10g.onnx` 파일이 없으면 전처리가 동작하지 않습니다.
