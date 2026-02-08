# Hecto Deepfake Challenge 제출 코드

## 1. 문서 목적
이 문서는 `likelihood_submission` 제출 패키지의 실행 방법, 대회 규칙 대응 현황, 운영 전 최종 점검 항목을 정리한 문서입니다.

---

## 2. 프로젝트 개요 및 규칙 대응 범위
본 저장소는 이미지/영상 딥페이크 이진 분류(Real/Fake) 제출을 위한 구조입니다.

- 최종 판별 모델 가중치: `model/model.pt` 단일 파일
- 추론 엔트리포인트: `inference.py`
- 학습 엔트리포인트: `train.py`
- 전처리(face crop/alignment): `retinaface/det_10g.onnx` 기반

영상은 프레임 단위로 분해 후 이미지 단위로 추론하고, 프레임별 독립 추론 결과를 후처리로 집계합니다.

---

## 3. 실제 프로젝트 구조
아래는 현재 저장소 기준 실제 구조입니다.

```text
likelihood_submission/
├── model/
│   └── model.pt
├── backbone/
│   └── facebook/dinov3-vith16plus-pretrain-lvd1689m/
├── retinaface/
│   └── det_10g.onnx
├── src/
│   ├── models.py
│   ├── dataset.py
│   └── utils.py
├── config/
│   ├── config.yaml
│   └── dataset_json/
├── env/
│   ├── Dockerfile
│   └── requirements.txt
├── train_data/
├── test_data/
├── train.py
├── inference.py
└── README.md
```

---

## 4. 제출 요건 충족 상태(파일 기준)
| 항목 | 상태 | 비고 |
|---|---|---|
| `model/model.pt` | 충족 | 최종 단일 분류 모델 가중치 |
| `config/config.yaml` | 충족 | 하이퍼파라미터/경로 |
| `env/Dockerfile` | 충족 | 제출 재현용 컨테이너 정의 |
| `env/requirements.txt` | 충족 | 파이썬 의존성 |
| `train.py` | 충족 | 학습 엔트리포인트 |
| `inference.py` | 충족 | 추론 엔트리포인트 |
| `README.md` | 충족 | 본 문서 |
| `src/models.py`, `src/dataset.py`, `src/utils.py` | 충족(희망) | 모듈 분리 구조 |

---

## 5. 환경 및 리소스
### 5.1 저장소 Docker 베이스
`env/Dockerfile` 기준 베이스 이미지는 아래와 같습니다.

- `nvcr.io/nvidia/pytorch:24.08-py3`

---

## 6. Docker 사용법
### 6.1 이미지 빌드
프로젝트 루트(`likelihood_submission`)에서 실행:

```bash
docker build -f env/Dockerfile -t likelihood-submission:latest .
```

### 6.2 컨테이너 실행
`--gpus all`, `--shm-size=90g`등을 포함해 실행:

예:
```bash
docker run --gpus all --shm-size=90g --rm -it \
  -v $(pwd):/likelihood_submission \
  -w /likelihood_submission \
  likelihood-submission:latest bash
```

### 6.3 컨테이너 내부에서 추론 실행
```bash
python inference.py
```

### 6.4 학습데이터 다운로드
학습데이터를 train_data에 위치시키면 됩니다.
학습 데이터 링크: https://drive.google.com/file/d/1xSmhkJoTs_QNQ9wkRuK2Uoz__Eeh456V/view?usp=sharing

예시 디렉토리 구조(`train_data/train_data`처럼 중첩되지 않도록 주의):
```text
train_data/
├── Celeb-DF-v1/
├── Celeb-DF-v2/
├── FaceForensics++/
├── DFDC/
├── DFDCP/
├── UADFV/
├── ffhq/
├── celeba_data/
├── deepfacelab/
├── faceswap/
├── one_shot_free/
├── wav2lip/
├── StyleGAN2/
├── StyleGAN3/
├── StyleGANXL/
├── VQGAN/
├── DiT/
├── SiT/
├── MidJourney/
├── pixart/
└── ... (기타 데이터 폴더들)
```


### 6.5 컨테이너 내부에서 학습 실행
```bash
python train.py
```

---

## 7. 추론 실행 가이드
### 7.1 실행(실행인자 기본값 권장)
```bash
python inference.py
```

주의: GPU 종류, 드라이버/CUDA/패키지 버전, 영상 디코딩 환경(ffmpeg 등)에 따라 얼굴 크롭 결과가 미세하게 달라질 수 있습니다.
따라서 비트 단위의 완전한 재현성은 보장되지 않습니다.

### 7.2 입출력 형식
- 출력 파일: `result/submission.csv`
- 출력 컬럼:
  - `filename`
  - `prob`

### 7.3 기본 경로
- 입력 평가 데이터: `./test_data`
- 전처리 결과: `./cropped`
- 학습된 분류 모델 가중치: `./model` (내부 `model.pt`)
- RetinaFace ONNX(크롭모델): `./retinaface/det_10g.onnx`
- 결과 CSV: `./result/submission.csv`

### 7.4 주요 추론 옵션
```bash
python inference.py \
  --clean_cropped \
  --batch_size 64
  --skip_preprocess
```

- `--batch_size`: 분류 추론 배치 크기
- `--clean_cropped`: 이전 크롭 결과 삭제 후 전처리 재실행
- `--skip_preprocess`: 크롭/전처리 결과인 cropped/ 가 이미 존재한다면 전처리를 무시하고 추론만 진행

---

## 8. 실측 벤치마크(참고값)
측정 일시: **2026-02-06**  
명령: `python inference.py --input_root test_data --clean_cropped ...`  
데이터: `test_data` 500개(영상 255, 이미지 245)

- 전처리 시간: `00:09:02`
- 추론+CSV 시간: `00:00:59`
- 총 시간: `00:10:01`
- 결과 CSV: 500행, `filename` 유니크 500

중요:
- 위 수치는 현재 측정 장비 기준 참고값입니다.
- **대회 최종 판단은 주최측 환경에서 재측정해야 합니다.**

---
