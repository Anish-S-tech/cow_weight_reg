# CowScale — Cattle Weight Estimation from a Photo

Estimate a cow's live weight from a single side-view photo. CowScale combines a
**YOLOv8** cow detector with an **EfficientNetV2** regression model, and ships as
an offline **Android app** plus a lightweight **Flask web server**.

<p>
  <img alt="Android" src="https://img.shields.io/badge/Android-7.0%2B-3DDC84?logo=android&logoColor=white">
  <img alt="Kotlin" src="https://img.shields.io/badge/Kotlin-1.9-7F52FF?logo=kotlin&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow Lite" src="https://img.shields.io/badge/TFLite%20%2F%20LiteRT-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="ONNX Runtime" src="https://img.shields.io/badge/ONNX%20Runtime-005CED?logo=onnx&logoColor=white">
</p>

---

## Features

- **Fully on-device Android app**: no internet needed. Pick a photo from the gallery or use the live camera.
- **Cow detection guard**: rejects images with no cow or with more than one cow before estimating.
- **Test-time augmentation**: the server averages predictions over flipped and brightness/contrast variants for more stable results.
- **Weight range**: returns a point estimate plus a ±20 kg range.
- **Reproducible training**: Kaggle notebooks for data prep, training (EfficientNetV2-B0 / S, MobileNet) and ensembling.

## How it works

```
 photo ──► YOLOv8n (COCO class 19 "cow") ──► exactly one cow? ──► resize 224×224
                                                                        │
            weight (kg) ± range  ◄── calibration ×2.1 ◄── EfficientNetV2 regressor (TFLite)
```

| Stage       | Model                          | Runtime (Android)  | Runtime (server)   |
|-------------|--------------------------------|--------------------|--------------------|
| Detection   | YOLOv8n                        | ONNX Runtime 1.22  | Ultralytics        |
| Regression  | EfficientNetV2-B0 → TFLite     | LiteRT 1.4         | TensorFlow Lite    |

> **Calibration note:** the regressor was trained on lighter cattle (~150–250 kg).
> Raw outputs are multiplied by a calibration factor of `2.1` for heavier
> deployment cattle. Treat results as estimates, not a replacement for a scale.

## Project structure

```
.
├── CowWeightApp/            # Android app (Kotlin, CameraX, ONNX Runtime, LiteRT)
│   └── app/src/main/
│       ├── assets/          # yolov8n.onnx, clean_weight_model.tflite
│       └── java/com/cowweight/app/
├── server/                  # Flask web app (upload a photo from any phone on the LAN)
│   ├── app.py
│   └── templates/index.html
├── models/                  # Trained weights used by the server and scripts
│   ├── clean_weight_model.tflite
│   ├── effnetv2b0_best.keras
│   └── norm_state.json      # target normalisation stats (log-weight mean/std)
├── scripts/                 # CLI inference, batch evaluation and model export
├── training/                # Kaggle notebooks + runners for remote GPU training
│   └── notebooks/
└── requirements.txt
```

## Getting started

### 1. Android app

Requirements: Android Studio (Ladybug or newer), JDK 17, Android SDK 35.

```bash
cd CowWeightApp
./gradlew assembleDebug            # Windows: gradlew.bat assembleDebug
```

Install `app/build/outputs/apk/debug/app-debug.apk`, or open `CowWeightApp/`
in Android Studio and press **Run**. Minimum Android version: **7.0 (API 24)**.

<details>
<summary>Signed release build</summary>

1. Copy `keystore.properties.example` to `keystore.properties` and fill in your keystore details.
2. Run `./gradlew assembleRelease`.
3. Share `app/build/outputs/apk/release/app-release.apk`.

`keystore.properties` and `*.jks` are gitignored. Never commit them.
</details>

### 2. Web server

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python server/app.py
```

Open `http://<your-pc-ip>:5000` from a phone on the same Wi-Fi network and upload
a photo. The YOLOv8 segmentation weights (`yolov8n-seg.pt`) download
automatically on first run.

**API:** `POST /predict` with multipart field `image`:

```json
{ "weight_kg": 412.3, "range_low": 392, "range_high": 432, "confidence": 0.91 }
```

### 3. Command-line scripts

```bash
# Single image, TFLite model (same pipeline as the app)
python scripts/test_pipeline.py --image path/to/cow.jpg [--truth 420]

# Single image, full Keras model
python scripts/test_keras_model.py --image path/to/cow.jpg

# Evaluate on a labelled test split
python scripts/batch_test.py --csv test_split.csv --img_dir path/to/images \
                             --model models/effnetv2b0_best.keras

# Re-export YOLOv8n for mobile
python scripts/export_yolo_tflite.py
```

## Training

Training runs on Kaggle GPUs using the
[Cattle Weight Detection Model Dataset (12k)](https://www.kaggle.com/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k)
(8,084 train / 1,011 val / 1,011 test images).

| Notebook | Purpose |
|---|---|
| `01_dataset_exploration` | Dataset statistics and visual checks |
| `02a_data_prep` | Cleaning, splits, normalisation stats |
| `02b / 02c / 02d_train_*` | EfficientNetV2-B0, EfficientNetV2-S and MobileNet regressors |
| `02e_ensemble` | Ensemble evaluation |
| `03_camera_pipeline` | Detection + regression end-to-end pipeline |
| `04_local_pretrain_finetune` | Local pre-training / fine-tuning experiments |

To push and run notebooks remotely, place your Kaggle API token at
`~/.kaggle/kaggle.json`, then:

```bash
python training/kaggle_runner.py --notebook training/notebooks/02b_train_b0.ipynb --gpu
python training/auto_runner.py     # runs the training notebooks and auto-retries on known errors
```

## Limitations

- Works best on a clear **side-view** photo with the whole animal visible.
- Only one cow may be in the frame.
- Accuracy depends on breed and body condition. The calibration factor was tuned for heavier cattle than the training set.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EfficientNetV2](https://arxiv.org/abs/2104.00298) (Tan & Le, 2021)
- Kaggle dataset by *sadhliroomyprime*
