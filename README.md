
<div align="center">

# Lightweight Face Anti-Spoof (MobileNetV4, ONNX)

[![License](https://img.shields.io/badge/License-Apache%202.0-black.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-black)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-black)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-black)](https://onnx.ai/)

</div>

![Face Anti-Spoofing Header](assets/header.png)

A face anti-spoof (liveness) classifier that predicts three classes: **real**, **photo attack**, and **video attack**. Originally implemented in [Suri](https://github.com/johnraivenolazo/suri) and kept here as a standalone training/export repo.

---

## Quick Start

**Run the demo with webcam:**

```bash
python demo.py --camera
```

**Run the demo with an image:**

```bash
python demo.py --image path/to/image.jpg
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- OpenCV
- ONNX Runtime
- See `requirements.txt` for full list

---

## Usage

### Demo (Inference Only)

**Webcam:**
```bash
python demo.py --camera
```

**Single image:**
```bash
python demo.py --image path/to/image.jpg
```

**Custom models:**
```bash
python demo.py --image image.jpg \
  --face-model models/face_detection_yunet_2023mar.onnx \
  --antispoof-model models/best_224.onnx \
  --threshold 0.6
```

The demo uses YuNet for face detection and the trained antispoof model for liveness detection. Results show colored bounding boxes: **green** for "Real" faces (live_score ≥ threshold), **red** for "Spoof" faces. Default threshold is 0.5.

### Training

Train your own model:

```bash
python train.py --data-root Cropped_Dataset --save-dir models
```

**Expected dataset structure:**
- `Cropped_Dataset/metas/labels/train_label.json`
- `Cropped_Dataset/metas/labels/test_label.json`

**Common options:**
```bash
python train.py \
  --data-root Cropped_Dataset \
  --save-dir models \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-3 \
  --weighted-sampler \
  --pretrained
```

### Data Preparation

Prepare your dataset by cropping faces:

```bash
python data_prep.py --orig_dir /path/to/dataset_root --crop_dir Cropped_Dataset
```

![Data preparation overview](assets/data_prep.png)

**Dataset requirements:**
- Image files (`.jpg` or `.png`)
- Bounding box files: `image.jpg` → `image_BB.txt` (format: `x y w h`)
- Label JSON files: `metas/labels/train_label.json` and `metas/labels/test_label.json`

![Data preparation overview](assets/data_prep2.png)

**Options:**
```bash
python data_prep.py \
  --orig_dir /path/to/dataset_root \
  --crop_dir Cropped_Dataset \
  --size 224 \
  --bbox_inc 1.5 \
  --spoof_types 0 1 2 3 7 8 9
```

### Export to ONNX

Convert PyTorch checkpoint to ONNX:

```bash
python to_onnx.py --input model.pth --output model.onnx
```

**Options:**
```bash
python to_onnx.py \
  --input model.pth \
  --output model.onnx \
  --model-name mobilenetv4_conv_small.e2400_r224_in1k \
  --num-classes 3 \
  --image-size 224
```

---

## Model Performance

The pretrained model achieves **99% recall** on live (real) faces. Trained on a balanced dataset with aggressive augmentation and class-weighted sampling.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Real | 0.8252 | 0.9920 | 0.9009 | 19,923 |
| Photo | 0.9417 | 0.8316 | 0.8832 | 15,104 |
| Video | 0.8738 | 0.7386 | 0.8005 | 14,619 |
| **Accuracy** | | | **0.8686** | 49,646 |
| **Macro avg** | 0.8802 | 0.8541 | 0.8616 | 49,646 |
| **Weighted avg** | 0.8749 | 0.8686 | 0.8660 | 49,646 |

[View pre-trained models](models/)

---

## Features

- **Backbone**: MobileNetV4 (feature extractor) with a 3-class classifier head
- **Clean Architecture**: Organized codebase following clean architecture principles
- **Export**: `to_onnx.py` exports a `.pth` checkpoint to a `.onnx` file
- **Dataset prep**: `data_prep.py` crops faces using bounding boxes and produces a fixed-size dataset

---

## Project Structure

```
src/
├── core/                # Core logic and entities
│   ├── labels.py        # LabelSpec and label handling
│   └── models.py        # Model configuration and creation
├── app/                 # Use cases and application logic
│   ├── inference.py     # Face detection and antispoof inference
│   └── training.py      # Model training and evaluation
├── infra/               # External dependencies and implementations
│   ├── data.py          # Data loading and transforms
│   ├── preprocess.py    # Image preprocessing utilities
│   ├── checkpoint.py    # Model checkpoint management
│   ├── sampler.py       # Data sampling strategies
│   ├── data_prep.py     # Dataset preparation
│   └── export_onnx.py   # ONNX model export
```

---

## License

Apache-2.0. See `LICENSE`.
