# RetinaViT-AD: Explainable Hybrid CNN-Transformer for Retinal Disease Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.0+-red?style=flat-square&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Preprint-yellow?style=flat-square)

**A Lightweight Hybrid CNN-Transformer Framework for Explainable Multi-Class Retinal Disease Classification with Cross-Domain Generalization**

[📄 Paper](#) · [📊 Dataset](#-dataset) · [🚀 Quick Start](#-installation--usage) · [📈 Results](#-results)

</div>

---

## 📌 Overview

**RetinaViT-AD** is a lightweight hybrid deep learning framework for automated multi-class retinal disease classification from fundus images. The model integrates:

- An **EfficientNet-B3** convolutional backbone for local feature extraction
- A **Global Attention Module (GAM)** for channel- and spatial-wise feature recalibration
- A **Vision Transformer (ViT) encoder** for long-range global context modelling
- A **Multi-Head Feature Fusion** layer to consolidate complementary representations
- **Grad-CAM** explainability for clinical transparency

With only **1.589M trainable parameters** and **~18ms inference latency**, RetinaViT-AD is designed for real-world deployment in portable screening devices and tele-ophthalmology platforms.

---

## 🏗️ Architecture

```
Input (224×224×3)
      │
      ▼
┌─────────────────┐
│  EfficientNet-B3 │  ← Local feature extraction (MBConv blocks)
│  CNN Backbone    │
└────────┬────────┘
         │  F_CNN ∈ ℝ^(h×w×1536)
         ▼
┌─────────────────┐
│ Global Attention │  ← Channel Attention + Spatial Attention
│   Module (GAM)   │
└────────┬────────┘
         │  F_GAM
         ▼
┌─────────────────┐
│Vision Transformer│  ← 8-head MHSA, D=1536, L=1 layer
│    Encoder       │
└────────┬────────┘
         │  X_trans
         ▼
┌─────────────────┐
│ Multi-Head Fusion│  ← Concat(F_GAM, X_trans) → 2 fusion blocks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Classification   │  ← GAP → BN → Dropout → FC → Softmax
│     Head         │  (8 disease classes)
└─────────────────┘
         │
         ▼
    Grad-CAM XAI
```

---

## 📊 Dataset

### Primary: ODIR-5K
| Property | Detail |
|----------|--------|
| Source | [Kaggle – ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) |
| Patients | 5,000 |
| Images | Bilateral fundus photographs (left + right eye) |
| Classes | 8 (Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other) |
| Cameras | Canon, Zeiss, Kowa |
| Split | 70% Train / 10% Val / 20% Test (stratified) |

### External Validation (Cross-Domain)
| Dataset | Purpose |
|---------|---------|
| [DDR](https://github.com/nkicsl/DDR-dataset) | Diabetic Retinopathy |
| [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection) | Diabetic Retinopathy grading |
| [MESSIDOR](https://www.adcis.net/en/third-party/messidor/) | Diabetic Retinopathy |
Main used External dataset link : https://www.kaggle.com/datasets/mohammadasimbluemoon/diabeticretinopathyddraptos2019messidorbalanced

> ⚠️ **No images from external datasets were used during training.** External evaluation is strictly zero-shot (no retraining).

---

## 📈 Results

### Internal Performance (ODIR-5K Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | **95.69%** |
| Precision | **95.81%** |
| Recall | **95.69%** |
| F1-Score | **95.73%** |
| Mean AUC | **0.9575** |

### Comparison with Baselines

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| VGG16 | 77.94% | 72.28% | 77.94% | 74.16% |
| InceptionV3 | 75.50% | 76.75% | 75.50% | 75.97% |
| ResNet50 | 85.50% | 85.92% | 85.50% | 85.52% |
| DenseNet121 | 86.63% | 86.62% | 86.63% | 86.45% |
| EfficientNet-B3 | 90.31% | 90.36% | 90.31% | 90.31% |
| **RetinaViT-AD (Ours)** | **95.69%** | **95.81%** | **95.69%** | **95.73%** |

### Cross-Domain Generalization (Zero-Shot)

| Dataset | Accuracy | AUC |
|---------|----------|-----|
| Internal (ODIR-5K, 8-class) | 95.69% | 0.9575 |
| External (DDR + APTOS + MESSIDOR, 5-class) | 92.40% | 0.9522 |

### Robustness & Reliability

| Evaluation | Result |
|------------|--------|
| 5-Fold Cross-Validation | 0.9204 ± 0.0158 |
| Expected Calibration Error (ECE) | 0.041 |
| Brier Score | 0.012 |
| Bootstrap 95% CI | [0.9113 – 0.9373] |
| Statistical Significance (paired t-test) | p = 0.006 |

### Computational Efficiency

| Metric | Value |
|--------|-------|
| Trainable Parameters | 1.589M |
| Model Size | 6.3 MB |
| Inference Time | ~18 ms/image |
| Training Time | ~34 sec/epoch |
| FLOPs | ~1.9 GFLOPs |
| GPU Memory | ~3.2 GB |

---

## 💻 Installation & Usage

### Prerequisites

```bash
Python >= 3.8
TensorFlow >= 2.0
Keras >= 2.0
OpenCV
NumPy
Pandas
```

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/RetinaViT-AD.git
cd RetinaViT-AD
```



### 3. Prepare Dataset

Download ODIR-5K from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) and organize as follows:

```
data/
├── ODIR-5K/
│   ├── Training Images/
│   ├── Testing Images/
│   └── full_df.csv
└── external/
    ├── DiabeticRetinopathyDDR+APTOS2019+Messidor+Balanced

```
External Dataset Link- https://www.kaggle.com/datasets/mohammadasimbluemoon/diabeticretinopathyddraptos2019messidorbalanced

### 4. Preprocess Data

```python
python preprocess.py \
  --data_dir data/ODIR-5K \
  --output_dir data/processed \
  --img_size 224 \
  --split_ratio 0.7 0.1 0.2
```

### 5. Train the Model

```python
python train.py \
  --data_dir data/processed \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --backbone efficientnetb3 \
  --num_classes 8
```

### 6. Train with 5-Fold Cross-Validation

```python
python train_crossval.py \
  --data_dir data/processed \
  --n_folds 5 \
  --epochs 40 \
  --batch_size 32
```

### 7. Evaluate on External Dataset (Cross-Domain)

```python
python evaluate_external.py \
  --model_path checkpoints/retinavit_ad_best.h5 \
  --external_dir data/external \
  --num_classes 5
```



---

## 🧩 Model Components

### Global Attention Module (GAM)

The GAM applies channel and spatial attention sequentially:

```
F_GAM = F_CNN ⊙ CA(F_CNN) ⊙ SA(F_CNN)
```

- **Channel Attention**: Recalibrates inter-channel responses via global average pooling → FC → Sigmoid
- **Spatial Attention**: Highlights lesion-prone spatial locations (optic disc, macula, vascular regions)

### Vision Transformer Encoder

- **Heads**: 8 parallel attention heads
- **Embedding Dim**: D = 1536 (d_k = 192 per head)
- **FFN Dim**: 2048
- **Layers**: L = 1 (deliberately lightweight)
- **Activation**: Swish
- **Normalization**: Pre-Layer Norm with residual connections

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input Size | 224 × 224 × 3 |
| CNN Backbone | EfficientNet-B3 (ImageNet pretrained) |
| Transformer Dim | 1536 |
| Attention Heads | 8 |
| Dropout Rate | 0.1 – 0.4 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Epochs | 50 |
| Loss | Categorical Cross-Entropy |

---

## 🔬 Explainability (Grad-CAM)

RetinaViT-AD integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize disease-specific attention regions:

```
α_k^c = (1/Z) Σ_ij (∂y^c / ∂A^k_ij)
L^c_XAI = ReLU(Σ_k α_k^c · A^k)
```

Heatmaps consistently align with clinically established retinal biomarkers:
- **Optic disc boundaries** (Glaucoma)
- **Microaneurysms & hemorrhagic lesions** (Diabetic Retinopathy)
- **Drusen deposits & RPE changes** (AMD)
- **Vascular irregularities** (Hypertension)

---



---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **ODIR-5K** dataset provided via Kaggle
- **EfficientNet-B3** pretrained weights from ImageNet via TensorFlow/Keras
- External validation datasets: DDR, APTOS 2019, MESSIDOR
- Experiments conducted on dual NVIDIA Tesla T4 GPUs

---

<div align="center">
⭐ If you find this work useful, please consider starring this repository!
</div>
