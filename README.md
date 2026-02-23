# RetinaViT-AD: Explainable Multi-Modal Attention Fusion Network for Retinal Disease Detection

**RetinaViT-AD** is a state-of-the-art hybrid deep learning framework designed for the early and accurate detection of vision-threatening conditions, specifically **Glaucoma** and **Diabetic Retinopathy (DR)**, from retinal fundus images. Unlike traditional CNNs that rely on local receptive fields, this model integrates a **Vision Transformer (ViT)** and a **Global Attention Module (GAM)** to capture long-range dependencies and clinically significant features.

## 🚀 Key Features

* 
**Hybrid Architecture**: Combines a pre-trained **EfficientNet-B3** backbone for local feature extraction with a **Vision Transformer** for global contextual modeling.


* 
**Global Attention Module (GAM)**: Enhances feature representation by focusing on critical retinal regions through simultaneous **Channel and Spatial Attention** mechanisms.


* **5-Fold Cross-Validation**: Implemented to ensure the model's robustness and stability across different data subsets, providing a reliable measure of performance.
* **Explainable AI (XAI)**: Integrated to provide transparency in clinical decision-making, allowing ophthalmologists to visualize the specific retinal areas driving the model's predictions.
* 
**Multi-Class Detection**: Capable of classifying eight clinically significant categories: Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, and others.



## 📊 Dataset: ODIR-5K

The model is trained and validated on the **Ocular Disease Intelligent Recognition (ODIR-5K)** link : https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k dataset:

* 
**Scale**: Fundus images from 5,000 patients.


* 
**Real-world Diversity**: Images captured across multiple hospitals using various fundus cameras (Canon, Zeiss, Kowa).


* 
**Preprocessing**: Includes multi-stage resizing (), intensity normalization, oversampling for class imbalance, and on-the-fly data augmentation (rotation, flipping, zooming).



## 🏗️ Model Architecture

1. 
**Feature Extraction**: EfficientNet-B3 backbone captures hierarchical local features.


2. 
**Attention Enhancement**: GAM applies channel and spatial weights to emphasize informative channels and locations.


3. 
**Global Modeling**: Transformer blocks with Multi-Head Self-Attention capture global dependencies across the image.


4. 
**Multi-Head Fusion**: Fuses diverse feature subspaces for a richer final representation.



## 📈 Performance (Summary)

| Metric | Value |
| --- | --- |
| **Accuracy** | 91.69% |
| **Precision** | 91.83% |
| **Recall** | 91.69% |
| **F1-Score** | 91.73% |
| **AUC** | 0.98 - 1.00 (Class-wise) |

Performance highlights from.

## 💻 Installation & Usage

### Prerequisites

* Python 3.8+ 


* TensorFlow 2.0+ / Keras 


* OpenCV, NumPy, Pandas 



### Training with 5-Fold Cross-Validation



### XAI Visualization

To generate interpretability heatmaps for a specific fundus image



---
