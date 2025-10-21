# 🎭 Multimodal Sentiment Analysis for Emotion and Reaction

This repository contains the implementation of a **multimodal emotion recognition system** combining **audio** and **visual** information for classifying human emotions.  
It integrates CNN-based feature extraction (EfficientFace, Conv1D) with **cross-modal fusion mechanisms** (Intermediate Attention and Late Transformer) for improved emotional understanding.

---

## 🧠 Overview

The system aims to recognize human emotions from short video clips by combining two modalities:
- **Visual stream**: facial expressions extracted from video frames.
- **Audio stream**: Mel-frequency cepstral coefficients (MFCC) features extracted from speech.

Two fusion strategies were implemented and compared:
1. **Intermediate Attention (IA)** — introduces cross-attention and temporal gating between modalities.
2. **Late Transformer (LT)** — merges embeddings via Transformer-based fusion after independent processing.

---

## ⚙️ Architecture

**1. Input Processing**
- Video → detected faces (MTCNN) → EfficientFace → Conv1D temporal modeling.  
- Audio → MFCC extraction → Conv1D temporal modeling.

**2. Fusion**
- IA: bidirectional cross-attention + temporal gate  
- LT: late-stage Transformer encoder

**3. Output**
- Fully Connected → Softmax layer → Emotion classification (8 classes)

---

## 📊 Dataset

- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
- 24 professional actors, 8 emotion categories:
  - Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprise
- 15-frame video clips synchronized with 1–2 second audio segments.

---

## 🚀 Training & Evaluation

| Configuration | Accuracy | Macro F1 |
|----------------|-----------|-----------|
| Audio-only | 45.2% | 43.8% |
| Video-only | 44.7% | 43.1% |
| IA Fusion | **79.8%** | **78.5%** |
| LT Fusion | 77.6% | 76.3% |

- Loss: CrossEntropy  
- Optimizer: Adam (lr=1e-4)  
- Batch size: 8, Epochs: 50  
- Hardware: NVIDIA GPU (≥12GB VRAM recommended)

---

## 📁 Project Structure

```
akaaka_mutil_model_fullcode/
│
├── main.py                # Main training script
├── validation.py          # Model evaluation
├── dataset.py             # RAVDESS data loading and preprocessing
├── model.py               # CNN + Transformer architecture
├── utils.py               # Helper functions
├── results/               # Logs, metrics, confusion matrices
├── ravdess_preprocessing/ # Audio/video extraction scripts
├── requirements.txt       # Dependencies
└── README.md              # This file
```

---

## 🧩 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 💡 Usage

### 1. Training
```bash
python main.py
```

### 2. Validation
```bash
python validation.py
```

### 3. Run with specific fusion mode
```bash
python main_aka.py --fusion IA     # Intermediate Attention
python main_aka.py --fusion LT     # Late Transformer
```

---

## 🧠 Citation

If you use or reference this work, please cite:
```
@project{leminh2025multimodal,
  title={Multimodal Sentiment Analysis for Emotion and Reaction},
  author={Le Minh, Le Chau Giang},
  year={2025},
  note={RAVDESS-based multimodal fusion for emotion recognition}
}
```

---

## 🪴 Acknowledgements

- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [EfficientFace model](https://github.com/zengqunzhao/EfficientFace)
- [PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models)

---

## 📫 Contact

For academic or research collaboration:
**Le Minh** – [GitHub Profile](https://github.com/leminh2810)
