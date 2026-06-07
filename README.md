Motorcycle Helmet Detection using YOLOv8[README.md](https://github.com/user-attachments/files/27536029/README.md)

# 🪖 Real-Time Helmet & Head Detection using YOLOv8

[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/AniHug/Helmet_Detection)
[![Model](https://img.shields.io/badge/Model-YOLOv8m-green)](https://github.com/ultralytics/ultralytics)
[![mAP50](https://img.shields.io/badge/mAP50-0.968-brightgreen)](https://github.com/Ani-Lit/Helmet-Detection-using-YOLO)
[![mAP50--95](https://img.shields.io/badge/mAP50--95-0.645-brightgreen)](https://github.com/Ani-Lit/Helmet-Detection-using-YOLO)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> A production-grade computer vision system that detects safety helmets and bare heads in real time , built for construction and industrial safety monitoring.

---

## 🚀 Live Demo

👉 **[Try it here → huggingface.co/spaces/AniHug/Helmet_Detection](https://huggingface.co/spaces/AniHug/Helmet_Detection)**

Upload any image and the model will detect:
- 🪖 **Helmet** — person wearing a safety helmet
- 👤 **Head** — person without a helmet

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **mAP50** | **0.968** |
| **mAP50-95** | **0.645** |
| Helmet (mAP50) | 0.983 |
| Head (mAP50) | 0.953 |
| Precision | 0.946 |
| Recall | 0.930 |
| Training Time | 5.27 hours |
| Hardware | Dual Kaggle T4 GPUs |

> Trained on 25,000+ annotated instances across train/valid/test splits.

---

## 🏗️ Project Architecture

```
Input Image / Webcam
        ↓
  YOLOv8m Model
  (Fine-tuned)
        ↓
  Bounding Boxes
  + Class Labels
  + Confidence Scores
        ↓
  Streamlit / HuggingFace
  Live Demo Output
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Model | YOLOv8m (Ultralytics) |
| Training | Kaggle T4 GPU x2 |
| Dataset | Roboflow (custom filtered) |
| Deployment | Hugging Face Spaces |
| Interface | Streamlit |
| Language | Python 3.12 |

---

## 📁 Repository Structure

```
Helmet-Detection-using-YOLO/
│
├── Helmet_Detection.ipynb   # Full training pipeline notebook
├── inference.py             # Run inference on images
├── requirements.txt         # Dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Ani-Lit/Helmet-Detection-using-YOLO.git
cd Helmet-Detection-using-YOLO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run inference on an image
```bash
python inference.py --source your_image.jpg --conf 0.6
```

---

## 🎯 Training Pipeline

The model was trained using the following configuration:

```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device='0,1',       # Dual T4 GPU
    patience=10,
    name='helmet_detector'
)
```

**Dataset:**
- Source: Roboflow Universe
- Classes: `helmet`, `head`
- Total images: 5,000+
- Split: 70% train / 20% valid / 10% test

---

## 📈 Training Results

| Epoch | mAP50 | mAP50-95 |
|-------|-------|----------|
| 10 | 0.821 | 0.412 |
| 20 | 0.903 | 0.521 |
| 34 (best) | **0.968** | **0.645** |
| 44 (final) | 0.968 | 0.645 |

> EarlyStopping triggered at epoch 44 — best weights saved at epoch 34.

---

## 🔍 Use Cases

- ✅ Construction site safety monitoring
- ✅ Industrial workplace compliance
- ✅ Real-time CCTV helmet detection
- ✅ Safety audit automation

---

## ⚠️ Known Limitations

- Model may produce false positives on similar headwear (caps, turbans, hoods)
- Optimized for safety/hard helmets specifically
- Performance may vary in low-light conditions

---

## 👨‍💻 Author

**Anirudha Kumar**
B.Tech CSE (Artificial Intelligence) — Shoolini University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/anirudha-kumar-08b216345/)
[![HuggingFace](https://img.shields.io/badge/🤗-AniHug-orange)](https://huggingface.co/AniHug)
[![GitHub](https://img.shields.io/badge/GitHub-Ani--Lit-black)](https://github.com/Ani-Lit)

---

## 📄 License

This project is licensed under the MIT License.

---

> ⭐ If you find this useful, please star the repository!
