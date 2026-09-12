# Plant Disease Detector 🌿

An end-to-end deep learning web application designed to identify crop leaf diseases from user-uploaded images. Built with **PyTorch**, **Flask**, and **JavaScript**, this application leverages a custom Convolutional Neural Network (CNN) trained on the PlantVillage dataset to deliver fast, automated disease diagnosis and confidence scoring.

---

## 📑 Table of Contents
- [Features](#-features)
- [Problem Statement](#-problem-statement)
- [Tech Stack](#-tech-stack)
- [Dataset Overview](#-dataset-overview)
- [Model Architecture & Pipeline](#-model-architecture--pipeline)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Workflow](#-usage-workflow)
- [Future Enhancements](#-future-enhancements)

---

## ✨ Features
- **Instant Leaf Disease Classification:** Upload a plant leaf image to detect healthy or diseased states across 19 categories.
- **Confidence Scoring:** Calculates real-time model confidence percentages using Softmax output probabilities.
- **In-Memory Preprocessing:** Handles image channel conversion and scaling directly via PIL memory streams for fast inference.
- **Responsive UI:** Clean, intuitive frontend built with HTML5, CSS3, and JavaScript.

---

## 🎯 Problem Statement
Plant diseases significantly reduce crop yields if left undetected. Traditional disease identification relies on manual visual inspection by farmers or agricultural experts, which is time-consuming, expensive, and prone to human error. This project automates disease identification through computer vision, enabling rapid assessment to prevent widespread crop damage.

---

## 🛠️ Tech Stack
- **Language:** Python 3.x, JavaScript (ES6+), HTML5, CSS3
- **Deep Learning Framework:** PyTorch (`torch`, `torchvision`)
- **Backend Framework:** Flask
- **Image Processing:** Pillow (PIL)
- **Data Source:** Kaggle (PlantVillage Dataset)

---

## 📊 Dataset Overview
The model is trained on a subset of the **PlantVillage dataset**, containing over 50,000 labeled images of healthy and diseased plant leaves across multiple crop types (e.g., Tomato, Potato, Corn, Apple, Grape).

- **Total Selected Categories:** 19 disease and health classes.
- **Sample Target Classes:**
  - Tomato Early Blight / Tomato Late Blight
  - Potato Early Blight / Potato Late Blight
  - Corn Common Rust
  - Apple Black Rot

---

## 🧠 Model Architecture & Pipeline

### 1. Convolutional Neural Network (CNN) Layers
- **Conv1:** $3 \to 32$ channels ($3 \times 3$ kernel) — Detects low-level features (edges, basic color blotches, textures).
- **Conv2:** $32 \to 64$ channels ($3 \times 3$ kernel) — Extracts intermediate shapes and texture combinations.
- **Conv3:** $64 \to 128$ channels ($3 \times 3$ kernel) — Extracts complex, disease-specific features (lesions, spot patterns).
- **Activations & Downsampling:** Every convolution is followed by a **ReLU** activation function ($\text{ReLU}(x) = \max(0, x)$) and a **Max Pooling** layer for spatial reduction.
- **Fully Connected Layers:** Flattened feature vectors pass through dense layers mapped to **19 output neurons**.

### 2. Image Preprocessing Pipeline
1. **Format Standardization:** Converts input image to `RGB` format via PIL.
2. **Resizing:** Resizes image to $224 \times 224$ pixels.
3. **Tensor Conversion:** `torchvision.transforms.ToTensor()` converts pixel values to PyTorch float tensors and scales ranges from $[0, 255]$ to $[0, 1]$.
4. **Batch Dimension:** `unsqueeze(0)` adds a batch dimension, producing a 4D tensor with shape `(1, 3, 224, 224)`.

### 3. Training & Optimization
- **Loss Function:** `CrossEntropyLoss` (Multi-class classification penalty)
- **Optimizer:** `Adam` (Adaptive Moment Estimation)
- **Epochs:** 10 epochs (monitored via validation accuracy to prevent overfitting)

---

## 📁 Project Structure

```text
plant-disease-detector/
│
├── app.py                 # Flask backend API routing and entry point
├── model.py               # PyTorch CNN model architecture definition
├── best_model.pth         # Saved weights from trained PyTorch model
├── requirements.txt       # Python dependency requirements
│
├── static/
│   ├── css/
│   │   └── style.css      # User interface styles
│   └── js/
│       └── main.js        # Dynamic AJAX requests & UI updates
│
└── templates/
    └── index.html         # Frontend HTML upload interface