# ♻️ Smart Waste Classification System – Streamlit Deployment

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Deep Learning](https://img.shields.io/badge/DeepLearning-CNN-orange)
![Model](https://img.shields.io/badge/Model-MobileNetV2-green)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-red)
![Platform](https://img.shields.io/badge/Platform-Streamlit-success)

A Deep Learning-based web application that automatically classifies waste images into **Organic** and **Recyclable** categories using **Transfer Learning with MobileNetV2**.

---

# 🚀 Live Streamlit App

👉 Click below to use the deployed application:

🔗 Add your Streamlit Cloud URL here

---

# 🚀 Run Notebook in Google Colab

Click below to open the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](ADD_YOUR_COLAB_LINK_HERE)

---

# 📘 Project Overview

Waste segregation is an important step in environmental sustainability and efficient recycling.

This project uses **Transfer Learning** with **MobileNetV2** to classify waste images into:

- Organic Waste
- Recyclable Waste

The system includes:

- Data Preprocessing
- Image Augmentation
- Transfer Learning
- Model Training
- Model Evaluation
- Streamlit Deployment
- Power BI Dashboard

---

# 🎯 Objective

The main objectives of this project are:

🔹 Automate waste classification using Deep Learning

🔹 Improve waste segregation efficiency

🔹 Reduce manual sorting efforts

🔹 Promote recycling awareness

🔹 Deploy a real-time prediction system using Streamlit

🔹 Visualize prediction insights using Power BI

---

# 📂 Dataset Description

| Component | Description |
|------------|------------|
| Problem Type | Image Classification |
| Classes | Organic, Recyclable |
| Input Size | 224 × 224 × 3 |
| Model Input | RGB Images |
| Framework | TensorFlow / Keras |

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

✔ Image loading

✔ Image resizing (224 × 224)

✔ Pixel normalization

✔ Data augmentation

✔ Training-validation split

✔ Batch generation using ImageDataGenerator

✔ Class balancing using class weights

---

# 🏗 Model Architecture

## Framework Used

- TensorFlow
- Keras

## Architecture Used

- Convolutional Neural Network (CNN)
- MobileNetV2 (Transfer Learning)

## Transfer Learning

The pre-trained MobileNetV2 model trained on ImageNet was used.

```python
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
```

## Classification Head

- GlobalAveragePooling2D
- Dense (128 neurons, ReLU)
- Dropout (0.5)
- Dense (2 neurons, Softmax)

---

# 🔄 Transfer Learning Strategy

✔ Pre-trained MobileNetV2 loaded

✔ ImageNet weights utilized

✔ Original classifier removed

✔ Base model frozen

✔ Custom classification layers added

✔ Fine-tuned for waste classification

---

# 🤖 Model Training

The model was trained using:

- Adam Optimizer
- Categorical Crossentropy Loss
- Accuracy Metric

Training included:

- Early Stopping
- Learning Rate Reduction
- Class Weights

---

# 📊 Model Evaluation

Evaluation metrics used:

✔ Accuracy

✔ Loss

✔ Confusion Matrix

✔ Classification Report

✔ Learning Curves

✔ Error Analysis

---

# 🏆 Final Model

### MobileNetV2 Transfer Learning Model

Architecture:

```text
MobileNetV2
       ↓
GlobalAveragePooling2D
       ↓
Dense (128, ReLU)
       ↓
Dropout (0.5)
       ↓
Dense (2, Softmax)
```

---

# 🌐 Streamlit Application Features

## Home Page

✔ Upload waste image

✔ Predict waste type

✔ Confidence score

## Prediction Page

✔ Organic prediction

✔ Recyclable prediction

✔ Disposal recommendations

## History Page

✔ Prediction history

✔ Excel export

## Dashboard Page

✔ Waste distribution

✔ Prediction analytics

✔ Interactive charts

---

# 📊 Power BI Dashboard

The project includes a professional Power BI dashboard with:

✔ Total Predictions KPI

✔ Organic Count KPI

✔ Recyclable Count KPI

✔ Average Confidence KPI

✔ Waste Distribution Donut Chart

✔ Prediction Trend Analysis

✔ Prediction History Table

---

# ⚠️ Limitations

- Limited to two waste categories
- Model performance depends on image quality
- Dataset diversity affects generalization
- Real-world waste may contain mixed categories

---

# 🛠 Tech Stack

| Tool | Purpose |
|--------|---------|
| Python | Programming Language |
| TensorFlow | Deep Learning |
| Keras | Model Development |
| NumPy | Numerical Computation |
| Pandas | Data Processing |
| Matplotlib | Visualization |
| Streamlit | Web Application |
| Power BI | Dashboard Analytics |
| Google Colab | Model Training |

---

# 📁 Repository Structure

```text
Smart-Waste-Classification/

│
├── app.py
├── waste_classification_model.keras
├── prediction_history.csv
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── Waste_Classification.ipynb
│
├── powerbi/
│   ├── Waste_Classification_Dashboard.pbix
│   └── dashboard_screenshot.png
│
├── images/
│   ├── home_background.jpg
│   ├── prediction_background.jpg
│   ├── history_background.jpg
│   └── dashboard_background.jpg
```

---

# 🚀 How to Run the Project

## 1️⃣ Clone Repository

```bash
git clone YOUR_REPOSITORY_LINK
cd Smart-Waste-Classification
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

# 📌 Academic Submission

This project was developed as part of a Deep Learning and Data Science program to demonstrate image classification using Convolutional Neural Networks, Transfer Learning, Streamlit deployment, and Power BI dashboard development.

---

# 📈 Future Work

✔ Add additional waste categories

✔ Fine-tune MobileNetV2 layers

✔ Integrate real-time camera prediction

✔ Deploy using Docker

✔ Cloud deployment using AWS or Azure

✔ Improve dashboard analytics

---

# 👤 Author

**Name:** Laya Mary Joy

**Organization:** Entri Elevate

**Year:** May 20,2026

---

# ⭐ Acknowledgment

Special thanks to **Entri Elevate** for providing guidance and support throughout the development of this Deep Learning project.
