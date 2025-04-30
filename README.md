# 💅 Nail Deficiency Detection Web App  
> Detect nail health issues and get dietary recommendations using deep learning!

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]()

---

## 📌 Overview

This web app uses a trained CNN model to **analyze nail images** and detect signs of health conditions like **Beau’s Line**, **Black Line**, and **Clubbing**. It also provides personalized **nutritional recommendations** and **causes** based on the diagnosis.

🧠 Built using:
- TensorFlow / Keras
- Flask
- OpenCV
- HTML/CSS/JS

---

## 🎯 Features

✅ Upload or capture nail images  
✅ Detect symptoms using deep learning  
✅ Get medical causes and food suggestions  
✅ Secure Login/Register system  
✅ Lightweight and easy to run locally

---

## 📂 Dataset

The model was trained on a custom nail image dataset organized as:
🔗 **[Download the Dataset] (https://www.kaggle.com/datasets/nikhilgurav21/nail-disease-detection-dataset)

---
## 🧠 Model Details

- **Input Size**: 128x128 pixels  
- **Architecture**: CNN (or MobileNetV2 pretrained)  
- **Framework**: TensorFlow/Keras  
- **File**: `nail_disease_cnn_model.h5`

---
## 🗃️ Prediction Output

| Condition      | Deficiencies             | Reason                                             | Food Suggestions                             |
|----------------|--------------------------|----------------------------------------------------|-----------------------------------------------|
| **Beau’s Line**| Zinc, Protein            | Severe illness, malnutrition, or trauma            | Lean meats, eggs, dairy, nuts, whole grains   |
| **Black Line** | Iron, Vitamin B12        | Trauma, endocarditis, vitamin deficiency           | Spinach, red meat, lentils, fish              |
| **Clubbing**   | Vitamin C                | Chronic lung/heart disease                         | Citrus, bell peppers, strawberries, greens    |
| **Normal**     | None                     | Healthy nail bed                                   | Balanced diet                                 |

---
## ⚙️ How to Run Locally

1️⃣ Clone the repo  
2️⃣ Install dependencies  
3️⃣ Run Flask app
```bash
git clone https://github.com/yourusername/nail-deficiency-detection.git
cd nail-deficiency-detection

python -m venv venv
venv\Scripts\activate        # On Windows
# source venv/bin/activate   # On Linux/macOS

pip install -r requirements.txt
python app.py



