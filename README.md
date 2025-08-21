<h1 align="center">
  <a href="https://github.com/iPelo/Logistic_Regression">
    BREAST-CANCER-CLASSIFIER (Logistic Regression from Scratch)
  </a>
</h1>

<p align="center">Early Detection with Logistic Regression – Built from Scratch</p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/iPelo/Logistic_Regression?style=for-the-badge" alt="Last Commit">
  <img src="https://img.shields.io/github/languages/top/iPelo/Logistic_Regression?style=for-the-badge" alt="Top Language">
  <img src="https://img.shields.io/github/languages/count/iPelo/Logistic_Regression?style=for-the-badge" alt="Language Count">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Markdown-000000?logo=markdown&logoColor=white&style=for-the-badge" alt="Markdown">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white&style=for-the-badge" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557c?logo=plotly&logoColor=white&style=for-the-badge" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style=for-the-badge" alt="Pandas">
</p>

---

## 📑 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)

---

## 📜 Overview

**Breast-Cancer-Classifier** is an educational project implementing **logistic regression from scratch** to classify tumors as **benign (0)** or **malignant (1)**.  

The core model is defined in `logisctic_regression.py` and trained with **gradient descent** on the **WDBC (Wisconsin Diagnostic Breast Cancer)** dataset.  

Outputs include:  
- 📊 **Confusion Matrix Plot**  
- 📈 **ROC Curve with AUC**  
- ✅ Performance metrics (Accuracy, Precision, Recall, F1-score)  

This demonstrates how to build a classifier without relying on high-level ML libraries like scikit-learn.  

---

## 🧬 Dataset

We use the **WDBC dataset** from the UCI Machine Learning Repository:  

- **Features:** 30 numeric attributes (cell nucleus characteristics).  
- **Target:** `diagnosis` (M = malignant, B = benign).  

> 📌 The dataset is included in the repo under the `data/` directory (`wdbc.csv` / `wdbc.json`).

---

## 🚀 Getting Started

### 📦 Prerequisites
- **Python 3.10+**
- **pip** (optional: **Conda**)

---

### ⚙ Installation
```bash
# Clone the repository
git clone https://github.com/iPelo/Breast-Cancer-Classifier

# Enter the project directory
cd Breast-Cancer-Classifier

# Install dependencies
pip install -r requirements.txt