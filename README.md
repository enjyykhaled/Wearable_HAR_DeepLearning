# Wearable Human Activity Recognition (Motion Mindset)

Deep Learning Final Project (A1403) — Fall 2026  
**Institution:** Pharos University in Alexandria (PUA) — Faculty of Computer Science & Artificial Intelligence  
**Instructor:** Dr. Sahar Ghanem  
**Team Name:** Motion Mindset  

---

## Team Members & IDs
* **Engy Khaled** (202302900) — Team Lead & Advanced Hybrid Model
* **Habiba Abdulhameed** (202301140) — Classic Bi-LSTM Model & Dataset Scaling
* **Rahaf Mohamed** (202301923) — Time-Series Transformer & Hyperparameter / Error Analysis
* **Hana Yasser** (202302665) — Random Forest Baseline & Data Preprocessing / EDA

---

## 1. Project Overview
This project implements a complete, leakage-free deep learning pipeline for six-class Human Activity Recognition (HAR) using smartphone inertial sensor windows (UCI HAR Dataset). The pipeline compares a classical Random Forest baseline with three deep learning architectures: a Bidirectional LSTM, a Time-Series Transformer, and an Advanced Hybrid 1D-CNN + Bi-LSTM + Custom Temporal Attention model.

---

## 2. Environment Setup
Make sure you have Python 3.9+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. Dataset & Preprocessing
* **Dataset:** UCI HAR Dataset (10,299 raw inertial windows, 128 time steps, 9 sensor channels).
* **Split Strategy:** Subject-wise split ensuring **no data leakage** (Subjects 1, 3, 5, and 6 used for validation; official test subjects isolated).
* **Standardization:** Channel-wise mean and standard deviation computed strictly on the training partition.

---

## 4. Training Commands
To train the models and generate results:
* **Baseline:** `python src/models/baseline.py`
* **Bi-LSTM:** `python src/models/bilstm.py`
* **Transformer:** `python src/models/transformer.py`
* **Advanced Hybrid:** `python src/models/advanced_hybrid.py`

---

## 5. Evaluation & Results
All summary metrics, ablation studies, and model comparisons are stored in the `results/` directory. The primary source of truth for evaluation is:
* `results/final_4models_comparison.csv`

---

## 6. Live Demonstration (Streamlit App)
To launch the interactive live demonstration interface:
```bash
streamlit run demo/app.py
```

---

## 7. Hardware & Performance Summary
* **Training Hardware:** Local development environment / Visual Studio Code (VS Code).
* **Inference Speed:** ~0.267 ms/window for the advanced hybrid model.
* **Model Size:** All neural model checkpoints are under 1 MiB (`saved_models/`).

---

## 8. Credits & References
* UCI HAR Dataset (ESANN 2013 dataset paper).
* Foundational architectures: Hochreiter & Schmidhuber (1997), Vaswani et al. (2017), Bahdanau et al. (2014).