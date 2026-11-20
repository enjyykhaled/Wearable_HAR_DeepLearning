# 🏃‍♂️ Wearable Human Activity Recognition (HAR) — Motion Mindset

A leakage-free deep learning framework for **Human Activity Recognition (HAR)** using raw smartphone inertial sensor time-series data. Developed as the final project for **Deep Learning (A1403)** at **Pharos University in Alexandria**, this repository benchmarks classical machine learning against recurrent, transformer, and custom hybrid deep learning architectures, paired with a strict validation-only ablation protocol and an interactive Streamlit demo.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Repository Structure](#-repository-structure)
3. [Installation & Quick Start](#-installation--quick-start)
4. [Dataset & Leakage-Free Pipeline](#-dataset--leakage-free-pipeline)
5. [Evaluated Model Architectures](#-evaluated-model-architectures)
6. [Benchmark Results (Mean ± SD)](#-benchmark-results-mean--sd)
7. [Validation-Only Ablation & Tuning Results](#-validation-only-ablation--tuning-results)
8. [Team & Course Information](#-team--course-information)

---

## 🎯 Project Overview

Recognizing human physical activities from wearable sensors presents two core challenges:
* **Temporal Data Leakage:** Standard random window splits leak overlapping sliding windows ($50\%$ overlap) from the same subject across train and test sets, artificially inflating accuracy.
* **Multi-Scale Temporal Dynamics:** Inertial signals contain both rapid local spikes (e.g., heel strikes) and long-range sequential dependencies, while static postures (sitting vs. standing) differ primarily in subtle gravity-vector orientation.

### Key Contributions
* **Strict Subject-Wise Partitioning:** Zero subject overlap between Train ($17$ subjects), Validation ($4$ subjects: `1, 3, 5, 6`), and Official Test ($9$ subjects) splits, with training-only Z-score standardization.
* **Custom Hybrid Architecture:** Combines a `1D-CNN` local feature extractor, a `Bi-LSTM` sequence encoder, and a hand-written Keras `TemporalAttentionLayer` (`src/models/custom_layers.py`) to dynamically weight informative time steps.
* **Reproducible Multi-Seed Evaluation:** All models are trained from scratch and evaluated across **3 random seeds (`42, 123, 2026`)** using population standard deviation ($\text{ddof}=0$).
* **Interactive Streamlit Demo:** Supports browsing held-out test windows as well as uploading external 6-channel or 9-channel CSV sensor recordings.

---

## 📂 Repository Structure

```text
Wearable_HAR_DeepLearning/
├── data/
│   └── UCI HAR Dataset/                          # Raw inertial signals & subject labels (not tracked in Git)
├── notebooks/
│   ├── eda_and_models.ipynb                      # Part 1: EDA, leakage-free splits, RF & Bi-LSTM baselines
│   ├── 02_modern_and_advanced_models.ipynb       # Part 2: Transformer, Hybrid model & 4-model benchmark
│   └── 03_hyperparameter_and_ablation_studies.ipynb # Part 3: Validation-only tuning, data & ablation studies
├── src/
│   └── models/
│       └── custom_layers.py                      # Hand-written Keras TemporalAttentionLayer
├── results/
│   ├── final_4models_comparison.csv              # Unified 4-model validation & test metrics (Mean ± SD)
│   ├── study_hyperparameters.csv                 # Bi-LSTM grid search logs (units × learning rate)
│   ├── study_data_strategy.csv                   # Z-score vs. raw vs. Gaussian jitter augmentation logs
│   └── study_ablation.csv                        # Sensor channel & hybrid component ablation logs
├── checkpoints/
│   ├── bilstm_har.keras                          # Saved Bi-LSTM model checkpoint
│   ├── transformer_har.keras                     # Saved Time-Series Transformer checkpoint
│   └── cnn_bilstm_attn_har.keras                 # Saved Hybrid model checkpoint
├── demo/
│   ├── app.py                                    # Interactive Streamlit inference & timeline visualization app
│   └── input_utils.py                            # Preprocessing & 0.3 Hz gravity-removal helpers for CSV uploads
├── requirements.txt                              # Python dependencies
└── README.md                                     # Project documentation
```

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/enjyykhaled/Wearable_HAR_DeepLearning.git
cd Wearable_HAR_DeepLearning
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the UCI HAR Dataset
Download the [UCI HAR Dataset (v1.0)](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) and extract it into `data/UCI HAR Dataset/` so that `train/` and `test/` folders (containing `Inertial Signals/` and `subject_*.txt`) are accessible by the data loader.

### 4. Run the Notebooks (In Order)
1. **`notebooks/eda_and_models.ipynb`**: Runs data integrity checks, builds subject-wise splits, and trains Random Forest & Bi-LSTM.
2. **`notebooks/02_modern_and_advanced_models.ipynb`**: Trains the Time-Series Transformer and Hybrid model, generating `results/final_4models_comparison.csv`.
3. **`notebooks/03_hyperparameter_and_ablation_studies.ipynb`**: Executes all validation-only tuning and ablation studies.

### 5. Launch the Streamlit Web App
```bash
streamlit run demo/app.py
```

---

## 📊 Dataset & Leakage-Free Pipeline

* **Source:** UCI Human Activity Recognition Using Smartphones ($30$ volunteers aged $19\text{--}48$, waist-mounted Samsung Galaxy S II at $50\text{ Hz}$).
* **Input Tensor Shape:** `(N, 128, 9)` — **9 inertial sensor channels** sampled into $2.56\text{-second}$ windows ($128$ steps, $50\%$ overlap):
  * Channels `0–2`: Body Acceleration ($X, Y, Z$)
  * Channels `3–5`: Angular Velocity / Gyroscope ($X, Y, Z$)
  * Channels `6–8`: Total Acceleration ($X, Y, Z$)
* **Subject-Wise Split ($10,299$ Total Windows):**

| Partition | Subjects | Walking | Upstairs | Downstairs | Sitting | Standing | Laying | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | $17$ subjects | $960$ | $863$ | $793$ | $1,088$ | $1,147$ | $1,186$ | **$6,037$** |
| **Validation** | $4$ subjects (`1, 3, 5, 6`) | $266$ | $210$ | $193$ | $198$ | $227$ | $221$ | **$1,315$** |
| **Test (Official)** | $9$ subjects | $496$ | $471$ | $420$ | $491$ | $532$ | $537$ | **$2,947$** |

* **Preprocessing:** Channel-wise mean and standard deviation are computed strictly on the $6,037$ training windows before standardizing validation and test sets.

---

## 🧠 Evaluated Model Architectures

All neural models are trained from scratch using **Adam** (`lr=1e-3`, `clipnorm=1.0`), `sparse_categorical_crossentropy`, `batch_size=64`, up to $40$ epochs with `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)`:

1. **Baseline — Random Forest (`100` trees):** Flattens each $128 \times 9$ window into $1,152$ tabular features (`n_jobs=-1`).
2. **Classic DL — Bidirectional LSTM:** Single `Bidirectional(LSTM(64, dropout=0.2))` layer + `Dropout(0.3)` + 6-class `Softmax` ($38,662$ params).
3. **Modern DL — Time-Series Transformer:** Dense projection ($d_{\text{model}}=64$) + learned positional embeddings ($128$ steps) + 1 Encoder block ($4$ heads, $d_{\text{ff}}=128$, `dropout=0.2`, residual `LayerNorm`) + `GlobalAveragePooling1D` + `Dropout(0.3)` ($34,502$ params).
4. **Advanced Hybrid — 1D-CNN + Bi-LSTM + Custom Attention:**
   * `Conv1D(64, kernel_size=5, padding="same", activation="relu")` $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling1D(2)` $\rightarrow$ `Dropout(0.2)`
   * `Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))`
   * Custom `TemporalAttentionLayer(attention_dim=64)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(6, activation="softmax")` ($78,342$ params).

---

## 📈 Benchmark Results (Mean ± SD)

Evaluated across **3 random seeds (`42, 123, 2026`)** on CPU (`results/final_4models_comparison.csv`):

| Model | Params | Size (KiB) | Val Acc (%) | Val Macro-F1 (%) | Test Acc (%) | Test Macro-F1 (%) | Train Time (s) | Inference (ms/win) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Random Forest (100 trees)** | N/A | N/A | $92.95 \pm 0.41$ | $92.87 \pm 0.40$ | $83.49 \pm 0.40$ | $83.26 \pm 0.43$ | $1.56$ | $0.013$ |
| **2. Classic Bi-LSTM** | $38,662$ | $491.4$ | $94.20 \pm 0.75$ | $94.06 \pm 0.77$ | $91.23 \pm 0.98$ | $91.36 \pm 0.99$ | $76.43$ | $0.362$ |
| **3. Time-Series Transformer** | $34,502$ | $661.9$ | $93.33 \pm 0.74$ | $93.16 \pm 0.74$ | $88.73 \pm 0.85$ | $88.69 \pm 0.88$ | $112.54$ | $0.256$ |
| **4. Hybrid (CNN + Bi-LSTM + Attn)** | $78,342$ | $977.3$ | **$95.29 \pm 0.43$** | **$95.14 \pm 0.45$** | **$91.99 \pm 0.94$** | **$92.13 \pm 0.94$** | $60.07$ | $0.267$ |

### 🔍 Key Takeaways
* **Deep Models vs. Baseline (H1 Supported):** All three neural architectures outperform the Random Forest baseline by **$5.24$ to $8.50$ percentage points** on unseen test subjects.
* **Hybrid Ranking (H2):** The Hybrid model achieves the highest overall accuracy ($91.99\%$) and Macro-F1 ($92.13\%$), clearly outperforming the Transformer ($+3.26\%$) across all seeds and showing a consistent positive trend over the standalone Bi-LSTM ($+0.76\%$).
* **Primary Error Mode:** Confusion matrices reveal that `LAYING` achieves $100\%$ F1, while ~$84\%$ of classification errors occur between `SITTING` and `STANDING` due to subtle differences in static phone tilt.

---

## 🔬 Validation-Only Ablation & Tuning Results

All ablation and hyperparameter experiments (`notebooks/03_hyperparameter_and_ablation_studies.ipynb`) were executed strictly on the **4-subject validation split** (`results/study_ablation.csv`):

| Model | Setting / Removed Component | Params | Val Acc (%) | Val Macro-F1 (%) | $\Delta$ Acc vs. Full |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Bi-LSTM** | Full model ($9$ channels) | $38,662$ | $94.20 \pm 0.75$ | $94.06 \pm 0.77$ | $0.00$ |
| **Bi-LSTM** | Without Gyroscope (`[3,4,5]`) | $37,126$ | $94.32 \pm 0.45$ | $94.18 \pm 0.42$ | $+0.12$ |
| **Bi-LSTM** | Without Total Acceleration (`[6,7,8]`) | $37,126$ | $83.57 \pm 1.46$ | $83.22 \pm 1.56$ | **$-10.63$** |
| **Bi-LSTM** | Without Body Acceleration (`[0,1,2]`) | $37,126$ | $92.57 \pm 0.56$ | $92.48 \pm 0.56$ | $-1.63$ |
| **Hybrid** | Full (`CNN + Bi-LSTM + Attention`) | $78,342$ | **$95.29 \pm 0.43$** | **$95.14 \pm 0.45$** | $0.00$ |
| **Hybrid** | Without Attention (`GlobalAvgPool1D`) | $70,022$ | $94.78 \pm 0.28$ | $94.63 \pm 0.29$ | $-0.51$ |
| **Hybrid** | Without `1D-CNN` front-end | $46,982$ | $95.01 \pm 0.19$ | $94.84 \pm 0.18$ | $-0.28$ |

* **Sensor Importance (H3):** Removing **Total Acceleration** causes a severe **$-10.63\%$** drop in validation accuracy because it carries the gravitational orientation vector essential for distinguishing static postures, whereas removing the gyroscope caused no drop.

---

## 👥 Team & Course Information

* **University:** Pharos University in Alexandria — Faculty of Computer Science & Artificial Intelligence (AI Department)
* **Course:** Deep Learning (`A1403`) — Fall 2026 | **Lab:** `D03`
* **Instructor:** Dr. Sahar Ghanem
* **Project Idea:** `9` — Wearable Human Activity Recognition
* **Repository:** [github.com/enjyykhaled/Wearable_HAR_DeepLearning](https://github.com/enjyykhaled/Wearable_HAR_DeepLearning)

### Team **Motion Mindset**
| Name | Student ID | Primary Role & Contributions |
| :--- | :---: | :--- |
| **Engy Khaled** *(Team Lead)* | `202302900` | Advanced Hybrid Architecture (`1D-CNN + Bi-LSTM + Custom Temporal Attention`) & Integration |
| **Habiba Abdulhameed** | `202301140` | Classic `Bi-LSTM` Architecture & Leakage-Free Dataset Scaling Pipeline |
| **Rahaf Mohamed** | `202301923` | Modern `Time-Series Transformer`, Hyperparameter Experiments & Error Analysis |
| **Hana Yasser** | `202302665` | `Random Forest` Baseline, Data Preprocessing & Exploratory Data Analysis (EDA) |
