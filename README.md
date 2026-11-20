# Wearable Human Activity Recognition — Motion Mindset

**Course:** Deep Learning (AI403) — Fall 2026, Pharos University in Alexandria (Faculty of Computer Science & AI)
**Instructor:** Dr. Sahar Ghanem  |  **Idea chosen:** #9 — Wearable human activity recognition  |  **Team:** Motion Mindset

| Member | ID | Main role |
|---|---|---|
| Engy Khaled (Team Lead) | 202302900 | Advanced hybrid model |
| Habiba Abdulhameed | 202301140 | Bi-LSTM model, dataset scaling |
| Rahaf Mohamed | 202301923 | Transformer, hyperparameters, error analysis |
| Hana Yasser | 202302665 | Random Forest baseline, preprocessing, EDA |

## 1. Task and best result
Classify 2.56 s smartphone inertial windows (128 steps × 9 channels, 50 Hz) into 6 activities.
Best model: **1D-CNN + Bi-LSTM + custom temporal attention** — test accuracy **91.99 ± 0.94 %**, macro-F1 **92.13 ± 0.94 %** (3 seeds).

| Model | Test Acc (%) | Test Macro-F1 (%) |
|---|---|---|
| Random Forest (baseline) | 83.49 ± 0.40 | 83.26 ± 0.43 |
| Bi-LSTM | 91.23 ± 0.98 | 91.36 ± 0.99 |
| Time-Series Transformer | 88.73 ± 0.85 | 88.69 ± 0.88 |
| CNN + Bi-LSTM + Attention | 91.99 ± 0.94 | 92.13 ± 0.94 |

Full table (parameters, size, times): `results/final_4models_comparison.csv`.

## 2. Environment setup
Python 3.9+ (the saved models were built with TensorFlow/Keras, see `requirements.txt`).
```bash
pip install -r requirements.txt
```

## 3. Dataset (UCI HAR) — download and preprocessing
The raw dataset is **not** stored in the repository.
```bash
python data/download.py                 # downloads and extracts to data/UCI HAR Dataset/
python src/datasets/loader.py           # builds the splits, prints shapes (sanity check)
```
* Source: UCI Machine Learning Repository, *Human Activity Recognition Using Smartphones* (Anguita et al., ESANN 2013). Check the licence/terms on the dataset page.
* Split (subject-wise, no leakage): official train subjects minus validation subjects 1, 3, 5, 6 → train (17 subjects) / validation (4 subjects); the official 9 test subjects are used only for final evaluation.
* Z-score statistics are computed on the training partition only.

## 4. Training and evaluation
Training and evaluation are run from the notebooks (model builders live in `src/models/`):

| Step | Where |
|---|---|
| EDA, data pipeline, Random Forest baseline, Bi-LSTM (3 seeds) | `notebooks/eda_and_models.ipynb` |
| Transformer, advanced hybrid, final 4-model comparison (3 seeds) | `notebooks/02_modern_and_advanced_models.ipynb` |
| Hyperparameter study, data-strategy study, ablation, error analysis | `notebooks/<ADD NOTEBOOK NAME>.ipynb`  ← **TODO: add the notebook that produced `results/hyperparam_and_data_study.csv`, `ablation_study.csv`, `error_analysis_summary.csv`** |

Run order: `eda_and_models.ipynb` → `02_modern_and_advanced_models.ipynb` → the study notebook. Seeds: 42, 123, 2026.
Outputs: `results/*.csv`, `results/*.png`, `saved_models/*.keras`.

## 5. Live demo
```bash
streamlit run demo/app.py
```
Two input modes: (a) a window from the held-out test subjects, (b) **your own CSV** — 9 columns in UCI order, or 6 columns (total-acc xyz + gyro xyz, body-acc derived by gravity removal), ≥128 rows at 50 Hz. Sample CSV files can be downloaded from the app sidebar.

## 6. Hardware and expected runtimes
* Hardware: **TODO — CPU model / GPU / RAM / OS used for training and timing**
* Per-model training time and inference latency: see `results/final_4models_comparison.csv` (e.g. hybrid ≈ 60 s training, ≈ 0.27 ms/window inference).
* All neural checkpoints are < 1 MiB (`saved_models/`).

## 7. Large files
* Checkpoints / extra files: **TODO — shared drive link**

## 8. Repository layout
```
README.md  requirements.txt
data/        download script only (no raw data)
src/datasets/ loader and split        src/models/ model builders + custom attention layer
notebooks/   experiments              results/ CSV tables and figures
saved_models/ trained checkpoints     demo/ Streamlit app (+ input_utils.py)
report/      report and slides        contributions/ individual logs
```

## 9. Credits
* UCI HAR dataset — Anguita et al., ESANN 2013.
* Architectures: Hochreiter & Schmidhuber (1997, LSTM), Vaswani et al. (2017, Transformer), Bahdanau et al. (2014, additive attention).
* Libraries: TensorFlow/Keras, scikit-learn, Streamlit, Plotly. The temporal-attention layer (`src/models/custom_layers.py`) was written by the team.
