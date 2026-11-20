# Wearable Human Activity Recognition (HAR): Deep Learning Technical Report

## 1. Data Pipeline & Integrity Verification

* **Dataset:** UCI Human Activity Recognition (HAR) Using Smartphones, comprising **9 inertial sensor channels** (Body Acceleration X/Y/Z, Angular Velocity/Gyroscope X/Y/Z, and Total Acceleration X/Y/Z) sampled at **50 Hz** into **128-step sliding windows**.
* **Leakage-Free Subject-Wise Split:** To prevent temporal data leakage caused by overlapping windows from the same individual, the dataset ($10,299$ total windows across 6 activities) was strictly partitioned by subject:
  * **Train Set:** $6,037$ windows ($17$ subjects)
  * **Validation Set:** $1,315$ windows ($4$ held-out subjects: $1, 3, 5, 6$)
  * **Test Set:** $2,947$ windows ($9$ held-out subjects)
* **Preprocessing:** Zero missing/NaN values and zero subject overlap were verified. Z-score standardization ($\mu = 0, \sigma = 1$) was computed strictly on the training set before transforming validation and test tensors.
* **Class Distribution:** Balanced across all six activities:
  * `LAYING` ($18.88\%$)
  * `STANDING` ($18.51\%$)
  * `SITTING` ($17.25\%$)
  * `WALKING` ($16.72\%$)
  * `WALKING_UPSTAIRS` ($14.99\%$)
  * `WALKING_DOWNSTAIRS` ($13.65\%$)

---

## 2. Evaluated Model Architectures

All deep learning models were trained using an identical protocol: **Adam optimizer** ($\text{lr} = 10^{-3}$, $\text{clipnorm} = 1.0$), `sparse_categorical_crossentropy`, $\text{batch\_size} = 64$, up to **$40$ epochs** with `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)`, evaluated across **3 random seeds (`42, 123, 2026`)**.

1. **Baseline: Random Forest (100 trees)** — Non-deep learning tabular baseline.
2. **Classic DL: Bidirectional LSTM** — Single Bi-LSTM layer ($64$ units, $\text{dropout} = 0.2$) followed by `Dropout(0.3)` and a 6-class Softmax head.
3. **Modern DL: Time-Series Transformer** — Dense channel projection ($d_{\text{model}} = 64$) + learnable positional embeddings + 1 Transformer Encoder block ($4$ heads, $d_{\text{ff}} = 128$, $\text{dropout} = 0.2$, residual LayerNorm connections) + `GlobalAveragePooling1D`.
4. **Advanced Hybrid: 1D-CNN + Bi-LSTM + Custom Attention**:
   * **Local Feature Extraction:** `Conv1D(64, kernel_size=5, padding="same", activation="relu")` $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling1D(2)` $\rightarrow$ `Dropout(0.2)`.
   * **Sequence Modeling:** `Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))`.
   * **Dynamic Time-Step Weighting:** Hand-written `TemporalAttentionLayer(attention_dim=64)` imported from `src/models/custom_layers.py` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(6, softmax)`.

---

## 3. Unified 4-Model Benchmark Comparison (Mean ± SD)

| Model | Params | Size (KB) | Val Accuracy (%) | Val Macro-F1 (%) | Test Accuracy (%) | Test Macro-F1 (%) | Train Time (s) | Inference (ms/win) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline: Random Forest (100 trees)** | N/A | N/A | $92.95 \pm 0.41$ | $92.87 \pm 0.40$ | $83.49 \pm 0.40$ | $83.26 \pm 0.43$ | $1.56$ | $0.013$ |
| **2. Classic DL: Bidirectional LSTM** | $38,662$ | $491.4$ | $94.20 \pm 0.75$ | $94.06 \pm 0.77$ | $91.23 \pm 0.98$ | $91.36 \pm 0.99$ | $76.43$ | $0.362$ |
| **3. Modern DL: Time-Series Transformer** | $34,502$ | $661.9$ | $93.33 \pm 0.74$ | $93.26 \pm 0.75$ | $88.73 \pm 0.85$ | $88.69 \pm 0.88$ | $112.54$ | $0.256$ |
| **4. Advanced Hybrid: 1D-CNN + Bi-LSTM + Custom Attn** | $78,342$ | $977.3$ | **$95.29 \pm 0.43$** | **$95.24 \pm 0.44$** | **$91.99 \pm 0.94$** | **$92.13 \pm 0.94$** | $60.07$ | $0.267$ |

### Key Findings from Benchmark

* **Generalization Gap:** Random Forest achieved strong validation accuracy ($92.95\%$) but dropped sharply to **$83.49\%$** on unseen test subjects. All three deep learning architectures generalized significantly better across unseen subjects.
* **Top Performer & Efficiency:** The **Advanced Hybrid** achieved the highest validation accuracy (**$95.29\%$**), highest test accuracy (**$91.99\%$**), and highest test Macro-F1 (**$92.13\%$**). Moreover, because `MaxPooling1D(2)` halved the sequence length from $128$ to $64$ before the Bi-LSTM, it trained faster (**$60.07\text{s}$**) than both the standalone Bi-LSTM (**$76.43\text{s}$**) and the Transformer (**$112.54\text{s}$**).
* **Transformer vs. Recurrent Hybrids:** Without convolutional inductive bias for local high-frequency sensor motifs, the lightweight Transformer ($88.73\%$ test accuracy) trailed both the Classic Bi-LSTM ($91.23\%$) and Advanced Hybrid ($91.99\%$) on this $6,037$-window dataset.

---

## 4. Validation-Only Ablation & Tuning Protocol (Part 3)

`Part 3` isolates all design decisions strictly on the **validation set** (leaving the $9$ test subjects completely untouched to prevent selection bias) across $4$ controlled experiments:

1. **Hyperparameter Grid (Bi-LSTM):** Evaluates hidden units $\in \{32, 64, 128\} \times$ learning rates $\in \{10^{-3}, 3\times 10^{-4}\}$ (`study_hyperparameters.csv`).
2. **Data Strategy (Bi-LSTM, 64 units):** Compares train-only Z-score scaling, raw unscaled inputs, and Z-score + Gaussian jitter augmentation ($\sigma = 0.05$, doubling training samples) (`study_data_strategy.csv`).
3. **Sensor Channel Ablation (H3):** Evaluates dropping Gyroscope `[3,4,5]`, Total Acceleration `[6,7,8]`, or Body Acceleration `[0,1,2]` against the full 9-channel input (`study_ablation.csv`).
4. **Hybrid Component Ablation (H2):** Tests removing the custom `TemporalAttentionLayer` (replacing it with `GlobalAveragePooling1D`) and removing the `1D-CNN` front-end (`study_ablation.csv`).
