# Water Pipeline Leak Prediction — Data Engineering & ML/Deep Learning Pipeline

## Overview
This repository demonstrates an **end‑to‑end workflow** for predicting water pipeline leaks using historical **maintenance work orders** and **pipeline asset attributes** (sensor‑light setting). It is part of my Honours research at the **University of KwaZulu‑Natal** (Supervisor: **Prof. Okuthe Paul Kogeda**).

The project integrates:
- **Data Engineering & EDA** (cleaning, merging, feature construction, imbalance analysis)  
- **Modeling** with a **strong classical baseline (Logistic Regression)** and **deep learning** variants (Tabular MLP, Text BiLSTM, and late‑fusion Text+Tabular)  
- **Threshold calibration** on the **Precision–Recall (PR) curve** to balance precision and recall under class imbalance

> Results below summarize the manuscript’s findings and figures (see `Water_Pipeline_Leak_Prediction.pdf` in this repo).

---

## Repository Structure

| File | Description |
| --- | --- |
| **`data_engineering.ipynb`** | Data ingestion, cleaning, deduplication, label creation, **feature engineering** (age, diameter, material, system/subsystem, temporal features, historical indicators), and EDA (distributions, class imbalance, seasonal trends). |
| **`water_pipeline_leak_pred.ipynb`** | **Unified modeling pipeline** combining text (TF‑IDF) + categorical (one‑hot/embeddings) + numeric (scaled) features; classifier training; **PR‑curve threshold tuning**; ROC/PR plots; confusion matrix; calibration. |
| **`Water_Pipeline_Leak_Prediction.pdf`** | Research paper with full methodology, architecture diagrams, ablations, and **final quantitative results** (classical vs deep models). |

---

## Data Sources (Summary)
- **Work‑order records (2020–present):** free‑text descriptions (Problem/Location/Work), activity codes, priorities, timestamps.  
- **Pipeline asset register:** material, diameter, length, installation year, system/sub‑system.
- **Labeling:** Leak events identified by **keyword rules + maintenance codes** → `target_leak` (binary).  
- **Imbalance:** ~18% positives (leak).

---

## Data Engineering Pipeline (what `data_engineering.ipynb` does)
- **Cleaning & Merge:** deduplicate, align IDs, merge work‑orders ↔ assets.  
- **Feature Engineering:**  
  - **Text:** concatenate fields → `text_all`.  
  - **Categorical:** order/priority/activity/system/subsystem/material category.  
  - **Numeric:** `feat_age` (current year − install year), diameter, length, cost indicators.  
  - **Temporal:** year, month; aggregate counts for seasonality; optional time‑based splits.  
- **EDA:** Distributions (material/age/diameter), class balance, yearly/monthly trends, interactions (e.g., age×material).

---

## Modeling (what `water_pipeline_leak_pred.ipynb` implements)
- **Classical Baseline:** `LogisticRegression(class_weight='balanced')` with a **scikit‑learn Pipeline**:  
  - **Text:** TF‑IDF (n‑grams) on `text_all`  
  - **Categorical:** One‑hot encoding  
  - **Numeric:** Standardization  
- **Deep Learning Variants (per manuscript):**  
  - **Tabular MLP** (numerics + embedded categoricals)  
  - **Text BiLSTM** (tokenized text with trainable embeddings)  
  - **Fusion BiLSTM** (concatenate text + tabular latent vectors)

**Training/Eval:** 80/20 stratified split; early stopping on validation; **threshold selection by maximizing F1 on PR curve**; report ROC‑AUC, PR‑AUC, Precision, Recall, F1; probability **calibration** + confusion matrices.

---

## Results (from the manuscript)
| Model | ROC‑AUC | PR‑AUC | Precision | Recall | F1 |
| --- | ---:| ---:| ---:| ---:| ---:|
| **Logistic Regression (baseline)** | **0.995** | **0.977** | **0.907** | **0.936** | **0.921** |
| **Tabular MLP** | 0.816 | 0.647 | 0.671 | 0.635 | 0.652 |
| **Text BiLSTM** | **0.999** | **0.997** | **0.977** | **0.982** | **0.980** |
| **Fusion BiLSTM (Text+Tabular)** | **0.999** | **0.996** | **0.979** | **0.978** | **0.979** |

**Key insight:** free‑text maintenance narratives carry the **dominant predictive signal**; tabular features add **marginal calibration gains**. (See PR/ROC, calibration, and confusion matrices in the PDF.)

---

## Tech Stack
- **Python**, **Jupyter**  
- **pandas**, **NumPy**  
- **scikit‑learn** (TF‑IDF, OneHotEncoder, StandardScaler, Logistic Regression)  
- **TensorFlow/Keras** (BiLSTM/MLP in manuscript experiments)  
- **matplotlib**, **seaborn**, **joblib**

---

## Getting Started
```bash
# 1) Clone
git clone https://github.com/Samukelo09/water-pipeline-leak-prediction.git
cd water-pipeline-leak-prediction

# 2) Create & activate environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 3) Install deps
pip install -r requirements.txt

# 4) Run notebooks
jupyter notebook data_engineering.ipynb
jupyter notebook water_pipeline_leak_pred.ipynb
```

> Tip: For reproducible runs, set a global seed and persist the fitted pipeline with `joblib.dump(...)`.

---

## Citation & Acknowledgements
If you reference this work, please cite the accompanying manuscript in this repository.  
Acknowledgements: **NRF bursary**, **Umgeni Water** (data access), and **Prof. Okuthe Paul Kogeda** (supervision).

---

##  Author
**Samukelo Mkhize** · University of KwaZulu‑Natal  
📧 [Email](mailto:samkelomanager@gmail.com) · 🌐 [Website](https://samukelo09.github.io/My-Personal-Portfolio/) · 🔗 [GitHub](https://github.com/Samukelo09)
