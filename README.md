## 🧠 ML Healthcare Portfolio

A collection of end‑to‑end machine‑learning and deep‑learning projects focused on healthcare diagnostics.

## 📁 Repository Structure

```
.
├── 001_Cervical_Cancer_Predection_with_ML/
├── 002_COVID19_Prediction_from_Chest_Xray_Images_with_CNN/
├── 007_Breast_Cancer_Prediction_with_ML/
└── README.md
```

> Tip: Each project folder is self‑contained with code, data pointers, and results artifacts.

---

## 🚀 Quick Start

### 1) Clone & Setup

```bash
# clone
git clone <your-repo-url>.git
cd <your-repo-name>

# (recommended) create a virtual env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install common dependencies
pip install -r requirements.txt
```

> If a project includes its own `requirements.txt`/`environment.yml`, prefer that for exact reproducibility.

### 2) Run

Most projects provide a Jupyter notebook and a Python entry‑point script. Typical patterns:

```bash
# Notebooks
jupyter lab  # or: jupyter notebook

# Training / inference scripts (if available)
python train.py   # trains the model
python infer.py   # runs inference on sample data
```

---

## 🧪 Datasets & Ethics

* Datasets are referenced via links or expected in a local `data/` folder inside each project.
* Use datasets responsibly and comply with licenses/IRB/consent requirements.
* Models are **assistive**, not a replacement for clinical judgment.

---

## 📊 Projects

### 001 — Cervical Cancer Prediction (Classical ML)

**Goal:** Predict risk of cervical cancer using tabular clinical features.

**Highlights**

* Data cleaning: missing‑value strategies, outlier handling, class‑imbalance remedies (e.g., SMOTE/weighted loss).
* Feature engineering: encoding categorical variables, scaling, correlation/variance filters.
* Models: Logistic Regression, Random Forest, XGBoost (with cross‑validation).
* Evaluation: ROC‑AUC, PR‑AUC (for imbalance), F1, calibration plots.
* Explainability: permutation importance / SHAP for feature attributions.

**How to run**

* Open the notebook in `001_Cervical_Cancer_Predection_with_ML/` and run cells top‑to‑bottom **or**
* Use `python train.py` (if provided) with CLI flags, e.g.:

  ```bash
  python train.py --model xgboost --test-size 0.2 --seed 42
  ```

**Expected folders**

```
001_Cervical_Cancer_Predection_with_ML/
├── data/              # raw/processed CSVs (ignored in git by default)
├── notebooks/
├── src/
├── models/            # saved models (.pkl/.joblib)
└── reports/           # metrics, plots
```

---

### 002 — COVID‑19 Detection from Chest X‑rays (CNN)

**Goal:** Classify chest X‑ray images as COVID‑19 vs. Normal/Other using a convolutional neural network.

**Highlights**

* Data pipeline: train/val/test splits, on‑the‑fly augmentations, balanced sampling.
* Architectures: Custom CNN and/or transfer learning (e.g., ResNet/DenseNet).
* Training: early stopping, learning‑rate scheduling, mixed precision (optional).
* Evaluation: accuracy, F1, ROC‑AUC; confusion matrix; per‑class precision/recall.
* Explainability: Grad‑CAM heatmaps for visual sanity checks.

**How to run**

```bash
cd 002_COVID19_Prediction_from_Chest_Xray_Images_with_CNN
# notebook workflow
jupyter lab
# or script workflow
python train.py --arch resnet18 --epochs 25 --img-size 224 --batch-size 32
python infer.py --weights runs/best.pt --input samples/
```

**Data layout (example)**

```
002_COVID19_Prediction_from_Chest_Xray_Images_with_CNN/
├── data/
│   ├── train/
│   │   ├── covid/
│   │   └── normal/
│   ├── val/
│   └── test/
├── src/
├── runs/        # checkpoints & logs
└── reports/
```

---

### 007 — Breast Cancer Prediction (Classical ML)

**Goal:** Predict malignancy using features (e.g., Wisconsin Breast Cancer dataset).

**Highlights**

* Preprocessing: imputation, scaling, feature selection.
* Models: Logistic Regression, SVM, Random Forest, Gradient Boosting.
* Model selection: nested CV / grid‑search with stratification.
* Evaluation: ROC‑AUC, sensitivity/specificity at clinically meaningful thresholds.
* Interpretability: SHAP/coefficients for feature influence.

**How to run**

* Use the notebook in `007_Breast_Cancer_Prediction_with_ML/` **or**

```bash
python train.py --model svm --kernel rbf --C 2.0
```

---

## 🔧 Common Configuration

Create a `.env` file in the project root for settings shared across projects:

```
SEED=42
NUM_WORKERS=4
LOG_LEVEL=INFO
```

Projects read environment variables if available; otherwise, sensible defaults are used.

---

## 🧰 Tooling

* **Python:** 3.9+
* **Core:** numpy, pandas, scikit‑learn, matplotlib, pillow
* **DL (for CNN):** torch/torchvision or tensorflow/keras (as used in the project)
* **Utilities:** jupyter, tqdm, shap, imbalanced‑learn

> Exact versions are pinned in each project’s `requirements.txt` when reproducibility is critical.

---

## ✅ Reproducibility Checklist

*

---

## 📈 Results (Snapshot)

Add your latest key results below as you run experiments:

| Project         | Model     | ROC‑AUC | F1   | Notes                         |
| --------------- | --------- | ------- | ---- | ----------------------------- |
| Cervical Cancer | XGBoost   | 0.94    | 0.88 | Weighted loss, SMOTE          |
| COVID‑19 X‑ray  | ResNet18  | 0.97    | 0.95 | 224², Grad‑CAM sanity‑checked |
| Breast Cancer   | SVM (RBF) | 0.99    | 0.98 | Calibrated probabilities      |

> Replace with your actual numbers/plots; include links to `reports/`.

---

## 📜 License

This repository is released under the MIT License (see `LICENSE`).

---

## 🙌 Acknowledgments

* Open‑source dataset providers & maintainers
* Authors of baseline architectures and libraries

---

## 📨 Contact

**Ayushi Sharma**
Email: [ayushi66sharma@gmail.com](mailto:ayushi66sharma@gmail.com)

 --
