# 🏦 Loan Approval Prediction System (LoanGuard AI)

A complete end-to-end Machine Learning project to predict loan approval using Python and classical machine learning algorithms.

---

## 📌 Project Overview
The **Loan Approval Prediction System** analyzes applicant profiles (income, education, employment status, credit history, loan amount, and property area) to evaluate credit risk and predict whether a loan will be approved or rejected, along with approval probability and financial risk factor explanations.

---

## 📂 Project Directory Structure

```text
Loan_Approval_Prediction/
│── dataset/
│   └── loan_data.csv                    # Dataset used for training and evaluation
│── notebooks/
│   └── EDA.ipynb                        # Complete Exploratory Data Analysis notebook
│── models/
│   ├── best_model.joblib                # Champion end-to-end ML pipeline model
│   └── model_metadata.json              # Model architecture metadata and metrics
│── src/
│   ├── data_preprocessing.py            # Preprocessing, feature engineering & ColumnTransformer
│   ├── model_training.py                # Base classifiers, Voting & Stacking ensembles, PCA, GridSearchCV
│   └── evaluation.py                    # Multi-model ROC/PR curves, stratified CV & metrics
│── tests/
│   └── test_pipeline.py                 # Automated unit tests
│── outputs/
│   ├── model_comparison_test.csv        # Test set performance table
│   ├── model_comparison_cv.csv          # 5-Fold Stratified CV scores
│   ├── roc_curves_comparison.png        # Multi-model ROC curves
│   ├── precision_recall_comparison.png  # Multi-model Precision-Recall curves
│   ├── feature_importance_top15.png     # Top 15 most important features
│   ├── model_comparison_benchmark.png   # Benchmark performance bar chart
│   └── cm_*.png                         # Confusion matrices for all models
│── web_app.py                           # Zero-dependency interactive web dashboard
│── train.py                             # Master training and benchmarking script
│── predict.py                           # Standalone prediction module & risk factor analysis
│── app.py                               # Interactive Terminal / CLI interface
│── requirements.txt                     # Core dependencies
│── README.md                            # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### 2. Setup Virtual Environment & Install Dependencies
```bash
cd Loan_Approval_Prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Train the Machine Learning Pipeline
Executes data preprocessing, domain feature engineering, trains 6 classical algorithms + 2 ensembles, performs 5-fold Stratified CV, benchmarks PCA, tunes the Random Forest model with GridSearchCV, saves all visualization plots to `outputs/`, and saves the champion model to `models/best_model.joblib`:
```bash
python train.py
```

### 2. Launch the Interactive Web Dashboard
Launches a modern, zero-dependency browser UI powered by Python's standard library (`http.server`):
```bash
python web_app.py
```
Open your web browser and visit: **`http://localhost:8080`**

### 3. Run the Interactive CLI Prediction Tool
Enter applicant details directly in the terminal:
```bash
python app.py
```

### 4. Run Automated Unit Tests
Verify mathematical calculations, transformer shapes, and prediction boundaries:
```bash
python -m unittest discover tests
```

### 5. View Exploratory Data Analysis (EDA) Notebook
Open and explore `notebooks/EDA.ipynb` with Jupyter:
```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## 🎯 Expected Outcomes & Sample Outputs

### 1. Training Pipeline Expected Output (`python train.py`)
```text
============================================================
LOAN APPROVAL PREDICTION SYSTEM - ENHANCED ML PIPELINE
============================================================

[Step 1/8] Loading Data & Performing Domain Feature Engineering...
Train samples: 491 | Test samples: 123
[Step 2/8] Constructing Base Model Pipelines...
[Step 3/8] Building Classical Ensemble Pipelines (Voting & Stacking)...
[Step 4/8] Training All Base Models and Ensembles...
[Step 5/8] Evaluating Models & Generating ROC/PR Curves...

--- Test Set Benchmark Performance ---
                   Model  Accuracy  Precision   Recall  F1 Score  ROC-AUC   PR-AUC
Voting Classifier (Soft)  0.845528   0.836735 0.964706  0.896175 0.865635 0.907496
           Random Forest  0.886179   0.881720 0.964706  0.921348 0.856192 0.888875
     Stacking Classifier  0.869919   0.848485 0.988235  0.913043 0.843653 0.893387
  Support Vector Machine  0.861789   0.846939 0.976471  0.907104 0.847678 0.908636
    Gaussian Naive Bayes  0.845528   0.836735 0.964706  0.896175 0.838080 0.909360
     Logistic Regression  0.829268   0.880952 0.870588  0.875740 0.850774 0.879006
     K-Nearest Neighbors  0.853659   0.860215 0.941176  0.898876 0.798607 0.845975
           Decision Tree  0.707317   0.845070 0.705882  0.769231 0.721981 0.780858

[Step 6/8] Running 5-Fold Stratified Cross-Validation on Full Dataset...
[Step 7/8] Benchmarking with PCA Dimensionality Reduction...
[Step 8/8] Hyperparameter Tuning for Random Forest Pipeline...

============================================================
*** CHAMPION MODEL SELECTED: Voting Classifier (Soft) (ROC-AUC: 0.8656) ***
============================================================
Saved complete end-to-end pipeline model to: models/best_model.joblib
Training pipeline finished successfully!
```

---

### 2. CLI Inference Expected Output (`python app.py`)

**Case A: Approved Applicant (Prime Credit & Income Profile)**
```text
============================================================
     [LOANGUARD AI] LOAN APPROVAL PREDICTION SYSTEM
============================================================
Press Enter to accept defaults, or enter custom applicant data:

Gender (Male/Female) [Default: Male]: Male
Married Status (Yes/No) [Default: Yes]: Yes
Dependents (0/1/2/3+) [Default: 0]: 0
Education (Graduate/Not Graduate) [Default: Graduate]: Graduate
Self Employed (Yes/No) [Default: No]: No
Applicant Monthly Income ($) [Default: 5400]: 6500
Co-applicant Monthly Income ($) [Default: 1800]: 2200
Loan Amount in Thousands ($k) [Default: 140]: 160
Loan Term in Months [Default: 360]: 360
Credit History (1.0 for Good, 0.0 for Bad) [Default: 1.0]: 1.0
Property Area (Urban/Semiurban/Rural) [Default: Semiurban]: Semiurban

------------------------------------------------------------
Evaluating application through Champion Machine Learning Pipeline...
------------------------------------------------------------

DECISION:  >>> LOAN APPROVED <<<
Approval Probability: 91.82%
Rejection Risk:       8.18%

[+] Positive Financial Factors:
   * Positive Credit History record
   * Healthy Disposable Income ($675/mo after EMI)
   * Adequate household income profile

============================================================
```

**Case B: Rejected Applicant (Delinquent Credit / Subprime Profile)**
```text
DECISION:  >>> LOAN REJECTED <<<
Approval Probability: 14.30%
Rejection Risk:       85.70%

[-] Identified Risk Factors:
   * Poor Credit History (Significant risk factor)
   * High Debt Burden (EMI of $555/mo exceeds 50% of monthly income $400/mo)
```

---

### 3. Web Dashboard Expected Outcome (`python web_app.py`)
- Real-time instant underwriting verdict (`LOAN APPROVED` in emerald or `LOAN REJECTED` in rose).
- SVG animated circular score gauge displaying exact approval probability percentage (e.g. `92%`).
- Dynamic pill badges showing positive financial factors vs risk flags.
- Instant preset profile buttons (`Prime Applicant`, `Moderate Risk`, `Subprime Credit`, `Self-Employed`).

---

### 4. Unit Test Expected Outcome (`python -m unittest discover tests`)
```text
...
----------------------------------------------------------------------
Ran 3 tests in 0.097s

OK
```

---

## 📊 Model Performance Benchmark

| Model | Test Accuracy | Precision | Recall | F1 Score | Test ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Voting Classifier (Soft)** 🏆 | **84.55%** | 83.67% | 96.47% | 0.8962 | **0.8656** | **0.9075** |
| **Random Forest** | **88.62%** | 88.17% | 96.47% | 0.9213 | 0.8562 | 0.8889 |
| **Tuned Random Forest (GridSearch)** | **86.18%** | 91.46% | 88.24% | 0.8982 | 0.8632 | 0.8931 |
| **Stacking Classifier** | **86.99%** | 84.85% | 98.82% | 0.9130 | 0.8437 | 0.8934 |
| **Support Vector Machine (SVM)** | **86.18%** | 84.69% | 97.65% | 0.9071 | 0.8477 | 0.9086 |
| **Logistic Regression** | **82.93%** | 88.10% | 87.06% | 0.8757 | 0.8508 | 0.8790 |
| **Gaussian Naive Bayes** | **84.55%** | 83.67% | 96.47% | 0.8962 | 0.8381 | 0.9094 |
| **K-Nearest Neighbors** | **85.37%** | 86.02% | 94.12% | 0.8989 | 0.7986 | 0.8460 |
| **Decision Tree** | **70.73%** | 84.51% | 70.59% | 0.7692 | 0.7220 | 0.7809 |

---

## 🛠️ Technologies Used
- **Python 3.11+**
- **pandas** - Data wrangling and transformation
- **numpy** - Numerical calculations
- **scikit-learn** - Preprocessing pipelines, classical ML models, ensembles, and metrics
- **matplotlib** & **seaborn** - Publication-quality visualizations
- **joblib** - Model persistence and serialization
- **unittest** - Automated testing framework
