# Wine Quality Multi-Class Classification

A machine learning pipeline that trains and compares three classifiers — XGBoost, SVM, and Logistic Regression — on a wine quality dataset, using SMOTE oversampling and randomized hyperparameter search to select the best model.

---

## Overview

This project tackles a multi-class classification problem where the goal is to predict wine quality scores from physicochemical features. The pipeline handles class imbalance via SMOTE, searches for optimal hyperparameters using cross-validated randomized search, and selects the best model based on validation accuracy.

---

## Pipeline

1. Load and inspect the dataset
2. Preprocess labels and remove duplicates
3. Split into train and test sets (stratified)
4. Apply SMOTE oversampling to the training set only
5. Define pipelines for XGBoost, SVM, and Logistic Regression
6. Run randomized hyperparameter search with stratified k-fold cross-validation
7. Select the best model and evaluate on the test set

---

## Dataset

- **File:** `data/wine_quality.csv`
- **Target column:** `quality` (integer scores, remapped to start from 0 for XGBoost compatibility)
- **Issue:** Class imbalance — some quality scores have significantly fewer samples than others
- **Preprocessing:** Drop unnamed index column, remove duplicate rows

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
scipy
```

Install with:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn scipy
```

---

## Class Imbalance Handling

SMOTE (Synthetic Minority Oversampling Technique) is applied exclusively to the training set to avoid introducing synthetic data into the evaluation set.

| Parameter | Value |
|-----------|-------|
| Strategy | `auto` (upsample all minority classes to match the majority) |
| `k_neighbors` | 3 (reduced from default 5 due to very small minority classes) |
| Random state | 52 |

The test set retains the original class distribution to reflect real-world conditions.

---

## Models and Pipelines

Three classifiers are evaluated. Each is wrapped in a scikit-learn `Pipeline`:

**XGBoost**
- No scaling required (tree-based model)
- Evaluation metric: `mlogloss` (standard for multi-class problems)

**SVM**
- StandardScaler applied before the classifier
- Initial kernel: linear (others explored during search)

**Logistic Regression**
- StandardScaler applied before the classifier
- `max_iter=1000` to ensure convergence

Scaling is included for SVM and Logistic Regression because both rely on distance computations and gradient descent, which are sensitive to feature magnitude.

---

## Hyperparameter Search

`RandomizedSearchCV` is used with `n_iter=10` and 5-fold `StratifiedKFold` cross-validation. Scoring metric: accuracy.

| Model | Parameters Searched |
|-------|---------------------|
| XGBoost | `n_estimators` [50–500], `max_depth` [1–20], `learning_rate` [0.01–0.31], `reg_lambda` [0–10] |
| SVM | `C` [0–10], `kernel` [linear, rbf] |
| Logistic Regression | `C` [0–10], `solver` [lbfgs, liblinear] |

The best model is selected based on the highest mean cross-validation accuracy across all classifiers.

---

## Results

The best-performing model is **XGBoost**, achieving approximately **85% validation accuracy** with a standard deviation of 0.003, indicating stable performance across folds.

On the test set, performance is stronger for majority classes (quality scores 5, 6, 7) and weaker for minority classes, which is expected when SMOTE-balanced training data does not fully reflect the natural test distribution.

---

## Conclusions and Recommendations

1. XGBoost is the best model for this multi-class task, with high and stable validation accuracy.
2. Minority class metrics on the test set are lower than majority class metrics — a known limitation of oversampling approaches where synthetic training data diverges from the real test distribution.
3. Potential improvements include a more exhaustive hyperparameter search, undersampling as an alternative or complement to SMOTE, or a hybrid over/undersampling strategy.
4. An ensemble approach — separate models for majority and minority classes — could also improve minority-class performance.

---

## Project Structure

```
.
├── data/
│   └── wine_quality.csv    # Input dataset
├── notebook.ipynb          # Main notebook with full pipeline
└── README.md               # This file
```
