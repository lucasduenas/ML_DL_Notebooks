# Sustainable Development Goals (SDG) Text Classification Using Machine Learning

A multi-class text classification pipeline that automatically assigns text to one of 16 Sustainable Development Goal (SDG) categories, using TF-IDF vectorization, dimensionality reduction, and a comparison of three classical ML classifiers.

---

## Overview

This project builds a machine learning system to classify Spanish-language text into the correct SDG category. The pipeline covers text preprocessing, feature extraction, dimensionality reduction, hyperparameter search, and model selection. Three classifiers are trained and compared; the best is selected based on cross-validated accuracy.

---

## Objective

Develop a model capable of automatically assigning a text to its corresponding Sustainable Development Goal category — a 16-class classification problem.

---

## Dataset

- **File:** `microproyecto2_data/Train_textosODS.xlsx`
- **Input column:** `textos` — raw text in Spanish
- **Target column:** `ODS` — SDG category label (values 1–16, remapped to 0–15 for XGBoost compatibility)
- **Quality checks:** No null values or duplicate rows detected
- **Note:** The dataset contains class imbalance — some SDG categories have significantly more samples than others

---

## Requirements

```bash
pip install pandas numpy nltk scikit-learn xgboost
```

Additionally, download the required NLTK resources before running:

```python
import nltk
nltk.download('stopwords')
```

| Library | Purpose |
|---------|---------|
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| NLTK | Text preprocessing (stopwords, tokenization, stemming) |
| Scikit-learn | Pipelines, vectorization, SVD, models, evaluation |
| XGBoost | Gradient boosting classifier |

---

## Key Parameters

| Parameter | Value |
|-----------|-------|
| TF-IDF max features | 5000 |
| SVD components | 100 |
| Cross-validation folds | 5 |
| Random search iterations | 10 |
| Test set size | 20% |
| Random state | 52 |

---

## Workflow

### 1. Data Loading and Exploration

The dataset is loaded from Excel and explored for class distribution, missing values, and duplicates. No cleaning was required beyond label remapping.

### 2. Dataset Splitting

| Split | Proportion |
|-------|------------|
| Train | 80% |
| Test | 20% |

Splitting is done with `train_test_split()` using `random_state=52` for reproducibility.

### 3. Text Preprocessing

A custom preprocessing function is applied to each text before vectorization. Steps applied in order:

1. Convert to lowercase
2. Tokenize using `RegexpTokenizer(r'\w+')`
3. Remove punctuation
4. Remove Spanish stopwords (NLTK)
5. Apply stemming with `PorterStemmer`
6. Reconstruct into a single processed string

### 4. Feature Engineering

**TF-IDF Vectorization**  
Converts preprocessed text into numerical vectors. Vocabulary limited to the top 5000 terms by frequency-weighted relevance.

**Dimensionality Reduction — Truncated SVD**  
Reduces the TF-IDF matrix from 5000 to 100 components, lowering computational cost, removing noisy features, and reducing overfitting risk.

### 5. Classification Models

Each model is wrapped in a scikit-learn `Pipeline` that includes preprocessing, TF-IDF, SVD, and the classifier.

**XGBoost**
- Evaluation metric: `mlogloss`
- Hyperparameters searched: `n_estimators` [50, 100, 200], `max_depth` [3, 5, 7], `learning_rate` [0.01, 0.1, 0.2]

**SVM**
- Initial kernel: linear, probability output enabled
- Hyperparameters searched: `C` [0.1, 1, 10], `kernel` [linear, rbf]

**Logistic Regression**
- `max_iter=1000` to ensure convergence
- Hyperparameters searched: `C` [0.1, 1, 10], `solver` [lbfgs, liblinear]

### 6. Hyperparameter Search and Model Selection

`RandomizedSearchCV` is used with 5-fold `StratifiedKFold` cross-validation. StratifiedKFold ensures all 16 classes are represented in each fold.

The model with the highest mean validation accuracy is selected and evaluated on the test set using `classification_report()`.

---

## Results

The best-performing model was **Logistic Regression** with `solver=lbfgs` and `C=10`.

- High overall classification accuracy
- Balanced performance across most SDG categories
- Good generalization without signs of overfitting

**Categories with lower recall:** ODS 8, ODS 9, and ODS 11 showed higher misclassification rates, likely due to class imbalance and overlapping vocabulary between related goal categories.

---

## Outputs

| Output | Description |
|--------|-------------|
| Class distribution analysis | Sample counts per SDG category |
| Hyperparameter tuning summaries | Results per model and configuration |
| Best model selection | Model name and winning hyperparameters |
| Classification report | Precision, recall, F1, and accuracy for all 16 classes |

---

## How to Run

1. Install dependencies:

```bash
pip install pandas numpy nltk scikit-learn xgboost
```

2. Download NLTK resources:

```python
import nltk
nltk.download('stopwords')
```

3. Place the dataset at:

```
microproyecto2_data/Train_textosODS.xlsx
```

4. Run all notebook cells sequentially.

---

## Project Structure

```
.
├── microproyecto2_data/
│   └── Train_textosODS.xlsx    # Input dataset
├── notebook.ipynb              # Main notebook with full pipeline
└── README.md                   # This file
```
