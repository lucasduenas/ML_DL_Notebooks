Sustainable Development Goals (ODS) Text Classification Using Machine Learning
Project Overview

This project implements a multi-class text classification system to automatically classify textual data into one of the 16 Sustainable Development Goals (ODS) categories.

The model pipeline includes:

Text preprocessing
TF-IDF vectorization
Dimensionality reduction using Truncated SVD
Training and comparison of multiple classification models
Hyperparameter optimization
Selection of the best-performing model

Three machine learning models are evaluated:

XGBoost
Support Vector Machine (SVM)
Logistic Regression

The final model is selected based on classification accuracy using cross-validation.

Objective

The objective of this project is to develop a machine learning model capable of automatically classifying text into the appropriate Sustainable Development Goal (ODS) category.

This is a multi-class classification problem involving 16 target categories.

Dataset

The dataset is loaded from an Excel file:

Train_textosODS.xlsx

It contains:

textos → Text data used as input features
ODS → Target category label (values from 1 to 16)
Label Adjustment

Since some machine learning models require labels starting from zero, the ODS labels are transformed:

Original labels: 1–16
Transformed labels: 0–15

This transformation ensures compatibility with XGBoost.

Technologies and Libraries Used

The following libraries are used:

Pandas — Data manipulation
NumPy — Numerical operations
NLTK — Text preprocessing
Scikit-learn — Machine learning models and evaluation
XGBoost — Gradient boosting classifier

Key components:

TF-IDF Vectorization
Dimensionality Reduction (Truncated SVD)
Machine Learning Pipelines
Hyperparameter Optimization
Cross-Validation

Required installations:

pip install pandas numpy nltk scikit-learn xgboost

Workflow Summary

The notebook follows five main stages.

1. Import Required Libraries

All required libraries for:

Text preprocessing
Feature extraction
Model training
Evaluation

are imported.

NLTK components used:

Spanish stopwords
Tokenization tools
Stemming using PorterStemmer

2. Data Loading and Exploration

The dataset is loaded from an Excel file and explored to understand its structure.

Class Distribution Analysis

The distribution of ODS categories is analyzed using:

value_counts()

This reveals that:

The dataset contains class imbalance
Some categories have significantly more samples than others

This observation is important for model evaluation and interpretation.

Data Quality Verification

The dataset is checked for:

Missing values
Duplicate records

Results:

No null values detected
No duplicated rows detected

Therefore, no data removal was required.

2.1 Dataset Splitting

The dataset is divided into:

Training set: 80%
Test set: 20%

Method used:

train_test_split()

Random state:

52

This ensures reproducibility of results.

3. Pipeline Construction

Custom pipelines are created to standardize the workflow for each model.

Each pipeline includes:

Text preprocessing
TF-IDF vectorization
Dimensionality reduction
Classification model
3.1 Text Preprocessing

A custom preprocessing function is created to clean and normalize text data.

Preprocessing Steps

Each text undergoes:

Conversion to lowercase
Tokenization using regular expressions
Removal of punctuation
Removal of Spanish stopwords
Word stemming using PorterStemmer
Reconstruction into processed text

Tokenizer used:

RegexpTokenizer(r'\w+')

Stopwords:

Spanish stopwords from NLTK.

Stemming method:

PorterStemmer.

This step reduces noise and improves feature quality.

3.2 Feature Engineering Pipeline
TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical vectors.

Configuration:

max_features = 5000

This limits the vocabulary to the most relevant 5000 terms.

TF-IDF helps:

Emphasize meaningful words
Reduce the influence of common words
Dimensionality Reduction Using Truncated SVD

Truncated Singular Value Decomposition (SVD) is used to reduce dimensionality.

Configuration:

n_components = 100

Benefits:

Reduces computational cost
Removes noisy features
Improves model efficiency
Reduces overfitting risk
3.3 Classification Models

Three classification models are tested.

XGBoost Classifier

Configuration includes:

Evaluation metric: mlogloss

Hyperparameters tuned:

Number of estimators: 50, 100, 200
Maximum depth: 3, 5, 7
Learning rate: 0.01, 0.1, 0.2

XGBoost is known for strong performance on structured datasets.

Support Vector Machine (SVM)

Initial configuration:

Kernel: Linear
Probability output enabled

Hyperparameters tuned:

Regularization parameter C: 0.1, 1, 10
Kernel type: Linear, RBF

SVM allows non-linear decision boundaries when using RBF kernel.

Logistic Regression

Configuration:

max_iter = 1000

Hyperparameters tuned:

Regularization parameter C: 0.1, 1, 10
Solver type: lbfgs, liblinear

Logistic Regression is effective for multi-class classification.

3.4 Hyperparameter Search

Hyperparameter optimization is performed using:

RandomizedSearchCV

Configuration:

Number of iterations: 10
Cross-validation folds: 5
StratifiedKFold used
Scoring metric: Accuracy

StratifiedKFold ensures:

Balanced representation of classes
Reliable validation performance

4. Model Selection

Each pipeline is trained using cross-validation.

For each model:

Randomized hyperparameter search is performed
Validation results are recorded
Best hyperparameters are selected

The final model is chosen based on:

Highest validation accuracy.

Final Model Evaluation

The best model is evaluated on the test dataset using:

classification_report()

Metrics reported:

Precision
Recall
F1-score
Accuracy

These metrics are reported for all 16 ODS categories.

Results Summary

The best-performing model was:

Logistic Regression

Best configuration:

Solver: lbfgs
Regularization parameter C = 10

This configuration produced:

High classification accuracy
Balanced performance across most categories
Good generalization without overfitting
Observed Challenges

Some categories showed lower recall performance, particularly:

ODS 8
ODS 9
ODS 11

Possible causes:

Class imbalance
Similar textual patterns between categories

These factors increased misclassification rates.

Key Parameters

TF-IDF Maximum Features:

5000

SVD Components:

100

Cross-Validation Splits:

5

Random Search Iterations:

10

Test Size:

20%

Outputs Generated

The notebook produces:

Class distribution analysis
Model training results
Hyperparameter tuning summaries
Best model selection
Classification report
How to Run This Notebook

Step 1: Install dependencies

pip install pandas numpy nltk scikit-learn xgboost

Step 2: Download required NLTK resources

nltk.download('stopwords')

Step 3: Ensure the dataset file is available:

microproyecto2_data/Train_textosODS.xlsx

Step 4: Run the notebook sequentially.