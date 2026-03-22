# IMDb Movie Review Sentiment Classification Using Recurrent Neural Networks

An end-to-end NLP deep learning pipeline that trains an RNN with Word2Vec embeddings to classify IMDb movie reviews as positive or negative, with automated hyperparameter tuning via Keras Tuner.

---

## Overview

This project builds a binary sentiment classifier using LSTM-based Recurrent Neural Networks. Word2Vec embeddings trained on the review corpus are used as the input representation. The architecture is optimized through random hyperparameter search and evaluated using standard classification metrics.

---

## Objective

Develop a sentiment classification model that accurately labels IMDb movie reviews as **positive** or **negative** using RNNs and Word2Vec embeddings.

---

## Dataset

- **Source:** [IMDb Movie Ratings Sentiment Analysis on Kaggle](https://www.kaggle.com/) (downloaded automatically via KaggleHub)
- **Input:** Movie review text
- **Target:** Sentiment label — `0` (negative) or `1` (positive)
- Each row is a labeled review used for supervised binary classification.

---

## Requirements

```bash
pip install tensorflow keras keras-tuner gensim nltk kagglehub visualkeras scikit-learn matplotlib seaborn pandas numpy
```

The following NLTK resources are downloaded automatically at runtime:

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | RNN model development |
| Keras Tuner | Hyperparameter optimization |
| Gensim | Word2Vec embedding training |
| NLTK | Text preprocessing |
| Scikit-learn | Evaluation metrics |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Visualization |
| KaggleHub | Dataset download |
| VisualKeras | Neural network architecture visualization |

---

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Max sequence length | 150 words |
| Batch size | 64 |
| Epochs | 20 |
| Early stopping patience | 5 epochs |
| Word2Vec vector size | 100 |
| Word2Vec algorithm | Skip-gram |
| Word2Vec window | 5 |
| Word2Vec min frequency | 5 |

---

## Workflow

### 1. Data Extraction and Exploratory Analysis

The dataset is downloaded via KaggleHub and loaded into a Pandas DataFrame. EDA includes:

- Sentiment distribution bar chart (verifies class balance)
- Review length histogram (informs padding and sequence length decisions)

### 2. Text Preprocessing

Each review is cleaned and normalized through the following steps in order:

1. Convert to lowercase
2. Remove HTML tags
3. Remove URLs
4. Remove numbers and special characters
5. Expand contractions
6. Tokenize into words
7. Remove stopwords
8. Lemmatize tokens
9. Remove short tokens

The output is a list of cleaned tokens per review.

### 3. Word Embedding Using Word2Vec

A Word2Vec model is trained on the preprocessed tokens using the Skip-gram algorithm. The resulting 100-dimensional word vectors capture semantic relationships between words and serve as the embedding layer input to the RNN.

### 4. Data Preparation

Tokens are converted to integer indices from the Word2Vec vocabulary. Sequences are padded or truncated to a fixed length of 150 tokens.

| Split | Proportion |
|-------|------------|
| Train | 60% |
| Validation | 20% |
| Test | 20% |

Stratified splitting is used to preserve class distribution across all sets.

### 5. Model Architecture and Hyperparameter Tuning

The RNN is built dynamically using **Keras Tuner with Random Search** (10 maximum trials). The architecture includes:

- Embedding layer initialized with pretrained Word2Vec vectors
- 1 to 3 LSTM layers (optionally bidirectional)
- Dropout after each LSTM layer
- 1 to 3 Dense layers with dropout
- Sigmoid output layer for binary classification

**Tuned hyperparameters:**

| Component | Parameter | Search Space |
|-----------|-----------|-------------|
| LSTM | Number of layers | 1–3 |
| LSTM | Units per layer | 32–256 |
| LSTM | Bidirectional | True / False |
| LSTM | Dropout rate | 0.0–0.5 |
| Dense | Number of layers | 1–3 |
| Dense | Units per layer | 16–256 |
| Dense | Dropout rate | 0.0–0.5 |
| Training | Learning rate | 1e-4 to 1e-2 |
| Training | Optimizer | Adam, RMSprop |

Search objective: validation accuracy. Early stopping applied during each trial.

### 6. Model Training

The best configuration from the search is trained for up to 20 epochs with early stopping (patience = 5). The best checkpoint is saved automatically.

Training curves (accuracy and loss vs. epochs) are generated to monitor convergence and detect overfitting.

### 7. Model Evaluation

The trained model is evaluated on the held-out test set. Outputs include:

- Test accuracy and loss
- Classification report (precision, recall, F1-score)
- Confusion matrix (true/false positives and negatives)
- VisualKeras architecture diagram

---

## Outputs

| Output | Description |
|--------|-------------|
| Sentiment distribution plot | Bar chart of class counts |
| Review length histogram | Word count distribution across reviews |
| Training history plots | Accuracy and loss curves per epoch |
| Classification report | Per-class precision, recall, F1 |
| Confusion matrix | Heatmap of predictions vs. true labels |
| `best_tuned_model.h5` | Saved trained model weights |
| `best_hyperparameters.txt` | Best configuration from Keras Tuner |
| `tuned_model_training_history.png` | Training curve figure |
| `tuned_model_confusion_matrix.png` | Confusion matrix figure |

---

## How to Run

1. Install dependencies:

```bash
pip install tensorflow keras keras-tuner gensim nltk kagglehub visualkeras scikit-learn matplotlib seaborn pandas numpy
```

2. Run all notebook cells sequentially. NLTK resources and the dataset download automatically on first execution.

---

## Project Structure

```
.
├── notebook.ipynb                      # Main notebook with full pipeline
├── best_tuned_model.h5                 # Saved best model
├── best_hyperparameters.txt            # Best hyperparameter configuration
├── tuned_model_training_history.png    # Training curves
├── tuned_model_confusion_matrix.png    # Confusion matrix
└── README.md                           # This file
```
