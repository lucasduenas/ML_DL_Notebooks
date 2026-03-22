# BBC News Multi-Class Classification using Transformers

A deep learning pipeline that fine-tunes a DistilBERT-based classifier to categorize BBC news articles into multiple topic categories.

---

## Overview

This project builds a multi-class text classification model using a pretrained DistilBERT transformer to generate sentence embeddings, which are then passed through a custom neural network for final classification. The model is trained and evaluated on a BBC news dataset sourced from Kaggle.

---

## Pipeline

1. Import required libraries
2. Load and explore the dataset
3. Split data into train, validation, and test sets
4. Load the pretrained DistilBERT tokenizer and model
5. Build and train a classification neural network on top of the embeddings
6. Evaluate results quantitatively and qualitatively

---

## Dataset

- **Source:** [BBC Articles Dataset on Kaggle](https://www.kaggle.com/datasets/jacopoferretti/bbc-articles-dataset)
- **File:** `bbc_news_text_complexity_summarization.csv`
- **Columns used:** `text`, `labels`
- **Classes:** 5 news categories (business, entertainment, politics, sport, tech)
- The dataset is balanced across all categories.

---

## Requirements

```
kagglehub
pandas
numpy
torch
scikit-learn
transformers
matplotlib
seaborn
```

Install all dependencies with:

```bash
pip install kagglehub pandas numpy torch scikit-learn transformers matplotlib seaborn
```

---

## Data Split

| Set        | Proportion |
|------------|------------|
| Train      | 70%        |
| Validation | 15%        |
| Test       | 15%        |

Splits are stratified by label to preserve class distribution. Random state is fixed at `42` for reproducibility.

---

## Model Architecture

**Embedding layer:** DistilBERT (`distilbert-base-uncased`) — frozen, used only for inference. The `[CLS]` token representation (768-dimensional) is extracted as the sentence embedding.

**Classification head:**

```
Linear(768 -> 256)
ReLU
Dropout(0.3)
Linear(256 -> 5)
```

**Loss function:** CrossEntropyLoss (includes softmax internally)  
**Optimizer:** Adam, learning rate `2e-5`  
**Epochs:** 700

---

## Key Implementation Notes

- Texts are truncated to a maximum of 512 tokens and padded to uniform length within each batch.
- Attention masks are passed to DistilBERT so padding tokens are ignored during encoding.
- Embeddings are computed in batches of 32 with `torch.no_grad()` to reduce memory usage.
- The DistilBERT backbone is kept frozen; only the classification head is trained.
- GPU is used automatically if available (`cuda`), otherwise falls back to CPU.

---

## Results

The model achieves strong performance across all five categories on the held-out test set, with high precision, recall, and F1-score. No significant overfitting or class bias is observed. Minor confusion occurs between the `politics` and `business` categories in isolated cases, which is expected given topical overlap.

Evaluation outputs:
- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmap
- List of misclassified articles with true and predicted labels

---

## Project Structure

```
.
├── notebook.ipynb          # Main notebook with full pipeline
└── README.md               # This file
```

---

## Usage

1. Clone or download the repository.
2. Install dependencies listed above.
3. Run all cells in `notebook.ipynb` sequentially.
4. The dataset will be downloaded automatically via the Kaggle API on first run.

> A valid Kaggle account and API token (`~/.kaggle/kaggle.json`) are required to download the dataset through `kagglehub`.