# Brain Tumor MRI Classification Using Convolutional Neural Networks

A deep learning image classification model using CNNs to identify brain tumor types from MRI scans, with automated hyperparameter tuning via Keras Tuner.

---

## Overview

This project trains a Convolutional Neural Network to classify brain MRI images into four categories: glioma, meningioma, pituitary tumor, and healthy tissue. The notebook covers a complete deep learning workflow including data exploration, preprocessing, architecture search, training, and evaluation.

---

## Objective

Develop a CNN-based classifier capable of accurately identifying the type of brain tumor — or the absence of one — from MRI images.

---

## Dataset

- **Source:** [Brain Tumor MRI Scans on Kaggle](https://www.kaggle.com/) (downloaded automatically via KaggleHub)
- **Classes:** `glioma`, `meningioma`, `pituitary`, `no_tumor`
- Each class contains labeled MRI images representing different tumor types or healthy tissue.

---

## Requirements

```bash
pip install tensorflow keras keras-tuner visualkeras kagglehub scikit-learn matplotlib pillow numpy
```

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | CNN model creation and training |
| Keras Tuner | Hyperparameter optimization |
| Scikit-learn | Evaluation metrics |
| Matplotlib | Data visualization |
| NumPy | Numerical operations |
| Pillow (PIL) | Image handling |
| KaggleHub | Dataset download |
| VisualKeras | Neural network architecture visualization |

---

## Key Parameters

| Parameter | Value |
|-----------|-------|
| `SEED` | 42 |
| `IMAGE_SIZE` | (112, 112) |
| `BATCH_SIZE` | 32 |
| `EPOCHS` | 40 |

---

## Workflow

### 1. Data Extraction and Exploratory Analysis

The dataset is downloaded from Kaggle via KaggleHub. The following EDA steps are performed:

- **Class distribution:** Bar plots and pie charts showing the number of images per class, used to verify dataset balance.
- **Image dimension analysis:** All image sizes are inspected to detect inconsistencies and confirm whether resizing is necessary.
- **Random image visualization:** Sample images from each class are displayed to validate labels, inspect quality, and understand visual differences between tumor types.

### 2. Dataset Splitting

Images are loaded and resized using `image_dataset_from_directory()`.

| Split | Proportion |
|-------|------------|
| Train | 80% |
| Validation | 10% |
| Test | 10% |

Performance optimizations applied: dataset caching, shuffling, and prefetching.

### 3. Model Creation and Hyperparameter Tuning

A CNN is built dynamically using **Keras Tuner with Random Search** to find the best architecture configuration.

**Tuned hyperparameters:**

| Parameter | Search Space |
|-----------|-------------|
| Convolutional layers | 2 to 4 |
| Filters per layer | 32, 64, 128, 256 |
| Dense layer units | 64 to 512 |
| Dropout rate | 0.0 to 0.5 |
| Learning rate | 1e-4 to 1e-2 |

**CNN architecture components:**

- Rescaling layer (normalizes pixel values to [0, 1])
- Multiple Conv2D + MaxPooling layers
- Flatten layer
- Dense layer
- Optional Dropout layer
- Output layer with softmax activation (4 classes)

**Search configuration:**

- Maximum trials: 10
- Early stopping patience: 3 epochs
- Selection objective: validation accuracy

### 4. Model Training

The best model from the search is trained on the full training set. The following metrics are tracked and plotted across epochs:

- Training and validation accuracy
- Training and validation loss

These plots help identify overfitting, underfitting, or convergence issues.

### 5. Model Evaluation

The trained model is evaluated on the held-out test set. Outputs include:

- **Classification report:** Precision, recall, F1-score, and accuracy per class.
- **Confusion matrix:** Visualizes correct predictions and misclassifications across all four classes.
- **Architecture visualization:** The final CNN structure is rendered using VisualKeras, showing layer order, feature map dimensions, and model depth.

---

## Outputs

| Output | Description |
|--------|-------------|
| Class distribution plots | Bar and pie charts per class |
| Random image samples | Visual inspection of each category |
| Image dimension analysis | Summary of dataset image sizes |
| CNN architecture diagram | VisualKeras layer visualization |
| Training curves | Accuracy and loss vs. epochs |
| Classification report | Per-class precision, recall, F1 |
| Confusion matrix | Heatmap of predictions vs. true labels |

---

## How to Run

1. Install dependencies:

```bash
pip install tensorflow keras keras-tuner visualkeras kagglehub scikit-learn matplotlib pillow numpy
```

2. Configure your Kaggle API credentials (`~/.kaggle/kaggle.json`).

3. Run all notebook cells sequentially. The dataset downloads automatically on first execution.

---

## Project Structure

```
.
├── notebook.ipynb      # Main notebook with full pipeline
└── README.md           # This file
```
