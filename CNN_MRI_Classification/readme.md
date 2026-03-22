Brain Tumor MRI Classification Using Convolutional Neural Networks (CNN)
Project Overview

This project implements a Deep Learning image classification model using Convolutional Neural Networks (CNNs) to classify brain MRI scans into four categories:

Glioma
Meningioma
Pituitary Tumor
Healthy Tissue (No Tumor)

The model is trained using MRI brain scan images and optimized through hyperparameter tuning to identify the best-performing architecture.

This notebook demonstrates a complete deep learning workflow, including data exploration, preprocessing, model tuning, training, evaluation, and results analysis.

Objective

The objective of this project is to develop a CNN-based classification model capable of accurately identifying the type of brain tumor from MRI images.

Dataset

The dataset is automatically downloaded using KaggleHub.

Dataset: Brain Tumor MRI Scans
Source: Kaggle

The dataset contains labeled MRI images organized into four classes:

glioma
meningioma
pituitary
no_tumor

Each class contains MRI images representing different tumor types or healthy tissue.

Technologies and Libraries Used

The following Python libraries are used:

TensorFlow / Keras — CNN model creation and training
Keras Tuner — Hyperparameter optimization
Scikit-learn — Model evaluation metrics
Matplotlib — Data visualization
NumPy — Numerical operations
Pillow (PIL) — Image handling
KaggleHub — Dataset download
VisualKeras — Neural network visualization

Required installations:

pip install tensorflow keras keras-tuner visualkeras kagglehub scikit-learn matplotlib pillow numpy

Workflow Summary

The notebook follows six main stages.

1. Import Required Libraries

All required libraries for:

Data processing
Image handling
Model development
Visualization
Performance evaluation

are imported at the beginning of the notebook.

2. Data Extraction and Exploratory Analysis

The dataset is downloaded from Kaggle using KaggleHub.

Several exploratory data analysis (EDA) steps are performed to understand the dataset.

Class Distribution Visualization

A function is created to:

Count the number of images per class
Generate bar plots showing the number of images per class
Generate pie charts showing percentage distribution

This step verifies whether the dataset is balanced across categories.

Image Dimension Analysis

All image sizes are analyzed to:

Detect inconsistent dimensions
Confirm whether resizing is necessary

This ensures compatibility with CNN input requirements.

Random Image Visualization

Random images from each class are displayed to:

Validate class labels
Inspect image quality
Understand visual differences between tumor types
3. Dataset Splitting

The dataset is split into:

80% Training
10% Validation
10% Testing

Images are resized to:

IMAGE_SIZE = (112, 112)

TensorFlow's image_dataset_from_directory() function is used to:

Load image data
Apply resizing
Create batches

Performance optimization techniques used:

Dataset caching
Data shuffling
Prefetching

These optimizations improve training efficiency.

4. Model Creation and Hyperparameter Tuning

A CNN model is dynamically created using Keras Tuner with Random Search to identify the best model configuration.

Tuned Hyperparameters

The following parameters are optimized:

Number of convolutional layers: 2 to 4
Number of filters per layer: 32, 64, 128, 256
Dense layer units: 64 to 512
Dropout rate: 0.0 to 0.5
Learning rate: 1e-4 to 1e-2
CNN Architecture Components

The CNN includes:

Rescaling layer (normalizes pixel values)
Multiple Conv2D layers
MaxPooling layers
Flatten layer
Dense layer
Dropout layer (optional)
Output layer with softmax activation (4 classes)
Training Optimization

Early stopping is used to prevent overfitting.

Configuration:

Maximum trials: 10
Patience: 3 epochs
Objective: Validation accuracy

The best model is selected automatically based on validation performance.

5. Model Training

The selected best model is trained using the training dataset.

Metrics tracked during training:

Training accuracy
Validation accuracy
Training loss
Validation loss

These metrics are plotted across epochs to visualize learning progress.

Training History Visualization

Two plots are generated:

Accuracy vs Epochs
Loss vs Epochs

These plots help identify:

Overfitting
Underfitting
Training convergence
6. Model Evaluation

The trained model is evaluated using the test dataset.

Predictions are generated for test images and compared against true labels.

Classification Metrics

A classification report is generated including:

Precision
Recall
F1-score
Accuracy

These metrics evaluate overall model performance and class-level prediction quality.

Confusion Matrix

A confusion matrix is generated to visualize:

Correct predictions
Misclassifications
Performance across individual classes

This provides deeper insight into classification errors.

Model Visualization

The final CNN architecture is visualized using VisualKeras.

This visualization displays:

Layer structure
Feature map dimensions
Model depth

It provides a graphical overview of the neural network architecture.

Key Parameters

SEED = 42
IMAGE_SIZE = (112, 112)
BATCH_SIZE = 32
EPOCHS = 40

Outputs Generated

The notebook produces the following outputs:

Class distribution plots
Random image samples
Image dimension analysis
CNN architecture visualization
Training performance graphs
Classification report
Confusion matrix
How to Run This Notebook

Step 1: Install dependencies

pip install tensorflow keras keras-tuner visualkeras kagglehub scikit-learn matplotlib pillow numpy

Step 2: Configure Kaggle API credentials.

Step 3: Run the notebook sequentially.

The dataset will be downloaded automatically during execution.