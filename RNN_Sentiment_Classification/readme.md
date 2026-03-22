IMDb Movie Review Sentiment Classification Using Recurrent Neural Networks (RNN)
Project Overview

This project implements a Deep Learning model using Recurrent Neural Networks (RNNs) to perform sentiment analysis on movie reviews from the IMDb platform. The goal is to classify each review as either positive or negative.

The workflow includes text preprocessing, word embedding generation using Word2Vec, sequence preparation, hyperparameter tuning of RNN architectures, model training, and evaluation using classification metrics.

This notebook demonstrates an end-to-end Natural Language Processing (NLP) deep learning pipeline.

Objective

The objective of this project is to develop a sentiment classification model capable of accurately classifying IMDb movie reviews as:

Positive
Negative

using Recurrent Neural Networks and Word2Vec embeddings.

Dataset

The dataset is downloaded automatically using KaggleHub.

Dataset: IMDb Movie Ratings Sentiment Analysis
Source: Kaggle

The dataset contains:

Movie reviews (text format)
Sentiment labels
0 → Negative
1 → Positive

Each row represents a labeled movie review used for supervised learning.

Technologies and Libraries Used

The following libraries are used in this project:

TensorFlow / Keras — RNN model development
Keras Tuner — Hyperparameter optimization
Gensim — Word2Vec embedding training
NLTK — Text preprocessing
Scikit-learn — Model evaluation
Pandas — Data manipulation
NumPy — Numerical operations
Matplotlib — Visualization
Seaborn — Data visualization
KaggleHub — Dataset download
VisualKeras — Neural network visualization

Required installations:

pip install tensorflow keras keras-tuner gensim nltk kagglehub visualkeras scikit-learn matplotlib seaborn pandas numpy

Workflow Summary

The notebook follows eight main stages.

1. Import Required Libraries

All necessary libraries for:

Data loading
Text preprocessing
Word embedding
Model development
Visualization
Evaluation

are imported at the beginning.

NLTK resources downloaded:

punkt
stopwords
wordnet

These are required for tokenization, stopword removal, and lemmatization.

2. Data Extraction and Exploratory Analysis

The dataset is downloaded using KaggleHub and loaded into a Pandas DataFrame.

Initial exploratory analysis includes:

Dataset structure inspection
Summary statistics
Sentiment distribution visualization
Review length analysis
Sentiment Distribution

A bar chart is generated to show:

Number of positive reviews
Number of negative reviews

This step verifies whether the dataset is balanced.

Review Length Analysis

The number of words per review is calculated and visualized using a histogram.

This helps determine:

Typical review lengths
Maximum sequence size needed
Padding requirements
3. Text Preprocessing

Text preprocessing is performed to clean and standardize movie reviews before embedding generation.

Preprocessing Steps

Each review undergoes:

Conversion to lowercase
Removal of HTML tags
Removal of URLs
Removal of numbers and special characters
Expansion of contractions
Tokenization into words
Stopword removal
Lemmatization
Removal of short tokens

The result is a list of cleaned tokens for each review.

4. Word Embedding Using Word2Vec

A Word2Vec model is trained using the preprocessed tokens.

Configuration:

Algorithm: Skip-gram
Vector size: 100
Context window: 5
Minimum word frequency: 5

Word2Vec learns numerical vector representations of words based on their context.

These embeddings are later used as input to the neural network.

5. Data Preparation for RNN

Processed text is converted into numerical sequences using Word2Vec vocabulary indices.

Sequence Processing Steps
Convert tokens to integer indices
Limit sequence length
Pad sequences to fixed length

Maximum sequence length:

150 words

Dataset Splitting

Data is split into:

60% Training
20% Validation
20% Testing

Stratified splitting is used to maintain label distribution.

6. Model Definition and Hyperparameter Tuning

A configurable RNN architecture is defined and optimized using Keras Tuner Random Search.

Model Architecture Components

The model includes:

Embedding layer using pretrained Word2Vec vectors
One to three LSTM layers
Optional Bidirectional LSTM layers
Dropout layers
Dense fully connected layers
Sigmoid output layer for binary classification
Tuned Hyperparameters

The following parameters are optimized:

LSTM Parameters:

Number of LSTM layers: 1–3
Units per LSTM layer: 32–256
Bidirectional usage: True/False
Dropout rate per LSTM layer: 0.0–0.5

Dense Layers:

Number of dense layers: 1–3
Units per dense layer: 16–256
Dropout rate per dense layer: 0.0–0.5

Training Parameters:

Learning rate: 1e-4 to 1e-2
Optimizer: Adam or RMSprop

Random Search Configuration:

Maximum trials: 10
Objective: Validation accuracy

Early stopping is used to prevent overfitting.

7. Model Training

The best hyperparameter configuration is used to train the final model.

Training settings:

Batch size: 64
Epochs: 20
Early stopping patience: 5 epochs
Model Checkpointing

The best-performing model is saved automatically using:

best_tuned_model.h5

Training Visualization

Two plots are generated:

Accuracy vs Epochs
Loss vs Epochs

These plots help monitor:

Model learning progress
Overfitting
Training convergence

The training history figure is saved as:

tuned_model_training_history.png

Model Visualization

The final network architecture is visualized using VisualKeras.

This visualization shows:

LSTM layers
Dense layers
Model depth
Feature flow
8. Model Evaluation

The trained model is evaluated using the test dataset.

Evaluation Metrics

The following metrics are computed:

Test Accuracy
Test Loss
Precision
Recall
F1-score

A classification report is generated to summarize performance.

Confusion Matrix

A confusion matrix is created to visualize prediction performance.

It shows:

True positives
True negatives
False positives
False negatives

The confusion matrix is saved as:

tuned_model_confusion_matrix.png

Additional Metrics

Additional performance metrics calculated:

Precision
Recall
F1-score

These metrics provide deeper insight into model effectiveness.

Key Parameters

Random Seed:

42

Maximum Sequence Length:

150

Batch Size:

64

Epochs:

20

Word2Vec Vector Size:

100

Outputs Generated

The notebook produces the following outputs:

Sentiment distribution plot
Review length histogram
Preprocessed text examples
Trained Word2Vec model
Tuned RNN model architecture
Training history plots
Classification report
Confusion matrix
Saved trained model file
Best hyperparameters file

Generated files:

best_tuned_model.h5
best_hyperparameters.txt
tuned_model_training_history.png
tuned_model_confusion_matrix.png

How to Run This Notebook

Step 1: Install dependencies

pip install tensorflow keras keras-tuner gensim nltk kagglehub visualkeras scikit-learn matplotlib seaborn pandas numpy

Step 2: Download required NLTK resources.

The notebook automatically downloads:

punkt
stopwords
wordnet

Step 3: Run the notebook sequentially.

The dataset will be downloaded automatically during execution.