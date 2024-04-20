Project Overview

This repository contains the implementation of machine learning models developed for the CS57700 course homework assignment. The objective of this project is to build classifiers capable of detecting emotions from text, specifically analyzing tweets. This involves the development of a logistic regression classifier and a multi-layer neural network from scratch, using Python.

Repository Contents

main.py: Core script containing the implementation of the logistic regression and neural network models.
train.csv: Training dataset comprising 1200 labeled tweets across six emotional categories (joy, love, sadness, anger, fear, surprise).
test.csv: Unlabeled test dataset with 800 tweets for model evaluation.
test_lr.csv and test_nn.csv: Output files from the logistic regression and neural network models, respectively, containing the predicted emotions for the test dataset.
Features and Preprocessing

Text Preprocessing: Utilizes advanced techniques such as word embeddings and a bag-of-words model to convert text data into a format suitable for machine learning.
Feature Engineering: Implements custom feature extraction to enhance model performance, leveraging libraries like NLTK for text preprocessing while avoiding ML-specific libraries for model building.
Model Training and Evaluation

Cross-Validation: Employs cross-validation techniques to ensure robust model training and parameter tuning.



