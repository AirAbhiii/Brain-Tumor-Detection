
# Brain Tumor Detection Using Convolutional Neural Networks

This project focuses on detecting brain tumors from MRI scans using deep learning techniques, particularly Convolutional Neural Networks (CNNs). The model is trained to classify MRI images into two categories: Tumor and No Tumor.

## Project Overview

Brain tumor detection from medical images is a crucial task in healthcare. This project automates the classification process using a CNN model built with TensorFlow and Keras. The model achieves high accuracy in distinguishing between tumor and non-tumor MRI scans.

## Tech Stack

Deep Learning  
TensorFlow  
Keras  
NumPy  
OpenCV  
Matplotlib  
Scikit-learn  
Python

## Dataset

The dataset used for training and validation consists of MRI images divided into two folders:

- yes (images showing presence of a tumor)  
- no (images without a tumor)

The dataset can be downloaded from open medical image repositories or Kaggle.

## Model Architecture

The model is a CNN consisting of:

- Two convolutional layers with ReLU activation  
- MaxPooling layers to reduce spatial dimensions  
- Flatten layer  
- Dense (fully connected) layer  
- Dropout layer to prevent overfitting  
- Final softmax layer for binary classification
Training
Loss Function: Categorical Crossentropy

Optimizer: Adam

Metrics: Accuracy

Epochs: 10 (can be adjusted)

Validation split: 20 
  
Results
The model achieves high accuracy when trained on well-labeled MRI datasets. Performance can be further improved with data augmentation and fine-tuning.



File Structure
.
├── brain_tumor_dataset/
├── braintumor.h5
├── model.py
├── predict.py
├── requirements.txt
└── README.md
