# Breast Cancer Diagnosis using SVM and Naive Bayes

## Overview
This project focuses on building and comparing two machine learning models—Support Vector Machines (SVM) and Naive Bayes—for diagnosing breast cancer using the Breast Cancer Wisconsin Dataset. The goal is to predict whether a tumor is malignant or benign based on features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

## Dataset
The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains 569 samples with 30 features describing characteristics of cell nuclei. The target variable is binary:
- `0`: Benign
- `1`: Malignant

## Methodology
1. **Data Preprocessing**:
   - Dropped irrelevant columns (`id` and `Unnamed: 32`).
   - Encoded the target variable (`diagnosis`) into binary values.
   - Removed highly correlated features to avoid multicollinearity.
   - Normalized the dataset using MinMaxScaler.

2. **Model Training**:
   - **SVM**: Trained with a polynomial kernel, `C=1000`, `degree=1`, and `gamma=0.01`.
   - **Naive Bayes**: Trained with `alpha=2.0` and `fit_prior=False`.

3. **Evaluation**:
   - Split the dataset into training (60%), validation (20%), and testing (20%) sets.
   - Evaluated models using accuracy, confusion matrix, and classification report.

## Results
- **SVM**:
  - Test Accuracy: 97.37%
  - Confusion Matrix: High true positive and true negative rates with minimal misclassifications.

- **Naive Bayes**:
  - Test Accuracy: 82.46%
  - Confusion Matrix: 62 true negatives, 32 true positives, 14 false positives, and 6 false negatives.

## Conclusion
The SVM model outperformed the Naive Bayes model, achieving higher accuracy and better classification performance. However, Naive Bayes is computationally faster and may be suitable for scenarios where training time is a constraint.
