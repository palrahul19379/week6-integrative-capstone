# Week 6 — Integrative Capstone Project and Evaluation

## Wine Classification and Unsupervised Segmentation

This project implements a complete Data Science pipeline in Python using the publicly available Wine Recognition dataset. It combines data acquisition, data cleaning, exploratory analysis, supervised machine learning, unsupervised learning, evaluation, interpretation, and recommendations.

### Project Objectives
- Acquire and inspect a public dataset.
- Check data quality and perform preprocessing.
- Explore class distribution and feature characteristics.
- Compare multiple supervised classification models.
- Evaluate models using accuracy, precision, recall, F1-score, and cross-validation.
- Apply K-Means clustering as an unsupervised analysis.
- Select the number of clusters using silhouette score.
- Visualize clusters with PCA.
- Interpret feature importance and critically discuss limitations.

### Models
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (RBF)

### Actual Results
**Best supervised model:** Random Forest

- Test Accuracy: **1.0000**
- Weighted Precision: **1.0000**
- Weighted Recall: **1.0000**
- Weighted F1: **1.0000**
- 5-Fold CV Mean Accuracy: **0.9665**

### Unsupervised Results
- Best K-Means k: **3**
- Best silhouette score: **0.2849**

### Files
- `Week6_Integrative_Capstone_Wine_Classification_Clustering.ipynb` — complete executed notebook
- `week6_capstone.py` — Python implementation
- `requirements.txt` — required packages
- `assets/` — project visualizations

### Dataset
The dataset is accessed reproducibly using `sklearn.datasets.load_wine()`. It contains 178 observations, 13 numeric features, and 3 classes.

### Important Note
This is an educational Data Science project. The results should not be treated as evidence for real-world decisions without independent validation, domain expertise, and appropriate governance.
