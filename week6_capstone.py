"""
Week 6 Integrative Capstone Project
Wine Classification + K-Means Segmentation

Educational implementation of a complete Data Science pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Data acquisition
wine = load_wine(as_frame=True)
df = wine.frame.copy()
df["target_name"] = df["target"].map(dict(enumerate(wine.target_names)))

# 2. Cleaning
df = df.drop_duplicates()
X = df[wine.feature_names]
y = df["target"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Supervised models
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42))
    ]),
    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ])
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="weighted"),
        "Recall": recall_score(y_test, pred, average="weighted"),
        "F1": f1_score(y_test, pred, average="weighted"),
        "CV Mean": cross_val_score(model, X, y, cv=5).mean()
    })

results = pd.DataFrame(rows).sort_values("F1", ascending=False)
print(results.round(4))

# 5. Unsupervised learning
X_scaled = StandardScaler().fit_transform(X)
scores = {}
for k in range(2, 6):
    labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

best_k = max(scores, key=scores.get)
labels = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit_predict(X_scaled)

print("\nSilhouette scores:", {k: round(v, 4) for k, v in scores.items()})
print("Selected k:", best_k)

# 6. PCA visualization coordinates
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("PCA explained variance:", np.round(pca.explained_variance_ratio_, 4))
