import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Set a random seed for reproducibility
RANDOM_STATE = 42

# --- 1. Data Acquisition and Preparation ---
# Generate a synthetic binary classification dataset
# 1000 samples, 10 features, 2 informative features, 2 classes
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    weights=[0.5, 0.5], # Ensure balanced classes
    flip_y=0.01, # Small amount of label noise
    random_state=RANDOM_STATE
)

print("--- Dataset Information ---")
print(f"Generated {X.shape[0]} samples with {X.shape[1]} features.")
print(f"Class distribution (0/1): {np.bincount(y)}")

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples\n")


# --- 2. Model Instantiation and Training ---
# Initialize a LogisticRegression model
# Using 'liblinear' solver which is good for small datasets and binary classification
model = LogisticRegression(random_state=RANDOM_STATE, solver='liblinear')

# Train the model on the training data
print("Training Logistic Regression model...")
model.fit(X_train, y_train)
print("Model training complete.\n")


# --- 3. Prediction on Test Data ---
# Predict class labels on the test set
y_pred = model.predict(X_test)

# Predict class probabilities for the positive class (class 1) on the test set
# model.predict_proba returns probabilities for [class 0, class 1], we need class 1
y_proba = model.predict_proba(X_test)[:, 1]


# --- 4. Performance Metric Calculation and Display ---
print("--- Model Evaluation Metrics (Test Set) ---")

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:  {accuracy:.4f}")

# Calculate Precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")

# Calculate Recall
recall = recall_score(y_test, y_pred)
print(f"Recall:    {recall:.4f}")

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print(f"F1-score:  {f1:.4f}")

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC:   {roc_auc:.4f}\n")


# --- 5. ROC Curve Visualization ---
# Compute False Positive Rate (fpr) and True Positive Rate (tpr) for ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr, tpr, color='darkorange', lw=2,
    label=f'ROC curve (AUC = {roc_auc:.2f})'
)
# Plot the diagonal line representing a random classifier
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

print("--- ROC Curve Plot Displayed ---")