import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Set a random state for reproducibility across all steps
RANDOM_STATE = 42

# 1. Generate Synthetic Multi-class Classification Dataset
# At least 1000 samples, 10 features, 3 distinct classes
X, y = make_classification(
    n_samples=1200,        # Number of samples (at least 1000)
    n_features=10,         # Total number of features
    n_informative=5,       # Number of informative features
    n_redundant=2,         # Number of redundant features
    n_repeated=0,          # Number of duplicated features
    n_classes=3,           # Number of distinct classes
    n_clusters_per_class=1, # Number of clusters per class
    weights=[0.33, 0.33, 0.34], # Approximate equal weights for classes
    class_sep=0.8,         # How separated the classes are
    random_state=RANDOM_STATE # For reproducibility
)

print(f"Dataset generated: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes.")

# 2. Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

# 3. Train a RandomForestClassifier on the training data
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
print("\nTraining RandomForestClassifier...")
model.fit(X_train, y_train)
print("Training complete.")

# 4. Predict class labels on the test set
y_pred = model.predict(X_test)

# 5. Print a detailed classification report
print("\n--- Classification Report (Test Set) ---")
print(classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in sorted(np.unique(y))]))

# 6. Plot the confusion matrix for the test set predictions
print("Displaying Confusion Matrix...")
# Ensure appropriate figure size for better visualization
plt.figure(figsize=(8, 7))
disp = ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    display_labels=[f'Class {i}' for i in sorted(np.unique(y))], # Labels for the plot
    cmap='Blues',
    normalize='true' # Normalize by true labels to show proportions
)
disp.ax_.set_title("Normalized Confusion Matrix (Test Set)")
disp.ax_.set_xlabel("Predicted Label")
disp.ax_.set_ylabel("True Label")

# Adjust layout to prevent labels from overlapping
plt.tight_layout()
plt.show()
print("Script finished.")