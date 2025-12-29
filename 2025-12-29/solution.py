import numpy as np
from scipy.stats import loguniform

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer, classification_report

# Set a random_state for reproducibility across all steps
GLOBAL_RANDOM_STATE = 42

# 1. Generate a synthetic binary classification dataset
#    - At least 1000 samples
#    - 5 informative features
#    - Significant class imbalance (90% majority, 10% minority)
X, y = make_classification(
    n_samples=1500,  # More than 1000 samples
    n_features=10,   # Total features
    n_informative=5, # 5 informative features
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    weights=[0.9, 0.1], # 90% majority (class 0), 10% minority (class 1)
    flip_y=0, # No label flipping for clear class distinction
    random_state=GLOBAL_RANDOM_STATE
)

# 2. Split the dataset into training and testing sets (e.g., 70/30 split)
#    - Stratified to maintain class imbalance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=GLOBAL_RANDOM_STATE
)

# 3. Define a custom scoring function using sklearn.metrics.make_scorer
#    - Prioritizes the F1-score for the *minority class* (class 1)
#    - `average='binary'` and `pos_label=1` are crucial for this.
minority_f1_scorer = make_scorer(f1_score, average='binary', pos_label=1)

# 4. Construct an sklearn.pipeline.Pipeline
#    - First applies StandardScaler
#    - Then fits a LogisticRegression model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logisticregression', LogisticRegression(random_state=GLOBAL_RANDOM_STATE, solver='liblinear'))
])

# 5. Define a hyperparameter distribution for RandomizedSearchCV
#    - Tune LogisticRegression's C parameter using scipy.stats.loguniform
param_distributions = {
    'logisticregression__C': loguniform(1e-3, 1e2) # C from 0.001 to 100 on a log scale
}

# 6. Perform RandomizedSearchCV
#    - With the pipeline, defined parameter distributions
#    - 3-fold cross-validation (`cv=3`)
#    - The custom minority class F1-scorer
#    - Explore at least 10 different parameter settings (`n_iter=10`)
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=10, # Number of parameter settings that are sampled
    cv=3,      # 3-fold cross-validation
    scoring=minority_f1_scorer, # Our custom F1-scorer for the minority class
    random_state=GLOBAL_RANDOM_STATE, # For reproducibility of the search itself
    n_jobs=-1  # Use all available CPU cores
)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)

# 7. Report the best C value found, the corresponding best cross-validation score,
#    and print a full classification_report for the test set.

print("--- Randomized Search Results ---")
best_c_value = random_search.best_params_['logisticregression__C']
best_cv_score = random_search.best_score_
print(f"Best C value found: {best_c_value:.4f}")
print(f"Best cross-validation minority F1-score: {best_cv_score:.4f}")

# Get the best estimator from the search
best_estimator = random_search.best_estimator_

# Make predictions on the test set using the best estimator
y_pred = best_estimator.predict(X_test)

print("\n--- Test Set Classification Report (Best Estimator) ---")
print(classification_report(y_test, y_pred))