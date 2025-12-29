Here are the implementation steps for the task:

1.  **Generate and Split Dataset:**
    *   Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with at least 1000 samples, 5 informative features, a significant class imbalance (e.g., `weights=[0.9, 0.1]` for 90% majority, 10% minority), and a fixed `random_state` for reproducibility.
    *   Split the generated dataset into training and testing sets (e.g., 70% training, 30% testing) using `sklearn.model_selection.train_test_split`. Ensure the split is stratified to maintain the class imbalance in both sets and use a fixed `random_state`.

2.  **Define Custom Minority Class F1 Scorer:**
    *   Import `f1_score` from `sklearn.metrics` and `make_scorer` from `sklearn.metrics`.
    *   Create a custom scoring function using `make_scorer` that calculates the F1-score specifically for the *minority class* (class 1). Explicitly set `average='binary'` and `pos_label=1` to correctly focus on the performance of the positive (minority) class.

3.  **Construct Machine Learning Pipeline:**
    *   Import `Pipeline` from `sklearn.pipeline`, `StandardScaler` from `sklearn.preprocessing`, and `LogisticRegression` from `sklearn.linear_model`.
    *   Define an `sklearn.pipeline.Pipeline` consisting of two sequential steps:
        1.  A `StandardScaler` for standardizing features.
        2.  A `LogisticRegression` model, with a specified `random_state` and `solver='liblinear'` for consistency.

4.  **Configure and Execute Randomized Search:**
    *   Import `RandomizedSearchCV` from `sklearn.model_selection` and `loguniform` from `scipy.stats`.
    *   Define a dictionary for the hyperparameter distribution to tune the `C` parameter of the `LogisticRegression` model within the pipeline. Use `scipy.stats.loguniform` (e.g., spanning `1e-3` to `1e2`). Remember to prefix the parameter name with the estimator's name in the pipeline (e.g., `'logisticregression__C'`).
    *   Instantiate `RandomizedSearchCV` using the defined pipeline, the hyperparameter distribution, the custom minority class F1-scorer, 3-fold cross-validation (`cv=3`), and a specified number of different parameter settings to sample (e.g., `n_iter=10`).
    *   Fit `RandomizedSearchCV` to the training data.

5.  **Report Best Parameters and Evaluate on Test Set:**
    *   Retrieve and print the best hyperparameter `C` value found by `RandomizedSearchCV` (accessible via `best_params_`).
    *   Retrieve and print the corresponding best cross-validation score achieved (accessible via `best_score_`).
    *   Obtain the best estimator found by `RandomizedSearchCV` (accessible via `best_estimator_`).
    *   Use this best estimator to make predictions on the unseen test set.
    *   Generate and print a full `classification_report` from `sklearn.metrics` comparing the test set true labels with the best estimator's predictions, providing a comprehensive evaluation of performance across both classes.