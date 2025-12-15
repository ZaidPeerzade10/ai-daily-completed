Here are the implementation steps to complete the task:

1.  **Generate Synthetic Dataset:** Create a synthetic regression dataset with 1000 samples, 5 informative features, and a small amount of noise using `sklearn.datasets.make_regression`. Subsequently, generate 5 completely random, uninformative features (e.g., using `np.random.rand`) and concatenate them horizontally with the informative features to form a single feature matrix `X` of 10 features. Store the target variable as `y`.

2.  **Construct ML Pipeline:** Define an `sklearn.pipeline.Pipeline` consisting of three sequential steps:
    *   `StandardScaler` for feature scaling.
    *   `SelectKBest` with `f_regression` as the score function for feature selection.
    *   `LinearRegression` as the final estimator.

3.  **Define Hyperparameter Grid for Feature Selection:** Create a dictionary for `param_grid` to be used with `GridSearchCV`. The parameter to tune will be `k` for `SelectKBest`, specified as `'selectkbest__k'`. Provide a list of integer values (e.g., `[3, 5, 7, 10]`) to explore different numbers of features to select.

4.  **Perform Hyperparameter Tuning with GridSearchCV:** Initialize `sklearn.model_selection.GridSearchCV` using the defined pipeline, the hyperparameter grid, `neg_mean_squared_error` as the scoring metric, and 3-fold cross-validation. Fit `GridSearchCV` to your generated feature matrix `X` and target `y`.

5.  **Report Best Results:** After `GridSearchCV` completes, retrieve and report the following:
    *   The best `k` value found for `SelectKBest` (from `best_params_`).
    *   The corresponding best cross-validation score (from `best_score_`), converting the `neg_mean_squared_error` to a positive Mean Squared Error for interpretability.
    *   The indices of the features selected by the `SelectKBest` step within the best estimator (access `best_estimator_.named_steps['selectkbest'].get_support(indices=True)`).