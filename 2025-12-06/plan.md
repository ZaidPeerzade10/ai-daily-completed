Here are the implementation steps for the task:

1.  **Generate the Synthetic Dataset:** Create a synthetic regression dataset using `sklearn.datasets.make_regression` with at least 500 samples, 3 informative features, and a small amount of noise. Assign the features (X) and target (y) to variables.
2.  **Construct ML Pipelines:**
    *   Define `pipeline_simple` as an `sklearn.pipeline.Pipeline` consisting of `StandardScaler` followed by `LinearRegression`.
    *   Define `pipeline_poly` as an `sklearn.pipeline.Pipeline` consisting of `PolynomialFeatures` with `degree=2`, then `StandardScaler`, then `LinearRegression`.
3.  **Evaluate Pipelines with Cross-Validation:**
    *   For `pipeline_simple`, use `sklearn.model_selection.cross_val_score` to evaluate it with 5-fold cross-validation (`cv=5`) and `neg_mean_squared_error` as the scoring metric. Store the resulting scores.
    *   Repeat the cross-validation evaluation for `pipeline_poly`, storing its scores separately.
4.  **Process and Print Results:**
    *   For both sets of scores obtained from cross-validation, convert the negative mean squared error values to positive mean squared error values (by multiplying by -1).
    *   Calculate the mean and standard deviation of these positive MSE scores for `pipeline_simple`.
    *   Calculate the mean and standard deviation of these positive MSE scores for `pipeline_poly`.
    *   Print the calculated mean and standard deviation of the MSE for `pipeline_simple` and `pipeline_poly`, clearly labeling which results correspond to which pipeline.