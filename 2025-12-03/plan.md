Here are the implementation steps for the task:

1.  **Generate Synthetic Dataset:** Use `sklearn.datasets.make_regression` to create a synthetic regression dataset. Ensure it has at least 500 samples and 5 features. Store the features (X) and the target variable (y).

2.  **Construct ML Pipeline:** Create an `sklearn.pipeline.Pipeline` instance. Define two sequential steps within the pipeline:
    *   The first step should be an `sklearn.preprocessing.StandardScaler` for feature scaling. Assign it a descriptive name (e.g., 'scaler').
    *   The second step should be an `sklearn.linear_model.Ridge` regressor. Assign it a descriptive name (e.g., 'ridge').

3.  **Define Hyperparameter Grid:** Create a dictionary to specify the hyperparameter grid for `GridSearchCV`. For the `Ridge` regressor within your pipeline, define a list of at least three distinct `alpha` values (e.g., `[0.1, 1.0, 10.0]`) to tune. Remember to prefix the parameter name with the name you assigned to the `Ridge` step in your pipeline (e.g., `'ridge__alpha'`).

4.  **Execute Grid Search Cross-Validation:** Initialize `sklearn.model_selection.GridSearchCV`. Pass the pipeline created in step 2, the hyperparameter grid from step 3, `scoring='neg_mean_squared_error'` as the evaluation metric, and `cv=3` for 3-fold cross-validation. Then, fit the `GridSearchCV` object to your generated features (X) and target (y) data.

5.  **Report Best Results:** After the `GridSearchCV` has finished fitting, extract and report the best hyperparameters found using the `best_params_` attribute. Also, retrieve the corresponding best score using the `best_score_` attribute. Since `neg_mean_squared_error` is used, multiply this best score by -1 to convert it into a positive Mean Squared Error (MSE) before reporting the final result.