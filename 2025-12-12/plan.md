Here are the steps to follow for the task:

1.  **Generate Synthetic Dataset and Initial Preparation**:
    Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification`. Ensure at least 1000 samples, exactly 4 numerical features, and 2 classes, setting a `random_state` for reproducibility. Convert the generated features and target into a pandas DataFrame, assigning clear column names like `feature_0` through `feature_3` for features and `target` for the label.

2.  **Feature Discretization**:
    Select two of the original numerical features (e.g., `feature_0` and `feature_1`). Apply `pandas.cut` to each of these selected features to discretize them into 3 to 4 bins, assigning meaningful categorical labels (e.g., 'low', 'medium', 'high'). Add these two newly created binned categorical features to the DataFrame, giving them distinct names (e.g., `binned_feature_0`, `binned_feature_1`).

3.  **Define Preprocessing with ColumnTransformer**:
    Identify the *remaining original* numerical features (the two features that were *not* binned). Also, identify the *newly created* binned categorical features. Construct an `sklearn.compose.make_column_transformer` (or `ColumnTransformer`) with the following transformations:
    *   Apply `sklearn.preprocessing.StandardScaler` to the identified remaining original numerical features.
    *   Apply `sklearn.preprocessing.OneHotEncoder` with `handle_unknown='ignore'` to the identified newly created binned categorical features.

4.  **Construct the Machine Learning Pipeline**:
    Create an `sklearn.pipeline.Pipeline` that sequentially combines the preprocessing steps and the classification model. The first step in the pipeline should be the `ColumnTransformer` defined in the previous step. The second step should be an `sklearn.ensemble.GradientBoostingClassifier`, initialized with a `random_state` for reproducibility.

5.  **Evaluate Pipeline Performance using Cross-Validation**:
    Use `sklearn.model_selection.cross_val_score` to evaluate the complete pipeline. Perform 5-fold cross-validation on the prepared features (excluding the original target column) and the target label. Specify `accuracy` as the scoring metric. Finally, calculate and report the mean accuracy and the standard deviation of the accuracy scores obtained from the cross-validation.