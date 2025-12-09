Here are the implementation steps to follow for this task:

1.  **Generate Synthetic Dataset and Prepare Features:**
    *   Generate a synthetic binary classification dataset using `sklearn.datasets.make_classification` with at least 1000 samples and 5 numerical features.
    *   Create a new feature representing a high-cardinality categorical variable. To do this, generate a numerical array of random integers (e.g., between 0 and 99) for the same number of samples, ensuring a significant number of unique values (e.g., 50-100). Convert these integers into unique string representations (e.g., using f-strings like "Category\_0", "Category\_1", etc.) to simulate a true categorical feature.
    *   Combine the numerical features, the newly created categorical feature, and the target variable into a single pandas DataFrame. Explicitly name your columns to clearly distinguish numerical from categorical features.

2.  **Split Data into Training and Testing Sets:**
    *   Separate the features (X) and the target variable (y) from your DataFrame.
    *   Split `X` and `y` into training and testing sets using `sklearn.model_selection.train_test_split`. Aim for a 70/30 split, ensure stratification on the target variable, and set a `random_state` for reproducibility.
    *   Identify and store the list of numerical feature names and the name of the high-cardinality categorical feature for use in the `ColumnTransformer`.

3.  **Define Pipeline for OneHot Encoding:**
    *   Create a `sklearn.compose.ColumnTransformer` named `preprocessor_onehot`.
        *   For all numerical features, apply `sklearn.preprocessing.StandardScaler`.
        *   For the high-cardinality categorical feature, apply `sklearn.preprocessing.OneHotEncoder`, explicitly setting `handle_unknown='ignore'` to handle categories unseen during training.
    *   Construct `pipeline_onehot_encoding` by chaining this `preprocessor_onehot` with a `sklearn.linear_model.LogisticRegression` model. Set `solver='liblinear'` and a `random_state` in the Logistic Regression model for consistent results.

4.  **Define Pipeline for Ordinal Encoding:**
    *   Create a second `sklearn.compose.ColumnTransformer` named `preprocessor_ordinal`.
        *   For all numerical features, apply `sklearn.preprocessing.StandardScaler`.
        *   For the high-cardinality categorical feature, apply `sklearn.preprocessing.OrdinalEncoder`, explicitly setting `handle_unknown='use_encoded_value'` and `unknown_value=-1` to manage unseen categories during evaluation.
    *   Construct `pipeline_ordinal_encoding` by chaining this `preprocessor_ordinal` with a `sklearn.linear_model.LogisticRegression` model. As before, set `solver='liblinear'` and a `random_state` for reproducibility.

5.  **Train Models and Evaluate Performance:**
    *   Fit both `pipeline_onehot_encoding` and `pipeline_ordinal_encoding` on your training data (`X_train`, `y_train`).
    *   Use each trained pipeline to make predictions on the test features (`X_test`).
    *   Calculate the `accuracy_score` and `f1_score` (from `sklearn.metrics`) for each pipeline's predictions against the true test labels (`y_test`).
    *   Clearly report the accuracy score and F1-score for `pipeline_onehot_encoding` (stating it used OneHot Encoding) and for `pipeline_ordinal_encoding` (stating it used Ordinal Encoding), allowing for a direct comparison of the encoding strategies.