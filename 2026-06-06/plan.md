Here are the implementation steps for developing the machine learning pipeline:

1.  **Generate Synthetic Data**:
    *   Create two pandas DataFrames: `loan_applicants_df` (1000-1500 rows) and `payment_history_df` (20000-30000 rows), populating them with the specified columns and data types.
    *   Ensure `application_date` in `loan_applicants_df` and `payment_date` in `payment_history_df` are appropriate datetime objects.
    *   Implement realistic data patterns: defaulting applicants (`_actual_default_status=1`, ~10-15% of total) should generally have lower `income`, lower `credit_score_at_application`, higher `loan_amount_requested`, and a higher frequency of `is_late_payment=1` events in their historical payment records *before* their `application_date`.
    *   Strictly ensure all `payment_date` entries for an `applicant_id` are earlier than the corresponding `application_date` for that applicant.
    *   Sort `payment_history_df` by `applicant_id` then `payment_date` for consistency.

2.  **Load Data into SQLite & Perform Time-Windowed SQL Feature Engineering**:
    *   Initialize an in-memory SQLite database connection using `sqlite3`.
    *   Load the `loan_applicants_df` into a table named `applicants` and `payment_history_df` into a table named `payments`.
    *   Construct and execute a single SQL query that leverages `LEFT JOIN` between `applicants` and `payments`. The query must:
        *   Select all static columns from the `applicants` table.
        *   For each applicant, aggregate historical payment data from the `payments` table *only for payments made strictly before the applicant's `application_date` and within the 12 months preceding it*.
        *   Calculate `num_late_payments_prev_12m_at_app`, `avg_payment_amount_prev_12m_at_app`, and `num_payments_prev_12m_at_app` using appropriate `COUNT` and `AVG` aggregations.
        *   Determine `days_since_last_payment_at_app` by finding the difference in days between `application_date` and the applicant's most recent `payment_date` prior to application.
        *   Extract `day_of_week_application` and `month_of_application` from `application_date`.
        *   Use `COALESCE` or similar functions to handle `NULL` values resulting from `LEFT JOIN` or aggregations (e.g., 0 for counts/averages, 9999 for `days_since_last_payment_at_app`).
    *   Fetch the results of this query into a new pandas DataFrame, `loan_features_df`.

3.  **Pandas Feature Engineering, Target Creation, and Data Splitting**:
    *   In `loan_features_df`, convert the `application_date` column to pandas datetime objects.
    *   Address any remaining `NaN` values: fill numerical historical aggregated features with 0.0 or 0, `days_since_last_payment_at_app` with 9999, and `income`/`credit_score_at_application` NaNs with their respective mean values.
    *   Calculate `debt_to_income_ratio` as `loan_amount_requested` divided by `income` (adding a small epsilon to income to prevent division by zero). Fill any `NaN` or `inf` values in this new column with 0.
    *   Define the feature set `X` (including numerical and one-hot encoded categorical features like `education`) and the target variable `y` by directly using the `_actual_default_status` column and renaming it to `is_default`.
    *   Split `X` and `y` into training and testing sets (e.g., 70/30 split) using `train_test_split`, ensuring `random_state` is set and the split is stratified on `y` to maintain class balance.

4.  **Exploratory Data Visualization**:
    *   Create a violin plot (or box plot) using `matplotlib` or `seaborn` to visualize the distribution of `credit_score_at_application` for applicants who did not default (0) versus those who did default (1). Ensure proper titles and axis labels.
    *   Generate a stacked bar chart showing the proportion of `is_default` (0 or 1) across different categories of `education`. Provide clear titles and labels.

5.  **Machine Learning Pipeline Construction, Training, and Prediction**:
    *   Construct an `sklearn.pipeline.Pipeline`.
    *   Inside the pipeline, use an `sklearn.compose.ColumnTransformer` for preprocessing:
        *   Apply `SimpleImputer` (strategy='mean') followed by `StandardScaler` to all numerical features.
        *   Apply `OneHotEncoder` (with `handle_unknown='ignore'`) to the categorical `education` feature.
    *   Append `HistGradientBoostingClassifier` as the final estimator in the pipeline, setting `random_state=42` and `class_weight='balanced'` to address class imbalance.
    *   Train the complete machine learning pipeline on the training data (`X_train`, `y_train`).
    *   Use the trained pipeline to predict probabilities for the positive class (default) on the `X_test` set.

6.  **Model Evaluation**:
    *   Calculate and print the `roc_auc_score` using the true labels (`y_test`) and the predicted probabilities for the positive class from the test set.
    *   Generate and print a `classification_report` for the test set, using the true labels (`y_test`) and the predicted class labels (obtained by applying a threshold to the predicted probabilities, or directly using `pipeline.predict(X_test)`).