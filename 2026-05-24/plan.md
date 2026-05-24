Here's a structured, numbered list of implementation steps for developing the churn prediction machine learning pipeline:

1.  **Generate Synthetic Datasets**:
    *   Create three pandas DataFrames: `customers_df`, `content_df`, and `viewing_history_df`, adhering to the specified row counts and column definitions.
    *   For `customers_df`, ensure `churn_date` is present for 15-20% of customers, is after `signup_date`, within the last 12 months, and `NaT` otherwise.
    *   For `viewing_history_df`, ensure `view_date` is always after the customer's `signup_date` and before their `churn_date` (if applicable).
    *   Implement realistic patterns: `Premium` plan users should exhibit higher `duration_minutes` and more frequent views. For churned customers, simulate a clear drop-off in `duration_minutes` and `num_views` in the 1-2 months preceding their `churn_date`.
    *   Finally, sort `viewing_history_df` by `customer_id` and then `view_date`.

2.  **Load Data into SQLite & SQL Feature Engineering**:
    *   Initialize an in-memory SQLite database connection.
    *   Load the `customers_df`, `content_df`, and `viewing_history_df` into corresponding tables named `customers`, `content`, and `viewing_history`.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 1 month prior to the latest `view_date` found in `viewing_history_df`.
    *   Construct a single SQL query to extract features for *each customer*. This query should perform a `LEFT JOIN` from `customers` to aggregated `viewing_history` data.
    *   For the 30-day window immediately *preceding* `GLOBAL_PREDICTION_CUTOFF_DATE`, calculate: `total_view_duration_prev_30d`, `num_views_prev_30d`, `num_unique_content_prev_30d`, `num_unique_genres_prev_30d` (joining with `content` for `genre`).
    *   Calculate `days_since_last_view_at_cutoff`, returning 9999 if no views exist before or on the cutoff date.
    *   Include static customer attributes: `customer_id`, `signup_date`, `subscription_plan`, `region`, `age_group`.
    *   Ensure proper handling of `NULL` values (e.g., using `COALESCE`) for customers with no viewing activity in the window, returning 0 for counts/sums and 9999 for `days_since_last_view_at_cutoff`.

3.  **Pandas Feature Engineering & Binary Target Creation**:
    *   Fetch the results of the SQL query into a pandas DataFrame (`customer_features_df`).
    *   Convert date columns (`signup_date`, `current_cutoff_date`) to datetime objects.
    *   Handle `NaN` values: fill numerical aggregated features with 0 or 0.0, and `days_since_last_view_at_cutoff` with 9999.
    *   Calculate `customer_tenure_at_cutoff_days` as the difference in days between `current_cutoff_date` and `signup_date`.
    *   Calculate `avg_view_duration_per_view_prev_30d` and handle potential `NaN` or `inf` results (e.g., from division by zero) by filling with 0.
    *   Merge `churn_date` from the original `customers_df` into `customer_features_df`.
    *   Create the binary target variable `will_churn_in_next_30_days`: Assign 1 if the customer's `churn_date` falls within the 30-day window *immediately following* their `current_cutoff_date`, and 0 otherwise (including for customers who never churned or churned outside this specific window).
    *   Define feature sets `X` (numerical: aggregated viewing stats, tenure; categorical: `subscription_plan`, `region`, `age_group`) and target `y`.
    *   Split `X` and `y` into training and testing sets (e.g., 70/30) using `train_test_split`, ensuring `random_state=42` and `stratify=y` to maintain class distribution.

4.  **Exploratory Data Analysis & Visualization**:
    *   Generate a violin plot (or box plot) to visually compare the distribution of `total_view_duration_prev_30d` between customers who `will_churn_in_next_30_days` (target=1) and those who will not (target=0). Ensure clear labels and a title.
    *   Create a stacked bar chart to visualize the proportion of churners (1) vs. non-churners (0) within each `subscription_plan` category. Add appropriate labels and a title.

5.  **Machine Learning Pipeline Development & Training**:
    *   Construct an `sklearn.pipeline.Pipeline`.
    *   Include a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   Apply `SimpleImputer` (strategy='mean') and `StandardScaler` to numerical features.
        *   Apply `OneHotEncoder` (handle_unknown='ignore') to categorical features.
    *   Set the final estimator of the pipeline to `sklearn.ensemble.HistGradientBoostingClassifier`, configuring it with `random_state=42` and `class_weight='balanced'` to address potential class imbalance.
    *   Train the entire pipeline using `X_train` and `y_train`.

6.  **Model Evaluation**:
    *   Use the trained pipeline to predict probabilities for the positive class (churn, class 1) on the `X_test` dataset.
    *   Calculate and print the `roc_auc_score` for the test set predictions against `y_test`.
    *   Generate and print a `classification_report` to provide a comprehensive view of precision, recall, f1-score, and support for both classes on the test set.