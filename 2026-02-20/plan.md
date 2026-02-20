Here are the implementation steps for the task:

1.  **Generate Synthetic Data with Fraud Patterns:**
    *   Create the `users_df` DataFrame (500-700 rows) with `user_id`, `signup_date`, `region`, `age`, and `is_fraudulent`. Ensure `is_fraudulent` has an approximate 5-10% fraud rate.
    *   Create the `transactions_df` DataFrame (5000-8000 rows) with `transaction_id`, `user_id` (sampled from `users_df`), `transaction_date`, `amount`, `merchant_category`, `location_country`.
    *   Implement realistic fraud patterns:
        *   For fraudulent users, generate transactions with higher average `amount`s or a mix of very large and normal `amount`s.
        *   Simulate bursts of activity by generating more frequent transactions for fraudulent users within shorter periods.
        *   Introduce transactions from multiple distinct `location_country`s for fraudulent users within a short timeframe.
        *   Ensure `transaction_date` is always after the respective `signup_date`.
        *   For fraudulent users, concentrate transaction activity closer to their `signup_date` and have it potentially stop abruptly, contrasting with more prolonged activity for non-fraudulent users.
    *   Finally, sort `transactions_df` by `user_id` then `transaction_date`.

2.  **Load Data into SQLite and Perform SQL Feature Engineering:**
    *   Initialize an in-memory SQLite database connection using `sqlite3`.
    *   Load `users_df` into a SQL table named `users` and `transactions_df` into a table named `transactions`.
    *   Determine `global_analysis_date` (e.g., `max(transaction_date)` from `transactions_df` + 60 days) and `feature_cutoff_date` (`global_analysis_date` - 90 days) using pandas on the `transactions_df`.
    *   Write and execute a single SQL query that performs the following for *each user*, aggregating transaction behavior *before* the `feature_cutoff_date`:
        *   `LEFT JOIN` `users` and `transactions` tables to ensure all users are included.
        *   Filter transactions to only include those with `transaction_date` earlier than `feature_cutoff_date`.
        *   Calculate `total_spend_pre_cutoff`, `num_transactions_pre_cutoff`, `avg_transaction_value_pre_cutoff`, `max_transaction_value_pre_cutoff`, `num_unique_merchant_categories_pre_cutoff`, and `num_unique_location_countries_pre_cutoff`.
        *   Calculate `days_since_last_transaction_pre_cutoff` (difference in days between `feature_cutoff_date` and `MAX(transaction_date)` before cutoff).
        *   Calculate `transaction_span_days_pre_cutoff` (difference in days between `MIN(transaction_date)` and `MAX(transaction_date)` before cutoff).
        *   Include static user attributes: `user_id`, `age`, `region`, `signup_date`, `is_fraudulent`.
        *   Handle `NULL` values for users with no transactions before cutoff: `0` for sums/counts/averages/max, and `NULL` for date difference features.

3.  **Pandas Feature Engineering and Data Preparation:**
    *   Fetch the results of the SQL query into a pandas DataFrame, `user_fraud_features_df`.
    *   Handle `NaN` values resulting from the SQL query:
        *   Fill `total_spend_pre_cutoff`, `num_transactions_pre_cutoff`, `avg_transaction_value_pre_cutoff`, `max_transaction_value_pre_cutoff`, `num_unique_merchant_categories_pre_cutoff`, `num_unique_location_countries_pre_cutoff`, `transaction_span_days_pre_cutoff` with `0`.
        *   Fill `days_since_last_transaction_pre_cutoff` with a large sentinel value (e.g., `9999`).
    *   Convert `signup_date` to datetime objects.
    *   Calculate `account_age_at_cutoff_days`: number of days between `signup_date` and `feature_cutoff_date`.
    *   Calculate `transaction_frequency_pre_cutoff`: `num_transactions_pre_cutoff` / (`account_age_at_cutoff_days` + 1).
    *   Calculate `avg_transaction_per_span_pre_cutoff`: `num_transactions_pre_cutoff` / (`transaction_span_days_pre_cutoff` + 1) for users with `transaction_span_days_pre_cutoff` > 0, else 0.
    *   Define the feature matrix `X` (all engineered numerical and categorical features) and the target vector `y` (`is_fraudulent`).
    *   Split `X` and `y` into training and testing sets (e.g., 70/30) using `train_test_split`, ensuring `random_state=42` and `stratify=y` for balanced class distribution.

4.  **Data Visualization for Fraud Pattern Insights:**
    *   Create a violin plot (or box plot) to visualize the distribution of `total_spend_pre_cutoff` for users where `is_fraudulent=0` versus `is_fraudulent=1`. Include clear labels and a title.
    *   Generate a stacked bar chart showing the proportion of `is_fraudulent` (0 or 1) across different `region` values. Ensure appropriate axis labels and a title.

5.  **Build, Train, and Evaluate ML Pipeline:**
    *   Construct an `sklearn.pipeline.Pipeline`.
    *   Within the pipeline, define a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For all numerical features (including `age`, `account_age_at_cutoff_days`, `total_spend_pre_cutoff`, `num_transactions_pre_cutoff`, `avg_transaction_value_pre_cutoff`, `max_transaction_value_pre_cutoff`, `num_unique_merchant_categories_pre_cutoff`, `num_unique_location_countries_pre_cutoff`, `days_since_last_transaction_pre_cutoff`, `transaction_span_days_pre_cutoff`, `transaction_frequency_pre_cutoff`, `avg_transaction_per_span_pre_cutoff`): apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For the categorical feature `region`: apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   Set the final estimator in the pipeline to `sklearn.ensemble.HistGradientBoostingClassifier(random_state=42)`.
    *   Train the complete pipeline using the `X_train` and `y_train` sets.
    *   Predict probabilities for the positive class (fraudulent, class 1) on the `X_test` set.
    *   Calculate and print the `roc_auc_score` and a detailed `classification_report` for the test set predictions against `y_test`.