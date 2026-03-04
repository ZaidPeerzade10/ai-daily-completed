Here are the implementation steps for the user premium conversion prediction task:

1.  **Generate Synthetic Data with Realistic Patterns:**
    *   Create three pandas DataFrames: `users_df` (500-700 rows, `user_id`, `signup_date`, `industry`, `company_size`, all initially 'Free'), `feature_usage_df` (5000-8000 rows, `usage_id`, `user_id`, `usage_date`, `feature_name`, `duration_minutes`), and `premium_conversions_df` (100-200 rows, `conversion_id`, `user_id`, `conversion_date`).
    *   Ensure all `usage_date` and `conversion_date` values are strictly *after* their respective `signup_date`.
    *   Simulate realistic patterns: identify a subset of users (e.g., 15-25%) who will convert. For these converting users, ensure their `feature_usage_df` entries generally show higher overall `duration_minutes` and `num_usage_events`, more frequent use of 'premium-adjacent' features (e.g., 'Report_Gen', 'Data_Export_Limited'), and their `conversion_date` occurs after a period of significant usage.
    *   Sort `feature_usage_df` first by `user_id`, then by `usage_date`.

2.  **Load into SQLite and Perform SQL Feature Engineering:**
    *   Establish an in-memory SQLite database connection using `sqlite3`.
    *   Load `users_df` into a table named `users` and `feature_usage_df` into a table named `feature_usage`.
    *   Determine a `global_analysis_date` by taking the maximum `usage_date` from `feature_usage_df` and adding 60 days. Calculate `feature_cutoff_date` as `global_analysis_date` minus 180 days.
    *   Write and execute a single SQL query that aggregates free-tier feature usage *before* the `feature_cutoff_date` for *each user*:
        *   `LEFT JOIN` the `users` and `feature_usage` tables.
        *   Include `user_id`, `signup_date`, `industry`, `company_size` directly from the `users` table.
        *   Aggregate the following features from `feature_usage` (considering only records where `usage_date` is before `feature_cutoff_date`): `total_usage_duration_pre_cutoff`, `num_usage_events_pre_cutoff`, `avg_duration_per_event_pre_cutoff`, `num_unique_features_used_pre_cutoff`, `count_report_gen_pre_cutoff`, `count_data_export_pre_cutoff`.
        *   Calculate `days_since_last_usage_pre_cutoff` as the number of days between `feature_cutoff_date` and the maximum `usage_date` for the user before the cutoff.
        *   Ensure the `LEFT JOIN` includes all users, returning `0` for counts/sums, `0.0` for averages, and `NULL` for `days_since_last_usage_pre_cutoff` if a user had no usage before the cutoff.

3.  **Pandas Feature Engineering and Binary Target Creation:**
    *   Fetch the results of the SQL query into a pandas DataFrame, named `user_conversion_features_df`.
    *   Handle `NaN` values: Fill `total_usage_duration_pre_cutoff`, `num_usage_events_pre_cutoff`, `count_report_gen_pre_cutoff`, `count_data_export_pre_cutoff`, and `num_unique_features_used_pre_cutoff` with `0`. Fill `avg_duration_per_event_pre_cutoff` with `0.0`. For `days_since_last_usage_pre_cutoff` (where `NULL` indicates no usage before cutoff), fill with `user_account_age_at_cutoff_days` + 30 days (or a large sentinel like 9999).
    *   Convert `signup_date` to datetime objects.
    *   Calculate new features: `user_account_age_at_cutoff_days` (days between `signup_date` and `feature_cutoff_date`), `usage_frequency_pre_cutoff` (`num_usage_events_pre_cutoff` divided by `user_account_age_at_cutoff_days + 1`), and `avg_daily_usage_duration_pre_cutoff` (`total_usage_duration_pre_cutoff` divided by `user_account_age_at_cutoff_days + 1`).
    *   Create the binary target column `converted_to_premium_in_next_90_days`. For each user, check `premium_conversions_df` to see if a `conversion_date` exists that falls *between* `feature_cutoff_date` and `feature_cutoff_date + timedelta(days=90)`. Set to `1` if converted, `0` otherwise.
    *   Define `X` (all engineered numerical and categorical features) and `y` (`converted_to_premium_in_next_90_days`). Split `X` and `y` into training and testing sets (e.g., 70/30) using `sklearn.model_selection.train_test_split`, ensuring `random_state=42` and `stratify` on `y`.

4.  **Data Visualization:**
    *   Create a violin plot (or box plot) using a suitable visualization library (e.g., Seaborn or Matplotlib) to compare the distribution of `total_usage_duration_pre_cutoff` for users who converted (`1`) versus those who did not (`0`).
    *   Generate a stacked bar chart showing the proportions of `converted_to_premium_in_next_90_days` (0 or 1) across different `industry` categories.
    *   Ensure both plots have clear titles, axis labels, and legends.

5.  **ML Pipeline, Training, and Evaluation:**
    *   Construct an `sklearn.pipeline.Pipeline` consisting of two main stages:
        *   A `sklearn.compose.ColumnTransformer` for preprocessing:
            *   For numerical features (e.g., `user_account_age_at_cutoff_days`, `total_usage_duration_pre_cutoff`, `num_usage_events_pre_cutoff`, `avg_duration_per_event_pre_cutoff`, `num_unique_features_used_pre_cutoff`, `count_report_gen_pre_cutoff`, `count_data_export_pre_cutoff`, `days_since_last_usage_pre_cutoff`, `usage_frequency_pre_cutoff`, `avg_daily_usage_duration_pre_cutoff`), apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
            *   For categorical features (`industry`, `company_size`), apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
        *   The final estimator in the pipeline should be `sklearn.ensemble.HistGradientBoostingClassifier(random_state=42)`.
    *   Train the complete pipeline using `X_train` and `y_train`.
    *   Predict probabilities for the positive class (class 1) on the `X_test` set.
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a detailed `sklearn.metrics.classification_report` using the test set predictions.