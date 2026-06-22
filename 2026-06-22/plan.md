Here are the steps to develop the machine learning pipeline:

1.  **Synthetic Data Generation**:
    *   Generate three Pandas DataFrames: `users_df` (1000-1500 rows with `user_id`, `signup_date`, `device_type`, `browser`), `pages_df` (50-100 rows with `page_id`, `page_name`, `page_type`), and `sessions_df` (20000-30000 rows with `session_id`, `user_id`, `start_datetime`, `landing_page_id`, `session_duration_seconds`, `had_conversion`).
    *   Ensure data consistency: `start_datetime` for each session must be strictly after the corresponding `user_id`'s `signup_date`.
    *   Simulate realistic patterns: Introduce a positive correlation between `session_duration_seconds` and `had_conversion`. `had_conversion` should be true for approximately 10-15% of sessions.
    *   Sort `sessions_df` by `user_id` then `start_datetime` in ascending order to facilitate time-series aggregations.

2.  **SQLite Database & SQL Feature Engineering**:
    *   Initialize an in-memory SQLite database connection using `sqlite3`.
    *   Load `users_df`, `pages_df`, and `sessions_df` into database tables named `users`, `pages`, and `sessions` respectively.
    *   Determine `GLOBAL_PREDICTION_CUTOFF_DATE` as 7 days prior to the latest `start_datetime` in the `sessions` table.
    *   Construct a single SQL query to extract features for all sessions starting *after* the `GLOBAL_PREDICTION_CUTOFF_DATE`. For each of these sessions, the query should:
        *   Include current session details (`session_id`, `start_datetime`, `session_duration_seconds`, `had_conversion`, `user_id`).
        *   Join with `users` and `pages` tables to include static user details (`signup_date`, `device_type`, `browser`) and landing page details (`page_name`, `page_type`).
        *   Calculate time-windowed historical aggregates for the *respective `user_id` based on sessions occurring 30 days prior to or on `GLOBAL_PREDICTION_CUTOFF_DATE`*. These aggregates are:
            *   `avg_session_duration_prev_30d`: Average `session_duration_seconds`.
            *   `num_sessions_prev_30d`: Count of sessions.
            *   `conversion_rate_prev_30d`: Average of `had_conversion` (treated as 0 or 1).
            *   `days_since_last_session_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `start_datetime` for that `user_id` *before or on* the cutoff date. Return 9999 if no prior sessions are found within the specified historical window.
        *   Handle `NULL` values for historical aggregates for users with no prior activity by defaulting `avg_session_duration_prev_30d` and `conversion_rate_prev_30d` to 0.0, `num_sessions_prev_30d` to 0, and `days_since_last_session_at_cutoff` to 9999. Use `julianday()` for date comparisons in SQL.

3.  **Pandas Feature Engineering & Binary Target Creation**:
    *   Fetch the results of the SQL query into a Pandas DataFrame, `session_features_df`.
    *   Convert `signup_date` and `start_datetime` columns to datetime objects.
    *   Fill `NaN` values resulting from the SQL query's historical aggregations: 0.0 for `avg_session_duration_prev_30d` and `conversion_rate_prev_30d`, 0 for `num_sessions_prev_30d`, and 9999 for `days_since_last_session_at_cutoff`.
    *   Calculate a new feature: `user_tenure_at_session_start_days`, which is the difference in days between `start_datetime` and `signup_date` for each session.
    *   Create the binary target variable `is_bounce`: Assign 1 if `session_duration_seconds` is less than or equal to the 20th percentile of all `session_duration_seconds` in the `session_features_df` AND `had_conversion` is False for that session; otherwise, assign 0.
    *   After `is_bounce` creation, drop `session_duration_seconds` and `had_conversion` columns from `session_features_df` to prevent target leakage.
    *   Define the feature set `X` (including numerical features like historical aggregates and tenure, and categorical features like `device_type`, `browser`, `page_name`, `page_type`) and the target `y` (`is_bounce`).
    *   Split `X` and `y` into training (70%) and testing (30%) sets using `sklearn.model_selection.train_test_split`, ensuring `random_state=42` and `stratify=y` for class balance.

4.  **Data Visualization**:
    *   Using Matplotlib or Seaborn, create two visualizations from the `session_features_df` (or `X_train`/`y_train` if preferred for smaller sample):
        *   A violin plot (or box plot) showing the distribution of `user_tenure_at_session_start_days` for sessions where `is_bounce` is 0 ('Not Bounce') versus 1 ('Bounce'). Ensure clear labels and a descriptive title.
        *   A stacked bar chart illustrating the proportions of 'Not Bounce' (0) and 'Bounce' (1) sessions across different `device_type` categories. Provide appropriate labels, title, and legend.

5.  **ML Pipeline Construction & Evaluation**:
    *   Construct an `sklearn.pipeline.Pipeline` that incorporates a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features, apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features, apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   Append `sklearn.ensemble.HistGradientBoostingClassifier` as the final estimator in the pipeline. Set `random_state=42` and `class_weight='balanced'` to handle potential class imbalance in the target.
    *   Train the complete pipeline using `X_train` and `y_train`.
    *   Predict the probabilities of the positive class (class 1, i.e., 'Bounce') on the `X_test` set.
    *   Calculate and print the `sklearn.metrics.roc_auc_score` and a comprehensive `sklearn.metrics.classification_report` using the test set predictions and true labels (`y_test`).