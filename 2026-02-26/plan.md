Here are the implementation steps for developing the machine learning pipeline to predict long-term user retention:

1.  **Generate Synthetic Dataset**:
    *   Create three pandas DataFrames: `users_df`, `onboarding_events_df`, and `future_activity_df`.
    *   `users_df` should contain `user_id`, `signup_date`, `assigned_onboarding_variant` ('Control', 'Variant_A', 'Variant_B'), and `referral_source` (e.g., 'Organic', 'Paid_Ad', 'Referral').
    *   `onboarding_events_df` should include `event_id`, `user_id`, `event_date` (within 7 days of `signup_date`), `event_type` (e.g., 'step_1_completed', 'profile_filled', 'tutorial_viewed', 'payment_info_added'), and `duration_seconds`.
    *   `future_activity_df` should have `activity_id`, `user_id`, `activity_date` (between 30 and 90 days after `signup_date`), and `activity_type`.
    *   Ensure all date columns are `datetime` objects. Simulate realistic patterns where `assigned_onboarding_variant` influences onboarding completion, and users completing critical steps are more likely to appear in `future_activity_df`.

2.  **Load Data into SQLite and SQL Feature Engineering**:
    *   Initialize an in-memory SQLite database connection using `sqlite3`.
    *   Load `users_df` into a table named `users` and `onboarding_events_df` into `onboarding_events`.
    *   Execute a single SQL query to aggregate onboarding behavior for *each user* within the first 7 days of their `signup_date`. The query must:
        *   `LEFT JOIN` `users` with aggregated `onboarding_events`.
        *   Calculate `num_onboarding_events`, `total_onboarding_duration`, `avg_step_duration`, `num_critical_steps_completed` (for 'profile_filled' or 'payment_info_added'), `days_to_first_onboarding_event` (from `signup_date` to `MIN(event_date)`), and `onboarding_completion_rate` (`num_critical_steps_completed` / 2.0).
        *   Return `user_id`, `signup_date`, `assigned_onboarding_variant`, `referral_source`, and all aggregated features.
        *   Properly handle users with no onboarding events within the window, ensuring they are included with 0s for counts/sums, 0.0s for averages/rates, and `NULL` for `days_to_first_onboarding_event`. Use `strftime('%J', ...)` for day differences.

3.  **Pandas Post-SQL Processing, Target Creation, and Data Splitting**:
    *   Fetch the results of the SQL query into a pandas DataFrame (`user_onboarding_features_df`).
    *   Handle `NaN` values resulting from the SQL aggregation: fill count/sum features (`num_onboarding_events`, `total_onboarding_duration`, `num_critical_steps_completed`) with 0; fill rate/average features (`avg_step_duration`, `onboarding_completion_rate`) with 0.0; fill `days_to_first_onboarding_event` with a large sentinel value (e.g., 9999).
    *   Ensure `signup_date` is a datetime object.
    *   Create the binary target `is_retained_90_days`: For each `user_id`, set to 1 if there is *any* record in `future_activity_df` where `activity_date` is between `signup_date + 30 days` and `signup_date + 90 days`. Otherwise, set to 0. Merge this target back into `user_onboarding_features_df`.
    *   Define feature sets `X` (numerical and categorical columns) and target `y` (`is_retained_90_days`).
    *   Split the data into `X_train`, `X_test`, `y_train`, `y_test` using `train_test_split` with `random_state=42` and `stratify=y`.

4.  **Exploratory Data Visualization**:
    *   Create a stacked bar chart showing the proportion of `is_retained_90_days` (0 vs. 1) for each `assigned_onboarding_variant`. Label axes and provide a clear title (e.g., '90-Day Retention Rate by Onboarding Variant').
    *   Generate a violin plot (or box plot) illustrating the distribution of `total_onboarding_duration` for users who were `is_retained_90_days=0` versus those who were `is_retained_90_days=1`. Ensure appropriate labels and titles.

5.  **Machine Learning Pipeline Construction, Training, and Evaluation**:
    *   Build an `sklearn.pipeline.Pipeline` starting with a `sklearn.compose.ColumnTransformer`.
    *   Within the `ColumnTransformer`:
        *   Apply `SimpleImputer(strategy='mean')` followed by `StandardScaler` to the numerical features.
        *   Apply `OneHotEncoder(handle_unknown='ignore')` to the categorical features (`assigned_onboarding_variant`, `referral_source`).
    *   Add `sklearn.linear_model.LogisticRegression` as the final estimator in the pipeline, configured with `random_state=42`, `solver='liblinear'`, and `class_weight='balanced'`.
    *   Train the complete pipeline using `X_train` and `y_train`.
    *   Predict probabilities for the positive class (retention, class 1) on the `X_test` set.
    *   Calculate and print the `roc_auc_score` and a `classification_report` for the test set predictions.