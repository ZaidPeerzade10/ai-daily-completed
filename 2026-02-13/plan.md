Here are the implementation steps for your employee attrition prediction task:

1.  **Generate Synthetic Data with Churn Simulation:**
    *   **Employees DataFrame (`employees_df`):** Create 500-700 rows. Assign `employee_id` (unique integers), `hire_date` (random dates over the last 5-10 years), `department`, `job_role`, `salary` (float, with a bias for higher roles/departments), `gender`, and `is_churned` (binary, with 15-25% churn rate).
    *   **Performance Reviews DataFrame (`performance_reviews_df`):** Create 1000-1500 rows. Assign `review_id` (unique integers), `employee_id` (sampled from `employees_df`), `review_date` (after `hire_date`), and `rating` (1-5, biased towards 3-5).
    *   **Training Completion DataFrame (`training_completion_df`):** Create 800-1200 rows. Assign `completion_id` (unique integers), `employee_id` (sampled from `employees_df`), `completion_date` (after `hire_date`), and `course_category`.
    *   **Simulate Churn Behavior:** For employees marked `is_churned=1`:
        *   Generate their `rating`s to be generally lower (e.g., mean 2.5-3.0 vs. 3.5-4.0 for non-churned).
        *   Generate fewer `performance_reviews` and `training_completion` records.
        *   Ensure `review_date`s and `completion_date`s are concentrated earlier in their tenure, with few or no records within the last 6-12 months before a potential `analysis_date` (which will be defined later). For non-churned employees, activity should be more consistent throughout their tenure.

2.  **Load Data into SQLite and Perform SQL Feature Engineering:**
    *   **Database Setup:** Create an in-memory SQLite database connection using `sqlite3`. Load `employees_df`, `performance_reviews_df`, and `training_completion_df` into tables named `employees`, `reviews`, and `training` respectively.
    *   **Define Analysis Dates:** Calculate `global_analysis_date` using pandas as the maximum of all `review_date` and `completion_date` values across the generated data, then add 90 days. Define `feature_cutoff_date` as `global_analysis_date` minus 180 days. Convert these dates to a string format suitable for SQL comparison.
    *   **SQL Query Construction:** Write a single SQL query that:
        *   `LEFT JOIN`s `employees` with `reviews` and `training` tables on `employee_id`.
        *   Filters `reviews` and `training` records to only include those where `review_date` or `completion_date` respectively are *before* the `feature_cutoff_date`.
        *   Groups the results by `employee_id` and other static employee attributes.
        *   Aggregates the following features for each employee based on activity *before* the `feature_cutoff_date`: `avg_performance_rating_pre_cutoff`, `num_reviews_pre_cutoff`, `days_since_last_review_pre_cutoff` (calculated as `feature_cutoff_date - MAX(review_date)`), `num_trainings_pre_cutoff`, and `days_since_last_training_pre_cutoff` (calculated as `feature_cutoff_date - MAX(completion_date)`).
        *   Handles `NULL` values using `COALESCE` or `IFNULL`: `0` for `num_reviews_pre_cutoff` and `num_trainings_pre_cutoff`; `0.0` for `avg_performance_rating_pre_cutoff` (if no reviews); and `NULL` for `days_since_last_review_pre_cutoff` and `days_since_last_training_pre_cutoff` if no activities occurred before the cutoff.
        *   Selects `employee_id`, `department`, `job_role`, `salary`, `gender`, `hire_date`, `is_churned`, and all aggregated features.

3.  **Pandas Feature Engineering and Data Preparation:**
    *   **Load SQL Results:** Fetch the results of the SQL query into a pandas DataFrame, named `employee_features_df`.
    *   **Handle `NaN` Values:**
        *   Fill `num_reviews_pre_cutoff` and `num_trainings_pre_cutoff` `NaN`s with `0`.
        *   Fill `avg_performance_rating_pre_cutoff` `NaN`s with a neutral value like `3.0`.
        *   For `days_since_last_review_pre_cutoff` and `days_since_last_training_pre_cutoff`, fill `NaN`s with a large sentinel value (e.g., `3650` days), indicating no activity for a very long time.
    *   **Calculate Tenure:** Convert the `hire_date` column to datetime objects. Calculate `tenure_at_cutoff_days` as the number of days between `hire_date` and the `feature_cutoff_date`.
    *   **Define Features and Target:** Define your feature set `X` to include all numerical features (`salary`, `tenure_at_cutoff_days`, `avg_performance_rating_pre_cutoff`, `num_reviews_pre_cutoff`, `days_since_last_review_pre_cutoff`, `num_trainings_pre_cutoff`, `days_since_last_training_pre_cutoff`) and categorical features (`department`, `job_role`, `gender`). Define your target variable `y` as `is_churned`.
    *   **Train-Test Split:** Split `X` and `y` into training and testing sets (e.g., 70/30 split) using `sklearn.model_selection.train_test_split`, ensuring `random_state=42` and `stratify=y` for class balance.

4.  **Data Visualization:**
    *   **Salary Distribution by Churn:** Create a violin plot (or box plot) to compare the distribution of `salary` for employees where `is_churned=0` versus `is_churned=1`. Label axes and provide a clear title.
    *   **Churn Proportion by Department:** Generate a stacked bar chart that visualizes the proportion of churned (1) and non-churned (0) employees within each `department`. Ensure appropriate labels and a descriptive title.

5.  **Build and Evaluate ML Pipeline:**
    *   **Pipeline Setup:** Create an `sklearn.pipeline.Pipeline`.
        *   **Preprocessing Step:** Within the pipeline, use `sklearn.compose.ColumnTransformer` for preprocessing:
            *   For numerical features, apply a `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by a `sklearn.preprocessing.StandardScaler`.
            *   For categorical features, apply a `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
        *   **Estimator Step:** The final estimator in the pipeline should be an `sklearn.ensemble.GradientBoostingClassifier` with `random_state=42`, `n_estimators=100`, and `learning_rate=0.1`.
    *   **Training and Prediction:** Train the complete pipeline on your `X_train` and `y_train` datasets. Predict probabilities for the positive class (churned, `is_churned=1`) on the `X_test` set.
    *   **Evaluation:** Calculate and print the `sklearn.metrics.roc_auc_score` for the test set predictions. Also, generate and print a `sklearn.metrics.classification_report` to assess precision, recall, and F1-score for both classes.