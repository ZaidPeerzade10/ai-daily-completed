Here are the implementation steps for developing a machine learning pipeline to predict streaming service customer churn:

1.  **Generate Synthetic Data**:
    *   Create three Pandas DataFrames: `customers_df`, `content_df`, and `viewing_history_df`.
    *   Populate `customers_df` (1000-1500 rows) with `customer_id`, `signup_date` (random 3-5 years back), `subscription_plan`, `region`, `age_group`. For 15-20% of customers, generate a `churn_date` within the last 12 months, ensuring it's after their `signup_date`; `NaT` otherwise.
    *   Populate `content_df` (100-200 rows) with `content_id`, `title`, `genre`, and `avg_rating`.
    *   Populate `viewing_history_df` (20000-30000 rows) with `view_id`, `customer_id`, `content_id`, `view_date`, and `duration_minutes`.
    *   Ensure `view_date` for each entry is after the customer's `signup_date` and before their `churn_date` (if applicable).
    *   **Simulate realistic patterns**: Implement logic for 'Premium' plan users to generally have higher `duration_minutes` and more frequent views. For customers with a `churn_date`, simulate a noticeable drop-off in `duration_minutes` and view frequency 1-2 months leading up to their `churn_date` compared to their earlier activity.
    *   Sort `viewing_history_df` by `customer_id` then `view_date`.

2.  **Load Data into SQLite and SQL Feature Engineering**:
    *   Establish an in-memory SQLite database connection using `sqlite3`.
    *   Load `customers_df`, `content_df`, and `viewing_history_df` into respective tables: `customers`, `content`, and `viewing_history`.
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as 1 month prior to the latest `view_date` in the generated `viewing_history_df`.
    *   Write a single SQL query to retrieve `customer_id`, `signup_date`, `subscription_plan`, `region`, `age_group`, and compute the following features for *each customer*, aggregated over the 30 days immediately preceding `GLOBAL_PREDICTION_CUTOFF_DATE`:
        *   `current_cutoff_date` (the `GLOBAL_PREDICTION_CUTOFF_DATE`).
        *   `total_view_duration_prev_30d` (sum of `duration_minutes`).
        *   `num_views_prev_30d` (count of `view_id`s).
        *   `num_unique_content_prev_30d` (count of distinct `content_id`s).
        *   `num_unique_genres_prev_30d` (count of distinct `genre`s, obtained by joining `viewing_history` with `content`).
        *   `days_since_last_view_at_cutoff`: Number of days between `current_cutoff_date` and the most recent `view_date` *before or on* the cutoff. Return 9999 if no views before cutoff.
    *   Use `LEFT JOIN` operations to ensure all customers are included, even those with no activity in the window, and handle `NULL`s for aggregated features by defaulting to 0 or 0.0 using `COALESCE`.

3.  **Pandas Feature Transformation and Binary Target Creation**:
    *   Fetch the results of the SQL query into a pandas DataFrame, `customer_features_df`.
    *   Convert `signup_date` and `current_cutoff_date` columns to datetime objects.
    *   Handle `NaN` values resulting from no activity: Fill `total_view_duration_prev_30d`, `num_views_prev_30d`, `num_unique_content_prev_30d`, `num_unique_genres_prev_30d` with 0 or 0.0. Fill `days_since_last_view_at_cutoff` with 9999.
    *   Calculate `customer_tenure_at_cutoff_days`: The difference in days between `current_cutoff_date` and `signup_date`.
    *   Calculate `avg_view_duration_per_view_prev_30d`: `total_view_duration_prev_30d` / `num_views_prev_30d`. Fill any `NaN` or `inf` resulting from division by zero with 0.
    *   Merge the original `customers_df` (specifically the `churn_date` column) back into `customer_features_df` using `customer_id`.
    *   Create the binary target variable `will_churn_in_next_30_days`: Set to 1 if the customer's `churn_date` falls within the 30-day period *immediately following* their `current_cutoff_date`; otherwise, set to 0. Fill `NaN`s (for customers who didn't churn or whose churn date is outside this specific window) with 0.
    *   Define numerical features (e.g., `total_view_duration_prev_30d`, `num_views_prev_30d`, `customer_tenure_at_cutoff_days`, etc.) and categorical features (e.g., `subscription_plan`, `region`, `age_group`) for `X`. Define `y` as `will_churn_in_next_30_days`.
    *   Split the dataset into training and testing sets (e.g., 70/30) using `sklearn.model_selection.train_test_split`, ensuring `random_state=42` and `stratify=y` for consistent and balanced splits.

4.  **Exploratory Data Analysis (EDA) with Visualizations**:
    *   Create a violin plot (or box plot) using `matplotlib.pyplot` and `seaborn` to compare the distribution of `total_view_duration_prev_30d` for customers who `will_churn_in_next_30_days` (target = 1) versus those who will not (target = 0). Include appropriate labels and a descriptive title.
    *   Generate a stacked bar chart showing the proportion of churners (1) and non-churners (0) within each category of `subscription_plan`. Ensure clear labels, title, and legend.

5.  **Machine Learning Pipeline Construction, Training, and Evaluation**:
    *   Build an `sklearn.pipeline.Pipeline` starting with a `sklearn.compose.ColumnTransformer`.
    *   Configure the `ColumnTransformer`:
        *   For numerical features, apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by `sklearn.preprocessing.StandardScaler`.
        *   For categorical features, apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   Set the final estimator in the pipeline to `sklearn.ensemble.HistGradientBoostingClassifier`, initialized with `random_state=42` and `class_weight='balanced'` to address potential target imbalance.
    *   Train the complete pipeline using `X_train` and `y_train`.
    *   Use the trained pipeline to predict churn probabilities for the positive class (class 1) on the `X_test` dataset.
    *   Calculate and print the `sklearn.metrics.roc_auc_score` using the predicted probabilities and `y_test`.
    *   Generate and print a `sklearn.metrics.classification_report` on the test set, using predicted class labels (derived from probabilities with a suitable threshold, typically 0.5) and `y_test`, to provide a comprehensive evaluation including precision, recall, and F1-score.