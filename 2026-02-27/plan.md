Here are the implementation steps for the Ad Click-Through Rate (CTR) Prediction task, designed for a Python ML engineer:

1.  **Generate Synthetic Data with Realistic Biases:**
    *   Create `users_df` using Pandas and Numpy, generating 500-700 unique user IDs, random `signup_date`s (last 5 years), `age` (18-70), `gender` (e.g., 'Male', 'Female', 'Other'), `device_type` (e.g., 'Mobile', 'Desktop', 'Tablet'), and `ad_blocker_enabled` (0/1).
    *   Create `ads_df` with 50-100 unique ad IDs, `advertiser_category` (e.g., 'Finance', 'Gaming', 'Retail', 'Travel'), `ad_format` (e.g., 'Banner', 'Video', 'Native'), `target_audience_age_min` (18-50), and `target_audience_age_max` (30-70, ensuring `max > min`).
    *   Generate `impressions_df` with 5000-8000 unique impression IDs. Randomly sample `user_id`s from `users_df` and `ad_id`s from `ads_df`.
    *   Assign `impression_date`s to be *after* the corresponding user's `signup_date`.
    *   Simulate `was_clicked` (binary target, 0 or 1) with an overall CTR of 2-5%. Introduce biases:
        *   Significantly lower CTR for `ad_blocker_enabled=1` users.
        *   Higher CTR for users whose `age` falls within the ad's `target_audience_age_min`/`max` range.
        *   Generally higher CTR for specific `advertiser_category`s (e.g., 'Gaming') or `ad_format`s (e.g., 'Video').
        *   Introduce a sequential bias: Users who have a history of prior clicks (within their impressions before the current one) should have a slightly higher propensity to click again. This will require an iterative approach or a clever post-processing step during data generation to simulate this history.
    *   Ensure `impressions_df` is sorted by `user_id` then `impression_date` for consistent sequential processing in the next steps.

2.  **Load Data into SQLite and Engineer Event-Level Features with SQL:**
    *   Establish an in-memory SQLite database connection using `sqlite3`.
    *   Load `users_df`, `ads_df`, and the sorted `impressions_df` into three separate tables named `users`, `ads`, and `impressions`, respectively.
    *   Construct a single comprehensive SQL query that joins `users`, `ads`, and `impressions`. For each impression event, the query must calculate the following using SQL window functions (`OVER (PARTITION BY ... ORDER BY ...)`):
        *   `user_prior_impressions`: Count of *previous* impressions for the same user.
        *   `user_prior_clicks`: Count of *previous* clicks for the same user.
        *   `user_prior_ctr`: Calculated as `user_prior_clicks` / `user_prior_impressions`, handling division by zero (return 0.0 if no prior impressions).
        *   `days_since_last_user_click`: Days between the current `impression_date` and the user's *most recent prior click date*. If no prior clicks, use days between `signup_date` and `impression_date`. Use `LAG()` and `COALESCE()` for this.
        *   `ad_prior_impressions`: Count of *previous* impressions for the same ad across all users.
        *   `ad_prior_clicks`: Count of *previous* clicks for the same ad across all users.
        *   `ad_prior_ctr`: Calculated as `ad_prior_clicks` / `ad_prior_impressions`, handling division by zero (return 0.0 if no prior impressions).
    *   Include all original columns from `impressions`, `users`, and `ads` tables in the final selection (`impression_id`, `user_id`, `ad_id`, `impression_date`, `was_clicked`, `age`, `gender`, `device_type`, `ad_blocker_enabled`, `advertiser_category`, `ad_format`, `target_audience_age_min`, `target_audience_age_max`, `signup_date`).

3.  **Pandas Post-Processing, Feature Engineering, and Data Split:**
    *   Fetch the results of the SQL query into a new pandas DataFrame, `ad_features_df`.
    *   Handle any remaining `NaN` values:
        *   Fill `user_prior_impressions`, `user_prior_clicks`, `ad_prior_impressions`, `ad_prior_clicks` with 0.
        *   Fill `user_prior_ctr` and `ad_prior_ctr` with 0.0.
        *   Fill `days_since_last_user_click` with a large sentinel value (e.g., 9999) if any `NaN`s are present after SQL processing (these would typically represent a user's very first impression or first impression after a long period without clicks).
    *   Convert `signup_date` and `impression_date` columns to pandas datetime objects.
    *   Calculate `user_account_age_at_impression_days`: The number of days between `signup_date` and `impression_date`.
    *   Create a binary feature `is_user_in_target_audience`: Set to 1 if the user's `age` is between `target_audience_age_min` and `target_audience_age_max` (inclusive) for the specific ad, otherwise 0.
    *   Define the feature matrix `X` and the target vector `y`:
        *   `X`: `age`, `user_account_age_at_impression_days`, `user_prior_impressions`, `user_prior_clicks`, `user_prior_ctr`, `days_since_last_user_click`, `ad_prior_impressions`, `ad_prior_clicks`, `ad_prior_ctr`, `target_audience_age_min`, `target_audience_age_max` (numerical); `gender`, `device_type`, `ad_blocker_enabled`, `advertiser_category`, `ad_format`, `is_user_in_target_audience` (categorical).
        *   `y`: `was_clicked`.
    *   Split `X` and `y` into training (70%) and testing (30%) sets using `sklearn.model_selection.train_test_split`, ensuring `random_state=42` and `stratify=y` to maintain the class distribution.

4.  **Visualize Key Feature Relationships with Click-Through Rate:**
    *   Generate a violin plot (or box plot) to compare the distribution of `user_prior_ctr` for `was_clicked=0` versus `was_clicked=1`. Label axes appropriately and provide a descriptive title.
    *   Create a stacked bar chart showing the proportion of `was_clicked` (0 or 1) for each unique `ad_format`. Ensure the plot is clearly labeled and titled to show how click-through rates vary by ad format.

5.  **Build and Evaluate an ML Pipeline for CTR Prediction:**
    *   Construct an `sklearn.pipeline.Pipeline` that orchestrates preprocessing and model training.
    *   Inside the pipeline, use an `sklearn.compose.ColumnTransformer` to apply different preprocessing steps to numerical and categorical features:
        *   For numerical features, apply `sklearn.preprocessing.SimpleImputer` with a `strategy='mean'` to handle any potential missing values, followed by `sklearn.preprocessing.StandardScaler` for standardization.
        *   For categorical features, apply `sklearn.preprocessing.OneHotEncoder` with `handle_unknown='ignore'` to convert them into a numerical format.
    *   Append `sklearn.ensemble.HistGradientBoostingClassifier` as the final estimator in the pipeline, setting `random_state=42` for reproducibility.
    *   Train the complete pipeline on the `X_train` and `y_train` datasets.
    *   Predict the probabilities for the positive class (class 1, i.e., `was_clicked=1`) on the `X_test` dataset.
    *   Calculate and print the `roc_auc_score` of the predictions against `y_test`.
    *   Generate and print a `classification_report` for the test set, providing precision, recall, f1-score, and support for both classes.