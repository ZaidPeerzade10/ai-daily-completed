As a senior data science mentor, I've outlined a comprehensive machine learning pipeline for predicting social media post virality. This approach covers data simulation, feature engineering from various sources (SQL and Pandas), data visualization, and a robust machine learning workflow using scikit-learn.

Here are the detailed steps:

1.  **Generate Synthetic Dataset for Posts and Interactions:**
    Create two pandas DataFrames: `posts_df` (1000-1500 rows) and `interactions_df` (30000-50000 rows).
    *   `posts_df` should include `post_id`, `user_id`, `post_date` (random dates over 1-2 years), `category`, `num_hashtags`, `sentiment_score` (between -1 and 1), and `user_follower_count`.
    *   `interactions_df` should include `interaction_id`, `post_id` (sampled from `posts_df`), `interaction_timestamp`, and `interaction_type` ('like', 'comment', 'share', 'view').
    *   Crucially, simulate realistic patterns: ensure `interaction_timestamp` is always *after* its respective `post_date`. For 5-10% of posts, specifically engineer 'viral' behavior by assigning significantly higher interaction counts (especially shares and comments) within the first 7 days. Posts from users with higher follower counts should have a higher baseline of interactions, and higher sentiment scores should correlate with more likes. Finally, sort `interactions_df` by `post_id` then `interaction_timestamp`.

2.  **Load Data into SQLite and Extract Early Engagement Features via SQL:**
    Establish an in-memory SQLite database connection. Load `posts_df` into a table named `posts` and `interactions_df` into a table named `interactions`.
    *   Define an `early_engagement_window_hours` variable set to 24.
    *   Write a single SQL query to join the `posts` and `interactions` tables. For each post, aggregate its interaction behavior *within the first 24 hours* (`interaction_timestamp` within `post_date` and `post_date + 24 hours`).
    *   The query should calculate: `num_likes_first_24h`, `num_comments_first_24h`, `num_shares_first_24h` (counts for specific interaction types), `total_interactions_first_24h` (total across all types), and `unique_users_first_24h` (count of distinct `user_id`s interacting).
    *   Include static post attributes (`post_id`, `post_date`, `category`, `num_hashtags`, `sentiment_score`, `user_follower_count`).
    *   Utilize a `LEFT JOIN` to ensure all posts are represented, even those with no interactions in the first 24 hours, and use `COALESCE` to replace `NULL` aggregated counts with 0. Leverage SQLite's `julianday()` function for robust date/time comparisons.

3.  **Pandas Feature Engineering and Target Variable Creation:**
    Fetch the results of the SQL query into a pandas DataFrame (`post_early_features_df`).
    *   Handle `NaN` values in the newly aggregated early engagement features (`num_likes_first_24h`, etc.) by filling them with 0.
    *   Convert the `post_date` column to datetime objects.
    *   Create two new features:
        *   `engagement_rate_first_24h`: calculated as `total_interactions_first_24h` / (`user_follower_count` + 1). Handle any `NaN` or `inf` values by replacing them with 0.
        *   `share_comment_ratio_first_24h`: calculated as `num_shares_first_24h` / (`num_comments_first_24h` + 1). Handle any `NaN` or `inf` values by replacing them with 0.
    *   **Create the Binary Target `will_go_viral`**: Define a `viral_window_days` as 7. First, calculate the *total active interactions* (sum of 'like', 'comment', 'share' interactions, excluding 'view') for *each post* within 7 days of its `post_date` using the original `interactions_df`. Then, determine the 90th percentile of these total active interactions across all posts. A post is labeled `will_go_viral = 1` if its total active interactions within the 7-day window exceed this 90th percentile threshold; otherwise, it's `0`. Merge this binary target back to `post_early_features_df`.
    *   Define your feature set `X` (numerical: `num_hashtags`, `sentiment_score`, `user_follower_count`, all `_first_24h` features; categorical: `category`) and your target `y` (`will_go_viral`). Split `X` and `y` into training and testing sets (e.g., 70/30) using `sklearn.model_selection.train_test_split`, ensuring `random_state=42` and `stratify` on `y` to maintain class balance.

4.  **Visualize Key Feature Relationships:**
    Generate two distinct plots to gain insights into the relationship between features and the viral status.
    *   Create a violin plot (or box plot) to visualize the distribution of `engagement_rate_first_24h` for posts that `will_go_viral = 0` versus those that `will_go_viral = 1`. Ensure clear labels and a descriptive title.
    *   Generate a stacked bar chart displaying the proportion of `will_go_viral` (0 or 1) for each distinct `category`. Provide appropriate labels and a title.

5.  **Build and Evaluate an ML Pipeline for Binary Classification:**
    Construct an `sklearn.pipeline.Pipeline` for the classification task.
    *   Integrate a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features, apply a `sklearn.preprocessing.SimpleImputer` with a `strategy='mean'` followed by a `sklearn.preprocessing.StandardScaler`.
        *   For categorical features, use `sklearn.preprocessing.OneHotEncoder` with `handle_unknown='ignore'`.
    *   The final estimator in the pipeline should be a `sklearn.ensemble.HistGradientBoostingClassifier`, initialized with `random_state=42`.
    *   Train this pipeline using your `X_train` and `y_train` data.
    *   Predict the probabilities for the positive class (class 1) on your `X_test` data.
    *   Finally, calculate and print the `sklearn.metrics.roc_auc_score` and a detailed `sklearn.metrics.classification_report` for the test set predictions to evaluate the model's performance.