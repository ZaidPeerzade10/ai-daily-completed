Here are the implementation steps for developing the machine learning pipeline:

1.  **Generate Synthetic Data**:
    *   Create three Pandas DataFrames: `users_df` (500-800 rows), `content_df` (100-150 rows), and `interactions_df` (10000-15000 rows), populating them with the specified columns and data types.
    *   Ensure data consistency and simulate realistic patterns: `interaction_date` must always be after `signup_date`. Implement differential behavior for `Premium` users (higher `duration_minutes`, more `like` interactions) and `Mobile`/`Desktop` users (genre preferences, potentially duration adjustments). Ensure content with higher `avg_rating` attracts more `view` interactions.
    *   Define `current_prediction_date = pd.to_datetime('2024-03-01')`. All `interaction_date` values in `interactions_df` must be less than or equal to this date.
    *   Sort `interactions_df` first by `user_id` and then by `interaction_date` in ascending order.

2.  **Load into SQLite & SQL Feature Engineering (Historical Interaction Patterns)**:
    *   Establish an in-memory SQLite database connection using `sqlite3`.
    *   Load `users_df`, `content_df`, and `interactions_df` into SQL tables named `users`, `content`, and `interactions` respectively.
    *   Write and execute a single SQL query that performs the following for *each user*:
        *   Define `history_cutoff_date` as `current_prediction_date - 30 days` (e.g., `DATE('2024-03-01', '-30 days')`).
        *   Aggregate historical features for user interactions occurring within the 60-day window *ending at* `history_cutoff_date` (i.e., `interaction_date` between `history_cutoff_date - 60 days` and `history_cutoff_date`). The aggregated features should be: `num_interactions_prev_60d`, `total_duration_prev_60d`, `num_unique_genres_prev_60d`, and `avg_content_rating_prev_60d`.
        *   Include static user attributes: `user_id`, `signup_date`, `region`, `subscription_tier`, `device_type`, `age`.
        *   Use a `LEFT JOIN` from the `users` table to ensure all users are included in the result. For users with no activity in the 60-day window, their aggregated features should default to 0 for counts/sums and 0.0 for averages.
        *   Fetch the query results into a Pandas DataFrame.

3.  **Pandas Feature Engineering & Multi-Class Target Creation (Next Preferred Genre)**:
    *   Load the SQL query results into a Pandas DataFrame, e.g., `user_history_features_df`.
    *   Handle `NaN` values for the aggregated features: fill `num_interactions_prev_60d`, `total_duration_prev_60d`, `num_unique_genres_prev_60d` with 0, and `avg_content_rating_prev_60d` with 0.0.
    *   Convert the `signup_date` column to datetime objects. Recalculate `current_prediction_date` and `history_cutoff_date` as Pandas `datetime` objects.
    *   Calculate a new feature `interaction_frequency_prev_60d` by dividing `num_interactions_prev_60d` by 60.0. Fill any resulting `NaN`s with 0.
    *   **Create the Multi-Class Target `next_preferred_genre`**:
        *   Identify interactions from the original `interactions_df` (joined with `content_df` to get `genre`) that occur *between* `history_cutoff_date` (exclusive) and `history_cutoff_date + 30 days` (inclusive).
        *   For each user, determine the `genre` with the highest total `duration_minutes` within this 30-day future window. This will be the `next_preferred_genre`.
        *   Merge this `next_preferred_genre` information with `user_history_features_df` using a left join based on `user_id`.
        *   For any user who has no interactions in this future 30-day window, assign their `next_preferred_genre` target as 'No Future Preference'.
    *   Define the feature matrix `X` (including numerical features like `num_interactions_prev_60d`, `total_duration_prev_60d`, `num_unique_genres_prev_60d`, `avg_content_rating_prev_60d`, `age`, `interaction_frequency_prev_60d`; and categorical features like `region`, `subscription_tier`, `device_type`) and the target vector `y` (`next_preferred_genre`).
    *   Split `X` and `y` into training and testing sets (e.g., 70/30) using `sklearn.model_selection.train_test_split`, ensuring `random_state=42` and `stratify` on `y` to maintain class balance.

4.  **Data Visualization**:
    *   Generate a violin plot (or box plot) that illustrates the distribution of `avg_content_rating_prev_60d` for each unique category within the `next_preferred_genre` target. Ensure the plot has appropriate titles and axis labels.
    *   Create a stacked bar chart displaying the proportional distribution of `next_preferred_genre` across the different `subscription_tier` values. Ensure the chart includes clear titles and axis labels.

5.  **ML Pipeline & Evaluation (Multi-Class)**:
    *   Construct an `sklearn.pipeline.Pipeline`.
    *   Within the pipeline, include a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features, apply a `sklearn.preprocessing.SimpleImputer(strategy='mean')` followed by a `sklearn.preprocessing.StandardScaler`.
        *   For categorical features, apply a `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`.
    *   Set `sklearn.ensemble.HistGradientBoostingClassifier(random_state=42)` as the final estimator in the pipeline.
    *   Train the complete pipeline using `X_train` and `y_train`.
    *   Use the trained pipeline to predict the `next_preferred_genre` for `X_test`.
    *   Calculate and print the `sklearn.metrics.accuracy_score` and a comprehensive `sklearn.metrics.classification_report` based on the `y_test` and the predictions.