Here are the implementation steps for developing a machine learning pipeline to predict the popularity tier of online courses:

1.  **Synthetic Data Generation**:
    *   Generate `courses_df` with `course_id`, `release_date` (random dates over 3 years), `category`, `instructor_experience_years`, `difficulty_level`, and `price`. To simulate popularity, assign an intrinsic `popularity_factor` to each course, which will bias enrollment counts later.
    *   Generate `users_df` with `user_id`, `signup_date` (random dates over 5 years), and `region`.
    *   Generate `enrollments_df` with `enrollment_id`, `user_id`, `course_id`, `enrollment_date`, `completion_percentage`, and `time_spent_hours`.
        *   Crucially, ensure `enrollment_date` for each record is after both the respective `user_id`'s `signup_date` and `course_id`'s `release_date`.
        *   Simulate realistic patterns: Use the `popularity_factor` from `courses_df` to bias the number of enrollments for each course. Higher `instructor_experience_years` or specific `category`s (e.g., 'Data Science', 'Programming') should correlate with more enrollments. 'Advanced' courses should generally have lower enrollment counts but higher `completion_percentage` and `time_spent_hours` for those who do enroll. Ensure `time_spent_hours` generally correlates positively with `completion_percentage`.
        *   Sort `enrollments_df` by `course_id` then `enrollment_date`.

2.  **SQL-like Early Performance Metric Aggregation (First 30 Days)**:
    *   For each course in `courses_df`, define an `initial_popularity_cutoff_date` as `release_date + 30 days`.
    *   Using operations analogous to SQL, `LEFT JOIN` `courses_df` with `enrollments_df` on `course_id`.
    *   Filter the joined data to include only enrollments where `enrollment_date` falls between `release_date` (inclusive) and `initial_popularity_cutoff_date` (inclusive) for that specific course.
    *   `GROUP BY course_id` on the filtered early enrollments to calculate the following metrics, ensuring `IFNULL`-like handling (e.g., filling with 0 for counts/sums, 0.0 for averages) for courses with no enrollments in the first 30 days:
        *   `num_enrollments_30d` (count of enrollments)
        *   `num_unique_users_30d` (count of distinct users)
        *   `avg_completion_30d` (average `completion_percentage`)
        *   `avg_time_spent_30d` (average `time_spent_hours`)
        *   `days_to_first_enrollment`: Calculate the difference in days between the *earliest* `enrollment_date` within the 30-day window and the `release_date`. For courses with no enrollments in the first 30 days, fill this with a sentinel value (e.g., 30 or 31), indicating enrollment occurred after the window or never.

3.  **Pandas-based Target Variable and Additional Feature Engineering**:
    *   **Target Variable Creation (`popularity_tier`)**:
        *   Calculate `total_enrollments_all_time` for each course from the *original, unfiltered* `enrollments_df`.
        *   Assign courses with `total_enrollments_all_time == 0` to the 'Low_LTV' popularity tier.
        *   For courses with `total_enrollments_all_time > 0`, calculate thresholds by finding the 33rd and 66th percentiles of `total_enrollments_all_time` among these non-zero values.
        *   Use `np.select` to assign the remaining courses to tiers: 'Medium' (below 33rd percentile), 'High' (between 33rd and 66th percentiles), and 'Very_High' (above 66th percentile).
    *   **Feature Consolidation**: Merge the `courses_df` with the aggregated early performance metrics (from Step 2) and the newly created `popularity_tier`.
    *   Fill any remaining `NaN` values in the aggregated early performance metrics (e.g., `avg_completion_30d`, `avg_time_spent_30d`) with appropriate defaults (0.0 for averages). Ensure `days_to_first_enrollment` `NaN`s are handled with the sentinel value.

4.  **Machine Learning Model Training and Evaluation**:
    *   **Data Preparation**:
        *   Separate the merged DataFrame into features (X) and the target variable (`popularity_tier`, y).
        *   Identify categorical features (`category`, `difficulty_level`). Apply One-Hot Encoding to these features.
        *   Scale numerical features (e.g., `price`, `instructor_experience_years`, and all aggregated early metrics) using a standard scaler, although tree-based models like `HistGradientBoostingClassifier` are less sensitive to scaling.
    *   **Train-Test Split**: Split the preprocessed data into training and testing sets using `train_test_split`, ensuring to use `stratify=y` to maintain the class distribution of `popularity_tier` in both sets.
    *   **Model Training**: Initialize and train a `HistGradientBoostingClassifier` model on the training data.
    *   **Evaluation**: Predict the `popularity_tier` on the test set. Evaluate the model's performance using suitable multi-class classification metrics such as accuracy, precision, recall, F1-score (macro or weighted average), and a full classification report.