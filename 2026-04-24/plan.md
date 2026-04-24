Here are the implementation steps for the given task, broken down for a Python ML engineer:

1.  **Generate Synthetic Data with Realistic Patterns:**
    *   Create `machines_df` (500-700 rows) with `machine_id`, `machine_type` ('TypeA', 'TypeB', 'TypeC'), `location` ('North', 'South', 'East', 'West'), and `installation_date` (random dates over the last 3 years).
    *   For 20-30% of these machines, randomly assign a `major_error_date` (a random date *after* `installation_date` and within the last 6 months) and add this column to `machines_df`.
    *   Generate `telemetry_df` (15000-25000 rows) with `telemetry_id`, `machine_id` (sampled from `machines_df`), `timestamp` (random dates *after* respective `installation_date`), `temperature` (20-100), and `vibration` (0-10).
        *   Implement machine-type specific baselines: 'TypeA' machines should have slightly higher baseline `vibration`, and 'TypeC' machines higher `temperature`.
        *   For machines with a `major_error_date`, ensure that `temperature` and `vibration` readings show a noticeable increasing trend (e.g., +5-15% above baseline) in the 7-14 days *before* their `major_error_date`.
    *   Create `maintenance_df` (2000-3000 rows) with `maintenance_id`, `machine_id` (sampled from `machines_df`), `maintenance_date` (random dates *after* respective `installation_date`), and `maintenance_type` ('Routine Check', 'Lubrication', 'Component Repair', 'Software Update').
        *   Ensure that a significant portion of `maintenance_type='Component Repair'` entries occur *after* (e.g., 0-7 days after) some of the high sensor readings or known `major_error_date` for a machine.
    *   Finally, sort `telemetry_df` by `machine_id` then `timestamp`, and `maintenance_df` by `machine_id` then `maintenance_date`.

2.  **Load Data into SQLite and Perform SQL Feature Engineering:**
    *   Initialize an in-memory SQLite database using the `sqlite3` module.
    *   Load `machines_df`, `telemetry_df`, and `maintenance_df` into tables named `machines`, `telemetry`, and `maintenance` respectively.
    *   Construct a single SQL query using Common Table Expressions (CTEs) to achieve the following:
        *   First, determine the `latest_telemetry_date` for each machine by finding the maximum `timestamp` in the `telemetry` table.
        *   `LEFT JOIN` the `machines` table with subqueries that aggregate `telemetry` and `maintenance` data.
        *   For each machine, aggregate features *within the 30-day window ending at its `latest_telemetry_date`*: `avg_temp_prev_30d`, `max_vibration_prev_30d`, `num_telemetry_readings_prev_30d`, `num_maintenances_prev_30d`, and `num_repairs_prev_30d` (where `maintenance_type` is 'Component Repair').
        *   Include `machine_id`, `machine_type`, `location`, `installation_date`, and `current_observation_date` (which is the `latest_telemetry_date`).
        *   Use `julianday()` for robust date comparisons in `WHERE` clauses for the 30-day window.
        *   Apply `COALESCE` to aggregated numerical features to ensure 0 or 0.0 for machines with no activity in the window (due to `LEFT JOIN` generating `NULL`s).

3.  **Pandas Feature Engineering and Binary Target Creation:**
    *   Fetch the results of the SQL query from step 2 into a pandas DataFrame, naming it `machine_features_df`.
    *   Handle `NaN` values in numerical aggregated features (e.g., `avg_temp_prev_30d`, `max_vibration_prev_30d`, counts) by filling them with 0 or 0.0.
    *   Convert `installation_date` and `current_observation_date` columns to `datetime` objects.
    *   Calculate `days_since_installation_at_obs` as the difference in days between `current_observation_date` and `installation_date`.
    *   Calculate `telemetry_frequency_prev_30d` by dividing `num_telemetry_readings_prev_30d` by 30.0. Fill any `NaN` or `inf` resulting from this calculation with 0.
    *   **Create the Binary Target `major_error_in_next_7_days`**: Merge the original `major_error_date` column from `machines_df` (which you generated in Step 1) into `machine_features_df` using `machine_id`. For each machine, determine if its `major_error_date` falls within the 7-day period *immediately following* its `current_observation_date`. Assign 1 if true, 0 otherwise. Fill any `NaN`s (for machines without a `major_error_date` or whose error date is outside the relevant window) with 0.
    *   Define feature sets `X` (numerical: `avg_temp_prev_30d`, `max_vibration_prev_30d`, `num_telemetry_readings_prev_30d`, `num_maintenances_prev_30d`, `num_repairs_prev_30d`, `days_since_installation_at_obs`, `telemetry_frequency_prev_30d`; categorical: `machine_type`, `location`) and target `y` (`major_error_in_next_7_days`).
    *   Split `X` and `y` into training and testing sets (e.g., 70/30 split) using `train_test_split`, ensuring `random_state=42` and `stratify=y` for class balance.

4.  **Data Visualization for Insight:**
    *   Create a violin plot (or box plot) to compare the distribution of `max_vibration_prev_30d` for instances where `major_error_in_next_7_days` is 0 versus 1. Include appropriate titles and axis labels.
    *   Generate a stacked bar chart showing the proportion of `major_error_in_next_7_days` (0 or 1) for each distinct `machine_type`. Ensure the chart is clearly labeled and titled.

5.  **Build and Evaluate an ML Pipeline for Binary Classification:**
    *   Construct an `sklearn.pipeline.Pipeline` starting with a `sklearn.compose.ColumnTransformer`.
    *   Within the `ColumnTransformer`:
        *   Apply `SimpleImputer(strategy='mean')` followed by `StandardScaler` to the numerical features.
        *   Apply `OneHotEncoder(handle_unknown='ignore')` to the categorical features.
    *   Append `HistGradientBoostingClassifier(random_state=42)` as the final estimator in the pipeline.
    *   Train this pipeline using the `X_train` and `y_train` datasets.
    *   Predict probabilities for the positive class (class 1) on the `X_test` dataset.
    *   Calculate and print the `roc_auc_score` for the test set.
    *   Generate and print a `classification_report` for the test set predictions (you may need to convert probabilities to binary predictions using a threshold, e.g., 0.5, for the classification report).