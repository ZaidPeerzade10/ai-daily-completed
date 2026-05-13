Here are the implementation steps for developing the machine learning pipeline to predict loan defaults:

1.  **Generate Synthetic Dataframes**:
    *   Create `applicants_df` with `applicant_id` (1000-1500 unique integers), `age` (20-70), `income` (2000-15000), `credit_score` (300-850), and `employment_status` (e.g., 'Employed', 'Self-Employed', 'Unemployed', 'Retired').
    *   Create `loans_df` with `loan_id` (1000-1500 unique integers), `applicant_id` (randomly sampled from `applicants_df`), `loan_amount` (1000-50000), `loan_term_months` (12-60), `interest_rate` (0.05-0.20), `loan_date` (random dates over the last 5 years), and `default_date`. For `default_date`, assign a random date after `loan_date` (within the last 2 years) for ~10-15% of loans, and `NaT` otherwise. Ensure `loan_id` is unique.
    *   Create `payments_df` with `payment_id` (15000-25000 unique integers), `loan_id` (randomly sampled from `loans_df`), `payment_date` (random dates after respective `loan_date` and before `default_date` if applicable), and `paid_amount` (100-2000).
    *   **Simulate realistic patterns**: Ensure `payment_date` is always after `loan_date` and strictly before `default_date` for defaulted loans. For defaulted loans, simulate fewer or smaller `paid_amount`s, especially in the 3-6 months leading up to the `default_date`. Introduce correlations where lower `credit_score` and `income` generally lead to higher `interest_rate` and an increased likelihood of a `default_date` being present.
    *   Sort `payments_df` by `loan_id` then `payment_date`.

2.  **Load into SQLite & SQL Feature Engineering**:
    *   Initialize an in-memory SQLite database using `sqlite3`.
    *   Load `applicants_df`, `loans_df`, and `payments_df` into SQL tables named `applicants`, `loans`, and `payments` respectively. Ensure date columns are stored in a format compatible with SQL date functions (e.g., ISO 8601 strings).
    *   Define `GLOBAL_PREDICTION_CUTOFF_DATE` as `payments_df['payment_date'].max() - pd.Timedelta(months=3)`.
    *   Construct and execute a single SQL query to retrieve the following for *each loan*:
        *   `loan_id`, `applicant_id`, `loan_amount`, `loan_term_months`, `interest_rate`, `loan_date`, `age`, `income`, `credit_score`, `employment_status`.
        *   `current_cutoff_date` (the `GLOBAL_PREDICTION_CUTOFF_DATE` as a constant column).
        *   Aggregations for payments made within the 6 months *immediately preceding* `GLOBAL_PREDICTION_CUTOFF_DATE`: `num_payments_prev_6m` (count), `total_paid_prev_6m` (sum), `avg_payment_prev_6m` (average). Use `LEFT JOIN` from `loans` to `payments` and `COALESCE` to ensure all loans are included, with 0 for counts/sums/averages if no payments exist in the window.
        *   `days_since_last_payment_at_cutoff`: Number of days between `GLOBAL_PREDICTION_CUTOFF_DATE` and the most recent `payment_date` *before or on* the cutoff. Return 9999 if no payments are found before or on the cutoff date.
        *   `outstanding_balance_at_cutoff`: `loan_amount` minus the sum of all `paid_amount`s for payments *before or on* `GLOBAL_PREDICTION_CUTOFF_DATE`. Return `loan_amount` if no payments were made before or on the cutoff.

3.  **Pandas Feature Engineering & Binary Target Creation**:
    *   Fetch the results of the SQL query into a pandas DataFrame, `loan_features_df`.
    *   Convert `loan_date` and `current_cutoff_date` columns in `loan_features_df` to datetime objects.
    *   Handle `NaN` values: Fill `num_payments_prev_6m`, `total_paid_prev_6m`, `avg_payment_prev_6m` with 0 or 0.0. Fill `days_since_last_payment_at_cutoff` with 9999 (if not already handled by SQL).
    *   Calculate `loan_age_at_cutoff_days`: The number of days between `loan_date` and `current_cutoff_date`.
    *   Calculate `payment_frequency_prev_6m`: `num_payments_prev_6m` divided by 180.0. Fill any `NaN` or `inf` results with 0.
    *   Merge the original `default_date` from `loans_df` (using `loan_id`) into `loan_features_df`.
    *   Create the binary target `will_default_in_next_90_days`: This should be 1 if the loan's `default_date` falls within the 90-day period *immediately following* `current_cutoff_date`, and 0 otherwise. Handle `NaT` `default_date` values by treating them as non-defaults.
    *   Define feature sets: `X_numerical` (e.g., `loan_amount`, `age`, `credit_score`, `num_payments_prev_6m`, etc.), `X_categorical` (e.g., `employment_status`), and the target `y` (`will_default_in_next_90_days`).
    *   Split the data into training and testing sets (e.g., 70/30) using `sklearn.model_selection.train_test_split`, ensuring `random_state=42` and `stratify=y` to maintain class balance.

4.  **Data Visualization**:
    *   Generate a violin plot (or box plot) using `matplotlib.pyplot` and `seaborn` to visualize the distribution of `credit_score` for loans that `will_default_in_next_90_days` (1) versus those that will not (0). Ensure the plot has appropriate titles and axis labels.
    *   Create a stacked bar chart showing the proportion of `will_default_in_next_90_days` (0 or 1) for each `employment_status` category. Include clear labels and a title for the plot.

5.  **ML Pipeline & Evaluation**:
    *   Construct an `sklearn.pipeline.Pipeline` with a `sklearn.compose.ColumnTransformer` for preprocessing:
        *   For numerical features (from `X_numerical`): Apply `sklearn.preprocessing.SimpleImputer(strategy='mean')` to handle any remaining missing values, followed by `sklearn.preprocessing.StandardScaler` for standardization.
        *   For categorical features (from `X_categorical`): Apply `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')` to convert them into a numerical format.
    *   Append `sklearn.ensemble.HistGradientBoostingClassifier` as the final estimator to the pipeline, setting `random_state=42`.
    *   Train the complete pipeline on `X_train` and `y_train`.
    *   Predict probabilities for the positive class (class 1) on the `X_test` set.
    *   Calculate and print the `sklearn.metrics.roc_auc_score` for the test set.
    *   Generate and print a `sklearn.metrics.classification_report` for the test set predictions (after converting probabilities to binary predictions using a suitable threshold, typically 0.5, or a threshold optimized for your specific business goal).