Here are the steps to develop the machine learning pipeline for transactional fraud prediction:

1.  **Generate Synthetic Data with Initial Fraud Patterns:**
    Create three Pandas DataFrames: `users_df`, `merchants_df`, and `transactions_df` as specified. Populate them with realistic-looking synthetic data including user profiles, merchant characteristics, and transaction details. For `transactions_df`, initially assign `is_fraud` based on static user and merchant attributes (e.g., 'Bronze' user tier, high merchant risk score, large transaction amounts, 'Online' transaction type). Ensure the `transactions_df` is sorted by `user_id` then `transaction_date` for subsequent sequential processing.

2.  **Simulate Advanced Sequential Fraud and Load into SQLite:**
    Refine the `is_fraud` column in `transactions_df` to incorporate sequential fraud patterns:
    *   Significantly increase the probability of `is_fraud=1` for users who have *prior* transactions already marked as fraudulent.
    *   Slightly increase fraud probability for a user's *first* transaction with a specific merchant.
    After this advanced fraud simulation, load the `users_df`, `merchants_df`, and the fraud-enriched `transactions_df` into an in-memory SQLite database, creating tables named `users`, `merchants`, and `transactions` respectively.

3.  **Perform SQL-Based Sequential Feature Engineering:**
    Construct a single SQL query to join `users`, `merchants`, and `transactions` tables. For each transaction, calculate advanced sequential features relative to the transaction's date using window functions (e.g., `LAG` and `SUM/AVG/COUNT` with `PARTITION BY user_id/merchant_id ORDER BY transaction_date`):
    *   User-specific prior aggregates: `user_prior_num_transactions`, `user_prior_total_spend`, `user_avg_prior_transaction_amount`, `user_prior_num_fraud_transactions`, `days_since_last_user_transaction` (or `days_since_signup_at_transaction` for first transactions).
    *   Merchant-specific prior aggregates: `merchant_prior_num_transactions`, `merchant_avg_prior_transaction_amount`, `merchant_prior_num_fraud_transactions`.
    *   Include all static user, merchant, and current transaction attributes. Handle `NULL`s for first transactions (e.g., for prior aggregates or dates) appropriately.

4.  **Pandas Post-Processing, Additional Feature Engineering, and Data Split:**
    Fetch the results of the SQL query into a Pandas DataFrame (`transaction_features_df`). Perform the following operations:
    *   Handle `NaN` values resulting from the SQL query by imputing prior counts/sums/averages with 0 (or 0.0) and `days_since_last_user_transaction` with `days_since_signup_at_transaction` where applicable.
    *   Convert date columns to datetime objects.
    *   Calculate `days_since_signup_at_transaction`.
    *   Engineer additional features: `user_prior_fraud_rate`, `merchant_prior_fraud_rate`, and `amount_deviation_from_user_avg_prior`. Handle division by zero for rates.
    *   Define the feature matrix `X` (including all numerical and categorical features) and the target vector `y` (`is_fraud`).
    *   Split `X` and `y` into training and testing sets using `train_test_split` with `random_state=42` and `stratify=y` to maintain the fraud rate distribution.

5.  **Visualize Key Relationships with Fraud:**
    Generate two distinct visualizations to explore the relationship between features and the `is_fraud` target:
    *   A violin plot (or box plot) showing the distribution of `amount` for non-fraudulent (0) versus fraudulent (1) transactions.
    *   A stacked bar chart illustrating the proportion of `is_fraud` (0 or 1) across different merchant `category` values. Ensure both plots have clear titles and axis labels.

6.  **Build, Train, and Evaluate ML Pipeline:**
    Construct an `sklearn.pipeline.Pipeline` with a `ColumnTransformer` for robust preprocessing:
    *   Apply `SimpleImputer(strategy='mean')` followed by `StandardScaler` to numerical features.
    *   Apply `OneHotEncoder(handle_unknown='ignore')` to categorical features.
    The final estimator in the pipeline should be a `HistGradientBoostingClassifier` (or `RandomForestClassifier`/`XGBoostClassifier`), configured with `random_state=42` and `class_weight='balanced'` to handle class imbalance. Train this pipeline on `X_train` and `y_train`. Finally, predict fraud probabilities on `X_test` and evaluate the model's performance by printing the `roc_auc_score` and a detailed `classification_report`.