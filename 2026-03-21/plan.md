Here are the implementation steps for developing a machine learning pipeline to predict loan default:

1.  **Data Loading and Initial Merging**:
    Load the `applicants_df`, `loans_df`, and `repayments_df` datasets into pandas DataFrames. Perform an initial merge of `applicants_df` with `loans_df` on `applicant_id` to create a foundational dataset containing core applicant and loan details for each loan.

2.  **Core Applicant and Loan Feature Engineering**:
    From the merged `applicants` and `loans` data, engineer relevant features. This includes calculating derived numerical features such as `debt_to_income_ratio` (if income is considered with other debts, though not explicitly in schema, it's a good example), `loan_to_income_ratio`, or `age_at_application`. Identify and prepare categorical features (`loan_purpose`, `employment_status`) for later encoding.

3.  **Early Repayment Behavior Feature Engineering**:
    Process the `repayments_df` to extract features specifically related to a loan's early repayment performance. For each loan, calculate metrics within a defined "early behavior window" (e.g., the first 90 days after `disbursement_date`). Examples include:
    *   Number of late payments within the window.
    *   Total amount underpaid within the window (`expected_amount_per_month` - `amount_paid`).
    *   Average payment ratio (`amount_paid` / `expected_amount_per_month`) within the window.
    *   Boolean flags indicating any late payments or underpayments in the window.
    *   The time (in days) from `disbursement_date` to the first actual repayment.

4.  **Target Variable Creation (`is_bad_loan`)**:
    For each unique `loan_id`, calculate the `is_bad_loan` target variable. This involves:
    *   Determining the total expected repayment amount over the full loan term (principal + total interest). This can be approximated for synthetic data or calculated precisely using standard amortization formulas.
    *   Aggregating the `amount_paid` for *all* repayments associated with that `loan_id` from `repayments_df`.
    *   Defining `is_bad_loan` as `1` if the total `amount_paid` is significantly less than the total expected repayment (e.g., less than 90% or 80%), and `0` otherwise.

5.  **Final Feature Set Assembly and Preprocessing**:
    Consolidate all engineered features (from steps 2 and 3) along with the `is_bad_loan` target into a single, comprehensive master dataframe. Apply necessary preprocessing steps:
    *   Handle any missing values using appropriate imputation strategies.
    *   Encode categorical features (e.g., using One-Hot Encoding or Target Encoding).
    *   Scale numerical features (e.g., using StandardScaler or MinMaxScaler).

6.  **Model Training, Prediction, and Evaluation**:
    Split the preprocessed master dataframe into training and testing sets (e.g., 80% train, 20% test). Select and train a suitable binary classification model (e.g., Logistic Regression, Random Forest, Gradient Boosting Machine like XGBoost or LightGBM). Make predictions on the test set and evaluate the model's performance using relevant metrics for binary classification, paying attention to potential class imbalance. Key metrics include AUC-ROC, Precision, Recall, F1-score, and a Confusion Matrix.