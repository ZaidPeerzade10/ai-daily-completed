import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# --- 1. Data Generation (Synthetic) ---
np.random.seed(42)
random.seed(42)

N_LOANS = 650  # Between 500-700 for applicants and loans
N_REPAYMENTS_MAX = 7500  # Between 5000-8000 for repayments

# Generate applicants_df
applicant_ids = np.arange(1000, 1000 + N_LOANS)
application_dates = [datetime.now() - timedelta(days=np.random.randint(365*1, 365*5)) for _ in range(N_LOANS)]
credit_scores = np.random.randint(300, 850, N_LOANS)
incomes = np.random.randint(30000, 200000, N_LOANS)
loan_purposes = np.random.choice(['Home', 'Car', 'Education', 'Debt_Consolidation', 'Other'], N_LOANS, p=[0.2, 0.2, 0.15, 0.3, 0.15])
employment_statuses = np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], N_LOANS, p=[0.7, 0.2, 0.1])

applicants_df = pd.DataFrame({
    'applicant_id': applicant_ids,
    'application_date': application_dates,
    'credit_score': credit_scores,
    'income': incomes,
    'loan_purpose': loan_purposes,
    'employment_status': employment_statuses
})

# Generate loans_df
loan_ids = np.arange(2000, 2000 + N_LOANS)
loan_amounts = np.random.uniform(5000, 50000, N_LOANS)
loan_term_months = np.random.randint(12, 61, N_LOANS) # Max 60 months

# Bias interest_rate based on credit_score, employment_status, loan_purpose
interest_rates = np.zeros(N_LOANS)
base_rate = np.random.uniform(0.05, 0.15, N_LOANS)

for i in range(N_LOANS):
    rate = base_rate[i]
    if applicants_df.loc[i, 'credit_score'] < 600:
        rate += np.random.uniform(0.03, 0.08)
    if applicants_df.loc[i, 'employment_status'] == 'Unemployed':
        rate += np.random.uniform(0.05, 0.10)
    if applicants_df.loc[i, 'loan_purpose'] == 'Debt_Consolidation':
        rate += np.random.uniform(0.01, 0.03)
    interest_rates[i] = min(rate, 0.25) # Cap interest rate at 25%

disbursement_dates = [app_date + timedelta(days=np.random.randint(7, 30)) for app_date in applicants_df['application_date']]

loans_df = pd.DataFrame({
    'loan_id': loan_ids,
    'applicant_id': applicants_df['applicant_id'], # 1:1 for simplicity
    'loan_amount': loan_amounts,
    'loan_term_months': loan_term_months,
    'interest_rate': interest_rates,
    'disbursement_date': disbursement_dates
})

# Generate repayments_df
all_repayments = []
repayment_id_counter = 1

for index, loan in loans_df.iterrows():
    loan_id = loan['loan_id']
    disbursement_date = loan['disbursement_date']
    loan_amount = loan['loan_amount']
    loan_term_months = loan['loan_term_months']
    interest_rate = loan['interest_rate']

    # Approximated expected_amount_per_month based on hint
    expected_amount_per_month_base = loan_amount / loan_term_months
    expected_amount_per_month = expected_amount_per_month_base * (1 + interest_rate / 12)

    # Get applicant details for this loan to apply bias
    applicant_details = applicants_df[applicants_df['applicant_id'] == loan['applicant_id']].iloc[0]
    credit_score = applicant_details['credit_score']
    employment_status = applicant_details['employment_status']
    loan_purpose = applicant_details['loan_purpose']

    for month in range(1, loan_term_months + 1):
        repayment_date = disbursement_date + timedelta(days=month * 30) # Approximate monthly
        
        amount_paid = expected_amount_per_month
        is_late = 0 # Default to not late

        # Introduce bias for late/underpayments
        risk_score = 0
        if credit_score < 600: risk_score += 0.4
        if employment_status == 'Unemployed': risk_score += 0.6
        if loan_purpose == 'Debt_Consolidation': risk_score += 0.3
        
        if np.random.rand() < (0.1 + risk_score * 0.2): # Higher chance of issue for risky loans
            is_late = 1
            # Amount paid is significantly less
            amount_paid = expected_amount_per_month * np.random.uniform(0.5, 0.8)
        elif np.random.rand() < (0.05 + risk_score * 0.1): # Small underpayment even if not marked late
            amount_paid = expected_amount_per_month * np.random.uniform(0.9, 0.98)
            
        all_repayments.append({
            'repayment_id': repayment_id_counter,
            'loan_id': loan_id,
            'repayment_date': repayment_date,
            'expected_amount_per_month': expected_amount_per_month,
            'amount_paid': amount_paid,
            'is_late': is_late
        })
        repayment_id_counter += 1

        if len(all_repayments) >= N_REPAYMENTS_MAX:
            break
    if len(all_repayments) >= N_REPAYMENTS_MAX:
        break

repayments_df = pd.DataFrame(all_repayments)

# Convert date columns to datetime objects
applicants_df['application_date'] = pd.to_datetime(applicants_df['application_date'])
loans_df['disbursement_date'] = pd.to_datetime(loans_df['disbursement_date'])
repayments_df['repayment_date'] = pd.to_datetime(repayments_df['repayment_date'])


# --- 2. Data Loading and Initial Merging ---
# Merge applicants and loans data
main_df = pd.merge(loans_df, applicants_df, on='applicant_id', how='left')


# --- 3. Core Applicant and Loan Feature Engineering ---
main_df['loan_to_income_ratio'] = main_df['loan_amount'] / main_df['income']
# Using loan_amount as proxy for 'debt' for simplicity, as no other debt info is given
main_df['debt_to_income_ratio'] = main_df['loan_amount'] / main_df['income'] 
main_df['days_from_application_to_disbursement'] = (main_df['disbursement_date'] - main_df['application_date']).dt.days


# --- 4. Early Repayment Behavior Feature Engineering ---
# Merge repayments with loan details to get disbursement_date
repayments_with_loan_dates = pd.merge(repayments_df, loans_df[['loan_id', 'disbursement_date']], on='loan_id', how='left')
repayments_with_loan_dates['days_since_disbursement'] = (repayments_with_loan_dates['repayment_date'] - repayments_with_loan_dates['disbursement_date']).dt.days

# Filter for early behavior window (first 90 days)
early_window_repayments = repayments_with_loan_dates[repayments_with_loan_dates['days_since_disbursement'] <= 90].copy()

# Add temporary column for underpayment flag before aggregation
early_window_repayments['is_underpaid_repayment'] = (early_window_repayments['amount_paid'] < early_window_repayments['expected_amount_per_month']).astype(int)

# Calculate early repayment numerical features
early_repayment_numerical_features = early_window_repayments.groupby('loan_id').agg(
    num_early_late_payments=('is_late', 'sum'),
    total_expected_early_amount=('expected_amount_per_month', 'sum'),
    total_paid_early_amount=('amount_paid', 'sum')
).reset_index()

early_repayment_numerical_features['total_early_underpaid_amount'] = early_repayment_numerical_features['total_expected_early_amount'] - early_repayment_numerical_features['total_paid_early_amount']
early_repayment_numerical_features['total_early_underpaid_amount'] = early_repayment_numerical_features['total_early_underpaid_amount'].apply(lambda x: max(0, x)) # Only count actual underpayments

early_repayment_numerical_features['avg_early_payment_ratio'] = early_repayment_numerical_features['total_paid_early_amount'] / early_repayment_numerical_features['total_expected_early_amount']
# Handle potential division by zero if total_expected_early_amount is zero (though unlikely with positive expected amounts)
early_repayment_numerical_features.loc[early_repayment_numerical_features['total_expected_early_amount'] == 0, 'avg_early_payment_ratio'] = 1.0


# Additional early repayment boolean flags
early_repayment_flags = early_window_repayments.groupby('loan_id').agg(
    has_early_late_payment=('is_late', lambda x: (x > 0).any()),
    has_early_underpayment=('is_underpaid_repayment', lambda x: (x > 0).any())
).reset_index()

early_repayment_flags['has_early_late_payment'] = early_repayment_flags['has_early_late_payment'].astype(int)
early_repayment_flags['has_early_underpayment'] = early_repayment_flags['has_early_underpayment'].astype(int)

# Days to first repayment (from all repayments, not just early window)
first_repayment_timing = repayments_with_loan_dates.groupby('loan_id')['days_since_disbursement'].min().reset_index()
first_repayment_timing.rename(columns={'days_since_disbursement': 'days_to_first_repayment'}, inplace=True)


# Merge early repayment features into main_df
main_df = pd.merge(main_df, early_repayment_numerical_features[['loan_id', 'num_early_late_payments', 'total_early_underpaid_amount', 'avg_early_payment_ratio']], on='loan_id', how='left')
main_df = pd.merge(main_df, early_repayment_flags, on='loan_id', how='left')
main_df = pd.merge(main_df, first_repayment_timing, on='loan_id', how='left')

# Fill NaNs for loans with no early repayments (or no repayments at all)
early_features_to_fill_zero = ['num_early_late_payments', 'total_early_underpaid_amount', 'has_early_late_payment', 'has_early_underpayment']
main_df[early_features_to_fill_zero] = main_df[early_features_to_fill_zero].fillna(0)
main_df['avg_early_payment_ratio'] = main_df['avg_early_payment_ratio'].fillna(1.0) # Assume perfect payment if no early repayment data
main_df['days_to_first_repayment'] = main_df['days_to_first_repayment'].fillna(365 * 10) # A large number for loans with no repayments ever


# --- 5. Target Variable Creation (`is_bad_loan`) ---
# Calculate total expected repayment for each loan
# Following the simplified formula from hint for 'expected_amount_per_month', then multiplied by term
# total_expected_payment = expected_amount_per_month * loan_term_months
main_df['total_expected_repayment'] = (main_df['loan_amount'] / main_df['loan_term_months']) * (1 + main_df['interest_rate'] / 12) * main_df['loan_term_months']

# Aggregate total amount paid for each loan from all repayments
total_paid_per_loan = repayments_df.groupby('loan_id')['amount_paid'].sum().reset_index()
total_paid_per_loan.rename(columns={'amount_paid': 'total_amount_paid'}, inplace=True)

main_df = pd.merge(main_df, total_paid_per_loan, on='loan_id', how='left')

# Fill NaN for total_amount_paid if a loan never had any repayments (e.g., due to N_REPAYMENTS_MAX cutoff)
main_df['total_amount_paid'] = main_df['total_amount_paid'].fillna(0)

# Define 'is_bad_loan'
# If total paid is significantly less than total expected (e.g., < 90%)
BAD_LOAN_THRESHOLD_RATIO = 0.9
main_df['is_bad_loan'] = ((main_df['total_amount_paid'] / main_df['total_expected_repayment']) < BAD_LOAN_THRESHOLD_RATIO).astype(int)

# Filter out loans that are not present in the repayments_df at all (e.g., if N_REPAYMENTS_MAX was hit very early)
# Loans with total_expected_repayment = 0 (highly unlikely with generated data) would cause division by zero;
# if main_df['total_expected_repayment'] is 0, it should be filtered or handled.
# Assuming total_expected_repayment is always positive for valid loans.
main_df = main_df[main_df['loan_id'].isin(repayments_df['loan_id'].unique()) | (main_df['total_amount_paid'] == 0)].copy()

num_bad_loans = main_df['is_bad_loan'].sum()
print(f"Total loans evaluated: {len(main_df)}")
print(f"Number of bad loans (target=1): {num_bad_loans}")
print(f"Proportion of bad loans: {num_bad_loans / len(main_df):.2f}\n")

# --- 6. Final Feature Set Assembly and Preprocessing ---
# Select features and target
features = [
    'credit_score', 'income', 'loan_amount', 'loan_term_months', 'interest_rate',
    'loan_to_income_ratio', 'debt_to_income_ratio', 'days_from_application_to_disbursement',
    'num_early_late_payments', 'total_early_underpaid_amount', 'avg_early_payment_ratio',
    'has_early_late_payment', 'has_early_underpayment', 'days_to_first_repayment',
    'loan_purpose', 'employment_status'
]
target = 'is_bad_loan'

X = main_df[features]
y = main_df[target]

# Identify categorical and numerical features
categorical_features = ['loan_purpose', 'employment_status']
numerical_features = [col for col in features if col not in categorical_features]

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any, though none expected here)
)

# Create a machine learning pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("--- Model Evaluation ---")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
print(f"  True Negatives (TN): {conf_matrix[0, 0]}")
print(f"  False Positives (FP): {conf_matrix[0, 1]}")
print(f"  False Negatives (FN): {conf_matrix[1, 0]}")
print(f"  True Positives (TP): {conf_matrix[1, 1]}")

# Display feature importances if using RandomForestClassifier
if isinstance(model.named_steps['classifier'], RandomForestClassifier):
    print("\n--- Feature Importances (Top 10) ---")
    # Get feature names after one-hot encoding
    ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names)
    
    importances = model.named_steps['classifier'].feature_importances_
    feature_importances_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
    print(feature_importances_df.head(10).to_string(index=False))