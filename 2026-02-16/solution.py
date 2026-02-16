import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


# --- 1. Generate Synthetic Data ---
print("--- Generating Synthetic Data ---")

# General parameters
N_APPLICANTS = np.random.randint(500, 701)
N_LOANS = np.random.randint(1000, 1501)
N_PAYMENTS = np.random.randint(5000, 8001)
# Fixed 'today' for reproducibility, simulating a current date for data generation
TODAY = pd.to_datetime('2023-10-26') 

# Applicants DataFrame
applicants_data = {
    'applicant_id': np.arange(N_APPLICANTS),
    'application_date': TODAY - pd.to_timedelta(np.random.randint(0, 5 * 365, N_APPLICANTS), unit='D'),
    'age': np.random.randint(18, 71, N_APPLICANTS),
    'income': np.random.uniform(25000, 200000, N_APPLICANTS).round(2),
    # Credit score biased towards higher scores (using beta distribution)
    'credit_score': np.round(np.random.beta(a=5, b=2, size=N_APPLICANTS) * (850 - 300) + 300).astype(int),
    'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired', 'Student'], N_APPLICANTS),
    'residence_type': np.random.choice(['Rent', 'Own', 'Mortgage', 'Other'], N_APPLICANTS)
}
applicants_df = pd.DataFrame(applicants_data)

# Loans DataFrame
loans_data = {
    'loan_id': np.arange(N_LOANS),
    # Sample applicant_ids, allowing some applicants to have multiple loans
    'applicant_id': np.random.choice(applicants_df['applicant_id'], N_LOANS, replace=True),
    'loan_amount': np.random.uniform(5000, 50000, N_LOANS).round(2),
    'interest_rate': np.random.uniform(0.05, 0.20, N_LOANS).round(4),
    'loan_term_months': np.random.randint(12, 61, N_LOANS),
    'loan_type': np.random.choice(['Personal', 'Auto', 'Home_Equity', 'Business', 'Education'], N_LOANS),
}
loans_df = pd.DataFrame(loans_data)

# Merge application_date and credit_score for disbursement_date and default logic
loans_df = loans_df.merge(applicants_df[['applicant_id', 'application_date', 'credit_score']], on='applicant_id', how='left')

# Disbursement date always after application_date, and not too far in the future/past
loans_df['disbursement_date'] = loans_df.apply(lambda row: row['application_date'] + pd.to_timedelta(np.random.randint(10, 91), unit='D'), axis=1)
# Ensure disbursement date allows for at least a few months of payments before TODAY
loans_df['disbursement_date'] = loans_df['disbursement_date'].apply(lambda d: min(d, TODAY - pd.Timedelta(days=90)))

# Generate is_default with bias
default_rate_target = 0.12 # Approximate 10-15% default rate
loans_df['is_default'] = 0

# Factors influencing default: lower credit score, higher interest rate
# Create a 'propensity to default' score for each loan
loans_df['default_propensity'] = (1 - (loans_df['credit_score'] - 300) / (850 - 300)) * 0.7 + \
                                 (loans_df['interest_rate'] - 0.05) / (0.20 - 0.05) * 0.3

# Scale propensity to match desired default rate by setting a threshold
sorted_propensity = loans_df['default_propensity'].sort_values(ascending=False)
threshold_index = int(len(loans_df) * default_rate_target)
if threshold_index > 0:
    propensity_threshold = sorted_propensity.iloc[threshold_index - 1]
    loans_df['is_default'] = (loans_df['default_propensity'] >= propensity_threshold).astype(int)
else: # Fallback if for some reason threshold_index is 0
    loans_df['is_default'] = (np.random.rand(len(loans_df)) < default_rate_target).astype(int)

# Drop temporary columns used for generating default status
loans_df = loans_df.drop(columns=['application_date', 'credit_score', 'default_propensity'])

# Payments DataFrame
payments_list = []
payment_id_counter = 0

# Determine an effective maximum payment date (e.g., up to TODAY)
MAX_PAYMENT_DATE = TODAY

for idx, loan in loans_df.iterrows():
    loan_id = loan['loan_id']
    disbursement_date = loan['disbursement_date']
    loan_amount = loan['loan_amount']
    loan_term_months = loan['loan_term_months']
    is_default = loan['is_default']
    
    # Calculate expected monthly payment
    monthly_payment = loan_amount / loan_term_months
    
    # Determine the actual number of payments to simulate
    num_payments_to_generate = loan_term_months
    
    # If defaulted, payments might stop earlier
    if is_default:
        # Defaulted loans pay for a random fraction of the term, e.g., 30-70%
        # Ensure at least 1 payment for defaulted loans if possible
        min_default_months = max(1, int(loan_term_months * 0.3))
        max_default_months = int(loan_term_months * 0.7)
        if min_default_months > max_default_months: # Edge case for very short terms
            effective_default_month = min_default_months
        else:
            effective_default_month = np.random.randint(min_default_months, max_default_months + 1)
        num_payments_to_generate = min(num_payments_to_generate, effective_default_month)

    # Generate payments for this loan
    for i in range(num_payments_to_generate):
        # Approximate monthly payment date
        payment_date = disbursement_date + pd.to_timedelta((i + 1) * 30, unit='D') 
        
        # Ensure payment date is not in the future beyond MAX_PAYMENT_DATE
        if payment_date > MAX_PAYMENT_DATE:
            break
            
        amount_paid = monthly_payment * np.random.uniform(0.9, 1.1) # Some variance around expected payment
        
        is_late = 0
        late_prob = 0.05 # Base late probability for non-defaulted loans
        if is_default:
            # Higher chance of late, increasing towards the point of default
            late_prob = 0.25 + (i / max(1, num_payments_to_generate)) * 0.2 
        
        if np.random.rand() < late_prob:
            is_late = 1
            # Late payments might also be slightly less than expected
            amount_paid = monthly_payment * np.random.uniform(0.7, 1.05) if np.random.rand() < 0.5 else amount_paid 
            
        payments_list.append({
            'payment_id': payment_id_counter,
            'loan_id': loan_id,
            'payment_date': payment_date,
            'amount_paid': amount_paid.round(2),
            'is_late': is_late
        })
        payment_id_counter += 1

payments_df = pd.DataFrame(payments_list)

# If no payments were generated (highly unlikely with current logic, but as a safeguard)
if payments_df.empty:
    print("Warning: No payments were generated. Creating a single dummy payment.")
    first_loan = loans_df.iloc[0]
    payments_df = pd.DataFrame([{
        'payment_id': 0, 'loan_id': first_loan['loan_id'],
        'payment_date': first_loan['disbursement_date'] + pd.Timedelta(days=30),
        'amount_paid': (first_loan['loan_amount'] / first_loan['loan_term_months']).round(2),
        'is_late': 0
    }])

# Ensure date types are correct
applicants_df['application_date'] = pd.to_datetime(applicants_df['application_date'])
loans_df['disbursement_date'] = pd.to_datetime(loans_df['disbursement_date'])
payments_df['payment_date'] = pd.to_datetime(payments_df['payment_date'])

print(f"Generated {len(applicants_df)} applicants, {len(loans_df)} loans, {len(payments_df)} payments.")
print(f"Actual default rate in generated loans: {loans_df['is_default'].mean():.2%}")
print("\nApplicants head:")
print(applicants_df.head())
print("\nLoans head:")
print(loans_df.head())
print("\nPayments head:")
print(payments_df.head())


# --- 2. Load into SQLite & SQL Feature Engineering ---
print("\n--- Loading data into SQLite and SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

applicants_df.to_sql('applicants', conn, if_exists='replace', index=False)
loans_df.to_sql('loans', conn, if_exists='replace', index=False)
payments_df.to_sql('payments', conn, if_exists='replace', index=False)

# Determine global_analysis_date and feature_cutoff_date
global_analysis_date_pd = payments_df['payment_date'].max() + pd.Timedelta(days=60)
feature_cutoff_date_pd = global_analysis_date_pd - pd.Timedelta(days=180)

global_analysis_date_str = global_analysis_date_pd.strftime('%Y-%m-%d')
feature_cutoff_date_str = feature_cutoff_date_pd.strftime('%Y-%m-%d')

print(f"Global Analysis Date: {global_analysis_date_str}")
print(f"Feature Cutoff Date: {feature_cutoff_date_str}")

sql_query = f"""
SELECT
    l.loan_id,
    a.applicant_id,
    a.age,
    a.income,
    a.credit_score,
    a.employment_status,
    a.residence_type,
    l.loan_amount,
    l.interest_rate,
    l.loan_term_months,
    l.loan_type,
    l.disbursement_date,
    COALESCE(COUNT(p.payment_id), 0) AS num_payments_pre_cutoff,
    COALESCE(SUM(p.amount_paid), 0.0) AS total_amount_paid_pre_cutoff,
    COALESCE(AVG(p.amount_paid), 0.0) AS avg_payment_value_pre_cutoff,
    COALESCE(SUM(CASE WHEN p.is_late = 1 THEN 1 ELSE 0 END), 0) AS num_late_payments_pre_cutoff,
    CASE
        WHEN MAX(p.payment_date) IS NOT NULL 
             THEN JULIANDAY('{feature_cutoff_date_str}') - JULIANDAY(MAX(p.payment_date))
        ELSE NULL
    END AS days_since_last_payment_pre_cutoff,
    -- Calculate loan age at cutoff in days; ensure it's not negative
    MAX(0, JULIANDAY('{feature_cutoff_date_str}') - JULIANDAY(l.disbursement_date)) AS loan_age_at_cutoff_days
FROM
    loans l
LEFT JOIN
    applicants a ON l.applicant_id = a.applicant_id
LEFT JOIN
    payments p ON l.loan_id = p.loan_id AND p.payment_date < '{feature_cutoff_date_str}'
GROUP BY
    l.loan_id, a.applicant_id, a.age, a.income, a.credit_score, a.employment_status,
    a.residence_type, l.loan_amount, l.interest_rate, l.loan_term_months, l.loan_type,
    l.disbursement_date
ORDER BY
    l.loan_id;
"""

loan_features_df = pd.read_sql_query(sql_query, conn)
print("\nSQL Feature Engineering Results Head:")
print(loan_features_df.head())


# --- 3. Pandas Feature Engineering & Binary Target Creation ---
print("\n--- Pandas Feature Engineering & Binary Target Creation ---")

# Handle NaN values resulting from LEFT JOIN in SQL
loan_features_df['num_payments_pre_cutoff'] = loan_features_df['num_payments_pre_cutoff'].fillna(0)
loan_features_df['total_amount_paid_pre_cutoff'] = loan_features_df['total_amount_paid_pre_cutoff'].fillna(0.0)
loan_features_df['avg_payment_value_pre_cutoff'] = loan_features_df['avg_payment_value_pre_cutoff'].fillna(0.0)
loan_features_df['num_late_payments_pre_cutoff'] = loan_features_df['num_late_payments_pre_cutoff'].fillna(0)

# For days_since_last_payment_pre_cutoff, fill NaN (meaning no payments before cutoff)
# with a value indicating "no payment since loan start" or a large sentinel.
# Using loan_age_at_cutoff_days + 30 makes sense: if no payments, it's at least loan_age_at_cutoff_days since a payment
loan_features_df['days_since_last_payment_pre_cutoff'] = loan_features_df['days_since_last_payment_pre_cutoff'].fillna(
    loan_features_df['loan_age_at_cutoff_days'] + 30 
).clip(lower=0) # Ensure no negative values after fillna

# Convert disbursement_date to datetime objects for consistency
loan_features_df['disbursement_date'] = pd.to_datetime(loan_features_df['disbursement_date'])

# Calculate payment_frequency_pre_cutoff
# Add 1 to denominator to prevent division by zero for very new loans (loan_age_at_cutoff_days = 0)
loan_features_df['payment_frequency_pre_cutoff'] = loan_features_df['num_payments_pre_cutoff'] / (
    loan_features_df['loan_age_at_cutoff_days'].replace(0, 1) + 1e-6 # Add small epsilon to prevent issues with 0
)
loan_features_df['payment_frequency_pre_cutoff'] = loan_features_df['payment_frequency_pre_cutoff'].fillna(0)

# Calculate ratio_late_payments_pre_cutoff
# Replace 0 num_payments_pre_cutoff with 1 to avoid division by zero
loan_features_df['ratio_late_payments_pre_cutoff'] = loan_features_df['num_late_payments_pre_cutoff'] / (
    loan_features_df['num_payments_pre_cutoff'].replace(0, 1)
)
loan_features_df['ratio_late_payments_pre_cutoff'] = loan_features_df['ratio_late_payments_pre_cutoff'].fillna(0)


# Create the Binary Target `is_default` by merging from original loans_df
loan_features_df = loan_features_df.merge(loans_df[['loan_id', 'is_default']], on='loan_id', how='left')

print("\nLoan Features DataFrame head after Pandas FE:")
print(loan_features_df.head())
print(f"Number of loans after feature engineering: {len(loan_features_df)}")
print(f"Default rate in feature engineered data: {loan_features_df['is_default'].mean():.2%}")

# Define features `X` and target `y`
numerical_features = [
    'age', 'income', 'credit_score', 'loan_amount', 'interest_rate', 'loan_term_months',
    'loan_age_at_cutoff_days', 'num_payments_pre_cutoff', 'total_amount_paid_pre_cutoff',
    'avg_payment_value_pre_cutoff', 'num_late_payments_pre_cutoff',
    'days_since_last_payment_pre_cutoff', 'payment_frequency_pre_cutoff',
    'ratio_late_payments_pre_cutoff'
]
categorical_features = [
    'employment_status', 'residence_type', 'loan_type'
]

X = loan_features_df[numerical_features + categorical_features]
y = loan_features_df['is_default']

# Split into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Training default rate: {y_train.mean():.2%}")
print(f"Test default rate: {y_test.mean():.2%}")


# --- 4. Data Visualization ---
print("\n--- Data Visualization ---")

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Violin plot showing distribution of credit_score for defaulted vs non-defaulted loans
sns.violinplot(x='is_default', y='credit_score', data=loan_features_df, ax=axes[0])
axes[0].set_title('Credit Score Distribution by Loan Default Status')
axes[0].set_xlabel('Is Default (0: No, 1: Yes)')
axes[0].set_ylabel('Credit Score')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Non-Defaulted', 'Defaulted'])

# Stacked bar chart showing proportion of is_default across different loan_type values
# Calculate proportions of default status within each loan type
default_by_loan_type = loan_features_df.groupby('loan_type')['is_default'].value_counts(normalize=True).unstack()
default_by_loan_type.plot(kind='bar', stacked=True, ax=axes[1], cmap='viridis')
axes[1].set_title('Proportion of Default by Loan Type')
axes[1].set_xlabel('Loan Type')
axes[1].set_ylabel('Proportion')
axes[1].tick_params(axis='x', rotation=45, ha='right')
axes[1].legend(title='Is Default', labels=['Non-Defaulted', 'Defaulted'])

plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---
print("\n--- ML Pipeline & Evaluation ---")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Impute missing numerical values with the mean
    ('scaler', StandardScaler())                 # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full machine learning pipeline
# 1. Preprocessing (imputation, scaling, one-hot encoding)
# 2. Classifier (Logistic Regression with balanced class weights)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'))
])

# Train the pipeline on the training data
print("Training the Logistic Regression model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Predict class labels on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model performance
print("\n--- Model Evaluation on Test Set ---")
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n--- Script Finished ---")