import pandas as pd
import numpy as np
import sqlite3
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# --- 1. Synthetic Data Generation ---

# Set random seed for reproducibility
np.random.seed(42)

# Global variables for data generation
N_APPLICANTS = np.random.randint(1000, 1500)
N_LOANS = np.random.randint(1000, 1500)
N_PAYMENTS = np.random.randint(15000, 25000)
DEFAULT_RATE_BASE = 0.12 # Target base default rate

# Generate applicants_df
applicants_df = pd.DataFrame({
    'applicant_id': np.arange(N_APPLICANTS) + 1,
    'age': np.random.randint(20, 71, N_APPLICANTS),
    'income': np.random.uniform(2000, 15000, N_APPLICANTS).round(2),
    'credit_score': np.random.randint(300, 851, N_APPLICANTS),
    'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired'], N_APPLICANTS, p=[0.6, 0.2, 0.1, 0.1])
})

# Simulate correlations for loan generation
# Lower credit_score/income -> higher default risk
# Normalize credit_score and income to [0,1]
norm_credit_score = (applicants_df['credit_score'] - 300) / 550
norm_income = (applicants_df['income'] - 2000) / 13000

# Higher risk for lower credit score and income
applicants_df['default_risk_factor'] = (1 - norm_credit_score) * 0.6 + (1 - norm_income) * 0.4
applicants_df['default_risk_factor'] = np.clip(applicants_df['default_risk_factor'], 0, 1) # Ensure within [0,1]

# Generate loans_df
loan_ids = np.arange(N_LOANS) + 1
applicant_ids = np.random.choice(applicants_df['applicant_id'], N_LOANS, replace=True)

# Merge default risk factor for loan-level correlation
loans_pre_df = pd.DataFrame({'loan_id': loan_ids, 'applicant_id': applicant_ids})
loans_pre_df = loans_pre_df.merge(applicants_df[['applicant_id', 'default_risk_factor']], on='applicant_id', how='left')

loans_df = pd.DataFrame({
    'loan_id': loans_pre_df['loan_id'],
    'applicant_id': loans_pre_df['applicant_id'],
    'loan_amount': np.random.uniform(1000, 50000, N_LOANS).round(2),
    'loan_term_months': np.random.randint(12, 61, N_LOANS),
})

# Correlate interest rate with default risk factor
min_ir, max_ir = 0.05, 0.20
# Higher default risk factor -> higher interest rate
loans_df['interest_rate'] = min_ir + (max_ir - min_ir) * loans_pre_df['default_risk_factor'] + np.random.uniform(-0.02, 0.02, N_LOANS)
loans_df['interest_rate'] = np.clip(loans_df['interest_rate'], min_ir, max_ir).round(4)

# Generate loan_date
today = pd.to_datetime('today').normalize()
five_years_ago = today - pd.DateOffset(years=5)
loans_df['loan_date'] = pd.to_datetime([five_years_ago + pd.Timedelta(days=np.random.randint(0, (today - five_years_ago).days)) for _ in range(N_LOANS)])

# Generate default_date
# Probability of default biased by default_risk_factor
# Scale default_risk_factor to generate actual default probabilities around the base rate
default_probabilities = DEFAULT_RATE_BASE * (1 + loans_pre_df['default_risk_factor'])
default_probabilities = np.clip(default_probabilities, 0.01, 0.4) # Ensure some min/max default prob

will_default = np.random.rand(N_LOANS) < default_probabilities

default_dates_list = []
for i, row in loans_df.iterrows():
    if will_default[i]:
        # Default within 6 months to 2 years after loan_date, but not before today
        start_default_period = row['loan_date'] + pd.DateOffset(months=6)
        end_default_period = row['loan_date'] + pd.DateOffset(years=2)
        
        # Default date must be before or on today
        end_default_period = min(end_default_period, today)
        
        if start_default_period < end_default_period:
            # Generate a random date in the valid window
            days_in_period = (end_default_period - start_default_period).days
            if days_in_period > 0:
                default_date = start_default_period + pd.Timedelta(days=np.random.randint(0, days_in_period + 1))
            else:
                default_date = pd.NaT # Window is too short or invalid
        else: 
            default_date = pd.NaT # Start date is already past end date/today
    else:
        default_date = pd.NaT
    default_dates_list.append(default_date)

loans_df['default_date'] = default_dates_list

# Final check for default_date validity: must be after loan_date
loans_df.loc[loans_df['default_date'].notna() & (loans_df['default_date'] <= loans_df['loan_date']), 'default_date'] = pd.NaT

# Generate payments_df
payments_data = []
payment_id_counter = 1
for _ in range(N_PAYMENTS):
    loan_idx = np.random.randint(0, N_LOANS)
    loan_id = loans_df.loc[loan_idx, 'loan_id']
    loan_amount = loans_df.loc[loan_idx, 'loan_amount']
    loan_date = loans_df.loc[loan_idx, 'loan_date']
    default_date = loans_df.loc[loan_idx, 'default_date']
    loan_term_months = loans_df.loc[loan_idx, 'loan_term_months']

    # Payment date must be after loan_date
    start_payment_date = loan_date + pd.Timedelta(days=1)
    
    # And before default_date if it exists, otherwise up to today
    end_payment_date = default_date if pd.notna(default_date) else today
    
    if start_payment_date >= end_payment_date:
        continue # Skip payment if loan ended or defaulted too quickly or no valid window

    days_for_payment = (end_payment_date - start_payment_date).days
    if days_for_payment <= 0:
        continue # No valid days for payment

    payment_date = start_payment_date + pd.Timedelta(days=np.random.randint(0, days_for_payment + 1))

    # Simulate smaller payments before default
    # Base payment: monthly installment roughly
    paid_amount_base = loan_amount / loan_term_months + np.random.uniform(-50, 50)
    
    paid_amount_multiplier = np.random.uniform(0.8, 1.2) # Normal payments
    if pd.notna(default_date) and (default_date - payment_date).days < 180: # If within 6 months of default
        paid_amount_multiplier = np.random.uniform(0.3, 0.8) # Smaller payments

    paid_amount = round(paid_amount_base * paid_amount_multiplier, 2)
    paid_amount = np.clip(paid_amount, 10, 2000) # Ensure a reasonable range

    payments_data.append({
        'payment_id': payment_id_counter,
        'loan_id': loan_id,
        'payment_date': payment_date,
        'paid_amount': paid_amount
    })
    payment_id_counter += 1

payments_df = pd.DataFrame(payments_data)

# Sort payments_df
payments_df = payments_df.sort_values(by=['loan_id', 'payment_date']).reset_index(drop=True)

print("--- Synthetic Data Generated ---")
print("Applicants head:\n", applicants_df.head())
print("Loans head:\n", loans_df.head())
print("Payments head:\n", payments_df.head())
print(f"Total applicants: {len(applicants_df)}")
print(f"Total loans: {len(loans_df)}")
print(f"Total payments: {len(payments_df)}")
print(f"Number of defaulted loans (with a valid default_date): {loans_df['default_date'].count()}")


# --- 2. Load into SQLite & SQL Feature Engineering ---

conn = sqlite3.connect(':memory:')

# Convert date columns to string for SQLite
applicants_df.to_sql('applicants', conn, index=False, if_exists='replace')
loans_df['loan_date_str'] = loans_df['loan_date'].dt.strftime('%Y-%m-%d')
loans_df['default_date_str'] = loans_df['default_date'].dt.strftime('%Y-%m-%d')
loans_df.drop(columns=['loan_date', 'default_date']).to_sql('loans', conn, index=False, if_exists='replace')

payments_df['payment_date_str'] = payments_df['payment_date'].dt.strftime('%Y-%m-%d')
payments_df.drop(columns=['payment_date']).to_sql('payments', conn, index=False, if_exists='replace')

# Define GLOBAL_PREDICTION_CUTOFF_DATE
# Ensure payments_df is not empty before getting max date
if not payments_df.empty:
    GLOBAL_PREDICTION_CUTOFF_DATE = payments_df['payment_date'].max() - pd.Timedelta(months=3)
else:
    # Fallback to a sensible date if no payments were generated
    GLOBAL_PREDICTION_CUTOFF_DATE = today - pd.Timedelta(months=3) 
GLOBAL_PREDICTION_CUTOFF_DATE_STR = GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')

print(f"\n--- SQLite & SQL Feature Engineering ---")
print(f"Global Prediction Cutoff Date: {GLOBAL_PREDICTION_CUTOFF_DATE_STR}")

sql_query = f"""
WITH PaymentAggregates_6m AS (
    SELECT
        p.loan_id,
        COUNT(p.payment_id) AS num_payments_prev_6m,
        SUM(p.paid_amount) AS total_paid_prev_6m,
        AVG(p.paid_amount) AS avg_payment_prev_6m
    FROM payments p
    WHERE p.payment_date_str BETWEEN date('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}', '-6 months') AND '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}'
    GROUP BY p.loan_id
),
PaymentAggregates_AllTime AS (
    SELECT
        p.loan_id,
        MAX(p.payment_date_str) AS last_payment_date_at_cutoff,
        SUM(p.paid_amount) AS total_paid_at_cutoff
    FROM payments p
    WHERE p.payment_date_str <= '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}'
    GROUP BY p.loan_id
)
SELECT
    l.loan_id,
    l.applicant_id,
    l.loan_amount,
    l.loan_term_months,
    l.interest_rate,
    l.loan_date_str AS loan_date,
    a.age,
    a.income,
    a.credit_score,
    a.employment_status,
    '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' AS current_cutoff_date,
    COALESCE(pa6m.num_payments_prev_6m, 0) AS num_payments_prev_6m,
    COALESCE(pa6m.total_paid_prev_6m, 0.0) AS total_paid_prev_6m,
    COALESCE(pa6m.avg_payment_prev_6m, 0.0) AS avg_payment_prev_6m,
    CASE
        WHEN paat.last_payment_date_at_cutoff IS NOT NULL
        THEN JULIANDAY('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') - JULIANDAY(paat.last_payment_date_at_cutoff)
        ELSE 9999
    END AS days_since_last_payment_at_cutoff,
    COALESCE(l.loan_amount - COALESCE(paat.total_paid_at_cutoff, 0.0), l.loan_amount) AS outstanding_balance_at_cutoff
FROM loans l
LEFT JOIN applicants a ON l.applicant_id = a.applicant_id
LEFT JOIN PaymentAggregates_6m pa6m ON l.loan_id = pa6m.loan_id
LEFT JOIN PaymentAggregates_AllTime paat ON l.loan_id = paat.loan_id;
"""

loan_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("\nLoan Features from SQL head:\n", loan_features_df.head())
print(f"Number of loans after SQL feature engineering: {len(loan_features_df)}")


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n--- Pandas Feature Engineering & Target Creation ---")

# Convert date columns to datetime objects
loan_features_df['loan_date'] = pd.to_datetime(loan_features_df['loan_date'])
loan_features_df['current_cutoff_date'] = pd.to_datetime(loan_features_df['current_cutoff_date'])

# Handle NaN values for numerical aggregated features (already mostly handled by SQL COALESCE)
loan_features_df['num_payments_prev_6m'] = loan_features_df['num_payments_prev_6m'].fillna(0).astype(int)
loan_features_df['total_paid_prev_6m'] = loan_features_df['total_paid_prev_6m'].fillna(0.0)
loan_features_df['avg_payment_prev_6m'] = loan_features_df['avg_payment_prev_6m'].fillna(0.0)
loan_features_df['days_since_last_payment_at_cutoff'] = loan_features_df['days_since_last_payment_at_cutoff'].fillna(9999).astype(int)
loan_features_df['outstanding_balance_at_cutoff'] = loan_features_df['outstanding_balance_at_cutoff'].fillna(loan_features_df['loan_amount']) 

# Calculate loan_age_at_cutoff_days
loan_features_df['loan_age_at_cutoff_days'] = (loan_features_df['current_cutoff_date'] - loan_features_df['loan_date']).dt.days

# Calculate payment_frequency_prev_6m
# Use 180.0 to represent 6 months in days
loan_features_df['payment_frequency_prev_6m'] = loan_features_df['num_payments_prev_6m'] / 180.0
loan_features_df['payment_frequency_prev_6m'] = loan_features_df['payment_frequency_prev_6m'].replace([np.inf, -np.inf], 0).fillna(0)

# Create the Binary Target `will_default_in_next_90_days`
# Merge original default_date from loans_df (which has the true default_date values)
loan_features_df = loan_features_df.merge(loans_df[['loan_id', 'default_date']], on='loan_id', how='left', suffixes=('', '_original'))

default_window_start = loan_features_df['current_cutoff_date']
default_window_end = loan_features_df['current_cutoff_date'] + pd.Timedelta(days=90)

loan_features_df['will_default_in_next_90_days'] = (
    (loan_features_df['default_date'].notna()) &
    (loan_features_df['default_date'] > default_window_start) &
    (loan_features_df['default_date'] <= default_window_end)
).astype(int)

# Drop temporary default_date column
loan_features_df = loan_features_df.drop(columns=['default_date'])

print("\nLoan features with target variable head:\n", loan_features_df.head())
print("\nTarget variable distribution:\n", loan_features_df['will_default_in_next_90_days'].value_counts())

# Define features X and target y
numerical_features = [
    'loan_amount', 'loan_term_months', 'interest_rate', 'age', 'income',
    'credit_score', 'num_payments_prev_6m', 'total_paid_prev_6m',
    'avg_payment_prev_6m', 'days_since_last_payment_at_cutoff',
    'outstanding_balance_at_cutoff', 'loan_age_at_cutoff_days',
    'payment_frequency_prev_6m'
]
categorical_features = ['employment_status']

X = loan_features_df[numerical_features + categorical_features]
y = loan_features_df['will_default_in_next_90_days']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# --- 4. Data Visualization ---

print("\n--- Data Visualization ---")

plt.figure(figsize=(14, 6))

# Violin plot for credit_score vs. will_default_in_next_90_days
plt.subplot(1, 2, 1)
sns.violinplot(x='will_default_in_next_90_days', y='credit_score', data=loan_features_df, palette='viridis')
plt.title('Credit Score Distribution by Default Status')
plt.xlabel('Will Default in Next 90 Days (0: No, 1: Yes)')
plt.ylabel('Credit Score')

# Stacked bar chart for employment_status vs. will_default_in_next_90_days
plt.subplot(1, 2, 2)
# Calculate proportions
default_proportions = loan_features_df.groupby('employment_status')['will_default_in_next_90_days'].value_counts(normalize=True).unstack(fill_value=0)
default_proportions.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='coolwarm')
plt.title('Proportion of Default by Employment Status')
plt.xlabel('Employment Status')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Will Default', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---

print("\n--- ML Pipeline & Evaluation ---")

# Create ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create the ML pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Convert probabilities to binary predictions for classification report (using default threshold 0.5)
y_pred = (y_pred_proba > 0.5).astype(int)

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\n--- Pipeline execution complete ---")