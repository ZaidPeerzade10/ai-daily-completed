import pandas as pd
import numpy as np
import sqlite3
import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report


# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

print("--- 1. Generating Synthetic Data ---")

# 1. Generate Synthetic Data
num_applicants = np.random.randint(1000, 1501)
applicant_ids = np.arange(10000, 10000 + num_applicants)

# Generate application dates over the last 3 years
end_date = datetime.now()
start_date = end_date - timedelta(days=3 * 365)
application_dates = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(num_applicants)]
application_dates.sort() # Sort to simulate chronological applications for potential realism

loan_applicants_df = pd.DataFrame({
    'applicant_id': applicant_ids,
    'age': np.random.randint(18, 71, num_applicants),
    'income': np.random.uniform(20000, 200000, num_applicants),
    'education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], num_applicants, p=[0.3, 0.4, 0.2, 0.1]),
    'loan_amount_requested': np.random.uniform(5000, 100000, num_applicants),
    'loan_term_months': np.random.randint(12, 61, num_applicants),
    'credit_score_at_application': np.random.randint(300, 851, num_applicants),
    'application_date': application_dates,
    '_actual_default_status': 0 # Initialize all as non-default, then adjust
})

# Introduce realistic patterns for defaulters (~10-15%)
num_defaulters = int(num_applicants * np.random.uniform(0.10, 0.15))
defaulter_indices = np.random.choice(loan_applicants_df.index, num_defaulters, replace=False)
loan_applicants_df.loc[defaulter_indices, '_actual_default_status'] = 1

# Adjust defaulter characteristics
loan_applicants_df.loc[defaulter_indices, 'income'] = np.random.uniform(20000, 80000, num_defaulters) # Lower income
loan_applicants_df.loc[defaulter_indices, 'credit_score_at_application'] = np.random.randint(300, 650, num_defaulters) # Lower credit score
loan_applicants_df.loc[defaulter_indices, 'loan_amount_requested'] = np.random.uniform(50000, 150000, num_defaulters) # Higher loan requested

# Generate payment history
num_payments = np.random.randint(20000, 30001)
payment_history_data = []

# To ensure payment_date < application_date for each applicant,
# iterate through applicants and generate payments
for _, applicant_row in loan_applicants_df.iterrows():
    app_id = applicant_row['applicant_id']
    app_date = pd.to_datetime(applicant_row['application_date'])
    is_default = applicant_row['_actual_default_status']

    # Each applicant makes between 10 and 50 payments prior to application
    num_applicant_payments = np.random.randint(10, 51)
    
    # Define a historical window (e.g., up to 2 years before application)
    earliest_payment_date = app_date - timedelta(days=2 * 365) 

    for _ in range(num_applicant_payments):
        # Ensure payment_date is strictly before application_date
        # Randomly choose a date between earliest_payment_date and app_date - 1 day
        payment_date = earliest_payment_date + timedelta(days=random.randint(0, (app_date - earliest_payment_date).days - 1))
        
        amount_paid = np.random.uniform(100, 2000)
        
        # Defaulters tend to have more late payments
        late_payment_prob = 0.15 if is_default == 0 else 0.40 # Higher prob for defaulters
        is_late_payment = np.random.choice([0, 1], p=[1 - late_payment_prob, late_payment_prob])
        
        payment_history_data.append({
            'payment_id': len(payment_history_data),
            'applicant_id': app_id,
            'payment_date': payment_date,
            'amount_paid': amount_paid,
            'is_late_payment': is_late_payment
        })

payment_history_df = pd.DataFrame(payment_history_data)

# Ensure payment_history_df is sorted for clarity
payment_history_df = payment_history_df.sort_values(by=['applicant_id', 'payment_date']).reset_index(drop=True)

# Convert application_date to string for SQLite compatibility
loan_applicants_df['application_date'] = loan_applicants_df['application_date'].dt.strftime('%Y-%m-%d')
payment_history_df['payment_date'] = payment_history_df['payment_date'].dt.strftime('%Y-%m-%d')

print(f"Generated {len(loan_applicants_df)} loan applicants.")
print(f"Generated {len(payment_history_df)} payment records.")
print(f"Default rate: {loan_applicants_df['_actual_default_status'].mean():.2%}")
print("Sample of loan_applicants_df:")
print(loan_applicants_df.head())
print("\nSample of payment_history_df:")
print(payment_history_df.head())


print("\n--- 2. Loading into SQLite & SQL Feature Engineering ---")

# 2. Load into SQLite & SQL Feature Engineering
conn = sqlite3.connect(':memory:')
loan_applicants_df.to_sql('applicants', conn, index=False, if_exists='replace')
payment_history_df.to_sql('payments', conn, index=False, if_exists='replace')

sql_query = """
SELECT
    a.applicant_id,
    a.age,
    a.income,
    a.education,
    a.loan_amount_requested,
    a.loan_term_months,
    a.credit_score_at_application,
    a.application_date,
    a._actual_default_status,
    -- Time-based features from application_date
    CAST(STRFTIME('%w', a.application_date) AS INTEGER) AS day_of_week_application,
    CAST(STRFTIME('%m', a.application_date) AS INTEGER) AS month_of_application,
    -- Aggregated historical features (12 months prior to application)
    COALESCE(SUM(CASE WHEN p.is_late_payment = 1 THEN 1 ELSE 0 END), 0) AS num_late_payments_prev_12m_at_app,
    COALESCE(AVG(p.amount_paid), 0.0) AS avg_payment_amount_prev_12m_at_app,
    COALESCE(COUNT(p.payment_id), 0) AS num_payments_prev_12m_at_app,
    -- Days since last payment (before application date)
    -- Use a subquery to find MAX payment_date BEFORE application_date, then calculate difference
    COALESCE(
        CAST(JULIANDAY(a.application_date) - JULIANDAY((
            SELECT MAX(p_sub.payment_date)
            FROM payments p_sub
            WHERE p_sub.applicant_id = a.applicant_id
            AND p_sub.payment_date < a.application_date
        )) AS INTEGER),
        9999 -- Value if no prior payments
    ) AS days_since_last_payment_at_app
FROM
    applicants a
LEFT JOIN
    payments p ON a.applicant_id = p.applicant_id
    AND p.payment_date < a.application_date
    AND p.payment_date >= DATE(a.application_date, '-12 months') -- Payments within 12 months prior
GROUP BY
    a.applicant_id,
    a.age,
    a.income,
    a.education,
    a.loan_amount_requested,
    a.loan_term_months,
    a.credit_score_at_application,
    a.application_date,
    a._actual_default_status
ORDER BY
    a.applicant_id;
"""

loan_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Features DataFrame created with {len(loan_features_df)} rows and {len(loan_features_df.columns)} columns.")
print("Sample of loan_features_df (after SQL feature engineering):")
print(loan_features_df.head())


print("\n--- 3. Pandas Feature Engineering & Binary Target Creation ---")

# 3. Pandas Feature Engineering & Binary Target Creation
# Convert application_date to datetime objects
loan_features_df['application_date'] = pd.to_datetime(loan_features_df['application_date'])

# Handle NaNs: Fill numerical historical aggregated features with 0.0 or 0
loan_features_df['num_late_payments_prev_12m_at_app'].fillna(0, inplace=True)
loan_features_df['avg_payment_amount_prev_12m_at_app'].fillna(0.0, inplace=True)
loan_features_df['num_payments_prev_12m_at_app'].fillna(0, inplace=True)
loan_features_df['days_since_last_payment_at_app'].fillna(9999, inplace=True)

# Calculate debt_to_income_ratio
loan_features_df['debt_to_income_ratio'] = loan_features_df['loan_amount_requested'] / (loan_features_df['income'] + 1e-6)
# Handle any NaN or inf values from division by very small income
loan_features_df['debt_to_income_ratio'].replace([np.inf, -np.inf], 0, inplace=True)
loan_features_df['debt_to_income_ratio'].fillna(0, inplace=True)

# Define features X and target y
numerical_features = [
    'age', 'income', 'loan_amount_requested', 'loan_term_months',
    'credit_score_at_application', 'num_late_payments_prev_12m_at_app',
    'avg_payment_amount_prev_12m_at_app', 'num_payments_prev_12m_at_app',
    'days_since_last_payment_at_app', 'day_of_week_application',
    'month_of_application', 'debt_to_income_ratio'
]
categorical_features = ['education']

# Combine all feature names
all_features = numerical_features + categorical_features

X = loan_features_df[all_features].copy()
y = loan_features_df['_actual_default_status'].copy().rename('is_default')

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"Default rate in y_train: {y_train.mean():.2%}")
print(f"Default rate in y_test: {y_test.mean():.2%}")


print("\n--- 4. Data Visualization ---")

# 4. Data Visualization
plt.figure(figsize=(14, 6))

# Violin plot for credit_score_at_application vs. is_default
plt.subplot(1, 2, 1)
sns.violinplot(x=y, y=X['credit_score_at_application'], palette='viridis')
plt.title('Credit Score Distribution by Default Status')
plt.xlabel('Default Status (0: No Default, 1: Default)')
plt.ylabel('Credit Score at Application')

# Stacked bar chart for proportion of is_default across education levels
plt.subplot(1, 2, 2)
education_default_prop = loan_features_df.groupby('education')['_actual_default_status'].value_counts(normalize=True).unstack().fillna(0)
education_default_prop.plot(kind='bar', stacked=True, color=['lightgreen', 'salmon'], ax=plt.gca())
plt.title('Proportion of Default Status by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Default Status', labels=['No Default', 'Default'])
plt.tight_layout()
plt.show()


print("\n--- 5. ML Pipeline & Evaluation ---")

# 5. ML Pipeline & Evaluation
# Preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Impute NaNs with mean (for income/credit_score not already filled)
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, though not expected here
)

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42, class_weight='balanced'))
])

# Train the pipeline
print("Training the model pipeline...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
y_pred = model_pipeline.predict(X_test) # For classification report

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred)

print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:")
print(class_report)

print("\n--- Pipeline execution complete ---")