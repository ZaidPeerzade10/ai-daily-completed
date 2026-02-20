import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer  # Corrected import path for SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Ensure reproducibility
np.random.seed(42)

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

print("--- Generating Synthetic Data ---")

# 1.1 users_df
num_users = np.random.randint(500, 701)
signup_start_date = datetime.date(2018, 1, 1)
signup_end_date = datetime.date(2023, 1, 1)
days_in_signup_range = (signup_end_date - signup_start_date).days

users_df = pd.DataFrame({
    'user_id': np.arange(num_users),
    'signup_date': [signup_start_date + datetime.timedelta(days=np.random.randint(days_in_signup_range)) for _ in range(num_users)],
    'region': np.random.choice(['North', 'South', 'East', 'West'], num_users),
    'age': np.random.randint(18, 71, num_users),
    'is_fraudulent': np.random.choice([0, 1], num_users, p=[0.92, 0.08]) # Approx 8% fraud rate
})
users_df['signup_date'] = pd.to_datetime(users_df['signup_date']) # Convert to datetime objects

print(f"Generated {len(users_df)} users.")
print(users_df.head())

# 1.2 transactions_df
num_transactions_target = np.random.randint(5000, 8001)
transactions_data = []

# Simulate transaction patterns for each user
for _, user_row in users_df.iterrows():
    user_id = user_row['user_id']
    signup_date = user_row['signup_date']
    is_fraudulent = user_row['is_fraudulent']

    num_user_transactions = np.random.randint(5, 25) # Base transactions per user
    if is_fraudulent:
        num_user_transactions = np.random.randint(10, 50) # More transactions for fraudulent users
        
    for i in range(num_user_transactions):
        # Ensure transaction_date is after signup_date
        days_after_signup_max = (datetime.datetime.now() - datetime.timedelta(days=30) - signup_date).days
        if days_after_signup_max < 1: days_after_signup_max = 1 # Ensure at least 1 day

        days_after_signup = np.random.randint(1, min(days_after_signup_max, 365 * 3)) # Up to 3 years after signup, not into future

        if is_fraudulent:
            # Fraudulent patterns: more concentrated activity closer to signup
            days_after_signup = np.random.randint(1, min(days_after_signup_max, 90)) # Within first 3 months
            if np.random.rand() < 0.3: # Burst activity (multiple transactions on same/next day)
                days_after_signup = np.random.randint(1, min(days_after_signup_max, 7)) # Very short window
        
        transaction_date = signup_date + datetime.timedelta(days=days_after_signup)
        
        amount = np.random.uniform(10.0, 500.0) # Base amount
        merchant_category = np.random.choice(['Groceries', 'Retail', 'Dining', 'Travel', 'Online_Service'])
        location_country = np.random.choice(['USA', 'Canada', 'UK', 'Mexico', 'Japan'])

        if is_fraudulent:
            # Higher amounts / very large amounts
            if np.random.rand() < 0.4: # 40% chance of higher amount for fraud
                amount = np.random.uniform(200.0, 5000.0)
            if np.random.rand() < 0.1: # 10% chance of very large amount
                amount = np.random.uniform(2000.0, 15000.0) # Significantly higher

            # More diverse location countries for fraudulent users
            if np.random.rand() < 0.5 and i % 3 == 0: # Change country frequently
                location_country = np.random.choice(['USA', 'Canada', 'UK', 'Mexico', 'Japan', 'Germany', 'France', 'Australia'])


        transactions_data.append({
            'user_id': user_id,
            'transaction_date': transaction_date,
            'amount': amount,
            'merchant_category': merchant_category,
            'location_country': location_country
        })

transactions_df = pd.DataFrame(transactions_data)
# Filter down to the desired number of transactions if too many were generated
if len(transactions_df) > num_transactions_target:
    transactions_df = transactions_df.sample(n=num_transactions_target, random_state=42).reset_index(drop=True)
else: # If too few, resample existing with replacement
    transactions_df = transactions_df.sample(n=num_transactions_target, replace=True, random_state=42).reset_index(drop=True)


transactions_df['transaction_id'] = np.arange(len(transactions_df))
transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])

# Sort transactions_df
transactions_df = transactions_df.sort_values(by=['user_id', 'transaction_date']).reset_index(drop=True)

print(f"Generated {len(transactions_df)} transactions.")
print(transactions_df.head())

# --- 2. Load into SQLite & SQL Feature Engineering ---
print("\n--- Loading Data into SQLite and SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

users_df.to_sql('users', conn, index=False, if_exists='replace')
transactions_df.to_sql('transactions', conn, index=False, if_exists='replace')

# Determine global_analysis_date and feature_cutoff_date
max_transaction_date = transactions_df['transaction_date'].max()
global_analysis_date = max_transaction_date + pd.Timedelta(days=60)
feature_cutoff_date = global_analysis_date - pd.Timedelta(days=90)

print(f"Global Analysis Date: {global_analysis_date.strftime('%Y-%m-%d')}")
print(f"Feature Cutoff Date: {feature_cutoff_date.strftime('%Y-%m-%d')}")

# SQL Query
sql_query = f"""
SELECT
    u.user_id,
    u.age,
    u.region,
    u.signup_date,
    u.is_fraudulent,
    COALESCE(SUM(t.amount), 0.0) AS total_spend_pre_cutoff,
    COALESCE(COUNT(t.transaction_id), 0) AS num_transactions_pre_cutoff,
    COALESCE(AVG(t.amount), 0.0) AS avg_transaction_value_pre_cutoff,
    COALESCE(MAX(t.amount), 0.0) AS max_transaction_value_pre_cutoff,
    COALESCE(COUNT(DISTINCT t.merchant_category), 0) AS num_unique_merchant_categories_pre_cutoff,
    COALESCE(COUNT(DISTINCT t.location_country), 0) AS num_unique_location_countries_pre_cutoff,
    CASE
        WHEN MAX(t.transaction_date) IS NULL THEN NULL
        ELSE CAST(strftime('%J', '{feature_cutoff_date.strftime('%Y-%m-%d')}') - strftime('%J', MAX(t.transaction_date)) AS INTEGER)
    END AS days_since_last_transaction_pre_cutoff,
    CASE
        WHEN MIN(t.transaction_date) = MAX(t.transaction_date) THEN 0
        WHEN MIN(t.transaction_date) IS NULL THEN NULL
        ELSE CAST(strftime('%J', MAX(t.transaction_date)) - strftime('%J', MIN(t.transaction_date)) AS INTEGER)
    END AS transaction_span_days_pre_cutoff
FROM
    users u
LEFT JOIN
    transactions t ON u.user_id = t.user_id
                 AND t.transaction_date < '{feature_cutoff_date.strftime('%Y-%m-%d')}'
GROUP BY
    u.user_id, u.age, u.region, u.signup_date, u.is_fraudulent
ORDER BY
    u.user_id;
"""

user_fraud_features_df_sql = pd.read_sql_query(sql_query, conn)
print("\nSQL Feature Engineering Results (head):")
print(user_fraud_features_df_sql.head())

# --- 3. Pandas Feature Engineering & Binary Target Creation ---
print("\n--- Pandas Feature Engineering ---")

user_fraud_features_df = user_fraud_features_df_sql.copy()

# Handle NaN values
user_fraud_features_df['days_since_last_transaction_pre_cutoff'] = user_fraud_features_df['days_since_last_transaction_pre_cutoff'].fillna(9999)
user_fraud_features_df['transaction_span_days_pre_cutoff'] = user_fraud_features_df['transaction_span_days_pre_cutoff'].fillna(0)

# Convert signup_date to datetime objects
user_fraud_features_df['signup_date'] = pd.to_datetime(user_fraud_features_df['signup_date'])

# Calculate account_age_at_cutoff_days
user_fraud_features_df['account_age_at_cutoff_days'] = (feature_cutoff_date - user_fraud_features_df['signup_date']).dt.days
# Ensure minimum of 1 day to avoid issues with very new accounts or division by zero in frequency calc
user_fraud_features_df['account_age_at_cutoff_days'] = user_fraud_features_df['account_age_at_cutoff_days'].apply(lambda x: max(x, 1)) 

# Calculate transaction_frequency_pre_cutoff
user_fraud_features_df['transaction_frequency_pre_cutoff'] = user_fraud_features_df['num_transactions_pre_cutoff'] / (user_fraud_features_df['account_age_at_cutoff_days'] + 1)

# Calculate avg_transaction_per_span_pre_cutoff
user_fraud_features_df['avg_transaction_per_span_pre_cutoff'] = user_fraud_features_df.apply(
    lambda row: row['num_transactions_pre_cutoff'] / (row['transaction_span_days_pre_cutoff'] + 1)
    if row['transaction_span_days_pre_cutoff'] > 0 else 0, axis=1
)

print(user_fraud_features_df.head())
print(f"DataFrame shape after Pandas FE: {user_fraud_features_df.shape}")

# Define features X and target y
features = [
    'age', 'region', 'total_spend_pre_cutoff', 'num_transactions_pre_cutoff',
    'avg_transaction_value_pre_cutoff', 'max_transaction_value_pre_cutoff',
    'num_unique_merchant_categories_pre_cutoff', 'num_unique_location_countries_pre_cutoff',
    'days_since_last_transaction_pre_cutoff', 'transaction_span_days_pre_cutoff',
    'account_age_at_cutoff_days', 'transaction_frequency_pre_cutoff',
    'avg_transaction_per_span_pre_cutoff'
]
X = user_fraud_features_df[features]
y = user_fraud_features_df['is_fraudulent']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nTrain set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Fraudulent cases in training set: {y_train.sum()} ({y_train.mean():.2%})")
print(f"Fraudulent cases in test set: {y_test.sum()} ({y_test.mean():.2%})")

# --- 4. Data Visualization ---
print("\n--- Data Visualization ---")

# Plot 1: Box plot of total_spend_pre_cutoff by is_fraudulent
plt.figure(figsize=(8, 6))
# Using pandas boxplot directly on the dataframe for convenience
user_fraud_features_df.boxplot(column='total_spend_pre_cutoff', by='is_fraudulent', ax=plt.gca(), grid=False)
plt.suptitle('') # Suppress the default suptitle from pandas boxplot
plt.title('Distribution of Total Spend Pre-Cutoff by Fraud Status')
plt.xlabel('Is Fraudulent (0: No, 1: Yes)')
plt.ylabel('Total Spend Pre-Cutoff')
plt.yscale('log') # Use log scale due to potential large range
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
plt.close()

# Plot 2: Stacked bar chart of proportion of is_fraudulent across regions
plt.figure(figsize=(10, 6))
region_fraud_proportion = pd.crosstab(user_fraud_features_df['region'], user_fraud_features_df['is_fraudulent'], normalize='index')
region_fraud_proportion.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='coolwarm')
plt.title('Proportion of Fraudulent Users by Region')
plt.xlabel('Region')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Is Fraudulent', labels=['Not Fraudulent', 'Fraudulent'])
plt.tight_layout()
plt.show()
plt.close()

# --- 5. ML Pipeline & Evaluation (Binary Classification) ---
print("\n--- ML Pipeline & Evaluation ---")

# Define numerical and categorical features for preprocessing
numerical_features = [
    'age', 'total_spend_pre_cutoff', 'num_transactions_pre_cutoff',
    'avg_transaction_value_pre_cutoff', 'max_transaction_value_pre_cutoff',
    'num_unique_merchant_categories_pre_cutoff', 'num_unique_location_countries_pre_cutoff',
    'days_since_last_transaction_pre_cutoff', 'transaction_span_days_pre_cutoff',
    'account_age_at_cutoff_days', 'transaction_frequency_pre_cutoff',
    'avg_transaction_per_span_pre_cutoff'
]
categorical_features = ['region']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, though not expected here
)

# Create the full pipeline with preprocessing and a classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the model
print("\nTraining the ML model...")
model.fit(X_train, y_train)
print("Model training complete.")

# Predict probabilities and classes on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred, target_names=['Not Fraudulent (0)', 'Fraudulent (1)'])

print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:")
print(class_report)

print("\n--- Script Finished ---")