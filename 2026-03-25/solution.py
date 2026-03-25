import pandas as pd
import numpy as np
import datetime
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings

# Suppress warnings for cleaner output, specifically for plot layouts
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data ---

# Parameters for data generation
N_USERS = np.random.randint(500, 701)
N_MERCHANTS = np.random.randint(100, 201)
N_TRANSACTIONS = np.random.randint(10000, 15001)
FRAUD_RATE_TARGET = 0.04 # Target overall fraud rate

# 1.1. Users DataFrame
user_ids = np.arange(1, N_USERS + 1)
signup_dates = pd.to_datetime('now') - pd.to_timedelta(np.random.randint(0, 5*365, size=N_USERS), unit='D')
user_tiers = np.random.choice(['Bronze', 'Silver', 'Gold'], size=N_USERS, p=[0.4, 0.35, 0.25])
regions = np.random.choice(['North', 'South', 'East', 'West', 'International'], size=N_USERS, p=[0.2, 0.2, 0.2, 0.2, 0.2])
users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': signup_dates,
    'user_tier': user_tiers,
    'region': regions
})

# 1.2. Merchants DataFrame
merchant_ids = np.arange(1, N_MERCHANTS + 1)
categories = np.random.choice(['Electronics', 'Travel', 'Groceries', 'Utilities', 'Online_Services', 'Fashion', 'Dining'], size=N_MERCHANTS, p=[0.15, 0.1, 0.2, 0.1, 0.2, 0.15, 0.1])
risk_scores = np.random.randint(0, 101, size=N_MERCHANTS)
is_international_merchant = np.random.choice([0, 1], size=N_MERCHANTS, p=[0.8, 0.2])
merchants_df = pd.DataFrame({
    'merchant_id': merchant_ids,
    'category': categories,
    'risk_score': risk_scores,
    'is_international': is_international_merchant
})

# 1.3. Transactions DataFrame
transactions_df = pd.DataFrame({
    'transaction_id': np.arange(1, N_TRANSACTIONS + 1),
    'user_id': np.random.choice(users_df['user_id'], size=N_TRANSACTIONS),
    'merchant_id': np.random.choice(merchants_df['merchant_id'], size=N_TRANSACTIONS),
    'amount': np.round(np.random.uniform(10, 5000, size=N_TRANSACTIONS), 2),
    'transaction_type': np.random.choice(['Card_Present', 'Online', 'ATM_Withdrawal', 'Mobile_App'], size=N_TRANSACTIONS, p=[0.3, 0.4, 0.1, 0.2]),
})

# Merge user signup dates to transactions_df to ensure transaction_date > signup_date
transactions_df = transactions_df.merge(users_df[['user_id', 'signup_date']], on='user_id', how='left')

# Generate transaction_date after signup_date for each user
def generate_random_date_after_signup(row):
    start_date = row['signup_date']
    end_date = pd.to_datetime('now')
    # Ensure there's at least one day difference
    if start_date >= end_date:
        return end_date
    return start_date + pd.to_timedelta(np.random.randint(1, (end_date - start_date).days + 1), unit='D')

transactions_df['transaction_date'] = transactions_df.apply(generate_random_date_after_signup, axis=1)

# Sort transactions by user_id and transaction_date for sequential fraud simulation and SQL processing
transactions_df = transactions_df.sort_values(by=['user_id', 'transaction_date']).reset_index(drop=True)

# Merge static user and merchant data into transactions for fraud simulation
transactions_df = transactions_df.merge(users_df[['user_id', 'user_tier', 'region']], on='user_id', how='left')
transactions_df = transactions_df.merge(merchants_df[['merchant_id', 'category', 'risk_score', 'is_international']], on='merchant_id', how='left')

# 1.4. Simulate realistic fraud patterns
# Initialize base fraud probability
transactions_df['fraud_prob'] = FRAUD_RATE_TARGET * 0.5 # Start with a lower base probability

# Apply static biases
transactions_df.loc[transactions_df['user_tier'] == 'Bronze', 'fraud_prob'] *= 2.0
transactions_df.loc[transactions_df['region'] == 'International', 'fraud_prob'] *= 1.5
transactions_df.loc[transactions_df['risk_score'] > 70, 'fraud_prob'] *= 2.0
transactions_df.loc[transactions_df['is_international'] == 1, 'fraud_prob'] *= 1.5
transactions_df.loc[transactions_df['amount'] > 1000, 'fraud_prob'] *= 2.0
transactions_df.loc[transactions_df['transaction_type'] == 'Online', 'fraud_prob'] *= 1.5

# Cap probability to a reasonable range
transactions_df['fraud_prob'] = transactions_df['fraud_prob'].clip(0.005, 0.7)

# Apply sequential fraud patterns (approximate vectorized simulation)
# Users who previously committed fraud are more likely to commit fraud again
# First, assign an initial 'is_fraud' based on static biases
transactions_df['is_fraud_initial_guess'] = (np.random.rand(len(transactions_df)) < transactions_df['fraud_prob']).astype(int)

# Calculate if a user had prior fraud based on initial guesses (must be sorted!)
transactions_df['user_prior_fraud_count_guess'] = transactions_df.groupby('user_id')['is_fraud_initial_guess'].cumsum().shift(1).fillna(0)
transactions_df['user_had_prior_fraud_flag'] = (transactions_df['user_prior_fraud_count_guess'] > 0).astype(int)

# Boost fraud_prob if user had prior fraud
transactions_df.loc[transactions_df['user_had_prior_fraud_flag'] == 1, 'fraud_prob'] *= 3.0

# A user's first transaction with a new merchant has slightly higher fraud probability
transactions_df['is_first_merchant_transaction'] = transactions_df.groupby(['user_id', 'merchant_id']).cumcount() == 0
transactions_df.loc[transactions_df['is_first_merchant_transaction'], 'fraud_prob'] *= 1.2

# Final Cap on fraud_prob and assignment of final is_fraud
transactions_df['fraud_prob'] = transactions_df['fraud_prob'].clip(0.005, 0.9)
transactions_df['is_fraud'] = (np.random.rand(len(transactions_df)) < transactions_df['fraud_prob']).astype(int)

# Clean up temporary columns used for simulation
transactions_df = transactions_df.drop(columns=['fraud_prob', 'is_fraud_initial_guess', 'user_prior_fraud_count_guess',
                                                'user_had_prior_fraud_flag', 'is_first_merchant_transaction',
                                                'user_tier', 'region', 'category', 'risk_score', 'is_international'])

print(f"Generated {len(users_df)} users, {len(merchants_df)} merchants, {len(transactions_df)} transactions.")
print(f"Overall fraud rate: {transactions_df['is_fraud'].mean():.2%}")


# --- 2. Load into SQLite & SQL Feature Engineering ---

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Convert DataFrames to SQL tables
users_df.to_sql('users', conn, index=False, if_exists='replace')
merchants_df.to_sql('merchants', conn, index=False, if_exists='replace')
# Store dates as TEXT in SQLite for simplicity, will convert back to datetime in Pandas
transactions_df.to_sql('transactions', conn, index=False, if_exists='replace', dtype={'transaction_date': 'TEXT', 'signup_date': 'TEXT'})

# SQL query for feature engineering using window functions
sql_query = """
SELECT
    t.transaction_id,
    t.user_id,
    t.merchant_id,
    t.transaction_date,
    t.amount,
    t.transaction_type,
    t.is_fraud,
    u.user_tier,
    u.region,
    m.category,
    m.risk_score,
    m.is_international,
    u.signup_date,
    
    -- User-specific prior features
    COALESCE(COUNT(t.transaction_id) OVER (PARTITION BY t.user_id ORDER BY t.transaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS user_prior_num_transactions,
    COALESCE(SUM(t.amount) OVER (PARTITION BY t.user_id ORDER BY t.transaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0.0) AS user_prior_total_spend,
    COALESCE(AVG(t.amount) OVER (PARTITION BY t.user_id ORDER BY t.transaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0.0) AS user_avg_prior_transaction_amount,
    COALESCE(SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY t.user_id ORDER BY t.transaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS user_prior_num_fraud_transactions,
    
    -- Days since last user transaction (or since signup if first transaction)
    -- LAG(t.transaction_date, 1, u.signup_date) uses signup_date as default for the first transaction
    JULIANDAY(t.transaction_date) - JULIANDAY(LAG(t.transaction_date, 1, u.signup_date) OVER (PARTITION BY t.user_id ORDER BY t.transaction_date)) AS days_since_last_user_transaction,
    
    -- Merchant-specific prior features
    COALESCE(COUNT(t.transaction_id) OVER (PARTITION BY t.merchant_id ORDER BY t.transaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS merchant_prior_num_transactions,
    COALESCE(AVG(t.amount) OVER (PARTITION BY t.merchant_id ORDER BY t.transaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0.0) AS merchant_avg_prior_transaction_amount,
    COALESCE(SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY t.merchant_id ORDER BY t.transaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS merchant_prior_num_fraud_transactions
FROM
    transactions t
JOIN
    users u ON t.user_id = u.user_id
JOIN
    merchants m ON t.merchant_id = m.merchant_id
ORDER BY t.user_id, t.transaction_date;
"""

transaction_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Engineered {len(transaction_features_df)} transactions with SQL features.")
print(f"Columns after SQL feature engineering: {transaction_features_df.columns.tolist()}")

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Convert date columns to datetime objects
transaction_features_df['signup_date'] = pd.to_datetime(transaction_features_df['signup_date'])
transaction_features_df['transaction_date'] = pd.to_datetime(transaction_features_df['transaction_date'])

# Calculate days_since_signup_at_transaction
transaction_features_df['days_since_signup_at_transaction'] = (transaction_features_df['transaction_date'] - transaction_features_df['signup_date']).dt.days

# Handle any potential remaining NaNs (e.g., if there are no prior transactions for an aggregate)
# The SQL query uses COALESCE, so these should primarily be 0.
transaction_features_df['user_prior_num_transactions'] = transaction_features_df['user_prior_num_transactions'].fillna(0).astype(int)
transaction_features_df['user_prior_total_spend'] = transaction_features_df['user_prior_total_spend'].fillna(0.0)
transaction_features_df['user_avg_prior_transaction_amount'] = transaction_features_df['user_avg_prior_transaction_amount'].fillna(0.0)
transaction_features_df['user_prior_num_fraud_transactions'] = transaction_features_df['user_prior_num_fraud_transactions'].fillna(0).astype(int)
transaction_features_df['merchant_prior_num_transactions'] = transaction_features_df['merchant_prior_num_transactions'].fillna(0).astype(int)
transaction_features_df['merchant_avg_prior_transaction_amount'] = transaction_features_df['merchant_avg_prior_transaction_amount'].fillna(0.0)
transaction_features_df['merchant_prior_num_fraud_transactions'] = transaction_features_df['merchant_prior_num_fraud_transactions'].fillna(0).astype(int)
transaction_features_df['days_since_last_user_transaction'] = transaction_features_df['days_since_last_user_transaction'].fillna(0) # Should be handled by SQL LAG, but for safety

# Calculate user_prior_fraud_rate
transaction_features_df['user_prior_fraud_rate'] = transaction_features_df['user_prior_num_fraud_transactions'] / \
                                                   transaction_features_df['user_prior_num_transactions'].replace(0, 1.0)
transaction_features_df['user_prior_fraud_rate'] = transaction_features_df['user_prior_fraud_rate'].fillna(0.0)

# Calculate merchant_prior_fraud_rate
transaction_features_df['merchant_prior_fraud_rate'] = transaction_features_df['merchant_prior_num_fraud_transactions'] / \
                                                       transaction_features_df['merchant_prior_num_transactions'].replace(0, 1.0)
transaction_features_df['merchant_prior_fraud_rate'] = transaction_features_df['merchant_prior_fraud_rate'].fillna(0.0)

# Calculate amount_deviation_from_user_avg_prior
transaction_features_df['amount_deviation_from_user_avg_prior'] = transaction_features_df['amount'] - transaction_features_df['user_avg_prior_transaction_amount']
transaction_features_df['amount_deviation_from_user_avg_prior'] = transaction_features_df['amount_deviation_from_user_avg_prior'].fillna(0.0)


# Define features X and target y
numerical_features = [
    'amount', 'risk_score', 'user_prior_num_transactions', 'user_prior_total_spend',
    'user_avg_prior_transaction_amount', 'user_prior_num_fraud_transactions',
    'days_since_last_user_transaction', 'merchant_prior_num_transactions',
    'merchant_avg_prior_transaction_amount', 'merchant_prior_num_fraud_transactions',
    'days_since_signup_at_transaction', 'user_prior_fraud_rate',
    'merchant_prior_fraud_rate', 'amount_deviation_from_user_avg_prior'
]
categorical_features = [
    'transaction_type', 'user_tier', 'region', 'category', 'is_international'
]

# Filter features to ensure they exist in the DataFrame
numerical_features = [f for f in numerical_features if f in transaction_features_df.columns]
categorical_features = [f for f in categorical_features if f in transaction_features_df.columns]

X = transaction_features_df[numerical_features + categorical_features]
y = transaction_features_df['is_fraud']

print(f"\nFinal feature set includes {len(numerical_features)} numerical and {len(categorical_features)} categorical features.")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
print(f"Fraud rate in train set: {y_train.mean():.2%}, in test set: {y_test.mean():.2%}")


# --- 4. Data Visualization ---

print("\nGenerating visualizations...")
plt.figure(figsize=(14, 6))

# Violin plot for amount vs. is_fraud
plt.subplot(1, 2, 1)
sns.violinplot(x='is_fraud', y='amount', data=transaction_features_df, palette='viridis')
plt.title('Distribution of Transaction Amount by Fraud Status')
plt.xlabel('Is Fraud')
plt.ylabel('Amount')
plt.yscale('log') # Log scale for better visualization due to skewed amounts
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Stacked bar chart for is_fraud proportion across different categories
plt.subplot(1, 2, 2)
fraud_by_category = transaction_features_df.groupby('category')['is_fraud'].value_counts(normalize=True).unstack().fillna(0)
fraud_by_category.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'], ax=plt.gca())
plt.title('Fraud Proportion by Merchant Category')
plt.xlabel('Merchant Category')
plt.ylabel('Proportion')
plt.legend(title='Is Fraud', labels=['Not Fraud', 'Fraud'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

print("\nBuilding and training ML pipeline...")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (e.g., IDs) if any, though none should be left
)

# ML Pipeline with HistGradientBoostingClassifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42, class_weight='balanced'))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# For classification_report, we need binary predictions. Use a default threshold of 0.5.
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate the model
print("\nModel Evaluation on Test Set:")
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\nClassification Report (default 0.5 threshold):")
print(classification_report(y_test, y_pred))

print("\nScript finished.")