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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# --- 1. Generate Synthetic Data ---

# Define parameters for data generation
np.random.seed(42)
num_users = np.random.randint(500, 701)
num_transactions = np.random.randint(5000, 8001)
today = pd.to_datetime('today').normalize() # Ensure dates are consistent and capped at today

# Users DataFrame
user_ids = np.arange(1, num_users + 1)
signup_dates = today - pd.to_timedelta(np.random.randint(0, 365 * 5 + 1, num_users), unit='days')
regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_users)
ages = np.random.randint(18, 71, num_users)

users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': signup_dates,
    'region': regions,
    'age': ages
})

# Transactions DataFrame
transaction_ids = np.arange(1, num_transactions + 1)
user_ids_transactions = np.random.choice(users_df['user_id'], num_transactions, replace=True)

# Map signup dates to transaction user_ids
signup_dates_map = users_df.set_index('user_id')['signup_date'].to_dict()
transaction_signup_dates = pd.Series(user_ids_transactions).map(signup_dates_map)

# Generate transaction dates ensuring they are after signup_date and not in the future
# Generate random days after signup date for each transaction
# Max days after signup ensures transaction dates don't go too far into the future beyond 'today' cap.
max_days_after_signup = (today - transaction_signup_dates).dt.days.clip(lower=1)
days_after_signup = np.random.randint(1, max_days_after_signup + 1, num_transactions)
transaction_dates = transaction_signup_dates + pd.to_timedelta(days_after_signup, unit='days')

amounts = np.random.uniform(10.0, 2000.0, num_transactions)
merchant_categories = np.random.choice(['Groceries', 'Retail', 'Dining', 'Travel', 'Online_Service', 'Utilities', 'Entertainment'], num_transactions)
location_countries = np.random.choice(['USA', 'Canada', 'UK', 'Mexico', 'Japan', 'Germany', 'France'], num_transactions)

transactions_df = pd.DataFrame({
    'transaction_id': transaction_ids,
    'user_id': user_ids_transactions,
    'transaction_date': transaction_dates,
    'amount': amounts,
    'merchant_category': merchant_categories,
    'location_country': location_countries
})

# Sort transactions_df by user_id and then transaction_date for sequential processing
transactions_df.sort_values(by=['user_id', 'transaction_date'], inplace=True)
transactions_df.reset_index(drop=True, inplace=True)

print(f"Generated {len(users_df)} users and {len(transactions_df)} transactions.")
print("Users DataFrame head:")
print(users_df.head())
print("\nTransactions DataFrame head (sorted by user_id, transaction_date):")
print(transactions_df.head())

# Convert datetime columns to string for SQLite compatibility
users_df['signup_date_str'] = users_df['signup_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
transactions_df['transaction_date_str'] = transactions_df['transaction_date'].dt.strftime('%Y-%m-%d %H:%M:%S')


# --- 2. Load into SQLite & SQL Feature Engineering ---

conn = sqlite3.connect(':memory:')

users_df[['user_id', 'signup_date_str', 'region', 'age']].to_sql('users', conn, if_exists='replace', index=False)
transactions_df[['transaction_id', 'user_id', 'transaction_date_str', 'amount', 'merchant_category', 'location_country']].to_sql('transactions', conn, if_exists='replace', index=False)

# SQL Query for sequential features using window functions
sql_query = """
SELECT
    t.transaction_id,
    t.user_id,
    t.transaction_date_str AS transaction_date,
    t.amount,
    t.merchant_category,
    t.location_country,
    u.region,
    u.age,
    u.signup_date_str AS signup_date,
    -- Prior average spend
    COALESCE(AVG(t.amount) OVER (PARTITION BY t.user_id ORDER BY t.transaction_date_str ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0.0) AS user_avg_spend_prior,
    -- Prior max spend
    COALESCE(MAX(t.amount) OVER (PARTITION BY t.user_id ORDER BY t.transaction_date_str ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0.0) AS user_max_spend_prior,
    -- Prior number of transactions
    COALESCE(COUNT(t.transaction_id) OVER (PARTITION BY t.user_id ORDER BY t.transaction_date_str ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS user_num_transactions_prior,
    -- Days since last transaction or signup date if first
    ROUND(
        JULIANDAY(t.transaction_date_str) -
        JULIANDAY(LAG(t.transaction_date_str, 1, u.signup_date_str) OVER (PARTITION BY t.user_id ORDER BY t.transaction_date_str))
    ) AS days_since_last_transaction
FROM
    transactions t
JOIN
    users u ON t.user_id = u.user_id
ORDER BY
    t.user_id, t.transaction_date_str;
"""

transaction_features_df = pd.read_sql_query(sql_query, conn)

print("\nSQL Feature Engineering Results (head):")
print(transaction_features_df.head())
print("\nSQL Feature Engineering Results (tail, showing sequential features):")
print(transaction_features_df.tail())

conn.close()

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Convert date strings back to datetime objects
transaction_features_df['transaction_date'] = pd.to_datetime(transaction_features_df['transaction_date'])
transaction_features_df['signup_date'] = pd.to_datetime(transaction_features_df['signup_date'])

# Handle NaN values for prior features
# SQL COALESCE should handle most, but ensure consistency
transaction_features_df['user_avg_spend_prior'] = transaction_features_df['user_avg_spend_prior'].fillna(0.0)
transaction_features_df['user_max_spend_prior'] = transaction_features_df['user_max_spend_prior'].fillna(0.0)
transaction_features_df['user_num_transactions_prior'] = transaction_features_df['user_num_transactions_prior'].fillna(0).astype(int)
# Fill days_since_last_transaction NaNs with a large sentinel value (e.g., 9999) if any remain
transaction_features_df['days_since_last_transaction'] = transaction_features_df['days_since_last_transaction'].fillna(9999).astype(float)


# Calculate amount_vs_avg_prior_ratio
# Use np.where to handle division by zero for first transactions
transaction_features_df['amount_vs_avg_prior_ratio'] = np.where(
    transaction_features_df['user_avg_spend_prior'] > 0,
    transaction_features_df['amount'] / transaction_features_df['user_avg_spend_prior'],
    1.0 # If no prior average (first transaction or avg is 0), ratio is 1.0 (current amount compared to itself)
)

# Create is_first_transaction flag
transaction_features_df['is_first_transaction'] = (transaction_features_df['user_num_transactions_prior'] == 0).astype(int)

# Create Binary Target `is_suspicious`
# A transaction is 'suspicious' (1) if:
# 1. amount > 1000 (absolute high amount)
# OR
# 2. (amount_vs_avg_prior_ratio > 2.5 AND days_since_last_transaction < 1.0 AND user_num_transactions_prior > 0)
transaction_features_df['is_suspicious'] = (
    (transaction_features_df['amount'] > 1000) |
    (
        (transaction_features_df['amount_vs_avg_prior_ratio'] > 2.5) &
        (transaction_features_df['days_since_last_transaction'] < 1.0) &
        (transaction_features_df['user_num_transactions_prior'] > 0)
    )
).astype(int)

print(f"\nTotal suspicious transactions: {transaction_features_df['is_suspicious'].sum()}")
print(f"Percentage suspicious: {transaction_features_df['is_suspicious'].mean() * 100:.2f}%")

# Define features X and target y
numerical_features = ['age', 'amount', 'user_avg_spend_prior', 'user_max_spend_prior', 
                      'user_num_transactions_prior', 'days_since_last_transaction', 
                      'amount_vs_avg_prior_ratio']
categorical_features = ['region', 'merchant_category', 'location_country']
binary_features_for_encoding = ['is_first_transaction'] # Treated as categorical for OneHotEncoding

X = transaction_features_df[numerical_features + categorical_features + binary_features_for_encoding]
y = transaction_features_df['is_suspicious']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nShape of X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"Proportion of suspicious in y_train: {y_train.mean():.4f}")
print(f"Proportion of suspicious in y_test: {y_test.mean():.4f}")


# --- 4. Data Visualization ---

plt.style.use('seaborn-v0_8-darkgrid')

# Plot 1: Violin plot of amount_vs_avg_prior_ratio for suspicious vs. non-suspicious
plt.figure(figsize=(10, 6))
sns.violinplot(x='is_suspicious', y='amount_vs_avg_prior_ratio', data=transaction_features_df)
plt.title('Distribution of Amount vs. Avg Prior Ratio by Suspicious Status')
plt.xlabel('Is Suspicious (0: No, 1: Yes)')
plt.ylabel('Amount vs. Avg Prior Ratio')
plt.ylim(-0.5, 10) # Clip y-axis for better visualization due to potential outliers
plt.show()

# Plot 2: Stacked bar chart of proportion of is_suspicious across merchant_category
plt.figure(figsize=(12, 7))
category_suspicion_proportions = pd.crosstab(transaction_features_df['merchant_category'], 
                                              transaction_features_df['is_suspicious'], 
                                              normalize='index')
category_suspicion_proportions.plot(kind='bar', stacked=True, color=['lightcoral', 'darkred'], ax=plt.gca())
plt.title('Proportion of Suspicious Transactions by Merchant Category')
plt.xlabel('Merchant Category')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Is Suspicious', labels=['Not Suspicious', 'Suspicious'])
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---

# Preprocessing steps for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features + binary_features_for_encoding)
    ],
    remainder='passthrough'
)

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1))
])

# Train the pipeline
model_pipeline.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to binary predictions

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
classification_rep = classification_report(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"ROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:")
print(classification_rep)

print("\nScript finished successfully.")