import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import path for SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print("Starting solution script...")

# --- Task 1: Generate Synthetic Data (Pandas/Numpy) ---

print("\n--- Task 1: Generating Synthetic Data ---")

# 1.1 users_df
num_users = np.random.randint(500, 701)
signup_start_date = datetime.now() - timedelta(days=5*365) # Last 5 years
users_data = {
    'user_id': np.arange(num_users),
    'signup_date': [signup_start_date + timedelta(days=np.random.randint(0, (datetime.now() - signup_start_date).days + 1)) for _ in range(num_users)],
    'age': np.random.randint(18, 71, num_users), # 18-70 inclusive
    'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_users)
}
users_df = pd.DataFrame(users_data)

# 1.2 transactions_df
num_transactions_target = np.random.randint(8000, 12001)
transaction_data = []
fraud_rate_target = 0.02 # Aim for ~2% fraud rate
transaction_idx = 0

# Store user signup dates for efficient lookup
user_signup_dates = users_df.set_index('user_id')['signup_date'].to_dict()

# Identify a small number of users who will be 'fraudsters'
num_fraudster_users = max(5, int(num_users * 0.01)) 
fraudster_user_ids = np.random.choice(users_df['user_id'], num_fraudster_users, replace=False)

while transaction_idx < num_transactions_target:
    current_user_id = np.random.choice(users_df['user_id'])
    signup_date = user_signup_dates[current_user_id]

    is_fraud = 0
    # Increase chance of fraud for designated fraudsters
    if current_user_id in fraudster_user_ids and np.random.rand() < 0.25: # 25% of their tx are likely fraudulent
        is_fraud = 1
    elif np.random.rand() < fraud_rate_target * 0.5: # Base fraud rate for others
        is_fraud = 1

    # Simulate base transaction date (always after signup)
    days_since_signup_max = (datetime.now() - signup_date).days
    if days_since_signup_max < 1: # If signup date is today, ensure at least a second difference
        transaction_date = signup_date + timedelta(seconds=np.random.randint(1, 3600)) # Within an hour
    else:
        transaction_date = signup_date + timedelta(days=np.random.randint(1, days_since_signup_max + 1))

    amount = np.random.uniform(10.0, 5000.0)
    merchant_category = np.random.choice(['Groceries', 'Retail', 'Dining', 'Travel', 'Online_Service', 'Utilities', 'Entertainment'])
    location_country = np.random.choice(['USA', 'Canada', 'UK', 'Mexico', 'Japan', 'Germany', 'France', 'Australia'])

    if is_fraud:
        # Fraud pattern 1: Higher amounts
        amount = np.random.uniform(2000.0, 10000.0) 

        # Fraud pattern 2: Rapid succession & country hopping
        if np.random.rand() < 0.6: # 60% chance to generate a burst
            num_burst_tx = np.random.randint(1, 4) # 1 to 3 additional transactions
            burst_time_delta = timedelta(minutes=np.random.randint(1, 121)) # within 2 hours
            burst_countries = np.random.choice(['USA', 'Canada', 'UK', 'Mexico', 'Japan', 'Germany', 'France', 'Australia'], num_burst_tx + 1, replace=True)

            # First transaction of the burst
            transaction_data.append({
                'transaction_id': transaction_idx,
                'user_id': current_user_id,
                'transaction_date': transaction_date,
                'amount': amount,
                'merchant_category': merchant_category,
                'location_country': burst_countries[0],
                'is_fraudulent': 1
            })
            transaction_idx += 1

            for i in range(num_burst_tx):
                if transaction_idx >= num_transactions_target: break
                
                transaction_date_burst = transaction_date + timedelta(minutes=np.random.randint(1, 121))
                # Ensure it's still after signup_date
                transaction_date_burst = max(transaction_date_burst, signup_date + timedelta(seconds=1))

                transaction_data.append({
                    'transaction_id': transaction_idx,
                    'user_id': current_user_id,
                    'transaction_date': transaction_date_burst,
                    'amount': np.random.uniform(amount * 0.5, amount * 1.5), 
                    'merchant_category': np.random.choice(['Groceries', 'Retail', 'Dining', 'Travel', 'Online_Service', 'Utilities', 'Entertainment']),
                    'location_country': burst_countries[i+1], 
                    'is_fraudulent': 1
                })
                transaction_idx += 1
            continue # Skip adding the initial single transaction below
        
        # Fraud pattern 3: Concentrated shortly after signup (for non-burst fraud)
        if np.random.rand() < 0.3: 
            transaction_date = signup_date + timedelta(days=np.random.randint(1, 60)) # Within first 2 months
            transaction_date = max(transaction_date, signup_date + timedelta(seconds=1)) # Ensure after signup

    # Add the single transaction if not part of a burst and target not reached
    if transaction_idx < num_transactions_target:
        transaction_data.append({
            'transaction_id': transaction_idx,
            'user_id': current_user_id,
            'transaction_date': transaction_date,
            'amount': amount,
            'merchant_category': merchant_category,
            'location_country': location_country,
            'is_fraudulent': is_fraud
        })
        transaction_idx += 1

transactions_df = pd.DataFrame(transaction_data)
# Ensure exact number of transactions
transactions_df = transactions_df.iloc[:num_transactions_target] 
transactions_df['transaction_id'] = np.arange(len(transactions_df)) # Re-index transaction_id uniquely

# Sort transactions for sequential processing in SQL
transactions_df = transactions_df.sort_values(by=['user_id', 'transaction_date']).reset_index(drop=True)

print(f"Generated {len(users_df)} users and {len(transactions_df)} transactions.")
print(f"Actual fraud rate: {transactions_df['is_fraudulent'].mean():.2%}")
print("Synthetic data generation complete.")

# --- Task 2: Load into SQLite & SQL Feature Engineering ---

print("\n--- Task 2: Loading into SQLite & SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')
users_df.to_sql('users', conn, index=False, if_exists='replace')
transactions_df.to_sql('transactions', conn, index=False, if_exists='replace')

sql_query = """
WITH UserTransactionFeatures AS (
    SELECT
        t.transaction_id,
        t.user_id,
        t.transaction_date,
        t.amount,
        t.merchant_category,
        t.location_country,
        t.is_fraudulent,
        u.age,
        u.region,
        u.signup_date,
        -- Previous transaction date for days_since_last_user_transaction
        LAG(t.transaction_date, 1) OVER (PARTITION BY t.user_id ORDER BY t.transaction_date) AS prev_transaction_date,
        -- Prior transactions in last 30 days
        COUNT(t.transaction_id) OVER (
            PARTITION BY t.user_id
            ORDER BY JULIANDAY(t.transaction_date)
            RANGE BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS user_prior_num_transactions_30d,
        -- Total spend in last 30 days
        SUM(t.amount) OVER (
            PARTITION BY t.user_id
            ORDER BY JULIANDAY(t.transaction_date)
            RANGE BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS user_prior_total_spend_30d,
        -- Average amount of last 5 prior transactions
        AVG(t.amount) OVER (
            PARTITION BY t.user_id
            ORDER BY t.transaction_date
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS user_avg_amount_last_5_tx,
        -- Concatenated string of countries for last 5 prior transactions (to be processed in Pandas)
        GROUP_CONCAT(t.location_country) OVER (
            PARTITION BY t.user_id
            ORDER BY t.transaction_date
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS user_prior_countries_last_5_tx_str
    FROM
        transactions t
    JOIN
        users u ON t.user_id = u.user_id
)
SELECT
    transaction_id,
    user_id,
    transaction_date,
    amount,
    merchant_category,
    location_country,
    is_fraudulent,
    age,
    region,
    signup_date,
    user_prior_num_transactions_30d,
    COALESCE(user_prior_total_spend_30d, 0.0) AS user_prior_total_spend_30d,
    user_avg_amount_last_5_tx,
    -- Calculate days since last transaction or since signup if first transaction for the user
    COALESCE(
        JULIANDAY(transaction_date) - JULIANDAY(prev_transaction_date),
        JULIANDAY(transaction_date) - JULIANDAY(signup_date)
    ) AS days_since_last_user_transaction,
    user_prior_countries_last_5_tx_str
FROM
    UserTransactionFeatures
ORDER BY
    user_id, transaction_date;
"""

transaction_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"SQL feature engineering complete. {len(transaction_features_df)} transactions with engineered features.")
print("First 5 rows of SQL-engineered data:")
print(transaction_features_df.head())

# --- Task 3: Pandas Feature Engineering & Binary Target Creation ---

print("\n--- Task 3: Pandas Feature Engineering & Binary Target Creation ---")

# Convert dates to datetime objects
transaction_features_df['signup_date'] = pd.to_datetime(transaction_features_df['signup_date'])
transaction_features_df['transaction_date'] = pd.to_datetime(transaction_features_df['transaction_date'])

# Handle NaN values for SQL-engineered features
transaction_features_df['user_prior_num_transactions_30d'] = transaction_features_df['user_prior_num_transactions_30d'].fillna(0).astype(int)
# user_prior_total_spend_30d is already COALESCE'd to 0.0 in SQL
transaction_features_df['user_avg_amount_last_5_tx'] = transaction_features_df['user_avg_amount_last_5_tx'].fillna(0.0)

# Calculate user_num_unique_countries_last_5_tx from the string
def count_unique_countries_from_str(country_str):
    if pd.isna(country_str) or not country_str.strip():
        return 0
    return len(set(country_str.split(',')))

transaction_features_df['user_num_unique_countries_last_5_tx'] = transaction_features_df['user_prior_countries_last_5_tx_str'].apply(count_unique_countries_from_str)
transaction_features_df.drop('user_prior_countries_last_5_tx_str', axis=1, inplace=True)

# Calculate user_account_age_at_transaction_days
transaction_features_df['user_account_age_at_transaction_days'] = (
    (transaction_features_df['transaction_date'] - transaction_features_df['signup_date']).dt.days
)

# Fill any remaining NaNs in days_since_last_user_transaction with user_account_age_at_transaction_days
# This primarily handles cases where signup_date == transaction_date (SQL COALESCE yields 0, or very small float)
# And ensures non-negativity
transaction_features_df['days_since_last_user_transaction'] = transaction_features_df['days_since_last_user_transaction'].fillna(
    transaction_features_df['user_account_age_at_transaction_days']
).apply(lambda x: max(x, 0))


# Calculate amount_to_avg_prior_ratio
# Handle division by zero: if user_avg_amount_last_5_tx is 0, ratio should reflect no prior average.
# Replacing 0 with NaN will result in NaN, which is then filled with 0.0, as per requirements.
transaction_features_df['amount_to_avg_prior_ratio'] = transaction_features_df['amount'] / transaction_features_df['user_avg_amount_last_5_tx'].replace(0, np.nan)
transaction_features_df['amount_to_avg_prior_ratio'] = transaction_features_df['amount_to_avg_prior_ratio'].fillna(0.0)
transaction_features_df.replace([np.inf, -np.inf], 0.0, inplace=True) # Replace any remaining inf with 0

# Calculate transaction_velocity_30d
# Add 1 to denominator to avoid division by zero if days_since_last_user_transaction is 0
transaction_features_df['transaction_velocity_30d'] = transaction_features_df['user_prior_num_transactions_30d'] / (transaction_features_df['days_since_last_user_transaction'] + 1)
transaction_features_df['transaction_velocity_30d'] = transaction_features_df['transaction_velocity_30d'].fillna(0.0)
transaction_features_df.replace([np.inf, -np.inf], 0.0, inplace=True)

print("Pandas feature engineering complete.")
print("First 5 rows with new Pandas-engineered features:")
print(transaction_features_df[['transaction_id', 'amount', 'user_avg_amount_last_5_tx', 'amount_to_avg_prior_ratio', 
                               'transaction_velocity_30d', 'is_fraudulent']].head())

# Define features (X) and target (y)
numerical_features = [
    'amount', 'age', 'user_account_age_at_transaction_days',
    'user_prior_num_transactions_30d', 'user_prior_total_spend_30d',
    'user_avg_amount_last_5_tx', 'days_since_last_user_transaction',
    'user_num_unique_countries_last_5_tx', 'amount_to_avg_prior_ratio',
    'transaction_velocity_30d'
]
categorical_features = ['region', 'merchant_category', 'location_country']

# Ensure all feature columns exist before proceeding
missing_num_features = [f for f in numerical_features if f not in transaction_features_df.columns]
missing_cat_features = [f for f in categorical_features if f not in transaction_features_df.columns]
if missing_num_features or missing_cat_features:
    raise ValueError(f"Missing features: {missing_num_features + missing_cat_features}")

# Drop rows with NaN in critical categorical features if any exist after previous processing steps
transaction_features_df.dropna(subset=categorical_features, inplace=True)

X = transaction_features_df[numerical_features + categorical_features]
y = transaction_features_df['is_fraudulent']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")
print(f"Fraud rate in training: {y_train.mean():.2%}, in testing: {y_test.mean():.2%}")

# --- Task 4: Data Visualization ---

print("\n--- Task 4: Data Visualization ---")

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Violin plot for amount distribution vs. is_fraudulent
sns.violinplot(x='is_fraudulent', y='amount', data=transaction_features_df, palette='viridis', ax=axes[0])
axes[0].set_title('Distribution of Transaction Amount by Fraud Status')
axes[0].set_xlabel('Is Fraudulent')
axes[0].set_ylabel('Transaction Amount')
axes[0].set_xticks(ticks=[0, 1], labels=['Not Fraudulent', 'Fraudulent'])

# Plot 2: Stacked bar chart for fraud proportion across location_country
country_fraud_proportions = transaction_features_df.groupby('location_country')['is_fraudulent'].value_counts(normalize=True).unstack().fillna(0)
country_fraud_proportions.plot(kind='bar', stacked=True, ax=axes[1], cmap='coolwarm')
axes[1].set_title('Proportion of Fraudulent Transactions by Country')
axes[1].set_xlabel('Location Country')
axes[1].set_ylabel('Proportion')
axes[1].legend(title='Is Fraudulent', labels=['Not Fraudulent', 'Fraudulent'])
axes[1].tick_params(axis='x', rotation=45, ha='right')

plt.tight_layout()
plt.show()

print("Data visualizations displayed.")

# --- Task 5: ML Pipeline & Evaluation (Binary Classification) ---

print("\n--- Task 5: ML Pipeline & Evaluation ---")

# Preprocessing steps
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
    remainder='drop' # Drop any columns not specified
)

# Create the ML pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("Training the ML model...")
pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1] 
# Predict class labels (using default threshold 0.5)
y_pred = pipeline.predict(X_test) 

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud'])

print("\n--- Model Evaluation ---")
print(f"ROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:")
print(class_report)

print("\n--- Script Finished ---")