import pandas as pd
import numpy as np
import sqlite3
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set a random seed for reproducibility
np.random.seed(42)

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

# Generate users_df
num_users = np.random.randint(500, 701)
user_ids = np.arange(1, num_users + 1)
signup_start_date = datetime.date.today() - datetime.timedelta(days=5 * 365)
signup_dates = [signup_start_date + datetime.timedelta(days=np.random.randint(0, 5 * 365)) for _ in range(num_users)]
regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_users)
ages = np.random.randint(18, 71, num_users)
acquisition_channels = np.random.choice(['Organic', 'Social', 'Referral', 'Paid_Ad', 'Email'], num_users)

users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': pd.to_datetime(signup_dates),
    'region': regions,
    'age': ages,
    'acquisition_channel': acquisition_channels
})

# Generate transactions_df
num_transactions = np.random.randint(3000, 5001)
transaction_ids = np.arange(1, num_transactions + 1)

# Sample user_ids, ensuring some users have many, some few, some none
# Start with a base set of users that definitely have transactions
active_users = np.random.choice(users_df['user_id'], int(num_users * 0.8), replace=False)
transaction_user_ids = np.random.choice(active_users, num_transactions, replace=True, p=np.random.dirichlet(np.ones(len(active_users)) * 5)) # Skew distribution

# Create a temporary df to link transaction_date to signup_date
temp_transactions = pd.DataFrame({'user_id': transaction_user_ids})
temp_transactions = temp_transactions.merge(users_df[['user_id', 'signup_date']], on='user_id', how='left')

# Generate transaction_dates after signup_date
# For each transaction, pick a date between signup_date + 1 day and signup_date + random_days (max 3 years)
max_transaction_offset_days = (datetime.date.today() - signup_start_date).days # max possible days from earliest signup
transaction_dates = [
    row['signup_date'] + pd.to_timedelta(np.random.randint(1, min(3*365, (datetime.datetime.now() - row['signup_date']).days) + 1), unit='days')
    for idx, row in temp_transactions.iterrows()
]

amounts = np.random.uniform(10.0, 1000.0, num_transactions)
product_categories = np.random.choice(['Electronics', 'Books', 'Clothing', 'Groceries', 'Services', 'Home Goods'], num_transactions)

transactions_df = pd.DataFrame({
    'transaction_id': transaction_ids,
    'user_id': transaction_user_ids,
    'transaction_date': pd.to_datetime(transaction_dates),
    'amount': amounts,
    'product_category': product_categories
})

# Ensure no future transactions relative to "today" and clean up any edge cases
transactions_df = transactions_df[transactions_df['transaction_date'] <= pd.Timestamp.now()]

print("--- Synthetic Data Generated ---")
print("Users DataFrame Head:\n", users_df.head())
print("\nTransactions DataFrame Head:\n", transactions_df.head())
print(f"\nTotal users: {len(users_df)}")
print(f"Total transactions: {len(transactions_df)}")


# --- 2. Load into SQLite & SQL Feature Engineering ---

conn = sqlite3.connect(':memory:')
users_df.to_sql('users', conn, index=False, if_exists='replace')
transactions_df.to_sql('transactions', conn, index=False, if_exists='replace')

# Determine global_analysis_date and feature_cutoff_date using pandas first
# Convert transaction_date in transactions_df to datetime if not already
transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])

global_analysis_date_pd = transactions_df['transaction_date'].max() + pd.DateOffset(days=60)
feature_cutoff_date_pd = global_analysis_date_pd - pd.DateOffset(days=90)

# Convert to string for SQL query
global_analysis_date_str = global_analysis_date_pd.strftime('%Y-%m-%d')
feature_cutoff_date_str = feature_cutoff_date_pd.strftime('%Y-%m-%d')

print(f"\nGlobal Analysis Date: {global_analysis_date_str}")
print(f"Feature Cutoff Date: {feature_cutoff_date_str}")

sql_query = f"""
SELECT
    u.user_id,
    u.age,
    u.region,
    u.acquisition_channel,
    u.signup_date,
    COALESCE(SUM(CASE WHEN t.transaction_date < '{feature_cutoff_date_str}' THEN t.amount ELSE 0 END), 0.0) AS total_spend_pre_cutoff,
    COALESCE(COUNT(CASE WHEN t.transaction_date < '{feature_cutoff_date_str}' THEN t.transaction_id ELSE NULL END), 0) AS num_transactions_pre_cutoff,
    COALESCE(AVG(CASE WHEN t.transaction_date < '{feature_cutoff_date_str}' THEN t.amount ELSE NULL END), 0.0) AS avg_transaction_value_pre_cutoff,
    COALESCE(COUNT(DISTINCT CASE WHEN t.transaction_date < '{feature_cutoff_date_str}' THEN t.product_category ELSE NULL END), 0) AS num_unique_categories_pre_cutoff,
    (JULIANDAY('{feature_cutoff_date_str}') - JULIANDAY(MAX(CASE WHEN t.transaction_date < '{feature_cutoff_date_str}' THEN t.transaction_date ELSE NULL END))) AS days_since_last_transaction_pre_cutoff
FROM
    users u
LEFT JOIN
    transactions t ON u.user_id = t.user_id
GROUP BY
    u.user_id, u.age, u.region, u.acquisition_channel, u.signup_date
ORDER BY
    u.user_id;
"""

user_features_df = pd.read_sql_query(sql_query, conn)
conn.close() # Close SQLite connection

print("\n--- SQL Feature Engineering Results (Head) ---")
print(user_features_df.head())
print(f"\nNumber of users with engineered features: {len(user_features_df)}")


# --- 3. Pandas Feature Engineering & Multi-Class Target Creation ---

# Handle NaN values
user_features_df['total_spend_pre_cutoff'] = user_features_df['total_spend_pre_cutoff'].fillna(0.0)
user_features_df['num_transactions_pre_cutoff'] = user_features_df['num_transactions_pre_cutoff'].fillna(0)
user_features_df['num_unique_categories_pre_cutoff'] = user_features_df['num_unique_categories_pre_cutoff'].fillna(0)
user_features_df['avg_transaction_value_pre_cutoff'] = user_features_df['avg_transaction_value_pre_cutoff'].fillna(0.0)

# Convert signup_date to datetime objects
user_features_df['signup_date'] = pd.to_datetime(user_features_df['signup_date'])

# Calculate account_age_at_cutoff_days
user_features_df['account_age_at_cutoff_days'] = (feature_cutoff_date_pd - user_features_df['signup_date']).dt.days

# Fill days_since_last_transaction_pre_cutoff NaN with a sentinel value
# For users with no transactions before cutoff, days_since_last_transaction_pre_cutoff will be NaN.
# Fill it with their account age at cutoff + 30 days as a proxy for "very long time ago / no activity"
user_features_df['days_since_last_transaction_pre_cutoff'] = user_features_df['days_since_last_transaction_pre_cutoff'].fillna(
    user_features_df['account_age_at_cutoff_days'] + 30
)
# Ensure days_since_last_transaction_pre_cutoff is not negative if feature_cutoff_date_pd is slightly before max transaction date
user_features_df['days_since_last_transaction_pre_cutoff'] = user_features_df['days_since_last_transaction_pre_cutoff'].apply(lambda x: max(0, x))


# Calculate Future Spend
future_transactions_df = transactions_df[
    (transactions_df['transaction_date'] >= feature_cutoff_date_pd) &
    (transactions_df['transaction_date'] < global_analysis_date_pd)
].copy()

total_spend_future = future_transactions_df.groupby('user_id')['amount'].sum().reset_index()
total_spend_future.rename(columns={'amount': 'total_spend_future'}, inplace=True)

user_features_df = user_features_df.merge(total_spend_future, on='user_id', how='left')
user_features_df['total_spend_future'] = user_features_df['total_spend_future'].fillna(0.0)

# Create the Multi-Class Target `future_spending_tier`
non_zero_future_spend = user_features_df[user_features_df['total_spend_future'] > 0]['total_spend_future']
if len(non_zero_future_spend) > 0:
    q33 = non_zero_future_spend.quantile(0.33)
    q66 = non_zero_future_spend.quantile(0.66)
else: # Fallback for extremely rare synthetic data case where no future spend occurs
    q33 = 0.0
    q66 = 0.0

def assign_spending_tier(spend):
    if spend == 0:
        return 'No_Future_Spend'
    elif spend <= q33:
        return 'Low_Spender'
    elif spend <= q66:
        return 'Medium_Spender'
    else:
        return 'High_Spender'

user_features_df['future_spending_tier'] = user_features_df['total_spend_future'].apply(assign_spending_tier)

print("\n--- Pandas Feature Engineering & Target Creation Results (Head) ---")
print(user_features_df.head())
print("\nFuture Spending Tier Value Counts:\n", user_features_df['future_spending_tier'].value_counts())


# Define features X and target y
numerical_features = [
    'age',
    'total_spend_pre_cutoff',
    'num_transactions_pre_cutoff',
    'avg_transaction_value_pre_cutoff',
    'num_unique_categories_pre_cutoff',
    'days_since_last_transaction_pre_cutoff',
    'account_age_at_cutoff_days'
]
categorical_features = ['region', 'acquisition_channel']

X = user_features_df[numerical_features + categorical_features]
y = user_features_df['future_spending_tier']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")


# --- 4. Data Visualization ---

plt.style.use('ggplot')
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Violin plot: total_spend_pre_cutoff vs. future_spending_tier
sns.violinplot(x='future_spending_tier', y='total_spend_pre_cutoff', data=user_features_df, ax=axes[0],
               order=['No_Future_Spend', 'Low_Spender', 'Medium_Spender', 'High_Spender'])
axes[0].set_title('Distribution of Pre-Cutoff Spend by Future Spending Tier')
axes[0].set_xlabel('Future Spending Tier')
axes[0].set_ylabel('Total Spend Before Cutoff')
axes[0].ticklabel_format(style='plain', axis='y') # Disable scientific notation on y-axis

# Stacked bar chart: future_spending_tier across different regions
region_tier_counts = user_features_df.groupby(['region', 'future_spending_tier']).size().unstack(fill_value=0)
# Reorder columns for consistency
tier_order = ['No_Future_Spend', 'Low_Spender', 'Medium_Spender', 'High_Spender']
region_tier_counts = region_tier_counts[tier_order]
region_tier_counts.plot(kind='bar', stacked=True, ax=axes[1], cmap='viridis')
axes[1].set_title('Future Spending Tier Distribution by Region')
axes[1].set_xlabel('Region')
axes[1].set_ylabel('Number of Users')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Future Spend Tier', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

print("\n--- Visualizations Generated ---")


# --- 5. ML Pipeline & Evaluation (Multi-Class) ---

# Preprocessing steps for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing steps for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'))
])

# Train the model
print("\n--- Training Machine Learning Model ---")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nClassification Report:\n", classification_rep)

print("\nScript execution complete.")