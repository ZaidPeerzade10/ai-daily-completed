import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

print("--- Generating Synthetic Data ---")

# 1.1 users_df
num_users = np.random.randint(500, 701)
current_date = pd.Timestamp.today()
signup_start_date = current_date - pd.DateOffset(years=3)

users_data = {
    'user_id': np.arange(1, num_users + 1),
    'signup_date': [signup_start_date + pd.to_timedelta(np.random.randint(0, (current_date - signup_start_date).days), unit='D') for _ in range(num_users)],
    'region': np.random.choice(['North', 'South', 'East', 'West'], num_users, p=[0.25, 0.25, 0.3, 0.2]), # Bias East slightly
    'marketing_channel': np.random.choice(['Organic', 'Paid Search', 'Social Media', 'Referral'], num_users, p=[0.3, 0.35, 0.2, 0.15]), # Bias Paid Search
    'age_group': np.random.choice(['18-24', '25-34', '35-49', '50+'], num_users, p=[0.2, 0.4, 0.3, 0.1])
}
users_df = pd.DataFrame(users_data)
users_df['signup_date'] = pd.to_datetime(users_df['signup_date']).dt.strftime('%Y-%m-%d') # Format for SQLite

print(f"Generated users_df with {len(users_df)} rows.")

# 1.2 products_df
num_products = np.random.randint(100, 151)
categories = ['Electronics', 'Books', 'Home & Garden', 'Apparel', 'Food']

products_data = {
    'product_id': np.arange(1001, 1001 + num_products),
    'category': np.random.choice(categories, num_products)
}
products_df = pd.DataFrame(products_data)

# Bias unit_price for 'Electronics'
products_df['unit_price'] = products_df.apply(
    lambda row: np.random.uniform(200.0, 1000.0) if row['category'] == 'Electronics'
                else np.random.uniform(10.0, 300.0), axis=1
)
products_df['cost_price'] = products_df['unit_price'] * np.random.uniform(0.3, 0.7, num_products)

print(f"Generated products_df with {len(products_df)} rows.")

# 1.3 transactions_df
num_transactions = np.random.randint(10000, 15001)

transactions_list = []
# Pre-merge users_df for quick access to signup_date and channels/regions for bias
users_with_dates = users_df.set_index('user_id')['signup_date'].apply(pd.to_datetime)
users_with_channel_region = users_df.set_index('user_id')[['marketing_channel', 'region']]

# Prepare products_df for quick lookup
product_lookup = products_df.set_index('product_id')

# Simulate realistic patterns:
# 'Paid Search' users or users from certain 'region's might have slightly higher average 'amount's.
# 'Electronics' category should have higher average 'unit_price' (already handled in products_df).
# Ensure transaction_date is always after signup_date.

transaction_ids = np.arange(1, num_transactions + 1)
user_ids_sampled = np.random.choice(users_df['user_id'], num_transactions)
product_ids_sampled = np.random.choice(products_df['product_id'], num_transactions)
quantities = np.random.randint(1, 6, num_transactions) # 1-5 quantity

transactions_df = pd.DataFrame({
    'transaction_id': transaction_ids,
    'user_id': user_ids_sampled,
    'product_id': product_ids_sampled,
    'quantity': quantities
})

# Merge signup_date and product_details
transactions_df = transactions_df.merge(users_with_dates.rename('signup_date'), on='user_id')
transactions_df = transactions_df.merge(users_with_channel_region, on='user_id')
transactions_df = transactions_df.merge(product_lookup[['unit_price', 'category']], on='product_id')

# Generate transaction_date after signup_date, spanning up to 2 years post-signup
transactions_df['transaction_date'] = transactions_df.apply(
    lambda row: row['signup_date'] + pd.to_timedelta(np.random.randint(1, 730), unit='D'), # Max 2 years after signup
    axis=1
)

# Calculate base amount
transactions_df['amount'] = transactions_df['quantity'] * transactions_df['unit_price']

# Apply bias to amount: 'Paid Search' or 'East' region users have slightly higher amounts
bias_mask = (transactions_df['marketing_channel'] == 'Paid Search') | (transactions_df['region'] == 'East')
transactions_df.loc[bias_mask, 'amount'] *= np.random.uniform(1.1, 1.3, bias_mask.sum())

transactions_df['signup_date'] = transactions_df['signup_date'].dt.strftime('%Y-%m-%d') # Revert to string for consistency
transactions_df['transaction_date'] = transactions_df['transaction_date'].dt.strftime('%Y-%m-%d')

# Drop temporary columns used for merging/biasing
transactions_df = transactions_df.drop(columns=['unit_price', 'category', 'marketing_channel', 'region'])

# Sort transactions_df
transactions_df = transactions_df.sort_values(by=['user_id', 'transaction_date']).reset_index(drop=True)

print(f"Generated transactions_df with {len(transactions_df)} rows.")

# --- 2. Load into SQLite & SQL Feature Engineering (Early User Behavior) ---

print("\n--- Loading Data to SQLite and SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite
users_df.to_sql('users', conn, index=False, if_exists='replace')
products_df.to_sql('products', conn, index=False, if_exists='replace')
transactions_df.to_sql('transactions', conn, index=False, if_exists='replace')

# SQL Query for early user behavior
# For each user, define their early_behavior_cutoff_date as signup_date + 30 days.
# Aggregate transaction behavior within their first 30 days post-signup.
# Ensure all users are included using LEFT JOIN.
sql_query = """
SELECT
    u.user_id,
    u.signup_date,
    u.region,
    u.marketing_channel,
    u.age_group,
    COALESCE(agg.num_transactions_first_30d, 0) AS num_transactions_first_30d,
    COALESCE(agg.total_spend_first_30d, 0.0) AS total_spend_first_30d,
    COALESCE(agg.avg_transaction_amount_first_30d, 0.0) AS avg_transaction_amount_first_30d,
    COALESCE(agg.num_unique_products_first_30d, 0) AS num_unique_products_first_30d,
    COALESCE(agg.num_unique_categories_first_30d, 0) AS num_unique_categories_first_30d,
    COALESCE(agg.days_with_transactions_first_30d, 0) AS days_with_transactions_first_30d
FROM users u
LEFT JOIN (
    SELECT
        t.user_id,
        COUNT(t.transaction_id) AS num_transactions_first_30d,
        SUM(t.amount) AS total_spend_first_30d,
        AVG(t.amount) AS avg_transaction_amount_first_30d,
        COUNT(DISTINCT t.product_id) AS num_unique_products_first_30d,
        COUNT(DISTINCT p.category) AS num_unique_categories_first_30d,
        COUNT(DISTINCT t.transaction_date) AS days_with_transactions_first_30d
    FROM transactions t
    JOIN products p ON t.product_id = p.product_id
    JOIN users u_sub ON t.user_id = u_sub.user_id
    WHERE julianday(t.transaction_date) <= julianday(DATE(u_sub.signup_date, '+30 days'))
    GROUP BY t.user_id
) agg ON u.user_id = agg.user_id;
"""

user_early_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Generated user_early_features_df with {len(user_early_features_df)} rows and early behavior features.")
print("\nFirst 5 rows of user_early_features_df:")
print(user_early_features_df.head())

# --- 3. Pandas Feature Engineering & Regression Target Creation (CLV) ---

print("\n--- Pandas Feature Engineering & CLV Target Creation ---")

# Handle NaN values (COALESCE in SQL already handled the main ones, but being explicit for averages)
user_early_features_df['avg_transaction_amount_first_30d'] = user_early_features_df['avg_transaction_amount_first_30d'].fillna(0.0)

# Convert signup_date to datetime objects
user_early_features_df['signup_date'] = pd.to_datetime(user_early_features_df['signup_date'])

# Calculate spend_frequency_first_30d
user_early_features_df['spend_frequency_first_30d'] = user_early_features_df['num_transactions_first_30d'] / 30.0
user_early_features_df['spend_frequency_first_30d'] = user_early_features_df['spend_frequency_first_30d'].fillna(0)

# Create the Regression Target clv_6_months
# Convert original transactions_df dates to datetime for calculation
transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
users_df['signup_date'] = pd.to_datetime(users_df['signup_date']) # Convert original signup_date

# Calculate cutoff dates for CLV
clv_dates_df = users_df[['user_id', 'signup_date']].copy()
clv_dates_df['early_behavior_end'] = clv_dates_df['signup_date'] + pd.DateOffset(days=30)
clv_dates_df['clv_6_months_end'] = clv_dates_df['early_behavior_end'] + pd.DateOffset(days=180) # 6 months = 180 days

# Merge transaction data with CLV cutoff dates
transactions_with_clv_dates = transactions_df.merge(clv_dates_df, on='user_id')

# Filter transactions for the CLV period
clv_transactions = transactions_with_clv_dates[
    (transactions_with_clv_dates['transaction_date'] > transactions_with_clv_dates['early_behavior_end']) &
    (transactions_with_clv_dates['transaction_date'] <= transactions_with_clv_dates['clv_6_months_end'])
]

# Calculate clv_6_months
clv_6_months_agg = clv_transactions.groupby('user_id')['amount'].sum().reset_index()
clv_6_months_agg.rename(columns={'amount': 'clv_6_months'}, inplace=True)

# Merge CLV target with user early features
user_early_features_df = user_early_features_df.merge(clv_6_months_agg, on='user_id', how='left')
user_early_features_df['clv_6_months'] = user_early_features_df['clv_6_months'].fillna(0)

print(f"CLV target 'clv_6_months' created. Average CLV: {user_early_features_df['clv_6_months'].mean():.2f}")
print("\nFirst 5 rows of user_early_features_df with CLV:")
print(user_early_features_df.head())

# Define features (X) and target (y)
numerical_features = [
    'num_transactions_first_30d',
    'total_spend_first_30d',
    'avg_transaction_amount_first_30d',
    'num_unique_products_first_30d',
    'num_unique_categories_first_30d',
    'days_with_transactions_first_30d',
    'spend_frequency_first_30d'
]
categorical_features = ['region', 'marketing_channel', 'age_group']

X = user_early_features_df[numerical_features + categorical_features]
y = user_early_features_df['clv_6_months']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nData split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")

# --- 4. Data Visualization ---

print("\n--- Generating Data Visualizations ---")

plt.figure(figsize=(14, 6))

# Scatter plot: total_spend_first_30d vs. clv_6_months
plt.subplot(1, 2, 1)
sns.regplot(x='total_spend_first_30d', y='clv_6_months', data=user_early_features_df,
            scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Total Spend in First 30 Days vs. 6-Month CLV')
plt.xlabel('Total Spend First 30 Days')
plt.ylabel('6-Month CLV')

# Box plot: clv_6_months across marketing_channel
plt.subplot(1, 2, 2)
sns.boxplot(x='marketing_channel', y='clv_6_months', data=user_early_features_df)
plt.title('6-Month CLV Distribution by Marketing Channel')
plt.xlabel('Marketing Channel')
plt.ylabel('6-Month CLV')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# --- 5. ML Pipeline & Evaluation (Regression) ---

print("\n--- Building and Evaluating ML Pipeline ---")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create the full ML pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

# Train the pipeline
print("Training the ML pipeline...")
model_pipeline.fit(X_train, y_train)
print("Pipeline training complete.")

# Predict on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation on Test Set:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")
print("\n--- Script Finished ---")