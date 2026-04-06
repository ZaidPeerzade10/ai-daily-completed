import pandas as pd
import numpy as np
import sqlite3
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) # For seaborn plots on small data or specific scenarios

print("--- Starting Product Return Prediction Pipeline ---")

# 1. Generate Synthetic Data
# Set random seed for reproducibility
np.random.seed(42)

# --- Customers DataFrame ---
num_customers = np.random.randint(500, 701)
print(f"Generating {num_customers} synthetic customers...")
customers_df = pd.DataFrame({
    'customer_id': np.arange(num_customers)
})

# Fix for signup_date generation: convert Timestamps to numerical for uniform, then back
today = pd.to_datetime('today')
three_years_ago = today - pd.DateOffset(years=3)
start_timestamp = three_years_ago.value
end_timestamp = today.value
random_timestamps = np.random.uniform(start_timestamp, end_timestamp, num_customers)
customers_df['signup_date'] = pd.to_datetime(random_timestamps)

customers_df['region'] = np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_customers)
customers_df['loyalty_status'] = np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], num_customers, p=[0.4, 0.3, 0.2, 0.1])

# --- Products DataFrame ---
num_products = np.random.randint(100, 151)
print(f"Generating {num_products} synthetic products...")
products_df = pd.DataFrame({
    'product_id': np.arange(num_products)
})
products_df['category'] = np.random.choice(
    ['Electronics', 'Books', 'Apparel', 'Home Goods', 'Groceries', 'Beauty'],
    num_products,
    p=[0.2, 0.15, 0.25, 0.15, 0.1, 0.15] # Apparel slightly more common for return bias
)
products_df['unit_price'] = np.random.uniform(20.0, 1500.0, num_products)
products_df['brand'] = np.random.choice([f'Brand{chr(65 + i)}' for i in range(5)], num_products) # BrandA to BrandE

# --- Transactions DataFrame ---
num_transactions = np.random.randint(10000, 15001)
print(f"Generating {num_transactions} synthetic transactions...")
transactions_df = pd.DataFrame({
    'transaction_id': np.arange(num_transactions),
    'customer_id': np.random.choice(customers_df['customer_id'], num_transactions),
    'product_id': np.random.choice(products_df['product_id'], num_transactions),
    'quantity': np.random.randint(1, 4, num_transactions)
})

# Merge signup_date and other details for purchase_date and is_returned logic
temp_df = transactions_df.merge(customers_df[['customer_id', 'signup_date', 'loyalty_status']], on='customer_id', how='left')
temp_df = temp_df.merge(products_df[['product_id', 'category', 'unit_price', 'brand']], on='product_id', how='left')

# Generate purchase_date after signup_date
max_days_after_signup = (today - three_years_ago).days
random_days_offset = np.random.randint(1, max_days_after_signup, num_transactions)
transactions_df['purchase_date'] = temp_df['signup_date'] + pd.to_timedelta(random_days_offset, unit='days')

# Ensure purchase_date is not in the future
transactions_df['purchase_date'] = transactions_df['purchase_date'].apply(lambda x: min(x, today))

# Simulate is_returned with biases
# Base return probability
p_return = np.full(num_transactions, 0.08) # Base 8%

# Bias by unit_price
p_return[temp_df['unit_price'] > 700] += 0.07 # High price, higher return
p_return[temp_df['unit_price'] > 1200] += 0.05 # Very high price, even higher return

# Bias by category
p_return[temp_df['category'] == 'Apparel'] += 0.08
p_return[temp_df['category'] == 'Electronics'] += 0.03 # Electronics might have specific return reasons

# Bias by brand
p_return[temp_df['brand'] == 'BrandA'] += 0.05 # Specific brands might have higher returns

# Bias by loyalty_status (newer customers / Bronze)
p_return[temp_df['loyalty_status'] == 'Bronze'] += 0.04

# Calculate days_since_signup_at_purchase for temporary use
days_since_signup_at_purchase_temp = (transactions_df['purchase_date'] - temp_df['signup_date']).dt.days
p_return[days_since_signup_at_purchase_temp < 180] += 0.05 # Newer customers (less than 6 months)

# Bias by quantity (minor inverse relationship)
p_return[temp_df['quantity'] == 1] += 0.02 # Buying only one might indicate less commitment

# Clamp probabilities between 0 and 1
p_return = np.clip(p_return, 0.01, 0.7) # Min 1% max 70% return prob

transactions_df['is_returned'] = (np.random.rand(num_transactions) < p_return).astype(int)

# Check overall return rate
overall_return_rate = transactions_df['is_returned'].mean()
print(f"Synthetic data generated. Overall return rate: {overall_return_rate:.2%}")
if not (0.10 <= overall_return_rate <= 0.15):
    print("Warning: Overall return rate is outside the 10-15% target range. Adjusting base probability for better fit.")
    # Attempt to adjust base probability if it's far off
    adjustment_factor = 0.125 / overall_return_rate # Target 12.5%
    if adjustment_factor < 0.5 or adjustment_factor > 2: # Prevent extreme adjustments
        adjustment_factor = 1.0 # Don't adjust much if already very off
    
    adjusted_p_return = np.clip(p_return * adjustment_factor, 0.01, 0.7)
    transactions_df['is_returned'] = (np.random.rand(num_transactions) < adjusted_p_return).astype(int)
    overall_return_rate = transactions_df['is_returned'].mean()
    print(f"Adjusted return rate: {overall_return_rate:.2%}")


# Sort transactions_df
transactions_df.sort_values(by=['customer_id', 'purchase_date'], inplace=True)
transactions_df.reset_index(drop=True, inplace=True)

print("\n--- Data Generation Complete ---")

# 2. Load into SQLite & SQL Feature Engineering
print("Loading data into in-memory SQLite database and performing SQL feature engineering...")
conn = sqlite3.connect(':memory:')

customers_df.to_sql('customers', conn, index=False, if_exists='replace')
products_df.to_sql('products', conn, index=False, if_exists='replace')

# Convert datetime columns to string before loading into SQLite for consistency
# SQLite 'DATE' type stores as TEXT, so ISO format is good.
transactions_df['purchase_date'] = transactions_df['purchase_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
customers_df['signup_date'] = customers_df['signup_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

transactions_df.to_sql('transactions', conn, index=False, if_exists='replace')

sql_query = """
SELECT
    t.transaction_id,
    t.customer_id,
    t.product_id,
    t.purchase_date,
    t.quantity,
    t.is_returned,
    c.signup_date,
    c.region,
    c.loyalty_status,
    p.category,
    p.unit_price,
    p.brand,
    (t.quantity * p.unit_price) AS transaction_value,
    CAST(JULIANDAY(t.purchase_date) - JULIANDAY(c.signup_date) AS INTEGER) AS days_since_signup_at_purchase
FROM
    transactions t
JOIN
    customers c ON t.customer_id = c.customer_id
JOIN
    products p ON t.product_id = p.product_id
"""

transaction_features_df = pd.read_sql_query(sql_query, conn)
conn.close()
print("SQL feature engineering complete. Data loaded into pandas DataFrame.")

# 3. Pandas Feature Engineering & Binary Target Creation
print("Performing Pandas feature engineering and preparing data for ML...")
# Convert date columns to datetime objects
transaction_features_df['signup_date'] = pd.to_datetime(transaction_features_df['signup_date'])
transaction_features_df['purchase_date'] = pd.to_datetime(transaction_features_df['purchase_date'])

# Handle NaNs: days_since_signup_at_purchase might be negative if there were edge cases or future dates,
# but our generation avoids that. If any NaNs appear (e.g. from SQL conversion issues), impute.
# For simplicity, fill with median if any are truly NaN after conversion.
# In this specific case, `days_since_signup_at_purchase` from JULIANDAY difference should not be NaN if dates are valid.
if transaction_features_df['days_since_signup_at_purchase'].isnull().any():
    median_days = transaction_features_df['days_since_signup_at_purchase'].median()
    transaction_features_df['days_since_signup_at_purchase'].fillna(median_days, inplace=True)
    print(f"NaNs in days_since_signup_at_purchase imputed with median: {median_days}")

# Define features (X) and target (y)
numerical_features = ['quantity', 'unit_price', 'transaction_value', 'days_since_signup_at_purchase']
categorical_features = ['region', 'loyalty_status', 'category', 'brand']
all_features = numerical_features + categorical_features

X = transaction_features_df[all_features]
y = transaction_features_df['is_returned']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")
print(f"Training set return rate: {y_train.mean():.2%}")
print(f"Testing set return rate: {y_test.mean():.2%}")


# 4. Data Visualization
print("\n--- Generating Data Visualizations ---")
# Use a BytesIO object to capture plots without displaying them in an interactive environment
# This is useful for scripts run in non-GUI environments or where plots are saved.
plt.ioff() # Turn off interactive plotting

# Plot 1: Violin plot of unit_price vs. is_returned
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.violinplot(x='is_returned', y='unit_price', data=transaction_features_df, ax=ax1, palette='viridis')
ax1.set_title('Distribution of Unit Price for Returned vs. Non-Returned Items')
ax1.set_xlabel('Is Returned (0=No, 1=Yes)')
ax1.set_ylabel('Unit Price')
plt.tight_layout()
buffer1 = io.BytesIO()
plt.savefig(buffer1, format='png')
buffer1.seek(0)
print("Violin plot generated (saved to buffer).")

# Plot 2: Stacked bar chart of proportion of is_returned across different categories
fig2, ax2 = plt.subplots(figsize=(10, 7))
category_return_proportions = pd.crosstab(
    transaction_features_df['category'],
    transaction_features_df['is_returned'],
    normalize='index'
)
category_return_proportions.plot(kind='bar', stacked=True, ax=ax2, colormap='Paired')
ax2.set_title('Proportion of Returned vs. Non-Returned Items by Category')
ax2.set_xlabel('Product Category')
ax2.set_ylabel('Proportion')
ax2.legend(title='Is Returned', labels=['No', 'Yes'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
buffer2 = io.BytesIO()
plt.savefig(buffer2, format='png')
buffer2.seek(0)
print("Stacked bar chart generated (saved to buffer).")

plt.close('all') # Close all plots to free up memory
print("Visualization buffers are available (not printed to stdout directly for script execution).")


# 5. ML Pipeline & Evaluation
print("\n--- Building and Training ML Pipeline ---")

# Preprocessing steps for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline
# HistGradientBoostingClassifier chosen for performance and handling of mixed data types
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', HistGradientBoostingClassifier(random_state=42))])

# Train the pipeline
model_pipeline.fit(X_train, y_train)
print("ML Pipeline training complete.")

# Predict probabilities for the positive class (class 1: returned)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Predict classes for the classification report
y_pred = model_pipeline.predict(X_test)

print("\n--- Model Evaluation ---")
# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score on Test Set: {roc_auc:.4f}")

# Print Classification Report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred, target_names=['Not Returned (0)', 'Returned (1)']))

print("\n--- Pipeline Execution Complete ---")