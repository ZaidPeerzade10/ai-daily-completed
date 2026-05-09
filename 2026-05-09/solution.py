import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Not explicitly used for plotting dates, but good to import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data ---
print("--- Generating Synthetic Data ---")

# --- Stores DataFrame ---
num_stores = np.random.randint(100, 151)
store_types = ['Hypermarket', 'Supermarket', 'Convenience']
regions = ['North', 'South', 'East', 'West', 'Central']
store_type_probs = [0.2, 0.5, 0.3] # Hypermarkets are fewer but larger contributors

stores_df = pd.DataFrame({
    'store_id': range(num_stores),
    'store_type': np.random.choice(store_types, size=num_stores, p=store_type_probs),
    'region': np.random.choice(regions, size=num_stores),
})
# Random opening dates over the last 5 years, ensuring not in future
today = pd.Timestamp.now()
stores_df['opening_date'] = pd.to_datetime(today - pd.to_timedelta(np.random.randint(365 * 1, 365 * 5, size=num_stores), unit='D'))
stores_df['opening_date'] = stores_df['opening_date'].apply(lambda x: x if x < today else today - pd.Timedelta(days=1))


# --- Products DataFrame ---
num_products = np.random.randint(50, 71)
product_categories = ['Dairy', 'Bakery', 'Produce', 'Snacks', 'Beverages', 'HomeGoods']
# Simulate higher transaction frequency for Produce/Dairy by increasing their probability of being chosen
product_category_probs = [0.2, 0.15, 0.25, 0.15, 0.15, 0.1] # Produce, Dairy more likely

products_df = pd.DataFrame({
    'product_id': range(num_products),
    'product_category': np.random.choice(product_categories, size=num_products, p=product_category_probs),
})
products_df['unit_cost'] = np.random.uniform(1.0, 50.0, size=num_products)
products_df['retail_price'] = products_df['unit_cost'] * np.random.uniform(1.5, 3.0, size=num_products)


# --- Sales DataFrame ---
num_sales = np.random.randint(50000, 80001)

# Generate base sales structure
sales_data = {
    'sale_id': range(num_sales),
    'store_id': np.random.choice(stores_df['store_id'], size=num_sales),
    'product_id': np.random.choice(products_df['product_id'], size=num_sales),
}
sales_df = pd.DataFrame(sales_data)

# Merge to get necessary info for realistic sales generation
sales_df = pd.merge(sales_df, stores_df[['store_id', 'opening_date', 'store_type']], on='store_id', how='left')
sales_df = pd.merge(sales_df, products_df[['product_id', 'product_category', 'retail_price']], on='product_id', how='left')

# Define the maximum allowed sale date (2 weeks prior to now)
max_allowed_sale_date = today - pd.Timedelta(weeks=2)

# Generate sale_date ensuring it's after opening_date and before max_allowed_sale_date
sales_df['days_since_opening_for_sale'] = sales_df.apply(
    lambda row: np.random.randint(1, (max_allowed_sale_date - row['opening_date']).days + 1)
    if (max_allowed_sale_date - row['opening_date']).days >= 1 else 1, axis=1
)
sales_df['sale_date'] = sales_df['opening_date'] + pd.to_timedelta(sales_df['days_since_opening_for_sale'], unit='D')
sales_df['sale_date'] = sales_df['sale_date'].apply(lambda x: x if x <= max_allowed_sale_date else max_allowed_sale_date)


# Simulate quantity_sold
sales_df['quantity_sold'] = np.random.randint(1, 10, size=num_sales)
# 'Hypermarket' stores should have generally higher quantity_sold per sale
hypermarket_mask = sales_df['store_type'] == 'Hypermarket'
sales_df.loc[hypermarket_mask, 'quantity_sold'] += np.random.randint(3, 7, size=hypermarket_mask.sum())
# 'Produce' and 'Dairy' categories might have lower quantity_sold per transaction
produce_dairy_mask = sales_df['product_category'].isin(['Produce', 'Dairy'])
sales_df.loc[produce_dairy_mask, 'quantity_sold'] = sales_df.loc[produce_dairy_mask, 'quantity_sold'].apply(lambda x: max(1, x - np.random.randint(0, 3))) # Reduce by 0-2, min 1

sales_df['revenue'] = sales_df['quantity_sold'] * sales_df['retail_price']

# Clean up temporary columns and sort
sales_df = sales_df.drop(columns=['opening_date', 'store_type', 'product_category', 'retail_price', 'days_since_opening_for_sale'])
sales_df = sales_df.sort_values(by=['store_id', 'product_id', 'sale_date']).reset_index(drop=True)

print(f"Generated {len(stores_df)} stores, {len(products_df)} products, {len(sales_df)} sales records.")
print("Stores Head:\n", stores_df.head())
print("Products Head:\n", products_df.head())
print("Sales Head:\n", sales_df.head())

# --- 2. Load into SQLite & SQL Feature Engineering ---
print("\n--- Loading Data into SQLite and SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

# Convert datetime to string for SQLite storage
stores_df['opening_date'] = stores_df['opening_date'].dt.strftime('%Y-%m-%d')
sales_df['sale_date'] = sales_df['sale_date'].dt.strftime('%Y-%m-%d')

stores_df.to_sql('stores', conn, index=False, if_exists='replace')
products_df.to_sql('products', conn, index=False, if_exists='replace')
sales_df.to_sql('sales', conn, index=False, if_exists='replace')

# Determine GLOBAL_PREDICTION_CUTOFF_DATE
latest_sale_date_str = pd.read_sql("SELECT MAX(sale_date) FROM sales", conn).iloc[0, 0]
GLOBAL_PREDICTION_CUTOFF_DATE = pd.to_datetime(latest_sale_date_str) - pd.Timedelta(weeks=2)
print(f"Global Prediction Cutoff Date: {GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}")

# SQL Query for feature engineering
sql_query = f"""
WITH AllStoreCategoryCombos AS (
    SELECT
        s.store_id,
        s.store_type,
        s.region,
        s.opening_date,
        p.product_category
    FROM stores AS s
    CROSS JOIN (SELECT DISTINCT product_category FROM products) AS p
),
SalesAggregates AS (
    SELECT
        sa.store_id,
        pr.product_category,
        SUM(CASE WHEN julianday(sa.sale_date) BETWEEN julianday('{GLOBAL_PREDICTION_CUTOFF_DATE - pd.Timedelta(days=30)}') AND julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}') THEN sa.quantity_sold ELSE 0 END) AS total_quantity_prev_30d_raw,
        SUM(CASE WHEN julianday(sa.sale_date) BETWEEN julianday('{GLOBAL_PREDICTION_CUTOFF_DATE - pd.Timedelta(days=30)}') AND julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}') THEN sa.revenue ELSE 0 END) AS total_revenue_prev_30d_raw,
        COUNT(DISTINCT CASE WHEN julianday(sa.sale_date) BETWEEN julianday('{GLOBAL_PREDICTION_CUTOFF_DATE - pd.Timedelta(days=30)}') AND julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}') THEN sa.sale_date ELSE NULL END) AS num_sales_days_prev_30d_raw,
        MAX(CASE WHEN julianday(sa.sale_date) <= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}') THEN sa.sale_date ELSE NULL END) AS last_sale_date_at_cutoff_raw,
        AVG(CASE WHEN julianday(sa.sale_date) BETWEEN julianday('{GLOBAL_PREDICTION_CUTOFF_DATE - pd.Timedelta(days=30)}') AND julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}') THEN sa.quantity_sold ELSE NULL END) AS avg_quantity_per_sale_prev_30d_raw
    FROM sales AS sa
    JOIN products AS pr ON sa.product_id = pr.product_id
    GROUP BY sa.store_id, pr.product_category
)
SELECT
    ascc.store_id,
    ascc.store_type,
    ascc.region,
    ascc.opening_date,
    ascc.product_category,
    '{GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}' AS current_cutoff_date,
    COALESCE(sa.total_quantity_prev_30d_raw, 0) AS total_quantity_prev_30d,
    COALESCE(sa.total_revenue_prev_30d_raw, 0.0) AS total_revenue_prev_30d,
    COALESCE(sa.num_sales_days_prev_30d_raw, 0) AS num_sales_days_prev_30d,
    CASE
        WHEN sa.last_sale_date_at_cutoff_raw IS NOT NULL THEN CAST(julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}') - julianday(sa.last_sale_date_at_cutoff_raw) AS INTEGER)
        ELSE 9999
    END AS days_since_last_sale_at_cutoff,
    COALESCE(sa.avg_quantity_per_sale_prev_30d_raw, 0.0) AS avg_quantity_per_sale_prev_30d
FROM AllStoreCategoryCombos AS ascc
LEFT JOIN SalesAggregates AS sa
    ON ascc.store_id = sa.store_id
    AND ascc.product_category = sa.product_category
ORDER BY ascc.store_id, ascc.product_category;
"""

store_category_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Generated {len(store_category_features_df)} store-category feature combinations.")
print("Features DataFrame Head:\n", store_category_features_df.head())


# --- 3. Pandas Feature Engineering & Regression Target Creation ---
print("\n--- Pandas Feature Engineering & Regression Target Creation ---")

store_category_features_df['opening_date'] = pd.to_datetime(store_category_features_df['opening_date'])
store_category_features_df['current_cutoff_date'] = pd.to_datetime(store_category_features_df['current_cutoff_date'])

# Handle NaNs (if any, though SQL COALESCE should largely prevent this)
numerical_agg_cols = [
    'total_quantity_prev_30d', 'total_revenue_prev_30d',
    'num_sales_days_prev_30d', 'avg_quantity_per_sale_prev_30d'
]
for col in numerical_agg_cols:
    store_category_features_df[col] = store_category_features_df[col].fillna(0)
store_category_features_df['days_since_last_sale_at_cutoff'] = store_category_features_df['days_since_last_sale_at_cutoff'].fillna(9999).astype(int)


# Calculate days_since_store_opened_at_cutoff
store_category_features_df['days_since_store_opened_at_cutoff'] = (
    store_category_features_df['current_cutoff_date'] - store_category_features_df['opening_date']
).dt.days
store_category_features_df['days_since_store_opened_at_cutoff'] = store_category_features_df['days_since_store_opened_at_cutoff'].clip(lower=0)


# Calculate avg_daily_quantity_prev_30d
store_category_features_df['avg_daily_quantity_prev_30d'] = store_category_features_df['total_quantity_prev_30d'] / 30.0
store_category_features_df['avg_daily_quantity_prev_30d'] = store_category_features_df['avg_daily_quantity_prev_30d'].replace([np.inf, -np.inf], 0).fillna(0)


# Create Regression Target: next_7d_category_sales_quantity
target_start_date = GLOBAL_PREDICTION_CUTOFF_DATE + pd.Timedelta(days=1)
target_end_date = GLOBAL_PREDICTION_CUTOFF_DATE + pd.Timedelta(days=7)

# Convert sales_df['sale_date'] back to datetime for pandas ops
sales_df['sale_date'] = pd.to_datetime(sales_df['sale_date'])

target_sales_df = sales_df[
    (sales_df['sale_date'] >= target_start_date) &
    (sales_df['sale_date'] <= target_end_date)
].copy()

# Merge product category info back to target_sales_df for aggregation
# (original sales_df no longer has product_category after initial merge and drop)
target_sales_df = pd.merge(target_sales_df, products_df[['product_id', 'product_category']], on='product_id', how='left')

next_7d_category_sales = target_sales_df.groupby(['store_id', 'product_category'])['quantity_sold'].sum().reset_index()
next_7d_category_sales.rename(columns={'quantity_sold': 'next_7d_category_sales_quantity'}, inplace=True)

# Merge target into main features DataFrame
store_category_features_df = pd.merge(
    store_category_features_df,
    next_7d_category_sales,
    on=['store_id', 'product_category'],
    how='left'
)
store_category_features_df['next_7d_category_sales_quantity'] = store_category_features_df['next_7d_category_sales_quantity'].fillna(0)

print("Features and Target DataFrame Head:\n", store_category_features_df.head())
print("Target distribution (top 10 values, showing sparsity):\n", store_category_features_df['next_7d_category_sales_quantity'].value_counts().head(10))

# Define features (X) and target (y)
numerical_features = [
    'total_quantity_prev_30d', 'total_revenue_prev_30d', 'num_sales_days_prev_30d',
    'days_since_last_sale_at_cutoff', 'avg_quantity_per_sale_prev_30d',
    'days_since_store_opened_at_cutoff', 'avg_daily_quantity_prev_30d'
]
categorical_features = ['store_type', 'region', 'product_category']

X = store_category_features_df[numerical_features + categorical_features]
y = store_category_features_df['next_7d_category_sales_quantity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining data shape: {X_train.shape}, Test data shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}, Test target shape: {y_test.shape}")


# --- 4. Data Visualization ---
print("\n--- Data Visualization ---")

# Scatter plot: total_quantity_prev_30d vs. next_7d_category_sales_quantity
plt.figure(figsize=(10, 6))
plt.scatter(
    np.log1p(store_category_features_df['total_quantity_prev_30d']),
    np.log1p(store_category_features_df['next_7d_category_sales_quantity']),
    alpha=0.5
)
plt.title('Log1p(Total Quantity Prev 30d) vs. Log1p(Next 7d Category Sales Quantity)')
plt.xlabel('Log1p(Total Quantity Prev 30d)')
plt.ylabel('Log1p(Next 7d Category Sales Quantity)')
plt.grid(True)
plt.show()

# Box plot: next_7d_category_sales_quantity across different store_type values
plt.figure(figsize=(12, 7))
store_category_features_df.boxplot(column='next_7d_category_sales_quantity', by='store_type', grid=False, rot=45)
plt.title('Next 7d Category Sales Quantity by Store Type')
plt.suptitle('') # Suppress the default matplotlib suptitle
plt.xlabel('Store Type')
plt.ylabel('Next 7d Category Sales Quantity')
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---
print("\n--- ML Pipeline & Evaluation ---")

# Preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Handles potential NaNs introduced by avg/division
    ('scaler', StandardScaler())
])

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
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

# Train the model
print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation on Test Set:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")