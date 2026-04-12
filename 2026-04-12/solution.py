import pandas as pd
import numpy as np
import sqlite3
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set random seed for reproducibility
np.random.seed(42)
# Set matplotlib backend for non-interactive environment
plt.switch_backend('Agg')

# 1. Generate Synthetic Data
# --- Customers DataFrame ---
num_customers = np.random.randint(1000, 1501)
customer_ids = np.arange(1, num_customers + 1)
signup_dates = pd.to_datetime('2021-01-01') + pd.to_timedelta(np.random.randint(0, 3 * 365, size=num_customers), unit='D')
customer_segments = np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=num_customers, p=[0.4, 0.3, 0.2, 0.1])
customers_df = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates,
    'customer_segment': customer_segments
})

# --- Warehouses DataFrame ---
num_warehouses = np.random.randint(10, 16)
warehouse_ids = np.arange(1, num_warehouses + 1)
location_cities = np.random.choice([
    'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
    'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte'
], size=num_warehouses, replace=False)
operational_capacity_pct = np.random.uniform(70.0, 100.0, size=num_warehouses)
warehouses_df = pd.DataFrame({
    'warehouse_id': warehouse_ids,
    'location_city': location_cities,
    'operational_capacity_pct': operational_capacity_pct
})

# --- Orders DataFrame ---
num_orders = np.random.randint(15000, 20001)
order_ids = np.arange(1, num_orders + 1)
customer_ids_sampled = np.random.choice(customers_df['customer_id'], size=num_orders)
warehouse_ids_sampled = np.random.choice(warehouses_df['warehouse_id'], size=num_orders)

# Temporarily merge signup_date to ensure order_date > signup_date for generation
temp_orders_df = pd.DataFrame({
    'order_id': order_ids,
    'customer_id': customer_ids_sampled,
    'warehouse_id': warehouse_ids_sampled
}).merge(customers_df[['customer_id', 'signup_date']], on='customer_id', how='left')

# Generate order_date after signup_date
order_dates = temp_orders_df['signup_date'] + pd.to_timedelta(np.random.randint(1, 3 * 365 + 100, size=num_orders), unit='D')
# Ensure all order dates are in the past, up to yesterday to allow future delivery dates for actual_delivery_date
max_order_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
order_dates[order_dates > max_order_date] = max_order_date - pd.to_timedelta(np.random.randint(0, 30, size=(order_dates > max_order_date).sum()), unit='D')
temp_orders_df['order_date'] = order_dates

total_order_value = np.random.uniform(20.0, 500.0, size=num_orders)
shipping_methods = np.random.choice(['Standard', 'Express', 'Priority'], size=num_orders, p=[0.6, 0.3, 0.1])
destination_regions = np.random.choice(['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest'], size=num_orders, p=[0.2, 0.2, 0.25, 0.2, 0.15])

orders_df = pd.DataFrame({
    'order_id': order_ids,
    'customer_id': customer_ids_sampled,
    'order_date': temp_orders_df['order_date'],
    'total_order_value': total_order_value,
    'shipping_method': shipping_methods,
    'warehouse_id': warehouse_ids_sampled,
    'destination_region': destination_regions
})

# Simulate realistic delivery patterns and delays
# Merge necessary columns for delay calculation (customer_segment, operational_capacity_pct)
orders_df_temp = orders_df.merge(customers_df[['customer_id', 'customer_segment']], on='customer_id', how='left')
orders_df_temp = orders_df_temp.merge(warehouses_df[['warehouse_id', 'operational_capacity_pct']], on='warehouse_id', how='left')

# Define base expected delivery days
base_expected_delivery_days_map = {
    'Standard': 6,
    'Express': 3,
    'Priority': 1.5
}
orders_df_temp['base_expected_delivery_days'] = orders_df_temp['shipping_method'].map(base_expected_delivery_days_map)

# Initialize random_noise_days
orders_df_temp['random_noise_days'] = np.random.normal(loc=0.5, scale=1.0, size=num_orders)
orders_df_temp['random_noise_days'] = orders_df_temp['random_noise_days'].clip(lower=-0.5) # Minimum noise to avoid too early delivery

# Add conditional noise biases
# Bias 1: 'Standard' shipping or 'Midwest' destination are slightly more prone to larger random_noise_days
cond_standard_midwest = (orders_df_temp['shipping_method'] == 'Standard') | (orders_df_temp['destination_region'] == 'Midwest')
orders_df_temp.loc[cond_standard_midwest, 'random_noise_days'] += np.random.uniform(1.0, 3.0, size=cond_standard_midwest.sum())

# Bias 2: Orders from warehouses with lower operational_capacity_pct (< 80%) are more likely to experience delays
cond_low_capacity = orders_df_temp['operational_capacity_pct'] < 80
orders_df_temp.loc[cond_low_capacity, 'random_noise_days'] += np.random.uniform(0.5, 2.0, size=cond_low_capacity.sum())

# Bias 3: Orders from 'Bronze' customer_segment might also have higher delay risk
cond_bronze_segment = orders_df_temp['customer_segment'] == 'Bronze'
orders_df_temp.loc[cond_bronze_segment, 'random_noise_days'] += np.random.uniform(0.5, 1.5, size=cond_bronze_segment.sum())

# Calculate actual_delivery_date
orders_df_temp['actual_delivery_date'] = orders_df_temp['order_date'] + pd.to_timedelta(orders_df_temp['base_expected_delivery_days'] + orders_df_temp['random_noise_days'], unit='D')

# Define is_delayed
# A shipment is_delayed=1 if actual_delivery_date is more than (base_expected_days + 1.5 days) from order_date
delay_threshold_extra_days = 1.5
orders_df_temp['is_delayed'] = ((orders_df_temp['actual_delivery_date'] - orders_df_temp['order_date']).dt.total_seconds() / (24 * 3600)) > (orders_df_temp['base_expected_delivery_days'] + delay_threshold_extra_days)
orders_df_temp['is_delayed'] = orders_df_temp['is_delayed'].astype(int)

# Transfer actual_delivery_date and is_delayed to the final orders_df
orders_df['actual_delivery_date'] = orders_df_temp['actual_delivery_date']
orders_df['is_delayed'] = orders_df_temp['is_delayed']

# Sort orders_df
orders_df = orders_df.sort_values(by=['customer_id', 'order_date']).reset_index(drop=True)

# Print initial DataFrame info
print("--- Synthetic Data Generation Summary ---")
print(f"Customers generated: {len(customers_df)}")
print(f"Warehouses generated: {len(warehouses_df)}")
print(f"Orders generated: {len(orders_df)}")
print(f"Overall delay rate: {orders_df['is_delayed'].mean() * 100:.2f}%")
print("\nCustomers_df head:")
print(customers_df.head())
print("\nWarehouses_df head:")
print(warehouses_df.head())
print("\nOrders_df head:")
print(orders_df.head())

# 2. Load into SQLite & SQL Feature Engineering
conn = sqlite3.connect(':memory:')

customers_df.to_sql('customers', conn, index=False, if_exists='replace')
warehouses_df.to_sql('warehouses', conn, index=False, if_exists='replace')
orders_df.to_sql('orders', conn, index=False, if_exists='replace')

sql_query = """
SELECT
    o.order_id,
    o.customer_id,
    o.order_date,
    o.total_order_value,
    o.shipping_method,
    o.destination_region,
    o.is_delayed,
    c.customer_segment,
    w.operational_capacity_pct,
    o.actual_delivery_date,
    -- Engineered features
    CAST(strftime('%w', o.order_date) AS INTEGER) AS order_day_of_week_sql, -- SQLite %w is 0=Sunday, 1=Monday...
    CAST(strftime('%m', o.order_date) AS INTEGER) AS order_month,
    julianday(o.order_date) - julianday(c.signup_date) AS days_since_customer_signup_at_order
FROM orders AS o
JOIN customers AS c ON o.customer_id = c.customer_id
JOIN warehouses AS w ON o.warehouse_id = w.warehouse_id
ORDER BY o.customer_id, o.order_date;
"""

order_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

# Adjust order_day_of_week from SQLite's 0=Sunday, 1=Monday to 1=Monday, 7=Sunday
order_features_df['order_day_of_week'] = order_features_df['order_day_of_week_sql'].replace(0, 7) # Replace Sunday (0) with 7
order_features_df['order_day_of_week'] = order_features_df['order_day_of_week'].replace({1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}) # Map 1-6 to themselves
# It's actually: (SQLite %w + 6) % 7 + 1. For instance, 0 (Sun) -> (0+6)%7+1 = 0+1 = 1 (Mon). No, this is wrong.
# If SQLite gives 0=Sunday, 1=Monday, ..., 6=Saturday
# Desired: 1=Monday, ..., 6=Saturday, 7=Sunday
# Map: 1->1, 2->2, ..., 6->6, 0->7
order_features_df['order_day_of_week'] = order_features_df['order_day_of_week_sql'].map({
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 0: 7
})
order_features_df.drop(columns=['order_day_of_week_sql'], inplace=True)


print("\n--- SQL Feature Engineering Result ---")
print("Order_features_df head after SQL query:")
print(order_features_df.head())
print(f"Shape of order_features_df: {order_features_df.shape}")

# 3. Pandas Feature Engineering & Binary Target Creation
order_features_df['order_date'] = pd.to_datetime(order_features_df['order_date'])
order_features_df['actual_delivery_date'] = pd.to_datetime(order_features_df['actual_delivery_date'])

# Calculate delivery_lead_time_actual_days (for visualization, not ML features to prevent leakage)
order_features_df['delivery_lead_time_actual_days'] = (order_features_df['actual_delivery_date'] - order_features_df['order_date']).dt.total_seconds() / (24 * 3600)

# Calculate expected_delivery_days (for visualization, not ML features to prevent leakage)
expected_delivery_days_map = {
    'Standard': 6,
    'Express': 3,
    'Priority': 1.5
}
order_features_df['expected_delivery_days'] = order_features_df['shipping_method'].map(expected_delivery_days_map).fillna(0.0)

# Calculate order_value_per_day_since_signup
order_features_df['order_value_per_day_since_signup'] = order_features_df['total_order_value'] / (order_features_df['days_since_customer_signup_at_order'] + 1)
order_features_df['order_value_per_day_since_signup'] = order_features_df['order_value_per_day_since_signup'].fillna(0.0)


# Define features X and target y
numerical_features = [
    'total_order_value',
    'order_day_of_week',
    'order_month',
    'days_since_customer_signup_at_order',
    'operational_capacity_pct',
    'order_value_per_day_since_signup'
]
categorical_features = [
    'shipping_method',
    'destination_region',
    'customer_segment'
]

# Ensure all selected features exist in the DataFrame
missing_num_features = [f for f in numerical_features if f not in order_features_df.columns]
missing_cat_features = [f for f in categorical_features if f not in order_features_df.columns]
if missing_num_features or missing_cat_features:
    raise ValueError(f"Missing features: Numerical: {missing_num_features}, Categorical: {missing_cat_features}")

X = order_features_df[numerical_features + categorical_features]
y = order_features_df['is_delayed']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\n--- Feature Engineering & Data Split ---")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("X_train head (selected features):")
print(X_train.head())
print(f"Target distribution in y_train:\n{y_train.value_counts(normalize=True)}")
print(f"Target distribution in y_test:\n{y_test.value_counts(normalize=True)}")


# 4. Data Visualization
print("\n--- Data Visualization (generating plots) ---")

# Violin plot for delivery_lead_time_actual_days vs is_delayed
plt.figure(figsize=(10, 6))
sns.violinplot(x='is_delayed', y='delivery_lead_time_actual_days', data=order_features_df)
plt.title('Distribution of Actual Delivery Lead Time by Delay Status')
plt.xlabel('Is Delayed (0: No, 1: Yes)')
plt.ylabel('Actual Delivery Lead Time (Days)')
# Save plot to buffer
buf_violin = io.BytesIO()
plt.savefig(buf_violin, format='png')
buf_violin.seek(0)
print("Violin plot generated and saved to buffer.")
plt.close() # Close plot to free memory

# Stacked bar chart for is_delayed across shipping_method
shipping_method_delay_prop = order_features_df.groupby('shipping_method')['is_delayed'].value_counts(normalize=True).unstack().fillna(0)
# Ensure both 0 and 1 columns exist, even if one is all zeros
for col in [0, 1]:
    if col not in shipping_method_delay_prop.columns:
        shipping_method_delay_prop[col] = 0.0

# Reorder columns to ensure 0 comes before 1 for stacking (non-delayed then delayed)
shipping_method_delay_prop = shipping_method_delay_prop[[0, 1]]

plt.figure(figsize=(10, 6))
shipping_method_delay_prop.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e']) # Blue for not delayed, Orange for delayed
plt.title('Proportion of Delayed Orders by Shipping Method')
plt.xlabel('Shipping Method')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Is Delayed', labels=['No', 'Yes'])
plt.tight_layout()
# Save plot to buffer
buf_bar = io.BytesIO()
plt.savefig(buf_bar, format='png')
buf_bar.seek(0)
print("Stacked bar chart generated and saved to buffer.")
plt.close() # Close plot to free memory


# 5. ML Pipeline & Evaluation (Binary Classification)
print("\n--- ML Pipeline & Evaluation ---")

# Preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# Create the ML pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("Training the model pipeline...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict probabilities on the test set for ROC AUC
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Predict classes for classification report (using default threshold 0.5)
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate and print ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

# Calculate and print Classification Report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

print("\n--- Script Execution Complete ---")