import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

print("Starting ML Pipeline for Product Return Prediction...\n")

# --- 1. Synthetic Data Generation ---
print("1. Generating Synthetic Data...")

# Seed for reproducibility
np.random.seed(42)
num_customers = 500
num_products = 200
num_orders = 5000
# num_order_items is implicitly determined by the number of items per order loop
num_returns_to_simulate_target = 2000 # Aim for a good number of positive samples

# Customers
customer_signup_dates = pd.to_datetime(pd.date_range(start='2020-01-01', end='2023-12-31', periods=num_customers))
customers_df = pd.DataFrame({
    'customer_id': range(1, num_customers + 1),
    'customer_signup_date': np.random.choice(customer_signup_dates, num_customers, replace=False),
    'loyalty_status': np.random.choice(['Bronze', 'Silver', 'Gold'], num_customers, p=[0.5, 0.3, 0.2]),
    'gender': np.random.choice(['Male', 'Female', 'Other'], num_customers, p=[0.48, 0.48, 0.04]),
    'age': np.random.randint(18, 70, num_customers)
})

# Products
product_categories = ['Electronics', 'Apparel', 'Home Goods', 'Books', 'Groceries', 'Beauty']
products_df = pd.DataFrame({
    'product_id': range(1, num_products + 1),
    'product_name': [f'Product_{i}' for i in range(1, num_products + 1)],
    'product_category': np.random.choice(product_categories, num_products, p=[0.2, 0.3, 0.2, 0.1, 0.1, 0.1]),
    'unit_price': np.round(np.random.uniform(5, 500, num_products), 2)
})

# Orders
order_dates = pd.to_datetime(pd.date_range(start='2022-01-01', end='2024-03-01', periods=num_orders))
orders_df = pd.DataFrame({
    'order_id': range(1, num_orders + 1),
    'customer_id': np.random.choice(customers_df['customer_id'], num_orders),
    'order_date': np.random.choice(order_dates, num_orders, replace=False),
    'shipping_cost': np.round(np.random.uniform(3, 20, num_orders), 2),
    'discount_applied': np.random.uniform(0, 0.2, num_orders)
})
orders_df = orders_df.sort_values('order_date').reset_index(drop=True) # Ensure order dates are sequential for time-based features

# Order Items
order_items_data = []
for order_id in orders_df['order_id']:
    num_items_in_order = np.random.randint(1, 6) # 1 to 5 items per order
    for _ in range(num_items_in_order):
        product = products_df.sample(1).iloc[0]
        quantity = np.random.randint(1, 4)
        order_items_data.append({
            'order_id': order_id,
            'product_id': product['product_id'],
            'quantity': quantity,
            'item_price': np.round(product['unit_price'] * quantity, 2) # Total price for this item type in the order
        })
order_items_df = pd.DataFrame(order_items_data)
order_items_df['order_item_id'] = range(1, len(order_items_df) + 1)

# Calculate order_total in orders_df
order_totals = order_items_df.groupby('order_id')['item_price'].sum().reset_index()
order_totals.rename(columns={'item_price': 'order_total_items'}, inplace=True)
orders_df = orders_df.merge(order_totals, on='order_id', how='left')
orders_df['order_total'] = orders_df['order_total_items'] + orders_df['shipping_cost'] - (orders_df['order_total_items'] * orders_df['discount_applied'])
orders_df['order_total'] = orders_df['order_total'].apply(lambda x: max(x, 0.01)) # Ensure total is not zero for division
orders_df.drop(columns=['order_total_items'], inplace=True)

# Merge order_items with order and product details for return simulation logic
order_items_enriched = order_items_df.merge(orders_df[['order_id', 'customer_id', 'order_date']], on='order_id', how='left')
order_items_enriched = order_items_enriched.merge(products_df[['product_id', 'product_category']], on='product_id', how='left')
order_items_enriched = order_items_enriched.merge(customers_df[['customer_id', 'customer_signup_date']], on='customer_id', how='left')

# Simulate Returns with stronger signals
returns_data = []
return_prob_base = 0.02 # Base return probability for any item
return_prob_apparel_multiplier = 4.0 # Apparel is 4x more likely to be returned
return_prob_new_customer_multiplier = 2.5 # New customers are 2.5x more likely
new_customer_threshold_days = 180 # Customer signed up within last 6 months is "new"

order_items_enriched['return_prob'] = return_prob_base
order_items_enriched['is_apparel'] = (order_items_enriched['product_category'] == 'Apparel')
order_items_enriched['is_new_customer'] = (order_items_enriched['order_date'] - order_items_enriched['customer_signup_date']).dt.days < new_customer_threshold_days

# Apply multipliers
order_items_enriched.loc[order_items_enriched['is_apparel'], 'return_prob'] *= return_prob_apparel_multiplier
order_items_enriched.loc[order_items_enriched['is_new_customer'], 'return_prob'] *= return_prob_new_customer_multiplier
# Max out return prob at 1.0 (though capped lower to ensure randomness)
order_items_enriched['return_prob'] = order_items_enriched['return_prob'].clip(upper=0.7) # Cap to avoid 100% and allow some randomness

# Determine which items are returned
returned_items_idx = np.random.rand(len(order_items_enriched)) < order_items_enriched['return_prob']
returned_order_items = order_items_enriched[returned_items_idx].copy()

# Simulate return dates for returned items
for idx, row in returned_order_items.iterrows():
    order_date = row['order_date']
    # Ensure return date is always after order_date
    return_delay_days = np.random.randint(1, 61) # Returns within 1 to 60 days
    return_date = order_date + datetime.timedelta(days=return_delay_days)
    returns_data.append({
        'return_id': len(returns_data) + 1,
        'order_item_id': row['order_item_id'],
        'return_date': return_date
    })
returns_df = pd.DataFrame(returns_data)

# Convert all relevant date columns to datetime objects
customers_df['customer_signup_date'] = pd.to_datetime(customers_df['customer_signup_date'])
orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
if not returns_df.empty:
    returns_df['return_date'] = pd.to_datetime(returns_df['return_date'])

print(f"  - Generated {len(customers_df)} customers, {len(products_df)} products, {len(orders_df)} orders, {len(order_items_df)} order items.")
print(f"  - Simulated {len(returns_df)} returns based on custom logic.")

# --- 2. SQL-based Feature Engineering with Prediction Cutoff ---
print("\n2. Performing SQL-based Feature Engineering with Prediction Cutoff...")

# Establish GLOBAL_PREDICTION_CUTOFF_DATE
# Choose a cutoff that leaves a reasonable amount of data for the prediction set
# and enough historical data for features.
GLOBAL_PREDICTION_CUTOFF_DATE = pd.to_datetime('2023-08-01')
print(f"  - Global Prediction Cutoff Date: {GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d')}")

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Prepare dataframes for SQL - convert datetime to string for julianday()
customers_df_sql = customers_df.copy()
products_df_sql = products_df.copy()
orders_df_sql = orders_df.copy()
order_items_df_sql = order_items_df.copy()
returns_df_sql = returns_df.copy() if not returns_df.empty else pd.DataFrame(columns=['return_id', 'order_item_id', 'return_date'])

orders_df_sql['order_date'] = orders_df_sql['order_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
returns_df_sql['return_date'] = returns_df_sql['return_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Write dataframes to SQL tables
customers_df_sql.to_sql('customers', conn, index=False, if_exists='replace')
products_df_sql.to_sql('products', conn, index=False, if_exists='replace')
orders_df_sql.to_sql('orders', conn, index=False, if_exists='replace')
order_items_df_sql.to_sql('order_items', conn, index=False, if_exists='replace')
returns_df_sql.to_sql('returns', conn, index=False, if_exists='replace')

# Convert cutoff date to Julian day for SQL comparison
cutoff_julianday = pd.to_datetime(GLOBAL_PREDICTION_CUTOFF_DATE).to_julian_date()

sql_query = f"""
WITH historical_data AS (
    SELECT
        oi.order_item_id,
        o.customer_id,
        p.product_id,
        p.product_category,
        o.order_date,
        CASE WHEN r.order_item_id IS NOT NULL THEN 1.0 ELSE 0.0 END AS is_returned
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN returns r ON oi.order_item_id = r.order_item_id
    WHERE julianday(o.order_date) <= {cutoff_julianday}
)
SELECT
    t_oi.order_item_id,
    COALESCE(ca.customer_avg_return_rate_prev_6m, 0.0) AS customer_avg_return_rate_prev_6m,
    COALESCE(pa.product_avg_return_rate_all_time_at_cutoff, 0.0) AS product_avg_return_rate_all_time_at_cutoff,
    COALESCE(pca.category_avg_return_rate_all_time_at_cutoff, 0.0) AS category_avg_return_rate_all_time_at_cutoff
FROM order_items t_oi -- t_oi for "target order item"
JOIN orders t_o ON t_oi.order_id = t_o.order_id
JOIN products t_p ON t_oi.product_id = t_p.product_id
LEFT JOIN (
    SELECT
        customer_id,
        AVG(is_returned) AS customer_avg_return_rate_prev_6m
    FROM historical_data
    WHERE julianday(order_date) >= ({cutoff_julianday} - 180) -- 6 months prior to cutoff
    GROUP BY customer_id
) ca ON t_o.customer_id = ca.customer_id
LEFT JOIN (
    SELECT
        product_id,
        AVG(is_returned) AS product_avg_return_rate_all_time_at_cutoff
    FROM historical_data
    GROUP BY product_id
) pa ON t_oi.product_id = pa.product_id
LEFT JOIN (
    SELECT
        product_category,
        AVG(is_returned) AS category_avg_return_rate_all_time_at_cutoff
    FROM historical_data
    GROUP BY product_category
) pca ON t_p.product_category = pca.product_category
WHERE julianday(t_o.order_date) > {cutoff_julianday}
ORDER BY t_oi.order_item_id;
"""

# Execute the SQL query
sql_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"  - Engineered SQL features for {len(sql_features_df)} order items (those after cutoff).")
print("  - Sample SQL features:\n", sql_features_df.head())


# --- 3. Pandas Feature Engineering and Target Variable Creation ---
print("\n3. Performing Pandas Feature Engineering and Target Variable Creation...")

# Merge prediction set order_items with orders, customers, products for further pandas FE
# Need to use the original dataframes with datetime objects for date arithmetic
prediction_set_order_items = order_items_df.merge(orders_df, on='order_id', how='inner')
prediction_set_order_items = prediction_set_order_items.merge(products_df, on='product_id', how='inner')
prediction_set_order_items = prediction_set_order_items.merge(customers_df, on='customer_id', how='inner')

# Filter for order items after the cutoff date
prediction_set_order_items = prediction_set_order_items[
    prediction_set_order_items['order_date'] > GLOBAL_PREDICTION_CUTOFF_DATE
].copy()

# Merge with SQL features
prediction_set_order_items = prediction_set_order_items.merge(sql_features_df, on='order_item_id', how='left')

# Merge with returns_df to create the target variable
final_df = prediction_set_order_items.merge(returns_df[['order_item_id', 'return_date']], on='order_item_id', how='left')

# Target variable: will_be_returned_in_next_30_days
final_df['return_window_end'] = final_df['order_date'] + datetime.timedelta(days=30)
final_df['will_be_returned_in_next_30_days'] = (
    (final_df['return_date'].notna()) &
    (final_df['return_date'] <= final_df['return_window_end'])
).astype(int)

# Pandas-based Feature Engineering
final_df['days_since_customer_signup_at_order'] = (final_df['order_date'] - final_df['customer_signup_date']).dt.days
final_df['item_value_percentage_of_order'] = (final_df['item_price'] / final_df['order_total']).fillna(0)
final_df['item_value_percentage_of_order'] = final_df['item_value_percentage_of_order'].replace([np.inf, -np.inf], 0) # Handle potential inf if order_total was 0.01

# Time-based features from order_date
final_df['order_day_of_week'] = final_df['order_date'].dt.dayofweek # Monday=0, Sunday=6
final_df['order_month'] = final_df['order_date'].dt.month
final_df['order_is_weekend'] = (final_df['order_day_of_week'] >= 5).astype(int)

# Handle missing values for SQL-engineered features (should be 0.0 from COALESCE, but double-check)
for col in ['customer_avg_return_rate_prev_6m', 'product_avg_return_rate_all_time_at_cutoff', 'category_avg_return_rate_all_time_at_cutoff']:
    final_df[col] = final_df[col].fillna(0.0)

# Drop auxiliary columns used for target/feature creation, or which are identifiers
final_df.drop(columns=['return_date', 'return_window_end', 'customer_signup_date', 'order_date',
                       'product_name', 'gender', 'age', # Gender and age were generated but not explicitly requested as features
                       'shipping_cost', 'discount_applied', 'order_total'], # Keep order_total, it's used for item_value_percentage_of_order
              errors='ignore', inplace=True)

# Convert categorical features to 'category' dtype
categorical_cols = ['product_category', 'loyalty_status', 'order_day_of_week', 'order_month']
for col in categorical_cols:
    final_df[col] = final_df[col].astype('category')

print(f"  - Final dataset for ML has {len(final_df)} rows and {len(final_df.columns)} columns.")
print("  - Sample of final features and target:\n", final_df.head())
print(f"  - Target distribution:\n{final_df['will_be_returned_in_next_30_days'].value_counts(normalize=True)}")


# --- 4. Data Visualization ---
print("\n4. Performing Data Visualization (plots are generated but not displayed in console)...")

# Set plot style
sns.set_style("whitegrid")
plt.figure(figsize=(18, 12))

# Plot 1: Target distribution
plt.subplot(3, 3, 1)
sns.countplot(x='will_be_returned_in_next_30_days', data=final_df)
plt.title('Distribution of Target Variable')
plt.ylabel('Count')
plt.xlabel('Returned in Next 30 Days')

# Plot 2: Item Price vs. Target
plt.subplot(3, 3, 2)
sns.violinplot(x='will_be_returned_in_next_30_days', y='item_price', data=final_df)
plt.title('Item Price vs. Returns')
plt.ylabel('Item Price')
plt.xlabel('Returned in Next 30 Days')

# Plot 3: Customer Avg Return Rate vs. Target
plt.subplot(3, 3, 3)
sns.boxplot(x='will_be_returned_in_next_30_days', y='customer_avg_return_rate_prev_6m', data=final_df)
plt.title('Customer Avg Return Rate (Prev 6M) vs. Returns')
plt.ylabel('Customer Avg Return Rate')
plt.xlabel('Returned in Next 30 Days')

# Plot 4: Product Category vs. Target (stacked bar)
plt.subplot(3, 3, 4)
category_returns = final_df.groupby(['product_category', 'will_be_returned_in_next_30_days']).size().unstack(fill_value=0)
category_returns_norm = category_returns.div(category_returns.sum(axis=1), axis=0)
category_returns_norm.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Return Proportion by Product Category')
plt.ylabel('Proportion')
plt.xlabel('Product Category')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Returned in 30 Days', labels=['No', 'Yes'], loc='upper left')

# Plot 5: Loyalty Status vs. Target (stacked bar)
plt.subplot(3, 3, 5)
loyalty_returns = final_df.groupby(['loyalty_status', 'will_be_returned_in_next_30_days']).size().unstack(fill_value=0)
loyalty_returns_norm = loyalty_returns.div(loyalty_returns.sum(axis=1), axis=0)
loyalty_returns_norm.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Return Proportion by Loyalty Status')
plt.ylabel('Proportion')
plt.xlabel('Loyalty Status')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Returned in 30 Days', labels=['No', 'Yes'], loc='upper left')

# Plot 6: Days Since Signup vs. Target
plt.subplot(3, 3, 6)
sns.boxplot(x='will_be_returned_in_next_30_days', y='days_since_customer_signup_at_order', data=final_df)
plt.title('Days Since Customer Signup vs. Returns')
plt.ylabel('Days Since Signup')
plt.xlabel('Returned in Next 30 Days')

# Plot 7: Item Value Percentage of Order vs. Target
plt.subplot(3, 3, 7)
sns.boxplot(x='will_be_returned_in_next_30_days', y='item_value_percentage_of_order', data=final_df)
plt.title('Item Value % of Order vs. Returns')
plt.ylabel('Item Value % of Order')
plt.xlabel('Returned in Next 30 Days')

plt.tight_layout()
# In a self-contained script meant to run without a display, plt.show() will cause errors.
# We'll generate the plots but not display them. If running locally with a display, uncomment plt.show().
# plt.show()
plt.close('all') # Close figures to free memory and avoid display issues.


# --- 5. Scikit-learn ML Pipeline Construction and Training ---
print("\n5. Building and Training Scikit-learn ML Pipeline...")

# Define features (X) and target (y)
X = final_df.drop(columns=['order_item_id', 'customer_id', 'order_id', 'product_id', 'will_be_returned_in_next_30_days'])
y = final_df['will_be_returned_in_next_30_days']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='category').columns.tolist()

print(f"  - Numerical features: {numerical_features}")
print(f"  - Categorical features: {categorical_features}")

# Split data into training and testing sets, stratified by target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  - Training data size: {len(X_train)} samples")
print(f"  - Test data size: {len(X_test)} samples")
print(f"  - Training target (1/0) distribution:\n{y_train.value_counts(normalize=True)}")
print(f"  - Test target (1/0) distribution:\n{y_test.value_counts(normalize=True)}")


# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, or 'drop'
)

# Create the full ML pipeline
# Using class_weight='balanced' to address class imbalance for HistGradientBoostingClassifier.
# Also increasing max_iter (number of boosting stages) and learning_rate to give the model more learning capacity.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42, class_weight='balanced', max_iter=250, learning_rate=0.15)) # Increased max_iter and learning_rate
])

# Train the model
print("  - Training model...")
model_pipeline.fit(X_train, y_train)
print("  - Model training complete.")

# --- 6. Model Evaluation and Interpretation ---
print("\n6. Evaluating Model Performance...")

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Calculate ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"  - ROC AUC Score on Test Set: {roc_auc:.4f}")

# Predict classes based on a threshold (default 0.5)
y_pred = model_pipeline.predict(X_test)

# Generate and print classification report
print("\n  - Classification Report on Test Set:")
print(classification_report(y_test, y_pred))

# Plot ROC curve
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc_val = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
# As with other plots, plt.show() is commented out for script execution without display.
# plt.show()
plt.close('all') # Close figure

print("\n7. Pipeline Execution Complete.")
print("\nSummary of results:")
print(f"  ROC AUC Score: {roc_auc:.4f}")
print("  Check the Classification Report above for detailed precision, recall, and F1-score for both classes.")
print("  The synthetic data generation was enhanced to create stronger, more learnable signals for returns based on product category ('Apparel') and customer tenure (new customers).")
print("  The model uses HistGradientBoostingClassifier with class_weight='balanced', increased `max_iter`, and `learning_rate` to improve its ability to learn from imbalanced data, combined with a robust ColumnTransformer for preprocessing.")