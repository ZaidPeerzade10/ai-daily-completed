import pandas as pd
import numpy as np
import sqlite3
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Suppress warnings for cleaner output in production-like scripts
import warnings
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

print("1. Generating Synthetic Data...")

# Define number of products
num_products = np.random.randint(500, 801)
product_ids = np.arange(1, num_products + 1)
categories = ['Electronics', 'Apparel', 'Books', 'Food']

# products_df
products_df = pd.DataFrame({
    'product_id': product_ids,
    'category': np.random.choice(categories, num_products),
    'base_price': np.round(np.random.uniform(10.0, 500.0, num_products), 2),
    'reorder_lead_time_days': np.random.randint(3, 15, num_products)
})

# daily_sales_df
num_sales_records = np.random.randint(10000, 15001)
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=180) # Last 6 months
date_range = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]

# Create a mapping for category sales bias
category_sales_bias = {
    'Electronics': 1.5,
    'Apparel': 1.2,
    'Books': 0.8,
    'Food': 2.0
}

# Generate initial daily sales
daily_sales_data = []
for _ in range(num_sales_records):
    pid = np.random.choice(product_ids)
    s_date = np.random.choice(date_range)
    # Get category for sales bias
    category = products_df[products_df['product_id'] == pid]['category'].iloc[0]
    
    base_qty = np.random.randint(1, 50)
    
    # Apply category bias
    qty_sold = base_qty * category_sales_bias[category]
    
    # Apply seasonality for Apparel in Nov/Dec (assuming current date allows Nov/Dec in last 6 months)
    if category == 'Apparel' and s_date.month in [11, 12]:
        qty_sold *= 1.5 # 50% increase
        
    daily_sales_data.append({
        'product_id': pid,
        'sale_date': s_date,
        'quantity_sold': int(qty_sold)
    })

daily_sales_df = pd.DataFrame(daily_sales_data)
daily_sales_df['sale_date'] = pd.to_datetime(daily_sales_df['sale_date'])
daily_sales_df = daily_sales_df.sort_values(by=['product_id', 'sale_date']).reset_index(drop=True)

# current_inventory_df
current_inventory_df = pd.DataFrame({
    'product_id': product_ids,
    'stock_on_hand': np.random.randint(100, 2001, num_products)
})

# Simulate realistic stock-out patterns: bias stock_on_hand for some products
# Calculate average daily sales for each product over the entire period to inform stock biasing
product_avg_overall_daily_sales = daily_sales_df.groupby('product_id')['quantity_sold'].mean().reset_index()
product_avg_overall_daily_sales.rename(columns={'quantity_sold': 'avg_overall_daily_sales'}, inplace=True)

# Merge to get average sales into inventory df
current_inventory_df = current_inventory_df.merge(product_avg_overall_daily_sales, on='product_id', how='left')
current_inventory_df['avg_overall_daily_sales'] = current_inventory_df['avg_overall_daily_sales'].fillna(0) # For products with no sales

# Identify 10-15% of products to bias for low stock
num_biased_products = int(num_products * np.random.uniform(0.10, 0.15))
biased_product_ids = np.random.choice(product_ids, num_biased_products, replace=False)

# Adjust stock_on_hand for biased products
for pid in biased_product_ids:
    avg_sales_for_product = current_inventory_df[current_inventory_df['product_id'] == pid]['avg_overall_daily_sales'].iloc[0]
    if avg_sales_for_product > 0:
        # Set stock_on_hand to cover only a few days of their average sales, or slightly less than 30 days
        # This makes them likely to stock out within 30 days if sales continue at average rate
        new_stock = int(avg_sales_for_product * np.random.uniform(10, 25)) # Covers 10-25 days of avg sales
        current_inventory_df.loc[current_inventory_df['product_id'] == pid, 'stock_on_hand'] = max(1, new_stock) # Ensure stock is at least 1
    else:
        # If no sales, still give it some low stock to be considered, but it won't stock out due to sales
        current_inventory_df.loc[current_inventory_df['product_id'] == pid, 'stock_on_hand'] = np.random.randint(10, 50)

# Drop the temporary avg_overall_daily_sales column
current_inventory_df.drop(columns=['avg_overall_daily_sales'], inplace=True)

print(f"Generated {len(products_df)} products, {len(daily_sales_df)} daily sales records, {len(current_inventory_df)} inventory records.")
print("Products head:\n", products_df.head())
print("Daily Sales head:\n", daily_sales_df.head())
print("Current Inventory head:\n", current_inventory_df.head())

# --- 2. Load into SQLite & SQL Feature Engineering ---

print("\n2. Loading data into SQLite and performing SQL Feature Engineering...")

conn = sqlite3.connect(':memory:')

products_df.to_sql('products', conn, index=False, if_exists='replace')
daily_sales_df.to_sql('daily_sales', conn, index=False, if_exists='replace', dtype={'sale_date': 'TEXT'}) # Store date as TEXT for julianday

# Ensure inventory is loaded correctly (no 'if_exists' for new table)
current_inventory_df.to_sql('inventory', conn, index=False, if_exists='replace')

# Determine analysis_date
analysis_date_str = pd.read_sql_query("SELECT MAX(sale_date) FROM daily_sales", conn).iloc[0,0]
analysis_date = pd.to_datetime(analysis_date_str).date()
print(f"Analysis Date (today for prediction): {analysis_date}")

sql_query = f"""
WITH ProductSalesFiltered AS (
    SELECT
        product_id,
        sale_date,
        quantity_sold
    FROM
        daily_sales
    WHERE
        sale_date <= '{analysis_date_str}'
),
SalesLast7d AS (
    SELECT
        product_id,
        CAST(SUM(quantity_sold) AS REAL) / COUNT(DISTINCT sale_date) AS avg_sales_last_7d
    FROM
        ProductSalesFiltered
    WHERE
        julianday(sale_date) >= julianday('{analysis_date_str}') - 6 -- 7 days inclusive
    GROUP BY
        product_id
),
SalesLast30d AS (
    SELECT
        product_id,
        SUM(quantity_sold) AS total_sales_last_30d,
        COUNT(DISTINCT sale_date) AS num_selling_days_last_30d
    FROM
        ProductSalesFiltered
    WHERE
        julianday(sale_date) >= julianday('{analysis_date_str}') - 29 -- 30 days inclusive
    GROUP BY
        product_id
)
SELECT
    p.product_id,
    p.category,
    p.base_price,
    p.reorder_lead_time_days,
    i.stock_on_hand,
    COALESCE(s7.avg_sales_last_7d, 0.0) AS avg_sales_last_7d,
    COALESCE(s30.total_sales_last_30d, 0) AS total_sales_last_30d,
    COALESCE(s30.num_selling_days_last_30d, 0) AS num_selling_days_last_30d
FROM
    products p
LEFT JOIN
    inventory i ON p.product_id = i.product_id
LEFT JOIN
    SalesLast7d s7 ON p.product_id = s7.product_id
LEFT JOIN
    SalesLast30d s30 ON p.product_id = s30.product_id;
"""

product_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("SQL Feature Engineering complete. Resulting DataFrame head:\n", product_features_df.head())
print(f"DataFrame shape: {product_features_df.shape}")

# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n3. Performing Pandas Feature Engineering and Binary Target Creation...")

# Ensure numerical aggregated features are handled (COALESCE should do this, but defensive coding)
numerical_agg_cols = ['avg_sales_last_7d', 'total_sales_last_30d', 'num_selling_days_last_30d']
for col in numerical_agg_cols:
    if product_features_df[col].isnull().any():
        product_features_df[col] = product_features_df[col].fillna(0) # Fill potential remaining NaNs with 0

# Calculate sales_velocity_30d
epsilon = 1e-6 # Small constant to prevent division by zero
product_features_df['sales_velocity_30d'] = product_features_df['total_sales_last_30d'] / (product_features_df['num_selling_days_last_30d'] + epsilon)
product_features_df['sales_velocity_30d'].replace([np.inf, -np.inf], 0, inplace=True)
product_features_df['sales_velocity_30d'].fillna(0, inplace=True)

# Calculate stock_to_avg_daily_sales_7d_ratio
large_ratio_fill = 9999 # A large number to indicate very high stock relative to sales (or no sales)
product_features_df['stock_to_avg_daily_sales_7d_ratio'] = product_features_df['stock_on_hand'] / (product_features_df['avg_sales_last_7d'] * 30 + epsilon)
product_features_df['stock_to_avg_daily_sales_7d_ratio'].replace([np.inf, -np.inf], large_ratio_fill, inplace=True)
product_features_df['stock_to_avg_daily_sales_7d_ratio'].fillna(large_ratio_fill, inplace=True)


# Create the Binary Target `will_stock_out_in_next_30_days`
# A product is considered to stock out if its stock_on_hand is less than or equal to (avg_sales_last_7d * 30)
# If avg_sales_last_7d is 0, the product will not stock out due to sales, so set its target to 0.
product_features_df['will_stock_out_in_next_30_days'] = np.where(
    (product_features_df['avg_sales_last_7d'] > 0) & 
    (product_features_df['stock_on_hand'] <= (product_features_df['avg_sales_last_7d'] * 30)), 
    1, 
    0
)

print("Pandas Feature Engineering and Target Creation complete. DataFrame head with new features:\n", product_features_df.head())
print(f"Number of stock-out predictions (1): {product_features_df['will_stock_out_in_next_30_days'].sum()}")
print(f"Number of non-stock-out predictions (0): {len(product_features_df) - product_features_df['will_stock_out_in_next_30_days'].sum()}")


# Define features X and target y
numerical_features = [
    'base_price', 'reorder_lead_time_days', 'stock_on_hand',
    'avg_sales_last_7d', 'total_sales_last_30d', 'num_selling_days_last_30d',
    'sales_velocity_30d', 'stock_to_avg_daily_sales_7d_ratio'
]
categorical_features = ['category']

X = product_features_df[numerical_features + categorical_features]
y = product_features_df['will_stock_out_in_next_30_days']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train/Test split: {len(X_train)} training samples, {len(X_test)} testing samples.")
print(f"Target distribution in training set:\n{y_train.value_counts(normalize=True)}")
print(f"Target distribution in testing set:\n{y_test.value_counts(normalize=True)}")


# --- 4. Data Visualization ---

print("\n4. Generating Data Visualizations (plots will be displayed)...")

plt.figure(figsize=(14, 6))

# Violin plot for stock_to_avg_daily_sales_7d_ratio
plt.subplot(1, 2, 1)
sns.violinplot(
    x='will_stock_out_in_next_30_days',
    y='stock_to_avg_daily_sales_7d_ratio',
    data=product_features_df,
    palette='viridis',
    inner='quartile'
)
plt.title('Stock-to-Avg-Daily-Sales Ratio vs. Stock-Out')
plt.xlabel('Will Stock Out in Next 30 Days (0=No, 1=Yes)')
plt.ylabel('Stock-to-Avg-Daily-Sales (7d) Ratio')
# Limit y-axis for better visualization if there are extreme outliers
plt.ylim(0, product_features_df['stock_to_avg_daily_sales_7d_ratio'].quantile(0.98) * 1.2) # Show up to 98th percentile for clarity


# Stacked bar chart for category vs. stock-out proportion
plt.subplot(1, 2, 2)
category_stock_out_proportion = product_features_df.groupby('category')['will_stock_out_in_next_30_days'].value_counts(normalize=True).unstack().fillna(0)
category_stock_out_proportion.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Proportion of Stock-Out by Category')
plt.xlabel('Category')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Will Stock Out', labels=['No', 'Yes'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("Visualizations displayed.")

# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

print("\n5. Building ML Pipeline and Evaluating...")

# Create preprocessing pipelines for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Handles potential NaNs, though StandardScaler also implicitly does with mean
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop any columns not specified (like product_id if it were in X)
)

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)
print("ML Pipeline trained successfully.")

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Convert probabilities to binary predictions using a 0.5 threshold for classification report
y_pred = (y_pred_proba >= 0.5).astype(int)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
classification_rep = classification_report(y_test, y_pred)

print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:")
print(classification_rep)

print("\nML Pipeline execution complete.")