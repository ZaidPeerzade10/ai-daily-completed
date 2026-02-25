import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---
np.random.seed(42)

# 1.1 products_df
n_products = np.random.randint(100, 201)
product_ids = np.arange(1, n_products + 1)
categories = ['Electronics', 'Apparel', 'Books', 'Home Goods', 'Food']
brands = ['BrandA', 'BrandB', 'BrandC', 'Generic']

products_data = {
    'product_id': product_ids,
    'category': np.random.choice(categories, n_products),
    'brand': np.random.choice(brands, n_products, p=[0.25, 0.25, 0.25, 0.25]),
    'base_price': np.round(np.random.uniform(20.0, 500.0, n_products), 2)
}
products_df = pd.DataFrame(products_data)

# Generate launch dates for the last 3 years
today = pd.Timestamp.now()
products_df['launch_date'] = today - pd.to_timedelta(np.random.randint(0, 3 * 365, n_products), unit='D')
# Ensure launch_date is always before or equal to today, and normalize to date only
products_df['launch_date'] = products_df['launch_date'].dt.normalize()

print("--- Generated products_df head ---")
print(products_df.head())
print(f"Products DataFrame shape: {products_df.shape}\n")

# 1.2 sales_df
n_sales = np.random.randint(4000, 6001)

# Base sales data
sales_data = {
    'sale_id': np.arange(1, n_sales + 1),
    'product_id': np.random.choice(products_df['product_id'], n_sales, 
                                   p=products_df['base_price'].rank(ascending=False).values / 
                                     products_df['base_price'].rank(ascending=False).sum())
}
sales_df = pd.DataFrame(sales_data)

# Merge with product launch dates and categorical features for sales pattern simulation
sales_df = sales_df.merge(products_df[['product_id', 'launch_date', 'category', 'brand']], on='product_id', how='left')

# Generate sale dates after launch_date but before today
# Sales can occur between 1 day and 2 years after launch, capped by today
sales_df['days_offset'] = np.random.randint(1, 365 * 2, len(sales_df)) 
sales_df['sale_date'] = sales_df['launch_date'] + pd.to_timedelta(sales_df['days_offset'], unit='D')
sales_df['sale_date'] = sales_df.apply(lambda row: min(row['sale_date'], today), axis=1) # Cap at today
sales_df['sale_date'] = sales_df['sale_date'].dt.normalize()

# Filter out sales that might have ended up before launch date due to capping or edge cases
sales_df = sales_df[sales_df['sale_date'] >= sales_df['launch_date']].copy()
# Re-assign unique sale_ids after filtering and reset index
sales_df.reset_index(drop=True, inplace=True)
sales_df['sale_id'] = np.arange(1, len(sales_df) + 1)

# Simulate realistic sales patterns for quantity_sold and discount_applied_percent
sales_df['quantity_sold'] = np.random.randint(1, 6, len(sales_df)) # Base quantity

# Bias quantity_sold by category/brand
category_qty_bias = {'Electronics': 1.8, 'Apparel': 1.2, 'Books': 0.7, 'Home Goods': 1.0, 'Food': 1.5}
brand_qty_bias = {'BrandA': 1.5, 'BrandB': 1.0, 'BrandC': 0.8, 'Generic': 0.5}

sales_df['quantity_sold'] = sales_df.apply(
    lambda row: round(row['quantity_sold'] * category_qty_bias.get(row['category'], 1) * brand_qty_bias.get(row['brand'], 1)), axis=1
)
sales_df['quantity_sold'] = sales_df['quantity_sold'].astype(int).clip(1, 10) # Clip to 1-10

# Bias discount_applied_percent (mostly 0, occasional non-zero)
sales_df['discount_applied_percent'] = np.random.choice(
    [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    size=len(sales_df),
    p=[0.7, 0.1, 0.08, 0.05, 0.03, 0.02, 0.02]
)

# Discounts should generally lead to higher quantity_sold
discount_effect_multiplier = sales_df['discount_applied_percent'].apply(lambda x: 1 + x / 100 * 0.5) # Max 1.15x for 30% discount
sales_df['quantity_sold'] = (sales_df['quantity_sold'] * discount_effect_multiplier).round().astype(int)
sales_df['quantity_sold'] = sales_df['quantity_sold'].clip(1, 10) # Re-clip after discount effect

# Drop temporary columns used for generation
sales_df.drop(columns=['launch_date', 'category', 'brand', 'days_offset'], inplace=True)

print("--- Generated sales_df head ---")
print(sales_df.head())
print(f"Sales DataFrame shape: {sales_df.shape}\n")


# --- 2. Load into SQLite & SQL Feature Engineering (Early Product Performance) ---
conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite
products_df.to_sql('products', conn, index=False, if_exists='replace')
sales_df.to_sql('sales', conn, index=False, if_exists='replace')

# Determine global_analysis_date and feature_cutoff_date
global_analysis_date_pd = sales_df['sale_date'].max() + pd.to_timedelta(60, unit='D')
feature_cutoff_date_pd = global_analysis_date_pd - pd.to_timedelta(120, unit='D')

global_analysis_date_str = global_analysis_date_pd.strftime('%Y-%m-%d')
feature_cutoff_date_str = feature_cutoff_date_pd.strftime('%Y-%m-%d')

print(f"Global Analysis Date: {global_analysis_date_str}")
print(f"Feature Cutoff Date: {feature_cutoff_date_str}\n")

# SQL Query for pre-cutoff features
sql_query = f"""
SELECT
    p.product_id,
    p.category,
    p.brand,
    p.base_price,
    p.launch_date,
    COALESCE(SUM(s.quantity_sold), 0) AS total_quantity_sold_pre_cutoff,
    COALESCE(COUNT(s.sale_id), 0) AS num_sales_events_pre_cutoff,
    COALESCE(AVG(s.discount_applied_percent), 0.0) AS avg_discount_pre_cutoff,
    COALESCE(COUNT(DISTINCT s.sale_date), 0) AS num_unique_sale_days_pre_cutoff,
    CASE
        WHEN MIN(s.sale_date) IS NOT NULL THEN CAST(JULIANDAY('{feature_cutoff_date_str}') - JULIANDAY(MIN(s.sale_date)) AS INTEGER)
        ELSE NULL
    END AS days_since_first_sale_pre_cutoff
FROM
    products p
LEFT JOIN
    sales s ON p.product_id = s.product_id AND s.sale_date < '{feature_cutoff_date_str}'
GROUP BY
    p.product_id, p.category, p.brand, p.base_price, p.launch_date
ORDER BY
    p.product_id;
"""

product_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("--- product_features_df (from SQL) head ---")
print(product_features_df.head())
print(f"Product Features DataFrame shape: {product_features_df.shape}\n")

# --- 3. Pandas Feature Engineering & Multi-Class Target Creation (Future Sales Performance) ---

# 3.1 Handle NaN values in product_features_df
# Aggregated numerical features will be 0 due to COALESCE in SQL, but for safety:
product_features_df['total_quantity_sold_pre_cutoff'].fillna(0, inplace=True)
product_features_df['num_sales_events_pre_cutoff'].fillna(0, inplace=True)
product_features_df['num_unique_sale_days_pre_cutoff'].fillna(0, inplace=True)
product_features_df['avg_discount_pre_cutoff'].fillna(0.0, inplace=True)

# 3.2 Convert launch_date and calculate product_age_at_cutoff_days
product_features_df['launch_date'] = pd.to_datetime(product_features_df['launch_date'])
product_features_df['product_age_at_cutoff_days'] = (feature_cutoff_date_pd - product_features_df['launch_date']).dt.days
product_features_df['product_age_at_cutoff_days'] = product_features_df['product_age_at_cutoff_days'].clip(lower=0) # Age cannot be negative

# Fill days_since_first_sale_pre_cutoff with a sentinel value
# If no sales pre-cutoff, fill with a large value relative to product age
sentinel_value = product_features_df['product_age_at_cutoff_days'] + 30
# For products launched after cutoff date, this value doesn't make sense, set to 0 then.
product_features_df['days_since_first_sale_pre_cutoff'] = product_features_df.apply(
    lambda row: row['days_since_first_sale_pre_cutoff'] 
                if pd.notna(row['days_since_first_sale_pre_cutoff'])
                else (row['product_age_at_cutoff_days'] + 30 if row['product_age_at_cutoff_days'] > 0 else 0),
    axis=1
)
product_features_df['days_since_first_sale_pre_cutoff'] = product_features_df['days_since_first_sale_pre_cutoff'].astype(int)

# 3.3 Calculate sales_frequency_pre_cutoff
product_features_df['sales_frequency_pre_cutoff'] = product_features_df['num_sales_events_pre_cutoff'] / (product_features_df['product_age_at_cutoff_days'] + 1)
product_features_df['sales_frequency_pre_cutoff'].fillna(0, inplace=True) # Handle division by zero for age 0

# 3.4 Create the Multi-Class Target `future_sales_tier`
# Calculate total_quantity_sold_future
future_sales_df = sales_df[
    (sales_df['sale_date'] >= feature_cutoff_date_pd) &
    (sales_df['sale_date'] <= global_analysis_date_pd)
].copy()

total_quantity_sold_future = future_sales_df.groupby('product_id')['quantity_sold'].sum().reset_index()
total_quantity_sold_future.rename(columns={'quantity_sold': 'total_quantity_sold_future'}, inplace=True)

product_features_df = product_features_df.merge(total_quantity_sold_future, on='product_id', how='left')
product_features_df['total_quantity_sold_future'].fillna(0, inplace=True)

# Define sales tiers based on percentiles of non-zero future sales
non_zero_future_sales = product_features_df[product_features_df['total_quantity_sold_future'] > 0]['total_quantity_sold_future']

if len(non_zero_future_sales) >= 3: # Ensure enough data points for percentiles
    p33 = non_zero_future_sales.quantile(0.33)
    p66 = non_zero_future_sales.quantile(0.66)
else: # Fallback for very sparse future sales data, prevent errors with small datasets
    if len(non_zero_future_sales) > 0:
        p33 = non_zero_future_sales.min()
        p66 = non_zero_future_sales.max()
        if p33 == p66: # If all non-zero sales are the same value
            p33 = p33 * 0.5 if p33 > 0 else 1 # Ensure p33 is smaller for tiering
            p66 = p66 * 1.5 if p66 > 0 else 2 # Ensure p66 is larger
    else: # No non-zero sales
        p33 = 1
        p66 = 2

def assign_sales_tier(sales_qty, p33, p66):
    if sales_qty == 0:
        return 'No_Sales'
    elif sales_qty <= p33:
        return 'Low_Sales'
    elif sales_qty <= p66:
        return 'Medium_Sales'
    else:
        return 'High_Sales'

product_features_df['future_sales_tier'] = product_features_df['total_quantity_sold_future'].apply(
    lambda x: assign_sales_tier(x, p33, p66)
)

print("\n--- Product Features with Target (future_sales_tier) head ---")
print(product_features_df.head())
print(f"Product Features DataFrame shape: {product_features_df.shape}\n")
print("Future Sales Tier distribution:")
print(product_features_df['future_sales_tier'].value_counts())
print(f"Percentiles for tiering: 33rd={p33:.2f}, 66th={p66:.2f}\n")

# Define features `X` and target `y`
numerical_features = [
    'base_price', 'product_age_at_cutoff_days', 'total_quantity_sold_pre_cutoff',
    'num_sales_events_pre_cutoff', 'avg_discount_pre_cutoff',
    'num_unique_sale_days_pre_cutoff', 'days_since_first_sale_pre_cutoff',
    'sales_frequency_pre_cutoff'
]
categorical_features = ['category', 'brand']

X = product_features_df[numerical_features + categorical_features]
y = product_features_df['future_sales_tier']

# Split into training and testing sets
# Handle cases where stratification might not be possible for very small classes
if len(y.unique()) > 1 and all(y.value_counts() > 1): # Ensure at least 2 samples per class for stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
else:
    print("Warning: Cannot use stratify due to small number of samples per class.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}\n")

# --- 4. Data Visualization ---

plt.figure(figsize=(14, 6))

# Violin plot for sales_frequency_pre_cutoff vs future_sales_tier
plt.subplot(1, 2, 1)
# Ensure consistent order for tiers in plots
tier_order = ['No_Sales', 'Low_Sales', 'Medium_Sales', 'High_Sales']
sns.violinplot(x='future_sales_tier', y='sales_frequency_pre_cutoff', data=product_features_df, order=tier_order)
plt.title('Sales Frequency Pre-Cutoff by Future Sales Tier')
plt.xlabel('Future Sales Tier')
plt.ylabel('Sales Frequency (Pre-Cutoff)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Stacked bar chart for future_sales_tier across different brands
plt.subplot(1, 2, 2)
brand_tier_distribution = pd.crosstab(product_features_df['brand'], product_features_df['future_sales_tier'], normalize='index')
# Reorder columns to match tier_order
brand_tier_distribution = brand_tier_distribution.reindex(columns=tier_order, fill_value=0)
brand_tier_distribution.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Proportion of Future Sales Tiers by Brand')
plt.xlabel('Brand')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Future Sales Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("--- Visualization plots displayed ---\n")

# --- 5. ML Pipeline & Evaluation (Multi-Class) ---

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, not applicable here.
)

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'))
])

# Train the model
print("--- Training RandomForestClassifier pipeline ---")
model_pipeline.fit(X_train, y_train)
print("--- Training complete ---\n")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=0) # zero_division=0 to handle classes with no predicted samples gracefully

print("--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy:.4f}\n")
print("Classification Report:\n", class_report)
print("--- Script Finished ---")