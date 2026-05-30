import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

# Suppress matplotlib warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Synthetic Data Generation (Pandas/Numpy) ---

print("--- Generating Synthetic Data ---")

# Products DataFrame
n_products = np.random.randint(1000, 1501)
product_ids = np.arange(1, n_products + 1)
categories = ['Electronics', 'Apparel', 'Home Goods', 'Books', 'Groceries', 'Beauty']
brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'BrandF']
# Products launched over the last 3-5 years
start_date_products_generation = pd.Timestamp.now() - pd.DateOffset(years=5)
end_date_products_generation = pd.Timestamp.now() - pd.DateOffset(years=3)

products_data = {
    'product_id': product_ids,
    'category': np.random.choice(categories, n_products),
    'brand': np.random.choice(brands, n_products),
    'base_price': np.round(np.random.uniform(10, 1000, n_products), 2),
    'launch_date': start_date_products_generation + pd.to_timedelta(
        np.random.randint(0, (end_date_products_generation - start_date_products_generation).days, n_products), unit='D'
    )
}
products_df = pd.DataFrame(products_data)

# Simulate higher prices/sales for certain categories/brands
category_price_multipliers = {'Electronics': 1.5, 'Apparel': 1.2, 'Home Goods': 1.0, 'Books': 0.8, 'Groceries': 0.7, 'Beauty': 1.1}
brand_price_multipliers = {'BrandA': 1.3, 'BrandB': 1.1, 'BrandC': 1.0, 'BrandD': 0.9, 'BrandE': 0.8, 'BrandF': 0.7}
products_df['base_price'] = products_df.apply(
    lambda row: row['base_price'] * category_price_multipliers[row['category']] * brand_price_multipliers[row['brand']],
    axis=1
)
products_df['base_price'] = np.round(products_df['base_price'], 2)


# Sales DataFrame
n_sales = np.random.randint(20000, 30001)
max_sale_date_overall = pd.Timestamp.now() - pd.Timedelta(weeks=2)

# To simulate more sales for certain products, we sample product_ids with replacement
# based on a 'popularity' score. Older products are generally less active *recently*,
# but here we're creating a history so all products should have sales.
# A simple way to get more sales for some products is to use a non-uniform distribution.
# Let's make newer products slightly more likely to have sales events
product_age_at_max_sale = (max_sale_date_overall - products_df['launch_date']).dt.days.clip(lower=1)
# Using inverse square root to give a moderate preference to newer products, but still allow old products to have sales
product_activity_weights = 1 / np.sqrt(product_age_at_max_sale)
product_activity_weights_norm = product_activity_weights / product_activity_weights.sum()

sampled_product_ids = np.random.choice(products_df['product_id'], size=n_sales, p=product_activity_weights_norm)
sampled_products_df_lookup = products_df.set_index('product_id').loc[sampled_product_ids].reset_index()

# Generate sale_date: biased towards launch date for a product, but within the overall window
time_deltas = (max_sale_date_overall - sampled_products_df_lookup['launch_date']).dt.days.values
# Generate a beta distribution for each sale: skewed towards earlier dates in product's life
# (a=0.5, b=5 makes it heavily skewed towards 0, i.e., closer to launch_date)
beta_samples = np.random.beta(a=0.5, b=5, size=n_sales)
generated_sale_dates = sampled_products_df_lookup['launch_date'] + pd.to_timedelta(beta_samples * time_deltas, unit='D')
generated_sale_dates = generated_sale_dates.apply(lambda x: min(x, max_sale_date_overall)) # Ensure not past max_sale_date_overall

# Base quantity sold
base_qty = np.random.randint(1, 15, size=n_sales)

# Simulate varying sales patterns: category/brand boosts
category_qty_multipliers = {'Electronics': 1.8, 'Apparel': 1.2, 'Home Goods': 1.0, 'Books': 0.6, 'Groceries': 0.5, 'Beauty': 1.0}
brand_qty_multipliers = {'BrandA': 1.5, 'BrandB': 1.2, 'BrandC': 1.0, 'BrandD': 0.8, 'BrandE': 0.7, 'BrandF': 0.6}

categories_for_sales = sampled_products_df_lookup['category'].values
brands_for_sales = sampled_products_df_lookup['brand'].values

qty_multipliers = np.array([category_qty_multipliers[cat] * brand_qty_multipliers[brand] for cat, brand in zip(categories_for_sales, brands_for_sales)])
quantities_sold = (base_qty * qty_multipliers).astype(int).clip(1, 25) # Clip to reasonable range

# Simulate mild seasonality (e.g., boost sales around holidays in Nov/Dec)
# A simple sine wave can model this. Peak sales around day 330 of the year (late Nov)
days_in_year = generated_sale_dates.dt.dayofyear
seasonality_factor = (np.sin((days_in_year - 330) / 365 * 2 * np.pi) + 1.5) / 2 + 0.5 # Factor between ~0.5 and ~1.5
quantities_sold = (quantities_sold * seasonality_factor).astype(int).clip(1, 25)

# Simulate sales gradually decreasing for older products within their lifecycle:
# Penalize quantity sold for sales events that occur long after launch
sale_product_age_at_sale = (generated_sale_dates - sampled_products_df_lookup['launch_date']).dt.days
decay_factor_per_day = 0.002 # 0.2% reduction per day of age
quantity_decay_adjust = (1 - sale_product_age_at_sale * decay_factor_per_day).clip(0.3, 1.0) # Min 30% of original qty
quantities_sold = (quantities_sold * quantity_decay_adjust).astype(int).clip(1, 25)


sales_df = pd.DataFrame({
    'sale_id': np.arange(1, n_sales + 1),
    'product_id': sampled_product_ids,
    'sale_date': generated_sale_dates,
    'quantity_sold': quantities_sold
})

# Sort sales_df as required
sales_df = sales_df.sort_values(by=['product_id', 'sale_date']).reset_index(drop=True)

print(f"Generated {products_df.shape[0]} products.")
print(f"Generated {sales_df.shape[0]} sales records (target: 20k-30k).")
print(f"Latest sale_date in dataset: {sales_df['sale_date'].max()}")

# --- 2. Load into SQLite & SQL Feature Engineering (Time-Windowed Sales Aggregations) ---

print("\n--- Performing SQL-based Feature Engineering ---")

conn = sqlite3.connect(':memory:')
products_df.to_sql('products', conn, index=False)
sales_df.to_sql('sales', conn, index=False)

# Define GLOBAL_PREDICTION_CUTOFF_DATE
# FIX from previous attempt: Use pd.DateOffset for months
GLOBAL_PREDICTION_CUTOFF_DATE = sales_df['sale_date'].max() - pd.DateOffset(months=2)
GLOBAL_PREDICTION_CUTOFF_DATE_STR = GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d %H:%M:%S')

# SQL query for feature engineering
sql_query = f"""
WITH ProductSalesPrev30D AS (
    SELECT
        s.product_id,
        AVG(s.quantity_sold) AS avg_qty_sold_prev_30d,
        SUM(s.quantity_sold) AS total_qty_sold_prev_30d,
        COUNT(s.sale_id) AS num_sales_prev_30d
    FROM sales s
    WHERE s.sale_date > datetime('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}', '-30 days')
      AND s.sale_date <= datetime('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}')
    GROUP BY s.product_id
),
LastSaleBeforeCutoff AS (
    SELECT
        s.product_id,
        MAX(s.sale_date) AS last_sale_date_at_cutoff
    FROM sales s
    WHERE s.sale_date <= datetime('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}')
    GROUP BY s.product_id
)
SELECT
    p.product_id,
    p.category,
    p.brand,
    p.base_price,
    p.launch_date,
    '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' AS current_cutoff_date,
    COALESCE(ps30.avg_qty_sold_prev_30d, 0.0) AS avg_qty_sold_prev_30d,
    COALESCE(ps30.total_qty_sold_prev_30d, 0) AS total_qty_sold_prev_30d,
    COALESCE(ps30.num_sales_prev_30d, 0) AS num_sales_prev_30d,
    COALESCE(
        CAST(JULIANDAY('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') - JULIANDAY(lsbc.last_sale_date_at_cutoff) AS REAL),
        9999.0
    ) AS days_since_last_sale_at_cutoff
FROM products p
LEFT JOIN ProductSalesPrev30D ps30 ON p.product_id = ps30.product_id
LEFT JOIN LastSaleBeforeCutoff lsbc ON p.product_id = lsbc.product_id
ORDER BY p.product_id;
"""

product_features_df = pd.read_sql_query(sql_query, conn)
conn.close() # Close SQLite connection

print(f"Features calculated up to cutoff date: {GLOBAL_PREDICTION_CUTOFF_DATE}")
print(f"Product features DataFrame head:\n{product_features_df.head()}")


# --- 3. Pandas Feature Engineering & Multi-class Target Creation ---

print("\n--- Performing Pandas-based Feature Engineering & Target Creation ---")

# Convert dates to datetime objects
product_features_df['launch_date'] = pd.to_datetime(product_features_df['launch_date'])
product_features_df['current_cutoff_date'] = pd.to_datetime(product_features_df['current_cutoff_date'])

# Handle NaNs: base_price could theoretically have NaNs if not handled during generation or if product_df was empty
if product_features_df['base_price'].isnull().any():
    median_base_price = product_features_df['base_price'].median()
    product_features_df['base_price'].fillna(median_base_price, inplace=True)

# Calculate product_age_at_cutoff_days
product_features_df['product_age_at_cutoff_days'] = (product_features_df['current_cutoff_date'] - product_features_df['launch_date']).dt.days
# Ensure age is not negative (e.g., if launch_date is after cutoff due to synthetic data generation nuances)
product_features_df['product_age_at_cutoff_days'] = product_features_df['product_age_at_cutoff_days'].clip(lower=0)


# Create the Multi-class Target `next_14d_demand_category`
target_start_date = GLOBAL_PREDICTION_CUTOFF_DATE + pd.Timedelta(days=1)
target_end_date = GLOBAL_PREDICTION_CUTOFF_DATE + pd.Timedelta(days=14)

sales_in_target_window = sales_df[
    (sales_df['sale_date'] >= target_start_date) & # Sales after cutoff
    (sales_df['sale_date'] <= target_end_date)
].copy()

next_14d_total_qty_sold = sales_in_target_window.groupby('product_id')['quantity_sold'].sum().reset_index()
next_14d_total_qty_sold.rename(columns={'quantity_sold': 'next_14d_total_qty_sold'}, inplace=True)

product_features_df = pd.merge(
    product_features_df,
    next_14d_total_qty_sold,
    on='product_id',
    how='left'
)
product_features_df['next_14d_total_qty_sold'].fillna(0, inplace=True)

# Categorize next_14d_total_qty_sold
# Adjust thresholds based on actual data distribution to ensure reasonable class balance
# Example: Using quantiles to inform thresholds could be robust, but for this task, fixed thresholds are given.
demand_bins = [-1, 10, 50, float('inf')] # -1 to include 0-10, lower bound inclusive
demand_labels = ['Low', 'Medium', 'High']
product_features_df['next_14d_demand_category'] = pd.cut(
    product_features_df['next_14d_total_qty_sold'],
    bins=demand_bins,
    labels=demand_labels,
    right=True, # (a, b], so 10 is in 'Low', 50 is in 'Medium'
    include_lowest=True
)

# Convert target labels to numerical for stratified split and model training
le = LabelEncoder()
product_features_df['next_14d_demand_category_encoded'] = le.fit_transform(product_features_df['next_14d_demand_category'])
# Store original labels for report
target_names = le.classes_ # Ordered according to LabelEncoder (e.g., 'High', 'Low', 'Medium')

# Define features X and target y
numerical_features = [
    'base_price',
    'avg_qty_sold_prev_30d',
    'total_qty_sold_prev_30d',
    'num_sales_prev_30d',
    'days_since_last_sale_at_cutoff',
    'product_age_at_cutoff_days'
]
categorical_features = ['category', 'brand']

X = product_features_df[numerical_features + categorical_features]
y = product_features_df['next_14d_demand_category_encoded']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Target distribution in full dataset:\n{product_features_df['next_14d_demand_category'].value_counts(normalize=True)}")
# Print with original label names for clarity
y_train_counts = pd.Series(y_train).map(dict(zip(range(len(target_names)), target_names)))
y_test_counts = pd.Series(y_test).map(dict(zip(range(len(target_names)), target_names)))
print(f"Target distribution in training set:\n{y_train_counts.value_counts(normalize=True)}")
print(f"Target distribution in test set:\n{y_test_counts.value_counts(normalize=True)}")


# --- 4. Data Visualization (Matplotlib/Seaborn) ---

print("\n--- Generating Data Visualizations ---")

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Violin plot: Distribution of total_qty_sold_prev_30d for each demand category
sns.violinplot(
    x='next_14d_demand_category',
    y='total_qty_sold_prev_30d',
    data=product_features_df,
    order=demand_labels, # Ensure correct order of categories: Low, Medium, High
    palette='viridis',
    ax=axes[0]
)
axes[0].set_title('Distribution of Previous 30-Day Sales by Next 14-Day Demand Category')
axes[0].set_xlabel('Next 14-Day Demand Category')
axes[0].set_ylabel('Total Quantity Sold (Previous 30 Days)')

# Stacked bar chart: Proportion of next_14d_demand_category for different category values
category_demand_counts = product_features_df.groupby(['category', 'next_14d_demand_category']).size().unstack(fill_value=0)
# Normalize to get proportions for stacking
category_demand_proportions = category_demand_counts.div(category_demand_counts.sum(axis=1), axis=0)
# Ensure columns are in the desired order (Low, Medium, High)
category_demand_proportions = category_demand_proportions[demand_labels]
category_demand_proportions.plot(
    kind='bar',
    stacked=True,
    ax=axes[1],
    cmap='viridis',
    edgecolor='black'
)
axes[1].set_title('Proportion of Demand Categories by Product Category')
axes[1].set_xlabel('Product Category')
axes[1].set_ylabel('Proportion')
axes[1].tick_params(axis='x', rotation=45, ha='right')
axes[1].legend(title='Demand Category', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation (Multi-class Classification) ---

print("\n--- Building and Training ML Pipeline ---")

# Preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop other columns not specified (like product_id, dates)
)

# Create the full machine learning pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("Training the HistGradientBoostingClassifier model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
print("\n--- Classification Report (Test Set) ---")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\n--- Model Information ---")
# HistGradientBoostingClassifier does not have direct feature importances like RandomForest,
# and it's behind a ColumnTransformer. Retrieving full feature names after one-hot encoding
# and linking them back to the model's internal structure can be complex.
# We'll just list the initial features and note the nature of the model.

print(f"Numerical features used: {numerical_features}")
print(f"Categorical features (pre-one-hot encoding): {categorical_features}")
print("The model is a HistGradientBoostingClassifier, an ensemble of decision trees.")
print("It automatically handles missing values (imputed by mean, then scaled) and categorical features (one-hot encoded) via the preprocessing pipeline.")
print("Feature importance for this model within a ColumnTransformer pipeline requires more advanced techniques like permutation importance, which are beyond standard libraries.")