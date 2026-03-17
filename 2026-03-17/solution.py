import pandas as pd
import numpy as np
import sqlite3
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import based on feedback
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Generate Synthetic Data ---
print("--- Generating Synthetic Data ---")

# 1.1 products_df
num_products = np.random.randint(100, 201)
categories = ['Electronics', 'Software', 'Apparel', 'HomeGoods', 'Books', 'Games']
product_ids = np.arange(1, num_products + 1)

products_data = {
    'product_id': product_ids,
    'launch_date': pd.to_datetime('2021-01-01') + pd.to_timedelta(np.random.randint(0, 3 * 365, num_products), unit='D'),
    'category': np.random.choice(categories, num_products),
    'initial_price': np.random.uniform(20.0, 5000.0, num_products),
    'marketing_spend_at_launch': np.random.uniform(1000.0, 50000.0, num_products)
}
products_df = pd.DataFrame(products_data)

# Introduce a "hidden_success_score" to bias interactions and sales
products_df['hidden_success_score'] = (
    products_df['marketing_spend_at_launch'] / products_df['marketing_spend_at_launch'].max() * 0.5 +
    np.where(products_df['category'] == 'Software', 0.3,
             np.where(products_df['category'] == 'Electronics', 0.2, 0.1)) +
    np.random.rand(num_products) * 0.2
)
products_df['hidden_success_score'] = products_df['hidden_success_score'].clip(0.1, 1.0) # Ensure score is within reasonable bounds

print(f"Generated {len(products_df)} products.")

# 1.2 user_interactions_df
num_interactions = np.random.randint(10000, 15001)
interaction_types = ['View', 'Add_to_Cart', 'Wishlist', 'Share', 'View'] # 'View' is more common

interaction_records = []
current_interaction_id = 1
for _, product in products_df.iterrows():
    # Number of interactions biased by marketing spend and hidden success score
    base_interactions = int(product['marketing_spend_at_launch'] / 500) + int(product['hidden_success_score'] * 50)
    num_prod_interactions = np.random.randint(max(10, base_interactions // 2), base_interactions * 2)

    for _ in range(num_prod_interactions):
        interaction_date = product['launch_date'] + pd.to_timedelta(np.random.randint(1, 15), unit='D')
        
        # Bias interaction type based on success score
        if np.random.rand() < product['hidden_success_score'] * 0.5: # Higher chance for 'Add_to_Cart'/'Wishlist' for successful products
            interaction_type = np.random.choice(['Add_to_Cart', 'Wishlist', 'Share'])
        else:
            interaction_type = np.random.choice(interaction_types)

        duration_seconds = 0.0
        if interaction_type == 'View':
            # Longer duration for higher success score
            duration_seconds = np.random.uniform(5, 600 * product['hidden_success_score'])
            duration_seconds = max(5.0, duration_seconds) # Min duration

        interaction_records.append({
            'interaction_id': current_interaction_id,
            'user_id': np.random.randint(1, 10001), # Simulate different users
            'product_id': product['product_id'],
            'interaction_date': interaction_date,
            'interaction_type': interaction_type,
            'duration_seconds': duration_seconds
        })
        current_interaction_id += 1

user_interactions_df = pd.DataFrame(interaction_records)
# Ensure interaction_date is within 14 days of launch_date (already done in generation but good to double check)
user_interactions_df = pd.merge(user_interactions_df, products_df[['product_id', 'launch_date']], on='product_id', how='left')
user_interactions_df = user_interactions_df[user_interactions_df['interaction_date'] <= user_interactions_df['launch_date'] + pd.to_timedelta(14, unit='D')]
user_interactions_df.drop(columns='launch_date', inplace=True)
print(f"Generated {len(user_interactions_df)} user interactions.")

# 1.3 sales_df
num_sales = np.random.randint(1000, 2001)
sales_records = []
current_sale_id = 1

# Aggregate early interactions to bias sales generation
early_engagement = user_interactions_df.groupby('product_id').agg(
    total_add_to_cart=pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Add_to_Cart').sum()),
    total_wishlist=pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Wishlist').sum())
).reset_index()
products_df_with_engagement = pd.merge(products_df, early_engagement, on='product_id', how='left').fillna(0)
products_df_with_engagement['early_engagement_score'] = products_df_with_engagement['total_add_to_cart'] * 2 + products_df_with_engagement['total_wishlist']
products_df_with_engagement['early_engagement_score'] = products_df_with_engagement['early_engagement_score'].clip(0, 50) # Cap score

for _, product in products_df_with_engagement.iterrows():
    # Number of sales biased by hidden success score and early engagement
    base_sales = int(product['hidden_success_score'] * 5) + int(product['early_engagement_score'] * 0.5)
    num_prod_sales = np.random.randint(0, max(1, base_sales * 2)) # Some products might have no sales

    for _ in range(num_prod_sales):
        sale_date = product['launch_date'] + pd.to_timedelta(np.random.randint(1, 61), unit='D')
        quantity_sold = np.random.randint(1, 10) # Base quantity

        # Increase quantity for highly successful products
        if product['hidden_success_score'] > 0.7:
            quantity_sold += np.random.randint(1, 5)

        sales_records.append({
            'sale_id': current_sale_id,
            'product_id': product['product_id'],
            'sale_date': sale_date,
            'quantity_sold': quantity_sold
        })
        current_sale_id += 1

sales_df = pd.DataFrame(sales_records)
if not sales_df.empty:
    # Ensure sales_date is within 60 days of launch_date (already done in generation but good to double check)
    sales_df = pd.merge(sales_df, products_df[['product_id', 'launch_date']], on='product_id', how='left')
    sales_df = sales_df[sales_df['sale_date'] <= sales_df['launch_date'] + pd.to_timedelta(60, unit='D')]
    sales_df.drop(columns='launch_date', inplace=True)
print(f"Generated {len(sales_df)} sales records.")

print("\nSample products_df:")
print(products_df.head())
print("\nSample user_interactions_df:")
print(user_interactions_df.head())
print("\nSample sales_df:")
print(sales_df.head())


# --- 2. Load into SQLite & SQL Feature Engineering ---
print("\n--- SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

products_df.to_sql('products', conn, index=False, if_exists='replace')
user_interactions_df.to_sql('user_interactions', conn, index=False, if_exists='replace')

sql_query = """
WITH ProductInteractions AS (
    SELECT
        i.product_id,
        SUM(CASE WHEN i.interaction_type = 'View' THEN 1 ELSE 0 END) AS total_views_first_14d_raw,
        SUM(CASE WHEN i.interaction_type = 'Add_to_Cart' THEN 1 ELSE 0 END) AS total_add_to_cart_first_14d_raw,
        SUM(CASE WHEN i.interaction_type = 'Wishlist' THEN 1 ELSE 0 END) AS total_wishlist_first_14d_raw,
        AVG(CASE WHEN i.interaction_type = 'View' THEN i.duration_seconds ELSE NULL END) AS avg_view_duration_first_14d_raw,
        COUNT(DISTINCT i.user_id) AS num_unique_users_interacting_first_14d_raw,
        MIN(julianday(i.interaction_date) - julianday(p.launch_date)) AS days_from_launch_to_first_interaction_raw
    FROM
        user_interactions AS i
    JOIN
        products AS p ON i.product_id = p.product_id
    WHERE
        julianday(i.interaction_date) BETWEEN julianday(p.launch_date) AND julianday(DATE(p.launch_date, '+14 days'))
    GROUP BY
        i.product_id
)
SELECT
    p.product_id,
    p.launch_date,
    p.category,
    p.initial_price,
    p.marketing_spend_at_launch,
    COALESCE(pi.total_views_first_14d_raw, 0) AS total_views_first_14d,
    COALESCE(pi.total_add_to_cart_first_14d_raw, 0) AS total_add_to_cart_first_14d,
    COALESCE(pi.total_wishlist_first_14d_raw, 0) AS total_wishlist_first_14d,
    COALESCE(pi.avg_view_duration_first_14d_raw, 0.0) AS avg_view_duration_first_14d,
    COALESCE(pi.num_unique_users_interacting_first_14d_raw, 0) AS num_unique_users_interacting_first_14d,
    pi.days_from_launch_to_first_interaction_raw AS days_from_launch_to_first_interaction
FROM
    products AS p
LEFT JOIN
    ProductInteractions AS pi ON p.product_id = pi.product_id;
"""

product_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Generated {len(product_features_df)} product features using SQL.")
print("\nSample product_features_df after SQL aggregation:")
print(product_features_df.head())


# --- 3. Pandas Feature Engineering & Multi-Class Target Creation ---
print("\n--- Pandas Feature Engineering & Target Creation ---")

# Handle NaN values from SQL aggregation
product_features_df['launch_date'] = pd.to_datetime(product_features_df['launch_date'])
product_features_df['total_views_first_14d'] = product_features_df['total_views_first_14d'].fillna(0).astype(int)
product_features_df['total_add_to_cart_first_14d'] = product_features_df['total_add_to_cart_first_14d'].fillna(0).astype(int)
product_features_df['total_wishlist_first_14d'] = product_features_df['total_wishlist_first_14d'].fillna(0).astype(int)
product_features_df['num_unique_users_interacting_first_14d'] = product_features_df['num_unique_users_interacting_first_14d'].fillna(0).astype(int)
product_features_df['avg_view_duration_first_14d'] = product_features_df['avg_view_duration_first_14d'].fillna(0.0)
product_features_df['days_from_launch_to_first_interaction'] = product_features_df['days_from_launch_to_first_interaction'].fillna(14).astype(int)

# Calculate additional pandas features
product_features_df['total_interactions_first_14d'] = (
    product_features_df['total_views_first_14d'] +
    product_features_df['total_add_to_cart_first_14d'] +
    product_features_df['total_wishlist_first_14d']
)
product_features_df['interaction_frequency_per_day_first_14d'] = product_features_df['total_interactions_first_14d'] / 14.0
product_features_df['interaction_frequency_per_day_first_14d'] = product_features_df['interaction_frequency_per_day_first_14d'].fillna(0.0)

# Create the Multi-Class Target `product_success_tier`
products_launch_dates = products_df[['product_id', 'launch_date']].copy()
sales_df_merged = pd.merge(sales_df, products_launch_dates, on='product_id', how='left')

# Filter sales within 60 days
sales_in_first_60d = sales_df_merged[
    (sales_df_merged['sale_date'] >= sales_df_merged['launch_date']) &
    (sales_df_merged['sale_date'] <= sales_df_merged['launch_date'] + pd.to_timedelta(60, unit='D'))
]

total_sales_60d = sales_in_first_60d.groupby('product_id')['quantity_sold'].sum().reset_index()
total_sales_60d.rename(columns={'quantity_sold': 'total_sales_in_first_60d'}, inplace=True)

product_features_df = pd.merge(product_features_df, total_sales_60d, on='product_id', how='left')
product_features_df['total_sales_in_first_60d'] = product_features_df['total_sales_in_first_60d'].fillna(0)

# Calculate percentiles on non-zero sales
non_zero_sales = product_features_df[product_features_df['total_sales_in_first_60d'] > 0]['total_sales_in_first_60d']
if not non_zero_sales.empty:
    p33 = non_zero_sales.quantile(0.33)
    p66 = non_zero_sales.quantile(0.66)
else: # Fallback for edge case where all sales are zero or no sales at all
    p33 = 1 # Smallest possible positive value for tiers
    p66 = 2


def assign_success_tier(row):
    sales = row['total_sales_in_first_60d']
    if sales == 0:
        return 'Low_Success'
    elif sales <= p33:
        return 'Medium_Success'
    elif sales <= p66:
        return 'High_Success'
    else:
        return 'Very_High_Success'

product_features_df['product_success_tier'] = product_features_df.apply(assign_success_tier, axis=1)

# Define features and target
numerical_features = [
    'initial_price', 'marketing_spend_at_launch', 'total_views_first_14d',
    'total_add_to_cart_first_14d', 'total_wishlist_first_14d', 'avg_view_duration_first_14d',
    'num_unique_users_interacting_first_14d', 'days_from_launch_to_first_interaction',
    'total_interactions_first_14d', 'interaction_frequency_per_day_first_14d'
]
categorical_features = ['category']

X = product_features_df[numerical_features + categorical_features]
y = product_features_df['product_success_tier']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nDataFrame shape: {product_features_df.shape}")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train class distribution:\n{y_train.value_counts(normalize=True)}")
print(f"y_test class distribution:\n{y_test.value_counts(normalize=True)}")
print("\nSample product_features_df with target:")
print(product_features_df[['product_id', 'total_sales_in_first_60d', 'product_success_tier']].head())


# --- 4. Data Visualization ---
print("\n--- Data Visualization ---")
plt.style.use('ggplot')

# Violin plot of total_add_to_cart_first_14d by product_success_tier
plt.figure(figsize=(10, 6))
sns.violinplot(x='product_success_tier', y='total_add_to_cart_first_14d', data=product_features_df,
               order=['Low_Success', 'Medium_Success', 'High_Success', 'Very_High_Success'])
plt.title('Distribution of Early Add-to-Cart by Product Success Tier')
plt.xlabel('Product Success Tier')
plt.ylabel('Total Add to Cart (First 14 Days)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Stacked bar chart of product_success_tier proportions across categories
category_success_proportions = product_features_df.groupby('category')['product_success_tier'].value_counts(normalize=True).unstack().fillna(0)
# Ensure all tiers are present for consistent plotting, even if some categories don't have all tiers
all_tiers = ['Low_Success', 'Medium_Success', 'High_Success', 'Very_High_Success']
for tier in all_tiers:
    if tier not in category_success_proportions.columns:
        category_success_proportions[tier] = 0.0
category_success_proportions = category_success_proportions[all_tiers] # Reorder columns

plt.figure(figsize=(12, 7))
category_success_proportions.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Proportion of Product Success Tiers by Category')
plt.xlabel('Product Category')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Success Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---
print("\n--- Machine Learning Pipeline & Evaluation ---")

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
    ])

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the model
print("Training the HistGradientBoostingClassifier model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred, zero_division=0)

print(f"\nTest Set Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report_str)

print("\n--- Pipeline Execution Complete ---")