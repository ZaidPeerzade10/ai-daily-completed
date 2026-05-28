import pandas as pd
import numpy as np
import sqlite3
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

print("--- 1. Generating Synthetic Data ---")

# 1. Generate Synthetic Data
# Products DataFrame
num_products = np.random.randint(1000, 1501)
categories = ['Electronics', 'Fashion', 'Home Goods', 'Beauty', 'Books', 'Sports']
brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']

product_data = {
    'product_id': range(1, num_products + 1),
    'product_name': [f'Product_{i}' for i in range(1, num_products + 1)],
    'category': np.random.choice(categories, num_products),
    'brand': np.random.choice(brands, num_products),
    'base_cost': np.random.uniform(10, 500, num_products),
    'release_date': pd.to_datetime('2020-01-01') + pd.to_timedelta(np.random.randint(0, 3 * 365, num_products), unit='D')
}
products_df = pd.DataFrame(product_data)

# Introduce some NaN in base_cost
nan_indices = np.random.choice(products_df.index, int(num_products * 0.02), replace=False)
products_df.loc[nan_indices, 'base_cost'] = np.nan

# Historical Listings DataFrame
num_listings = np.random.randint(20000, 30001)

# Base conversion rates for categories/brands to simulate performance differences
category_cr_effect = {cat: np.random.uniform(0.8, 1.2) for cat in categories}
brand_cr_effect = {brand: np.random.uniform(0.7, 1.3) for brand in brands}

listing_data = {
    'listing_id': range(1, num_listings + 1),
    'product_id': np.random.choice(products_df['product_id'], num_listings),
    'listing_date': pd.NaT,  # Will fill this later
    'listed_price': np.zeros(num_listings),
    'impressions': np.random.randint(100, 5000, num_listings),
    'conversions': np.zeros(num_listings).astype(int)
}
historical_listings_df = pd.DataFrame(listing_data)

# Join with products_df to get release_date, category, brand, base_cost for conversion simulation
historical_listings_df = historical_listings_df.merge(
    products_df[['product_id', 'release_date', 'category', 'brand', 'base_cost']],
    on='product_id',
    how='left'
)

# Simulate listing_date after release_date
min_listing_delay_days = 1 # Minimum 1 day after release
max_listing_delay_days = 900 # Up to 900 days after release
historical_listings_df['listing_date'] = historical_listings_df.apply(
    lambda row: row['release_date'] + pd.to_timedelta(np.random.randint(min_listing_delay_days, max_listing_delay_days), unit='D'),
    axis=1
)

# Ensure listing_date doesn't exceed a reasonable max (e.g., today)
max_overall_date = pd.to_datetime('now') - pd.to_timedelta(30, unit='D') # Ensure enough future data
historical_listings_df['listing_date'] = historical_listings_df['listing_date'].apply(
    lambda x: x if x < max_overall_date else max_overall_date - pd.to_timedelta(np.random.randint(0,30), unit='D')
)

# Simulate listed_price (generally higher than base_cost)
price_multiplier = np.random.uniform(1.2, 3.0, num_listings)
historical_listings_df['listed_price'] = historical_listings_df['base_cost'] * price_multiplier
# Add some noise to listed_price
historical_listings_df['listed_price'] = historical_listings_df['listed_price'] * np.random.uniform(0.9, 1.1, num_listings)
historical_listings_df['listed_price'] = historical_listings_df['listed_price'].clip(lower=10) # Minimum price

# Simulate conversions
# Base conversion rate (e.g., 1-15%)
base_cr = np.random.uniform(0.01, 0.15, num_listings)

# Apply effects:
# Positive correlation with impressions (implicit from impressions * CR)
# Inverse correlation with listed_price
price_to_cost = (historical_listings_df['listed_price'] / (historical_listings_df['base_cost'] + 1e-6)).fillna(1)
price_effect = 1 / (1 + price_to_cost / 5) # higher price, lower CR (e.g. at price/cost ratio 5, effect is 1/2)

# Category and Brand effects
cat_effect = historical_listings_df['category'].map(category_cr_effect)
brand_effect = historical_listings_df['brand'].map(brand_cr_effect)

# Slight positive trend over time (e.g., +5% per year)
time_effect = 1 + (historical_listings_df['listing_date'] - historical_listings_df['listing_date'].min()).dt.days / 365 * 0.05

# Combine effects and calculate conversions
simulated_cr = base_cr * cat_effect * brand_effect * price_effect * time_effect
simulated_cr = simulated_cr.clip(0.001, 0.25) # Clip CR to a realistic range

historical_listings_df['conversions'] = (historical_listings_df['impressions'] * simulated_cr).round().astype(int)
historical_listings_df['conversions'] = historical_listings_df['conversions'].clip(lower=0, upper=historical_listings_df['impressions'])

# Drop temporary columns used for simulation
historical_listings_df = historical_listings_df.drop(columns=['release_date', 'category', 'brand', 'base_cost'])

# Sort historical_listings_df by listing_date
historical_listings_df = historical_listings_df.sort_values(by='listing_date').reset_index(drop=True)

print(f"Generated {len(products_df)} products and {len(historical_listings_df)} historical listings.")
print(f"Products head:\n{products_df.head()}")
print(f"Historical listings head:\n{historical_listings_df.head()}")

print("\n--- 2. Load into SQLite & SQL Feature Engineering ---")

# 2. Load into SQLite & SQL Feature Engineering
conn = sqlite3.connect(':memory:')
products_df.to_sql('products', conn, index=False)
historical_listings_df.to_sql('listings', conn, index=False)

# Define GLOBAL_PREDICTION_CUTOFF_DATE
# Adjusted to be closer to max_listing_date to get more data for prediction.
max_listing_date_db = pd.read_sql("SELECT MAX(listing_date) FROM listings", conn).iloc[0, 0]
max_listing_date_dt = pd.to_datetime(max_listing_date_db)
GLOBAL_PREDICTION_CUTOFF_DATE = (max_listing_date_dt - timedelta(weeks=3)).strftime('%Y-%m-%d %H:%M:%S') # Adjusted from 2 months to 3 weeks

print(f"Max listing date in historical data: {max_listing_date_dt}")
print(f"Global Prediction Cutoff Date: {GLOBAL_PREDICTION_CUTOFF_DATE}")

# SQL Query for time-windowed aggregations and future listings
sql_query = f"""
WITH CategoryAggregates AS (
    SELECT
        p.category,
        CAST(SUM(CASE WHEN l.impressions = 0 THEN 0.0 ELSE CAST(l.conversions AS REAL) * 1.0 / l.impressions END) AS REAL) / NULLIF(COUNT(l.listing_id), 0) AS avg_cat_cr,
        COUNT(l.listing_id) AS num_cat_listings,
        AVG(l.listed_price) AS avg_cat_listed_price
    FROM
        listings l
    JOIN
        products p ON l.product_id = p.product_id
    WHERE
        l.listing_date >= datetime('{GLOBAL_PREDICTION_CUTOFF_DATE}', '-60 days') AND
        l.listing_date < '{GLOBAL_PREDICTION_CUTOFF_DATE}'
    GROUP BY
        p.category
),
BrandAggregates AS (
    SELECT
        p.brand,
        CAST(SUM(CASE WHEN l.impressions = 0 THEN 0.0 ELSE CAST(l.conversions AS REAL) * 1.0 / l.impressions END) AS REAL) / NULLIF(COUNT(l.listing_id), 0) AS avg_brand_cr,
        COUNT(l.listing_id) AS num_brand_listings
    FROM
        listings l
    JOIN
        products p ON l.product_id = p.product_id
    WHERE
        l.listing_date >= datetime('{GLOBAL_PREDICTION_CUTOFF_DATE}', '-60 days') AND
        l.listing_date < '{GLOBAL_PREDICTION_CUTOFF_DATE}'
    GROUP BY
        p.brand
)
SELECT
    l.listing_id,
    l.product_id,
    p.category,
    p.brand,
    p.base_cost,
    p.release_date,
    l.listing_date,
    l.listed_price,
    l.impressions,
    l.conversions,
    -- Time-based features from current listing_date
    CAST(strftime('%w', l.listing_date) AS INTEGER) AS day_of_week, -- 0=Sunday, 6=Saturday
    CAST(strftime('%H', l.listing_date) AS INTEGER) AS hour_of_day,
    CAST(strftime('%m', l.listing_date) AS INTEGER) AS month_of_year,
    -- Historical aggregated features, COALESCE to handle NULLs from LEFT JOIN
    COALESCE(CA.avg_cat_cr, 0.0) AS avg_category_cr_prev_60d,
    COALESCE(CA.num_cat_listings, 0) AS num_listings_category_prev_60d,
    COALESCE(BA.avg_brand_cr, 0.0) AS avg_brand_cr_prev_60d,
    COALESCE(BA.num_brand_listings, 0) AS num_listings_brand_prev_60d,
    COALESCE(CA.avg_cat_listed_price, 0.0) AS avg_listed_price_category_prev_60d
FROM
    listings l
JOIN
    products p ON l.product_id = p.product_id
LEFT JOIN
    CategoryAggregates CA ON p.category = CA.category
LEFT JOIN
    BrandAggregates BA ON p.brand = BA.brand
WHERE
    l.listing_date >= '{GLOBAL_PREDICTION_CUTOFF_DATE}'
ORDER BY
    l.listing_date;
"""

listing_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Generated {len(listing_features_df)} features for listings after cutoff date.")
print(f"Listing features head:\n{listing_features_df.head()}")
print(f"Listing features descriptive stats:\n{listing_features_df.describe()}")


print("\n--- 3. Pandas Feature Engineering & Multi-class Target Creation ---")

# 3. Pandas Feature Engineering & Multi-class Target Creation
# Convert date/datetime columns
listing_features_df['release_date'] = pd.to_datetime(listing_features_df['release_date'])
listing_features_df['listing_date'] = pd.to_datetime(listing_features_df['listing_date'])

# Handle NaN values for numerical historical aggregates (should be covered by COALESCE, but defensive)
historical_agg_cols = [
    'avg_category_cr_prev_60d', 'num_listings_category_prev_60d',
    'avg_brand_cr_prev_60d', 'num_listings_brand_prev_60d',
    'avg_listed_price_category_prev_60d'
]
for col in historical_agg_cols:
    if 'cr' in col or 'price' in col:
        listing_features_df[col] = listing_features_df[col].fillna(0.0)
    else: # For counts
        listing_features_df[col] = listing_features_df[col].fillna(0).astype(int)

# Handle base_cost NaNs with mean
listing_features_df['base_cost'] = listing_features_df['base_cost'].fillna(listing_features_df['base_cost'].mean())

# Ensure impressions are not zero for conversion rate calculation
# Replace 0 with 1 to avoid division by zero, assuming 1 impression results in 0 conversion for 0 impressions
listing_features_df['impressions'] = listing_features_df['impressions'].replace(0, 1)

# Calculate listing_age_days
listing_features_df['listing_age_days'] = (listing_features_df['listing_date'] - listing_features_df['release_date']).dt.days
listing_features_df['listing_age_days'] = listing_features_df['listing_age_days'].clip(lower=0) # Age cannot be negative

# Calculate price_to_cost_ratio
listing_features_df['price_to_cost_ratio'] = listing_features_df['listed_price'] / (listing_features_df['base_cost'] + 1e-6)
# Fill NaN/inf
listing_features_df['price_to_cost_ratio'] = listing_features_df['price_to_cost_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Create Multi-class Target `conversion_category`
listing_features_df['conversion_rate'] = listing_features_df['conversions'] / listing_features_df['impressions']
listing_features_df['conversion_rate'] = listing_features_df['conversion_rate'].fillna(0) # In case impressions was nan or similar

# Define thresholds
conversion_rate_bins = [-np.inf, 0.03, 0.10, np.inf]
conversion_rate_labels = ['Low', 'Medium', 'High']
listing_features_df['conversion_category'] = pd.cut(
    listing_features_df['conversion_rate'],
    bins=conversion_rate_bins,
    labels=conversion_rate_labels,
    right=True
)
print(f"Conversion category distribution:\n{listing_features_df['conversion_category'].value_counts()}")

# Define features X and target y
numerical_features = [
    'base_cost', 'listed_price', 'avg_category_cr_prev_60d',
    'num_listings_category_prev_60d', 'avg_brand_cr_prev_60d',
    'num_listings_brand_prev_60d', 'avg_listed_price_category_prev_60d',
    'day_of_week', 'hour_of_day', 'month_of_year',
    'listing_age_days', 'price_to_cost_ratio'
]
categorical_features = ['category', 'brand']

X = listing_features_df[numerical_features + categorical_features]
y = listing_features_df['conversion_category']

if len(listing_features_df) < 20 or y.nunique() < 2:
    print("\nWarning: Insufficient data or target classes for meaningful ML training. Skipping ML and visualization steps.")
else:
    # Split into training and testing sets
    if y.nunique() < 2:
        print("Warning: Less than 2 unique classes in target 'conversion_category'. Cannot stratify. Splitting without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"\nTraining set size: {len(X_train)} rows")
    print(f"Test set size: {len(X_test)} rows")
    print(f"Target distribution in training set:\n{y_train.value_counts(normalize=True)}")
    print(f"Target distribution in test set:\n{y_test.value_counts(normalize=True)}")

    print("\n--- 4. Data Visualization ---")

    # 4. Data Visualization
    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot 1: Violin plot of listed_price by conversion_category
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='conversion_category', y='listed_price', data=listing_features_df, order=conversion_rate_labels, palette='viridis')
    plt.title('Distribution of Listed Price by Conversion Category')
    plt.xlabel('Conversion Category')
    plt.ylabel('Listed Price')
    plt.show()

    # Plot 2: Stacked bar chart of conversion_category proportion by product category
    category_conversion_proportion = listing_features_df.groupby('category')['conversion_category'].value_counts(normalize=True).unstack(fill_value=0)
    
    # CRITICAL FIX: Ensure all target columns ('Low', 'Medium', 'High') are present, fill missing with 0
    # Reindex to ensure consistent column order and fill missing categories for robustness
    category_conversion_proportion = category_conversion_proportion.reindex(columns=conversion_rate_labels, fill_value=0)

    category_conversion_proportion.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
    plt.title('Proportion of Conversion Categories per Product Category')
    plt.xlabel('Product Category')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Conversion Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    print("\n--- 5. ML Pipeline & Evaluation ---")

    # 5. ML Pipeline & Evaluation
    # Preprocessing with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any, though not expected here
    )

    # ML Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42))
    ])

    # Train the model
    print("Training the ML model...")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # Predict on the test set
    y_pred = model_pipeline.predict(X_test)

    # Evaluate the model
    print("\nClassification Report for Test Set:")
    # Ensure all labels are present in the report, even if not predicted
    print(classification_report(y_test, y_pred, labels=conversion_rate_labels, zero_division=0))

print("\n--- Script Finished ---")