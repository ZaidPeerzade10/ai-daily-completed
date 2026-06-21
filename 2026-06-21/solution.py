import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

# 1. Synthetic Data Generation

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for data generation
NUM_PRODUCTS = np.random.randint(500, 1001)
NUM_CUSTOMERS = np.random.randint(1000, 1501)
NUM_REVIEWS = np.random.randint(20000, 30001)

# Generate product data
product_ids = np.arange(NUM_PRODUCTS)
product_names = [f"Product_{i+1}" for i in product_ids]
categories = np.random.choice(['Electronics', 'Books', 'Clothing', 'Home', 'Food', 'Outdoors'], NUM_PRODUCTS)
prices = np.random.uniform(10.0, 1000.0, NUM_PRODUCTS)

# Launch dates over the last 3-5 years, ensuring reviews can happen after launch
end_date_products = datetime.now() - timedelta(days=90) 
start_date_products = end_date_products - timedelta(days=5*365)
launch_dates = pd.to_datetime([start_date_products + timedelta(days=np.random.randint(0, (end_date_products - start_date_products).days)) for _ in product_ids])

products_df = pd.DataFrame({
    'product_id': product_ids,
    'product_name': product_names,
    'category': categories,
    'price': prices,
    'launch_date': launch_dates
})

# Generate customer data
customer_ids = np.arange(NUM_CUSTOMERS)
signup_dates = pd.to_datetime([datetime.now() - timedelta(days=np.random.randint(0, 7*365)) for _ in customer_ids])

customers_df = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates
})

# Generate review data
review_ids = np.arange(NUM_REVIEWS)
review_product_ids = np.random.choice(products_df['product_id'], NUM_REVIEWS)
review_customer_ids = np.random.choice(customers_df['customer_id'], NUM_REVIEWS)

# Simulate rating distribution (more 4-5 stars)
ratings = np.random.choice([1, 2, 3, 4, 5], NUM_REVIEWS, p=[0.05, 0.1, 0.2, 0.35, 0.3])

# Create temporary reviews_df to join with launch_date for realistic review_date generation
temp_reviews_df = pd.DataFrame({
    'review_id': review_ids,
    'product_id': review_product_ids,
    'customer_id': review_customer_ids,
    'rating': ratings
})

# Merge launch_date to reviews_df temporarily
temp_reviews_df = pd.merge(temp_reviews_df, products_df[['product_id', 'launch_date']], on='product_id', how='left')

# Generate review_date, ensuring it's after launch_date and not in the future
min_review_delta_days = 1 # Reviews must be at least 1 day after launch
max_review_delta_days = (datetime.now() - products_df['launch_date'].min()).days + 30 # Max range for review dates

temp_reviews_df['review_date_offset'] = np.random.randint(min_review_delta_days, max_review_delta_days, NUM_REVIEWS)
temp_reviews_df['review_date'] = temp_reviews_df['launch_date'] + pd.to_timedelta(temp_reviews_df['review_date_offset'], unit='D')

# Ensure review_date is not in the future
temp_reviews_df['review_date'] = temp_reviews_df['review_date'].apply(lambda x: min(x, datetime.now()))

# Drop temporary columns
reviews_df = temp_reviews_df[['review_id', 'product_id', 'customer_id', 'review_date', 'rating']]

# Sort reviews_df as required
reviews_df = reviews_df.sort_values(by=['product_id', 'review_date']).reset_index(drop=True)

print(f"Generated {len(products_df)} products, {len(customers_df)} customers, {len(reviews_df)} reviews.")
print("Products Head:\n", products_df.head())
print("Reviews Head:\n", reviews_df.head())

# 2. Load into SQLite & SQL Feature Engineering

conn = sqlite3.connect(':memory:')

products_df.to_sql('products', conn, if_exists='replace', index=False, dtype={'launch_date': 'TEXT'})
customers_df.to_sql('customers', conn, if_exists='replace', index=False, dtype={'signup_date': 'TEXT'})
reviews_df.to_sql('reviews', conn, if_exists='replace', index=False, dtype={'review_date': 'TEXT'})

# Define GLOBAL_PREDICTION_CUTOFF_DATE
latest_review_date = pd.to_datetime(reviews_df['review_date']).max()
GLOBAL_PREDICTION_CUTOFF_DATE = latest_review_date - pd.Timedelta(days=45)
GLOBAL_PREDICTION_CUTOFF_DATE_STR = GLOBAL_PREDICTION_CUTOFF_DATE.strftime('%Y-%m-%d %H:%M:%S')

print(f"\nGLOBAL_PREDICTION_CUTOFF_DATE: {GLOBAL_PREDICTION_CUTOFF_DATE_STR}")

# SQL Query for feature engineering
sql_query = f"""
WITH ProductBase AS (
    SELECT
        p.product_id,
        p.category,
        p.price,
        p.launch_date
    FROM products p
),
ReviewsBeforeCutoff AS (
    SELECT
        r.product_id,
        r.review_id,
        r.rating,
        r.review_date
    FROM reviews r
    WHERE r.review_date <= '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}'
),
ReviewsPrev90d AS (
    SELECT
        r.product_id,
        r.rating,
        r.review_date,
        r.review_id
    FROM ReviewsBeforeCutoff r
    WHERE julianday(r.review_date) > julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') - 90
),
LastReviewDates AS (
    SELECT
        product_id,
        MAX(review_date) AS last_review_date_at_cutoff
    FROM ReviewsBeforeCutoff
    GROUP BY product_id
)
SELECT
    pb.product_id,
    pb.category,
    pb.price,
    pb.launch_date,
    COALESCE(AVG(r90.rating), 0.0) AS avg_rating_prev_90d,
    COALESCE(COUNT(r90.review_id), 0) AS num_reviews_prev_90d,
    CASE
        WHEN lrd.last_review_date_at_cutoff IS NOT NULL
        THEN CAST(julianday('{GLOBAL_PREDICTION_CUTOFF_DATE_STR}') - julianday(lrd.last_review_date_at_cutoff) AS INTEGER)
        ELSE 9999
    END AS days_since_last_review_at_cutoff,
    COALESCE(COUNT(rbc.review_id), 0) AS total_reviews_since_launch,
    '{GLOBAL_PREDICTION_CUTOFF_DATE_STR}' AS GLOBAL_PREDICTION_CUTOFF_DATE
FROM ProductBase pb
LEFT JOIN ReviewsPrev90d r90 ON pb.product_id = r90.product_id
LEFT JOIN ReviewsBeforeCutoff rbc ON pb.product_id = rbc.product_id
LEFT JOIN LastReviewDates lrd ON pb.product_id = lrd.product_id
GROUP BY pb.product_id, pb.category, pb.price, pb.launch_date
ORDER BY pb.product_id;
"""

product_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("\nProduct Features (SQL Aggregations) Head:\n", product_features_df.head())
print("Product Features Info:\n", product_features_df.info())


# 3. Pandas Feature Engineering & Multi-class Target Creation

# Convert date columns to datetime objects
product_features_df['launch_date'] = pd.to_datetime(product_features_df['launch_date'])
product_features_df['GLOBAL_PREDICTION_CUTOFF_DATE'] = pd.to_datetime(product_features_df['GLOBAL_PREDICTION_CUTOFF_DATE'])

# Handle NaN values (COALESCE in SQL mostly handles this, but for robustness)
product_features_df['avg_rating_prev_90d'] = product_features_df['avg_rating_prev_90d'].fillna(0.0)
product_features_df['num_reviews_prev_90d'] = product_features_df['num_reviews_prev_90d'].fillna(0).astype(int)
product_features_df['total_reviews_since_launch'] = product_features_df['total_reviews_since_launch'].fillna(0).astype(int)
product_features_df['days_since_last_review_at_cutoff'] = product_features_df['days_since_last_review_at_cutoff'].fillna(9999).astype(int)

# Calculate product_age_at_cutoff_days
product_features_df['product_age_at_cutoff_days'] = (
    product_features_df['GLOBAL_PREDICTION_CUTOFF_DATE'] - product_features_df['launch_date']
).dt.days
# Ensure product age is not negative, should not happen with current data generation logic
product_features_df['product_age_at_cutoff_days'] = product_features_df['product_age_at_cutoff_days'].apply(lambda x: max(0, x))


# Create the Multi-class Target `next_30d_rating_category`
future_window_start = GLOBAL_PREDICTION_CUTOFF_DATE
future_window_end = GLOBAL_PREDICTION_CUTOFF_DATE + pd.Timedelta(days=30)

# Filter original reviews for the future window
reviews_in_future_window = reviews_df[
    (reviews_df['review_date'] > future_window_start) &
    (reviews_df['review_date'] <= future_window_end)
].copy() # Use .copy() to avoid SettingWithCopyWarning

# Calculate average rating for each product in the future window
future_avg_ratings = reviews_in_future_window.groupby('product_id')['rating'].mean().reset_index()
future_avg_ratings.rename(columns={'rating': 'avg_rating_future_30d'}, inplace=True)

# Merge with product_features_df. Use inner join to exclude products with no reviews in the future window
product_features_df = pd.merge(
    product_features_df,
    future_avg_ratings,
    on='product_id',
    how='inner' 
)

# Define rating categories based on percentiles of avg_rating_future_30d
p33 = product_features_df['avg_rating_future_30d'].quantile(0.33)
p66 = product_features_df['avg_rating_future_30d'].quantile(0.66)

def categorize_rating(avg_rating):
    if avg_rating <= p33:
        return 'Low'
    elif avg_rating <= p66:
        return 'Medium'
    else:
        return 'High'

product_features_df['next_30d_rating_category'] = product_features_df['avg_rating_future_30d'].apply(categorize_rating)

print(f"\nDataFrame after target creation (first 5 rows with target):\n", product_features_df[['product_id', 'avg_rating_future_30d', 'next_30d_rating_category']].head())
print("\nTarget category distribution:\n", product_features_df['next_30d_rating_category'].value_counts(normalize=True))

# Define features X and target y
features = [
    'price', 'avg_rating_prev_90d', 'num_reviews_prev_90d',
    'days_since_last_review_at_cutoff', 'total_reviews_since_launch',
    'product_age_at_cutoff_days', 'category'
]
target = 'next_30d_rating_category'

X = product_features_df[features]
y = product_features_df[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print(f"X_train columns: {X_train.columns.tolist()}")


# 4. Data Visualization

# Set style for plots
sns.set_style("whitegrid")
plt.figure(figsize=(15, 6))

# Plot 1: Violin plot of price vs. next_30d_rating_category
plt.subplot(1, 2, 1)
sns.violinplot(x='next_30d_rating_category', y='price', data=product_features_df, palette='viridis', order=['Low', 'Medium', 'High'])
plt.title('Product Price Distribution by Next 30-Day Rating Category')
plt.xlabel('Next 30-Day Rating Category')
plt.ylabel('Price')

# Plot 2: Stacked bar chart of next_30d_rating_category across categories
plt.subplot(1, 2, 2)
category_sentiment_proportion = product_features_df.groupby('category')['next_30d_rating_category'].value_counts(normalize=True).unstack().fillna(0)
# Ensure columns are in the desired order, handling cases where a category might be missing for a class
for col in ['Low', 'Medium', 'High']:
    if col not in category_sentiment_proportion.columns:
        category_sentiment_proportion[col] = 0
category_sentiment_proportion = category_sentiment_proportion[['Low', 'Medium', 'High']] 
category_sentiment_proportion.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Proportion of Rating Categories by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Rating Category')
plt.tight_layout()
plt.show()

# 5. ML Pipeline & Evaluation (Multi-class Classification)

# Define numerical and categorical features
numerical_features = [
    'price', 'avg_rating_prev_90d', 'num_reviews_prev_90d',
    'days_since_last_review_at_cutoff', 'total_reviews_since_launch',
    'product_age_at_cutoff_days'
]
categorical_features = ['category']

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
    ])

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the model
print("\nTraining the HistGradientBoostingClassifier model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
print("\n--- Model Evaluation ---")
print(classification_report(y_test, y_pred))

print("\nScript execution complete.")