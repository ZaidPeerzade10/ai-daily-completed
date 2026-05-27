import pandas as pd
import numpy as np
import datetime
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys

# Ensure consistent output for plots in a non-interactive environment
plt.switch_backend('Agg')

# Set a random seed for reproducibility
np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- Starting Synthetic Data Generation ---")

# --- 1. Synthetic Data Generation ---

# 1.1 customers_df
num_customers = np.random.randint(1000, 1501)
customer_ids = np.arange(1, num_customers + 1)
# Generate signup dates over the last 3-5 years
today = datetime.date.today()
signup_start_date = today - pd.DateOffset(years=5)
signup_end_date = today - pd.DateOffset(years=3)
signup_dates_temp = pd.to_datetime(pd.date_range(start=signup_start_date, end=signup_end_date, periods=num_customers))
np.random.shuffle(signup_dates_temp.values) # Shuffle to make dates random
loyalty_statuses = np.random.choice(['Bronze', 'Silver', 'Gold'], size=num_customers, p=[0.5, 0.3, 0.2])

customers_df = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates_temp,
    'loyalty_status': loyalty_statuses
})
# Ensure 'signup_date' is a proper datetime object for comparisons
customers_df['signup_date'] = pd.to_datetime(customers_df['signup_date'])


# 1.2 products_df
num_products = np.random.randint(200, 301)
product_ids = np.arange(1, num_products + 1)
product_names = [f'Product_{i}' for i in product_ids]
categories = np.random.choice(['Electronics', 'Books', 'Clothing', 'Home Goods', 'Groceries', 'Beauty'], size=num_products, p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
price_usd = np.random.uniform(10, 1000, size=num_products).round(2)

products_df = pd.DataFrame({
    'product_id': product_ids,
    'product_name': product_names,
    'category': categories,
    'price_usd': price_usd
})

# 1.3 reviews_df
num_reviews = np.random.randint(15000, 25001)
review_ids = np.arange(1, num_reviews + 1)

# Keywords for review text generation
positive_keywords = ['great', 'excellent', 'love it', 'fantastic', 'amazing', 'highly recommend', 'perfect', 'superb', 'happy']
neutral_keywords = ['okay', 'decent', 'average', 'fine', 'acceptable', 'not bad', 'fair', 'regular']
negative_keywords = ['bad', 'disappointing', 'terrible', 'awful', 'horrible', 'poor quality', 'frustrating', 'not good', 'unhappy']
generic_keywords = ['product', 'item', 'delivery', 'service', 'price', 'quality', 'features', 'easy to use', 'worth', 'time']

reviews_data = []
# Pre-sample customer and product IDs for efficiency
customer_id_samples = np.random.choice(customers_df['customer_id'], size=num_reviews, replace=True)
product_id_samples = np.random.choice(products_df['product_id'], size=num_reviews, replace=True)

# Map customer/product IDs to their attributes for faster lookup
customer_signup_map = customers_df.set_index('customer_id')['signup_date'].to_dict()
customer_loyalty_map = customers_df.set_index('customer_id')['loyalty_status'].to_dict()
product_category_map = products_df.set_index('product_id')['category'].to_dict()

# Generate review dates for the last 2 years, then select from these
review_date_pool_start = today - pd.DateOffset(years=2)
review_date_pool_end = today
review_date_pool = pd.to_datetime(pd.date_range(start=review_date_pool_start, end=review_date_pool_end, freq='D'))

for i in range(num_reviews):
    cust_id = customer_id_samples[i]
    prod_id = product_id_samples[i]
    
    customer_signup_date = customer_signup_map[cust_id]
    
    # Ensure review_date is after signup_date and within the last 2 years
    eligible_review_dates = review_date_pool[review_date_pool > customer_signup_date]
    
    if len(eligible_review_dates) == 0:
        # Fallback: if no valid dates in pool, take signup_date + 1 day (ensure it's not in future)
        review_date = customer_signup_date + pd.Timedelta(days=1)
        if review_date.date() > today:
            review_date = pd.to_datetime(today)
    else:
        review_date = np.random.choice(eligible_review_dates)

    # Simulate rating correlation and biases
    rating = np.random.randint(1, 6)
    
    # Adjust rating based on loyalty status (Gold customers tend to give slightly higher ratings)
    loyalty = customer_loyalty_map[cust_id]
    if loyalty == 'Gold' and rating < 5 and np.random.rand() < 0.25: # 25% chance to bump up rating for Gold
        rating += 1
    rating = min(rating, 5) # Cap at 5

    # Adjust rating based on category (e.g., Electronics might have more extreme ratings)
    category = product_category_map[prod_id]
    if category == 'Electronics':
        if np.random.rand() < 0.15: # 15% chance for extreme rating
            rating = np.random.choice([1, 5])
            if loyalty == 'Gold' and rating == 1: # Gold customers less likely to give 1
                rating = 2 # bump to 2 if it was 1
    
    # Generate review text based on rating
    review_text_words = []
    num_words = np.random.randint(15, 50) # Review length

    if rating <= 2: # Negative
        review_text_words.extend(np.random.choice(negative_keywords, size=np.random.randint(2, 5), replace=True))
    elif rating == 3: # Neutral
        review_text_words.extend(np.random.choice(neutral_keywords, size=np.random.randint(2, 4), replace=True))
    else: # Positive (rating >= 4)
        review_text_words.extend(np.random.choice(positive_keywords, size=np.random.randint(2, 5), replace=True))
    
    # Add generic words
    words_needed = num_words - len(review_text_words)
    if words_needed > 0:
        review_text_words.extend(np.random.choice(generic_keywords, size=words_needed, replace=True))
    np.random.shuffle(review_text_words)
    review_text = ' '.join(review_text_words) + '.'

    reviews_data.append({
        'review_id': review_ids[i],
        'customer_id': cust_id,
        'product_id': prod_id,
        'review_date': review_date,
        'rating': rating,
        'review_text': review_text
    })

reviews_df = pd.DataFrame(reviews_data)

# Sort reviews_df by customer_id then review_date for consistent historical aggregations
reviews_df = reviews_df.sort_values(by=['customer_id', 'review_date']).reset_index(drop=True)

print(f"Generated {len(customers_df)} customers, {len(products_df)} products, {len(reviews_df)} reviews.")
print("\nCustomers DataFrame head:")
print(customers_df.head())
print("\nProducts DataFrame head:")
print(products_df.head())
print("\nReviews DataFrame head:")
print(reviews_df.head())

print("\n--- Loading Data into SQLite & SQL Feature Engineering ---")

# --- 2. Load into SQLite & SQL Feature Engineering ---
conn = sqlite3.connect(':memory:')

# Ensure dates are stored in a format SQLite can handle as text, then convert back later
customers_df['signup_date'] = customers_df['signup_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
reviews_df['review_date'] = reviews_df['review_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

customers_df.to_sql('customers', conn, index=False, if_exists='replace')
products_df.to_sql('products', conn, index=False, if_exists='replace')
reviews_df.to_sql('reviews', conn, index=False, if_exists='replace')

sql_query = """
WITH ProductAggregates AS (
    SELECT
        product_id,
        AVG(rating) AS product_avg_rating_all_time,
        COUNT(rating) AS product_num_reviews_all_time
    FROM reviews
    GROUP BY product_id
),
CustomerHistoricalAggregates AS (
    SELECT
        review_id,
        customer_id,
        review_date,
        rating,
        -- Customer's average rating BEFORE this review
        -- Uses 1 PRECEDING to get the window excluding the current row, ordered by review_date
        AVG(rating) OVER (PARTITION BY customer_id ORDER BY review_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS customer_avg_rating_prev,
        -- Customer's number of reviews BEFORE this review
        COUNT(rating) OVER (PARTITION BY customer_id ORDER BY review_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS customer_num_reviews_prev
    FROM reviews
)
SELECT
    r.review_id,
    r.review_text,
    r.rating,
    c.loyalty_status,
    p.category,
    p.price_usd,
    COALESCE(cha.customer_avg_rating_prev, 0.0) AS customer_avg_rating_prev,
    COALESCE(cha.customer_num_reviews_prev, 0) AS customer_num_reviews_prev,
    COALESCE(pa.product_avg_rating_all_time, 0.0) AS product_avg_rating_all_time,
    COALESCE(pa.product_num_reviews_all_time, 0) AS product_num_reviews_all_time
FROM reviews AS r
LEFT JOIN customers AS c ON r.customer_id = c.customer_id
LEFT JOIN products AS p ON r.product_id = p.product_id
LEFT JOIN CustomerHistoricalAggregates AS cha ON r.review_id = cha.review_id
LEFT JOIN ProductAggregates AS pa ON r.product_id = pa.product_id
ORDER BY r.customer_id, r.review_date;
"""

review_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("\nFeatures DataFrame head after SQL aggregations:")
print(review_features_df.head())
print("\nFeatures DataFrame info after SQL aggregations:")
review_features_df.info()

print("\n--- Starting Pandas Feature Engineering & Multi-class Target Creation ---")

# --- 3. Pandas Feature Engineering & Multi-class Target Creation ---

# Handle NaN values (COALESCE in SQL already provided defaults, but ensuring types)
review_features_df['customer_avg_rating_prev'] = review_features_df['customer_avg_rating_prev'].astype(float)
review_features_df['customer_num_reviews_prev'] = review_features_df['customer_num_reviews_prev'].astype(int)
review_features_df['product_avg_rating_all_time'] = review_features_df['product_avg_rating_all_time'].astype(float)
review_features_df['product_num_reviews_all_time'] = review_features_df['product_num_reviews_all_time'].astype(int)

# Calculate review_text_length
review_features_df['review_text_length'] = review_features_df['review_text'].apply(len)

# Create the Multi-class Target sentiment_category
def get_sentiment_category(rating):
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else: # rating >= 4
        return 'Positive'

review_features_df['sentiment_category'] = review_features_df['rating'].apply(get_sentiment_category)

print("\nFeatures DataFrame head after Pandas FE:")
print(review_features_df.head())
print("\nSentiment Category distribution:")
print(review_features_df['sentiment_category'].value_counts())

# Define features X and target y
numerical_features = [
    'price_usd',
    'customer_avg_rating_prev',
    'customer_num_reviews_prev',
    'product_avg_rating_all_time',
    'product_num_reviews_all_time',
    'review_text_length'
]
categorical_features = ['loyalty_status', 'category']
text_feature = 'review_text'

X = review_features_df[numerical_features + categorical_features + [text_feature]]
y = review_features_df['sentiment_category']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTrain set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")


print("\n--- Starting Data Visualization ---")

# --- 4. Data Visualization ---

# Redirect matplotlib output to a buffer to prevent direct display in a non-interactive environment
# This allows the script to run without a GUI backend for matplotlib, but indicates plots were generated.
buffer = io.BytesIO()

plt.figure(figsize=(10, 6))
sns.violinplot(x='sentiment_category', y='review_text_length', data=review_features_df, palette='viridis', order=['Negative', 'Neutral', 'Positive'])
plt.title('Distribution of Review Text Length by Sentiment Category')
plt.xlabel('Sentiment Category')
plt.ylabel('Review Text Length')
plt.tight_layout()
plt.savefig(buffer, format='png') # Save to buffer
plt.close() # Close plot to free memory

print("\nGenerated Plot 1: Violin plot of Review Text Length by Sentiment Category (image saved to internal buffer)")

plt.figure(figsize=(12, 7))
category_sentiment_counts = review_features_df.groupby(['category', 'sentiment_category']).size().unstack(fill_value=0)
# Ensure all sentiment categories are present, even if counts are 0, for consistent columns
for sentiment in ['Negative', 'Neutral', 'Positive']:
    if sentiment not in category_sentiment_counts.columns:
        category_sentiment_counts[sentiment] = 0
category_sentiment_counts = category_sentiment_counts[['Negative', 'Neutral', 'Positive']] # Order columns for consistency
category_sentiment_proportions = category_sentiment_counts.divide(category_sentiment_counts.sum(axis=1), axis=0)
category_sentiment_proportions.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Proportion of Sentiment Category by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(buffer, format='png') # Save to buffer
plt.close() # Close plot to free memory

print("Generated Plot 2: Stacked bar chart of Sentiment Category Proportions by Product Category (image saved to internal buffer)")


print("\n--- Starting ML Pipeline & Evaluation ---")

# --- 5. ML Pipeline & Evaluation ---

# Preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing for text feature
# FunctionTransformer to extract the text column as a 1D array of strings for TfidfVectorizer
text_transformer = Pipeline(steps=[
    ('selector', FunctionTransformer(lambda x: x.values.astype(str), accept_sparse=False)),
    ('tfidf', TfidfVectorizer(max_features=1000))
])

# Create a ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_feature)
    ],
    remainder='drop' # Drop any other columns not specified
)

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

print("\nTraining ML Pipeline...")
# Train the model
model_pipeline.fit(X_train, y_train)
print("ML Pipeline training complete.")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
print("\n--- Classification Report for Test Set ---")
print(classification_report(y_test, y_pred))

print("\n--- Script execution completed ---")