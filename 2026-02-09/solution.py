import pandas as pd
import numpy as np
import sqlite3
import datetime
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style("whitegrid")

print("--- 1. Generating Synthetic Data ---")

# 1. Generate Synthetic Data (Pandas/Numpy)
# Products DataFrame
num_products = random.randint(100, 150)
products_df = pd.DataFrame({
    'product_id': np.arange(1, num_products + 1),
    'category': np.random.choice(['Electronics', 'Books', 'Clothing', 'HomeGoods', 'Food'], size=num_products),
    'price': np.round(np.random.uniform(50.0, 500.0, size=num_products), 2),
    'release_date': pd.to_datetime('today') - pd.to_timedelta(np.random.randint(0, 3 * 365, size=num_products), unit='D')
})

# Reviews DataFrame
num_reviews = random.randint(800, 1200)
reviews_df = pd.DataFrame({
    'review_id': np.arange(1, num_reviews + 1),
    'product_id': np.random.choice(products_df['product_id'], size=num_reviews),
    'user_id': np.random.randint(1, 201, size=num_reviews),
    'rating': np.random.choice([1, 2, 3, 4, 5], size=num_reviews, p=[0.1, 0.15, 0.25, 0.25, 0.25]) # Biased towards 3-5
})

# Merge to get release_date for review_date generation
reviews_df = reviews_df.merge(products_df[['product_id', 'release_date']], on='product_id', how='left')

# Generate review_date always after release_date, but not too far into the future
now = pd.Timestamp.now()
reviews_df['review_date'] = reviews_df.apply(
    lambda row: row['release_date'] + pd.Timedelta(days=np.random.randint(1, (now - row['release_date']).days + 1))
    if (now - row['release_date']).days > 0 else row['release_date'] + pd.Timedelta(days=1),
    axis=1
)
# Ensure review_date doesn't go past today
reviews_df['review_date'] = reviews_df['review_date'].apply(lambda x: min(x, now))


# Synthetically generate review_text based on rating
positive_words = ['excellent', 'great', 'loved it', 'high quality', 'perfect', 'amazing', 'fantastic', 'superb', 'highly recommend']
neutral_words = ['ok', 'fine', 'average', 'decent', 'acceptable', 'fair', 'neither good nor bad']
negative_words = ['bad', 'terrible', 'broken', 'disappointing', 'poor quality', 'unacceptable', 'frustrating', 'waste of money']
generic_words = ['product', 'item', 'delivery was fast', 'would buy again', 'met expectations', 'easy to use', 'good value']

def generate_review_text(rating):
    text_parts = []
    if rating >= 4:
        text_parts.append(random.choice(positive_words))
        text_parts.append(random.choice(generic_words))
        if random.random() < 0.3: # Add another positive for high ratings
            text_parts.append(random.choice(positive_words))
    elif rating == 3:
        text_parts.append(random.choice(neutral_words))
        text_parts.append(random.choice(generic_words))
        if random.random() < 0.2: # Sometimes a mixed sentiment
            text_parts.append(random.choice(random.choice([positive_words, negative_words])))
    else: # rating <= 2
        text_parts.append(random.choice(negative_words))
        text_parts.append(random.choice(generic_words))
        if random.random() < 0.3: # Add another negative for low ratings
            text_parts.append(random.choice(negative_words))
    
    # Add some additional generic filler words to make it longer
    for _ in range(random.randint(0, 3)):
        text_parts.append(random.choice(generic_words))
        
    random.shuffle(text_parts)
    return ' '.join(text_parts).capitalize() + '.'

reviews_df['review_text'] = reviews_df['rating'].apply(generate_review_text)

# Drop the temporary release_date column
reviews_df.drop(columns=['release_date'], inplace=True)

print(f"Generated {len(products_df)} products and {len(reviews_df)} reviews.")
print("Products head:\n", products_df.head())
print("\nReviews head:\n", reviews_df.head())

print("\n--- 2. Loading into SQLite & SQL Feature Engineering ---")

# 2. Load into SQLite & SQL Feature Engineering
conn = sqlite3.connect(':memory:')

# Convert datetime columns to string for SQLite storage
products_df['release_date_str'] = products_df['release_date'].dt.strftime('%Y-%m-%d')
reviews_df['review_date_str'] = reviews_df['review_date'].dt.strftime('%Y-%m-%d')

products_df.to_sql('products', conn, if_exists='replace', index=False, columns=['product_id', 'category', 'price', 'release_date_str'])
reviews_df.to_sql('reviews', conn, if_exists='replace', index=False, columns=['review_id', 'product_id', 'user_id', 'review_date_str', 'rating', 'review_text'])

# Determine global_analysis_date
global_analysis_date_pd = reviews_df['review_date'].max() + pd.Timedelta(days=30)
global_analysis_date_str = global_analysis_date_pd.strftime('%Y-%m-%d')

print(f"Global Analysis Date: {global_analysis_date_str}")

# SQL Query
sql_query = f"""
SELECT
    p.product_id,
    p.category,
    p.price,
    p.release_date_str AS release_date,
    AVG(CASE WHEN r.review_date_str <= '{global_analysis_date_str}' THEN r.rating ELSE NULL END) AS avg_rating,
    COUNT(CASE WHEN r.review_date_str <= '{global_analysis_date_str}' THEN r.review_id ELSE NULL END) AS num_reviews,
    JULIANDAY('{global_analysis_date_str}') - JULIANDAY(MAX(CASE WHEN r.review_date_str <= '{global_analysis_date_str}' THEN r.review_date_str ELSE NULL END)) AS days_since_last_review,
    GROUP_CONCAT(CASE WHEN r.review_date_str <= '{global_analysis_date_str}' THEN r.review_text ELSE NULL END, ' ') AS concatenated_reviews_text
FROM
    products p
LEFT JOIN
    reviews r ON p.product_id = r.product_id
GROUP BY
    p.product_id, p.category, p.price, p.release_date_str
ORDER BY
    p.product_id;
"""

product_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("SQL Feature Engineering complete. Resulting DataFrame head:")
print(product_features_df.head())
print(f"Shape: {product_features_df.shape}")

print("\n--- 3. Pandas Feature Engineering & Binary Target Creation ---")

# 3. Pandas Feature Engineering & Binary Target Creation
# Handle NaN values
product_features_df['num_reviews'] = product_features_df['num_reviews'].fillna(0).astype(int)
product_features_df['avg_rating'] = product_features_df['avg_rating'].fillna(3.0)
product_features_df['days_since_last_review'] = product_features_df['days_since_last_review'].fillna(365 * 5) # Large sentinel
product_features_df['concatenated_reviews_text'] = product_features_df['concatenated_reviews_text'].fillna('')

# Convert release_date to datetime objects
product_features_df['release_date'] = pd.to_datetime(product_features_df['release_date'])

# Calculate product_age_at_analysis_days
product_features_df['product_age_at_analysis_days'] = (global_analysis_date_pd - product_features_df['release_date']).dt.days
# Ensure age is not negative (for products released very close to or after global_analysis_date, though should be handled by synthetic data)
product_features_df['product_age_at_analysis_days'] = product_features_df['product_age_at_analysis_days'].apply(lambda x: max(x, 1))

# Extract Text Features
positive_keywords_re = re.compile(r'\b(' + '|'.join(positive_words) + r')\b', re.IGNORECASE)
negative_keywords_re = re.compile(r'\b(' + '|'.join(negative_words) + r')\b', re.IGNORECASE)

def count_keywords(text, keyword_regex):
    if not isinstance(text, str):
        return 0
    return len(keyword_regex.findall(text))

product_features_df['positive_word_count'] = product_features_df['concatenated_reviews_text'].apply(
    lambda x: count_keywords(x, positive_keywords_re)
)
product_features_df['negative_word_count'] = product_features_df['concatenated_reviews_text'].apply(
    lambda x: count_keywords(x, negative_keywords_re)
)

# Calculate review_density
product_features_df['review_density'] = product_features_df['num_reviews'] / (product_features_df['product_age_at_analysis_days'] + 1)

# Create the Binary Target is_successful_product
# 70th percentile of num_reviews among products with at least one review
reviews_with_data = product_features_df[product_features_df['num_reviews'] > 0]
if not reviews_with_data.empty:
    min_reviews_for_70_percentile = reviews_with_data['num_reviews'].quantile(0.70)
else:
    min_reviews_for_70_percentile = 0 # Fallback if no products have reviews

product_features_df['is_successful_product'] = (
    (product_features_df['avg_rating'] >= 4.0) &
    (product_features_df['num_reviews'] >= min_reviews_for_70_percentile)
).astype(int)

# Define features X and target y
numerical_features = [
    'price', 'product_age_at_analysis_days', 'avg_rating', 'num_reviews',
    'days_since_last_review', 'positive_word_count', 'negative_word_count', 'review_density'
]
categorical_features = ['category']

X = product_features_df[numerical_features + categorical_features]
y = product_features_df['is_successful_product']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\nEngineered Features head (X_train):\n", X_train.head())
print("\nTarget distribution (y_train):\n", y_train.value_counts(normalize=True))
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

print("\n--- 4. Data Visualization ---")

# 4. Data Visualization
plt.figure(figsize=(14, 6))

# Plot 1: Violin plot of avg_rating for successful vs. unsuccessful products
plt.subplot(1, 2, 1)
sns.violinplot(x='is_successful_product', y='avg_rating', data=product_features_df)
plt.title('Distribution of Average Rating by Product Success')
plt.xlabel('Is Successful Product (0=No, 1=Yes)')
plt.ylabel('Average Rating')
plt.xticks([0, 1], ['Unsuccessful', 'Successful'])

# Plot 2: Stacked bar chart of success proportion across categories
plt.subplot(1, 2, 2)
category_success_counts = product_features_df.groupby('category')['is_successful_product'].value_counts(normalize=True).unstack().fillna(0)
category_success_counts.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Proportion of Successful Products by Category')
plt.xlabel('Category')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Is Successful', labels=['Unsuccessful', 'Successful'])

plt.tight_layout()
plt.show()

print("Visualization plots displayed.")

print("\n--- 5. ML Pipeline & Evaluation (Binary Classification) ---")

# 5. ML Pipeline & Evaluation (Binary Classification)
# Preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ML Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Train the pipeline
print("Training the Gradient Boosting Classifier pipeline...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# Predict probabilities and classes on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

# Calculate and print evaluation metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
classification_rep = classification_report(y_test, y_pred)

print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:\n", classification_rep)

print("\n--- Script Finished ---")