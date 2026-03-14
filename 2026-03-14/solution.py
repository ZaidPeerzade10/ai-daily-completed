import pandas as pd
import numpy as np
import datetime
import sqlite3
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import from sklearn.impute
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

print("--- Generating Synthetic Data ---")

# 1.1 users_df
num_users = np.random.randint(500, 701)
signup_dates = pd.to_datetime('today') - pd.to_timedelta(np.random.randint(0, 5 * 365, num_users), unit='days')
account_tiers = np.random.choice(['Bronze', 'Silver', 'Gold'], num_users, p=[0.5, 0.3, 0.2])

users_df = pd.DataFrame({
    'user_id': np.arange(num_users),
    'signup_date': signup_dates,
    'reputation_score': np.random.randint(0, 101, num_users),
    'account_tier': account_tiers
})
print(f"Generated {len(users_df)} users.")

# 1.2 products_df
num_products = np.random.randint(100, 201)
release_dates = pd.to_datetime('today') - pd.to_timedelta(np.random.randint(0, 3 * 365, num_products), unit='days')

products_df = pd.DataFrame({
    'product_id': np.arange(num_products),
    'product_category': np.random.choice(['Electronics', 'Books', 'Home&Kitchen', 'Apparel', 'Sports&Outdoors', 'Groceries'], num_products),
    'price': np.round(np.random.uniform(10, 1000, num_products), 2),
    'release_date': release_dates
})
print(f"Generated {len(products_df)} products.")

# 1.3 reviews_df
num_reviews = np.random.randint(8000, 12001)

# Sample user_ids and product_ids
review_user_ids = np.random.choice(users_df['user_id'], num_reviews)
review_product_ids = np.random.choice(products_df['product_id'], num_reviews)

# Generate base review dates (will be adjusted)
base_review_dates = pd.to_datetime('today') - pd.to_timedelta(np.random.randint(0, 3 * 365, num_reviews), unit='days')

reviews_df = pd.DataFrame({
    'review_id': np.arange(num_reviews),
    'user_id': review_user_ids,
    'product_id': review_product_ids,
    'review_date': base_review_dates,
    'rating': np.random.randint(1, 6, num_reviews)
})

# Merge to get signup_date and release_date for review_date adjustment and helpfulness simulation
reviews_df = reviews_df.merge(users_df[['user_id', 'signup_date', 'reputation_score', 'account_tier']], on='user_id', how='left')
reviews_df = reviews_df.merge(products_df[['product_id', 'release_date', 'product_category']], on='product_id', how='left')

# Ensure review_date is after signup_date and release_date
reviews_df['review_date'] = np.maximum(
    reviews_df['review_date'],
    np.maximum(reviews_df['signup_date'], reviews_df['release_date']) + pd.to_timedelta(1, unit='days') # At least 1 day after
)

# Simulate review text
positive_phrases = [
    "Absolutely love this product!", "Highly recommend, worth every penny.", "Fantastic quality and design.",
    "Exceeded my expectations, truly impressed.", "A game-changer, so glad I bought it.",
    "Works perfectly, no issues at all.", "Great value for money, very happy.", "Smooth experience, brilliant performance."
]
negative_phrases = [
    "Very disappointed with this purchase.", "Not worth the price, poor quality.", "Broke after a few uses.",
    "Wouldn't recommend, don't waste your money.", "Completely failed to meet expectations.",
    "Terrible experience, very frustrating.", "Much lower quality than advertised.", "Regret buying this, useless."
]
neutral_phrases = [
    "It's an OK product, nothing special.", "Does the job, but could be better.", "Average performance for the price.",
    "Functions as described, no complaints.", "Neither good nor bad, just functional.",
    "Standard item, as expected.", "Could improve in some areas.", "It is what it is."
]

def generate_review_text(rating):
    num_sentences = np.random.randint(1, 5) # 1 to 4 sentences
    sentences = []
    for _ in range(num_sentences):
        if rating >= 4: # Positive sentiment
            sentences.append(random.choice(positive_phrases))
        elif rating <= 2: # Negative sentiment
            sentences.append(random.choice(negative_phrases))
        else: # Neutral sentiment
            sentences.append(random.choice(neutral_phrases))
    return " ".join(sentences) + "."

reviews_df['review_text'] = reviews_df['rating'].apply(generate_review_text)
reviews_df['review_length_chars'] = reviews_df['review_text'].str.len()

# Simulate is_helpful with biases
reviews_df['is_helpful_prob'] = 0.12 # Base helpfulness probability (overall 10-15%)

# Bias 1: Users with higher reputation_score or 'Gold' account_tier
reviews_df['is_helpful_prob'] += reviews_df['reputation_score'] / 200 # Max +0.5
reviews_df.loc[reviews_df['account_tier'] == 'Gold', 'is_helpful_prob'] += 0.1
reviews_df.loc[reviews_df['account_tier'] == 'Silver', 'is_helpful_prob'] += 0.05

# Bias 2: Longer review_text tends to be more helpful
reviews_df['is_helpful_prob'] += np.minimum(reviews_df['review_length_chars'] / 500, 0.2) # Max +0.2

# Bias 3: Reviews written closer to the release_date of the product tend to be more helpful
reviews_df['days_since_product_release_at_review'] = (reviews_df['review_date'] - reviews_df['release_date']).dt.days
reviews_df['is_helpful_prob'] -= np.minimum(reviews_df['days_since_product_release_at_review'] / 365 / 5, 0.1) # Max -0.1

# Bias 4: Reviews with extreme rating (1 or 5) might be less frequently helpful unless combined with long text
extreme_rating_short_text = (reviews_df['rating'].isin([1, 5])) & (reviews_df['review_length_chars'] < 100)
reviews_df.loc[extreme_rating_short_text, 'is_helpful_prob'] -= 0.05

# Clip probabilities to [0.01, 0.99] to avoid issues with np.random.rand
reviews_df['is_helpful_prob'] = np.clip(reviews_df['is_helpful_prob'], 0.01, 0.99) 

# Generate 'is_helpful' based on probabilities
reviews_df['is_helpful'] = (np.random.rand(len(reviews_df)) < reviews_df['is_helpful_prob']).astype(int)

# Drop temporary columns used for simulation, keeping only required ones for reviews_df
reviews_df = reviews_df[['review_id', 'user_id', 'product_id', 'review_date', 'rating', 'review_text', 'is_helpful']]

# Sort reviews_df by user_id then review_date
reviews_df = reviews_df.sort_values(by=['user_id', 'review_date']).reset_index(drop=True)
print(f"Generated {len(reviews_df)} reviews.")
print(f"Overall helpfulness rate: {reviews_df['is_helpful'].mean():.2%}")


# --- 2. Load into SQLite & SQL Feature Engineering (Review-Level Context) ---

print("\n--- Loading Data into SQLite and SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

users_df.to_sql('users', conn, index=False, if_exists='replace')
products_df.to_sql('products', conn, index=False, if_exists='replace')
reviews_df.to_sql('reviews', conn, index=False, if_exists='replace')

sql_query = """
WITH UserAgg AS (
    SELECT
        review_id,
        user_id,
        review_date,
        rating,
        LAG(review_date, 1) OVER (PARTITION BY user_id ORDER BY review_date) AS prev_user_review_date,
        COUNT(review_id) OVER (PARTITION BY user_id ORDER BY review_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_reviews_count,
        COALESCE(AVG(CAST(rating AS REAL)) OVER (PARTITION BY user_id ORDER BY review_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0.0) AS user_avg_prior_rating
    FROM reviews
),
ProductAgg AS (
    SELECT
        review_id,
        product_id,
        review_date,
        rating,
        MIN(review_date) OVER (PARTITION BY product_id) AS product_overall_first_review_date,
        COUNT(review_id) OVER (PARTITION BY product_id ORDER BY review_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS product_prior_reviews_count,
        COALESCE(AVG(CAST(rating AS REAL)) OVER (PARTITION BY product_id ORDER BY review_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0.0) AS product_avg_prior_rating
    FROM reviews
)
SELECT
    r.review_id,
    r.user_id,
    r.product_id,
    r.review_date,
    r.rating,
    r.review_text,
    r.is_helpful,
    u.reputation_score,
    u.account_tier,
    u.signup_date,
    p.product_category,
    p.price,
    p.release_date,

    ua.user_prior_reviews_count,
    ua.user_avg_prior_rating,
    CASE
        WHEN ua.prev_user_review_date IS NOT NULL THEN julianday(r.review_date) - julianday(ua.prev_user_review_date)
        ELSE julianday(r.review_date) - julianday(u.signup_date)
    END AS days_since_last_user_review,

    pa.product_prior_reviews_count,
    pa.product_avg_prior_rating,
    CASE
        WHEN pa.product_prior_reviews_count > 0 THEN julianday(pa.product_overall_first_review_date) - julianday(p.release_date)
        ELSE NULL
    END AS days_since_product_first_review
FROM reviews AS r
JOIN users AS u ON r.user_id = u.user_id
JOIN products AS p ON r.product_id = p.product_id
JOIN UserAgg AS ua ON r.review_id = ua.review_id
JOIN ProductAgg AS pa ON r.review_id = pa.review_id
ORDER BY r.user_id, r.review_date;
"""

review_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"SQL query executed and fetched {len(review_features_df)} rows with engineered features.")
print(f"Sample of SQL engineered features:\n{review_features_df[['review_id', 'user_id', 'product_id', 'review_date', 'user_prior_reviews_count', 'user_avg_prior_rating', 'days_since_last_user_review', 'product_prior_reviews_count', 'product_avg_prior_rating', 'days_since_product_first_review']].head()}")


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n--- Pandas Feature Engineering & Binary Target Creation ---")

# Convert date columns to datetime objects
date_cols = ['signup_date', 'release_date', 'review_date']
for col in date_cols:
    review_features_df[col] = pd.to_datetime(review_features_df[col])

# Calculate additional Pandas features
review_features_df['user_account_age_at_review_days'] = (review_features_df['review_date'] - review_features_df['signup_date']).dt.days
review_features_df['days_since_product_release_at_review'] = (review_features_df['review_date'] - review_features_df['release_date']).dt.days
review_features_df['review_length_chars'] = review_features_df['review_text'].str.len()

# Handle `days_since_product_first_review` NaNs
# As per SQL, this is NULL if `product_prior_reviews_count` is 0 (i.e., current review is the product's first).
# In this case, fill it with `days_since_product_release_at_review`.
# Otherwise (if other NaNs somehow exist), fill with a large sentinel (9999).
review_features_df['days_since_product_first_review'] = review_features_df.apply(
    lambda row: row['days_since_product_release_at_review']
    if pd.isna(row['days_since_product_first_review']) and row['product_prior_reviews_count'] == 0
    else row['days_since_product_first_review'], axis=1
)
review_features_df['days_since_product_first_review'] = review_features_df['days_since_product_first_review'].fillna(9999)


# Calculate rating_deviation_from_product_mean
global_mean_rating = review_features_df['rating'].mean()
review_features_df['rating_deviation_from_product_mean'] = review_features_df.apply(
    lambda row: row['rating'] - row['product_avg_prior_rating'] if row['product_avg_prior_rating'] != 0 else row['rating'] - global_mean_rating, axis=1
)

# Define features X and target y
numerical_features = [
    'rating', 'price', 'reputation_score',
    'user_prior_reviews_count', 'user_avg_prior_rating', 'days_since_last_user_review',
    'product_prior_reviews_count', 'product_avg_prior_rating', 'days_since_product_first_review',
    'user_account_age_at_review_days', 'days_since_product_release_at_review', 'review_length_chars',
    'rating_deviation_from_product_mean'
]
categorical_features = ['account_tier', 'product_category']

X = review_features_df[numerical_features + categorical_features]
y = review_features_df['is_helpful']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"Class distribution in y_train:\n{y_train.value_counts(normalize=True)}")
print(f"Class distribution in y_test:\n{y_test.value_counts(normalize=True)}")


# --- 4. Data Visualization ---

print("\n--- Data Visualization ---")

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(14, 6))

# Plot 1: Violin plot of review_length_chars for helpful vs. unhelpful reviews
plt.subplot(1, 2, 1)
sns.violinplot(x='is_helpful', y='review_length_chars', data=review_features_df, palette='viridis')
plt.title('Review Length Characters Distribution by Helpfulness')
plt.xlabel('Is Helpful (0: No, 1: Yes)')
plt.ylabel('Review Length (Characters)')

# Plot 2: Stacked bar chart of helpfulness proportion across product_category
plt.subplot(1, 2, 2)
crosstab_norm = pd.crosstab(review_features_df['product_category'], review_features_df['is_helpful'], normalize='index')
crosstab_norm.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='coolwarm')
plt.title('Proportion of Helpful Reviews by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Is Helpful', labels=['No', 'Yes'])

plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

print("\n--- ML Pipeline & Evaluation ---")

# Create preprocessing pipelines for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
print("Training the HistGradientBoostingClassifier model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict probabilities for the positive class (class 1) on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Predict classes for the classification report
y_pred = model_pipeline.predict(X_test)

# Calculate and print evaluation metrics
print("\n--- Model Evaluation on Test Set ---")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n--- Script Finished ---")