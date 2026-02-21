import pandas as pd
import numpy as np
import datetime
import sqlite3
from collections import defaultdict
import random

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Generate Synthetic Data ---
print("1. Generating Synthetic Data...")

# Users DataFrame
num_users = np.random.randint(500, 701)
user_ids = np.arange(num_users)
signup_dates = pd.to_datetime('2018-01-01') + pd.to_timedelta(np.random.randint(0, 5*365, num_users), unit='D')

users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': signup_dates,
    'age': np.random.randint(18, 71, num_users),
    'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_users),
    'subscription_tier': np.random.choice(['Free', 'Basic', 'Premium'], num_users, p=[0.5, 0.3, 0.2])
})

# Interactions DataFrame
num_interactions = np.random.randint(5000, 8001)
sampled_user_ids = np.random.choice(users_df['user_id'], num_interactions, replace=True)

# Map user_id to signup_date for efficient date generation
user_signup_map = users_df.set_index('user_id')['signup_date'].to_dict()

interactions_data = []
for i in range(num_interactions):
    user_id = sampled_user_ids[i]
    signup_date = user_signup_map[user_id]
    # Ensure interaction_date is after signup_date
    days_after_signup = np.random.randint(1, 3 * 365 + 1) # Interactions up to 3 years after signup
    interaction_date = signup_date + pd.to_timedelta(days_after_signup, unit='D')
    interactions_data.append({
        'interaction_id': i,
        'user_id': user_id,
        'interaction_date': interaction_date,
        'channel': np.random.choice(['Chat', 'Email', 'Survey', 'Social_Media', 'Phone']),
    })
interactions_df = pd.DataFrame(interactions_data)

# Text and Sentiment generation
positive_words = ['excellent', 'happy', 'resolved', 'great', 'fast', 'smooth', 'love', 'fantastic', 'easy', 'pleasure', 'fixed', 'recommend']
negative_words = ['frustrated', 'issue', 'slow', 'bad', 'problem', 'unhappy', 'difficult', 'broken', 'disappointed', 'wait', 'terrible']
neutral_words = ['ok', 'question', 'feedback', 'inquiry', 'regarding', 'information', 'check', 'update', 'status', 'hello', 'request']

# Simulate realistic sentiment patterns for users
# Assign a predominant sentiment to a subset of users
biased_user_fraction = 0.3 # 30% of users have a sentiment bias
biased_user_ids = np.random.choice(users_df['user_id'], int(num_users * biased_user_fraction), replace=False)
user_sentiment_bias = {uid: np.random.choice(['Positive', 'Negative'], p=[0.6, 0.4]) for uid in biased_user_ids}

sentiment_labels = []
interaction_texts = []

for idx, row in interactions_df.iterrows():
    user_id = row['user_id']
    
    # Determine sentiment based on user bias or randomly
    if user_id in user_sentiment_bias:
        bias_type = user_sentiment_bias[user_id]
        if bias_type == 'Positive':
            sentiment = np.random.choice(['Positive', 'Neutral', 'Negative'], p=[0.7, 0.2, 0.1])
        else: # Negative bias
            sentiment = np.random.choice(['Negative', 'Neutral', 'Positive'], p=[0.7, 0.2, 0.1])
    else:
        sentiment = np.random.choice(['Positive', 'Neutral', 'Negative'], p=[0.4, 0.3, 0.3]) # Default probabilities
    
    sentiment_labels.append(sentiment)
    
    # Generate text based on sentiment
    text_parts = []
    if sentiment == 'Positive':
        text_parts.extend(random.sample(positive_words, k=np.random.randint(1, 3)))
        if np.random.rand() < 0.5: text_parts.append(random.choice(neutral_words))
    elif sentiment == 'Negative':
        text_parts.extend(random.sample(negative_words, k=np.random.randint(1, 3)))
        if np.random.rand() < 0.5: text_parts.append(random.choice(neutral_words))
    else: # Neutral
        text_parts.extend(random.sample(neutral_words, k=np.random.randint(1, 3)))
        if np.random.rand() < 0.3: text_parts.append(random.choice(positive_words + negative_words + neutral_words))
        
    random.shuffle(text_parts)
    interaction_texts.append(" ".join(text_parts).capitalize() + ".")

interactions_df['sentiment_label'] = sentiment_labels
interactions_df['interaction_text'] = interaction_texts

# Sort interactions_df by user_id then interaction_date for sequential processing in SQL
interactions_df = interactions_df.sort_values(by=['user_id', 'interaction_date']).reset_index(drop=True)

print(f"Generated {len(users_df)} users and {len(interactions_df)} interactions.")
print("Users head:")
print(users_df.head())
print("\nInteractions head:")
print(interactions_df.head())

# --- 2. Load into SQLite & SQL Feature Engineering ---
print("\n2. Loading into SQLite & SQL Feature Engineering (Prior User Sentiment History)...")

conn = sqlite3.connect(':memory:')
users_df.to_sql('users', conn, index=False, if_exists='replace')
interactions_df.to_sql('interactions', conn, index=False, if_exists='replace')

sql_query = """
WITH RankedInteractions AS (
    SELECT
        i.interaction_id,
        i.user_id,
        i.interaction_date,
        i.sentiment_label,
        i.channel,
        i.interaction_text,
        u.signup_date,
        u.age,
        u.region,
        u.subscription_tier,
        -- Calculate previous sentiment aggregates
        SUM(CASE WHEN i.sentiment_label = 'Positive' THEN 1 ELSE 0 END) OVER (PARTITION BY i.user_id ORDER BY i.interaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_positive_interactions,
        SUM(CASE WHEN i.sentiment_label = 'Negative' THEN 1 ELSE 0 END) OVER (PARTITION BY i.user_id ORDER BY i.interaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_negative_interactions,
        COUNT(i.interaction_id) OVER (PARTITION BY i.user_id ORDER BY i.interaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_prior_total_interactions,
        -- Get previous interaction date for days since last
        LAG(i.interaction_date, 1) OVER (PARTITION BY i.user_id ORDER BY i.interaction_date) AS prev_interaction_date
    FROM
        interactions i
    JOIN
        users u ON i.user_id = u.user_id
)
SELECT
    interaction_id,
    user_id,
    interaction_date,
    channel,
    interaction_text,
    sentiment_label,
    age,
    region,
    subscription_tier,
    signup_date,
    COALESCE(user_prior_total_interactions, 0) AS user_prior_total_interactions,
    COALESCE(user_prior_positive_interactions, 0) AS user_prior_positive_interactions,
    COALESCE(user_prior_negative_interactions, 0) AS user_prior_negative_interactions,
    (CAST(COALESCE(user_prior_positive_interactions, 0) AS REAL) + 1.0) / (CAST(COALESCE(user_prior_negative_interactions, 0) AS REAL) + 1.0) AS user_prior_sentiment_ratio_pos_neg,
    -- Calculate days since last user interaction or signup_date for first interaction
    CAST(JULIANDAY(interaction_date) - JULIANDAY(COALESCE(prev_interaction_date, signup_date)) AS INTEGER) AS days_since_last_user_interaction
FROM
    RankedInteractions
ORDER BY
    user_id, interaction_date
;
"""

interaction_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("SQL Feature Engineering complete. Resulting DataFrame head:")
print(interaction_features_df.head())
print(f"DataFrame shape: {interaction_features_df.shape}")
print(interaction_features_df.info())

# --- 3. Pandas Feature Engineering & Multi-Class Target Creation ---
print("\n3. Pandas Feature Engineering & Multi-Class Target Creation...")

# Ensure correct types and handle any remaining NaNs after SQL
# `user_prior_total_interactions`, `user_prior_positive_interactions`, `user_prior_negative_interactions` should be filled with 0 for first interactions
interaction_features_df['user_prior_total_interactions'] = interaction_features_df['user_prior_total_interactions'].fillna(0).astype(int)
interaction_features_df['user_prior_positive_interactions'] = interaction_features_df['user_prior_positive_interactions'].fillna(0).astype(int)
interaction_features_df['user_prior_negative_interactions'] = interaction_features_df['user_prior_negative_interactions'].fillna(0).astype(int)
# `user_prior_sentiment_ratio_pos_neg` should be filled with 1.0 (Laplace smoothed 0+1 / 0+1) for first interactions
interaction_features_df['user_prior_sentiment_ratio_pos_neg'] = interaction_features_df['user_prior_sentiment_ratio_pos_neg'].fillna(1.0)
# `days_since_last_user_interaction` should be handled by SQL's COALESCE. If any NaNs remain (edge cases), fill with a large sentinel.
interaction_features_df['days_since_last_user_interaction'] = interaction_features_df['days_since_last_user_interaction'].fillna(9999).astype(int)

# Convert date columns to datetime objects
interaction_features_df['signup_date'] = pd.to_datetime(interaction_features_df['signup_date'])
interaction_features_df['interaction_date'] = pd.to_datetime(interaction_features_df['interaction_date'])

# Calculate user_account_age_at_interaction_days
interaction_features_df['user_account_age_at_interaction_days'] = (
    interaction_features_df['interaction_date'] - interaction_features_df['signup_date']
).dt.days

# Extract Text Features using TfidfVectorizer
print("  Applying TfidfVectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_features=500)
# Note: tfidf_vectorizer will be fitted on X_train only, then transformed on X_test.
# Here we just initialize. The actual fit/transform is done after train/test split.

# Define features (X) and target (y)
# X_for_split will include 'interaction_text' which will be handled separately later
X_for_split = interaction_features_df.drop(columns=[
    'interaction_id', 'user_id', 'sentiment_label', 'signup_date', 'interaction_date'
])
y = interaction_features_df['sentiment_label']

# Split data into training and testing sets
print("  Splitting data into training and testing sets (70/30, stratified by sentiment_label)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_for_split, y, test_size=0.3, random_state=42, stratify=y
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- 4. Data Visualization ---
print("\n4. Generating Data Visualizations...")

plt.figure(figsize=(15, 6))

# Plot 1: Stacked bar chart of sentiment_label across different channel values
plt.subplot(1, 2, 1)
sentiment_by_channel = interaction_features_df.groupby(['channel', 'sentiment_label']).size().unstack(fill_value=0)
sentiment_by_channel.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Sentiment Distribution by Channel')
plt.xlabel('Channel')
plt.ylabel('Number of Interactions')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment')
plt.tight_layout()

# Plot 2: Violin plot of user_prior_sentiment_ratio_pos_neg for each sentiment_label
plt.subplot(1, 2, 2)
# Add a small constant to ratio to handle log(0) if any unhandled cases exist
sns.violinplot(x='sentiment_label', y='user_prior_sentiment_ratio_pos_neg', data=interaction_features_df, palette='muted', ax=plt.gca())
plt.title('Prior Sentiment Ratio by Current Interaction Sentiment')
plt.xlabel('Current Interaction Sentiment')
plt.ylabel('Prior Positive/Negative Sentiment Ratio (Log Scale)')
plt.yscale('log') # Log scale for better visualization of skewed ratio
plt.tight_layout()
plt.show()

# --- 5. ML Pipeline & Evaluation (Multi-Class Classification) ---
print("\n5. Building ML Pipeline & Evaluating Model...")

# Define numerical and categorical features for the ColumnTransformer
# 'interaction_text' is excluded from these lists as it's processed separately by TF-IDF
numerical_features = [
    'age', 'user_account_age_at_interaction_days',
    'user_prior_total_interactions', 'user_prior_positive_interactions',
    'user_prior_negative_interactions', 'user_prior_sentiment_ratio_pos_neg',
    'days_since_last_user_interaction'
]
categorical_features = ['region', 'subscription_tier', 'channel']

# Create a ColumnTransformer for preprocessing structured features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ],
    remainder='drop' # Drop any other columns (like 'interaction_text' which is processed separately)
)

# Process text features using TfidfVectorizer
print("  Fitting and transforming TF-IDF features...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['interaction_text'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['interaction_text'])

# Process structured features using the preprocessor
print("  Fitting and transforming structured features...")
# Drop 'interaction_text' from X_train/X_test before passing to preprocessor
X_train_structured = X_train.drop(columns=['interaction_text'])
X_test_structured = X_test.drop(columns=['interaction_text'])

X_train_structured_processed = preprocessor.fit_transform(X_train_structured)
X_test_structured_processed = preprocessor.transform(X_test_structured)

# Combine processed structured features with TF-IDF features
print("  Combining structured and TF-IDF features...")
X_train_processed = hstack([X_train_structured_processed, X_train_tfidf])
X_test_processed = hstack([X_test_structured_processed, X_test_tfidf])

print(f"Combined X_train_processed shape: {X_train_processed.shape}")
print(f"Combined X_test_processed shape: {X_test_processed.shape}")

# Initialize the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the classifier
print("  Training RandomForestClassifier...")
classifier.fit(X_train_processed, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test_processed)

# Evaluate the model
print("\nModel Evaluation:")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n--- Script Finished ---")