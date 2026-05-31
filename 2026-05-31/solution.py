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
from sklearn.metrics import roc_auc_score, classification_report
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn') # For potential feature name warnings

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

print("--- 1. Generating Synthetic Data ---")

np.random.seed(42)

# Authors DataFrame
num_authors = np.random.randint(200, 301)
authors_df = pd.DataFrame({
    'author_id': range(1, num_authors + 1),
    'author_tier': np.random.choice(['Junior', 'Mid', 'Senior'], size=num_authors, p=[0.3, 0.5, 0.2]),
    'past_avg_article_engagement': np.random.uniform(0, 100, size=num_authors)
})

# Articles DataFrame
num_articles = np.random.randint(1000, 1501)
start_date = datetime.now() - timedelta(days=3 * 365) # Last 3 years
end_date = datetime.now() - timedelta(days=30) # End a month ago to allow for event generation
time_range_seconds = int((end_date - start_date).total_seconds())

articles_df = pd.DataFrame({
    'article_id': range(1, num_articles + 1),
    'author_id': np.random.choice(authors_df['author_id'], size=num_articles),
    'publish_date': [start_date + timedelta(seconds=np.random.randint(time_range_seconds)) for _ in range(num_articles)],
    'category': np.random.choice(['Politics', 'Tech', 'Sports', 'Lifestyle', 'Finance', 'Science'], size=num_articles),
    'sentiment_score': np.random.uniform(0.1, 0.9, size=num_articles),
    'word_count': np.random.randint(200, 1501, size=num_articles),
})

# Map author tier to a numerical value for correlation
tier_mapping = {'Junior': 1, 'Mid': 2, 'Senior': 3}
articles_with_author_info = articles_df.merge(authors_df[['author_id', 'author_tier', 'past_avg_article_engagement']], on='author_id', how='left')
articles_with_author_info['author_tier_numeric'] = articles_with_author_info['author_tier'].map(tier_mapping)

# Simulate _actual_48h_engagement_score with correlations
# Base score
articles_with_author_info['_actual_48h_engagement_score'] = (
    50 * articles_with_author_info['sentiment_score'] +
    0.05 * articles_with_author_info['word_count'] +
    10 * articles_with_author_info['author_tier_numeric'] +
    0.5 * articles_with_author_info['past_avg_article_engagement'] +
    np.random.normal(0, 20, num_articles) # Noise
)
# Ensure scores are non-negative and within a reasonable range
articles_with_author_info['_actual_48h_engagement_score'] = articles_with_author_info['_actual_48h_engagement_score'].clip(lower=0, upper=1000)

# Merge back into original articles_df for target creation and remove temporary columns
articles_df = articles_df.merge(
    articles_with_author_info[['article_id', '_actual_48h_engagement_score']], 
    on='article_id', how='left'
)
# Now articles_df has the actual engagement score. The author info will be joined again later in SQL.


# Engagement Events DataFrame
num_engagement_events = np.random.randint(20000, 30001)
event_types = ['view', 'like', 'share', 'comment']
engagement_events_data = []

# Identify articles that are likely to have high engagement for event generation bias
high_engagement_threshold_for_event_bias = articles_df['_actual_48h_engagement_score'].quantile(0.7)
high_engagement_article_ids = articles_df[articles_df['_actual_48h_engagement_score'] >= high_engagement_threshold_for_event_bias]['article_id'].tolist()

# Ensure enough events are generated to meet the lower bound of num_engagement_events
# Generate events such that more occur in the first 6 hours and for high engagement articles
target_total_events = num_engagement_events * 1.5 # Generate more events initially and then sample down
generated_events_count = 0

while generated_events_count < target_total_events:
    article_row = articles_df.sample(n=1).iloc[0]
    article_id = article_row['article_id']
    publish_date = article_row['publish_date']
    
    # Base number of events per article, higher for high engagement articles
    if article_id in high_engagement_article_ids:
        num_events_for_article = np.random.randint(15, 40)
        event_type_probs = [0.5, 0.25, 0.15, 0.1] # More likes/shares/comments
    else:
        num_events_for_article = np.random.randint(8, 25)
        event_type_probs = [0.7, 0.15, 0.1, 0.05] # More views
        
    for _ in range(num_events_for_article):
        # 70% chance event is in first 6 hours, 30% in remaining 42 hours
        if np.random.rand() < 0.7: 
            event_offset_seconds = np.random.randint(1, int(timedelta(hours=6).total_seconds()))
        else:
            event_offset_seconds = np.random.randint(int(timedelta(hours=6).total_seconds()), int(timedelta(hours=48).total_seconds()))
        
        event_timestamp = publish_date + timedelta(seconds=event_offset_seconds)
        
        engagement_events_data.append({
            'event_id': len(engagement_events_data) + 1,
            'article_id': article_id,
            'event_timestamp': event_timestamp,
            'event_type': np.random.choice(event_types, p=event_type_probs)
        })
    generated_events_count = len(engagement_events_data)

engagement_events_df = pd.DataFrame(engagement_events_data)
# Sample down to the desired number of events if too many were generated
if len(engagement_events_df) > num_engagement_events:
    engagement_events_df = engagement_events_df.sample(n=num_engagement_events, random_state=42).reset_index(drop=True)

# Sort engagement_events_df
engagement_events_df = engagement_events_df.sort_values(by=['article_id', 'event_timestamp']).reset_index(drop=True)

print(f"Generated {len(articles_df)} articles, {len(authors_df)} authors, {len(engagement_events_df)} engagement events.")

# --- 2. Load into SQLite & SQL Feature Engineering ---

print("\n--- 2. Loading Data to SQLite & SQL Feature Engineering ---")

conn = sqlite3.connect(':memory:')

# Load DataFrames into SQLite tables
# Removed datetime_format argument as it's not supported by to_sql for SQLite and caused TypeError
articles_df.to_sql('articles', conn, index=False, if_exists='replace')
authors_df.to_sql('authors', conn, index=False, if_exists='replace')
engagement_events_df.to_sql('engagement_events', conn, index=False, if_exists='replace')

# Determine GLOBAL_PREDICTION_CUTOFF_DATE
latest_event_ts_str = pd.read_sql_query("SELECT MAX(event_timestamp) FROM engagement_events", conn).iloc[0, 0]
latest_event_ts = datetime.strptime(latest_event_ts_str, '%Y-%m-%d %H:%M:%S')

GLOBAL_PREDICTION_CUTOFF_DATE = latest_event_ts - timedelta(weeks=4)
print(f"Latest event timestamp: {latest_event_ts}")
print(f"Global prediction cutoff date: {GLOBAL_PREDICTION_CUTOFF_DATE}")

# SQL Query for Feature Engineering
# Using julianday for date arithmetic and comparison
sql_query = f"""
WITH LatestOverallEvent AS (
    SELECT MAX(event_timestamp) AS max_event_ts FROM engagement_events
),
ArticleBase AS (
    SELECT
        a.article_id,
        a.publish_date,
        a.category,
        a.sentiment_score,
        a.word_count,
        a._actual_48h_engagement_score,
        au.author_tier,
        au.past_avg_article_engagement
    FROM articles AS a
    JOIN authors AS au ON a.author_id = au.author_id
),
Engagement6h AS (
    SELECT
        ee.article_id,
        SUM(CASE WHEN ee.event_type = 'view' THEN 1 ELSE 0 END) AS num_views_first_6h,
        SUM(CASE WHEN ee.event_type = 'like' THEN 1 ELSE 0 END) AS num_likes_first_6h,
        SUM(CASE WHEN ee.event_type = 'share' THEN 1 ELSE 0 END) AS num_shares_first_6h,
        SUM(CASE WHEN ee.event_type = 'comment' THEN 1 ELSE 0 END) AS num_comments_first_6h
    FROM engagement_events AS ee
    JOIN articles AS a ON ee.article_id = a.article_id
    WHERE julianday(ee.event_timestamp) >= julianday(a.publish_date)
      AND julianday(ee.event_timestamp) <= julianday(a.publish_date, '+6 hours')
    GROUP BY ee.article_id
)
SELECT
    ab.article_id,
    ab.publish_date,
    ab.category,
    ab.sentiment_score,
    ab.word_count,
    ab._actual_48h_engagement_score,
    ab.author_tier,
    ab.past_avg_article_engagement,
    COALESCE(e6.num_views_first_6h, 0) AS num_views_first_6h,
    COALESCE(e6.num_likes_first_6h, 0) AS num_likes_first_6h,
    COALESCE(e6.num_shares_first_6h, 0) AS num_shares_first_6h,
    COALESCE(e6.num_comments_first_6h, 0) AS num_comments_first_6h,
    CAST(strftime('%w', ab.publish_date) AS INTEGER) AS publish_day_of_week, -- 0 for Sunday, 6 for Saturday
    CAST(strftime('%H', ab.publish_date) AS INTEGER) AS publish_hour_of_day
FROM ArticleBase AS ab
LEFT JOIN Engagement6h AS e6 ON ab.article_id = e6.article_id
CROSS JOIN LatestOverallEvent AS loe
WHERE
    julianday(ab.publish_date) <= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}')
AND
    julianday(ab.publish_date, '+6 hours') <= julianday('{GLOBAL_PREDICTION_CUTOFF_DATE}')
AND
    julianday(ab.publish_date, '+48 hours') <= julianday(loe.max_event_ts);
"""

article_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Extracted features for {len(article_features_df)} articles after applying time window filters.")
if not article_features_df.empty:
    print("Sample of features after SQL engineering:")
    print(article_features_df.head())
else:
    print("No articles found after SQL feature engineering. Check data generation and time window filters.")
    exit() # Exit if no data to process


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

print("\n--- 3. Pandas Feature Engineering & Binary Target Creation ---")

# Convert publish_date to datetime objects
article_features_df['publish_date'] = pd.to_datetime(article_features_df['publish_date'])

# Handle NaN values
# Fill numerical aggregated features with 0 (already handled by COALESCE in SQL, but good for robustness)
for col in ['num_views_first_6h', 'num_likes_first_6h', 'num_shares_first_6h', 'num_comments_first_6h']:
    article_features_df[col] = article_features_df[col].fillna(0)

# Fill sentiment_score with its mean, word_count with its median
article_features_df['sentiment_score'] = article_features_df['sentiment_score'].fillna(article_features_df['sentiment_score'].mean())
article_features_df['word_count'] = article_features_df['word_count'].fillna(article_features_df['word_count'].median())
# Fill past_avg_article_engagement for authors not found (shouldn't happen with inner join but good practice)
article_features_df['past_avg_article_engagement'] = article_features_df['past_avg_article_engagement'].fillna(0)


# Calculate engagement_rate_first_6h
article_features_df['engagement_rate_first_6h'] = (
    article_features_df['num_likes_first_6h'] +
    article_features_df['num_shares_first_6h'] +
    article_features_df['num_comments_first_6h']
) / (article_features_df['num_views_first_6h'] + 1e-6) # Add epsilon to avoid division by zero

# Fill any NaN or inf resulting from division with 0
article_features_df['engagement_rate_first_6h'] = article_features_df['engagement_rate_first_6h'].replace([np.inf, -np.inf], 0).fillna(0)


# Create the Binary Target 'will_be_high_engagement'
# Adjust threshold dynamically to ensure reasonable class balance (e.g., 15-25% positive class)
target_percentile = 0.80
max_iterations = 20 # Prevent infinite loops
for i in range(max_iterations):
    engagement_score_threshold = article_features_df['_actual_48h_engagement_score'].quantile(target_percentile)
    article_features_df['will_be_high_engagement'] = (article_features_df['_actual_48h_engagement_score'] > engagement_score_threshold).astype(int)
    positive_class_ratio = article_features_df['will_be_high_engagement'].mean()
    if 0.15 <= positive_class_ratio <= 0.25:
        break
    elif positive_class_ratio < 0.15:
        target_percentile = max(0.05, target_percentile - 0.01) # Lower percentile to get more positives
    else: # positive_class_ratio > 0.25
        target_percentile = min(0.95, target_percentile + 0.01) # Raise percentile to get fewer positives
    
    if i == max_iterations - 1:
        print("Warning: Could not achieve desired class balance within iteration limits.")

print(f"High engagement threshold: > {engagement_score_threshold:.2f} (Target percentile: {target_percentile*100:.0f}th)")
print(f"Positive class ratio for 'will_be_high_engagement': {positive_class_ratio:.2f}")


# Define features X and target y
numerical_features = [
    'sentiment_score', 'word_count', 'past_avg_article_engagement',
    'num_views_first_6h', 'num_likes_first_6h', 'num_shares_first_6h', 'num_comments_first_6h',
    'engagement_rate_first_6h', 'publish_day_of_week', 'publish_hour_of_day'
]
categorical_features = ['category', 'author_tier']

# Ensure all features exist in the dataframe before splitting
all_features = numerical_features + categorical_features
for f in all_features:
    if f not in article_features_df.columns:
        print(f"Error: Feature '{f}' not found in the dataframe. Exiting.")
        exit()

X = article_features_df[all_features]
y = article_features_df['will_be_high_engagement']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")
print("Sample of processed features (X_train head):")
print(X_train.head())

# --- 4. Data Visualization (Matplotlib/Seaborn) ---

print("\n--- 4. Generating Data Visualizations ---")

plt.figure(figsize=(14, 6))

# Plot 1: Violin plot of engagement_rate_first_6h vs. will_be_high_engagement
plt.subplot(1, 2, 1)
sns.violinplot(x='will_be_high_engagement', y='engagement_rate_first_6h', data=article_features_df)
plt.title('Engagement Rate (First 6h) by High Engagement Status')
plt.xlabel('High Engagement (0=No, 1=Yes)')
plt.ylabel('Engagement Rate (First 6h)')
plt.xticks([0, 1], ['Not High Engagement', 'High Engagement'])


# Plot 2: Stacked bar chart of category proportions
plt.subplot(1, 2, 2)
# Ensure categories are not empty
if not article_features_df['category'].empty:
    category_engagement_counts = article_features_df.groupby('category')['will_be_high_engagement'].value_counts(normalize=True).unstack().fillna(0)
    category_engagement_counts.plot(kind='bar', stacked=True, ax=plt.gca(), color=['lightcoral', 'lightseagreen'])
    plt.title('Proportion of High Engagement by Category')
    plt.xlabel('Category')
    plt.ylabel('Proportion')
    plt.legend(title='High Engagement', labels=['No', 'Yes'])
else:
    plt.text(0.5, 0.5, "No categories to display", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title('Proportion of High Engagement by Category (No data)')

plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

print("\n--- 5. Building & Evaluating ML Pipeline ---")

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')), # Impute before OHE
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create the ML Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42, class_weight='balanced'))
])

# Train the pipeline
print("Training the model...")
pipeline.fit(X_train, y_train)
print("Model training complete.")

# Predict probabilities for the positive class on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
# Predict hard labels for the classification report
y_pred = pipeline.predict(X_test) 

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

print("\n--- Pipeline Execution Complete ---")