import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# --- 1. Generate Synthetic Data ---

np.random.seed(42)

# Parameters for data generation
N_posts = np.random.randint(1000, 1501)
N_interactions = np.random.randint(30000, 50001)
N_users = 200 # For post creators
N_interacting_users = 1000 # For unique_users_first_24h metric
viral_post_ratio = 0.08 # Approximately 5-10% of posts are 'viral'
post_categories = ['News', 'Humor', 'DIY', 'Tech', 'Fashion', 'Sports', 'Food', 'Travel']

# Generate posts_df
posts_data = {
    'post_id': np.arange(N_posts),
    'user_id': np.random.randint(1, N_users + 1, N_posts),
    'post_date': pd.to_datetime(pd.Timestamp.now() - pd.to_timedelta(np.random.randint(0, 730, N_posts), unit='D')),
    'category': np.random.choice(post_categories, N_posts),
    'num_hashtags': np.random.randint(0, 11, N_posts),
    'sentiment_score': np.random.uniform(-1.0, 1.0, N_posts),
    'user_follower_count': np.random.randint(100, 100001, N_posts)
}
posts_df = pd.DataFrame(posts_data)
posts_df = posts_df.sort_values(by='post_date').reset_index(drop=True) # Ensure post_date is somewhat ordered

# Simulate 'viral' posts characteristics
viral_post_ids = np.random.choice(posts_df['post_id'], int(N_posts * viral_post_ratio), replace=False)
posts_df['is_viral_candidate'] = posts_df['post_id'].isin(viral_post_ids)

# Generate interactions_df
interactions_list = []
interaction_types = ['like', 'comment', 'share', 'view']
interaction_type_weights_base = np.array([0.5, 0.2, 0.1, 0.2]) # Base probabilities for like, comment, share, view

current_interaction_id = 0
for _, post in posts_df.iterrows():
    post_id = post['post_id']
    post_date = post['post_date']
    user_follower_count = post['user_follower_count']
    sentiment_score = post['sentiment_score']
    is_viral_candidate = post['is_viral_candidate']

    # Base number of interactions, influenced by follower count
    base_interactions_count = np.random.poisson(20 + user_follower_count / 5000)
    
    # Apply viral boost
    if is_viral_candidate:
        base_interactions_count = int(base_interactions_count * np.random.uniform(3, 8)) # 3-8x more interactions
        
    num_interactions_for_post = max(1, base_interactions_count) # At least one interaction

    # Generate interactions for this post
    for _ in range(num_interactions_for_post):
        # Initial timestamp, decaying exponentially
        interaction_timestamp = post_date + pd.to_timedelta(np.random.exponential(scale=3*24*3600), unit='s') 

        # For viral candidates, concentrate interactions heavily in the first 7 days
        if is_viral_candidate and interaction_timestamp > post_date + pd.Timedelta(7, 'D'):
            # If it's a viral candidate and interaction is too late, generate within 7 days
            interaction_timestamp = post_date + pd.to_timedelta(np.random.uniform(0, 7*24*3600), unit='s')

        # Ensure interaction_timestamp is strictly after post_date
        while interaction_timestamp <= post_date:
             interaction_timestamp = post_date + pd.to_timedelta(np.random.uniform(0, 1), unit='s') # Smallest positive delta

        # Adjust interaction type probabilities based on sentiment and virality
        type_weights = np.copy(interaction_type_weights_base)
        
        # Sentiment score -> likes
        if sentiment_score > 0.5:
            type_weights[0] += 0.2 
        elif sentiment_score < -0.5:
            type_weights[0] -= 0.1 

        # Viral candidates -> more shares/comments
        if is_viral_candidate:
            type_weights[1] += 0.15 # Comments
            type_weights[2] += 0.15 # Shares
        
        type_weights = np.maximum(0, type_weights) # Ensure no negative weights
        type_weights /= type_weights.sum() # Normalize weights

        interaction_type = np.random.choice(interaction_types, p=type_weights)
        
        interactions_list.append({
            'interaction_id': current_interaction_id,
            'post_id': post_id,
            'interacting_user_id': np.random.randint(1, N_interacting_users + 1), # User who interacted
            'interaction_timestamp': interaction_timestamp,
            'interaction_type': interaction_type
        })
        current_interaction_id += 1

interactions_df = pd.DataFrame(interactions_list)
# Adjust interactions_df size to be within the desired range, if generation resulted in more.
if len(interactions_df) > N_interactions:
    interactions_df = interactions_df.sample(N_interactions, random_state=42).reset_index(drop=True)

interactions_df = interactions_df.sort_values(by=['post_id', 'interaction_timestamp']).reset_index(drop=True)
print(f"Generated {len(posts_df)} posts and {len(interactions_df)} interactions.")


# --- 2. Load into SQLite & SQL Feature Engineering ---

conn = sqlite3.connect(':memory:')
posts_df.drop(columns=['is_viral_candidate']).to_sql('posts', conn, index=False) # Drop temp column
interactions_df.to_sql('interactions', conn, index=False)

early_engagement_window_hours = 24

sql_query = f"""
SELECT
    p.post_id,
    p.post_date,
    p.category,
    p.num_hashtags,
    p.sentiment_score,
    p.user_follower_count,
    COALESCE(SUM(CASE WHEN i.interaction_type = 'like' THEN 1 ELSE 0 END), 0) AS num_likes_first_24h,
    COALESCE(SUM(CASE WHEN i.interaction_type = 'comment' THEN 1 ELSE 0 END), 0) AS num_comments_first_24h,
    COALESCE(SUM(CASE WHEN i.interaction_type = 'share' THEN 1 ELSE 0 END), 0) AS num_shares_first_24h,
    COALESCE(COUNT(i.interaction_id), 0) AS total_interactions_first_24h,
    COALESCE(COUNT(DISTINCT i.interacting_user_id), 0) AS unique_users_first_24h
FROM
    posts p
LEFT JOIN
    interactions i ON p.post_id = i.post_id
    AND julianday(i.interaction_timestamp) BETWEEN julianday(p.post_date) AND julianday(p.post_date) + ({early_engagement_window_hours} / 24.0)
GROUP BY
    p.post_id, p.post_date, p.category, p.num_hashtags, p.sentiment_score, p.user_follower_count
ORDER BY
    p.post_id;
"""

post_early_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("\n--- Early Engagement Features (First 5 Rows) ---")
print(post_early_features_df.head())
print(f"Shape of early features DataFrame: {post_early_features_df.shape}")


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Convert post_date to datetime objects
post_early_features_df['post_date'] = pd.to_datetime(post_early_features_df['post_date'])

# Handle potential NaN values from SQL (COALESCE already takes care of 0 for counts, but for safety)
cols_to_fill_zero = [
    'num_likes_first_24h', 'num_comments_first_24h', 'num_shares_first_24h',
    'total_interactions_first_24h', 'unique_users_first_24h'
]
post_early_features_df[cols_to_fill_zero] = post_early_features_df[cols_to_fill_zero].fillna(0)

# Calculate engagement_rate_first_24h
post_early_features_df['engagement_rate_first_24h'] = post_early_features_df['total_interactions_first_24h'] / (post_early_features_df['user_follower_count'] + 1)
post_early_features_df['engagement_rate_first_24h'] = post_early_features_df['engagement_rate_first_24h'].replace([np.inf, -np.inf], 0).fillna(0)

# Calculate share_comment_ratio_first_24h
post_early_features_df['share_comment_ratio_first_24h'] = post_early_features_df['num_shares_first_24h'] / (post_early_features_df['num_comments_first_24h'] + 1)
post_early_features_df['share_comment_ratio_first_24h'] = post_early_features_df['share_comment_ratio_first_24h'].replace([np.inf, -np.inf], 0).fillna(0)


# Create the Binary Target `will_go_viral`
viral_window_days = 7

# Merge post_date into interactions_df to calculate the target relative to post creation
interactions_with_post_date = interactions_df.merge(posts_df[['post_id', 'post_date']], on='post_id', how='left')

# Define end of viral window for each interaction
interactions_with_post_date['viral_window_end'] = interactions_with_post_date['post_date'] + pd.Timedelta(viral_window_days, 'days')

# Filter interactions within the viral window and active types
active_interactions_within_window = interactions_with_post_date[
    (interactions_with_post_date['interaction_timestamp'] >= interactions_with_post_date['post_date']) &
    (interactions_with_post_date['interaction_timestamp'] <= interactions_with_post_date['viral_window_end']) &
    (interactions_with_post_date['interaction_type'].isin(['like', 'comment', 'share']))
]

# Calculate total active interactions per post within the viral window
total_active_interactions_7d = active_interactions_within_window.groupby('post_id')['interaction_id'].count().reset_index(name='total_active_interactions_7d')

# Get the 90th percentile threshold
if not total_active_interactions_7d.empty:
    virality_threshold = total_active_interactions_7d['total_active_interactions_7d'].quantile(0.90)
    print(f"\n90th percentile virality threshold (total active interactions in 7 days): {virality_threshold:.2f}")
else:
    virality_threshold = 0 # If no interactions, no viral posts

# Determine viral status for all posts (left join to ensure all posts are included)
posts_viral_status_df = posts_df[['post_id']].copy()
posts_viral_status_df = posts_viral_status_df.merge(total_active_interactions_7d, on='post_id', how='left')
posts_viral_status_df['total_active_interactions_7d'] = posts_viral_status_df['total_active_interactions_7d'].fillna(0)
posts_viral_status_df['will_go_viral'] = (posts_viral_status_df['total_active_interactions_7d'] > virality_threshold).astype(int)

# Merge the target with post_early_features_df
post_early_features_df = post_early_features_df.merge(posts_viral_status_df[['post_id', 'will_go_viral']], on='post_id', how='left')

# Define features X and target y
numerical_features = [
    'num_hashtags', 'sentiment_score', 'user_follower_count',
    'num_likes_first_24h', 'num_comments_first_24h', 'num_shares_first_24h',
    'total_interactions_first_24h', 'unique_users_first_24h',
    'engagement_rate_first_24h', 'share_comment_ratio_first_24h'
]
categorical_features = ['category']

X = post_early_features_df[numerical_features + categorical_features]
y = post_early_features_df['will_go_viral']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"Viral posts in target (1s): {y.sum()} ({y.mean()*100:.2f}%)")
print(f"Viral posts in training set (1s): {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Viral posts in test set (1s): {y_test.sum()} ({y_test.mean()*100:.2f}%)")


# --- 4. Data Visualization ---

print("\n--- Generating Visualizations ---")
plt.style.use('seaborn-v0_8-darkgrid') 

# Plot 1: Engagement Rate First 24h vs. Viral Status
plt.figure(figsize=(8, 6))
sns.violinplot(x='will_go_viral', y='engagement_rate_first_24h', data=post_early_features_df)
plt.title('Engagement Rate (First 24h) Distribution by Viral Status')
plt.xlabel('Will Go Viral (0=No, 1=Yes)')
plt.ylabel('Engagement Rate in First 24 Hours')
plt.xticks([0, 1], ['Not Viral', 'Viral'])
plt.tight_layout()
plt.show()

# Plot 2: Proportion of Viral Posts across Categories (Stacked Bar Chart)
category_virality_proportions = post_early_features_df.groupby('category')['will_go_viral'].value_counts(normalize=True).unstack(fill_value=0)
category_virality_proportions.rename(columns={0: 'Not Viral', 1: 'Viral'}, inplace=True)

plt.figure(figsize=(10, 7))
category_virality_proportions.plot(kind='bar', stacked=True, color=['lightcoral', 'skyblue'])
plt.title('Proportion of Viral vs. Non-Viral Posts by Category')
plt.xlabel('Category')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Viral Status')
plt.tight_layout()
plt.show()


# --- 5. ML Pipeline & Evaluation ---

print("\n--- Building and Evaluating ML Pipeline ---")

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create the ML pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred)

print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")
print("\nClassification Report on Test Set:")
print(class_report)

print("\n--- Script Finished ---")