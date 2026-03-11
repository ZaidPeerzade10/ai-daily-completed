import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # CORRECTED: Import SimpleImputer from sklearn.impute
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Ensure plots don't try to open a GUI window
plt.switch_backend('Agg')

# --- 1. Generate Synthetic Data (Pandas/Numpy) ---

# Random seed for reproducibility
np.random.seed(42)

# 1.1 users_df
num_users = np.random.randint(500, 701)
users_df = pd.DataFrame({
    'user_id': np.arange(num_users),
    'signup_date': pd.to_datetime(pd.Series(np.random.uniform(
        pd.to_datetime('2019-01-01').timestamp(),
        pd.to_datetime('2023-12-31').timestamp(),
        num_users
    )).apply(datetime.datetime.fromtimestamp).dt.date),
    'reputation_score': np.random.randint(0, 101, num_users),
    'account_status': np.random.choice(['Active', 'Suspended', 'New', 'Banned'], num_users, p=[0.7, 0.1, 0.15, 0.05])
})

# 1.2 content_df
num_content = np.random.randint(5000, 8001)
content_data = {
    'content_id': np.arange(num_content),
    'user_id': np.random.choice(users_df['user_id'], num_content),
    'word_count': np.random.randint(10, 1001, num_content),
    'contains_link': np.random.choice([0, 1], num_content, p=[0.7, 0.3]),
    'has_keywords': np.random.choice([0, 1], num_content, p=[0.85, 0.15])
}
content_df = pd.DataFrame(content_data)

# Join with users_df to get signup_date and user details for each content
content_df = content_df.merge(users_df[['user_id', 'signup_date', 'reputation_score', 'account_status']], on='user_id', how='left')

# Generate post_date after signup_date
content_df['post_date'] = content_df.apply(
    lambda row: pd.to_datetime(row['signup_date']) + pd.Timedelta(days=np.random.randint(1, 365*5)), axis=1
).dt.date

# Sort content_df by user_id and post_date for sequential spam pattern simulation
content_df = content_df.sort_values(by=['user_id', 'post_date']).reset_index(drop=True)

# Simulate realistic spam patterns for 'is_spam' (will be assigned to moderation_df)
content_df['spam_probability'] = 0.03 # Base spam rate

# Bias 1: Users with lower reputation_score or account_status='Suspended'/'Banned' are more likely to post spam
content_df['spam_probability'] += (1 - content_df['reputation_score'] / 100) * 0.1
content_df.loc[content_df['account_status'] == 'Suspended', 'spam_probability'] += 0.2
content_df.loc[content_df['account_status'] == 'Banned', 'spam_probability'] += 0.4

# Bias 2: Content with contains_link=1 or has_keywords=1 is more likely to be spam
content_df.loc[content_df['contains_link'] == 1, 'spam_probability'] += 0.15
content_df.loc[content_df['has_keywords'] == 1, 'spam_probability'] += 0.15

# Bias 3: Spam often has extreme word_count (very low or very high)
mean_word_count = content_df['word_count'].mean()
std_word_count = content_df['word_count'].std()
content_df['word_count_deviation'] = np.abs(content_df['word_count'] - mean_word_count) / std_word_count
content_df.loc[content_df['word_count_deviation'] > 2, 'spam_probability'] += 0.15

# Bias 4: Users who have previously posted spam are more likely to post spam again
# This needs to be done sequentially
prior_spam_counts = {} # user_id -> spam_count
for idx, row in content_df.iterrows():
    user_id = row['user_id']
    if user_id not in prior_spam_counts:
        prior_spam_counts[user_id] = 0
    
    # Increase spam probability if user has prior spam posts
    if prior_spam_counts[user_id] > 0:
        content_df.loc[idx, 'spam_probability'] += min(0.3, prior_spam_counts[user_id] * 0.1) # Max 0.3 boost

    # Temporarily decide if this post *would be* spam to update prior_spam_counts for next iteration
    # This is a hacky way to get the sequential effect without knowing the final 'is_spam' yet
    if np.random.rand() < row['spam_probability']:
        prior_spam_counts[user_id] += 1

# Clip probabilities to [0, 1]
content_df['spam_probability'] = np.clip(content_df['spam_probability'], 0.01, 0.9)


# 1.3 moderation_df
num_moderations = np.random.randint(500, 801)
moderation_content_sample = content_df.sample(num_moderations, random_state=42, replace=False).copy()

# Assign 'is_spam' based on 'spam_probability'
moderation_content_sample['is_spam'] = (np.random.rand(len(moderation_content_sample)) < moderation_content_sample['spam_probability']).astype(int)

# Ensure overall spam rate is within 5-10% by potentially adjusting a few samples if needed
actual_spam_rate = moderation_content_sample['is_spam'].mean()
target_spam_rate = 0.08 # Aim for 8%
if actual_spam_rate < target_spam_rate - 0.01 or actual_spam_rate > target_spam_rate + 0.01:
    diff_count = int(abs(target_spam_rate - actual_spam_rate) * len(moderation_content_sample))
    if actual_spam_rate < target_spam_rate: # Need more spam
        indices_to_flip = moderation_content_sample[moderation_content_sample['is_spam'] == 0].index.to_numpy()
        if len(indices_to_flip) > diff_count:
            np.random.shuffle(indices_to_flip)
            moderation_content_sample.loc[indices_to_flip[:diff_count], 'is_spam'] = 1
    else: # Need less spam
        indices_to_flip = moderation_content_sample[moderation_content_sample['is_spam'] == 1].index.to_numpy()
        if len(indices_to_flip) > diff_count:
            np.random.shuffle(indices_to_flip)
            moderation_content_sample.loc[indices_to_flip[:diff_count], 'is_spam'] = 0

moderation_df = pd.DataFrame({
    'moderation_id': np.arange(len(moderation_content_sample)),
    'content_id': moderation_content_sample['content_id'].values,
    'is_spam': moderation_content_sample['is_spam'].values
})

# Ensure moderation_date is after post_date
moderation_df = moderation_df.merge(content_df[['content_id', 'post_date']], on='content_id', how='left')
moderation_df['moderation_date'] = moderation_df.apply(
    lambda row: pd.to_datetime(row['post_date']) + pd.Timedelta(days=np.random.randint(1, 365)), axis=1
).dt.date
moderation_df = moderation_df.drop(columns=['post_date'])

# Final cleanup of content_df before SQL load (remove temp columns)
content_df_for_sql = content_df.drop(columns=['spam_probability', 'word_count_deviation'])

print("--- Synthetic Data Generated ---")
print(f"Users: {len(users_df)} rows")
print(f"Content: {len(content_df_for_sql)} rows")
print(f"Moderation: {len(moderation_df)} rows")
print(f"Overall spam rate in moderation_df: {moderation_df['is_spam'].mean():.2%}")
print("-" * 30)


# --- 2. Load into SQLite & SQL Feature Engineering ---
conn = sqlite3.connect(':memory:')

# Convert date columns to string for SQLite
users_df['signup_date'] = users_df['signup_date'].astype(str)
content_df_for_sql['signup_date'] = content_df_for_sql['signup_date'].astype(str)
content_df_for_sql['post_date'] = content_df_for_sql['post_date'].astype(str)
moderation_df['moderation_date'] = moderation_df['moderation_date'].astype(str)

users_df.to_sql('users', conn, index=False, if_exists='replace')
content_df_for_sql.to_sql('content', conn, index=False, if_exists='replace')
moderation_df.to_sql('moderation', conn, index=False, if_exists='replace')


# SQL Query for feature engineering
# The ORDER BY clause in window functions needs to be consistent and include content_id
# for deterministic ordering in case of same post_date.
# LAG(c.post_date, 1, u.signup_date) provides the previous post date or signup date for the first post.
sql_query = """
WITH ContentWithLaggedFeatures AS (
    SELECT
        c.content_id,
        c.user_id,
        c.post_date,
        c.word_count,
        c.contains_link,
        c.has_keywords,
        m.is_spam,
        u.reputation_score,
        u.account_status,
        u.signup_date,
        -- Calculate prior total posts
        SUM(1) OVER (
            PARTITION BY c.user_id
            ORDER BY c.post_date, c.content_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS user_prior_total_posts,
        -- Calculate prior spam posts (CORRECTED: used m.is_spam, not m_prior.is_spam)
        SUM(CASE WHEN m.is_spam = 1 THEN 1 ELSE 0 END) OVER (
            PARTITION BY c.user_id
            ORDER BY c.post_date, c.content_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS user_prior_spam_posts,
        -- Get previous post date, or signup_date if it's the first post
        COALESCE(
            LAG(c.post_date, 1) OVER (
                PARTITION BY c.user_id
                ORDER BY c.post_date, c.content_id
            ),
            u.signup_date
        ) AS prev_post_date
    FROM
        content c
    JOIN
        users u ON c.user_id = u.user_id
    LEFT JOIN
        moderation m ON c.content_id = m.content_id
)
SELECT
    content_id,
    user_id,
    post_date,
    word_count,
    contains_link,
    has_keywords,
    is_spam,
    reputation_score,
    account_status,
    signup_date,
    COALESCE(user_prior_total_posts, 0) AS user_prior_total_posts,
    COALESCE(user_prior_spam_posts, 0) AS user_prior_spam_posts,
    CAST(COALESCE(user_prior_spam_posts, 0) AS REAL) / NULLIF(COALESCE(user_prior_total_posts, 0), 0) AS user_prior_spam_ratio,
    JULIANDAY(post_date) - JULIANDAY(prev_post_date) AS days_since_last_user_post
FROM
    ContentWithLaggedFeatures
ORDER BY
    user_id, post_date, content_id;
"""

content_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("--- SQL Feature Engineering Complete ---")
print(f"Engineered DataFrame shape: {content_features_df.shape}")
print(content_features_df[['user_id', 'post_date', 'user_prior_total_posts', 'user_prior_spam_posts', 'days_since_last_user_post', 'is_spam']].head())
print("-" * 30)


# --- 3. Pandas Feature Engineering & Binary Target Creation ---

# Handle NaN values for prior features
content_features_df['user_prior_total_posts'] = content_features_df['user_prior_total_posts'].fillna(0).astype(int)
content_features_df['user_prior_spam_posts'] = content_features_df['user_prior_spam_posts'].fillna(0).astype(int)
content_features_df['user_prior_spam_ratio'] = content_features_df['user_prior_spam_ratio'].fillna(0.0)

# Convert dates to datetime objects
content_features_df['signup_date'] = pd.to_datetime(content_features_df['signup_date'])
content_features_df['post_date'] = pd.to_datetime(content_features_df['post_date'])

# Calculate days_since_signup_at_post
content_features_df['days_since_signup_at_post'] = (content_features_df['post_date'] - content_features_df['signup_date']).dt.days

# For days_since_last_user_post, SQL query should have correctly filled initial ones using signup_date.
# Any remaining NaNs would imply an issue, but we'll fill with days_since_signup_at_post as a fallback.
content_features_df['days_since_last_user_post'] = content_features_df['days_since_last_user_post'].fillna(
    content_features_df['days_since_signup_at_post']
)

# Convert 'is_spam' target to binary (if it's not already, and fillna for content not moderated)
content_features_df['is_spam'] = content_features_df['is_spam'].fillna(0).astype(int)

# Define features (X) and target (y)
numerical_features = [
    'word_count',
    'reputation_score',
    'user_prior_total_posts',
    'user_prior_spam_posts',
    'user_prior_spam_ratio',
    'days_since_last_user_post',
    'days_since_signup_at_post'
]
categorical_features = [
    'contains_link', # Technically binary, but OneHotEncoder handles it correctly
    'has_keywords',  # Technically binary, but OneHotEncoder handles it correctly
    'account_status'
]

X = content_features_df[numerical_features + categorical_features]
y = content_features_df['is_spam']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("--- Pandas Feature Engineering & Data Split Complete ---")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"Spam rate in y_train: {y_train.mean():.2%}")
print(f"Spam rate in y_test: {y_test.mean():.2%}")
print("-" * 30)


# --- 4. Data Visualization ---

print("--- Generating Visualizations ---")

plt.figure(figsize=(10, 6))
sns.violinplot(x='is_spam', y='reputation_score', data=content_features_df)
plt.title('Reputation Score Distribution for Spam vs. Non-Spam Content')
plt.xlabel('Is Spam (0: No, 1: Yes)')
plt.ylabel('Reputation Score')
plt.tight_layout()
plt.savefig('reputation_score_vs_spam.png')
plt.close()
print("Saved 'reputation_score_vs_spam.png'")

# Stacked bar chart for account_status vs is_spam
status_spam_counts = content_features_df.groupby(['account_status', 'is_spam']).size().unstack(fill_value=0)
status_spam_proportions = status_spam_counts.apply(lambda x: x / x.sum(), axis=1)

plt.figure(figsize=(10, 6))
status_spam_proportions.plot(kind='bar', stacked=True, color=['lightgreen', 'salmon'])
plt.title('Proportion of Spam by Account Status')
plt.xlabel('Account Status')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Is Spam', labels=['Non-Spam', 'Spam'])
plt.tight_layout()
plt.savefig('account_status_vs_spam_stacked_bar.png')
plt.close()
print("Saved 'account_status_vs_spam_stacked_bar.png'")
print("-" * 30)


# --- 5. ML Pipeline & Evaluation (Binary Classification) ---

# Preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the ML pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

print("--- Training ML Pipeline ---")
model_pipeline.fit(X_train, y_train)
print("Pipeline training complete.")

# Predict probabilities on the test set
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
y_pred = model_pipeline.predict(X_test) # For classification report

print("\n--- Model Evaluation ---")
# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test ROC AUC Score: {roc_auc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("-" * 30)