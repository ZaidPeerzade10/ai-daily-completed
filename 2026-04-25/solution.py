import pandas as pd
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Corrected import location as per feedback
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate Synthetic Data (Pandas/Numpy)
np.random.seed(42)

# Define current prediction date
current_prediction_date = pd.to_datetime('2024-03-01')

# Users DataFrame
num_users = np.random.randint(500, 801)
signup_dates = pd.to_datetime(current_prediction_date - pd.to_timedelta(np.random.randint(0, 3 * 365, num_users), unit='days'))
regions = ['North', 'South', 'East', 'West']
subscription_tiers = ['Free', 'Basic', 'Premium']
device_types = ['Mobile', 'Desktop', 'Tablet']
ages = np.random.randint(18, 66, num_users)

users_df = pd.DataFrame({
    'user_id': range(1, num_users + 1),
    'signup_date': signup_dates,
    'region': np.random.choice(regions, num_users),
    'subscription_tier': np.random.choice(subscription_tiers, num_users, p=[0.4, 0.35, 0.25]),
    'device_type': np.random.choice(device_types, num_users, p=[0.5, 0.3, 0.2]),
    'age': ages
})

# Content DataFrame
num_content = np.random.randint(100, 151)
genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Documentary', 'Fantasy', 'Horror', 'Thriller']
avg_ratings = np.random.uniform(1.0, 5.0, num_content)
production_years = np.random.randint(current_prediction_date.year - 15, current_prediction_date.year + 1, num_content)

content_df = pd.DataFrame({
    'content_id': range(1, num_content + 1),
    'genre': np.random.choice(genres, num_content),
    'avg_rating': avg_ratings,
    'production_year': production_years
})

# Interactions DataFrame
num_interactions = np.random.randint(10000, 15001)

interaction_user_ids = np.random.choice(users_df['user_id'], num_interactions, replace=True)
interaction_content_ids = np.random.choice(content_df['content_id'], num_interactions, replace=True)
interaction_types = ['view', 'like', 'share', 'bookmark']

# Generate interaction dates ensuring they are after signup_date and before current_prediction_date
temp_interactions = pd.DataFrame({
    'user_id': interaction_user_ids,
    'content_id': interaction_content_ids,
    'interaction_type': np.random.choice(interaction_types, num_interactions, p=[0.7, 0.15, 0.1, 0.05])
})

temp_interactions = temp_interactions.merge(users_df[['user_id', 'signup_date', 'subscription_tier', 'device_type']], on='user_id')
temp_interactions = temp_interactions.merge(content_df[['content_id', 'avg_rating', 'genre']], on='content_id')

# Simulate interaction dates and duration_minutes with patterns
interaction_dates = []
duration_minutes = []

for idx, row in temp_interactions.iterrows():
    # Ensure interaction_date is after signup_date and before current_prediction_date
    start_date = row['signup_date'] + pd.Timedelta(days=1)
    end_date = current_prediction_date
    
    if start_date >= end_date: # Handle edge case where signup is too recent
        interaction_dates.append(start_date) # Or skip, but for synthetic data, force
    else:
        # Spread interactions over time for users
        date_range_days = (end_date - start_date).days
        if date_range_days <= 0: # Ensure at least one day
             interaction_dates.append(start_date)
        else:
            random_days = np.random.randint(0, date_range_days + 1)
            interaction_dates.append(start_date + pd.Timedelta(days=random_days))

    # Duration minutes patterns
    base_duration = np.random.randint(1, 61) # Base duration 1-60 minutes
    if row['subscription_tier'] == 'Premium':
        base_duration += np.random.randint(10, 31) # Premium users watch longer
    
    # Device type influence
    if row['device_type'] == 'Mobile' and row['genre'] in ['Comedy', 'Documentary']:
        base_duration = max(1, base_duration - np.random.randint(0, 20)) # Mobile users might watch shorter for some genres
    elif row['device_type'] == 'Desktop' and row['genre'] in ['Sci-Fi', 'Drama']:
        base_duration = min(120, base_duration + np.random.randint(0, 30)) # Desktop users might watch longer for some genres
    
    # Higher avg_rating content should generally have more view interactions and potentially longer duration
    # This is more complex to simulate directly in duration, but can be done implicitly by higher overall engagement.
    # For duration, we'll give a slight boost for highly-rated content
    if row['avg_rating'] > 4.0:
        base_duration = min(120, base_duration + np.random.randint(0, 15))

    duration_minutes.append(min(120, max(1, base_duration))) # Ensure duration is between 1 and 120

interactions_df = pd.DataFrame({
    'interaction_id': range(1, num_interactions + 1),
    'user_id': temp_interactions['user_id'],
    'content_id': temp_interactions['content_id'],
    'interaction_date': interaction_dates,
    'interaction_type': temp_interactions['interaction_type'],
    'duration_minutes': duration_minutes
})

# Simulate 'like' interactions for Premium users
premium_users = users_df[users_df['subscription_tier'] == 'Premium']['user_id'].values
num_premium_interactions_to_boost = int(num_interactions * 0.05) # Boost 5% of interactions
if len(premium_users) > 0 and num_premium_interactions_to_boost > 0:
    # Randomly select some interactions and change them to 'like' if user is premium
    # This needs to be done carefully to avoid changing existing 'like' types or overwriting
    # Let's target non-like interactions for premium users to change to 'like'
    eligible_indices = interactions_df[
        (interactions_df['user_id'].isin(premium_users)) &
        (interactions_df['interaction_type'] != 'like')
    ].index
    if len(eligible_indices) > 0:
        indices_to_change = np.random.choice(eligible_indices, min(num_premium_interactions_to_boost, len(eligible_indices)), replace=False)
        interactions_df.loc[indices_to_change, 'interaction_type'] = 'like'

# Sort interactions_df
interactions_df = interactions_df.sort_values(by=['user_id', 'interaction_date']).reset_index(drop=True)

print("--- Synthetic Data Generated ---")
print("Users DF Head:\n", users_df.head())
print("Content DF Head:\n", content_df.head())
print("Interactions DF Head:\n", interactions_df.head())
print(f"Number of Users: {len(users_df)}")
print(f"Number of Content Items: {len(content_df)}")
print(f"Number of Interactions: {len(interactions_df)}")

# 2. Load into SQLite & SQL Feature Engineering (Historical Interaction Patterns)
conn = sqlite3.connect(':memory:')

users_df.to_sql('users', conn, index=False, if_exists='replace')
content_df.to_sql('content', conn, index=False, if_exists='replace')
interactions_df.to_sql('interactions', conn, index=False, if_exists='replace')

history_cutoff_date_str = (current_prediction_date - pd.Timedelta(30, 'days')).strftime('%Y-%m-%d')
history_window_start_date_str = (current_prediction_date - pd.Timedelta(90, 'days')).strftime('%Y-%m-%d') # history_cutoff_date - 60 days

sql_query = f"""
SELECT
    u.user_id,
    u.signup_date,
    u.region,
    u.subscription_tier,
    u.device_type,
    u.age,
    COALESCE(SUM(CASE WHEN i.interaction_date BETWEEN DATE('{history_window_start_date_str}') AND DATE('{history_cutoff_date_str}') THEN 1 ELSE 0 END), 0) AS num_interactions_prev_60d,
    COALESCE(SUM(CASE WHEN i.interaction_date BETWEEN DATE('{history_window_start_date_str}') AND DATE('{history_cutoff_date_str}') THEN i.duration_minutes ELSE 0 END), 0) AS total_duration_prev_60d,
    COALESCE(COUNT(DISTINCT CASE WHEN i.interaction_date BETWEEN DATE('{history_window_start_date_str}') AND DATE('{history_cutoff_date_str}') THEN c.genre ELSE NULL END), 0) AS num_unique_genres_prev_60d,
    COALESCE(AVG(CASE WHEN i.interaction_date BETWEEN DATE('{history_window_start_date_str}') AND DATE('{history_cutoff_date_str}') THEN c.avg_rating ELSE NULL END), 0.0) AS avg_content_rating_prev_60d
FROM
    users u
LEFT JOIN
    interactions i ON u.user_id = i.user_id
LEFT JOIN
    content c ON i.content_id = c.content_id
GROUP BY
    u.user_id, u.signup_date, u.region, u.subscription_tier, u.device_type, u.age
ORDER BY
    u.user_id;
"""

user_history_features_df = pd.read_sql_query(sql_query, conn)
conn.close()

print("\n--- SQL Feature Engineering Results (Historical) ---")
print(user_history_features_df.head())
print(f"Shape of user_history_features_df: {user_history_features_df.shape}")

# 3. Pandas Feature Engineering & Multi-Class Target Creation (Next Preferred Genre)
# Handle NaN values for aggregated features (SQL COALESCE should mostly handle this, but for robustness)
user_history_features_df['num_interactions_prev_60d'] = user_history_features_df['num_interactions_prev_60d'].fillna(0).astype(int)
user_history_features_df['total_duration_prev_60d'] = user_history_features_df['total_duration_prev_60d'].fillna(0).astype(int)
user_history_features_df['num_unique_genres_prev_60d'] = user_history_features_df['num_unique_genres_prev_60d'].fillna(0).astype(int)
user_history_features_df['avg_content_rating_prev_60d'] = user_history_features_df['avg_content_rating_prev_60d'].fillna(0.0)

# Convert signup_date to datetime
user_history_features_df['signup_date'] = pd.to_datetime(user_history_features_df['signup_date'])

# Recalculate dates as Pandas datetime objects
current_prediction_date = pd.to_datetime('2024-03-01')
history_cutoff_date = current_prediction_date - pd.Timedelta(30, 'days')

# Calculate interaction_frequency_prev_60d
user_history_features_df['interaction_frequency_prev_60d'] = user_history_features_df['num_interactions_prev_60d'] / 60.0
user_history_features_df['interaction_frequency_prev_60d'] = user_history_features_df['interaction_frequency_prev_60d'].fillna(0)

# Create the Multi-Class Target `next_preferred_genre`
target_window_start = history_cutoff_date
target_window_end = history_cutoff_date + pd.Timedelta(30, 'days')

# Filter interactions for the target window
future_interactions = interactions_df[
    (interactions_df['interaction_date'] > target_window_start) &
    (interactions_df['interaction_date'] <= target_window_end)
].copy()

# Join with content_df to get genre
future_interactions = future_interactions.merge(content_df[['content_id', 'genre']], on='content_id')

# Calculate total duration per user per genre in the future window
genre_duration_agg = future_interactions.groupby(['user_id', 'genre'])['duration_minutes'].sum().reset_index()

# Find the genre with the highest total duration for each user
# Using idxmax() after sorting to handle ties (first one wins)
next_preferred_genre_series = genre_duration_agg.loc[genre_duration_agg.groupby('user_id')['duration_minutes'].idxmax()]
next_preferred_genre_df = next_preferred_genre_series[['user_id', 'genre']].rename(columns={'genre': 'next_preferred_genre'})

# Merge with user_history_features_df
user_history_features_df = user_history_features_df.merge(next_preferred_genre_df, on='user_id', how='left')

# Assign 'No Future Preference' to users with no interactions in the target window
user_history_features_df['next_preferred_genre'] = user_history_features_df['next_preferred_genre'].fillna('No Future Preference')

print("\n--- Pandas Feature Engineering & Target Creation ---")
print("User History Features with Target Head:\n", user_history_features_df.head())
print("Target Value Counts:\n", user_history_features_df['next_preferred_genre'].value_counts())

# Define features X and target y
numerical_features = [
    'num_interactions_prev_60d', 'total_duration_prev_60d',
    'num_unique_genres_prev_60d', 'avg_content_rating_prev_60d',
    'age', 'interaction_frequency_prev_60d'
]
categorical_features = ['region', 'subscription_tier', 'device_type']

X = user_history_features_df[numerical_features + categorical_features]
y = user_history_features_df['next_preferred_genre']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")


# 4. Data Visualization
print("\n--- Generating Visualizations ---")
plt.style.use('seaborn-v0_8-darkgrid') # Use a consistent style

# Violin plot: distribution of avg_content_rating_prev_60d for each next_preferred_genre
plt.figure(figsize=(12, 7))
sns.violinplot(x='next_preferred_genre', y='avg_content_rating_prev_60d', data=user_history_features_df, inner='quartile', palette='viridis')
plt.title('Distribution of Average Content Rating (Prev 60 Days) by Next Preferred Genre')
plt.xlabel('Next Preferred Genre')
plt.ylabel('Avg Content Rating (Prev 60 Days)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Stacked bar chart: proportion of next_preferred_genre across subscription_tier
genre_tier_pivot = user_history_features_df.groupby('subscription_tier')['next_preferred_genre'].value_counts(normalize=True).unstack(fill_value=0)
genre_tier_pivot.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
plt.title('Proportion of Next Preferred Genre by Subscription Tier')
plt.xlabel('Subscription Tier')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.legend(title='Next Preferred Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 5. ML Pipeline & Evaluation (Multi-Class)
print("\n--- Building and Evaluating ML Pipeline ---")

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train the pipeline
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report:\n", report)

print("\n--- Pipeline Execution Complete ---")